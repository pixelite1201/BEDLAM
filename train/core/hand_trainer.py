import os
import cv2
import torch
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from . import config
from ..dataset.dataset_hand import DatasetHand
from ..utils.train_utils import set_seed
from ..utils.eval_utils import reconstruction_error
from ..utils.image_utils import denormalize_images
from ..utils.renderer_cam import render_image_group
from ..utils.renderer import Renderer
from ..models.hand import Hand
from ..losses.hand_loss import HandLoss
from ..models.head.mano import MANO


class HandTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(HandTrainer, self).__init__()

        self.hparams.update(hparams)

        self.model = Hand(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )

        self.loss_fn = HandLoss(hparams=self.hparams)
        self.val_ds = self.val_dataset()
        self.save_itr = 0
        self.mano = MANO(model_path=config.MANO_MODEL_DIR, use_pca=False, is_rhand=True)
        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.mano.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb, dataloader_nb=0, vis=False, save=True, mesh_save_dir=None):
        images = batch['img']
        pred = self(images)
        #self.gt_projection(batch, pred, batch_nb)#,x=0)
        #self.visualize(batch, pred, batch_nb, dataloader_nb)
        #self.visualize_mesh(batch, pred, batch_nb, dataloader_nb,x=0)
        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)
        self.log('train_loss', loss, logger=True, sync_dist=True)

        for k, v in loss_dict.items():
            self.log(k, v, logger=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb=0, vis=False, save=True, mesh_save_dir=None):
        images = batch['img']
        dataset_index = batch['dataset_index'].detach().cpu().numpy()
        with torch.no_grad():
            pred = self(images)
        # self.visualize_mesh(batch, pred, batch_nb, dataloader_nb,x=0)
        # self.visualize(batch, pred, batch_nb, dataloader_nb)
        # self.gt_projection(batch, pred, batch_nb)#,x=0)

        pred_joints = pred['joints3d']
        gt_joints = batch['joints3d']

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).cpu().numpy()
        # Reconstuction_error

        r_error, _ = reconstruction_error(
            gt_joints.cpu().numpy(),
            pred_joints.cpu().numpy(),
            reduction=None
        )
        val_mpjpe = error.mean(-1)
        val_pampjpe = r_error.mean(-1)
        loss_dict = {}
        val_dataset_names = self.hparams.DATASET.VAL_DS.split('_')

        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset
            ds_idx = val_dataset_names.index(ds.dataset)

            idxs = np.where(dataset_index == ds_idx)
            loss_dict[ds_name + '_mpjpe'] = list(val_mpjpe[idxs])
            loss_dict[ds_name + '_pampjpe'] = list(val_pampjpe[idxs])

        return loss_dict

    def validation_epoch_end(self, outputs):
        logger.info(f'***** Epoch {self.current_epoch} *****')
        val_log = {}

        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset
            mpjpe = 1000 * np.hstack(np.array([x[ds_name + '_mpjpe'] for x in outputs])).mean()
            pampjpe = (1000 * np.hstack(np.array([x[ds_name + '_pampjpe'] for x in outputs]))).mean()
            if self.trainer.is_global_zero:
                logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))

                val_log[ds_name + '_val_mpjpe'] = mpjpe
                val_log[ds_name + '_val_pampjpe'] = pampjpe
        self.log('val_loss', val_log[self.val_ds[0].dataset + '_val_pampjpe'], logger=True, sync_dist=True, rank_zero_only=True)

        for k, v in val_log.items():
            self.log(k, v, logger=True, sync_dist=True, rank_zero_only=True)

    def test_step(self, batch, batch_nb, dataloader_nb=0):
        return self.validation_step(batch, batch_nb, dataloader_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD,
        )

    def train_dataset(self):
        options = self.hparams.DATASET
        dataset_names = options.DATASETS_AND_RATIOS.split('_')
        dataset_list = [DatasetHand(options, ds) for ds in dataset_names]
        train_ds = ConcatDataset(dataset_list)
        return train_ds

    def train_dataloader(self):
        set_seed(self.hparams.SEED_VALUE)
        self.train_ds = self.train_dataset()

        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
        )

    def val_dataset(self):
        datasets = self.hparams.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_datasets = []
        for dataset_name in datasets:
            val_datasets.append(
                DatasetHand(
                    options=self.hparams.DATASET,
                    dataset=dataset_name,
                    is_train=False,
                )
            )

        return val_datasets

    def val_dataloader(self):
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds,
                    batch_size=self.hparams.DATASET.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.hparams.DATASET.NUM_WORKERS,
                    drop_last=True
                )
            )
        return dataloaders

    def test_dataloader(self):
        return self.val_dataloader()
                
    def visualize_mesh(self, input_batch, output, batch_idx, dataloader_nb, x=-1):
        # images = input_batch['img']
        import trimesh
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
        os.makedirs(save_dir, exist_ok=True)
        
        pred_vertices = output['vertices'].detach().cpu().numpy()
        gt_vertices = input_batch['vertices'].detach().cpu().numpy()
        if x!=-1:
            i=x
            imgname = input_batch['imgname'][i].split('/')[-1]
            gt = trimesh.Trimesh(vertices=gt_vertices[i],faces=self.mano.faces, process=False)
            gt.visual.face_colors = [200, 200, 250, 100]
            gt.visual.vertex_colors = [200, 200, 250, 100]

            pred = trimesh.Trimesh(vertices=pred_vertices[i],faces=self.mano.faces,process=False)
            save_filename = os.path.join(save_dir, f'{self.current_epoch:04d}_{dataloader_nb:02d}_'
                                f'{batch_idx:05d}_{i:02d}_{os.path.basename(imgname)}')
            gt.export(save_filename+str(i)+'_gt.obj')
            pred.export(save_filename+str(i)+'_pred.obj')
               

        else:
            for i in range(pred_vertices.shape[0]):
                imgname = input_batch['imgname'][i].split('/')[-1]
                gt = trimesh.Trimesh(vertices=gt_vertices[i],faces=self.mano.faces, process=False)
                gt.visual.face_colors = [200, 200, 250, 100]
                gt.visual.vertex_colors = [200, 200, 250, 100]

                pred = trimesh.Trimesh(vertices=pred_vertices[i],faces=self.mano.faces,process=False)
                save_filename = os.path.join(save_dir, f'{self.current_epoch:04d}_{dataloader_nb:02d}_'
                                    f'{batch_idx:05d}_{i:02d}_{os.path.basename(imgname)}')
                gt.export(save_filename+str(i)+'_gt.obj')
                pred.export(save_filename+str(i)+'_pred.obj')    

    def visualize(self, input_batch, output, batch_idx, dataloader_nb):
        images = input_batch['img']
        img = denormalize_images(images)[0]
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
        os.makedirs(save_dir, exist_ok=True)
        pred_vertices = output['vertices'].detach()
        pred_kp_2d = output['joints2d'].detach().clone()
        pred_cam_t = output['pred_cam_t'].detach()
        for i,_ in enumerate(pred_vertices):
            if i>1:
                break
            images_pred = self.renderer.visualize_tb(
                pred_vertices[i:i+1],
                pred_cam_t[i:i+1],
                images[i:i+1],
                sideview=True,
            )

            save_filename = os.path.join(save_dir, f'result_{self.current_epoch:04d}_'
                                                f'{dataloader_nb:02d}_{batch_idx:05d}_{i}.jpg')
            if save_filename is not None:
                images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
                images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)
                cv2.imwrite(save_filename, cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB))

    def gt_projection(self, input_batch, output, batch_idx, max_save_img=1):
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images_gt')
        os.makedirs(save_dir, exist_ok=True)
        img_size = input_batch['orig_shape'].rot90().T

        gt_keypoints_2d_full_img = input_batch['keypoints_orig']
        focal_length = input_batch['focal_length']
        gt_out = self.mano(
        betas=input_batch['betas'],
        right_hand_pose=input_batch['pose'][:, 3:],
        global_orient=input_batch['pose'][:, :3]
        )
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        translation = input_batch['translation'][:,:3]


        for i in range(len(input_batch['imgname'])):

            imgname = input_batch['imgname'][i]
            cy, cx = input_batch['orig_shape'][i] // 2
            save_filename = os.path.join(save_dir, f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')

            rendered_img = render_image_group(
                image=read_img(imgname),
                camera_translation=translation[i],
                vertices=gt_vertices[i],
                focal_length=focal_length[i],
                camera_center=(cx, cy),
                camera_rotation=None,
                save_filename=save_filename,
          #      keypoints_2d=input_batch['keypoints_orig'][i].cpu().numpy(),
                faces = self.mano.faces,

            )
            # if i >= (max_save_img - 1):
            #     break