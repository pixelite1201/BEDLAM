import os
import cv2
import torch
import smplx
import pickle
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from . import constants
from . import config
from .constants import NUM_JOINTS_SMPL
from ..dataset.dataset_smpl import DatasetHMR
from ..dataset.mixed_dataset import MixedDataset
from ..utils.train_utils import set_seed
from ..utils.eval_utils import reconstruction_error
from ..utils.image_utils import denormalize_images
from ..utils.renderer_cam import render_image_group
from ..utils.renderer import Renderer
from ..utils.geometry import estimate_translation_fullimg
from ..models.hmr import HMR
from ..models.head.smpl_head import SMPL as SMPL49
from ..losses.losses import HMRLoss


class HMRTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(HMRTrainer, self).__init__()

        self.hparams.update(hparams)
        self.model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )
        self.loss_fn = HMRLoss(hparams=self.hparams)

        self.smplx = smplx.SMPLX(config.SMPLX_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE, create_transl=False, num_betas=11)
        self.smpl = smplx.SMPL(config.SMPL_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE, create_transl=False)
        self.smpl_49 = SMPL49(config.SMPL_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE, create_transl=False)
        # Initialize the training datasets only in training mode
        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()

        self.val_ds = self.val_dataset()
        self.save_itr = 0

        self.smplx2smpl = pickle.load(open(config.SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None], dtype=torch.float32)
        self.downsample_mat_smplx = pickle.load(open(config.DOWNSAMPLE_MAT_SMPLX_PATH, 'rb')).cuda()

        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.smplx.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )
        self.testing_gt_vis = self.hparams.TESTING.GT_VIS
        self.testing_wp_vis = self.hparams.TESTING.WP_VIS
        self.testing_fp_vis = self.hparams.TESTING.FP_VIS
        self.testing_mesh_vis = self.hparams.TESTING.MESH_VIS

        self.training_gt_vis = self.hparams.TRAINING.GT_VIS
        self.training_wp_vis = self.hparams.TRAINING.WP_VIS
        self.training_fp_vis = self.hparams.TRAINING.FP_VIS
        self.training_mesh_vis = self.hparams.TRAINING.MESH_VIS

    def forward(self, x, bbox_center, bbox_scale, img_w, img_h, fl=None):
        return self.model(x, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h, fl=fl) 

    def training_step(self, batch, batch_nb, dataloader_nb=0):
        # for param_group in self.optimizers().param_groups:
        #     print(param_group['lr'])
        # GT data
        images = batch['img']
        gt_betas = batch['betas']
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]
        fl = batch['focal_length']
        gt_pose = batch['pose']
        # Calculate joints and vertices using just 22 pose param for SMPL
        gt_out = self.smpl_49(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3]
        )
        batch['vertices'] = gt_out.vertices
        batch['joints3d'] = gt_out.joints

        # Prediction
        pred = self(images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h, fl=fl)

        # Visualization for debugging
        if self.training_gt_vis:
            self.gt_projection(batch, pred, batch_nb)
        if self.training_wp_vis:
            self.weak_perspective_projection(batch, pred, batch_nb, dataloader_nb)
        if self.training_fp_vis:
            self.perspective_projection(batch, pred, batch_nb)
        if self.training_mesh_vis:
            self.visualize_mesh(batch, pred, batch_nb, dataloader_nb, pred['smplx_vertices'], batch['vertices'])

        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)
        self.log('train_loss', loss, logger=True, sync_dist=True)

        for k, v in loss_dict.items():
            self.log(k, v, logger=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb=0, vis=False, save=True, mesh_save_dir=None):

        images = batch['img']
        batch_size = images.shape[0]
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        dataset_names = batch['dataset_name']
        dataset_index = batch['dataset_index'].detach().cpu().numpy()
        val_dataset_names = self.hparams.DATASET.VAL_DS.split('_')
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]
        J_regressor_batch_smpl = self.J_regressor[None, :].expand(batch['img'].shape[0], -1, -1)
        joint_mapper_h36m = constants.H36M_TO_J14
        joint_mapper_gt = constants.J24_TO_J14

        pred = self(images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
        pred_cam_vertices = pred['vertices']

        if 'rich' in dataset_names[0]:
            # For rich vertices are generated in dataset.py because gender is needed
            gt_cam_vertices = batch['vertices']
            gt_keypoints_3d = batch['joints']
            pred_keypoints_3d = torch.matmul(self.smpl.J_regressor, pred_cam_vertices)
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
        elif 'h36m' in dataset_names[0]:
            gt_cam_vertices = batch['vertices']
            # # Get 14 predicted joints from the mesh
            gt_keypoints_3d = batch['S']
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            gt_keypoints_3d = gt_keypoints_3d - ((gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)
            # # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            # pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            # pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_keypoints_3d = pred_keypoints_3d - ((pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)
        else:
            # For 3dpw vertices are generated in dataset.py because gender is needed
            gt_cam_vertices = batch['vertices']
            # Get 14 predicted joints from the mesh
            gt_keypoints_3d = torch.matmul(J_regressor_batch_smpl, gt_cam_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
        error_verts = torch.sqrt(((pred_cam_vertices - gt_cam_vertices) ** 2).sum(dim=-1)).cpu().numpy()

        # Reconstuction_error (PA-MPJPE)
        r_error, _ = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None
        )
        val_mpjpe = error.mean(-1)
        val_pampjpe = r_error.mean(-1)
        val_pve = error_verts.mean(-1)

        # Visualize results
        if self.testing_gt_vis:
            self.gt_projection(batch, pred, batch_nb)
        if self.testing_mesh_vis:
            self.visualize_mesh(batch, pred, batch_nb, dataloader_nb, pred_cam_vertices, gt_cam_vertices)
        if self.testing_wp_vis:
            self.weak_perspective_projection(batch, pred, batch_nb, dataloader_nb)
        if self.testing_fp_vis:
            self.perspective_projection(batch, pred, batch_nb)

        loss_dict = {}

        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset
            ds_idx = val_dataset_names.index(ds.dataset)
            idxs = np.where(dataset_index == ds_idx)
            loss_dict[ds_name+'_mpjpe'] = list(val_mpjpe[idxs])
            loss_dict[ds_name+'_pampjpe'] = list(val_pampjpe[idxs])
            loss_dict[ds_name+'_pve'] = list(val_pve[idxs])

        return loss_dict

    def validation_epoch_end(self, outputs):
        logger.info(f'***** Epoch {self.current_epoch} *****')
        val_log = {}

        if len(self.val_ds) > 1:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000*np.hstack(np.array([val[ds_name+'_mpjpe'] for x in outputs for val in x])).mean()
                pampjpe = 1000*np.hstack(np.array([val[ds_name+'_pampjpe'] for x in outputs for val in x])).mean()
                pve = 1000*np.hstack(np.array([val[ds_name+'_pve'] for x in outputs for val in x])).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name+'_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name+'_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name+'_PVE: ' + str(pve))
                    val_log[ds_name+'_val_mpjpe'] = mpjpe
                    val_log[ds_name+'_val_pampjpe'] = pampjpe
                    val_log[ds_name+'_val_pve'] = pve
        else:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000*np.hstack(np.array([x[ds_name+'_mpjpe'] for x in outputs])).mean()
                pampjpe = (1000*np.hstack(np.array([x[ds_name+'_pampjpe'] for x in outputs]))).mean()
                pve = (1000*np.hstack(np.array([x[ds_name+'_pve'] for x in outputs]))).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name+'_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name+'_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name+'_PVE: ' + str(pve))

                    val_log[ds_name+'_val_mpjpe'] = mpjpe
                    val_log[ds_name+'_val_pampjpe'] = pampjpe
                    val_log[ds_name+'_val_pve'] = pve

        self.log('val_loss', val_log[self.val_ds[0].dataset+'_val_pampjpe'], logger=True, sync_dist=True)#rank_zero_only=True)
        self.log('val_loss_mpjpe', val_log[self.val_ds[0].dataset+'_val_mpjpe'], logger=True, sync_dist=True)# rank_zero_only=True)

        for k, v in val_log.items():
            self.log(k, v, logger=True, sync_dist=True)# rank_zero_only=True)

    def gt_projection(self, input_batch, output, batch_idx, max_save_img=1):
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images_gt')
        os.makedirs(save_dir, exist_ok=True)
        focal_length = input_batch['focal_length']
        img_size = input_batch['orig_shape'].rot90().T
        gt_keypoints_2d_full_img = input_batch['keypoints_orig']

        gt_out = self.smpl_49(
            betas=input_batch['betas'],
            body_pose=input_batch['pose'][:, 3:],
            global_orient=input_batch['pose'][:, :3]
            )

        gt_model_joints = gt_out.joints.detach()
        gt_vertices = gt_out.vertices

        translation = estimate_translation_fullimg(
            gt_model_joints[:, :].cpu(),
            gt_keypoints_2d_full_img[:, :].cpu(),
            focal_length.cpu(),
            img_size.cpu(),
            use_all_joints=True
        )
        for i in range(len(input_batch['imgname'])):
            imgname = input_batch['imgname'][i]
            cv_img = cv2.imread(imgname)
            cy, cx = input_batch['orig_shape'][i] // 2
            save_filename = os.path.join(save_dir, f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')

            rendered_img = render_image_group(
                image=cv_img,
                camera_translation=translation[i],
                vertices=gt_vertices[i],
                focal_length=focal_length[i],
                camera_center=(cx, cy),
                camera_rotation=None,
                save_filename=save_filename,
                keypoints_2d=input_batch['keypoints_orig'][i].cpu().numpy(),
                faces=self.smpl.faces,
            )
            if i >= (max_save_img - 1):
                break

    def perspective_projection(self, input_batch, output, batch_idx, max_save_img=1):
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images_cliff')
        os.makedirs(save_dir, exist_ok=True)

        translation = output['pred_cam_t'].detach()
        vertices = output['vertices'].detach()

        for i in range(len(input_batch['imgname'])):
            cy, cx = input_batch['orig_shape'][i] // 2
            img_h, img_w = cy*2, cx*2
            imgname = input_batch['imgname'][i]
            save_filename = os.path.join(save_dir, f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')
            focal_length_ = (img_w * img_w + img_h * img_h) ** 0.5  # Assumed fl
            focal_length = (focal_length_, focal_length_)

            rendered_img = render_image_group(
                image=cv2.imread(imgname),
                camera_translation=translation[i],
                vertices=vertices[i],
                focal_length=focal_length,
                camera_center=(cx, cy),
                camera_rotation=None,
                save_filename=save_filename,
                faces=self.smpl.faces,
                keypoints_2d=input_batch['keypoints_orig'][i].cpu().numpy(),
            )
            if i >= (max_save_img - 1):
                break

    def visualize_mesh(self, input_batch, output, batch_idx, dataloader_nb, pc, gc, max_save=1):
        import trimesh
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
        os.makedirs(save_dir, exist_ok=True)
        
        pred_vertices = pc.detach().cpu().numpy()
        gt_vertices = gc.detach().cpu().numpy()
        for i in range(pred_vertices.shape[0]):
            imgname = input_batch['imgname'][i].split('/')[-1]
            gt = trimesh.Trimesh(vertices=gt_vertices[i]*np.array([1, -1, -1]), faces=self.smpl.faces, process=False)
            gt.visual.face_colors = [200, 200, 250, 100]
            gt.visual.vertex_colors = [200, 200, 250, 100]

            pred = trimesh.Trimesh(vertices=pred_vertices[i]*np.array([1, -1, -1]), faces=self.smplx.faces, process=False)
            save_filename = os.path.join(save_dir, f'{self.current_epoch:04d}_{dataloader_nb:02d}_'
                                f'{batch_idx:05d}_{i:02d}_{os.path.basename(imgname)}')
            gt.export(save_filename+str(i)+'_gt.obj')
            pred.export(save_filename+str(i)+'_pred.obj')   
            if i >= (max_save - 1):
                break

    def weak_perspective_projection(self, input_batch, output, batch_idx, dataloader_nb):
        pred_vertices = output['vertices'].detach()
        images = input_batch['img']
        images = denormalize_images(images)
        pred_cam_t = output['pred_cam_t'].detach()
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')

        os.makedirs(save_dir, exist_ok=True)
        for i, _ in enumerate(pred_vertices):
            if i > 1:
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

    def test_step(self, batch, batch_nb, dataloader_nb=0):
        return self.validation_step(batch, batch_nb, dataloader_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        if self.hparams.OPTIMIZER.TYPE == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.OPTIMIZER.LR, momentum=0.9)
        else:
            return torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.OPTIMIZER.LR,
                weight_decay=self.hparams.OPTIMIZER.WD,
                # eps=1.0,
                # amsgrad=True
            )

    def train_dataset(self):
        train_ds = MixedDataset(
            options=self.hparams.DATASET,
            is_train=True
        )
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
            drop_last=True
        )

    def val_dataset(self):
        datasets = self.hparams.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_datasets = []
        for dataset_name in datasets:
            val_datasets.append(
                DatasetHMR(
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
