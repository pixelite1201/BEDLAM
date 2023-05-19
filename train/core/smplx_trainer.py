
import os
import cv2
import json
import torch
import pickle
import smplx
import numpy as np
from loguru import logger

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Normalize

from . import config
from . import constants
from ..models.head.smplx_head_cam_full import SMPLX
from ..dataset.datasetx import DatasetHMR
from ..utils.abs2rel import pose_abs2rel

from ..utils.eval_utils import reconstruction_error
from ..utils.image_utils import denormalize_images
from ..utils.image_utils import read_img
from ..utils.renderer_cam import render_image_group
from ..utils.renderer import Renderer
from ..models.hmr import HMR
from ..models.hand import Hand
from kornia.geometry.transform.imgwarp import (
    get_perspective_transform, warp_affine
)
from ..models.hmrx import HMRX
from ..losses.lossesx import HMRXLoss
SCALE_FACTOR_HAND_BBOX = 3.0


def get_bbox_valid(joints, img_height, img_width, rescale):
    #Get bbox using keypoints
    img_width = img_width.unsqueeze(-1).unsqueeze(-1)
    img_height = img_height.unsqueeze(-1).unsqueeze(-1)

    min_coords, _ = torch.min(joints, dim=1)
    xmin, ymin = min_coords[:, 0], min_coords[:, 1]
    max_coords, _ = torch.max(joints, dim=1)
    xmax, ymax = max_coords[:, 0], max_coords[:, 1]
    center = torch.stack([xmax + xmin, ymax + ymin], dim=-1) * 0.5
    width = (xmax - xmin)
    height = (ymax - ymin)
    # Convert the bounding box to a square box
    scale = torch.max(width, height).unsqueeze(-1)
    scale *= rescale
    return center, scale


def crop_tensor(image, center, bbox_size, crop_size, interpolation='bilinear', align_corners=False):

    dtype = image.dtype
    device = image.device
    batch_size = image.shape[0]
    # points: top-left, top-right, bottom-right, bottom-left    
    src_pts = torch.zeros([4, 2], dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    src_pts[:, 0, :] = center - bbox_size * 0.5  # / (self.crop_size - 1)
    src_pts[:, 1, 0] = center[:, 0] + bbox_size[:, 0] * 0.5
    src_pts[:, 1, 1] = center[:, 1] - bbox_size[:, 0] * 0.5
    src_pts[:, 2, :] = center + bbox_size * 0.5
    src_pts[:, 3, 0] = center[:, 0] - bbox_size[:, 0] * 0.5
    src_pts[:, 3, 1] = center[:, 1] + bbox_size[:, 0] * 0.5

    DST_PTS = torch.tensor([[
        [0, 0],
        [crop_size - 1, 0],
        [crop_size - 1, crop_size - 1],
        [0, crop_size - 1],
    ]], dtype=dtype, device=device).expand(batch_size, -1, -1)
    # estimate transformation between points
    dst_trans_src = get_perspective_transform(src_pts, DST_PTS)

    # warp images 
    cropped_image = warp_affine(
        image, dst_trans_src[:, :2, :], (crop_size, crop_size),
        mode=interpolation, align_corners=align_corners)

    tform = torch.transpose(dst_trans_src, 2, 1)
    return cropped_image, tform


class SMPLXTrainer(pl.LightningModule):
    def __init__(self, hparams, config_tune=None):
        super(SMPLXTrainer, self).__init__()
        self.hparams.update(hparams)

        self.body_model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )
        self.hand_model = Hand(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )
        for param in self.body_model.parameters():
            param.requires_grad = False
        for param in self.hand_model.parameters():
            param.requires_grad = False

        self.fullbody_model = HMRX(backbone=self.hparams.MODEL.BACKBONE,
                                   img_res=self.hparams.DATASET.IMG_RES,
                                   pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
                                   hparams=self.hparams)

        self.loss_fn = HMRXLoss(hparams=self.hparams)

        self.smplx_rotmat = SMPLX(config.SMPLX_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE,
                                  use_pca=False, flat_hand_mean=True)
        self.smplx = smplx.SMPLX(config.SMPLX_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE,
                                 create_transl=False, use_pca=False, flat_hand_mean=True, num_betas=11)
        self.smpl = smplx.SMPL(config.SMPL_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE,
                               create_transl=False)

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.smplx.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR)

        self.smplx2smpl = pickle.load(open(config.SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None], dtype=torch.float32)
        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )
        self.J_regressor_smplx = torch.mm(self.J_regressor, self.smplx2smpl[0]).cuda()
        self.testing_gt_vis = self.hparams.TESTING.GT_VIS
        self.testing_wp_vis = self.hparams.TESTING.WP_VIS
        self.testing_fp_vis = self.hparams.TESTING.FP_VIS
        self.testing_mesh_vis = self.hparams.TESTING.MESH_VIS

        self.training_gt_vis = self.hparams.TRAINING.GT_VIS
        self.training_wp_vis = self.hparams.TRAINING.WP_VIS
        self.training_fp_vis = self.hparams.TRAINING.FP_VIS

        self.flip_vector = torch.ones((1, 9), dtype=torch.float32)
        self.flip_vector[:, [1, 2, 3, 6]] *= -1
        self.flip_vector = self.flip_vector.reshape(1, 3, 3).cuda()

        self.val_ds = self.val_dataset()
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def forward(self, body_feat, lhand_feat, rhand_feat, body_pose, body_shape, body_cam, left_hand_pose, right_hand_pose, bbox_center, bbox_scale, img_w, img_h, fl=None):
        return self.fullbody_model(body_feat, lhand_feat, rhand_feat, body_pose, body_shape, body_cam, left_hand_pose, right_hand_pose, bbox_center, bbox_scale, img_w, img_h, fl=fl)

    def training_step(self, batch, batch_nb, dataloader_nb=0):
        images = batch['img']
        gt_betas = batch['betas']
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]
        fl = batch['focal_length']
        self.body_model.eval()
        self.hand_model.eval()

        with torch.no_grad():
            body_pred = self.body_model(images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h, fl=fl)
            right_hand_crop = batch['rhand_img']
            right_hand_pred = self.hand_model(right_hand_crop)
            # Flip left hand image before feedint to hand network
            left_hand_crop = torch.flip(batch['lhand_img'], [3])
            left_hand_pred = self.hand_model(left_hand_crop)
            # Flip predicted right hand pose to left hand 
            left_hand_pred['pred_pose'] = left_hand_pred['pred_pose'] * self.flip_vector.unsqueeze(0)

        full_body_pred = self(body_pred['body_feat'], left_hand_pred['hand_feat'], right_hand_pred['hand_feat'], body_pred['pred_pose'], body_pred['pred_shape'], body_pred['pred_cam'],left_hand_pred['pred_pose'], right_hand_pred['pred_pose'],bbox_center, bbox_scale, img_w, img_h,fl=fl)
  
        gt_pose = batch['pose']
        lhand_pose = batch['lhand_pose']
        rhand_pose = batch['rhand_pose']

        gt_out = self.smplx(
            betas=gt_betas.float(),
            body_pose=gt_pose[:, 3:].float(),
            left_hand_pose=lhand_pose.float(),
            right_hand_pose=rhand_pose.float(),
            global_orient=gt_pose[:, :3].float()
        )

        batch['vertices'] = gt_out.vertices
        batch['joints3d'] = gt_out.joints
        loss, loss_dict = self.loss_fn(pred=full_body_pred, gt=batch)
        self.log('train_loss', loss, logger=True, sync_dist=True)

        if self.training_gt_vis:
            self.gt_projection(batch, full_body_pred, batch_nb)#,x=0)
        if self.training_wp_vis:
            self.weak_perspective_projection(batch, full_body_pred, batch_nb, dataloader_nb)
        if self.training_fp_vis:
            self.perspective_projection(batch, full_body_pred, batch_nb, fl=fl)

        for k, v in loss_dict.items():
            self.log(k, v, logger=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb=0, vis=False, save=True, mesh_save_dir=None):
  
        images = batch['img']
        batch_size = images.shape[0]
        dataset_names = batch['dataset_name']
        dataset_index = batch['dataset_index'].detach().cpu().numpy()
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        val_dataset_names = self.hparams.DATASET.VAL_DS.split('_')
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]

        body_pred = self.body_model(images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)

        lhand_joints = body_pred['joints2d'][:, 25:40]
        rhand_joints = body_pred['joints2d'][:, 40:55]
        center_r, scale_r = get_bbox_valid(rhand_joints, img_h, img_w, SCALE_FACTOR_HAND_BBOX)
        right_hand_crop, _ = crop_tensor(batch['img_full'], center_r, scale_r, 224)
        right_hand_crop = self.normalize_img(right_hand_crop)
        right_hand_pred = self.hand_model(right_hand_crop)
        # Flip left hand image before feedint to hand network
        center_l, scale_l = get_bbox_valid(lhand_joints, img_h, img_w, SCALE_FACTOR_HAND_BBOX)
        left_hand_crop, _ = crop_tensor(batch['img_full'], center_l, scale_l, 224)
        left_hand_crop = self.normalize_img(left_hand_crop)
        left_hand_crop = torch.flip(left_hand_crop, [3])
        left_hand_pred = self.hand_model(left_hand_crop)
        # Flip predicted right hand pose to left hand 
        left_hand_pred['pred_pose'] = left_hand_pred['pred_pose'] * self.flip_vector.unsqueeze(0)

        full_body_pred = self(body_pred['body_feat'], left_hand_pred['hand_feat'],
                              right_hand_pred['hand_feat'], body_pred['pred_pose'],
                              body_pred['pred_shape'], body_pred['pred_cam'],
                              left_hand_pred['pred_pose'], right_hand_pred['pred_pose'],
                              bbox_center, bbox_scale, img_w, img_h)

        if self.testing_gt_vis:
            self.gt_projection(batch, full_body_pred, batch_nb)
        if self.testing_wp_vis:
            self.weak_perspective_projection(batch, full_body_pred, batch_nb, dataloader_nb)
        if self.testing_fp_vis:
            self.perspective_projection(batch, full_body_pred, batch_nb)


        gt_pose = batch['pose']
        lhand_pose = batch['lhand_pose']
        rhand_pose = batch['rhand_pose']
        gt_betas = batch['betas']

        gt_out_cam = self.smplx(
            betas=gt_betas.float(),
            body_pose=gt_pose[:, 3:].float(),
            left_hand_pose=lhand_pose.float(),
            right_hand_pose=rhand_pose.float(),
            global_orient=gt_pose[:, :3].float()
        )
        gt_cam_vertices = gt_out_cam.vertices
        gt_joints = gt_out_cam.joints
        if self.testing_mesh_vis:
            self.visualize_mesh(batch, full_body_pred, batch_nb, dataloader_nb, 
                                full_body_pred['vertices'], gt_cam_vertices, x=0)
            
        pred_cam_vertices = full_body_pred['vertices']
        pred_joints = full_body_pred['joints3d']
        # Calculate v2v, hand joints, body joints error
        error_verts = torch.sqrt(((pred_cam_vertices - gt_cam_vertices) ** 2).sum(dim=-1)).cpu().numpy()
        error_body_joints = torch.sqrt(((pred_joints[:, :24] - gt_joints[:, :24]) ** 2).sum(dim=-1)).cpu().numpy()
        r_error, _ = reconstruction_error(pred_joints[:, :24].cpu().numpy(),
                                            gt_joints[:, :24].cpu().numpy(),
                                            reduction=None)
        pred_rhand_joints = pred_joints[:, [21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]] - pred_joints[:, [21]]
        gt_rhand_joints = gt_joints[:, [21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]] - gt_joints[:, [21]]
        pred_lhand_joints = pred_joints[:, [20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]] - pred_joints[:, [20]]
        gt_lhand_joints = gt_joints[:, [20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]] - gt_joints[:, [20]]
    
        error_lhand_joints = torch.sqrt(((pred_lhand_joints - gt_lhand_joints) ** 2).sum(dim=-1)).cpu().numpy()
        error_rhand_joints = torch.sqrt(((pred_rhand_joints - gt_rhand_joints) ** 2).sum(dim=-1)).cpu().numpy()

        val_mpjpe = error_body_joints.mean(-1)
        val_pampjpe = r_error.mean(-1)      
        val_pve = error_verts.mean(-1)

        val_lhand = error_lhand_joints.mean(-1)
        val_rhand = error_rhand_joints.mean(-1)
        loss_dict = {}

        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset
            ds_idx = val_dataset_names.index(ds.dataset)
            idxs = np.where(dataset_index == ds_idx)
            loss_dict[ds_name + '_mpjpe'] = list(val_mpjpe[idxs])
            loss_dict[ds_name + '_pampjpe'] = list(val_pampjpe[idxs])
            loss_dict[ds_name + '_lhand'] = list(val_lhand[idxs])
            loss_dict[ds_name + '_rhand'] = list(val_rhand[idxs])
            loss_dict[ds_name + '_pve'] = list(val_pve[idxs])

        return loss_dict
   
    def validation_epoch_end(self, outputs):
        logger.info(f'***** Epoch {self.current_epoch} *****')
        val_log = {}
        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset

            if len(self.val_ds) > 1:
                mpjpe = 1000 * np.hstack(np.array([val[ds_name + '_mpjpe'] for x in outputs for val in x])).mean()
                pampjpe = (1000 * np.hstack(np.array([val[ds_name + '_pampjpe'] for x in outputs for val in x]))).mean()
                lhand = 1000 * np.hstack(np.array([val[ds_name + '_lhand'] for x in outputs for val in x])).mean()
                rhand = (1000 * np.hstack(np.array([val[ds_name + '_rhand'] for x in outputs for val in x]))).mean()
                pve = (1000 * np.hstack(np.array([val[ds_name + '_pve'] for x in outputs for val in x]))).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_LHAND: ' + str(lhand))
                    logger.info(ds_name + '_RHAND: ' + str(rhand))
                    logger.info(ds_name + '_PVE: ' + str(pve))

                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_lhand'] = lhand
                    val_log[ds_name + '_val_rhand'] = rhand
                    val_log[ds_name + '_val_pve'] = pve
            else:
                mpjpe = 1000 * np.hstack(np.array([x[ds_name + '_mpjpe'] for x in outputs])).mean()
                pampjpe = (1000 * np.hstack(np.array([x[ds_name + '_pampjpe'] for x in outputs]))).mean()
                lhand = 1000 * np.hstack(np.array([x[ds_name + '_lhand'] for x in outputs])).mean()
                rhand = (1000 * np.hstack(np.array([x[ds_name + '_rhand'] for x in outputs]))).mean()
                pve = (1000 * np.hstack(np.array([x[ds_name + '_pve'] for x in outputs]))).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_LHAND: ' + str(lhand))
                    logger.info(ds_name + '_RHAND: ' + str(rhand))
                    logger.info(ds_name + '_PVE: ' + str(pve))

                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_lhand'] = lhand
                    val_log[ds_name + '_val_rhand'] = rhand
                    val_log[ds_name + '_val_pve'] = pve


        self.log('val_loss', val_log[self.val_ds[0].dataset + '_val_pampjpe'], logger=True, sync_dist=True, rank_zero_only=True)
        self.log('val_loss_mpjpe', val_log[self.val_ds[0].dataset + '_val_mpjpe'], logger=True, sync_dist=True, rank_zero_only=True)

        for k, v in val_log.items():
            self.log(k, v, logger=True, sync_dist=True, rank_zero_only=True)

    def gt_projection(self, input_batch, output, batch_idx, max_save_img=1):
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images_gt')
        os.makedirs(save_dir, exist_ok=True)

        focal_length = input_batch['focal_length']

        gt_out = self.smplx(betas=input_batch['betas'],
                            body_pose=input_batch['pose'][:, 3:],
                            global_orient=input_batch['pose'][:, :3]
                            )
        gt_vertices = gt_out.vertices

        translation = input_batch['translation'][:,:3]

        for i in range(len(input_batch['imgname'])):
            dataset_name = input_batch['dataset_name'][i]
            imgname = input_batch['imgname'][i]
            cv_img = read_img(imgname)
            if 'closeup' in dataset_name:
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
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
                faces=self.smplx.faces,

            )
            if i >= (max_save_img - 1):
                break

    def perspective_projection(self, input_batch, output, batch_idx, fl=None, max_save_img=1):
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images_cliff')
        os.makedirs(save_dir, exist_ok=True)

        img_size = input_batch['orig_shape'].rot90().T

        translation = output['pred_cam_t'].detach()
        vertices = output['vertices'].detach()

        for i in range(len(input_batch['imgname'])):
            cy, cx = input_batch['orig_shape'][i] // 2
            img_h, img_w = cy * 2, cx * 2
            imgname = input_batch['imgname'][i]
            save_filename = os.path.join(save_dir, f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')
            if fl is None:
                focal_length_ = (img_w * img_w + img_h * img_h) ** 0.5  
                focal_length = (focal_length_,focal_length_)
            else:
                focal_length = fl[:,0]
            rendered_img = render_image_group(
                image=read_img(imgname),
                camera_translation=translation[i],
                vertices=vertices[i],
                focal_length=focal_length,
                camera_center=(cx, cy),
                camera_rotation=None,
                save_filename=save_filename,
                faces = self.smplx.faces,
                # keypoints_2d=input_batch['keypoints_orig'][i].cpu().numpy()[25:],
            )
            if i >= (max_save_img - 1):
                break

    def visualize_mesh(self, input_batch, output, batch_idx, dataloader_nb,pc, gc, x=-1):
        # images = input_batch['img']
        import trimesh
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
        os.makedirs(save_dir, exist_ok=True)
        
        pred_vertices = pc.detach().cpu().numpy()
        gt_vertices = gc.detach().cpu().numpy()
        if x!=-1:
            i=x
            imgname = input_batch['imgname'][i].split('/')[-1]
            gt = trimesh.Trimesh(vertices=gt_vertices[i]*np.array([1,-1,-1]),faces=self.smplx.faces, process=False)
            gt.visual.face_colors = [200, 200, 250, 100]
            gt.visual.vertex_colors = [200, 200, 250, 100]

            pred = trimesh.Trimesh(vertices=pred_vertices[i]*np.array([1,-1,-1]),faces=self.smplx.faces,process=False)
            save_filename = os.path.join(save_dir, f'{self.current_epoch:04d}_{dataloader_nb:02d}_'
                                f'{batch_idx:05d}_{i:02d}_{os.path.basename(imgname)}')
            gt.export(save_filename+str(i)+'_gt.obj')
            pred.export(save_filename+str(i)+'_pred.obj')
        else:
            for i in range(pred_vertices.shape[0]):
                imgname = input_batch['imgname'][i].split('/')[-1]
                gt = trimesh.Trimesh(vertices=gt_vertices[i]*np.array([1,-1,-1]),faces=self.smpl.faces, process=False)
                gt.visual.face_colors = [200, 200, 250, 100]
                gt.visual.vertex_colors = [200, 200, 250, 100]

                pred = trimesh.Trimesh(vertices=pred_vertices[i]*np.array([1,-1,-1]),faces=self.smplx.faces,process=False)
                save_filename = os.path.join(save_dir, f'{self.current_epoch:04d}_{dataloader_nb:02d}_'
                                    f'{batch_idx:05d}_{i:02d}_{os.path.basename(imgname)}')
                gt.export(save_filename+str(i)+'_gt.obj')
                pred.export(save_filename+str(i)+'_pred.obj')       
                    
    def weak_perspective_projection(self, input_batch, output, batch_idx, dataloader_nb):
        pred_vertices = output['vertices'].detach()
        
        gtkps = input_batch['keypoints'].clone()
        pred_kp_2d = output['joints2d'].detach().clone()

        images = input_batch['img']
        images = denormalize_images(images)
        pred_cam_t = output['pred_cam_t'].detach()
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
        os.makedirs(save_dir, exist_ok=True)
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
                                    
            # save_filename2 = os.path.join(save_dir, f'result_{self.current_epoch:04d}_'
            #                                     f'{dataloader_nb:02d}_{batch_idx:05d}_{i}_{1}.jpg')
            # from matplotlib import pyplot as plt
            # temp = images[i].cpu().numpy().transpose(1, 2, 0) * 255
            # temp = np.clip(temp, 0, 255).astype(np.uint8)
    
            # fig = plt.figure(dpi=300)
            # ax = fig.add_subplot(111)
            # ax.imshow(temp)
            # pred_kp_2d[i] = (pred_kp_2d[i]+1)*224/2
            # for k in range(len(pred_kp_2d[i])):
            #     ax.scatter(pred_kp_2d[i][k, 0].detach().cpu().numpy(), pred_kp_2d[i][k, 1].detach().cpu().numpy(), s=1.0)
            # plt.savefig(save_filename2)
            # plt.close()
            # fig = plt.figure(dpi=300)
            # ax = fig.add_subplot(111)
            # ax.imshow(temp)
   
            # gtkps[i] = (gtkps[i]+1)*224/2
            # for k in range(len(gtkps[i])):
            #     ax.scatter(gtkps[i][k, 0].detach().cpu().numpy(), gtkps[i][k, 1].detach().cpu().numpy(), s=1.0)
            # plt.savefig(save_filename)
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
        options = self.hparams.DATASET
        dataset_names = options.DATASETS_AND_RATIOS.split('_')
        dataset_list = [DatasetHMR(options, ds) for ds in dataset_names]
        train_ds = ConcatDataset(dataset_list)

        return train_ds

    def train_dataloader(self):
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