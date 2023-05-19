import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from .head.smplx_head_cam_full import SMPLXHeadCamFull
from ..utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d
from ..utils.abs2rel import pose_abs2rel
from ..core import constants


class HMRX(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            focal_length=5000.,
            img_res=224,
            pretrained_ckpt=None,
            use_cam=False,
            p=0.0,
            config=None,
            hparams=None,
            num_input_features=1440,
    ):
        super(HMRX, self).__init__()
        self.hparams = hparams
        npose = (2 + 16) * 6
        self.fc1 = nn.Linear(num_input_features + npose, 256)
        self.fc1_l = nn.Linear(num_input_features + npose, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_l = nn.Linear(256, 128)
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()
        self.relu = nn.ReLU()
        self.decpose = nn.Linear(128, 12)
        self.decpose_l = nn.Linear(128, 12)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decpose_l.weight, gain=0.01)
        self.lelbow_index_6d = constants.LELBOW
        self.relbow_index_6d = constants.RELBOW
        self.lwrist_index_6d = constants.LWRIST
        self.rwrist_index_6d = constants.RWRIST
        self.smplx_head = SMPLXHeadCamFull()

    def forward(
            self,
            body_feat,
            lhand_feat,
            rhand_feat,
            body_pose,
            body_shape,
            body_cam,
            lhand_pose,
            rhand_pose,
            bbox_center, 
            bbox_scale, 
            img_w, 
            img_h, 
            fl=None):
        batch_size = body_feat.shape[0]

        if fl is not None:
            focal_length = fl
        else:
            focal_length = (img_w * img_w + img_h * img_h) ** 0.5
            focal_length = focal_length.repeat(2).view(batch_size,2)
        cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
        cam_intrinsics[:, 0, 0]  = focal_length[:, 0]
        cam_intrinsics[:, 1, 1]  = focal_length[:, 1]
        cam_intrinsics[:, 0, 2] = img_w/2.
        cam_intrinsics[:, 1, 2] = img_h/2.

        right_wrist = pose_abs2rel(body_pose[:,[0]], body_pose[:,1:], rhand_pose[:,[0]], abs_joint='right_wrist')
        left_wrist = pose_abs2rel(body_pose[:,[0]], body_pose[:,1:], lhand_pose[:,[0]], abs_joint='left_wrist')
        rhand_pose[:,0] = right_wrist
        lhand_pose[:,0] = left_wrist
        body_pose_6d = rotmat_to_rot6d(body_pose)
        #For right hand
        ind1 = self.relbow_index_6d
        ind2 = self.rwrist_index_6d
        indices=np.concatenate((np.arange(ind1*6,ind1*6+6),np.arange(ind2*6,ind2*6+6)))

        rhand_pose_sub = rhand_pose
        rhand_pose_6d_sub  = rotmat_to_rot6d(rhand_pose_sub)
        features_r = torch.cat((body_feat, rhand_feat),-1)   

        body_pose_r = body_pose[:, [ind1, ind2]]
        body_pose_6d_r = rotmat_to_rot6d(body_pose_r)


        xc = torch.cat([features_r, body_pose_6d_r, rhand_pose_6d_sub],1)
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.relu(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        body_pose_6d[:, indices] = self.decpose(xc) + body_pose_6d_r

        #For left hand
        ind1 = self.lelbow_index_6d
        ind2 = self.lwrist_index_6d
        indices=np.concatenate((np.arange(ind1*6,ind1*6+6),np.arange(ind2*6,ind2*6+6)))

        lhand_pose_sub = lhand_pose
        lhand_pose_6d_sub = rotmat_to_rot6d(lhand_pose_sub)
        features_l = torch.cat((body_feat, lhand_feat),-1)
       
        body_pose_l = body_pose[:, [ind1, ind2]]  ###########
        body_pose_6d_l = rotmat_to_rot6d(body_pose_l)

        xc = torch.cat([features_l, body_pose_6d_l, lhand_pose_6d_sub], 1)
        xc = self.fc1_l(xc)
        xc = self.drop1(xc)
        xc = self.relu(xc)
        xc = self.fc2_l(xc)
        xc = self.relu(xc)
        body_pose_6d[:, indices] = self.decpose_l(xc) + body_pose_6d_l  #############

        #Todo Convert from 6d
        body_pose = rot6d_to_rotmat(body_pose_6d).view(batch_size, -1, 3, 3)
        smplx_output = self.smplx_head(body_pose=body_pose, lhand_pose = lhand_pose[:,1:], rhand_pose=rhand_pose[:,1:], 
        shape=body_shape, cam=body_cam, cam_intrinsics=cam_intrinsics,bbox_scale=bbox_scale,bbox_center=bbox_center, img_w=img_w, img_h=img_h, normalize_joints2d=False)

        smplx_output['pred_pose']=body_pose
        smplx_output['pred_lhand_pose']=lhand_pose[:, 1:]
        smplx_output['pred_rhand_pose']=rhand_pose[:, 1:]
        smplx_output['pred_shape']=body_shape
        smplx_output['pred_cam']=body_cam

        return smplx_output
