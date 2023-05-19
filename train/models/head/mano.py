import torch
import torch.nn as nn

from smplx import MANOLayer
from smplx.utils import SMPLXOutput

from ...core import config
from ...utils.geometry import perspective_projection, convert_weak_perspective_to_perspective, batch_rodrigues

from dataclasses import dataclass
from typing import Optional

# Modified from https://github.com/HongwenZhang/PyMAF-X
class MANO(MANOLayer):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if 'pose2rot' not in kwargs:
            kwargs['pose2rot'] = True
        pose_keys = ['global_orient', 'right_hand_pose']
        batch_size = kwargs['global_orient'].shape[0]
        if kwargs['pose2rot']:
            for key in pose_keys:
                if key in kwargs:
                    kwargs[key] = batch_rodrigues(kwargs[key].contiguous().view(-1, 3)).view([batch_size, -1, 3, 3])
        kwargs['hand_pose'] = kwargs.pop('right_hand_pose')
        mano_output = super().forward(*args, **kwargs)
        return mano_output

class MANOHead(nn.Module):
    def __init__(self, focal_length=5000., img_res=224):
        super(MANOHead, self).__init__()    
        self.mano = MANO(model_path=config.MANO_MODEL_DIR, use_pca=False, is_rhand=True)
        self.add_module('mano', self.mano)
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, rotmat, shape, cam=None, normalize_joints2d=False):

        smpl_output = self.mano(
            right_hand_pose=rotmat[:, 1:].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            betas=shape,
            pose2rot=False,
        )

        output = {
            'vertices': smpl_output.vertices,
            'joints3d': smpl_output.joints,
        }

        if cam is not None:
            joints3d = output['joints3d']
            batch_size = joints3d.shape[0]
            device = joints3d.device
            cam_t = convert_weak_perspective_to_perspective(
                cam,
                focal_length=self.focal_length,
                img_res=self.img_res,
            )
            joints2d = perspective_projection(
                joints3d,
                rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                translation=cam_t,
                focal_length=self.focal_length,
                camera_center=torch.zeros(batch_size, 2, device=device)
            )
            if normalize_joints2d:
                # Normalize keypoints to [-1,1]
                joints2d = joints2d / (self.img_res / 2.) 

            output['joints2d'] = joints2d
            output['pred_cam_t'] = cam_t

        return output
