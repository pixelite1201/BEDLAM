import torch
import torch.nn as nn

from .smplx_local import SMPLX
from ...core import config
from ...core.constants import NUM_JOINTS_SMPLX
from ...utils.geometry import perspective_projection, convert_weak_perspective_to_perspective


class SMPLXHead(nn.Module):
    def __init__(self, focal_length=5000., img_res=224):

        super(SMPLXHead, self).__init__()
        self.smplx = SMPLX(config.SMPLX_MODEL_DIR, num_betas=11)
        self.add_module('smplx', self.smplx)
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, rotmat, shape, cam=None, normalize_joints2d=False):

        smpl_output = self.smplx(
            betas=shape,
            body_pose=rotmat[:, 1:NUM_JOINTS_SMPLX].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
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
                joints2d = joints2d / (self.img_res / 2.) 

            output['joints2d'] = joints2d
            output['pred_cam_t'] = cam_t

        return output
