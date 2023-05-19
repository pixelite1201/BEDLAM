import torch
import torch.nn as nn

from smplx import SMPLXLayer as SMPLX_
from smplx.utils import SMPLXOutput
from ...core import config


class SMPLX(SMPLX_):
    def __init__(self, *args, **kwargs):
        super(SMPLX, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        smplx_output = super(SMPLX, self).forward(*args, **kwargs)
        output = SMPLXOutput(vertices=smplx_output.vertices,
                             global_orient=smplx_output.global_orient,
                             body_pose=smplx_output.body_pose,
                             joints=smplx_output.joints,
                             betas=smplx_output.betas,
                             full_pose=smplx_output.full_pose)
        return output


class SMPLXHeadCamFull(nn.Module):
    def __init__(self, focal_length=5000., img_res=224):
        super(SMPLXHeadCamFull, self).__init__()    
        self.smplx = SMPLX(config.SMPLX_MODEL_DIR, flat_hand_mean=True, num_betas=11)
        self.add_module('smplx', self.smplx)
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, body_pose, lhand_pose, rhand_pose, shape, cam, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h, normalize_joints2d=False):

        smpl_output = self.smplx(
            betas=shape,
            body_pose=body_pose[:, 1:].contiguous(),
            global_orient=body_pose[:, 0].unsqueeze(1).contiguous(),
            left_hand_pose=lhand_pose.contiguous(),
            right_hand_pose=rhand_pose.contiguous(),
            pose2rot=False,
        )

        output = {
            'vertices': smpl_output.vertices,
            'joints3d': smpl_output.joints,
        }
        joints3d = output['joints3d']
        batch_size = joints3d.shape[0]
        device = joints3d.device

        cam_t = convert_pare_to_full_img_cam(
            pare_cam=cam,
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
            crop_res=self.img_res,
        )

        joints2d = perspective_projection(
            joints3d,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )

        output['joints2d'] = joints2d
        output['pred_cam_t'] = cam_t

        return output


def perspective_projection(points, rotation, translation, cam_intrinsics):
    K = cam_intrinsics
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:,:,-1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points.float())
    return projected_points[:, :, :-1]


def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t