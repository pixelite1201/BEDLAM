import torch
import numpy as np
import torch.nn as nn

from ...core.config import SMPL_MEAN_PARAMS
from ...core.constants import NUM_JOINTS_HAND
from ...utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d, batch_rodrigues

BN_MOMENTUM = 0.1


class HMRHand(nn.Module):
    def __init__(
            self,
            mean_pose_params,
            num_input_features,
    ):
        super(HMRHand, self).__init__()

        self.npose = NUM_JOINTS_HAND * 6
        self.num_input_features = num_input_features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(num_input_features, 1024)
        self.dechand = nn.Linear(1024, NUM_JOINTS_HAND * 6)
        self.deccam = nn.Linear(1024, 3)
        self.decshape = nn.Linear(1024, 10)

        nn.init.xavier_uniform_(self.dechand.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)

        mean_params = np.load(SMPL_MEAN_PARAMS)

        rhand_mean_rot6d = rotmat_to_rot6d(batch_rodrigues(mean_pose_params.view(-1, 3)).view([-1, 3, 3]))
        init_rhand = rhand_mean_rot6d.reshape(-1).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_rhand', init_rhand)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_shape', init_shape)

    def forward(
            self,
            features,
            n_iter=3
    ):

        batch_size = features.shape[0]
        init_rhand = self.init_rhand.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)

        xf = self.avgpool(features)
        xf = xf.view(xf.size(0), -1)
        pred_pose = init_rhand
        pred_cam = init_cam
        pred_shape = init_shape

        xc = self.fc1(xf)
        pred_pose = self.dechand(xc) + pred_pose
        pred_cam = self.deccam(xc) + pred_cam
        pred_shape = self.decshape(xc) + pred_shape

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, NUM_JOINTS_HAND, 3, 3)

        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_pose_6d': pred_pose,
            'pred_shape': pred_shape,
            'hand_feat': xf,
            'hand_feat2': xc,

        }

        return output
