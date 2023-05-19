import torch
import torch.nn as nn

from ..core.constants import NUM_JOINTS_HAND
from ..utils.geometry import batch_rodrigues
from ..utils.eval_utils import compute_similarity_transform_batch


class HandLoss(nn.Module):
    def __init__(
            self,
            hparams=None,
    ):
        super(HandLoss, self).__init__()
        self.criterion_mse = nn.MSELoss()
        self.criterion_mse_noreduce = nn.MSELoss(reduction='none')
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l1_noreduce = nn.L1Loss(reduction='none')
        self.hparams = hparams

        self.loss_weight = self.hparams.MODEL.LOSS_WEIGHT
        self.shape_loss_weight = self.hparams.MODEL.SHAPE_LOSS_WEIGHT
        self.pose_loss_weight = self.hparams.MODEL.POSE_LOSS_WEIGHT
        self.joint_loss_weight = self.hparams.MODEL.JOINT_LOSS_WEIGHT
        self.keypoint_loss_weight_2d = self.hparams.MODEL.KEYPOINT_LOSS_WEIGHT
        self.beta_loss_weight = self.hparams.MODEL.BETA_LOSS_WEIGHT
        self.num_joints = NUM_JOINTS_HAND

    def forward(self, pred, gt):

        if self.hparams.TRIAL.criterion == 'mse':
            self.criterion = self.criterion_mse
            self.criterion_noreduce = self.criterion_mse_noreduce
        elif self.hparams.TRIAL.criterion == 'l1':
            self.criterion = self.criterion_l1
            self.criterion_noreduce = self.criterion_l1_noreduce

        pred_vertices = pred['vertices']
        pred_joints = pred['joints3d']
        pred_kps = pred['joints2d']
        pred_rotmat = pred['pred_pose']
        pred_beta = pred['pred_shape']
        pred_cam = pred['pred_cam']
        gt_vertices = gt['vertices']
        gt_joints = gt['joints3d']
        gt_kps = gt['keypoints']
        gt_pose = gt['pose']
        gt_beta = gt['betas']

        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            criterion=self.criterion_l1,
        )

        loss_regr_pose, loss_regr_beta = smpl_losses(
            pred_rotmat,
            gt_pose,
            pred_beta,
            gt_beta,
            criterion=self.criterion,
            num_joints=self.num_joints,

        )

        loss_keypoints = projected_keypoint_loss(
            pred_kps,
            gt_kps,
            criterion=self.criterion,
        )

        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            criterion=self.criterion,
        )

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight_2d
        loss_keypoints_3d *= self.joint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_beta *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_beta': loss_regr_beta,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }
        loss = sum(loss for loss in loss_dict.values())
        loss *= self.loss_weight
        return loss, loss_dict


def projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):

    loss = criterion(pred_keypoints_2d, gt_keypoints_2d)
    return loss


def keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):

    loss = criterion(pred_keypoints_2d, gt_keypoints_2d)
    return loss


def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        criterion,
):
    gt_keypoints_3d = gt_keypoints_3d.clone()
    pred_keypoints_3d = pred_keypoints_3d
    if len(gt_keypoints_3d) > 0:
        return (criterion(pred_keypoints_3d, gt_keypoints_3d))
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_keypoints_3d.device)


def reconstruction_error(pred_keypoints_3d, gt_keypoints_3d, criterion):
    pred_keypoints_3d = pred_keypoints_3d.detach().cpu().numpy()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].detach().cpu().numpy()

    pred_keypoints_3d_hat = compute_similarity_transform_batch(pred_keypoints_3d, gt_keypoints_3d)
    return criterion(torch.tensor(pred_keypoints_3d_hat), torch.tensor(gt_keypoints_3d)).mean()


def shape_loss(
        pred_vertices,
        gt_vertices,
        criterion,
):
    pred_vertices_with_shape = pred_vertices
    gt_vertices_with_shape = gt_vertices

    if len(gt_vertices_with_shape) > 0:
        return criterion(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)


def smpl_losses(
        pred_rotmat,
        gt_pose,
        pred_betas,
        gt_betas,
        criterion,
        num_joints
):
    pred_rotmat_valid = pred_rotmat[:,1:]
    gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).view(-1, num_joints, 3, 3)[:, 1:]
    pred_betas_valid = pred_betas
    gt_betas_valid = gt_betas

    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = (criterion(pred_rotmat_valid, gt_rotmat_valid))
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
    return loss_regr_pose, loss_regr_betas
