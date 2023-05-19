import torch
import torch.nn as nn
from ..utils.geometry import batch_rodrigues, rotmat_to_rot6d
from ..utils.eval_utils import compute_similarity_transform_batch
from ..core.constants import NUM_JOINTS_SMPLX, NUM_JOINTS_SMPL

class HMRLoss(nn.Module):
    def __init__(
            self,
            hparams=None,
    ):
        super(HMRLoss, self).__init__()
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
        if self.hparams.TRIAL.version == 'real': # Using SMPL
            self.num_joints = 49
        else:
            self.num_joints = 24

    def forward(self, pred, gt):
        if self.hparams.TRIAL.criterion == 'mse':
            self.criterion = self.criterion_mse
            self.criterion_noreduce = self.criterion_mse_noreduce
        elif self.hparams.TRIAL.criterion == 'l1':
            self.criterion = self.criterion_l1
            self.criterion_noreduce = self.criterion_l1_noreduce
        img_size = gt['orig_shape'].rot90().T.unsqueeze(1)
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['joints3d'][:, :self.num_joints]
        pred_keypoints_2d = pred['joints2d'][:, :self.num_joints]
        pred_vertices = pred['vertices']
        gt_betas = gt['betas']
        gt_joints = gt['joints3d'][:, :self.num_joints]
        gt_vertices = gt['vertices']
        gt_pose = gt['pose']

        if self.hparams.TRIAL.bedlam_bbox:
            # Use full image keypoints
            pred_keypoints_2d[:, :, :2] = 2 * (pred_keypoints_2d[:, :, :2] / img_size) - 1
            gt_keypoints_2d = gt['keypoints_orig']
            gt_keypoints_2d[:, :, :2] = 2 * (gt_keypoints_2d[:, :, :2] / img_size) - 1
        else:
            # Use crop keypoints
            gt_keypoints_2d = gt['keypoints']

        loss_keypoints = projected_keypoint_loss(
            pred_keypoints_2d,
            gt_keypoints_2d,
            criterion=self.criterion_noreduce,
        )

        if self.hparams.TRIAL.bedlam_bbox:
            loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] * 200.).unsqueeze(-1)
            loss_keypoints = loss_keypoints * loss_keypoints_scale.unsqueeze(1)
            loss_keypoints = loss_keypoints.mean()
        else:
            loss_keypoints = loss_keypoints.mean()

        # Compute 2D reprojection loss for the keypoints
        if self.hparams.DATASET.proj_verts:
            pred_proj_verts = pred['pred_proj_verts']
            gt_proj_verts = gt['proj_verts_orig']
            gt_proj_verts[:, :, :2] = 2 * (gt_proj_verts[:, :, :2] / img_size) - 1
            pred_proj_verts[:, :, :2] = 2 * (pred_proj_verts[:, :, :2] / img_size) - 1

            loss_projverts = projected_verts_loss(
                pred_proj_verts,
                gt_proj_verts,
                criterion=self.criterion_noreduce,
            )

            if self.hparams.TRIAL.bedlam_bbox:
                loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] * 200.).unsqueeze(-1)
                loss_projverts = loss_projverts * loss_keypoints_scale.unsqueeze(1)
                loss_projverts = loss_projverts.mean()
            else:
                loss_projverts = loss_projverts.mean()
  
        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            criterion=self.criterion,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            criterion=self.criterion,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            criterion=self.criterion_l1,
        )

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight_2d
        loss_keypoints_3d *= self.joint_loss_weight

        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()

        if self.hparams.TRIAL.losses_abl == 'param':
            loss_dict = {
                        'loss/loss_keypoints': loss_keypoints,
                        'loss/loss_regr_pose': loss_regr_pose,
                        'loss/loss_regr_betas': loss_regr_betas,
                        'loss/loss_cam': loss_cam,
                    }
        elif self.hparams.TRIAL.losses_abl == 'param_keypoints':
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_regr_pose': loss_regr_pose,
                'loss/loss_regr_betas': loss_regr_betas,
                'loss/loss_cam': loss_cam,
            }
        elif self.hparams.TRIAL.losses_abl == 'keypoints':
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_cam': loss_cam,
            }
        elif self.hparams.TRIAL.losses_abl == 'param_verts':
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                # 'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_regr_pose': loss_regr_pose,
                'loss/loss_regr_betas': loss_regr_betas,
                'loss/loss_shape': loss_shape,
                'loss/loss_cam': loss_cam,
            }
        elif self.hparams.TRIAL.losses_abl == 'verts':
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                # 'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_shape': loss_shape,
                'loss/loss_cam': loss_cam,
            }
        elif self.hparams.DATASET.proj_verts:
            loss_projverts *= self.keypoint_loss_weight_2d

            loss_dict = {
                'loss/loss_projverts': loss_projverts,
                'loss/loss_keypoints': loss_keypoints,
                'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_regr_pose': loss_regr_pose,
                'loss/loss_regr_betas': loss_regr_betas,
                'loss/loss_shape': loss_shape,
                'loss/loss_cam': loss_cam,
            }

        else: # param+keypoints+verts
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_regr_pose': loss_regr_pose,
                'loss/loss_regr_betas': loss_regr_betas,
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

    conf = gt_keypoints_2d[:, :, -1]
    conf[conf == -2] = 0
    conf = conf.unsqueeze(-1)
    loss = conf * criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])
    return loss


def projected_verts_loss(
        pred_proj_verts,
        gt_proj_verts,
        criterion,
):
    loss = criterion(pred_proj_verts, gt_proj_verts[:, :, :-1])
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
        pred_betas,
        gt_pose,
        gt_betas,
        criterion,
):
    pred_rotmat_valid = pred_rotmat[:,1:]
    batch_size = pred_rotmat_valid.shape[0]
    gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).view(batch_size, -1, 3, 3)[:, 1:]
    pred_betas_valid = pred_betas
    gt_betas_valid = gt_betas

    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = (criterion(pred_rotmat_valid, gt_rotmat_valid))
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
    return loss_regr_pose, loss_regr_betas
