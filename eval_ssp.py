import numpy as np
import pickle
import os
import torch
import sys
from smplx import SMPL

# Evaluation code from https://github.com/akashsengupta1997/SSP-3D
SMPL_MODEL_DIR = '/ps/project/alignment/models/smpl'
def compute_pve_neutral_pose_scale_corrected(predicted_smpl_shape, target_smpl_shape, gender):
    """
    Given predicted and target SMPL shape parameters, computes neutral-pose per-vertex error
    after scale-correction (to account for scale vs camera depth ambiguity).
    :param predicted_smpl_parameters: predicted SMPL shape parameters tensor with shape (1, 10)
    :param target_smpl_parameters: target SMPL shape parameters tensor with shape (1, 10)
    :param gender: gender of target
    """
    smpl_male = SMPL(SMPL_MODEL_DIR, batch_size=1, gender='male')
    smpl_female = SMPL(SMPL_MODEL_DIR, batch_size=1, gender='female')

    # Get neutral pose vertices
    if gender == 'm':
        pred_smpl_neutral_pose_output = smpl_male(betas=predicted_smpl_shape)
        target_smpl_neutral_pose_output = smpl_male(betas=target_smpl_shape)
    elif gender == 'f':
        pred_smpl_neutral_pose_output = smpl_female(betas=predicted_smpl_shape)
        target_smpl_neutral_pose_output = smpl_female(betas=target_smpl_shape)

    pred_smpl_neutral_pose_vertices = pred_smpl_neutral_pose_output.vertices
    target_smpl_neutral_pose_vertices = target_smpl_neutral_pose_output.vertices

    # Rescale such that RMSD of predicted vertex mesh is the same as RMSD of target mesh.
    # This is done to combat scale vs camera depth ambiguity.
    pred_smpl_neutral_pose_vertices_rescale = scale_and_translation_transform_batch(pred_smpl_neutral_pose_vertices,
                                                                                    target_smpl_neutral_pose_vertices)

    # Compute PVE-T-SC
    pve_neutral_pose_scale_corrected = np.linalg.norm(pred_smpl_neutral_pose_vertices_rescale
                                                      - target_smpl_neutral_pose_vertices.detach().cpu().numpy(),
                                                      axis=-1)  # (1, 6890)

    return pve_neutral_pose_scale_corrected


def scale_and_translation_transform_batch(P, T):
    """
    First normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """
    P = P.detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed

if __name__ == '__main__':
    df = np.load('data/ssp_3d_test.npz')
    inp_path = sys.argv[1]
    total_err=[]

    for i in range(len(df['image'])):
        pred = pickle.load(open(os.path.join(inp_path, str(i)+'.pkl'),'rb'))
        err = compute_pve_neutral_pose_scale_corrected(pred['betas'].detach().cpu(),torch.tensor(df['shape'][i].reshape(-1,10)),df['gender'][i])
        total_err.append(err.mean())
    print('SSP-3D PVE-T-SC error:', np.array(total_err).mean()*1000)
