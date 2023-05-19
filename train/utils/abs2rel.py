import torch
from .geometry import batch_rodrigues, batch_rot2aa
def pose_abs2rel(global_pose, body_pose, abs_joint_pose, abs_joint = 'right_wrist'):
    ''' change absolute pose to relative pose
    Basic knowledge for SMPLX kinematic tree:
            absolute pose = parent pose * relative pose
    '''

    if abs_joint == 'head':
        kin_chain = [15, 12, 9, 6, 3, 0]
    elif abs_joint == 'right_wrist':
        # Pelvis -> Spine 1, 2, 3 -> right Collar -> right shoulder
        # -> right elbow -> right wrist
        kin_chain = [21, 19, 17, 14, 9, 6, 3, 0]
    elif abs_joint == 'left_wrist':
        # Pelvis -> Spine 1, 2, 3 -> Left Collar -> Left shoulder
        # -> Left elbow -> Left wrist
        kin_chain = [20, 18, 16, 13, 9, 6, 3, 0]
    else:
        raise NotImplementedError(
            f'pose_rel2abs does not support: {abs_joint}')

    #####
    batch_size = global_pose.shape[0]
    dtype = global_pose.dtype
    device = global_pose.device
    full_pose = torch.cat([global_pose, body_pose], dim=1)
    rel_rot_mat = torch.eye(
        3, device=device,
        dtype=dtype).unsqueeze_(dim=0).repeat(batch_size, 1, 1)
    for idx in kin_chain[1:]:
        rel_rot_mat = torch.bmm(full_pose[:, idx], rel_rot_mat)
        # rel_rot_mat = torch.matmul(full_pose[:, idx], rel_rot_mat)

    # This contains the absolute pose of the parent
    abs_parent_pose = rel_rot_mat.detach()

    # abs_head = parents(abs_neck) * rel_head ==> rel_head =  abs_neck.T * abs_head
    rel_joint_pose = torch.matmul(
        abs_parent_pose.reshape(-1, 3, 3).transpose(1, 2),
        abs_joint_pose.reshape(-1, 3, 3))
    # Replace the new relative pose
    return rel_joint_pose

def pose_rel2abs(global_pose, body_pose, abs_joint = 'right_wrist'):
    ##### change relative head pose to absolute head pose

    full_pose = torch.cat([global_pose, body_pose])
    full_pose  = batch_rodrigues(full_pose.view(-1,3))
    if abs_joint == 'head':
        kin_chain = [15, 12, 9, 6, 3, 0]
    elif abs_joint == 'right_wrist':
        # Pelvis -> Spine 1, 2, 3 -> right Collar -> right shoulder
        # -> right elbow -> right wrist
        kin_chain = [21, 19, 17, 14, 9, 6, 3, 0]
    elif abs_joint == 'left_wrist':
        # Pelvis -> Spine 1, 2, 3 -> Left Collar -> Left shoulder
        # -> Left elbow -> Left wrist
        kin_chain = [20, 18, 16, 13, 9, 6, 3, 0]
    else:
        raise NotImplementedError(
            f'pose_rel2abs does not support: {abs_joint}')
    rel_rot_mat = torch.eye(3, device=full_pose.device,
                            dtype=full_pose.dtype)
    # dec_wrist_pose = torch.matmul(
    #         parent_rots.reshape(-1, 3, 3).transpose(1, 2),
    #         dec_wrist_pose_abs.reshape(-1, 3, 3)
    #     )
    for idx in kin_chain:
        # rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
        rel_rot_mat = torch.matmul(full_pose[idx], rel_rot_mat)
    abs_pose = rel_rot_mat[None,:,:]
    # abs_head = parents(abs_neck) * rel_head ==> rel_head =  abs_neck.T * abs_head
    # rel_head_pose = torch.matmul(abs_neck_pose.reshape(-1, 3, 3).transpose(1, 2), abs_head_pose.reshape(-1, 3, 3))
    # body_pose[:, 15-1, :, :] = rel_head_pose
    # import ipdb; ipdb.set_trace()
    abs_pose = batch_rot2aa(abs_pose)
    return abs_pose
