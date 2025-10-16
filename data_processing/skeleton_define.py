import numpy as np 
from collections import OrderedDict

def joint_mapping(source_format, target_format, all_matched=True):
    mapping = np.ones(len(target_format),dtype=np.int32)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    
    if all_matched:
        assert (mapping<0).sum() == 0, f"set all matched in joint mapping, but found unmatched {mapping}"
    return np.array(mapping)

def convert_kp2ds(kp2ds, kp_num=None, thresh=None, maps=None):
    if kp_num is None:
        kp_num = len(maps)
    kp2ds_new = np.zeros((kp_num, 3))
    kp2ds_new[:len(maps)] = kp2ds[maps]
    kp2ds_new[:len(maps)][maps==-1] = -1

    if thresh is not None:
        kp2ds_new[:, 2] = kp2ds_new[:, 2] > thresh
        #kp2ds_new[kp2ds_new[:, 2]<0.5] = -1

    return kp2ds_new

# keypoints 2D 44 = 25 Openpose + 14 lsp + 5?
JOINT44_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'Nose', # 0
    'Neck', # 1
    'RShoulder', # 2
    'RElbow', # 3
    'RWrist', # 4
    'LShoulder', # 5
    'LElbow', # 6
    'LWrist', # 7
    'MidHip', # 8
    'RHip', # 9
    'RKnee', # 10
    'RAnkle', # 11
    'LHip', # 12
    'LKnee', # 13
    'LAnkle', # 14
    'REye', # 15
    'LEye', # 16
    'REar', # 17
    'LEar', # 18
    'LBigToe', # 19
    'LSmallToe', # 20
    'LHeel', # 21
    'RBigToe', # 22
    'RSmallToe', # 23
    'RHeel', # 24
    # 14 LSP joints
    'R_Ankle',
    'R_Knee',
    'R_Hip',
    'L_Hip',
    'L_Knee',
    'L_Ankle',
    'R_Wrist',
    'R_Elbow',
    'R_Shoulder',
    'L_Shoulder',
    'L_Elbow',
    'L_Wrist',
    'Neck_LSP',
    'HeadTop_LSP',
    # other dataset's joints
    'Pelvis_MPII',
    'Thorax_MPII',
    'Spine_H36M',
    'Jaw_H36M',
    'Head_H36M',
    #'Nose_other'
]

body44 = {
    'nose': 0, 
    'neck': 1, 
    'r_shoulder': 2, 
    'r_elbow': 3, 
    'r_wrist': 4, 
    'l_shoulder': 5, 
    'l_elbow': 6, 
    'l_wrist': 7, 
    'midhip': 8, 
    'r_hip': 9, 
    'r_knee': 10, 
    'r_ankle': 11, 
    'l_hip': 12, 
    'l_knee': 13, 
    'l_ankle': 14, 
    'r_eye': 15, 
    'l_eye': 16, 
    'r_ear': 17, 
    'l_ear': 18, 
    'l_big_toe': 19, 
    'l_small_toe': 20, 
    'l_heel': 21, 
    'r_big_toe': 22, 
    'r_small_toe': 23, 
    'r_heel': 24, 

    'r_lsp_ankle': 25, 'r_lsp_knee': 26, 
    'r_lsp_hip': 27, 'l_lsp_hip': 28, 
    'l_lsp_knee': 29, 'l_lsp_ankle': 30, 
    'r_lsp_wrist': 31, 'r_lsp_elbow': 32, 
    'r_lsp_shoulder': 33, 'l_shoulder': 34, 
    'l_lsp_elbow': 35, 'l_lsp_wrist': 36, 

    'neck_lsp': 37, 'headtop_lsp': 38, 
    'pelvis_mpii': 39, 'thorax_mpii': 40, 'spine_h36m': 41, 'jaw_h36m': 42, 'head_h36m': 43}


SMPLX_Joints = {
    0: 'pelvis',
    # CAUTION: body pose doesn't include pelvis, index -1
     1: 'l_hip',
     2: 'r_hip',
     3: 'spine1',
     4: 'l_knee',
     5: 'r_knee',
     6: 'spine2',
     7: 'l_ankle',
     8: 'r_ankle',
     9: 'spine3',
    10: 'l_foot',
    11: 'r_foot',
    12: 'neck',
    13: 'l_collar',
    14: 'r_collar',
    15: 'head',
    16: 'l_shoulder',
    17: 'r_shoulder',
    18: 'l_elbow',
    19: 'r_elbow',
    20: 'l_wrist',
    21: 'r_wrist',

    22: 'jaw',
    23: 'l_eye',
    24: 'r_eye',

    25: 'l_index1',
    26: 'l_index2',
    27: 'l_index3',
    28: 'l_middle1',
    29: 'l_middle2',
    30: 'l_middle3',
    31: 'l_pinky1',
    32: 'l_pinky2',
    33: 'l_pinky3',
    34: 'l_ring1',
    35: 'l_ring2',
    36: 'l_ring3',
    37: 'l_thumb1',
    38: 'l_thumb2',
    39: 'l_thumb3',
    40: 'r_index1',
    41: 'r_index2',
    42: 'r_index3',
    43: 'r_middle1',
    44: 'r_middle2',
    45: 'r_middle3',
    46: 'r_pinky1',
    47: 'r_pinky2',
    48: 'r_pinky3',
    49: 'r_ring1',
    50: 'r_ring2',
    51: 'r_ring3',
    52: 'r_thumb1',
    53: 'r_thumb2',
    54: 'r_thumb3'
}


rtmpose134 = {
 'l_hand_root': 92,
 'l_thumb1': 93,
 'l_thumb2': 94,
 'l_thumb3': 95,
 'l_thumb4': 96,
 'l_index1': 97,
 'l_index2': 98,
 'l_index3': 99,
 'l_index4': 100,
 'l_middle1': 101,
 'l_middle2': 102,
 'l_middle3': 103,
 'l_middle4': 104,
 'l_ring1': 105,
 'l_ring2': 106,
 'l_ring3': 107,
 'l_ring4': 108,
 'l_pinky1': 109,
 'l_pinky2': 110,
 'l_pinky3': 111,
 'l_pinky4': 112,

 'r_hand_root': 113,
 'r_thumb1': 114,
 'r_thumb2': 115,
 'r_thumb3': 116,
 'r_thumb4': 117,
 'r_index1': 118,
 'r_index2': 119,
 'r_index3': 120,
 'r_index4': 121,
 'r_middle1': 122,
 'r_middle2': 123,
 'r_middle3': 124,
 'r_middle4': 125,
 'r_ring1': 126,
 'r_ring2': 127,
 'r_ring3': 128,
 'r_ring4': 129,
 'r_pinky1': 130,
 'r_pinky2': 131,
 'r_pinky3': 132,
 'r_pinky4': 133
}

smplx42_lr_hand_ids = {'l_thumb4': 0, 'l_index4': 1, 'l_middle4': 2, 'l_ring4': 3, 'l_pinky4': 4, 
    'r_thumb4': 5, 'r_index4': 6, 'r_middle4': 7, 'r_ring4': 8, 'r_pinky4': 9, 
    'l_index1': 10, 'l_index2': 11, 'l_index3': 12, 'l_middle1': 13, 'l_middle2': 14, 'l_middle3': 15, 
    'l_pinky1': 16, 'l_pinky2': 17, 'l_pinky3': 18, 'l_ring1': 19, 'l_ring2': 20, 'l_ring3': 21, 
    'l_thumb1': 22, 'l_thumb2': 23, 'l_thumb3': 24, 'r_index1': 25, 'r_index2': 26, 'r_index3': 27, 
    'r_middle1': 28, 'r_middle2': 29, 'r_middle3': 30, 'r_pinky1': 31, 'r_pinky2': 32, 'r_pinky3': 33, 
    'r_ring1': 34, 'r_ring2': 35, 'r_ring3': 36, 'r_thumb1': 37, 'r_thumb2': 38, 'r_thumb3': 39, 
    'l_hand_root': 40, 'r_hand_root': 41}

smplx_lr_hand_inds = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

hand42 = {'r_thumb4': 0, 'r_thumb3': 1, 'r_thumb2': 2, 'r_thumb1': 3, 
          'r_index4': 4, 'r_index3': 5, 'r_index2': 6, 'r_index1': 7, 
          'r_middle4': 8, 'r_middle3': 9, 'r_middle2': 10, 'r_middle1': 11, 
          'r_ring4': 12, 'r_ring3': 13, 'r_ring2': 14, 'r_ring1': 15, 
          'r_pinky4': 16, 'r_pinky3': 17, 'r_pinky2': 18, 'r_pinky1': 19,   'r_hand_root': 20, 
          'l_thumb4': 21, 'l_thumb3': 22, 'l_thumb2': 23, 'l_thumb1': 24, 
          'l_index4': 25, 'l_index3': 26, 'l_index2': 27, 'l_index1': 28, 
          'l_middle4': 29, 'l_middle3': 30, 'l_middle2': 31, 'l_middle1': 32, 
          'l_ring4': 33, 'l_ring3': 34, 'l_ring2': 35, 'l_ring1': 36, 
          'l_pinky4': 37, 'l_pinky3': 38, 'l_pinky2': 39, 'l_pinky1': 40,   'l_hand_root': 41}

rhand21 = {'r_thumb4': 0, 'r_thumb3': 1, 'r_thumb2': 2, 'r_thumb1': 3, 
          'r_index4': 4, 'r_index3': 5, 'r_index2': 6, 'r_index1': 7, 
          'r_middle4': 8, 'r_middle3': 9, 'r_middle2': 10, 'r_middle1': 11, 
          'r_ring4': 12, 'r_ring3': 13, 'r_ring2': 14, 'r_ring1': 15, 
          'r_pinky4': 16, 'r_pinky3': 17, 'r_pinky2': 18, 'r_pinky1': 19,   'r_hand_root': 20}

mano_hand21 = {
    'r_hand_root': 0, 
    'r_thumb1': 1, 'r_thumb2': 2, 'r_thumb3': 3, 'r_thumb4': 4,
    'r_index1': 5, 'r_index2': 6, 'r_index3': 7, 'r_index4': 8, 
    'r_middle1': 9, 'r_middle2': 10, 'r_middle3': 11, 'r_middle4': 12, 
    'r_ring1': 13, 'r_ring2': 14, 'r_ring3': 15, 'r_ring4': 16, 
    'r_pinky1': 17, 'r_pinky2': 18, 'r_pinky3': 19, 'r_pinky4': 20,
}


COCO133_JOINTS = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,

    "l_big_toe": 17,
    "l_small_toe": 18,
    "l_heel": 19,
    "r_big_toe": 20,
    "r_small_toe": 21,
    "r_heel": 22,

    "face_0": 23,
    "face_1": 24,
    "face_2": 25,
    "face_3": 26,
    "face_4": 27,
    "face_5": 28,
    "face_6": 29,
    "face_7": 30,
    "face_8": 31,
    "face_9": 32,
    "face_10": 33,
    "face_11": 34,
    "face_12": 35,
    "face_13": 36,
    "face_14": 37,
    "face_15": 38,
    "face_16": 39,
    "face_17": 40,
    "face_18": 41,
    "face_19": 42,
    "face_20": 43,
    "face_21": 44,
    "face_22": 45,
    "face_23": 46,
    "face_24": 47,
    "face_25": 48,
    "face_26": 49,
    "face_27": 50,
    "face_28": 51,
    "face_29": 52,
    "face_30": 53,
    "face_31": 54,
    "face_32": 55,
    "face_33": 56,
    "face_34": 57,
    "face_35": 58,
    "face_36": 59,
    "face_37": 60,
    "face_38": 61,
    "face_39": 62,
    "face_40": 63,
    "face_41": 64,
    "face_42": 65,
    "face_43": 66,
    "face_44": 67,
    "face_45": 68,
    "face_46": 69,
    "face_47": 70,
    "face_48": 71,
    "face_49": 72,
    "face_50": 73,
    "face_51": 74,
    "face_52": 75,
    "face_53": 76,
    "face_54": 77,
    "face_55": 78,
    "face_56": 79,
    "face_57": 80,
    "face_58": 81,
    "face_59": 82,
    "face_60": 83,
    "face_61": 84,
    "face_62": 85,
    "face_63": 86,
    "face_64": 87,
    "face_65": 88,
    "face_66": 89,
    "face_67": 90,

    "l_hand_root": 91,
    "l_thumb1": 92,
    "l_thumb2": 93,
    "l_thumb3": 94,
    "l_thumb4": 95,
    "l_index1": 96,
    "l_index2": 97,
    "l_index3": 98,
    "l_index4": 99,
    "l_middle1": 100,
    "l_middle2": 101,
    "l_middle3": 102,
    "l_middle4": 103,
    "l_ring1": 104,
    "l_ring2": 105,
    "l_ring3": 106,
    "l_ring4": 107,
    "l_pinky1": 108,
    "l_pinky2": 109,
    "l_pinky3": 110,
    "l_pinky4": 111,

    "r_hand_root": 112,
    "r_thumb1": 113,
    "r_thumb2": 114,
    "r_thumb3": 115,
    "r_thumb4": 116,
    "r_index1": 117,
    "r_index2": 118,
    "r_index3": 119,
    "r_index4": 120,
    "r_middle1": 121,
    "r_middle2": 122,
    "r_middle3": 123,
    "r_middle4": 124,
    "r_ring1": 125,
    "r_ring2": 126,
    "r_ring3": 127,
    "r_ring4": 128,
    "r_pinky1": 129,
    "r_pinky2": 130,
    "r_pinky3": 131,
    "r_pinky4": 132
}
{
                14: "hip",
                15: "left_knee",
                16: "right_knee",
                17: "left_ankle",
                18: "right_ankle",
                19: "left_big_toe",
                20: "left_small_toe",
                21: "left_heel",
                22: "right_big_toe",
                23: "right_small_toe",
                24: "right_heel",
            },
COCO25 = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "neck": 5,
    "l_shoulder": 6,
    "r_shoulder": 7,
    "l_elbow": 8,
    "r_elbow": 9,
    "l_wrist": 10,
    "r_wrist": 11,
    
    "l_hip": 12,
    "r_hip": 13,
    "midhip": 14,
    "l_knee": 15,
    "r_knee": 16,
    "l_ankle": 17,
    "r_ankle": 18,

    "l_big_toe": 19,
    "l_small_toe": 20,
    "l_heel": 21,
    "r_big_toe": 22,
    "r_small_toe": 23,
    "r_heel": 24}

# ID_KP2Ds_inds = [JOINT44_NAMES.index(joint_name) for joint_name in \
#                 ['Nose', 'Neck',
#                 'RShoulder', 'RElbow', 'RWrist',
#                 'LShoulder', 'LElbow', 'LWrist',
#                 'RHip', 'RKnee', 'RAnkle',
#                 'LHip', 'LKnee', 'LAnkle']]

ID_KP2Ds_inds = [JOINT44_NAMES.index(joint_name) for joint_name in \
                ['Neck',
                'RWrist',
                'LWrist']]

BODY_KP2Ds_PROMPT_inds = [JOINT44_NAMES.index(joint_name) for joint_name in \
                ['Nose', 'REye', 'LEye',
                'RShoulder', 'RElbow', 'RWrist',
                'LShoulder', 'LElbow', 'LWrist',
                'RHip', 'RKnee', 'RAnkle',
                'LHip', 'LKnee', 'LAnkle', 
                'LBigToe', 'LHeel', 'RBigToe', 'RHeel']]

Hips_inds = [JOINT44_NAMES.index(joint_name) for joint_name in \
                                    ['RHip', 'LHip', 'MidHip']]

invalid_smpl_joint_inds = [JOINT44_NAMES.index(joint_name) for joint_name in \
            ['REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']]

# need to learn to adapt to different prompt
# rtmpose134_to_openpose25 = np.array([0, -1, 2, 3, 4, 5, 6, 7, -1, 
#                     8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
#                     18, 19, 20, 21, 22, 23])
# remove left / right hip, which are very not aligned.  to avoid wrong supervision & fat shape.
rtmpose134_to_openpose25 = np.array([0, -1, 2, 3, 4, 5, 6, 7, -1, 
                    -1, 9, 10, -1, 12, 13, 14, 15, 16, 17, 
                    18, 19, 20, 21, 22, 23])

smplx42_to_hand42_old = [5, 39, 38, 37, 6, 27, 26, 25, 7, 30, 29, 28, 8, 36, 35, 34, 9, 33, 32, 31, 41, 
                  0, 24, 23, 22, 1, 12, 11, 10, 2, 15, 14, 13, 3, 21, 20, 19, 4, 18, 17, 16, 40]

smplx42_to_hand42 = joint_mapping(smplx42_lr_hand_ids, hand42, all_matched=True)

assert (np.array(smplx42_to_hand42_old) - np.array(smplx42_to_hand42)).sum()==0, "smplx42_to_hand42 goes wrong!"

mano_hand21_to_hand21 = joint_mapping(mano_hand21, rhand21, all_matched=True)

hand_root_inds = [rhand21[joint_name] for joint_name in \
                ['r_thumb1', 'r_index1', 'r_middle1', 'r_ring1', 'r_pinky1', 'r_hand_root']]

rtmpose134_to_hand42 = joint_mapping(rtmpose134, hand42, all_matched=True)

coco133_to_body44 = joint_mapping(COCO133_JOINTS, body44, all_matched=False)
coco133_to_hand42 = joint_mapping(COCO133_JOINTS, hand42, all_matched=True)

coco25_to_body44 = joint_mapping(COCO25, body44, all_matched=False)