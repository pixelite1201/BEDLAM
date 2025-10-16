
import numpy as np
import torch
import smplx
from smplx import SMPLX as _SMPLX
from smplx.utils import ModelOutput, SMPLOutput
from smplx.lbs import vertices2joints
from vertex_joint_selector import VertexJointSelector
from vertex_ids import vertex_ids as VERTEX_IDS

SMPL_MODEL_DIR = 'bedlam_data/body_models/SMPL_python_v.1.1.0/smpl/models'
SMPLX2SMPL = 'bedlam_data/utils/smplx2smpl.pkl'
SMPL2J19 = 'bedlam_data/utils/SMPL_to_J19.pkl'

class SMPLX(_SMPLX):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPLX, self).__init__(*args, **kwargs)
        smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        import pickle
        self.smpl = smplx.SMPL(model_path=SMPL_MODEL_DIR)
        self.smplx2smpl = pickle.load(open(SMPLX2SMPL, 'rb'))
        J_regressor_extra = pickle.load(open(SMPL2J19,'rb'))
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))

        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None], dtype=torch.float32).repeat(1, 1, 1) #Todo provide batch size?
        self.J_regressor_smplx2smpljoints = torch.mm(self.smpl.J_regressor, self.smplx2smpl[0])
        self.J_regressor_extra = torch.mm(self.J_regressor_extra, self.smplx2smpl[0])
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))


    def forward(self,openpose_smplx=False,openpose=False, *args, **kwargs):
        #Openpose_smplx by default
        kwargs['get_skin'] = True
   
        smplx_output = super(SMPLX, self).forward(*args, **kwargs)
        smplx_output_vertices, smplx_output_joints = smplx_output.vertices, smplx_output.joints
        if openpose_smplx:
            smpl_24_joints = vertices2joints(self.J_regressor_smplx2smpljoints, smplx_output_vertices)
            smpl_output_extra_joints = smplx_output_joints[:,55:76]
            smpl_output_joints = torch.cat([smpl_24_joints, smpl_output_extra_joints], dim=1)
            joints = smpl_output_joints[:, self.joint_map, :]
            extra_joints = vertices2joints(self.J_regressor_extra, smplx_output_vertices)
            joints = torch.cat([joints, extra_joints, smplx_output_joints], dim=1)
        else:
            joints = smplx_output_joints

        output = SMPLOutput(vertices=smplx_output_vertices,
                            joints=joints)
        return output
