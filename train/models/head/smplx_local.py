from smplx import SMPLXLayer as SMPLX_
from smplx.utils import SMPLXOutput
from ...core.constants import NUM_JOINTS_SMPLX


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
