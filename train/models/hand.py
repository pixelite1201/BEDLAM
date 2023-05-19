import torch.nn as nn

from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32, hrnet_w48

from .head.mano import MANOHead
from .head.hmr_hand import HMRHand
from ..core.config import PRETRAINED_CKPT_FOLDER

class Hand(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            img_res=224,
            focal_length=5000,
            pretrained_ckpt=None,
            hparams=None
    ):
        super(Hand, self).__init__()
        self.hparams = hparams

        # Initialize backbone
        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            pretrained_ckpt = backbone + '-' + pretrained_ckpt
            pretrained_ckpt_path = PRETRAINED_CKPT_FOLDER[pretrained_ckpt]
            self.backbone = eval(backbone)(
                pretrained_ckpt_path=pretrained_ckpt_path,
                downsample=True,
                use_conv=(use_conv == 'conv'),
            ) 
        else:
            pass

        self.manohead = MANOHead(img_res=img_res)
        self.head = HMRHand(mean_pose_params=self.manohead.mano.pose_mean, 
                            num_input_features=get_backbone_info(backbone)['n_output_channels'],)

    def forward(
            self,
            images,
    ):
        features = self.backbone(images)

        hmr_output = self.head(features)
        mano_output = self.manohead(
            rotmat=hmr_output['pred_pose'],
            cam=hmr_output['pred_cam'],
            shape=hmr_output['pred_shape'],
            normalize_joints2d=True,
        )

        mano_output.update(hmr_output)

        return mano_output