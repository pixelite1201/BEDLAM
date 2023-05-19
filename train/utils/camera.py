# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple

import torch
import torch.nn as nn
import numpy as np

from smplx.lbs import transform_mat


PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])


def create_camera(camera_type='persp', **kwargs):
    if camera_type.lower() == 'persp':
        return PerspectiveCamera(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))


class PerspectiveCamera(nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None, focal_length=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        self.render_cam = np.eye(4)

        if focal_length is None or type(focal_length) == float:
            focal_length = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length is None else
                focal_length,
                dtype=dtype)

        focal_length = nn.Parameter(focal_length, requires_grad=False)

        self.register_parameter('focal_length', focal_length)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        rotation = nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)

    def forward(self, points):
        device = points.device

        # with torch.no_grad():
        camera_mat = torch.zeros([self.batch_size, 2, 2],
                                 dtype=self.dtype, device=points.device).float()
        camera_mat[:, 0, 0] = self.focal_length
        camera_mat[:, 1, 1] = self.focal_length

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1)).to(device).float()
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points


def convert_pare_to_full_img_cam(pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf
    s, tx, ty = pare_cam[0]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[0] - (img_w / 2)) / (s * bbox_height)
    cy = 2 * (bbox_center[1] - (img_h / 2)) / (s * bbox_height)

    cam_t = [tx + cx, ty + cy, tz]

    return torch.tensor(cam_t, dtype=pare_cam.dtype, device=pare_cam.device).expand_as(pare_cam)


def convert_focal_length_to_bbox_coords(f_img, bbox_height, img_res):
    '''
        Converts the original focal length to bbox focal length
        f_img: int
            focal length in original image space
        bbox_height: int
            bbox_height in original image space
        img_res: int
            resized bbox resolution
    '''
    r = bbox_height / img_res
    f_bbox = f_img / (r * img_res)
    return f_bbox


def convert_focal_length_to_image_coords(f_bbox, bbox_height, img_res):
    '''
        Converts the bbox focal length to image focal length
        f_bbox: int
            focal length in bbox space
        bbox_height: int
            bbox_height in original image space
        img_res: int
            resized bbox resolution
    '''
    return 0