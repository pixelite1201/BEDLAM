import os
import cv2
import time

from . import kp_utils
from .vis_utils import draw_skeleton, visualize_joint_error, visualize_joint_uncertainty, \
    visualize_heatmaps, visualize_segm_masks

from .mesh_utils import get_checkerboard_plane, get_world_mesh_list

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] \
#     if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

import torch
import trimesh
import pyrender
import numpy as np
from smplx import SMPL
from PIL import Image
from torchvision.utils import make_grid
from typing import List, Set, Dict, Tuple, Optional

from .vis_utils import get_colors
from ..core.config import SMPL_MODEL_DIR
from cam_reg.vis_utils import show_horizon_line


def view_animation(
        image: np.ndarray,
        camera_translation: torch.tensor,
        vertices: torch.tensor,
        camera_rotation: torch.tensor,
        focal_length: Tuple,
        camera_center: Tuple,
        mesh_color: str = 'gray',
        alpha: float = 1.0,
        faces: np.ndarray = None,
        add_ground_plane: bool = True,
        gif_filename: str = '',
        add_image: bool = False,
):
    to_numpy = lambda x: x.detach().cpu().numpy()

    if np.max(image) > 10:
        image = image / 255.

    camera_translation = to_numpy(camera_translation)
    camera_rotation = to_numpy(camera_rotation)
    vertices = to_numpy(vertices)

    if faces is None:
        smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=1,
            create_transl=False
        )
        faces = smpl.faces
        del smpl

    mesh_color = get_colors()[mesh_color]
    cam_color = get_colors()['green']

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        alphaMode='OPAQUE',
        baseColorFactor=(mesh_color[0] / 255., mesh_color[1] / 255., mesh_color[2] / 255., alpha))

    camera_translation[0] *= -1.

    mesh = trimesh.Trimesh(vertices, faces, process=False)

    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    if add_ground_plane:
        ground_mesh = pyrender.Mesh.from_trimesh(
            get_checkerboard_plane(plane_width=6, num_boxes=8),
            smooth=False,
        )
        pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        # pose[:3, 3] = [0, -1, 0]
        pose[:3, 3] = np.array([0, mesh.bounds[0, 1], 0])

        scene.add(ground_mesh, pose=pose, name='ground_plane')

    camera_pose = np.eye(4)

    camera_pose[:3, :3] = camera_rotation
    camera_pose[:3, 3] = camera_rotation @ camera_translation

    ######## add cam mesh
    cam_mesh = trimesh.load('data/slr_camera.obj', process=False)
    # cam_mesh = trimesh.load('data/SLR_Camera_V1_L3.123c9eb1ce10-76b6-4cc9-88b7-06885d8e450d/10124_SLR_Camera_SG_V1_Iteration2.obj', process=False)
    cam_mesh.apply_scale(0.004)
    cam_mesh.apply_transform(
        trimesh.transformations.rotation_matrix(
            np.radians(-90), [1, 0, 0])
    )
    cam_mesh.apply_transform(
        trimesh.transformations.rotation_matrix(
            np.radians(180), [0, 1, 0])
    )

    cam_mesh.visual.face_colors = np.ones_like(cam_mesh.faces) * cam_color

    cam_mesh = pyrender.Mesh.from_trimesh(cam_mesh, smooth=False)
    scene.add(cam_mesh, name='cam_mesh', pose=camera_pose)

    cam_coord_mesh = pyrender.Mesh.from_trimesh(
        get_world_mesh_list(axisRadius=0.05, axisHeight=0.8, add_plane=False),
        smooth=False,
    )
    scene.add(cam_coord_mesh, name='cam_coord_mesh', pose=camera_pose)

    ######## add cam mesh

    # viewer_cam_pose = np.eye(4)
    # viewer_cam_pose[:3, 3] += [0, 0, camera_pose[2,3] + 1]

    viewer_cam_pose = camera_pose.copy()

    if gif_filename.endswith('hmr.gif'):
        viewer_cam_pose[2, 3] = 5
        viewer_cam_pose[0, 3] = 0
    else:
        viewer_cam_pose[:3, 3] += 1
        viewer_cam_pose[0, 3] = 0

    # camera = pyrender.IntrinsicsCamera(fx=500, fy=500,
    #                                    cx=camera_center[0], cy=camera_center[1])

    # camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
    #                                    cx=camera_center[0], cy=camera_center[1])

    camera = pyrender.IntrinsicsCamera(fx=750, fy=750,
                                       cx=1000//2, cy=1000//2)

    scene.add(camera, pose=viewer_cam_pose)

    ##### add image #####
    if add_image:
        # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3)
        # light_pose = np.eye(4)
        # light_pose[:3, 3] = np.array([0, -1, -1.5])
        # scene.add(light, pose=light_pose)

        image_mesh = create_image_mesh(image, scale=4.0)
        image_mesh_pose = np.eye(4)
        image_mesh_pose[:3, 3] = [0, -image_mesh.bounds[0, 1], -3]
        scene.add(image_mesh, name='img_mesh', pose=image_mesh_pose)

    # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    # scene.add(light)
    #
    # light_pose = np.eye(4)
    #
    # light_pose[:3, 3] = np.array([0, -1, 1])
    # scene.add(light, pose=light_pose)
    #
    # light_pose[:3, 3] = np.array([0, 1, 1])
    # scene.add(light, pose=light_pose)
    #
    # light_pose[:3, 3] = np.array([1, 1, 2])
    # scene.add(light, pose=light_pose)

    viewer = pyrender.Viewer(
        scene,
        viewport_size=(1000, 1000),
        use_raymond_lighting=True,
        use_direct_lighting=True,
        lighting_intensity=5.0,
        rotate=True,
        rotate_axis=[0,1,0],
        view_center=[0,0,0],
        # show_world_axis=True,
        # viewport_size=(image.shape[1], image.shape[0]),
        # cull_faces=False,
        run_in_thread=True,
        record=True,
    )

    if gif_filename != '':
        time.sleep(30)
        viewer.close_external()
        print(f'Saving {gif_filename}')
        viewer.save_gif(filename=gif_filename)


def create_image_mesh(image, scale=1.0):

    def make_square(im, min_size=256, fill_color=(1, 1, 1, 1)):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGBA', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    if not isinstance(image, Image.Image):
        if image.max() < 2:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    image = make_square(image)

    screen = trimesh.primitives.Box(
        center=[0, 0, -0.0001],
        extents=[1, 1, 0.0002]
    )
    uv = screen.vertices[:, :2]  # np.random.rand(rectangle.vertices.shape[0], 2)
    uv = (uv - uv.min()) / np.ptp(uv)

    uv[screen.vertices[:, 2] < 0] = 0.0

    material = trimesh.visual.texture.SimpleMaterial(image=image)
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=image, material=material)
    # assert(len(uvs) == len(vertices))
    mesh = trimesh.Trimesh(vertices=screen.vertices,
                           faces=screen.faces,
                           visual=color_visuals,
                           validate=True,
                           process=True)

    mesh.apply_scale(scale)
    screen_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    return screen_mesh