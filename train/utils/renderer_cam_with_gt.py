import os
import cv2

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] \
#     if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

import torch
import trimesh
import pyrender
import numpy as np
from smplx import SMPL
from torchvision.utils import make_grid
from typing import List, Set, Dict, Tuple, Optional


SMPL_MODEL_DIR='/ps/project/alignment/models/smpl'

def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
        'pinkish': np.array([204, 77, 77]),
    }
    return colors


def get_checkerboard_plane(plane_width=4, num_boxes=9, center=True):
    
    pw = plane_width/num_boxes
    white = [220, 220, 220, 255]
    black = [35, 35, 35, 255]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            ground = trimesh.primitives.Box(
                center=[0, 0, -0.0001],
                extents=[pw, pw, 0.0002]
            )

            if center:
                c = c[0]+(pw/2)-(plane_width/2), c[1]+(pw/2)-(plane_width/2)
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.apply_translation([c[0], c[1], 0])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.visual.face_colors = black if ((i+j) % 2) == 0 else white
            meshes.append(ground)

    return meshes


def render_overlay_image(
        image: np.ndarray,
        camera_translation: np.ndarray,
        vertices: np.ndarray,
        camera_rotation: np.ndarray,
        focal_length: Tuple,
        camera_center: Tuple,
        mesh_color: str = 'gray',
        alpha: float = 1.0,
        faces: np.ndarray = None,
        sideview_angle: int = 0,
        mesh_filename: str = None,
        add_ground_plane: bool = True,
        correct_ori: bool = True,
) -> np.ndarray:
    if faces is None:
        smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=1,
            create_transl=False
        )
        faces = smpl.faces
        del smpl

    mesh_color = get_colors()[mesh_color]

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        alphaMode='OPAQUE',
        baseColorFactor=(mesh_color[0] / 255., mesh_color[1] / 255., mesh_color[2] / 255., alpha))

    camera_translation[0] *= -1.

    mesh = trimesh.Trimesh(vertices, faces, process=False)

    if correct_ori:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

    if sideview_angle > 0:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(sideview_angle), [0, 1, 0])
        mesh.apply_transform(rot)

    if mesh_filename:
        mesh.export(mesh_filename)
        if not mesh_filename.endswith('_rot.obj'):
            np.save(mesh_filename.replace('.obj', '.npy'), camera_translation)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)


    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=np.ones(3) * 0)
    scene.add(mesh, 'mesh')

    if add_ground_plane:
        ground_mesh = pyrender.Mesh.from_trimesh(
            get_checkerboard_plane(),
            smooth=False,
        )
        pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        # pose[:3, 3] = [0, -1, 0]
        pose[:3, 3] = np.array([0, mesh.bounds[0, 1], 0])

        scene.add(ground_mesh, pose=pose, name='ground_plane')

    camera_pose = np.eye(4)
    if camera_rotation is not None:
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] =  camera_rotation @ camera_translation
    else:
        camera_pose[:3, 3] =  camera_translation

    camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
                                       cx=camera_center[0], cy=camera_center[1])

    scene.add(camera, pose=camera_pose)
    # Create light source
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    # for DirectionalLight, only rotation matters
    light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
    scene.add(light, pose=light_pose)
    light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
    scene.add(light, pose=light_pose)


    # light_pose = np.eye(4)

    # light_pose[:3, 3] = np.array([0, -1, 1])
    # scene.add(light, pose=light_pose)

    # light_pose[:3, 3] = np.array([0, 1, 1])
    # scene.add(light, pose=light_pose)

    # light_pose[:3, 3] = np.array([1, 1, 2])
    # scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=image.shape[1],
        viewport_height=image.shape[0],
        point_size=1.0
    )

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    valid_mask = (rend_depth > 0)[:, :, None]
    output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
    return output_img


def render_overlay_image_multiple(
        image: np.ndarray,
        camera_translation: np.ndarray,
        vertices: np.ndarray,
        gt_vertices: np.ndarray,
        camera_rotation: np.ndarray,
        focal_length: Tuple,
        camera_center: Tuple,
        mesh_color: str = 'gray',
        alpha: float = 1.0,
        faces: np.ndarray = None,
        sideview_angle: int = 0,
        mesh_filename: str = None,
        add_ground_plane: bool = True,
        correct_ori: bool = True,
) -> np.ndarray:
    if faces is None:
        smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=1,
            create_transl=False
        )
        faces = smpl.faces
        del smpl

    mesh_color = get_colors()[mesh_color]
    mesh_color2 = np.array([145, 191, 219])

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        alphaMode='OPAQUE',
        baseColorFactor=(mesh_color[0] / 255., mesh_color[1] / 255., mesh_color[2] / 255., alpha))

    material2 = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        alphaMode='OPAQUE',
        baseColorFactor=(mesh_color2[0] / 255., mesh_color2[1] / 255., mesh_color2[2] / 255., alpha))
    camera_translation[0] *= -1.

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh_gt = trimesh.Trimesh(gt_vertices, faces, process=False)

    if correct_ori:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh_gt.apply_transform(rot)

    if sideview_angle > 0:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(sideview_angle), [0, 1, 0])
        mesh.apply_transform(rot)
        mesh_gt.apply_transform(rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    mesh_gt = pyrender.Mesh.from_trimesh(mesh_gt, material=material2)


    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=np.ones(3) * 0)
    scene.add(mesh, 'mesh')
    scene.add(mesh_gt, 'mesh_gt')

    if add_ground_plane:
        ground_mesh = pyrender.Mesh.from_trimesh(
            get_checkerboard_plane(),
            smooth=False,
        )
        pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        # pose[:3, 3] = [0, -1, 0]
        pose[:3, 3] = np.array([0, mesh.bounds[0, 1], 0])

        scene.add(ground_mesh, pose=pose, name='ground_plane')

    camera_pose = np.eye(4)
    if camera_rotation is not None:
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] =  camera_rotation @ camera_translation
    else:
        camera_pose[:3, 3] =  camera_translation

    camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
                                       cx=camera_center[0], cy=camera_center[1])

    scene.add(camera, pose=camera_pose)
    # Create light source
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    # for DirectionalLight, only rotation matters
    light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
    scene.add(light, pose=light_pose)
    light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
    scene.add(light, pose=light_pose)


    # light_pose = np.eye(4)

    # light_pose[:3, 3] = np.array([0, -1, 1])
    # scene.add(light, pose=light_pose)

    # light_pose[:3, 3] = np.array([0, 1, 1])
    # scene.add(light, pose=light_pose)

    # light_pose[:3, 3] = np.array([1, 1, 2])
    # scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=image.shape[1],
        viewport_height=image.shape[0],
        point_size=1.0
    )

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    valid_mask = (rend_depth > 0)[:, :, None]
    output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
    return output_img


def render_image_group(
        image: np.ndarray,
        camera_translation: torch.tensor,
        vertices: torch.tensor,
        gt_vertices: torch.tensor,
        camera_rotation: torch.tensor,
        focal_length: Tuple,
        camera_center: Tuple,
        mesh_color: str = 'pinkish',
        alpha: float = 1.0,
        faces: np.ndarray = None,
        mesh_filename: str = None,
        save_filename: str = None,
        keypoints_2d: np.ndarray = None,
        cam_params: np.ndarray = None,
        correct_ori: bool = True,
):
    to_numpy = lambda x: x.detach().cpu().numpy()

    if np.max(image) > 10:
        image = image / 255.


    camera_translation = (camera_translation)
    if camera_rotation is not None:
        camera_rotation = (camera_rotation)
    vertices = (vertices)

    # input image to this step should be between [0,1]
    overlay_img = render_overlay_image(
        image=image,
        camera_translation=camera_translation,
        vertices=vertices,
        camera_rotation=camera_rotation,
        focal_length=focal_length,
        camera_center=camera_center,
        mesh_color=mesh_color,
        alpha=alpha,
        faces=faces,
        mesh_filename=mesh_filename,
        sideview_angle=0,
        add_ground_plane=False,
        correct_ori=correct_ori,
    )

    # side_img = render_overlay_image_multiple(
    #     image=np.zeros_like(image),
    #     camera_translation=camera_translation,
    #     vertices=vertices,
    #     gt_vertices=gt_vertices,
    #     camera_rotation=camera_rotation,
    #     focal_length=focal_length,
    #     camera_center=camera_center,
    #     mesh_color=mesh_color,
    #     alpha=alpha,
    #     faces=faces,
    #     mesh_filename=mesh_filename,
    #     sideview_angle=0,
    #     correct_ori=correct_ori,

    # )

    # side_img2 = render_overlay_image_multiple(
    #     image=np.zeros_like(image),
    #     camera_translation=camera_translation,
    #     vertices=vertices,
    #     gt_vertices=gt_vertices,
    #     camera_rotation=camera_rotation,
    #     focal_length=focal_length,
    #     camera_center=camera_center,
    #     mesh_color=mesh_color,
    #     alpha=alpha,
    #     faces=faces,
    #     mesh_filename=mesh_filename,
    #     sideview_angle=270,
    #     correct_ori=correct_ori,

    # )

    # concatenate images horizontally
    output_img = np.concatenate([image, overlay_img], axis=0)

    if save_filename is not None:
        images_save = output_img * 255
        images_save = np.clip(images_save, 0, 255).astype(np.uint8)
        cv2.imwrite(save_filename, cv2.cvtColor(images_save, cv2.COLOR_BGR2RGB))

    return output_img #, overlay_img


class RendererCam:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self, img_res=224, faces=None, mesh_color='pink'):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res,
            viewport_height=img_res,
            point_size=1.0
        )
        # self.focal_length = focal_length
        self.img_res = img_res
        # self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.mesh_color = get_colors()[mesh_color]

    def _set_focal_length(self, focal_length):
        self.focal_length = focal_length

    def _set_camera_center(self, cam_center):
        self.camera_center = cam_center

    def _set_mesh_color(self, mesh_color):
        self.mesh_color = get_colors()[mesh_color]

    def _update_renderer(self, img_res):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res[1],
            viewport_height=img_res[0],
            point_size=1.0
        )

    def visualize_tb(
            self,
            vertices,
            camera_translation,
            images,
            kp_2d=None,
            heatmaps=None,
            segm_masks=None,
            skeleton_type='smpl',
            nb_max_img=8,
            sideview=False,
            vertex_colors=None,
            joint_labels=None,
            joint_uncertainty=None,
            alpha=1.0,
            camera_rotation=None,
            focal_length=None,
            cam_center=None,
            multi_sideview=False,
            mesh_filename=None,
            unnormalize_keypoints=True,
    ):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()

        if camera_rotation is not None:
            camera_rotation = camera_rotation.cpu().numpy()

        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))

        if kp_2d is not None:
            kp_2d = kp_2d.cpu().numpy()

        rend_imgs = []
        nb_max_img = min(nb_max_img, vertices.shape[0])
        num_sideview = 0
        for i in range(nb_max_img):
            rend_img = torch.from_numpy(
                np.transpose(self.__call__(
                    vertices[i],
                    camera_translation[i],
                    images_np[i],
                    vertex_colors=None if vertex_colors is None else vertex_colors[i],
                    joint_labels=None if joint_labels is None else joint_labels[i],
                    alpha=alpha,
                    camera_rotation=camera_rotation[i],
                    focal_length=None if focal_length is None else focal_length[i],
                    cam_center=None if cam_center is None else cam_center[i],
                    mesh_filename=mesh_filename,
                ), (2, 0, 1))
            ).float()

            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)

            if sideview:
                if multi_sideview:
                    for angle in [270, 180, 90]:
                        side_img = torch.from_numpy(
                            np.transpose(
                                self.__call__(
                                    vertices[i],
                                    camera_translation[i],
                                    np.ones_like(images_np[i]),
                                    vertex_colors=None if vertex_colors is None else vertex_colors[i],
                                    joint_labels=None if joint_labels is None else joint_labels[i],
                                    alpha=alpha,
                                    sideview=True,
                                    camera_rotation=camera_rotation[i],
                                    focal_length=None if focal_length is None else focal_length[i],
                                    cam_center=None if cam_center is None else cam_center[i],
                                    sideview_angle=angle,
                                    mesh_filename=mesh_filename.replace('.obj', f'_{angle:03d}_rot.obj')
                                    if mesh_filename else None
                                ),
                                (2, 0, 1)
                            )
                        ).float()
                        rend_imgs.append(side_img)
                        num_sideview += 1
                else:
                    side_img = torch.from_numpy(
                        np.transpose(
                            self.__call__(
                                vertices[i],
                                camera_translation[i],
                                np.ones_like(images_np[i]),
                                vertex_colors=None if vertex_colors is None else vertex_colors[i],
                                joint_labels=None if joint_labels is None else joint_labels[i],
                                alpha=alpha,
                                sideview=True,
                                camera_rotation=camera_rotation[i],
                                focal_length=None if focal_length is None else focal_length[i],
                                cam_center=None if cam_center is None else cam_center[i],
                                mesh_filename=mesh_filename.replace('.obj', f'_270_rot.obj')
                                if mesh_filename else None
                            ),
                            (2, 0, 1)
                        )
                    ).float()
                    rend_imgs.append(side_img)
                    num_sideview += 1


        nrow = 2
        if kp_2d is not None: nrow += 1
        if sideview: nrow += num_sideview
        if joint_labels is not None: nrow += 1
        if joint_uncertainty is not None: nrow += 1
        if heatmaps is not None: nrow += 1
        if segm_masks is not None: nrow += 1

        # nrow = len(rend_imgs)

        rend_imgs = make_grid(rend_imgs, nrow=nrow)
        return rend_imgs

    def __call__(
            self, vertices, camera_translation, image, vertex_colors=None,
            sideview=False, joint_labels=None, alpha=1.0, camera_rotation=None,
            sideview_angle=270, mesh_filename=None, mesh_inp=None,
            focal_length=None, cam_center=None,
    ):

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(self.mesh_color[0] / 255., self.mesh_color[1] / 255., self.mesh_color[2] / 255., alpha))

        camera_translation[0] *= -1.

        # if camera_rotation is not None:
        #     rot_mat = camera_rotation.squeeze()
        #     translation = camera_translation.squeeze()
        #     vertices = np.matmul(vertices, rot_mat.T)

        if mesh_inp:
            mesh = mesh_inp
        else:
            if vertex_colors is not None:
                mesh = trimesh.Trimesh(vertices, self.faces, vertex_colors=vertex_colors, process=False)
            else:
                mesh = trimesh.Trimesh(vertices, self.faces, vertex_colors=vertex_colors, process=False)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        # if camera_rotation is not None:
        #     mesh_pose = np.eye(4)
        #     mesh_pose[:3, :3] = camera_rotation
        #     mesh.apply_transform(mesh_pose)

        if sideview:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(sideview_angle), [0, 1, 0])
            mesh.apply_transform(rot)

        if mesh_filename:
            mesh.export(mesh_filename)
            if not mesh_filename.endswith('_rot.obj'):
                np.save(mesh_filename.replace('.obj', '.npy'), camera_translation)

        if vertex_colors is not None:
            mesh = pyrender.Mesh.from_trimesh(mesh)
        elif mesh_inp is not None:
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        else:
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        if camera_rotation is not None:
            camera_pose[:3, :3] = camera_rotation
            camera_pose[:3, 3] = camera_rotation @ camera_translation
        else:
            camera_pose[:3, 3] = camera_translation

        if focal_length is None:
            camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                               cx=self.camera_center[0], cy=self.camera_center[1])
        elif cam_center is None:
            camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
                                               cx=self.camera_center[0], cy=self.camera_center[1])
        else:
            camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
                                               cx=cam_center[1], cy=cam_center[0])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        # if joint_labels is not None:
        #     for joint, err in joint_labels.items():
        #         add_joints(scene, joints=joint, radius=err)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:, :, None]
        output_img = (color[:, :, :3] * valid_mask +
                      (1 - valid_mask) * image)
        return output_img


def add_joints(scene, joints, radius=0.005, color=[0.1, 0.1, 0.9, 1.0]):
    sm = trimesh.creation.uv_sphere(radius=radius)
    sm.visual.vertex_colors = color
    tfs = np.tile(np.eye(4), (1, 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)
