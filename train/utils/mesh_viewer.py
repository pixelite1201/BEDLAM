import trimesh
import pyrender
import numpy as np
from PIL import Image

from .mesh_utils import get_checkerboard_plane, get_world_mesh_list


class MeshViewer(object):

    def __init__(
            self,
            width=1200,
            height=800,
            body_color=(1.0, 1.0, 0.9, 1.0),
            bg_color=(0.0, 0.0, 0.0, 1.0),
            add_ground_plane=False,
            add_world_coord=False,
            add_cam_mesh=False,
            image=None,
            registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        # import trimesh
        # import pyrender

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.body_color = body_color
        self.scene = pyrender.Scene(bg_color=bg_color,
                                    ambient_light=(0.3, 0.3, 0.3))

        self.add_ground_plane = add_ground_plane
        self.add_world_coord = add_world_coord
        self.add_cam_mesh = add_cam_mesh

        if add_ground_plane:
            ground_mesh = pyrender.Mesh.from_trimesh(
                get_checkerboard_plane(),
                smooth=False,
            )
            pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
            pose[:3, 3] = [0,-1,0]
            self.scene.add(ground_mesh, pose=pose, name='ground_plane')

        if add_world_coord:
            world_mesh = pyrender.Mesh.from_trimesh(get_world_mesh_list(add_plane=False), smooth=False)
            self.scene.add(world_mesh)

        if add_cam_mesh:
            cam_mesh_pose = np.eye(4)
            cam_mesh_pose[:3, 3] = np.array([0, 0, 3])
            cam_mesh = pyrender.Mesh.from_trimesh(
                get_world_mesh_list(axisRadius=0.01, axisHeight=0.2, add_plane=False),
                smooth=False,
            )
            self.scene.add(cam_mesh, name='cam_mesh', pose=cam_mesh_pose)

        if image is not None:
            self.add_image(image=image)

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0,
                                        aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3])
        self.scene.add(pc, pose=camera_pose)

        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True,
                                      viewport_size=(width, height),
                                      cull_faces=False,
                                      run_in_thread=True,
                                      registered_keys=registered_keys)

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                    wireframe=False, transform=None):

        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces)

        # rotate around x axis 180
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        if transform is not None:
            # rot = self.transf(np.radians(270), [1, 0, 0])
            mesh.apply_transform(transform)

        return self.trimesh_to_pymesh(mesh, material=material)

    def update_mesh(self, vertices, faces, transform=None, cam_pose=None, image=None):
        if not self.viewer.is_active:
            return

        self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name == 'body_mesh':
                self.scene.remove_node(node)
                break

        body_mesh = self.create_mesh(
            vertices, faces, color=self.body_color, transform=transform
        )
        self.scene.add(body_mesh, name='body_mesh')

        if self.add_cam_mesh:
            if cam_pose is not None:
                for node in self.scene.get_nodes():
                    if node.name == 'cam_mesh':
                        self.scene.set_pose(node, pose=cam_pose)

        if self.add_ground_plane:
            for node in self.scene.get_nodes():
                if node.name == 'ground_plane':
                    plane_pose = self.scene.get_pose(node)
                    plane_pose[:3, 3] = np.array([0, body_mesh.bounds[0,1], 0])
                    self.scene.set_pose(node, pose=plane_pose)

        self.viewer.render_lock.release()

    def add_image(self, image, cam_pose=None):

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

        screen_mesh = pyrender.Mesh.from_trimesh(screen, smooth=True)

        self.scene.add(screen_mesh, name='img_mesh')