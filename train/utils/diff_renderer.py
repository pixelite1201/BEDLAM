import torch
import numpy as np
import torch.nn as nn

from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        DirectionalLights,
        BlendParams,
        HardFlatShader,
        MeshRasterizer,
        TexturesVertex,
    )
from pytorch3d.structures import Meshes

from .smpl_uv import get_tenet_texture
from .image_utils import get_default_camera


class MeshRendererWithDepth(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        mask = (fragments.zbuf > -1).float()

        zbuf = fragments.zbuf.view(images.shape[0], -1)
        # print(images.shape, zbuf.shape)
        depth = (zbuf - zbuf.min(-1, keepdims=True).values) / \
                (zbuf.max(-1, keepdims=True).values - zbuf.min(-1, keepdims=True).values)
        depth = depth.reshape(*images.shape[:3] + (1,))

        images = torch.cat([images[:, :, :, :3], mask, depth], dim=-1)
        return images


class DifferentiableRenderer(nn.Module):
    def __init__(
            self,
            img_size,
            focal_length,
            device='cuda',
            background_color=(0.0, 0.0, 0.0),
            texture_mode='smplpix',
            smpl_faces=None,
    ):
        super(DifferentiableRenderer, self).__init__()
        self.x = 'a'
        self.img_size = img_size
        self.device = device
        self.focal_length = focal_length
        K, R = get_default_camera(focal_length, img_size)

        T = torch.tensor([[0, 0, 2.5 * self.focal_length / self.img_size]])
        self.background_color = background_color
        self.renderer = None
        smpl_faces = smpl_faces

        face_colors = get_tenet_texture(mode=texture_mode).to(device).float()
        vertex_colors = torch.from_numpy(
            np.load(f'data/body_models/smpl/{texture_mode}_vertex_colors.npy')[:,:3]
        ).unsqueeze(0).to(device).float()

        self.register_buffer('K', K)
        self.register_buffer('R', R)
        self.register_buffer('T', T)
        self.register_buffer('face_colors', face_colors)
        self.register_buffer('vertex_colors', vertex_colors)
        self.register_buffer('smpl_faces', smpl_faces)

        self.set_requires_grad(False)

    def set_requires_grad(self, val=False):
        self.K.requires_grad_(val)
        self.R.requires_grad_(val)
        self.T.requires_grad_(val)
        self.face_colors.requires_grad_(val)
        self.vertex_colors.requires_grad_(val)
        self.smpl_faces.requires_grad_(val)

    def forward(self, vertices, faces=None, R=None, T=None):
        raise NotImplementedError


class Pytorch3D(DifferentiableRenderer):
    def __init__(
            self,
            img_size,
            focal_length,
            device='cuda',
            background_color=(0.0, 0.0, 0.0),
            texture_mode='smplpix',
            smpl_faces=None,
    ):
        super(Pytorch3D, self).__init__(
            img_size,
            focal_length,
            device=device,
            background_color=background_color,
            texture_mode=texture_mode,
            smpl_faces=smpl_faces,
        )

        # this R converts the camera from pyrender NDC to
        # OpenGL coordinate frame. It is basicall R(180, X) x R(180, Y)
        # I manually defined it here for convenience
        self.R = self.R @ torch.tensor(
            [[[ -1.0,  0.0, 0.0],
              [  0.0, -1.0, 0.0],
              [  0.0,  0.0, 1.0]]],
            dtype=self.R.dtype, device=self.R.device,
        )

        cameras = PerspectiveCameras(
            device=self.device,
            focal_length=self.focal_length,
            principal_point=((self.img_size // 2, self.img_size // 2),),
            R=self.R,
            T=self.T,
            image_size=((self.img_size, self.img_size),)
        )

        for param in cameras.parameters():
            param.requires_grad_(False)

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = DirectionalLights(
            device=self.device,
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),),
            direction=((0, 1, 0),),
        )

        blend_params = BlendParams(background_color=self.background_color)

        shader = HardFlatShader(device=self.device,
                                cameras=cameras,
                                blend_params=blend_params,
                                lights=lights)

        self.textures = TexturesVertex(verts_features=self.vertex_colors)
        self.renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=shader,
        )

    def forward(self, vertices, faces=None, R=None, T=None):
        batch_size = vertices.shape[0]
        if faces is None:
            faces = self.smpl_faces.expand(batch_size, -1, -1)

        if R is None:
            R = self.R.expand(batch_size, -1, -1)

        if T is None:
            T = self.T.expand(batch_size, -1)

        # convert camera translation to pytorch3d coordinate frame
        T = torch.bmm(R, T.unsqueeze(-1)).squeeze(-1)

        textures = TexturesVertex(
            verts_features=self.vertex_colors.expand(batch_size, -1, -1)
        )
        # we may need to rotate the mesh
        meshes = Meshes(verts=vertices, faces=faces, textures=textures)
        images = self.renderer(meshes, R=R, T=T)
        images = images.permute(0, 3, 1, 2)
        return images


class NeuralMeshRenderer(DifferentiableRenderer):
    def __init__(self, *args, **kwargs):
        import neural_renderer as nr

        super(NeuralMeshRenderer, self).__init__(*args, **kwargs)

        self.neural_renderer = nr.Renderer(
            dist_coeffs=None,
            orig_size=self.img_size,
            image_size=self.img_size,
            light_intensity_ambient=1,
            light_intensity_directional=0,
            anti_aliasing=False,
        )

    def forward(self, vertices, faces=None, R=None, T=None):
        batch_size = vertices.shape[0]
        if faces is None:
            faces = self.smpl_faces.expand(batch_size, -1, -1)

        if R is None:
            R = self.R.expand(batch_size, -1, -1)

        if T is None:
            T = self.T.expand(batch_size, -1)
        rgb, depth, mask = self.neural_renderer(
            vertices,
            faces,
            textures=self.face_colors.expand(batch_size, -1, -1, -1, -1, -1),
            K=self.K.expand(batch_size, -1, -1),
            R=R,
            t=T.unsqueeze(1),
        )
        return torch.cat([rgb, depth.unsqueeze(1), mask.unsqueeze(1)], dim=1)