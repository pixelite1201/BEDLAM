import torch
import trimesh
import numpy as np
import skimage.io as io
from PIL import Image
from smplx import SMPL
from matplotlib import cm as mpl_cm, colors as mpl_colors
from trimesh.visual.color import face_to_vertex_color, vertex_to_face_color

from ..core.config import SMPL_MODEL_DIR
from .colorwheel import make_color_wheel_image


def get_smpl_uv():
    uv_obj = 'data/body_models/smpl_uv_20200910/smpl_uv.obj'

    uv_map = []
    with open(uv_obj) as f:
        for line in f.readlines():
            if line.startswith('vt'):
                coords = [float(x) for x in line.split(' ')[1:]]
                uv_map.append(coords)

    uv_map = np.array(uv_map)

    return uv_map


def show_uv_texture():
    # image = io.imread('data/body_models/smpl_uv_20200910/smpl_uv_20200910.png')
    image = make_color_wheel_image(1024, 1024)
    image = Image.fromarray(image)

    uv = np.load('data/body_models/smpl_uv_20200910/uv_table.npy') # get_smpl_uv()
    material = trimesh.visual.texture.SimpleMaterial(image=image)
    tex_visuals = trimesh.visual.TextureVisuals(uv=uv, image=image, material=material)

    smpl = SMPL(SMPL_MODEL_DIR)

    faces = smpl.faces
    verts = smpl().vertices[0].detach().numpy()

    # assert(len(uv) == len(verts))
    print(uv.shape)
    vc = tex_visuals.to_color().vertex_colors
    fc = trimesh.visual.color.vertex_to_face_color(vc, faces)
    face_colors = fc.copy()
    fc = fc.astype(float)
    vc = vc.astype(float)
    fc[:,:3] = fc[:,:3] / 255.
    vc[:,:3] = vc[:,:3] / 255.
    print(fc[:,:3].max(), fc[:,:3].min(), fc[:,:3].mean())
    print(vc[:, :3].max(), vc[:, :3].min(), vc[:, :3].mean())
    np.save('data/body_models/smpl/color_wheel_face_colors.npy', fc)
    np.save('data/body_models/smpl/color_wheel_vertex_colors.npy', vc)
    print(fc.shape)
    mesh = trimesh.Trimesh(verts, faces, validate=True, process=False, face_colors=face_colors)
    # mesh = trimesh.load('data/body_models/smpl_uv_20200910/smpl_uv.obj', process=False)
    # mesh.visual = tex_visuals

    # import ipdb; ipdb.set_trace()
    # print(vc.shape)
    mesh.show()


def show_colored_mesh():
    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    smpl = SMPL(SMPL_MODEL_DIR)

    faces = smpl.faces
    verts = smpl().vertices[0].detach().numpy()

    m = trimesh.Trimesh(verts, faces, process=False)

    mode = 1
    if mode == 0:
        # mano_segm_labels = m.triangles_center
        face_labels = m.triangles_center
        face_colors = (face_labels - face_labels.min()) / np.ptp(face_labels)

    elif mode == 1:
        # print(face_labels.shape)
        face_labels = m.triangles_center
        face_labels = np.argsort(np.linalg.norm(face_labels, axis=-1))
        face_colors = np.ones((13776, 4))
        face_colors[:, 3] = 1.0
        face_colors[:, :3] = cm(norm_gt(face_labels))[:, :3]
    elif mode == 2:
        # breakpoint()
        fc = np.load('data/body_models/smpl_uv_20200910/data/vertex_texture.npy')[0, :, 0, 0, 0, :]
        face_colors = np.ones((13776, 4))
        face_colors[:, :3] = fc
    mesh = trimesh.Trimesh(verts, faces, process=False, face_colors=face_colors)
    mesh.show()


def get_tenet_texture(mode='smplpix'):
    # mode = 'smplpix', 'decomr'

    smpl = SMPL(SMPL_MODEL_DIR)

    faces = smpl.faces
    verts = smpl().vertices[0].detach().numpy()

    m = trimesh.Trimesh(verts, faces, process=False)
    if mode == 'smplpix':
        # mano_segm_labels = m.triangles_center
        face_labels = m.triangles_center
        face_colors = (face_labels - face_labels.min()) / np.ptp(face_labels)
        texture = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)
        texture[0, :, 0, 0, 0, :] = face_colors[:, :3]
        texture = torch.from_numpy(texture).float()
    elif mode == 'decomr':
        texture = np.load('data/body_models/smpl_uv_20200910/data/vertex_texture.npy')
        texture = torch.from_numpy(texture).float()
    elif mode == 'colorwheel':
        face_colors = np.load('data/body_models/smpl/color_wheel_face_colors.npy')
        texture = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)
        texture[0, :, 0, 0, 0, :] = face_colors[:, :3]
        texture = torch.from_numpy(texture).float()
    else:
        raise ValueError(f'{mode} is not defined!')


    return texture


def save_tenet_textures(mode='smplpix'):
    # mode = 'smplpix', 'decomr'

    smpl = SMPL(SMPL_MODEL_DIR)

    faces = smpl.faces
    verts = smpl().vertices[0].detach().numpy()

    m = trimesh.Trimesh(verts, faces, process=False)

    if mode == 'smplpix':
        # mano_segm_labels = m.triangles_center
        face_labels = m.triangles_center
        face_colors = (face_labels - face_labels.min()) / np.ptp(face_labels)
        texture = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)
        texture[0, :, 0, 0, 0, :] = face_colors[:, :3]
        texture = torch.from_numpy(texture).float()

        vert_colors = face_to_vertex_color(m, face_colors).astype(float) / 255.0

    elif mode == 'decomr':
        texture = np.load('data/body_models/smpl_uv_20200910/data/vertex_texture.npy')
        texture = torch.from_numpy(texture).float()
        face_colors = texture[0, :, 0, 0, 0, :]
        vert_colors = face_to_vertex_color(m, face_colors).astype(float) / 255.0

    elif mode == 'colorwheel':
        face_colors = np.load('data/body_models/smpl/color_wheel_face_colors.npy')
        texture = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)
        texture[0, :, 0, 0, 0, :] = face_colors[:, :3]
        texture = torch.from_numpy(texture).float()
        face_colors[:, :3] *= 255
        vert_colors = face_to_vertex_color(m, face_colors).astype(float) / 255.0
    else:
        raise ValueError(f'{mode} is not defined!')

    print(vert_colors.shape, vert_colors.max())
    np.save(f'data/body_models/smpl/{mode}_vertex_colors.npy', vert_colors)
    return texture