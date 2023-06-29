import cv2
import os
import sys
import smplx
import torch
import numpy as np
from train.core.config import DATASET_FOLDERS, DATASET_FILES
from train.utils.renderer_pyrd import Renderer
MODEL_FOLDER = 'data/body_models/smplx/models/'
SCENES = ['agora-bfh', 'agora-body', 'zoom-suburbd', 'closeup-suburba', 'closeup-suburbb', 'closeup-suburbc', 'closeup-suburbd',
        'closeup-gym', 'zoom-gym', 'static-gym', 'static-office', 'orbit-office', 'orbit-archviz-15', 'orbit-archviz-19', 'orbit-archviz-12',
        'orbit-archviz-10', 'static-hdri', 'static-hdri-zoomed', 'staticzoomed-suburba-frameocc', 'zoom-suburbb-frameocc', 'static-hdri-frameocc',
        'orbit-archviz-objocc', 'pitchup-stadium', 'static-hdri-bmi', 'closeup-suburbb-bmi', 'closeup-suburbc-bmi', 'static-suburbd-bmi', 'zoom-gym-bmi',
        'pitchdown-stadium', 'static-office-hair', 'zoom-suburbd-hair', 'static-gym-hair', 'orbit-stadium-bmi']

smplx_model_male = smplx.create(MODEL_FOLDER, model_type='smplx',
                            gender='neutral',
                            ext='npz',
                            flat_hand_mean=True,
                            num_betas=11,
                            use_pca=False)
smplx_model_female = smplx.create(MODEL_FOLDER, model_type='smplx',
                                  gender='female',
                                  ext='npz',
                                  num_betas=11,
                                  flat_hand_mean=True,
                                  use_pca=False)  

smplx_model_neutral = smplx.create(MODEL_FOLDER, model_type='smplx',
                                   gender='neutral',
                                   ext='npz',
                                   flat_hand_mean=True,
                                   num_betas=11,
                                   use_pca=False)  

def get_smplx_vertices(poses, betas, trans, gender):

    if gender == 'male':
        model_out = smplx_model_male(betas=torch.tensor(betas).unsqueeze(0).float(),
                              global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                              body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                              left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
                              right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
                              jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                              leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                              reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                              transl=torch.tensor(trans).unsqueeze(0))
# from psbody.mesh import Mesh
    elif gender == 'female':
        model_out = smplx_model_female(betas=torch.tensor(betas).unsqueeze(0).float(),
                              global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                              body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                              left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
                              right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
                              jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                              leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                              reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                              transl=torch.tensor(trans).unsqueeze(0))
    elif gender == 'neutral':
        model_out = smplx_model_neutral(betas=torch.tensor(betas).unsqueeze(0).float(),
                              global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                              body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                              left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
                              right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
                              jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                              leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                              reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                              transl=torch.tensor(trans).unsqueeze(0))
    else:
        print('Please provide gender as male or female')
    return model_out.vertices[0], model_out.joints[0]

    
def get_transform(center, scale, res, rot=0):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    # Upper left point

    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    from skimage.transform import resize
    # resize image
    new_img = resize(new_img, res)
    return img, new_img


def visualize_crop(image_path, center, scale, verts, focal_length, 
                   output_dir, rotate_flag=False):
    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w, c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                        faces=smplx_model_neutral.faces)
    front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
                                            bg_img_rgb=img[:, :, ::-1].copy())

    img, crop_img = crop(front_view[:, :, ::-1], center, scale, res=(224, 224))

    cv2.imwrite(os.path.join(output_dir, 'crop_'+str(i)+'_'+image_path.split('/')[-4]+image_path.split('/')[-1]), crop_img)


def visualize(image_path, verts, focal_length, output_dir, rotate_flag=False):

    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w, c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                        faces=smplx_model_neutral.faces)
    front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
                                            bg_img_rgb=img[:, :, ::-1].copy())

    cv2.imwrite(os.path.join(output_dir, str(i)+'_'+image_path.split('/')[-4]+image_path.split('/')[-1]), front_view[:, :, ::-1])


if __name__ == '__main__':

    scene_name = 'static-hdri' # You can provide a different scene. All scenes are listed in SCENES
    output_dir = sys.argv[1]   # Output folder to save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if 'closeup' in scene_name: # Since the original image are rotated
        rotate_flag = True
    else:
        rotate_flag = False
    base_img_path = DATASET_FOLDERS[scene_name]
    npz_file = DATASET_FILES[1][scene_name]
    dataframe = np.load(npz_file)
    for i, imgname in enumerate(dataframe['imgname']):
        pose = dataframe['pose_cam'][i]
        beta = dataframe['shape'][i]
        body_trans_cam = dataframe['trans_cam'][i]
        verts_3d, joints_3d = get_smplx_vertices(pose, beta, body_trans_cam, 'neutral')
        cam_trans = dataframe['cam_ext'][i][:, 3][:3]
        verts_3d = verts_3d.detach().cpu() + cam_trans
        focal_length = dataframe['cam_int'][i][0][0]
        visualize(os.path.join(base_img_path, imgname), verts_3d, focal_length, output_dir, rotate_flag)

        # # Visualize crop used in training
        center = dataframe['center'][i]
        scale = dataframe['scale'][i]
        visualize_crop(os.path.join(base_img_path, imgname), center, scale, verts_3d, focal_length, output_dir, rotate_flag)

        # Using world pose/ To understand camera extrinsic and world pose
        pose_world = dataframe['pose_world'][i]
        body_trans_world = dataframe['trans_world'][i]
        cam_ext = dataframe['cam_ext'][i]
        vertices3d_world, joints3d_world = get_smplx_vertices(pose_world, beta, body_trans_world, 'neutral')
        vertices3d_world = vertices3d_world.detach().cpu().numpy()
        vertices3d = np.matmul(cam_ext[:3, :3], vertices3d_world.T).T
        verts_cam = vertices3d + cam_trans
        visualize(os.path.join(base_img_path, imgname), torch.tensor(verts_cam), focal_length, output_dir, rotate_flag)
