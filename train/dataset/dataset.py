import cv2
import os
import torch
import pickle
import numpy as np
from loguru import logger
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from skimage.transform import resize
from ..core import constants, config
from ..core.constants import NUM_JOINTS_SMPLX
from ..core.config import DATASET_FILES, DATASET_FOLDERS
from ..utils.image_utils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa, random_crop, read_img
from smplx import SMPL, SMPLX

class DatasetHMR(Dataset):

    def __init__(self, options, dataset, use_augmentation=True, is_train=True):
        super(DatasetHMR, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN,
                                       std=constants.IMG_NORM_STD)
        self.data = np.load(DATASET_FILES[is_train][dataset],
                            allow_pickle=True)
        self.imgname = self.data['imgname']
        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        if self.is_train:
            if '3dpw-train-smplx' in self.dataset:
                self.pose_cam = self.data['smplx_pose'][:, :NUM_JOINTS_SMPLX*3].astype(np.float)
                self.betas = self.data['smplx_shape'][:, :11].astype(np.float)
            else:
                self.pose_cam = self.data['pose_cam'][:, :NUM_JOINTS_SMPLX*3].astype(np.float)
                self.betas = self.data['shape'].astype(np.float)

            # For AGORA and 3DPW num betas are 10
            if self.betas.shape[-1] == 10:
                self.betas = np.hstack((self.betas, np.zeros((self.betas.shape[0], 1))))

            if 'cam_int' in self.data:
                self.cam_int = self.data['cam_int']
            else:
                self.cam_int = np.zeros((self.imgname.shape[0], 3, 3))
            if 'cam_ext' in self.data:
                self.cam_ext = self.data['cam_ext']
            else:
                self.cam_ext = np.zeros((self.imgname.shape[0], 4, 4))
            if 'trans_cam' in self.data:
                self.trans_cam = self.data['trans_cam']

        else:
            if 'h36m' in self.dataset: # H36m doesn't have pose and shape param only 3d joints
                self.joints = self.data['S']
                self.pose_cam = np.zeros((self.imgname.shape[0], 66))
                self.betas = np.zeros((self.imgname.shape[0], 11))
            else:
                self.pose_cam = self.data['pose_cam'].astype(np.float)
                self.betas = self.data['shape'].astype(np.float)
        
        if self.is_train:
            if '3dpw-train-smplx' in self.dataset: # Only for 3dpw training
                self.joint_map = constants.joint_mapping(constants.COCO_18, constants.SMPL_24)
                self.keypoints = np.zeros((len(self.imgname), 24, 3))
                self.keypoints = self.data['gtkps'][:, self.joint_map]
                self.keypoints[:, self.joint_map == -1] = -2
            else:
                full_joints = self.data['gtkps']
                self.keypoints = full_joints[:, :24]
        else:
            self.keypoints = np.zeros((len(self.imgname), 24, 3))

        if 'proj_verts' in self.data:
            self.proj_verts = self.data['proj_verts']
        else:
            self.proj_verts = np.zeros((len(self.imgname), 437, 3))

        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm'
                                    else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

        # evaluation variables
        if not self.is_train:
            if 'width' in self.data: # For closeup image stored in rotated format
                self.width = self.data['width']
            self.joint_mapper_h36m = constants.H36M_TO_J14
            self.joint_mapper_gt = constants.J24_TO_J14
            self.J_regressor = torch.from_numpy(np.load(
                               config.JOINT_REGRESSOR_H36M)).float()
            self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                                  gender='male',
                                  create_transl=False)
            self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                                    gender='female',
                                    create_transl=False)
            self.smplx_male = SMPLX(config.SMPLX_MODEL_DIR,
                                    gender='male')
            self.smplx_female = SMPLX(config.SMPLX_MODEL_DIR,
                                      gender='female')
            self.smplx2smpl = pickle.load(open(config.SMPLX2SMPL, 'rb'))
            self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None],
                                           dtype=torch.float32)
        if self.is_train and 'agora' not in self.dataset and '3dpw' not in self.dataset: # first 80% is training set 20% is validation
            self.length = int(self.scale.shape[0] * self.options.CROP_PERCENT)
        else:
            self.length = self.scale.shape[0]
        logger.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def scale_aug(self):
        sc = 1            # scaling
        if self.is_train:
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.SCALE_FACTOR,
                    max(1-self.options.SCALE_FACTOR,
                    np.random.randn()*self.options.SCALE_FACTOR+1))
        return sc

    def rgb_processing(self, rgb_img_full, center, scale, img_res, kp2d=None):

        if self.is_train and self.options.ALB: 
            aug_comp = [A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                        A.ImageCompression(20, 100, p=0.1),
                        A.RandomRain(blur_value=4, p=0.1),
                        A.MotionBlur(blur_limit=(3, 15),  p=0.2),
                        A.Blur(blur_limit=(3, 10), p=0.1),
                        A.RandomSnow(brightness_coeff=1.5,
                        snow_point_lower=0.2, snow_point_upper=0.4)]
            aug_mod = [A.CLAHE((1, 11), (10, 10), p=0.2), A.ToGray(p=0.2),
                       A.RandomBrightnessContrast(p=0.2),
                       A.MultiplicativeNoise(multiplier=[0.5, 1.5],
                       elementwise=True, per_channel=True, p=0.2),
                       A.HueSaturationValue(hue_shift_limit=20,
                       sat_shift_limit=30, val_shift_limit=20,
                       always_apply=False, p=0.2),
                       A.Posterize(p=0.1),
                       A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                       A.Equalize(mode='cv', p=0.1)]
            albumentation_aug = A.Compose([A.OneOf(aug_comp,
                                           p=self.options.ALB_PROB),
                                           A.OneOf(aug_mod,
                                           p=self.options.ALB_PROB)])          
            rgb_img_full = albumentation_aug(image=rgb_img_full)['image']

        rgb_img = crop(rgb_img_full, center, scale, [img_res, img_res])

        rgb_img = np.transpose(rgb_img.astype('float32'),
                               (2, 0, 1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale):
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [self.options.IMG_RES,
                                   self.options.IMG_RES])
        kp[:, :-1] = 2. * kp[:, :-1] / self.options.IMG_RES - 1.
        kp = kp.astype('float32')
        return kp


    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        keypoints_orig = self.keypoints[index].copy()
        if self.options.proj_verts:
            proj_verts_orig = self.proj_verts[index].copy()
            item['proj_verts_orig'] = torch.from_numpy(proj_verts_orig).float()
            proj_verts = self.proj_verts[index].copy()
        # Apply scale augmentation
        sc = self.scale_aug()
        # apply crop augmentation
        if self.is_train and self.options.CROP_FACTOR > 0:
            rand_no = np.random.rand()
            if rand_no < self.options.CROP_PROB:
                center, scale = random_crop(center, scale,
                                            crop_scale_factor=1-self.options.CROP_FACTOR,
                                            axis='y')

        imgname = os.path.join(self.img_dir, self.imgname[index])
        try:
            cv_img = read_img(imgname)
        except Exception as E:
            print(E)
            logger.info(f'@{imgname}@ from {self.dataset}')
        if self.is_train and 'closeup' in self.dataset:

            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

        orig_shape = np.array(cv_img.shape)[:2]
        pose = self.pose_cam[index].copy()
        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.j2d_processing(keypoints, center, sc * scale)
        if self.options.proj_verts:
            proj_verts = self.j2d_processing(proj_verts, center, sc * scale)
            item['proj_verts'] = torch.from_numpy(proj_verts).float()
       
        # Process image
        try:
            img = self.rgb_processing(cv_img, center, sc*scale, kp2d=keypoints,
                                      img_res=self.options.IMG_RES)
        except Exception as E:
            logger.info(f'@{imgname} from {self.dataset}')
            print(E)

        img = torch.from_numpy(img).float()
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(self.betas[index]).float()
        item['imgname'] = imgname
        if self.is_train:
            if 'cam_int' in self.data.files:
                item['focal_length'] = torch.tensor([self.cam_int[index][0, 0], self.cam_int[index][1, 1]])
            if self.dataset == '3dpw-train-smplx':
                item['focal_length'] = torch.tensor([1961.1, 1969.2])
            # Will be 0 for 3dpw-train-smplx
            item['cam_ext'] = self.cam_ext[index]
            item['translation'] = self.cam_ext[index][:, 3]
            if 'trans_cam' in self.data.files:
                item['translation'][:3] += self.trans_cam[index]

        item['keypoints_orig'] = torch.from_numpy(keypoints_orig).float()
        item['keypoints'] = torch.from_numpy(keypoints).float()
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        if not self.is_train:
            if '3dpw' in self.dataset:
                if self.gender[index] == 1:
                    gt_smpl_out = self.smpl_female(
                                global_orient=item['pose'].unsqueeze(0)[:, :3],
                                body_pose=item['pose'].unsqueeze(0)[:, 3:],
                                betas=item['betas'].unsqueeze(0))
                    gt_vertices = gt_smpl_out.vertices
                else:
                    gt_smpl_out = self.smpl_male(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0))
                    gt_vertices = gt_smpl_out.vertices

                item['vertices'] = gt_vertices[0].float()
            elif 'rich' in self.dataset:
                if self.gender[index] == 1:
                    model = self.smpl_female
                    gt_smpl_out = self.smplx_female(
                                global_orient=item['pose'].unsqueeze(0)[:, :3],
                                body_pose=item['pose'].unsqueeze(0)[:, 3:NUM_JOINTS_SMPLX*3],
                                betas=item['betas'].unsqueeze(0),)
                    gt_vertices = gt_smpl_out.vertices 
                else:
                    model = self.smpl_male
                    gt_smpl_out = self.smplx_male(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:NUM_JOINTS_SMPLX*3],
                        betas=item['betas'].unsqueeze(0),
                    )
                    gt_vertices = gt_smpl_out.vertices

                gt_vertices = torch.matmul(self.smplx2smpl, gt_vertices)
                item['joints'] = torch.matmul(model.J_regressor, gt_vertices[0])
                item['vertices'] = gt_vertices[0].float()
            elif 'h36m' in self.dataset:
                item['joints'] = self.joints[index]
                item['vertices'] = torch.zeros((6890, 3)).float()
            else:
                item['vertices'] = torch.zeros((6890, 3)).float()

        if not self.is_train:
            item['dataset_index'] = self.options.VAL_DS.split('_').index(self.dataset)

        return item

    def __len__(self):
        if self.is_train and 'agora' not in self.dataset and '3dpw' not in self.dataset:
            return int(self.options.CROP_PERCENT * len(self.imgname))
        else:
            return len(self.imgname)
