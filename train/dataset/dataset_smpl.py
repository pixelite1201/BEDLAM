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
from ..core.constants import NUM_JOINTS_SMPL, NUM_JOINTS_SMPLX
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
        if not self.is_train and 'h36m' in self.dataset:
            self.pose_3d = self.data['S']
        if 'betas' in self.data:
            self.betas = self.data['shape'].astype(np.float)
        else:
            self.betas = np.zeros((len(self.imgname), 10))

        if 'pose_cam' in self.data:
            self.pose = self.data['pose_cam'].astype(np.float)
        elif 'pose' in self.data:
            self.pose = self.data['pose'].astype(np.float)
        else:
            self.pose = np.zeros((len(self.imgname), 24, 3))
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm'
                                    else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

        # evaluation variables
        if not self.is_train:
            self.joint_mapper_h36m = constants.H36M_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.H36M_TO_J14
            self.joint_mapper_gt = constants.J24_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.J24_TO_J14
            self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
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

        if self.is_train:
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
            rgb_img_full = cv2.cvtColor(rgb_img_full, cv2.COLOR_BGR2RGB)
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
            logger.info(f'@{imgname} from {self.dataset}')

        orig_shape = np.array(cv_img.shape)[:2]
        pose = self.pose[index].copy()
        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.j2d_processing(keypoints, center, sc * scale)
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
        item['keypoints_orig'] = torch.from_numpy(keypoints_orig).float()
        item['keypoints'] = torch.from_numpy(keypoints).float()
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        img_h, img_w, _ = cv_img.shape
        estimated_fl = (img_w * img_w + img_h * img_h) ** 0.5

        if self.is_train:
            if 'h36m' in self.dataset:
                item['focal_length'] = np.array([1145, 1145]) # Approximate fl from h3.6m
            elif 'mpi-inf-3dhp' in self.dataset:
                item['focal_length'] = np.array([1500, 1500]) # Approximate fl from mpi-inf
            elif '3dpw' in self.dataset:
                item['focal_length'] = np.array([1965, 1965])
            else:
                item['focal_length'] = np.array([estimated_fl, estimated_fl])
        item['focal_length'] = np.array([estimated_fl, estimated_fl])
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
            else:
                item['vertices'] = torch.zeros((6890, 3)).float()

        if not self.is_train:
            item['dataset_index'] = self.options.VAL_DS.split('_').index(self.dataset)
        if not self.is_train and 'h36m' in self.dataset:
            item['S'] = self.pose_3d[index]
        return item

    def __len__(self):
        if self.is_train:
            return int(self.options.CROP_PERCENT * len(self.imgname))
        else:
            return len(self.imgname)
