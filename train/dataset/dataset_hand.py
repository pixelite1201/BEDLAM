import torch
import numpy as np
import pickle
from os.path import join
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from skimage.transform import resize

from ..core import constants, config
from ..core.config import DATASET_FILES, DATASET_FOLDERS
from ..utils.image_utils import crop, transform, random_crop, read_img
from ..models.head.mano import MANO


class DatasetHand(Dataset):

    def __init__(self, options, dataset,
                 use_augmentation=True, is_train=True, num_images=0):
        super(DatasetHand, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]
        self.mano = MANO(model_path=config.MANO_MODEL_DIR, use_pca=False, is_rhand=True)
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(DATASET_FILES[is_train][dataset].replace('.npz', '-hands.npz').replace('all_npz_12_training','all_npz_12_hands'), allow_pickle=True)

        #Just taking right hand
        self.right_hand_detect = np.asarray(self.data['right_hand']).astype(np.bool)
        self.cam_int = self.data['cam_int'][self.right_hand_detect]
        self.scale = self.data['scale_hand'][self.right_hand_detect][:, 1]
        self.center = self.data['center_hand'][self.right_hand_detect][:, 1]
        self.pose_cam = self.data['right_hand_abs_pose'][self.right_hand_detect]
        self.betas = self.data['shape'][self.right_hand_detect][:, :10]
        self.imgname = self.data['imgname'][self.right_hand_detect]
        self.keypoints = self.data['gtkps'][self.right_hand_detect][:, [21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54], :2]
        if 'cam_ext' in self.data:
            self.cam_ext = self.data['cam_ext'][self.right_hand_detect]
        if 'trans_cam' in self.data:
            self.trans_cam = self.data['trans_cam'][self.right_hand_detect]

        self.use_augmentation = use_augmentation
        
        self.length = self.scale.shape[0]
        logger.info(f'Loaded {self.dataset} dataset, num samples {self.length}')


    def rgb_processing(self, rgb_img_full, center, scale, img_res, kp2d=None):

        rgb_img = crop(rgb_img_full, center, scale, [img_res, img_res])
        rgb_img_full = resize(rgb_img_full, [img_res, img_res])

        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        rgb_img_full = np.transpose(rgb_img_full.astype('float32'), (2, 0, 1)) / 255.0

        return rgb_img, rgb_img_full

    def j2d_processing(self, kp, center, scale):
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale, [self.options.IMG_RES, self.options.IMG_RES])
        kp[:, :] = 2. * kp[:, :] / self.options.IMG_RES - 1.
        kp = kp.astype('float32')
        return kp

    def scale_aug(self):
        sc = 1
        if self.is_train:
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1 + self.options.SCALE_FACTOR, max(1 - self.options.SCALE_FACTOR, 
                     np.random.randn() * self.options.SCALE_FACTOR+1))
        return sc

    def __getitem__(self, index):
        item = {}

        pose = self.pose_cam[index].copy()
        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(self.betas[index]).float()
        center = self.center[index]
        scale = self.scale[index]
        if 'cam_ext' in self.data.files:
            item['translation'] = self.cam_ext[index][:, 3][:3]
        if 'trans_cam' in self.data.files:
            item['translation'][:3] += self.trans_cam[index]
        output = self.mano(betas=item['betas'].unsqueeze(0),
                           global_orient=item['pose'][:3].unsqueeze(0),
                           right_hand_pose=item['pose'][3:].unsqueeze(0))
        item['vertices'] = output.vertices[0].detach().cpu().numpy()
        item['joints3d'] = output.joints[0].detach().cpu().numpy()

        keypoints = self.keypoints[index]

        keypoints = keypoints.copy()
        keypoints_orig = keypoints.copy()

        if self.is_train and self.options.CROP_FACTOR > 0:
            rand_no = np.random.rand()
            if rand_no < self.options.CROP_PROB:
                center, scale = random_crop(center, scale, crop_scale_factor=1 - self.options.CROP_FACTOR, axis='y')
        sc = self.scale_aug()

        imgname = join(self.img_dir, self.imgname[index])
        try:
            cv_img = read_img(imgname)
        except Exception as E:
            logger.info(E)
            logger.info(imgname)

        orig_shape = np.array(cv_img.shape)[:2]

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.j2d_processing(keypoints, center, sc * scale)

        # Process image
        img, img_full = self.rgb_processing(cv_img, center, sc * scale, kp2d=keypoints, img_res=self.options.IMG_RES)

        img = torch.from_numpy(img).float()
        img_full = torch.from_numpy(img_full).float()

        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['img_full'] = self.normalize_img(img_full)
        item['imgname'] = imgname
        item['focal_length'] = torch.tensor([self.cam_int[index][0, 0], self.cam_int[index][1, 1]])
        item['keypoints_orig'] = torch.from_numpy(keypoints_orig).float()
        item['keypoints'] = torch.from_numpy(keypoints).float()
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
 
        if not self.is_train:
            item['dataset_index'] = self.options.VAL_DS.split('_').index(self.dataset)

        return item

    def __len__(self):
        return len(self.imgname)
