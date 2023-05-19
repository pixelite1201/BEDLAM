
import os
import cv2
import torch
import tqdm
from loguru import logger
import numpy as np
from . import constants
from multi_person_tracker import MPT
from torchvision.transforms import Normalize
from glob import glob
from train.utils.train_utils import load_pretrained_model
from train.utils.vibe_image_utils import get_single_image_crop_demo

from .config import update_hparams
from ..models.hmr import HMR
from ..models.head.smplx_cam_head import SMPLXCamHead
from ..utils.renderer_cam import render_image_group
from ..utils.renderer_pyrd import Renderer
from ..utils.image_utils import crop


class Tester:
    def __init__(self, args):
        self.args = args
        self.model_cfg = update_hparams(args.cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.bboxes_dict = {}

        self.model = self._build_model()
        self.smplx_cam_head = SMPLXCamHead(img_res=self.model_cfg.DATASET.IMG_RES).to(self.device)
        self._load_pretrained_model()
        self.model.eval()

    def _build_model(self):
        self.hparams = self.model_cfg
        model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        ).to(self.device)
        return model

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        logger.info(f'Loading pretrained model from {self.args.ckpt}')
        ckpt = torch.load(self.args.ckpt)['state_dict']
        load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
        logger.info(f'Loaded pretrained weights from \"{self.args.ckpt}\"')

    def run_detector(self, all_image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=False,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = []
        for fold_id, image_folder in enumerate(all_image_folder):
            bboxes.append(mot.detect(image_folder))

        return bboxes

    def load_yolov5_bboxes(self, all_bbox_folder):
        # run multi object tracker
        for fold_id, bbox_folder in enumerate(all_bbox_folder):
            for bbox_file in os.listdir(bbox_folder):
                bbox = np.loadtxt(os.path.join(bbox_folder, bbox_file))
                fname = os.path.join('/'.join(bbox_folder.split('/')[-3:-1]),bbox_file.replace('.txt','.png'))
                self.bboxes_dict[fname] = bbox

    @torch.no_grad()
    def run_on_image_folder(self, all_image_folder, detections, output_folder, visualize_proj=True):
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
            ]
            image_file_names = (sorted(image_file_names))
            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                
                dets = detections[fold_idx][img_idx]
                if len(dets) < 1:
                    continue

                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
                                         self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []

                for det_idx, det in enumerate(dets):
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
                    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
                hmr_output = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)

                focal_length = (img_w * img_w + img_h * img_h) ** 0.5
                pred_vertices_array = (hmr_output['vertices'] + hmr_output['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
                renderer = Renderer(focal_length=focal_length[0], img_w=img_w[0], img_h=img_h[0],
                                    faces=self.smplx_cam_head.smplx.faces,
                                    same_mesh_color=False)
                front_view = renderer.render_front_view(pred_vertices_array,
                                                        bg_img_rgb=img.copy())

                # save rendering results
                basename = img_fname.split('/')[-1]
                filename = basename + "pred_%s.jpg" % 'bedlam'
                filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                front_view_path = os.path.join(output_folder, filename)
                orig_path = os.path.join(output_folder, filename_orig)
                logger.info(f'Writing output files to {output_folder}')
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                cv2.imwrite(orig_path, img[:, :, ::-1])
                renderer.delete()

    @torch.no_grad()
    def run_on_hbw_folder(self, all_image_folder, detections, output_folder, data_split='test', visualize_proj=True):
        img_names = []
        verts = []
        image_file_names = []
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
            ]
            image_file_names = (sorted(image_file_names))
            print(image_folder, len(image_file_names))

            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                if detections:
                    dets = detections[fold_idx][img_idx]
                    if len(dets) < 1:
                        img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                        template_verts = self.smplx_cam_head.smplx().vertices[0].detach().cpu().numpy()
                        verts.append(template_verts)
                        continue
                else:
                    match_fname = '/'.join(img_fname.split('/')[-3:])
                    if match_fname not in self.bboxes_dict.keys():
                        img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                        template_verts = self.smplx_cam_head.smplx().vertices[0].detach().cpu().numpy()
                        verts.append(template_verts)
                        continue
                    dets = self.bboxes_dict[match_fname]
                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(1, 3, self.model_cfg.DATASET.IMG_RES,
                                         self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)
                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []
                if len(dets.shape)==1:
                    dets = np.expand_dims(dets, 0)
                for det_idx, det in enumerate(dets):
                    if det_idx>=1:
                        break
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
                    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()

                hmr_output = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
                img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                template_verts = self.smplx_cam_head.smplx(betas=hmr_output['pred_shape'], pose2rot=False).vertices[0].detach().cpu().numpy()
                verts.append(template_verts)
                if visualize_proj:
                    focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                    pred_vertices_array = (hmr_output['vertices'][0] + hmr_output['pred_cam_t']).unsqueeze(0).detach().cpu().numpy()
                    renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                        faces=self.smplx_cam_head.smplx.faces,
                                        same_mesh_color=False)
                    front_view = renderer.render_front_view(pred_vertices_array,
                                                            bg_img_rgb=img.copy())

                    # save rendering results
                    basename = img_fname.split('/')[-3]+'_'+img_fname.split('/')[-2]+'_'+img_fname.split('/')[-1]
                    filename = basename + "pred_%s.jpg" % 'bedlam'
                    filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                    front_view_path = os.path.join(output_folder, filename)
                    orig_path = os.path.join(output_folder, filename_orig)
                    logger.info(f'Writing output files to {output_folder}')
                    cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                    cv2.imwrite(orig_path, img[:, :, ::-1])
                    renderer.delete()
        np.savez(os.path.join(output_folder, data_split + '_hbw_prediction.npz'), image_name=img_names, v_shaped=verts)
               
    def run_on_dataframe(self, dataframe_path, output_folder, visualize_proj=True):
        dataframe = np.load(dataframe_path)
        centers = dataframe['center']
        scales = dataframe['scale']
        image = dataframe['image']
        for ind, center in tqdm.tqdm(enumerate(centers)):
            center = centers[ind]
            scale = scales[ind]
            img = image[ind]
            orig_height, orig_width = img.shape[:2]
            rgb_img = crop(img, center, scale, [self.hparams.DATASET.IMG_RES, self.hparams.DATASET.IMG_RES])

            rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
            rgb_img = torch.from_numpy(rgb_img).float().cuda()
            rgb_img = self.normalize_img(rgb_img)

            img_h = torch.tensor(orig_height).repeat(1).cuda().float()
            img_w = torch.tensor(orig_width).repeat(1).cuda().float()
            center = torch.tensor(center).cuda().float()
            scale = torch.tensor(scale).cuda().float()

            hmr_output = self.model(rgb_img.unsqueeze(0), bbox_center=center.unsqueeze(0), bbox_scale=scale.unsqueeze(0), img_w=img_w, img_h=img_h)
            # Need to convert SMPL-X meshes to SMPL using conversion tool before calculating error
            import trimesh
            mesh = trimesh.Trimesh(vertices=hmr_output['vertices'][0].detach().cpu().numpy(),faces=self.smplx_cam_head.smplx.faces)
            output_mesh_path = os.path.join(output_folder, str(ind)+'.obj')
            mesh.export(output_mesh_path)

            if visualize_proj:
                focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                pred_vertices_array = (hmr_output['vertices'][0] + hmr_output['pred_cam_t']).unsqueeze(0).detach().cpu().numpy()
                renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                    faces=self.smplx_cam_head.smplx.faces,
                                    same_mesh_color=False)
                front_view = renderer.render_front_view(pred_vertices_array,
                                                        bg_img_rgb=img.copy())

                # save rendering results
                basename = str(ind)
                filename = basename + "pred_%s.jpg" % 'bedlam'
                filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                front_view_path = os.path.join(output_folder, filename)
                orig_path = os.path.join(output_folder, filename_orig)
                logger.info(f'Writing output files to {output_folder}')
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                cv2.imwrite(orig_path, img[:, :, ::-1])
                renderer.delete()