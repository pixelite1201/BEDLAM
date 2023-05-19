import os
import sys
import argparse
from loguru import logger
from glob import glob
from train.core.testerx import Tester

os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.append('')

def main(args):

    input_image_folder = args.image_folder
    output_path = args.output_folder
    os.makedirs(output_path, exist_ok=True)

    logger.add(
        os.path.join(output_path, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')

    tester = Tester(args)
    if args.eval_dataset == 'bedlam':
        all_image_folder = glob(os.path.join(input_image_folder, '*', 'png', '*'))
        detections = tester.run_detector(all_image_folder)
        tester.run_on_image_folder(all_image_folder, detections, output_path, args.display, args.save_result, args.eval_dataset)
    elif args.eval_dataset == 'agora':
        all_image_folder = [input_image_folder]
        detections = tester.run_detector(all_image_folder)
        tester.run_on_image_folder(all_image_folder, detections, output_path, args.display, args.save_result, args.eval_dataset)
    else:
        all_image_folder = [input_image_folder]
        detections = tester.run_detector(all_image_folder)
        tester.run_on_image_folder(all_image_folder, detections, output_path, args.display, args.save_result)

    del tester.model

    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/demo_bedlam_cliff_x.yaml',
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='data/ckpt/bedlam_cliff_x.ckpt',
                        help='checkpoint path')

    parser.add_argument('--image_folder', type=str, default='demo_images',
                        help='input image folder')

    parser.add_argument('--output_folder', type=str, default='demo_images/results',
                        help='output folder to write results')

    parser.add_argument('--tracker_batch_size', type=int, default=1,
                        help='batch size of object detector used for bbox tracking')
                        
    parser.add_argument('--display', action='store_true',
                        help='visualize the 3d body projection on image')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--dataframe_path', type=str, default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--save_result', action='store_true', help='Save verts, joints, joints2d in pkl file to evaluate')

    args = parser.parse_args()
    main(args)
