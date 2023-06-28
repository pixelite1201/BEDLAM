import argparse
import sys
import logging
import os

from glob import glob
import numpy as np
import pickle
import zipfile


logging.basicConfig(level=logging.DEBUG)


def assert_type(var, gt_type, key):
    if not isinstance(var, gt_type):
        raise TypeError('{} should be of type {}'.format(key, gt_type))


def assert_shape(var, gt_shape, key):
    if var.shape != gt_shape:
        raise ValueError('{} should be of shape {} but you are providing {}'.format(key, gt_shape, var.shape))


def check_smplx(pred_file):
    pred_param = pickle.load(open(pred_file, 'rb'), encoding='latin1')
        
    if 'allSmplJoints3d' in pred_param.keys() and 'verts' in pred_param.keys():
        joints3d = pred_param['allSmplJoints3d'].squeeze()
        verts3d = pred_param['verts'].squeeze()
        # Instance should be numpy array
        assert_type(verts3d, np.ndarray, 'verts')
        assert_type(joints3d, np.ndarray, 'allSmplJoints3d')
        # SMPL-X vertices shape should be (10475,3)
        assert_shape(verts3d, (10475, 3), 'verts')
        if len(joints3d.shape) != 2 or joints3d.shape[1] != 3 or joints3d.shape[0] < 127:
            raise ValueError('joints should be of shape (127,3) but you ar providing {}'.format(joints3d.shape))
        if joints3d.shape[0] > 127:
            logging.warning(' Only first 127 3d joints will be used for body, hands and face evaluation but you are providing {} joints'.format(joints3d.shape[0]))

    else:
        raise KeyError('allSMPLJoints3d and verts needs to be provided in key. Please check the ReadMe for details and run the evaluation code on github')

    assert 'joints' in pred_param.keys()
    joints = pred_param['joints']
    assert_type(pred_param['joints'], np.ndarray, 'joints')
    # Only first 24 joints will be used for matching
    if len(joints.shape) != 2 or joints.shape[1] < 2 or joints.shape[0] < 24:
        raise ValueError('joints should be of shape (24,2) but you are providing {}'.format(joints.shape))
    if joints.shape[0] > 24:
        logging.warning(' Only first 24 projected joints in joints key will be used in matching but you are providing {} joints'.format(joints.shape[0]))


def check_pred_file(*args):
    """Function to check the prediction file"""

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--predZip', type=str,
                        default='')
    parser.add_argument('--extractZipFolder', type=str,
                        default='')

    args = parser.parse_args(*args)
    path_to_zip_file = args.predZip
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(args.extractZipFolder)

    all_files = glob(os.path.join(args.extractZipFolder, 'predictions', '*'))
    if len(all_files) == 0:
        raise EOFError('No files are present inside zip')

    for pred_file in all_files:
        logging.info('Reading file {}'.format(pred_file))
        check_smplx(pred_file)


if __name__ == '__main__':
    check_pred_file(sys.argv[1:])
    logging.info('If you reach here then your zip folder is ready to submit')