import os
import sys
import torch
import time
import yaml
import argparse
import numpy as np
import pytorch_lightning as pl
from loguru import logger
from flatten_dict import flatten, unflatten
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import ProgressBar
from train.utils.os_utils import copy_code
from train.core.smplx_trainer import SMPLXTrainer
from train.utils.train_utils import set_seed, update_hparams
#from pytorch_lightning.loggers import WandbLogger

sys.path.append('.')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix + '.', '')] = value
    return stripped_state_dict


def load_valid(model, pretrained_file, skip_list=None):

    pretrained_dict = torch.load(pretrained_file)['state_dict']
    pretrained_dict = strip_prefix_if_present(pretrained_dict, prefix='model')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict1)
    model.load_state_dict(model_dict)


def train(hparams, fast_dev_run=False):
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(hparams.SEED_VALUE)
    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    # # copy the code for reproducibility
    # copy_code(
    #     output_folder=log_dir,
    #     curr_folder=os.path.dirname(os.path.abspath(__file__))
    # )

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')
    experiment_loggers = []
    
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        log_graph=False,
    )
    experiment_loggers.append(tb_logger)

    model = SMPLXTrainer(hparams=hparams).to(device)

    body_model_pretrained = hparams.TRAINING.body_ckpt
    hand_model_pretrained = hparams.TRAINING.hand_ckpt

    load_valid(model.body_model, body_model_pretrained)
    load_valid(model.hand_model, hand_model_pretrained)
    
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=True,
        save_top_k=5,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=1,
        logger=experiment_loggers,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        callbacks=[ckpt_callback, ProgressBar(refresh_rate=200)],
        default_root_dir=log_dir,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
    )

    if hparams.TRAINING.RESUME and hparams.TRAINING.TEST_BEFORE_TRAINING:
        logger.info(f'Running an initial on {hparams.DATASET.VAL_DS} test before training')
        trainer.test(model, ckpt_path=hparams.TRAINING.RESUME)

    # logger.info('*** Testing before training ***')
    # trainer.test(model, ckpt_path=hparams.TRAINING.RESUME)
    logger.info('*** Started training ***')

    if args.test:
        trainer.test(model, ckpt_path=hparams.TRAINING.RESUME)
    else:
        trainer.fit(model, ckpt_path=hparams.TRAINING.RESUME)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--log_dir', type=str, help='log dir path', default='./logs')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from where it left off')
    parser.add_argument('--hand_ckpt', type=str, help='path to pretrained hand model checkpoint')
    parser.add_argument('--body_ckpt', type=str, help='path to pretrained body model checkpoint')

    parser.add_argument('--fdr', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--start_with_test', action='store_true')

    args = parser.parse_args()
    
    logger.info(f'Input arguments: \n {args}')
    torch.cuda.empty_cache()

    # Update hparams with config file and from args
    hparams = update_hparams(args.cfg)
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = os.path.join(args.log_dir, hparams.EXP_NAME+logtime)
    os.makedirs(logdir, exist_ok=True)
    hparams.LOG_DIR = logdir

    hparams.TRAINING.hand_ckpt = args.hand_ckpt
    hparams.TRAINING.body_ckpt = args.body_ckpt

    # Load the last checkpoint using the epoch id.
    if args.resume and hparams.TRAINING.RESUME is None:
        ckpt_files = []
        for root, dirs, files in os.walk(args.log_dir, topdown=False):
            for f in files:
                if f.endswith('.ckpt'):
                    ckpt_files.append(os.path.join(root, f))

        epoch_idx = [int(x.split('=')[-1].split('.')[0]) for x in ckpt_files]
        if len(epoch_idx) == 0:
            ckpt_file = None
        else:
            last_epoch_idx = np.argsort(epoch_idx)[-1]
            ckpt_file = ckpt_files[last_epoch_idx]
        logger.info('Loading CKPT', ckpt_file)
        hparams.TRAINING.RESUME = ckpt_file

    if args.start_with_test:
        hparams.TRAINING.TEST_BEFORE_TRAINING = True

    if args.test:
        hparams.RUN_TEST = True

    def save_dict_to_yaml(obj, filename, mode='w'):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save final config
    save_dict_to_yaml(
        unflatten(flatten(hparams)),
        os.path.join(hparams.LOG_DIR, 'config_to_run.yaml')
    )

    train(hparams, fast_dev_run=args.fdr)
