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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import ProgressBar
from train.utils.os_utils import copy_code
from train.core.hand_trainer import HandTrainer
from train.utils.train_utils import set_seed, update_hparams
#from pytorch_lightning.loggers import WandbLogger

sys.path.append('.')


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
    # initialize tensorboard logger
    # tb_logger = TensorBoardLogger(
    #     save_dir=log_dir,
    #     log_graph=False,
    # )
    # experiment_loggers.append(tb_logger)

    # wandb_logger = WandbLogger(project=log_dir.split('/')[-3], name=log_dir.split('/')[-2])
    # experiment_loggers.append(wandb_logger)
    model = HandTrainer(hparams=hparams).to(device)

    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=True,
        save_top_k=5,
        mode='min',
    )

    # Because the last checkpoint might not always be the best,
    # we load temporary trainer instance from the log dir and this provides us
    # the path to the best checkpoint
    # Refer https://github.com/Lightning-AI/lightning/issues/3107

    if hparams.TRAINING.RESUME is not None:
        trainer_temp = pl.Trainer(default_root_dir=log_dir, callbacks=[ckpt_callback])
        trainer_temp.checkpoint_connector.resume_start(hparams.TRAINING.RESUME)
        trainer_temp.checkpoint_connector.restore_callbacks()
        if trainer_temp.callbacks[3].best_model_path:
            hparams.TRAINING.RESUME = trainer_temp.callbacks[3].best_model_path
        trainer_temp = None
        
    trainer = pl.Trainer(
        #accelerator='gpu',
        #devices=1,
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
