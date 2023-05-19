import torch
import numpy as np
from loguru import logger

from ..core import config
from .dataset_smpl import DatasetHMR
from ..utils.train_utils import parse_datasets_ratios

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):

        datasets_ratios = parse_datasets_ratios(options.DATASETS_AND_RATIOS)
        hl = len(datasets_ratios) // 2
        self.dataset_list = datasets_ratios[:hl]
        self.dataset_ratios = datasets_ratios[hl:]

        assert len(self.dataset_list) == len(self.dataset_ratios), 'Number of datasets and ratios should be equal'

        # self.dataset_list = ['h36m', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        # self.dataset_dict = {
        #     'h36m': 0, 'mpii': 1, 'lspet': 2,
        #     'coco-cam': 3, 'mpi-inf-3dhp': 4, '3doh': 5, 'mannequin': 6,
        # }
        itw_datasets = ['mpii', 'coco']

        self.datasets = [DatasetHMR(options, ds, **kwargs)
                         for ds in self.dataset_list]
        self.length = max([len(ds) for ds in self.datasets])

        self.partition = []

        for idx, (ds_name, ds_ratio) in enumerate(zip(self.dataset_list, self.dataset_ratios)):
            r = ds_ratio
            self.partition.append(r)

        logger.info(f'Using these datasets: {self.dataset_list}')
        logger.info(f'Ratios of datasets: {self.partition}')

        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
