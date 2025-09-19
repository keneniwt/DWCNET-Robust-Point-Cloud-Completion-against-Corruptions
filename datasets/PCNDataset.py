import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io3 import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class PCN(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS
        
        # Add clean partial points path
        self.clean_partial_points_path = getattr(config, 'CLEAN_PARTIAL_POINTS_PATH', None)

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if self.cars:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial', 'clean_partial']  # Add clean_partial to sampling
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'clean_partial', 'gt']  # Add clean_partial to mirroring
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'clean_partial', 'gt']  # Add clean_partial to tensor conversion
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial', 'clean_partial']  # Add clean_partial to sampling
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'clean_partial', 'gt']  # Add clean_partial to tensor conversion
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                file_entry = {
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s),
                }
                
                # Add clean partial paths if available
                if self.clean_partial_points_path is not None:
                    file_entry['clean_partial_path'] = [
                        self.clean_partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ]
                
                file_list.append(file_entry)

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == 'train' else 0

        # Load all point cloud types
        point_types = ['partial', 'gt']
        if 'clean_partial_path' in sample:
            point_types.append('clean_partial')

        for ri in point_types:
            file_path = sample['%s_path' % ri]
            if isinstance(file_path, list):
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        # Return all available point clouds
        if 'clean_partial' in data:
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['clean_partial'], data['gt'])
        else:
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)