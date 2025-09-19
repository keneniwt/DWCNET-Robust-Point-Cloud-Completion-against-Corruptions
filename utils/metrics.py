import logging
import torch
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferDistanceL2_split
import os
from extensions.emd import emd_module as emd
import numpy as np
from scipy.spatial import cKDTree


def compute_point_cloud_distance(source, target):
    """
    Compute the distance from each point in the source point cloud to the nearest point in the target point cloud.

    Parameters:
    - source: numpy.ndarray, shape (batch_size, num_points, 3), the source point clouds.
    - target: numpy.ndarray, shape (batch_size, num_points, 3), the target point clouds.

    Returns:
    - distances: numpy.ndarray, shape (batch_size, num_points), an array of distances for each point cloud in the batch.
    """
    # Ensure the input arrays are NumPy arrays
    source = np.asarray(source)
    target = np.asarray(target)

    # Validate input shapes
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source must be a 3D array with shape (batch_size, num_points, 3).")
    if target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("Target must be a 3D array with shape (batch_size, num_points, 3).")
    if source.shape[0] != target.shape[0]:
        raise ValueError("Batch sizes of source and target must match.")

    batch_size = source.shape[0]
    num_points = source.shape[1]
    distances = np.zeros((batch_size, num_points))

    # Iterate over the batch dimension
    for i in range(batch_size):
        # Get the i-th point cloud in the batch
        source_cloud = source[i]  # Shape: (num_points, 3)
        target_cloud = target[i]  # Shape: (num_points, 3)

        # Build a KDTree from the target point cloud
        kdtree = cKDTree(target_cloud)

        # Query the nearest neighbor for each point in the source point cloud
        dist, _ = kdtree.query(source_cloud, k=1)
        distances[i] = dist  # Store distances for this batch

    return distances

class Metrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'EMDistance',
        'enabled': True,
        'eval_func': 'cls._get_emd_distance',
        'eval_object': emd.emdModule(),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'Fidelity',
        'enabled': True,
        'eval_func': 'cls._get_fidelity_distance',
        'eval_object': ChamferDistanceL2_split(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }]


    @classmethod
    def get(cls, pred, gt, require_emd=False):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            if not require_emd and 'emd' in item['eval_func']:
                _values[i] = torch.tensor(0.).to(gt.device)
            else:
                eval_func = eval(item['eval_func'])
                result = eval_func(pred, gt)
                # Handle both tuple and single-value returns
                _values[i] = result[0] if isinstance(result, tuple) else result
                
        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]
    
    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):

        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        device = pred.device
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)
            
            #pred = np.asarray(pred)
            #gt = np.asarray(gt)

            dist1 = compute_point_cloud_distance(pred,gt)
            dist2 = compute_point_cloud_distance(gt,pred)
            
            #print(type(dist1))
            #print(type(dist2))
            
             # Compute recall and precision
            recall = np.mean(dist2 < th)  # Equivalent to sum(d < th)/len(dist2)
            precision = np.mean(dist1 < th)
            
            # Compute F-score (harmonic mean)
            if recall + precision > 0:
                f_score = 2 * recall * precision / (recall + precision)
            else:
                f_score = 0.0
            result_tensor = torch.tensor(f_score).to(device)  
                
            return result_tensor
    '''
            # Compute recall and precision
            recall = float(np.sum(dist2 < th)) / float(len(dist2))
            precision = float(np.sum(dist1 < th)) / float(len(dist1))

            # Compute F-score
            if recall + precision > 0:
                result = 2 * recall * precision / (recall + precision)
            else:
                result = 0.0

            # Convert result to a PyTorch tensor and move to the specified device
            result_tensor = torch.tensor(result, dtype=torch.float32).to(device)
            return result_tensor
'''

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """Convert a tensor to point cloud data without using Open3D."""
        tensor = tensor.cpu().numpy()  # Move tensor to CPU and convert to NumPy array

        # If the input tensor has a batch dimension, return a list of point clouds
        if tensor.ndim == 3:  # Batched input (batch_size, num_points, 3)
            ptclouds = [tensor[i] for i in range(tensor.shape[0])]  # List of point clouds
            return ptclouds
        elif tensor.ndim == 2:  # Single point cloud (num_points, 3)
            return tensor  # Return the point cloud directly
        else:
            raise ValueError("Input tensor must have shape (batch_size, num_points, 3) or (num_points, 3).")


    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[2]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_emd_distance(cls, pred, gt, eps=0.005, iterations=100):
        emd_loss = cls.ITEMS[3]['eval_object']
        dist, _ = emd_loss(pred, gt, eps, iterations)
        emd_out = torch.mean(torch.sqrt(dist))
        return emd_out * 1000
    
    @classmethod
    def _get_fidelity_distance(cls, pred, gt):
        """Computes fidelity using Chamfer Distance L2 between prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted point cloud (B x N x 3)
            gt (torch.Tensor): Ground truth point cloud (B x M x 3)
        
        Returns:
            torch.Tensor: Single scalar value of mean distance * 1000
        """
        fidelity_loss = cls.ITEMS[4]['eval_object']
        dist, _ = fidelity_loss(pred, gt)  # We only need the first distance
        return torch.mean(dist) * 1000  # Return single scaled value
    
    '''
    @classmethod
    def _get_fidelity_distance(cls, pred, gt):
        """Computes fidelity using Chamfer Distance L2 between prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted point cloud (B x N x 3)
            gt (torch.Tensor): Ground truth point cloud (B x M x 3)
        
        Returns:
            float: Mean Chamfer Distance (L2) multiplied by 1000 for scaling
        """
        criterion = ChamferDistanceL2_split(ignore_zeros=True)
        dist, _ = criterion(pred, gt)
        fidelity_out = torch.mean(dist)
        return fidelity_out * 1000  # Scale for readability
    
    @classmethod
    def get_Fidelity():
        """Calculates average fidelity error across all samples."""
        metric = []
        for sample in Samples:
            input_data = torch.from_numpy(np.load(os.path.join(Data_path, sample, 'input.npy'))).unsqueeze(0).cuda()
            pred_data = torch.from_numpy(np.load(os.path.join(Data_path, sample, 'pred.npy'))).unsqueeze(0).cuda()
            metric.append(cls._get_fidelity_distance(input_data, pred_data))
        
        avg_fidelity = sum(metric) / len(metric)
        print(f'Fidelity is {avg_fidelity:.6f}')
        return avg_fidelity
    
    '''

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())


    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value





