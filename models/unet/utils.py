# coding=utf-8
import h5py
import numpy as np
import os
import random
import time
import torch
from functools import wraps
from torch.utils.data import Dataset, DataLoader


class UNetDatasetSingle(Dataset):
    def __init__(self, file_name,
                 saved_folder,
                 initial_step=10,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1):
        
        # file path, HDF5 file is assumed
        file_path = os.path.join(saved_folder, file_name)

        # read data from HDF5 file
        with h5py.File(file_path, 'r') as f:
            if "tensor" not in f.keys(): # CFD datasets
                spatial_dim = len(f["density"].shape) - 2
                self.data = None
                if spatial_dim == 1:
                    for i, key in enumerate(["density", "pressure", "Vx"]):
                        _data = np.array(f[key], dtype=np.float32) # [num_sample, t, x1, ..., xd]
                        _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
                        _data = np.transpose(_data, (0, 2, 1))
                        if i == 0:
                            data_shape = list(_data.shape)
                            data_shape.append(3)
                            self.data = np.zeros(data_shape, dtype=np.float32)
                        self.data[..., i] = _data
                elif spatial_dim == 2:
                    for i, key in enumerate(["density", "pressure", "Vx", "Vy"]):
                        _data = np.array(f[key], dtype=np.float32)
                        _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        if i == 0:
                            data_shape = list(_data.shape)
                            data_shape.append(4)
                            self.data = np.zeros(data_shape, dtype=np.float32)
                        self.data[..., i] = _data
                else: # spatial_dim == 3
                    for i, key in enumerate(["density", "pressure", "Vx", "Vy", "Vz"]):
                        _data = np.array(f[key], dtype=np.float32)
                        _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        if i == 0:
                            data_shape = list(_data.shape)
                            data_shape.append(5)
                            self.data = np.zeros(data_shape, dtype=np.float32)
                        self.data[..., i] = _data
            else:
                _data = np.array(f["tensor"], dtype=np.float32)
                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution] # (num_sample, t, x)
                    _data = np.transpose(_data[:, :, :], (0, 2, 1)) # convert to (num_sample, x, t)
                    self.data = _data[:, :, :, None]  # (num_sample, x, t, v)
                elif len(_data.shape) == 4:
                    if "nu" in f.keys(): # 2D darcy flow
                        # label
                        _data = _data[::reduced_batch, :, ::reduced_resolution, ::reduced_resolution] # (num_sample, 1, x1, x2)
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1)) # (num_sample, x1, x2, 1)
                        self.data = _data
                        # nu
                        _data = np.array(f['nu'], dtype=np.float32) # (num_sample, x1, x2)
                        _data = _data[::reduced_batch, None, ::reduced_resolution, ::reduced_resolution] # (num_sample, 1, x1, x2)
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1)) # (num_sample, x1, x2, 1)
                        # data
                        self.data = np.concatenate([_data, self.data], axis=-1) # (num_sample, x1, x2, 2)
                        self.data = self.data[:, :, :, :, None]  # (num_sample, x1, x2, 2, v)
                    else:  
                        _data = _data[::reduced_batch, :, ::reduced_resolution, ::reduced_resolution] # (num_sample, t, x1, x2)
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1)) # (num_sample, x1, ..., xd, t)
                        self.data = _data[:, :, :, :, None] # (num_sample, x1, ..., xd, t, v)
                else: # TODO 3D
                    pass

        # define the max number of samples
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        # construct train/test dataset
        test_idx = int(num_samples_max * (1-test_ratio))
        if if_test:
            self.data = self.data[test_idx:num_samples_max]
        else:
            self.data = self.data[:test_idx]

        # time steps used as initial conditions
        self.initial_step = initial_step

        self.data = torch.tensor(self.data)


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        return self.data[idx,...,:self.initial_step,:], self.data[idx]
    


class UNetDatasetMult(Dataset):
    def __init__(self, file_name, 
                 saved_folder,
                 initial_step=10,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1
                ):
        # file path, HDF5 file is assumed
        self.file_path = os.path.join(saved_folder, file_name)
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t

        # extract list of seeds
        with h5py.File(self.file_path, 'r') as f:
            seed_list = sorted(f.keys())
        seed_list = seed_list[::reduced_batch]

        # define the max number of samples
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, len(seed_list))
        else:
            num_samples_max = len(seed_list)

        # construct train/test dataset
        test_idx = int(num_samples_max * (1-test_ratio))
        if if_test:
            self.seed_list = np.array(seed_list[test_idx:num_samples_max])
        else:
            self.seed_list = np.array(seed_list[:test_idx])
            
        # time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.seed_list)
    

    def __getitem__(self, idx):
        # open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.seed_list[idx]]
            data = np.array(seed_group["data"], dtype=np.float32) # (t, x1, ..., xd, v)
            if len(data.shape) == 3: # 1D
                data = data[::self.reduced_resolution_t, ::self.reduced_resolution, :]
            elif len(data.shape) == 4: # 2D
                data = data[::self.reduced_resolution_t, ::self.reduced_resolution, ::self.reduced_resolution, :]
            else: # TODO 3D
                pass
            data = torch.tensor(data)
            ## convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape)-1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

        return data[..., :self.initial_step, :], data


def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn