# coding=utf-8
import os
import h5py
import torch
import torch.nn.functional as F
from torch.nn import Identity
from torch.utils.data import Dataset
import numpy as np
import random
import math as mt
import collections
from functools import wraps
from time import time

def _get_act(act):
    if callable(act):
        return act

    if act == 'tanh':
        func = torch.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    elif act == 'none':
        func = Identity()
    else:
        raise ValueError(f'{act} is not supported')
    return func

def _get_initializer(initializer: str = "Glorot normal"):

    INITIALIZER_DICT = {
        "Glorot normal": torch.nn.init.xavier_normal_,
        "Glorot uniform": torch.nn.init.xavier_uniform_,
        "He normal": torch.nn.init.kaiming_normal_,
        "He uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    return INITIALIZER_DICT[initializer]


class DeepOnetDatasetSingle(Dataset):
    def __init__(self, file_name,
                 saved_folder,
                 initial_step=1,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1,
                ):

        # file path, HDF5 file is assumed
        file_path = os.path.abspath(saved_folder + file_name)

        # read data from HDF5 file
        with h5py.File(file_path, 'r') as f:
            ## data dim: [num_sample, t, x1, ..., xd] (The field dimension is 1)
            # or [num_sample, t, x1, ..., xd, v] (The field dimension is v)
            if 'tensor' in f.keys():  # scalar equations
                ## data dim = [n, t, x1, ..., xd]
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                # recorrect the nt
                nt=min(_data.shape[1],f['t-coordinate'].shape[0]) if f.get('t-coordinate',None) else 1

                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [n, x1, ..., xd, t]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch
                    x = np.array(f["x-coordinate"], dtype='f')
                    t = np.array(f["t-coordinate"], dtype='f')[:nt] if f.get('t-coordinate',None) else np.array([0],dtype='f')
                    x = torch.tensor(x, dtype=torch.float)
                    t = torch.tensor(t, dtype=torch.float)
                    X, T = torch.meshgrid((x, t),indexing='ij')
                    self.grid = torch.stack((X,T),axis=-1)[::reduced_resolution,::reduced_resolution_t]


                if len(_data.shape) == 4:  
                    if nt==1:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [n, x1, ..., xd, t]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [n, x1, ..., xd, t]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, None, :]  # batch, x, y, t, ch
                        x = np.array(f["x-coordinate"], dtype='f')
                        y = np.array(f["y-coordinate"], dtype='f')
                        t = np.array(f["t-coordinate"], dtype='f')[:nt] if f.get('t-coordinate',None) else np.array([0],dtype='f')
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        t = torch.tensor(t, dtype=torch.float)
                        X, Y, T = torch.meshgrid((x, y, t),indexing='ij')
                        self.grid = torch.stack((X, Y, T), axis=-1)[::reduced_resolution, ::reduced_resolution,::reduced_resolution_t]
                    else:
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data=_data[:,:,:,:,None]
                        x = np.array(f["x-coordinate"], dtype='f')
                        y = np.array(f["y-coordinate"], dtype='f')
                        t = np.array(f["t-coordinate"], dtype='f')[:nt] if f.get('t-coordinate',None) else np.array([0],dtype='f')
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        t = torch.tensor(t, dtype=torch.float)
                        X, Y, T = torch.meshgrid((x, y, t),indexing='ij')
                        self.grid = torch.stack((X, Y, T), axis=-1)[::reduced_resolution, ::reduced_resolution,::reduced_resolution_t]

            else:  # NS equation
                _data = np.array(f['density'], dtype=np.float32)  # density: [batch, time, x1,..,xd]
                idx_cfd = _data.shape
                # recorrect the nt
                nt=min(_data.shape[1],f['t-coordinate'].shape[0])
                if len(idx_cfd)==3:  # 1D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          3],
                                        dtype=np.float32)
                    #density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,2] = _data   # batch, x, t, ch

                    x = np.array(f["x-coordinate"], dtype='f')
                    t = np.array(f["t-coordinate"], dtype='f')[:nt]
                    x = torch.tensor(x, dtype=torch.float)
                    t = torch.tensor(t, dtype=torch.float)[:nt]
                    X, T = torch.meshgrid((x, t),indexing='ij')
                    self.grid = torch.stack((X,T),axis=-1)[::reduced_resolution,::reduced_resolution_t]

                if len(idx_cfd)==4:  # 2D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          4],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,3] = _data   # batch, x, t, ch

                    x = np.array(f["x-coordinate"], dtype='f')
                    y = np.array(f["y-coordinate"], dtype='f')
                    t = np.array(f["t-coordinate"], dtype='f')[:nt]
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    t = torch.tensor(t, dtype=torch.float)[:nt]
                    X, Y, T = torch.meshgrid((x, y, t),indexing='ij')
                    self.grid = torch.stack((X, Y, T), axis=-1)[::reduced_resolution, ::reduced_resolution,::reduced_resolution_t]

        self.dx=x[reduced_resolution]-x[0]
        self.dt=t[reduced_resolution_t]-t[0] if t.shape[0]>1  else None
        self.tmax=t[-1] if t.shape[0]>1  else None

        # Define the max number of samples
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        # Construct train/test dataset
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
        return self.data[idx,...,:self.initial_step,:], self.data[idx], self.grid



class DeepOnetDatasetMult(Dataset):
    def __init__(self, file_name,
                 saved_folder,
                 initial_step=1,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1,
                ):
        # file path, HDF5 file is assumed
        self.file_path = os.path.abspath(saved_folder + file_name)
        self.sub = reduced_resolution
        self.sub_t = reduced_resolution_t
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as f:
            seed_list = sorted(f.keys())
            grid_x,grid_t=f[seed_list[0]]['grid']['x'],f[seed_list[0]]['grid']['t']
            self.dx=grid_x[self.sub]-grid_x[0]
            self.dt=grid_t[self.sub_t]-grid_t[0]
            self.tmax= grid_t[-1]
            data=f[seed_list[0]]["data"]
            self.dim = len(data.shape) - 2
            if self.dim == 1:
                self.grid=np.array(grid_x)[::reduced_resolution,None]
                x = torch.tensor(grid_x, dtype=torch.float)
                t = torch.tensor(grid_t, dtype=torch.float)
                X,T=torch.meshgrid(x, t, indexing='ij')
                grid = torch.stack((X,T),axis=-1)
                self.grid = grid[::self.sub,::self.sub_t]


            elif self.dim == 2:
                grid_y=f[seed_list[0]]['grid']['y']
                x = torch.tensor(grid_x, dtype=torch.float)
                y = torch.tensor(grid_y, dtype=torch.float)
                t = torch.tensor(grid_t, dtype=torch.float)
                X, Y, T = torch.meshgrid((x, y, t),indexing='ij')
                grid = torch.stack((X,Y,T),axis=-1)
                self.grid = grid[::self.sub,::self.sub,::self.sub_t]

        seed_list = seed_list[::reduced_batch]

        # Define the max number of samples
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, len(seed_list))
        else:
            num_samples_max = len(seed_list)

        # Construct test dataset
        test_idx = int(num_samples_max * (1-test_ratio))
        if if_test:
            self.seed_list = np.array(seed_list[test_idx:num_samples_max])
        else:
            self.seed_list = np.array(seed_list[:test_idx])
        self.initial_step = initial_step


    def __len__(self):
        return len(self.seed_list)


    def __getitem__(self, idx):
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.seed_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1,len(data.shape)-1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

            # Extract spatial dimension of data
            dim = len(data.shape) - 2

            # x, y and t are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                data=data[::self.sub,::self.sub_t]
            elif dim == 2:
                data=data[::self.sub,::self.sub,::self.sub_t]

        return data[...,:self.initial_step,:], data, self.grid    # return (a, u , grid)

def timer(func):
    @wraps(func)
    def func_wrapper(*args,**kwargs):
        start_time=time()
        result=func(*args,**kwargs)
        end_time=time()
        print(f"{func.__name__} cost time: {end_time-start_time:.4f} s")
        return result, end_time-start_time
    return func_wrapper

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True

def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count

def to_device(li,device):
    if isinstance(li,collections.Iterable):
        for i in range(len(li)):
            li[i]=li[i].to(device)
        return li
    else: # object
        return li.to(device)


def generate_input(a,grid):
    # a : shape of [bs, x1, ..., xd, init_t, v]
    # grid : shape of [bs, x1, ..., xd, t, d+1]
    T_step=grid.shape[-2]
    a_shape = list(a.shape)[:-2]+[-1] #(bs, x1, ..., xd, -1)
    a=a.reshape(a_shape).unsqueeze(-2)   # (bs, x1, ..., xd, 1, init_t*v )
    temp=[1]*len(a.shape)
    temp[-2]=T_step  # [1,...,t,1]
    input=a.repeat(temp)   # (bs, x1, ..., xd, t, init_t*v )
    input=torch.concat([input,grid], dim=-1)  # (bs, x1, ..., xd, t, init_t*v+d+1)
    return input 