import h5py
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_cluster import radius_graph
from typing import Tuple
import random

import time
from functools import wraps, reduce

class PDE(object):
    def __init__(self, 
                 name: str,
                 variables: dict,
                 temporal_domain: Tuple,
                 resolution_t: int,
                 spatial_domain: list,
                 resolution: list,
                 reduced_resolution_t: int=1,
                 reduced_resolution: int=1):
        super().__init__()
        self.name = name
        self.tmin = temporal_domain[0]
        self.tmax = temporal_domain[1]
        self.resolution_t = resolution_t // reduced_resolution_t
        self.spatial_domain = spatial_domain
        self.resolution = [res // reduced_resolution for res in resolution]
        self.spatial_dim = len(spatial_domain)
        self.variables = variables

    def __repr__(self):
        return self.name
    

class MPNNDatasetSingle(Dataset):
    def __init__(self, 
                 file_name: str,
                 saved_folder: str,
                 reduced_resolution: int=1,
                 reduced_resolution_t: int=1,
                 reduced_batch: int=1,
                 if_test: bool=False,
                 test_ratio: float=0.1,
                 num_samples_max: int=-1,
                 variables: dict={}) -> None:

        super().__init__()

        # file path
        file_path = os.path.abspath(saved_folder + file_name)

        # read data and coordinates from HDF5 file
        with h5py.File(file_path, 'r') as f:
            if "tensor" not in f.keys(): # TODO CFD datasets
                spatial_dim = len(f["density"].shape) - 2
                self.data = None
                if spatial_dim == 1:
                    self.coordinates = torch.from_numpy(f["x-coordinate"][::reduced_resolution][:, None])
                    for i, key in enumerate(["density", "pressure", "Vx"]):
                        _data = np.array(f[key], dtype=np.float32)
                        _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
                        if i == 0:
                            data_shape = list(_data.shape)
                            data_shape.append(3)
                            self.data = np.empty(data_shape, dtype=np.float32)
                        self.data[..., i] = _data
                elif spatial_dim == 2:
                    # coordinates: (x*y, 2)
                    x = torch.from_numpy(f["x-coordinate"][::reduced_resolution])
                    y = torch.from_numpy(f["y-coordinate"][::reduced_resolution])
                    X, Y = torch.meshgrid(x, y, indexing="ij")
                    self.coordinates = torch.stack([X.ravel(), Y.ravel()], dim=-1)
                    for i, key in enumerate(["density", "pressure", "Vx", "Vy"]):
                        _data = np.array(f[key], dtype=np.float32)
                        _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution]
                        if i == 0:
                            data_shape = list(_data.shape)
                            data_shape.append(4)
                            self.data = np.empty(data_shape, dtype=np.float32)
                        self.data[..., i] = _data
                else:
                    # coordinates: (x*y, 3)
                    x = torch.from_numpy(f["x-coordinate"][::reduced_resolution])
                    y = torch.from_numpy(f["y-coordinate"][::reduced_resolution])
                    z = torch.from_numpy(f["z-coordinate"][::reduced_resolution])
                    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
                    self.coordinates = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)
                    for i, key in enumerate(["density", "pressure", "Vx", "Vy", "Vz"]):
                        _data = np.array(f[key], dtype=np.float32)
                        _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution]
                        if i == 0:
                            data_shape = list(_data.shape)
                            data_shape.append(5)
                            self.data = np.empty(data_shape, dtype=np.float32)
                        self.data[..., i] = _data
            else:
                _data = np.array(f["tensor"], dtype=np.float32) # (num_samples, t, x1, ..., xd)
                if len(_data.shape) == 3:  # 1D
                    # coordinates: (x, 1)
                    self.coordinates = torch.from_numpy(f["x-coordinate"][::reduced_resolution][:, None])
                    # data: (num_sample, t, x, 1)
                    self.data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, None] # (num_samples, t, x1, ..., xd, 1)
                elif len(_data.shape) == 4: # 2D (ignore darcy flow)
                    # coordinates: (x*y, 2)
                    x = torch.from_numpy(f["x-coordinate"][::reduced_resolution])
                    y = torch.from_numpy(f["y-coordinate"][::reduced_resolution])
                    X, Y = torch.meshgrid(x, y, indexing="ij")
                    self.coordinates = torch.stack([X.ravel(), Y.ravel()], dim=-1)
                    # data: (num_sample, t, x, y, 1)
                    self.data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, None]
                else:
                    self.data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, None] # (num_samples, t, x1, ..., xd, 1)
                    # coordinates: (x*y, 3)
                    x = torch.from_numpy(f["x-coordinate"][::reduced_resolution])
                    y = torch.from_numpy(f["y-coordinate"][::reduced_resolution])
                    z = torch.from_numpy(f["z-coordinate"][::reduced_resolution])
                    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
                    self.coordinates = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)
                    # data: (num_sample, t, x, y, z, 1)
                    self.data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution, None]
        
        # PDE parameters
        self.variables = variables

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

        # To tensor
        self.data = torch.tensor(self.data)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # data: (bs, t, num_points, v) coordinates: (bs, num_points, spatial_dim) variables: {parm1: (bs), parm2: (bs), ...}
        return torch.flatten(self.data[idx], start_dim=1, end_dim=-2), self.coordinates, self.variables
    


class MPNNDatasetMult(Dataset):
    def __init__(self, 
                 file_name: str, 
                 saved_folder: str,
                 reduced_resolution: int=1,
                 reduced_resolution_t: int=1,
                 reduced_batch: int=1,
                 if_test: bool=False,
                 test_ratio: float=0.1,
                 num_samples_max: int=-1,
                 variables: dict={}
                ):
        # file path, HDF5 file is assumed
        file_path = os.path.abspath(saved_folder + file_name)
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t
        self.variables = variables

        # Extract list of seeds
        self.file_handle = h5py.File(file_path, 'r')
        seed_list = sorted(self.file_handle.keys())
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

    def __len__(self):
        return len(self.seed_list)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        seed_group = self.file_handle[self.seed_list[idx]]
        data = np.array(seed_group["data"], dtype=np.float32) # (t, x1, .., xd, v)
        if len(data.shape) == 3: # 1D
            coordinates = torch.from_numpy(seed_group["grid"]["x"][::self.reduced_resolution][:, None]) # (x, 1)
            data = data[::self.reduced_resolution_t, ::self.reduced_resolution, :]
        elif len(data.shape) == 4: # 2D
            x = torch.from_numpy(seed_group["grid"]["x"][::self.reduced_resolution])
            y = torch.from_numpy(seed_group["grid"]["y"][::self.reduced_resolution])
            X, Y = torch.meshgrid(x, y, indexing="ij")
            coordinates = torch.stack([X.ravel(), Y.ravel()], dim=-1)
            data = data[::self.reduced_resolution_t, ::self.reduced_resolution, ::self.reduced_resolution, :]
        else:
            x = torch.from_numpy(seed_group["grid"]["x"][::self.reduced_resolution])
            y = torch.from_numpy(seed_group["grid"]["y"][::self.reduced_resolution])
            z = torch.from_numpy(seed_group["grid"]["z"][::self.reduced_resolution])
            X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
            coordinates = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)
            data = data[::self.reduced_resolution_t, ::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution, :]
        data = torch.tensor(data)
        # data: (bs, t, num_points, v) coordinates: (bs, num_points, spatial_dim) variables: {parm1: (bs), parm2: (bs), ...}
        return torch.flatten(data, start_dim=1, end_dim=-2), coordinates, self.variables



class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE,
                 neighbors: int = 2,
                 time_window: int = 25) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE to solve
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.nt = pde.resolution_t
        self.nx = reduce(lambda x, y: x*y, self.pde.resolution)
        print("nt:", self.nt, "nx:", self.nx)

    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time steps
        Args:
            datapoints (torch.Tensor): trajectory with shape [bs, t, x1, ..., xd, v]
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]
            data = torch.cat((data, d[None, :]), dim=0)
            labels = torch.cat((labels, l[None, :]), dim=0)

        return data, labels # (bs, tw, x1, ..., xd, v)
    
    def create_graph(self,
                     data: torch.Tensor,
                     labels: torch.Tensor,
                     x: torch.Tensor,
                     variables: dict,
                     steps: list) -> Data:
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor
            variables (dict): dictionary of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        t = torch.linspace(self.pde.tmin, self.pde.tmax, self.nt)
        u, x_pos, t_pos, y, batch, pde_variables = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):
            # data_batch: [tw, nx, v] , labels_batch: [tw, nx, v]
            u = torch.cat((u, torch.transpose(data_batch, 0, 1)), dim=0) # u: [bs*nx, tw, v]
            y = torch.cat((y, torch.transpose(labels_batch, 0, 1)), dim=0) # y: [bs*nx, tw, v]
            x_pos = torch.cat((x_pos, x[0]), dim=0)
            t_pos = torch.cat((t_pos, torch.ones(self.nx) * t[step]), dim=0)
            batch = torch.cat((batch, torch.ones(self.nx) * b), dim=0)
            # pde variables
            batch_variables = torch.tensor([variables[k][b] for k in variables]).unsqueeze(0).repeat(self.nx, 1) # [num_variables] -> [1, num_variables] -> [bs*nx, num_variables]
            pde_variables = torch.cat((pde_variables, batch_variables), dim=0)
        # print(f"u: {u.shape}, y: {y.shape}, x_pos: {x_pos.shape}, t_pos:{t_pos.shape}, batch: {batch.shape}, pde_variables: {pde_variables.shape}")
        
        # edge index
        x_min, x_max = self.pde.spatial_domain[0]
        res = self.pde.resolution[0]
        dx = (x_max - x_min) / res
        if self.pde.spatial_dim == 1:
            radius = self.n * dx + dx / 10
        elif self.pde.spatial_dim == 2:
            radius = self.n * dx * np.sqrt(2) + dx / 10
        else: # TODO 3D
            pass
        edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        # print(f"edge index: {edge_index.shape}")

        # build graph data
        graph = Data(x=u, edge_index=edge_index)
        graph.y = y
        graph.x_pos = x_pos
        graph.t_pos = t_pos
        graph.batch = batch.long()
        graph.variables = pde_variables.float()

        graph.validate(raise_on_error=True) # validate graph data

        return graph
    

    def create_next_graph(self,
                          graph: Data,
                          pred: torch.Tensor,
                          labels: torch.Tensor,
                          steps: list) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        graph.x = pred # pred: [bs*nx, tw]

        # update labels and timesteps
        t = torch.linspace(self.pde.tmin, self.pde.tmax, self.nt)
        y, t_pos = torch.Tensor(), torch.Tensor()
        for (labels_batch, step) in zip(labels, steps):
            y = torch.cat((y, torch.transpose(labels_batch, 0, 1)), dim=0)
            t_pos = torch.cat((t_pos, torch.ones(self.nx) * t[step]), dim=0)
        graph.y = y
        graph.t_pos = t_pos

        graph.validate(raise_on_error=True) # validate graph data

        return graph
    

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def timer(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        from time import time
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        cost_time = end_time - start_time
        print(f'{func.__name__} cost time: {cost_time:.4f} s')
        return result
    return func_wrapper


def timeit(runs=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0
            for _ in range(runs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                total_time += (end_time - start_time)
                print(f"cost time: {end_time - start_time:.4f} s")
            average_time = total_time / runs
            print(f"total cost time: {total_time:.4f} s, average cost time: {average_time:.4f} s")
            return result
        return wrapper
    return decorator


def to_PDEBench_format(graph_data: torch.Tensor, batch_size: int, pde: PDE):
    """Convert graph data to PDEBench formart [bs, x1, ..., xd, t, v]
    Args:
        graph_data (torch.Tensor): input/output data of model with shape [bs*nx, t, v]
        batch_size (int): batch size
        pde (PDE): PDE to solve
    """
    assert (len(graph_data.shape) == 3)
    output_shape = [batch_size]
    for v in pde.resolution:
        output_shape.append(v)
    output_shape.append(graph_data.shape[-2])
    output_shape.append(graph_data.shape[-1])
    
    return graph_data.reshape(output_shape)


# test dataloader
if __name__ == "__main__":
    # launch slowly but iter quickly
    file_name = "2D_Allen-Cahn_0.0001_1.hdf5"
    saved_folder = "/home/data2/fluidbench/2D/Allen-Cahn/"
    # file_name = "1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5"
    # saved_folder = "/data1/zhouziyang/datasets/pdebench/1D/CFD/Train/"
    variables = {"c1": 0.0001, "c2": 1}

    tic = time.time()
    dataset = MPNNDatasetSingle(file_name, saved_folder, variables=variables)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    toc = time.time()
    wait_time = toc-tic

    tic = time.time()
    for u, x, variables in dataloader:
        print(u.shape, x.shape, variables)
    toc = time.time()
    print(f"Time for waiting: {wait_time}s, Time for one epoch: {toc-tic}s")

    # launch quickly but iter slowly
    # file_name = "2D_diff-react_NA_NA.h5"
    # saved_folder = "/home/data2/fluidbench/2D/diffusion-reaction/"
    # variables = {}

    # tic = time.time()
    # dataset = MPNNDatasetMult(file_name, saved_folder, variables=variables)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    # toc = time.time()
    # wait_time = toc-tic

    # tic = time.time()
    # for u, x, variables in dataloader:
    #     print(u.shape, x.shape, variables)
    # toc = time.time()
    # print(f"Time for waiting: {wait_time}, Time for one epoch: {toc-tic}s")