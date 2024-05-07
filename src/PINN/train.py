import deepxde as dde
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import torch
import h5py

from typing import Tuple

from utils import (
    PINNDatasetRadialDambreak,
    PINNDatasetDiffReact,
    PINNDataset2D,
    PINNDatasetDiffSorption,
    PINNDataset1Dpde,
    PINNDataset2Dpde,
    PINNDataset3Dpde,
    PINNDatasetBurgers,
    PINNDatasetAC,
    PINNDatasetDarcy
)
from pde_definitions import (
    pde_diffusion_reaction,
    pde_swe2d,
    pde_diffusion_sorption,
    pde_adv1d,
    pde_diffusion_reaction_1d,
    pde_burgers1D,
    pde_CFD1d,
    pde_CFD2d,
    pde_CFD3d,
    pde_burgers2D,
    pde_Allen_Cahn,
    pde_Cahn_Hilliard,
    pde_Allen_Cahn2d,
    pde_darcy_flow,
    pde_euler,
    pde_BS,
    pde_Maxwell
)

from metrics import metrics, metric_func, L1RE


def setup_diffusion_sorption(filename, root_path, seed):
    # TODO: read from dataset config file
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 500.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    D = 5e-4

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    bc_d = dde.icbc.DirichletBC(
        geomtime,
        lambda x: 1.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
    )

    def operator_bc(inputs, outputs, X):
        # compute u_t
        du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return outputs - D * du_x

    bc_d2 = dde.icbc.OperatorBC(
        geomtime,
        operator_bc,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0),
    )

    dataset = PINNDatasetDiffSorption(filename, root_path, seed)

    initial_input, initial_u = dataset.get_initial_condition()
    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1])

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_sorption,
        [ic, bc_d, bc_d2, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")

    def transform_output(x, y):
        return torch.relu(y)

    net.apply_output_transform(transform_output)

    model = dde.Model(data, net)

    return model, dataset

def setup_diffusion_reaction(filename, root_path, seed):

    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0, 5.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

    dataset = PINNDatasetDiffReact(filename, root_path, seed)
    initial_input, initial_u, initial_v = dataset.get_initial_condition()

    ic_data_u = dde.icbc.PointSetBC(initial_input, initial_u, component=0)
    ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_reaction,
        [bc, ic_data_u, ic_data_v, bc_data_u, bc_data_v],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_burgers(filename, root_path, seed, aux_params):
    pde = lambda x, y: pde_burgers2D(x, y, aux_params[0])

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((0, 0), (2, 2))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

    dataset = PINNDatasetBurgers(filename, root_path, seed)
    initial_input, initial_u, initial_v = dataset.get_initial_condition()

    ic_data_u = dde.icbc.PointSetBC(initial_input, initial_u, component=0)
    ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    # ratio = int(len(dataset) * 0.3)

    # data_split, _ = torch.utils.data.random_split(
    #     dataset,
    #     [ratio, len(dataset) - ratio],
    #     generator=torch.Generator(device="cuda").manual_seed(42),
    # )

    # data_gt = data_split[:]

    # bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    # bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic_data_u, ic_data_v],# bc_data_u, bc_data_v],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset


def setup_AC2d(filename, root_path, seed, aux_params):
    pde = lambda x, y: pde_Allen_Cahn2d(x, y, aux_params[0], aux_params[1])

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((0, 0), (1.0, 1.0))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # ic = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial)

    dataset = PINNDatasetAC(filename, root_path, seed)
    initial_input, initial_u = dataset.get_initial_condition()

    ic_data_u = dde.icbc.PointSetBC(initial_input.cpu(), initial_u)
    bc_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    bc_y = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary)
    # ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    # ratio = int(len(dataset) * 0.3)

    # data_split, _ = torch.utils.data.random_split(
    #     dataset,
    #     [ratio, len(dataset) - ratio],
    #     generator=torch.Generator(device="cuda").manual_seed(42),
    # )

    # data_gt = data_split[:]

    # bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    # bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_u, bc_x, bc_y],# bc_data_u, bc_data_v],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_darcy_2d(filename, root_path, seed, aux_params):
    

    # TODO: read from dataset config file
    geom = dde.geometry.Cuboid((0., 0., 0.), (1., 1., 1.))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # ic = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial)

    dataset = PINNDatasetDarcy(filename, root_path, seed)

    pde = lambda x, y: pde_darcy_flow(x, y, aux_params[0])
    initial_input, initial_u = dataset.get_initial_condition()
    # print(initial_input.shape)

    ic_data_u = dde.icbc.PointSetBC(initial_input.cpu(), initial_u)
    bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
    # bc_y = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary)
    # ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    # ratio = int(len(dataset) * 0.3)

    # data_split, _ = torch.utils.data.random_split(
    #     dataset,
    #     [ratio, len(dataset) - ratio],
    #     generator=torch.Generator(device="cuda").manual_seed(42),
    # )

    # data_gt = data_split[:]

    # bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    # bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_u, bc],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([4] + [40] * 6 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_BS_2d(filename, root_path, seed, aux_params):
    

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((0, 0), (1.0, 1.0))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # ic = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial)

    dataset = PINNDatasetAC(filename, root_path, seed)

    pde = lambda x, y: pde_BS(x, y)
    initial_input, initial_u = dataset.get_initial_condition()
    # print(initial_input.shape)

    ic_data_u = dde.icbc.PointSetBC(initial_input.cpu(), initial_u)
    # bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
    bc_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    bc_y = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary)
    # ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    # ratio = int(len(dataset) * 0.3)

    # data_split, _ = torch.utils.data.random_split(
    #     dataset,
    #     [ratio, len(dataset) - ratio],
    #     generator=torch.Generator(device="cuda").manual_seed(42),
    # )

    # data_gt = data_split[:]

    # bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    # bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_u],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset


def setup_swe_2d(filename, root_path, seed) -> Tuple[dde.Model, PINNDataset2D]:

    dataset = PINNDatasetRadialDambreak(filename, root_path, seed)

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-2.5, -2.5), (2.5, 2.5))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic_h = dde.icbc.IC(
        geomtime,
        dataset.get_initial_condition_func(),
        lambda _, on_initial: on_initial,
        component=0,
    )
    ic_u = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=1
    )
    ic_v = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=2
    )

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde_swe2d,
        [bc, ic_h, ic_u, ic_v, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def _boundary_r(x, on_boundary, xL, xR):
    return (on_boundary and np.isclose(x[0], xL)) or (on_boundary and np.isclose(x[0], xR))

def setup_pde1D(filename="1D_Advection_Sols_beta0.1.hdf5",
                root_path='data',
                val_batch_idx=0,
                input_ch=2,
                output_ch=1,
                hidden_ch=40,
                xL=-1.,
                xR=1.,
                if_periodic_bc=True,
                aux_params=[0.1]):

    # TODO: read from dataset config file
    geom = dde.geometry.Interval(xL, xR)
    boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, xL, xR)
    if filename[0] == 'R':
        timedomain = dde.geometry.TimeDomain(0, 1.0)
        pde = lambda x, y : pde_diffusion_reaction_1d(x, y, aux_params[0], aux_params[1])
    else:
        if filename.split('_')[1]=="Allen-Cahn":
            print('AC')
            timedomain = dde.geometry.TimeDomain(0, 1.0)
            pde = lambda x, y: pde_Allen_Cahn(x, y, aux_params[0], aux_params[1])
        elif filename.split('_')[1]=="Cahn-Hilliard":
            print('CH')
            timedomain = dde.geometry.TimeDomain(0, 1.0)
            pde = lambda x, y: pde_Cahn_Hilliard(x, y, aux_params[0], aux_params[1])
        elif filename.split('_')[1][0]=='A':
            print('Adv')
            timedomain = dde.geometry.TimeDomain(0, 2.0)
            pde = lambda x, y: pde_adv1d(x, y, aux_params[0])
        elif filename.split('_')[1][0] == 'B':
            timedomain = dde.geometry.TimeDomain(0, 2.0)
            pde = lambda x, y: pde_burgers1D(x, y, aux_params[0])
        elif filename.split('_')[1][0]=='C':
            timedomain = dde.geometry.TimeDomain(0, 1.0)
            pde = lambda x, y: pde_CFD1d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset1Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    if filename.split('_')[1][0] == 'C' and filename.split('_')[1][1] != 'a':
        ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,0].unsqueeze(1), component=0)
        ic_data_v = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,1].unsqueeze(1), component=1)
        ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,2].unsqueeze(1), component=2)
    else:
        ic_data_u = dde.icbc.PointSetBC(initial_input.cpu(), initial_u, component=0)
    # prepare boundary condition
    if if_periodic_bc:
        if filename.split('_')[1][0] == 'C' and filename.split('_')[1][1] != 'a':
            bc_D = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            bc_V = dde.icbc.PeriodicBC(geomtime, 1, boundary_r)
            bc_P = dde.icbc.PeriodicBC(geomtime, 2, boundary_r)

            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_d, ic_data_v, ic_data_p, bc_D, bc_V, bc_P],
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000,
            )
        else:
            print('PB is used.')
            bc = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_u, bc],
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000,
            )
    else:
        ic = dde.icbc.IC(
            geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
        )
        bd_input, bd_uL, bd_uR = dataset.get_boundary_condition()
        bc_data_uL = dde.icbc.PointSetBC(bd_input.cpu(), bd_uL, component=0)
        bc_data_uR = dde.icbc.PointSetBC(bd_input.cpu(), bd_uR, component=0)

        data = dde.data.TimePDE(
            geomtime,
            pde,
            [ic, bc_data_uL, bc_data_uR],
            num_domain=1000,
            num_boundary=1000,
            num_initial=5000,
        )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_CFD2D(filename="2D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=4,
                hidden_ch=40,
                xL=0.,
                xR=1.,
                yL=0.,
                yR=1.,
                if_periodic_bc=True,
                aux_params=[1.6667]):

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0., 1.0)
    pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset2Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    # prepare boundary condition
    bc_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    bc_y = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_p, bc_x, bc_y],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_CFD3D(filename="3D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=4,
                hidden_ch=40,
                aux_params=[1.6667]):

    # TODO: read from dataset config file
    geom = dde.geometry.Cuboid((0., 0., 0.), (1., 1., 1.))
    timedomain = dde.geometry.TimeDomain(0., 1.0)
    pde = lambda x, y: pde_CFD3d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset3Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    ic_data_vz = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,4].unsqueeze(1), component=4)
    # prepare boundary condition
    bc_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    bc_y = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary)
    bc_z = dde.icbc.PeriodicBC(geomtime, 2, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_vz, ic_data_p, bc_x, bc_y, bc_z],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_Euler3D(filename="Turb_M01.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=4,
                hidden_ch=40,
                aux_params=[1.6667]):

    # TODO: read from dataset config file
    geom = dde.geometry.Cuboid((0., 0., 0.), (1., 1., 1.))
    timedomain = dde.geometry.TimeDomain(0., 1.0)
    pde = lambda x, y: pde_euler(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset3Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    ic_data_vz = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,4].unsqueeze(1), component=4)
    # prepare boundary condition
    bc_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    bc_y = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary)
    bc_z = dde.icbc.PeriodicBC(geomtime, 2, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_vz, ic_data_p, bc_x, bc_y, bc_z],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_Maxwell3D(filename="Turb_M01.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=4,
                hidden_ch=40,
                aux_params=[1.6667]):

    # TODO: read from dataset config file
    geom = dde.geometry.Cuboid((0., 0., 0.), (1.575e-05, 1.575e-05, 1.575e-05))
    timedomain = dde.geometry.TimeDomain(0., 1.6682531e-13)
    pde = lambda x, y: pde_Maxwell(x, y)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    data_path = os.path.join(root_path, filename)
    h5_file = h5py.File(data_path, "r")

        
    seed_group = h5_file['0900']['data']
    # print(h5_file['0900']['grid']['y'][-1])

    # p = input()

    dataset = seed_group
    # prepare initial condition
    # initial_input, initial_u = dataset.get_initial_condition()
    # ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    # ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    # ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    # ic_data_vz = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    # ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,4].unsqueeze(1), component=4)
    # prepare boundary condition
    # bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=0)
    # bc2 = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=1)
    # bc3 = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=2)
    # bc4 = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=3)
    # bc5 = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=4)
    # bc6 = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=5)
    bc1 = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    bc2 = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary)
    bc3 = dde.icbc.PeriodicBC(geomtime, 2, lambda _, on_boundary: on_boundary)
    # bc4 = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, component= 1)
    # bc5 = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, component= 1)
    # bc6 = dde.icbc.PeriodicBC(geomtime, 2, lambda _, on_boundary: on_boundary, component= 1)
    # bc7 = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, component= 2)
    # bc8 = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, component= 2)
    # bc9 = dde.icbc.PeriodicBC(geomtime, 2, lambda _, on_boundary: on_boundary, component= 2)
    # bc10 = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, component= 3)
    # bc11 = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, component= 3)
    # bc12 = dde.icbc.PeriodicBC(geomtime, 2, lambda _, on_boundary: on_boundary, component= 3)
    # bc_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    # bc_y = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary)
    # bc_z = dde.icbc.PeriodicBC(geomtime, 2, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc1, bc2, bc3],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def _run_training(scenario, epochs, learning_rate, model_update, flnm,
                  input_ch, output_ch,
                  root_path, val_batch_idx, if_periodic_bc, aux_params,
                  if_single_run,
                  seed):
    if scenario == "swe2d":
        model, dataset = setup_swe_2d(filename=flnm, root_path=root_path, seed=seed)
        n_components = 1
    elif scenario == "darcy2d":
        model, dataset = setup_darcy_2d(filename=flnm, root_path=root_path, seed=seed, aux_params = aux_params)
        n_components = 1
    elif scenario == "BS2d":
        model, dataset = setup_BS_2d(filename=flnm, root_path=root_path, seed=seed, aux_params = aux_params)
        n_components = 1
    elif scenario == "diff-react":
        model, dataset = setup_diffusion_reaction(filename=flnm, root_path=root_path, seed=seed)
        n_components = 2
    elif scenario == "diff-sorp":
        model, dataset = setup_diffusion_sorption(filename=flnm, root_path = root_path, seed=seed)
        n_components = 1
    elif scenario == 'burgers':
        model, dataset = setup_burgers(filename=flnm, root_path=root_path, seed=seed, aux_params = aux_params)
        n_components = 2
    elif scenario == "AC2d":
        model, dataset = setup_AC2d(filename=flnm, root_path=root_path, seed=seed, aux_params = aux_params)
        n_components = 1
    elif scenario == "pde1D":
        model, dataset = setup_pde1D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     if_periodic_bc=if_periodic_bc,
                                     aux_params=aux_params)
        if flnm.split('_')[1][0] == 'C' and flnm.split('_')[1][1] != 'a':
            n_components = 3
        else:
            n_components = 1
    elif scenario == "CFD2D":
        model, dataset = setup_CFD2D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     aux_params=aux_params)
        n_components = 4
    elif  scenario == "CFD3D":
        model, dataset = setup_CFD3D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     aux_params=aux_params)
        n_components = 5
    elif  scenario == "Euler3D":
        model, dataset = setup_Euler3D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     aux_params=aux_params)
        n_components = 5
    elif  scenario == "Max3D":
        model, dataset = setup_Maxwell3D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     aux_params=aux_params)
        n_components = 5
    else:
        raise NotImplementedError(f"PINN training not implemented for {scenario}")

    # filename
    if if_single_run:
        model_name = flnm +'_' + str(val_batch_idx) + "_PINN"
    else:
        model_name = flnm[:-5] + "_PINN"

    print(model_name)

    checker = dde.callbacks.ModelCheckpoint(
        f"{model_name}.pt", save_better_only=True, period=5000
    )

    model.compile("adam", 
                  lr=learning_rate,
                  decay=('step', 2000, 0.6)
                  )
    losshistory, train_state = model.train(
        epochs=epochs, display_every=model_update, callbacks=[checker]
    )

    dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir='./outputs', loss_fname=model_name+"loss.dat")
    test_input, test_gt = dataset.get_test_data(
        n_last_time_steps=10, n_components=n_components
    )
    # select only n_components of output
    # dirty hack for swe2d where we predict more components than we have data on
    test_pred = torch.tensor(model.predict(test_input.cpu())[:, :n_components])

    # prepare data for metrics eval
    test_pred = dataset.unravel_tensor(
        test_pred, n_last_time_steps=10, n_components=n_components
    )
    test_gt = dataset.unravel_tensor(
        test_gt, n_last_time_steps=10, n_components=n_components
    )

    if if_single_run:
        errs = metric_func(test_pred, test_gt)
        errors = [np.array(err.cpu()) for err in errs]
        err_L1RE = L1RE(test_pred, test_gt).cpu()
        print(errors)
        print(err_L1RE)

        pickle.dump(errors, open(model_name + ".pickle", "wb"))
        return

        # plot sample
        plot_input = dataset.generate_plot_input(time=1.0)
        if scenario == "pde1D":
            xdim = dataset.xdim
            dim = 1
        elif scenario == "diff-sorp":
            dim = 1
            xdim = dataset.config["sim"]["xdim"]
        elif scenario == "diff-react":
            dim = 2
            xdim = dataset.config["sim"]["xdim"]
            ydim = dataset.config["sim"]["ydim"]
        elif scenario == "burgers":
            dim = 2
            xdim = dataset.config["sim"]["xdim"]
            ydim = dataset.config["sim"]["ydim"]
        else:
            dim = dataset.config["plot"]["dim"]
            xdim = dataset.config["sim"]["xdim"]
            if dim == 2:
                ydim = dataset.config["sim"]["ydim"]

        y_pred = model.predict(plot_input)[:, 0]
        if dim == 1:
            plt.figure()
            plt.plot(y_pred)
        elif dim == 2:
            im_data = y_pred.reshape(xdim, ydim)
            plt.figure()
            plt.imshow(im_data)

        plt.savefig(f"{model_name}.png")

        # TODO: implement function to get specific timestep from dataset
        # y_true = dataset[:][1][-xdim * ydim :]
    else:
        return test_pred, test_gt, model_name

def run_training(scenario, epochs, learning_rate, model_update, flnm,
                 input_ch=1, output_ch=1,
                 root_path='../data/', val_num=10, if_periodic_bc=True, aux_params=[None], seed='0000', val_batch_idx = 9000):

    if val_num == 1:  # single job
        _run_training(scenario, epochs, learning_rate, model_update, flnm,
                      input_ch, output_ch, 
                      root_path, val_batch_idx, if_periodic_bc, aux_params,
                      if_single_run=True, seed=seed)
    else:
        for val_batch_idx in range(-1, -val_num, -1):
            test_pred, test_gt, model_name = _run_training(scenario, epochs, learning_rate, model_update, flnm,
                                                           input_ch, output_ch,
                                                           root_path, val_batch_idx, if_periodic_bc, aux_params,
                                                           if_single_run=False, seed=seed)
            if val_batch_idx == -1:
                pred, target = test_pred.unsqueeze(0), test_gt.unsqueeze(0)
            else:
                pred = torch.cat([pred, test_pred.unsqueeze(0)], 0)
                target = torch.cat([target, test_gt.unsqueeze(0)], 0)

        errs = metric_func(test_pred, test_gt)
        errors = [np.array(err.cpu()) for err in errs]
        err_L1RE = L1RE(test_pred, test_gt).cpu()
        print(errors)
        print(err_L1RE)
        pickle.dump(errors, open(model_name + ".pickle", "wb"))


if __name__ == "__main__":
    # run_training(
    #     scenario="diff-sorp",
    #     epochs=100,
    #     learning_rate=1e-3,
    #     model_update=500,
    #     flnm="2D_diff-sorp_NA_NA_0000.h5",
    #     seed="0000",
    # )
    run_training(
        scenario="diff-react",
        epochs=100,
        learning_rate=1e-3,
        model_update=500,
        flnm="2D_diff-react_NA_NA.h5",
        seed="0000",
    )
    # run_training(
    #     scenario="swe2d",
    #     epochs=100,
    #     learning_rate=1e-3,
    #     model_update=500,
    #     flnm="radial_dam_break_0000.h5",
    #     seed="0000",
    # )
    # run_training(
    #         scenario=cfg.args.scenario,
    #         epochs=cfg.args.epochs,
    #         learning_rate=cfg.args.learning_rate,
    #         model_update=cfg.args.model_update,
    #         flnm=cfg.args.filename,
    #         seed=cfg.args.seed,
    #         input_ch=cfg.args.input_ch,
    #         output_ch=cfg.args.output_ch,
    #         root_path=cfg.args.root_path,
    #         val_num=cfg.args.val_num,
    #         if_periodic_bc=cfg.args.if_periodic_bc,
    #         aux_params=cfg.args.aux_params
    #     )
