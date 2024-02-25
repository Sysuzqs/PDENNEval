# coding=utf-8
import argparse
import os
from torch.utils.data import DataLoader
import yaml
import sys

from train import get_model, get_dataset

sys.path.append('..')
from lipschitz_utils import *
from seqlip import optim_nn_pca_greedy


def get_unet_forward_paths(model):
    paths = []
    # Concatenate makes branches
    for i in range(1, 5):
        path = []
        for j in range(1, i+1):
            path.append(f"encoder{j}")
        for j in range(i, 0, -1):
            path.append(f"decoder{j}")
            if j-1 > 0:
                path.append(f"upconv{j-1}")
        path.append('conv')
        paths.append(path)
    # The longest path
    path = []
    for name in model._modules:
        if 'pool' in name:
            continue
        else:
            path.append(name)
    paths.append(path)
    # Print all paths
    for path in paths:
        print(path)

    return paths


if __name__ == "__main__":
    # Parse command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("sv_root", type=str, help="Root path of saved singular value")
    parser.add_argument("--n_sv", type=int, default=1, help="The number of singular values used to estimate. (default: 1)")
    cmd_args = parser.parse_args()
    print(f"n_sv=={cmd_args.n_sv}")

    # Read config file and init
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.path.basename(args["model_path"])
    saved_root = os.path.join(cmd_args.sv_root, args['pde_name'], model_name)

    # Get model input size from dataset
    train_data, val_data = get_dataset(args)
    dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    x, y = next(iter(dataloader))
    input_shape = list(x.shape)[:-2]
    input_shape.append(-1) # (bs, x1, ..., xd, -1)
    model_input = x.reshape(input_shape)
    input_permute = [0, -1]
    input_permute.extend([i for i in range(1, len(model_input.shape)-1)])
    model_input = model_input.permute(input_permute)
    input_size = list(model_input.shape)
    print(f"The input size of model is {input_size}")

    # Get model
    spatial_dim = len(input_size) - 2
    model = get_model(spatial_dim, args)

    # Get forward paths 
    paths = get_unet_forward_paths(model)
    print(f"There are {len(paths)} paths in model.")

    # Estimate the upper bound of lipschitz constant
    print(f'Start estimate the lipschitz constant upper bound of {model_name}.')
    print("="*50)
    max_sv = cmd_args.n_sv
    lips = []
    lip_spectrals = []
    for path in paths:
        print(f"Path: {path}")
        # get layers in path
        conv_layers = []
        bn_layers = []
        for block in path:
            if 'coder' in block or 'bottleneck' in block:
                conv_layers.append(f'{block}-{0}')
                bn_layers.append(f'{block}-{1}')
                conv_layers.append(f'{block}-{3}')
                bn_layers.append(f'{block}-{4}')
            else:
                conv_layers.append(f'{block}')

        # Estimate
        lip = 1
        lip_spectral = 1
        # seqlip for convolution layer
        for i in range(len(conv_layers)-1):
            U = torch.load(os.path.join(saved_root, f'feat-left-sing-{conv_layers[i]}'))
            su = torch.load(os.path.join(saved_root, f'feat-singular-{conv_layers[i]}'))
            n_su = min(len(su), max_sv)
            U = torch.cat(U[:n_su], dim=0).view(n_su, -1)
            su = su[:n_su]

            V = torch.load(os.path.join(saved_root, f'feat-right-sing-{conv_layers[i+1]}'))
            sv = torch.load(os.path.join(saved_root, f'feat-singular-{conv_layers[i+1]}'))
            n_sv = min(len(sv), max_sv)
            V = torch.cat(V[:n_sv], dim=0).view(n_sv, -1)
            sv = sv[:n_sv]

            U, V = U.cpu(), V.cpu()

            if i == 0:
                sigmau = torch.diag(torch.Tensor(su))
            else:
                sigmau = torch.diag(torch.sqrt(torch.Tensor(su)))

            if i == len(conv_layers)-2:
                sigmav = torch.diag(torch.Tensor(sv))
            else:
                sigmav = torch.diag(torch.sqrt(torch.Tensor(sv)))

            expected = sigmau[0,0] * sigmav[0,0]
            lip_spectral *= float(expected)
            
            # deal with concatenate
            if "decoder" in conv_layers[i+1] and "encoder" in conv_layers[i]:
                print(f'Deal with concatenate between {conv_layers[i]} and {conv_layers[i+1]}')
                if 'bottleneck' in path:
                    V = V[:,:U.t().shape[0]]
                else:
                    V = V[:,-U.t().shape[0]:]

            # optimize
            try:
                approx, _ = optim_nn_pca_greedy(sigmav @ V, U.t() @ sigmau, verbose=False)
                print(f'Approximation: {approx}')
                lip *= float(approx)
            except:
                print(f'Probably something went wrong when dealing with layer {conv_layers[i]} and {conv_layers[i+1]}')
                print(f"Use expected value: {expected}")
                lip *= float(expected)

        # batch norm layer
        for i in range(len(bn_layers)):
            sigma = torch.load(os.path.join(saved_root, f'feat-singular-{bn_layers[i]}'))
            print(f"batch layer{i}: {sigma}")
            lip_spectral *= sigma
            lip *= sigma
            
        # append
        lip_spectrals.append(lip_spectral)
        lips.append(lip)
        print(f"Lipschitz spectral of this path: {lip_spectral}")
        print(f"Estimated upper bound of this path: {lip}")
        print('='*50)

    print(f"Lipschitz spectral of model: {sum(lip_spectrals)}")
    print(f"The Lipschitz constant upper bound of model: {sum(lips)}")
    print('Done.')