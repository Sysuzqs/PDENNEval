# coding=utf-8
import argparse
from functools import reduce
import os
import sys
from torch.utils.data import DataLoader
import yaml

from train import get_model, get_dataset
from utils import setup_seed

sys.path.append('..')
from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method, lipschitz_bn



def compute_module_input_output_size(model, input_size):
    """ Tag all modules with `input_sizes` and 'output_size' attribute
    """
    def hook_fn(module, args, output):
        module.input_sizes = [x.size() for x in args]
        module.output_size = output.size()
    execute_through_model(hook_fn, model, input_size=input_size)


def spec_unet(module, input, output):
    max_sv = 20
    if is_convolution_or_linear(module):
        input_dim = reduce(lambda x, y: x*y ,module.input_sizes[0])
        output_dim = reduce(lambda x, y: x*y , module.output_size)
        max_rank = min(input_dim, output_dim)
        n_sv = min(max_rank, max_sv)
        s, u, v = k_generic_power_method(module.forward, module.input_sizes[0], n_sv, 
                                         max_iter=500, verbose=True, use_cuda=True)
        module.spectral_norm = s
        module.u = u
        module.v = v
        print(module.__class__.__name__, s)

    if is_batch_norm(module):
        s = lipschitz_bn(module)
        module.spectral_norm = s
        print(module.__class__.__name__, s)


def save_singular(unet, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        
    for name in unet._modules:
        module = getattr(unet, name)
        if issubclass(module.__class__, torch.nn.Sequential):
            for i in range(len(module)):
                if hasattr(module[i], 'spectral_norm'):
                    torch.save(module[i].spectral_norm, open(os.path.join(root_path, f'feat-singular-{name}-{i}'), 'wb'))
                if hasattr(module[i], 'u'):
                    torch.save(module[i].u, open(os.path.join(root_path, f'feat-left-sing-{name}-{i}'), 'wb'))
                    torch.save(module[i].v, open(os.path.join(root_path, f'feat-right-sing-{name}-{i}'), 'wb'))
        else:
            if hasattr(module, 'spectral_norm'):
                    torch.save(module.spectral_norm, open(os.path.join(root_path, f'feat-singular-{name}'), 'wb'))
            if hasattr(module, 'u'):
                torch.save(module.u, open(os.path.join(root_path, f'feat-left-sing-{name}'), 'wb'))
                torch.save(module.v, open(os.path.join(root_path, f'feat-right-sing-{name}'), 'wb'))


if __name__ == '__main__':
    # parse command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("sv_root", type=str, help="Root path to save singular value")
    cmd_args = parser.parse_args()

    # read config file
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)

    # initialize
    setup_seed(args["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args["model_path"])

    # get model input size from dataset
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
    print(f"input_size: {input_size}")

    # get model
    spatial_dim = len(input_size) - 2
    # if spatial_dim == 3:
    #     torch.backends.cudnn.enabled = False
    model = get_model(spatial_dim, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
 
    # compute singular value
    # compute_module_input_sizes(model, input_size)
    compute_module_input_output_size(model, input_size)
    execute_through_model(spec_unet, model)

    # save singular value
    model_name = os.path.basename(args["model_path"])
    saved_root = os.path.join(cmd_args.sv_root, args['pde_name'], model_name)
    save_singular(model, saved_root)