
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.io as sci

import argparse

try:
    from scipy.stats import qmc
except:
    result = subprocess.run(["bash", '-c', 'pip install scipy'])
    from scipy.stats import qmc

from tqdm import *


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train High Dimension Poisson Equation using WAN", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--s",
        type=int,
        default=12160,
        help="The random seed"
    )
    parser.add_argument(
        "--d",
        type=int,
        default=2,
        help="The dimension of space"
    )
    parser.add_argument(
        "--co",
        type=str,
        default="0",
        help="The number of gpu used",
    )
    parser.add_argument(
        "--i",
        type=int,
        default=200,
        help="The iteration number"
    )
    parser.add_argument(
        "--b",
        type=int,
        default=1000,
        help="The beta number"
    )
    return parser.parse_args(arg_list)


def sampleCubeMC(dim, l_bounds, u_bounds, N=100):
    '''Monte Carlo Sampling

    Get the sampling points by Monte Carlo Method.

    Args:
        dim:      The dimension of space
        l_bounds: The lower boundary
        u_bounds: The upper boundary
        N:        The number of sample points

    Returns:
        numpy.array: An array of sample points
    '''
    sample = []
    for i in range(dim):
        sample.append( np.random.uniform(l_bounds[i], u_bounds[i], [N, 1]) ) 
    data = np.concatenate(sample, axis=1)
    return data


args = get_arguments()
torch.manual_seed(args.s)
np.random.seed(args.s)
DIMENSION = args.d
a = [-1 for _ in range(DIMENSION)]
b = [ 1 for _ in range(DIMENSION)]
# Netword
DIM_INPUT  = DIMENSION
NUM_UNIT   = 30
DIM_OUTPUT = 1
NUM_BLOCKS = 4
# Optimizer
IS_DECAY           = 0
LEARN_RATE         = 1e-3
LEARN_FREQUENCY    = 50
LEARN_LOWWER_BOUND = 1e-5
LEARN_DECAY_RATE   = 0.99
LOSS_FN            = nn.MSELoss()
# Training
CUDA_ORDER      = args.co
NUM_INT_SAMPLE  = 10000
NUM_BD_SAMPLE   = 50
NUM_TRAIN_TIMES = 1
NUM_ITERATION   = args.i
# Testing
NUM_TEST_SAMPLE = 10000
TEST_FREQUENCY  = 1
# Loss weight
BETA = args.b
# Save model
IS_SAVE_MODEL = 1


class PoissonEquation(object):
    def __init__(self, dimension, device):
        self.D = dimension
        self.device = device

    def f(self, X):
        f = torch.ones([X.shape[0], 1])
        return f.reshape(-1, 1).detach().to(self.device)

    def g(self, X):
        u = torch.zeros([X.shape[0], 1])
        return u.reshape(-1, 1).detach().to(self.device)

    def interior(self, N=100):
        # np.spacing(1)
        X = []
        l_bounds = [-1, -1]
        u_bounds = [0, 0]
        X.append(sampleCubeMC(self.D, l_bounds, u_bounds, N // 3))
        l_bounds = [-1, 0]
        u_bounds = [0, 1]
        X.append(sampleCubeMC(self.D, l_bounds, u_bounds, N // 3))
        l_bounds = [0, -1]
        u_bounds = [1, 0]
        X.append(sampleCubeMC(self.D, l_bounds, u_bounds, N // 3))
        X = torch.FloatTensor(np.concatenate(X, axis=0))
        return X.requires_grad_(True).to(self.device)

    def boundary(self, n=100):
        # sample on boundary of [-1,1]^2\[0,1]^2
        x_boundary = []
        line1 = np.random.uniform(-1, 1, 2 * n)
        line2 = np.random.uniform(-1, 0, n)
        line3 = np.random.uniform(0, 1, n)
        x_boundary.append(np.stack([line1, -np.ones(2 * n)], axis=1))
        x_boundary.append(np.stack([-np.ones(2 * n), line1], axis=1))
        x_boundary.append(np.stack([line2, np.ones(n)], axis=1))
        x_boundary.append(np.stack([np.ones(n), line2], axis=1))
        x_boundary.append(np.stack([line3, np.zeros(n)], axis=1))
        x_boundary.append(np.stack([np.zeros(n), line3], axis=1))
        x_boundary = np.concatenate(x_boundary, axis=0)
        x_boundary = torch.FloatTensor(x_boundary).requires_grad_(True).to(self.device)
        return x_boundary


def fun_w(x, low=-1.0, up=1.0, mid=0.0):
    dim = 2
    I1 = 0.210987
    x_list = torch.split(x, 1, dim=1)

    h_len_mid = (mid - low) / 2.0
    h_len = (up - low) / 2.0

    x_scale_list1 = []
    x_scale_list2 = []

    for i in range(dim):
        if i == 0:
            x_scale1 = (x_list[i] - low - h_len_mid) / h_len_mid
            x_scale2 = (x_list[i] - low - h_len) / h_len
        elif i == 1:
            x_scale1 = (x_list[i] - low - h_len) / h_len
            x_scale2 = (x_list[i] - low - h_len_mid) / h_len_mid
        else:
            x_scale1 = (x_list[i] - low - h_len) / h_len
            x_scale2 = (x_list[i] - low - h_len) / h_len

        x_scale_list1.append(x_scale1)
        x_scale_list2.append(x_scale2)

    z_x1_list = []
    z_x2_list = []

    for i in range(dim):
        supp_x1 = torch.gt(1 - torch.abs(x_scale_list1[i]), 0.0)
        supp_x2 = torch.gt(1 - torch.abs(x_scale_list2[i]), 0.0)

        z_x1 = torch.where(supp_x1, torch.exp(1 / (torch.pow(x_scale_list1[i], 2) - 1)) / I1,
                           torch.zeros_like(x_scale_list1[i]))
        z_x2 = torch.where(supp_x2, torch.exp(1 / (torch.pow(x_scale_list2[i], 2) - 1)) / I1,
                           torch.zeros_like(x_scale_list2[i]))

        z_x1_list.append(z_x1)
        z_x2_list.append(z_x2)

    w1_val = torch.tensor(1.0)
    w2_val = torch.tensor(1.0)

    for i in range(dim):
        w1_val = torch.mul(w1_val, z_x1_list[i])
        w2_val = torch.mul(w2_val, z_x2_list[i])

    w_val = torch.add(w1_val, w2_val)

    # Compute gradients using torch.autograd.grad
    dw = torch.autograd.grad(w_val, x, grad_outputs=torch.ones_like(w_val), create_graph=True)[0]
    dw = torch.where(torch.isnan(dw), torch.zeros_like(dw), dw)

    return w_val, dw


def loss(eq, model_u, model_v, x_int, x_bd, beta):
    # square = 0
    # square = 2 ** eq.D
    u = model_u(x_int)
    v = model_v(x_int)

    f = eq.f(x_int)
    w, dw = fun_w(x_int)
    wv = w * v
    du = torch.autograd.grad(u, x_int,
                             grad_outputs=torch.ones_like(u),
                             create_graph=True,
                             retain_graph=True)[0]
    dv = torch.autograd.grad(v, x_int,
                             grad_outputs=torch.ones_like(v),
                             create_graph=True,
                             retain_graph=True)[0]
    du_dw = torch.sum(du * dw, 1).reshape(-1, 1)
    du_dv = torch.sum(du * dv, 1).reshape(-1, 1)
    du_dwv = v * du_dw + w * du_dv

    norm = torch.sum(wv ** 2)

    loss_l1 = torch.sum(du_dwv) - torch.sum(f * wv)
    loss_int = torch.pow(loss_l1, 2) / norm

    u_theta = model_u(x_bd)
    g = eq.g(x_bd)
    loss_bd = torch.mean(torch.abs(u_theta-g))
    loss_u = loss_int + beta * loss_bd
    loss_v = -torch.log(loss_int)
    return loss_u, loss_v


# Test function
def TEST(model, x_test, u_real):
    with torch.no_grad():
        u_pred  = model(x_test)
        Error   =  u_real - u_pred
        L2error = torch.sqrt(torch.sum(Error*Error) / torch.sum(u_real*u_real) )
        Maxerror = torch.max(torch.abs(Error))
    return L2error.cpu().detach().numpy(), Maxerror.cpu().detach().numpy()


class MLP(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_width=40):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x.to(torch.float32))


def plot_2D(model, device, title):
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    data = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
    data = torch.FloatTensor(data).to(device)
    Z = model(data).cpu().detach().numpy().reshape(100,100)
    Z[50:,50:] = np.NaN
    plt.figure()
    plt.pcolormesh(X, Y, Z, vmin=0.0, vmax=0.16, cmap= 'rainbow') # 'coolwarm') #
    plt.colorbar()
    plt.savefig(title)
    plt.close()


def train_pipeline():
    # define device
    DEVICE = torch.device(f"cuda:{CUDA_ORDER}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    # define equation
    Eq = PoissonEquation(DIMENSION, DEVICE)
    # define model
    # torch.set_default_dtype(torch.float64)
    n0 = 1
    model_u = MLP(in_channels=args.d).to(DEVICE)
    model_v = MLP(in_channels=args.d).to(DEVICE)

    optu = torch.optim.Adam(model_u.parameters(), lr=0.001)
    optv = torch.optim.Adam(model_v.parameters(), lr=0.001)

    x_int = Eq.interior(NUM_INT_SAMPLE)
    x_bd = Eq.boundary(NUM_BD_SAMPLE)

    test_dataset = sci.loadmat('data-Lshape.mat')['piocg1']
    test_data = test_dataset[:, :2]
    test_data = torch.FloatTensor(test_data).to(DEVICE)
    test_u = test_dataset[:,2:3]
    test_u = torch.FloatTensor(test_u).to(DEVICE)

    x_test = Eq.interior(NUM_TEST_SAMPLE)

    elapsed_time = 0
    training_history = []
    epoch_list = []
    min_l2 = 100
    for step in tqdm(range(NUM_ITERATION + 1)):
        if IS_DECAY and step and step % LEARN_FREQUENCY == 0:
            for p in optu.param_groups:
                if p['lr'] > LEARN_LOWWER_BOUND:
                    p['lr'] = p['lr'] * LEARN_DECAY_RATE
                    print(f"Learning Rate: {p['lr']}")
            for p in optv.param_groups:
                if p['lr'] > LEARN_LOWWER_BOUND:
                    p['lr'] = p['lr'] * LEARN_DECAY_RATE
                    print(f"Learning Rate: {p['lr']}")

        start_time = time.time()
        loss_u, loss_v = loss(Eq, model_u, model_v, x_int, x_bd, BETA)
        start_t = time.time()
        if step % 2 == 0:
            optu.zero_grad()
            loss_u.backward()
            optu.step()
        else:
            optv.zero_grad()
            loss_v.backward()
            optv.step()
        end_t = time.time() - start_t
        epoch_list.append(end_t)
        elapsed_time = elapsed_time + time.time() - start_time
        if step % TEST_FREQUENCY == 0:
            loss_u = loss_u.cpu().detach().numpy()
            loss_v = loss_v.cpu().detach().numpy()
            L2error, Maxerror = TEST(model_u, test_data, test_u)

            tqdm.write(f'Step: {step:>5} | '
                       f'Loss_u: {loss_u:>12.5f} | '
                       f'Loss_v: {loss_v:>10.5f} | '
                       f'L2 error: {L2error:>7.5e} | '
                       f'Max error: {Maxerror:>7.5e} |'
                       f'Time: {elapsed_time:>7.2f} |')
            training_history.append([step, loss_u, loss_v, L2error, Maxerror, elapsed_time])
            if L2error < min_l2:
                min_l2 = L2error
                torch.save({'u_model_state_dict': model_u.state_dict(),
                            'v_model_state_dict': model_v.state_dict(),
                            'min': min_l2
                            },
                           f'{DIMENSION}D-'+'check_point_me.pt'
                           )
                print('Save model min L2error!')
    training_history = np.array(training_history)
    print('l2r_min:', np.min(training_history[:, 3]))
    history = pd.DataFrame(training_history[:, 3])
    history.to_csv('WAN.csv')
    loss_history = pd.DataFrame(training_history[:, 1])
    loss_history.to_csv(f'WAN-{DIMENSION}D-loss_history.csv')
    dir_path = os.getcwd() + f'/PossionEQ/{DIMENSION}D/'

    epoch_list = np.array(epoch_list)

    np.savetxt('epoch_time-'+f'{DIMENSION}D'+'.csv', epoch_list, delimiter=",",
                header    ="epoch_time",
                comments  ='')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    if IS_SAVE_MODEL:
        torch.save(model_u.state_dict(), dir_path + f'{DIMENSION}D'+'WAN-U_net')
        print('Weak Adversarial Network Saved!')

    metric = []
    with torch.no_grad():
        model_me = torch.load(f'{DIMENSION}D-'+'check_point_me.pt')
        model_u.load_state_dict(model_me['u_model_state_dict'])
        start_time = time.time()
        _ = model_u(x_test)
        end_time = time.time() - start_time
        metric.append(end_time)
        print(f'infer time: {end_time}')
        L2error, Maxerror = TEST(model_u, test_data, test_u)
        metric.append(L2error)
        metric.append(Maxerror)
        metric_list = np.array(metric)

        np.savetxt('metric-' + f'{DIMENSION}D' + '.csv', metric_list, delimiter=",",
                   header="infer_time, L2error, Maxerror",
                   comments='')
        print(f'L2 error: {L2error}'
              f'Max error: {Maxerror}')
    plot_2D(model_u, DEVICE, 'WAN-Lshape.png')
    return model_u


if __name__ == "__main__":
    model = train_pipeline()
    print('over')
