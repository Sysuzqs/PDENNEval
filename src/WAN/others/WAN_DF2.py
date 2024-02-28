
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import argparse
import h5py

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
        default=20000,
        help="The iteration number"
    )
    parser.add_argument(
        "--b",
        type=int,
        default=1,
        help="The beta number"
    )
    return parser.parse_args(arg_list)


args = get_arguments()
torch.manual_seed(args.s)
np.random.seed(args.s)

DIMENSION = args.d
a = [0, 1]
b = [0, 1]
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


class DarcyFlowEquation(object):
    def __init__(self, dimension, device):
        self.D = dimension
        self.device = device

    def f(self, X):
        f = torch.ones([X.shape[0], 1])
        return f.reshape(-1, 1).detach().to(self.device)

    def g(self, X):
        u = torch.zeros([X.shape[0], 1])
        return u.reshape(-1, 1).detach().to(self.device)

    def interior(self, dataset, N=2000):
        with h5py.File(dataset, 'r') as f:
            x = f['x-coordinate'][:]
            y = f['y-coordinate'][:]
            x_mesh, y_mesh = np.meshgrid(x, y)
            xy = np.concatenate([x_mesh.flatten()[:, None], y_mesh.flatten()[:, None]], axis=1)
            idx = np.random.choice(xy.shape[0] - 1, N, replace=False)
            idx = np.sort(idx)
            xy_train = torch.Tensor(xy[idx, :]).requires_grad_(True).to(self.device)
            coef_a = f['nu'][0, :].flatten()[:, None]
            A = torch.Tensor(coef_a[idx, :]).to(self.device)
        return xy_train, A

    def boundary(self, n=100):
        data = []
        for i in range(self.D):
            x = np.random.uniform(a[i], b[i], [2 * n, self.D])
            x[:n, i] = b[i]
            x[n:, i] = a[i]
            data.append(x)
        X = np.concatenate(data, axis=0)
        return torch.FloatTensor(X).requires_grad_(True).to(self.device)

    def generate_test(self, dataset, N=10000):
        with h5py.File(dataset, 'r') as f:
            x = f['x-coordinate'][:]
            y = f['y-coordinate'][:]
            x_mesh, y_mesh = np.meshgrid(x, y)
            xy = np.concatenate([x_mesh.flatten()[:, None], y_mesh.flatten()[:, None]], axis=1)
            idx = np.random.choice(xy.shape[0] - 1, N, replace=False)
            idx = np.sort(idx)
            xy_test = torch.Tensor(xy[idx, :]).to(self.device)
            u = f['tensor'][0, 0, :].flatten()[:, None]
            u = torch.Tensor(u[idx, :]).to(self.device)
        return xy_test, u


def fun_w(x, eq):
    I1 = 0.210987
    x_list = torch.split(x, 1, 1)
    up = 1.0
    low = 0.0
    # **************************************************
    x_scale_list = []
    h_len = (up - low) / 2.0
    for i in range(eq.D):
        x_scale = (x_list[i] - low - h_len) / h_len
        x_scale_list.append(x_scale)
    # ************************************************
    z_x_list = []
    for i in range(eq.D):
        supp_x = torch.greater(1 - torch.abs(x_scale_list[i]), 0)
        z_x = torch.where(supp_x, torch.exp(1 / (torch.pow(x_scale_list[i], 2) - 1)) / I1,
                        torch.zeros_like(x_scale_list[i]))
        z_x_list.append(z_x)
    # ***************************************************
    w_val = 1
    for i in range(eq.D):
        w_val = w_val*z_x_list[i]
    dw = torch.autograd.grad(w_val, x,
                                grad_outputs=torch.ones_like(w_val),
                                create_graph=True,
                                retain_graph=True)[0]
    dw = torch.where(torch.isnan(dw), torch.zeros_like(dw), dw)
    return (w_val, dw)


def loss(eq, model_u, model_v, x_int, A, x_bd, beta):

    u = model_u(x_int)
    v = model_v(x_int)

    f = eq.f(x_int)
    w, dw = fun_w(x_int, eq)
    wv = w * v
    du = torch.autograd.grad(u, x_int,
                             grad_outputs=torch.ones_like(u),
                             create_graph=True,
                             retain_graph=True)[0]
    dv = torch.autograd.grad(v, x_int,
                             grad_outputs=torch.ones_like(v),
                             create_graph=True,
                             retain_graph=True)[0]
    # du_dw = torch.sum(du * dw, 1).reshape(-1, 1)
    # du_dv = torch.sum(du * dv, 1).reshape(-1, 1)
    lap_uwv = torch.sum((A * du) * (dw * v + w * dv), 1)

    norm = torch.mean(wv ** 2)

    loss_l1 = torch.mean(lap_uwv) - torch.mean(f * wv)
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


def train_pipeline():
    # define device
    DEVICE = torch.device(f"cuda:{CUDA_ORDER}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    # define equation
    equation = 'DF2'
    dataset = '/home/data2/fluidbench/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5'
    Eq = DarcyFlowEquation(DIMENSION, DEVICE)
    # define model
    model_u = MLP(in_channels=args.d).to(DEVICE)
    model_v = MLP(in_channels=args.d).to(DEVICE)

    optu = torch.optim.Adam(model_u.parameters(), lr=0.0001)
    optv = torch.optim.Adam(model_v.parameters(), lr=0.014)

    x_int, A = Eq.interior(dataset, NUM_INT_SAMPLE)
    x_bd = Eq.boundary(NUM_BD_SAMPLE)

    x_test, u_real = Eq.generate_test(dataset, NUM_TEST_SAMPLE)

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
        loss_u, loss_v = loss(Eq, model_u, model_v, x_int, A, x_bd, BETA)
        start_t = time.time()
        for _ in range(2):
            loss_u, loss_v = loss(Eq, model_u, model_v, x_int, A, x_bd, BETA)
            optv.zero_grad()
            loss_v.backward()
            optv.step()
        for _ in range(1):
            loss_u, loss_v = loss(Eq, model_u, model_v, x_int, A, x_bd, BETA)
            optu.zero_grad()
            loss_u.backward()
            optu.step()
        end_t = time.time() - start_t
        epoch_list.append(end_t)
        elapsed_time = elapsed_time + time.time() - start_time
        if step % TEST_FREQUENCY == 0:
            loss_u = loss_u.cpu().detach().numpy()
            loss_v = loss_v.cpu().detach().numpy()
            L2error, Maxerror = TEST(model_u, x_test, u_real)
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
                           equation+'-check_point_me.pt'
                           )
                print('Save model min L2error!')
    training_history = np.array(training_history)
    print('l2r_min:', np.min(training_history[:, 3]))
    history = pd.DataFrame(training_history[:, 3])
    history.to_csv(f'WAN-{equation}.csv')
    loss_history = pd.DataFrame(training_history[:, 1])
    loss_history.to_csv(f'WAN-{equation}-loss_history.csv')
    dir_path = os.getcwd() + f'/{equation}/'

    epoch_list = np.array(epoch_list)

    np.savetxt('epoch_time-'+equation+'.csv', epoch_list, delimiter=",",
                header    ="epoch_time",
                comments  ='')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if IS_SAVE_MODEL:
        torch.save(model_u.state_dict(), dir_path + 'WAN-U_net')
        print('Weak Adversarial Network Saved!')

    metric = []
    with torch.no_grad():
        model_me = torch.load(equation+'-check_point_me.pt')
        model_u.load_state_dict(model_me['u_model_state_dict'])
        start_time = time.time()
        _ = model_u(x_test)
        end_time = time.time() - start_time
        metric.append(end_time)
        print(f'infer time: {end_time}')
        L2error, Maxerror = TEST(model_u, x_test, u_real)
        metric.append(L2error)
        metric.append(Maxerror)
        metric_list = np.array(metric)

        np.savetxt('metric-' + equation + '.csv', metric_list, delimiter=",",
                   header="infer_time, L2error, Maxerror",
                   comments='')
        print(f'L2 error: {L2error}'
              f'Max error: {Maxerror}')
    return model_u


if __name__ == "__main__":
    model = train_pipeline()
    print('over')
