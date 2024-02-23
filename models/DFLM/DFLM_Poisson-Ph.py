# %% 引入模块
import csv
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame
from tqdm.notebook import tqdm
import seaborn as sns
import argparse
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% 配置方程
class linearpossion(object):
    def __init__(self, dimension):
        self.dim = dimension
        self.low = -1
        self.high = 1
        self.sigma = np.sqrt(2)

    def bound_cond(self, x):
        return self.u_exact(x)

    def f(self, x):
        f = np.pi**2 * np.sum( np.sin( np.pi * x / 2 ), axis = 1).reshape(-1,1) / 4
        return f
    def u_exact(self, x):
        u = np.sum(np.sin(np.pi * x / 2), axis=1, keepdims=True)
        return u
    def unif_sample_domain(self, num_sample):
        return np.random.uniform(low=self.low, high=self.high, size=(num_sample, self.dim))

    def unif_sample_bound(self, num_sample):
        x = np.random.uniform(low=self.low, high=self.high, size=(num_sample, self.dim))
        for i in range(num_sample):
            idx_num1 = random.randint(0, self.dim - 1)
            if idx_num1 == 0:
                idx_num2 = random.randint(1, self.dim - 1)
            else:
                idx_num2 = random.randint(0, self.dim - 1)

            idx1 = random.sample(list(range(self.dim)), idx_num1)
            x[i, idx1] = 1

            idx2 = random.sample(list(range(self.dim)), idx_num2)
            x[i, idx2] = -1
        return x

    def unif_sample_bound2(self, n=50):
        x_boundary = []
        for i in range( self.dim ):
            x = np.random.uniform(-1, 1, [2*n, self.dim]) 
            x[:n,i] = 1
            x[n:,i] = -1
            x_boundary.append(x)
        x_boundary = np.concatenate(x_boundary, axis=0)
        # x_boundary = torch.FloatTensor(x_boundary).requires_grad_(True).to(self.device)
        return x_boundary
        # x_boundary = []
        # for i in range(self.dim):
        #     x = np.random.uniform(low=self.low, high=self.high, size=(num_sample // (2 * self.dim), self.dim))
        #     x[:, i] = self.high
        #     x_boundary.append(x)
        #     x = np.random.uniform(low=self.low, high=self.high, size=(num_sample // (2 * self.dim), self.dim))
        #     x[:, i] = self.low
        #     x_boundary.append(x)
        # x_boundary = np.concatenate(x_boundary, axis=0)
        # return x_boundary

    def is_in_domain(self, x):
        sup_flag = np.all(x < self.high, axis=1, keepdims=True)
        sub_flag = np.all(x > self.low, axis=1, keepdims=True)
        flag = sup_flag * sub_flag
        return flag

    def exit_estimate(self, x0, x1):
        sup_bound = np.ones_like(x0)
        sub_bound = np.ones_like(x0) * -1
        delta_x = x1 - x0
        sup_alpha = (sup_bound - x0) / delta_x
        sub_alpha = (sub_bound - x0) / delta_x
        alpha = np.concatenate((sup_alpha, sub_alpha), axis=1)
        alpha = min(alpha[alpha > 0])
        x_e = x0 + alpha * delta_x
        return x_e

    def transit(self, x0, delta_t):
        M = x0.shape[0]
        D = x0.shape[1]
        delta_W = np.sqrt(delta_t) * np.random.normal(size=(M, D))
        x1 = x0 + self.sigma * delta_W
        return x1

    def D(self, x0, delta_t):
        return np.ones((x0.shape[0], 1))

    def R(self, x0, x1, delta_t):
        return (self.f(x0) + self.f(x1)) * delta_t * 0.5
        # return

    def spread(self, x0, J, delta_t):
        M = x0.shape[0]
        D = x0.shape[1]
        delta_W = np.sqrt(delta_t) * np.random.normal(size=(M * J, D))
        # delta_W = np.reshape(delta_W, (M, m, D))
        x0 = np.expand_dims(x0, axis=1)
        x0 = np.broadcast_to(x0, (M, J, D))
        x0 = np.reshape(x0, (M * J, D))
        x1 = x0 + self.sigma * delta_W
        return x1

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Parser
parser = argparse.ArgumentParser(description='DFVM')
parser.add_argument('--dimension', type=int, default=100, metavar='N',
                    help='dimension of the problem (default: 100)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
NUM_TRAIN_SAMPLE = 2000
NUM_TEST_SAMPLE  = 10000
NUM_BOUND_SAMPLE = 50
DIMENSION        = parser.parse_args().dimension     # Dimension
seed             = parser.parse_args().seed          # Random Seed
LEARNING_RATE    = 0.0001
NUM_ITERATION    = 20000
Delta_t          = 0.0005
m = 10

setup_seed(seed)

# %% 额外函数
def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    # if type(m) == nn.Parameter:
    #     torch.nn.init.xavier_uniform_(m.params)
# %% 配置训练

Eq = linearpossion(DIMENSION)
U = MLP(DIMENSION).to(device)
# U.apply(weights_init)
optU = optim.Adam(U.parameters(), lr=LEARNING_RATE)


# %% 开始训练
start_time = time.time()
elapsed_time = 0
training_history = []

Loss=0
X0 = Eq.unif_sample_domain(NUM_TRAIN_SAMPLE)
Xb = Eq.unif_sample_bound2(NUM_BOUND_SAMPLE)
Lossb = 0
for step in range(NUM_ITERATION + 1):
    torch.cuda.empty_cache()
    # if step and step % 50 == 0:
    #     for p in optU.param_groups:
    #         if p['lr'] > 1e-5:
    #             p['lr'] = p['lr']*0.99
    #             print(f"Learning Rate: {p['lr']}")

    if step % 1 == 0:
        end = time.time() - start_time
        epoch_time = end - elapsed_time
        elapsed_time = end
        X0t = Eq.unif_sample_domain(NUM_TEST_SAMPLE)
        begin = time.time()
        Y0t = U(torch.FloatTensor(X0t).to(device)).reshape(-1, 1)
        inference_time = time.time() - begin
        Y0t = Y0t.cpu().detach().numpy()
        U0t = Eq.u_exact(X0t).reshape(-1, 1)
        L2 = np.sqrt(np.mean((Y0t - U0t) ** 2) / np.mean(U0t ** 2))
        L1 = np.max(np.abs(U0t-Y0t))

        if step % 1000 == 0:
            print(f'\nStep: {step:>5}, '\
                f'Loss: {Loss:>10.5f}, '\
                f'L2: {L2:.6f},\n'\
                f'L1: {L1:.6f},\n'\
                f'Time: {elapsed_time:.2f}')
        training_history.append([step, Loss, L2, L1, elapsed_time, epoch_time, inference_time])
##############################################################################################################################################################

    Loss = 0

    Y0 = U(torch.FloatTensor(X0).to(device))
    Xm = Eq.spread(X0,m, Delta_t)
    X0m = np.expand_dims(X0, axis=1)
    X0m = np.broadcast_to(X0m, (NUM_TRAIN_SAMPLE, m, DIMENSION))
    X0m = np.reshape(X0m, (NUM_TRAIN_SAMPLE * m, DIMENSION))
    Y0m = U(torch.FloatTensor(Xm).to(device))

    flag = Eq.is_in_domain(Xm)
    if np.any(flag == False):
        X0m_out = X0m[flag.squeeze(-1) == False, :]
        Xm_out = Xm[flag.squeeze(-1) == False, :]
        Xm_new = Eq.exit_estimate(X0m_out, Xm_out)
        Y0m[flag.squeeze(-1) == False, :] = torch.FloatTensor(Eq.bound_cond(Xm_new)).to(device)
        Xm[flag.squeeze(-1) == False, :] = Xm_new

    D_t = Eq.D(X0m, Delta_t)
    R_t = Eq.R(X0m, Xm, Delta_t)
    R_t = np.reshape(R_t, (D_t.shape[0], 1))
    Target = - torch.FloatTensor(R_t).to(device)*torch.FloatTensor(D_t).to(device) + torch.FloatTensor(D_t).to(device) *Y0m
    Target = torch.reshape(Target, (NUM_TRAIN_SAMPLE, m, 1))
    Y0_pred = 1 / m * torch.sum(Target, dim=1)
    Loss = Loss + 1 / NUM_TRAIN_SAMPLE * torch.sum(torch.square(Y0_pred - Y0))
    
    Yb = U(torch.FloatTensor(Xb).to(device))
    Gb = torch.FloatTensor(Eq.bound_cond(Xb)).to(device)
    Loss = Loss + 1000/NUM_BOUND_SAMPLE * torch.sum(torch.square(Yb - Gb))

    optU.zero_grad()
    Loss.backward()
    optU.step()

training_history = np.array(torch.FloatTensor(training_history).cpu().detach())
print('Min L2:', np.min(training_history[:,2]))
print('Mean L2:', np.mean(training_history[:,2]))

# save files
record_time = time.localtime()
dir_path  = os.getcwd() + f'/PossionEQ_seed{seed}/'
np.savetxt(dir_path+'{:d}DIM-DFLM-global-[{:0>2d}{:0>2d}{:0>2d}].csv'.format(
    # Eq.__class__.__name__,
    DIMENSION,
    record_time.tm_mday,
    record_time.tm_hour,
    record_time.tm_min,),
            training_history,
            delimiter=",",
            header="step, loss, L2, Time, epoch_time, inference_time",
            comments='')
print('Training History Saved!')

torch.save(U.state_dict(), dir_path+f'{DIMENSION}DIM-DFLM-global')
print('Model Saved!')
