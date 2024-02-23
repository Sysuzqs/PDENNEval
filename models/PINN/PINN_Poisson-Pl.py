# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import *
import os
import argparse
from GenerateData import *


# Parser
parser = argparse.ArgumentParser(description='DFVM')
parser.add_argument('--dimension', type=int, default=100, metavar='N',
                    help='dimension of the problem (default: 100)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
seed = parser.parse_args().seed
# Omega 空间域
DIMENSION = parser.parse_args().dimension
a = [-1 for _ in range(DIMENSION)]
b = [ 1 for _ in range(DIMENSION)]
# Netword
DIM_INPUT  = DIMENSION   # 输入维数
NUM_UNIT   = 40         # 单层神经元个数
DIM_OUTPUT = 1           # 输出维数
NUM_LAYERS = 6           # 模型层数
# Optimizer
IS_DECAY           = 0
LEARN_RATE         = 1e-3    # 学习率
LEARN_FREQUENCY    = 50     # 学习率变化间隔
LEARN_LOWWER_BOUND = 1e-5
LEARN_DECAY_RATE   = 0.99
LOSS_FN            = nn.MSELoss()
# Training
CUDA_ORDER = "8"
NUM_TRAIN_SMAPLE   = 10000    # 训练集大小
NUM_TRAIN_TIMES    = 1       # 训练样本份数
NUM_ITERATION      = 100000  # 单份样本训练次数
# Re-sampling
IS_RESAMPLE = 0
SAMPLE_FREQUENCY   = 2000     # 重采样间隔
# Testing
NUM_TEST_SAMPLE    = 10000
TEST_FREQUENCY     = 1     # 输出间隔
# Loss weight
BETA = 1000                 # 边界损失函数权重
# Save model
IS_SAVE_MODEL = 1


class PossionEquation(object):
    def __init__(self, dimension, device):
        self.D      = dimension
        self.device = device

    def f(self, X):
        f = torch.ones(X.shape[0], 1).to(self.device)
        return f.detach()

    def g(self, X):
        g = torch.zeros(X.shape[0], 1).to(self.device)
        return g.detach()

    def u_exact(self, X):
        x = X[:,0]
        u = torch.where(x<0.5, x.pow(2), (x-1).pow(2))
        return u.reshape(-1,1).detach()

    # 区域内部的采样
    def interior(self, N=100):
        X = []
        l_bounds = [-1, -1]
        u_bounds = [ 0,  0]
        X.append( sampleCubeMC(self.D, l_bounds, u_bounds, N//3) )
        l_bounds = [-1,  0]
        u_bounds = [ 0,  1]
        X.append( sampleCubeMC(self.D, l_bounds, u_bounds, N//3) )
        l_bounds = [ 0, -1]
        u_bounds = [ 1,  0]
        X.append( sampleCubeMC(self.D, l_bounds, u_bounds, N//3) )
        X = torch.FloatTensor( np.concatenate(X, axis=0) )
        # X = torch.FloatTensor( sampleCubeQMC(self.D, l_bounds, u_bounds, N) )
        return X.requires_grad_(True).to(self.device)

    # 边界采样
    def boundary(self, n=100):
        # sample on boundary of [-1,1]^2\[0,1]^2
        x_boundary = []
        line1 = np.random.uniform(-1, 1, 2*n)
        line2 = np.random.uniform(-1, 0, n)
        line3 = np.random.uniform( 0, 1, n)
        x_boundary.append( np.stack([line1, -np.ones(2*n)], axis=1) )
        x_boundary.append( np.stack([-np.ones(2*n), line1], axis=1) )
        x_boundary.append( np.stack([line2, np.ones(n)], axis=1) )
        x_boundary.append( np.stack([np.ones(n), line2], axis=1) )
        x_boundary.append( np.stack([line3, np.zeros(n)], axis=1) )
        x_boundary.append( np.stack([np.zeros(n), line3], axis=1) )
        x_boundary = np.concatenate(x_boundary, axis=0)
        x_boundary = torch.FloatTensor(x_boundary).requires_grad_(True).to(self.device)
        return x_boundary


# 内部损失函数
def loss_interior(Eq, model, x):
    u = model(x)
    du = torch.autograd.grad(u, x, 
                            grad_outputs = torch.ones_like(u), 
                            create_graph = True, 
                            retain_graph = True)[0]
    # 计算各维二阶偏导
    laplace = torch.zeros_like(u)
    for i in range( Eq.D ):
        d2u = torch.autograd.grad(du[:,i], x, 
                                grad_outputs = torch.ones_like(du[:,i]), 
                                create_graph = True, 
                                retain_graph = True)[0][:,i]
        laplace += d2u.reshape(-1, 1)
    f_test  = Eq.f(x).detach().reshape(-1, 1)
    return LOSS_FN(-laplace, f_test)

# 边界损失函数
def loss_boundary(Eq, model, x_boundary):
    u_theta    = model(x_boundary).reshape(-1,1)
    u_bd       = Eq.g(x_boundary).reshape(-1,1)
    loss_bd    = LOSS_FN(u_theta, u_bd) 
    return loss_bd

# Test function
def TEST(Eq, model, x_test, u_real):
    with torch.no_grad():
        begin  = time.time()
        u_pred = model(x_test).reshape(1,-1)
        end    = time.time()
        Error  =  u_real - u_pred
        L2error  = torch.sqrt( torch.mean(Error*Error) )/ torch.sqrt( torch.mean(u_real*u_real) )
        MaxError = torch.max(torch.abs(Error))
    return L2error.cpu().detach().numpy(), MaxError.cpu().detach().numpy(), end-begin

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
    import matplotlib.pyplot as plt
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
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_pipeline():
    # define device
    DEVICE = torch.device(f"cuda:{CUDA_ORDER}" if torch.cuda.is_available() else "cpu")
    print(f"当前启用 {DEVICE}")
    # define equation
    Eq = PossionEquation(DIMENSION, DEVICE)
    # define model
    # torch.set_default_dtype(torch.float64)
    model = MLP(DIM_INPUT, DIM_OUTPUT, NUM_UNIT).to(DEVICE)
    optA   = torch.optim.Adam(model.parameters(), lr=LEARN_RATE) 

    x = Eq.interior(NUM_TRAIN_SMAPLE)
    x_boundary = Eq.boundary(100)

    data = np.loadtxt('./L-shape.csv', delimiter=',')
    test_input = torch.FloatTensor(data[:,0:2]).to(DEVICE)
    test_output = torch.FloatTensor(data[:,2]).to(DEVICE)

    # 网络迭代
    elapsed_time     = 0    # 计时
    training_history = []    # 记录数据

    for step in tqdm(range(NUM_ITERATION+1)):
        if IS_DECAY and step and step % LEARN_FREQUENCY == 0:
            for p in optA.param_groups:
                if p['lr'] > LEARN_LOWWER_BOUND:
                    p['lr'] = p['lr']*LEARN_DECAY_RATE
                    print(f"Learning Rate: {p['lr']}")

        start_time = time.time()
        loss_int = loss_interior(Eq, model, x)
        loss_bd  = loss_boundary(Eq, model, x_boundary)
        loss     = loss_int + BETA*loss_bd

        optA.zero_grad()
        loss.backward()
        optA.step()

        epoch_time = time.time() - start_time
        elapsed_time = elapsed_time + epoch_time
        if step % TEST_FREQUENCY == 0:
                loss_int     = loss_int.cpu().detach().numpy()
                loss_bd      = loss_bd.cpu().detach().numpy()
                loss         = loss.cpu().detach().numpy()
                L2error,ME,T = TEST(Eq, model, test_input, test_output)
                if step and step % 10000 == 0:
                    plot_2D(model, DEVICE, f'./Figures/PINN-{step}.png')
                    tqdm.write( f'\nStep: {step:>5}, '
                                f'Loss_int: {loss_int:>10.5f}, '
                                f'Loss_bd: {loss_bd:>10.5f}, '
                                f'Loss: {loss:>10.5f}, '                                     
                                f'L2 error: {L2error:.5f}, '                                     
                                f'Time: {elapsed_time:.2f}')
                training_history.append([step, L2error, ME, loss, elapsed_time, epoch_time, T])

    training_history = np.array(training_history)
    print(np.min(training_history[:,1]))
    print(np.min(training_history[:,2]))

    save_time = time.localtime()
    save_time = f'[{save_time.tm_mday:0>2d}{save_time.tm_hour:0>2d}{save_time.tm_min:0>2d}]'
    dir_path  = os.getcwd() + f'/PossionEQ_seed{seed}/'
    file_name = f'{DIMENSION}DIM-PINN-{BETA}weight-{NUM_ITERATION}itr-{LEARN_RATE}lr.csv'
    file_path = dir_path + file_name

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.savetxt(file_path, training_history,
                delimiter =",",
                header    ="step, L2error, MaxError, loss, elapsed_time, epoch_time, inference_time",
                comments  ='')
    print('Training History Saved!')

    if IS_SAVE_MODEL:
        torch.save(model.state_dict(), dir_path + f'{DIMENSION}DIM-PINN_net')
        print('PINN Network Saved!')


if __name__ == "__main__":
    setup_seed(seed)
    train_pipeline()
