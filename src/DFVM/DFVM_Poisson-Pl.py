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
# Omega space domain
DIMENSION = parser.parse_args().dimension
a = [-1 for _ in range(DIMENSION)]
b = [1 for _ in range(DIMENSION)]
# Finite Volume
EPSILON   = 1e-4         # Neighborhood size
BDSIZE    = 1
# Network
DIM_INPUT  = DIMENSION   # Input dimension
NUM_UNIT   = 40          # Number of neurons per layer
DIM_OUTPUT = 1           # Output dimension
NUM_LAYERS = 6           # Number of model layers
# Optimizer
IS_DECAY   = 0
LEARN_RATE         = 1e-3    # Learning rate
LEARN_FREQUENCY    = 50      # Learning rate change interval
LEARN_LOWWER_BOUND = 1e-5
LEARN_DECAY_RATE   = 0.99
LOSS_FN            = nn.MSELoss()
# Training
CUDA_ORDER = "7"
NUM_TRAIN_SMAPLE   = 10000    # Training set size
NUM_TRAIN_TIMES    = 1        # Number of training samples
NUM_ITERATION      = 100000   # Number of training iterations per sample
# Re-sampling
IS_RESAMPLE = 0
SAMPLE_FREQUENCY   = 2000     # Resampling interval
# Testing
NUM_TEST_SAMPLE    = 10000
TEST_FREQUENCY     = 1        # Output interval
# Loss weight
BETA = 1000                   # Boundary loss function weight
# Save model
IS_SAVE_MODEL = 1

class PossionQuation(object):
    def __init__(self, dimension, epsilon, bd_size, device):
        self.D      = dimension
        self.E      = epsilon
        self.B      = bd_size
        self.bdsize = 2**bd_size - 1
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

    # Sampling within the region
    def interior(self, N=100):
        eps = self.E # np.spacing(1)
        X = []
        l_bounds = [-1, -1]
        u_bounds = [0, 0]
        X.append(sampleCubeMC(self.D, l_bounds, u_bounds, N//3))
        l_bounds = [-1, 0]
        u_bounds = [0, 1]
        X.append(sampleCubeMC(self.D, l_bounds, u_bounds, N//3))
        l_bounds = [0, -1]
        u_bounds = [1, 0]
        X.append(sampleCubeMC(self.D, l_bounds, u_bounds, N//3))
        X = torch.FloatTensor(np.concatenate(X, axis=0))
        # X = torch.FloatTensor( sampleCubeQMC(self.D, l_bounds, u_bounds, N) )
        return X.requires_grad_(True).to(self.device)

    # Boundary sampling
    def boundary(self, n=100):
        # sample on boundary of [-1,1]^2\[0,1]^2
        x_boundary = []
        line1 = np.random.uniform(-1, 1, 2*n)
        line2 = np.random.uniform(-1, 0, n)
        line3 = np.random.uniform(0, 1, n)
        x_boundary.append(np.stack([line1, -np.ones(2*n)], axis=1))
        x_boundary.append(np.stack([-np.ones(2*n), line1], axis=1))
        x_boundary.append(np.stack([line2, np.ones(n)], axis=1))
        x_boundary.append(np.stack([np.ones(n), line2], axis=1))
        x_boundary.append(np.stack([line3, np.zeros(n)], axis=1))
        x_boundary.append(np.stack([np.zeros(n), line3], axis=1))
        x_boundary = np.concatenate(x_boundary, axis=0)
        x_boundary = torch.FloatTensor(x_boundary).requires_grad_(True).to(self.device)
        return x_boundary

    # Randomly take bdsize points in the neighborhood of point x
    def neighborhood(self, x, size):
        l_bounds = [t-self.E for t in x.cpu().detach().numpy()]
        u_bounds = [t+self.E for t in x.cpu().detach().numpy()]
        sample   = sampleCubeQMC(self.D, l_bounds, u_bounds, size)
        sample   = torch.FloatTensor(sample).to(self.device)
        return sample

    def neighborhoodBD(self, X):
        lb = [-1 for _ in range(self.D-1)]
        ub = [1 for _ in range(self.D-1)]
        x_QMC   = sampleCubeQMC(self.D-1, lb, ub, self.B)
        x_nbound = []
        for i in range(self.D):
            x_nbound.append(np.insert(x_QMC, i, [1], axis=1))
            x_nbound.append(np.insert(x_QMC, i, [-1], axis=1))
        x_nbound = np.concatenate(x_nbound, axis=0).reshape(1, -1, self.D)
        x_nbound = torch.FloatTensor(x_nbound).to(self.device)
        X = torch.unsqueeze(X, dim=1)
        X = X.expand(-1, x_nbound.shape[1], x_nbound.shape[2])
        X_bound = X + self.E*x_nbound
        X_bound = X_bound.reshape(-1, self.D)
        return X_bound.detach().requires_grad_(True)

    def outerNormalVec(self):
        bd_dir = torch.zeros(2*self.D*self.bdsize, self.D)
        for i in range(self.D):
            bd_dir[2*i*self.bdsize : (2*i+1)*self.bdsize, i] = 1
            bd_dir[(2*i+1)*self.bdsize : 2*(i+1)*self.bdsize, i] = -1
        bd_dir = bd_dir.reshape(1,-1)
        return bd_dir.detach().requires_grad_(True).to(self.device)


class DFVMsolver(object):
    def __init__(self, Equation, model, device):
        self.Eq     = Equation
        self.model  = model
        self.device = device

    # Compute divergence u = u_theta
    def Nu(self, X):
        u = self.model(X)
        Du = torch.autograd.grad(outputs     = [u], 
                                inputs       = [X], 
                                grad_outputs = torch.ones_like(u),
                                allow_unused = True,
                                retain_graph = True,
                                create_graph = True)[0]
        return Du

    # Compute boundary integral at each sampling point
    def integrate_BD(self, X, x_bd, bd_dir):
        n = len(X)
        integrate_bd = torch.zeros(n, 1).to(self.device)
        # Compute gradient
        Du = self.Nu(x_bd).reshape(n, -1)
        # Multiply gradient with normal vector matrix to get directional derivative
        integrate_bd = torch.sum(Du*bd_dir, 1)/(2*self.Eq.E*self.Eq.bdsize)
        return integrate_bd

    # Compute volume integral
    def integrate_F(self, X):
        n = len(X)
        integrate_f = torch.zeros([n, 1]).to(self.device)
        for i in range(n):
            x_neighbor     = self.Eq.neighborhood(X[i], 1)
            res            = self.Eq.f(x_neighbor)
            integrate_f[i] = torch.mean(res)
        return integrate_f.detach().requires_grad_(True)

    # Boundary loss function
    def loss_boundary(self, x_boundary):
        u_theta    = self.model(x_boundary).reshape(-1,1)
        u_bound    = self.Eq.g(x_boundary).reshape(-1,1)
        loss_bd    = LOSS_FN(u_theta, u_bound) 
        return loss_bd

    # Test function
    def TEST(self, x_test, u_real):
        with torch.no_grad():
            begin  = time.time()
            u_pred = self.model(x_test).reshape(1,-1)
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
    plt.pcolormesh(X, Y, Z, vmin=0.0, vmax=0.16, cmap='rainbow') # 'coolwarm') # 
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
    print(f"Current device {DEVICE}")
    # define equation
    Eq = PossionQuation(DIMENSION, EPSILON, BDSIZE, DEVICE)
    # define model
    # torch.set_default_dtype(torch.float64)
    model = MLP(DIM_INPUT, DIM_OUTPUT, NUM_UNIT).to(DEVICE)
    optA   = torch.optim.Adam(model.parameters(), lr=LEARN_RATE) 
    solver = DFVMsolver(Eq, model, DEVICE)

    x      = Eq.interior(NUM_TRAIN_SMAPLE)
    x_bd   = Eq.neighborhoodBD(x)
    print(x_bd.shape)
    int_f  = solver.integrate_F(x)
    bd_dir = Eq.outerNormalVec()
    x_boundary = Eq.boundary(100)
    
    data = np.loadtxt('./L-shape.csv', delimiter=',')
    test_input = torch.FloatTensor(data[:,0:2]).to(DEVICE)
    test_output = torch.FloatTensor(data[:,2]).to(DEVICE)

    # Network iteration
    elapsed_time     = 0    # Timing
    training_history = []    # Record data

    for step in tqdm(range(NUM_ITERATION+1)):
        if IS_DECAY and step and step % LEARN_FREQUENCY == 0:
            for p in optA.param_groups:
                if p['lr'] > LEARN_LOWWER_BOUND:
                    p['lr'] = p['lr']*LEARN_DECAY_RATE
                    print(f"Learning Rate: {p['lr']}")

        start_time = time.time()
        int_bd   = solver.integrate_BD(x, x_bd, bd_dir).reshape(-1,1)
        loss_int = LOSS_FN(-int_bd, int_f)
        loss_bd  = solver.loss_boundary(x_boundary)
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
            L2error,ME,T = solver.TEST(test_input, test_output)
            if step and step%10000 == 0:
                plot_2D(model, DEVICE, f'./Figures/DFVM-{step}.png')
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
    file_name = f'{DIMENSION}DIM-DFVM-{BETA}weight-{NUM_ITERATION}itr-{EPSILON}R-{BDSIZE}bd-{LEARN_RATE}lr.csv'
    file_path = dir_path + file_name

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.savetxt(file_path, training_history,
                delimiter =",",
                header    ="step, L2error, MaxError, loss, elapsed_time, epoch_time, inference_time",
                comments  ='')
    print('Training History Saved!')

    if IS_SAVE_MODEL:
        torch.save(model.state_dict(), dir_path + f'{DIMENSION}DIM-DFVM_net')
        print('DFVM Network Saved!')


if __name__ == "__main__":
    setup_seed(seed)
    train_pipeline()
