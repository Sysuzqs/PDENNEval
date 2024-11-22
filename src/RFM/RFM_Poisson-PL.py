# %%
import argparse
import numpy as np
import os
import scipy as sci
import torch
import torch.nn as nn
from sklearn.metrics import max_error
import pandas as pd
import time


# %%
# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--num-basis", type=int, default=800, help="The number of basis/feature functions (default: 800).")
argparser.add_argument("--scale", type=float, default=0.1, help="The scale of NN (default: 0.1).")
argparser.add_argument("--seed", type=int, default=2024, help="The random seed (default: 2024).")
argparser.add_argument("--save-dir", type=str, default="checkpoint/Poisson-PL", help="The directory path to save model (default: checkpoint/Poisson-PL).")
args = argparser.parse_args()
print(args)

n_basis_func = args.num_basis  # define the number of feature functions


# %%
# Set random seed and deterministic behavior
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)
start_time = time.time()


# %%
# Define the network architecture
class Net(nn.Module):
    def __init__(self, hidden_size, scale):
        # hidden_size: The hidden layer size that is equal to the number of basis/feature functions
        super(Net, self).__init__()
        self.input_layers = torch.nn.Linear(2, hidden_size)
        self.input_layers.weight.data.uniform_(-scale, scale)
        self.input_layers.bias.data.uniform_(-scale, scale)
        print("The shape of input layer weights:", self.input_layers.weight.shape)

    def forward(self, x):
        y = torch.tanh(8*torch.pi*self.input_layers(x))
        return y

net = Net(n_basis_func, args.scale)


# %%
# Read data from csv file
data = pd.read_csv('Lshape.csv')
data_cleaned = data[data['U'] != 0]

# Interior points
x_test = torch.tensor(data_cleaned['X'].values)
y_test = torch.tensor(data_cleaned['Y'].values)
X = torch.stack([x_test, y_test], dim=1)
X.requires_grad_(True)

# Ground truth
u_domain_exact = torch.tensor(data_cleaned['U'].values).unsqueeze(1)

# Sample initial points
size_x_initial = 200
x_initial = np.concatenate([np.linspace(-1, -1, size_x_initial).reshape(-1,1), 
                            np.linspace(-1, 1, size_x_initial).reshape(-1,1)], axis=1)
x_initial = torch.from_numpy(x_initial)

# Sample boundary points
size_x_bound = 200
x_bound1 = np.concatenate([np.linspace(-1, 1, size_x_bound).reshape(-1,1), 
                           np.linspace(-1, -1, size_x_bound).reshape(-1,1)], axis=1)
x_bound1 = torch.from_numpy(x_bound1)
x_bound2 = np.concatenate([np.linspace(-1, 0, size_x_bound).reshape(-1,1), 
                           np.linspace(1, 1, size_x_bound).reshape(-1,1)], axis=1)
x_bound2 = torch.from_numpy(x_bound2)
x_bound3 = np.concatenate([np.linspace(0, 1, size_x_bound).reshape(-1,1), 
                           np.linspace(0, 0, size_x_bound).reshape(-1,1)], axis=1)
x_bound3 = torch.from_numpy(x_bound3)
x_bound4 = np.concatenate([np.linspace(0, 0, size_x_bound).reshape(-1,1), 
                           np.linspace(0, 1, size_x_bound).reshape(-1,1)], axis=1)
x_bound4 = torch.from_numpy(x_bound4)
x_bound5 = np.concatenate([np.linspace(1, 1, size_x_bound).reshape(-1,1), 
                           np.linspace(-1, 0, size_x_bound).reshape(-1,1)], axis=1)
x_bound5 = torch.from_numpy(x_bound5)
X_bound = torch.cat((x_initial, x_bound1, x_bound2, x_bound3, x_bound4, x_bound5), dim=0)


# %%
# Calculate first and second order derivatives
dx = []
dy = []
dxx = []
dyy = []
for i in range(n_basis_func):
    d = torch.autograd.grad(
        net(X)[:,i : i + 1], 
        X, 
        grad_outputs=torch.ones_like(net(X)[:, i : i + 1]), 
        create_graph=True)[0]   
    dx.append(d[:,0:1].detach())      
    dy.append(d[:,1:2].detach())

    # The second derivative with respect to x
    dx2 = torch.autograd.grad(d[:,0:1], 
                               X, 
                               grad_outputs=torch.ones_like(d[:,0:1]), 
                               create_graph=True)[0]
    dxx.append(dx2[:,0:1].detach())
    
    # The second derivative with respect to y
    dy2 = torch.autograd.grad(d[:,1:2], 
                               X, 
                               grad_outputs=torch.ones_like(d[:,1:2]), 
                               create_graph=True)[0]
    dyy.append(dy2[:,1:2].detach())

dx = torch.cat(dx,dim=1)
dy = torch.cat(dy,dim=1) # [n，m]
dxx = torch.cat(dxx,dim=1)
dyy = torch.cat(dyy,dim=1) # [n，m]


# %% Construct loss function
# The interior point conditions
u_domain_dxx_pred = dxx.detach().cpu().numpy()
u_domain_dyy_pred = dyy.detach().cpu().numpy()
f_domain_exact = np.ones((X.shape[0], 1))

# The boundary condition
u_bound_pred = net(X_bound.to(X.device))
u_bound_pred = u_bound_pred.detach().cpu().numpy()
u_bound_exact = np.zeros((X_bound.shape[0], 1))

# AC = f
A = np.vstack([-u_domain_dxx_pred-u_domain_dyy_pred, u_bound_pred])
f = np.vstack([f_domain_exact, u_bound_exact])


# %%
# Perform least-squares approximation to obtain coefficient matrix C
C = sci.linalg.lstsq(A, f)[0] # (M，1)


# %%
# Calculate Metrics
u_pred = (net(X).detach().cpu().numpy() @ C) # Calculate predicted values
u_domain_exact = u_domain_exact.detach().cpu().numpy() # Calculate the ground truth value of collection points

# Calculate relative error
rel_error = np.linalg.norm(u_pred-u_domain_exact, 2) / np.linalg.norm(u_domain_exact, 2)
print("Relative Error:", rel_error)

# Calculate max error
Max_error = max_error(u_pred, u_domain_exact)
print("Max Error:", Max_error)


# %%
# Record execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time}s")


# %%
# Save basis/feature functions and coefficient matrix C
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
torch.save(net.state_dict(), os.path.join(args.save_dir, f"network_{args.num_basis}basis_scale{args.scale}_seed{args.seed}.pth"))
np.save(os.path.join(args.save_dir, f"C_{args.num_basis}basis_scale{args.scale}_seed{args.seed}.npy"), C)