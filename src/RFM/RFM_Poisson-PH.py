# %%
import argparse
import numpy as np
import os
import scipy as sci
from sklearn.metrics import max_error
import time
import torch
import torch.nn as nn
from torch import Tensor, sin


# %%
# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-d", "--dimension", type=int, default=3, help="The dimension of equation to be solved (default: 3).")
argparser.add_argument("--num-basis", type=int, default=1000, help="The number of basis/feature functions (default: 1000).")
argparser.add_argument("--scale", type=float, default=1.0, help="The scale of NN (default: 1.0).")
argparser.add_argument("--seed", type=int, default=2024, help="The random seed (default: 2024).")
argparser.add_argument("--save-dir", type=str, default="checkpoint/Poisson-PH", help="The directory path to save model (default: checkpoint/Poisson-PH).")
args = argparser.parse_args()
print(args)

# Define the dimension of equation
dimension = args.dimension
n_basis_func = args.num_basis


# %%
# Set random seed and deterministic behavior 2024/1018/1021/2010
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
    def __init__(self, hidden_size, dimension, scale):
        # hidden_size: The hidden layer size that is equal to the number of basis/feature functions
        super(Net, self).__init__()
        self.input_layers = torch.nn.Linear(dimension, hidden_size)
        self.input_layers.weight.data.uniform_(-scale, scale)
        self.input_layers.bias.data.uniform_(-scale, scale)
        print("self.input_layers.weight.shape:", self.input_layers.weight.shape)

    def forward(self, x):
        y = torch.tanh(self.input_layers(x))
        return y

net = Net(n_basis_func, dimension, args.scale)


# %%
# Define boundary of PDE
xl = -1.0 # left boundary: [xl, xr]^d
xr = 1.0 # right boundary: [xl, xr]^d

# Sample interior points
n_in = 3000 # The number of interior points
points_in = np.random.uniform(xl, xr, (n_in, dimension))
X = torch.from_numpy(points_in)
X.requires_grad_(True)

# Sample boundary points
n_bc = 400 # The number of boundary points
points_bc_raw = np.random.uniform(xl, xr, (2*n_bc*dimension, dimension))
for i in range(n_bc):
    # range for this is [i*2d, (i+1)*2d]
    for j in range(dimension):
        points_bc_raw[i*2*dimension+j, j] = xl
        points_bc_raw[i*2*dimension+dimension+j, j] = xr
X_bound = torch.from_numpy(points_bc_raw)


# %%
# Calculate first and second order derivatives
D_list = [[] for _ in range (dimension)]
DD_list = [[] for _ in range (dimension)]

def compute_grad_hessian(net, X, index):
    d = torch.autograd.grad(
        net(X)[:, index:index+1], 
        X, 
        grad_outputs=torch.ones_like(net(X)[:, index:index+1]), 
        create_graph=True)[0]  
    for i in range (dimension):    
        D_list[i].append(d[:, i:i+1].detach()) # D_list[i]: The first-order derivative of the i-th dimension
        
    # Calculate the second derivative
    def compute_second_derivatives(d, index):
        d2 = torch.autograd.grad(d[:, index:index+1],
                                 X, 
                                 grad_outputs=torch.ones_like(d[:, index:index+1]), 
                                 create_graph=True)[0]
        return d2[:,index:index+1].detach()
    for i in range (dimension):
        # DD_list[i]: The second-order derivative of the i-th dimension
        DD_list[i].append(compute_second_derivatives(d, i))
    
for i in range(n_basis_func):
    compute_grad_hessian(net, X, i)

# The first derivative
D = []
for i in range(len(D_list)):
    D.append(torch.stack(D_list[i]))
D = torch.stack(D).squeeze().permute(0, 2, 1)

# The second derivative
DD = []
for i in range(len(DD_list)):
    DD.append(torch.stack(DD_list[i]))
DD = torch.stack(DD).squeeze().permute(0, 2, 1)


# %% Ground Truth
# The expression of ground truth
def real_solution(p: torch.Tensor):
    sum = 0 
    for i in range(dimension):
        sum += p[:, i:i+1]
    return (1/dimension*sum)**2 + torch.sin(1/dimension*sum)

# The expression of boundary condition
def boundary(p: Tensor):
    return real_solution(p)

# The expression of initial condition
def initial(p: Tensor):
    return real_solution(p)


# %% Construct loss function
# The interior point conditions
f_domain = 0
for i in range(dimension):
    f_domain += DD[i].detach().cpu().numpy()
sum_X = 0
for i in range(dimension):
    sum_X += X[:, i:i+1]
f_domain_exact = np.zeros((X.shape[0], 1))

# The boundary condition
u_bound_pred = net(X_bound)
u_bound_pred = u_bound_pred.detach().cpu().numpy()
u_bound_exact = boundary(X_bound).detach().cpu().numpy()

# AC = f
A = np.vstack([f_domain + 1 / dimension * (sin(1/dimension*(sum_X)) - 2).detach().cpu().numpy(), u_bound_pred])
f = np.vstack([f_domain_exact, u_bound_exact])


# %%
# Perform least-squares approximation to obtain coefficient matrix C
C = sci.linalg.lstsq(A, f)[0] # (M, 1)


# %%
# Calculate Metrics
u_pred = (net(X).detach().cpu().numpy() @ C) # Calculate prediction, net(x) is equivalent to $\psi(x) in formula (2.8) of the paper
u_domain_exact = real_solution(X).detach().cpu().numpy() # Ground Truth of interior points

# Calculate L2 relative error
rel_error = np.linalg.norm(u_pred-u_domain_exact, 2) / np.linalg.norm(u_domain_exact, 2)
print("L2 Relative Error:", rel_error)

# Calculate max error
Max_error = max_error(u_pred, u_domain_exact)
print("Max Error:", Max_error)


# %%
# Record execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time}s")


# %%
# Save basis/feature functions and coefficient matrix C
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
torch.save(net.state_dict(), os.path.join(args.save_dir, f"network_{dimension}D_{args.num_basis}basis_scale{args.scale}_seed{args.seed}.pth"))
np.save(os.path.join(args.save_dir, f"C_{dimension}D_{args.num_basis}basis_scale{args.scale}_seed{args.seed}.npy"), C)