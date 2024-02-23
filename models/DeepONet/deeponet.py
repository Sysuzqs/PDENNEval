# coding=utf-8
import torch
import torch.nn as nn
from utils import _get_act, _get_initializer


class MLP(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = _get_act(activation)
        initializer = _get_initializer(kernel_initializer)
        initializer_zero = _get_initializer("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=torch.float32
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x

class Modified_MLP(nn.Module):
    def __init__(self, layer_sizes, activation, kernel_initializer) -> None:
        super().__init__()
        self.activation = _get_act(activation)
        initializer = _get_initializer(kernel_initializer)
        initializer_zero = _get_initializer("zeros")
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=torch.float32
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)
        self.linear1=torch.nn.Linear(layer_sizes[0], layer_sizes[1], dtype=torch.float32)
        self.linear2=torch.nn.Linear(layer_sizes[0], layer_sizes[1], dtype=torch.float32)
        initializer(self.linear1.weight),initializer(self.linear2.weight)
        initializer_zero(self.linear1.bias),initializer_zero(self.linear2.bias)
    def forward(self, inputs):
        U = self.activation(self.linear1(inputs))
        V = self.activation(self.linear2(inputs))
        for linear in self.linears[:-1]:
            outputs=torch.sigmoid(linear(inputs))
            inputs= outputs*U + (1-outputs)* V
        outputs = self.linears[-1](inputs)
        return outputs


class DeepONet(nn.Module):
    """Deep operator network.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = _get_act(activation["branch"])
            self.activation_trunk = _get_act(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = _get_act(activation)
        if callable(layer_sizes_branch[0]):
            # User-defined network
            self.branch = layer_sizes_branch[0]
        else:
            # Fully connected network
            self.branch = MLP(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = MLP(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,bi->b", x_func, x_loc)
        x = torch.unsqueeze(x, 1)
        # Add bias
        x += self.b
        return x

class DeepONetCartesianProd(nn.Module):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        base_model = "MLP"    # or Modified_MLP 
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = _get_act(activation["branch"])
            self.activation_trunk = _get_act(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = _get_act(activation)
        base_model= MLP if base_model=="MLP" else Modified_MLP
        if callable(layer_sizes_branch[0]):
            # User-defined network
            self.branch = layer_sizes_branch[0]
        else:
            self.branch = base_model(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = base_model(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)
        # Add bias
        x += self.b
        return x

class DeepONetCartesianProd2D(DeepONetCartesianProd): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        in_channel_branch: int,
        query_dim: int ,
        out_channel: int,
        activation: str = "relu",
        kernel_initializer: str = "Glorot normal",
        base_model = "MLP"):
        layer_sizes_branch = [in_channel_branch*size**2]+[128]*4+[128*out_channel]
        layer_sizes_trunk= [query_dim]+[128]*4+[128*out_channel]
        super().__init__(layer_sizes_branch,layer_sizes_trunk,activation,kernel_initializer,base_model)
        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]  
        batchsize=x_func.shape[0]
        x_func = x_func.reshape([batchsize,-1]) 
        grid_shape = x_loc.shape[:-1]
        x_loc = x_loc.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=x_loc.shape[0]
        # Branch net to encode the input function
        x_func = self.branch(x_func.reshape([batchsize,-1]))
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x_func = x_func.reshape([batchsize,self.out_channel,-1])
        x_loc = x_loc.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x_func, x_loc)
        # Add bias
        x += self.b
        return x.reshape([-1,*grid_shape,self.out_channel])

class DeepONetCartesianProd1D(DeepONetCartesianProd):
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size :int,
        in_channel_branch: int,
        query_dim: int ,
        out_channel: int,
        activation: str = "relu",
        kernel_initializer: str = "Glorot normal",
        base_model="MLP"):
        layer_sizes_branch= [in_channel_branch*size]+[128]*4+[128*out_channel]
        layer_sizes_trunk= [query_dim]+[128]*3+[128*out_channel]
        super().__init__(layer_sizes_branch,layer_sizes_trunk,activation,kernel_initializer,base_model)
        self.out_channel = out_channel
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))
        self.query_dim=query_dim

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        grid_shape = x_loc.shape[:-1]
        x_loc = x_loc.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=x_loc.shape[0]
        batchsize=x_func.shape[0]
        x_func = x_func.reshape([batchsize,-1])
        # Branch net to encode the input function
        
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x_func = x_func.reshape([batchsize,self.out_channel,-1])
        x_loc = x_loc.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x_func, x_loc)
        # Add bias
        x += self.b
        return x.reshape([-1,*grid_shape,self.out_channel])