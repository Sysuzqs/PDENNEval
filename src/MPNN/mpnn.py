# coding=utf-8
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, InstanceNorm
from utils import PDE

class Swish(nn.Module):
    """Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)
    

class GNN_Layer(MessagePassing):
    """Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 spatial_dim: int,
                 n_variables: int):
        """Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            spatial_dim (int): number of dimension of spatial domain  
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean') # node_dim: The axis along which to propagate. (default: -2)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        assert (spatial_dim == 1 or spatial_dim == 2 or spatial_dim == 3)

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + spatial_dim + n_variables, hidden_features), 
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features), 
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features), 
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features), 
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update



class MPNN(nn.Module):
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layers: int = 6,
                 eq_variables: dict = {}):
        """Initialize MPNN
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE to solve
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MPNN, self).__init__()
        # arguments
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.time_window = time_window
        self.eq_variables = eq_variables

        # encoder
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window+self.pde.spatial_dim+1+len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
            )

        # processor
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            spatial_dim=self.pde.spatial_dim,
            n_variables=len(self.eq_variables)+1 # (time is treated as equation variable)
            ) for _ in range(self.hidden_layers)))
        
        # decoder
        if(self.time_window==10): # NEW ADD
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 18, stride=5),
                Swish(),
                nn.Conv1d(8, 1, 14, stride=1)
                )
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 15, stride=4),
                Swish(),
                nn.Conv1d(8, 1, 10, stride=1)
                )
        if (self.time_window==25):
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 16, stride=3),
                Swish(),
                nn.Conv1d(8, 1, 14, stride=1)
                )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 12, stride=2),
                Swish(),
                nn.Conv1d(8, 1, 10, stride=1)
                )
            
    def forward(self, data: Data, v: int=0) -> torch.Tensor:
        """Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window, v].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
            v (int): physics variable to solve
        Returns:
            torch.Tensor: predictive solutions with shape [batch*n_nodes, time_window]
        """
        u = data.x[..., v] # [bs*nx, tw]
        x_pos = data.x_pos # [bs*nx, spatial_dim]
        t_pos = data.t_pos # [bs*nx]
        edge_index = data.edge_index # [2, num_edges]
        batch = data.batch # [bs*nx]
        variables = data.variables # [bs*nx, num_variables]

        # normalize temporal and spatial coordinate
        t_pos = t_pos / self.pde.tmax
        for d in range(self.pde.spatial_dim):
            x_pos[:, d] = x_pos[:, d] / (self.pde.spatial_domain[d][1] - self.pde.spatial_domain[d][0])

        # encode
        node_input = torch.cat((u, x_pos, t_pos.unsqueeze(1), variables), dim=-1)
        f = self.embedding_mlp(node_input) # [bs*nx, hidden_dim]

        # process
        for i in range(self.hidden_layers):
            # f = self.gnn_layers[i](f, u, x_pos, variables, edge_index, batch)
            f = self.gnn_layers[i](f, u, x_pos, torch.cat((t_pos.unsqueeze(1), variables), dim=-1), edge_index, batch)

        # decode
        dt = (self.pde.tmax - self.pde.tmin) / self.pde.resolution_t
        dt = (torch.ones(1, self.time_window) * dt).to(f.device) # [1, tw]
        dt = torch.cumsum(dt, dim=1) # [1, tw]
        # [bs*nx, hidden_dim] -> [bs*nx, 1, hidden_dim]) -> conv1d -> [bs*nx, 1, tw] -> [bs*nx, tw] (diff)
        diff = self.output_mlp(f[:, None]).squeeze(1)
        # [bs*nx] -> [tw, bs*nx] -> [bs*nx, tw]
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff
        # print(out.shape)
        return out