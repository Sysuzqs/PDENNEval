# coding=utf-8
# Codes for section: Results on Darcy Flow Equation

import torch
import torch.nn as nn
import torch.nn.functional as F
from integral_operators import *

###############
#  UNO^dagger achitechtures
###############
class UNO1d(nn.Module):
    def __init__(self, num_channels, width, pad = 9, factor = 1, initial_step = 10):
        super(UNO1d, self).__init__()

        self.in_width = num_channels * initial_step + 1 # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) 

        self.conv0 = OperatorBlock_1D(self.width, 2*factor*self.width,24, 11)

        self.conv1 = OperatorBlock_1D(2*factor*self.width, 4*factor*self.width, 12, 5, Normalize = True)

        self.conv2 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 12, 5)

        self.conv4 = OperatorBlock_1D(4*factor*self.width, 2*factor*self.width, 24, 5, Normalize = True)

        self.conv5 = OperatorBlock_1D(4*factor*self.width, self.width, 57, 11) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 1*self.width)
        self.fc2 = nn.Linear(1*self.width, num_channels)

    def forward(self, x, grid):
        x = torch.cat((x, grid), dim=-1)

        x_fc_1 = self.fc_n1(x)
        x_fc_1 = F.gelu(x_fc_1)

        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 2, 1)
        # scale = math.ceil(x_fc0.shape[-1]/43)
        x_fc0 = F.pad(x_fc0, [0,self.padding])
        
        D1 = x_fc0.shape[-1]
        x_c0 = self.conv0(x_fc0,D1//2)

        x_c1 = self.conv1(x_c0,D1//4)

        x_c2 = self.conv2(x_c1,D1//4)
        
        x_c4 = self.conv4(x_c2 ,D1//2)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x_c5 = self.conv5(x_c4,D1)
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)


        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding]


        x_c5 = x_c5.permute(0, 2, 1)

        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)
        return x_out.unsqueeze(-2)

class UNO2d(nn.Module):
    def __init__(self, num_channels, width, pad = 6, factor = 1, initial_step = 10):
        super(UNO2d, self).__init__()

        self.in_width = num_channels * initial_step + 2 # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) 

        self.conv0 = OperatorBlock_2D(self.width, 2*factor*self.width,36, 36, 17, 17)

        self.conv1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 18, 18, 7, 7, Normalize = True)

        self.conv2 = OperatorBlock_2D(4*factor*self.width, 4*factor*self.width, 18, 18, 7, 7)

        self.conv4 = OperatorBlock_2D(4*factor*self.width, 2*factor*self.width, 36, 36, 7, 7, Normalize = True)

        self.conv5 = OperatorBlock_2D(4*factor*self.width, self.width, 81, 81, 17, 17) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 1*self.width)
        self.fc2 = nn.Linear(1*self.width, num_channels)

    def forward(self, x, grid):
        x = torch.cat((x, grid), dim=-1)

        x_fc_1 = self.fc_n1(x)
        x_fc_1 = F.gelu(x_fc_1)

        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        # scale = math.ceil(x_fc0.shape[-1]/85)
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0,D1//2,D2//2)

        x_c1 = self.conv1(x_c0,D1//4,D2//4)

        x_c2 = self.conv2(x_c1,D1//4,D2//4)
        
        x_c4 = self.conv4(x_c2 ,D1//2,D2//2)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x_c5 = self.conv5(x_c4,D1,D2)
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)


        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]


        x_c5 = x_c5.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)
        
        return x_out.unsqueeze(-2)

class UNO3d(nn.Module):
    def __init__(self, num_channels, width, pad = 5, factor = 1, initial_step = 10):
        super(UNO3d, self).__init__()

        self.in_width = num_channels * initial_step + 3 # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) 

        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,36, 36, 36, 17, 17, 17)

        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 18, 18, 18, 7, 7, 7, Normalize = True)

        self.conv2 = OperatorBlock_3D(4*factor*self.width, 4*factor*self.width, 18, 18, 18, 7, 7, 7)

        self.conv4 = OperatorBlock_3D(4*factor*self.width, 2*factor*self.width, 36, 36, 36, 7, 7, 7, Normalize = True)

        self.conv5 = OperatorBlock_3D(4*factor*self.width, self.width, 81, 81, 81, 17, 17, 17) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 1*self.width)
        self.fc2 = nn.Linear(1*self.width, num_channels)

    def forward(self, x, grid):
        x = torch.cat((x, grid), dim=-1)

        x_fc_1 = self.fc_n1(x)
        x_fc_1 = F.gelu(x_fc_1)

        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        # scale = math.ceil(x_fc0.shape[-1]/85)
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding, 0,self.padding])
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0,D1//2,D2//2,D3//2)

        x_c1 = self.conv1(x_c0,D1//4,D2//4,D3//4)

        x_c2 = self.conv2(x_c1,D1//4,D2//4,D3//4)
        
        x_c4 = self.conv4(x_c2 ,D1//2,D2//2,D3//2)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x_c5 = self.conv5(x_c4,D1,D2,D3)
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)


        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding, :-self.padding]


        x_c5 = x_c5.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)
        
        return x_out.unsqueeze(-2)

class UNO_maxwell(nn.Module):
    def __init__(self, num_channels, width, pad = 5, factor = 1, initial_step = 10):
        super(UNO_maxwell, self).__init__()

        self.in_width = num_channels * initial_step # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) 

        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,24, 24, 24, 10, 10, 10)

        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 12, 12, 12, 5, 5, 5, Normalize = True)

        self.conv2 = OperatorBlock_3D(4*factor*self.width, 4*factor*self.width, 12, 12, 12, 5, 5, 5)

        self.conv4 = OperatorBlock_3D(4*factor*self.width, 2*factor*self.width, 12, 12, 12, 5, 5, 5, Normalize = True)

        self.conv5 = OperatorBlock_3D(4*factor*self.width, self.width, 53, 53, 53, 10, 10, 10) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 1*self.width)
        self.fc2 = nn.Linear(1*self.width, num_channels)

    def forward(self, x, grid):
        x_fc_1 = self.fc_n1(x)
        x_fc_1 = F.gelu(x_fc_1)

        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        # scale = math.ceil(x_fc0.shape[-1]/85)
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding, 0,self.padding])
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0,D1//2,D2//2,D3//2)

        x_c1 = self.conv1(x_c0,D1//4,D2//4,D3//4)

        x_c2 = self.conv2(x_c1,D1//4,D2//4,D3//4)
        
        x_c4 = self.conv4(x_c2 ,D1//2,D2//2,D3//2)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x_c5 = self.conv5(x_c4,D1,D2,D3)
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)


        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding, :-self.padding]


        x_c5 = x_c5.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)
        
        return x_out.unsqueeze(-2)
    