import torch
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def metric_func(pred, target, if_mean=True, Lx=1., Ly=1., Lz=1., iLow=4, iHigh=12):
    """
    code for calculate metrics discussed in the Brain-storming session
    RMSE, normalized RMSE, max error, RMSE at the boundaries, conserved variables, RMSE in Fourier space, temporal sensitivity
    """
    pred, target = pred.to(device), target.to(device)
    # (batch, nx^i..., timesteps, nc)
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]
    # print(pred.size(), target.size())

    # RMSE
    err_mean = torch.sqrt(torch.mean((pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2, dim=2))
    err_RMSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)

    err_CSV = torch.sqrt(torch.mean(
        (torch.sum(pred.view([nb, nc, -1, nt]), dim=2) - torch.sum(target.view([nb, nc, -1, nt]), dim=2)) ** 2,
        dim=0))
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz
    # worst case in all the data
    err_Max = torch.max(torch.max(
        torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])), dim=2)[0], dim=0)[0]

    if len(idxs) == 4:  # 1D
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.), dim=0)
    elif len(idxs) == 5:  # 2D
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (2 * nx + 2 * ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    elif len(idxs) == 6:  # 3D
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0] - target[:, :, :, :, 0]) ** 2
        err_BD_z += (pred[:, :, :, :, -1] - target[:, :, :, :, -1]) ** 2
        # print(err_BD_x.size(), [nb, -1, nt])
        err_BD = torch.sum(err_BD_x.reshape([nc, -1, nt]), dim=[-2, -1]) \
                 + torch.sum(err_BD_y.reshape([nc, -1, nt]), dim=[-2, -1]) \
                 + torch.sum(err_BD_z.reshape([nc, -1, nt]), dim=[-2, -1])
        err_BD = err_BD / (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        print(err_BD.size())
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)

    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = torch.sqrt(torch.mean(torch.abs(pred_F - target_F) ** 2, axis=0)) / nx * Lx
    if len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i ** 2 + j ** 2))
                if it > min(nx // 2, ny // 2) - 1:
                    continue
                err_F[:, :, it] += _err_F[:, :, i, j]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny) * Lx * Ly
    elif len(idxs) == 6:  # 3D
        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2, nz // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                for k in range(nz // 2):
                    it = mt.floor(mt.sqrt(i ** 2 + j ** 2 + k ** 2))
                    if it > min(nx // 2, ny // 2, nz // 2) - 1:
                        continue
                    err_F[:, :, it] += _err_F[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny * nz) * Lx * Ly * Lz

    err_F = torch.zeros([nc, 3, nt]).to(device)
    err_F[:,0] += torch.mean(_err_F[:,:iLow], dim=1)  # low freq
    err_F[:,1] += torch.mean(_err_F[:,iLow:iHigh], dim=1)  # middle freq
    err_F[:,2] += torch.mean(_err_F[:,iHigh:], dim=1)  # high freq

    if if_mean:
        # return torch.mean(err_RMSE, dim=[0, -1]), \
        #        torch.mean(err_nRMSE, dim=[0, -1]), \
        #        torch.mean(err_CSV, dim=[0, -1]), \
        #        torch.mean(err_Max, dim=[0, -1]), \
        #        torch.mean(err_BD, dim=[0, -1]), \
        #        torch.mean(err_F, dim=[0, -1])
        return torch.mean(err_RMSE, dim=-1), \
               torch.mean(err_nRMSE, dim=-1), \
               torch.mean(err_CSV, dim=-1), \
               torch.mean(err_Max, dim=-1), \
               torch.mean(err_BD, dim=-1), \
               torch.mean(err_F, dim=-1)
    else:
        return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F

def metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot, model_name, x_min,
            x_max, y_min, y_max, t_min, t_max, mode='FNO', initial_step=None, ):
    if mode=='Unet':
        with torch.no_grad():
            itot = 0
            for xx, yy in val_loader:
                xx = xx.to(device)
                yy = yy.to(device)

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                    inp = inp.permute(temp_shape)
                    
                    y = yy[..., t:t+1, :]
                
                    temp_shape = [0]
                    temp_shape.extend([i for i in range(2,len(inp.shape))])
                    temp_shape.append(1)
                    im = model(inp).permute(temp_shape).unsqueeze(-2)
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                    = metric_func(pred, yy, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

                if itot == 0:
                    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F \
                        = _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F
                    pred_plot = pred[:1]
                    target_plot = yy[:1]
                    val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                else:
                    err_RMSE += _err_RMSE
                    err_nRMSE += _err_nRMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F
                    
                    mean_dim = [i for i in range(len(yy.shape)-2)]
                    mean_dim.append(-1)
                    mean_dim = tuple(mean_dim)
                    val_l2_time += torch.sqrt(torch.mean((pred-yy)**2, dim=mean_dim))
                
                itot += 1

    elif mode=='FNO':
        with torch.no_grad():
            itot = 0
            for xx, yy, grid in val_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                grid = grid.to(device)

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    y = yy[..., t:t + 1, :]
                    im = model(inp, grid)
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                    = metric_func(pred, yy, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)
                if itot == 0:
                    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F \
                        = _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F
                    pred_plot = pred[:1]
                    target_plot = yy[:1]
                    val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                else:
                    err_RMSE += _err_RMSE
                    err_nRMSE += _err_nRMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F
                    
                    mean_dim = [i for i in range(len(yy.shape)-2)]
                    mean_dim.append(-1)
                    mean_dim = tuple(mean_dim)
                    val_l2_time += torch.sqrt(torch.mean((pred-yy)**2, dim=mean_dim))

                itot += 1

    elif mode == "PINN":
        raise NotImplementedError


    err_RMSE = np.array(err_RMSE.data.cpu()/itot)
    err_nRMSE = np.array(err_nRMSE.data.cpu()/itot)
    err_CSV = np.array(err_CSV.data.cpu()/itot)
    err_Max = np.array(err_Max.data.cpu()/itot)
    err_BD = np.array(err_BD.data.cpu()/itot)
    err_F = np.array(err_F.data.cpu()/itot)
    print('RMSE: {0:.5f}'.format(err_RMSE))
    print('normalized RMSE: {0:.5f}'.format(err_nRMSE))
    print('RMSE of conserved variables: {0:.5f}'.format(err_CSV))
    print('Maximum value of rms error: {0:.5f}'.format(err_Max))
    print('RMSE at boundaries: {0:.5f}'.format(err_BD))
    print('RMSE in Fourier space: {0}'.format(err_F))
    
    val_l2_time = val_l2_time/itot
    
    if plot:
        dim = len(yy.shape) - 3
        plt.ioff()
        if dim == 1:
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(pred_plot[...,channel_plot].squeeze().detach().cpu(),
                       extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,channel_plot].min(), target_plot[...,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Prediction", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$x$", fontsize=30)
            ax.set_xlabel("$t$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_pred.pdf'
            plt.savefig(filename)
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(target_plot[...,channel_plot].squeeze().detach().cpu(),
                       extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,channel_plot].min(), target_plot[...,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$x$", fontsize=30)
            ax.set_xlabel("$t$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_data.pdf'
            plt.savefig(filename)
    
        elif dim == 2:
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(pred_plot[...,-1,channel_plot].squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,-1,channel_plot].min(), target_plot[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Prediction", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_pred.pdf'
            plt.savefig(filename)
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(target_plot[...,-1,channel_plot].squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,-1,channel_plot].min(), target_plot[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_data.pdf'
            plt.savefig(filename)
    
        # plt.figure(figsize=(8,8))
        # plt.semilogy(torch.arange(initial_step,yy.shape[-2]),
        #              val_l2_time[initial_step:].detach().cpu())
        # plt.xlabel('$t$', fontsize=30)
        # plt.ylabel('$MSE$', fontsize=30)
        # plt.title('MSE vs unrolled time steps', fontsize=30)
        # plt.tight_layout()
        # filename = model_name + '_mse_time.pdf'
        # plt.savefig(filename)
        
        filename = model_name + 'mse_time.npz'
        np.savez(filename, t=torch.arange(initial_step,yy.shape[-2]).cpu(),
                 mse=val_l2_time[initial_step:].detach().cpu())

    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


# LpLoss Function  
class LpLoss(object):
    """
    Lp loss function 
    """
    def __init__(self, p=2, reduction='mean'):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert p > 0
        self.p = p
        self.reduction = reduction
    def __call__(self, x, y, eps=1e-20):
        num_examples = x.size()[0]
        _diff = x.view(num_examples,-1) - y.view(num_examples,-1)
        _diff = torch.norm(_diff, self.p, 1)
        _norm = eps + torch.norm(y.view(num_examples,-1), self.p, 1)
        if self.reduction in ['mean']:
            return torch.mean(_diff/_norm)
        if self.reduction in ['sum']:
            return torch.sum(_diff/_norm)
        return _diff/_norm

# FftLoss Function  
class FftLpLoss(object):
    """
    loss function in Fourier space

    June 2022, F.Alesiani
    """
    def __init__(self, p=2, reduction='mean'):
        super(FftLpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert p > 0
        self.p = p
        self.reduction = reduction
    def __call__(self, x, y, flow=None,fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:]
        dims = list(range(1,len(x.shape)))
        xf = torch.fft.fftn(x,dim=dims)
        yf = torch.fft.fftn(y,dim=dims)
        if flow is None: flow = 0
        if fhigh is None: fhigh = np.max(xf.shape[1:])

        if len(others_dims) ==1:
            xf = xf[:,flow:fhigh]
            yf = yf[:,flow:fhigh]        
        if len(others_dims) ==2:
            xf = xf[:,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh]
        if len(others_dims) ==3:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh]
        if len(others_dims) ==4:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]

        _diff = xf - yf.reshape(xf.shape)
        _diff = torch.norm(_diff.reshape(num_examples,-1), self.p, 1)
        _norm = eps + torch.norm(yf.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction in ['mean']:
            return torch.mean(_diff/_norm)
        if self.reduction in ['sum']:
            return torch.sum(_diff/_norm)
        return _diff/_norm

import torch.nn.functional as F
# FftLoss Function  
class FftMseLoss(object):
    """
    loss function in Fourier space

    June 2022, F.Alesiani
    """
    def __init__(self, reduction='mean'):
        super(FftMseLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        self.reduction = reduction
    def __call__(self, x, y, flow=None,fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:-2]
        for d in others_dims:
            assert (d>1), "we expect the dimension to be the same and greater the 1"
        # print(others_dims)
        dims = list(range(1,len(x.shape)-1))
        xf = torch.fft.fftn(x,dim=dims)
        yf = torch.fft.fftn(y,dim=dims)
        if flow is None: flow = 0
        if fhigh is None: fhigh = np.max(xf.shape[1:])

        if len(others_dims) ==1:
            xf = xf[:,flow:fhigh]
            yf = yf[:,flow:fhigh]        
        if len(others_dims) ==2:
            xf = xf[:,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh]
        if len(others_dims) ==3:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh]
        if len(others_dims) ==4:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
        _diff = xf - yf
        _diff = _diff.reshape(num_examples,-1).abs()**2
        if self.reduction in ['mean']:
            return torch.mean(_diff).abs()
        if self.reduction in ['sum']:
            return torch.sum(_diff).abs()
        return _diff.abs()


import torch.nn as nn
def inverse_metrics(u0,x,pred_u0,y):
    """
    computes all the metrics in the base and fourier space
    u0: esimated initial condition, 
    pred_u0: prediction from the esimated initial condition, pred_u0 = model(u0)
    x: true initial condition
    y: true prediction, y = model(x)

    June 2022, F.Alesiani
    """
    
    mseloss_fn = nn.MSELoss(reduction="mean")
    l2loss_fn = LpLoss(p=2,reduction='mean')
    l3loss_fn = LpLoss(p=3,reduction='mean')
    
    fftmseloss_fn = FftMseLoss(reduction="mean")
    fftl2loss_fn = FftLpLoss(p=2,reduction="mean")
    fftl3loss_fn = FftLpLoss(p=3,reduction="mean")    

    #initial condition
    mseloss_u0 = mseloss_fn(u0.view(1, -1), x.view(1, -1)).item()
    l2loss_u0 = l2loss_fn(u0.view(1, -1), x.view(1, -1)).item()
    l3loss_u0 = l3loss_fn(u0.view(1, -1), x.view(1, -1)).item()
    
    
    fmid = u0.shape[1]//4

    fftmseloss_u0 = fftmseloss_fn(u0, x).item()
    fftmseloss_low_u0 = fftmseloss_fn(u0, x, 0, fmid).item()
    fftmseloss_mid_u0 = fftmseloss_fn(u0, x, fmid, 2*fmid).item()
    fftmseloss_hi_u0 = fftmseloss_fn(u0, x, 2*fmid).item()
    
    fftl2loss_u0 = fftl2loss_fn(u0, x).item()
    fftl2loss_low_u0 = fftl2loss_fn(u0, x, 0, fmid).item()
    fftl2loss_mid_u0 = fftl2loss_fn(u0, x, fmid, 2*fmid).item()
    fftl2loss_hi_u0 = fftl2loss_fn(u0, x, 2*fmid).item()

    fftl3loss_u0 = fftl3loss_fn(u0, x).item()
    fftl3loss_low_u0 = fftl3loss_fn(u0, x, 0, fmid).item()
    fftl3loss_mid_u0 = fftl3loss_fn(u0, x, fmid, 2*fmid).item()
    fftl3loss_hi_u0 = fftl3loss_fn(u0, x, 2*fmid).item()    

    #prediction 
    mseloss_pred_u0 = mseloss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
    l2loss_pred_u0 = l2loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
    l3loss_pred_u0 = l3loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()

    fmid = pred_u0.shape[1]//4
    pred_u0 = pred_u0.squeeze(-1)
    y = y.squeeze(-1)
    
    fftmseloss_pred_u0 = fftmseloss_fn(pred_u0, y).item()
    fftmseloss_low_pred_u0 = fftmseloss_fn(pred_u0, y, 0, fmid).item()
    fftmseloss_mid_pred_u0 = fftmseloss_fn(pred_u0, y, fmid, 2*fmid).item()
    fftmseloss_hi_pred_u0 = fftmseloss_fn(pred_u0, y, 2*fmid).item()
    
    fftl2loss_pred_u0 = fftl2loss_fn(pred_u0, y).item()
    fftl2loss_low_pred_u0 = fftl2loss_fn(pred_u0, y, 0, fmid).item()
    fftl2loss_mid_pred_u0 = fftl2loss_fn(pred_u0, y, fmid, 2*fmid).item()
    fftl2loss_hi_pred_u0= fftl2loss_fn(pred_u0, y, 2*fmid).item()

    fftl3loss_pred_u0 = fftl3loss_fn(pred_u0, y).item()
    fftl3loss_low_pred_u0 = fftl3loss_fn(pred_u0, y, 0, fmid).item()
    fftl3loss_mid_pred_u0 = fftl3loss_fn(pred_u0, y, fmid, 2*fmid).item()
    fftl3loss_hi_pred_u0 = fftl3loss_fn(pred_u0, y, 2*fmid).item()   

    metric = {
        'mseloss_u0': mseloss_u0
        ,'l2loss_u0': l2loss_u0
        ,'l3loss_u0': l3loss_u0

        ,'mseloss_pred_u0': mseloss_pred_u0
        ,'l2loss_pred_u0': l2loss_pred_u0
        ,'l3loss_pred_u0': l3loss_pred_u0


        ,'fftmseloss_u0': fftmseloss_u0
        ,'fftmseloss_low_u0': fftmseloss_low_u0
        ,'fftmseloss_mid_u0': fftmseloss_mid_u0
        ,'fftmseloss_hi_u0': fftmseloss_hi_u0



        ,'fftmseloss_pred_u0': fftmseloss_pred_u0
        ,'fftmseloss_low_pred_u0': fftmseloss_low_pred_u0
        ,'fftmseloss_mid_pred_u0': fftmseloss_mid_pred_u0
        ,'fftmseloss_hi_pred_u0': fftmseloss_hi_pred_u0

        ,'fftl2loss_u0': fftl2loss_u0
        ,'fftl2loss_low_u0': fftl2loss_low_u0
        ,'fftl2loss_mid_u0': fftl2loss_mid_u0
        ,'fftl2loss_hi_u0': fftl2loss_hi_u0

        ,'fftl2loss_pred_u0': fftl2loss_pred_u0
        ,'fftl2loss_low_pred_u0': fftl2loss_low_pred_u0
        ,'fftl2loss_mid_pred_u0': fftl2loss_mid_pred_u0
        ,'fftl2loss_hi_pred_u0': fftl2loss_hi_pred_u0

        ,'fftl3loss_u0': fftl3loss_u0
        ,'fftl3loss_low_u0': fftl3loss_low_u0
        ,'fftl3loss_mid_u0': fftl3loss_mid_u0
        ,'fftl3loss_hi_u0': fftl3loss_hi_u0

        ,'fftl3loss_pred_u0': fftl3loss_pred_u0
        ,'fftl3loss_low_pred_u0': fftl3loss_low_pred_u0
        ,'fftl3loss_mid_pred_u0': fftl3loss_mid_pred_u0
        ,'fftl3loss_hi_pred_u0': fftl3loss_hi_pred_u0        
    }    

    return metric

def bMSE(pred, target):
    """return mean square error or root mean square error in a batch

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc, nt = pred.shape[0], pred.shape[1], pred.shape[-1]
    spatial_dim = len(pred.shape) - 3
    if spatial_dim == 1:
        bd_square_error = (pred[:, :, 0, :] - target[:, :, 0, :])**2
        bd_square_error += (pred[:, :, -1, :] - target[:, :, -1, :])**2 # (bs, v, t)
        bd_mean_square_error = bd_square_error / 2.
    elif spatial_dim == 2:
        bd_x_square_error = (pred[:, :, 0, :, :] - target[:, :, 0, :, :])**2
        bd_x_square_error += (pred[:, :, -1, :, :] - target[:, :, -1, :, :])**2
        bd_x_mean_square_error = torch.mean(bd_x_square_error / 2., dim=2) # (bs, v, t)
        bd_y_square_error = (pred[:, :, :, 0, :] - target[:, :, :, 0, :])**2
        bd_y_square_error += (pred[:, :, :, -1, :] - target[:, :, :, -1, :])**2
        bd_y_mean_square_error = torch.mean(bd_y_square_error / 2., dim=2) # (bs, v, t)
        bd_mean_square_error = (bd_x_mean_square_error + bd_y_mean_square_error) / 2.
    else: # spatial_dim == 3
        bd_x_square_error = (pred[:, :, 0, :, :, :] - target[:, :, 0, :, :, :])**2
        bd_x_square_error += (pred[:, :, -1, :, :, :] - target[:, :, -1, :, :, :])**2
        bd_x_mean_square_error = torch.mean(bd_x_square_error.view([nb, nc, -1, nt]) / 2., dim=2) # (bs, v, t)
        bd_y_square_error = (pred[:, :, :, 0, :, :] - target[:, :, :, 0, :, :])**2
        bd_y_square_error += (pred[:, :, :, -1, :, :] - target[:, :, :, -1, :, :])**2
        bd_y_mean_square_error = torch.mean(bd_y_square_error.view([nb, nc, -1, nt]) / 2., dim=2) # (bs, v, t)
        bd_z_square_error = (pred[:, :, :, :, 0, :] - target[:, :, :, :, 0, :])**2
        bd_z_square_error += (pred[:, :, :, :, -1, :] - target[:, :, :, :, -1, :])**2
        bd_z_mean_square_error = torch.mean(bd_y_square_error.view([nb, nc, -1, nt]) / 2., dim=2) # (bs, v, t)
        bd_mean_square_error = (bd_x_mean_square_error + bd_y_mean_square_error + bd_z_mean_square_error) / 3.
    return torch.mean(bd_mean_square_error, dim=0) # (v, t)


def bRMSE(pred, target):
    return torch.sqrt(bMSE(pred, target))

def L1RE(pred, target):
    """l1 relative error (backup)

    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    squared: bool, default=True. If True returns MSE value, if False returns RMSE value.
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape)
    print(pred.shape)
    target = target.permute(temp_shape)
    nb, nc, nt = pred.shape[0], pred.shape[1], pred.shape[-1]
    l1_errors = torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt]))
    res = torch.sum(l1_errors, dim=2) / torch.sum(torch.abs(target.view([nb, nc, -1, nt])), dim=2)
    return torch.mean(res, dim = [0, -1])