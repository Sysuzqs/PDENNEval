# coding=utf-8
import numpy as np
import torch
import torch.nn as nn


def burger_1d_loss(pred, a, u, dx, dt):  #
    # pred,u: [bs,x,t,1]
    # ic and f loss only, because bc can be infered from data
    v_coeff= 0.001
    loss_fn = nn.MSELoss(reduction="mean")
    #ic
    ic_loss=loss_fn(pred[:,:,0],u[:,:,0])
    #f_loss
    # pred: [bs,x,t,1]
    du_t=(pred[:,:,2:]-pred[:,:,:-2])/(2*dt)
    du_x=(pred[:,2:]-pred[:,:-2])/(2*dx)
    du_xx=(pred[:, 2:] -2*pred[:,1:-1] +pred[:, :-2]) / (dx**2)
    Df = du_t[:,1:-1] + du_x[:,:,1:-1]*pred[:,1:-1,1:-1] - v_coeff/np.pi*du_xx[:,:,1:-1]
    f_loss=loss_fn(Df, torch.zeros_like(Df))
    return ic_loss, f_loss
    
def adv_1d_loss(pred, a, u, dx, dt):
    beta=0.1
    loss_fn = nn.MSELoss(reduction="mean")
    #ic
    ic_loss=loss_fn(pred[:,:,0],u[:,:,0])
    #f_loss
    # pred: [bs,x,t,1]
    du_t = (pred[:,:,2:]-pred[:,:,:-2])/(2*dt)
    du_x = (pred[:,2:]-pred[:,:-2])/(2*dx)
    Df = du_t[:,1:-1]+beta*du_x[:,:,1:-1]
    f_loss=loss_fn(Df,torch.zeros_like(Df))
    return ic_loss, f_loss

def diff_sorp_1d_loss(pred, a, u, dx, dt):  #
    D: float = 5e-4
    por: float = 0.29
    rho_s: float = 2880
    k_f: float = 3.5e-4
    n_f: float = 0.874
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss = loss_fn(pred[:,:,0],u[:,:,0])

    du_xx=(pred[:, 2:] -2*pred[:,1:-1] +pred[:, :-2]) / (dx**2)
    du_t = (pred[:,:,2:]-pred[:,:,:-2])/(2*dt)
    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (pred[:,1:-1,1:-1].clip(min=0) + 1e-6) ** (n_f - 1)
    Df= du_t[:,1:-1]-D/retardation_factor*du_xx[:,:,1:-1]

    f_loss=loss_fn(Df,torch.zeros_like(Df))
    return ic_loss, f_loss

def diff_react_1d_loss(pred, a, u, dx, dt):  #
    nu=0.5
    rho=1.0
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss = loss_fn(pred[:,:,0],u[:,:,0])

    du_xx=(pred[:, 2:] -2*pred[:,1:-1] +pred[:, :-2]) / (dx**2)
    du_t = (pred[:,:,2:]-pred[:,:,:-2])/(2*dt)
    Df= du_t[:,1:-1]-nu*du_xx[:,:,1:-1]-rho*pred[:,1:-1,1:-1]*(1.0-pred[:,1:-1,1:-1])
    f_loss = loss_fn(Df,torch.zeros_like(Df))
    return ic_loss, f_loss

def CFD_1d_loss(pred, a, u, dx, dt):
    zeta=0.1
    eta=0.1
    gamma = 5.0/3.0
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss = loss_fn(pred[:,:,0],u[:,:,0])

    h = pred[..., 0]  # rho
    p = pred[..., 1]  # p
    v = pred[..., 2]  # v
    E = p/(gamma - 1.) + 0.5 * h * v**2
    Fx = v * (E + p)

    dhv_x=((h*v)[:,2:]-(h*v)[:,:-2])/(2*dx)
    dh_t=(h[:,:,2:]-h[:,:,:-2])/(2*dt)
    dv_x=(v[:,2:]-v[:,:-2])/(2*dx)
    dv_t=(v[:,:,2:]-v[:,:,:-2])/(2*dt)
    dp_x=(p[:,2:]-p[:,:-2])/(2*dx)
    dFx_x=(Fx[:,2:]-Fx[:,:-2])/(2*dx)
    dE_t=(E[:,:,2:]-E[:,:,:-2])/(2*dt)
    dv_xx=(v[:, 2:] -2*v[:,1:-1] +v[:, :-2]) / (dx**2)

    eq1=dh_t[:,1:-1] + dhv_x[:,:,1:-1]
    eq2=h[:,1:-1,1:-1] * (dv_t[:,1:-1] + v[:,1:-1,1:-1] * dv_x[:,:,1:-1]) + dp_x[:,:,1:-1] - eta*dv_xx[:,:,1:-1] - (zeta+eta/3.0)*dv_xx[:,:,1:-1]
    eq3=dE_t[:,1:-1] + dFx_x[:,:,1:-1]
    f_loss = loss_fn(eq1,torch.zeros_like(eq1))+loss_fn(eq2,torch.zeros_like(eq2))+loss_fn(eq3,torch.zeros_like(eq3))
    return ic_loss, f_loss

def darcy_loss(pred, a, u, dx, dt=None):
    beta=1.0
    loss_fn = nn.MSELoss(reduction="mean")
    a=a.squeeze(-1)  # [bs, x, x, 1]
    du_x_= (pred[:,1:]-pred[:,:-1])/dx   # 1/2 step point
    du_y_= (pred[:,:,1:]-pred[:,:,:-1])/dx   # 1/2 step point
    ax_=(a[:,:-1]+a[:,1:])/2       # interpolating

    ay_=(a[:,:,1:]+a[:,:,:-1])/2
    Df= - ((ax_*du_x_)[:,1:,1:-1]-(ax_*du_x_)[:,:-1,1:-1]+(ay_*du_y_)[:,1:-1,1:]-(ay_*du_y_)[:,1:-1,:-1])/dx-beta
    f_loss= loss_fn(Df,torch.zeros_like(Df))
    return torch.tensor(0.0), f_loss  

def diff_react_2d_loss(pred, a, u, dx, dt):  #
    def reaction_1(u1,u2):
        k=5e-3
        return u1 - u1**3 - k - u2
    def reaction_2(u1,u2):
        return u1 - u2
    loss_fn = nn.MSELoss(reduction="mean")
    d1 = 1e-3
    d2 = 5e-3
    u1=pred[...,0]
    u2=pred[...,1]
    ic_loss=loss_fn(u1[:,:,:,0],u[:,:,:,0,0])+loss_fn(u2[:,:,:,0],u[:,:,:,0,1])

    du1_xx= (u1[:,2:] -2*u1[:,1:-1] +u1[:,:-2])/(dx**2)
    du1_yy= (u1[:,:,2:] -2*u1[:,:,1:-1] +u1[:,:,:-2])/(dx**2)
    du2_xx = (u2[:,2:] -2*u2[:,1:-1] +u2[:,:-2])/(dx**2)
    du2_yy = (u2[:,:,2:] -2*u2[:,:,1:-1] +u2[:,:,:-2])/(dx**2)
    du1_t = (u1[:,:,:,2:]-u1[:,:,:,:-2])/(2*dt)
    du2_t = (u2[:,:,:,2:]-u2[:,:,:,:-2])/(2*dt)
    eq1 = du1_t[:,1:-1,1:-1]-d1*(du1_xx[:,:,1:-1,1:-1]+du1_yy[:,1:-1,:,1:-1])-reaction_1(u1,u2)[:,1:-1,1:-1,1:-1]
    eq2 = du2_t[:,1:-1,1:-1]-d2*(du2_xx[:,:,1:-1,1:-1]+du2_yy[:,1:-1,:,1:-1])-reaction_2(u1,u2)[:,1:-1,1:-1,1:-1]
    f_loss = loss_fn(eq1,torch.zeros_like(eq1))+loss_fn(eq2,torch.zeros_like(eq2))

    return ic_loss, f_loss
    
def swe_2d_loss(pred, a, u, dx, dt):
    g=1.0
    h=pred[...,0:1]
    u1=pred[...,1:2]
    u2=pred[...,2:3]
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss=loss_fn(h[:,:,:,0],u[:,:,:,0])+ loss_fn(u1[:,:,:,0],torch.zeros_like(u1[:,:,:,0]))+ \
                loss_fn(u2[:,:,:,0],torch.zeros_like(u2[:,:,:,0]))
    dh_x = (h[:,2:]-h[:,:-2])/(2*dx)
    dh_y = (h[:,:,2:]-h[:,:,:-2])/(2*dx)
    dh_t = (h[:,:,:,2:]-h[:,:,:,:-2])/(2*dt)
    du1_x = (u1[:,2:]-u1[:,:-2])/(2*dx)
    du1_y = (u1[:,:,2:]-u1[:,:,:-2])/(2*dx)
    du1_t = (u1[:,:,:,2:]-u1[:,:,:,:-2])/(2*dt)
    du2_x = (u2[:,2:]-u2[:,:-2])/(2*dx)
    du2_y = (u2[:,:,2:]-u2[:,:,:-2])/(2*dx)
    du2_t = (u2[:,:,:,2:]-u2[:,:,:,:-2])/(2*dt)

    eq1 = dh_t[:,1:-1,1:-1] + dh_x[:,:,1:-1,1:-1] * u1[:,1:-1,1:-1,1:-1] + h[:,1:-1,1:-1,1:-1] * du1_x[:,:,1:-1,1:-1] + \
        dh_y[:,1:-1,:,1:-1] * u2[:,1:-1,1:-1,1:-1] + h[:,1:-1,1:-1,1:-1] * du2_y[:,1:-1,:,1:-1]
    eq2 = du1_t[:,1:-1,1:-1] + u1[:,1:-1,1:-1,1:-1] * du1_x[:,:,1:-1,1:-1] + u2[:,1:-1,1:-1,1:-1] * du1_y[:,1:-1,:,1:-1] + g * dh_x[:,:,1:-1,1:-1]
    eq3 = du2_t[:,1:-1,1:-1] + u1[:,1:-1,1:-1,1:-1] * du2_x[:,:,1:-1,1:-1] + u2[:,1:-1,1:-1,1:-1] * du2_y[:,1:-1,:,1:-1] + g * dh_y[:,1:-1,:,1:-1]

    f_loss = loss_fn(eq1,torch.zeros_like(eq1))+ loss_fn(eq2,torch.zeros_like(eq2))+ \
        loss_fn(eq3,torch.zeros_like(eq3))
    return ic_loss, f_loss

def CFD_2d_loss(pred, a, u, dx, dt):
    eta=0.1
    zeta=0.1
    gamma=5.0/3.0
    h = pred[..., 0:1]  # rho
    p = pred[..., 1:2]  # p
    u1 = pred[..., 2:3]  # vx
    u2 = pred[..., 3:4]  # vy
    E = p/(gamma - 1.) + 0.5 * h * (u1**2 + u2**2)
    Fx = u1 * (E + p)
    Fx = Fx
    Fy = u2 * (E + p)
    Fy = Fy
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss = loss_fn(h[...,0,:],u[...,0,0:1])+loss_fn(u1[...,0,:],u[...,0,1:2])+\
         loss_fn(u2[...,0,:],u[...,0,2:3])+loss_fn(p[...,0,:],u[...,0,3:4])
    # non conservative form
    dhu1_x = ((h*u1)[:,2:]-(h*u1)[:,:-2])/(2*dx)
    dhu2_y = ((h*u2)[:,:,2:]-(h*u2)[:,:,:-2])/(2*dx)
    dh_t = (h[:,:,:,2:]-h[:,:,:,:-2])/(2*dt)
    du1_x = (u1[:,2:]-u1[:,:-2])/(2*dx)
    du1_y = (u1[:,:,2:]-u1[:,:,:-2])/(2*dx)
    du1_t = (u1[:,:,:,2:]-u1[:,:,:,:-2])/(2*dt)
    du2_x = (u2[:,2:]-u2[:,:-2])/(2*dx)
    du2_y = (u2[:,:,2:]-u2[:,:,:-2])/(2*dx)
    du2_t = (u2[:,:,:,2:]-u2[:,:,:,:-2])/(2*dt)
    dp_x = (p[:,2:]-p[:,:-2])/(2*dx)
    dp_y = (p[:,:,2:]-p[:,:,:-2])/(2*dx)
    dFx_x = (Fx[:,2:]-Fx[:,:-2])/(2*dx)
    dFy_y = (Fy[:,:,2:]-Fy[:,:,:-2])/(2*dx)
    dE_t = (E[:,:,:,2:]-E[:,:,:,:-2])/(2*dt)

    du1_xx= (u1[:,2:]-2*u1[:,1:-1]+u1[:,:-2])/(dx**2)
    du1_yy= (u1[:,:,2:]-2*u1[:,:,1:-1]+u1[:,:,:-2])/(dx**2)
    du1_xy= (du1_x[:,:,2:]-du1_x[:,:,:-2])/dx
    du2_xx= (u2[:,2:]-2*u2[:,1:-1]+u2[:,:-2])/(dx**2)
    du2_yy= (u2[:,:,2:]-2*u2[:,:,1:-1]+u2[:,:,:-2])/(dx**2)
    du2_xy= (du2_x[:,:,2:]-du2_x[:,:,:-2])/dx
    eq1 = dh_t[:,1:-1,1:-1] + dhu1_x[:,:,1:-1,1:-1] + dhu2_y[:,1:-1,:,1:-1]
    eq2 = h[:,1:-1,1:-1,1:-1] * (du1_t[:,1:-1,1:-1] + u1[:,1:-1,1:-1,1:-1] * du1_x[:,:,1:-1,1:-1] + u2[:,1:-1,1:-1,1:-1] * du1_y[:,1:-1,:,1:-1]) + dp_x[:,:,1:-1,1:-1] - eta*(du1_xx[:,:,1:-1,1:-1]+du1_yy[:,1:-1,:,1:-1])-(zeta+eta/3.0)*(du1_xx[:,:,1:-1,1:-1]+du2_xy[:,:,:,1:-1])
    eq3 = h[:,1:-1,1:-1,1:-1] * (du2_t[:,1:-1,1:-1] + u1[:,1:-1,1:-1,1:-1] * du2_x[:,:,1:-1,1:-1] + u2[:,1:-1,1:-1,1:-1] * du2_y[:,1:-1,:,1:-1]) + dp_y[:,1:-1,:,1:-1] - eta*(du2_xx[:,:,1:-1,1:-1]+du2_yy[:,1:-1,:,1:-1])-(zeta+eta/3.0)*(du1_xy[:,:,:,1:-1]+du2_yy[:,1:-1,:,1:-1])
    eq4 = dE_t[:,1:-1,1:-1] + dFx_x[:,:,1:-1,1:-1] + dFy_y[:,1:-1,:,1:-1]

    f_loss= loss_fn(eq1,torch.zeros_like(eq1))+loss_fn(eq2,torch.zeros_like(eq2))+\
         loss_fn(eq3,torch.zeros_like(eq3))+loss_fn(eq4,torch.zeros_like(eq4))
    return ic_loss, f_loss
 
def Allen_Cahn_loss(pred, a, u, dx, dt):
    c1=0.0001
    c2=5.0
    loss_fn = nn.MSELoss(reduction="mean")
    #ic
    ic_loss=loss_fn(pred[:,:,0],u[:,:,0])
    #f_loss
    # pred: [bs,x,t,1]
    du_t=(pred[:,:,2:]-pred[:,:,:-2])/(2*dt)
    du_xx=(pred[:, 2:] -2*pred[:,1:-1] +pred[:, :-2]) / (dx**2)
    Df = du_t[:,1:-1]  - c1 * du_xx[:,:,1:-1] + c2 *( pred[:,1:-1,1:-1] ** 3 -  pred[:,1:-1,1:-1])
    f_loss=loss_fn(Df, torch.zeros_like(Df))
    return ic_loss, f_loss

def Cahn_Hilliard_loss(pred, a, u, dx, dt):
    gamma1=1.0e-6
    gamma2=0.01
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss=loss_fn(pred[:,:,0],u[:,:,0])

    du_t=(pred[:,:,2:]-pred[:,:,:-2])/(2*dt)
    du_xx=(pred[:, 2:] -2*pred[:,1:-1] +pred[:, :-2]) / (dx**2)
    h=gamma2*(pred[:,1:-1]**3-pred[:,1:-1])-gamma1*du_xx
    dh_xx=(h[:, 2:] -2*h[:,1:-1] +h[:, :-2]) / (dx**2)
    Df = du_t[:,2:-2]-dh_xx[:,:,1:-1]
    f_loss=loss_fn(Df, torch.zeros_like(Df))
    return ic_loss,f_loss

def burger_2d_loss(pred, a, u, dx, dt):
    v_coeff= 0.001
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss=loss_fn(pred[:,:,:,0],u[:,:,:,0])
    
    u1=pred[...,0:1]
    u2=pred[...,1:2]
    du1_t=(u1[:,:,:,2:]-u1[:,:,:,:-2])/(2*dt)
    du2_t=(u2[:,:,:,2:]-u2[:,:,:,:-2])/(2*dt)
    du1_x=(u1[:,2:]-u1[:,:-2])/(2*dx)
    du1_xx=(u1[:, 2:] -2*u1[:,1:-1] +u1[:, :-2]) / (dx**2)
    du1_y=(u1[:,:,2:]-u1[:,:,:-2])/(2*dx)
    du1_yy=(u1[:,:,2:] -2*u1[:,:,1:-1] +u1[:,:,:-2]) / (dx**2)
    du2_x=(u2[:,2:]-u2[:,:-2])/(2*dx)
    du2_xx=(u2[:, 2:] -2*u2[:,1:-1] +u2[:, :-2]) / (dx**2)
    du2_y=(u2[:,:,2:]-u2[:,:,:-2])/(2*dx)
    du2_yy=(u2[:,:,2:] -2*u2[:,:,1:-1] +u2[:,:,:-2]) / (dx**2)
    eq1= du1_t[:,1:-1,1:-1]+ u1[:,1:-1,1:-1,1:-1]*du1_x[:,:,1:-1,1:-1]+u2[:,1:-1,1:-1,1:-1]*du1_y[:,1:-1,:,1:-1]-v_coeff*(du1_xx[:,:,1:-1,1:-1]+du1_yy[:,1:-1,:,1:-1])
    eq2= du2_t[:,1:-1,1:-1]+ u1[:,1:-1,1:-1,1:-1]*du2_x[:,:,1:-1,1:-1]+u2[:,1:-1,1:-1,1:-1]*du2_y[:,1:-1,:,1:-1]-v_coeff*(du2_xx[:,:,1:-1,1:-1]+du2_yy[:,1:-1,:,1:-1])

    f_loss=loss_fn(eq1,torch.zeros_like(eq1))+loss_fn(eq2,torch.zeros_like(eq2))
    return ic_loss, f_loss

def Allen_Cahn_2d_loss(pred, a, u, dx, dt):
    c1=0.0001
    c2=1.0
    loss_fn = nn.MSELoss(reduction="mean")
    #ic
    ic_loss=loss_fn(pred[:,:,:,0],u[:,:,:,0])
    #f_loss
    # pred: [bs,x,y,t,1]

    du_t=(pred[:,:,:,2:]-pred[:,:,:,:-2])/(2*dt)
    du_xx=(pred[:, 2:] -2*pred[:,1:-1] +pred[:, :-2]) / (dx**2)
    du_yy=(pred[:,:,2:] -2*pred[:,:,1:-1] +pred[:,:,:-2]) / (dx**2)
    Df = du_t[:,1:-1,1:-1]  - c1 * (du_xx[:,:,1:-1,1:-1]+du_yy[:,1:-1,:,1:-1]) + c2 *( pred[:,1:-1,1:-1,1:-1] ** 3 -  pred[:,1:-1,1:-1,1:-1])
    f_loss=loss_fn(Df, torch.zeros_like(Df))
    return ic_loss, f_loss

def Black_Scholes_loss(pred, a, u, dx, dt, grid):
    r=0.05
    rho=0.4
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss=loss_fn(pred[:,:,:,0],u[:,:,:,0])

    du_x=(pred[:,2:]-pred[:,:-2])/(2*dx)
    du_y=(pred[:,:,2:]-pred[:,:,:-2])/(2*dx)
    du_t=(pred[:,:,:,2:]-pred[:,:,:,:-2])/(2*dt)
    du_xx=(pred[:, 2:] -2*pred[:,1:-1] +pred[:, :-2]) / (dx**2)
    du_yy=(pred[:,:,2:] -2*pred[:,:,1:-1] +pred[:,:,:-2]) / (dx**2)
    # breakpoint()
    Df=du_t[:,1:-1,1:-1] + 0.5*(rho**2 * grid[:,1:-1,1:-1,1:-1,0:1]**2*du_xx[:,:,1:-1,1:-1]+ grid[:,1:-1,1:-1,1:-1,1:2]**2*du_yy[:,1:-1,:,1:-1])  \
        - r*(pred[:,1:-1,1:-1,1:-1] - grid[:,1:-1,1:-1,1:-1,0:1]*du_x[:,:,1:-1,1:-1]- grid[:,1:-1,1:-1,1:-1,1:2]*du_y[:,1:-1,:,1:-1])
    f_loss=loss_fn(Df, torch.zeros_like(Df))
    return ic_loss, f_loss

def pde_loss(pred,a,u,train_args, grid=None):
    if train_args["scenario"] == '1D_Burgers':
        return burger_1d_loss(pred,a,u,train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "1D_Advection":
        return adv_1d_loss(pred,a,u,train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "1D_diffusion_sorption":
        return diff_sorp_1d_loss(pred,a,u,train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "1D_diffusion_reaction":
        return diff_react_1d_loss(pred, a, u,train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "1D_compressible_NS":
        return CFD_1d_loss(pred, a, u,train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "2D_DarcyFlow":
        return darcy_loss(pred,a,u,train_args["dx"])
    elif train_args["scenario"] == "2D_diffusion_reaction":
        return diff_react_2d_loss(pred,a,u,train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "2D_shallow_water":
        return swe_2d_loss(pred,a,u,train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "2D_Compressible_NS":
        return CFD_2d_loss(pred,a,u,train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "1D_Allen_Cahn":
        return Allen_Cahn_loss(pred, a, u, train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "1D_Cahn_Hilliard":
        return Cahn_Hilliard_loss(pred, a, u, train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "2D_Burgers":
        return burger_2d_loss(pred, a, u, train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "2D_Allen-Cahn":
        return Allen_Cahn_2d_loss(pred, a, u, train_args["dx"],train_args["dt"])
    elif train_args["scenario"] == "2D_black-scholes":
        return Black_Scholes_loss(pred, a, u, train_args["dx"],train_args["dt"], grid)
 
# lossmap={
#     '1D_Burgers':burger_1d_loss,
#     "1D_Advection":adv_1d_loss,
#     "1D_diffusion_sorption":diff_sorp_1d_loss,
#     "1D_diffusion_reaction":diff_react_1d_loss,
#     "1D_compressible_NS": CFD_1d_loss,
#     "2D_DarcyFlow":darcy_loss,
#     "2D_diffusion_reaction":diff_react_2d_loss,
#     "2D_shallow_water":swe_2d_loss,
#     "2D_Compressible_NS":CFD_2d_loss,
#     "1D_Allen_Cahn":Allen_Cahn_loss,
#     "1D_Cahn_Hilliard":Cahn_Hilliard_loss,
#     "2D_Burgers":burger_2d_loss
# }

# class Pdeloss:
#     def __init__(self, train_args):
#         super().__init__()
#         self.dx=train_args["dx"]
#         self.dt=train_args["dt"]
#         self.pdeloss=lossmap[train_args["scenario"]]
#     def __call__(self, pred, a, u):
#         return self.pdeloss(pred,a,u,self.dx,self.dt)
