import deepxde as dde
import numpy as np
import torch


def reaction_1(u1, u2):
    k = 5e-3

    return u1 - (u1 * u1 * u1) - k - u2


def reaction_2(u1, u2):
    return u1 - u2


def pde_diffusion_reaction(x, y):

    d1 = 1e-3
    d2 = 5e-3

    du1_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)
    du1_yy = dde.grad.hessian(y, x, i=1, j=1, component=0)
    du2_xx = dde.grad.hessian(y, x, i=0, j=0, component=1)
    du2_yy = dde.grad.hessian(y, x, i=1, j=1, component=1)

    # TODO: check indices of jacobian
    du1_t = dde.grad.jacobian(y, x, i=0, j=2)
    du2_t = dde.grad.jacobian(y, x, i=1, j=2)

    u1 = y[..., 0].unsqueeze(1)
    u2 = y[..., 1].unsqueeze(1)

    eq1 = du1_t - reaction_1(u1, u2) - d1 * (du1_xx + du1_yy)
    eq2 = du2_t - reaction_2(u1, u2) - d2 * (du2_xx + du2_yy)

    return [eq1 , eq2]


def pde_diffusion_sorption(x, y):
    D: float = 5e-4
    por: float = 0.29
    rho_s: float = 2880
    k_f: float = 3.5e-4
    n_f: float = 0.874

    du1_xx = dde.grad.hessian(y, x, i=0, j=0)
    du1_t = dde.grad.jacobian(y, x, i=0, j=1)

    u1 = y[..., 0].unsqueeze(1)

    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u1 + 1e-6) ** (
        n_f - 1
    )

    return du1_t - D / retardation_factor * du1_xx
    
    
def pde_swe1d():
    raise NotImplementedError


def pde_swe2d(x, y):
    g = 1.0

    h_x = dde.grad.jacobian(y, x, i=0, j=0)
    h_y = dde.grad.jacobian(y, x, i=0, j=1)
    h_t = dde.grad.jacobian(y, x, i=0, j=2)
    u_x = dde.grad.jacobian(y, x, i=1, j=0)
    u_y = dde.grad.jacobian(y, x, i=1, j=1)
    u_t = dde.grad.jacobian(y, x, i=1, j=2)
    v_x = dde.grad.jacobian(y, x, i=2, j=0)
    v_y = dde.grad.jacobian(y, x, i=2, j=1)
    v_t = dde.grad.jacobian(y, x, i=2, j=2)

    h = y[..., 0].unsqueeze(1)
    u = y[..., 1].unsqueeze(1)
    v = y[..., 2].unsqueeze(1)

    eq1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
    eq2 = u_t + u * u_x + v * u_y + g * h_x
    eq3 = v_t + u * v_x + v * v_y + g * h_y

    return [eq1 , eq2 , eq3]

def pde_adv1d(x, y, beta):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    return dy_t + beta * dy_x

def pde_diffusion_reaction_1d(x, y, nu, rho):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - nu * dy_xx - rho * y * (1. - y)

def pde_burgers1D(x, y, nu):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - nu / np.pi * dy_xx

def pde_CFD1d(x, y, gamma):
    zeta = 0.1
    eta = 0.1

    h = y[..., 0].unsqueeze(1)  # rho
    u = y[..., 1].unsqueeze(1)  # v
    p = y[..., 2].unsqueeze(1)  # p
    E = p/(gamma - 1.) + 0.5 * h * u**2
    E = E.unsqueeze(1)
    Fx = u * (E + p)
    Fx = Fx.unsqueeze(1)

    # non conservative form
    hu_x = dde.grad.jacobian(h * u, x, i=0, j=0)
    h_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_x = dde.grad.jacobian(y, x, i=1, j=0)
    u_t = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    E_t = dde.grad.jacobian(E, x, i=0, j=1)
    u_xx = dde.grad.jacobian(u_x, x, i=0, j=0)

    eq1 = h_t + hu_x
    eq2 = h * (u_t + u * u_x) + p_x - eta*u_xx - (zeta + eta/3.0)*u_xx
    eq3 = E_t + Fx_x

    return [eq1 , eq2 , eq3]

def pde_CFD2d(x, y, gamma):
    zeta = 0.1
    eta = 0.1

    h = y[..., 0].unsqueeze(1)  # rho
    ux = y[..., 1].unsqueeze(1)  # vx
    uy = y[..., 2].unsqueeze(1)  # vy
    p = y[..., 3].unsqueeze(1)  # p
    E = p/(gamma - 1.) + 0.5 * h * (ux**2 + uy**2)
    E = E.unsqueeze(1)
    Fx = ux * (E + p)
    Fx = Fx.unsqueeze(1)
    Fy = uy * (E + p)
    Fy = Fy.unsqueeze(1)

    # non conservative form
    hu_x = dde.grad.jacobian(h * ux, x, i=0, j=0)
    hu_y = dde.grad.jacobian(h * uy, x, i=0, j=1)
    h_t = dde.grad.jacobian(y, x, i=0, j=2)
    ux_x = dde.grad.jacobian(y, x, i=1, j=0)
    ux_y = dde.grad.jacobian(y, x, i=1, j=1)
    ux_t = dde.grad.jacobian(y, x, i=1, j=2)
    uy_x = dde.grad.jacobian(y, x, i=2, j=0)
    uy_y = dde.grad.jacobian(y, x, i=2, j=1)
    uy_t = dde.grad.jacobian(y, x, i=2, j=2)
    p_x = dde.grad.jacobian(y, x, i=3, j=0)
    p_y = dde.grad.jacobian(y, x, i=3, j=1)
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    Fy_y = dde.grad.jacobian(Fy, x, i=0, j=1)
    E_t = dde.grad.jacobian(E, x, i=0, j=2)

    ux_xx = dde.grad.jacobian(ux_x, x, i = 0, j = 0)
    ux_yy = dde.grad.jacobian(ux_y, x, i = 0, j = 1)
    uy_xx = dde.grad.jacobian(uy_x, x, i = 0, j = 0)
    uy_yy = dde.grad.jacobian(uy_y, x, i = 0, j = 1)

    eq1 = h_t + hu_x + hu_y
    eq2 = h * (ux_t + ux * ux_x + uy * ux_y) + p_x - eta*(ux_xx + ux_yy) - (zeta + eta/3.0)*(ux_xx + ux_yy)
    eq3 = h * (uy_t + ux * uy_x + uy * uy_y) + p_y - eta*(uy_xx + uy_yy) - (zeta + eta/3.0)*(uy_xx + uy_yy)
    eq4 = E_t + Fx_x + Fy_y

    

    return [eq1 , eq2 , eq3 , eq4]

def pde_CFD3d(x, y, gamma):
    zeta = 1e-8
    eta = 1e-8

    h = y[..., 0].unsqueeze(1)  # rho
    ux = y[..., 1].unsqueeze(1)  # vx
    uy = y[..., 2].unsqueeze(1)  # vy
    uz = y[..., 3].unsqueeze(1)  # vz
    p = y[..., 4].unsqueeze(1)  # p
    E = p/(gamma - 1.) + 0.5 * h * (ux**2 + uy**2 + uz**2)
    E = E.unsqueeze(1)
    Fx = ux * (E + p)
    Fx = Fx.unsqueeze(1)
    Fy = uy * (E + p)
    Fy = Fy.unsqueeze(1)
    Fz = uz * (E + p)
    Fz = Fz.unsqueeze(1)

    # print(Fx.shape)

    # non conservative form
    hu_x = dde.grad.jacobian(h * ux, x, i=0, j=0)
    hu_y = dde.grad.jacobian(h * uy, x, i=0, j=1)
    hu_z = dde.grad.jacobian(h * uy, x, i=0, j=2)
    h_t = dde.grad.jacobian(y, x, i=0, j=3)
    ux_x = dde.grad.jacobian(y, x, i=1, j=0)
    ux_y = dde.grad.jacobian(y, x, i=1, j=1)
    ux_z = dde.grad.jacobian(y, x, i=1, j=2)
    ux_t = dde.grad.jacobian(y, x, i=1, j=3)
    uy_x = dde.grad.jacobian(y, x, i=2, j=0)
    uy_y = dde.grad.jacobian(y, x, i=2, j=1)
    uy_z = dde.grad.jacobian(y, x, i=2, j=2)
    uy_t = dde.grad.jacobian(y, x, i=2, j=3)
    uz_x = dde.grad.jacobian(y, x, i=3, j=0)
    uz_y = dde.grad.jacobian(y, x, i=3, j=1)
    uz_z = dde.grad.jacobian(y, x, i=3, j=2)
    uz_t = dde.grad.jacobian(y, x, i=3, j=3)
    p_x = dde.grad.jacobian(y, x, i=4, j=0)
    p_y = dde.grad.jacobian(y, x, i=4, j=1)
    p_z = dde.grad.jacobian(y, x, i=4, j=2)
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    Fy_y = dde.grad.jacobian(Fy, x, i=0, j=1)
    Fz_z = dde.grad.jacobian(Fz, x, i=0, j=2)
    E_t = dde.grad.jacobian(E, x, i=0, j=3)

    ux_xx = dde.grad.jacobian(ux_x, x, i = 0, j = 0)
    ux_yy = dde.grad.jacobian(ux_y, x, i = 0, j = 1)
    ux_zz = dde.grad.jacobian(ux_z, x, i = 0, j = 2)
    uy_xx = dde.grad.jacobian(uy_x, x, i = 0, j = 0)
    uy_yy = dde.grad.jacobian(uy_y, x, i = 0, j = 1)
    uy_zz = dde.grad.jacobian(uy_z, x, i = 0, j = 2)
    uz_xx = dde.grad.jacobian(uz_x, x, i = 0, j = 0)
    uz_yy = dde.grad.jacobian(uz_y, x, i = 0, j = 1)
    uz_zz = dde.grad.jacobian(uz_z, x, i = 0, j = 2)

    eq1 = h_t + hu_x + hu_y + hu_z
    eq2 = h * (ux_t + ux * ux_x + uy * ux_y + uz * ux_z) + p_x - eta*(ux_xx + ux_yy + ux_zz) - (zeta + eta/3.0)*(ux_xx + ux_yy + ux_zz)
    eq3 = h * (uy_t + ux * uy_x + uy * uy_y + uz * uy_z) + p_y - eta*(uy_xx + uy_yy + uy_zz) - (zeta + eta/3.0)*(uy_xx + uy_yy + uy_zz)
    eq4 = h * (uz_t + ux * uz_x + uy * uz_y + uz * uz_z) + p_z - eta*(uz_xx + uz_yy + uz_zz) - (zeta + eta/3.0)*(uz_xx + uz_yy + uz_zz)
    eq5 = E_t + Fx_x + Fy_y + Fz_z

    # print(Fx.shape, eq1.shape, eq2.shape, eq3.shape, eq4.shape, eq5.shape)

    return [eq1 , eq2 , eq3 , eq4 , eq5]


def pde_burgers2D(x, y, nu):
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    du_t = dde.grad.jacobian(y, x, i=0, j=2)
    du_xx = dde.grad.hessian(du_x, x, i=0, j=0)
    du_yy = dde.grad.hessian(du_y, x, i=0, j=1)

    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dv_t = dde.grad.jacobian(y, x, i=1, j=2)
    dv_xx = dde.grad.hessian(dv_x, x, i=0, j=0)
    dv_yy = dde.grad.hessian(dv_y, x, i=0, j=1)
    
    eq1 = du_t + y[:, 0:1]*du_x + y[:, 1:2]*du_y - nu*(du_xx+du_yy)
    eq2 = dv_t + y[:, 0:1]*dv_x + y[:, 1:2]*dv_y - nu*(dv_xx+dv_yy)

    return [eq1, eq2]

def pde_Allen_Cahn(x, y, c1, c2):
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_xx = dde.grad.jacobian(u_x, x, i=0, j=0)
    phi = y**3 - y

    return u_t - c1*u_xx + c2*phi

def pde_Allen_Cahn2d(x, y, c1, c2):
    u_t = dde.grad.jacobian(y, x, i=0, j=2)
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.jacobian(u_x, x, i=0, j=0)
    u_yy = dde.grad.jacobian(u_y, x, i=0, j=1)
    phi = y**3 - y

    return u_t - c1*(u_xx+u_yy) + c2*phi

def pde_Cahn_Hilliard(x, y, gamma1, gamma2):
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_xx = dde.grad.jacobian(u_x, x, i=0, j=0)
    phi = y**3 - y
    p = gamma2*phi - gamma1*u_xx
    p_x = dde.grad.jacobian(p, x, i=0, j=0)
    p_xx = dde.grad.jacobian(p_x, x, i=0, j=0)

    return u_t - p_xx


def pde_darcy_flow(x, y, beta):
    # x_list = x[..., :2].clone().detach()
    # idx = list(map(int, x_list*128+0.5))
    # idx = x_list[..., :2]
    # print(nu_map)
    # idx = list(map(int, idx*128+0.5))

    u_x = dde.grad.jacobian(y, x, i = 0, j = 0)
    u_y = dde.grad.jacobian(y, x, i = 0, j = 1)
    # u_t = dde.grad.jacobian(y, x, i = 0, j = 2)
    a = x[..., 3].unsqueeze(1)
    au_x = a*u_x
    # print(au_x.shape, x.shape)
    au_y = a*u_y
    au_xx = dde.grad.jacobian(au_x, x, i = 0, j = 0)
    au_yy = dde.grad.jacobian(au_y, x, i = 0, j = 1)

    return (au_xx + au_yy) - beta


def pde_euler(x, y, gamma):
    zeta = 0
    eta = 0

    print(x[1])

    h = y[..., 0].unsqueeze(1)  # rho
    ux = y[..., 1].unsqueeze(1)  # vx
    uy = y[..., 2].unsqueeze(1)  # vy
    uz = y[..., 3].unsqueeze(1)  # vz
    p = y[..., 4].unsqueeze(1)  # p
    E = p/(gamma - 1.) + 0.5 * h * (ux**2 + uy**2 + uz**2)
    E = E.unsqueeze(1)
    Fx = ux * (E + p)
    Fx = Fx.unsqueeze(1)
    Fy = uy * (E + p)
    Fy = Fy.unsqueeze(1)
    Fz = uz * (E + p)
    Fz = Fz.unsqueeze(1)

    # print(Fx.shape)

    # non conservative form
    hu_x = dde.grad.jacobian(h * ux, x, i=0, j=0)
    hu_y = dde.grad.jacobian(h * uy, x, i=0, j=1)
    hu_z = dde.grad.jacobian(h * uy, x, i=0, j=2)
    h_t = dde.grad.jacobian(y, x, i=0, j=3)
    ux_x = dde.grad.jacobian(y, x, i=1, j=0)
    ux_y = dde.grad.jacobian(y, x, i=1, j=1)
    ux_z = dde.grad.jacobian(y, x, i=1, j=2)
    ux_t = dde.grad.jacobian(y, x, i=1, j=3)
    uy_x = dde.grad.jacobian(y, x, i=2, j=0)
    uy_y = dde.grad.jacobian(y, x, i=2, j=1)
    uy_z = dde.grad.jacobian(y, x, i=2, j=2)
    uy_t = dde.grad.jacobian(y, x, i=2, j=3)
    uz_x = dde.grad.jacobian(y, x, i=3, j=0)
    uz_y = dde.grad.jacobian(y, x, i=3, j=1)
    uz_z = dde.grad.jacobian(y, x, i=3, j=2)
    uz_t = dde.grad.jacobian(y, x, i=3, j=3)
    p_x = dde.grad.jacobian(y, x, i=4, j=0)
    p_y = dde.grad.jacobian(y, x, i=4, j=1)
    p_z = dde.grad.jacobian(y, x, i=4, j=2)
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    Fy_y = dde.grad.jacobian(Fy, x, i=0, j=1)
    Fz_z = dde.grad.jacobian(Fz, x, i=0, j=2)
    E_t = dde.grad.jacobian(E, x, i=0, j=3)

    ux_xx = dde.grad.jacobian(ux_x, x, i = 0, j = 0)
    ux_yy = dde.grad.jacobian(ux_y, x, i = 0, j = 1)
    ux_zz = dde.grad.jacobian(ux_z, x, i = 0, j = 2)
    uy_xx = dde.grad.jacobian(uy_x, x, i = 0, j = 0)
    uy_yy = dde.grad.jacobian(uy_y, x, i = 0, j = 1)
    uy_zz = dde.grad.jacobian(uy_z, x, i = 0, j = 2)
    uz_xx = dde.grad.jacobian(uz_x, x, i = 0, j = 0)
    uz_yy = dde.grad.jacobian(uz_y, x, i = 0, j = 1)
    uz_zz = dde.grad.jacobian(uz_z, x, i = 0, j = 2)

    eq1 = h_t + hu_x + hu_y + hu_z
    eq2 = h * (ux_t + ux * ux_x + uy * ux_y + uz * ux_z) + p_x 
    eq3 = h * (uy_t + ux * uy_x + uy * uy_y + uz * uy_z) + p_y 
    eq4 = h * (uz_t + ux * uz_x + uy * uz_y + uz * uz_z) + p_z 
    eq5 = E_t + Fx_x + Fy_y + Fz_z

    # print(Fx.shape, eq1.shape, eq2.shape, eq3.shape, eq4.shape, eq5.shape)

    return [eq1 , eq2 , eq3 , eq4 , eq5]

def pde_BS(x, y):
    x1 = x[..., 0]
    x2 = x[..., 1]
    t = x[..., 2]

    u_t = dde.grad.jacobian(y, x, i=0, j=2)
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.jacobian(u_x, x, i=0, j=0)
    u_yy = dde.grad.jacobian(u_y, x, i=0, j=1)

    eq = u_t + 0.08*(x1**2*u_xx + x2**2*u_yy) - 0.05*y +0.05*(x1*u_x+x2*u_y)

    return eq


def pde_Maxwell(x, y):
    # raise NotImplementedError
    # E1 = y[..., 0]
    # E2 = y[..., 1]
    # E3 = y[..., 2]
    # H1 = y[..., 0]
    # H2 = y[..., 1]
    # H3 = y[..., 2]
    e0 = 8.854187817e-12

    E1_x = dde.grad.jacobian(y, x, i = 0, j = 0)
    E2_y = dde.grad.jacobian(y, x, i = 1, j = 1)
    E3_z = dde.grad.jacobian(y, x, i = 2, j = 2)
    H1_x = dde.grad.jacobian(y, x, i = 3, j = 0)
    H2_y = dde.grad.jacobian(y, x, i = 4, j = 1)
    H3_z = dde.grad.jacobian(y, x, i = 5, j = 2)

    divD = E1_x+E2_y+E3_z
    divH = H1_x+H2_y+H3_z
    return [divD, divH]