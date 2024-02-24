# coding=utf-8
# taken from https://github.com/avirmaux/lipEstimation
import itertools
import numpy as np
import numpy.random as rd
import scipy as sp
from scipy.optimize import minimize
import torch


def spectral_norm_sq(mat):
    """ Return the square of the spectral norm of `mat` """
    return sp.linalg.norm(mat, ord=2)


def _diag(vec, width, height):
    """ Return rectangle matrix of shape (m, n) with vector v on _diagonal
    """
    diag = np.zeros((width, height))
    idx = np.arange(len(vec), dtype=np.int)
    diag[idx, idx] = vec
    return diag


def optim_bf(mat_l, mat_r, verbose=True):
    """ Compute maximum spectral norm for |U d(sigma) V|
    with sigma binary _diagonal

    Algorithm: bruteforce

    TODO: change name

    Performances::
    size 5  ~ 10ms
    size 10 ~ 80ms
    size 15 ~ 4s
    size 20 ~ 2mn20
    """
    from tqdm import tqdm
    max_norm = 0
    word_max = None
    for sigma in tqdm(itertools.product([0, 1], repeat=mat_l.shape[1]),
                      total=2**mat_l.shape[1],
                      disable=(not verbose)):
        norm = spectral_norm_sq(mat_l @ _diag(sigma, mat_l.shape[1], mat_r.shape[0]) @ mat_r)
        if norm >= max_norm:
            max_norm = norm
            word_max = sigma
    return max_norm, word_max


def f(sigma, mat_l, mat_r):
    return mat_l @ _diag(sigma, mat_l.shape[1], mat_r.shape[0]) @ mat_r


def f_spec(sigma, mat_l, mat_r):
    """ Return the spectral norm of mat_l @ diag(sigma) @ mat_r """
    return spectral_norm_sq(f(sigma, mat_l, mat_r))


def f_spec_grad(sigma, mat_l, mat_r):
    """ Compute the gradient of `f_spec` with respect to sigma
    """
    mat = f(sigma, mat_l, mat_r)
    u, d, v = sp.linalg.svd(mat)
    u0 = u[:, 0]
    v0 = v[0, :]  # right singular vectors are rows
    grad_sn = np.outer(u0, v0)

    grad = np.zeros(len(sigma))
    for k in range(grad.shape[0]):
        grad[k] = (np.outer(mat_l[:, k], mat_r[k, :]) * grad_sn).sum()
    return grad


def optim_approx(mat_l, mat_r, verbose=True):
    """ Return approximation of the following optimization problem:

    max | U d(sigma) V|
    where |.| is the spectral norm, with sigma being in the cube [0, 1]

    Note that it is a maximization of a convex function with constraints.

    TODO: change name
    """
    n = mat_l.shape[1]
    fun = lambda s: -f_spec(s, mat_l, mat_r)
    f_grad = lambda s: -f_spec_grad(s, mat_l, mat_r)

    bounds = [(0, 1)] * n
    x0 = rd.rand(n)
    options = {'disp': verbose,
               'maxcor': 20,
               'maxfun': 1e6}

    res = minimize(fun=fun,
                   x0=x0,
                   jac=f_grad,
                   method='L-BFGS-B',
                   bounds=bounds,
                   options=options)
    return -res.fun, res.x.astype(np.int)

def optim_greedy(mat_l, mat_r, verbose=True):
    """ Greedy algorithm to perform the following optimization problem:

    max | U d(sigma) V|
    where |.| is the spectral norm, with sigma being in the cube [0, 1]
    """
    from tqdm import tqdm
    n = mat_l.shape[1]
    sigma = np.ones(n, dtype=np.int)
    stop_criterion = False
    current_spec  = f_spec(sigma, mat_l, mat_r)

    highest_loop = current_spec
    it = 0
    while not stop_criterion:
        it += 1
        previous = highest_loop
        highest_idx = -1
        for i in range(n):
            change = 1 - sigma[i] # if 1 then 0, if 0 then 1
            sigma[i] = change
            spec = f_spec(sigma, mat_l, mat_r)
            if highest_loop < spec:
                highest_loop = spec
                highest_idx = i
                current_spec = spec
            sigma[i] = 1 - change # change back

        if highest_idx < 0:
            stop_criterion = True
        else:
            sigma[highest_idx] = 1 - sigma[highest_idx]
            if verbose:
                sign_change = '+' if sigma[highest_idx] > 0.5 else '-'
                print('[{}] {} Best at position {}:  {:.4f} > {:.4f}'.format(
                    it,
                    sign_change,
                    highest_idx,
                    highest_loop,
                    previous))
    return current_spec, sigma


def optim_nn_greedy(f_l, f_r, input_size, use_cuda=False, max_iter=200, verbose=True):
    """ Greedy algorithm to perform the following optimization problem:

    INPUT:
        * `f_l` linear operator
        * `f_r` linear operator
        * `input_size` size of the input

    max | f_l d(sigma) f_r|
    where |.| is the spectral norm, with sigma being in the cube [0, 1]
    and A_1 and A_2 linear operators defined by a neural network.
    """
    import torch
    from max_eigenvalue import generic_power_method
    from tqdm import tqdm
    x = torch.randn(input_size)
    if use_cuda:
        x = x.cuda()
    sigma = torch.ones(f_r(x).size())
    if use_cuda:
        sigma = sigma.cuda()
    sigma_flat = sigma.view(-1)  # new tensor with same data
    stop_criterion = False

    def spectral_norm(sigma, f_l, f_r):
        ''' Return spectral norm sith specified `sigma` '''
        s, _, _ = generic_power_method(lambda x: f_l(f_r(x) * sigma),
                input_size=input_size,
                max_iter=max_iter,
                use_cuda=use_cuda)
        return s.data[0]
    current_spec  = spectral_norm(sigma, f_l, f_r)

    highest_loop = current_spec
    highest_idx = -1
    it = 0
    while not stop_criterion:
        it += 1
        previous = highest_loop
        highest_idx = -1
        for i in tqdm(range(sigma_flat.size()[0])):
            change = 1 - sigma_flat[i] # if 1 then 0, if 0 then 1
            sigma_flat[i] = change
            spec = spectral_norm(sigma, f_l, f_r)
            if highest_loop < spec:
                highest_loop = spec
                highest_idx = i
                current_spec = spec
            sigma_flat[i] = 1 - change

        if highest_idx == -1:
            stop_criterion = True
        else:
            sigma_flat[highest_idx] = 1 - sigma_flat[highest_idx]
            if verbose:
                sign_change = '+' if sigma[highest_idx] > 0.5 else '-'
                print('[{}] {} Best at position {}:  {:.4f} > {:.4f}'.format(
                    it,
                    sign_change,
                    highest_idx,
                    highest_loop,
                    previous))
    return current_spec, sigma


def optim_nn_pca_greedy(U, V, max_iteration=10, verbose=True):
    """ U is k x n and V is n x k

    Goal of this optimisation method is to get an approximation of the upper
    bound using only a few of the singular vectors associated to the highest
    singular values.
    """
    from tqdm import tqdm
    k = U.shape[0]
    n = U.shape[1]

    sigma = np.ones(n)
    M = torch.mm(U, V)
    current_spec = sp.linalg.norm(M, 2)
    stop_criterion = False
    it = 0
    while not stop_criterion:
        it += 1
        n_changes = 0
        n_changes_p = 0
        n_changes_n = 0
        previous = current_spec
        highest_idx = -1
        for i in tqdm(range(len(sigma))):
            change = 1 - sigma[i] # if 1 then 0, if 0 then 1
            m_change = torch.ger(U[:, i], V[i, :])
            tmpM = M + (2 * change - 1) * m_change
            spec = sp.linalg.norm(tmpM, 2)
            if current_spec < spec:
                highest_idx = i
                current_spec = spec
                M = tmpM
                n_changes += 1
                if change > 0.5:
                    n_changes_p += 1
                else:
                    n_changes_n += 1
                sigma[i] = change

        if verbose:
            print('[{}] {} updates: + {}, - {} | {:.4f} > {:.4f}'.format(
                it,
                n_changes,
                n_changes_p,
                n_changes_n,
                current_spec,
                previous))

        if it > max_iteration or highest_idx == -1:
            stop_criterion = True
    return current_spec, sigma
