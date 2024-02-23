# coding=utf-8
# taken from https://github.com/avirmaux/lipEstimation
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def generic_power_method(affine_fun, input_size, eps=1e-8,
                         max_iter=500, use_cuda=False):
    """ Return the highest singular value of the linear part of
    `affine_fun` and it's associated left / right singular vectors.

    INPUT:
        * `affine_fun`: an affine function
        * `input_size`: size of the input
        * `eps`: stop condition for power iteration
        * `max_iter`: maximum number of iterations
        * `use_cuda`: set to True if CUDA is present

    OUTPUT:
        * `eigenvalue`: maximum singular value of `affine_fun`
        * `v`: the associated left singular vector
        * `u`: the associated right singular vector

    NOTE:
        This algorithm is not deterministic, depending of the random
        initialisation, the returned eigenvectors are defined up to the sign.

        If affine_fun is a PyTorch model, beware of setting to `False` all
        parameters.requires_grad.

    TEST::
        >>> conv = nn.Conv2d(3, 8, 5)
        >>> for p in conv.parameters(): p.requires_grad = False
        >>> s, u, v = generic_power_method(conv, [1, 3, 28, 28])
        >>> bias = conv(torch.zeros([1, 3, 28, 28]))
        >>> linear_fun = lambda x: conv(x) - bias
        >>> torch.norm(linear_fun(v) - s * u) # should be very small

    TODO: more tests with CUDA
    """
    zeros = torch.zeros(input_size)
    if use_cuda:
        zeros = zeros.cuda()
    bias = affine_fun(Variable(zeros))
    linear_fun = lambda x: affine_fun(x) - bias

    def norm(x, p=2):
        """ Norm for each batch

        FIXME: avoid it?
        """
        norms = Variable(torch.zeros(x.shape[0]))
        if use_cuda:
            norms = norms.cuda()
        for i in range(x.shape[0]):
            norms[i] = x[i].norm(p=p)
        return norms

    # Initialise with random values
    v = torch.randn(input_size)
    v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
    if use_cuda:
        v = v.cuda()

    stop_criterion = False
    it = 0
    while not stop_criterion:
        previous = v
        v = _norm_gradient_sq(linear_fun, v)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    # Compute Rayleigh product to get eivenvalue
    u = linear_fun(Variable(v))  # unormalized left singular vector
    eigenvalue = norm(u)
    u = u.div(eigenvalue)
    return eigenvalue, u, v


def _norm_gradient_sq(linear_fun, v):
    v = Variable(v, requires_grad=True)
    loss = torch.norm(linear_fun(v))**2
    loss.backward(retain_graph=True)
    return v.grad.data


def k_generic_power_method(affine_fun, input_size, n_singular_values, eps=1e-8,
                           max_iter=500, use_cuda=False, verbose=True):
    """ Return the k highest eigenvalues of the linear part of
    `affine_fun` and it's left / right associated singular vectors.

    INPUT:
        * `affine_fun`: an affine function
        * `input_size`: size of the input
        * `n_singular_values`: number singular values to compute
        * `eps`: stop condition for power iteration
        * `max_iter`: maximum number of iterations
        * `use_cuda`: set to True if CUDA is present

    OUTPUT:
        * `eigenvalue`: maximum eigenvalue of `model`
        * `v`: the associated left singular vector
        * `u`: the associated right singular vector

    NOTE:
        This algorithm is not deterministic, depending of the random
        initialisation, the eigenvectors are defined up to the sign.

        If affine_fun is a PyTorch model, beware of setting to `False` all
        parameters.requires_grad.

    TODO: more tests with CUDA
    """
    if verbose:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x
    zeros = torch.zeros(input_size)
    if use_cuda:
        zeros = zeros.cuda()
    bias = affine_fun(Variable(zeros))

    right_list = []
    left_list = []
    sing_list = []

    # List of k functions
    funcs = [lambda x, i: x]
    linear_fun = lambda x, i: affine_fun(funcs[i](x, i)) - bias

    def norm(x, p=2):
        """ Norm for each batch

        FIXME: avoid it?
        """
        norms = Variable(torch.zeros(x.shape[0]))
        for i in range(x.shape[0]):
            norms[i] = x[i].norm(p=p)
        return norms

    for i in tqdm(range(n_singular_values)):
        # Initialise with random values
        v = torch.randn(input_size)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
        if use_cuda:
            v = v.cuda()

        stop_criterion = False
        it = 0
        while not stop_criterion:
            previous = v
            v = _k_norm_gradient_sq(linear_fun, v, i)
            v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
            stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
            it += 1
        # Found highest singular value
        u = linear_fun(Variable(v), i)  # unormalized left singular vector
        
        if use_cuda:
            u = u.cpu()
        eigenvalue = norm(u).cpu()
        u = u.div(eigenvalue)

        left_list.append(u)
        right_list.append(v)
        sing_list.append(eigenvalue)

        def fun(x, i):
            t = funcs[i](x, i-1)
            return t - (t.data * right_list[i-1]).sum() * Variable(right_list[i-1])
        funcs.append(fun)
    return sing_list, left_list, right_list


def _k_norm_gradient_sq(linear_fun, v, i):
    v = Variable(v, requires_grad=True)
    loss = torch.norm(linear_fun(v, i))**2
    loss.backward()
    return v.grad.data


def max_eigenvalue(module):
    return generic_power_method(module, module.input_sizes[0])


def _power_method_matrix(matrix, eps=1e-6, max_iter=300, use_cuda=False):
    """ Return square of maximal singular value of `matrix`
    """
    M = matrix.t() @ matrix
    v = torch.randn(M.shape[1], 1)
    stop_criterion = False
    it = 0
    while not stop_criterion:
        previous = v
        v = M @ v
        v = v / torch.norm(v)
        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    # Compute Rayleigh product to get eivenvalue
    eigenvalue = torch.norm(matrix @ v)
    return eigenvalue, v


def lipschitz_bn(bn_layer):
    """ Return Lipschitz constant of BatchNormalization

    Note that as an affine transformation, one could use
    `generic_power_method`. This is much faster.
    """
    return max(abs(bn_layer.weight / torch.sqrt(bn_layer.running_var + bn_layer.eps)))
