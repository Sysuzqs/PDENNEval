# coding=utf-8
# taken from https://github.com/avirmaux/lipEstimation
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from max_eigenvalue import max_eigenvalue, generic_power_method


def compute_module_input_sizes(model, input_size):
    """ Tag all modules with `input_sizes` attribute
    """
    def store_input_sizes(self, input, output):
        # print(self.__class__.__name__, len(input), input[0].shape)
        self.input_sizes = [x.size() for x in input]
    execute_through_model(store_input_sizes, model, input_size=input_size)


def execute_through_model(function, model, input_size=None, backwards=False):
    """ Execute `function` through the model
    """
    use_cuda = next(model.parameters()).is_cuda
    if input_size is None:
        input_size = model.input_sizes[0]
    handles = []
    for module in model.modules():
        if backwards:
            handle = module.register_backward_hook(function)
        else:
            handle = module.register_forward_hook(function)
        handles.append(handle)

    if backwards:
        input = Variable(torch.zeros(*input_size), requires_grad=True)
        if use_cuda:
            input = input.cuda()
        f = model(input).norm()
        f.backward()
    else:
        input = Variable(torch.zeros(*input_size))
        if use_cuda:
            input = input.cuda()
        model(input)

    # Remove hooks
    for handle in handles:
        handle.remove()


def get_input_size(data, batch_size=1):
    data_load = torch.utils.data.DataLoader(data, batch_size=batch_size)
    for batch_idx, (one_image, _) in enumerate(data_load):
        return one_image.size()


def gradient_norm(model, v, requires_grad=True):
    """ Return the gradient of `model` at point `v`

    FIXME: make requires_grad attribute usefull
    """
    use_cuda = next(model.parameters()).is_cuda

    v.requires_grad = False
    def norm_diff(e):
        return model(v + e * 1e-5) * 1e5

    s, _, _ = generic_power_method(norm_diff, v.size(), use_cuda=use_cuda)
    return s


def jacobian(model, v, requires_grad=True):
    """ Return the jacobian of `model` at point `v`

    The model should return a matrix of shape [batch_size, dim_output].
    The jacobian is of shape [dim_input, dmi_output].

    ALGORITHM:
        We stack the gradients after composing by every evaluation function in
        all dimensions of the output.
    """
    use_cuda = next(model.parameters()).is_cuda
    dim_input = v.view(1, -1).shape[1]
    if str(type(v)).find('Variable') == -1:
        v = Variable(v, requires_grad=True)
    if len(v.size()) == 1:
        v = v.unsqueeze(0) # Make batch of size 1
    f = model(v)
    # f is of shape [1, dim_output]
    dim_output = f.shape[1]
    if use_cuda:
        grad_outputs = grad_outputs.cuda()

    jacobian = torch.zeros(dim_input, dim_output)
    for i in range(dim_output):
        grad_i = autograd.grad(outputs=f[0, i], inputs=v,
                               grad_outputs=torch.Tensor([1]),
                               create_graph=False,
                               retain_graph=True,
                               only_inputs=False)[0]
        grad = grad_i.data.view(-1)
        jacobian[:, i] = grad
    return jacobian


def spectral_jacobian(model, v):
    """ Return a spectral norms for every element in the batch

    v is of shape [batch, dim_input]
    """
    from max_eigenvalue import _power_method_matrix
    spectral_batch = torch.zeros(v.size(0))
    for i in range(v.size(0)):
        jac = jacobian(model, v[i, :])
        spectral_batch[i] = _power_method_matrix(jac)[0]
    return spectral_batch


def is_convolution_or_linear(layer):
    """ Return True if `layer` is a convolution or a linear layer
    """
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        return True
    if classname.find('Linear') != -1:
        return True
    return False


def is_batch_norm(layer):
    """ Return True if `layer` is a batch normalisation layer
    """
    classname = layer.__class__.__name__
    return classname.find('BatchNorm') != -1


def random_dataset(input_size, dataset_size, scale=1, fn=lambda x: x):
    tensor_dataset = scale * torch.randn(dataset_size, *input_size)
    return torch.utils.data.TensorDataset(tensor_dataset, fn(tensor_dataset))


def spectral_norm_module(module):
    """ Compute the spectral norm of a module and write is as as attribute.

    EXAMPLE:

        >>> net.apply(spectral_norm_module)

    """
    if not is_convolution_or_linear(module):
        return
    else:
        s, u, v = max_eigenvalue(module)
        module.spectral_norm = s
        module.left_singular = u
        module.right_singular = v


def unset_grad(model):
    """ Set `requires_grad` attribute of all parameters to False
    """
    for p in model.parameters():
        p.requires_grad = False
