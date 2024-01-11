import torch

def MSE(pred, target):
    """return mean square error

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    res = torch.mean(errors**2, dim=2)
    return res # (bs, v)


def RMSE(pred, target):
    """return root mean square error

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    return torch.sqrt(MSE(pred, target)) # (bs, v)


def L2RE(pred, target):
    """l2 relative error (nMSE in PDEBench)

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
    return torch.sqrt(res) # (bs, v)

def MaxError(pred, target):
    """return max error in a batch

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    errors = torch.abs(pred - target)
    nc = errors.shape[-1]
    res, _ = torch.max(errors.reshape([-1, nc]), dim=0) # retain the last dim
    return res # (v)