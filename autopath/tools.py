import torch

def align_matrices_pairwise(x, y):
    _x = torch.tensor(x)
    _y = torch.tensor(y)
    X = _x.repeat_interleave(_y.shape[0], dim=0)
    Y = _y.repeat(_x.shape[0], 1)
    return X.numpy(), Y.numpy()
