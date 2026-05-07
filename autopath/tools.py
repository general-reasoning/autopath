import torch


def tensors_to_device(tensors, device, *, detach: bool = False):
    _tensors = {k: v.to(device) for k, v in tensors.items()}
    if detach:
        _tensors = {k: v.detach() for k, v in _tensors.items()} 
    return _tensors


def cat_tensor_dicts(tensor_dicts):
    tensors = {k: [] for k in tensor_dicts[0].keys()}
    for tensor_dict in tensor_dicts:
        for k, v in tensor_dict.items():
            tensors[k].append(v)
    _tensors = {k: torch.cat(v) for k, v in tensors.items()}
    return _tensors


def align_matrices_pairwise(x, y):
    _x = torch.tensor(x)
    _y = torch.tensor(y)
    X = _x.repeat_interleave(_y.shape[0], dim=0)
    Y = _y.repeat(_x.shape[0], 1)
    return X.numpy(), Y.numpy()
