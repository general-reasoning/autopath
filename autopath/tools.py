import json
import os

import torch

from streaming.base.format.mds.reader import MDSReader


def mds_readers(shards_dir):
    """Return a list of :class:`MDSReader` for MDS shards in *shards_dir*.

    Each reader uses plain file I/O — no shared memory — making it
    safe for iterating over thousands of bags without leaking file
    descriptors.

    Parameters
    ----------
    shards_dir : str
        Path to the directory containing ``index.json`` and ``.mds``
        shard files.

    Returns
    -------
    list[MDSReader]
    """
    index_path = os.path.join(shards_dir, 'index.json')
    with open(index_path) as f:
        index = json.load(f)
    return [MDSReader.from_json(shards_dir, split=None, obj=s)
            for s in index['shards']]


def read_mds_samples(shards_dir, *, column=None):
    """Bulk-read samples from MDS shards using :class:`MDSReader`.

    Parameters
    ----------
    shards_dir : str
        Path to the MDS shards directory.
    column : str | None
        If given, yield only that column's value per sample.
        Otherwise yield the full sample dict.

    Yields
    ------
    value
        Column value or full sample dict.
    """
    for reader in mds_readers(shards_dir):
        for i in range(len(reader)):
            sample = reader.get_item(i)
            yield sample[column] if column else sample


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
