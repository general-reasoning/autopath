import warnings

# dinov2's layer modules emit UserWarning at import time to announce whether
# xFormers is available.  These fire in every fresh worker process (Ray,
# multiprocessing spawn/forkserver) and clutter the output.  Suppress them
# here, before any import can pull in dinov2, so the filter is always active.
warnings.filterwarnings('ignore', message=r'xFormers is (available|not available|disabled)',
                        category=UserWarning)

import torch

import dbx

# Register safe globals for torch.load weights_only=True (PyTorch 2.6+)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        dbx.Logger, 
        dbx.Datablock, 
        dbx.Datablock.Bid
    ])
