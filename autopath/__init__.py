import torch
import dbx

# Register safe globals for torch.load weights_only=True (PyTorch 2.6+)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        dbx.Logger, 
        dbx.Datablock, 
        dbx.IntRange, 
        dbx.FloatRange, 
        dbx.BoolRange, 
        dbx.Datablock.Bid
    ])
