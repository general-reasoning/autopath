import torch.distributed
import dinov2.distributed
from dinov2.distributed import (
    get_global_size,
    get_global_rank,
    get_local_size,
    get_local_rank,
    is_enabled,
    is_main_process,
)

#TODO: spell out args and kwargs and emable multigpu/multinode training
#TODO: pass distributed initialization params through ssl.yml --> cfg
def initialize(*args, **kwargs):
    dinov2.distributed.enable(*args, **kwargs)

def tear_down():
    torch.distributed.destroy_process_group()
