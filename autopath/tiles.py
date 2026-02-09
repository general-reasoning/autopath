
import functools
from typing import Optional

import torchvision


import dbx
from dbx import Logger

from autopath.databits import Shard, Bag, Clip, ClipDatasetBuilder


logger = Logger()


class TileShard(Shard):
    @functools.cached_property
    def tiles(self):
        return self.tensor
    
    @property
    def labels(self):
        raise NotImplementedError()
    

class TileBag(TileShard, Bag):
    def __init__(self, *args, **kwargs):
        TileShard.__init__(self, *args, **kwargs)


def tileset(tilebagclip: Clip, 
            transform: Optional[torchvision.transforms.Compose] = None,
            *,
            debug: bool = False,
            verbose: bool = False,
            log = None,
):
    tilebagclip = dbx.eval_term(tilebagclip)
    transform = dbx.eval_term(transform)
    return ClipDatasetBuilder(spec=dict(clip=tilebagclip, transform=transform,), 
                       debug=debug, 
                       verbose=verbose,
                       log=log,
    ).dataset()