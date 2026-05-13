from functools import partial
from typing import Any, TypeVar, Optional, Callable, List


import torch

from dinov2.data import (
    SamplerType,
    make_data_loader, 
    collate_data_and_cast, 
    MaskingGenerator,
)

T = TypeVar("T")

def make_dataloader(
    dataset,
    *,
    global_crops_size: int = 224,
    student_patch_size: int = 16,
    ibot_mask_ratio_min_max: list[float] = [0.1, 0.5],
    ibot_mask_sample_probability: float = 0.5,
    batch_size: int = 4,
    num_workers: int = 1,
    start_iter: int = 0,
    shuffle: bool = True,
    sampler_type: Optional[SamplerType] = SamplerType.SHARDED_INFINITE,
    sampler_advance: int = 0,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    drop_last: bool = True,
    inputs_dtype: str = torch.half,
):
    img_size = global_crops_size
    patch_size = student_patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=ibot_mask_ratio_min_max,
        mask_probability=ibot_mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=sampler_advance,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return data_loader
