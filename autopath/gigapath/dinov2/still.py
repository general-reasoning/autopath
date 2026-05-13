# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import argparse
from dataclasses import dataclass, field, asdict
from functools import partial
import logging
import math
import os
import sys
from typing import Callable, Optional

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch
from torch.utils.tensorboard import SummaryWriter


from dinov2.data import SamplerType
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator, SamplerType, make_data_loader
from dinov2 import distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.utils import CosineScheduler

import dbx
from dbx import Datablock

from autopath.pancan.tiles import PancanTileSet
from .backbone import gigapath_tile_backbone
from .backbone import GigapathVisionTransformer
from .ssl import SSL
from . import distributed


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def default(x):
    return field(default_factory=x)

@dataclass 
class BLOCK_PRECISION:
    param_dtype: str = "fp16"
    reduce_dtype: str = "fp16"
    buffer_dtype: str = "fp32"

@dataclass
class BLOCK:
    sharding_strategy: str = "SHARD_GRAD_OP"
    mixed_precision: BLOCK_PRECISION = default(BLOCK_PRECISION)

@dataclass
class BRANCH:
    backbone: BLOCK = default(BLOCK)
    dino_head: BLOCK = default(BLOCK)
    ibot_head: BLOCK = default(BLOCK)

@dataclass   
class COMPUTE_PRECISION:
    grad_scaler: bool = True
    teacher: BRANCH = BRANCH(
        backbone=BLOCK(
            sharding_strategy="SHARD_GRAD_OP",
            mixed_precision=BLOCK_PRECISION(
                param_dtype="fp16",
                reduce_dtype="fp16",
                buffer_dtype="fp32",
            )
        ),
        dino_head=BLOCK(
            sharding_strategy="SHARD_GRAD_OP",
            mixed_precision=BLOCK_PRECISION(
                param_dtype="fp16",
                reduce_dtype="fp16",
                buffer_dtype="fp32",
            )
        ),
        ibot_head=BLOCK(
            sharding_strategy="SHARD_GRAD_OP",
            mixed_precision=BLOCK_PRECISION(
                param_dtype="fp16",
                reduce_dtype="fp16",
                buffer_dtype="fp32",
            )
        ),
    )
    student: BRANCH = BRANCH(
        backbone=BLOCK(
            sharding_strategy="SHARD_GRAD_OP",
            mixed_precision=BLOCK_PRECISION(
                param_dtype="fp16",
                reduce_dtype="fp16",
                buffer_dtype="fp32",
            )
        ),
        dino_head=BLOCK(
            mixed_precision=BLOCK_PRECISION(
                param_dtype="fp16",
                reduce_dtype="fp32",
                buffer_dtype="fp32",
            )
        ),
        ibot_head=BLOCK(
            mixed_precision=BLOCK_PRECISION(
                param_dtype="fp16",
                reduce_dtype="fp32",
                buffer_dtype="fp32",
            )
        ),
    )

@dataclass
class DINO:
    loss_weight: float = 1.0
    head_n_prototypes: int = 65536
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048
    koleo_loss_weight: float = 0.1

@dataclass
class IBOT:
    loss_weight: float = 1.0
    mask_sample_probability: float = 0.5
    mask_ratio_min_max: list[float] = field(default_factory=lambda: [0.1, 0.5])
    separate_head: bool = False
    head_n_prototypes: int = 65536
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048


@dataclass
class TRAIN:
    batch_size_per_gpu: int = 4
    shuffle: bool = True
    sampler_type: Optional[SamplerType] = SamplerType.SHARDED_INFINITE.value
    sampler_advance: int = 0
    start_iter: int = 0
    saveckp_freq: int = 20
    seed: int = 0
    OFFICIAL_EPOCH_LENGTH: int = 1250
    centering: str = "centering" # or "sinkhorn_knopp"
    tie_student_teacher_heads: bool = False
    freeze_backbone: bool = False

@dataclass
class STUDENT:
    drop_path_rate: float = 0.3
    drop_path_uniform: bool = True
    arch: str = "vit_large"
    patch_size: int = 16
    pretrained_weights: str = ""
    ffn_layer: str = "mlp"
    block_chunks: int = 0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    num_register_tokens: int = 0
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1

@dataclass
class TEACHER:
    momentum_teacher: float = 0.992
    final_momentum_teacher: float = 1
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30

@dataclass
class OPTIM:
    epochs: int = 100
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    base_lr: float = 0.004
    lr: float = 0.
    warmup_epochs: int = 10
    min_lr: float = 1.0e-06
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    scaling_rule: str = "sqrt_wrt_1024"
    patch_embed_lr_mult: float = 0.2
    layerwise_decay: float = 0.9
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

@dataclass
class CROPS:
    global_crops_scale: list[float] = field(default_factory=lambda: [0.32, 1.0])
    local_crops_number: int = 8
    local_crops_scale: list[float] = field(default_factory=lambda: [0.05, 0.32])
    global_crops_size: int = 224
    local_crops_size: int = 96

@dataclass
class EVALUATION:
    eval_period_iterations: int = 12500

@dataclass
class ARCH:
    compute_precision: COMPUTE_PRECISION = default(COMPUTE_PRECISION)
    dino: DINO = default(DINO)
    ibot: IBOT = default(IBOT)
    train: TRAIN = default(TRAIN)
    student: STUDENT = default(STUDENT)
    teacher: TEACHER = default(TEACHER)
    optim: OPTIM = default(OPTIM)
    crops: CROPS = default(CROPS)
    evaluation: EVALUATION = default(EVALUATION)

                
def dino_tile_dataloader(dataset, 
                        *, 
                        batch_size: int, 
                        num_workers: int,
                        shuffle: bool,
                        sampler_type: Optional[SamplerType],
                        sampler_advance: int,
                        start_iter: int,
                        log: dbx.Logger = dbx.Logger(),
) -> torch.utils.data.DataLoader:
    log.debug(f"{shuffle=}, {sampler_type=}, {sampler_advance=}, {start_iter=}")
    global_crops_size: int = 224
    student_patch_size: int = 16
    ibot_mask_ratio_min_max: list[float] = [0.1, 0.5]
    ibot_mask_sample_probability: float = 0.5
    collate_fn: Optional[Callable] = None
    drop_last: bool = True
    inputs_dtype: str = torch.half

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
            shuffle=shuffle,
            seed=start_iter,  # TODO: Fix this -- cfg.train.seed
            sampler_type=sampler_type,
            sampler_advance=sampler_advance,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    return data_loader


class GigaqStill(Datablock):
    VERSION = 1
    TOPICFILES = {'arch': 'arch.yaml', 'ckpts': None, 'tensorboard': None, 'training_metrics': None, 'breadcrumbs': 'breadcrumbs'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        arch: ARCH
        dataset: torch.utils.data.Dataset
        backbone: GigapathVisionTransformer = gigapath_tile_backbone()

    def __init__(self, *args, num_data_workers: int = 0, **kwargs):
        super().__init__(*args, num_data_workers=num_data_workers, **kwargs)

    def __build__(self):
        try:
            distributed.initialize()
            data_loader = dino_tile_dataloader(
                    self.config.dataset, 
                    batch_size=self.config.arch.train.batch_size_per_gpu,
                    shuffle=self.config.arch.train.shuffle,
                    sampler_type=SamplerType(self.config.arch.train.sampler_type),
                    sampler_advance=self.config.arch.train.sampler_advance,
                    start_iter=self.config.arch.train.start_iter,
                    num_workers=self.num_data_workers,
                    log=self.log,
            )    

            dbx.write_yaml(asdict(self.config.arch), self.path('arch', ensure_dirpath=True))
            model = SSL(self.config.arch).to(torch.device("cuda"))
            model.prepare_for_distributed_training()

            logger.info("Model:\n{}".format(model))

            writer = SummaryWriter(self.dirpath('tensorboard', ensure=True))
            model.train()
            inputs_dtype = torch.half
            fp16_scaler = model.fp16_scaler  # for mixed precision training

            # setup optimizer
            optimizer = self.build_optimizer(model.get_params_groups())
            (
                lr_schedule,
                wd_schedule,
                momentum_schedule,
                teacher_temp_schedule,
                last_layer_lr_schedule,
            ) = self.build_schedulers()

            # checkpointer
            checkpointer = FSDPCheckpointer(model, self.dirpath('ckpts'), optimizer=optimizer, save_to_disk=True)

            #start_iter = checkpointer.resume_or_load(self.config.arch.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
            start_iter = 0
            OFFICIAL_EPOCH_LENGTH = self.config.arch.train.OFFICIAL_EPOCH_LENGTH
            max_iter = self.config.arch.optim.epochs * OFFICIAL_EPOCH_LENGTH

            periodic_checkpointer = PeriodicCheckpointer(
                checkpointer,
                period=3 * OFFICIAL_EPOCH_LENGTH,
                max_iter=max_iter,
                max_to_keep=3,
            )

            # training loop

            iteration = start_iter

            logger.info("Starting training from iteration {}".format(start_iter))
            metrics_file = os.path.join(self.dirpath('training_metrics', ensure=True), 'training_metrics.json')
            metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
            header = "Training"
            for i, data in enumerate(metric_logger.log_every(
                data_loader,
                10,
                header,
                max_iter,
                start_iter,
            )):
                self.log.debug(f"{i}-th sample, iteration: {iteration}, max_iter: {max_iter}")
                current_batch_size = data["collated_global_crops"].shape[0] / 2
                if iteration > max_iter:
                    return

                # apply schedules

                lr = lr_schedule[iteration]
                wd = wd_schedule[iteration]
                mom = momentum_schedule[iteration]
                teacher_temp = teacher_temp_schedule[iteration]
                last_layer_lr = last_layer_lr_schedule[iteration]
                self.apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

                # compute losses

                optimizer.zero_grad(set_to_none=True)
                loss_dict, student_backbone_output = model.forward_backward(data, teacher_temp=teacher_temp)
                self.log.debug(f"loss_dict: keys: {loss_dict.keys()}")
                student_global_backbone_output, student_local_backbone_output = student_backbone_output
                self.log.debug(f"student_global_backbone_output: keys: {student_global_backbone_output.keys()}")
                self.log.debug(f"student_local_backbone_output: keys: {student_local_backbone_output.keys()}")

                # clip gradients

                if fp16_scaler is not None:
                    if self.config.arch.optim.clip_grad:
                        fp16_scaler.unscale_(optimizer)
                        for v in model.student.values():
                            v.clip_grad_norm_(self.config.arch.optim.clip_grad)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                else:
                    if self.config.arch.optim.clip_grad:
                        for v in model.student.values():
                            v.clip_grad_norm_(self.config.arch.optim.clip_grad)
                    optimizer.step()

                # perform teacher EMA update

                self.log.debug(f"Updating teacher: momentum: {mom}")
                model.update_teacher(mom)
                self.log.debug("Done updating teacher")

                # logging

                if distributed.get_global_size() > 1:
                    for v in loss_dict.values():
                        torch.distributed.all_reduce(v)
                loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

                if math.isnan(sum(loss_dict_reduced.values())):
                    logger.info("NaN detected")
                    raise AssertionError
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                metric_logger.update(lr=lr)
                metric_logger.update(wd=wd)
                metric_logger.update(mom=mom)
                metric_logger.update(last_layer_lr=last_layer_lr)
                metric_logger.update(current_batch_size=current_batch_size)
                metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
                writer.add_scalar("Loss/total", losses_reduced, iteration)
                for k, v in loss_dict.items():
                    writer.add_scalar(f"Loss/{k}", v, iteration)
                student_backbone_global_output, student_backbone_local_output = student_backbone_output
                #global_y =  student_backbone_global_output['x_prenorm']
                #local_y =  student_backbone_local_output['x_prenorm']
                #writer.add_image("student/global", global_y, global_step=iteration)
                #writer.add_image("student/local",  local_y, global_step=iteration)

                # checkpointing and testing

                if self.config.arch.evaluation.eval_period_iterations > 0 and (iteration + 1) % self.config.arch.evaluation.eval_period_iterations == 0:
                    self.checkpoint(self.config.arch, model, f"training_{iteration}")
                    torch.cuda.synchronize()
                periodic_checkpointer.step(iteration)

                iteration = iteration + 1
            metric_logger.synchronize_between_processes()
            #metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            
            self.leave_breadcrumbs_at_path(self.path('breadcrumbs'))
        finally:
            distributed.tear_down()
        return self

    def __read__(self, topic):
        if topic == 'arch':
            return dbx.read_yaml(self.path('arch'))
        else:
            raise ValueError(f"Unknown topic {topic}")

    def build_optimizer(self, params_groups):
        return torch.optim.AdamW(params_groups, betas=(self.config.arch.optim.adamw_beta1, self.config.arch.optim.adamw_beta2))

    def build_schedulers(self):
        OFFICIAL_EPOCH_LENGTH = self.config.arch.train.OFFICIAL_EPOCH_LENGTH
        lr = dict(
            base_value=self.config.arch.optim.lr,
            final_value=self.config.arch.optim.min_lr,
            total_iters=self.config.arch.optim.epochs * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=self.config.arch.optim.warmup_epochs * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=0,
        )
        wd = dict(
            base_value=self.config.arch.optim.weight_decay,
            final_value=self.config.arch.optim.weight_decay_end,
            total_iters=self.config.arch.optim.epochs * OFFICIAL_EPOCH_LENGTH,
        )
        momentum = dict(
            base_value=self.config.arch.teacher.momentum_teacher,
            final_value=self.config.arch.teacher.final_momentum_teacher,
            total_iters=self.config.arch.optim.epochs * OFFICIAL_EPOCH_LENGTH,
        )
        teacher_temp = dict(
            base_value=self.config.arch.teacher.teacher_temp,
            final_value=self.config.arch.teacher.teacher_temp,
            total_iters=self.config.arch.teacher.warmup_teacher_temp_epochs * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=self.config.arch.teacher.warmup_teacher_temp_epochs * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=self.config.arch.teacher.warmup_teacher_temp,
        )

        lr_schedule = CosineScheduler(**lr)
        wd_schedule = CosineScheduler(**wd)
        momentum_schedule = CosineScheduler(**momentum)
        teacher_temp_schedule = CosineScheduler(**teacher_temp)
        last_layer_lr_schedule = CosineScheduler(**lr)

        last_layer_lr_schedule.schedule[
            : self.config.arch.optim.freeze_last_layer_epochs * OFFICIAL_EPOCH_LENGTH
        ] = 0  # mimicking the original schedules

        logger.info("Schedulers ready.")

        return (
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            teacher_temp_schedule,
            last_layer_lr_schedule,
        )

    def apply_optim_scheduler(self, optimizer, lr, wd, last_layer_lr):
        for param_group in optimizer.param_groups:
            is_last_layer = param_group["is_last_layer"]
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            param_group["weight_decay"] = wd * wd_multiplier
            param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

    def checkpoint(self, model, iteration):
        new_state_dict = model.teacher.state_dict()

        if distributed.is_main_process():
            iterstring = str(iteration)
            eval_dir = os.path.join(self.config.arch.train.output_dir, "eval", iterstring)
            os.makedirs(eval_dir, exist_ok=True)
            # save teacher checkpoint
            teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
            torch.save({"teacher": new_state_dict}, teacher_ckp_path)
            
    