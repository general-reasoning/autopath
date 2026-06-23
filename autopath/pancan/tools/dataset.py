import functools

import tqdm

import numpy as np
import torch

from dbx import Logger


#TODO: DEPRECATE? FACTOR THRU BlockDataset?
class BagDataset(torch.utils.data.Dataset):
    def __init__(self, bags, bag_lens=None, transform=None, *, verbose: bool = False, debug: bool = False, log = None):
        self.bags = bags
        self._bag_lens = bag_lens
        self.transform = transform
        self.target_transform = None
        self.verbose = verbose
        self.debug = debug
        
        if log is None:
            self.log = Logger(verbose=self.verbose, debug=self.debug)
        else:
            self.log = log

        self._n_bags = len(self.bags)
        self.log.debug(f"Building dataset out of {self._n_bags} bags")
        if self._bag_lens is None:
            if self.verbose:
                bagsitor = tqdm.tqdm(self.bags)
                self.log.verbose(f"Computing _bag_lens")
            else:
                bagsitor = self.bags
            self._bag_lens = [len(bag) for bag in bagsitor]
        self.log.debug(f"Computing _bag_bounds")
        self._bag_bounds = np.cumsum(self._bag_lens)
        self.log.debug(f"{self._n_bags=}, {self._bag_bounds=}")
        self._bag_idx = None
        self._bag = None
        self._bag_label = None
        self._bag_slide = None

    @functools.lru_cache(maxsize=3)
    def bag(self, bag_idx):
        if bag_idx != self._bag_idx:
            self._bag_idx = bag_idx
            self._bag = self.bags[self._bag_idx]
        return self._bag

    def __len__(self):
        return self._bag_bounds[-1]

    def __getitem__(self, index):
        bag_idx = np.searchsorted(self._bag_bounds, index, side='right')
        bag_lo = self._bag_bounds[bag_idx-1] if bag_idx > 0 else 0
        bag_hi = self._bag_bounds[bag_idx]
        bag_len = self._bag_lens[bag_idx]
        idx = index - bag_lo
        self.log.debug(f"__getitem__: {index=}, {bag_idx=}, {bag_lo=}, {bag_len=}, {bag_hi=}, {idx=}")
        tensor = self.bag(bag_idx).tensor
        sample = tensor[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        label = self.bag(bag_idx).label
        return sample, label


class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, blocks, block_lens=None, transform=None, *, verbose: bool = False, debug: bool = False, log = None):
        self.blocks = blocks
        self._block_lens = block_lens
        self.transform = transform
        self.target_transform = None
        self.verbose = verbose
        self.debug = debug
        
        if log is None:
            self.log = Logger(verbose=self.verbose, debug=self.debug)
        else:
            self.log = log

        self._n_blocks = len(self.blocks)
        self.log.debug(f"Building dataset out of {self._n_blocks} blocks")
        if self._block_lens is None:
            if self.verbose:
                blocksitor = tqdm.tqdm(self.blocks)
                self.log.verbose(f"Computing _block_lens")
            else:
                blocksitor = self.blocks
            self._block_lens = [len(block) for block in blocksitor]
        self.log.debug(f"Computing _block_bounds")
        self._block_bounds = np.cumsum(self._block_lens)
        self.log.debug(f"{self._n_blocks=}, {self._block_bounds=}")
        self._block_idx = None
        self._block = None
        self._block_label = None
        self._block_slide = None

    @functools.lru_cache(maxsize=3)
    def block(self, block_idx):
        if block_idx != self._block_idx:
            self._block_idx = block_idx
            self._block = self.blocks[self._block_idx]
        return self._block

    def __len__(self):
        return self._block_bounds[-1]

    def __getitem__(self, index):
        block_idx = np.searchsorted(self._block_bounds, index, side='right')
        block_lo = self._block_bounds[block_idx-1] if block_idx > 0 else 0
        block_hi = self._block_bounds[block_idx]
        block_len = self._block_lens[block_idx]
        idx = index - block_lo
        self.log.detailed(f"{index=}, {block_idx=}, {block_lo=}, {block_len=}, {block_hi=}, {idx=}")
        tensor = self.block(block_idx).tensor
        sample = tensor[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        labels = self.block(block_idx).labels
        label = labels[idx]
        return sample, label

