# data/bucket_sampler.py
"""
SizeBucketedBatchSampler — batch samples of similar spatial size together.

Glacier crops range from 83x81 to 706x1052 pixels, so a randomly drawn batch has
to be zero-padded up to its largest member: 24% of a batch is padding on average,
and that fraction swings from 0 to 0.97 batch to batch. This sampler sorts by
area before batching so neighbours are near-equal in size (padding fraction 0.24
-> 0.018 at the default window), while keeping every epoch different: it shuffles
within contiguous windows and shuffles the batch order globally.

Every index is yielded exactly once per epoch; only who is paired with whom changes.

DDP is not supported: with more than one device, Lightning's
`use_distributed_sampler=True` would wrap this batch sampler incorrectly. Training
runs on a single device (`devices: [0]`), so no sharding is implemented.
"""

from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class SizeBucketedBatchSampler(Sampler[list[int]]):
    """
    Args:
        sizes:          (H, W) per sample, in dataset index order.
        batch_size:     samples per batch; the final batch may be smaller.
        window_batches: shuffling window, in batches. 1 is a strict area sort
                        (minimum padding, minimum diversity); >= n/batch_size
                        degenerates to a uniform shuffle (today's behaviour).
        shuffle:        False (val/test) still buckets, but deterministically.
        seed:           epoch e uses `default_rng(seed + e)`.
    """

    def __init__(
        self,
        sizes:          list[tuple[int, int]],
        batch_size:     int,
        *,
        window_batches: int  = 16,
        shuffle:        bool = True,
        seed:           int  = 0,
    ):
        self.area           = np.array([h * w for h, w in sizes], dtype=np.int64)
        self.batch_size     = batch_size
        self.window_batches = window_batches
        self.shuffle        = shuffle
        self.seed           = seed
        self.epoch          = 0

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.area)
        if not self.shuffle:
            order = np.argsort(self.area, kind="stable")
            for i in range(0, n, self.batch_size):
                yield order[i : i + self.batch_size].tolist()
            return

        # A fresh generator per epoch: reproducible under num_workers > 0, where
        # global numpy state is not shared with the main process.
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        idx = rng.permutation(n)                                    # randomise ties
        idx = idx[np.argsort(self.area[idx], kind="stable")]

        window = self.window_batches * self.batch_size
        for i in range(0, n, window):
            rng.shuffle(idx[i : i + window])

        batches = [idx[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        # Shuffling batch order keeps consecutive micro-batches under
        # accumulate_grad_batches from all coming from one size regime.
        for i in rng.permutation(len(batches)):
            yield batches[i].tolist()

    def __len__(self) -> int:
        return (len(self.area) + self.batch_size - 1) // self.batch_size
