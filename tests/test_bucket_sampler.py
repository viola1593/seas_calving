# tests/test_bucket_sampler.py

import numpy as np
import pytest
from data.bucket_sampler import SizeBucketedBatchSampler


def make_sizes(n=100, seed=0):
    """Synthetic (H, W) with the same 25x area spread as the real glaciers."""
    rng = np.random.default_rng(seed)
    h = rng.integers(80, 700, size=n)
    w = rng.integers(80, 1050, size=n)
    return [(int(a), int(b)) for a, b in zip(h, w)]


def distinct_sizes(n=100):
    """Sizes with strictly distinct areas, so the area sort is unambiguous."""
    return [(1, 10 * (i + 1)) for i in range(n)]


def epoch(sampler):
    return [list(b) for b in sampler]


# ── Test 1: an epoch is a partition of range(n) ──────────────────────────────

def test_epoch_covers_every_index_once():
    sampler = SizeBucketedBatchSampler(make_sizes(100), batch_size=2)
    for _ in range(3):
        batches = epoch(sampler)
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(100))
        assert len(batches) == len(sampler)


# ── Test 2: __len__ with a ragged final batch ────────────────────────────────

@pytest.mark.parametrize("n, bs, expected", [(100, 2, 50), (101, 2, 51), (100, 3, 34)])
def test_len_with_ragged_final_batch(n, bs, expected):
    sampler = SizeBucketedBatchSampler(make_sizes(n), batch_size=bs)
    assert len(sampler) == expected
    batches = epoch(sampler)
    assert len(batches) == expected
    assert sorted(len(b) for b in batches)[-1] == bs


# ── Test 3: epochs differ, seeds reproduce ───────────────────────────────────

def test_epochs_differ_but_seed_reproduces():
    sizes = make_sizes(100)
    a = SizeBucketedBatchSampler(sizes, batch_size=2, seed=7)
    e0, e1 = epoch(a), epoch(a)
    assert e0 != e1
    assert sorted(map(sorted, e0)) != sorted(map(sorted, e1))  # composition, not just order

    b = SizeBucketedBatchSampler(sizes, batch_size=2, seed=7)
    assert epoch(b) == e0
    assert epoch(b) == e1


# ── Test 4: the window_batches extremes ──────────────────────────────────────

def test_window_1_is_a_strict_area_sort():
    sizes = distinct_sizes(60)
    sampler = SizeBucketedBatchSampler(sizes, batch_size=3, window_batches=1)
    order = np.argsort([h * w for h, w in sizes], kind="stable")
    expected = {tuple(sorted(order[i : i + 3].tolist())) for i in range(0, 60, 3)}
    assert {tuple(sorted(b)) for b in epoch(sampler)} == expected


def test_window_larger_than_epoch_still_covers():
    sampler = SizeBucketedBatchSampler(make_sizes(100), batch_size=2, window_batches=1000)
    flat = [i for b in epoch(sampler) for i in b]
    assert sorted(flat) == list(range(100))


# ── Test 5: shuffle=False is deterministic ───────────────────────────────────

def test_no_shuffle_is_deterministic():
    sizes = make_sizes(50)
    a = SizeBucketedBatchSampler(sizes, batch_size=4, shuffle=False)
    b = SizeBucketedBatchSampler(sizes, batch_size=4, shuffle=False)
    assert epoch(a) == epoch(a)
    assert epoch(a) == epoch(b)

    areas = [max(sizes[i][0] * sizes[i][1] for i in batch) for batch in epoch(a)]
    assert areas == sorted(areas)   # still bucketed


# ── Test 6: fewer samples than batch_size ────────────────────────────────────

def test_fewer_samples_than_batch_size():
    sampler = SizeBucketedBatchSampler(make_sizes(3), batch_size=8)
    assert len(sampler) == 1
    batches = epoch(sampler)
    assert len(batches) == 1
    assert sorted(batches[0]) == [0, 1, 2]
