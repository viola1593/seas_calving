# tests/test_rollout_smoke.py
"""
Smoke test for the rollout engine of rollout_film.py (section 2e).

This is the ONE integration test. It touches GPU + zarr + the seed3 checkpoint,
so it is opt-in and excluded from the default `pytest tests/` run:

  * skipped unless the paper_M3_seed3 checkpoint exists, AND
  * skipped unless the env var ROLLOUT_SMOKE=1 is set (guards GPU use).

Run it on the cluster with a GPU, in the conda env (zarr importable):

  CUDA_VISIBLE_DEVICES=<gpu> ROLLOUT_SMOKE=1 \
      python -m pytest tests/test_rollout_smoke.py -v -s
"""

import os
from pathlib import Path

import numpy as np
import pytest

CKPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "seasonal" / "paper_M3_seed3"

pytestmark = [
    pytest.mark.skipif(
        not CKPT_DIR.exists(),
        reason=f"seed3 checkpoint not available at {CKPT_DIR}",
    ),
    pytest.mark.skipif(
        os.environ.get("ROLLOUT_SMOKE") != "1",
        reason="opt-in only: set ROLLOUT_SMOKE=1 (needs GPU + zarr) to run",
    ),
]


def test_smoke_rollout_one_feasible_glacier():
    """
    Load seed3, run one feasible glacier for a few steps, and assert:
      * CSV rows > 0 and required columns present,
      * autoreg_depth reaches K on >= 1 step,
      * lead_time_days is non-decreasing,
      * no NaN in model_active_mde at depth 0.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("zarr")

    from pathlib import Path
    from evaluate_film import find_best_checkpoint, load_module_and_config
    from data.dataset_v2 import CalvingFrontDatasetV2
    from rollout_film import (
        load_glacier_sample_indices,
        glacier_zarr_indices,
        gather_real_sdt_stack,
        compute_active_mask,
        rollout_glacier,
        records_to_dataframe,
        reaches_full_depth,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_best_checkpoint(CKPT_DIR, chkpt="best")
    module, cfg = load_module_and_config(ckpt, CKPT_DIR, device)
    module.eval()

    splits_dir = Path(cfg["data"]["splits_dir"])
    dataset = CalvingFrontDatasetV2(
        index_path=splits_dir / "test.json",
        config=cfg["data"]["config"],
        stats_path=cfg["data"]["stats_path"],
        augment=False,
    )

    # Pick the first glacier that reaches a fully-synthetic window.
    K = 5
    chosen = None
    for gid in sorted({dataset.samples[i]["glacier_id"] for i in range(len(dataset))}):
        indices = load_glacier_sample_indices(dataset, gid)
        if len(indices) >= K + 1:            # enough samples to fully detach
            chosen = (gid, indices)
            break
    assert chosen is not None, "no feasible glacier found in split"
    gid, indices = chosen

    store = dataset._get_store(dataset.samples[indices[0]]["zarr_path"])
    real_stack = gather_real_sdt_stack(store, glacier_zarr_indices(dataset, indices))
    frozen_mask = compute_active_mask(real_stack)

    records = rollout_glacier(module, dataset, gid, frozen_mask, delta_days=30)
    df = records_to_dataframe(records)

    # rows > 0
    assert len(df) > 0

    # required columns present
    required = {
        "glacier_id", "step", "autoreg_depth", "lead_time_days", "target_date",
        "model_active_mde", "n_frames_synthetic", "borrowed_dates", "max_date_offset",
    }
    assert required <= set(df.columns), f"missing: {required - set(df.columns)}"

    # autoreg_depth reaches K on at least one step
    assert reaches_full_depth(int(df["autoreg_depth"].max()), K)

    # lead_time_days non-decreasing
    lead = df["lead_time_days"].to_numpy()
    assert np.all(np.diff(lead) >= 0), "lead_time_days must be non-decreasing"

    # no NaN in model_active_mde at depth 0 (fully teacher-forced first step)
    depth0 = df[df["autoreg_depth"] == 0]["model_active_mde"].to_numpy()
    assert depth0.size > 0
    assert not np.isnan(depth0).any(), "depth-0 model_active_mde must be finite"