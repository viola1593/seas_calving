# tests/test_rollout_records.py
"""
Unit tests for the record/output functions of rollout_film.py (section 2d).
"""

from datetime import date

import pytest

from rollout_film import build_record, records_to_dataframe, reaches_full_depth


LEADING_COLS = ["glacier_id", "step", "autoreg_depth", "lead_time_days", "target_date"]
TRAILING_COLS = ["n_frames_synthetic", "borrowed_dates", "max_date_offset"]


def _metrics(mde, hausdorff):
    return {"active_mde": mde, "active_hausdorff": hausdorff}


def _make_record(glacier_id="G01", step=0, depth=0, lead=90, td=date(2020, 6, 1)):
    return build_record(
        glacier_id=glacier_id,
        step=step,
        autoreg_depth=depth,
        lead_time_days=lead,
        target_date=td,
        model_m=_metrics(10.0, 20.0),
        teacher_m=_metrics(11.0, 21.0),
        persist_m=_metrics(30.0, 40.0),
        n_frames_synthetic=depth,
        borrowed_dates=[date(2020, 1, 1)],
        max_date_offset=5,
    )


# ── build_record ──────────────────────────────────────────────────────────

def test_build_record_has_leading_and_trailing_keys():
    rec = _make_record()
    for col in LEADING_COLS + TRAILING_COLS:
        assert col in rec, f"missing key {col}"
    assert rec["glacier_id"] == "G01"
    assert rec["autoreg_depth"] == 0
    assert rec["lead_time_days"] == 90


def test_build_record_flattens_metric_dicts_with_prefixes():
    rec = _make_record()
    assert rec["model_active_mde"] == 10.0
    assert rec["model_active_hausdorff"] == 20.0
    assert rec["teacher_active_mde"] == 11.0
    assert rec["teacher_active_hausdorff"] == 21.0
    assert rec["persist_active_mde"] == 30.0
    assert rec["persist_active_hausdorff"] == 40.0
    # unprefixed metric names must not leak into the flat record
    assert "active_mde" not in rec


# ── records_to_dataframe ──────────────────────────────────────────────────

def test_dataframe_stable_and_ordered_columns():
    recs = [_make_record(step=0, depth=0), _make_record(step=1, depth=1, lead=180)]
    df = records_to_dataframe(recs)

    cols = list(df.columns)
    # leading identity columns in fixed order
    assert cols[: len(LEADING_COLS)] == LEADING_COLS
    # trailing bookkeeping columns in fixed order
    assert cols[-len(TRAILING_COLS):] == TRAILING_COLS
    # columns cover exactly the record keys
    assert set(cols) == set(recs[0].keys())
    # model_ before teacher_ before persist_
    assert cols.index("model_active_mde") < cols.index("teacher_active_mde")
    assert cols.index("teacher_active_mde") < cols.index("persist_active_mde")
    assert len(df) == 2


def test_dataframe_column_order_is_deterministic():
    recs = [_make_record(step=s) for s in range(3)]
    a = list(records_to_dataframe(recs).columns)
    b = list(records_to_dataframe(recs).columns)
    assert a == b


# ── headline feasibility filter (uses frozen reaches_full_depth) ──────────

def test_feasibility_filter_keeps_only_full_depth_glaciers():
    K = 5
    records = (
        [_make_record(glacier_id="A", step=s, depth=d) for s, d in enumerate([0, 3, 5])]
        + [_make_record(glacier_id="B", step=s, depth=d) for s, d in enumerate([0, 2, 4])]
    )
    df = records_to_dataframe(records)

    max_depth = df.groupby("glacier_id")["autoreg_depth"].max()
    headline = {g for g, d in max_depth.items() if reaches_full_depth(int(d), K)}

    assert headline == {"A"}          # A reaches depth 5 == K; B tops out at 4