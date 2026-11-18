# seascalving

Code for **Calving2DModel: seasonal (90-day) forecasting of 2D calving front positions** of Svalbard's marine-terminating glaciers without per-glacier calibration.

> Paper: Steidl, V., Li, T., Mitcham, T., Xiong, Z., Bamber, J., Zhu, X. X.: *Seasonal Forecasting of Marine-Terminating Glacier Calving Fronts in 2D With Deep Learning*, submitted.
> Data: *TODO Zenodo DOI*

The model forecasts calving front displacements as a signed distance field (SDF) of the calving front (SDF=0) over 90 days from five past front positions, geophysical fields (bed and surface elevation, thickness, surface ice speed, strain rate, bed slope) and scalar climate (MAR runoff, ocean temperature). Skill is measured against persistence. This work extends the dataset and preprocessing from CalvingDB (Li et al, 2026a; Li et al, 2026b) to the seasonal forecasting task. 


| Model | Code | Config |
|-------|------|--------|
| M3 — ConvLSTM + per-variable FiLM conditioning (Calving2DModel) | `models/convlstm_film_pervar.py` | `configs/paper_M3.yaml` |
| M1 — ConvLSTM, climate concatenated (architecture baseline) | `models/convlstm_seasonal.py` | `configs/paper_M1.yaml` |
| M2 — M3 without future forcing | `models/convlstm_film_pervar.py` | `configs/paper_M2.yaml` |

## References
Li, T., Steidl, V., Xiong, Z., Zhu, X. X., Mitcham, T., Cornford, S., & Bamber,
J. (2026a). CalvingDB: A Multi-Modal Benchmark for Marine-Terminating Glacier Calving Prediction in Svalbard. https://huggingface.co/datasets/enscg/calvdb.
Li, T., Steidl, V., Xiong, Z., Zhu, X. X., Mitcham, T., Cornford, S., & Bamber, J. (2026b). CalvingDB: A Multi-Modal Benchmark for Marine-Terminating Glacier Calving Prediction in Svalbard. https://github.com/icetianli/calvingdb.

## Setup

```bash
conda env create -f environment.yml
conda activate seascalving
```

All commands are run from the repository root.

## Data

Download the dataset (*TODO DOI*) and unpack it so the configs find it:

```
../glacier_data/TrainingDataset_processedV2/
    zarr/            one zarr store per glacier
    splits_seasonal/ train/val/test sample indices + normalisation_stats.json
```

`preprocessing/` rebuilds this from the raw calving front traces
(`00_create_glacier_mask.py` → `01_inventory.py` → `02_seasonal_observations.py` →
`03_seasonal_index.py` → `04_compute_stats.py`).

## Reproducing the paper

Train and evaluate five seeds of M3 and M1, including the per-seed input-ablation
caches (`MODELS`/`SEEDS` select a subset, `DRY_RUN=1` prints the commands):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_seed_replicates.sh
```

M2 is not in the default set; `MODELS=M3nf` trains it into
`experiments/seasonal/noforc/paper_M3nf_seed{0..4}`.

Single steps:

```bash
python train_film.py --config configs/paper_M3.yaml --seed 0 --run_name paper_M3_seed0 --device 0
python evaluate_film.py --experiment experiments/seasonal/paper_M3_seed0 --split test
python rollout_film.py --experiment experiments/seasonal/paper_M3_seed0 --split test
python rollout_centerline.py --experiment experiments/seasonal/paper_M3_seed0 --split test
python -m model_analysis.aggregate_input_importance_mae --experiment experiments/seasonal/paper_M3_seed0 --all
```

Cross-seed summaries and figures: `scripts/model_comparison/compare_models.py`,
`scripts/skill_robustness.py`, `scripts/plot_skill_per_glacier.py`,
`scripts/plot_skill_per_bin.py`, `scripts/linear_extrapolation_baseline.py`.

## Tests

```bash
python -m pytest tests
```

## Citation

*TODO BibTeX*

## License

Code: MIT (see `LICENSE`). The dataset has its own license, see its record.
