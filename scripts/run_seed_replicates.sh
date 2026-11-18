#!/usr/bin/env bash
# run_seed_replicates.sh — sequential, resumable, background-safe seed-replicate driver.
#
# Trains 5 seeds each of M3 (FiLM per-var) and M1 (seasonal baseline) and populates
# every per-seed cache the cross-seed scripts consume, under the naming contract
#   experiments/seasonal/paper_{M3,M1}_seed{0..4}
#
# Launch (background, survives SSH drop):
#   CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_seed_replicates.sh > driver.out 2>&1 &
# or in a tmux session. Once masked, the GPU is logical index 0 inside the process.
#
# Scope / control (env overrides, no need to edit the script):
#   MODELS="M3"      SEEDS="0 1"   bash scripts/run_seed_replicates.sh   # subset / re-run one seed
#   DRY_RUN=1                      bash scripts/run_seed_replicates.sh   # print the full command list, run nothing
#
# Idempotent: every step drops a .done sentinel on success and is skipped on re-run,
# so an interrupted 3-day run resumes where it stopped.

set -uo pipefail

: "${CUDA_VISIBLE_DEVICES:?set the GPU, e.g. CUDA_VISIBLE_DEVICES=2 (masked -> logical index 0 inside)}"

# ── Config ───────────────────────────────────────────────────────────────────
read -r -a SEEDS  <<< "${SEEDS:-0 1 2 3 4}"
read -r -a MODELS <<< "${MODELS:-M3 M1}"
DRY_RUN="${DRY_RUN:-0}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repository root
OUT=experiments/seasonal
LOG="$ROOT/run_seed_replicates.log"

# Per-model config path + whether to run attribution (M1 is skill-only).
# out_* must match the config's experiment.output_dir (train_film.py writes there).
# M3nf is M2 in the paper (M3 without future forcing).
cfg_M3=configs/paper_M3.yaml;   attr_M3=1;   out_M3=$OUT
cfg_M1=configs/paper_M1.yaml;   attr_M1=0;   out_M1=$OUT
cfg_M3nf=configs/paper_M2.yaml; attr_M3nf=0; out_M3nf=$OUT/noforc


cd "$ROOT"

RESULTS=()   # "M seedS: STATUS" lines, printed as a table at the end

log() { echo "[$(date +%FT%T)] $*" | tee -a "$LOG"; }

print_summary() {
  echo
  echo "================ seed-replicate status ================" | tee -a "$LOG"
  local line
  for line in "${RESULTS[@]}"; do
    echo "  $line" | tee -a "$LOG"
  done
  echo "=======================================================" | tee -a "$LOG"
}
trap print_summary EXIT

# Courtesy GPU check: warn (don't abort) if the masked GPU already looks busy.
gpu_courtesy_check() {
  [[ "$DRY_RUN" == 1 ]] && return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local stat
  stat=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
           --query-gpu=memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null) || return 0
  log "GPU $CUDA_VISIBLE_DEVICES status (used, total, util): $stat"
  local used=${stat%% MiB*}
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 1000 )); then
    log "WARNING: GPU $CUDA_VISIBLE_DEVICES already using ${used} MiB — another user may be on it."
  fi
}

# step <name> -- <cmd...>
#   Skips if <EXP>/.done/<name>.done exists. Otherwise runs, tees to
#   driver_logs/<name>.log, and writes the sentinel only on success.
#   Relies on `set -e` + `pipefail` (both active inside do_seed) to abort the
#   seed if the command fails before the sentinel is written.
step() {
  local name=$1; shift
  local sentinel="$EXP/.done/$name.done"
  if [[ -f "$sentinel" ]]; then
    log "[SKIP] $name (sentinel present)"
    return 0
  fi
  log "[RUN ] $name: $*"
  [[ "$DRY_RUN" == 1 ]] && return 0
  "$@" 2>&1 | tee "$EXP/driver_logs/$name.log"
  touch "$sentinel"
}

# do_seed <model> <seed> — one full per-(model,seed) pipeline in a subshell so
# `set -e` is local: a failing step aborts this seed only, the outer loop continues.
do_seed() (
  set -e
  local M=$1 S=$2
  local cfg_var="cfg_${M}";  local cfg=${!cfg_var}
  local attr_var="attr_${M}"; local attr=${!attr_var}
  local out_var="out_${M}";   local out=${!out_var}
  EXP="$out/paper_${M}_seed${S}"
  mkdir -p "$EXP/driver_logs" "$EXP/.done"

  log ">>> $M seed $S | config=$cfg | attribution=$attr | dir=$EXP"

  # 1. Train (deterministic seed-addressable dir via --run_name).
  step train \
    python train_film.py --config "$cfg" --seed "$S" \
      --run_name "paper_${M}_seed${S}" --device 0

  # 2. Skill eval on the held-out-glacier test split.
  step eval \
    python evaluate_film.py --experiment "$EXP" --split test

  # 3. Attribution — M3 only, in strict order: input-ablation requires the
  #    climate-ablate cache first.
  if [[ "$attr" == 1 ]]; then

    step attr_input_importance \
      python -m model_analysis.aggregate_input_importance_mae       --experiment "$EXP"                      --all --device cuda

  fi

  touch "$EXP/.done/seed_complete.done"
)

# ── Main loop: model-outer, seed-inner (seed 0 first => primary-seed early) ──
log "start | MODELS=${MODELS[*]} | SEEDS=${SEEDS[*]} | DRY_RUN=$DRY_RUN | GPU=$CUDA_VISIBLE_DEVICES"
gpu_courtesy_check

for M in "${MODELS[@]}"; do
  for S in "${SEEDS[@]}"; do
    out_var="out_${M}"
    EXP="${!out_var}/paper_${M}_seed${S}"
    if [[ -f "$EXP/.done/seed_complete.done" ]]; then
      st=SKIPPED
    elif do_seed "$M" "$S"; then
      st=OK
    else
      st=FAILED
    fi
    log "paper_${M}_seed${S} -> $st"
    RESULTS+=("paper_${M}_seed${S}: $st")
  done
done

log "done"
