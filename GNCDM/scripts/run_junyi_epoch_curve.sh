#!/usr/bin/env bash
# Junyi random_split epoch curves -> one combined ACC_new|ACC_old figure.
# Mirror math1 final_ep25: default 25 epochs.
#
#   cd GNCDM
#   bash scripts/run_junyi_epoch_curve.sh
#   bash scripts/run_junyi_epoch_curve.sh 25 cuda:0
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${1:-25}"
DEVICE="${2:-cuda:0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE#cuda:}}"

echo "[1/3] GNCDM (CLEAN-Full / Full-Replay / X-DER / C-LoRA-GNCDM) ep=${EPOCHS}"
python plot/plot_epoch_curve_gncdm_junyi.py --epochs "$EPOCHS"

echo "[2/3] Avalanche (EWC / DER++)"
python plot/plot_epoch_curve_avalanche_junyi.py --epochs "$EPOCHS"

echo "[3/3] Merge panels -> one figure"
python plot/plot_epoch_curve_final_junyi.py --epochs "$EPOCHS"

echo "Done: incremental_result/epoch_curve_junyi_random_split_final_ep${EPOCHS}.png"
