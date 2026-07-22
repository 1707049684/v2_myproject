#!/usr/bin/env bash
# Junyi random_split epoch curves (15 ep): ACC_new + ACC_old.
# Run on H100 server (needs torch+cuda and avalanche for EWC/DER++).
#
#   cd GNCDM
#   bash scripts/run_junyi_epoch_curve_ep15.sh
#   # or: bash scripts/run_junyi_epoch_curve_ep15.sh cuda:0
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-cuda:0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE#cuda:}}"
EPOCHS=15

echo "[1/3] GNCDM backbone curves (CLEAN-Full / Full-Replay / X-DER / C-LoRA-GNCDM)"
python plot/plot_epoch_curve_gncdm_junyi.py --epochs "$EPOCHS"

echo "[2/3] Avalanche curves (EWC / DER++)"
python plot/plot_epoch_curve_avalanche_junyi.py --epochs "$EPOCHS"

echo "[3/3] Merge + plot"
python plot/plot_epoch_curve_final_junyi.py --epochs "$EPOCHS"

echo "Done:"
echo "  incremental_result/epoch_curve_junyi_random_split_final_ep${EPOCHS}.png"
echo "  incremental_result/epoch_curve_junyi_random_split_final_old_ep${EPOCHS}.png"
