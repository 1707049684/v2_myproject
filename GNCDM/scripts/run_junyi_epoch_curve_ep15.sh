#!/usr/bin/env bash
# Thin wrapper: same as run_junyi_epoch_curve.sh with EPOCHS=15.
exec "$(dirname "$0")/run_junyi_epoch_curve.sh" 15 "${1:-cuda:0}"
