#!/usr/bin/env bash
# Extra five training seeds for a0910 user_split significance trials.
#
# Complements the original pack (seeds 1 7 21 42 84 under a0910_user/) with
# seeds 2 3 5 11 13, written to a separate output dir so the first pack is
# never overwritten. Support/query seed stays fixed at 7 (same as the original
# a0910/user profile). Analysis is skipped here; merge + ACC_overall tests are
# done offline after both packs exist.
#
# Usage (from anywhere; script cds into GNCDM/):
#   # Use the REAL EduCDM interpreter path on the server (do NOT copy /path/to/...).
#   # On this project's cloud box the working value was often:
#   #   ICD_PYTHON=/opt/conda/bin/python
#   ICD_PYTHON=/opt/conda/bin/python \
#     bash GNCDM/scripts/run_a0910_user_extra_seeds.sh --device cuda:0
#
# Optional flags:
#   --device cuda:0          torch device (default: auto if omitted)
#   --no-resume             do not pass --resume (default: resume on)
#   --icd-python PATH       EduCDM interpreter (default: $ICD_PYTHON or current python)
#   --output-dir PATH       override output root (default: .../a0910_user_extra)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNCDM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${GNCDM_DIR}/experiments/_core/run_statistical_trials.py"

EXTRA_SEEDS=(2 3 5 11 13)
DEVICE="${DEVICE:-}"
RESUME=1
ICD_PYTHON="${ICD_PYTHON:-}"
OUTPUT_DIR="${GNCDM_DIR}/incremental_result/significance_trials/a0910_user_extra"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-resume)
            RESUME=0
            shift
            ;;
        --icd-python)
            ICD_PYTHON="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${ICD_PYTHON}" ]]; then
    ICD_PYTHON="${PYTHON_BIN}"
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${GNCDM_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${GNCDM_DIR}"

CMD=(
    "${PYTHON_BIN}"
    "${RUNNER}"
    --dataset a0910
    --split user
    --seeds "${EXTRA_SEEDS[@]}"
    --output-dir "${OUTPUT_DIR}"
    --support-query-seed 7
    --support-frac 0.5
    --icd-python "${ICD_PYTHON}"
    --skip-analysis
)

if [[ -n "${DEVICE}" ]]; then
    CMD+=(--device "${DEVICE}")
fi
if [[ "${RESUME}" -eq 1 ]]; then
    CMD+=(--resume)
fi

printf '\n%s\n' "========================================================================"
printf 'a0910 user_split EXTRA seeds: %s\n' "${EXTRA_SEEDS[*]}"
printf 'support/query: seed=7 frac=0.5\n'
printf 'Results: %s\n' "${OUTPUT_DIR}"
printf 'ICD_PYTHON: %s\n' "${ICD_PYTHON}"
printf '%s\n\n' "========================================================================"
printf 'Command:\n'
printf '  %q' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"

printf '\nDone. Expected artifacts under %s:\n' "${OUTPUT_DIR}"
printf '  trials/seed_{2,3,5,11,13}.csv\n'
printf '  per_seed_results.csv\n'
printf 'Merge with a0910_user/ then run ACC_overall paired tests offline.\n'
