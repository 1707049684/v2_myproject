#!/usr/bin/env bash
# Run all fixed-hyperparameter, five-seed statistical-significance profiles.
#
# Usage:
#   bash GNCDM/scripts/run_fixed_significance_profiles.sh
#   bash GNCDM/scripts/run_fixed_significance_profiles.sh --device cuda:2
#   bash GNCDM/scripts/run_fixed_significance_profiles.sh --resume
#
# This entry always runs exactly these profiles with seeds 1, 7, 21, 42, 84:
#   math1/random, a0910/random, a0910/user, junyi/random.
# The fixed EWC/DER++/C-LoRA-GNCDM/X-DER parameters are defined in
# experiments/_core/run_statistical_trials.py.  No validation lambda sweep is run.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GNCDM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${GNCDM_DIR}/experiments/_core/run_statistical_trials.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
ICD_PYTHON="${ICD_PYTHON:-}"
DEVICE=""
OUTPUT_ROOT="${GNCDM_DIR}/incremental_result/significance_trials"
RESUME=0
SEEDS=(1 7 21 42 84)

usage() {
    cat <<'EOF'
Usage: bash GNCDM/scripts/run_fixed_significance_profiles.sh [options]

Run, in sequence, the fixed five-seed significance experiments for:
  1. math1/random
  2. a0910/random
  3. a0910/user
  4. junyi/random

Options:
  --device DEVICE       Torch device to use, for example cuda:0 or cuda:2.
                        By default, the Python runner auto-detects CUDA/CPU.
  --output-root DIR     Parent directory for the four result directories.
                        Default: GNCDM/incremental_result/significance_trials
  --icd-python PATH     Python executable in the isolated EduCDM environment.
                        Defaults to ICD_PYTHON, then PYTHON_BIN.
  --resume              Reuse valid completed seed_<seed>.csv trial files.
  --help, -h            Show this help message.

Environment:
  PYTHON_BIN=/path/to/python  Select the main experiment Python executable.
  ICD_PYTHON=/path/to/python  Select the isolated EduCDM Python executable.

Examples:
  bash GNCDM/scripts/run_fixed_significance_profiles.sh --device cuda:2
  ICD_PYTHON=/opt/icd-venv/bin/python bash GNCDM/scripts/run_fixed_significance_profiles.sh --resume
EOF
}

require_option_value() {
    if [[ $# -lt 2 || -z "${2}" || "${2}" == --* ]]; then
        echo "Missing value for ${1}." >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            require_option_value "$@"
            DEVICE="$2"
            shift 2
            ;;
        --output-root)
            require_option_value "$@"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --icd-python)
            require_option_value "$@"
            ICD_PYTHON="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --help | -h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${RUNNER}" ]]; then
    echo "Statistical runner not found: ${RUNNER}" >&2
    exit 1
fi

if [[ -z "${ICD_PYTHON}" ]]; then
    ICD_PYTHON="${PYTHON_BIN}"
fi

# Make a relative output path stable regardless of the directory from which the
# script was invoked.
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${GNCDM_DIR}/${OUTPUT_ROOT}"
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${GNCDM_DIR}"

DEVICE_ARGS=()
if [[ -n "${DEVICE}" ]]; then
    DEVICE_ARGS=(--device "${DEVICE}")
fi

RESUME_ARGS=()
if [[ "${RESUME}" -eq 1 ]]; then
    RESUME_ARGS=(--resume)
fi

ICD_ARGS=(--icd-python "${ICD_PYTHON}")

run_profile() {
    local dataset="$1"
    local split="$2"
    local profile="${dataset}_${split}"
    local output_dir="${OUTPUT_ROOT}/${profile}"
    local -a command=(
        "${PYTHON_BIN}"
        "${RUNNER}"
        --dataset "${dataset}"
        --split "${split}"
        --seeds "${SEEDS[@]}"
        --output-dir "${output_dir}"
    )

    command+=("${DEVICE_ARGS[@]}" "${RESUME_ARGS[@]}" "${ICD_ARGS[@]}")
    if [[ "${split}" == "user" ]]; then
        command+=(--support-query-seed 7 --support-frac 0.5)
    fi

    printf '\n%s\n' "========================================================================"
    printf 'Running fixed five-seed profile: %s/%s\n' "${dataset}" "${split}"
    printf 'Results: %s\n' "${output_dir}"
    printf '%s\n' "========================================================================"
    "${command[@]}"
}

run_profile math1 random
run_profile a0910 random
run_profile a0910 user
run_profile junyi random

printf '\nCompleted all fixed five-seed profiles. Results are under: %s\n' "${OUTPUT_ROOT}"
