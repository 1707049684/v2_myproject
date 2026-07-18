#!/usr/bin/env bash
# Run selected fixed-hyperparameter, five-seed statistical-significance profiles.
#
# Usage:
#   bash GNCDM/scripts/run_fixed_significance_profiles.sh --dataset math1 --device cuda:0
#   bash GNCDM/scripts/run_fixed_significance_profiles.sh --dataset a0910 --split random
#   bash GNCDM/scripts/run_fixed_significance_profiles.sh --all --device cuda:2
#
# The available fixed profiles use seeds 1, 7, 21, 42, 84:
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
DATASET=""
SPLIT=""
RUN_ALL=0
PROFILE_DATASETS=(math1 a0910 a0910 junyi)
PROFILE_SPLITS=(random random user random)

usage() {
    cat <<'EOF'
Usage: bash GNCDM/scripts/run_fixed_significance_profiles.sh \
  (--dataset {math1|a0910|junyi} [--split {random|user}] | --all) [options]

Select one dataset at a time, or explicitly request all profiles. The available
fixed five-seed profiles are:
  1. math1/random
  2. a0910/random
  3. a0910/user
  4. junyi/random

Selectors (choose exactly one):
  --dataset DATASET     Run all configured profiles for math1, a0910, or junyi.
                        With --split, run only that dataset/split profile.
  --split SPLIT         Restrict --dataset to random or user.
  --all                 Explicitly run all four configured profiles.

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
  bash GNCDM/scripts/run_fixed_significance_profiles.sh --dataset math1 --device cuda:0
  bash GNCDM/scripts/run_fixed_significance_profiles.sh --dataset a0910 --split random --device cuda:0
  bash GNCDM/scripts/run_fixed_significance_profiles.sh --dataset a0910 --device cuda:0
  ICD_PYTHON=/opt/icd-venv/bin/python bash GNCDM/scripts/run_fixed_significance_profiles.sh --dataset junyi --resume
  bash GNCDM/scripts/run_fixed_significance_profiles.sh --all --device cuda:0
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
        --dataset)
            require_option_value "$@"
            DATASET="$2"
            shift 2
            ;;
        --split)
            require_option_value "$@"
            SPLIT="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=1
            shift
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

if [[ -n "${SPLIT}" && -z "${DATASET}" ]]; then
    echo "--split can only be used together with --dataset." >&2
    usage >&2
    exit 2
fi

SELECTOR_COUNT=0
if [[ -n "${DATASET}" ]]; then
    SELECTOR_COUNT=$((SELECTOR_COUNT + 1))
fi
if [[ "${RUN_ALL}" -eq 1 ]]; then
    SELECTOR_COUNT=$((SELECTOR_COUNT + 1))
fi
if [[ "${SELECTOR_COUNT}" -ne 1 ]]; then
    echo "Choose exactly one selector: --dataset DATASET or --all." >&2
    usage >&2
    exit 2
fi

if [[ -n "${DATASET}" && "${DATASET}" != "math1" && "${DATASET}" != "a0910" && "${DATASET}" != "junyi" ]]; then
    echo "Unsupported dataset: ${DATASET}. Choose math1, a0910, or junyi." >&2
    exit 2
fi

if [[ -n "${SPLIT}" && "${SPLIT}" != "random" && "${SPLIT}" != "user" ]]; then
    echo "Unsupported split: ${SPLIT}. Choose random or user." >&2
    exit 2
fi

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

profile_is_selected() {
    local candidate_dataset="$1"
    local candidate_split="$2"

    if [[ "${RUN_ALL}" -eq 1 ]]; then
        return 0
    fi

    [[ "${candidate_dataset}" == "${DATASET}" ]] || return 1
    [[ -z "${SPLIT}" || "${candidate_split}" == "${SPLIT}" ]]
}

SELECTED_COUNT=0
for index in "${!PROFILE_DATASETS[@]}"; do
    dataset="${PROFILE_DATASETS[${index}]}"
    split="${PROFILE_SPLITS[${index}]}"
    if profile_is_selected "${dataset}" "${split}"; then
        run_profile "${dataset}" "${split}"
        SELECTED_COUNT=$((SELECTED_COUNT + 1))
    fi
done

if [[ "${SELECTED_COUNT}" -eq 0 ]]; then
    echo "No fixed profile is configured for ${DATASET}/${SPLIT}." >&2
    echo "Available profiles: math1/random, a0910/random, a0910/user, junyi/random." >&2
    exit 2
fi

printf '\nCompleted %d fixed five-seed profile(s). Results are under: %s\n' "${SELECTED_COUNT}" "${OUTPUT_ROOT}"
