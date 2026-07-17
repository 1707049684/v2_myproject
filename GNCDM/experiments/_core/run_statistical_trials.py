"""Run auditable multi-seed trials and formal paired significance analysis.

Run this module from ``GNCDM/`` so the data paths and imports resolve:

    python experiments/_core/run_statistical_trials.py \
      --dataset math1 --split random --seeds 1 7 21 42 84

The runner intentionally writes a new long per-seed table under
``incremental_result/significance_trials``. It never overwrites the legacy
``all_methods_*.csv`` point-estimate tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = CORE_DIR.parent
GNCDM_DIR = EXPERIMENTS_DIR.parent
REPO_ROOT = GNCDM_DIR.parent
for path in (str(GNCDM_DIR), str(EXPERIMENTS_DIR), str(CORE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import cl_baselines_random_split as random_baselines  # noqa: E402
import gncdm_clora_baseline as gncdm_clora  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from eval_all_methods_user_split import run_trial_for_statistics  # noqa: E402
from incremental.statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_PRIMARY_METRIC,
    analyze_paired_results,
    minimum_pairs_for_exact_holm,
    write_analysis_reports,
)
from run_incremental_a0910 import auto_new_concepts  # noqa: E402
from run_incremental_math1 import run_experiment, set_seed  # noqa: E402
from run_xder import run_xder, run_xder_user_split  # noqa: E402

DEFAULT_SEEDS = [1, 7, 21, 42, 84]
DEFAULT_TARGET = "CLEAN-Full"
GNCDM_INCREMENTAL_STRATEGIES = frozenset(
    {"Ours (Dynamic DNA)", "Ours (LoRA)", "Full Replay Oracle"}
)
REQUESTED_METHODS = [
    "G-NCDM(Anchor)",
    "EWC",
    "DER++",
    "C-LoRA",
    "X-DER",
    "Full-Replay",
    "C-LoRA-GNCDM",
    "ICD",
    "CLEAN-Full",
    "CLEAN-LoRA",
]
FIXED_BASELINE_PROFILES = {
    ("math1", "random"): {
        "source_table": "all_methods_math1_random_split.csv",
        "selection_source": "fixed_from_existing_result",
        "ewc_lambda": 10_000,
        "der_mem_size": 5_000,
        "clora_lambda": 10_000,
        "gncdm_clora_lambda": 10.0,
        "xder_mem_size": 5_000,
        "methods": list(REQUESTED_METHODS),
        "comparison_methods": [
            "EWC",
            "DER++",
            "C-LoRA",
            "X-DER",
            "C-LoRA-GNCDM",
            "ICD",
            "CLEAN-LoRA",
        ],
    },
    ("a0910", "random"): {
        "source_table": "all_methods_a0910_random_split.csv",
        "selection_source": "fixed_from_existing_result",
        "ewc_lambda": 10_000,
        "der_mem_size": 5_000,
        "clora_lambda": 10_000,
        "gncdm_clora_lambda": 10.0,
        "xder_mem_size": 5_000,
        "methods": list(REQUESTED_METHODS),
        "comparison_methods": [
            "EWC",
            "DER++",
            "C-LoRA",
            "X-DER",
            "C-LoRA-GNCDM",
            "ICD",
            "CLEAN-LoRA",
        ],
    },
    ("a0910", "user"): {
        "source_table": "all_methods_a0910_user_split.csv",
        "selection_source": "fixed_from_existing_result",
        "ewc_lambda": 10_000,
        "der_mem_size": 5_000,
        "clora_lambda": 10.0,
        "gncdm_clora_lambda": 10.0,
        "xder_mem_size": 5_000,
        "methods": list(REQUESTED_METHODS),
        "comparison_methods": [
            "EWC",
            "DER++",
            "C-LoRA",
            "X-DER",
            "C-LoRA-GNCDM",
            "ICD",
            "CLEAN-LoRA",
        ],
    },
    ("junyi", "random"): {
        "source_table": "all_methods_junyi_random_split.csv",
        "selection_source": "fixed_from_existing_result",
        "ewc_lambda": 1_000,
        "der_mem_size": 5_000,
        "clora_lambda": 10.0,
        "gncdm_clora_lambda": 0.1,
        "xder_mem_size": 5_000,
        "methods": list(REQUESTED_METHODS),
        "comparison_methods": [
            "EWC",
            "DER++",
            "C-LoRA",
            "X-DER",
            "C-LoRA-GNCDM",
            "ICD",
            "CLEAN-LoRA",
        ],
    },
}
METRIC_COLUMNS = [
    "AUC_old",
    "AUC_new",
    "RMSE_old",
    "RMSE_new",
    "ACC_old",
    "ACC_new",
    "F1_old",
    "F1_new",
    "TMD",
]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-free multi-seed incremental trials and exact paired significance tests."
    )
    parser.add_argument("--dataset", choices=("math1", "a0910", "junyi"), required=True)
    parser.add_argument("--split", choices=("random", "user"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to incremental_result/significance_trials/<dataset>_<split>.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse complete per-seed trial CSVs.")
    parser.add_argument(
        "--icd-python",
        default=os.environ.get("ICD_PYTHON"),
        help=(
            "Python executable from the isolated EduCDM environment. Defaults to ICD_PYTHON "
            "or the current interpreter."
        ),
    )
    parser.add_argument(
        "--support-query-seed",
        type=int,
        default=7,
        help="Fixed user-split support/query seed for the primary analysis.",
    )
    parser.add_argument("--support-frac", type=float, default=0.5)
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Do not request deterministic cuDNN behavior (recorded in the manifest).",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        help="Defaults to the fixed realistic baselines for the selected profile.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[DEFAULT_PRIMARY_METRIC],
        help="Predeclared metrics to test; default is Balanced_AUC only.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--strict-min-seeds",
        action="store_true",
        help="Reject underpowered runs instead of writing an exploratory five-seed report.",
    )
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Also test CLEAN-Full against Full-Replay; it is otherwise output-only as an oracle.",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args(argv)


def _fixed_baseline_profile(dataset: str, split: str) -> dict:
    try:
        return dict(FIXED_BASELINE_PROFILES[(dataset, split)])
    except KeyError as exc:
        supported = ", ".join(f"{name}/{mode}" for name, mode in FIXED_BASELINE_PROFILES)
        raise ValueError(
            "No fixed-hyperparameter profile is configured for "
            f"{dataset}/{split}. Supported profiles: {supported}."
        ) from exc


def _default_baselines(profile: dict) -> list[str]:
    return list(profile["comparison_methods"])


def _device_from_arg(value: str | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _path_fingerprint(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _protocol_id(
    config: dict,
    *,
    split: str,
    support_query_seed: int,
    support_frac: float,
    deterministic: bool,
    icd_python: str | None,
) -> str:
    protocol_config = {
        key: value for key, value in config.items() if key not in {"train", "valid", "test", "Q"}
    }
    payload = {
        "split": split,
        "config": protocol_config,
        "support_query_seed": support_query_seed if split == "user" else None,
        "support_frac": support_frac if split == "user" else None,
        "deterministic_cudnn": deterministic,
        "baseline_selection": config["fixed_baselines"]["selection_source"],
        "icd_python": icd_python,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _dataset_config(dataset: str, split: str) -> dict:
    if dataset == "math1":
        if split == "random":
            return {
                "name": "math1",
                "train": str(GNCDM_DIR / "data" / "math1_train_0.8_0.2.csv"),
                "valid": str(GNCDM_DIR / "data" / "math1_valid_0.8_0.2.csv"),
                "test": str(GNCDM_DIR / "data" / "math1_test_0.8_0.2.csv"),
                "Q": str(GNCDM_DIR / "data" / "math1_Q_matrix.npy"),
                "n_user": 4209,
                "n_item": 20,
                "n_know": 11,
                "new_concepts": [0, 1, 3, 6],
                "alpha": 0.20,
                "strategy_select_metric": "acc",
            }
        return {
            "name": "math1",
            "train": str(REPO_ROOT / "data" / "math1" / "user_split" / "train.csv"),
            "valid": str(REPO_ROOT / "data" / "math1" / "user_split" / "valid.csv"),
            "test": str(REPO_ROOT / "data" / "math1" / "user_split" / "test.csv"),
            "Q": str(GNCDM_DIR / "data" / "math1_Q_matrix.npy"),
            "n_user": 4209,
            "n_item": 20,
            "n_know": 11,
            "new_concepts": [0, 1, 3, 6],
            "alpha": 0.70,
        }

    root = REPO_ROOT / "data" / dataset
    q_path = root / "Q_matrix.npy"
    q_matrix = np.load(q_path)
    if dataset == "a0910":
        n_user, alpha = 4163, 0.10 if split == "random" else 0.60
    else:
        split_dir = root / f"new_{split}_split"
        n_user = max(
            int(pd.read_csv(split_dir / filename)["user_id"].max()) + 1
            for filename in ("train.csv", "valid.csv", "test.csv")
        )
        alpha = 0.10 if split == "random" else 0.60
    split_dir = root / f"new_{split}_split"
    return {
        "name": dataset,
        "train": str(split_dir / "train.csv"),
        "valid": str(split_dir / "valid.csv"),
        "test": str(split_dir / "test.csv"),
        "Q": str(q_path),
        "n_user": n_user,
        "n_item": int(q_matrix.shape[0]),
        "n_know": int(q_matrix.shape[1]),
        "new_concepts": auto_new_concepts(q_matrix, 0.34),
        "alpha": alpha,
        "strategy_select_metric": "acc",
    }


def _canonical_method(name: str) -> str:
    aliases = {
        "Base": "G-NCDM(Anchor)",
        "Ours (Dynamic DNA)": "CLEAN-Full",
        "Ours (LoRA)": "CLEAN-LoRA",
        "Full Replay Oracle": "Full-Replay",
    }
    if name in aliases:
        return aliases[name]
    if name.startswith("C-LoRA-GNCDM"):
        return "C-LoRA-GNCDM"
    if name.startswith("X-DER"):
        return "X-DER"
    if name.startswith("EWC"):
        return "EWC"
    if name.startswith("DER++"):
        return "DER++"
    if name.startswith("C-LoRA"):
        return "C-LoRA"
    return name


def _method_family(method: str) -> tuple[str, bool, str]:
    if method == "Full-Replay":
        return "oracle", True, "gncdm_concept"
    if method == "G-NCDM(Anchor)":
        return "anchor", False, "gncdm_concept"
    if method == "ICD":
        return "external_baseline", False, "icd_trait"
    if method in {"EWC", "DER++", "C-LoRA"}:
        return "continual_learning_baseline", False, "embedding"
    if method in {"C-LoRA-GNCDM", "X-DER"}:
        return "gncdm_baseline", False, "gncdm_concept"
    if method == "Base":
        return "base", False, "gncdm_concept"
    return "gncdm_strategy", False, "gncdm_concept"


def _selected_hparams(row: dict, method: str) -> dict:
    if "selected_lambda" in row:
        return {"lambda": row["selected_lambda"]}
    if method == "C-LoRA-GNCDM" and "ortho_lambda" in row:
        return {"lambda": row["ortho_lambda"]}
    if method in {"DER++", "X-DER"}:
        return {
            "mem_size": row.get("mem_size", random_baselines.MEM_SIZE),
        }
    return {}


def _canonicalize_rows(
    rows: list[dict],
    *,
    dataset: str,
    split: str,
    protocol: str,
    seed: int,
    support_query_seed: int | None,
    support_frac: float | None,
) -> pd.DataFrame:
    canonical_rows: list[dict] = []
    for row in rows:
        original_method = str(row.get("Method", row.get("Model", "")))
        if not original_method:
            raise ValueError(f"A trial row has no Method/Model field: {row}")
        method = _canonical_method(original_method)
        family, is_oracle, drift_space = _method_family(method)
        record = {
            "dataset": dataset,
            "split": split,
            "protocol": protocol,
            "seed": seed,
            "train_seed": seed,
            "support_query_seed": support_query_seed,
            "support_frac": support_frac,
            "method": method,
            "method_label": original_method,
            "method_family": family,
            "is_oracle": is_oracle,
            "drift_space": drift_space,
            "selected_hparams": json.dumps(_selected_hparams(row, method), sort_keys=True),
            "selection_source": row.get("selection_source", "fixed_or_not_applicable"),
        }
        for metric in METRIC_COLUMNS:
            source = "RD" if metric == "TMD" else metric
            record[metric] = pd.to_numeric(row.get(source, np.nan), errors="coerce")
        canonical_rows.append(record)
    frame = pd.DataFrame(canonical_rows)
    key = ["dataset", "split", "protocol", "seed", "method"]
    if frame.duplicated(key).any():
        duplicated = frame.loc[frame.duplicated(key, keep=False), key].to_dict("records")
        raise ValueError(f"Canonical trial has duplicate methods: {duplicated}")
    return frame


def _icd_script(dataset: str, split: str) -> Path:
    scripts = {
        ("math1", "random"): EXPERIMENTS_DIR / "run_icd_math1_A.py",
        ("a0910", "random"): EXPERIMENTS_DIR / "run_icd_a0910_A.py",
        ("a0910", "user"): EXPERIMENTS_DIR / "run_icd_a0910_A.py",
        ("junyi", "random"): EXPERIMENTS_DIR / "run_icd_junyi_A.py",
    }
    try:
        return scripts[(dataset, split)]
    except KeyError as exc:
        raise ValueError(f"No seed-aware ICD entry is configured for {dataset}/{split}.") from exc


def _icd_command(config: dict, args: argparse.Namespace, device: torch.device) -> list[str]:
    """Build the isolated-EduCDM command without importing its incompatible package."""

    dataset, split = config["name"], args.split
    command = [str(args.icd_python or sys.executable), str(_icd_script(dataset, split))]
    if dataset == "a0910":
        command.extend(
            [
                str(REPO_ROOT / "data" / "a0910"),
                str(device),
                "25",
                f"{split}_split",
            ]
        )
    elif dataset == "junyi":
        command.extend([str(REPO_ROOT / "data" / "junyi"), str(device), "25"])
    return command


def _validate_icd_interpreter(python_executable: str) -> None:
    """Fail before expensive trials when the isolated ICD environment is unavailable."""

    try:
        check = subprocess.run(
            [python_executable, "-c", "import EduCDM"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to start the ICD interpreter: {python_executable}") from exc
    if check.returncode != 0:
        detail = check.stderr.strip().splitlines()[-1] if check.stderr.strip() else "import failed"
        raise RuntimeError(
            "ICD is part of the requested five-seed roster but EduCDM is unavailable in "
            f"{python_executable!r}: {detail}. Pass --icd-python /path/to/icd-venv/bin/python "
            "(or set ICD_PYTHON) to use the isolated ICD environment."
        )


def _run_icd_trial(
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict:
    """Run ICD once in its own environment and normalize its single result row."""

    output_path = output_dir / "icd_raw" / f"seed_{seed}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "ICD_TRAIN_SEED": str(seed),
            "ICD_OUTPUT_CSV": str(output_path),
            "ICD_SUPPORT_QUERY_SEED": str(args.support_query_seed),
            "ICD_SUPPORT_FRAC": str(args.support_frac),
        }
    )
    command = _icd_command(config, args, device)
    print(f"\n=== External ICD: seed={seed} ===")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=GNCDM_DIR, env=environment)
    if not output_path.exists():
        raise RuntimeError(f"ICD completed without writing its expected result: {output_path}")
    rows = pd.read_csv(output_path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one ICD row in {output_path}, found {len(rows)}.")
    row = rows.iloc[0].to_dict()
    row["Method"] = "ICD"
    row["selection_source"] = "fixed_protocol_external_icd"
    return row


def _validate_trial_roster(frame: pd.DataFrame, expected_methods: Sequence[str]) -> pd.DataFrame:
    expected = list(expected_methods)
    expected_set = set(expected)
    observed_set = set(frame["method"])
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    if missing or unexpected:
        raise ValueError(
            "Trial methods do not match the fixed requested roster. "
            f"missing={missing}; unexpected={unexpected}."
        )
    order = {method: position for position, method in enumerate(expected)}
    return frame.sort_values("method", key=lambda methods: methods.map(order)).reset_index(
        drop=True
    )


def _run_random_trial(
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    output_dir: Path,
    deterministic: bool,
) -> list[dict]:
    set_seed(seed, deterministic=deterministic)
    ours_path = output_dir / "ours_raw" / f"seed_{seed}.csv"
    ours_rows = run_experiment(
        f"{config['name']}_random_split_seed_{seed}",
        "buf",
        config["train"],
        config["valid"],
        config["test"],
        config["Q"],
        device,
        n_user=config["n_user"],
        n_item_total=config["n_item"],
        n_know_total=config["n_know"],
        new_concepts=config["new_concepts"],
        alpha=config["alpha"],
        run_strategies=GNCDM_INCREMENTAL_STRATEGIES,
        strategy_select_metric=config.get("strategy_select_metric", "acc"),
        output_path=str(ours_path),
    )
    baseline_config = {
        "name": config["name"],
        "train": config["train"],
        "valid": config["valid"],
        "test": config["test"],
        "Q": config["Q"],
        "n_item": config["n_item"],
        "n_know": config["n_know"],
        "new_concepts": config["new_concepts"],
    }
    baseline_meta = random_baselines.load_random(baseline_config)
    fixed = config["fixed_baselines"]
    baseline_rows = random_baselines.run_fixed_baselines(
        baseline_meta,
        device,
        seed=seed,
        ewc_lambda=fixed["ewc_lambda"],
        der_mem_size=fixed["der_mem_size"],
        clora_lambda=fixed.get("clora_lambda"),
    )
    if fixed.get("gncdm_clora_lambda") is not None:
        baseline_rows.append(
            gncdm_clora.run_fixed_random_split(
                config,
                device,
                lambda_ortho=fixed["gncdm_clora_lambda"],
                seed=seed,
            )
        )
    if fixed.get("xder_mem_size") is not None:
        baseline_rows.append(
            run_xder(
                f"{config['name']}_random_split_seed_{seed}",
                config["name"],
                config["train"],
                config["valid"],
                config["test"],
                config["Q"],
                device,
                config["n_user"],
                config["n_item"],
                config["n_know"],
                config["new_concepts"],
                alpha=config["alpha"],
                buffer_size=fixed["xder_mem_size"],
                seed=seed,
                write_artifacts=False,
            )
        )
    if "ICD" in fixed["methods"]:
        baseline_rows.append(_run_icd_trial(config, args, device, seed, output_dir))
    return ours_rows + baseline_rows


def _write_manifest(
    output_dir: Path,
    *,
    config: dict,
    args: argparse.Namespace,
    protocol: str,
    device: torch.device,
) -> None:
    minimum_seed_count = minimum_pairs_for_exact_holm(
        args.alpha, len(args.baselines) * len(args.metrics)
    )
    exploratory_warning = None
    if len(args.seeds) < minimum_seed_count:
        exploratory_warning = (
            f"Only {len(args.seeds)} paired seeds were run. With exact two-sided sign-flip "
            f"testing and {len(args.baselines) * len(args.metrics)} planned Holm hypotheses, "
            f"at least {minimum_seed_count} seeds are needed for a rejection to be attainable."
        )
    manifest = {
        "created_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "dataset": config["name"],
        "split": args.split,
        "protocol": protocol,
        "train_seeds": args.seeds,
        "minimum_formal_seed_count": minimum_seed_count,
        "inference_status": "exploratory_underpowered" if exploratory_warning else "formal",
        "inference_warning": exploratory_warning,
        "support_query_seed": args.support_query_seed if args.split == "user" else None,
        "support_frac": args.support_frac if args.split == "user" else None,
        "deterministic_cudnn": not args.non_deterministic,
        "target": args.target,
        "baselines": args.baselines,
        "metrics": args.metrics,
        "alpha": args.alpha,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "baseline_hyperparameter_selection": config["fixed_baselines"]["selection_source"],
        "icd_python": args.icd_python,
        "oracle_policy": "Full-Replay is saved but excluded from the default comparison family.",
        "drift_policy": "TMD/RD is not comparable across gncdm_concept and embedding spaces.",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "data_fingerprints": {
            key: _path_fingerprint(config[key]) for key in ("train", "valid", "test", "Q")
        },
        "config": {
            key: value
            for key, value in config.items()
            if key not in {"train", "valid", "test", "Q"}
        },
    }
    (output_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _load_or_run_trial(
    *,
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    protocol: str,
    seed: int,
    deterministic: bool,
) -> pd.DataFrame:
    trial_path = output_dir / "trials" / f"seed_{seed}.csv"
    if args.resume and trial_path.exists():
        frame = pd.read_csv(trial_path)
        expected = {"dataset", "split", "protocol", "seed", "method"}
        expected_roster = set(config["fixed_baselines"]["methods"])
        cache_matches_protocol = (
            not frame.empty
            and expected.issubset(frame.columns)
            and frame["dataset"].eq(config["name"]).all()
            and frame["split"].eq(args.split).all()
            and frame["protocol"].eq(protocol).all()
            and frame["seed"].eq(seed).all()
            and set(frame["method"]) == expected_roster
        )
        if cache_matches_protocol:
            print(f"[resume] seed={seed}: {trial_path}")
            return frame
        print(f"[resume] seed={seed}: invalid cached trial; rerunning.")

    print(f"\n{'=' * 72}\nStatistical trial: {config['name']} {args.split} seed={seed}\n{'=' * 72}")
    if args.split == "random":
        rows = _run_random_trial(config, args, device, seed, output_dir, deterministic)
        support_query_seed, support_frac = None, None
    else:
        rows, _ = run_trial_for_statistics(
            config,
            device,
            train_seed=seed,
            support_query_seed=args.support_query_seed,
            support_frac=args.support_frac,
            deterministic=deterministic,
            fixed_baseline_config=config["fixed_baselines"],
            ours_strategies=GNCDM_INCREMENTAL_STRATEGIES,
        )
        fixed = config["fixed_baselines"]
        if fixed.get("gncdm_clora_lambda") is not None:
            rows.append(
                gncdm_clora.run_fixed_user_split(
                    config,
                    device,
                    lambda_ortho=fixed["gncdm_clora_lambda"],
                    seed=seed,
                    support_query_seed=args.support_query_seed,
                    support_frac=args.support_frac,
                )
            )
        if fixed.get("xder_mem_size") is not None:
            xder_row = run_xder_user_split(
                split_name=f"{config['name']}_user_split_seed_{seed}",
                ds_name=config["name"],
                train_path=config["train"],
                valid_path=config["valid"],
                test_path=config["test"],
                Q_path=config["Q"],
                device=device,
                n_user=config["n_user"],
                n_item_total=config["n_item"],
                n_know_total=config["n_know"],
                new_concepts=config["new_concepts"],
                alpha=config["alpha"],
                buffer_size=fixed["xder_mem_size"],
                seed=seed,
                support_query_seed=args.support_query_seed,
                support_frac=args.support_frac,
                artifact_dir=output_dir / "xder_raw" / f"seed_{seed}",
            )
            xder_row["selection_source"] = "fixed_from_user_specification"
            rows.append(xder_row)
        if "ICD" in config["fixed_baselines"]["methods"]:
            rows.append(_run_icd_trial(config, args, device, seed, output_dir))
        support_query_seed, support_frac = args.support_query_seed, args.support_frac
    frame = _canonicalize_rows(
        rows,
        dataset=config["name"],
        split=args.split,
        protocol=protocol,
        seed=seed,
        support_query_seed=support_query_seed,
        support_frac=support_frac,
    )
    frame = _validate_trial_roster(frame, config["fixed_baselines"]["methods"])
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(trial_path, index=False)
    return frame


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if args.split == "user" and not 0.0 < args.support_frac < 1.0:
        raise ValueError("--support-frac must lie strictly between zero and one.")

    config = _dataset_config(args.dataset, args.split)
    config["fixed_baselines"] = _fixed_baseline_profile(args.dataset, args.split)
    if args.baselines is None:
        args.baselines = _default_baselines(config["fixed_baselines"])
    if args.include_oracle and "Full-Replay" not in args.baselines:
        args.baselines.append("Full-Replay")
    available_methods = set(config["fixed_baselines"]["methods"])
    if args.target not in available_methods:
        raise ValueError(
            f"The fixed profile does not run target {args.target!r}. "
            f"Available methods: {', '.join(config['fixed_baselines']['methods'])}."
        )
    unknown_baselines = sorted(set(args.baselines).difference(available_methods))
    if unknown_baselines:
        raise ValueError(
            "The fixed profile does not run these requested baselines: "
            + ", ".join(unknown_baselines)
        )

    minimum_pairs = minimum_pairs_for_exact_holm(
        args.alpha, len(args.baselines) * len(args.metrics)
    )
    inference_warning = None
    if len(args.seeds) < minimum_pairs:
        inference_warning = (
            f"Exploratory five-seed analysis: {len(args.seeds)} paired seeds cannot attain a "
            f"Holm-adjusted two-sided exact rejection across {len(args.baselines) * len(args.metrics)} "
            f"planned hypotheses at alpha={args.alpha}. At least {minimum_pairs} seeds are required."
        )
        print(f"[STATISTICAL WARNING] {inference_warning}")
        if args.strict_min_seeds:
            raise ValueError(inference_warning)

    if "ICD" in config["fixed_baselines"]["methods"]:
        args.icd_python = args.icd_python or sys.executable
        _validate_icd_interpreter(args.icd_python)

    if any(metric in {"TMD", "RD"} for metric in args.metrics):
        baseline_spaces = {"EWC", "DER++", "C-LoRA", "ICD"}
        if baseline_spaces.intersection(args.baselines):
            raise ValueError(
                "TMD/RD is not comparable across G-NCDM, embedding-space, and ICD-trait baselines; "
                "remove EWC/DER++/C-LoRA/ICD or do not request TMD/RD."
            )

    deterministic = not args.non_deterministic
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else GNCDM_DIR
        / "incremental_result"
        / "significance_trials"
        / f"{args.dataset}_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)
    protocol = _protocol_id(
        config,
        split=args.split,
        support_query_seed=args.support_query_seed,
        support_frac=args.support_frac,
        deterministic=deterministic,
        icd_python=args.icd_python,
    )
    _write_manifest(output_dir, config=config, args=args, protocol=protocol, device=device)
    print(f"device={device}; protocol={protocol}; output={output_dir}")

    trials = [
        _load_or_run_trial(
            config=config,
            args=args,
            device=device,
            output_dir=output_dir,
            protocol=protocol,
            seed=seed,
            deterministic=deterministic,
        )
        for seed in args.seeds
    ]
    results = pd.concat(trials, ignore_index=True)
    per_seed_path = output_dir / "per_seed_results.csv"
    results.to_csv(per_seed_path, index=False)
    print(f"per-seed results: {per_seed_path}")

    if args.skip_analysis:
        return
    analysis = analyze_paired_results(
        results,
        target=args.target,
        baselines=args.baselines,
        metrics=args.metrics,
        alpha=args.alpha,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        minimum_pairs=minimum_pairs if args.strict_min_seeds else None,
    )
    if inference_warning:
        analysis.significance["inference_warning"] = inference_warning
    paths = write_analysis_reports(analysis, output_dir, stem="formal_significance")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
