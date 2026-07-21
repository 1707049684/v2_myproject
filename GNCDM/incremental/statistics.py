"""Seed-level statistical analysis for incremental-learning experiments.

The functions in this module intentionally depend only on NumPy and pandas so
that the statistical reports are reproducible in the project's base runtime.
They operate on a *long* per-seed table rather than on a single summary table.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP_REPS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20_260_716
DEFAULT_PRIMARY_METRIC = "ACC_overall"
DEFAULT_ALPHA = 0.05
MAX_EXACT_PAIRS = 20
# Legacy fixed-weight endpoint (kept as a derived column only; not the default test metric).
OLD_TASK_ACC_WEIGHT = 0.7
NEW_TASK_ACC_WEIGHT = 0.3
# Test-split interaction counts after strict bipartition (random_split). Used for
# ACC_overall = (n_old * ACC_old + n_new * ACC_new) / (n_old + n_new).
TEST_INTERACTION_COUNTS = {
    "math1": (10901, 5935),
    "junyi": (13997, 6398),
    "a0910": (37642, 16836),
}

REQUIRED_COLUMNS = ("dataset", "split", "seed", "method")
METRIC_DIRECTIONS = {
    "AUC_old": 1,
    "AUC_new": 1,
    "ACC_old": 1,
    "ACC_new": 1,
    "ACC_overall": 1,
    "F1_old": 1,
    "F1_new": 1,
    "Balanced_AUC": 1,
    "Balanced_ACC": 1,
    "RMSE_old": -1,
    "RMSE_new": -1,
    "TMD": -1,
    "RD": -1,
}


class StatisticalInputError(ValueError):
    """Raised when a per-seed result table cannot support a paired test."""


@dataclass(frozen=True)
class AnalysisResult:
    """Structured results returned by :func:`analyze_paired_results`."""

    summary: pd.DataFrame
    significance: pd.DataFrame


def _pool_counts_for_row(row: pd.Series) -> tuple[float, float] | None:
    """Return ``(n_old, n_new)`` used to pool ACC_overall for one result row."""

    if "n_old_test" in row.index and "n_new_test" in row.index:
        n_old, n_new = row["n_old_test"], row["n_new_test"]
        if pd.notna(n_old) and pd.notna(n_new) and float(n_old) + float(n_new) > 0:
            return float(n_old), float(n_new)
    dataset = str(row["dataset"]) if "dataset" in row.index and pd.notna(row["dataset"]) else ""
    if dataset in TEST_INTERACTION_COUNTS:
        return TEST_INTERACTION_COUNTS[dataset]
    # Equal-weight fallback for unit tests / ad-hoc tables without known counts.
    return 1.0, 1.0


def add_derived_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with predeclared derived metrics added.

    ``ACC_overall`` pools old/new test accuracy by each dataset's test-split
    interaction counts (see ``TEST_INTERACTION_COUNTS``), i.e.
    ``(n_old * ACC_old + n_new * ACC_new) / (n_old + n_new)``. This is the
    default primary endpoint. ``Balanced_AUC`` assigns equal weight to retained
    old-task and new-task discrimination. ``Balanced_ACC`` keeps the legacy
    fixed ``0.7 / 0.3`` task weighting and is not the default test metric; it is
    also not the classification metric commonly called balanced accuracy.
    Each derived metric is only defined when both of its source values are
    numeric.
    """

    frame = results.copy()
    for column in ("AUC_old", "AUC_new", "ACC_old", "ACC_new"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if {"AUC_old", "AUC_new"}.issubset(frame.columns):
        frame["Balanced_AUC"] = (frame["AUC_old"] + frame["AUC_new"]) / 2.0
    if {"ACC_old", "ACC_new"}.issubset(frame.columns):
        frame["Balanced_ACC"] = (
            OLD_TASK_ACC_WEIGHT * frame["ACC_old"] + NEW_TASK_ACC_WEIGHT * frame["ACC_new"]
        )
        overall = []
        for _, row in frame.iterrows():
            if pd.isna(row["ACC_old"]) or pd.isna(row["ACC_new"]):
                overall.append(np.nan)
                continue
            n_old, n_new = _pool_counts_for_row(row)
            overall.append(
                (n_old * float(row["ACC_old"]) + n_new * float(row["ACC_new"])) / (n_old + n_new)
            )
        frame["ACC_overall"] = overall
    return frame


def metric_direction(metric: str) -> int:
    """Return ``+1`` when larger is better and ``-1`` when smaller is better."""

    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        allowed = ", ".join(sorted(METRIC_DIRECTIONS))
        raise StatisticalInputError(
            f"Unknown metric {metric!r}. Allowed metrics: {allowed}."
        ) from exc


def validate_trial_results(results: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Validate and canonicalize a long table before statistical analysis.

    The function deliberately rejects duplicate trial keys and unequal seed
    sets later in the pairing stage. Silently intersecting seed sets would make
    an apparently paired p-value invalid.
    """

    missing = sorted(set(REQUIRED_COLUMNS).difference(results.columns))
    if missing:
        raise StatisticalInputError(
            "Per-seed results are missing required columns: " + ", ".join(missing)
        )

    frame = add_derived_metrics(results)
    if "protocol" not in frame:
        frame["protocol"] = "default"
    frame["protocol"] = frame["protocol"].fillna("default").astype(str)
    frame["method"] = frame["method"].astype(str)

    required_metrics = set(metrics)
    missing_metrics = sorted(required_metrics.difference(frame.columns))
    if missing_metrics:
        raise StatisticalInputError(
            "Per-seed results are missing requested metrics: " + ", ".join(missing_metrics)
        )
    for metric in required_metrics:
        metric_direction(metric)
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    key_columns = ["dataset", "split", "protocol", "seed", "method"]
    duplicate = frame.duplicated(key_columns, keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, key_columns].head(5).to_dict("records")
        raise StatisticalInputError(
            "Each (dataset, split, protocol, seed, method) must occur once; "
            f"duplicate examples: {examples}"
        )
    if frame[["dataset", "split", "seed", "method"]].isna().any().any():
        raise StatisticalInputError(
            "dataset, split, seed, and method cannot contain missing values."
        )
    return frame


def paired_exact_permutation(
    differences: Sequence[float], *, max_pairs: int = MAX_EXACT_PAIRS
) -> dict:
    """Run an exact two-sided paired sign-flip permutation test.

    ``differences`` must be oriented so that positive values favor the target
    method. The observed statistic is the mean difference. Enumerating all
    sign flips is equivalent to all pair-label swaps under the null hypothesis.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise StatisticalInputError("Paired permutation testing requires at least one difference.")
    if not np.isfinite(values).all():
        raise StatisticalInputError("Paired permutation testing received non-finite differences.")
    n_pairs = int(values.size)
    if n_pairs > max_pairs:
        raise StatisticalInputError(
            f"Exact sign-flip testing supports at most {max_pairs} pairs; got {n_pairs}."
        )

    observed_sum = abs(float(values.sum()))
    if math.isclose(observed_sum, 0.0, abs_tol=1e-15):
        return {
            "n": n_pairs,
            "statistic": float(values.mean()),
            "p_value": 1.0,
            "permutations": 1 << n_pairs,
        }

    total = 1 << n_pairs
    bit_positions = np.arange(n_pairs, dtype=np.uint64)
    extreme = 0
    # Chunking avoids allocating a 2**20 by 20 array for the largest allowed n.
    chunk_size = 65_536
    tolerance = max(1e-12, observed_sum * 1e-12)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        masks = np.arange(start, stop, dtype=np.uint64)[:, None]
        bits = ((masks >> bit_positions) & 1).astype(np.int8)
        signs = (bits * 2 - 1).astype(float)
        permuted_sums = np.abs(signs @ values)
        extreme += int(np.count_nonzero(permuted_sums >= observed_sum - tolerance))

    return {
        "n": n_pairs,
        "statistic": float(values.mean()),
        "p_value": extreme / total,
        "permutations": total,
    }


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for a paired mean.

    Resampling is performed on per-seed *paired differences*, never separately
    on the two methods, so pairing is preserved.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise StatisticalInputError("Paired bootstrap requires at least one difference.")
    if not np.isfinite(values).all():
        raise StatisticalInputError("Paired bootstrap received non-finite differences.")
    if reps < 1:
        raise StatisticalInputError("Bootstrap repetitions must be positive.")
    if not 0.0 < confidence < 1.0:
        raise StatisticalInputError("confidence must lie strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    batch_size = 4_096
    for start in range(0, reps, batch_size):
        stop = min(reps, start + batch_size)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Apply Holm's step-down correction while preserving original ordering."""

    raw = np.asarray(p_values, dtype=float)
    if raw.ndim != 1:
        raise StatisticalInputError("Holm adjustment requires a one-dimensional p-value sequence.")
    if raw.size == 0:
        return raw.copy()
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise StatisticalInputError(
            "Holm adjustment requires finite p-values between zero and one."
        )

    order = np.argsort(raw, kind="stable")
    sorted_raw = raw[order]
    adjusted_sorted = np.empty_like(sorted_raw)
    running_max = 0.0
    m = len(sorted_raw)
    for index, value in enumerate(sorted_raw):
        running_max = max(running_max, (m - index) * value)
        adjusted_sorted[index] = min(1.0, running_max)
    adjusted = np.empty_like(raw)
    adjusted[order] = adjusted_sorted
    return adjusted


def minimum_pairs_for_exact_holm(alpha: float, hypotheses: int) -> int:
    """Return the smallest ``n`` that can possibly reject after Holm correction.

    A two-sided exact sign-flip test has minimum attainable p-value
    ``2 / 2**n``. For the first Holm step to be rejectable among ``m`` planned
    hypotheses, that p-value must be strictly below ``alpha / m``.
    """

    if not 0.0 < alpha < 1.0:
        raise StatisticalInputError("alpha must lie strictly between zero and one.")
    if hypotheses < 1:
        raise StatisticalInputError("hypotheses must be a positive integer.")
    return math.floor(math.log2(2.0 * hypotheses / alpha)) + 1


def _cohens_dz(oriented_differences: np.ndarray) -> float:
    if len(oriented_differences) < 2:
        return float("nan")
    std = float(np.std(oriented_differences, ddof=1))
    mean = float(np.mean(oriented_differences))
    if math.isclose(std, 0.0, abs_tol=1e-15):
        if math.isclose(mean, 0.0, abs_tol=1e-15):
            return 0.0
        return math.copysign(math.inf, mean)
    return mean / std


def _paired_metric_values(
    group: pd.DataFrame, target: str, baseline: str, metric: str
) -> tuple[np.ndarray, np.ndarray, list]:
    target_rows = group.loc[group["method"] == target, ["seed", metric]].copy()
    baseline_rows = group.loc[group["method"] == baseline, ["seed", metric]].copy()
    if target_rows.empty:
        raise StatisticalInputError(f"Target method {target!r} is absent from a result group.")
    if baseline_rows.empty:
        raise StatisticalInputError(f"Baseline method {baseline!r} is absent from a result group.")
    if target_rows[metric].isna().any() or baseline_rows[metric].isna().any():
        raise StatisticalInputError(
            f"Metric {metric!r} has missing values for comparison {target!r} vs {baseline!r}."
        )

    target_rows = target_rows.set_index("seed").sort_index()
    baseline_rows = baseline_rows.set_index("seed").sort_index()
    target_seeds = set(target_rows.index.tolist())
    baseline_seeds = set(baseline_rows.index.tolist())
    if target_seeds != baseline_seeds:
        missing_from_target = sorted(baseline_seeds - target_seeds)
        missing_from_baseline = sorted(target_seeds - baseline_seeds)
        raise StatisticalInputError(
            f"Seed sets differ for {target!r} vs {baseline!r} on {metric!r}; "
            f"missing from target={missing_from_target}, missing from baseline={missing_from_baseline}."
        )
    seeds = target_rows.index.tolist()
    return (
        target_rows.loc[seeds, metric].to_numpy(dtype=float),
        baseline_rows.loc[seeds, metric].to_numpy(dtype=float),
        seeds,
    )


def _summary_table(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = ["dataset", "split", "protocol", "method"]
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        metadata = dict(zip(group_columns, keys, strict=True))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else float("nan"),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def analyze_paired_results(
    results: pd.DataFrame,
    *,
    target: str,
    baselines: Iterable[str],
    metrics: Sequence[str] = (DEFAULT_PRIMARY_METRIC,),
    alpha: float = DEFAULT_ALPHA,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    minimum_pairs: int | None = None,
) -> AnalysisResult:
    """Analyze predeclared paired comparisons in a per-seed result table.

    Holm correction is applied separately within every
    ``(dataset, split, protocol)`` family over all requested
    ``metric × baseline`` hypotheses. Exclude oracle methods from ``baselines``
    unless they are an explicitly declared exploratory comparison.
    """

    if not 0.0 < alpha < 1.0:
        raise StatisticalInputError("alpha must lie strictly between zero and one.")
    if minimum_pairs is not None and minimum_pairs < 1:
        raise StatisticalInputError("minimum_pairs must be positive when provided.")
    baselines = list(baselines)
    if not baselines:
        raise StatisticalInputError("At least one baseline must be supplied.")
    if target in baselines:
        raise StatisticalInputError("The target method cannot also be a baseline.")
    metrics = list(metrics)
    if not metrics:
        raise StatisticalInputError("At least one metric must be supplied.")

    frame = validate_trial_results(results, metrics)
    summary = _summary_table(frame, metrics)
    rows: list[dict] = []
    family_columns = ["dataset", "split", "protocol"]
    for keys, group in frame.groupby(family_columns, sort=True, dropna=False):
        metadata = dict(zip(family_columns, keys, strict=True))
        family_start = len(rows)
        for metric in metrics:
            direction = metric_direction(metric)
            for baseline in baselines:
                target_values, baseline_values, seeds = _paired_metric_values(
                    group, target, baseline, metric
                )
                if minimum_pairs is not None and len(seeds) < minimum_pairs:
                    raise StatisticalInputError(
                        f"{target!r} vs {baseline!r} has only {len(seeds)} paired seeds; "
                        f"this analysis requires at least {minimum_pairs}."
                    )
                raw_differences = target_values - baseline_values
                oriented_differences = direction * raw_differences
                permutation = paired_exact_permutation(oriented_differences)
                ci_low, ci_high = paired_bootstrap_ci(
                    oriented_differences,
                    reps=bootstrap_reps,
                    seed=bootstrap_seed,
                )
                tolerance = 1e-12
                rows.append(
                    {
                        **metadata,
                        "family": f"{metadata['dataset']}::{metadata['split']}::{metadata['protocol']}",
                        "metric": metric,
                        "target": target,
                        "baseline": baseline,
                        "n": len(seeds),
                        "seeds": ",".join(map(str, seeds)),
                        "target_mean": float(target_values.mean()),
                        "target_sd": float(target_values.std(ddof=1))
                        if len(target_values) > 1
                        else float("nan"),
                        "baseline_mean": float(baseline_values.mean()),
                        "baseline_sd": float(baseline_values.std(ddof=1))
                        if len(baseline_values) > 1
                        else float("nan"),
                        "raw_delta_mean": float(raw_differences.mean()),
                        "oriented_delta_mean": float(oriented_differences.mean()),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "cohens_dz": _cohens_dz(oriented_differences),
                        "wins": int(np.count_nonzero(oriented_differences > tolerance)),
                        "losses": int(np.count_nonzero(oriented_differences < -tolerance)),
                        "ties": int(np.count_nonzero(np.abs(oriented_differences) <= tolerance)),
                        "p_raw": permutation["p_value"],
                        "permutations": permutation["permutations"],
                        "bootstrap_reps": bootstrap_reps,
                        "bootstrap_seed": bootstrap_seed,
                    }
                )
        family_indices = list(range(family_start, len(rows)))
        adjusted = holm_adjust([rows[index]["p_raw"] for index in family_indices])
        for index, p_holm in zip(family_indices, adjusted, strict=True):
            rows[index]["p_holm"] = float(p_holm)
            rows[index]["reject_holm"] = bool(p_holm < alpha)
            rows[index]["alpha"] = alpha

    return AnalysisResult(summary=summary, significance=pd.DataFrame(rows))


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float | np.floating):
                cells.append("-" if not np.isfinite(value) else f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_analysis_reports(
    analysis: AnalysisResult,
    output_dir: str | Path,
    *,
    stem: str = "significance",
) -> dict[str, Path]:
    """Write machine-readable tables and a concise Markdown handoff report."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / f"{stem}_summary.csv"
    significance_path = directory / f"{stem}_tests.csv"
    report_path = directory / f"{stem}_report.md"
    analysis.summary.to_csv(summary_path, index=False)
    analysis.significance.to_csv(significance_path, index=False)

    report_lines = [
        "# Seed-level paired statistical analysis",
        "",
        "The test table uses an exact two-sided paired sign-flip permutation test. "
        "Positive oriented differences favor the target method; Holm correction is applied within each dataset/split/protocol family.",
        "",
    ]
    if "inference_warning" in analysis.significance.columns:
        warnings = analysis.significance["inference_warning"].dropna().unique().tolist()
        if warnings:
            report_lines.extend(["## Inference warning", "", *map(str, warnings), ""])
    if analysis.significance.empty:
        report_lines.append("No comparisons were requested.")
    else:
        columns = [
            "dataset",
            "split",
            "protocol",
            "metric",
            "target",
            "baseline",
            "n",
            "oriented_delta_mean",
            "ci95_low",
            "ci95_high",
            "p_raw",
            "p_holm",
            "reject_holm",
        ]
        report_lines.extend(_markdown_table(analysis.significance, columns))
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"summary": summary_path, "tests": significance_path, "report": report_path}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact paired significance tests from a canonical per-seed CSV."
    )
    parser.add_argument("--input", required=True, help="Canonical long per-seed result CSV.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for summary/test/report files."
    )
    parser.add_argument("--target", required=True, help="Proposed method to test.")
    parser.add_argument(
        "--baselines", nargs="+", required=True, help="Predeclared non-oracle baselines."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[DEFAULT_PRIMARY_METRIC],
        help=f"Metrics to test. Default: {DEFAULT_PRIMARY_METRIC}.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--strict-min-pairs",
        action="store_true",
        help="Reject a table whose seed count cannot attain a Holm-adjusted exact rejection.",
    )
    parser.add_argument("--stem", default="significance")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    minimum_pairs = minimum_pairs_for_exact_holm(
        args.alpha, len(args.baselines) * len(args.metrics)
    )
    results = pd.read_csv(args.input)
    inference_warning = None
    observed_seeds = int(results["seed"].nunique())
    if observed_seeds < minimum_pairs:
        inference_warning = (
            f"Exploratory analysis: {observed_seeds} paired seeds cannot attain a Holm-adjusted "
            f"two-sided exact rejection across {len(args.baselines) * len(args.metrics)} planned "
            f"hypotheses at alpha={args.alpha}. At least {minimum_pairs} seeds are required."
        )
        print(f"[STATISTICAL WARNING] {inference_warning}")
    analysis = analyze_paired_results(
        results,
        target=args.target,
        baselines=args.baselines,
        metrics=args.metrics,
        alpha=args.alpha,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        minimum_pairs=minimum_pairs if args.strict_min_pairs else None,
    )
    if inference_warning:
        analysis.significance["inference_warning"] = inference_warning
    paths = write_analysis_reports(analysis, args.output_dir, stem=args.stem)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
