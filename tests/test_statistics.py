"""Unit tests for the seed-level paired statistical analysis helpers."""

import numpy as np
import pandas as pd
import pytest
from incremental.statistics import (
    DEFAULT_PRIMARY_METRIC,
    NEW_TASK_ACC_WEIGHT,
    OLD_TASK_ACC_WEIGHT,
    StatisticalInputError,
    add_derived_metrics,
    analyze_paired_results,
    holm_adjust,
    minimum_pairs_for_exact_holm,
    paired_bootstrap_ci,
    paired_exact_permutation,
)


def test_exact_permutation_has_expected_small_sample_resolution():
    result = paired_exact_permutation([1.0] * 5)
    assert result["permutations"] == 32
    assert result["p_value"] == pytest.approx(2 / 32)


def test_exact_permutation_with_ten_uniform_wins():
    result = paired_exact_permutation([0.01] * 10)
    assert result["permutations"] == 1024
    assert result["p_value"] == pytest.approx(2 / 1024)


def test_exact_permutation_all_ties_is_not_significant():
    result = paired_exact_permutation([0.0, 0.0, 0.0])
    assert result["p_value"] == 1.0


def test_paired_bootstrap_constant_difference_has_degenerate_interval():
    low, high = paired_bootstrap_ci([0.02] * 6, reps=100, seed=7)
    assert low == pytest.approx(0.02)
    assert high == pytest.approx(0.02)


def test_holm_adjustment_preserves_original_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_minimum_pairs_accounts_for_holm_family_size():
    assert minimum_pairs_for_exact_holm(0.05, hypotheses=1) == 6
    assert minimum_pairs_for_exact_holm(0.05, hypotheses=6) == 8


def _results_frame():
    rows = []
    for seed in range(1, 6):
        rows.extend(
            [
                {
                    "dataset": "toy",
                    "split": "random",
                    "protocol": "p1",
                    "seed": seed,
                    "method": "Ours",
                    "AUC_old": 0.80,
                    "AUC_new": 0.80 + seed * 0.001,
                    "ACC_old": 0.70,
                    "ACC_new": 0.70 + seed * 0.001,
                },
                {
                    "dataset": "toy",
                    "split": "random",
                    "protocol": "p1",
                    "seed": seed,
                    "method": "Baseline A",
                    "AUC_old": 0.75,
                    "AUC_new": 0.75 + seed * 0.001,
                    "ACC_old": 0.65,
                    "ACC_new": 0.65 + seed * 0.001,
                },
                {
                    "dataset": "toy",
                    "split": "random",
                    "protocol": "p1",
                    "seed": seed,
                    "method": "Baseline B",
                    "AUC_old": 0.76,
                    "AUC_new": 0.76 + seed * 0.001,
                    "ACC_old": 0.66,
                    "ACC_new": 0.66 + seed * 0.001,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_balanced_acc_is_the_default_weighted_primary_metric():
    frame = add_derived_metrics(_results_frame())

    assert DEFAULT_PRIMARY_METRIC == "Balanced_ACC"
    ours = frame.loc[(frame["method"] == "Ours") & (frame["seed"] == 1)].iloc[0]
    assert OLD_TASK_ACC_WEIGHT == pytest.approx(0.7)
    assert NEW_TASK_ACC_WEIGHT == pytest.approx(0.3)
    assert ours["Balanced_ACC"] == pytest.approx(0.7 * 0.70 + 0.3 * 0.701)


def test_balanced_acc_analysis_uses_prespecified_task_weights():
    analysis = analyze_paired_results(
        _results_frame(),
        target="Ours",
        baselines=["Baseline A", "Baseline B"],
        metrics=["Balanced_ACC"],
        bootstrap_reps=100,
        bootstrap_seed=9,
    )

    assert set(analysis.summary["metric"]) == {"Balanced_ACC"}
    assert (analysis.significance["oriented_delta_mean"] > 0).all()


def test_analysis_outputs_expected_schema_and_holm_values():
    analysis = analyze_paired_results(
        _results_frame(),
        target="Ours",
        baselines=["Baseline A", "Baseline B"],
        metrics=["Balanced_AUC"],
        bootstrap_reps=100,
        bootstrap_seed=9,
    )
    assert set(analysis.summary.columns) >= {"method", "metric", "n", "mean", "sd"}
    assert set(analysis.significance.columns) >= {
        "baseline",
        "oriented_delta_mean",
        "ci95_low",
        "ci95_high",
        "p_raw",
        "p_holm",
        "reject_holm",
    }
    assert (analysis.significance["n"] == 5).all()
    assert (analysis.significance["oriented_delta_mean"] > 0).all()
    assert np.allclose(analysis.significance["p_holm"], 0.125)


def test_analysis_enforces_requested_minimum_pair_count():
    with pytest.raises(StatisticalInputError, match="requires at least 6"):
        analyze_paired_results(
            _results_frame(),
            target="Ours",
            baselines=["Baseline A"],
            metrics=["Balanced_AUC"],
            bootstrap_reps=10,
            minimum_pairs=6,
        )


def test_analysis_rejects_duplicate_trial_keys():
    frame = _results_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(StatisticalInputError, match="must occur once"):
        analyze_paired_results(
            duplicated,
            target="Ours",
            baselines=["Baseline A"],
            metrics=["Balanced_AUC"],
            bootstrap_reps=10,
        )


def test_analysis_rejects_unequal_seed_sets_instead_of_inner_joining():
    frame = _results_frame()
    frame = frame.loc[~((frame["method"] == "Baseline A") & (frame["seed"] == 5))].copy()
    with pytest.raises(StatisticalInputError, match="Seed sets differ"):
        analyze_paired_results(
            frame,
            target="Ours",
            baselines=["Baseline A"],
            metrics=["Balanced_AUC"],
            bootstrap_reps=10,
        )
