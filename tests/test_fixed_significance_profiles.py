"""Smoke tests for fixed-hyperparameter five-seed trial profiles."""

import argparse
import sys
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[1] / "GNCDM" / "experiments" / "_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import run_statistical_trials as trials  # noqa: E402


@pytest.mark.parametrize(
    ("dataset", "split", "ewc_lambda", "clora_lambda", "gncdm_lambda", "has_xder"),
    [
        ("math1", "random", 10_000, 10_000, 10.0, True),
        ("a0910", "random", 10_000, 10_000, 10.0, True),
        ("a0910", "user", 10_000, 10.0, 10.0, True),
        ("junyi", "random", 1_000, 10.0, 0.1, True),
    ],
)
def test_fixed_profiles_match_declared_hyperparameters(
    dataset, split, ewc_lambda, clora_lambda, gncdm_lambda, has_xder
):
    profile = trials._fixed_baseline_profile(dataset, split)
    assert profile["ewc_lambda"] == ewc_lambda
    assert profile["der_mem_size"] == 5_000
    assert profile["clora_lambda"] == clora_lambda
    assert profile["gncdm_clora_lambda"] == gncdm_lambda
    assert (profile["xder_mem_size"] is not None) is has_xder


def test_gncdm_clora_and_xder_keep_their_own_canonical_spaces():
    assert trials._canonical_method("C-LoRA-GNCDM(lambda=10)") == "C-LoRA-GNCDM"
    assert trials._canonical_method("C-LoRA-GNCDM (lambda=0.1)") == "C-LoRA-GNCDM"
    assert trials._canonical_method("X-DER (mem=5000)") == "X-DER"
    assert trials._method_family("C-LoRA-GNCDM")[2] == "gncdm_concept"
    assert trials._method_family("X-DER")[2] == "gncdm_concept"


def test_requested_roster_trains_lora_and_oracle_but_excludes_them_from_tests():
    profile = trials._fixed_baseline_profile("a0910", "random")
    assert profile["methods"] == trials.REQUESTED_METHODS
    assert "CLEAN-LoRA" in profile["methods"]
    assert "Full-Replay" in profile["methods"]
    assert trials._default_baselines(profile) == [
        "EWC",
        "DER++",
        "C-LoRA",
        "X-DER",
        "C-LoRA-GNCDM",
        "ICD",
    ]
    assert "CLEAN-LoRA" not in profile["comparison_methods"]
    assert "Full-Replay" not in profile["comparison_methods"]
    assert trials._canonical_method("Base") == "G-NCDM(Anchor)"
    assert trials._canonical_method("Ours (Dynamic DNA)") == "CLEAN-Full"
    assert trials._canonical_method("Ours (LoRA)") == "CLEAN-LoRA"
    assert trials._canonical_method("Full Replay Oracle") == "Full-Replay"
    assert trials._method_family("ICD")[2] == "icd_trait"
    assert trials.GNCDM_INCREMENTAL_STRATEGIES == {
        "Ours (Dynamic DNA)",
        "Ours (LoRA)",
        "Full Replay Oracle",
    }
    assert "Ours-Ablated" not in trials.GNCDM_INCREMENTAL_STRATEGIES
    assert "Naive FT (NFT)" not in trials.GNCDM_INCREMENTAL_STRATEGIES
    a0910_user = trials._fixed_baseline_profile("a0910", "user")
    assert a0910_user["xder_mem_size"] == 5_000
    assert a0910_user["methods"] == trials.REQUESTED_METHODS


def test_a0910_icd_command_preserves_the_fixed_split_and_device():
    args = argparse.Namespace(split="random", icd_python="/opt/icd/bin/python")
    command = trials._icd_command({"name": "a0910"}, args, trials.torch.device("cuda:2"))
    assert command[0] == "/opt/icd/bin/python"
    assert command[-3:] == ["cuda:2", "25", "random_split"]
    assert command[1].endswith("run_icd_a0910_A.py")


def test_five_seed_default_is_kept_for_the_fixed_profiles():
    assert trials.DEFAULT_SEEDS == [1, 7, 21, 42, 84]
