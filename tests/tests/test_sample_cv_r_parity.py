"""Parity checks against deterministic R sampleCV reference predictions."""

import json
from pathlib import Path

import numpy as np

from slide_py.sample_cv import sample_cv


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures/sample_cv/2026-11-01_r-reference/sample_cv.json"
)


def test_sample_cv_matches_r_reference():
    cases = json.loads(FIXTURE.read_text())["cases"]
    for case in cases:
        expected = case["expected"]
        result = sample_cv(
            np.asarray(case["z"], dtype=float),
            np.asarray(case["y"], dtype=float),
            case["marginal_indices"],
            np.asarray(case["interaction_pairs"], dtype=int).T
            if case["interaction_pairs"]
            else None,
            condition=case["condition"],
            k=4,
            iterations=max(1, len(case["splits"])),
            train_splits=[split["train_indices"] for split in case["splits"]]
            if case["splits"]
            else None,
        )
        assert result["guard"] == expected["guard"], case["name"]
        if expected["metric"] is None:
            assert result["metric"] is None, case["name"]
        else:
            assert np.isclose(result["metric"], expected["metric"], atol=1e-10), case["name"]
        assert result["last_rank"] == expected["last_rank"], case["name"]
        assert result["last_aliased_terms"] == expected["last_aliased_terms"], case["name"]
        if expected["guard"] == "ok":
            assert result["last_train_indices"] == expected["last_train_indices"], case["name"]
            assert result["term_order"] == expected["term_order"], case["name"]
            reference_coefficients = list(expected["last_coefficients"].values())
            assert len(result["last_coefficients"]) == len(reference_coefficients), case["name"]
            for actual, target in zip(result["last_coefficients"], reference_coefficients):
                if target is None:
                    assert actual is None, case["name"]
                else:
                    assert np.isclose(actual, target, atol=1e-9), case["name"]
        observed = [
            (row["iteration"], row["row_index"], row["y_true"], row["y_pred"])
            for row in result["predictions"]
        ]
        reference = [
            (row["iteration"], row["row_index"], row["y_true"], row["y_pred"])
            for row in expected["predictions"]
        ]
        assert len(observed) == len(reference), case["name"]
        for actual, target in zip(observed, reference):
            assert actual[:3] == target[:3], case["name"]
            assert np.isclose(actual[3], target[3], atol=1e-9), case["name"]


def test_sample_cv_empty_selection_is_unscorable():
    result = sample_cv(np.arange(8, dtype=float)[:, None], np.arange(8, dtype=float), [])
    assert result["guard"] == "no_marginals"
    assert result["metric"] is None
    assert result["predictions"] == []
