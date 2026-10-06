import pytest
import numpy as np

from benchmarks.analyze_paired_xtilde import aggregate_selection, gaussian_kkt, jaccard, prefix_stability, validate_hashes, zero_entry_ties
from benchmarks.replay_paired_xtilde import CHOLESKY_JITTER, CONTRASTS, contrast_summary, knockoff_factor, paired_vote_summary


def test_paired_vote_summary_preserves_pairing_and_direction():
    left = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=bool)
    same = paired_vote_summary(left, left)
    assert same["frequency_delta"].tolist() == [0, 0]
    assert same["paired_mcse"].tolist() == [0, 0]
    right = np.array([[0, 1], [0, 0], [0, 1], [0, 0]], dtype=bool)
    changed = paired_vote_summary(left, right)
    assert changed["frequency_delta"].tolist() == [0.5, 0]
    assert changed["rust_s_only"].tolist() == [2, 0]
    assert changed["r_s_only"].tolist() == [0, 0]
    assert changed["paired_mcse"].tolist() == pytest.approx([np.sqrt(1/12), 0])
    with pytest.raises(ValueError, match="equal draw-by-feature"):
        paired_vote_summary(left, right[:2])
    with pytest.raises(ValueError, match="at least two draws"):
        paired_vote_summary(left[:1], right[:1])


def test_tie_counterfactual_changes_statistics_not_original_fit():
    fit = {"w": [2.0, -3.0, 4.0], "entry_original": [2.0, 1.0, 4.0],
           "entry_knockoff": [2.0, 3.0, 2.0]}
    corrected = zero_entry_ties(fit, 0.1)
    assert corrected["w"] == [0.0, -3.0, 4.0]
    assert fit["w"] == [2.0, -3.0, 4.0]
    assert corrected["selected"] == [2]


def test_gaussian_kkt_distinguishes_solution_from_truncated_fit():
    x = np.array([[-1.0], [1.0]])
    y = np.array([1.0, 5.0])  # centered y = 2*x; nonzero intercept must not matter.
    lambdas = np.array([2.5, 1.0, 0.5])
    beta = np.array([[0.0, 1.0, 1.5]])
    assert gaussian_kkt(x, y, beta, lambdas)["max_kkt_absolute"] == pytest.approx(0)
    bad = gaussian_kkt(x, y, np.zeros_like(beta), lambdas)
    assert bad["max_kkt_absolute"] == pytest.approx(1.5)
    assert bad["max_kkt_over_lambda"] == pytest.approx(3.0)


def test_prefix_stability_reports_nested_counts_and_nonzero_uncertainty():
    draws = [{0} if i % 2 else set() for i in range(1000)]
    output = prefix_stability({"solver": draws}, 2, 0.1)["solver"]
    assert [row["draws"] for row in output] == [100, 250, 500, 1000]
    assert output[-1]["frequency"] == [0.5, 0.0]
    assert output[-1]["pointwise_95_low"][0] < 0.5 < output[-1]["pointwise_95_high"][0]
    assert output[-1]["pointwise_95_high"][1] > 0  # zero observations is not certainty.
    assert output[-1]["pointwise_95_high"][0] < output[0]["pointwise_95_high"][0]


def test_validate_hashes_enforces_every_paired_input():
    inputs = {
        "hashes": {"a.raw": "a", "z.raw": "z", "y.raw": "y"},
        "lf_names_sha256": "names",
    }
    manifest = {"iteration": "0", "xtilde_sha256": "x", "swap_sha256": "swap"}
    output = {
        "a_sha256": "a", "z_sha256": "z", "y_sha256": "y",
        "lf_names_sha256": "names", "xtilde_sha256": "x", "swap_sha256": "swap",
    }
    validate_hashes({"r": output}, manifest, inputs)
    with pytest.raises(RuntimeError, match="swap_sha256 mismatch"):
        validate_hashes({"r": output | {"swap_sha256": "wrong"}}, manifest, inputs)


def test_vote_aggregation_reproduces_find_opt_iter():
    result = aggregate_selection([{0, 2, 4}, {0, 2}, {0, 3}], feature_count=5, spec=0.5)
    assert result["votes"] == [3, 0, 2, 1, 1]
    assert result["frequent"] == [0, 2]
    assert result["findOptIter"] == {"selected": [0, 2], "optimal_iteration": 1}
    assert result["cumulative"][-1]["votes"] == result["votes"]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [({1, 2}, {2, 3}, 1 / 3), (set(), set(), 1.0), ({1}, set(), 0.0)],
)
def test_pairwise_jaccard(left, right, expected):
    assert jaccard(left, right) == pytest.approx(expected)


def test_knockoff_factor_keeps_zero_s_exact_and_matches_jitter_limit():
    sigma = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    s = np.array([0.4, 0.0, 0.5])
    target, exact = knockoff_factor(sigma, s, jitter=False)
    assert not exact[1].any() and not exact[:, 1].any() and not target[1].any()
    np.testing.assert_allclose(exact.T @ exact, target, atol=1e-15)
    jittered_target, jittered = knockoff_factor(sigma, s, jitter=True)
    np.testing.assert_allclose(jittered_target - target, CHOLESKY_JITTER * np.eye(3), atol=1e-16)
    assert jittered[1, 1] == pytest.approx(np.sqrt(CHOLESKY_JITTER))
    np.testing.assert_allclose(jittered, exact + np.sqrt(CHOLESKY_JITTER) * np.diag([0, 1, 0]), atol=1e-9)
    with pytest.raises(ValueError, match="non-negative"):
        knockoff_factor(sigma, np.array([0.4, -1e-12, 0.5]), jitter=False)
    with pytest.raises(ValueError, match="positive definite"):
        knockoff_factor(sigma, np.full(3, 2.0), jitter=False)


def test_contrast_summary_telescopes_and_cancels_identical_arms():
    rng = np.random.default_rng(3)
    selected = {arm: rng.random((50, 4)) < 0.3 for arm in {a for w in CONTRASTS.values() for a in w}}
    frame = contrast_summary(selected, CONTRASTS).set_index(["contrast", "feature_index"]).frequency_delta
    parts = frame["jitter_with_zeros"] + frame["zeros_exact"] + frame["nonzero_s_exact"]
    np.testing.assert_allclose(parts, frame["total"], atol=1e-15)
    nonzero = frame["tiny_s_exact"] + frame["substantive_s_exact"]
    np.testing.assert_allclose(nonzero, frame["nonzero_s_exact"], atol=1e-15)
    same = {arm: selected["rust_s_exact"] for arm in selected}
    flat = contrast_summary(same, CONTRASTS)
    assert (flat.frequency_delta == 0).all() and (flat.paired_mcse == 0).all() and (flat.nonzero_draws == 0).all()
