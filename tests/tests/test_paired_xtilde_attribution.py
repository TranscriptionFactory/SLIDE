import pytest

from benchmarks.analyze_paired_xtilde import aggregate_selection, jaccard, validate_hashes


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
