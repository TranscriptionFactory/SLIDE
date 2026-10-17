"""Pilot intervals must retain uncertainty even when ten observed runs agree."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


def test_comparison_mode_requires_test_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["analyze", str(tmp_path / "out"), "--arm", "r=pure_r:missing"])
    with pytest.raises(SystemExit) as error:
        analysis.main()
    assert error.value.code == 2

PATH = Path(__file__).parents[2] / "benchmarks" / "analyze_seed_replicates.py"
SPEC = importlib.util.spec_from_file_location("analyze_seed_replicates", PATH)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def runs(left, right):
    return pd.DataFrame([dict(arm=arm, cell="one", k=2, marginal=s, interaction=set(),
                              run=str(i), seed_dir="unused", backend=arm)
                         for arm, sets in (("a", left), ("r", right)) for i, s in enumerate(sets)])


def test_degenerate_agreement_does_not_certify_equivalence(monkeypatch):
    monkeypatch.setattr(analysis, "load_z", lambda _: (["Z1", "Z2"], np.ones((3, 2))))
    df = runs([set()] * 10, [set()] * 10)
    factors, summary = analysis.equivalence_intervals(df, [("a", "r")], 50, np.random.default_rng(1))
    assert len(factors) == 2  # Include factors never selected.
    assert set(factors.verdict) == {"inconclusive"}
    assert (factors.simultaneous_hi > 0.05).all()
    assert (summary.both_empty_pairs == 100).all()
    assert (summary.one_empty_pairs == 0).all()
    assert (summary.cross_jaccard == 1).all()
    assert summary.interval_scope.str.contains("not a verdict").all()


def test_extreme_difference_and_bootstrap_reproducibility(monkeypatch):
    monkeypatch.setattr(analysis, "load_z", lambda _: (["Z1", "Z2"], np.ones((3, 2))))
    df = runs([{"Z1"}] * 10, [set()] * 10)
    a, b = analysis.equivalence_intervals(df, [("a", "r")], 50, np.random.default_rng(2))
    aa, bb = analysis.equivalence_intervals(df, [("a", "r")], 50, np.random.default_rng(2))
    pd.testing.assert_frame_equal(a, aa)
    pd.testing.assert_frame_equal(b, bb)
    assert a.loc[a.factor == "Z1", "verdict"].item() == "different"
    marginal = b[b.kind == "marginal"].iloc[0]
    assert marginal.one_empty_pairs == 100
    assert marginal.overlap_deficit == 1
    assert marginal.count_difference == 1


def test_strict_alignment_rejects_permuted_factor_names(monkeypatch):
    monkeypatch.setattr(analysis, "load_z", lambda row: (
        ["Z1", "Z2"] if row.arm == "r" else ["Z2", "Z1"], np.ones((3, 2))))
    with pytest.raises(AssertionError, match="K/columns differ"):
        analysis.check_z(runs([set()] * 2, [set()] * 2), "r", 1e-8)


def test_all_empty_cells_retain_historical_comparison(monkeypatch):
    monkeypatch.setattr(analysis, "load_z", lambda _: (["Z1", "Z2"], np.ones((3, 2))))
    df = runs([set()] * 3, [set()] * 3).assign(
        n_marginal=0, n_interaction=0, wall_seconds=1.0, knockoff_seconds=1.0)
    _, factors = analysis.compare(df, [("a", "r")], 20, 20, np.random.default_rng(3))
    assert list(factors.factor) == ["Z1", "Z2"]
    assert (factors.a_rate == 0).all()


def test_truth_matching_sign_permutation_and_rank():
    truth = np.random.default_rng(5).normal(size=(40, 3))
    mapping, diag = analysis.match_truth(truth[:, [2, 0, 1]] * [-1, 1, -1], truth)
    assert mapping == {0: 2, 1: 0, 2: 1}
    assert diag['matched_abs_corr_min'] == pytest.approx(1)
    assert diag['max_principal_angle_degrees'] < 1e-10
    assert diag['unmatched_truth'] == diag['unmatched_estimated'] == 0
    _, rank_diag = analysis.match_truth(np.column_stack([truth[:, 0]] * 2), truth)
    assert rank_diag['estimated_rank'] == 1


def test_truth_unmatched_selection_bounds():
    truth = np.random.default_rng(6).normal(size=(100, 3))
    mapping, diag = analysis.match_truth(np.column_stack([truth[:, 0], np.ones(100)]), truth)
    assert mapping == {0: 0}
    assert diag['unmatched_estimated'] == 1
    bounds = analysis.truth_bounds({(0,), (1,)}, {(0,), (2,)}, mapping)
    assert bounds['fdp_lo'] == 0 and bounds['fdp_hi'] == 0.5
    assert bounds['power_lo'] == 0.5 and bounds['power_hi'] == 1


def test_truth_near_degenerate_cannot_certify_individual_factors():
    bounds = analysis.truth_bounds({(0,)}, {(0,)}, {0: 0}, identifiable=False)
    assert bounds['classification'] == 'inconclusive'
    assert bounds['fdp_lo'] == bounds['power_lo'] == 0
    assert bounds['fdp_hi'] == bounds['power_hi'] == 1


def test_truth_empty_null_and_identified_interactions():
    bounds = analysis.truth_bounds(set(), set(), {})
    assert bounds['fdp_lo'] == bounds['fdp_hi'] == 0
    assert np.isnan(bounds['power_lo']) and np.isnan(bounds['power_hi'])
    bounds = analysis.truth_bounds({(0, 1)}, {(0, 2)}, {0: 2, 1: 0})
    assert bounds['fdp_lo'] == bounds['fdp_hi'] == 0
    assert bounds['power_lo'] == bounds['power_hi'] == 1


def test_truth_summary_has_nonzero_uncertainty_at_zero_fdp(tmp_path, monkeypatch):
    import json
    truth = np.random.default_rng(7).normal(size=(40, 3))
    source = tmp_path / 'inputs' / 'ordinary_null' / 'seed_10000'
    source.mkdir(parents=True)
    np.savez(source / 'truth.npz', Z=truth)
    (source / 'truth.json').write_text(json.dumps(dict(near_degenerate=False,
        marginal_support=[], interaction_support=[])))
    monkeypatch.setattr(analysis, 'load', lambda _: pd.DataFrame([
        dict(seed=10000, cell='d=0.1 l=0.1', marginal=set(), interaction=set())]))
    monkeypatch.setattr(analysis, 'load_z', lambda _: (['Z1', 'Z2', 'Z3'], truth))
    datasets, summary = analysis.analyze_truth(
        [('rust', 'rust', [tmp_path / 'results' / 'ordinary_null' / 'rust'])], tmp_path / 'inputs')
    assert (datasets.fdp_exact == 0).all()
    fdp = summary[summary.endpoint.str.startswith('fdp')]
    assert (fdp['mean'] == 0).all() and (fdp.simultaneous_hi > 0).all()
    near = analysis.truth_bounds(set(), {(0,)}, {}, identifiable=False)
    assert near['fdp_exact'] == 0 and near['classification'] == 'inconclusive'


@pytest.mark.parametrize("identifiable", [True, False])
def test_global_null_nonempty_selection_is_false_even_without_matching(identifiable):
    bounds = analysis.truth_bounds({(0,), (1,)}, set(), {}, identifiable=identifiable)
    assert bounds['fdp_exact'] == bounds['fdp_lo'] == bounds['fdp_hi'] == 1
    assert bounds['unmatched_selected'] == 2
    assert np.isnan(bounds['power_lo'])


def test_projector_distance_detects_extra_dimensions_despite_zero_angles():
    truth = np.random.default_rng(8).normal(size=(40, 3))
    _, diag = analysis.match_truth(truth, truth[:, :1])
    assert diag['estimated_rank'] == 3 and diag['truth_rank'] == 1
    assert diag['max_principal_angle_degrees'] < 1e-10
    assert diag['normalized_projector_frobenius'] == pytest.approx(np.sqrt(2 / 3))


def test_uncertainty_plots_render_without_changing_tables(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(analysis, 'load_z', lambda _: (['Z1', 'Z2'], np.ones((3, 2))))
    factors, _ = analysis.equivalence_intervals(
        runs([{'Z1'}] * 10, [set()] * 10), [('a', 'r')], 20, np.random.default_rng(9))
    factors = pd.concat([factors, factors.assign(cell='empty', left_selected=0, difference=0,
                                               simultaneous_lo=-1, simultaneous_hi=1)], ignore_index=True)
    before = factors.copy(deep=True)
    analysis.plot_equivalence(factors, tmp_path)
    pd.testing.assert_frame_equal(before, factors)
    summary = pd.DataFrame([dict(cell='one', arm=arm, regime=regime, kind='marginal', endpoint=endpoint,
                                mean=mean, simultaneous_lo=0, simultaneous_hi=1)
                            for arm, regime in [('rust', 'ordinary_null'), ('python', 'near_sparse')]
                            for endpoint, mean in [('fdp_lo', 0), ('fdp_hi', 1),
                                                   ('power_lo', np.nan), ('power_hi', np.nan)]])
    before = summary.copy(deep=True)
    analysis.plot_truth(summary, tmp_path)
    pd.testing.assert_frame_equal(before, summary)
    images = list(tmp_path.glob('*_intervals_*.png'))
    assert len(images) == 3
    for path in images:
        with Image.open(path) as image:
            assert image.width > 500 and image.height > 300
            image.verify()



def test_truth_differences_pair_dataset_seeds_and_report_missing_coverage():
    data = pd.DataFrame([dict(arm=arm, seed=seed, regime="ordinary_sparse", cell="one", kind="marginal",
                              power_lo=lo, power_hi=hi, fdp_lo=lo, fdp_hi=hi)
                         for arm, seed, lo, hi in [("rust", 2, .6, .9), ("rust", 1, .1, .3),
                                                  ("rust", 3, 0, 1), ("pure_r", 1, .2, .4),
                                                  ("pure_r", 2, .3, .5), ("pure_r", 4, 0, 1)]])
    result = analysis.truth_backend_differences(data).set_index("endpoint")
    assert result.loc['power_difference_lo', 'mean'] == pytest.approx(-.1)
    assert result.loc['power_difference_hi', 'mean'] == pytest.approx(.35)
    assert (result.paired_datasets == 2).all() and (result.datasets == 2).all()
    assert (result.backend_only_seeds == '[3]').all()
    assert (result.reference_only_seeds == '[4]').all()
    assert (result.simultaneous_lo == -1).all() and (result.simultaneous_hi == 1).all()
