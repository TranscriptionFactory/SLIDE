import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from slide_py.knockoffs import Knockoffs
from slide_py.slide import OptimizeSLIDE, _format_interaction_pairs


def test_interaction_cap_matches_r_on_fixed_draws(monkeypatch):
    from slide_py.knockoff.stats import glmnet
    from slide_py.knockoff.filter import knockoff_threshold

    if not glmnet.HAS_GLMNET:
        pytest.skip("R parity requires the vendored glmnet extension")
    fixture = Path(__file__).resolve().parents[1] / "fixtures/slide_core/interaction_cap"
    reference = json.loads((fixture / "r_reference.json").read_text())
    z = np.loadtxt(fixture / "z.csv", delimiter=",")
    y = np.loadtxt(fixture / "y.csv")
    specs = []

    def fixed_draw_selection(*, z, y, spec, **kwargs):
        specs.append(spec)
        np.testing.assert_allclose(y, reference["corrected_y"], atol=1e-14, rtol=0)
        counts = np.zeros(z.shape[1], dtype=int)
        for draw in range(4):
            xt = np.loadtxt(fixture / f"xtilde/xt_{draw}.csv", delimiter=",")
            swap = np.fromfile(fixture / f"xtilde/swap_{draw}.raw", np.uint8).astype(bool)
            w = glmnet.stat_glmnet_lambdasmax(z, xt, y, swap=swap, nlambda=30)
            counts += w >= knockoff_threshold(w, fdr=kwargs["fdr"], offset=0)
        np.testing.assert_array_equal(counts, reference["counts"])
        return np.flatnonzero(counts >= kwargs["niter"] * spec)

    monkeypatch.setattr(Knockoffs, "select_short_freq", staticmethod(fixed_draw_selection))
    slide = OptimizeSLIDE.__new__(OptimizeSLIDE)
    slide.latent_factors = pd.DataFrame(z)
    slide.marginal_idxs = np.array([0])
    slide.data = SimpleNamespace(Y=pd.Series(y))
    slide.find_interaction_LFs(None, spec=0.1, fdr=0.9, niter=4, f_size=100,
                              knockoff_backend="python", slide_selection=False)
    assert specs == reference["spec_ladder"]
    assert _format_interaction_pairs(slide.interaction_pairs) == reference["selected"]


def test_interaction_pairs_are_serialized_with_both_latent_factors():
    pairs = np.array([[44, 61], [59, 50]])

    assert _format_interaction_pairs(pairs) == ["Z44.Z59", "Z61.Z50"]


def test_interaction_subproblems_use_disjoint_seed_ranges(monkeypatch):
    seeds = []

    def fake_select(**kwargs):
        seeds.append(kwargs["base_seed"])
        return np.array([], dtype=int)

    monkeypatch.setattr(Knockoffs, "select_short_freq", staticmethod(fake_select))
    slide = OptimizeSLIDE.__new__(OptimizeSLIDE)
    slide.latent_factors = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3))
    slide.marginal_idxs = np.array([0, 1])
    slide.data = type("Data", (), {"Y": pd.Series([0.0, 1.0, 2.0, 3.0])})()

    slide.find_interaction_LFs(
        None, spec=0.1, fdr=0.1, niter=2, f_size=2,
        knockoff_backend="python", knockoff_base_seed=10,
    )

    assert seeds == [18, 26]


@pytest.mark.parametrize("backend", ["python", "r_knockoffs"])
@pytest.mark.parametrize("n_lfs, expected_widths", [(88, [87]), (173, [86, 86])])
def test_interaction_chunks_match_r_default(monkeypatch, backend, n_lfs, expected_widths):
    calls = []

    def fake_voting(X, y, **kwargs):
        calls.append((X.shape[1], kwargs["base_seed"]))
        return SimpleNamespace(selected=np.array([], dtype=int))

    module = importlib.import_module("slide_py.knockoff.filter")
    monkeypatch.setattr(module, "knockoff_filter_voting", fake_voting)
    slide = OptimizeSLIDE.__new__(OptimizeSLIDE)
    rng = np.random.default_rng(42)
    slide.latent_factors = pd.DataFrame(rng.normal(size=(24, n_lfs)))
    slide.marginal_idxs = np.array([5])
    slide.data = SimpleNamespace(Y=pd.Series(rng.normal(size=24)))

    slide.find_interaction_LFs(
        None, spec=0.1, fdr=0.1, niter=2, f_size=24,
        knockoff_backend=backend, knockoff_base_seed=42,
    )

    assert [width for width, _ in calls] == expected_widths
    assert [seed for _, seed in calls] == [
        42 + 2 * (n_lfs + 1) + 2 * i for i in range(len(expected_widths))
    ]


def test_marginal_chunks_still_use_supplied_size(monkeypatch):
    widths = []

    def fake_voting(X, y, **kwargs):
        widths.append(X.shape[1])
        return SimpleNamespace(selected=np.array([], dtype=int))

    module = importlib.import_module("slide_py.knockoff.filter")
    monkeypatch.setattr(module, "knockoff_filter_voting", fake_voting)
    rng = np.random.default_rng(42)
    Knockoffs.select_short_freq(
        z=rng.normal(size=(24, 88)), y=rng.normal(size=24),
        f_size=24, niter=2, backend="python",
    )

    assert widths == [22, 22, 22, 22]
