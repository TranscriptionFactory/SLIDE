import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from loveslide.knockoffs import Knockoffs
from loveslide.slide import OptimizeSLIDE, _format_interaction_pairs


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

    module = importlib.import_module("loveslide.knockoff.filter")
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

    module = importlib.import_module("loveslide.knockoff.filter")
    monkeypatch.setattr(module, "knockoff_filter_voting", fake_voting)
    rng = np.random.default_rng(42)
    Knockoffs.select_short_freq(
        z=rng.normal(size=(24, 88)), y=rng.normal(size=24),
        f_size=24, niter=2, backend="python",
    )

    assert widths == [22, 22, 22, 22]
