import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from slide_py.knockoffs import Knockoffs
from slide_py.slide import OptimizeSLIDE, _format_interaction_pairs, _validate_knockoff_seed_range


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
    candidates = []

    def fake_select(**kwargs):
        seeds.append(kwargs["base_seed"])
        candidates.append(kwargs["z"])
        return np.arange(kwargs["z"].shape[1])

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
    z = slide.latent_factors.values
    np.testing.assert_array_equal(candidates[0], z[:, [0]] * z[:, [1, 2]])
    np.testing.assert_array_equal(candidates[1], z[:, [1]] * z[:, [0, 2]])
    np.testing.assert_array_equal(slide.interaction_pairs, [[0, 0, 1, 1], [1, 2, 0, 2]])
    np.testing.assert_array_equal(slide.interaction_terms, np.column_stack(candidates))


@pytest.mark.parametrize("base_seed", [10, None])
def test_interaction_retries_restart_with_fresh_reproducible_seeds(monkeypatch, base_seed):
    calls = []

    def fake_select(**kwargs):
        calls.append((kwargs["spec"], kwargs["base_seed"]))
        return np.array([[0], [0, 1, 2], [2], [3]][(len(calls) - 1) % 4])

    monkeypatch.setattr(Knockoffs, "select_short_freq", staticmethod(fake_select))
    slide = OptimizeSLIDE.__new__(OptimizeSLIDE)
    slide.latent_factors = pd.DataFrame(np.arange(20, dtype=float).reshape(4, 5))
    slide.marginal_idxs = np.array([1, 3])
    slide.data = SimpleNamespace(Y=pd.Series([0.0, 1.0, 2.0, 3.0]))

    for _ in range(2):
        slide.find_interaction_LFs(
            None, spec=0.1, fdr=0.1, niter=2, f_size=2,
            knockoff_backend="python", knockoff_base_seed=base_seed,
        )
        np.testing.assert_array_equal(slide.interaction_pairs, [[1, 3], [3, 4]])

    expected_seeds = [22, 34, 94, 106] if base_seed is not None else [None] * 4
    assert calls[:4] == list(zip([0.1, 0.1, 0.2, 0.2], expected_seeds))
    assert calls[4:] == calls[:4]


@pytest.mark.parametrize("base_seed", [10, None])
@pytest.mark.parametrize("spec, sizes, expected", [
    (0.1, [2], [0, 1]),
    (0.1, [4, 4, 2], [0, 1]),
    (0.9, [4, 4], []),
])
def test_marginal_retries_use_fresh_reproducible_seeds(
    monkeypatch, base_seed, spec, sizes, expected
):
    calls = []

    def fake_select(**kwargs):
        size = sizes[len(calls) % len(sizes)]
        calls.append((kwargs["spec"], kwargs["base_seed"]))
        return np.arange(size)

    monkeypatch.setattr(Knockoffs, "select_short_freq", staticmethod(fake_select))
    slide = OptimizeSLIDE.__new__(OptimizeSLIDE)
    z = pd.DataFrame(np.arange(20, dtype=float).reshape(4, 5))
    slide.data = SimpleNamespace(Y=pd.Series([0.0, 1.0, 2.0, 3.0]))

    for _ in range(2):
        slide.find_standalone_LFs(
            z, spec=spec, fdr=0.1, niter=2, f_size=2,
            knockoff_backend="python", knockoff_base_seed=base_seed,
        )
        np.testing.assert_array_equal(slide.marginal_idxs, expected)

    first = calls[:len(sizes)]
    assert [value for value, _ in first] == pytest.approx(
        [spec + 0.1 * retry for retry in range(len(sizes))]
    )
    expected_seeds = [10 + 72 * retry for retry in range(len(sizes))]
    assert [seed for _, seed in first] == (
        expected_seeds if base_seed is not None else [None] * len(sizes)
    )
    assert calls[len(sizes):] == first
    # Each pass reserves one marginal block followed by five interaction blocks.
    marginal_draws = {
        seed + draw for _, seed in first if seed is not None for draw in range(12)
    }
    interaction_draws = {
        10 + (retry * 6 + position + 1) * 12 + draw
        for retry in range(len(sizes)) for position in range(5) for draw in range(12)
    }
    assert marginal_draws.isdisjoint(interaction_draws)


@pytest.mark.parametrize("backend", ["python", "r_knockoffs"])
def test_knockoff_seed_range_boundaries(backend):
    limit = 2 ** (32 if backend == "python" else 31) - 1
    for seed in [None, 0, limit - 11]:
        _validate_knockoff_seed_range(seed, 12, backend)
    for seed in [-1, limit - 10]:
        with pytest.raises(ValueError, match="reserved block includes chunk and stage-two draws"):
            _validate_knockoff_seed_range(seed, 12, backend)
    _validate_knockoff_seed_range(limit + 1, 12, "r")


@pytest.mark.parametrize("backend", ["python", "r_knockoffs"])
@pytest.mark.parametrize("stage", ["marginal", "interaction"])
def test_retry_seed_overflow_fails_before_selection(monkeypatch, backend, stage):
    calls = []

    def fake_select(**kwargs):
        calls.append(kwargs["base_seed"])
        return np.arange(4 if len(calls) == 1 else 1)

    monkeypatch.setattr(Knockoffs, "select_short_freq", staticmethod(fake_select))
    slide = OptimizeSLIDE.__new__(OptimizeSLIDE)
    slide.latent_factors = pd.DataFrame(np.arange(20, dtype=float).reshape(4, 5))
    slide.marginal_idxs = np.array([0])
    slide.data = SimpleNamespace(Y=pd.Series([0.0, 1.0, 2.0, 3.0]))
    limit = 2 ** (32 if backend == "python" else 31) - 1
    base_seed = limit - 11 - (12 if stage == "interaction" else 0)
    args = dict(spec=0.1, fdr=0.1, niter=2, f_size=2,
                knockoff_backend=backend, knockoff_base_seed=base_seed)
    with pytest.raises(ValueError, match="seed range"):
        if stage == "marginal":
            slide.find_standalone_LFs(slide.latent_factors, **args)
        else:
            slide.find_interaction_LFs(None, **args)
    assert calls == [limit - 11]


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
