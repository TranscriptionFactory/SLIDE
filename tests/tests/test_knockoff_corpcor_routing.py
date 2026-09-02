from pathlib import Path
from unittest.mock import patch

import numpy as np

from loveslide.knockoff.create import create_second_order
from loveslide.knockoff.filter import (
    _cov_shrink_corpcor,
    _prepare_knockoff_cache,
    knockoff_filter_voting,
)
from loveslide.knockoff.solve import create_solve_asdp
from loveslide.knockoff._parallel import _precompute_knockoff_params
from loveslide.knockoffs import Knockoffs


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "knockoff_corpcor"
    / "high_delta_n24_p88.npz"
)


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def test_corpcor_covariance_matches_high_delta_r_fixture():
    fixture = np.load(FIXTURE)

    sigma = _cov_shrink_corpcor(fixture["z"])

    assert _relative_error(sigma, fixture["r_sigma"]) <= 1e-8


def test_asdp_diag_s_matches_high_delta_r_fixture():
    fixture = np.load(FIXTURE)
    sigma = _cov_shrink_corpcor(fixture["z"])

    diag_s = create_solve_asdp(sigma)

    assert _relative_error(diag_s, fixture["r_diag_s"]) <= 1e-5


def test_low_n_cache_uses_corpcor_and_never_ledoit_wolf():
    fixture = np.load(FIXTURE)

    with patch(
        "loveslide.knockoff.filter._cov_shrink_corpcor",
        wraps=_cov_shrink_corpcor,
    ) as corpcor, patch(
        "sklearn.covariance.LedoitWolf",
        side_effect=AssertionError("native Python routing must not use LedoitWolf"),
    ):
        cache = _prepare_knockoff_cache(fixture["z"], method="asdp")

    corpcor.assert_called_once()
    assert _relative_error(cache["Sigma"], fixture["r_sigma"]) <= 1e-8


def test_direct_second_order_uses_corpcor_and_never_ledoit_wolf():
    fixture = np.load(FIXTURE)

    with patch(
        "loveslide.knockoff.filter._cov_shrink_corpcor",
        wraps=_cov_shrink_corpcor,
    ) as corpcor, patch(
        "sklearn.covariance.LedoitWolf",
        side_effect=AssertionError("native Python routing must not use LedoitWolf"),
    ):
        create_second_order(fixture["z"], method="asdp")

    corpcor.assert_called_once()


def test_parallel_precompute_uses_corpcor_and_never_ledoit_wolf():
    fixture = np.load(FIXTURE)

    with patch(
        "loveslide.knockoff.filter._cov_shrink_corpcor",
        wraps=_cov_shrink_corpcor,
    ) as corpcor, patch(
        "sklearn.covariance.LedoitWolf",
        side_effect=AssertionError("native Python routing must not use LedoitWolf"),
    ):
        _, sigma, _ = _precompute_knockoff_params(fixture["z"], method="asdp")

    corpcor.assert_called_once()
    assert _relative_error(sigma, fixture["r_sigma"]) <= 1e-8


def test_legacy_iterative_python_uses_corpcor_and_never_ledoit_wolf():
    fixture = np.load(FIXTURE)
    y = np.zeros(fixture["z"].shape[0])

    with patch(
        "loveslide.knockoff.filter._cov_shrink_corpcor",
        wraps=_cov_shrink_corpcor,
    ) as corpcor, patch(
        "sklearn.covariance.LedoitWolf",
        side_effect=AssertionError("native Python routing must not use LedoitWolf"),
    ), patch(
        "loveslide.knockoffs._single_knockoff_iteration_python",
        return_value=[],
    ):
        Knockoffs.filter_knockoffs_iterative_python(
            fixture["z"], y, niter=1, n_jobs=1
        )

    corpcor.assert_called_once()


def test_well_conditioned_cache_keeps_sample_covariance():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(80, 8))

    with patch(
        "loveslide.knockoff.filter._cov_shrink_corpcor",
        wraps=_cov_shrink_corpcor,
    ) as corpcor:
        cache = _prepare_knockoff_cache(z, method="asdp")

    corpcor.assert_not_called()
    np.testing.assert_allclose(cache["Sigma"], np.cov(z, rowvar=False, ddof=1))


def test_cached_voting_is_deterministic_across_serial_and_parallel():
    rng = np.random.default_rng(11)
    z = rng.normal(size=(60, 6))
    y = rng.normal(size=60)

    def statistic(_x, xk, _y):
        return xk.mean(axis=0)

    kwargs = dict(
        statistic=statistic,
        niter=4,
        spec=0.5,
        base_seed=42,
        slide_selection=False,
    )
    serial = knockoff_filter_voting(z, y, n_jobs=1, **kwargs)
    parallel = knockoff_filter_voting(z, y, n_jobs=2, **kwargs)

    np.testing.assert_array_equal(serial.selection_counts, parallel.selection_counts)
    np.testing.assert_array_equal(serial.selected, parallel.selected)
