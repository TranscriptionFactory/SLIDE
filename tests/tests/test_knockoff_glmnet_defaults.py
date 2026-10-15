import numpy as np

from loveslide.knockoff._vendor.glmnet import linear
from loveslide.knockoff.stats.glmnet import (
    _lasso_max_lambda_glmnet, _r_scale, stat_glmnet_lambdasmax,
)


def test_gaussian_knockoff_path_defaults_to_r_lambda_count():
    x = np.arange(24, dtype=float).reshape(8, 3)
    x_knockoff = x[:, ::-1] + np.array([0.1, -0.2, 0.3])
    y = np.arange(8, dtype=float)

    _, diagnostics = stat_glmnet_lambdasmax(
        x, x_knockoff, y, swap=np.zeros(3, dtype=bool), return_diagnostics=True
    )

    assert diagnostics["lambda"].shape == (500,)
    np.testing.assert_allclose(
        diagnostics["lambda"] / diagnostics["lambda"][0],
        (1 / 2000) ** (np.arange(500) / 500),
    )


def test_gaussian_algorithm_matches_r_variable_count_threshold(monkeypatch):
    real_elnet = linear.elnet
    selected_algorithms = []

    def record_algorithm(algo_flag, *args):
        selected_algorithms.append(algo_flag)
        return real_elnet(algo_flag, *args)

    monkeypatch.setattr(linear, "elnet", record_algorithm)
    rng = np.random.default_rng(17)
    base = rng.normal(size=(12, 3))
    for p in (499, 500):
        x = np.resize(base, (12, p))
        linear.ElasticNet(
            lambda_path=np.array([0.1]), n_splits=0, standardize=False
        ).fit(x, rng.normal(size=12))

    assert selected_algorithms == [1, 2]


def test_gaussian_preparation_matches_r_extended_reductions_and_scalar_power():
    x = np.array([[1e16, 0.1], [1, 0.2], [-1e16, 0.3], [4, 0.7]])
    y = np.array([1., 2., 4., 8.])
    # R 4.5.3 (Linux): scale(x), then the knockoff seven-point lambda formula.
    expected = np.array([
        [1.2247448713915889, -0.85552773820804517],
        [-3.0618621784789725e-17, -0.47529318789335845],
        [-1.2247448713915894, -0.095058637578671779],
        [3.3680483963268702e-16, 1.4258795636800752],
    ])
    scaled, means, _ = _r_scale(x, extended_precision=True)
    np.testing.assert_array_equal(means, [1.25, 0.325])
    np.testing.assert_array_equal(scaled, expected)
    _, diagnostic = _lasso_max_lambda_glmnet(x, y, nlambda=7, return_diagnostics=True)
    np.testing.assert_array_equal(diagnostic['standardized_design'], expected)
    np.testing.assert_allclose(diagnostic['lambda'], [
        2.3051719612827881, 0.77826520591901938, 0.26275555182751487,
        0.088710737022677191, 0.029950251510851286,
        0.010111713594871206, 0.0034138862502593077,
    ], rtol=2e-15, atol=0)
    # The existing non-Gaussian reduction remains unchanged.
    np.testing.assert_array_equal(_r_scale(x)[1], np.mean(x, axis=0))
