import numpy as np

from loveslide.knockoff.stats.glmnet import stat_glmnet_lambdasmax


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
