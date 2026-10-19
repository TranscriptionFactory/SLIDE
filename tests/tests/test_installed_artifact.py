"""Exercise both native solvers; missing extensions must fail, not skip."""

import os
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
import pytest


def test_native_solvers():
    from slide_py.dsdp_solver import pydsdp5
    from slide_py.knockoff._vendor.glmnet import _glmnet, ElasticNet, LogitNet
    from slide_py.knockoff.solve import create_solve_sdp

    for module in (pydsdp5, _glmnet):
        assert any(str(module.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    sigma = 0.8 * np.eye(4) + 0.2 * np.ones((4, 4))
    s = create_solve_sdp(sigma, solver="dsdp")
    assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
    assert np.linalg.eigvalsh(2 * sigma - np.diag(s)).min() >= -1e-6
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 4))
    y = x[:, 0] + rng.normal(scale=0.1, size=40)
    for model, response in [
        (ElasticNet(n_splits=0, lambda_path=np.array([0.1])), y),
        (LogitNet(n_splits=0, lambda_path=np.array([0.1])), (y > 0).astype(int)),
    ]:
        model.fit(x, response)
        assert np.isfinite(model.coef_path_).all()
        assert np.any(model.coef_path_ != 0)


@pytest.mark.skipif(os.environ.get("SLIDE_TEST_INSTALLED") != "1", reason="artifact job only")
def test_import_is_installed():
    import slide_py
    from slide_py.dsdp_solver import pydsdp5
    from slide_py.knockoff._vendor.glmnet import _glmnet

    installed = Path(sysconfig.get_path("platlib")).resolve()
    for module in (slide_py, pydsdp5, _glmnet):
        assert Path(module.__file__).resolve().is_relative_to(installed)
    package = Path(slide_py.__file__).resolve().parent
    assert (package / "slide_r" / "getLatentFactors.R").is_file()


@pytest.mark.skipif(os.environ.get("SLIDE_TEST_EDITABLE") != "1", reason="Pixi source job only")
def test_import_is_editable():
    import slide_py
    from slide_py.dsdp_solver import pydsdp5
    from slide_py.knockoff._vendor.glmnet import _glmnet

    source = Path(__file__).resolve().parents[2] / "src" / "slide_py"
    prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
    assert Path(sys.prefix).resolve() == prefix
    for module in (slide_py, pydsdp5, _glmnet):
        assert Path(module.__file__).resolve().is_relative_to(source)
    # Linux exposes actual loaded libraries, not only linker search paths.
    if sys.platform == "linux":
        libraries = {
            Path(line.split()[-1]).resolve()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "openblas" in line.lower() and "/" in line
        }
        assert libraries, "No loaded OpenBLAS library found"
        assert all(library.is_relative_to(prefix) for library in libraries), libraries


def test_native_pipeline(tmp_path):
    from test_pipeline import _EXAMPLE_DIR, _run_pipeline

    _, output = _run_pipeline(
        tmp_path / "native-pipeline", "python",
        _EXAMPLE_DIR / "HIV+cytokines_X.csv",
        _EXAMPLE_DIR / "HIV+cytokines_y.csv",
        niter=2, do_interacts=False, knockoff_method="sdp",
    )
    for name in ("A.csv", "z_matrix.csv", "sig_LFs.txt", "love_result.pkl"):
        assert (output / name).is_file()
