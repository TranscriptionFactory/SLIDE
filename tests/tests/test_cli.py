"""Tests for the ``slide-py`` command-line entry point."""

import numpy as np
import pandas as pd
import pytest

from slide_py import __version__
from slide_py.cli import load_config, main


def _write_dataset(tmp_path, n=40, k=3, per=5):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(n, k))
    x = np.repeat(z, per, axis=1) + 0.3 * rng.normal(size=(n, k * per))
    y = (z[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    index = [f"s{i}" for i in range(n)]
    pd.DataFrame(x, index=index, columns=[f"g{j}" for j in range(k * per)]).to_csv(
        tmp_path / "x.csv"
    )
    pd.DataFrame({"y": y}, index=index).to_csv(tmp_path / "y.csv")


def _write_config(tmp_path, body):
    config = tmp_path / "config.yaml"
    config.write_text(body)
    return config


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("slide-py ")
    assert __version__ in out


def test_run_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "slide-py run" in out
    assert "--config" in out and "--out-path" in out


def test_load_config_resolves_paths_and_wraps_grid(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path, "x_path: x.csv\ny_path: /abs/y.csv\ndelta: 0.1\nlambda: [0.5, 1.0]\n"
    )
    monkeypatch.chdir(tmp_path.parent)

    params = load_config(config, {"out_path": "results", "x_path": None})

    assert params["x_path"] == str((tmp_path / "x.csv").resolve())
    assert params["y_path"] == "/abs/y.csv"
    assert params["out_path"] == str((tmp_path.parent / "results").resolve())
    assert params["delta"] == [0.1]
    assert params["lambda"] == [0.5, 1.0]


def test_load_config_rejects_non_mapping(tmp_path):
    config = _write_config(tmp_path, "- x_path\n- y_path\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(config)


@pytest.mark.slow
def test_run_end_to_end(tmp_path):
    """Full pipeline on 40 x 15 synthetic data (~2 min: LOVE CV plus scoring)."""
    _write_dataset(tmp_path)
    config = _write_config(
        tmp_path,
        "x_path: x.csv\n"
        "y_path: y.csv\n"
        "delta: 0.1\n"
        "lambda: 0.5\n"
        "niter: 20\n"
        "spec: 0.2\n"
        "fdr: 0.2\n"
        "knockoff_backend: python\n"
        "love_backend: python\n",
    )
    out = tmp_path / "out"

    assert main(["run", "--config", str(config), "--out-path", str(out)]) == 0

    run_dir = out / "0.1_0.5_out"
    assert (run_dir / "z_matrix.csv").exists()
    assert (run_dir / "run_params.txt").exists()
    assert (out / "summary_table.csv").exists()
