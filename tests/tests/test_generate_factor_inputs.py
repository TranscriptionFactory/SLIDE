import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location(
    "generate_factor_inputs", Path(__file__).parents[2] / "benchmarks/scripts/generate_factor_inputs.py"
)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.mark.parametrize("near_degenerate", [False, True])
@pytest.mark.parametrize("null", [False, True])
def test_truth_and_reproducibility(near_degenerate, null):
    args = dict(n=12, p=30, k=4, seed=19, near_degenerate=near_degenerate, null=null)
    data = generator.generate(**args)
    repeated = generator.generate(**args)
    for key in ("X", "y", "A", "Z", "noise", "y_noise"):
        np.testing.assert_array_equal(data[key], repeated[key])
    np.testing.assert_array_equal(data["X"], data["Z"] @ data["A"].T + data["noise"])
    expected = data["y_noise"].copy()
    if not null:
        z = data["Z"]
        expected += 2 * z[:, 0] + 1.5 * z[:, 1] + 2 * z[:, 0] * z[:, 2]
    np.testing.assert_array_equal(data["y"], expected)
    np.testing.assert_array_equal(data["A"][:12], np.repeat(np.eye(4), 3, axis=0))
    assert np.linalg.matrix_rank(data["A"][:12]) == 4
    assert np.all(np.count_nonzero(data["A"][12:], axis=1) == 2)
    np.testing.assert_allclose(np.abs(data["A"]).sum(axis=1), 1)
    assert data["metadata"]["factor_covariance"][0][1] == (0.999 if near_degenerate else 0)
    assert data["metadata"]["marginal_support"] == ([] if null else [0, 1])
    assert data["metadata"]["interaction_support"] == ([] if null else [[0, 2]])


def test_export_and_no_overwrite(tmp_path):
    data = generator.generate(n=12, p=15, k=3)
    out = tmp_path / "inputs"
    generator.write_inputs(data, out)
    x = pd.read_csv(out / "X.csv", index_col=0)
    y = pd.read_csv(out / "Y.csv", index_col=0)
    assert x.index.tolist() == y.index.tolist() == [f"sample_{i}" for i in range(12)]
    np.testing.assert_allclose(x.to_numpy(), data["X"])
    np.testing.assert_allclose(y["Y"], data["y"])
    with np.load(out / "truth.npz") as truth:
        for key in truth.files:
            np.testing.assert_array_equal(truth[key], data[key])
    assert json.loads((out / "truth.json").read_text()) == data["metadata"]
    before = (out / "X.csv").read_bytes()
    with pytest.raises(FileExistsError):
        generator.write_inputs(data, out)
    assert (out / "X.csv").read_bytes() == before


@pytest.mark.parametrize("args", [{"n": 2}, {"k": 2}, {"p": 29, "k": 10}])
def test_invalid_dimensions(args):
    with pytest.raises(ValueError):
        generator.generate(**args)
