"""Generate known-truth factor inputs without response-based factor matching."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n=100, p=500, k=10, seed=42, near_degenerate=False, null=False):
    if n < 3 or k < 3 or p < 3 * k:
        raise ValueError("Require n >= 3, k >= 3, and p >= 3*k")
    rng = np.random.default_rng(seed)
    A = np.zeros((p, k))
    A[np.arange(3 * k), np.repeat(np.arange(k), 3)] = 1
    rows = np.arange(3 * k, p)
    first = rng.integers(k, size=len(rows))
    second = (first + rng.integers(1, k, size=len(rows))) % k
    A[rows, first] = 0.6 * rng.choice([-1, 1], size=len(rows))
    A[rows, second] = 0.4 * rng.choice([-1, 1], size=len(rows))
    covariance = np.eye(k)
    Z = rng.standard_normal((n, k))
    if near_degenerate:
        Z[:, 1] = 0.999 * Z[:, 0] + np.sqrt(1 - 0.999**2) * Z[:, 1]
        covariance[0, 1] = covariance[1, 0] = 0.999
    noise = 0.3 * rng.standard_normal((n, p))
    y_noise = rng.standard_normal(n)
    y = y_noise.copy()
    if not null:
        y += 2 * Z[:, 0] + 1.5 * Z[:, 1] + 2 * Z[:, 0] * Z[:, 2]
    return {
        "X": Z @ A.T + noise, "y": y,
        "A": A, "Z": Z, "noise": noise, "y_noise": y_noise,
        "metadata": {
            "n": n, "p": p, "k": k, "seed": seed,
            "near_degenerate": near_degenerate, "null": null,
            "factor_covariance": covariance.tolist(),
            "feature_noise_sd": 0.3, "response_noise_sd": 1.0,
            "factor_index_base": 0,
            "marginal_support": [] if null else [0, 1],
            "interaction_support": [] if null else [[0, 2]],
            "response_equation": "y_noise" if null else "2*Z0 + 1.5*Z1 + 2*Z0*Z2 + y_noise",
            "loading_rule": "Three pure unit indicators per factor; remaining rows have two distinct factors with absolute weights 0.6 and 0.4 and independent random signs",
            "rng": "numpy.random.default_rng",
        },
    }


def write_inputs(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ids = [f"sample_{i}" for i in range(data["metadata"]["n"])]
    pd.DataFrame(data["X"], index=ids,
                 columns=[f"feature_{i}" for i in range(data["metadata"]["p"])]).to_csv(out / "X.csv", index_label="sample")
    pd.DataFrame({"Y": data["y"]}, index=ids).to_csv(out / "Y.csv", index_label="sample")
    np.savez_compressed(out / "truth.npz", **{
        key: data[key] for key in ("A", "Z", "noise", "y_noise")
    })
    (out / "truth.json").write_text(json.dumps(data["metadata"], indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-degenerate", action="store_true")
    parser.add_argument("--null", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args())
    out = args.pop("out")
    write_inputs(generate(**args), out)


if __name__ == "__main__":
    main()
