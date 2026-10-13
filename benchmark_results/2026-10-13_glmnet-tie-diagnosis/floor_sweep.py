"""s-floor sweep: where does R glmnet stop tying a knockoff with its original?

For each relative floor f, the 17 near-zero s entries of chunk 0 (delta 0.01, lambda 1.0) are set to f * Sigma_jj
and the other entries are shrunk by the smallest g in GRID_G that keeps 2 Sigma - diag(s) PD (R's own fix-up is a
shrink too). 200 X~ draws per floor with the s-zero innovations/swaps (seeds of ../2026-10-13_s-zero-handling/check.py).
Each draw is scored by R knockoff:::lasso_max_lambda (r_stat.R), the current Rust paired_xtilde binary (tol 1e-4 and 1e-7)
and the entry-order Rust mirror (cdsim.rust_cd, tol 1e-7). Rows cover all 22 factors (W thresholds need the full vector). Per pair it records the standardized distance
delta_j = sqrt(2 (1 - corr(X_j, X~_j))) and the entry grid-step gap of the Rust fit.

usage: floor_sweep.py <scratch_dir> <draws> <procs> <rust_bin> <rscript>   (env: PYTHONPATH=src, R_LIBS)
"""
import json, logging, os, subprocess, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd

from loveslide.knockoff.create import create_gaussian
import cdsim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCR, DRAWS, PROCS, RUST_BIN, RSCRIPT = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
SEED, FDR, N, P = 20261013, 0.1, cdsim.N, cdsim.P
FLOORS = [1e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4]
GRID_G = [0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2]
STEP = np.log(2000) / 500  # log-lambda grid spacing (nlambda 500, ratio 1/2000)

src = ROOT / ".tmpfiles/2026-10-13_s-zero/inputs"
inputs = SCR / "inputs"
inputs.mkdir(parents=True, exist_ok=True)
for f in ("z.raw", "y.raw", "a.raw", "lf_names.txt", "swaps.raw"):
    (inputs / f).write_bytes((src / f).read_bytes())
z, y, _, sw = cdsim.load("rust_s")
sigma = np.loadtxt(ROOT / ".tmpfiles/2026-10-13_clarabel-diag-s/sigma_1.0_0.csv", delimiter=",")
s_tab = pd.read_csv(ROOT / "benchmark_results/2026-10-13_clarabel-diag-s/s_lam1.0_chunk0.csv", index_col=0)
diag = np.diag(sigma)
near_zero = s_tab["R"].to_numpy() / diag < 1e-6
base = s_tab["Rust_default"].to_numpy()


def floored_s(f):
    for g in GRID_G:
        s = np.where(near_zero, f * diag, base * (1 - g))
        if np.linalg.eigvalsh(2 * sigma - np.diag(s)).min() > 1e-9:
            return s, g
    raise RuntimeError(f"no feasible shrink for floor {f}")


def rust_bin(xt, swap, tol, max_iter):
    d = SCR / f"bin_{os.getpid()}"
    d.mkdir(exist_ok=True)
    np.asarray(xt, "<f8", order="C").tofile(d / "xt.raw")
    swap.astype(np.uint8).tofile(d / "sw.raw")
    env = os.environ | {"RAYON_NUM_THREADS": "1", "SLIDE_PAIRED_CACHED": "1"}
    subprocess.run([RUST_BIN, str(inputs / "z.raw"), str(d / "xt.raw"), str(inputs / "y.raw"), str(inputs / "a.raw"),
                    str(d / "sw.raw"), str(N), str(P), "1", str(P), str(inputs / "lf_names.txt"), str(FDR), "500",
                    str(tol), str(max_iter), str(d / "out.json")], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
    out = json.loads((d / "out.json").read_text())
    return np.array(out["w"]), np.array(out["entry_original"]), np.array(out["entry_knockoff"])


def score(args):
    d, xt = args
    w_rust, eo, ek = rust_bin(xt, sw[d], 1e-4, 1000)
    w_rust7 = rust_bin(xt, sw[d], 1e-7, 1000)[0]
    xs, yc = cdsim.prep(cdsim.augmented(z, xt, sw[d]), y)
    lam = cdsim.grid(xs, yc)
    e = cdsim.entry(cdsim.rust_cd(xs, yc, lam, 1e-7, 100000, active_order="entry")[0], lam)
    o, k = e[:P], e[P:]
    w_entry = np.where(o == k, 0.0, np.maximum(o, k) * np.sign(o - k)) * np.where(sw[d], -1, 1)
    gap = np.where((eo > 0) & (ek > 0), np.abs(np.log(np.where(eo > 0, eo, 1) / np.where(ek > 0, ek, 1))) / STEP, np.inf)
    return d, w_rust, w_rust7, w_entry, np.rint(gap)


rows, arm_rows = [], []
for f in FLOORS:
    s, g = floored_s(f)
    xts = []
    for d in range(DRAWS):
        np.random.seed(SEED + d)
        xts.append(create_gaussian(z, z.mean(0), sigma, diag_s=s))
    xts = np.asarray(xts)
    arm = f"floor_{f:g}"
    xts.astype("<f8").tofile(inputs / f"xt_{arm}.raw")
    subprocess.run([RSCRIPT, str(HERE.parent / "2026-10-13_s-zero-handling/r_stat.R"), str(inputs), arm, str(DRAWS),
                    str(N), str(P), str(FDR)], check=True)
    wr = pd.read_csv(inputs / f"w_r_{arm}.csv").to_numpy()
    zc = (z - z.mean(0)) / z.std(0)
    with Pool(PROCS) as pool:
        for d, w_rust, w_rust7, w_entry, gap in pool.imap_unordered(score, list(enumerate(xts))):
            xc = (xts[d] - xts[d].mean(0)) / xts[d].std(0)
            delta = np.sqrt(np.maximum(2 * (1 - (zc * xc).mean(0)), 0))
            for j in range(P):
                rows.append(dict(floor=f, shrink=g, draw=d, factor=f"Z{j}", near_zero=near_zero[j],
                                 s_rel=s[j] / diag[j], delta=delta[j], w_r=wr[d, j], r_entry=wr[d, P + 1 + j],
                                 w_rust=w_rust[j], w_rust7=w_rust7[j], w_entry=w_entry[j], rust_gap_steps=gap[j]))
    logging.info("floor %g (shrink %g) done", f, g)
pd.DataFrame(rows).to_csv(SCR / "floor_sweep.csv", index=False)
