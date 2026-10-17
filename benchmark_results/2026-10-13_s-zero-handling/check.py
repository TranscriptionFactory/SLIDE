"""Mechanism check: does s_j = 0 vs ~1e-9 change X~_j - X_j and W_j for Z20/Z21 (delta 0.01, lambda 1.0, chunk 0)?

usage: check.py <scratch_dir> <out_dir> <draws> <rust_bin> <rscript>
  (env: PYTHONPATH=src, R_LIBS with knockoff/glmnet; run under scripts/with-rust-blas-env.sh)

Python sampler (loveslide.knockoff.create.create_gaussian) for every arm; per draw d the innovations
(np.random.seed(SEED + d)) and swap mask (RandomState(SEED + 10**6 + d)) are shared by all arms.
Statistic: Python stat_glmnet_lambdasmax (nlambda 500) and the Rust paired_xtilde binary
(nlambda 500; tol/max_iter of the 2026-10-13 replay, 1e-7/1e5, and the production default 1e-4/1e3),
and R's knockoff:::lasso_max_lambda (glmnet, nlambda 500) on the same X~ and swap (r_stat.R).
"""
import json, logging, os, subprocess, sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy import linalg

from loveslide.knockoff.create import create_gaussian
from loveslide.knockoff.filter import knockoff_threshold
from loveslide.knockoff.stats.glmnet import stat_glmnet_lambdasmax

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
ROOT = Path(__file__).resolve().parents[2]
SCR, OUT, DRAWS, RUST_BIN, RSCRIPT = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
SEED, FDR = 20261013, 0.1
DESIGN = ROOT / ".tmpfiles/2026-10-13_s-only-d001/designs/marginal_chunk_1"
DIAG = ROOT / "benchmark_results/2026-10-13_clarabel-diag-s"
FOCUS = {"Z20": 20, "Z21": 21}
RUST_CFG = {"rust_replay": (1e-7, 100000), "rust_default": (1e-4, 1000)}

zf = pd.read_csv(DESIGN / "z_matrix.csv", index_col=0)
z = zf.to_numpy(float)
y = pd.read_csv(DESIGN / "Y.csv", index_col=0).iloc[:, 0].to_numpy(float)
n, p = z.shape
sigma = np.loadtxt(SCR.parent / "2026-10-13_clarabel-diag-s/sigma_1.0_0.csv", delimiter=",")
assert np.abs(sigma - np.cov(z, rowvar=False)).max() < 1e-12, "Sigma differs from np.cov(Z chunk 0)"
s_tab = pd.read_csv(DIAG / "s_lam1.0_chunk0.csv", index_col=0)
near_zero = s_tab["R"].to_numpy() / np.diag(sigma) < 1e-6
arms = {
    "rust_s": s_tab["Rust_default"].to_numpy(),
    "r_s": s_tab["R"].to_numpy(),
    "r_s_nearzero_to_0": np.where(near_zero, 0.0, s_tab["R"].to_numpy()),
    "rust_s_floor_1e-9": np.where(near_zero, np.maximum(s_tab["Rust_default"].to_numpy(), 1e-9),
                                  s_tab["Rust_default"].to_numpy()),
}

inputs = SCR / "inputs"
inputs.mkdir(parents=True, exist_ok=True)
for name, value in (("z.raw", z), ("y.raw", y), ("a.raw", np.zeros((1, p)))):
    np.asarray(value, dtype="<f8", order="C").tofile(inputs / name)
(inputs / "lf_names.txt").write_text("\n".join(zf.columns) + "\n")


def rust_w(xtilde, swap, tol, max_iter):
    xt, sw, js = inputs / "xtilde.raw", inputs / "swap.raw", inputs / "out.json"
    np.asarray(xtilde, dtype="<f8", order="C").tofile(xt)
    swap.astype(np.uint8).tofile(sw)
    env = os.environ | {"RAYON_NUM_THREADS": "1", "SLIDE_PAIRED_CACHED": "1"}
    subprocess.run([RUST_BIN, str(inputs / "z.raw"), str(xt), str(inputs / "y.raw"), str(inputs / "a.raw"),
                    str(sw), str(n), str(p), "1", str(p), str(inputs / "lf_names.txt"), str(FDR), "500",
                    str(tol), str(max_iter), str(js)], check=True, cwd=ROOT, env=env)
    out = json.loads(js.read_text())
    return np.array(out["w"]), out["threshold"], np.maximum(out["entry_original"], out["entry_knockoff"])


rows, arm_rows = [], []
for arm, s in arms.items():
    s_mat = np.diag(s)
    sigma_k = 2 * s_mat - s_mat @ linalg.solve(sigma, s_mat)
    try:
        linalg.cholesky(sigma_k, lower=True)
        jitter = False
    except linalg.LinAlgError:
        jitter = True
    shift = np.abs(z @ linalg.solve(sigma, s_mat))  # |mu_k - X| column contributions
    arm_rows.append(dict(arm=arm, chol_needs_jitter=jitter, n_exact_zero=int((s == 0).sum()),
                         **{f"s_{f}": s[j] for f, j in FOCUS.items()},
                         **{f"sigmak_{f}": sigma_k[j, j] for f, j in FOCUS.items()},
                         **{f"mushift_rel_{f}": np.linalg.norm(shift[:, j]) / np.linalg.norm(z[:, j])
                            for f, j in FOCUS.items()}))

xt_stack, swaps = {arm: [] for arm in arms}, []
for d in range(DRAWS):
    swap = np.random.RandomState(SEED + 10**6 + d).binomial(1, 0.5, p).astype(bool)
    for arm, s in arms.items():
        np.random.seed(SEED + d)
        xt = create_gaussian(z, z.mean(0), sigma, diag_s=s)
        xt_stack[arm].append(xt)
        wp, diag = stat_glmnet_lambdasmax(z, xt, y, nlambda=500, swap=swap, return_diagnostics=True)
        e = diag["entry_lambda_swapped"]
        w, entry = {"python": np.asarray(wp)}, {"python": np.maximum(e[:p], e[p:])}
        thr = {"python": knockoff_threshold(w["python"], fdr=FDR, offset=0)}
        for name, (tol, it) in RUST_CFG.items():
            w[name], t, entry[name] = rust_w(xt, swap, tol, it)
            thr[name] = np.inf if t is None else t
        for f, j in FOCUS.items():
            rel = np.linalg.norm(xt[:, j] - z[:, j]) / np.linalg.norm(z[:, j])
            for stat, wv in w.items():
                rows.append(dict(draw=d, arm=arm, factor=f, stat=stat, rel_diff=rel, swap=bool(swap[j]),
                                 w=wv[j], entry_max=entry[stat][j], selected=bool(wv[j] >= thr[stat])))
    swaps.append(swap)
    if (d + 1) % 50 == 0:
        logging.info("%d/%d draws", d + 1, DRAWS)

np.asarray(swaps, dtype=np.uint8).tofile(inputs / "swaps.raw")
for arm in arms:
    np.asarray(xt_stack[arm], dtype="<f8", order="C").tofile(inputs / f"xt_{arm}.raw")
    subprocess.run([RSCRIPT, str(Path(__file__).with_name("r_stat.R")), str(inputs), arm, str(DRAWS), str(n),
                    str(p), str(FDR)], check=True)
    wr = pd.read_csv(inputs / f"w_r_{arm}.csv").to_numpy()
    for d in range(DRAWS):
        for f, j in FOCUS.items():
            xt = xt_stack[arm][d]
            rows.append(dict(draw=d, arm=arm, factor=f, stat="r_glmnet",
                             rel_diff=np.linalg.norm(xt[:, j] - z[:, j]) / np.linalg.norm(z[:, j]),
                             swap=bool(swaps[d][j]), w=wr[d, j], entry_max=wr[d, p + 1 + j],
                             selected=bool(wr[d, j] >= wr[d, p])))
df = pd.DataFrame(rows)
df.to_csv(SCR / "draws.csv", index=False)
pd.DataFrame(arm_rows).to_csv(OUT / "arms.csv", index=False)
g = df.groupby(["arm", "factor", "stat"])
summ = pd.DataFrame({
    "rel_diff_median": g.rel_diff.median(), "rel_diff_max": g.rel_diff.max(),
    "frac_w_pos": g.w.apply(lambda v: (v > 0).mean()), "frac_w_zero": g.w.apply(lambda v: (v == 0).mean()),
    "frac_w_neg": g.w.apply(lambda v: (v < 0).mean()),
    "median_abs_w": g.w.apply(lambda v: v.abs().median()),
    "frac_entered": g.entry_max.apply(lambda v: (v > 0).mean()),
    "median_entry": g.entry_max.median(),
    "frac_sign_follows_swap": g.apply(lambda t: ((t.w > 0) == ~t.swap)[t.w != 0].mean(), include_groups=False),
    "frac_selected": g.selected.mean(),
}).reset_index()
summ.to_csv(OUT / "summary.csv", index=False)
# Paired agreement of W_j with the rust_s arm on the same draw.
piv = df.pivot_table(index=["draw", "factor", "stat"], columns="arm", values="w")
agree = pd.DataFrame({a: (np.sign(piv[a]) == np.sign(piv["rust_s"])).groupby(level=["factor", "stat"]).mean()
                      for a in arms}).reset_index()
agree.to_csv(OUT / "paired_sign_agreement_vs_rust_s.csv", index=False)
print(pd.DataFrame(arm_rows).to_string())
print(summ.to_string())
print(agree.to_string())
