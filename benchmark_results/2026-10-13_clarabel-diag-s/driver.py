"""Per-chunk diag_s at delta=0.01: R DSDP vs Python DSDP vs Rust Clarabel (tol 1e-6 default, 1e-8, 1e-10).

usage: driver.py <scratch_dir> <out_dir>   (env: R_LIBS, SLIDE_R_PY_ENV for Rscript path, PYTHONPATH=src)
"""
import os, subprocess, sys
import numpy as np, pandas as pd
from loveslide.knockoff.solve import create_solve_asdp

SCR, OUT = sys.argv[1], sys.argv[2]
REPO = os.getcwd()
PAR = f"{REPO}/.tmpfiles/2026-10-12_python-scale-bug"
ENV = "/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev"
ZP = {"0.1": f"{PAR}/results/python/seed_42/python/0.01_0.1_out/z_matrix.csv",
      "1.0": f"{PAR}/results/python/seed_42/python/0.01_1.0_out/z_matrix.csv"}
CH, NCH = 22, 8
TOLS = ["default", "1e-8", "1e-9", "1e-10"]
hx = float.fromhex

def chunk_cols(i, p):
    return range(i * CH, min(i * CH + CH, p))

# 1. covariances. lambda=0.1: R's own hex Sigma from the sdp-parity run; lambda=1.0: np.cov of the chunk.
sig, names = {}, {}
for lam, zp in ZP.items():
    z = pd.read_csv(zp, index_col=0)
    names[lam] = list(z.columns)
    for i in range(NCH):
        X = z.values[:, list(chunk_cols(i, z.shape[1]))]
        if lam == "0.1":
            S = np.array([[hx(t) for t in l.split()] for l in open(f"{PAR}/rSig_{i}.txt")])
            print(f"lam0.1 chunk{i}: max|rSig-np.cov|={np.abs(S-np.cov(X,rowvar=False)).max():.1e}")
        else:
            S = np.cov(X, rowvar=False)
        sig[lam, i] = S
        np.savetxt(f"{SCR}/sigma_{lam}_{i}.csv", S, delimiter=",", fmt="%.17g")

# 2. R DSDP
subprocess.run([f"{ENV}/bin/Rscript", f"{OUT}/r_solve.R", SCR], check=True)

# 3. Rust Clarabel, one test run per chunk per tolerance
rust_env = dict(os.environ)
FAILED = set()
for lam in ZP:
    for i in range(NCH):
        for tol in TOLS:
            e = dict(rust_env, SLIDE_S_SIGMA_CSV=f"{SCR}/sigma_{lam}_{i}.csv",
                     SLIDE_S_OUT=f"{SCR}/sRust_{tol}_{lam}_{i}.txt")
            if tol != "default":
                e["SLIDE_S_TOL"] = tol
            r = subprocess.run(["bash", "scripts/with-rust-blas-env.sh", "cargo", "test", "--release", "-p",
                            "slide-knockoff", "emit_diag_s", "--", "--ignored"],
                           env=e, capture_output=True, text=True)
            if r.returncode:
                msg = [l for l in r.stdout.splitlines() if "SDP solver failed" in l]
                print(f"rust FAILED lam={lam} chunk={i} tol={tol}: {msg[:1]}")
                FAILED.add((lam, i, tol))

# 4. tables
def rd(p):
    return np.loadtxt(p, ndmin=1)
rows, factors = [], []
for lam in ZP:
    for i in range(NCH):
        S = sig[lam, i]
        sol = {"R": rd(f"{SCR}/sR_{lam}_{i}.csv"), "Py": np.asarray(create_solve_asdp(S)).ravel()}
        for tol in TOLS:
            if (lam, i, tol) not in FAILED:
                sol[f"Rust_{tol}"] = rd(f"{SCR}/sRust_{tol}_{lam}_{i}.txt")
        cols = list(chunk_cols(i, len(names[lam])))
        for k, c in enumerate(cols):
            if names[lam][c] in ("Z20", "Z21", "Z73", "Z74"):
                factors.append(dict(lam=lam, chunk=i, factor=names[lam][c], local_idx=k,
                                    **{f"s_{m}": v[k] for m, v in sol.items()}))
        for a, b in [("Py", "R")] + [(f"Rust_{t}", "R") for t in TOLS] + [(f"Rust_{t}", "Py") for t in TOLS] \
                + [(f"Rust_{t}", f"Rust_{u}") for t, u in [("1e-8", "default"), ("1e-9", "1e-8")]]:
            if a not in sol or b not in sol:
                continue
            sa, sb = sol[a], sol[b]
            big = sb > 1e-3
            rel = np.abs(sa - sb)[big] / sb[big]
            rows.append(dict(lam=lam, chunk=i, a=a, b=b, n=len(sa),
                             max_abs=np.abs(sa - sb).max(),
                             max_abs_corrunits=np.abs((sa - sb) / np.diag(S)).max(),
                             max_rel_gt1e3=rel.max() if rel.size else np.nan, n_gt1e3=int(big.sum()),
                             nz_a=int((sa / np.diag(S) < 1e-6).sum()), nz_b=int((sb / np.diag(S) < 1e-6).sum()),
                             nz_ratio=(sa / np.diag(S)).sum() / max((sb / np.diag(S)).sum(), 1e-300) if False else np.nan))
        # near-zero ratio: median Rust/R over entries with R corr-unit s < 1e-6
        v = np.diag(S); tiny = sol["R"] / v < 1e-6
        for m in sol:
            if m == "R":
                continue
            q = sol[m][tiny] / sol["R"][tiny]
            rows.append(dict(lam=lam, chunk=i, a=m, b="R", n=int(tiny.sum()), nz_a=int((sol[m] / v < 1e-6).sum()),
                             nz_b=int(tiny.sum()), nz_ratio=np.nan, tiny_ratio_median=np.median(q) if q.size else np.nan,
                             tiny_ratio_min=q.min() if q.size else np.nan, tiny_ratio_max=q.max() if q.size else np.nan,
                             max_abs=np.nan, max_abs_corrunits=np.nan, max_rel_gt1e3=np.nan, n_gt1e3=-1))
        pd.DataFrame({m: v for m, v in sol.items()}).to_csv(f"{OUT}/s_lam{lam}_chunk{i}.csv", index_label="local_idx")
pd.DataFrame(rows).to_csv(f"{OUT}/pair_stats.csv", index=False)
pd.DataFrame(factors).to_csv(f"{OUT}/factors_Z20_Z21_Z73_Z74.csv", index=False)
