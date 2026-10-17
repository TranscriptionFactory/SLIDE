"""Score every saved draw with the Rust CD mirror (current index-order active set and the glmnet
entry-order variant) and the glmnet naive mirror; compare W to the recorded R glmnet W per draw.
usage: experiment.py <arm> <draws> <procs> <out_csv>"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from cdsim import *
sys.path.insert(0, str(ROOT / "src"))
from loveslide.knockoff.filter import knockoff_threshold

ARM, K, PROCS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
VARIANTS = {"rust_index_1e-4": ("index", 1e-4, 1000), "rust_entry_1e-4": ("entry", 1e-4, 1000),
            "rust_index_1e-7": ("index", 1e-7, 100000), "rust_entry_1e-7": ("entry", 1e-7, 100000)}
z, y, xt, sw = load(ARM)


def w_from_entry(e, swap):
    o, k = e[:P], e[P:]
    w = np.where(o == k, 0.0, np.maximum(o, k) * np.sign(o - k))
    return w * np.where(swap, -1, 1)


def one(dr):
    xs, yc = prep(augmented(z, xt[dr], sw[dr]), y)
    lam = grid(xs, yc)
    out = {name: w_from_entry(entry(rust_cd(xs, yc, lam, tol, it, active_order=o)[0], lam) * N, sw[dr])
           for name, (o, tol, it) in VARIANTS.items()}
    out["glmnet_mirror"] = w_from_entry(entry(glmnet_naive(xs, yc, lam)[0], lam) * N, sw[dr])
    return dr, out


if __name__ == "__main__":
    wr = pd.read_csv(INP / f"w_r_{ARM}.csv").to_numpy()
    rows = []
    with Pool(PROCS) as pool:
        for dr, out in pool.imap_unordered(one, range(K)):
            out["r_glmnet"] = wr[dr, :P]
            for name, w in out.items():
                t = knockoff_threshold(w, fdr=0.1, offset=0)
                for j in range(P):
                    rows.append(dict(arm=ARM, draw=dr, stat=name, factor=f"Z{j}", w=w[j], selected=bool(w[j] >= t)))
    pd.DataFrame(rows).sort_values(["draw", "stat", "factor"]).to_csv(OUT, index=False)
