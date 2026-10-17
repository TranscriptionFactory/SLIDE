"""Does keeping the candidate set in column order (glmnet ix) add agreement beyond the entry-order active set?
usage: sorted_check.py <draws>  (rust_s arm, tol 1e-7)"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from cdsim import *


def w_from_entry(e, swap):
    o, k = e[:P], e[P:]
    return np.where(o == k, 0.0, np.maximum(o, k) * np.sign(o - k)) * np.where(swap, -1, 1)
z, y, xt, sw = load("rust_s")
def one(dr):
    xs, yc = prep(augmented(z, xt[dr], sw[dr]), y); lam = grid(xs, yc)
    return [w_from_entry(entry(rust_cd(xs, yc, lam, 1e-7, 100000, active_order="entry", sorted_cand=s)[0], lam), sw[dr]) for s in (False, True)]
if __name__ == "__main__":
    K = int(sys.argv[1]); wr = pd.read_csv(INP / "w_r_rust_s.csv").to_numpy()[:K, :P]
    with Pool(4) as pool: res = pool.map(one, range(K))
    for i, name in enumerate(("entry", "entry+sorted_cand")):
        w = np.array([r[i] for r in res]); print(name, "all-factor sign agreement with R:", (np.sign(w) == np.sign(wr)).mean().round(4))
