"""Probe scale equivariance of Python's knockoff path on a real δ=0.01 chunk."""
import sys
import numpy as np, pandas as pd
from loveslide.knockoff.filter import _prepare_knockoff_cache, _sample_knockoffs_from_cache
from loveslide.knockoff.stats.glmnet import stat_glmnet_lambdasmax

Z = pd.read_csv(sys.argv[1], index_col=0).values
y = pd.read_csv(sys.argv[2], index_col=0).values.ravel()
lo, hi = int(sys.argv[3]), int(sys.argv[4])
X = Z[:, lo:hi]
d = X.std(0, ddof=0)
Xs = (X - X.mean(0)) / d          # StandardScaler (old path)
craw = _prepare_knockoff_cache(X, method="asdp")
csc = _prepare_knockoff_cache(Xs, method="asdp")
v = np.diag(craw["Sigma"])
print("s/var raw vs scaled  max rel diff:",
      np.max(np.abs(craw["diag_s"]/v - csc["diag_s"]/np.diag(csc["Sigma"]))))
print("min s/var:", (craw["diag_s"]/v).min(), " Sigma_k chol escalation? L raw vs D L_sc:",
      np.max(np.abs(craw["L"] - np.diag(d) @ csc["L"])) / np.abs(craw["L"]).max())
Sk = 2*np.diag(craw["diag_s"]) - np.diag(craw["diag_s"]) @ craw["SigmaInv_s"]
print("Sigma_k eig min/max (raw):", np.linalg.eigvalsh((Sk+Sk.T)/2)[[0,-1]])
np.random.seed(7); Xk_raw = _sample_knockoffs_from_cache(X, craw)
np.random.seed(7); Xk_sc = _sample_knockoffs_from_cache(Xs, csc)
back = Xk_sc * d + X.mean(0)
print("knockoff draw raw vs rescaled scaled, max rel diff:", np.max(np.abs(Xk_raw-back))/np.abs(Xk_raw).max())
rng = np.random.default_rng(3); swap = rng.random(X.shape[1]) < .5
W1 = stat_glmnet_lambdasmax(X, Xk_raw, y, swap=swap)
W2 = stat_glmnet_lambdasmax(X, back, y, swap=swap)   # same knockoffs, raw scale
W3 = stat_glmnet_lambdasmax(Xs, Xk_sc, y, swap=swap) # same knockoffs, scaled
print("W raw-draw vs back-transformed draw:", np.max(np.abs(W1-W2)))
print("W same knockoffs raw vs scaled:", np.max(np.abs(W2-W3)), " scale of W:", np.abs(W2).max())
