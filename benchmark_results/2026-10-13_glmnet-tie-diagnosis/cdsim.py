"""Python re-implementations of Rust slide-glmnet run_gaussian_cd and Fortran glmnet elnet2 (naive),
with per-update tracing of selected columns. Shared loader for the 2026-10-13 s-zero draws."""
import numpy as np
from pathlib import Path

ROOT = Path("/vast/djishnu/Aaron/1_general_use/omni-SLIDE")
INP = ROOT / ".tmpfiles/2026-10-13_s-zero/inputs"
N, P, DRAWS = 24, 22, 200


def load(arm="rust_s"):
    z = np.fromfile(INP / "z.raw", "<f8").reshape(N, P)
    y = np.fromfile(INP / "y.raw", "<f8")
    xt = np.fromfile(INP / f"xt_{arm}.raw", "<f8").reshape(DRAWS, N, P)
    sw = np.fromfile(INP / "swaps.raw", np.uint8).reshape(DRAWS, P).astype(bool)
    return z, y, xt, sw


def augmented(z, xt, swap):
    left = np.where(swap, xt, z)
    right = np.where(swap, z, xt)
    return np.hstack([left, right])


def prep(xa, y):
    xc = xa - xa.mean(0)
    xs = xc / np.sqrt((xc ** 2).sum(0) / (len(y) - 1))  # R scale(), ddof 1
    return xs, y - y.mean()


def grid(xs, yc, nlambda=500):
    n = len(yc)
    lmax = np.abs(xs.T @ yc).max() / n
    return lmax * (1 / 2000) ** (np.arange(nlambda) / nlambda)


def rust_cd(xs, yc, lam, tol=1e-4, max_iter=1000, trace=(), active_order="index", sorted_cand=False):
    """Mirror of slide-glmnet run_gaussian_cd + coordinate_descent_step + sweep_coordinates.
    active_order: "index" = current Rust (active set = nonzero candidates in column order);
    "entry" = glmnet ia: ever-nonzero coordinates in first-entry order, never dropped.
    sorted_cand: keep the candidate (strong) set in column order after KKT additions, as glmnet's ix mask does."""
    n, p2 = xs.shape
    norm = (xs ** 2).sum(0) / n
    beta, r = np.zeros(p2), yc.copy()
    thr = tol * (yc @ yc) / n
    path = np.zeros((p2, len(lam)))
    log = []
    ia, in_ia = [], np.zeros(p2, bool)

    def sweep(idx, lv, li, phase):
        mx = 0.0
        for j in idx:
            old = beta[j]
            g = xs[:, j] @ r / n + old * norm[j]
            new = np.sign(g) * max(abs(g) - lv, 0.0) / norm[j]
            beta[j] = new
            d = old - new
            if d != 0.0:
                r[:] += d * xs[:, j]
                if not in_ia[j]:
                    in_ia[j] = True
                    ia.append(j)
            mx = max(mx, norm[j] * d * d)
            if j in trace:
                log.append((li, phase, j, g, old, new))
        return mx

    for li, lv in enumerate(lam):
        if li == 0:
            cand = list(range(p2))
        else:
            t = 2 * lv - lam[li - 1]
            cand = [j for j in range(p2) if abs(beta[j]) > 0 or abs(xs[:, j] @ r / n) >= t]
        inc = np.zeros(p2, bool)
        inc[cand] = True
        while True:
            act = list(ia) if active_order == "entry" else [j for j in cand if abs(beta[j]) > 0]
            for _ in range(max_iter):
                mc = 0.0
                if act:
                    mc = sweep(act, lv, li, "active")
                    if mc >= thr:
                        continue
                cc = sweep(cand, lv, li, "cand")
                act = list(ia) if active_order == "entry" else [j for j in cand if abs(beta[j]) > 0]
                if max(cc, mc) < thr:
                    break
            viol = False
            for j in range(p2):
                if not inc[j] and abs(xs[:, j] @ r / n) > lv:
                    cand.append(j)
                    inc[j] = True
                    viol = True
            if sorted_cand:
                cand.sort()
            if not viol:
                break
        path[:, li] = beta
    return path, log


def glmnet_naive(xs, yc, lam, thr=1e-7, maxit=100000, trace=()):
    """Mirror of glmnet5.f90 elnet2 (naive update, the type Python's vendored glmnet runs when 2p > n)
    after standard1 with isd=0, intr=1: x <- x_c/sqrt(n), y <- y_c/sqrt(n)/ys, lambda <- lambda/ys."""
    n, p2 = xs.shape
    x = xs / np.sqrt(n)
    yy = yc / np.sqrt(n)
    ys = np.sqrt(yy @ yy)
    yy = yy / ys
    xv = (x ** 2).sum(0)
    ulam = lam / ys
    a, ix, ia = np.zeros(p2), np.zeros(p2, int), []
    iz = False
    g = np.abs(x.T @ yy)
    path = np.zeros((p2, len(lam)))
    log = []
    alm = 0.0
    for m, alm_new in enumerate(ulam):
        alm0, alm = alm, alm_new
        tlam = 2 * alm - alm0
        ix[(ix == 0) & (g > tlam)] = 1
        skip_full = iz  # glmnet: if(iz*jz.ne.0) go to 10360 -- start each later lambda on the active set
        while True:
            if skip_full:
                skip_full = False
                dlx = thr  # fall through to the inner loop
            else:
              # full sweep over strong set (label 10880)
              dlx = 0.0
              for k in range(p2):
                if ix[k] == 0:
                    continue
                gk = yy @ x[:, k]
                ak = a[k]
                u = gk + ak * xv[k]
                v = abs(u) - alm
                a[k] = np.sign(u) * v / xv[k] if v > 0 else 0.0
                if k in trace:
                    log.append((m, "full", k, u * ys, ak * ys, a[k] * ys))
                if a[k] == ak:
                    continue
                if k not in ia:
                    ia.append(k)
                d = a[k] - ak
                yy = yy - d * x[:, k]
                dlx = max(xv[k] * d * d, dlx)
              if dlx < thr:
                ixx = False
                for k in range(p2):
                    if ix[k] == 1:
                        continue
                    g[k] = abs(yy @ x[:, k])
                    if g[k] > alm:
                        ix[k] = 1
                        ixx = True
                if ixx:
                    continue
                break
            # inner loop over ever-active set in entry order (label 10360/10990)
            iz = True
            while True:
                dlx = 0.0
                for k in list(ia):
                    gk = yy @ x[:, k]
                    ak = a[k]
                    u = gk + ak * xv[k]
                    v = abs(u) - alm
                    a[k] = np.sign(u) * v / xv[k] if v > 0 else 0.0
                    if k in trace:
                        log.append((m, "inner", k, u * ys, ak * ys, a[k] * ys))
                    if a[k] == ak:
                        continue
                    d = a[k] - ak
                    yy = yy - d * x[:, k]
                    dlx = max(xv[k] * d * d, dlx)
                if dlx < thr:
                    break
        path[:, m] = a * ys
    return path, log


def entry(path, lam):
    nz = path != 0
    return np.where(nz.any(1), lam[nz.argmax(1)], 0.0)
