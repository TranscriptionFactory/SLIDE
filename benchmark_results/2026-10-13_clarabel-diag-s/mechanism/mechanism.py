"""Why does Clarabel return 0 where DSDP returns 1e-10..1e-8?  usage: mechanism.py <scratch> <mech_scratch> <out_dir>
Reproduces crates/slide-sdp (correlation SDP, same cones/scaling/settings) with the clarabel Python package,
records raw/clipped/final s, duals, DSDP gaptol sweep, and a high-accuracy reference."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp, clarabel, cvxpy as cp
from loveslide.knockoff.solve import create_solve_sdp as py_dsdp
SCR, MS, OUT = sys.argv[1:4]
FLOOR, CLIPTOL = 1e-10, 1e-12
D = 22

def clip(v):
    if v < 0 and v > -CLIPTOL: return 0.0
    if v > 1 and v < 1 + CLIPTOL: return 1.0
    return min(max(v, 0.0), 1.0)

def diag_idx(c): return c * (c + 3) // 2

def rhs_of(C):
    d = C.shape[0]; r = [1.0] * d + [0.0] * d
    for c in range(d):
        for rr in range(c + 1):
            r.append(2 * C[rr, c] * (1 if rr == c else np.sqrt(2)))
    return np.array(r)

def solve(C, tol, presolve=True, reduced=0.1, maxit=200):
    d = C.shape[0]; ntri = d * (d + 1) // 2
    A = sp.lil_matrix((2 * d + ntri, d))
    for c in range(d):
        A[c, c] = 1; A[d + c, c] = -1; A[2 * d + diag_idx(c), c] = 1
    st = clarabel.DefaultSettings(); st.verbose = False; st.max_iter = maxit
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = st.tol_infeas_abs = st.tol_infeas_rel = tol
    st.reduced_tol_gap_abs = st.reduced_tol_gap_rel = st.reduced_tol_feas = tol * reduced
    st.reduced_tol_infeas_abs = st.reduced_tol_infeas_rel = tol * reduced
    st.presolve_enable = presolve
    s = clarabel.DefaultSolver(sp.csc_matrix((d, d)), -np.ones(d), sp.csc_matrix(A), rhs_of(C),
                               [clarabel.NonnegativeConeT(2 * d), clarabel.PSDTriangleConeT(d)], st)
    return s.solve()

def feas(M, s, unit):
    # same closed form as enforce_feasibility
    sh = 2 * M - np.diag(s)
    lm = np.linalg.eigvalsh((sh + sh.T) / 2).min()
    if lm >= FLOOR: return s.copy(), 1.0, lm
    un = 2 * np.linalg.eigvalsh(M).min()
    g = float(np.clip((un - FLOOR) / (un - lm), 0, 1)) if un > lm else 0.0
    s2 = s * g
    if unit: s2 = np.array([clip(v) for v in s2])
    return s2, g, lm

def scs_ref(C):
    s = cp.Variable(C.shape[0])
    pr = cp.Problem(cp.Maximize(cp.sum(s)), [2 * C - cp.diag(s) >> 0, s >= 0, s <= 1])
    pr.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    return s.value

rows, summ = [], []
for lam in ["0.1", "1.0"]:
    z = pd.read_csv(f"{SCR}/../2026-10-12_python-scale-bug/results/python/seed_42/python/0.01_{lam}_out/z_matrix.csv" if False else
                    f"/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-10-12_python-scale-bug/results/python/seed_42/python/0.01_{lam}_out/z_matrix.csv", index_col=0)
    names = list(z.columns)
    for i in range(8):
        Sg = np.loadtxt(f"{SCR}/sigma_{lam}_{i}.csv", delimiter=","); d = Sg.shape[0]
        v = np.diag(Sg); sc = np.sqrt(v); C = Sg / np.outer(sc, sc)
        sol = solve(C, 1e-6); x = np.array(sol.x); zz = np.array(sol.z)
        clipped = np.array([clip(t) for t in x])
        cf, g1, lm1 = feas(C, clipped, True)
        fin, g2, lm2 = feas(Sg, cf * v, False)
        rust = np.loadtxt(f"{SCR}/sRust_default_{lam}_{i}.txt")
        big = rust / v > 1e-3
        z_lo = zz[d:2 * d]; z_hi = zz[:d]; Zjj = np.array([zz[2 * d + diag_idx(j)] for j in range(d)])
        ref = solve(C, 1e-10, presolve=False, reduced=1e-3, maxit=500)
        xr = np.array(ref.x); zr = np.array(ref.z)
        scs = scs_ref(C)
        dsdp = {f"{gt}_{mi}": np.loadtxt(f"{MS}/sR_{lam}_{i}_gt{gt}_mi{mi}.txt") / v for gt in ["1e-06", "1e-08", "1e-10"] for mi in [1000, 10000]}
        pyd = {gt: np.asarray(py_dsdp(Sg, gaptol=gt, maxit=1000)).ravel() / v for gt in [1e-6, 1e-8, 1e-10]}
        obj = lambda s_: float(np.sum(s_))
        summ.append(dict(lam=lam, chunk=i, status=str(sol.status), iters=sol.iterations, r_prim=sol.r_prim, r_dual=sol.r_dual,
                         raw_min=x.min(), n_raw_neg=int((x < 0).sum()), n_raw_zero_after_clip=int((x[clipped == 0] <= 0).sum()) if (clipped == 0).any() else 0,
                         gamma_corr=g1, lmin_corr=lm1, gamma_cov=g2, lmin_cov=lm2,
                         py_vs_rust_big=np.abs(fin[big] - rust[big]).max(), py_vs_rust_all=np.abs(fin - rust).max(),
                         n_final_zero_py=int((fin == 0).sum()), n_final_zero_rust=int((rust == 0).sum()),
                         obj_clarabel_raw=obj(x), obj_ref_clarabel=obj(xr), ref_status=str(ref.status), ref_iters=ref.iterations,
                         obj_scs=obj(scs), obj_rust_final=obj(rust / v), obj_R1e6=obj(dsdp["1e-06_1000"]),
                         obj_R1e10=obj(dsdp["1e-10_10000"]), obj_pyDSDP=obj(pyd[1e-6]),
                         scs_vs_ref_max=np.abs(scs - xr).max()))
        for j in range(d):
            rows.append(dict(lam=lam, chunk=i, local=j, factor=names[i * 22 + j], var=v[j],
                             R_gt1e6=dsdp["1e-06_1000"][j], R_gt1e8=dsdp["1e-08_1000"][j], R_gt1e10=dsdp["1e-10_1000"][j],
                             R_gt1e10_maxit1e4=dsdp["1e-10_10000"][j], Py_gt1e6=pyd[1e-6][j], Py_gt1e8=pyd[1e-8][j], Py_gt1e10=pyd[1e-10][j],
                             clar_raw=x[j], clar_clipped=clipped[j], clar_after_corr_feas=cf[j], clar_final_py=fin[j] / v[j], rust_final=rust[j] / v[j],
                             mu_lo=z_lo[j], nu_hi=z_hi[j], Zjj=Zjj[j], one_minus_Zjj=1 - Zjj[j],
                             ref_clar=xr[j], ref_mu_lo=zr[d + j], ref_Zjj=zr[2 * d + diag_idx(j)], scs=scs[j]))
pd.DataFrame(rows).to_csv(f"{OUT}/entries_all.csv", index=False)
pd.DataFrame(summ).to_csv(f"{OUT}/chunk_solver_summary.csv", index=False)
