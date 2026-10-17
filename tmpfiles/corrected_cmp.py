import pandas as pd, numpy as np, os
D="benchmark_results/20260601_155411_covgate_fix"
combos=[("0.01","0.1"),("0.01","1.0"),("0.1","0.1"),("0.1","1.0")]
def load_lfs(p):
    if not os.path.exists(p) or os.path.getsize(p)==0: return set()
    return set(np.loadtxt(p,dtype=str).reshape(-1).tolist())
def shift(s):  # -1 shift on "Z\d+"
    return {f"Z{int(x[1:])-1}" for x in s}
def jac(a,b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)
print(f"{'combo':10} {'ref(pureR)':>11} {'python':>7} {'rust':>5} | {'J(ref,py)':>9} {'J(ref,rust)':>11} {'J(ref,rk)':>9}")
for d,l in combos:
    base=f"{D}"
    ref=shift(load_lfs(f"{base}/pure_r/{d}_{l}_out/sig_LFs.txt"))
    py=load_lfs(f"{base}/python/{d}_{l}_out/sig_LFs.txt")
    rk=load_lfs(f"{base}/r_knockoffs/{d}_{l}_out/sig_LFs.txt")
    # rust from summary
    import json
    summ=json.load(open(f"{base}/benchmark_summary.json"))
    rk_key=f"rust_{d}_{l}"
    rust=set(summ["results"].get(rk_key,{}).get("sig_lfs",[]))
    print(f"{d+'_'+l:10} {str(sorted(ref,key=lambda z:int(z[1:]))):>11} {len(py):>7} {len(rust):>5} | {jac(ref,py):>9.3f} {jac(ref,rust):>11.3f} {jac(ref,rk):>9.3f}")
    print(f"   ref={sorted(ref,key=lambda z:int(z[1:]))} py={sorted(py,key=lambda z:int(z[1:]))} rust={sorted(rust,key=lambda z:int(z[1:]))}")
