import json,numpy as np
D="benchmark_results/20260601_155411_covgate_fix"
ck=json.load(open(f"{D}/rust/0.01_0.1_out/checkpoint.json"))
ev=None
for e in ck.get("evaluations",[]):
    if e and abs(e["delta"]-0.01)<1e-9 and abs(e["lambda"]-0.1)<1e-9: ev=e;break
m=ev["marginal"]
print("marginal keys:", list(m.keys()))
v=m.get("voting",{})
print("voting keys:", list(v.keys()) if isinstance(v,dict) else type(v))
for k in ("niter","spec","min_selections","threshold","optimal_iter","stage_one_union"):
    if isinstance(v,dict) and k in v:
        val=v[k]
        print(f"  {k} = {val if not isinstance(val,list) else f'list len {len(val)}'}" )
print("sig_lfs (rust):", m["sig_lfs"])
# selection frequency for the LFs reference kept but rust dropped
freq=v.get("selection_frequency",{})
counts=v.get("selection_counts")
if isinstance(freq,dict) and "data" in freq: freq=freq["data"]
print("freq type/len:", type(freq), len(freq) if hasattr(freq,'__len__') else '?')
for idx in [18,27,45,74,136]:
    fr = freq[idx] if hasattr(freq,'__getitem__') and len(freq)>idx else None
    ct = counts[idx] if counts and len(counts)>idx else None
    print(f"  Z{idx}: count={ct} freq={fr}")
print("stage_one_union (sample):", v.get("stage_one_union")[:30] if isinstance(v.get("stage_one_union"),list) else v.get("stage_one_union"))
