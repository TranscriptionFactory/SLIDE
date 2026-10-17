import pandas as pd, numpy as np
D="benchmark_results/20260601_155411_covgate_fix"
for combo in ["0.01_0.1","0.1_1.0"]:
    pr=pd.read_csv(f"{D}/pure_r/{combo}_out/z_matrix.csv", index_col=0)
    py=pd.read_csv(f"{D}/python/{combo}_out/z_matrix.csv", index_col=0)
    # Hypothesis: pure_r 'Z(i+1)' == python 'Zi'. Rename pure_r by -1.
    pr_shift = pr.copy()
    pr_shift.columns = [f"Z{int(c[1:])-1}" for c in pr.columns]
    common = sorted(set(pr_shift.columns)&set(py.columns), key=lambda c:int(c[1:]))
    a=pr_shift[common].values; b=py[common].values
    # per-column corr
    cors=[]
    for i in range(len(common)):
        v1,v2=a[:,i],b[:,i]
        if v1.std()<1e-10 or v2.std()<1e-10: continue
        cors.append(np.corrcoef(v1,v2)[0,1])
    cors=np.array(cors)
    overall=np.corrcoef(a.flatten(),b.flatten())[0,1]
    print(f"[{combo}] AFTER -1 shift: overall_r={overall:.4f} median_col_r={np.median(cors):.4f} min={cors.min():.4f} | frac>0.99={np.mean(cors>0.99):.3f}")
    # also check abs (sign flips common in factor models)
    print(f"   |col_r| median={np.median(np.abs(cors)):.4f} min={np.abs(cors).min():.4f}")
