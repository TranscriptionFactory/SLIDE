import pandas as pd, numpy as np
D="benchmark_results/20260601_155411_covgate_fix"
pr=pd.read_csv(f"{D}/pure_r/0.01_0.1_out/z_matrix.csv", index_col=0)
py=pd.read_csv(f"{D}/python/0.01_0.1_out/z_matrix.csv", index_col=0)
print("pure_r shape", pr.shape, "| python shape", py.shape)
print("pure_r cols[:6]", list(pr.columns[:6]), "last", list(pr.columns[-2:]))
print("python cols[:6]", list(py.columns[:6]), "last", list(py.columns[-2:]))
print("pure_r rows[:3]", list(pr.index[:3]))
print("python rows[:3]", list(py.index[:3]))
print("rows equal set?", set(map(str,pr.index))==set(map(str,py.index)))
print("rows same order?", list(map(str,pr.index))==list(map(str,py.index)))
print("cols equal set?", set(pr.columns)==set(py.columns))
