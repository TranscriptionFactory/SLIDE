"""Independent readback of the chunk-4 tiny/substantive split run (argv: run, Oct-6 full run, Oct-5 run); prints compact results only."""
import glob, json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, 'benchmarks')
from analyze_paired_xtilde import aggregate_selection
from replay_paired_xtilde import zero_entry_ties
run, repeat, prior = sys.argv[1:4]
d = f'{run}/marginal_chunk_4'
meta = json.load(open('.tmpfiles/2026-09-14_paired-convergence/results/marginal_chunk_4/inputs/metadata.json'))
names = meta['lf_names']; p = len(names)
summ = json.load(open(f'{d}/sampler_summary.json')); dec = summ['decomposition']
arms = list(dec['arms'])
sel = {a: [] for a in arms}; W = {a: [] for a in arms}
for a in arms:
    for i in range(summ['draws']):
        f = json.load(open(f'{d}/{a}/iter_{i:03d}.json'))
        c = zero_entry_ties(f, meta['fdr'])
        assert c['selected'] == f['selected'] and c['threshold'] == f['threshold']
        sel[a].append(set(f['selected'])); W[a].append(f['w'])
votes = {a: aggregate_selection(sel[a], p, meta['spec']) for a in arms}
for a in arms:
    assert [names[j] for j in votes[a]['findOptIter']['selected']] == dec['arms'][a]['findOptIter']
arms_csv = pd.read_csv(f'{d}/arms.csv')
for a in arms: assert arms_csv[f'{a}_votes'].tolist() == votes[a]['votes']
print('minimum votes', votes[arms[0]]['minimum_votes'])
z = 1.959963984540054; N = summ['draws']
for a in arms:
    v = np.array(votes[a]['votes']); near = [f'{names[j]}:{v[j]}' for j in np.flatnonzero(abs(v - 100) <= 15)]
    k = v[names.index('Z85')]; ph = k / N; den = 1 + z*z/N; c = (ph + z*z/(2*N))/den; h = z*np.sqrt(ph*(1-ph)/N + z*z/(4*N*N))/den
    print(f"{a:22s} n={len(dec['arms'][a]['findOptIter'])} Z85={k} [{100*(c-h):.2f},{100*(c+h):.2f}]% ties/draw={dec['arms'][a]['mean_positive_ties']:.2f} near={near}")
    print('   ', dec['arms'][a]['findOptIter'])
con = pd.read_csv(f'{d}/contrasts.csv')
for name, s in dec['contrasts'].items():
    c = con[con.contrast == name]; z85 = c[c.feature == 'Z85'].iloc[0]
    top = c.reindex(c.frequency_delta.abs().sort_values(ascending=False).index).head(3)
    extra = '' if 'left' not in s else f"sel_changed={s['selection_changed_draws']} thr_changed={s['threshold_changed_draws']} ties_changed={s['tie_pattern_changed_draws']} Wrms={s['w_rms_delta']:.3f} fOI_equal={s['findOptIter_equal']}"
    print(f"{name:24s} Z85 {100*z85.frequency_delta:+.1f}±{100*z85.paired_mcse:.2f}pp | top " +
          ', '.join(f"{r.feature} {100*r.frequency_delta:+.1f}±{100*r.paired_mcse:.2f}" for r in top.itertuples()) + ' | ' + extra)
# zero-set ties per arm
zero = ['Z68', 'Z70', 'Z72', 'Z79', 'Z82']
print('zero-set tie draws:', arms_csv.set_index('feature').loc[zero, [f'{a}_tie_draws' for a in arms]].to_string())
print('zero-set votes:', arms_csv.set_index('feature').loc[zero, [f'{a}_votes' for a in arms]].sum().to_dict())
Wa = {a: np.array(W[a]) for a in arms}
zi = [names.index(x) for x in zero]
for a in arms: print(f'{a}: zero-set W==0 frac {np.mean(Wa[a][:, zi] == 0):.3f}, W>0 frac {np.mean(Wa[a][:, zi] > 0):.3f}, W<0 frac {np.mean(Wa[a][:, zi] < 0):.3f}')
# cross-run repeatability of saved arms vs Oct 5
for a in ('rust_s', 'r_s'):
    bad = 0
    for i in range(N):
        x = json.load(open(f'{d}/{a}/iter_{i:03d}.json')); y = json.load(open(f'{prior}/marginal_chunk_4/{a}/iter_{i:03d}.json'))
        bad += any(x[k] != y[k] for k in x if k != 'wall_ms')
    print(f'{a} vs Oct-5 fits: {bad} differing fits of {N}')
# repeat run
bad = 0
old = [a for a in arms if Path(f'{repeat}/marginal_chunk_4/{a}').is_dir()]
for a in old:
    for i in range(N):
        x = json.load(open(f'{d}/{a}/iter_{i:03d}.json')); y = json.load(open(f'{repeat}/marginal_chunk_4/{a}/iter_{i:03d}.json'))
        bad += any(x[k] != y[k] for k in x if k != 'wall_ms')
print(f'Oct-6 full run, shared arms {old}: {bad} differing fits of {N*len(old)}')
