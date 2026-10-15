"""Fixed-draw fitter comparison; reuses production stats and existing replay/aggregation."""
import argparse, importlib.util, json, logging, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
# shortcut: fixed 24-row saved fixtures; extend the existing loaders for another study.
ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'benchmarks')]
from replay_paired_xtilde import fit_draw, paired_vote_summary
from analyze_paired_xtilde import aggregate_selection
from loveslide.knockoff.stats import glmnet as stat
from loveslide.knockoff.filter import knockoff_threshold
logging.basicConfig(level=logging.INFO, format='%(message)s')
p = argparse.ArgumentParser()
p.add_argument('out', type=Path)
p.add_argument('--draws', type=int)
p.add_argument('--designs', nargs='+', choices=['native_chunk_1','native_chunk_4','saved_stage2','saved_c8','saved_highdelta'])
p.add_argument('--python-production', action='store_true')
a = p.parse_args()
if a.draws is not None and a.draws < 2:
    p.error('--draws must be at least 2')
run = ROOT / '.tmpfiles/2026-10-15_gaussian-parity'
old_path = run / 'baseline-source/src/loveslide/knockoff/_vendor/glmnet/linear.py'
spec = importlib.util.spec_from_file_location('loveslide.knockoff._vendor.glmnet.baseline_linear', old_path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
new_class = stat.ElasticNet
spec = importlib.util.spec_from_file_location('loveslide.knockoff.stats.baseline_glmnet', run/'baseline-source/src/loveslide/knockoff/stats/glmnet.py')
baseline_stat = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline_stat)
arms = ['rust_baseline', 'rust_schedule', 'rust_rprep', 'rust_schedule_rprep',
        'python_baseline', 'python_algorithm', 'python_rprep', 'python_algorithm_rprep']
if a.python_production:
    arms = ['python_baseline','python_algorithm','python_arithmetic','python_algorithm_arithmetic']
source = ROOT / '.tmpfiles/2026-10-13_preliminary-selection-diagnosis/numerical'
summary = {}
for label, inp in [('native_chunk_1', source/'chunk_1'), ('native_chunk_4', source/'chunk_4'),
                   ('saved_stage2', ROOT/'.tmpfiles/diag_s2_001_10'), ('saved_c8', ROOT/'.tmpfiles/diag_c8'),
                   ('saved_highdelta', ROOT/'.tmpfiles/diag_d01_l01')]:
    if a.designs and label not in a.designs: continue
    n = 24
    if label == 'saved_highdelta':
        source_names = [f'Z{44+j}' for j in range(22)]
        m = 22
        z = np.loadtxt(inp/'Zc.txt'); y = np.loadtxt(inp/'Y.txt')
        augmented = np.load(inp/'augs.npy'); sw = np.load(inp/'swaps.npy').astype(bool)
        assert np.array_equal(np.where(sw[:,None,:],augmented[:,:,m:],augmented[:,:,:m]),np.broadcast_to(z,augmented[:,:,:m].shape))
        xt = np.where(sw[:,None,:],augmented[:,:,:m],augmented[:,:,m:])
    else:
        source_names = (inp/'lf_names.txt').read_text().splitlines()
        m = len(source_names)
        z = np.fromfile(inp/'z.raw', '<f8').reshape(n,m)
        y = np.fromfile(inp/'y.raw', '<f8')
        xtpath = inp/('xt_native.raw' if label.startswith('native') else 'xt.raw')
        swpath = inp/('swaps.raw' if label.startswith('native') else 'swap.raw' if label=='saved_stage2' else 'swap.txt')
        xt = np.fromfile(xtpath, '<f8').reshape(-1,n,m)
        sw = np.fromfile(swpath, np.uint8).reshape(-1,m).astype(bool)
    # The original chunk-4 capture used local labels; trace.rs/compare.py map its start to Z66.
    names = [f'Z{66+j}' for j in range(m)] if label=='native_chunk_4' else source_names
    assert len(sw)==len(xt) and y.shape==(n,)
    count = min(len(xt), a.draws) if a.draws else len(xt)
    out = a.out/label
    out.mkdir(parents=True, exist_ok=False)
    z.tofile(out/'z.raw'); y.tofile(out/'y.raw'); np.zeros((1,m)).tofile(out/'a.raw')
    (out/'lf_names.txt').write_text('\n'.join(source_names)+'\n')
    xt[:count].tofile(out/'xt_fixed.raw'); sw[:count].astype(np.uint8).tofile(out/'swaps.raw')
    with (out/'r.log').open('w') as log:
        subprocess.run([os.environ['ENV']+'/bin/Rscript', '-e',
          '''library(knockoff); library(glmnet);
          save_raw <- function(name, value) { con <- file(file.path(Sys.getenv('REPLAY_CAPTURE_DIR'),name),'ab'); writeBin(as.double(value),con); close(con) };
          trace('glmnet', where=asNamespace('glmnet'), print=FALSE,
            tracer=quote({ save_raw('r_x.raw',t(x)); save_raw('r_lambda.raw',lambda) }),
            exit=quote({ b <- as.matrix(returnValue()$beta); ix <- apply(b,1,function(v)match(TRUE,abs(v)>0)); ix[is.na(ix)]<-0; save_raw('r_entries.raw',ix-1) }));
          source("benchmark_results/2026-10-13_s-zero-handling/r_stat.R")''',
          str(out), 'fixed', str(count), str(n), str(m), '0.1'], check=True, stdout=log, stderr=log, env=os.environ | {'REPLAY_CAPTURE_DIR':str(out)})
    r = pd.read_csv(out/'w_r_fixed.csv').to_numpy()
    rx = np.fromfile(out/'r_x.raw','<f8').reshape(count,n,2*m)
    rl = np.fromfile(out/'r_lambda.raw','<f8').reshape(count,500)
    re = np.fromfile(out/'r_entries.raw','<f8').reshape(count,2*m).astype(int)
    prepared = out/'prepared'; prepared.mkdir()
    for name in ['y.raw','a.raw','lf_names.txt']:
        (prepared/name).symlink_to((out/name).resolve())
    statistics = {arm: [] for arm in arms}
    selections = {arm: [] for arm in arms+['r']}
    entries = {arm: [] for arm in arms}
    for d in range(count):
        selections['r'].append(set(np.flatnonzero(r[d,:m]>=r[d,m])))
        xt[d].tofile(out/'draw.raw'); sw[d].astype(np.uint8).tofile(out/'swap.raw')
        np.where(sw[d],rx[d,:,m:],rx[d,:,:m]).tofile(prepared/'z.raw')
        np.where(sw[d],rx[d,:,:m],rx[d,:,m:]).tofile(prepared/'draw.raw')
        rl[d].tofile(prepared/'lambda.raw')
        for arm in arms:
            if arm.startswith('rust'):
                binary = run/('schedule-prepared-xtilde' if 'schedule' in arm else 'baseline-prepared-xtilde')
                args = SimpleNamespace(rust_bin=binary, tolerance=1e-7, max_iter=100000, nlambda=500, prepared_lambda=prepared/'lambda.raw' if 'rprep' in arm else None)
                meta = {'shape':{'Z':[n,m], 'A':[1,m]}, 'rust_lambda_environment':'1', 'rust_cached':True, 'fdr':0.1}
                inputs = prepared if 'rprep' in arm else out
                result = fit_draw(args, meta, inputs, inputs/'draw.raw', out/'swap.raw', out/'last_fit.json')
                assert len(result['lambda'])==500
                if 'rprep' in arm: assert np.array_equal(result['lambda'],rl[d])
                w = np.asarray(result['w']) * n
                selected = set(result['selected'])
                beta = np.asarray(result['coefficients'])
            else:
                fitter = stat if 'arithmetic' in arm else baseline_stat
                fitter.ElasticNet = new_class if 'algorithm' in arm else old.ElasticNet
                if 'rprep' in arm:
                    model = fitter.ElasticNet(alpha=1,lambda_path=rl[d],standardize=False,fit_intercept=True,n_splits=0,tol=1e-7,max_iter=100000).fit(rx[d],y)
                    beta = model.coef_path_; nz=beta!=0
                    e=np.where(nz.any(axis=1),model.lambda_path_[nz.argmax(axis=1)]*n,0)
                    w=stat.correct_for_swap(stat.compute_signed_max_stat(e,m),sw[d])
                    diag={'coefficients':beta}
                else:
                    w, diag = fitter.stat_glmnet_lambdasmax(z,xt[d],y,swap=sw[d],nlambda=500,return_diagnostics=True)
                selected = set(np.flatnonzero(w>=knockoff_threshold(w,fdr=0.1,offset=0)))
                beta = diag['coefficients']
            nz = np.asarray(beta)!=0
            indices = np.where(nz.any(axis=1), nz.argmax(axis=1), -1)
            entries[arm].append(indices.tolist())
            statistics[arm].append(w.tolist()); selections[arm].append(selected)
        if d%100==0: logging.info('%s draw %s/%s',label,d,count)
    stat.ElasticNet = new_class
    cell = {'draws':count,'factors':names,'source':str(inp),'arms':{}}
    for arm in arms+['r']:
        agg = aggregate_selection(selections[arm],m,0.1); agg.pop('cumulative')
        if arm!='r':
            w = np.asarray(statistics[arm])
            agg['all_factor_sign_agreement_r'] = float((np.sign(w)==np.sign(r[:,:m])).mean())
            agg['selected_set_agreement_r'] = float(np.mean([x==y for x,y in zip(selections[arm],selections['r'])]))
            agg['entry_index_agreement_r'] = float((np.asarray(entries[arm])==re).mean())
            agg['entry_index0_count'] = int((np.asarray(entries[arm])==0).sum())
            if count>1:
                masks = lambda sets: [[j in s for j in range(m)] for s in sets]
                table = paired_vote_summary(masks(selections[arm]),masks(selections['r']),names=(arm,'r'))
                table.insert(0,'factor',names); table.to_csv(out/(arm+'_votes.csv'),index=False)
        cell['arms'][arm] = agg
    (out/'entries.json').write_text(json.dumps(entries))
    (out/'w.json').write_text(json.dumps(statistics))
    (out/'selected.json').write_text(json.dumps({k:[sorted(map(int,s)) for s in v] for k,v in selections.items()}))
    (out/'summary.json').write_text(json.dumps(cell,indent=2))
    summary[label] = cell
(a.out/'summary.json').write_text(json.dumps(summary,indent=2))
# Reuse the paired-vote plot; these fits hold saved draws fixed, not a population sample.
from replay_paired_xtilde import plot_sampler_votes
frames = []
for label, cell in summary.items():
    if cell['draws'] > 1:
        frame = pd.read_csv(a.out/label/(arms[-1]+'_votes.csv')).rename(columns={'factor':'feature'})
        frame['design'] = label
        frames.append(frame)
if frames:
    plot_sampler_votes(a.out, pd.concat(frames), title=arms[-1]+' minus R: fixed-draw selection votes',
                       note='Same saved draws and swaps; conditional Monte Carlo SE, not equivalence or full-pipeline inference.')
