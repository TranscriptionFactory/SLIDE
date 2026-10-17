#!/usr/bin/env bash
#SBATCH --job-name=tol-feas-analysis
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --cluster=htc
#SBATCH --constraint=turin
set -euo pipefail
cd /vast/djishnu/Aaron/1_general_use/omni-SLIDE
RUN=$PWD/.tmpfiles/2026-10-16_tol-feas-benchmark
mkdir -p "$RUN/scratch/analysis"
export TMPDIR="$RUN/scratch/analysis" TMP="$RUN/scratch/analysis" TEMP="$RUN/scratch/analysis" MPLCONFIGDIR="$RUN/scratch/analysis"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev/bin/python - <<'PY'
import json, logging, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path('benchmarks').resolve()))
from analyze_seed_replicates import load, check_z, summarize, compare, plot
from aggregate_seed_sweep import _jaccard
from benchmark_backends import _param_subdir
logging.basicConfig(level=logging.INFO, format='%(message)s')
root = Path('benchmark_results/2026-10-16_tol-feas-benchmark')
out = root / 'analysis'
out.mkdir(exist_ok=True)
candidate = Path('.tmpfiles/2026-10-16_tol-feas-benchmark/results/rust')
reference = Path('.tmpfiles/2026-10-16_backend-replicates-all-arms/results')
seeds = {42, *range(1000, 10000, 1000)}
cells = {(0.01,0.1), (0.01,1.0), (0.1,0.1), (0.1,1.0)}
statuses, events = [], []
for arm, parent in [('baseline', reference/'rust'), ('candidate', candidate)]:
    assert {int(p.name[5:]) for p in parent.glob('seed_*')} == seeds
    for seed in sorted(seeds):
        seed_dir = parent / f'seed_{seed}'
        timings = json.loads((seed_dir/'rust/timings.json').read_text())
        assert len(timings) == 4 and {(t['delta'],t['lam']) for t in timings} == cells
        for t in timings:
            log = seed_dir/'rust'/_param_subdir(t['delta'],t['lam'])/'rust.log'
            text = log.read_text()
            error = '; '.join(line.strip() for line in text.splitlines() if 'Error:' in line or 'SDP solver failed' in line)
            statuses.append(dict(arm=arm, seed=seed, delta=t['delta'], lam=t['lam'], status=t['status'], wall_seconds=t['wall_seconds'], error=t['error'], solver_error=error, log=str(log)))
            if arm == 'candidate':
                lines = [line for line in text.splitlines() if line.startswith('SDP_BENCH ')]
                for ordinal, line in enumerate(lines, 1):
                    fields = dict(item.split('=',1) for item in line.split()[1:])
                    events.append(dict(fixture='SSc_UnTx', seed=seed, delta=t['delta'],lam=t['lam'], ordinal=ordinal, **fields))
status = pd.DataFrame(statuses)
assert len(status) == 80
assert status.groupby(['arm','status']).size().to_dict() == {('baseline','success'):40, ('candidate','success'):30, ('candidate','failed'):10}
assert all((r.delta,r.lam)==(0.1,1.0) for r in status[status.status=='failed'].itertuples())
status.to_csv(root/'cell_status.csv',index=False)
events = pd.DataFrame(events)
assert len(events) == 575 and events.status.value_counts().to_dict() == {'Solved':565,'NumericalError':10}
events.to_csv(root/'solver_events.csv',index=False)
logging.info('Explicit status measurements: %s', status.groupby(['arm','status']).size().to_dict())
arms = [('candidate','rust',[candidate]), ('baseline','rust',[reference/'rust']), ('pure_r','pure_r',[reference/'pure_r'])]
df = load(arms)
# The existing loader returns zero for absent Rust stage-timing lines; these are unavailable.
for idx, row in df[df.backend=="rust"].iterrows():
    text = (Path(row.seed_dir)/"rust"/_param_subdir(row.delta,row.lam)/"rust.log").read_text()
    if not re.search(r"\[SLIDE timing\] Knockoff \w+: ([\d.]+)s", text):
        df.loc[idx, "knockoff_seconds"] = np.nan
paired=[]
for r in status[status.arm=='candidate'].itertuples():
    b = df[(df.arm=='baseline') & (df.seed==r.seed) & (df.delta==r.delta) & (df.lam==r.lam)].iloc[0]
    row = dict(seed=r.seed, delta=r.delta, lam=r.lam, candidate_status=r.status, baseline_wall_seconds=b.wall_seconds, candidate_wall_seconds=r.wall_seconds)
    row["runtime_ratio" if r.status=="success" else "time_to_failure_over_baseline_completion"] = r.wall_seconds/b.wall_seconds
    if r.status=='success':
        c = df[(df.arm=='candidate') & (df.seed==r.seed) & (df.delta==r.delta) & (df.lam==r.lam)].iloc[0]
        for kind in ['marginal','interaction']:
            row[f'{kind}_identical'] = c[kind] == b[kind]
            row[f'{kind}_jaccard'] = _jaccard(c[kind], b[kind])
            row[f'candidate_{kind}'] = ' '.join(sorted(c[kind]))
            row[f'baseline_{kind}'] = ' '.join(sorted(b[kind]))
        row.update(candidate_score=c.score, baseline_score=b.score, score_delta=c.score-b.score)
    paired.append(row)
paired = pd.DataFrame(paired)
assert len(paired)==40 and paired[paired.candidate_status=='failed'].marginal_jaccard.isna().all()
paired.to_csv(root/'paired_cells.csv', index=False)
# Explicit, predeclared exclusion: report conditional successful-cell distributions only.
df = df[~((df.delta==0.1)&(df.lam==1.0))].copy()
assert len(df)==90 and len(df.groupby(['arm','cell']))==9
assert all(set(g.seed)==seeds and len(g)==10 for _,g in df.groupby(['arm','cell']))
logging.info('CONDITIONAL SUCCESSFUL-CELL ANALYSIS: excludes d=0.1 l=1.0 for all arms; not full benchmark success.')
logging.info('Independent-arm tests and bootstrap CIs are approximate: Rust arms have paired seeds. R agreement is diagnostic only.')
check_z(df,'pure_r',1e-8).to_csv(out/'z_check.csv',index=False)
summary=summarize(df)
tests,factors=compare(df,[('candidate','baseline'),('candidate','pure_r'),('baseline','pure_r')],10000,10000,np.random.default_rng(0))
summary.to_csv(out/'summary.csv',index=False)
tests.to_csv(out/'tests.csv',index=False)
factors.to_csv(out/'factor_rates.csv',index=False)
df[['arm','cell','seed','run','node','k','wall_seconds','knockoff_seconds','score','n_marginal','n_interaction']].assign(marginal=df.marginal.map(lambda s:' '.join(sorted(s))), interaction=df.interaction.map(lambda s:' '.join(sorted(s)))).to_csv(out/'timings.csv',index=False)
plot(df,factors,[a[0] for a in arms],list(df.cell.unique()),out)
logging.info('SUMMARY\n%s',summary.to_string(index=False))
logging.info('RUNTIME CI\n%s',tests[(tests.left=='candidate')&(tests.right=='baseline')&(tests.kind=='runtime')].to_string(index=False))
logging.info('PAIRED\n%s',paired.groupby(['delta','lam'])[['marginal_identical','interaction_identical','score_delta']].agg(['count','sum','median']).to_string())
logging.info('SCORE QUARTILES\n%s',df.groupby(['arm','cell']).score.quantile([.25,.5,.75]).to_string())
logging.info('Unavailable scores by arm: %s',df.groupby('arm').score.apply(lambda s:int(s.isna().sum())).to_dict())
PY
