#!/usr/bin/env python3
"""Summarize selected comparable native-R cohorts; never run a backend."""
import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
from statistics import mean, median
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from aggregate_seed_sweep import _jaccard, _normalize_interaction, load_sweep


def metrics(sets):
    pairs = [_jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return dict(n_runs=len(sets), count_median=median(map(len, sets)),
                count_min=min(map(len, sets)), count_max=max(map(len, sets)),
                intersection_count=len(set.intersection(*sets)), union_count=len(set.union(*sets)),
                jaccard_mean=mean(pairs), jaccard_min=min(pairs), jaccard_max=max(pairs))


def self_check():
    assert metrics([{'Z1'}, {'Z1'}])['jaccard_mean'] == 1
    assert metrics([{'Z1'}, {'Z2'}])['jaccard_mean'] == 0
    assert metrics([set(), set()])['jaccard_mean'] == 1
    assert metrics([set(), {'Z1'}])['count_median'] == 0.5
    assert _normalize_interaction('Z10:Z2') == 'Z2.Z10'


def fingerprint(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    self_check()
    args.out.mkdir(parents=True, exist_ok=True)
    scratch_root = ROOT / '.tmpfiles'
    scratch_root.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='2026-09-28_r-history-', dir=scratch_root))
    os.environ['TMPDIR'] = str(scratch)
    os.environ['MPLCONFIGDIR'] = str(scratch / 'matplotlib')
    tempfile.tempdir = str(scratch)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    bundles = []
    for cohort, folder, rule in [
        ('aug19_stock', '20260819_high-delta-seeds', 'findOptIter'),
        ('aug20_frequency', '20260819_stability-rule-seeds', 'frequency_only'),
    ]:
        parent = ROOT / 'benchmark_results' / folder
        loaded = load_sweep(parent)
        assert loaded['seeds'] == [42, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        for seed in loaded['seeds']:
            bundles.append((cohort, rule, 500, seed, parent / f'seed_{seed}', loaded))
    for niter, seed in itertools.product((1000, 2000, 5000), (42, 1000, 2000)):
        bundles.append(('sep12_seed_sweep', 'findOptIter', niter, seed,
                        ROOT / f'.tmpfiles/2026-09-12_r-stability/results/niter_{niter}/seed_{seed}', None))
    for folder in ('2026-09-14_all-backend-parity', '2026-09-22_all-backend-parity-tiefix'):
        bundles.append(('sep14_22_repeats', 'findOptIter', 500, 42,
                        ROOT / '.tmpfiles' / folder / 'results', None))

    records, groups, seen_paths, seen_executions = [], {}, set(), set()
    for cohort, rule, niter, seed, path, loaded in bundles:
        summary_path = path / 'benchmark_summary.json'
        data = json.loads(summary_path.read_text())
        environment_path = path / 'pure_r/benchmark_environment.json'
        if not environment_path.exists():
            environment_path = path / 'benchmark_environment.json'
        environment = json.loads(environment_path.read_text())
        assert environment['seed'] == seed and environment['niter'] == niter
        if 'slide_selection' in environment:
            assert environment['slide_selection'] == (rule == 'findOptIter')
        results = {k: v for k, v in data['results'].items() if k.startswith('pure_r_')}
        deltas = (0.1,) if cohort == 'aug19_stock' else (0.01, 0.1)
        assert set(results) == {f'pure_r_{d}_{lam}' for d, lam in itertools.product(deltas, (0.1, 1.0))}
        for key, result in sorted(results.items()):
            _, delta, lam = key.rsplit('_', 2)
            cell = path / 'pure_r' / f'{delta}_{lam}_out'
            params = dict(line.split(': ', 1) for line in (cell / 'yaml_params.yaml').read_text().splitlines())
            assert float(params['delta']) == float(delta) and float(params['lambda']) == float(lam)
            assert int(params['SLIDE_iter']) == niter
            assert float(params['spec']) == float(params['fdr']) == 0.1
            assert params['do_interacts'] == 'yes'
            timing = [t for t in data['timings'] if t['backend'] == 'pure_r'
                      and t['delta'] == float(delta) and t['lam'] == float(lam)]
            assert len(timing) == 1 and timing[0]['status'] == 'success'
            source_path = str(cell.resolve())
            execution = params['out_path'].replace('/ix/', '/vast/').rstrip('/')
            assert source_path not in seen_paths and execution not in seen_executions, 'Duplicate execution/copy'
            seen_paths.add(source_path)
            seen_executions.add(execution)
            log_path = path / 'pure_r' / f'pure_r_{delta}_{lam}.log'
            log = log_path.read_text()
            if rule == 'frequency_only':
                assert 'SLIDE_R_SOURCE=vendor' in log and 'SLIDE_OPT_ITER=0' in log
            else:
                assert 'SLIDE_OPT_ITER=0' not in log
            assert int(params['ncore']) == (10 if cohort.startswith('aug') else 4)
            if cohort == 'sep14_22_repeats':
                assert '[pure_r] effective SLIDE workers: 4' in log
                assert '[pure_r] SLIDE_R_OPTIMIZE=1:' in log
                assert '[pure_r] SLIDE_R_SOURCE=installed:' in log
                assert environment['workers_by_backend']['pure_r'] == 4
                assert environment['inherited_environment']['SLIDE_R_OPTIMIZE'] == '1'
            for kind, field, filename in [('LF', 'sig_lfs', 'sig_LFs.txt'),
                                           ('Interaction', 'sig_interacts', 'sig_interacts.txt')]:
                raw = set((cell / filename).read_text().split())
                saved = set(result[field])
                if kind == 'Interaction':
                    raw = {_normalize_interaction(v) for v in raw}
                    saved = {_normalize_interaction(v) for v in saved}
                assert raw == saved, f'Raw/summary selection mismatch: {cell}'
                if loaded is not None:
                    loaded_kind = 'marginal' if kind == 'LF' else 'interaction'
                    assert dict(loaded['selections'][('pure_r', float(delta), float(lam))][loaded_kind])[seed] == raw
                group_key = (cohort, niter, rule, float(delta), float(lam), kind)
                groups.setdefault(group_key, []).append((source_path, seed, raw))
            record = dict(cohort=cohort, rule=rule, niter=niter, seed=seed, delta=float(delta), lam=float(lam),
                          source=source_path, original_output=execution, params=params, timing=timing[0],
                          files=[fingerprint(cell / f) for f in ['sig_LFs.txt', 'sig_interacts.txt', 'yaml_params.yaml', 'z_matrix.csv']],
                          summary=fingerprint(summary_path), log=fingerprint(log_path),
                          environment_file=fingerprint(environment_path), environment=environment,
                          seed_evidence='Validated saved environment seed against nominal execution label; worker RNG seeding not established')
            records.append(record)
    assert len(bundles) == 31 and len(records) == 104
    for cohort, niter, rule, delta, lam, kind in groups:
        matching = [r for r in records if (r['cohort'], r['niter'], r['rule'], r['delta'], r['lam'])
                    == (cohort, niter, rule, delta, lam)]
        assert len({r['files'][3]['sha256'] for r in matching}) == 1, 'LF universe changed within stratum'

    frozen = []
    for folder in ('2026-09-14_all-backend-parity', '2026-09-22_all-backend-parity-tiefix'):
        run = ROOT / '.tmpfiles' / folder
        paths = sorted(p for directory in ('R-library', 'inputs') for p in (run / directory).rglob('*') if p.is_file())
        paths += [run / 'source/benchmarks' / f for f in ('run_slide_r.R', 'benchmark_backends.py')]
        frozen.append({str(p.relative_to(run)): fingerprint(p) for p in paths})
    assert {k: v['sha256'] for k, v in frozen[0].items()} == {k: v['sha256'] for k, v in frozen[1].items()}
    summary, rates, pairs = [], [], []
    for key, values in sorted(groups.items()):
        base = dict(zip(('cohort', 'niter', 'rule', 'delta', 'lam', 'kind'), key))
        sets = [v[2] for v in values]
        summary.append(dict(base, **metrics(sets)))
        for item in sorted(set.union(*sets)):
            k = sum(item in s for s in sets)
            rates.append(dict(base, item=item, k=k, n=len(sets), selection_rate=k / len(sets)))
        for a, b in itertools.combinations(values, 2):
            pairs.append(dict(base, source_a=a[0], source_b=b[0], seed_a=a[1], seed_b=b[1], jaccard=_jaccard(a[2], b[2])))
    assert len(summary) == 44 and len(pairs) == 620
    write_csv(args.out / 'summary.csv', summary)
    write_csv(args.out / 'selection_rates.csv', rates)
    write_csv(args.out / 'pairwise_jaccards.csv', pairs)
    cohort_metadata = {
        'aug19_stock': dict(requested_workers=10, effective_workers=None,
            worker_evidence='Ten documented; no effective-worker log validation',
            r_source='installed', source_evidence='docs/backend-benchmark.md:84-94,108-122; per-execution environment git_sha is recorded, not a frozen package hash',
            optimization='Not recorded; predates September R interaction optimization'),
        'aug20_frequency': dict(requested_workers=10, effective_workers=None,
            worker_evidence='Ten documented; no effective-worker log validation',
            r_source='vendor', source_evidence='Every cell log validates vendor and SLIDE_OPT_ITER=0; per-execution environment git_sha recorded',
            optimization='Not recorded; predates September R interaction optimization'),
        'sep12_seed_sweep': dict(requested_workers=4, effective_workers=None,
            worker_evidence='Four requested/allocated; upstream worker-budget enforcement not verified',
            r_source='installed', source_evidence='docs/2026-09-12-r-stability-submission.md:26-35; frozen runner 096085e and installed SLIDE 0.1.1',
            optimization='Unmodified frozen runner predates interaction optimization'),
        'sep14_22_repeats': dict(requested_workers=4, effective_workers=4,
            worker_evidence='Every cell log and per-backend environment validate four effective workers',
            r_source='installed', source_evidence='Per-backend source_revision a74c521 / 605557c; R runner/harness/package/input hashes match',
            optimization='SLIDE_R_OPTIMIZE=1 validated in every cell log and per-backend environment'),
    }
    manifest = dict(cohorts=cohort_metadata, scope='Selected comparable cohorts, not a complete archive census',
                    bundles=len(bundles), cell_executions=len(records),
                    duplicate_policy='Reject repeated canonical cell paths or original YAML output paths; retain equal selections from separate executions.',
                    selection_rate_denominator='All runs in the stratum; rows cover observed union only, not all candidate features.',
                    matched_repeat_frozen_files=frozen, records=records)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    labels = {('aug19_stock', 500): 'Aug19 stock\n500 · n=10',
              ('aug20_frequency', 500): 'Aug20 frequency\n500 · n=10',
              ('sep12_seed_sweep', 1000): 'Sep12 stock\n1000 · n=3',
              ('sep12_seed_sweep', 2000): 'Sep12 stock\n2000 · n=3',
              ('sep12_seed_sweep', 5000): 'Sep12 stock\n5000 · n=3',
              ('sep14_22_repeats', 500): 'Sep14/22 stock\n500 · n=2'}
    df = pd.DataFrame(summary)
    df['cohort_label'] = [labels[(r.cohort, r.niter)] for r in df.itertuples()]
    df['cell'] = [f'δ={r.delta:g}, λ={r.lam:g}' for r in df.itertuples()]
    sns.set_theme(style='white', context='notebook')
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for ax, kind in zip(axes, ('LF', 'Interaction')):
        values = df[df.kind == kind].pivot(index='cell', columns='cohort_label', values='jaccard_mean').reindex(columns=list(labels.values()))
        sns.heatmap(values, vmin=0, vmax=1, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                    cbar_kws={'label': 'Mean pairwise Jaccard'})
        ax.set(title=f'{kind} selected-set agreement', xlabel='', ylabel='')
        ax.tick_params(axis='y', rotation=0)
    axes[-1].tick_params(axis='x', rotation=0)
    fig.suptitle('Native-R run variability across separate historical cohorts')
    fig.text(0.5, 0.015, 'Columns are separate strata, not exchangeable runs. n = runs per cell; 500/1000/2000/5000 = knockoff draws.\nBlank cells were not run. Pairwise comparisons share runs; no confidence intervals or FDR claims.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(args.out / 'stability.png', dpi=160)
    plt.close(fig)
    logging.info('Verified %d bundles, %d cells, %d strata, %d pairwise rows', len(bundles), len(records), len(summary), len(pairs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
