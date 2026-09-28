#!/usr/bin/env python3
"""Compare sampler geometry and streams on the five saved paired designs; no fits."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd


def moment_check(draws, mean, covariance):
    residuals = (draws - mean).reshape(-1, mean.shape[-1])
    count = len(residuals)
    empirical_mean = residuals.mean(axis=0)
    empirical_cov = residuals.T @ residuals / count
    variance = covariance.diagonal()
    mean_z = np.abs(empirical_mean) / np.sqrt(variance / count)
    cov_se = np.sqrt((np.outer(variance, variance) + covariance**2) / count)
    cov_z = np.abs(empirical_cov - covariance) / cov_se
    # Diagnostic six-SE screen, not a calibrated omnibus test or proof of parity.
    return float(mean_z.max()), float(cov_z.max())


def self_check():
    rng = np.random.default_rng(5)
    draws = rng.normal(size=(1000, 24, 3))
    mean = np.zeros((24, 3))
    assert max(moment_check(draws, mean, np.eye(3))) < 6
    assert max(moment_check(draws + 0.2, mean, np.eye(3))) > 6
    assert max(moment_check(draws * 1.2, mean, np.eye(3))) > 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rscript', type=Path, required=True)
    parser.add_argument('--rust-test', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=1000)
    args = parser.parse_args()
    if args.draws < 100: raise ValueError('Use at least 100 diagnostic draws')
    self_check()
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TMPDIR=str(args.out.resolve()), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', RAYON_NUM_THREADS='1', SLIDE_KNOCKOFF_MATCH_R_COV='1')
    env.pop('SLIDE_S_TOL', None)
    rows, manifest = [], {}
    names = [f'marginal_chunk_{i}' for i in range(1, 5)] + ['rust_stage_two_union']
    for name in names:
        inputs = args.fixtures / name / 'inputs'
        meta = json.loads((inputs / 'metadata.json').read_text())
        n, p = meta['shape']['Z']
        out = (args.out / name).resolve()
        out.mkdir()
        source = inputs / 'z.raw'
        manifest[name] = {'source': str(source.resolve()), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        shutil.copyfile(source, out / 'z.raw')
        with (out / 'r.log').open('w') as log:
            subprocess.run([str(args.rscript.resolve()), '--vanilla', str(Path(__file__).with_name('prepare.R').resolve()),
                            str(out), str(n), str(p), str(args.draws)], env=env, stdout=log, stderr=log, check=True)
        probe_env = dict(env, SLIDE_S_SIGMA_CSV=str(out / 'sigma.csv'), SLIDE_S_OUT=str(out / 'rust_s.csv'),
                         SLIDE_S_AUDIT_DIR=str(out), SLIDE_S_AUDIT_DRAWS=str(args.draws))
        with (out / 'rust.log').open('w') as log:
            subprocess.run([str(args.rust_test.resolve()), 'tests::emit_diag_s', '--exact', '--ignored', '--nocapture'],
                           env=probe_env, stdout=log, stderr=log, check=True)
        def raw(filename, shape):
            values = np.fromfile(out / filename, dtype='<f8').reshape(shape)
            if not np.isfinite(values).all(): raise ValueError(f'Nonfinite output: {filename}')
            return values
        sigma = np.loadtxt(out / 'sigma.csv', delimiter=',')
        rust_sigma = raw('rust_sigma.raw', (p,p))
        r_s = np.loadtxt(out / 'r_s.csv')
        rust_s = np.loadtxt(out / 'rust_s.csv')
        r_mean, r_cov, r_factor = (raw(f'r_{label}.raw', shape) for label, shape in
                                  [('mean',(n,p)), ('cov',(p,p)), ('factor',(p,p))])
        native_mean = raw('rust_native_mean.raw',(n,p))
        native_factor = raw('rust_native_factor.raw',(p,p))
        shared_mean = raw('rust_shared_s_mean.raw',(n,p))
        shared_factor = raw('rust_shared_s_factor.raw',(p,p))
        native = raw('rust_native_draws.raw',(args.draws,n,p))
        shared_sigma = raw('rust_shared_sigma_draws.raw',(args.draws,n,p))
        shared_sigma_factor = raw('rust_shared_sigma_factor.raw',(p,p))
        shared = raw('rust_shared_s_draws.raw',(args.draws,n,p))
        noise = raw('rust_noise_draws.raw',(args.draws,n,p))
        r_draws = raw('r_draws.raw',(args.draws,n,p))
        r_noise = raw('r_noise.raw',(args.draws,n,p))
        same_innovations = r_mean + noise @ r_factor
        native_cov = native_factor.T @ native_factor
        target_cov = 2*np.diag(rust_s)-np.diag(rust_s)@np.linalg.solve(sigma,np.diag(rust_s))
        r_mean_z, r_cov_z = moment_check(r_draws, r_mean, r_cov)
        native_mean_z, native_cov_z = moment_check(native, native_mean, native_cov)
        same_seed_rms = float(np.sqrt(np.mean((shared-r_draws)**2)))
        row = dict(design=name, n=n, p=p, draws=args.draws,
                   sigma_max_abs=float(np.max(np.abs(rust_sigma-sigma))),
                   s_max_variance_normalized=float(np.max(np.abs(rust_s-r_s)/sigma.diagonal())),
                   rust_s_zeros=int(sum(rust_s==0)), r_s_zeros=int(sum(r_s==0)),
                   conditional_mean_native_vs_r_max_abs=float(np.max(np.abs(native_mean-r_mean))),
                   conditional_cov_native_vs_r_max_abs=float(np.max(np.abs(native_cov-r_cov))),
                   native_cov_relative_frobenius_difference=float(np.linalg.norm(native_cov-r_cov)/np.linalg.norm(r_cov)),
                   shared_sigma_factor_cov_target_max_abs=float(np.max(np.abs(shared_sigma_factor.T@shared_sigma_factor-target_cov))),
                   shared_s_mean_max_abs=float(np.max(np.abs(shared_mean-r_mean))),
                   shared_s_cov_max_abs=float(np.max(np.abs(shared_factor.T@shared_factor-r_cov))),
                   shared_s_same_noise_max_abs=float(np.max(np.abs(shared-same_innovations))),
                   native_vs_shared_sigma_mean_max_abs=float(np.max(np.abs(native_mean-raw('rust_shared_sigma_mean.raw',(n,p))))),
                   same_noise_s_change_rms=float(np.sqrt(np.mean((shared_sigma-shared)**2))),
                   same_noise_sigma_roundoff_rms=float(np.sqrt(np.mean((native-shared_sigma)**2))),
                   same_s_different_stream_rms=same_seed_rms,
                   expected_independent_stream_rms=float(np.sqrt(2*np.trace(r_cov)/p)),
                   gaussian_stream_correlation=float(np.corrcoef(noise.ravel(),r_noise.ravel())[0,1]),
                   rust_mean_max_se=native_mean_z, rust_cov_max_se=native_cov_z,
                   r_mean_max_se=r_mean_z, r_cov_max_se=r_cov_z)
        np.testing.assert_allclose(rust_sigma, sigma, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(shared_mean, r_mean, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(shared_factor.T@shared_factor,r_cov,rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(shared,same_innovations,rtol=1e-7,atol=1e-8)
        assert max(r_mean_z,r_cov_z,native_mean_z,native_cov_z)<6, 'Sampler moment screen failed'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest[name]['sha256']
        manifest[name]['r_metadata'] = (out / 'r_metadata.txt').read_text().splitlines()
        manifest[name]['r_sampler_source_sha256'] = hashlib.sha256((out / 'r_sampler_source.txt').read_bytes()).hexdigest()
        rows.append(row)
        logging.info('%s: normalized S difference %.3g; common-noise error %.3g; stream RMS %.3g',
                     name,row['s_max_variance_normalized'],row['shared_s_same_noise_max_abs'],same_seed_rms)
    pd.DataFrame(rows).to_csv(args.out / 'summary.csv',index=False)
    manifest.update(rust_test_sha256=hashlib.sha256(args.rust_test.read_bytes()).hexdigest(),
                    draws=args.draws, seed_start=42, seed_policy='reseed each draw at 42+i; no swaps or fits',
                    rscript=str(args.rscript.resolve()), output=str(args.out.resolve()),
                    covariance_gate='SLIDE_KNOCKOFF_MATCH_R_COV=1',
                    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                   [Path(__file__), Path(__file__).with_name('prepare.R'),
                                    Path(__file__).resolve().parents[2] / 'crates/slide-knockoff/src/lib.rs']})
    (args.out / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
