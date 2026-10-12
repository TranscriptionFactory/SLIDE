# Lean FDR threshold and cached LOVE grid work

What changed:
- Rust `apply_fdr_threshold` (`crates/slide-love/src/love.rs`) and Python `_apply_fdr_threshold`
  (`src/loveslide/love_python/love/love.py`) find the BH cutoff from one sorted p^2 p-value
  buffer and filter in place, instead of index, adjusted-p and mask buffers. Same p-value and
  BH arithmetic as before (R `threshSigma` / statsmodels `fdr_bh`); kept entries are those at
  or below the largest kept sorted p-value, which is exact because adjusted p is monotone in rank.
- `LoveCache` / `love_cached`: slide-core passes one cache through the delta-major grid. The
  FDR-filtered covariance is computed once per run and the homogeneous pure-node estimate once
  per delta; each lambda reruns only the non-pure step. Only single-delta homogeneous calls are
  cached (a delta grid inside one call uses random CV).

Equivalence checks:
- Rust unit test compares the new filter bitwise with the old implementation (3 designs x 6
  thresholds, with ties and a constant column); a second test compares `love_cached` over a
  3x3 grid, a new input and a config change against uncached `love`. slide-love 51 and slide-core 24 + 1 parity test pass.
- Python test compares bitwise against statsmodels `multipletests(fdr_bh)`; the same 28
  pre-existing failures in `tests/tests/test_love_python.py` occur before and after.
- Old vs new Python filter on SSc (24 x 804) at thresholds 0.05/0.2/0.5: bitwise equal.
- Rust parity cells (delta 0.01/0.1, lambda 0.1), FDR build vs baseline 471b42b: full checkpoints equal.
- `run_grid.sh`: one 2x2 grid run with the cache build vs four single-cell baseline runs: all
  four evaluations (LOVE, Z, marginal, interactions, score) equal.

Python FDR on synthetic p=8000, n=50 (`fdr_mem.py`, one run each, htc, 2 CPUs):

| | time | peak RSS added by the call |
|---|---|---|
| old | 37.5 s | 4.57 GB |
| new | 17.1 s | 1.20 GB |

Output checksums identical. At p=804 the cache saves little (K=172 LOVE 2.8 s -> 2.2 s for the
second lambda) because the Dantzig step dominates; the cached covariance, FDR and pure-node
search are the parts that grow as p^2.
