# Compiled backend optimizations

## Python glmnet

The installed Python 3.11 extension still contained `-O0` in both its C and
Fortran DWARF compiler metadata. No optimized rebuild had replaced that binary.
The Meson project now defaults to `buildtype=release`; a fresh build uses `-O3`
without fast-math. The rebuilt Python 3.11 extension replaces the tracked binary;
the old binary lives in `archive/2026-09-12_glmnet-unoptimized/`.
The Python 3.9 binary is unchanged and was not rebuilt or verified.

Build environment: Python 3.11.15, NumPy 2.4.3, conda-forge GCC/gfortran, from
`/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev`.
Use that environment's `bin` on PATH and its `x86_64-conda-linux-gnu-cc` and
`x86_64-conda-linux-gnu-gfortran` as CC and FC. From the repository root:

```bash
export TMPDIR="$PWD/.tmpfiles/2026-09-12_compiled-optimizations"
mkdir -p "$TMPDIR"
meson setup "$TMPDIR/glmnet-release" src/loveslide/knockoff/_vendor/glmnet
ninja -C "$TMPDIR/glmnet-release"
```

Archive an existing installed binary before copying the rebuilt extension from
the build directory to the vendor directory. Use a fresh build directory on replay.
The existing Pixi build task also inherits the new release default, but its
pre-existing build-directory deletion is not used here.

Verification: the existing `tests/tests/test_knockoff_glmnet_defaults.py` passes
before and after rebuilding. Four fixed-seed Gaussian cases (24×87 and 80×30)
produce bitwise-identical coefficients, lambda paths and W statistics: 12 arrays,
maximum absolute difference 0. Compile commands confirm Fortran `-O3`.
The compiler reports warnings in the unchanged vendored Fortran source.

Paired inputs, verification script and outputs are in
`benchmark_results/2026-09-12_compiled-optimizations/`. With `PYTHONPATH=src`, run
`python <directory>/check_glmnet.py <new-output.npz> <directory>/glmnet-before.npz`.
The script asserts the compiled backend is available. Its single-pass timings
are smoke measurements, not a repeated performance benchmark.

## Rust Clarabel

Enabled `faer-sparse` on the existing Clarabel 0.11.1 dependency in `slide-sdp`.
Kept `direct_solve_method=auto`, tolerances, chunking and RNG unchanged.
Clarabel can now select Faer for expensive factorizations and QDLDL for smaller
ones. Cargo feature unification also enables this option for LOVE's Clarabel
dependency when built with the SDP crate. Cargo.lock records the new dependencies.

The release `profile_voting` example builds. Release SDP/knockoff tests report
23 passes and four ignored diagnostics; one reported pass is an existing
Python-comparison helper that returns without checking when its external input
is absent. The exercised tests cover feasibility, R fixture optimality and
seeded knockoff determinism. They do not establish bitwise QDLDL/Faer equivalence
on the saved 87-column interaction.

One native replay of the saved 24×87 interaction, 100 draws, with
`RAYON_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`, took 13.307 s for
standalone setup and 9.557 s for voting (zero selected predictors). The earlier
QDLDL voting measurement was 72.273 s. This new run overlapped compilation;
it is a preliminary smoke result, not a matched repeated speed benchmark.
The release binary contains Faer solver symbols and the Cargo feature tree
confirms `faer-sparse`; no new CPU sample verified the runtime backend selection.
See `faer-interaction-smoke.jsonl` and `rust-tests.log` in the artifact directory.

### Full CLI rebuild blocked

The full release CLI build failed in `highs-sys` while creating `libhighs.a`
(`ranlib: No such file or directory`). The additional LOVE test build also
failed while creating that archive (`Error running link command: Bus error`).
Stopped after this repeated build blocker; the full CLI has not been rebuilt
with Faer and LOVE tests have not completed. Logs remain under
`.tmpfiles/2026-09-12_compiled-optimizations/rust-build-example.log` and
`rust-love-tests.log`. Repairing this build is the next action, pending approval.

`cargo fmt --all -- --check` reports pre-existing formatting differences in
untouched Rust files. No Rust source files changed in this step; diff whitespace
checks pass. The original `.gitignore`, `pixi.toml` and `pixi.lock` edits remain
outside these commits.
