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
