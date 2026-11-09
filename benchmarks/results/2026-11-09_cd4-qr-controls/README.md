# CD4 QR-control Rust benchmark

This run replaces the SVD control fitter with the existing ordered,
reorthogonalized OLS fitter. Selection, sampleCV, random draws, parallel control
execution and scoring formulas are unchanged. Near-dependent designs can produce
different control scores because the rank criterion changes.

The run reuses the preceding 40-fit manifest: ten seeds × four delta/lambda
settings, 1,000 selection draws, 500 sampleCV holdouts (K=4), four workers and one
BLAS thread per worker. Fresh checkpoints include control reporting and export.
R, Python, SSc and final CV are not rerun. Previous results remain intact in
`../2026-11-08_cd4-rust-rerun/parallel-controls/`.

Fitting source: `aacac0ac649743345a71732f92321096de74fb61` (clean tracked tree when frozen).

## Reproduction

Exact fit arguments and fitting source revision are in `manifests/rust-cd4.json`.
`source/` freezes that revision's Cargo files, crates, Python source and benchmark
scripts. Build time is outside fit timers. Patient-level outputs, source, binary
and logs are local and ignored by Git.

Submit from the repository root:

```bash
sbatch --parsable --job-name=cd4-rust-qr --cluster=htc --partition=htc --constraint=turin --cpus-per-task=4 --mem=32G --time=01:30:00 --output="$PWD/benchmarks/results/2026-11-09_cd4-qr-controls/logs/%j.out" --wrap 'set -euo pipefail
source /vast/djishnu/Aaron/1_general_use/SLIDE/benchmarks/results/2026-11-09_cd4-qr-controls/environment.sh
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$RUN/source"
export CC="$ENV/bin/x86_64-conda-linux-gnu-cc" CXX="$ENV/bin/x86_64-conda-linux-gnu-c++"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$CC" LIBCLANG_PATH="$ENV/lib" CARGO_BUILD_JOBS=2
export PKG_CONFIG_PATH="$REPO/benchmarks/results/2026-11-02_ssc-cd4-retry-benchmarks/build/pkgconfig"
export CARGO_TARGET_DIR="$REPO/.scratch/2026-11-09_qr-controls/target"
toolchain="$HOME/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin"
export RUSTC="$toolchain/rustc" RUSTDOC="$toolchain/rustdoc" PATH="$toolchain:$PATH"
cargo build --release --locked -p slide-rs --bin slide
cp "$CARGO_TARGET_DIR/release/slide" "$RUN/bin/slide"
sha256sum --check --quiet "$REPO/benchmarks/results/2026-11-02_ssc-cd4-retry-benchmarks/input-files.sha256"
bash benchmarks/scripts/run_backend_array.sh --manifest "$RUN/manifests/rust-cd4.json"
'
```

## Status

Submitted Slurm job `12158255;htc`. Score tests passed (15 passed; one optional R weighted-GLM test ignored). No performance or R-equivalence claim yet.
