# CD4 Rust rerun

Fitting source: `1d972aa34baeffd7dc5fabe50120fa22e260a6ab` (clean tracked tree when frozen). Changes since
`81e1fc9`: buffered checkpoint I/O and saved, separately timed control reporting.

Forty Rust CD4 full fits: ten seeds × four delta/lambda settings, 1,000 selection
draws, 500 sampleCV holdouts (K=4), four workers and one BLAS thread per worker.
Fresh checkpoints include first-time control reporting and exports in every fit.
No R/Python/SSc/final-CV reruns. Inputs and native libraries use the November 2
campaign; original results remain intact.

## Reproduction

The maintained benchmark harness and manifest runner are reused. Exact fit
arguments are in `manifests/rust-cd4.json`; `source/` is a local Git archive of
this revision's Cargo files, crates, Python source and benchmark scripts.
Build is outside fit timers. `bin/`, `logs/` and patient-level `full-fit/` outputs
remain local. `analysis/` holds updated comparisons with unchanged R/Python runs.

Submit from the repository root:

```bash
sbatch --parsable --job-name=cd4-rust-rerun --cluster=htc --partition=htc --constraint=turin --cpus-per-task=4 --mem=32G --time=01:30:00 --output=/vast/djishnu/Aaron/1_general_use/SLIDE/benchmarks/results/2026-11-08_cd4-rust-rerun/logs/%j.out --wrap 'set -euo pipefail
source /vast/djishnu/Aaron/1_general_use/SLIDE/benchmarks/results/2026-11-08_cd4-rust-rerun/environment.sh
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$RUN/source"
export CC="$ENV/bin/x86_64-conda-linux-gnu-cc" CXX="$ENV/bin/x86_64-conda-linux-gnu-c++"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$CC" LIBCLANG_PATH="$ENV/lib" CARGO_BUILD_JOBS=2
export PKG_CONFIG_PATH="/vast/djishnu/Aaron/1_general_use/SLIDE/benchmarks/results/2026-11-02_ssc-cd4-retry-benchmarks/build/pkgconfig"
export CARGO_TARGET_DIR="$REPO/target"
toolchain="$HOME/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin"
export RUSTC="$toolchain/rustc" RUSTDOC="$toolchain/rustdoc" PATH="$toolchain:$PATH"
"$toolchain/cargo" build --release --locked -p slide-rs --bin slide
cp "$CARGO_TARGET_DIR/release/slide" "$RUN/bin/slide"
sha256sum "$RUN/bin/slide" > "$RUN/build-artifacts.sha256"
sha256sum --check --quiet /vast/djishnu/Aaron/1_general_use/SLIDE/benchmarks/results/2026-11-02_ssc-cd4-retry-benchmarks/input-files.sha256
bash benchmarks/scripts/run_backend_array.sh --manifest "$RUN/manifests/rust-cd4.json"
'
```

Status: prepared; no completion claim.

Submitted Slurm job `12153565;htc`: 4 CPUs, 32 GiB, 90-minute limit.

Job `12153565` failed during build before any fits: Conda Cairo pkg-config
required unavailable expat metadata. The original campaign pkg-config path
passes the dependency check. Replacement job `12153620;htc` uses that path, with
unchanged fit settings and runtime libraries.
