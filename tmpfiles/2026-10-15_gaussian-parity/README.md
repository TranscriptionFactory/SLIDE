# Fixed-draw Gaussian parity replay

Source fixes: `28b1679` and `6cbea3c`; baseline fitter `37bbf94`.
This retained diagnostic reuses `benchmarks/replay_paired_xtilde.py::fit_draw`,
`paired_vote_summary`, `plot_sampler_votes`, the existing selection aggregation,
production Python statistics and `r_stat.R`. The original replay command asserts
an old tie-zero counterfactual; that assertion is unsuitable for changed fitters.
This driver compares them without regenerating knockoffs. Saved paths and n=24
are deliberate limits of this investigation, not a new general benchmark API.

From the repository root:

```bash
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export TMPDIR="$PWD/.tmpfiles/2026-10-15_gaussian-parity"
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$TMPDIR/mpl"
DRIVER=tmpfiles/2026-10-15_gaussian-parity/replay.py
# Each output directory must be new. Use a new dated results directory for a new run.
"$ENV/bin/python" "$DRIVER" <new-output>/fixed-draws \
  --designs native_chunk_1 native_chunk_4 saved_stage2 saved_c8
"$ENV/bin/python" "$DRIVER" <new-output>/python-production --python-production \
  --designs native_chunk_1 native_chunk_4 saved_stage2 saved_c8
"$ENV/bin/python" "$DRIVER" <new-output>/high-delta-isolation --designs saved_highdelta
"$ENV/bin/python" "$DRIVER" <new-output>/high-delta-python --designs saved_highdelta --python-production
```

Optional `--draws 2` runs a smoke check. The driver expects rebuilt
`baseline-prepared-xtilde` and `schedule-prepared-xtilde` under TMPDIR, and a
`baseline-source` archive. Baseline and candidate use the same updated
`crates/slide-cli/examples/paired_xtilde.rs`, but different `slide-glmnet` source.
Build with `cargo build --release --locked -p slide-rs --example paired_xtilde`.
Use separate Cargo targets, or archive workspace fingerprints between source
snapshots; verify the copied binaries differ. Rebuild Fortran using the existing
Meson configuration into a new directory, archive the former extension, then
install the newly built extension. Do not reuse frozen benchmark binaries.

Results and caveats: `benchmark_results/2026-10-15_gaussian-parity/README.md`.
The default eight-arm run includes diagnostic R-supplied preparation, not a
production Rust arithmetic fix. `--python-production` isolates the two actual
Python fixes. It loads baseline Python code from the source archive and shares
the rebuilt Fortran extension across arms. Fixed-draw files, entry indices, W,
selected sets, vote summaries and plots remain in each output directory.
