# Heldout S-stage validation

Completed on 2026-10-16. No source changes, tuning, or vote replay.

Preselected inputs: `tests/fixtures/sdp/{corr3,cov4}/input_sigma.csv` and
`.tmpfiles/2026-10-12_python-scale-bug/rSig_4.txt` (not the diagnostic chunks).

| Fixture | Dimension | Sigma condition number | Fixed-G raw solver max abs Δs | Max postprocessor abs ΔS across raw sources | Min final shifted-Sigma eigenvalue |
| --- | ---: | ---: | ---: | ---: | ---: |
| corr3 | 3 | 9.572 | 2.030e-4 | 5.340e-7 | 2.431e-9 |
| cov4 | 4 | 4.986 | 2.178e-5 | 2.988e-7 | 2.300e-10 |
| rSig4 | 22 | 1.168e5 | 3.931e-6 | 1.082e-4 | 1.077e-10 |

All 18 crossed S arms are finite, nonnegative, and numerically feasible on Sigma.
Every Clarabel call returned `Solved`; every full R call returned `PDFeasible`.
R repeated ordinary calls were identical; extracted R postprocessing matched the
installed entrypoint exactly on each DSDP vector. Native Rust staged output
matched its public API exactly (diagnostic assertion). No errors occurred.

On rSig4, Clarabel clipping yields 9 zeros versus DSDP's 0. R's grid repair uses
shrink 0.999 for either raw solver. Rust's repair uses approximately 0.9998224
for fixed-G Clarabel and 0.9999988 for DSDP. Extra covariance repair is inactive
on every arm. Raw Δs is in normalized correlation units; final ΔS is in
covariance units, so their magnitudes are not directly comparable in this table.
These are S-stage checks, not claims about heldout selection behavior.

Native versus fixed-R-G Clarabel raw deltas: 0 (corr3), 2.431e-14 (cov4),
2.722e-8 (rSig4). DSDP raw passed through Rust native versus fixed-G repairs has
final S delta 0, 0, and 5.908e-13, respectively. No normalization correction was
chosen or tuned from these results.

## Artifacts and loading

- `stage_results.csv`: 18 rows with condition numbers, clipping zeros, both
  shrink stages, min eigenvalues, and native-S deltas.
- `comparisons.csv`: paired raw/postprocessor/normalization deltas.
- `.tmpfiles/2026-10-16_s-construction-attribution/heldout/`: full stage dumps,
  logs, `commands.json`, and disposable `run_heldout.py` driver.

CSV fixture inputs are parsed once with NumPy `loadtxt(..., delimiter=',')`
into binary64; rSig4 hexadecimal tokens are parsed with Python `float.fromhex`.
R receives those exact values as little-endian binary64 `rust_sigma.raw`.
Rust receives headerless `%.17e` CSV; exact binary64 roundtrips are asserted,
as is R's re-emitted Sigma. Thus R `read.csv` is not a separate original-input
parser. Each R call saves package paths, function bodies, and session provenance.
Covariance eigenvalues/conditions in tables use NumPy `eigvalsh`/`cond`.

## Reproduction

From repository root, run:

```sh
python .tmpfiles/2026-10-16_s-construction-attribution/heldout/run_heldout.py
```

The driver deliberately refuses to overwrite existing run directories. Preserve
this run and choose a new output directory before rerunning. Exact per-arm
commands and logs are listed in `commands.json` (not a maintained new workflow).
It reuses `tmpfiles/2026-10-16_s-construction-attribution/trace_r_asdp.R` and
`target/debug/deps/slide_sdp-f8f71c3c7b2cef83 emit_sdp_stages --ignored --nocapture`,
with `SDP_DIAG_DIR` set for each arm; it never invokes Cargo or rebuilds.
R executable: `/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev/bin/Rscript`.
R 4.5.3; knockoff 0.3.6 and Rdsdp 1.0.6 from that environment's library paths,
matching the primary attribution captures. No `R_LIBS` override is needed here.
Rust binary was built with rustc/cargo 1.99.0 and the project linker config.
`TMPDIR`, `TMP`, and `TEMP` point to the heldout project-local `scratch/`;
OpenBLAS/OMP/MKL/BLIS/vecLib thread counts are 1. The installed R reference uses
`is_posdef(..., tol=1e-9)` with the unchanged eps decade schedule.

Not checked: downstream draws, votes, full pipelines, or statistical power.
