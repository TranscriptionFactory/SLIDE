# Retained benchmark evidence

Start with the [runtime and latent-factor comparisons](../README.md#measured-backend-comparisons).
These measurements describe frozen source revisions, not a benchmark of the
current checkout. No benchmarks were rerun for the 2026-10-17 cleanup.

| Run | Evidence retained |
| --- | --- |
| [Production-readiness pilot](2026-10-16_production-readiness/README.md) | Latest repeated SSc/CD4 runtime, memory, significant-LF comparisons, partial truth results and paired solver replay |
| [Large-feature scaling](2026-10-16_large-feature-scaling/README.md) | Runtime and memory at 10,000–20,000 features, including R timeouts |
| [Torok binary](2026-10-17_torok-all-binary/README.md) | Real-data binary capacity, selected factors and interactions, runtime and memory |
| [Four-arm comparison](2026-10-16_backend-replicates-four-arms/README.md) | Significant-LF frequencies and set agreement, including Rust with R knockoffs; not replaced by the three-backend pilot |
| [Four-arm setup](2026-10-16_backend-replicates-all-arms/README.md) | Frozen configuration and provenance for the four-arm comparison, including the excluded timed-out arm |

Keep tables, figures, job records and reproduction commands together within each
run. Raw patient-level inputs and some large intermediate outputs remain on the
shared filesystem; this repository is not a self-contained replay bundle.

Older solver investigations and superseded runs are in the
[cleanup archive](../archive/2026-10-17_production-cleanup/README.md).
The pilot's replay still uses an archived R diagnostic and historical scratch
inputs. Fixture provenance also points into the archive.

Independent random draws, small replicate counts, cancelled simulations and
unverified SSc sample identity limit interpretation. Matching factor counts does
not establish matching selected sets, FDR control or production equivalence.
