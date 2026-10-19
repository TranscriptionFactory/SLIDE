# Benchmark results

Saved runtime, memory, and factor-selection comparisons for specific source
revisions. See the [benchmark guide](../README.md) for tools and summary tables,
and the [revision mapping](../README.md#source-revisions-after-history-cleanup)
to locate the measured source.

| Run | Contents |
| --- | --- |
| [Production-readiness pilot](2026-10-16_production-readiness/README.md) | Repeated SSc/CD4 runtime, memory, [full selected-LF counts and pairwise overlap](2026-10-16_production-readiness/README.md#backend-selection-details), partial truth results and paired solver replay |
| [Large-feature scaling](2026-10-16_large-feature-scaling/README.md) | Runtime and memory at 10,000–20,000 features, including R timeouts |
| [LS skin transcriptomics](2026-10-17_torok-all-binary/README.md) | Localized scleroderma/control binary capacity, selected factors and interactions, runtime and memory |
| [Backend replicate campaign](2026-10-16_backend-replicates/README.md) and [five-arm setup](2026-10-16_backend-replicates/SETUP.md) | Significant-LF frequencies and set agreement, including Rust with R knockoffs; frozen configuration and provenance, with partial Python-with-R coverage explicit |

- [Five-backend selection heatmap](2026-10-17_hybrid-selection/README.md): native and R-knockoff arms from the October 16 campaign, including partial Python-hybrid coverage.

## Reproduction requirements

Raw patient-level inputs and some large intermediate outputs remain on the
shared filesystem. Reproducing these runs requires access to those files.

Replay helpers are in [`../scripts/`](../scripts/); some still require historical
external controls and scratch inputs. Find older solver investigations and superseded runs in the historical
[cleanup archive](https://github.com/TranscriptionFactory/SLIDE/blob/0c030651bf4db5dd0d184e09f7274ab319d66bce/archive/2026-10-17_production-cleanup/README.md).
The pilot's frozen `replay.sbatch` needs the [historical replay scripts](../README.md#maintained-tools-and-saved-results)
and external scratch inputs; it is not runnable from the current scripts tree.
Fixture provenance links point to historical commits.

## Interpretation

Independent random draws, small replicate counts, cancelled simulations and
unverified SSc sample identity limit interpretation. Matching factor counts does
not establish matching selected sets, FDR control or production equivalence.
