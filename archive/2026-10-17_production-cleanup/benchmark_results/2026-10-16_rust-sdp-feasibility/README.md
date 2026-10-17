# Pure-Rust SDP feasibility assessment

Assessed 2026-10-16 against repository HEAD `60a8006`, bundled DSDP sources,
published Rust packages, and upstream documentation. This is source research,
not a solver experiment. No builds, benchmarks, new dependencies, or production
changes were made. Unlike the preceding paired attribution, this assessment
asks whether removing native numerical dependencies is practical and worthwhile.

## Recommendation

**Audit Clarabel configuration before replacement; evaluate an existing
pure-Rust SDP solver before authorizing a DSDP port.**
POUNCE `pounce-convex` 0.12.0 with its FERAL backend is the strongest candidate
found for a bounded trial. Totsu supplies another genuine pure-Rust SDP route,
but its archived status makes it a weaker adoption choice. No ready-made
pure-Rust DSDP implementation was found in the primary-source search; that is
not proof that none exists.

Keep two acceptance criteria separate:

- **Portability:** no separately provisioned or compiled numerical C/Fortran
  library in the chosen dependency graph. Normal Rust standard-library/OS
  linkage is a different requirement from eliminating BLAS/LAPACK or DSDP C.
- **R compatibility:** match the installed R construction and its downstream
  behavior within agreed numerical limits. A Rust implementation of the same
  mathematical SDP does not establish this.

The portability project is meaningful if native toolchain installation is a
product constraint. A full DSDP rewrite is not justified by the present evidence
alone. It would create a numerical solver maintenance obligation without
guaranteeing R agreement, and would not remove the project's other native
dependencies.

## What a DSDP port actually requires

The saved R call needs p dual variables, one order-p positive-semidefinite
block, and 2p scalar inequalities:

maximize sum(s), subject to 0 ≤ s ≤ 1 and 2G − diag(s) ⪰ 0.

R returns the dual vector `sol$y`. It then clips, applies its correlation-space
feasibility shrink, and rescales. The actual installed formulation is retained
in the [preceding attribution captures](../2026-10-16_s-construction-attribution/stages/chunk_1/r_installed_provenance.txt).

DSDP uses a dual-scaling interior-point algorithm. A port needs initialization,
barrier updates, search directions, line searches, convergence/error decisions,
and solution recovery, not just a replacement eigensolver. The algorithm
assembles and solves Schur-complement systems and uses factorizations for
feasibility tests. [DSDP API and algorithm guide](https://www.mcs.anl.gov/hs/software/DSDP/DSDP5-API-UserGuide.pdf).

The repository's C implementation makes the scope concrete:

| Component | Local evidence under `src/loveslide/dsdp_solver/dsdp/C/` |
| --- | --- |
| Construction and solve sequence | `pyreadsdpa.c:179–317` |
| Dual-scaling loop and directions | `allc/dualalg.c:121–214,370` |
| LP cone for scalar inequalities | `allc/dsdplp.c:509` |
| Internal residual and bound cones, Schur and CG setup | `allc/dsdpsetup.c:58–68` |
| Dense/sparse Schur dispatch | `allc/cholmat.c:321–342` |
| BLAS/LAPACK operations | `allinclude/dsdplapack.h:174–213` |

Required numerical operations include Cholesky factorization/solve/inversion,
symmetric eigenanalysis, triangular solves, matrix-vector products, rank
updates, and dot/norm/scaling operations. Pure-Rust linear algebra can supply
these operations, but replacing packed storage, operation order, or backend
dispatch can change the finite-precision trajectory.

The bundled core has **60 C files / 26,911 physical lines**, plus **26 headers /
2,272 lines**, excluding the Python frontend. The main iteration is only 545
lines, but depends on LP, PSD, Schur, CG, and numerical kernels. These counts
are a scope indicator, not a required translation count or schedule estimate.

We could omit Python/NumPy bindings, SDPA file parsing, command-line display,
and unused general-purpose variants. A specialized diagonal-SDP solver could
be much smaller than the full library, but would need its own numerical
validation and could not be assumed to reproduce DSDP. No runtime reachability
audit has established a minimum source subset.

The bundled `src/loveslide/dsdp_solver/LICENSE:1–10` declares GPL v3-or-later
unless overridden by nested notices; the workspace declares MIT. Source
provenance and licensing need review before translating or redistributing the
bundle. This is a record of notices, not a legal conclusion about upstream
DSDP. Identity between the bundled numerical sources and the installed Rdsdp
build has not been established.

## Native dependencies already present in Rust

Removing DSDP C would not make the current workspace native-dependency-free:

| Existing dependency/use | Evidence |
| --- | --- |
| `nalgebra-lapack` with OpenBLAS | `Cargo.toml:29` |
| BLAS-enabled `ndarray` | `Cargo.toml:30` |
| System OpenBLAS | `Cargo.toml:36`, `openblas-src` feature `system` |
| Clarabel SDP BLAS/LAPACK providers | `crates/slide-sdp/Cargo.toml:19–21` |
| LAPACK eigen, Cholesky, SVD/pseudoinverse | `crates/slide-types/src/lib.rs:5,166,204,224` |

Shared decompositions serve sampling, LOVE, and fitting, not only the SDP:
`crates/slide-knockoff/src/lib.rs:1113–1162`,
`crates/slide-love/src/pure_hetero.rs:329–388`, and
`crates/slide-core/src/lib.rs:1193,1218`.
Replacing those backends is a separate numerical change that needs validation
outside the two S-construction chunks. It should not be hidden inside a solver
swap. Rust-native `faer` supplies dense decompositions and is a possible backend,
not an SDP solver or a LAPACK-rounding emulator.
[faer documentation](https://docs.rs/faer/0.23.2/faer/).

## Off-the-shelf options

### First: configuration is not exhausted

Clarabel exposes tolerances, iteration limits, equilibration, KKT
regularization, iterative refinement, presolve, and step controls.
Configuration may improve robustness or change finite-precision S; the paired
experiment did not prove that every configuration would fail R agreement.
[Official settings](https://clarabel.org/stable/api_settings/).

The earlier 16-chunk tolerance experiment did test simple tightening: at
1e−9, 8 of 16 solves failed; at 1e−10, all 16 failed. Smaller tolerances drove
many small entries closer to zero rather than R's positive residues. Those
were earlier covariance fixtures, not a comprehensive configuration test on
the latest two diagnostic chunks. See the
[recorded experiment](../2026-10-13_clarabel-diag-s/README.md).

The local `clarabel_settings` function (`crates/slide-sdp/src/lib.rs:201–218`)
also sets reduced-accuracy tolerances to 0.1 times the main tolerances, making
them stricter rather than looser. That differs from upstream's fallback
settings and merits review, but is not an established cause or correction:
the latest captured solves returned `Solved`, not `AlmostSolved`.

A small hypothesis-driven configuration audit is cheaper than a solver
replacement. Do not search settings to force the two vote cutoffs, infer
that higher iteration limits alone fix numerical failure, or promise that a
configuration reproduces DSDP's trajectory. Configuration also cannot remove
the installed SDP backend's BLAS/LAPACK dependencies.

### POUNCE: first candidate to evaluate, not an approved replacement

The published `pounce-convex` 0.12.0 package contains a PSD cone implementation,
Nesterov–Todd scaling, and Rust cyclic-Jacobi eigenanalysis through
`pounce_linalg::symmetric_eigen`. Its `tests/sdp_cone.rs` solves small PSD
problems with `FeralSolverInterface`, including an infeasible example. These
are real SDP constraints, not merely a positive-semidefinite quadratic
objective. The research worker inspected the published archive; the coordinator
independently checked its manifest, PSD source, and test entrypoints in memory.
[Published package](https://crates.io/api/v1/crates/pounce-convex/0.12.0/download),
[PSD source](https://github.com/jkitchin/pounce/blob/main/crates/pounce-convex/src/cones/psd.rs).

Upstream documents a pure-Rust default numerical backend without system BLAS,
Fortran, or HSL. Select the FERAL route, not optional HSL. Upstream distinguishes
its established NLP workflow from small dense SDP support; the published SDP
tests call PSD its least-exercised symmetric cone at program level. That warrants
a trial, not a production-readiness claim for ill-conditioned knockoff matrices.
[POUNCE repository](https://github.com/jkitchin/pounce),
[published SDP tests](https://docs.rs/crate/pounce-convex/0.12.0/source/tests/sdp_cone.rs).

Important adoption costs:

- It is not DSDP's algorithm and has no demonstrated installed-R parity here.
- `OptimalInaccurate`, iteration limits, and numerical-failure statuses must
  remain visible; do not turn every returned vector into success.
  [Status API](https://docs.rs/crate/pounce-convex/0.12.0/source/src/qp.rs).
- Edition 2024 requires at least Rust 1.85, whereas this workspace declares
  Rust 1.79 (`Cargo.toml:20`). Dependencies could impose a higher minimum.
  [Published manifest](https://docs.rs/crate/pounce-convex/0.12.0/source/Cargo.toml),
  [Rust edition guide](https://doc.rust-lang.org/edition-guide/rust-2024/index.html).
- Its declared EPL-2.0 license needs a distribution review. The supporting
  crates include Rayon and Unix `libc` bindings; thread budgeting and each
  target's resolved dependency graph still need checking. Absence of numerical
  native libraries is not absence of OS interfaces.
  [Common manifest](https://docs.rs/crate/pounce-common/0.12.0/source/Cargo.toml).

No clean build, complete resolved dependency audit, or platform test was run.

### Totsu: genuine alternative, weaker maintenance case

`totsu` 0.10.2 / `totsu_core` 0.1.1 supplies SDP modeling and a first-order conic
solver. `FloatGeneric<f64>` implements its operations and Jacobi eigenanalysis
in Rust. Avoid its separate LAPACK and CUDA backends for this requirement.
The author describes the generic backend as slow, and the repository was
archived on 2026-03-07. It is a useful fallback/reference candidate, not a clear
maintenance win or DSDP-equivalent implementation.
[Totsu repository](https://github.com/convexbrain/Totsu),
[FloatGeneric source](https://github.com/convexbrain/Totsu/blob/master/solver_rust_conic/totsu_core/src/floatgeneric.rs).

### Clarabel: retain for comparison, not native-free SDP as configured

Installed Clarabel 0.11.1's `sdp` feature requires BLAS/LAPACK. Its
`faer-sparse` feature changes sparse KKT factorization, not the dense SDP-cone
backend. A Rust dense-backend conversion would be custom work and would retain
the Clarabel-versus-DSDP algorithm difference. It is not an existing toggle
that solves both requirements.
[Manifest](https://raw.githubusercontent.com/oxfordcontrol/Clarabel.rs/main/Cargo.toml),
[0.11.1 SDP scalar requirements](https://docs.rs/clarabel/0.11.1/src/clarabel/algebra/floats.rs.html).

## Is the change worth pursuing?

**Yes to a bounded portability evaluation; no to committing to a DSDP rewrite
yet.** The [paired attribution](../2026-10-16_s-construction-attribution/README.md)
shows consequential selection sensitivity, so the solver is worth examining.
It does not show that DSDP improves false-discovery control, power, or numerical
optimality. Earlier [boundary diagnostics](../2026-10-13_clarabel-diag-s/README.md)
support zero optima and solver-dependent small residues. Reproducing one
installation's residues is a different goal from solving the SDP accurately.

Proposed next step, requiring separate implementation approval:

0. Review the existing Clarabel overrides against its pinned defaults and
   recorded residuals/statuses. If a specific numerical mechanism warrants a
   small fixed configuration comparison, use the existing paired harness and
   heldouts; do not launch a broad tolerance sweep. This addresses numerical
   behavior, not native-library removal.
1. If native-library removal remains required, pin POUNCE's candidate packages
   in an isolated evaluation, inspect the
   resolved target-specific graph, and test builds without numerical native
   libraries on the supported targets. Do not change production defaults.
2. Reuse the existing capture/replay harness on the two diagnostic matrices
   and the preselected `corr3`, `cov4`, and near-singular `rSig4` fixtures.
   Record status, feasibility, objective/duality measures, raw S, zeros,
   factorization/jitter, runtime, and fixed-draw selections. Set numerical
   acceptance criteria before running; do not tune for Z20/Z74 votes.
3. Decide whether it meets the mathematical and portability requirements.
   Different R votes are evidence to explain, not automatically a solver defect.
   If exact reference behavior remains mandatory, neither candidate currently
   has evidence sufficient for adoption.
4. Only after an acceptable solver result, scope replacing the shared
   BLAS/LAPACK-backed operations separately. A full native-free claim needs
   end-to-end target builds and numerical validation, not a solver-only test.

If the candidate fails, stop and review the reason before choosing a custom
DSDP port or Clarabel backend work. A full rewrite should require an explicit
long-term commitment to numerical tests, platform support, and maintenance.

## Research method and limits

Reused the preceding attribution and existing bundled implementation; no new
analysis or replay script was needed. Read scoped manifests/source, counted
the bundled C files with `wc -l`, checked upstream primary documentation, and
inspected the published POUNCE archive without extracting it to disk. Research
handoff: `.agent-handoff/2026-10-16_rust-sdp-options.md` (not committed).

No implementation, benchmark, performance estimate, cross-platform build,
complete dependency-closure audit, source-equivalence proof, or legal review
was performed. No push or merge. Existing outputs and `job_11935828.sh` were
preserved.
