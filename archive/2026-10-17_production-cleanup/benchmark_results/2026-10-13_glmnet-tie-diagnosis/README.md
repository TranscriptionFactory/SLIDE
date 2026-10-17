# Why glmnet ties near-duplicate knockoff pairs and slide-glmnet does not, and the Rust change that matches it

Job 5 of `.agent-handoff/2026-10-13_clarabel-plan.md`. Follows `../2026-10-13_s-zero-handling/` (same 200 draws ×
4 s arms, δ=0.01, λ=1.0, chunk 0, n=24, 2p=44). Decision already taken: Rust must behave like R for s_j ≈ 0 (the
same behaviour, tolerances and thresholds; not a bit-for-bit port). R and Python stay as they are.

## R's effective rule, and what reproduces it (s-floor sweep)
**R's rule.** R sets W_j = 0 when X_j and X̃_j have their first nonzero coefficient at the same point of the λ grid.
The grid has nlambda 500 from λ_max down to λ_max/2000, a 1.53% step, and glmnet's convergence `thresh` is 1e-7.
This is **not a threshold** on distance, s_j or entry gap. It is a probability that falls smoothly as the pair
separates. Let δ_j = sqrt(2(1 − corr(X_j, X̃_j))) be the standardized distance between the two columns. On these
draws δ_j ≈ 0.966·sqrt(2 s_j/Σ_jj).

`floor_sweep.py` sets all 17 near-zero s of chunk 0 to f·Σ_jj, with f from 1e-9 to 1e-4. It shrinks the other
entries by the smallest factor that keeps 2Σ − diag(s) PD (0 to 10%), because a larger floor is infeasible without
that. It then scores 200 draws per floor with R, current Rust at tol 1e-4 and 1e-7, and the entry-order mirror at
tol 1e-7. `sweep_summary.py` writes `sweep_tie_curve.csv` and `sweep_rule_agreement.csv`.

R's tie rate, over near-duplicate pairs that R entered:

| s/Σ_jj floor | 1e-9 | 1e-8 | 3e-8 | 1e-7 | 3e-7 | 1e-6 | 3e-6 | 1e-5 | 1e-4 |
|---|---|---|---|---|---|---|---|---|---|
| median δ | 4.3e-5 | 1.4e-4 | 2.4e-4 | 4.3e-4 | 7.5e-4 | 1.4e-3 | 2.4e-3 | 4.3e-3 | 1.4e-2 |
| R tie rate | 0.76 | 0.75 | 0.73 | 0.69 | 0.63 | 0.51 | 0.37 | 0.23 | 0.08 |

- The rate is flat at about 0.75 for δ ≤ 3e-4 (s/Σ ≲ 3e-8) and crosses 0.5 near δ ≈ 1.5e-3 (s/Σ ≈ 1e-6). Above
  δ = 1e-2 it is below 0.1.
- Genuine knockoffs (s not near zero) tie in 0.3% of entered pairs.
- The 25% of near-copies that R does not tie are pairs that enter low on the path, where other updates separate
  the two columns.

**Candidate Rust rules.** Per-draw sign(W) agreement with R; selection rate at FDR 0.1.

| fitter / rule | Z20+Z21 agreement, over floors | near-dup agreement | Z20 sel. at 1e-9 / 1e-7 / 1e-6 / 1e-4 (R: .005/.025/.095/.075) |
|---|---|---|---|
| Rust now (tol 1e-4) | 0.41–0.76 | 0.54–0.71 | .335 / .285 / .13 / .115 |
| Rust, tol 1e-7 only | 0.44–0.99 | 0.72–0.93 | .30 / .095 / .115 / .075 |
| Rust + W=0 if δ < 3e-4 / 1e-3 / 2e-3 | ≤ 0.95 at small floors, 0.54–0.76 near τ | ≤ 0.76 | 0 / 0 / 0–.135 / .115 (too many zeros below τ, none above) |
| Rust + W=0 if entry gap ≤ 1 or ≤ 3 grid steps | 0.41–0.75 | 0.49–0.69 | ≈ Rust now (the near-copies' gap is ≫ 3 steps) |
| **entry-order active set + tol 1e-7** | **0.978–1.000** | **0.89–0.97** | **.010 / .030 / .100 / .075** |

R's tie is a property of the coordinate-descent path, not a function of δ. A δ rule makes every pair below τ
unselectable. R does not do that: it selects Z20 in 2.5–9.5% of iterations at floors 1e-7 to 1e-6, and it ties
pairs above τ. A δ rule therefore misses R in both directions, and the size of the miss depends on τ, a
tolerance R does not have. An entry-gap rule never fires on the pairs that matter, because Rust leaves the second
column out for over 100 steps. **The simplest change that reproduces R across the whole sweep is the fitter
change specified below.** Z20/Z21 selection rates then stay within 0.02 of R at every floor (largest gap: Z21 at 1e-6, 0.17 vs 0.15), and per-draw
agreement for those two factors is 0.978–1.0. The target is 0.98–1.0, the Python-vs-R rate on the s-zero arms;
Python was not run on the sweep.

## Does R's S-side handling matter for s_j ≈ 0?
- `create.solve_sdp` clips s to [0,1] and then shrinks it by (1 − s_eps) until 2Σ − diag(s) is PD (is_posdef
  tolerance 1e-9). For s ≈ 0 this only makes a tiny value tinier, so it has no behavioural effect.
- `create.gaussian` runs `chol(Σ_k)` with no jitter. With an exact s_j = 0 R would stop with an error. DSDP's
  positive residue (1e-10 to 1.7e-8) avoids that. Rust's 0 plus its 1e-10 jitter gives X̃ about as close to X as
  s/Σ ≈ 5e-11 would. This is the one R behaviour Rust lacks; it would be a hard failure, not a selection
  difference. Copying it is not recommended.
- **0 vs 1e-9 is behaviourally irrelevant.** DSDP residues of 1e-10 to 1.7e-8 give δ of 1.4e-5 to 1.8e-4.
  Clarabel's 0 gives δ ≈ 1.1e-5. Both lie on the flat part of R's tie curve (0.75–0.76), and after the fitter
  change the rust_s and r_s arms both match R per draw (table further down).
- Z74 (R s = 1.18e-8 at λ 0.1, 1.65e-8 at λ 1.0; Clarabel 0 or 3.8e-9): its δ is about 1.5e-4 under either S,
  where R's tie rate is 0.75 against 0.76. **Inferred** from the chunk-0 curve; chunk 3 was not swept. The S
  difference should therefore move Z74's tie probability by about 0.01. The 5.4% vs 11.6% Z74 vote gap seen in
  job 2 has to come from other S entries or from the fitter, not from Z74's own s_j ≈ 0.
- The fitter change is not a tie rule, but it does make the Rust-vs-R S difference irrelevant here, because the
  tie outcome does not depend on δ within that range.

## Mechanism
The difference comes from the order in which the coordinate descent visits the active set, plus the knockoff
tolerance. It does not come from the lambda grid, standardization, strong rules or the KKT check, which are the
same in both.

- glmnet (`glmnet5.f90` `elnet2` naive, which Python runs because 2p > n, and `elnet1` covariance, which R runs)
  sweeps its active set in **first-entry order** (`ia(1..nin)`, appended when a coefficient first leaves 0 and
  never removed). Each new λ starts with this inner loop (`if(iz*jz.ne.0) go to 10360`). Then a full sweep over
  the strong set runs in column order, then the KKT check.
- `slide-glmnet` `coordinate_descent_step` (`crates/slide-glmnet/src/lib.rs:1410`) rebuilds its active set as
  "candidates with β≠0" in **column order**, both at the start of the call and after each candidate sweep.

For a pair (X_j, X̃_j) with X̃_j − X_j ≈ 1e-5 relative, both columns cross λ at the same grid point. The full
sweep sets β_left first and β_right, if at all, to a small value (~3e-4 against ~1e-2). What happens next decides
whether the right column is still nonzero when the path is stored at that λ:

- glmnet: in the inner loop the right column comes **immediately after** the left one, because it entered right
  after it. After the left update the pair's gradient is exactly λ. The right column then sees u = λ + d + β_right·xv,
  with d = (x_right − x_left)ᵀr/n, so it keeps β_right ≠ 0 and grows it by d each sweep. Both columns enter at
  the same λ, so W = 0.
- Rust: in column order the columns j+1 … p+j−1 (21 and 22 here) are updated **between** the left and the right
  column. The right column takes up their effect on the residual. Its gradient drops below λ, it is set back to
  0 and stays out until a much later λ. The left column, which comes first in [X | X̃], always wins, so the sign
  of W follows the swap bit.

Instrumented step, draw 0, Z20 (cols 20/42, swap = False, λ index 20 = 3.9562136; `trace_draw0_z20.txt`):

| fitter | full/candidate sweep at λ20 | next active-set sweeps at λ20 | stored β (20, 42) at λ20 | entry idx (20, 42) |
|---|---|---|---|---|
| glmnet naive mirror (tol 1e-7) | 20: 0→1.32e-2; 42: grad 3.9565213 > λ → 3.21e-4 | order […, 18, 0, 22, 21, 43, 40, **20, 42**]; 42: grad 3.9565447, 3.9565680, … → 3.46e-4, 3.70e-4, 3.94e-4 | 2.46e-2, 6.64e-4 | 20, 20 → W = 0 |
| Rust mirror, tol 1e-7 | 20: 0→1.33e-2; 42: grad 3.9565480 > λ → 3.49e-4 | order [0, 18, **20, 21, 22, 42**]; 42: grad 3.9561952 < λ → **0**, stays 0 | 2.54e-2, 0 | 20, 107 → W = +λ20 |
| Rust mirror, tol 1e-4 (production) | 20: 0→1.09e-2; 42: grad 3.9552442 < λ → 0 (residual from the loose earlier fit) | — | 1.09e-2, 0 | 20, 125 → W = +λ20 |

## Mirrors and how they were validated (`cdsim.py`, `validate.py`)
`rust_cd` is a line-by-line Python copy of `run_gaussian_cd` / `coordinate_descent_step` / `sweep_coordinates`
(cached path, ddof-1 scaling, R grid). `glmnet_naive` copies `standard1` (isd=0, intr=1) and `elnet2`. The Rust
mirror reproduces the recorded `paired_xtilde` W for Z20/Z21: sign in 398/400 draws (rust_s arm), magnitude to
6.5e-16 relative where both are nonzero, at tol 1e-4 and 1e-7. The glmnet mirror reproduces the recorded Python
glmnet W in 20/20 checked cases. Against R (covariance mode) it agrees on 99.5–100% of Z20/Z21 draws and on
96.4–96.8% of all 22 factors.

## Candidate change, scored against R glmnet per draw (`experiment.py`, `summarize.py` → `focus_summary.csv`, `all_factor_sign_agreement.csv`)
`active_order="entry"` is the proposed change applied to the Rust mirror. The target is Python-glmnet-vs-R
agreement from the s-zero run.

| arm | factor | target | Rust now (1e-4) | entry order, 1e-4 | Rust 1e-7 | **entry order, 1e-7** | R sel. | entry 1e-7 sel. | Rust now sel. |
|---|---|---|---|---|---|---|---|---|---|
| rust_s | Z20 | 1.000 | 0.025 | 0.990 | 0.025 | **1.000** | 0 | 0 | 0.32 |
| rust_s | Z21 | 0.985 | 0.730 | 0.040 | 0.615 | **0.990** | 0 | 0 | 0.05 |
| r_s | Z20 | 1.000 | 0.085 | 0.970 | 0.090 | **1.000** | 0 | 0 | 0.305 |
| r_s | Z21 | 0.985 | 0.740 | 0.100 | 0.680 | **0.990** | 0 | 0 | 0.045 |
| r_s_nearzero_to_0 | Z20 / Z21 | 1.000 / 0.985 | 0.020 / 0.720 | 0.985 / 0.030 | 0.025 / 0.605 | **1.000 / 0.990** | 0 / 0 | 0 / 0 | 0.32 / 0.05 |
| rust_s_floor_1e-9 | Z20 / Z21 | 0.980 / 0.985 | 0.080 / 0.760 | 0.940 / 0.105 | 0.160 / 0.750 | **0.985 / 0.985** | 0.005 / 0.005 | 0.01 / 0.01 | 0.325 / 0.065 |

(The numbers are per-draw sign(W) agreement with R. "sel." is the selection rate at FDR 0.1, offset 0.)

Over all 22 factors, sign agreement with R is 0.667–0.671 for Rust now, 0.930–0.938 for entry order at 1e-7, and
0.964–0.968 for the glmnet mirror. Entry order on its own, at tol 1e-4, fixes Z20 but gives Z21 (the λ_max column)
±W in 90–96% of draws. **Both parts of the change are required.** Keeping the candidate list in column order after
KKT additions (`sorted_check.py`, 40 draws) raises all-factor agreement from 0.943 to 0.952; the glmnet mirror
gets 0.972 on the same draws. Inferred: most of the remaining gap is glmnet's monotone strong set and its
first-λ strong rule, plus covariance-mode versus naive-mode rounding in R.

## Spec: minimal Rust change
1. **Entry-ordered, never-shrinking active set** (`crates/slide-glmnet/src/lib.rs`)
   - `GaussianCdWorkspace` (1244ff): `active_indices` becomes the persistent glmnet `ia`. Add
     `in_active: Vec<bool>`, sized total_p. `reset()` clears both. The vector stays allocated and is no longer
     cleared per call.
   - `coordinate_descent_step` (1410): remove the two `active_indices.clear(); extend(filter β≠0)` blocks. After
     each candidate sweep, append, in `candidate_indices` order, every j with `beta[j] != 0 && !in_active[j]`. This
     equals glmnet's append-on-first-change, because a coordinate can only become nonzero in the candidate sweep and
     is visited once per sweep. The active sweep visits `active_indices` as stored, including coefficients that went
     back to 0 (glmnet does the same).
     ```
     loop iter in 0..max_iter:
         if !active.is_empty(): mc = sweep(active); if mc >= thr: continue
         cc = sweep(candidates)
         for j in candidates: if beta[j] != 0 && !in_active[j] { active.push(j); in_active[j] = true }
         if max(cc, mc) < thr: break
     ```
   - `run_gaussian_cd` (1280): after the KKT loop appends violators, sort `candidate_indices`. glmnet's `ix` is a
     mask, so its full sweep always runs in column order. This step is optional but faithful (+0.01 all-factor).
   - The `coordinate_descent_step` signature can stay (`active_indices: &mut Vec<usize>` plus a new `&mut [bool]`).
     `run_binomial_cd` is unchanged.
2. **Knockoff tolerance 1e-7**: set `DEFAULT_KNOCKOFF_TOLERANCE` (lib.rs:517) to 1e-7 = glmnet `thresh`, and keep
   max_iter at 1000 per λ. On these draws the iteration count is the same at 1000 and 1e5. The `SLIDE_KNOCKOFF_TOL`
   override stays.
3. `slide-knockoff` needs no change. `signed_max_statistic` (903) already returns 0 on exact ties, and the swap
   handling (371-399) is correct.

This is not a post-hoc tie patch. glmnet still gives nonzero W in 0.5–7% of draws for Z20/Z21, and the
entry-ordered fitter matches those draws one by one (agreement 0.985–1.0).

## Tests and fixtures
- Should pass unchanged (inferred; not run, since crates/ was read-only for this job):
  `cached_lambdasmax_matches_uncached`, `signed_max_statistic_zeroes_exact_entry_ties`, the grid tests,
  `paired_xtilde_grid_divergence.rs`, `gaussian_convergence_uses_weighted_squared_update_and_strict_threshold`
  (traced by hand: still 2 iterations), and `knockoff_voting_is_deterministic_with_cached_sampling`.
- Expected to change value, but they use tolerance or correlation asserts: `r_parity.rs`
  `gaussian_path_matches_r_glmnet_fixture` (default tol already 1e-7; β moves at the CD-order level, so it should
  move closer to glmnet) and `gaussian_knockoff_statistics_match_r_glmnet_fixture` (correlation ≥ threshold, so
  it should rise).
- Must change: `knockoff_cd_defaults_use_fast_path` (lib.rs:1845) asserts the 1e-4 default. Every Rust knockoff
  vote and replicate output (e.g. the 1.0.0 replicate rerun) also changes.
- New regression test, proposed: a 2-column near-duplicate design (x, x+1e-5·e) where both enter at the same
  grid λ, so W = 0 under both swaps.

## Performance
On 20 draws at tol 1e-7 the current `paired_xtilde` path takes 7833 CD iterations and 11.9 ms, against 505
iterations and 2.0 ms at 1e-4. That is about 6× per knockoff path at this size. Inferred: the entry-order change
alone costs little, at most 2p extra coordinate visits per active sweep for zeroed ever-active columns, as in
glmnet. It was not timed in Rust.

## Verification after implementation
Rebuild `paired_xtilde` (`crates/slide-cli/examples/paired_xtilde.rs`) and rerun
`../2026-10-13_s-zero-handling/check.py` (200 draws × 4 arms). Pass condition: per-draw sign(W) agreement with
`r_glmnet` for Z20 and Z21 is at least the Python-vs-R rate (1.0/0.985; 0.98 for Z20 in the floor arm), and the
Z20/Z21 selection rates are within 1/200 of R's. In the same run, check that all-factor sign agreement with R is
≥ 0.93, the `experiment.py` mirror figure. Then rerun `floor_sweep.py` with the rebuilt binary at tol 1e-7 in
place of the mirror. At every floor the Z20/Z21 agreement must be ≥ 0.975 and the Z20/Z21 selection rates must be
within 0.02 of R's. The mirror reached 0.978–1.0 and ≤ 0.02 (`sweep_rule_agreement.csv`).

## Reproduce (login node, about 6 min per arm with 4 processes)
```
PY=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev/bin/python
cd benchmark_results/2026-10-13_glmnet-tie-diagnosis
$PY validate.py 10                       # mirrors vs recorded W
$PY trace0.py 0 20 1e-7                  # instrumented step above
for a in rust_s r_s r_s_nearzero_to_0 rust_s_floor_1e-9; do $PY experiment.py $a 200 4 ../../.tmpfiles/2026-10-13_glmnet-tie/w_$a.csv; done
$PY summarize.py ../../.tmpfiles/2026-10-13_glmnet-tie .
$PY sorted_check.py 40
cd ../..   # s-floor sweep, about 2 min per floor
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev; B=benchmark_results/2026-10-13_glmnet-tie-diagnosis
PYTHONPATH=src:$B R_LIBS=$PWD/.tmpfiles/2026-10-06_replicates/R-library:$ENV/lib/R/library $PY $B/floor_sweep.py \
  $PWD/.tmpfiles/2026-10-13_glmnet-tie/sweep 200 4 $PWD/.tmpfiles/2026-10-13_s-only-d001/full/bin/paired_xtilde $ENV/bin/Rscript
PYTHONPATH=src $PY $B/sweep_summary.py .tmpfiles/2026-10-13_glmnet-tie/sweep $B
```

## Caveats
- R glmnet 5.0 runs compiled C++ (glmnetpp), which was not instrumented. Its use of entry order is **inferred**
  from the Fortran `elnet1` it ports (`ia(l)` loop, glmnet5.f90:929) and from the per-draw match.
- Why Z21 fails at tol 1e-4 even with entry order is **inferred, not traced**. At 1e-4 the pair gradients differ by
  2.8e-3 before entry (λ14, draw 0), against 7e-6 at 1e-7. The loose fit probably leaves a residual component along
  other near-duplicate pairs' difference directions.
