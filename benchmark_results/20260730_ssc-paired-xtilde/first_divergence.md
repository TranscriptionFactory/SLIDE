# First divergence

**Decision: B. Lasso-path/solver mismatch.**

| Stage | Result | Evidence |
|---|---|---|
| Raw A/Z/y and X-tilde | equal | All consumer SHA-256 values match; X-tilde reload max absolute difference is 0. |
| Centering/scaling | equal within floating arithmetic | All use sample SD (ddof=1); standardized summaries agree to roundoff. |
| Lambda grid / lasso path | DIFFERS | First grid difference: iteration 0, grid index 1; R=4.3741044063721892, Python=4.3741044063721901, Rust=4.3739711533280365. R/Python use k/nlambda; Rust uses k/(nlambda-1) and stops once all observed features have entered. |
| Lambda-entry convention | DIFFERS by representation | R/Python report lambda*n; Rust reports raw lambda. Multiplying Rust entries/W by n removes this global scale only; feature-specific grid/solver differences remain. |
| W / threshold / selections | first final divergence | Iteration 14, LF Z10 (index 10): W R=0.97199920444846644, Python=0.97199920444846655, Rust=0.03445192653503077 (Rust*n=0.82684623684073855); selected R/Python/Rust=False/False/True. |
| Vote aggregation / indexing | equal implementation | Zero-based indices map directly to canonical LF names; cumulative counts are the sums of per-iteration selections. |

## Z33/Z61 paired evidence

Across 20 byte-paired iterations:

| Backend | Z33 votes | Z44 votes | Z61 votes |
|---|---:|---:|---:|
| r | 6 | 10 | 7 |
| python | 6 | 10 | 7 |
| rust | 6 | 10 | 7 |

Z33 and Z61 have identical per-iteration selections and final vote counts across all three consumers in this paired run. Therefore their persistent native-R versus Rust preference is not produced by downstream voting when realized X-tilde and swap masks are fixed. It comes from knockoff generation/unpaired RNG streams. The distinct downstream path mismatch first changes a final selected set at iteration 14, but not the Z33/Z61 counts.

## Minimal proposed change (not implemented)

Match R/Python's Gaussian grid exponent `k / nlambda` instead of `k / (nlambda - 1)`, then rerun this fixture. Separately decide whether public Rust diagnostics should report `lambda*n`; that scale does not affect threshold selection. Production fix intentionally deferred for approval.
