# LOVE A-Matrix Sparsity Mismatch: Investigation Results

## Root Cause

**Rust's Dantzig LP solver (Clarabel) is an interior-point method that produces dense solutions.
Python's LP solver (HiGHS via scipy) uses simplex, which produces sparse vertex solutions.**

Interior-point solvers find solutions in the *interior* of the feasible region, where all
variables tend to be slightly nonzero (~1e-10). Simplex solvers find solutions at *vertices*
of the feasible polytope, where many variables are exactly zero. After L1 normalization
(`normalize_l1`), those tiny interior-point values get scaled up relative to each other,
preserving all nonzeros regardless of lambda.

## Why Rust's A nonzeros are lambda-independent

Both implementations use the Dantzig method when `pure_homo=True` (benchmarks set this).
Both override HT→Dantzig for `pure_homo=True`:
- Python: `love.py:210-211` — `if pure_homo and est_non_pure_row == "HT": actual_est_method = "Dantzig"`
- Rust: `love.rs:128-131` — `if config.pure_homo && ... == HardThreshold { Dantzig }`

The Dantzig LP per non-pure row:
```
min sum(beta_pos + beta_neg)
s.t. C_hat @ (beta_pos - beta_neg) - y <= lbd
     -C_hat @ (beta_pos - beta_neg) + y <= lbd
     beta_pos >= 0, beta_neg >= 0
```

Where `lbd = config.lbd[0] * opt_delta * sigma_bar_sup * se_est_J[i]`

**With a simplex solver (Python/HiGHS):** increasing lambda relaxes constraints → more
variables become exactly zero at the vertex solution → A is sparser.

**With an interior-point solver (Rust/Clarabel):** increasing lambda relaxes constraints →
variables become *smaller* but never exactly zero → after L1 normalization, all variables
remain nonzero → A nonzero count is constant.

## Evidence

| Config (delta, lambda) | Python A nnz | Rust A nnz | Pattern match |
|------------------------|-------------|------------|---------------|
| d=0.01, l=0.1          | 72,534      | 73,821     | 99.1%         |
| d=0.01, l=1.0          | 66,434      | 73,821     | 94.7%         |
| d=0.1, l=0.1           | 19,969      | 27,861     | 88.9%         |
| d=0.1, l=1.0           | 2,269       | 27,861     | 63.8%         |

Rust constant per-delta: 73,821 (d=0.01), 27,861 (d=0.1) — lambda has no effect.
Python varies: 72,534→66,434 (d=0.01), 19,969→2,269 (d=0.1) — lambda drives sparsity.

## Code Flow (Both Implementations)

### 1. Lambda enters LOVE
- **Python**: `call_love(lbd=...)` → `LOVE(lbd=lbd)` → `love.py:326`: `lbd * optDelta * sigma_bar_sup`
- **Rust**: `run_slide_evaluation()` → `lib.rs:983`: `love_config.lbd = vec![lambda]` → `love.rs:456`: `config.lbd[0] * opt_delta * sigma_bar_sup`

### 2. Dantzig LP per non-pure row
- **Python**: `EstAJDant()` → `Dantzig(C_hat, y, lbd * se_est_J[i])` → `linprog(method='highs')` — **simplex** (`est_nonpure.py:242`)
- **Rust**: `est_aj_dant()` → `dantzig(c_hat, y, lbd * se[i])` → Clarabel `DefaultSolver` — **interior point** (`nonpure.rs:286-297`)

### 3. L1 normalization (identical)
- **Python**: `est_nonpure.py:187-188` — `if sum(|AJ[i]|) > 1: AJ[i] /= sum(|AJ[i]|)`
- **Rust**: `nonpure.rs:323-327` — `if l1_norm > 1.0 { vector /= l1_norm }`

### 4. Hard Threshold path (NOT taken in benchmarks)
Both implementations have an identical HT path (`love.py:363-364`, `love.rs:492-496`) that
computes `threshold = mu * optDelta * max(sum(|Omega|))` and zeros entries ≤ threshold.
This path IS lambda-sensitive in both, but it's bypassed when `pure_homo=True`.

## Fix Options (Prioritized)

### Option 1: Post-LP sparsification threshold (Recommended)
Add a small-value threshold after `extract_beta()` in Rust's `dantzig()` or `est_aj_dant()`.
Zero out entries with |value| < epsilon (e.g., 1e-8) before L1 normalization.

**Location**: `crates/slide-love/src/nonpure.rs`, after line 314 (`extract_beta`)
or before line 175 (`normalize_l1`).

**Pros**: Minimal change, preserves Clarabel's speed, matches simplex behavior.
**Cons**: Introduces a threshold parameter. Need to validate epsilon choice.

```rust
// In est_aj_dant, after extract_beta, before normalize_l1:
let mut beta = beta.to_nalgebra();
// Threshold near-zero entries from interior-point solver
for value in beta.iter_mut() {
    if value.abs() < 1e-8 {
        *value = 0.0;
    }
}
normalize_l1(&mut beta);
```

### Option 2: Switch LP solver to simplex-based
Replace Clarabel with a simplex solver (e.g., `highs` Rust bindings, `minilp`, or `good_lp`).

**Pros**: Exact parity with Python. No threshold parameter needed.
**Cons**: Larger dependency change. Clarabel is used elsewhere (glasso, hetero_cv).
Simplex may be slower for some problem sizes.

### Option 3: Apply threshold in `dantzig()` function
Same as Option 1 but inside `dantzig()` itself, applied to the raw LP solution before
`extract_beta()`.

**Location**: `nonpure.rs:197`, threshold `solution` values < 1e-8 to zero.

**Pros**: Catches near-zero values at the source.
**Cons**: Thresholding `beta_pos` and `beta_neg` separately vs. thresholding `beta_pos - beta_neg` gives subtly different results.

## Key File References

| File | Lines | What |
|------|-------|------|
| `crates/slide-love/src/love.rs` | 128-131 | HT→Dantzig override for pure_homo |
| `crates/slide-love/src/love.rs` | 453-458 | Lambda passed to Dantzig |
| `crates/slide-love/src/nonpure.rs` | 140-181 | `est_aj_dant` — Dantzig + L1 norm |
| `crates/slide-love/src/nonpure.rs` | 183-198 | `dantzig` — LP formulation |
| `crates/slide-love/src/nonpure.rs` | 277-310 | `solve_lp` — Clarabel interior point |
| `crates/slide-love/src/nonpure.rs` | 312-317 | `extract_beta` — beta_pos - beta_neg |
| `crates/slide-love/src/nonpure.rs` | 323-328 | `normalize_l1` — L1 normalization |
| `src/loveslide/love_python/love/est_nonpure.py` | 155-190 | Python `EstAJDant` |
| `src/loveslide/love_python/love/est_nonpure.py` | 193-251 | Python `Dantzig` (scipy/HiGHS) |
| `src/loveslide/love.py` | 207-213 | Python HT→Dantzig override |
