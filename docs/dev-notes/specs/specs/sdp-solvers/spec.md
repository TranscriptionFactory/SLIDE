# SDP Solvers for Knockoff Generation

## 1. Equi-Correlated Solver (`create_solve_equi`)

**WHEN** `create_solve_equi(Sigma)` is called with a `p x p` correlation matrix,
**THEN** set `s_j = min(1, 2 * lambda_min(Sigma))` for all `j` and return `diag_s` as a length-`p` vector with that constant value.

**WHEN** `lambda_min(Sigma) >= 0.5`,
**THEN** all `s_j = 1.0`.

**WHEN** `lambda_min(Sigma) < 0.5`,
**THEN** all `s_j = 2 * lambda_min(Sigma)`, which is less than 1.

**WHEN** the equi-correlated solution is requested,
**THEN** always return a feasible result without iteration; no binary-search scaling is required.

---

## 2. SDP Solver (`create_solve_sdp`)

**WHEN** `create_solve_sdp(Sigma)` is called,
**THEN** solve the semidefinite program:
- Maximize `sum(s)`
- Subject to: `0 <= s_j <= 1` for all `j`
- Subject to: `2 * Sigma - diag(s)` is positive semidefinite

**WHEN** DSDP5 (C extension) is available in the environment,
**THEN** use DSDP5 as the primary solver.

**WHEN** DSDP5 is not available,
**THEN** fall back to CVXPY with the SCS backend.

**WHEN** the raw solver solution yields a `diag_s` such that `2 * Sigma - diag(s)` has any eigenvalue below `1e-10`,
**THEN** apply binary-search feasibility scaling: multiply `diag_s` by a scalar `gamma in (0, 1)` until all eigenvalues of `2 * Sigma - diag(s * gamma)` exceed `1e-10`, then set `diag_s = diag_s * gamma`.

**WHEN** binary-search scaling is applied,
**THEN** this is a correctness step to guarantee positive definiteness, not an optional optimization; it must always run when the eigenvalue floor is not met.

**WHEN** the solver reports infeasibility or fails to converge,
**THEN** raise a `RuntimeError` with the solver status and suggest falling back to equi-correlated.

### Correctness Invariants

**WHEN** `create_solve_sdp` returns `diag_s`,
**THEN** all of the following must hold:
1. `0 <= s_j <= 1` for all `j`
2. All eigenvalues of `2 * Sigma - diag(s)` are >= `1e-10`

---

## 3. ASDP Solver (`create_solve_asdp`)

**WHEN** `create_solve_asdp(Sigma)` is called,
**THEN** cluster the `p` variables using correlation-based grouping, solve an independent SDP for each cluster block, and assemble the full `diag_s` vector from per-block solutions.

**WHEN** clustering is performed,
**THEN** variables within the same cluster have higher within-cluster absolute correlation than between-cluster correlation; the specific algorithm (e.g., hierarchical, k-means on correlation distance) is an implementation detail.

**WHEN** per-block SDPs are solved,
**THEN** apply the same binary-search feasibility scaling as `create_solve_sdp` to each block's solution.

**WHEN** all block solutions are assembled,
**THEN** the returned `diag_s` must satisfy `0 <= s_j <= 1` for all `j` and `2 * Sigma - diag(s) >= 1e-10` (eigenvalue floor) globally.

**WHEN** `p` is large (e.g., `p > 500`),
**THEN** ASDP is expected to complete substantially faster than full SDP by exploiting block structure.

---

## 4. Solver Fallback Cascade

**WHEN** a solver is requested,
**THEN** attempt solvers in this order: DSDP5 (C extension) → CVXPY + SCS → raise `RuntimeError`.

**WHEN** DSDP5 raises an `ImportError` or `OSError`,
**THEN** log a warning and proceed to CVXPY + SCS without re-raising.

**WHEN** CVXPY + SCS also fails or is unavailable,
**THEN** raise `RuntimeError("No SDP solver available: DSDP5 and CVXPY+SCS both failed.")`.

**WHEN** targeting a Rust port,
**THEN** a native SDP solver (Clarabel or equivalent) must be used; the fallback cascade still applies in the same order of preference.

---

## 5. Minor Convergence Differences

**WHEN** comparing Python DSDP5 output to R's DSDP output on the same `Sigma`,
**THEN** minor numerical differences in `diag_s` values are acceptable, provided both solutions satisfy the correctness invariants above.

---

## 6. Rust Implementation Notes

### Clarabel CSC Constraint Matrix Construction

The knockoff SDP (and the Dantzig LP used in LOVE's non-pure row estimation) must be expressed in Clarabel's standard conic form. Clarabel requires the constraint matrix `A` in Compressed Sparse Column (CSC) format (`colptr`, `rowval`, `nzval` vectors).

For the Dantzig LP with `K` variables (split into positive/negative components, so `2K` decision variables and `4K` constraints):

```rust
// Build CSC constraint matrix column-by-column
let mut colptr = vec![0usize];
let mut rowval = Vec::new();
let mut nzval = Vec::new();
let mut nz_count = 0usize;

for col in 0..num_vars {
    for row in 0..num_constraints {
        let val = if row < k {
            // Block 1: [C_hat, -C_hat] (upper bound constraints)
            if col < k { c_hat[[row, col]] } else { -c_hat[[row, col - k]] }
        } else if row < 2 * k {
            // Block 2: [-C_hat, C_hat] (lower bound constraints)
            let r = row - k;
            if col < k { -c_hat[[r, col]] } else { c_hat[[r, col - k]] }
        } else {
            // Block 3: -I_{2K} (non-negativity bounds)
            let r = row - 2 * k;
            if r == col { -1.0 } else { 0.0 }
        };

        if val.abs() > 1e-12 {
            rowval.push(row);
            nzval.push(val);
            nz_count += 1;
        }
    }
    colptr.push(nz_count);
}

let a_csc = CscMatrix::new(num_constraints, num_vars, colptr, rowval, nzval);
```

For the knockoff SDP (`maximize sum(s)` subject to `2*Sigma - diag(s) >= 0`), the same CSC pattern applies but the constraint encodes the PSD condition via Clarabel's `PsdConeT`. The `s` variables map to diagonal entries of the slack matrix.
