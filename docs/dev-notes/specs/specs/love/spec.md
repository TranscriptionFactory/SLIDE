# LOVE Algorithm Module Specification

**Module:** `loveslide.love_python.love`
**Reference implementation:** R (LOVE package)
**Model:** X = A @ Z + E
- X: n x p feature matrix (standardized, zero-mean columns)
- A: p x K loading matrix (sparse, overlapping clusters; each pure variable nonzero in exactly one column)
- Z: n x K latent factors
- E: n x p noise with diagonal covariance Gamma

---

## 1. Delta Selection

### 1.1 `CV_delta` — Homogeneous Half-Split CV

**Purpose:** Select the optimal delta from a grid using a single random 50/50 data split.

**WHEN** called with `X` (n x p), `deltaGrids`, `diagonal`, `se_est`, `merge`:

**THEN**
- Split `X` by randomly sampling `n // 2` rows into `X1`; remaining rows into `X2`.
- Compute `Sigma1 = (X1.T @ X1) / len(X1)` with diagonal zeroed; compute `Sigma2 = (X2.T @ X2) / len(X2)` with diagonal intact.
- Compute `FindRowMax(abs(Sigma1))` once, producing `Ms` and `arg_Ms`.
- For each `delta` in `deltaGrids`:
  - Call `CalFittedSigma(Sigma1, delta, Ms, arg_Ms, se_est, diagonal, merge)`.
  - If `fittedValue` is `None`, assign `loss = inf`.
  - Otherwise compute `loss = 2 * offSum(Sigma2[pureVec, pureVec] - fittedValue, se_est[pureVec]) / (|pureVec| * (|pureVec| - 1))`.
  - If denominator is zero, assign `loss = inf`.
- Return `deltaGrids[argmin(loss)]`.

**WHEN** `|pureVec| <= 1` for every delta in `deltaGrids`:

**THEN** all losses are `inf`; return `deltaGrids[0]` (argmin of all-inf array is index 0).

---

### 1.2 `CalFittedSigma` — Fitted Covariance for One Delta

**Purpose:** Compute the fitted value `A_I @ C @ A_I'` restricted to pure variables.

**WHEN** called with `Sigma` (diagonal zeroed), `delta`, `Ms`, `arg_Ms`, `se_est`, `diagonal`, `merge`:

**THEN**
- Call `FindPureNode(abs(Sigma), delta, Ms, arg_Ms, se_est, merge)` to obtain `estPureIndices` (list of lists) and `pureVec` (flat array).
- If `singleton(estPureIndices)` is `True` (any group of size 1, or no groups), return `{'pureVec': None, 'fitted': None}`.
- Call `FindSignPureNode(estPureIndices, Sigma)` then `RecoverAI` to build `AI` (p x K, entries in {-1, 0, 1}).
- Call `EstC(Sigma, AI, diagonal)` to get `C` (K x K).
- Compute `subAI = AI[pureVec, :]`; return `{'pureVec': list(pureVec), 'fitted': subAI @ C @ subAI.T}`.

**WHEN** `len(estPureIndices) == 1`:

**THEN** `fitted` is `None` even if the single group has more than one member.

---

### 1.3 `KfoldCV_delta` — Heterogeneous K-Fold CV

**Purpose:** Select delta for the heterogeneous pure-loading path using K-fold cross-validation.

**WHEN** called with a single-element `delta` array:

**THEN**
- Skip CV entirely.
- Return a result dict with `delta_min = delta[0]`, `delta_1se = delta[0]`, `foldid = None`, `cv_mean = None`, `cv_sd = None`, `est_pure = Est_Pure(score_mat, delta[0])`, `score`, and `moments`.

**WHEN** `delta` is `None` and `max_pure` is `None`:

**THEN** set `max_pure = 1.0` if `n > p`, else `max_pure = 0.8`.

**WHEN** `delta` is `None`:

**THEN**
- Compute `R = corrcoef(X)` and `score_mat, moments = Score_mat(R, q, exact)`.
- Build delta grid: `delta_max = quantile(row_mins_excluding_NaN, max_pure)`, `delta_min_val = nanmin(score_mat)`, `delta = linspace(delta_max, delta_min_val, ndelta)` (descending to ascending).
- Row mins are computed over all columns of each row (excluding `NaN`), for rows 0 through p-2 only.

**WHEN** performing K-fold CV (multi-element delta):

**THEN**
- Randomly permute `n` sample indices; partition into `nfolds` groups of sizes `ceil/floor(n / nfolds)` (larger groups first).
- For each fold `i`:
  - `trainX = X[not fold_i]`, `validX = X[fold_i]`.
  - Compute `R1 = corrcoef(trainX)`, `R2 = corrcoef(validX)`.
  - Call `Score_mat(R1, q, exact)` to get `score_mat_1` and `moments_1`.
  - For `j=0`: call `Est_Pure(score_mat_1, delta[0])` to initialize `I`, `I_part`.
  - For `j>0`: restrict score matrix to `pre_I` (previous pure indices), call `Est_Pure` on sub-matrix, map indices back to original space; if `pre_I` is empty, `I = []`, `I_part = []`.
  - If `I_part` is empty at any `j`, break the inner loop (no loss assigned for this fold beyond `j`).
  - Otherwise call `Est_BI_C(moments_1, R1, I_part, I)` to get `B_hat`, `C_hat`, `B_left_inv`.
  - Build reconstructed `tmp_R1`: replace `[I, I]` block with `B_I @ C_hat @ B_I.T`; if non-pure variables exist (`J != empty`), fill `[J, J]` block with `tmp.T @ pinv(C_hat) @ tmp` where `tmp = B_left_inv @ R1[I, J]`.
  - Assign `loss[i, j] = offSum(tmp_R1 - R2, 1) / p / (p - 1)`.
- After all folds: `cv_mean = nanmean(loss, axis=0)`, `cv_sd = nanstd(loss, axis=0, ddof=1)`.
- `ind_min = nanargmin(cv_mean)`, `delta_min = delta[ind_min]`.
- `delta_1se` range: indices where `cv_mean <= cv_mean[ind_min] + cv_sd[ind_min]`; return `(delta[min_index], delta[max_index])`.
- `est_pure = Est_Pure(score_mat_full, delta_min)` where `score_mat_full` is from the full dataset `R`.
- Return full dict with `foldid`, `delta_min`, `delta_1se`, `delta`, `cv_mean`, `cv_sd`, `est_pure`, `score`, `moments`.

---

### 1.4 `CV_lbd` — Lambda Selection for Precision Matrix

**Purpose:** Select lambda for `estOmega` using a 50/50 data split and Gaussian log-likelihood loss.

**WHEN** called with `X`, `lbdGrids`, `AI`, `pureVec`, `diagonal`:

**THEN**
- Split `X` randomly 50/50 into `X1`, `X2`.
- Compute `Sigma1 = (X1.T @ X1) / len(X1)`, `Sigma2 = (X2.T @ X2) / len(X2)`.
- Compute `C1 = EstC(Sigma1, AI, diagonal)`, `C2 = EstC(Sigma2, AI, diagonal)`.
- For each `lbd` in `lbdGrids`:
  - `Omega = estOmega(lbd, C1)`.
  - If `det(Omega) <= 0`, assign `loss = inf`.
  - Otherwise `loss = sum(Omega * C2) - log(det(Omega))` (Frobenius inner product minus log-determinant).
- Return `lbdGrids[argmin(loss)]`.

---

## 2. Pure Variable Detection

### 2.1 Homogeneous Path (`pure_homo=True`)

#### 2.1.1 `FindRowMax`

**WHEN** called with a p x p matrix `Sigma`:

**THEN** return `M[i] = max(Sigma[i, :])` and `arg_M[i] = argmax(Sigma[i, :])` for each row `i`. Ties are broken by `argmax` (first occurrence).

---

#### 2.1.2 `FindRowMaxInd`

**Purpose:** Find columns in row `i` whose values are within `2*delta` of the row maximum.

**WHEN** called with row index `i`, row maximum `M`, argmax `arg_M`, the row vector, `delta`, and `se_est`:

**THEN**
- Compute threshold `lbd[j] = delta * se_est[i] * se_est[arg_M] + delta * se_est[i] * se_est[j]` for each `j`.
- Return indices where `M <= lbd[j] + vector[j]` (i.e., where the row max is reachable from column `j` within the adaptive tolerance).

---

#### 2.1.3 `TestPure`

**Purpose:** Verify that every element of candidate set `Si` is consistent with belonging to the same cluster as row `i`.

**WHEN** called with a row of `Sigma`, row index `rowInd`, candidate indices `Si`, `Ms`, `arg_Ms`, `delta`, `se_est`:

**THEN**
- For each `j` in `Si`: compute `delta_j = (se_est[rowInd] + se_est[arg_Ms[j]]) * se_est[j] * delta`.
- If `|Sigma_row[j] - Ms[j]| > delta_j` for any `j`, return `False`.
- If all pass, return `True`.

---

#### 2.1.4 `FindPureNode`

**WHEN** called with `off_Sigma` (absolute off-diagonal covariance), `delta`, `Ms`, `arg_Ms`, `se_est`, `merge`:

**THEN**
- For each row `i`:
  - Compute `Si = FindRowMaxInd(i, Ms[i], arg_Ms[i], off_Sigma[i, :], delta, se_est)`.
  - If `Si` is nonempty and `TestPure` passes, form candidate group `Si + [i]`.
  - If `merge=True`, add via intersection (`Merge`); else add via union (`Merge_union`).
- `pureVec` = sorted unique union of all group members.
- Return `{'pureInd': G, 'pureVec': pureVec}`.

**WHEN** `merge=True` and a new group intersects an existing group:

**THEN** replace the existing group with the intersection (discarding non-shared members).

**WHEN** `merge=False` and a new group intersects one or more existing groups:

**THEN** replace all intersecting groups with their union.

---

#### 2.1.5 `FindSignPureNode`

**Purpose:** Assign positive/negative signs to pure variable sub-partitions within each factor group.

**WHEN** called with `pureList` (list of lists) and `Sigma`:

**THEN**
- For each group `purei` (sorted ascending):
  - Designate `purei[0]` as the reference; assign it to `pos`.
  - For each subsequent element `j`: if `Sigma[purei[0], j] < 0`, assign to `neg`; else assign to `pos`.
  - Return `{'pos': pos_indices, 'neg': neg_indices}`.

**WHEN** a group has exactly one member:

**THEN** return `{'pos': [member], 'neg': []}`.

---

#### 2.1.6 `RecoverAI`

**WHEN** called with `estGroupList` (list of pos/neg dicts) and `p`:

**THEN**
- Construct a p x K matrix (K = len(estGroupList)) of zeros.
- Set `A[pos_indices, k] = 1`, `A[neg_indices, k] = -1` for each group `k`.
- Return `A`.

---

#### 2.1.7 `EstAI`

**WHEN** called with `Sigma`, `optDelta`, `se_est`, `merge`:

**THEN**
- Zero the diagonal of `|Sigma|` before passing to `FindPureNode`.
- Chain: `FindRowMax` → `FindPureNode` → `FindSignPureNode` → `RecoverAI`.
- Return `{'AI': A, 'pureVec': flat_sorted_array, 'pureSignInd': sign_partition_list}`.

---

#### 2.1.8 `EstC` — Covariance of Z (Homogeneous)

**WHEN** called with `Sigma` (p x p), `AI` (p x K), `diagonal`:

**THEN**
- For each factor `k`: `group_k = nonzero_rows(AI[:, k])`.
  - `C[k, k] = sum(|Sigma[group_k, group_k]|) - trace(|Sigma[group_k, group_k]|)` divided by `|group_k| * (|group_k| - 1)`.
  - If `|group_k| <= 1`, `C[k, k] = 0`.
- If `diagonal=False`: for `i < j`, `C[i, j] = C[j, i] = mean over (i_row in group_i, j_col in group_j) of AI[i_row, i] * Sigma[i_row, j_col] * AI[j_col, j]`.
- Return K x K symmetric matrix.

**Verified correctness:** Max absolute deviation from R reference < 5.83e-16 on deterministic test data.

---

### 2.2 Heterogeneous Path (`pure_homo=False`)

#### 2.2.1 `Score_mat`

**Purpose:** Compute pairwise parallel-row scores used to detect pure variables.

**WHEN** called with correlation matrix `R` (p x p) and `q=2`:

**THEN**
- Compute `M = R.T @ R` (cross-product).
- For each pair `(i, j)` with `i < j`:
  - `V_ij = M[[i,j], [i,j]] - R[[i,j], [i,j]].T @ R[[i,j], [i,j]]`.
  - If `V_ij[0,0] == 0` or `V_ij[1,1] == 0`: `score_mat[i, j] = 0`.
  - Otherwise: `score_mat[i, j] = sqrt(|min(V_ij[0,0], V_ij[1,1]) / (p-2) * (1 - V_ij[0,1]^2 / V_ij[0,0] / V_ij[1,1])|)`.
- All other entries are `NaN` (lower triangle and diagonal).
- Return `{'score': score_mat, 'moments': M}`.

**WHEN** `q=inf` (infinity score):

**THEN** for each pair `(i, j)`: `score_mat[i, j] = min(LP_Score(R_ij, 0, exact), LP_Score(R_ij, 1, exact))` where `R_ij` is the (p-2) x 2 submatrix excluding rows `i` and `j`.

**Verified correctness:** Max absolute deviation from R reference approximately 2e-15.

---

#### 2.2.2 `LP_Score`

**WHEN** `exact=False`:

**THEN** grid search over `v in linspace(-1, 1, 100)`; return `min_v max_k |v * R_ij[k, ind] + R_ij[k, other_ind]|`.

**WHEN** `exact=True`:

**THEN** solve LP: minimize `t` subject to `|v| <= 1`, `t >= ||v * R_ij[:, ind] + R_ij[:, other_ind]||_inf`. Variables: `[t, v_pos, v_neg]` where `v = v_pos - v_neg`. If LP fails, fall back to `exact=False`.

---

#### 2.2.3 `Est_Pure`

**WHEN** called with `score_mat` (upper triangular, NaN elsewhere) and `delta`:

**THEN**
- Find all pairs `(i, j)` where `score_mat[i, j] <= delta` and `i != j` — these are graph edges.
- Build undirected graph on `p` nodes; find all connected components.
- Keep only components with `>= 2` members.
- `I_part` = list of those components; `I` = sorted union of all members; `K` = len(I_part).
- Return `{'K': K, 'I': I, 'I_part': I_part}`.

---

#### 2.2.4 `Est_BI_C`

**Purpose:** Estimate loading magnitudes `B` and correlation matrix of Z for the heterogeneous path.

**KNOWN BUG:** Python diverges from R on test data (pureVec [0,1,2,3] vs R [1,2,3,4,5]). Rust port must match R reference.

**WHEN** called with `M` (R.T @ R), `R`, `I_part`, `I_set`:

**THEN**
- For each group `k` and consecutive pair `(I_k[ell], I_k[(ell+1) % n_k])` (cyclic):
  - Compute `cross_Vij = M[[i,j], [i,j]] - R[[i,j], [i,j]].T @ R[[i,j], [i,j]]`.
  - `B_square[i, k] = |R[i, j]| * sqrt(cross_Vij[0,0] / cross_Vij[1,1])` if `cross_Vij[1,1] != 0`, else 0.
  - `Gamma[i] = R[i, i] - B_square[i, k]` (1 - loading variance).
  - Signs: first element `ell=0` has sign +1; subsequent elements take `sign(R[i, I_k[0]])`.
- `B = sqrt(B_square) * signs` (element-wise by row).
- `BI = B[I_set, :]` (sign-adjusted).
- `B_left_inv = solve(BI.T @ BI, BI.T)` (left inverse; use `pinv` if singular).
- `C_hat = B_left_inv @ (R[I_set, I_set] - diag(Gamma[I_set])) @ B_left_inv.T`; then `fill_diagonal(C_hat, 1.0)`.
- Return `{'B': B, 'C': C_hat, 'B_left_inv': B_left_inv, 'Gamma': Gamma}`.

---

#### 2.2.5 `Re_Est_Pure`

**Purpose:** Post-selection step that may reduce K if data-splitting suggests a smaller model.

**WHEN** called with `X`, `Sigma`, `M`, `I_part`, `Gamma`:

**THEN**
- For each group in `I_part`, select the representative `L[k]` = member with the largest squared L2 norm of `M[i, -i]` (all columns except self).
- Call `Est_K(X, L_hat, Gamma[L_hat])` to estimate K by data-splitting eigendecomposition.
- If `K_est < len(I_part)` and `K_est >= 1`, call `Post_Est_Pure` to select `K_est` groups from `I_part`; otherwise return `I_part` unchanged.

---

#### 2.2.6 `Est_K`

**WHEN** called with `X`, `L_hat` (representative indices), `Gamma_LL`:

**THEN**
- Randomly split `X` 50/50 into `X1`, `X2`.
- Compute `R1 = corrcoef(X1)`, `R2 = corrcoef(X2)`.
- `M1 = R1[L_hat, L_hat] - diag(Gamma_LL)`, `M2 = R2[L_hat, L_hat] - diag(Gamma_LL)`.
- Eigendecompose `M1`; sort eigenvalues descending.
- For each `k = 1, ..., len(L_hat)`: rank-k reconstruction error `= ||U_k D_k U_k.T - M2||_F^2`.
- Return `argmin + 1` (1-indexed count).

---

#### 2.2.7 `Post_Est_Pure`

**WHEN** called with `Sigma`, `Gamma_LL`, `L_hat`, `I_part`, `K_tilde`:

**THEN**
- Compute `Theta_LL = Sigma[L_hat, L_hat] - diag(Gamma_LL * diag(Sigma)[L_hat])`.
- Greedily select `K_tilde` groups using maximum diagonal of Schur complement:
  1. Select `L_tilde[0]` = argmax of `diag(diag(Theta_LL))` (diagonal-only matrix).
  2. For each subsequent step, compute Schur complement of remaining indices relative to already-selected; add index with maximum diagonal.
- Return `[I_part[i] for i in L_tilde]`.

---

## 3. Non-Pure Row Estimation

### 3.1 `EstY` — Factor-to-NonPure Correlation

**WHEN** called with `Sigma` (p x p), `AI` (p x K), `pureVec`:

**THEN**
- For each pure variable `i`: multiply its row in `Sigma` by `sign(AI[i, nonzero_column])` to form `signed_sigma`.
- For each factor `k`: `sigma_TJ[k, :] = mean(signed_sigma[group_k, nonPureVec], axis=0)`.
- Return K x |J| matrix (J = non-pure indices).

---

### 3.2 `EstAJInv` — Soft-Threshold Non-Pure (ST method)

**WHEN** called with `Omega` (K x K), `Y` (K x |J|), `lbd`:

**THEN**
- For each non-pure variable `i`: solve LP soft-threshold problem `LP(Omega @ Y[:, i], lbd)`.
- Normalize row `i` if `||AJ[i, :]||_1 > 1`.
- Return |J| x K matrix.

---

### 3.3 `LP` — Soft-Threshold via LP

**WHEN** called with vector `y` (length K) and scalar `lbd`:

**THEN**
- Solve: minimize `sum(beta_pos + beta_neg)` subject to `beta_pos - beta_neg <= lbd + y` and `-beta_pos + beta_neg <= lbd - y`, all variables >= 0.
- Variables layout: `[beta_1_pos, ..., beta_K_pos, beta_1_neg, ..., beta_K_neg]`.
- Return `beta = beta_pos - beta_neg`.

**WHEN** LP solver fails (`result.success = False`):

**THEN** return soft-thresholded `y`: `sign(y) * max(|y| - lbd, 0)` (fallback, does not return `None`).

---

### 3.4 `EstAJDant` — Dantzig Non-Pure Estimation

**WHEN** called with `C_hat` (K x K), `Y` (K x |J|), `lbd`, `se_est_J` (|J|-vector):

**THEN**
- For each non-pure variable `i`: call `Dantzig(C_hat, Y[:, i], lbd * se_est_J[i])`.
- If `Dantzig` returns `None` for any row, return `None` immediately (propagate failure).
- Otherwise normalize each row `i` if `||AJ[i, :]||_1 > 1`.
- Return |J| x K matrix or `None`.

---

### 3.5 `Dantzig` — Dantzig Selector LP

**WHEN** called with `C_hat` (K x K), `y` (length K), `lbd`:

**THEN**
- Solve: minimize `sum(beta_pos + beta_neg)` subject to `C_hat @ (beta_pos - beta_neg) - y <= lbd` and `-C_hat @ (beta_pos - beta_neg) + y <= lbd`, all variables >= 0.
- Variables layout: `[beta_1_pos, ..., beta_K_pos, beta_1_neg, ..., beta_K_neg]`.
- Return `beta = beta_pos - beta_neg`.

**WHEN** LP solver fails:

**THEN** return `None` (matches R's `dantzig()` returning `NULL`; no fallback).

---

## 4. Precision Matrix Estimation

### 4.1 `estOmega` — Graphical Lasso via LP

**WHEN** called with `lbd` and `C` (K x K):

**THEN**
- For each column `i`: call `solve_row(i, C, lbd)` to estimate column `i` of Omega.
- Return K x K matrix assembled column by column.

**Verified correctness:** Max absolute deviation from R reference < 3.53e-13.

---

### 4.2 `solve_row`

**WHEN** called with `col_ind`, `C_hat` (K x K), `lbd`:

**THEN**
- Variables: `[t, omega_1_pos, omega_1_neg, ..., omega_K_pos, omega_K_neg]` (total 2K+1).
- Objective: minimize `t`.
- Constraints:
  1. `t >= 0`.
  2. `sum_j (omega_j_pos + omega_j_neg) <= t` (L1 norm constraint).
  3. For each row `k`: `(C_hat @ omega)_k <= e_col[k]` and `-(C_hat @ omega)_k <= -e_col[k]`, where `e_col` is the unit vector for column `col_ind` and the `lbd` parameter multiplies the `t` coefficient in these rows (`-lbd` for `t`).

**WHEN** LP solver fails on the first attempt:

**THEN** increment `lbd` by 0.01 and retry up to 10 total attempts. Emit a warning each retry.

**WHEN** all 10 attempts fail:

**THEN** emit a warning and return a zero vector of length K.

---

## 5. Main Entry Point

### 5.1 `LOVE`

**WHEN** called with any `X`:

**THEN** center columns: `X = X - mean(X, axis=0)` (no-op on already-standardized data, but always applied).

---

**WHEN** `pure_homo=True` and scalar `delta`:

**THEN** `deltaGrids = delta * sqrt(log(max(p, n)) / n)` (scalar or array broadcast).

**WHEN** `pure_homo=True` and `delta=None`:

**THEN** `deltaGrids = [0.5 * sqrt(log(max(p, n)) / n)]` (single value, skip CV).

**WHEN** `pure_homo=True` and `len(deltaGrids) > 1`:

**THEN** run `CV_delta` `rep_CV=50` times; `optDelta = median(selections)`.

**WHEN** `thresh_fdr` is not `None` (homogeneous path):

**THEN**
- Compute `R = corrcoef(X)`.
- Convert each correlation to t-statistic: `t = r * sqrt(n-2) / sqrt(1 - r^2)` (clip `r` to [-0.9999, 0.9999]).
- Compute two-tailed p-values from t-distribution with `df = n-2`.
- Apply Benjamini-Hochberg FDR correction at `thresh_fdr` level to flattened p-values.
- Zero correlations with adjusted p-value > `thresh_fdr`.
- Convert filtered correlation matrix back to covariance: `Sigma_ij = R_filtered_ij * std_i * std_j`.

**WHEN** any pure variable group contains exactly one member after `EstAI` (homogeneous path):

**THEN** log a warning, switch `merge=False` (union mode), re-run `CV_delta` (if grid > 1), and re-call `EstAI`.

**WHEN** `EstAI` returns empty `I_hat`:

**THEN** raise `RuntimeError("No pure variables found")`.

---

**WHEN** `pure_homo=False` (heterogeneous path):

**THEN**
- Compute `R_hat = corrcoef(X)`, `Sigma = cov(X)`.
- If `thresh_fdr` is not `None`: apply BH FDR correction to `R_hat` (same procedure as homogeneous path); convert filtered R back to covariance Sigma.
- Call `KfoldCV_delta(X, delta, ndelta, q, exact, nfolds, max_pure)`.
- If `len(est_I) >= 2`: call `Est_BI_C` then `Re_Est_Pure`; update `est_I` and `est_I_set`.
- If `len(est_I) == 0`: raise `RuntimeError("No pure variables found")`.
- Re-call `Est_BI_C` on (potentially updated) `est_I`.
- Compute `A_hat`: `B_hat = sqrt(diag(Sigma)) * B_hat`; `D_B = max(|B_hat|, axis=0)`; `A_hat = B_hat / D_B`.
- Compute `C_hat`: `C_hat = D_B[:, newaxis] * R_Z` (column-wise recycled multiply, not outer product).
- If `diagonal=True`: `C_hat = diag(diag(C_hat))`.
- `Gamma_hat = BI_C_res['Gamma'] * diag(Sigma)`.
- `I_hat_part = FindSignPureNode(original_I_part_from_CV, Sigma)` (use partition before `Re_Est_Pure`).

---

**WHEN** any pure variable Gamma < 0 after the pure-variable stage (both paths):

**THEN** clamp `Gamma_hat[Gamma_hat < 0] = 1e-2` (stage 1 clamp, before non-pure estimation).

---

**WHEN** all p variables are pure (`len(I_hat) == p`):

**THEN** skip non-pure estimation; `Omega = None`.

---

**WHEN** `est_non_pure_row="HT"` (default, non-pure variables exist):

**THEN**
- Compute `Omega` via `estOmega(optLbd, C_hat)` (or `diag(1 / diag(C_hat))` if `diagonal=True`).
- `Y = EstY(Sigma, A_hat, I_hat)`.
- `threshold = mu * optDelta * ||Omega||_{inf}` where `||Omega||_{inf} = max row-wise L1 norm`.
- `AJ = threshA((Omega @ Y).T, threshold)`.

**WHEN** `est_non_pure_row="ST"`:

**THEN** `AJ = EstAJInv(Omega, Y, threshold)` (same threshold as HT).

**WHEN** `est_non_pure_row="Dantzig"`:

**THEN**
- Compute `AI_abs = |A_hat[I_hat, :]|`.
- `cross_AI_inv = solve(AI_abs.T @ AI_abs, AI_abs.T)`.
- `sigma_bar_sup = max(cross_AI_inv @ se_est[I_hat])`.
- `AJ = EstAJDant(C_hat, Y, lbd * optDelta * sigma_bar_sup, sigma_bar_sup + se_est[J])`.
- If `AJ is None`: return `None` (propagate LP failure, matches R returning `NULL`).
- `Omega = None` for Dantzig path.

**WHEN** `diagonal=False` and `len(lbdGrids) > 1` (HT or ST path):

**THEN** run `CV_lbd` `rep_CV=50` times; `optLbd = median(selections)`.

---

**WHEN** non-pure rows have been assigned and `J` is nonempty:

**THEN** re-estimate Gamma for J:
- `Gamma_hat[J] = diag(Sigma[J, J]) - diag(A_J @ C_hat @ A_J.T)`.
- Clamp `Gamma_hat[Gamma_hat < 0] = 1e2` (stage 2 clamp, large value for non-pure).

---

**Return value** (success path):

```
{
    'K':        number of estimated clusters,
    'pureVec':  array of pure variable indices,
    'pureInd':  list of {'pos': ..., 'neg': ...} dicts for pure groups,
    'group':    list of {'pos': ..., 'neg': ...} dicts for all K clusters,
    'A':        p x K loading matrix,
    'C':        K x K covariance of Z,
    'Omega':    K x K precision of Z (None for Dantzig or all-pure),
    'Gamma':    p-vector of noise variances,
    'optDelta': selected delta value
}
```

---

## 6. Pre-Screening

### 6.1 `Screen_X`

**Purpose:** Remove pure noise features (those uncorrelated with all others) before running LOVE.

**KNOWN BUG:** Python `thresh_min = 0.077` vs R `thresh_min = 0.091` on identical test data. Rust port must match R reference.

**WHEN** called with a single-element `thresh_grid`:

**THEN** return `indices where row_scale < thresh_grid[0]` as a flat array (no CV, no dict).

**WHEN** `thresh_grid=None`:

**THEN**
- Compute `R = corrcoef(X)`; zero diagonal.
- `row_scale[i] = sum(R[i, :]^2)` (sum of squared off-diagonal correlations for row `i`).
- `thresh_range = [quantile(row_scale, 0), quantile(row_scale, max_prop)]`.
- `thresh_grid = linspace(thresh_range[0], thresh_range[1], nthresh)`.

**WHEN** performing K-fold CV (multi-element thresh_grid):

**THEN**
- Permute and partition `n` samples into `nfolds` groups.
- For each fold `i`: train/valid split.
  - Compute `R_train`, `R_valid` (correlation matrices, diagonal zeroed).
  - For each threshold `thresh`: zero rows/columns of `R_train` where `row_scale < thresh`; `loss[i, j] = mean((R_valid - pred_R)^2)`.
- `cv_mean = mean(loss, axis=0)`, `cv_sd = std(loss, axis=0, ddof=1)`.
- `thresh_min = thresh_grid[argmin(cv_mean)]`.
- `thresh_1se = thresh_grid[max(indices where cv_mean <= cv_mean[argmin] + cv_sd[argmin])]` (largest threshold within 1 SE).
- `noise_ind = indices where row_scale < thresh_min`.
- Return dict: `{'foldid', 'thresh_min', 'thresh_1se', 'thresh_grid', 'cv_mean', 'cv_sd', 'noise_ind'}`.

**Note:** `row_scale` is always computed on the full dataset `X`, not per-fold, for the noise detection criterion.

---

## 7. Z-Matrix Computation

**Purpose:** Recover latent factor scores from estimated parameters.

**WHEN** given `X` (n x p), `A` (p x K), `Gamma_inv = diag(1 / Gamma)`, `C` (K x K):

**THEN**
- `G_hat = A.T @ Gamma_inv @ A + C_inv` where `C_inv = pinv(C)`.
- `Z_hat = X @ Gamma_inv @ A @ pinv(G_hat)`.
- Return n x K matrix.

**Verified correctness:** Results are byte-identical across Python backends (numpy, scipy) and to R reference.

---

## 8. Correctness Criteria (from SSc Divergence Analysis)

These are the acceptance thresholds for the Rust port. A pass requires all criteria to hold simultaneously on the standard test fixture.

| Criterion | Threshold |
|-----------|-----------|
| `EstC` max absolute diff from R | < 1e-14 |
| `estOmega` max absolute diff from R | < 1e-12 |
| Score matrix max absolute diff from R | < 1e-14 |
| Z-matrix: byte-identical across backends | required |
| K (number of factors) must match R for same delta | exact match |
| Mean `|corr(Z_python, Z_R)|` at delta=0.01 | >= 0.998 |
| Min `|corr(Z_python, Z_R)|` at delta=0.01 | >= 0.987 |
| `pure_homo=False` pureVec | must match R exactly (currently BROKEN) |
| `Screen_X` `thresh_min` | must match R exactly (currently BROKEN) |

**Known divergences (must be resolved in Rust port):**
- `pure_homo=False`: Python `pureVec = [0,1,2,3]`, R `pureVec = [1,2,3,4,5]` on identical test data.
- `Screen_X`: Python `thresh_min = 0.077`, R `thresh_min = 0.091` on identical test data.

---

## 9. Parameter Reference

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `delta` | `None` | array or `None` | Leading constant(s) for thresholding grid; actual grid scaled by `sqrt(log(max(p,n))/n)` in homo path |
| `lbd` | `0.5` | float | Leading constant for non-pure row estimation threshold |
| `mu` | `0.5` | float | Leading constant for hard/soft-threshold of loading matrix |
| `thresh_fdr` | `None` | float or `None` | BH FDR threshold applied to correlation matrix before pure variable detection |
| `pure_homo` | `False` | bool | `True` for homogeneous loading magnitudes; `False` for heterogeneous |
| `diagonal` | `False` | bool | Force covariance of Z to be diagonal |
| `merge` | `False` | bool | `True` = intersection merging of pure groups; `False` = union |
| `rep_CV` | `50` | int | Number of CV repetitions for delta/lambda selection |
| `ndelta` | `50` | int | Length of auto-generated delta grid |
| `q` | `2` | int | Score type: `2` (L2-based) or `inf` (LP-based) |
| `exact` | `False` | bool | Use exact LP for inf-score; approximate grid-search otherwise |
| `max_pure` | `None` | float or `None` | Max proportion of pure variables for delta grid |
| `nfolds` | `10` | int | Number of folds for K-fold CV |
| `est_non_pure_row` | `"HT"` | str | Method for non-pure rows: `"HT"`, `"ST"`, or `"Dantzig"` |
