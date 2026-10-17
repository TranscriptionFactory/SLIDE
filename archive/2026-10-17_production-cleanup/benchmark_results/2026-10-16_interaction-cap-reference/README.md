# Interaction-cap R reference

The Gaussian arithmetic correction exposed a pre-existing response-preparation
mismatch: Rust regressed through the origin; R and Python include an intercept.
On the cap fixture the residual discrepancy reached 0.4515905. After correcting
Rust, R, Rust, and Python agree on counts [3,4,2,1,1,1,1,1,1,1,0], final pairs
Z0.Z1 / Z0.Z2 / Z0.Z3, and specificity retries 0.1 → 0.2 → 0.3. The original
three-interaction assertion passes without relaxing it.

The fixture uses four rows and twelve columns:
z[i,j] = sin((12i+j) × 0.37) + 0.11j; y[i] = cos(0.7i).
Interactions are z[:,0] × z[:,1:]. `corrected_y.csv` uses centered single-variable
OLS; R independently recomputes residuals with `correctIt` / `lm(y ~ z)`.
The retained fixture is `tests/fixtures/slide_core/interaction_cap/`.

## Reproduction

```bash
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export TMPDIR="$PWD/.tmpfiles/2026-10-16_gaussian-arithmetic"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OUT="$PWD/benchmark_results/2026-10-16_interaction-cap-reference"
mkdir -p "$OUT"
"$ENV/bin/python" - <<'PYGEN'
import os, shutil, numpy as np
from pathlib import Path
src = Path('tests/fixtures/slide_core/interaction_cap')
out = Path(os.environ['OUT'])
for name in ['z.csv', 'y.csv']:
    shutil.copy2(src / name, out / name)
z = np.loadtxt(src / 'z.csv', delimiter=',')
y = np.loadtxt(src / 'y.csv')
zc, yc = z[:, 0] - z[:, 0].mean(), y - y.mean()
np.savetxt(out / 'interactions.csv', z[:, 0, None] * z[:, 1:], delimiter=',', fmt='%.17g')
np.savetxt(out / 'corrected_y.csv', yc - zc * (zc @ yc / (zc @ zc)), fmt='%.17g')
PYGEN
SLIDE_ATTR_Z_CSV="$OUT/interactions.csv" SLIDE_ATTR_Y_CSV="$OUT/corrected_y.csv" \
SLIDE_ATTR_OUTDIR="$OUT" SLIDE_ATTR_NDRAWS=4 SLIDE_ATTR_BASE_SEED=94 \
SLIDE_ATTR_FDR=0.9 SLIDE_ATTR_SPEC=0.1 SLIDE_ATTR_NLAMBDA=30 SLIDE_ATTR_METHOD=equi \
  cargo test --release --locked -p slide-knockoff --lib emit_knockoff_attribution -- --ignored --nocapture
"$ENV/bin/Rscript" tests/fixtures/generate/generate_slide_core_fixtures.R --interaction-cap "$OUT"
```

R uses the saved native draws and swaps, a 30-point grid, and the frequency-only
selection mode from the original test, not stock findOptIter. Its method-3 cap
controller is the method-4 selection path without upsilon regression. This does
not establish parity of independently generated RNG streams or full pipelines.

Verification: the Rust core test retains its three-interaction assertion and now
checks the exact R vote vector and pair names. All 8 Python interaction tests pass,
including the fixed-draw R-reference test. R residuals for a constant marginal
also confirm the intercept-only behavior covered by the Rust response test.
