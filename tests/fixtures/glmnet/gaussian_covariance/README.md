# Gaussian covariance entry reference

The test input CSVs are the first saved `saved_c8` draw from
`benchmark_results/2026-10-15_rust-covariance/fixed-draws/saved_c8/`, using the
production cached-half API, default 500-point Rust lambda grid, sample
standardization and tolerance. `expected_entry.csv` is the zero-based first
nonzero lambda index for augmented column 5 (Z159). It is 380; the archived
baseline enters at 381.

Regenerate the raw-input CSVs and the explicit R 5.0 covariance expected entry
after running the full replay from the repository root:

```bash
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export SCRATCH="$PWD/.tmpfiles/2026-10-15_rust-covariance-verify"
export TMPDIR="$SCRATCH" TMP="$SCRATCH" TEMP="$SCRATCH"
"$ENV/bin/python" - <<'PY'
from pathlib import Path
import numpy as np
src=Path('benchmark_results/2026-10-15_rust-covariance/fixed-draws/saved_c8')
out=Path('tests/fixtures/glmnet/gaussian_covariance')
n,m=24,18
for name,data in [('input_x.csv',np.fromfile(src/'z.raw','<f8').reshape(n,m)),
                  ('input_xtilde.csv',np.fromfile(src/'xt_fixed.raw','<f8').reshape(-1,n,m)[0]),
                  ('input_y.csv',np.fromfile(src/'y.raw','<f8')),
                  ('swap.csv',np.fromfile(src/'swap.raw',np.uint8))]:
    np.savetxt(out/name,data,delimiter=',',fmt='%.17g')
PY
"$ENV/bin/Rscript" -e 'library(glmnet); a <- commandArgs(TRUE); raw <- readBin(a[1], "double", n=24*36); x <- t(matrix(raw, nrow=36, ncol=24)); y <- readBin(a[2], "double", n=24); lambda <- readBin(a[3], "double", n=500); fit <- glmnet(x,y,lambda=lambda,standardize=FALSE,intercept=TRUE,type.gaussian="covariance",control=list(thresh=1e-7,maxit=100000)); b <- as.matrix(fit$beta); e <- apply(b,1,function(z) { q <- which(z != 0); if(length(q)) q[1]-1 else 0 }); writeBin(as.double(e),a[4]); cat("glmnet",as.character(packageVersion("glmnet")),"Z159 entry",e[6],"\n")' \
  "$PWD/benchmark_results/2026-10-15_rust-covariance/fixed-draws/saved_c8/r_x.raw" "$PWD/benchmark_results/2026-10-15_rust-covariance/fixed-draws/saved_c8/y.raw" \
  "$PWD/benchmark_results/2026-10-15_rust-covariance/fixed-draws/saved_c8/r_lambda.raw" "$SCRATCH/r_entries_covariance.raw"
"$ENV/bin/python" - <<'PY'
from pathlib import Path
import numpy as np
entries=np.fromfile('.tmpfiles/2026-10-15_rust-covariance-verify/r_entries_covariance.raw','<f8').astype(int)
Path('tests/fixtures/glmnet/gaussian_covariance/expected_entry.csv').write_text(f'{entries[5]}\n')
PY
```

The explicit R call confirms that the saved augmented width is below 500 and
uses the covariance implementation, rather than relying only on automatic
dispatch. The expected value is read from R output, never from candidate output.

The historical `fixed-draws/` inputs above are not bundled in this checkout.
The historical investigation report is in
[`archive/2026-10-17_production-cleanup/benchmark_results/2026-10-15_rust-covariance/README.md`](https://github.com/TranscriptionFactory/SLIDE/blob/0c030651bf4db5dd0d184e09f7274ab319d66bce/archive/2026-10-17_production-cleanup/benchmark_results/2026-10-15_rust-covariance/README.md).
