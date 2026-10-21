# TUI visual outputs (2026-10-21)

Captured from `f4e35dc` plus the working-tree capture tape.
Commands (from the repository root, with VHS and ttyd on `PATH`):

```bash
module load ffmpeg # HPC only; load before Pixi
SLIDE_RS_BIN="$PWD/target/debug/slide" pixi run --locked -e rust scripts/capture-tui-screenshots.sh
# Recapture the same fixture after archiving the first screenshot directory:
SLIDE_RS_BIN="$PWD/target/debug/slide" pixi run --locked -e rust scripts/capture-tui-screenshots.sh --skip-prepare
```

Fixture: `/vast/djishnu/Aaron/1_general_use/SLIDE/.tmpfiles/2026-10-21_tui-smoke`
Exports: `/vast/djishnu/Aaron/1_general_use/SLIDE/.tmpfiles/2026-10-21_tui-smoke/exported-results`
Binary: `/vast/djishnu/Aaron/1_general_use/SLIDE/target/debug/slide`

The shared smoke workflow uses 40 HIV features, eight knockoff iterations,
one grid cell, binary response, and three-fold final CV. These are demonstration
outputs, not scientific conclusions. Interactions are disabled.

This capture expands the earlier eight-screen tape with covariance and block
heatmaps, signed feature text, a feature-correlation network, response dots,
exported OLS controls, fold scores, prediction bins, and ROC/PR output.
The “Permuted CV” screen shows randomized in-sample OLS controls, not held-out CV.

Capture size: 1600×960. Factor histogram titles clip on the right; the loading
matrix uses fallback feature IDs, and Overview reports `Config: unavailable`.
These screenshots preserve the current UI rather than hiding those limitations.

- [factor-1.png](factor-1.png)

- [factor-2.png](factor-2.png)

- [factor-3.png](factor-3.png)

- [factor-network.png](factor-network.png)

- [factor-text-plot.png](factor-text-plot.png)

- [love-block-correlation.png](love-block-correlation.png)

- [love-covariance.png](love-covariance.png)

- [love-diagnostics.png](love-diagnostics.png)

- [love-gamma.png](love-gamma.png)

- [love-loading-matrix.png](love-loading-matrix.png)

- [overview.png](overview.png)

- [performance-controls.png](performance-controls.png)

- [performance-folds.png](performance-folds.png)

- [performance-predictions.png](performance-predictions.png)

- [performance-roc-pr.png](performance-roc-pr.png)

- [performance.png](performance.png)

- [slide-response.png](slide-response.png)

- [slide-selection.png](slide-selection.png)

- [tui-demo.gif](tui-demo.gif)
