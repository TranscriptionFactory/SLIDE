# TUI visual outputs (2026-10-21)

Captured from the code finalized in `312a783` with the rebuilt `target/debug/slide`.
Command (this host):

```bash
export TMPDIR="$PWD/.tmpfiles"
export PATH="$PWD/.tmpfiles/2026-10-21_tui-capture:$PATH"
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
export SLIDE_TUI_WORKDIR="$PWD/.tmpfiles/2026-10-21_readability-smoke"
export SLIDE_TUI_SCREENSHOT_DIR="$PWD/docs/assets/tui/2026-10-21_readability"
scripts/with-rust-blas-env.sh scripts/capture-tui-screenshots.sh --skip-prepare
```

The PATH entry supplies cached ttyd; LD_PRELOAD selects the host C++ runtime
for Chromium. Other hosts need their own VHS dependencies. The output directory
must not exist before capture.
Fixture: `/vast/djishnu/Aaron/1_general_use/SLIDE/.tmpfiles/2026-10-21_readability-smoke`
Exports: `/vast/djishnu/Aaron/1_general_use/SLIDE/.tmpfiles/2026-10-21_readability-smoke/exported-results`
Binary: `./target/debug/slide`

The shared smoke workflow uses 40 HIV features, eight knockoff iterations,
one grid cell, binary response, and three-fold final CV. These are demonstration
outputs, not scientific conclusions. The binary-response fixture has
interactions disabled, so the Interaction Ranking view is empty. Screenshots
are static and do not demonstrate interactive inspection.

The tape captures the standard views at 160x40 plus Overview, LOVE loading,
factor, network, and Performance at 80x24 and 60x20. It records and checks the
actual pty sizes with `stty size` before each launch, without overriding the
emulator size. Separate VHS launches use sections of the same maintained tape.
LOVE includes all seven subviews.

Other captures include covariance and block heatmaps, signed feature text,
response dots, exported OLS controls, fold scores, prediction bins, and ROC/PR
output. “Permuted CV” shows randomized in-sample OLS controls, not held-out CV.

## Changes and limits

This gallery replaces the [restored-tab gallery](../archive/2026-10-21_restored-tabs/README.md).
It adds the two previously uncaptured LOVE sub-views and ten narrow captures.
Feature names now load in LOVE, duplicate names have stable feature IDs, and
Overview identifies the supplied config. The factor histogram statistics fit;
curves and network edges use higher-contrast half-block strokes. Narrow ROC/PR
views show pooled AUC, AP, and prevalence when axes would overlap.

All 31 PNGs were visually reviewed. Terminal dimensions were checked without
forcing a different PTY size. The five tabs and seven LOVE sub-views remain.
Fold-mean CV and pooled metrics remain distinct; exported controls are in-sample.

The smoke fixture does not visually cover populated interaction networks or
continuous residuals. Tests cover those rendering paths. Some sparse diagnostic
panels retain whitespace; narrow screens abbreviate secondary titles and network
names. Same-shaped replacement inputs cannot be verified by dimension checks
alone. No terminal-image protocols or new plotting dependencies were used.

## All captures

- [factor-1.png](factor-1.png)

- [factor-2.png](factor-2.png)

- [factor-3.png](factor-3.png)

- [factor-60x20.png](factor-60x20.png)

- [factor-80x24.png](factor-80x24.png)

- [factor-network-60x20.png](factor-network-60x20.png)

- [factor-network-80x24.png](factor-network-80x24.png)

- [factor-network.png](factor-network.png)

- [factor-text-plot.png](factor-text-plot.png)

- [love-block-correlation.png](love-block-correlation.png)

- [love-covariance.png](love-covariance.png)

- [love-diagnostics.png](love-diagnostics.png)

- [love-gamma.png](love-gamma.png)

- [love-group-correlation.png](love-group-correlation.png)

- [love-interaction-ranking.png](love-interaction-ranking.png)

- [love-loading-matrix-60x20.png](love-loading-matrix-60x20.png)

- [love-loading-matrix-80x24.png](love-loading-matrix-80x24.png)

- [love-loading-matrix.png](love-loading-matrix.png)

- [overview-60x20.png](overview-60x20.png)

- [overview-80x24.png](overview-80x24.png)

- [overview.png](overview.png)

- [performance-60x20.png](performance-60x20.png)

- [performance-80x24.png](performance-80x24.png)

- [performance-controls.png](performance-controls.png)

- [performance-folds.png](performance-folds.png)

- [performance-predictions.png](performance-predictions.png)

- [performance-roc-pr.png](performance-roc-pr.png)

- [performance.png](performance.png)

- [slide-interactions.png](slide-interactions.png)

- [slide-response.png](slide-response.png)

- [slide-selection.png](slide-selection.png)

- [tui-demo.gif](tui-demo.gif)
