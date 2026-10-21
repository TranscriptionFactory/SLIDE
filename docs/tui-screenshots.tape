# Reproducible release screenshots for the Ratatui dashboard.
# Generate the fixture first with:
#   scripts/capture-tui-screenshots.sh --prepare-only
# Then render with:
#   scripts/capture-tui-screenshots.sh --skip-prepare

Require bash

Set Shell "bash"
# Pixel dimensions are calibrated with ttyd; the wrapper verifies stty size.
Set Width 1670
Set Height 824
Set FontSize 16
Set Padding 18
Set TypingSpeed 5ms

Output "docs/assets/tui/tui-demo.gif"

Hide
Type "unset NO_COLOR; export TERM=xterm-256color COLORTERM=truecolor; clear"
Enter
Show

# Capture 160x40
Type ' stty size > "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/terminal-160x40.txt"; "${SLIDE_RS_BIN:-./target/debug/slide}" view --checkpoint "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/tui-smoke-checkpoint.json" --config "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/tui-smoke.yaml" --no-follow'
Enter
Wait+Screen /LOVE -> SLIDE Monitor/
Sleep 800 ms
Screenshot "docs/assets/tui/overview.png"
Sleep 300ms

Tab
Wait+Screen /Diagnostics/
Sleep 700 ms
Screenshot "docs/assets/tui/love-diagnostics.png"
Sleep 300ms

Type "2"
Wait+Screen /Factor Covariance/
Sleep 700 ms
Screenshot "docs/assets/tui/love-covariance.png"
Sleep 300ms

Type "3"
Wait+Screen /Loading Matrix/
Sleep 700 ms
Screenshot "docs/assets/tui/love-loading-matrix.png"
Sleep 300ms

Type "4"
Wait+Screen /Group Correlation/
Sleep 700 ms
Screenshot "docs/assets/tui/love-group-correlation.png"
Sleep 300ms

Type "5"
Wait+Screen /Block Correlation/
Sleep 700 ms
Screenshot "docs/assets/tui/love-block-correlation.png"
Sleep 300ms

Type "6"
Wait+Screen /Gamma/
Sleep 700 ms
Screenshot "docs/assets/tui/love-gamma.png"
Sleep 300ms

Type "7"
Wait+Screen /Interaction Ranking/
Sleep 700 ms
Screenshot "docs/assets/tui/love-interaction-ranking.png"
Sleep 300ms

Tab
Wait+Screen /Factors/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-1.png"
Sleep 300ms

Type "j"
Wait+Screen /factor 2/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-2.png"
Sleep 300ms

Type "j"
Wait+Screen /factor 3/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-3.png"
Sleep 300ms

Type "kkv"
Wait+Screen /Spearman/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-text-plot.png"
Sleep 300ms

Type "v"
Wait+Screen /Feature correlation network/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-network.png"
Sleep 300ms

Tab
Wait+Screen /SLIDE/
Sleep 700 ms
Screenshot "docs/assets/tui/slide-selection.png"
Sleep 300ms

Type "vv"
Wait+Screen /Factor AUC/
Sleep 700 ms
Screenshot "docs/assets/tui/slide-response.png"
Sleep 300ms

Type "v"
Wait+Screen /interaction/
Sleep 700 ms
Screenshot "docs/assets/tui/slide-interactions.png"
Sleep 300ms

Tab
Wait+Screen /Performance/
Sleep 700 ms
Screenshot "docs/assets/tui/performance.png"
Sleep 300ms

Type "v"
Wait+Screen /Exported OLS controls/
Sleep 700 ms
Screenshot "docs/assets/tui/performance-controls.png"
Sleep 300ms

Type "v"
Wait+Screen /Fold Scores/
Sleep 700 ms
Screenshot "docs/assets/tui/performance-folds.png"
Sleep 300ms

Type "v"
Wait+Screen /Predictions/
Sleep 700 ms
Screenshot "docs/assets/tui/performance-predictions.png"
Sleep 300ms

Type "v"
Wait+Screen /ROC/
Sleep 700 ms
Screenshot "docs/assets/tui/performance-roc-pr.png"
Sleep 300ms

Type "q"
Sleep 300ms

# Capture 80x24
Hide
Type 'stty size > "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/terminal-80x24.txt"; "${SLIDE_RS_BIN:-./target/debug/slide}" view --checkpoint "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/tui-smoke-checkpoint.json" --config "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/tui-smoke.yaml" --no-follow'
Enter
Sleep 800 ms
Show
Wait+Screen /LOVE -> SLIDE Monitor/
Sleep 500 ms
Screenshot "docs/assets/tui/overview-80x24.png"
Sleep 300ms
Tab
Type "3"
Wait+Screen /Loading Matrix/
Sleep 700 ms
Screenshot "docs/assets/tui/love-loading-matrix-80x24.png"
Sleep 300ms
Tab
Sleep 500 ms
Screenshot "docs/assets/tui/factor-80x24.png"
Sleep 300ms
Type "vv"
Wait+Screen /Feature correlation network/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-network-80x24.png"
Sleep 300ms
Tab
Tab
Sleep 500 ms
Screenshot "docs/assets/tui/performance-80x24.png"
Sleep 300ms

Type "q"
Sleep 300ms

# Capture 60x20
Hide
Type 'stty size > "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/terminal-60x20.txt"; "${SLIDE_RS_BIN:-./target/debug/slide}" view --checkpoint "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/tui-smoke-checkpoint.json" --config "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/tui-smoke.yaml" --no-follow'
Enter
Sleep 800 ms
Show
Wait+Screen /LOVE -> SLIDE Monitor/
Sleep 500 ms
Screenshot "docs/assets/tui/overview-60x20.png"
Sleep 300ms
Tab
Type "3"
Wait+Screen /Loading Matrix/
Sleep 700 ms
Screenshot "docs/assets/tui/love-loading-matrix-60x20.png"
Sleep 300ms
Tab
Sleep 500 ms
Screenshot "docs/assets/tui/factor-60x20.png"
Sleep 300ms
Type "vv"
Wait+Screen /Feature correlation network/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-network-60x20.png"
Sleep 300ms
Tab
Tab
Sleep 500 ms
Screenshot "docs/assets/tui/performance-60x20.png"
Sleep 300ms

Type "q"
