# Reproducible release screenshots for the Ratatui dashboard.
# Generate the fixture first with:
#   scripts/capture-tui-screenshots.sh --prepare-only
# Then render with:
#   vhs docs/tui-screenshots.tape

Require bash

Set Shell "bash"
Set Width 1600
Set Height 960
Set FontSize 16
Set Padding 18
Set TypingSpeed 5ms

Output "docs/assets/tui/tui-demo.gif"

Hide
Type "unset NO_COLOR; export TERM=xterm-256color COLORTERM=truecolor; clear"
Enter
Show

Type '"${SLIDE_RS_BIN:-./target/debug/slide}" view --checkpoint "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/tui-smoke-checkpoint.json" --config "${SLIDE_TUI_WORKDIR:-.tmpfiles/$(date +%F)_tui-smoke}/tui-smoke.yaml" --no-follow'
Enter
Wait+Screen /LOVE -> SLIDE Monitor/
Sleep 800 ms
Screenshot "docs/assets/tui/overview.png"

Tab
Wait+Screen /Diagnostics/
Sleep 700 ms
Screenshot "docs/assets/tui/love-diagnostics.png"

Type "2"
Wait+Screen /Factor Covariance/
Sleep 700 ms
Screenshot "docs/assets/tui/love-covariance.png"

Type "3"
Wait+Screen /Loading Matrix/
Sleep 700 ms
Screenshot "docs/assets/tui/love-loading-matrix.png"

Type "5"
Wait+Screen /Block Correlation/
Sleep 700 ms
Screenshot "docs/assets/tui/love-block-correlation.png"

Type "6"
Wait+Screen /Gamma/
Sleep 700 ms
Screenshot "docs/assets/tui/love-gamma.png"

Tab
Wait+Screen /Factors/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-1.png"

Type "j"
Wait+Screen /factor 2/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-2.png"

Type "j"
Wait+Screen /factor 3/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-3.png"

Type "kkv"
Wait+Screen /Spearman/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-text-plot.png"

Type "v"
Wait+Screen /Feature correlation network/
Sleep 700 ms
Screenshot "docs/assets/tui/factor-network.png"

Tab
Wait+Screen /SLIDE/
Sleep 700 ms
Screenshot "docs/assets/tui/slide-selection.png"

Type "vv"
Wait+Screen /Factor AUC/
Sleep 700 ms
Screenshot "docs/assets/tui/slide-response.png"

Tab
Wait+Screen /Performance/
Sleep 700 ms
Screenshot "docs/assets/tui/performance.png"

Type "v"
Wait+Screen /Exported OLS controls/
Sleep 700 ms
Screenshot "docs/assets/tui/performance-controls.png"

Type "v"
Wait+Screen /Fold Scores/
Sleep 700 ms
Screenshot "docs/assets/tui/performance-folds.png"

Type "v"
Wait+Screen /Predictions/
Sleep 700 ms
Screenshot "docs/assets/tui/performance-predictions.png"

Type "v"
Wait+Screen /ROC/
Sleep 700 ms
Screenshot "docs/assets/tui/performance-roc-pr.png"

Type "q"
