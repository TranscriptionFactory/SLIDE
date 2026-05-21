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

Output docs/assets/tui/tui-demo.gif

Hide
Type "clear"
Enter
Show

Type "./target/debug/slide tui --checkpoint .tmp/tui-smoke/tui-smoke-checkpoint.json --config .tmp/tui-smoke/tui-smoke.yaml --no-follow"
Enter
Wait+Screen /LOVE -> SLIDE Monitor/
Sleep 800 ms
Screenshot docs/assets/tui/overview.png

Tab
Wait+Screen /Diagnostics/
Sleep 700 ms
Screenshot docs/assets/tui/love-diagnostics.png

Type "3"
Wait+Screen /Loading Matrix/
Sleep 700 ms
Screenshot docs/assets/tui/love-loading-matrix.png

Tab
Wait+Screen /Factors/
Sleep 700 ms
Screenshot docs/assets/tui/factors.png

Tab
Wait+Screen /SLIDE/
Sleep 700 ms
Screenshot docs/assets/tui/slide-selection.png

Tab
Wait+Screen /Performance/
Sleep 700 ms
Screenshot docs/assets/tui/performance.png

Type "q"
