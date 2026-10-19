# Test fixtures

Reference outputs from R LOVE, R `glmnet`, and R `knockoff` for Rust parity
tests. Individual fixture READMEs describe their inputs, expected values, and
comparison limits.

See [fixture generation](generate/README.md) for regeneration instructions.

## Fixture groups

- `love/` — R LOVE outputs on the portable SSc snapshot
- `glmnet/` — R `glmnet` path and knockoff-stat snapshots
- `sdp/` — R `knockoff` solver diagonals
- `prescreen/` — R `Screen_X` outputs
- `slide_core/` — end-to-end SLIDE pipeline reference outputs on the bundled HIV cytokine example
