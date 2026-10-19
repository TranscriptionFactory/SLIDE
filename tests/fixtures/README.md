# Test Fixtures

Reference outputs from the canonical R LOVE package, R `glmnet`, and R `knockoff` for parity testing of the Rust implementation.

See [fixture generation](generate/README.md) for regeneration instructions.

Current fixture groups:
- `love/` — R LOVE outputs on the portable SSc snapshot
- `glmnet/` — R `glmnet` path and knockoff-stat snapshots
- `sdp/` — R `knockoff` solver diagonals
- `prescreen/` — R `Screen_X` outputs
- `slide_core/` — end-to-end SLIDE pipeline reference outputs on the bundled HIV cytokine example
