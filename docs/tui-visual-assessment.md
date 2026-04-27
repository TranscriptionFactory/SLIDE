# TUI Visual Assessment

This document assesses the current SLIDE terminal UI implementation in
`crates/slide-cli/src/tui.rs` and separates the visuals worth keeping from the
ones that should be simplified, hidden, or removed.

## Keep And Refine

### Four-tab structure

Keep the current top-level tabs:

- `Overview`
- `LOVE`
- `Factors`
- `SLIDE`

This is a useful mental model: operational progress, LOVE factor creation,
per-factor inspection, and SLIDE knockoff selection are distinct tasks. The
structure is implemented around `App::render` in `crates/slide-cli/src/tui.rs`.

Refinement: make the footer more context-aware. It currently always shows
factor navigation even outside the Factors tab. Showing only relevant keys per
tab would reduce visual noise.

### Overview tab

Keep the Overview tab as the primary monitoring surface. The current combination of
summary, completed evaluation table, score grid, and config identity is the
strongest part of the TUI.

Useful elements:

- Run summary: progress, current evaluation, best score, selected hyperparams.
- Completed evaluations table: direct comparison across grid points.
- Score grid: compact view of grid coverage and best regions.
- Phase timeline: useful live status indicator.

Refinement: reduce the always-visible config panel. It is valuable for
provenance, but it currently consumes a large portion of the Overview tab. Prefer a
collapsed identity block or a dedicated config/help view.

### Factors tab

Keep the Factors tab. It answers the main scientific inspection question:
"What explains this selected factor?"

Useful elements:

- Factor overview.
- Top features by loading.
- Top features by response association.
- Factor score distribution sparkline.

Refinement: the four equal columns are rigid. The feature tables usually need
more horizontal space than the overview and histogram. A better layout would
give the two feature tables most of the width and keep overview/histogram
compact.

### SLIDE tab

Keep the marginal selection frequency and factor-response chart.

Useful elements:

- Marginal SLIDE knockoff frequency bars.
- Factor-response correlation or AUC bars.
- Interaction selection frequency when interactions are enabled.

Refinement: sort bars by frequency or selection status, not only by factor
index. Use neutral gray for unselected bars rather than red; red implies failure
or error, while "not selected" is often normal.

### LOVE diagnostics

Keep the LOVE diagnostics view, but make it more compact.

Useful elements:

- Selected `delta`, `lambda`, and score.
- `k`, pure variable count, and pure coverage.
- Group size range and singleton count.
- `opt_delta`, median gamma, and max gamma.

Refinement: remove or shorten the heuristic prose. It is useful guidance, but
long prose in a TUI competes with the actual run data. Prefer one compact
warning line only when the condition is detected.

## Refine Heavily

### LOVE tab layout

The LOVE tab is currently overloaded. It renders four fixed summary panels
above seven switchable subviews. That makes the tab feel dense before the user
even reaches the selected subview.

Recommended direction:

- Replace the four fixed summary panels with compact metric cards or one small
  grid.
- Give the active LOVE subview most of the screen.
- Consider merging redundant summary signals into the Overview tab.

### Matrix heatmaps

The factor covariance, loading matrix, and group correlation views are currently
numeric tables with colored cells. This works only when `k` is small. As `k`
or feature count grows, the views become hard to scan and can overflow.

Recommended direction:

- Use block-style heatmaps for scanability.
- Keep numeric values only for selected or hovered cells if interactive support
  is added later.
- Cap visible rows/columns clearly, or add paging.
- Preserve labels for small matrices, but avoid dense labels for large ones.

### Phase timeline

The phase timeline is useful and should stay. It currently appears in both the
waiting view and run summary path.

Recommended direction:

- Treat it as a global status strip.
- Keep the labels stable across views.
- Show interaction progress only when relevant.

## Hide, Move To Advanced, Or Remove

### Interaction Network

Move this out of the default LOVE subview cycle or remove it.

The current rendering is an adjacency list, not really a network visualization.
It also overlaps with the interaction selection view. It may still be useful as
debug output, but it should not be a primary visual.

Recommended direction:

- Hide behind an advanced/debug key.
- Or replace with a ranked interaction table showing pair, frequency, and
  response association.

### Gamma distribution

Move Gamma Distribution to advanced diagnostics.

Gamma values can be diagnostically useful, but the top-30 bar chart is not a
primary monitoring need for most runs. It is also visually expensive relative to
the value it provides during routine monitoring.

Recommended direction:

- Keep median and max gamma in LOVE diagnostics.
- Move the full gamma chart to an advanced diagnostics view.

### Delta Profile

Remove Delta Profile from the always-visible LOVE layout.

It largely duplicates information already available in the score grid and best
run summary. It is lower-value than giving the active LOVE subview more room.

Recommended direction:

- Keep the best-per-delta logic if useful.
- Surface it as an optional summary or integrate it into the Overview tab.

### Config / Identity panel

Do not remove config identity entirely, but stop showing the full panel by
default.

It is important for reproducibility, especially when attaching to a completed
checkpoint. However, it is not something users need to read continuously while
monitoring a run.

Recommended direction:

- Collapse to one or two identity lines in the Overview tab.
- Add a dedicated config/details view if full metadata is needed.

## Priority Recommendations

1. Simplify the LOVE tab first. It has the most visual overload and the most
   duplicate information.
2. Keep the Overview and Factors tabs as the core experience.
3. Move Gamma Distribution and Interaction Network into advanced/debug views.
4. Rework matrix views into scalable heatmaps or paged tables.
5. Reduce persistent footer/config noise so the TUI feels more intentional.

## Verification

`cargo check -p slide-cli` passed at the time of this assessment.
