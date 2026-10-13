See the "Mechanism (Job 1b)" section of `../README.md`. Run: `r_gaptol.R <scratch> <mech_scratch>` (DSDP sweep, 3 gaptol x maxit 1000/10000),
then `mechanism.py <scratch> <mech_scratch> <out>` (needs `clarabel`, `cvxpy`/SCS in the dev env, PYTHONPATH=src).
Scratch inputs: `.tmpfiles/2026-10-13_clarabel-diag-s/` (sigma_*.csv, sRust_default_*.txt) and `.../mech/` (R sweep outputs).
`entries_all.csv`: every entry (344) with R/Py DSDP s per gaptol (correlation units), Clarabel raw/clipped/final, duals (mu_lo = multiplier
on s>=0, Zjj), reference Clarabel and SCS. `chunk_solver_summary.csv`: status, iterations, residuals, objectives per chunk.
