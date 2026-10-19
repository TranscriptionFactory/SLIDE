# Python with R knockoffs: extended time limit

Extended-time rerun of all ten seeds and four cells from the October 16 campaign.
It uses the same frozen source `c4726e34fde91e59429f88d2fd228f785ce845dc`, inputs,
R library, extensions, and `run.sbatch`, symlinked from
`/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-10-16_backend-replicates-all-arms`.
Only the submitted array range (40-49) and task time limit (two hours rather
than 20 minutes) change. Four CPUs, 4 GB, Turin nodes and ten concurrent tasks
remain unchanged. Outputs are new; do not pool old partial cells with reruns.

## Recorded submission

The command and variables below refer to the original shared-filesystem setup.
Submission from the SLIDE repository root:

sbatch --parsable --array=40-49%10 --time=02:00:00 --job-name=slide-python-r-long --output=\"$RUN/logs/%A_%a.out\" \"$RUN/run.sbatch\" \"$RUN\" \"$ENV\" \"$R_DEPS\"

RUN=/vast/djishnu/Aaron/1_general_use/SLIDE/.tmpfiles/2026-10-17_python-r-knockoffs-extended
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library

The script verifies input/extension hashes and rejects tasks unless all four
cells succeed. After completion, check all ten task exit codes, 40 successful
cells, and K/Z alignment with the original pure-R reference before publishing
updated selections, runtimes and the five-arm heatmap. Stock R worker streams
are not fully seed-controlled; rerun selections need not match earlier draws.

## Submission status (2026-10-17)

Submitted to HTC as array **11967812**, tasks 40–49, on 2026-10-17.
At this checkpoint, results were pending. The existing heatmap shows the
original partial run; this record does not confirm successful completion.
