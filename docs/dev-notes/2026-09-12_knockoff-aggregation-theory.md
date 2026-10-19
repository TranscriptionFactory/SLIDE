# Statistical guarantees for representative-draw selection

Date: 2026-09-12. Scope: knockoff aggregation theory, not an audit of the SLIDE
paper or R implementation. Here `findOptIter` means: form a frequency set,
maximize its overlap with a realized draw, then minimize draw size among ties.

## What the existing theorems establish

Barber and Candès distinguish ordinary knockoffs, which control a modified FDR
(Theorem 1), from knockoff+, which controls FDR (Theorem 2). Their fixed-X result
assumes the stated Gaussian linear model, valid knockoff construction, and
appropriate statistics. These are guarantees for the specified selection
procedure, not for arbitrary postprocessing of its output. Check the actual
threshold offset before claiming ordinary FDR control.
[Original paper, definitions 1–2 and theorems 1–2](https://arxiv.org/pdf/1404.5609).

Ren and Barber average knockoff e-values across draws and apply e-BH. Their
Theorem 3 proves finite-sample FDR control for that aggregation, assuming valid
knockoffs and statistics. They do not prove the same result for choosing a
representative draw. Their discussion distinguishes frequency-threshold
derandomization, with expected-false-discovery bounds, from FDR control. They
also flag the additional strong assumptions of earlier AKO work, including
i.i.d. null statistics. Thus “use aggregation” alone does not specify a valid
FDR procedure.
[Published paper, sections 1.2–1.4 and 3, theorem 3](https://academic.oup.com/jrsssb/article/86/1/122/7262479).

## Why per-run control alone does not settle `findOptIter`

The following is our mathematical illustration, not a SLIDE simulation or a
counterexample constructed from valid knockoff statistics.

Suppose there is one null feature. Each independent base run selects it with
probability 0.2 and otherwise selects nothing. Every base run has FDR = 0.2.
Let the frequency threshold be 0.1. With M runs, the feature becomes frequent
whenever its selection count K satisfies K/M ≥ 0.1. On that event, maximizing
overlap chooses a singleton rather than an empty draw. The minimum-size
tiebreak cannot change this. By the law of large numbers, K/M → 0.2, so the
representative-draw procedure's FDR approaches 1 as M grows.

This refutes the general inference “every draw controls FDR, therefore the
adaptively chosen draw controls FDR.” It does **not** prove SLIDE violates FDR:
a SLIDE-specific proof might exploit additional structure absent here.
Frequency-only thresholding has the same problem in this illustration.

## Methodological interpretation

Choosing a small realized set with strong frequency overlap is a reasonable
representative-set objective. Neither that objective nor improved seed overlap
establishes scientific optimality or an error-control theorem. Smallest size
does not establish smallest false-discovery proportion.

Keep the original rule when the task is faithful SLIDE reproduction. Describe
its inferential guarantees only as far as the SLIDE proof supports them. Before
replacing it for stability, distinguish Monte Carlo repeatability, power, FDR,
and predictive accuracy. A replacement such as e-value aggregation requires
checking its assumptions against SLIDE's latent-factor estimation, screening,
knockoff generator, and interaction stages; its theorem is not an automatic
guarantee for the full pipeline.

Verification: checked the cited theorem statements and the algebraic
illustration. No numerical experiment or backend modification was performed.
