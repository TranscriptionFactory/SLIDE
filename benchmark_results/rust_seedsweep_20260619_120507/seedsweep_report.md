# Rust knockoff base_seed sweep vs pure_r

Generated: 2026-06-20 13:30:17

Sweep dir: `rust_seedsweep_20260619_120507`; seeds: [1, 7, 42, 123, 2024]. Reference: pure_r (native-R SLIDE) from `lfcompare_tolfix_20260617_233303`.

Question: does borderline-LF selection wander across the knockoff RNG base_seed? If yes, the residual rust-vs-pure_r gap is irreducible Monte-Carlo noise; if the seed-averaged rust set converges toward pure_r's, that union is the honest parity number.

## delta=0.01, lambda=0.1

niter=500, spec threshold = 50 votes (10%). pure_r sig LFs: ['Z18', 'Z27', 'Z45', 'Z74', 'Z136']

### Per-LF selection_count across seeds

| LF | s1 | s7 | s42 | s123 | s2024 | mean | min | max | #sel | in pure_r |
|---|---|---|---|---|---|---|---|---|---|---|
| Z18 | 71 | 71 | 67 | 70 | 85 | 72.8 | 67 | 85 | 5/5 | yes |
| Z27 | 182 | 179 | 178 | 174 | 198 | 182.2 | 174 | 198 | 5/5 | yes |
| Z45 | 24 | 24 | 23 | 21 | 29 | 24.2 | 21 | 29 | 0/5 | yes |
| Z74 | 12 | 12 | 10 | 9 | 11 | 10.8 | 9 | 12 | 0/5 | yes |
| Z136 | 24 | 24 | 23 | 20 | 28 | 23.8 | 20 | 28 | 0/5 | yes |

### Per-seed sig-LF set vs pure_r

| Seed | rust sig LFs | Jaccard vs pure_r |
|------|--------------|------------------:|
| 1 | ['Z18', 'Z27'] | 0.4000 |
| 7 | ['Z18', 'Z27'] | 0.4000 |
| 42 | ['Z18', 'Z27'] | 0.4000 |
| 123 | ['Z18', 'Z27'] | 0.4000 |
| 2024 | ['Z18', 'Z27'] | 0.4000 |

Mean per-seed Jaccard vs pure_r: **0.4000**

### Seed-aggregated rust set vs pure_r

| Aggregation | set | Jaccard vs pure_r |
|-------------|-----|------------------:|
| union (any seed) | ['Z18', 'Z27'] | 0.4000 |
| majority of seeds | ['Z18', 'Z27'] | 0.4000 |
| mean count >= thresh | ['Z18', 'Z27'] | 0.4000 |

## delta=0.01, lambda=1.0

niter=500, spec threshold = 50 votes (10%). pure_r sig LFs: ['Z18', 'Z27', 'Z45', 'Z57', 'Z73', 'Z136']

### Per-LF selection_count across seeds

| LF | s1 | s7 | s42 | s123 | s2024 | mean | min | max | #sel | in pure_r |
|---|---|---|---|---|---|---|---|---|---|---|
| Z18 | 128 | 126 | 123 | 129 | 123 | 125.8 | 123 | 129 | 5/5 | yes |
| Z27 | 220 | 220 | 250 | 246 | 243 | 235.8 | 220 | 250 | 5/5 | yes |
| Z45 | 39 | 39 | 31 | 35 | 34 | 35.6 | 31 | 39 | 0/5 | yes |
| Z57 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0/5 | yes |
| Z73 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0/5 | yes |
| Z136 | 38 | 38 | 31 | 35 | 34 | 35.2 | 31 | 38 | 0/5 | yes |

### Per-seed sig-LF set vs pure_r

| Seed | rust sig LFs | Jaccard vs pure_r |
|------|--------------|------------------:|
| 1 | ['Z18', 'Z27'] | 0.3333 |
| 7 | ['Z18', 'Z27'] | 0.3333 |
| 42 | ['Z18', 'Z27'] | 0.3333 |
| 123 | ['Z18', 'Z27'] | 0.3333 |
| 2024 | ['Z18', 'Z27'] | 0.3333 |

Mean per-seed Jaccard vs pure_r: **0.3333**

### Seed-aggregated rust set vs pure_r

| Aggregation | set | Jaccard vs pure_r |
|-------------|-----|------------------:|
| union (any seed) | ['Z18', 'Z27'] | 0.3333 |
| majority of seeds | ['Z18', 'Z27'] | 0.3333 |
| mean count >= thresh | ['Z18', 'Z27'] | 0.3333 |

## delta=0.1, lambda=0.1

niter=500, spec threshold = 50 votes (10%). pure_r sig LFs: ['Z25', 'Z33', 'Z53', 'Z61', 'Z62']

### Per-LF selection_count across seeds

| LF | s1 | s7 | s42 | s123 | s2024 | mean | min | max | #sel | in pure_r |
|---|---|---|---|---|---|---|---|---|---|---|
| Z5 | 43 | 43 | 42 | 48 | 49 | 45.0 | 42 | 49 | 0/5 |  |
| Z10 | 13 | 13 | 13 | 26 | 24 | 17.8 | 13 | 26 | 0/5 |  |
| Z25 | 32 | 32 | 31 | 37 | 37 | 33.8 | 31 | 37 | 0/5 | yes |
| Z33 | 25 | 26 | 25 | 32 | 28 | 27.2 | 25 | 32 | 0/5 | yes |
| Z39 | 19 | 19 | 20 | 28 | 24 | 22.0 | 19 | 28 | 0/5 |  |
| Z44 | 57 | 57 | 56 | 67 | 66 | 60.6 | 56 | 67 | 5/5 |  |
| Z53 | 26 | 27 | 26 | 32 | 29 | 28.0 | 26 | 32 | 0/5 | yes |
| Z61 | 57 | 57 | 56 | 67 | 66 | 60.6 | 56 | 67 | 5/5 | yes |
| Z62 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0/5 | yes |

### Per-seed sig-LF set vs pure_r

| Seed | rust sig LFs | Jaccard vs pure_r |
|------|--------------|------------------:|
| 1 | ['Z44', 'Z61'] | 0.1667 |
| 7 | ['Z44', 'Z61'] | 0.1667 |
| 42 | ['Z44', 'Z61'] | 0.1667 |
| 123 | ['Z44', 'Z61'] | 0.1667 |
| 2024 | ['Z44', 'Z61'] | 0.1667 |

Mean per-seed Jaccard vs pure_r: **0.1667**

### Seed-aggregated rust set vs pure_r

| Aggregation | set | Jaccard vs pure_r |
|-------------|-----|------------------:|
| union (any seed) | ['Z44', 'Z61'] | 0.1667 |
| majority of seeds | ['Z44', 'Z61'] | 0.1667 |
| mean count >= thresh | ['Z44', 'Z61'] | 0.1667 |

## delta=0.1, lambda=1.0

niter=500, spec threshold = 50 votes (10%). pure_r sig LFs: ['Z33', 'Z44']

### Per-LF selection_count across seeds

| LF | s1 | s7 | s42 | s123 | s2024 | mean | min | max | #sel | in pure_r |
|---|---|---|---|---|---|---|---|---|---|---|
| Z5 | 22 | 22 | 31 | 34 | 44 | 30.6 | 22 | 44 | 0/5 |  |
| Z25 | 15 | 15 | 21 | 23 | 31 | 21.0 | 15 | 31 | 0/5 |  |
| Z33 | 16 | 16 | 23 | 26 | 33 | 22.8 | 16 | 33 | 0/5 | yes |
| Z39 | 14 | 14 | 21 | 23 | 30 | 20.4 | 14 | 30 | 0/5 |  |
| Z44 | 106 | 106 | 84 | 86 | 109 | 98.2 | 84 | 109 | 5/5 | yes |
| Z61 | 73 | 73 | 55 | 56 | 79 | 67.2 | 55 | 79 | 5/5 |  |
| Z69 | 23 | 23 | 30 | 33 | 43 | 30.4 | 23 | 43 | 0/5 |  |

### Per-seed sig-LF set vs pure_r

| Seed | rust sig LFs | Jaccard vs pure_r |
|------|--------------|------------------:|
| 1 | ['Z44', 'Z61'] | 0.3333 |
| 7 | ['Z44', 'Z61'] | 0.3333 |
| 42 | ['Z44', 'Z61'] | 0.3333 |
| 123 | ['Z44', 'Z61'] | 0.3333 |
| 2024 | ['Z44', 'Z61'] | 0.3333 |

Mean per-seed Jaccard vs pure_r: **0.3333**

### Seed-aggregated rust set vs pure_r

| Aggregation | set | Jaccard vs pure_r |
|-------------|-----|------------------:|
| union (any seed) | ['Z44', 'Z61'] | 0.3333 |
| majority of seeds | ['Z44', 'Z61'] | 0.3333 |
| mean count >= thresh | ['Z44', 'Z61'] | 0.3333 |

## Summary: Jaccard vs pure_r

| Delta | Lambda | mean per-seed | union | majority | mean>=thresh |
|------:|-------:|--------------:|------:|---------:|-------------:|
| 0.01 | 0.1 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| 0.01 | 1.0 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| 0.1 | 0.1 | 0.1667 | 0.1667 | 0.1667 | 0.1667 |
| 0.1 | 1.0 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
