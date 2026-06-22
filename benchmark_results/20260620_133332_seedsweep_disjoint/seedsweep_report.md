# Rust knockoff base_seed sweep vs pure_r

Generated: 2026-06-20 13:47:35

Sweep dir: `20260620_133332_seedsweep_disjoint`; seeds: [42, 1000, 2000, 3000, 4000]. Reference: pure_r (native-R SLIDE) from `lfcompare_tolfix_20260617_233303`.

Question: does borderline-LF selection wander across the knockoff RNG base_seed? If yes, the residual rust-vs-pure_r gap is irreducible Monte-Carlo noise; if the seed-averaged rust set converges toward pure_r's, that union is the honest parity number.

## delta=0.01, lambda=0.1

niter=500, spec threshold = 50 votes (10%). pure_r sig LFs: ['Z18', 'Z27', 'Z45', 'Z74', 'Z136']

### Per-LF selection_count across seeds

| LF | s42 | s1000 | s2000 | s3000 | s4000 | mean | min | max | #sel | in pure_r |
|---|---|---|---|---|---|---|---|---|---|---|
| Z18 | 67 | 64 | 91 | 68 | 72 | 72.4 | 64 | 91 | 5/5 | yes |
| Z27 | 178 | 174 | 210 | 173 | 171 | 181.2 | 171 | 210 | 5/5 | yes |
| Z45 | 23 | 24 | 27 | 27 | 34 | 27.0 | 23 | 34 | 0/5 | yes |
| Z74 | 10 | 11 | 11 | 14 | 16 | 12.4 | 10 | 16 | 0/5 | yes |
| Z136 | 23 | 22 | 25 | 25 | 30 | 25.0 | 22 | 30 | 0/5 | yes |

### Per-seed sig-LF set vs pure_r

| Seed | rust sig LFs | Jaccard vs pure_r |
|------|--------------|------------------:|
| 42 | ['Z18', 'Z27'] | 0.4000 |
| 1000 | ['Z18', 'Z27'] | 0.4000 |
| 2000 | ['Z18', 'Z27'] | 0.4000 |
| 3000 | ['Z18', 'Z27'] | 0.4000 |
| 4000 | ['Z18', 'Z27'] | 0.4000 |

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

| LF | s42 | s1000 | s2000 | s3000 | s4000 | mean | min | max | #sel | in pure_r |
|---|---|---|---|---|---|---|---|---|---|---|
| Z18 | 123 | 120 | 142 | 136 | 151 | 134.4 | 120 | 151 | 5/5 | yes |
| Z27 | 250 | 214 | 237 | 274 | 272 | 249.4 | 214 | 274 | 5/5 | yes |
| Z45 | 31 | 25 | 35 | 42 | 39 | 34.4 | 25 | 42 | 0/5 | yes |
| Z57 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0/5 | yes |
| Z73 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0/5 | yes |
| Z136 | 31 | 24 | 36 | 42 | 39 | 34.4 | 24 | 42 | 0/5 | yes |

### Per-seed sig-LF set vs pure_r

| Seed | rust sig LFs | Jaccard vs pure_r |
|------|--------------|------------------:|
| 42 | ['Z18', 'Z27'] | 0.3333 |
| 1000 | ['Z18', 'Z27'] | 0.3333 |
| 2000 | ['Z18', 'Z27'] | 0.3333 |
| 3000 | ['Z18', 'Z27'] | 0.3333 |
| 4000 | ['Z18', 'Z27'] | 0.3333 |

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

| LF | s42 | s1000 | s2000 | s3000 | s4000 | mean | min | max | #sel | in pure_r |
|---|---|---|---|---|---|---|---|---|---|---|
| Z5 | 42 | 40 | 48 | 43 | 53 | 45.2 | 40 | 53 | 1/5 |  |
| Z25 | 31 | 24 | 35 | 29 | 43 | 32.4 | 24 | 43 | 0/5 | yes |
| Z33 | 25 | 23 | 25 | 24 | 34 | 26.2 | 23 | 34 | 0/5 | yes |
| Z39 | 20 | 17 | 21 | 22 | 25 | 21.0 | 17 | 25 | 0/5 |  |
| Z44 | 56 | 53 | 66 | 52 | 62 | 57.8 | 52 | 66 | 5/5 |  |
| Z53 | 26 | 22 | 26 | 25 | 35 | 26.8 | 22 | 35 | 0/5 | yes |
| Z61 | 56 | 53 | 66 | 52 | 60 | 57.4 | 52 | 66 | 5/5 | yes |
| Z62 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0/5 | yes |

### Per-seed sig-LF set vs pure_r

| Seed | rust sig LFs | Jaccard vs pure_r |
|------|--------------|------------------:|
| 42 | ['Z44', 'Z61'] | 0.1667 |
| 1000 | ['Z44', 'Z61'] | 0.1667 |
| 2000 | ['Z44', 'Z61'] | 0.1667 |
| 3000 | ['Z44', 'Z61'] | 0.1667 |
| 4000 | ['Z5', 'Z44', 'Z61'] | 0.1429 |

Mean per-seed Jaccard vs pure_r: **0.1619**

### Seed-aggregated rust set vs pure_r

| Aggregation | set | Jaccard vs pure_r |
|-------------|-----|------------------:|
| union (any seed) | ['Z5', 'Z44', 'Z61'] | 0.1429 |
| majority of seeds | ['Z44', 'Z61'] | 0.1667 |
| mean count >= thresh | ['Z44', 'Z61'] | 0.1667 |

## delta=0.1, lambda=1.0

niter=500, spec threshold = 50 votes (10%). pure_r sig LFs: ['Z33', 'Z44']

### Per-LF selection_count across seeds

| LF | s42 | s1000 | s2000 | s3000 | s4000 | mean | min | max | #sel | in pure_r |
|---|---|---|---|---|---|---|---|---|---|---|
| Z5 | 31 | 37 | 44 | 28 | 33 | 34.6 | 28 | 44 | 0/5 |  |
| Z25 | 21 | 25 | 31 | 20 | 25 | 24.4 | 20 | 31 | 0/5 |  |
| Z33 | 23 | 29 | 33 | 21 | 27 | 26.6 | 21 | 33 | 0/5 | yes |
| Z39 | 21 | 25 | 30 | 20 | 24 | 24.0 | 20 | 30 | 0/5 |  |
| Z44 | 84 | 87 | 111 | 86 | 94 | 92.4 | 84 | 111 | 5/5 | yes |
| Z61 | 55 | 63 | 80 | 52 | 71 | 64.2 | 52 | 80 | 5/5 |  |
| Z69 | 30 | 37 | 43 | 27 | 31 | 33.6 | 27 | 43 | 0/5 |  |

### Per-seed sig-LF set vs pure_r

| Seed | rust sig LFs | Jaccard vs pure_r |
|------|--------------|------------------:|
| 42 | ['Z44', 'Z61'] | 0.3333 |
| 1000 | ['Z44', 'Z61'] | 0.3333 |
| 2000 | ['Z44', 'Z61'] | 0.3333 |
| 3000 | ['Z44', 'Z61'] | 0.3333 |
| 4000 | ['Z44', 'Z61'] | 0.3333 |

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
| 0.1 | 0.1 | 0.1619 | 0.1429 | 0.1667 | 0.1667 |
| 0.1 | 1.0 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
