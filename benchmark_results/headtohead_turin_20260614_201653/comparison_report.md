# SLIDE Backend Benchmark Report

Generated: 2026-06-14 20:22:48

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| pure_r | 0.01 | 0.1 | 53.36 | success |
| pure_r | 0.01 | 1.0 | 45.56 | success |
| pure_r | 0.1 | 0.1 | 16.90 | success |
| pure_r | 0.1 | 1.0 | 17.85 | success |
| rust | 0.01 | 0.1 | 57.07 | success |
| rust | 0.01 | 1.0 | 66.37 | success |
| rust | 0.1 | 0.1 | 31.67 | success |
| rust | 0.1 | 1.0 | 33.77 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| pure_r | 133.67 | 1.4x |
| rust | 188.88 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| pure_r | 0.01 | 0.1 | 172 | 5 | 14 |
| pure_r | 0.01 | 1.0 | 172 | 6 | 18 |
| pure_r | 0.1 | 0.1 | 88 | 2 | 3 |
| pure_r | 0.1 | 1.0 | 88 | 3 | 6 |
| rust | 0.01 | 0.1 | 172 | 2 | 3 |
| rust | 0.01 | 1.0 | 172 | 2 | 3 |
| rust | 0.1 | 0.1 | 88 | 2 | 6 |
| rust | 0.1 | 1.0 | 88 | 2 | 8 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
| pure_r | rust | 0.01 | 0.1 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.01 | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.1 | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |
| pure_r | rust | 0.1 | 1.0 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | N/A |
