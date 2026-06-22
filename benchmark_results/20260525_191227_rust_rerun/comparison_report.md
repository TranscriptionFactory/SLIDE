# SLIDE Backend Benchmark Report

Generated: 2026-05-25 19:17:50

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 73.48 | success |
| rust | 0.01 | 1.0 | 71.30 | success |
| rust | 0.1 | 0.1 | 72.29 | success |
| rust | 0.1 | 1.0 | 93.18 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 310.24 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 5 | 19 |
| rust | 0.01 | 1.0 | 172 | 5 | 20 |
| rust | 0.1 | 0.1 | 88 | 14 | 63 |
| rust | 0.1 | 1.0 | 88 | 12 | 55 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
