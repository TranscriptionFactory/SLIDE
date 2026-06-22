# SLIDE Backend Benchmark Report

Generated: 2026-06-17 17:43:29

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 117.61 | success |
| rust | 0.01 | 1.0 | 91.00 | success |
| rust | 0.1 | 0.1 | 26.77 | success |
| rust | 0.1 | 1.0 | 24.21 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 259.59 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 12 | 28 |
| rust | 0.01 | 1.0 | 172 | 7 | 14 |
| rust | 0.1 | 0.1 | 88 | 3 | 10 |
| rust | 0.1 | 1.0 | 88 | 2 | 15 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
