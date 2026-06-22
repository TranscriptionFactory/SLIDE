# SLIDE Backend Benchmark Report

Generated: 2026-05-28 23:21:20

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 203.01 | success |
| rust | 0.01 | 1.0 | 194.73 | success |
| rust | 0.1 | 0.1 | 88.49 | success |
| rust | 0.1 | 1.0 | 91.73 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 577.96 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 2 | 4 |
| rust | 0.01 | 1.0 | 172 | 2 | 2 |
| rust | 0.1 | 0.1 | 88 | 2 | 4 |
| rust | 0.1 | 1.0 | 88 | 2 | 7 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
