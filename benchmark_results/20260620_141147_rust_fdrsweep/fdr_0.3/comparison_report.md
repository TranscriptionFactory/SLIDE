# SLIDE Backend Benchmark Report

Generated: 2026-06-20 14:36:20

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 222.86 | success |
| rust | 0.01 | 1.0 | 191.67 | success |
| rust | 0.1 | 0.1 | 133.63 | success |
| rust | 0.1 | 1.0 | 126.60 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 674.76 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 22 | 145 |
| rust | 0.01 | 1.0 | 172 | 21 | 106 |
| rust | 0.1 | 0.1 | 88 | 21 | 146 |
| rust | 0.1 | 1.0 | 88 | 23 | 139 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
