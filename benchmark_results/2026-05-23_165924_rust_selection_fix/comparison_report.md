# SLIDE Backend Benchmark Report

Generated: 2026-05-23 17:10:29

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 158.79 | success |
| rust | 0.01 | 1.0 | 139.07 | success |
| rust | 0.1 | 0.1 | 146.38 | success |
| rust | 0.1 | 1.0 | 177.23 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 621.47 | 1.0x |

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
