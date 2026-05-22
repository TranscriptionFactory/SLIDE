# SLIDE Backend Benchmark Report

Generated: 2026-05-22 14:54:49

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 1482.84 | success |
| rust | 0.01 | 1.0 | 1675.74 | success |
| rust | 0.1 | 0.1 | 155.32 | success |
| rust | 0.1 | 1.0 | 152.68 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 3466.58 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 5 | 20 |
| rust | 0.01 | 1.0 | 172 | 6 | 23 |
| rust | 0.1 | 0.1 | 88 | 13 | 54 |
| rust | 0.1 | 1.0 | 88 | 13 | 58 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
