# SLIDE Backend Benchmark Report

Generated: 2026-05-28 21:43:14

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 407.19 | success |
| rust | 0.01 | 1.0 | 368.74 | success |
| rust | 0.1 | 0.1 | 196.76 | success |
| rust | 0.1 | 1.0 | 243.27 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 1215.97 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 3 | 7 |
| rust | 0.01 | 1.0 | 172 | 2 | 2 |
| rust | 0.1 | 0.1 | 88 | 2 | 4 |
| rust | 0.1 | 1.0 | 88 | 4 | 11 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
