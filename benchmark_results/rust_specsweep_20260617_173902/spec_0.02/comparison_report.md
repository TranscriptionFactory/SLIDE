# SLIDE Backend Benchmark Report

Generated: 2026-06-17 17:50:35

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 115.95 | success |
| rust | 0.01 | 1.0 | 202.93 | success |
| rust | 0.1 | 0.1 | 77.12 | success |
| rust | 0.1 | 1.0 | 27.80 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 423.80 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 11 | 64 |
| rust | 0.01 | 1.0 | 172 | 16 | 77 |
| rust | 0.1 | 0.1 | 88 | 6 | 40 |
| rust | 0.1 | 1.0 | 88 | 3 | 25 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
