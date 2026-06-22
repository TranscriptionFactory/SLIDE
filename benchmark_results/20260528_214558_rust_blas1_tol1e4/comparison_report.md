# SLIDE Backend Benchmark Report

Generated: 2026-05-28 22:16:23

## Timing

| Backend | Delta | Lambda | Seconds | Status |
|---------|------:|-------:|--------:|--------|
| rust | 0.01 | 0.1 | 609.85 | success |
| rust | 0.01 | 1.0 | 603.77 | success |
| rust | 0.1 | 0.1 | 275.45 | success |
| rust | 0.1 | 1.0 | 304.12 | success |

### Total time per backend

| Backend | Total (s) | Speedup vs slowest |
|---------|----------:|-------------------:|
| rust | 1793.19 | 1.0x |

## Results per backend

| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |
|---------|------:|-------:|--:|--------:|--------------:|
| rust | 0.01 | 0.1 | 172 | 2 | 4 |
| rust | 0.01 | 1.0 | 172 | 2 | 2 |
| rust | 0.1 | 0.1 | 88 | 2 | 4 |
| rust | 0.1 | 1.0 | 88 | 4 | 10 |

## Pairwise correctness comparison

| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard | Z overall r | Z median col r | Z min col r | A nonzero Jaccard |
|-----------|-----------|------:|-------:|---------------:|------------:|---------------:|------------:|------------------:|
