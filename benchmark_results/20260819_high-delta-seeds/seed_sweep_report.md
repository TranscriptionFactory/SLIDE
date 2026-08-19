# Backend seed-sweep parity

Seeds: 42, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000

Stable core: selected in at least 50% of available seeds.

| Backend | Reference | Delta | Lambda | Marginal Jaccard | Marginal recall | Interaction Jaccard | Interaction recall |
|---|---|---:|---:|---:|---:|---:|---:|
| python | pure_r | 0.1 | 0.1 | 0.500 | 0.667 | 0.333 | 1.000 |
| python | pure_r | 0.1 | 1.0 | 0.667 | 0.667 | 0.400 | 0.571 |
| rust | pure_r | 0.1 | 0.1 | 1.000 | 1.000 | 0.214 | 1.000 |
| rust | pure_r | 0.1 | 1.0 | 0.250 | 0.333 | 0.214 | 0.429 |
