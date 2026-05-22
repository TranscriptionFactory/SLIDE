"""Generate parameter grid for array jobs."""
import json

deltas = [0.01, 0.05, 0.1]
lambdas = [0.1, 0.5, 1.0]

params = []
for d in deltas:
    for l in lambdas:
        params.append({"delta": d, "lambda": l, "index": len(params)})

with open("benchmark_results/comprehensive/param_grid.json", "w") as f:
    json.dump(params, f, indent=2)

print(f"Generated {len(params)} parameter combinations")
for p in params:
    print(f"  [{p['index']}] delta={p['delta']}, lambda={p['lambda']}")
