#!/usr/bin/env bash
set -euo pipefail
OUT=$(cd "$(dirname "$0")" && pwd)
(set -o noclobber; printf 'backend,job_id\n' > "$OUT/jobs.csv")
previous=
for backend in rust python pure_r; do
    dependency=()
    [[ -z $previous ]] || dependency=(--dependency="afterany:$previous")
    id=$(sbatch --parsable --cluster=htc --partition=htc --constraint=turin \
        --nodelist=htc-n70 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=128G \
        --time=02:15:00 --job-name="torok-all-$backend" \
        --output="$OUT/logs/$backend-%j.out" "${dependency[@]}" "$OUT/run.sh" "$backend")
    previous=${id%%;*}
    printf '%s,%s\n' "$backend" "$previous" | tee -a "$OUT/jobs.csv"
done
