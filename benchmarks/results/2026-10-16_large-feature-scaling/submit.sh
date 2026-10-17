#!/usr/bin/env bash
set -euo pipefail
OUT=$(cd "$(dirname "$0")" && pwd)
# Exclusive creation prevents an accidental duplicate submission.
(set -o noclobber; printf 'backend,p,job_id\n' > "$OUT/jobs.csv")
previous=
for backend in rust python pure_r; do
    for p in 10000 15000 20000; do
        dependency=()
        [[ -z $previous ]] || dependency=(--dependency="afterany:$previous")
        id=$(sbatch --parsable --cluster=htc --partition=htc --constraint=turin \
            --nodelist=htc-n73 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=128G \
            --time=02:15:00 --job-name="scale-$backend-p$p" \
            --output="$OUT/logs/$backend-p$p-%j.out" "${dependency[@]}" \
            "$OUT/cell.sh" "$OUT/manifests/$backend-p$p.json")
        previous=${id%%;*}
        printf '%s,%s,%s\n' "$backend" "$p" "$previous" | tee -a "$OUT/jobs.csv"
    done
done
id=$(sbatch --parsable --cluster=htc --partition=htc --cpus-per-task=1 --mem=8G \
    --time=00:15:00 --job-name=scale-cost-analysis --dependency="afterany:$previous" \
    --output="$OUT/logs/analysis-%j.out" "$OUT/aggregate.sh")
printf 'analysis,,%s\n' "${id%%;*}" | tee -a "$OUT/jobs.csv"
