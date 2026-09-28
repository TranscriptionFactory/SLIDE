#!/usr/bin/env Rscript
# Mirror the installed SLIDE PSOCK setup; this probes RNG setup, not selection.
args <- commandArgs(trailingOnly=TRUE)
out <- args[[1]]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
slide_body <- deparse(get("SLIDE", asNamespace("SLIDE")))
stopifnot(any(grepl("makePSOCKcluster", slide_body, fixed=TRUE)))
stopifnot(!any(grepl("clusterSetRNGStream", slide_body, fixed=TRUE)))
writeLines(slide_body, file.path(out,"installed_SLIDE.txt"))
writeLines(deparse(get("secondKO", asNamespace("SLIDE"))), file.path(out,"installed_secondKO.txt"))
worker_probe <- function(seed_workers) {
    set.seed(42)
    cl <- parallel::makePSOCKcluster(4)
    on.exit(parallel::stopCluster(cl))
    if (seed_workers) parallel::clusterSetRNGStream(cl, 42)
    values <- parallel::clusterCall(cl, function() {
        initialized <- exists(".Random.seed", envir=.GlobalEnv, inherits=FALSE)
        c(initialized=initialized, rnorm(8))
    })
    do.call(rbind, values)
}
rows <- list()
for (seed_workers in c(FALSE, TRUE)) {
    a <- worker_probe(seed_workers)
    b <- worker_probe(seed_workers)
    same <- identical(a,b)
    rows[[length(rows)+1L]] <- data.frame(seed_workers=seed_workers,
        workers_initialized_a=sum(a[,1]), workers_initialized_b=sum(b[,1]),
        identical_worker_draws=same, max_abs_difference=max(abs(a[,-1]-b[,-1])))
    write.table(a, file.path(out,paste0("workers_",seed_workers,"_a.csv")),sep=",",row.names=FALSE,col.names=FALSE)
    write.table(b, file.path(out,paste0("workers_",seed_workers,"_b.csv")),sep=",",row.names=FALSE,col.names=FALSE)
    stopifnot(same == seed_workers)
}
write.csv(do.call(rbind,rows), file.path(out,"worker_rng.csv"), row.names=FALSE)
