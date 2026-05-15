#!/usr/bin/env Rscript
# =============================================================================
# run_slide_single.R - Run SLIDE analysis for a single stratum
# =============================================================================
#
# This script is designed to be called from Python pipelines to run SLIDE
# using the R SLIDE package. It handles a single stratum with configurable
# parameter sweeps.
#
# Usage:
#   Rscript run_slide_single.R --input-dir <dir> --output-dir <dir> [options]
#
# Required Arguments:
#   --input-dir     Directory containing x.csv and y.csv
#   --output-dir    Output directory for SLIDE results
#   --yaml-path     Path to params.yaml (optional, will be generated if not provided)
#
# Output:
#   Creates subdirectories like {delta}_{lambda}_out containing:
#   - Alloc_mat.csv: Allocation matrix (metabolites x latent factors)
#   - params.yaml: Parameters used for this run
#   - Various SLIDE diagnostic outputs
#
# Examples:
#   # Basic usage
#   Rscript run_slide_single.R --input-dir data/TBI_Male --output-dir outputs/TBI_Male
#
#   # Custom parameters
#   Rscript run_slide_single.R --input-dir data/HV --output-dir outputs/HV \
#     --lambda 0.5,0.1 --delta 0.1,0.15 --niter 500
#
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
})

library(devtools)
# devtools::install_github("jishnu-lab/SLIDE")
library(SLIDE)


# Define command line options
option_list = list(
  make_option(c("--yaml-path"), type = "character", default = "config.yaml",
              help = "Path to params.yaml (optional, will be generated if not provided)")
              
)

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

log_msg = function(...) {
  cat(paste0(..., "\n"))
}

# Validate required arguments
if (is.null(opt$`yaml-path`) ) {
  print_help(opt_parser)
  stop("--yaml-path required", call. = FALSE)
}

yaml_path = opt$`yaml-path`


if (!file.exists(yaml_path)) {
  stop("yaml path not found: ",yaml_path)
}


input_params = yaml::yaml.load_file(yaml_path)
output_dir = input_params$out_path

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


log_msg("\nRunning SLIDE parameter sweep...")

delta_vals = input_params$delta
lambda_vals = input_params$lambda
spec_vals = input_params$spec

loop_params = input_params

for (s in spec_vals) {
  for (l in lambda_vals) {
    for (d in delta_vals) {

      loop_params$lambda = l
      loop_params$delta = d
      loop_params$spec = s
      loop_params$out_path = paste0(output_dir, "/")

      combo_params_path = file.path(output_dir, "yaml_params.yaml")
      yaml::write_yaml(loop_params, combo_params_path)

      tryCatch({
        SLIDE::optimizeSLIDE(loop_params, sink_file = FALSE)
        log_msg("  -> Success")
        SLIDE::plotCorrelationNetworks(loop_params)
        SLIDE::SLIDEcv(combo_params_path, k = 5, nreps=10)

      }, error = function(e) {
        log_msg("  -> ERROR: ", conditionMessage(e))
        message(traceback())
      })
    }
  }
}
