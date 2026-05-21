#!/bin/bash
#SBATCH --job-name='natR'
#SBATCH -t 1-00:00
#SBATCH --nodes=1
#SBATCH --partition=htc
#SBATCH --cluster=htc
#SBATCH --ntasks=1
#SBATCH --mem=100g
#SBATCH --cpus-per-task=4

module load r/4.4.0

Rscript runSLIDE.R --yaml-path config.yaml