# R sampleCV reference fixtures

Generated with `Rscript tests/fixtures/generate/generate_sample_cv_fixtures.R`.
The generator sources production `R/sampleCV.R`, uses `K=4`, and supplies the listed explicit zero-based train splits (iteration count equals the split count). It records R `lm`/`predict`, pooled Spearman, and fixed-direction pROC AUC results. Cases cover reversed AUC (0), tied AUC (0.5), raw interactions, constant-response missing metric, ordered exact/near aliases, and the endpoint-union guard. R >= 4.4, jsonlite, and pROC are required.
