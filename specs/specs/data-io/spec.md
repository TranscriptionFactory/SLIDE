# Data I/O

---

## 1. Input Formats

### CSV Input

**WHEN** a CSV file is provided as input for `X`
**THEN** it is parsed as comma-separated with the first row treated as the header (feature names)

**WHEN** the first column of the CSV contains non-numeric values
**THEN** it is treated as sample IDs and excluded from the numeric matrix

**WHEN** the first column contains numeric values indistinguishable from features
**THEN** it is included in the numeric matrix and no sample ID is recorded unless explicitly flagged

### TSV Input

**WHEN** a TSV file is provided as input
**THEN** it is parsed identically to CSV except the delimiter is a tab character

**WHEN** the file extension is `.tsv` or `.txt`
**THEN** tab-separated parsing is used by default without requiring an explicit delimiter argument

**WHEN** the file extension is `.csv`
**THEN** comma-separated parsing is used by default

### Matrix Dimensions

**WHEN** `X` is loaded from file
**THEN** its shape is `n x p` where `n` is the number of samples and `p` is the number of features

**WHEN** `y` is loaded from file
**THEN** its shape is `n x 1` or a flat vector of length `n`

**WHEN** the number of rows in `X` does not match the length of `y`
**THEN** a descriptive error is raised before any computation begins, reporting both lengths

### Numeric Types

**WHEN** `X` is loaded
**THEN** all values are cast to f64

**WHEN** a cell in `X` cannot be parsed as a number
**THEN** an error is raised identifying the row, column, and offending value unless the `--impute` flag is set

**WHEN** `--impute` is set and missing values are present
**THEN** each missing value is replaced with the column mean computed from non-missing entries

### Response Variable

**WHEN** `y` is loaded and `response_type="binary"`
**THEN** all values must be 0 or 1; any other value raises a descriptive error

**WHEN** `y` is loaded and `response_type="continuous"`
**THEN** values are cast to f64 with no domain restriction

**WHEN** `y` contains NaN or missing entries
**THEN** an error is raised identifying the positions of missing values

---

## 2. Output Formats

### A.csv (Loading Matrix)

**WHEN** results are saved to `output_dir`
**THEN** `A.csv` is written as a `p x K` comma-separated file with a header row containing LF indices (0-indexed integers as strings)

**WHEN** `A.csv` is read back
**THEN** the reconstructed matrix has shape `p x K` matching the original within floating-point round-trip precision

### z_matrix.csv (Z Matrix)

**WHEN** results are saved to `output_dir`
**THEN** `z_matrix.csv` is written as an `n x K` comma-separated file with a header row containing LF indices (0-indexed)

**WHEN** `z_matrix.csv` is read back
**THEN** the reconstructed matrix has shape `n x K` and values match to within floating-point round-trip precision

### sig_LFs.txt

**WHEN** results are saved to `output_dir`
**THEN** `sig_LFs.txt` contains one 0-indexed integer per line, each representing a significant marginal latent factor

**WHEN** no marginal LFs are selected
**THEN** `sig_LFs.txt` is written as an empty file (zero bytes or a single newline)

**WHEN** `sig_LFs.txt` is read back
**THEN** the loaded list of integers exactly matches the saved list in the same order

### sig_interacts.txt

**WHEN** results are saved to `output_dir`
**THEN** `sig_interacts.txt` contains one interaction pair per line in the format `i,j` where `i` and `j` are 0-indexed integers

**WHEN** no interactions are detected
**THEN** `sig_interacts.txt` is written as an empty file

**WHEN** `sig_interacts.txt` is read back
**THEN** the loaded list of pairs exactly matches the saved list in the same order

### Parameters JSON

**WHEN** results are saved to `output_dir`
**THEN** a JSON file (e.g., `params.json`) is written containing all parameter values used in the run

**WHEN** the parameters JSON is loaded
**THEN** every parameter that was passed to the pipeline is recoverable, including delta, lambda, seed, response_type, do_interacts, and any algorithm-specific settings

**WHEN** a parameter was not explicitly set by the user
**THEN** its default value is recorded in the JSON so the run is fully reproducible from the file alone

---

## 3. Standardization

### Column-Wise Standardization

**WHEN** `standardize(X)` is called
**THEN** each column has its mean subtracted and is divided by its standard deviation, producing a matrix with column mean=0 and std=1

**WHEN** the standardized matrix is used in the pipeline
**THEN** the original per-column means and standard deviations are stored for later back-transformation

### Constant Columns

**WHEN** a column has zero variance (all values identical)
**THEN** that column is set to all zeros in the standardized matrix without raising an error

**WHEN** constant columns are set to zero
**THEN** a warning is emitted (via the logging system, not print) listing the zero-variance column indices

**WHEN** back-transformation is requested for a zero-variance column
**THEN** the original constant value is restored correctly using the stored mean

### Back-Transformation

**WHEN** `unstandardize(X_std, means, stds)` is called
**THEN** it returns `X_std * stds + means` column-wise, recovering the original scale

**WHEN** any `stds` entry is zero (from a constant column that was zeroed)
**THEN** the corresponding column in the output is set to the stored mean value without a divide-by-zero error

---

## 4. Serialization

### Checkpoint Format

**WHEN** an intermediate pipeline state is saved for checkpointing
**THEN** it is serialized using a binary format (e.g., bincode) that preserves all f64 values exactly

**WHEN** the checkpoint is written
**THEN** the file includes the pipeline stage name and the parameter set so the checkpoint is self-describing

**WHEN** two checkpoints are written for the same stage and parameter set
**THEN** the later checkpoint overwrites the earlier one without creating duplicate files

### Human-Readable Outputs

**WHEN** final results are written
**THEN** matrices (`A.csv`, `z_matrix.csv`) are in CSV format and selections (`sig_LFs.txt`, `sig_interacts.txt`) are in plain text

**WHEN** a human-readable output file already exists in `output_dir`
**THEN** it is overwritten without prompting unless a `--no-overwrite` flag is set

**WHEN** `--no-overwrite` is set and a file already exists
**THEN** an error is raised identifying the conflicting file path before any writes occur

### Resume from Checkpoint

**WHEN** a run is started with `--resume` and a valid checkpoint exists in `output_dir`
**THEN** the pipeline skips all stages completed before the checkpoint and resumes from the next stage

**WHEN** the checkpoint stage does not match any known pipeline stage
**THEN** an error is raised identifying the unrecognized stage name and the checkpoint file path

**WHEN** the checkpoint parameter set does not match the current run parameters
**THEN** an error is raised listing the differing parameters so the user can resolve the conflict

**WHEN** no checkpoint exists and `--resume` is passed
**THEN** the pipeline starts from the beginning without error (resume is a no-op when no checkpoint is found)
