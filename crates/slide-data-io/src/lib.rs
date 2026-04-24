#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use csv::{ReaderBuilder, WriterBuilder};
use log::warn;
use ndarray::{Array1, Array2};
use serde::Serialize;
use slide_types::{Matrix, Scalar, Vector};
use thiserror::Error;

const MISSING_TOKENS: [&str; 3] = ["", "na", "nan"];
const STANDARDIZATION_TOLERANCE: Scalar = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delimiter {
    Comma,
    Tab,
}

impl Delimiter {
    #[must_use]
    pub fn as_byte(self) -> u8 {
        match self {
            Self::Comma => b',',
            Self::Tab => b'\t',
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseType {
    Continuous,
    Binary,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MatrixReadOptions {
    pub delimiter: Option<Delimiter>,
    pub impute_missing: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResponseReadOptions {
    pub delimiter: Option<Delimiter>,
    pub response_type: Option<ResponseType>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedMatrix {
    pub matrix: Matrix,
    pub feature_names: Vec<String>,
    pub sample_ids: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StandardizedMatrix {
    pub matrix: Matrix,
    pub means: Vector,
    pub stds: Vector,
    pub zero_variance_columns: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputWriteOptions {
    pub no_overwrite: bool,
}

impl Default for OutputWriteOptions {
    fn default() -> Self {
        Self {
            no_overwrite: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum DataIoError {
    #[error("unsupported delimiter for '{path}'; expected .csv, .tsv, or .txt unless a delimiter is provided")]
    UnsupportedDelimiter { path: PathBuf },
    #[error("matrix input '{path}' is missing a header row")]
    MissingHeader { path: PathBuf },
    #[error("matrix input '{path}' has no feature columns after removing the sample-id column")]
    NoFeatureColumns { path: PathBuf },
    #[error(
        "non-numeric value in '{path}' at row {row}, column '{column}'{sample_id_suffix}: '{value}'"
    )]
    InvalidNumericValue {
        path: PathBuf,
        row: usize,
        column: String,
        sample_id_suffix: String,
        value: String,
    },
    #[error("cannot impute column '{column}' in '{path}' because all values are missing or NaN")]
    CannotImputeAllMissing { path: PathBuf, column: String },
    #[error("response input '{path}' must contain exactly one numeric column, found {columns}")]
    InvalidResponseWidth { path: PathBuf, columns: usize },
    #[error("binary response in '{path}' must contain only 0 or 1, found {value} at row {row}")]
    InvalidBinaryResponse {
        path: PathBuf,
        row: usize,
        value: Scalar,
    },
    #[error("response input '{path}' contains a missing value at row {row}")]
    MissingResponseValue { path: PathBuf, row: usize },
    #[error("row count mismatch: X has {matrix_rows} rows but y has {response_len} values")]
    RowCountMismatch {
        matrix_rows: usize,
        response_len: usize,
    },
    #[error(
        "cannot unstandardize matrix with shape {rows}x{cols}; expected {mean_len} means and {std_len} stds"
    )]
    UnstandardizeShapeMismatch {
        rows: usize,
        cols: usize,
        mean_len: usize,
        std_len: usize,
    },
    #[error("refusing to overwrite existing file '{path}' because no_overwrite is set")]
    OutputExists { path: PathBuf },
    #[error("interaction entry '{line}' must have the form 'i,j'")]
    InvalidInteractionLine { line: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

pub fn read_feature_matrix(
    path: impl AsRef<Path>,
    options: &MatrixReadOptions,
) -> Result<LoadedMatrix, DataIoError> {
    let path = path.as_ref();
    let delimiter = options
        .delimiter
        .or_else(|| infer_delimiter(path))
        .ok_or_else(|| DataIoError::UnsupportedDelimiter {
            path: path.to_path_buf(),
        })?;
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter.as_byte())
        .has_headers(true)
        .from_path(path)?;
    let headers = reader.headers()?.clone();
    if headers.is_empty() {
        return Err(DataIoError::MissingHeader {
            path: path.to_path_buf(),
        });
    }

    let records = reader.records().collect::<Result<Vec<_>, _>>()?;
    let has_sample_ids = first_column_is_sample_ids(&records);
    let column_offset = usize::from(has_sample_ids);
    if headers.len() <= column_offset {
        return Err(DataIoError::NoFeatureColumns {
            path: path.to_path_buf(),
        });
    }

    let feature_names = headers
        .iter()
        .skip(column_offset)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let mut sample_ids = has_sample_ids.then(Vec::new);
    let mut matrix_data = Vec::with_capacity(records.len() * feature_names.len());
    let mut missing_by_column = vec![Vec::new(); feature_names.len()];
    let mut column_sums = vec![0.0; feature_names.len()];
    let mut column_counts = vec![0usize; feature_names.len()];

    for (row_index, record) in records.iter().enumerate() {
        if let Some(ids) = &mut sample_ids {
            ids.push(record.get(0).unwrap_or_default().to_owned());
        }

        for (feature_index, column_name) in feature_names.iter().enumerate() {
            let raw_value = record
                .get(feature_index + column_offset)
                .unwrap_or_default();
            match parse_numeric_cell(raw_value) {
                Some(value) => {
                    matrix_data.push(value);
                    column_sums[feature_index] += value;
                    column_counts[feature_index] += 1;
                }
                None if options.impute_missing => {
                    matrix_data.push(0.0);
                    missing_by_column[feature_index].push(row_index);
                }
                None => {
                    return Err(DataIoError::InvalidNumericValue {
                        path: path.to_path_buf(),
                        row: row_index + 1,
                        column: column_name.clone(),
                        sample_id_suffix: sample_ids
                            .as_ref()
                            .and_then(|ids| ids.get(row_index))
                            .map_or_else(String::new, |id| format!(" (sample '{id}')")),
                        value: raw_value.to_owned(),
                    });
                }
            }
        }
    }

    if options.impute_missing {
        for (feature_index, missing_rows) in missing_by_column.iter().enumerate() {
            if missing_rows.is_empty() {
                continue;
            }
            if column_counts[feature_index] == 0 {
                return Err(DataIoError::CannotImputeAllMissing {
                    path: path.to_path_buf(),
                    column: feature_names[feature_index].clone(),
                });
            }
            let mean = column_sums[feature_index] / column_counts[feature_index] as Scalar;
            for row_index in missing_rows {
                let flat_index = row_index * feature_names.len() + feature_index;
                matrix_data[flat_index] = mean;
            }
        }
    }

    let matrix = Array2::from_shape_vec((records.len(), feature_names.len()), matrix_data)
        .unwrap_or_else(|_| unreachable!("validated matrix data must match its shape"));

    Ok(LoadedMatrix {
        matrix: Matrix::from(matrix),
        feature_names,
        sample_ids,
    })
}

pub fn read_response_vector(
    path: impl AsRef<Path>,
    options: &ResponseReadOptions,
) -> Result<Vector, DataIoError> {
    let path = path.as_ref();
    let delimiter = options
        .delimiter
        .or_else(|| infer_delimiter(path))
        .ok_or_else(|| DataIoError::UnsupportedDelimiter {
            path: path.to_path_buf(),
        })?;
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter.as_byte())
        .has_headers(true)
        .from_path(path)?;
    let headers = reader.headers()?.clone();
    let records = reader.records().collect::<Result<Vec<_>, _>>()?;

    let width = if records.is_empty() {
        headers.len()
    } else {
        records
            .iter()
            .map(csv::StringRecord::len)
            .max()
            .unwrap_or(0)
    };

    if !(1..=2).contains(&width) {
        return Err(DataIoError::InvalidResponseWidth {
            path: path.to_path_buf(),
            columns: width,
        });
    }

    let has_sample_ids = width == 2 && first_column_is_sample_ids(&records);
    let value_index = usize::from(has_sample_ids);
    if width - value_index != 1 {
        return Err(DataIoError::InvalidResponseWidth {
            path: path.to_path_buf(),
            columns: width,
        });
    }

    let mut values = Vec::with_capacity(records.len());
    for (row_index, record) in records.iter().enumerate() {
        let raw_value = record.get(value_index).unwrap_or_default();
        let value =
            parse_numeric_cell(raw_value).ok_or_else(|| DataIoError::MissingResponseValue {
                path: path.to_path_buf(),
                row: row_index + 1,
            })?;

        if value.is_nan() {
            return Err(DataIoError::MissingResponseValue {
                path: path.to_path_buf(),
                row: row_index + 1,
            });
        }

        if matches!(options.response_type, Some(ResponseType::Binary))
            && !matches!(value, 0.0 | 1.0)
        {
            return Err(DataIoError::InvalidBinaryResponse {
                path: path.to_path_buf(),
                row: row_index + 1,
                value,
            });
        }

        values.push(value);
    }

    Ok(Vector::from(Array1::from_vec(values)))
}

pub fn validate_row_counts(matrix: &Matrix, response: &Vector) -> Result<(), DataIoError> {
    if matrix.nrows() != response.len() {
        return Err(DataIoError::RowCountMismatch {
            matrix_rows: matrix.nrows(),
            response_len: response.len(),
        });
    }
    Ok(())
}

pub fn standardize(matrix: &Matrix) -> StandardizedMatrix {
    let (rows, cols) = matrix.shape();
    let source = matrix.as_ndarray();
    let mut standardized = Array2::zeros((rows, cols));
    let mut means = Vec::with_capacity(cols);
    let mut stds = Vec::with_capacity(cols);
    let mut zero_variance_columns = Vec::new();

    for column_index in 0..cols {
        let column = source.column(column_index);
        let mean = column.iter().copied().sum::<Scalar>() / rows as Scalar;
        let variance = if rows > 1 {
            column
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<Scalar>()
                / (rows - 1) as Scalar
        } else {
            0.0
        };
        let std = variance.sqrt();

        means.push(mean);
        stds.push(std);

        if std <= STANDARDIZATION_TOLERANCE {
            zero_variance_columns.push(column_index);
            continue;
        }

        for row_index in 0..rows {
            standardized[(row_index, column_index)] =
                (source[(row_index, column_index)] - mean) / std;
        }
    }

    if !zero_variance_columns.is_empty() {
        warn!(
            "standardize encountered zero-variance columns: {:?}",
            zero_variance_columns
        );
    }

    StandardizedMatrix {
        matrix: Matrix::from(standardized),
        means: Vector::from(Array1::from_vec(means)),
        stds: Vector::from(Array1::from_vec(stds)),
        zero_variance_columns,
    }
}

pub fn unstandardize(
    standardized: &Matrix,
    means: &Vector,
    stds: &Vector,
) -> Result<Matrix, DataIoError> {
    let (rows, cols) = standardized.shape();
    if means.len() != cols || stds.len() != cols {
        return Err(DataIoError::UnstandardizeShapeMismatch {
            rows,
            cols,
            mean_len: means.len(),
            std_len: stds.len(),
        });
    }

    let mut restored = Array2::zeros((rows, cols));
    for column_index in 0..cols {
        let mean = means.as_ndarray()[column_index];
        let std = stds.as_ndarray()[column_index];
        if std <= STANDARDIZATION_TOLERANCE {
            for row_index in 0..rows {
                restored[(row_index, column_index)] = mean;
            }
            continue;
        }

        for row_index in 0..rows {
            restored[(row_index, column_index)] =
                standardized.as_ndarray()[(row_index, column_index)] * std + mean;
        }
    }

    Ok(Matrix::from(restored))
}

pub fn write_named_matrix_csv(
    path: impl AsRef<Path>,
    matrix: &Matrix,
    header_names: &[String],
    options: &OutputWriteOptions,
) -> Result<(), DataIoError> {
    let path = path.as_ref();
    ensure_writable(path, options)?;
    if header_names.len() != matrix.ncols() {
        return Err(DataIoError::InvalidResponseWidth {
            path: path.to_path_buf(),
            columns: header_names.len(),
        });
    }

    let mut writer = WriterBuilder::new().from_path(path)?;
    writer.write_record(header_names)?;
    for row in matrix.as_ndarray().rows() {
        let record = row.iter().map(ToString::to_string).collect::<Vec<_>>();
        writer.write_record(&record)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_indexed_matrix_csv(
    path: impl AsRef<Path>,
    matrix: &Matrix,
    options: &OutputWriteOptions,
) -> Result<(), DataIoError> {
    let headers = (0..matrix.ncols())
        .map(|index| index.to_string())
        .collect::<Vec<_>>();
    write_named_matrix_csv(path, matrix, &headers, options)
}

pub fn write_sig_lfs(
    path: impl AsRef<Path>,
    indices: &[usize],
    options: &OutputWriteOptions,
) -> Result<(), DataIoError> {
    let path = path.as_ref();
    ensure_writable(path, options)?;
    let mut file = File::create(path)?;
    for index in indices {
        writeln!(file, "{index}")?;
    }
    Ok(())
}

pub fn read_sig_lfs(path: impl AsRef<Path>) -> Result<Vec<usize>, DataIoError> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        values.push(
            trimmed
                .parse::<usize>()
                .map_err(|_| DataIoError::InvalidInteractionLine { line: line.clone() })?,
        );
    }
    Ok(values)
}

pub fn write_sig_interactions(
    path: impl AsRef<Path>,
    interactions: &[(usize, usize)],
    options: &OutputWriteOptions,
) -> Result<(), DataIoError> {
    let path = path.as_ref();
    ensure_writable(path, options)?;
    let mut file = File::create(path)?;
    for (left, right) in interactions {
        writeln!(file, "{left},{right}")?;
    }
    Ok(())
}

pub fn read_sig_interactions(path: impl AsRef<Path>) -> Result<Vec<(usize, usize)>, DataIoError> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split(',');
        let left = parts
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| DataIoError::InvalidInteractionLine { line: line.clone() })?;
        let right = parts
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| DataIoError::InvalidInteractionLine { line: line.clone() })?;
        if parts.next().is_some() {
            return Err(DataIoError::InvalidInteractionLine { line });
        }
        pairs.push((left, right));
    }
    Ok(pairs)
}

pub fn write_params_json<T: Serialize>(
    path: impl AsRef<Path>,
    params: &T,
    options: &OutputWriteOptions,
) -> Result<(), DataIoError> {
    let path = path.as_ref();
    ensure_writable(path, options)?;
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, params)?;
    Ok(())
}

fn infer_delimiter(path: &Path) -> Option<Delimiter> {
    match path.extension().and_then(OsStr::to_str) {
        Some("csv") => Some(Delimiter::Comma),
        Some("tsv" | "txt") => Some(Delimiter::Tab),
        _ => None,
    }
}

fn first_column_is_sample_ids(records: &[csv::StringRecord]) -> bool {
    if records.is_empty() {
        return false;
    }
    records.iter().any(|record| {
        record
            .get(0)
            .is_some_and(|value| !value.trim().is_empty() && parse_numeric_cell(value).is_none())
    })
}

fn parse_numeric_cell(raw: &str) -> Option<Scalar> {
    let trimmed = raw.trim();
    if MISSING_TOKENS
        .iter()
        .any(|token| trimmed.eq_ignore_ascii_case(token))
    {
        return None;
    }

    trimmed
        .parse::<Scalar>()
        .ok()
        .filter(|value| !value.is_nan())
}

fn ensure_writable(path: &Path, options: &OutputWriteOptions) -> Result<(), DataIoError> {
    if options.no_overwrite && path.exists() {
        return Err(DataIoError::OutputExists {
            path: path.to_path_buf(),
        });
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        read_feature_matrix, read_response_vector, read_sig_interactions, read_sig_lfs,
        standardize, unstandardize, validate_row_counts, write_indexed_matrix_csv,
        write_params_json, write_sig_interactions, write_sig_lfs, DataIoError, Delimiter,
        MatrixReadOptions, OutputWriteOptions, ResponseReadOptions, ResponseType,
    };
    use ndarray::{arr1, arr2};
    use serde::Serialize;
    use slide_types::{Matrix, Vector};
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    const TOLERANCE: f64 = 1.0e-10;

    #[test]
    fn reads_csv_with_sample_ids() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let path = dir.path().join("x.csv");
        fs::write(&path, "id,f1,f2\ns1,1.0,2.0\ns2,3.0,4.0\n")
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        let loaded = read_feature_matrix(&path, &MatrixReadOptions::default())
            .unwrap_or_else(|error| panic!("read should succeed: {error}"));

        assert_eq!(loaded.feature_names, vec!["f1", "f2"]);
        assert_eq!(loaded.sample_ids, Some(vec!["s1".into(), "s2".into()]));
        assert_eq!(loaded.matrix.as_ndarray(), &arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    }

    #[test]
    fn keeps_numeric_first_column_as_feature() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let path = dir.path().join("x.csv");
        fs::write(&path, "f0,f1\n1.0,2.0\n3.0,4.0\n")
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        let loaded = read_feature_matrix(&path, &MatrixReadOptions::default())
            .unwrap_or_else(|error| panic!("read should succeed: {error}"));

        assert_eq!(loaded.feature_names, vec!["f0", "f1"]);
        assert_eq!(loaded.sample_ids, None);
        assert_eq!(loaded.matrix.as_ndarray(), &arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    }

    #[test]
    fn infers_tab_delimiter_for_tsv() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let path = dir.path().join("x.tsv");
        fs::write(&path, "id\tf1\tf2\ns1\t1\t2\ns2\t3\t4\n")
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        let loaded = read_feature_matrix(&path, &MatrixReadOptions::default())
            .unwrap_or_else(|error| panic!("read should succeed: {error}"));

        assert_eq!(loaded.matrix.as_ndarray(), &arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    }

    #[test]
    fn imputes_missing_feature_values_with_column_mean() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let path = dir.path().join("x.csv");
        fs::write(&path, "id,f1,f2\ns1,1.0,\ns2,3.0,5.0\ns3,5.0,7.0\n")
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        let loaded = read_feature_matrix(
            &path,
            &MatrixReadOptions {
                delimiter: Some(Delimiter::Comma),
                impute_missing: true,
            },
        )
        .unwrap_or_else(|error| panic!("read should succeed: {error}"));

        assert_eq!(
            loaded.matrix.as_ndarray(),
            &arr2(&[[1.0, 6.0], [3.0, 5.0], [5.0, 7.0]])
        );
    }

    #[test]
    fn reports_non_numeric_feature_location() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let path = dir.path().join("x.csv");
        fs::write(&path, "id,f1,f2\ns1,1.0,2.0\ns2,oops,4.0\n")
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        let error = read_feature_matrix(&path, &MatrixReadOptions::default())
            .err()
            .unwrap_or_else(|| panic!("read should fail"));

        match error {
            DataIoError::InvalidNumericValue {
                row,
                column,
                sample_id_suffix,
                value,
                ..
            } => {
                assert_eq!(row, 2);
                assert_eq!(column, "f1");
                assert_eq!(sample_id_suffix, " (sample 's2')");
                assert_eq!(value, "oops");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn reads_binary_response_with_sample_ids() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let path = dir.path().join("y.csv");
        fs::write(&path, "id,y\ns1,0\ns2,1\n")
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        let response = read_response_vector(
            &path,
            &ResponseReadOptions {
                delimiter: None,
                response_type: Some(ResponseType::Binary),
            },
        )
        .unwrap_or_else(|error| panic!("response read should succeed: {error}"));

        assert_eq!(response.as_ndarray(), &arr1(&[0.0, 1.0]));
    }

    #[test]
    fn rejects_non_binary_response_values() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let path = dir.path().join("y.csv");
        fs::write(&path, "y\n0\n2\n")
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        let error = read_response_vector(
            &path,
            &ResponseReadOptions {
                delimiter: None,
                response_type: Some(ResponseType::Binary),
            },
        )
        .err()
        .unwrap_or_else(|| panic!("response read should fail"));

        match error {
            DataIoError::InvalidBinaryResponse { row, value, .. } => {
                assert_eq!(row, 2);
                assert_eq!(value, 2.0);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn validates_matching_row_counts() {
        let matrix = Matrix::from(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let response = Vector::from(arr1(&[1.0]));

        let error = validate_row_counts(&matrix, &response)
            .err()
            .unwrap_or_else(|| panic!("validation should fail"));

        match error {
            DataIoError::RowCountMismatch {
                matrix_rows,
                response_len,
            } => {
                assert_eq!(matrix_rows, 2);
                assert_eq!(response_len, 1);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn standardize_zeroes_constant_columns_and_roundtrips() {
        let matrix = Matrix::from(arr2(&[[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]]));

        let standardized = standardize(&matrix);

        assert_eq!(standardized.zero_variance_columns, vec![1]);
        assert_approx_eq(
            standardized.matrix.as_ndarray(),
            &arr2(&[[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        );

        let restored = unstandardize(
            &standardized.matrix,
            &standardized.means,
            &standardized.stds,
        )
        .unwrap_or_else(|error| panic!("unstandardize should succeed: {error}"));

        assert_approx_eq(restored.as_ndarray(), matrix.as_ndarray());
    }

    #[test]
    fn writes_and_reads_selection_files() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let lfs_path = dir.path().join("sig_LFs.txt");
        let interactions_path = dir.path().join("sig_interacts.txt");

        write_sig_lfs(&lfs_path, &[1, 4, 9], &OutputWriteOptions::default())
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));
        write_sig_interactions(
            &interactions_path,
            &[(0, 3), (2, 5)],
            &OutputWriteOptions::default(),
        )
        .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        assert_eq!(
            read_sig_lfs(&lfs_path).unwrap_or_else(|error| panic!("{error}")),
            vec![1, 4, 9]
        );
        assert_eq!(
            read_sig_interactions(&interactions_path).unwrap_or_else(|error| panic!("{error}")),
            vec![(0, 3), (2, 5)]
        );
    }

    #[test]
    fn respects_no_overwrite_for_outputs() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let path = dir.path().join("A.csv");
        fs::write(&path, "existing\n")
            .unwrap_or_else(|error| panic!("write should succeed: {error}"));

        let error = write_indexed_matrix_csv(
            &path,
            &Matrix::from(arr2(&[[1.0, 2.0]])),
            &OutputWriteOptions { no_overwrite: true },
        )
        .err()
        .unwrap_or_else(|| panic!("write should fail"));

        match error {
            DataIoError::OutputExists {
                path: conflict_path,
            } => {
                assert_eq!(conflict_path, path);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn writes_indexed_matrix_and_json_params() {
        #[derive(Serialize)]
        struct Params {
            delta: f64,
            response_type: &'static str,
        }

        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let matrix_path = dir.path().join("z_matrix.csv");
        let params_path = dir.path().join("params.json");

        write_indexed_matrix_csv(
            &matrix_path,
            &Matrix::from(arr2(&[[1.5, 2.5], [3.5, 4.5]])),
            &OutputWriteOptions::default(),
        )
        .unwrap_or_else(|error| panic!("matrix write should succeed: {error}"));
        write_params_json(
            &params_path,
            &Params {
                delta: 0.2,
                response_type: "continuous",
            },
            &OutputWriteOptions::default(),
        )
        .unwrap_or_else(|error| panic!("json write should succeed: {error}"));

        let matrix_text = fs::read_to_string(&matrix_path)
            .unwrap_or_else(|error| panic!("read should succeed: {error}"));
        let params_text = fs::read_to_string(&params_path)
            .unwrap_or_else(|error| panic!("read should succeed: {error}"));

        assert!(matrix_text.starts_with("0,1\n"));
        assert!(params_text.contains("\"delta\": 0.2"));
        assert!(params_text.contains("\"response_type\": \"continuous\""));
    }

    fn assert_approx_eq(left: &ndarray::Array2<f64>, right: &ndarray::Array2<f64>) {
        assert_eq!(left.dim(), right.dim());
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            assert!((lhs - rhs).abs() <= TOLERANCE, "left={lhs}, right={rhs}");
        }
    }

    #[allow(dead_code)]
    fn _read(path: &Path) -> String {
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read should succeed: {error}"))
    }
}
