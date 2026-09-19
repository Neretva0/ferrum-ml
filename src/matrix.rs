#[derive(Debug, Clone)]
pub enum MatrixError {
    DimensionMismatch {
        context: String,
        expected: (usize, usize),
        got: (usize, usize),
    },
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::DimensionMismatch {
                context,
                expected,
                got,
            } => {
                write!(f, "{}: expected {:?}, got {:?}", context, expected, got)
            }
        }
    }
}

impl std::error::Error for MatrixError {}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }
}

pub fn mat_copy(dst: &mut Matrix, src: &Matrix) -> Result<(), MatrixError> {
    if dst.rows != src.rows || dst.cols != src.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_copy: destination and source dimensions mismatch".to_string(),
            expected: (dst.rows, dst.cols),
            got: (src.rows, src.cols),
        });
    }
    dst.data.copy_from_slice(&src.data);
    Ok(())
}

pub fn mat_clear(mat: &mut Matrix) {
    mat.data.fill(0.0);
}

pub fn mat_fill(mat: &mut Matrix, x: f32) {
    mat.data.fill(x);
}

pub fn mat_fill_random(mat: &mut Matrix, lower: f32, upper: f32) {
    for v in mat.data.iter_mut() {
        *v = rand::random::<f32>() * (upper - lower) + lower;
    }
}

pub fn mat_scale(mat: &mut Matrix, x: f32) {
    mat.data.iter_mut().for_each(|v| *v *= x);
}

pub fn mat_sum(mat: &Matrix) -> f32 {
    mat.data.iter().copied().sum()
}

pub fn mat_argmax(mat: &Matrix) -> usize {
    mat.data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

pub fn mat_add(out: &mut Matrix, a: &Matrix, b: &Matrix) -> Result<(), MatrixError> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_add: operand dimensions mismatch".to_string(),
            expected: (a.rows, a.cols),
            got: (b.rows, b.cols),
        });
    }
    if out.rows != a.rows || out.cols != a.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_add: output dimensions mismatch".to_string(),
            expected: (a.rows, a.cols),
            got: (out.rows, out.cols),
        });
    }

    out.data
        .iter_mut()
        .zip(a.data.iter().zip(b.data.iter()))
        .for_each(|(o, (a_val, b_val))| *o = a_val + b_val);
    Ok(())
}

pub fn mat_sub(out: &mut Matrix, a: &Matrix, b: &Matrix) -> Result<(), MatrixError> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_sub: operand dimensions mismatch".to_string(),
            expected: (a.rows, a.cols),
            got: (b.rows, b.cols),
        });
    }
    if out.rows != a.rows || out.cols != a.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_sub: output dimensions mismatch".to_string(),
            expected: (a.rows, a.cols),
            got: (out.rows, out.cols),
        });
    }

    out.data
        .iter_mut()
        .zip(a.data.iter().zip(b.data.iter()))
        .for_each(|(o, (a_val, b_val))| *o = a_val - b_val);
    Ok(())
}

pub fn mat_add_assign(a: &mut Matrix, b: &Matrix) -> Result<(), MatrixError> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_add_assign: operand dimensions mismatch".to_string(),
            expected: (a.rows, a.cols),
            got: (b.rows, b.cols),
        });
    }

    for (a_val, b_val) in a.data.iter_mut().zip(b.data.iter()) {
        *a_val += *b_val;
    }
    Ok(())
}
