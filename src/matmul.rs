use crate::matrix::{Matrix, MatrixError};

pub fn mat_mul_nn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.rows {
        for k in 0..a.cols {
            let a_ik = a.data[a.idx(i, k)];
            for j in 0..out.cols {
                let idx = out.idx(i, j);
                out.data[idx] += a_ik * b.data[b.idx(k, j)];
            }
        }
    }
}

pub fn mat_mul_nt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.rows {
        for j in 0..out.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.data[a.idx(i, k)] * b.data[b.idx(j, k)];
            }
            let idx = out.idx(i, j);
            out.data[idx] += sum;
        }
    }
}

pub fn mat_mul_tn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for k in 0..a.rows {
        for i in 0..out.rows {
            let a_ki = a.data[a.idx(k, i)];
            for j in 0..out.cols {
                let idx = out.idx(i, j);
                out.data[idx] += a_ki * b.data[b.idx(k, j)];
            }
        }
    }
}

pub fn mat_mul_tt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for k in 0..a.rows {
        for i in 0..out.rows {
            let a_ki = a.data[a.idx(k, i)];
            for j in 0..out.cols {
                let idx = out.idx(i, j);
                out.data[idx] += a_ki * b.data[b.idx(j, k)];
            }
        }
    }
}

pub fn mat_mul(
    out: &mut Matrix,
    a: &Matrix,
    b: &Matrix,
    zero_out: bool,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<(), MatrixError> {
    let a_rows = if transpose_a { a.cols } else { a.rows };
    let a_cols = if transpose_a { a.rows } else { a.cols };
    let b_rows = if transpose_b { b.cols } else { b.rows };
    let b_cols = if transpose_b { b.rows } else { b.cols };

    if a_cols != b_rows {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_mul: inner dimensions mismatch".to_string(),
            expected: (a_cols, a_cols),
            got: (b_rows, b_rows),
        });
    }
    if out.rows != a_rows || out.cols != b_cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_mul: output dimensions mismatch".to_string(),
            expected: (a_rows, b_cols),
            got: (out.rows, out.cols),
        });
    }

    if zero_out {
        crate::matrix::mat_clear(out);
    }

    let transpose = ((transpose_a as u32) << 1) | (transpose_b as u32);

    match transpose {
        0b00 => mat_mul_nn(out, a, b),
        0b01 => mat_mul_nt(out, a, b),
        0b10 => mat_mul_tn(out, a, b),
        0b11 => mat_mul_tt(out, a, b),
        _ => unreachable!(),
    }

    Ok(())
}
