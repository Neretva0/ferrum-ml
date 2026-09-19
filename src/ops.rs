use crate::matmul::mat_mul;
use crate::matrix::{Matrix, MatrixError};

pub fn mat_relu(out: &mut Matrix, input: &Matrix) -> Result<(), MatrixError> {
    if out.rows != input.rows || out.cols != input.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_relu: output and input dimensions mismatch".to_string(),
            expected: (input.rows, input.cols),
            got: (out.rows, out.cols),
        });
    }

    out.data
        .iter_mut()
        .zip(input.data.iter())
        .for_each(|(o, i)| *o = i.max(0.0));

    Ok(())
}

pub fn mat_softmax(out: &mut Matrix, input: &Matrix) -> Result<(), MatrixError> {
    if out.rows != input.rows || out.cols != input.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_softmax: output and input dimensions mismatch".to_string(),
            expected: (input.rows, input.cols),
            got: (out.rows, out.cols),
        });
    }

    let mut sum = 0.0f32;

    for (o, &i) in out.data.iter_mut().zip(input.data.iter()) {
        *o = i.exp();
        sum += *o;
    }

    crate::matrix::mat_scale(out, 1.0f32 / sum);

    Ok(())
}

pub fn mat_cross_entropy(out: &mut Matrix, p: &Matrix, q: &Matrix) -> Result<(), MatrixError> {
    if p.rows != q.rows || p.cols != q.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_cross_entropy: probability and target dimensions mismatch".to_string(),
            expected: (p.rows, p.cols),
            got: (q.rows, q.cols),
        });
    }
    if out.rows != p.rows || out.cols != p.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_cross_entropy: output and input dimensions mismatch".to_string(),
            expected: (p.rows, p.cols),
            got: (out.rows, out.cols),
        });
    }

    const EPSILON: f32 = 1e-10;

    for ((o, &p_val), &q_val) in out.data.iter_mut().zip(p.data.iter()).zip(q.data.iter()) {
        *o = if p_val < EPSILON {
            0.0
        } else {
            p_val * -(q_val + EPSILON).ln()
        };
    }

    Ok(())
}

pub fn mat_relu_add_grad(
    out: &mut Matrix,
    input: &Matrix,
    grad: &Matrix,
) -> Result<(), MatrixError> {
    if out.rows != input.rows || out.cols != input.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_relu_add_grad: output and input dimensions mismatch".to_string(),
            expected: (input.rows, input.cols),
            got: (out.rows, out.cols),
        });
    }
    if out.rows != grad.rows || out.cols != grad.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_relu_add_grad: output and grad dimensions mismatch".to_string(),
            expected: (grad.rows, grad.cols),
            got: (out.rows, out.cols),
        });
    }
    let size = out.rows * out.cols;
    for i in 0..size {
        out.data[i] += grad.data[i] * if input.data[i] > 0.0 { 1.0 } else { 0.0 };
    }
    Ok(())
}

pub fn mat_softmax_add_grad(
    out: &mut Matrix,
    softmax_out: &Matrix,
    grad: &Matrix,
) -> Result<(), MatrixError> {
    if softmax_out.rows != 1 && softmax_out.cols != 1 {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_softmax_add_grad: softmax output must be a vector".to_string(),
            expected: (softmax_out.rows, softmax_out.cols),
            got: (softmax_out.rows, softmax_out.cols),
        });
    }
    let size = softmax_out.rows.max(softmax_out.cols);
    let mut jacobian = Matrix::new(size, size);
    for i in 0..size {
        for j in 0..size {
            jacobian.data[j + i * size] = softmax_out.data[i]
                * if i == j {
                    1.0 - softmax_out.data[i]
                } else {
                    -softmax_out.data[j]
                };
        }
    }
    mat_mul(out, &jacobian, grad, false, false, false)?;
    Ok(())
}

pub fn mat_cross_entropy_add_grad(
    p_grad: &mut Matrix,
    q_grad: &mut Matrix,
    p: &Matrix,
    q: &Matrix,
    grad: &Matrix,
) -> Result<(), MatrixError> {
    if p.rows != q.rows || p.cols != q.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_cross_entropy_add_grad: probability and target dimensions mismatch"
                .to_string(),
            expected: (p.rows, p.cols),
            got: (q.rows, q.cols),
        });
    }
    let size = p.rows * p.cols;
    const EPSILON: f32 = 1e-10;

    if !p_grad.data.is_empty() {
        if p_grad.rows != p.rows || p_grad.cols != p.cols {
            return Err(MatrixError::DimensionMismatch {
                context: "mat_cross_entropy_add_grad: p_grad and probability dimensions mismatch"
                    .to_string(),
                expected: (p.rows, p.cols),
                got: (p_grad.rows, p_grad.cols),
            });
        }
        for i in 0..size {
            p_grad.data[i] += -grad.data[i] * (q.data[i] + EPSILON).ln();
        }
    }

    if !q_grad.data.is_empty() {
        if q_grad.rows != q.rows || q_grad.cols != q.cols {
            return Err(MatrixError::DimensionMismatch {
                context: "mat_cross_entropy_add_grad: q_grad and target dimensions mismatch"
                    .to_string(),
                expected: (q.rows, q.cols),
                got: (q_grad.rows, q_grad.cols),
            });
        }
        for i in 0..size {
            q_grad.data[i] += -grad.data[i] * p.data[i] / (q.data[i] + EPSILON);
        }
    }
    Ok(())
}
