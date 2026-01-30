use bitflags::bitflags;

#[derive(Debug, Clone)]
enum MatrixError {
    DimensionMismatch {
        context: String,
        expected: (usize, usize),
        got: (usize, usize),
    },
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::DimensionMismatch { context, expected, got } => {
                write!(f, "{}: expected {:?}, got {:?}", context, expected, got)
            }
        }
    }
}

impl std::error::Error for MatrixError {}

#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    // Create a new matrix filled with zeros
    fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    // Index helper for flat storage
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy)]
    struct ModelVarFlags: u32 {
        const NONE            = 0;
        const REQUIRES_GRAD   = 1 << 0;
        const PARAMETER       = 1 << 1;
        const INPUT           = 1 << 2;
        const OUTPUT          = 1 << 3;
        const DESIRED_OUTPUT  = 1 << 4;
        const COST            = 1 << 5;
    }
}


#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelVarOp {
    Null = 0,
    Create,
    UnaryStart,
    Relu,
    Softmax,
    BinaryStart,
    Add,
    Sub,
    MatMul,
    CrossEntropy,
}

const MODEL_VAR_MAX_INPUTS: usize = 2;

impl ModelVarOp {
    #[inline]
    fn num_inputs(self) -> usize {
        use ModelVarOp::*;

        match self {
            op if (op as u32) < UnaryStart as u32 => 0,
            op if (op as u32) < BinaryStart as u32 => 1,
            _ => 2,
        }
    }
}


struct ModelVar {
    index: usize,
    flags: ModelVarFlags,
    op: ModelVarOp,
    value: Matrix,
    gradient: Matrix,
    inputs: [Option<usize>; MODEL_VAR_MAX_INPUTS],
}

impl ModelVar {
    fn mv_unary_impl(
        model: &mut ModelContext,
        input: &ModelVar,
        rows: usize,
        cols: usize,
        mut flags: ModelVarFlags,
        op: ModelVarOp,
    ) -> Result<Self, MatrixError> {
        if input.flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            flags |= ModelVarFlags::REQUIRES_GRAD;
        }
        let mut out = Self::create(model, rows, cols, flags)?;

        out.op = op;
        out.inputs[0] = Some(input.index);

        Ok(out)
    }
}

impl ModelVar {
    fn mv_binary_impl(
        model: &mut ModelContext,
        a: &ModelVar,
        b: &ModelVar,
        rows: usize,
        cols: usize,
        mut flags: ModelVarFlags,
        op: ModelVarOp,
    ) -> Result<Self, MatrixError> {
        if a.flags.contains(ModelVarFlags::REQUIRES_GRAD) ||
           b.flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            flags |= ModelVarFlags::REQUIRES_GRAD;
        }
        let mut out = Self::create(model, rows, cols, flags)?;

        out.op = op;
        out.inputs[0] = Some(a.index);
        out.inputs[1] = Some(b.index);

        Ok(out)
    }
    
}

struct ModelProgram {
    vars: Vec<ModelVar>,
}

struct ModelContext {
    num_vars: usize,
    input_idx: usize,
    output_idx: usize,
    desired_output_idx: usize,
    cost_idx: usize,

    forward_program: ModelProgram,
    cost_program: ModelProgram,
}


fn mat_copy(dst: &mut Matrix, src: &Matrix) -> Result<(), MatrixError> {
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

fn mat_clear(mat: &mut Matrix) {
    mat.data.fill(0.0);
}

fn mat_fill(mat: &mut Matrix, x: f32) {
    mat.data.fill(x);
}

fn mat_scale(mat: &mut Matrix, x: f32) {
    mat.data.iter_mut().for_each(|v| *v *= x);
}

fn mat_sum(mat: &Matrix) -> f32 {
    mat.data.iter().copied().sum()
}

fn mat_add(out: &mut Matrix, a: &Matrix, b: &Matrix) -> Result<(), MatrixError> {
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

    out.data.iter_mut().zip(a.data.iter().zip(b.data.iter()))
        .for_each(|(o, (a_val, b_val))| *o = a_val + b_val);
    Ok(())
}

fn mat_sub(out: &mut Matrix, a: &Matrix, b: &Matrix) -> Result<(), MatrixError> {
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

    out.data.iter_mut().zip(a.data.iter().zip(b.data.iter()))
        .for_each(|(o, (a_val, b_val))| *o = a_val - b_val);
    Ok(())
}

// ---------------------- Matrix multiplication ----------------------

fn mat_mul_nn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
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

fn mat_mul_nt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
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

fn mat_mul_tn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
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

fn mat_mul_tt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for k in 0..a.cols {
        for i in 0..out.rows {
            let a_ki = a.data[a.idx(k, i)];
            for j in 0..out.cols {
                let idx = out.idx(i, j);
                out.data[idx] += a_ki * b.data[b.idx(j, k)];
            }
        }
    }
}

fn mat_mul(
    out: &mut Matrix, a: &Matrix, b: &Matrix, 
    zero_out: bool, transpose_a: bool, transpose_b: bool
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
        mat_clear(out);
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

// ---------------------- ReLU ----------------------

fn mat_relu(out: &mut Matrix, input: &Matrix) -> Result<(), MatrixError> {
    if out.rows != input.rows || out.cols != input.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_relu: output and input dimensions mismatch".to_string(),
            expected: (input.rows, input.cols),
            got: (out.rows, out.cols),
        });
    }

    out.data.iter_mut().zip(input.data.iter())
        .for_each(|(o, i)| *o = i.max(0.0));

    Ok(())
}

fn mat_softmax(out: &mut Matrix, input: &Matrix) -> Result<(), MatrixError> {
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

    mat_scale(out, 1.0f32 / sum);
    
    Ok(())
}

fn mat_cross_entropy(out: &mut Matrix, p: &Matrix, q: &Matrix) -> Result<(), MatrixError>{
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
    
    for ((o, &p_val), &q_val) in out.data
        .iter_mut()
        .zip(p.data.iter())
        .zip(q.data.iter())
    {
        *o = if p_val < EPSILON {
            0.0
        } else {
            p_val * -q_val.ln()
        };
    }

    Ok(())
}

impl ModelVar {
    fn create(model: &mut ModelContext, rows: usize, cols: usize, flags: ModelVarFlags) -> Result<Self, MatrixError> {
        let grad = if flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            Matrix::new(rows, cols)
        } else {
            Matrix::new(0, 0)
        };

        let out = ModelVar {
            index: model.num_vars,
            flags,
            op: ModelVarOp::Create,
            value: Matrix::new(rows, cols),
            gradient: grad,
            inputs: [None, None],
        };

        model.num_vars += 1;

        if flags.contains(ModelVarFlags::INPUT) {
            model.input_idx = out.index;
        }
        if flags.contains(ModelVarFlags::OUTPUT) {
            model.output_idx = out.index;
        }
        if flags.contains(ModelVarFlags::DESIRED_OUTPUT) {
            model.desired_output_idx = out.index;
        }
        if flags.contains(ModelVarFlags::COST) {
            model.cost_idx = out.index;
        }

        Ok(out)
    }
}

fn mv_relu(
    model: &mut ModelContext,
    input: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    ModelVar::mv_unary_impl(
        model,
        input,
        input.value.rows,
        input.value.cols,
        flags,
        ModelVarOp::Relu,
    )
}

fn mv_softmax(
    model: &mut ModelContext,
    input: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    ModelVar::mv_unary_impl(
        model,
        input,
        input.value.rows,
        input.value.cols,
        flags,
        ModelVarOp::Softmax,
    )
}

fn mv_add(
    model: &mut ModelContext,
    a: &ModelVar,
    b: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    if a.value.rows != b.value.rows || a.value.cols != b.value.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mv_add: operand dimensions mismatch".to_string(),
            expected: (a.value.rows, a.value.cols),
            got: (b.value.rows, b.value.cols),
        });
    }
    ModelVar::mv_binary_impl(
        model,
        a,
        b,
        a.value.rows,
        a.value.cols,
        flags,
        ModelVarOp::Add,
    )
}

fn mv_sub(
    model: &mut ModelContext,
    a: &ModelVar,
    b: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    if a.value.rows != b.value.rows || a.value.cols != b.value.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mv_sub: operand dimensions mismatch".to_string(),
            expected: (a.value.rows, a.value.cols),
            got: (b.value.rows, b.value.cols),
        });
    }
    ModelVar::mv_binary_impl(
        model,
        a,
        b,
        a.value.rows,
        a.value.cols,
        flags,
        ModelVarOp::Sub,
    )
}

fn mv_cross_entropy(
    model: &mut ModelContext,
    p: &ModelVar,
    q: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    if p.value.rows != q.value.rows || p.value.cols != q.value.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mv_cross_entropy: operand dimensions mismatch".to_string(),
            expected: (p.value.rows, p.value.cols),
            got: (q.value.rows, q.value.cols),
        });
    }
    ModelVar::mv_binary_impl(
        model,
        p,
        q,
        p.value.rows,
        p.value.cols,
        flags,
        ModelVarOp::CrossEntropy,
    )
}

fn mv_matmul(
    model: &mut ModelContext,
    a: &ModelVar,
    b: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    let a_rows = a.value.rows;
    let a_cols = a.value.cols;
    let b_rows = b.value.rows;
    let b_cols = b.value.cols;

    if a_cols != b_rows {
        return Err(MatrixError::DimensionMismatch {
            context: "mv_matmul: inner dimensions mismatch".to_string(),
            expected: (a_cols, a_cols),
            got: (b_rows, b_rows),
        });
    }

    ModelVar::mv_binary_impl(
        model,
        a,
        b,
        a_rows,
        b_cols,
        flags,
        ModelVarOp::MatMul,
    )
}


// ---------------------- Example ----------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Matrix::new(2, 3);
    let mut b = Matrix::new(3, 2);
    let mut out = Matrix::new(2, 2);

    mat_fill(&mut a, 1.0);
    mat_fill(&mut b, 2.0);

    mat_mul(&mut out, &a, &b, true, false, false)?;

    println!("Result: {:?}", out);
    Ok(())
}

