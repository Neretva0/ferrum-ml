use std::io::Write;
use rand::Rng;
use std::rc::Rc;
use std::cell::RefCell;

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


#[derive(Clone)]
struct ModelVar {
    index: usize,
    flags: ModelVarFlags,
    op: ModelVarOp,
    value: Rc<RefCell<Matrix>>,
    gradient: Rc<RefCell<Matrix>>,
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

        // Update the variable in all_vars
        model.all_vars[out.index] = out.clone();

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

        // Update the variable in all_vars
        model.all_vars[out.index] = out.clone();

        Ok(out)
    }
    
}

struct ModelProgram {
    vars: Vec<ModelVar>,
    size: usize
}

struct ModelContext {
    num_vars: usize,
    input_idx: usize,
    output_idx: usize,
    desired_output_idx: usize,
    cost_idx: usize,

    all_vars: Vec<ModelVar>,

    forward_program: ModelProgram,
    cost_program: ModelProgram,
}

struct ModelTrainingDesc {
    train_images: Vec<Matrix>,
    train_labels: Vec<Matrix>,
    test_images: Vec<Matrix>,
    test_labels: Vec<Matrix>,

    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
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

fn mat_fill_random(mat: &mut Matrix, lower: f32, upper: f32) {
    for v in mat.data.iter_mut() {
        *v = rand::random::<f32>() * (upper - lower) + lower;
    }
}

fn mat_scale(mat: &mut Matrix, x: f32) {
    mat.data.iter_mut().for_each(|v| *v *= x);
}

fn mat_sum(mat: &Matrix) -> f32 {
    mat.data.iter().copied().sum()
}

fn mat_argmax(mat: &Matrix) -> usize {
    mat.data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
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
            p_val * -(q_val + EPSILON).ln()
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
            value: Rc::new(RefCell::new(Matrix::new(rows, cols))),
            gradient: Rc::new(RefCell::new(grad)),
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

        model.all_vars.push(out.clone());

        Ok(out)
    }
}

fn mv_relu(
    model: &mut ModelContext,
    input: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    let rows = input.value.borrow().rows;
    let cols = input.value.borrow().cols;
    ModelVar::mv_unary_impl(
        model,
        input,
        rows,
        cols,
        flags,
        ModelVarOp::Relu,
    )
}

fn mv_softmax(
    model: &mut ModelContext,
    input: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    let rows = input.value.borrow().rows;
    let cols = input.value.borrow().cols;
    ModelVar::mv_unary_impl(
        model,
        input,
        rows,
        cols,
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
    let a_rows = a.value.borrow().rows;
    let a_cols = a.value.borrow().cols;
    let b_rows = b.value.borrow().rows;
    let b_cols = b.value.borrow().cols;
    if a_rows != b_rows || a_cols != b_cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mv_add: operand dimensions mismatch".to_string(),
            expected: (a_rows, a_cols),
            got: (b_rows, b_cols),
        });
    }
    ModelVar::mv_binary_impl(
        model,
        a,
        b,
        a_rows,
        a_cols,
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
    let a_rows = a.value.borrow().rows;
    let a_cols = a.value.borrow().cols;
    let b_rows = b.value.borrow().rows;
    let b_cols = b.value.borrow().cols;
    if a_rows != b_rows || a_cols != b_cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mv_sub: operand dimensions mismatch".to_string(),
            expected: (a_rows, a_cols),
            got: (b_rows, b_cols),
        });
    }
    ModelVar::mv_binary_impl(
        model,
        a,
        b,
        a_rows,
        a_cols,
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
    let p_rows = p.value.borrow().rows;
    let p_cols = p.value.borrow().cols;
    let q_rows = q.value.borrow().rows;
    let q_cols = q.value.borrow().cols;
    if p_rows != q_rows || p_cols != q_cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mv_cross_entropy: operand dimensions mismatch".to_string(),
            expected: (p_rows, p_cols),
            got: (q_rows, q_cols),
        });
    }
    ModelVar::mv_binary_impl(
        model,
        p,
        q,
        p_rows,
        p_cols,
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
    let a_rows = a.value.borrow().rows;
    let a_cols = a.value.borrow().cols;
    let b_rows = b.value.borrow().rows;
    let b_cols = b.value.borrow().cols;

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

fn mat_relu_add_grad(
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

fn mat_softmax_add_grad(
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
            jacobian.data[j+i*size] = softmax_out.data[i] * if i == j { 1.0 - softmax_out.data[i] } else { -softmax_out.data[j] };
            }
        }
    mat_mul(out, &jacobian, grad, false, false, false)?;
    Ok(())
}

fn mat_cross_entropy_add_grad(
    p_grad: &mut Matrix,
    q_grad: &mut Matrix,
    p: &Matrix,
    q: &Matrix,
    grad: &Matrix,
) -> Result<(), MatrixError> {
    if p.rows != q.rows || p.cols != q.cols {
        return Err(MatrixError::DimensionMismatch {
            context: "mat_cross_entropy_add_grad: probability and target dimensions mismatch".to_string(),
            expected: (p.rows, p.cols),
            got: (q.rows, q.cols),
        });
    }
    let size = p.rows * p.cols;
    const EPSILON: f32 = 1e-10;
    
    if !p_grad.data.is_empty() {
        if p_grad.rows != p.rows || p_grad.cols != p.cols {
            return Err(MatrixError::DimensionMismatch {
                context: "mat_cross_entropy_add_grad: p_grad and probability dimensions mismatch".to_string(),
                expected: (p.rows, p.cols),
                got: (p_grad.rows, p_grad.cols),
            });
        }
        // Gradient w.r.t. p (target): -log(q) * grad
        for i in 0..size {
            p_grad.data[i] += -grad.data[i] * (q.data[i] + EPSILON).ln();
        }
    }

    if !q_grad.data.is_empty() {
        if q_grad.rows != q.rows || q_grad.cols != q.cols {
            return Err(MatrixError::DimensionMismatch {
                context: "mat_cross_entropy_add_grad: q_grad and target dimensions mismatch".to_string(),
                expected: (q.rows, q.cols),
                got: (q_grad.rows, q_grad.cols),
            });
        }
        // Gradient w.r.t. q (prediction): -p/q * grad
        for i in 0..size {
            q_grad.data[i] += -grad.data[i] * p.data[i] / (q.data[i] + EPSILON);
        }
    }
    Ok(())
}

impl ModelContext {
    /// Creates a program (topologically sorted indices) to compute a target variable.
    pub fn create_program(&self, all_vars: &[ModelVar], out_var_idx: usize) -> ModelProgram {
        let mut program_indices = Vec::new();
        let mut visited = vec![false; self.num_vars];

        // Inner helper function for recursive DFS
        fn visit(
            cur_idx: usize,
            all_vars: &[ModelVar],
            visited: &mut Vec<bool>,
            program_indices: &mut Vec<usize>,
        ) {
            // Guard against out-of-bounds or already processed nodes
            if cur_idx >= visited.len() || visited[cur_idx] {
                return;
            }

            let cur_var = &all_vars[cur_idx];
            
            // 1. Mark as visited BEFORE visiting children to handle cycles (if your graph allows them)
            // or simply to skip redundant work.
            visited[cur_idx] = true;

            // 2. Visit all inputs (dependencies) first
            for input_opt in cur_var.inputs.iter() {
                if let Some(input_idx) = input_opt {
                    visit(*input_idx, all_vars, visited, program_indices);
                }
            }

            // 3. After all dependencies are added to the list, add this node
            program_indices.push(cur_idx);
        }

        visit(out_var_idx, all_vars, &mut visited, &mut program_indices);

        // Create a mapping from old indices to new indices
        let mut index_map = vec![None; self.num_vars];
        for (new_idx, &old_idx) in program_indices.iter().enumerate() {
            index_map[old_idx] = Some(new_idx);
        }

        // Convert the sorted indices into a Program
        let mut vars: Vec<ModelVar> = program_indices
            .iter()
            .map(|&idx| all_vars[idx].clone())
            .collect();
        
        // Remap the input indices
        for var in vars.iter_mut() {
            for input_opt in var.inputs.iter_mut() {
                if let Some(old_input_idx) = input_opt {
                    *input_opt = index_map[*old_input_idx];
                }
            }
        }
        
        let size = vars.len();
        ModelProgram {
            vars,
            size,
        }
    }
}

fn model_prog_compute(
    prog: &mut ModelProgram,
){
    for i in 0..prog.size {
        let op = prog.vars[i].op;
        let a_idx = prog.vars[i].inputs[0];
        let b_idx = prog.vars[i].inputs[1];

        let mut out_val = prog.vars[i].value.borrow_mut();

        match op {
            ModelVarOp::Add => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                let b = prog.vars[b_idx.unwrap()].value.borrow();
                mat_add(&mut out_val, &a, &b).unwrap();
            },
            ModelVarOp::Sub => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                let b = prog.vars[b_idx.unwrap()].value.borrow();
                mat_sub(&mut out_val, &a, &b).unwrap();
            },
            ModelVarOp::MatMul => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                let b = prog.vars[b_idx.unwrap()].value.borrow();
                mat_mul(&mut out_val, &a, &b, true, false, false).unwrap();
            },
            ModelVarOp::Relu => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                mat_relu(&mut out_val, &a).unwrap();
            },
            ModelVarOp::Softmax => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                mat_softmax(&mut out_val, &a).unwrap();
            },
            ModelVarOp::CrossEntropy => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                let b = prog.vars[b_idx.unwrap()].value.borrow();
                mat_cross_entropy(&mut out_val, &a, &b).unwrap();
            },
            _ => {}
        }
    }
        
    
}

fn model_prog_compute_grads(prog: &mut ModelProgram){
    // Clear gradients for non-parameter variables
    for i in 0..prog.size{
        let cur: &mut ModelVar = &mut prog.vars[i];
        if !cur.flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            continue;
        }
        // Skip parameters - they accumulate gradients across the batch
        if cur.flags.contains(ModelVarFlags::PARAMETER) {
            continue;
        }
        mat_clear(&mut cur.gradient.borrow_mut());
    }

    // Initialize output gradient to 1
    mat_fill(&mut prog.vars[prog.size-1].gradient.borrow_mut(), 1.0);

    for i in (0..prog.size).rev() {
        let num_inputs = prog.vars[i].op.num_inputs();
        let op = prog.vars[i].op;
        

        
        if !prog.vars[i].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            continue;
        }

        // Early exit if no inputs require gradients
        if num_inputs >= 1 {
            let a_idx = prog.vars[i].inputs[0];
            if let Some(idx) = a_idx {
                let a_requires_grad = prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD);
                
                if num_inputs == 1 && !a_requires_grad {
                    continue;
                }
                
                if num_inputs == 2 {
                    let b_idx = prog.vars[i].inputs[1];
                    if let Some(b_idx) = b_idx {
                        let b_requires_grad = prog.vars[b_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD);
                        if !a_requires_grad && !b_requires_grad {
                            continue;
                        }
                    }
                }
            }
        }

        // Extract values and gradient before mutating
        let cur_gradient = prog.vars[i].gradient.borrow();
        let a_idx = prog.vars[i].inputs[0];
        let b_idx = prog.vars[i].inputs[1];

        match op {
            ModelVarOp::Add => {
                if let Some(idx) = a_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let mut a_grad = prog.vars[idx].gradient.borrow_mut();
                        // Manually accumulate to avoid double borrow
                        for (a, c) in a_grad.data.iter_mut().zip(cur_gradient.data.iter()) { *a += c; }
                    }
                }
                if let Some(idx) = b_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let mut b_grad = prog.vars[idx].gradient.borrow_mut();
                        for (b, c) in b_grad.data.iter_mut().zip(cur_gradient.data.iter()) { *b += c; }
                    }
                }
            },
            ModelVarOp::Sub => {
                if let Some(idx) = a_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let mut a_grad = prog.vars[idx].gradient.borrow_mut();
                        for (a, c) in a_grad.data.iter_mut().zip(cur_gradient.data.iter()) { *a += c; }
                    }
                }
                if let Some(idx) = b_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let mut b_grad = prog.vars[idx].gradient.borrow_mut();
                        for (b, c) in b_grad.data.iter_mut().zip(cur_gradient.data.iter()) { *b -= c; }
                    }
                }
            },
            ModelVarOp::MatMul => {
                if let (Some(a_idx), Some(b_idx)) = (a_idx, b_idx) {
                    let b_value = prog.vars[b_idx].value.borrow();
                    let a_value = prog.vars[a_idx].value.borrow();
                    
                    if prog.vars[a_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        mat_mul(&mut prog.vars[a_idx].gradient.borrow_mut(), &cur_gradient, &b_value, false, false, true).unwrap();
                    }
                    if prog.vars[b_idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        mat_mul(&mut prog.vars[b_idx].gradient.borrow_mut(), &a_value, &cur_gradient, false, true, false).unwrap();
                    }
                }
            },
            ModelVarOp::CrossEntropy => {
                if let (Some(p_idx), Some(q_idx)) = (a_idx, b_idx) {
                    let p_value = prog.vars[p_idx].value.borrow();
                    let q_value = prog.vars[q_idx].value.borrow();
                    
                    mat_cross_entropy_add_grad(
                        &mut prog.vars[p_idx].gradient.borrow_mut(),
                        &mut prog.vars[q_idx].gradient.borrow_mut(),
                        &p_value,
                        &q_value,
                        &cur_gradient
                    ).unwrap();
                }
            },
            ModelVarOp::Relu => {
                if let Some(idx) = a_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let a_value = prog.vars[idx].value.borrow();
                        mat_relu_add_grad(&mut prog.vars[idx].gradient.borrow_mut(), &a_value, &cur_gradient).unwrap();
                    }
                }
            },
            ModelVarOp::Softmax => {
                if let Some(idx) = a_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let softmax_output = prog.vars[i].value.borrow();
                        mat_softmax_add_grad(&mut prog.vars[idx].gradient.borrow_mut(), &softmax_output, &cur_gradient).unwrap();
                    }
                }
            },
            _ => {}
        }
    }
                
}

impl ModelContext{
    fn create() -> Self {
        ModelContext {
            num_vars: 0,
            input_idx: 0,
            output_idx: 0,
            desired_output_idx: 0,
            cost_idx: 0,
            all_vars: Vec::new(),
            forward_program: ModelProgram { vars: Vec::new(), size: 0 },
            cost_program: ModelProgram { vars: Vec::new(), size: 0 },
        }
    }
}

fn modele_compile(model: &mut ModelContext){
    model.forward_program = model.create_program(&model.all_vars, model.output_idx);
    model.cost_program = model.create_program(&model.all_vars, model.cost_idx);
}

fn model_feedforward(model: &mut ModelContext){
    model_prog_compute(&mut model.forward_program);
}

fn create_mnist_model(model: &mut ModelContext) -> Result<(), MatrixError> {
    let input = ModelVar::create(model, 784, 1, ModelVarFlags::INPUT)?;

    let mut w0 = ModelVar::create(model, 16, 784, ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD)?;
    let mut w1 = ModelVar::create(model, 16, 16, ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD)?;
    let mut w2 = ModelVar::create(model, 10, 16, ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD)?;

    let bound0 = (6.0f32 / (784 + 16) as f32).sqrt();
    let bound1 = (6.0f32 / (16 + 16) as f32).sqrt();
    let bound2 = (6.0f32 / (16 + 10) as f32).sqrt();
    mat_fill_random(&mut w0.value.borrow_mut(), -bound0, bound0);
    mat_fill_random(&mut w1.value.borrow_mut(), -bound1, bound1);
    mat_fill_random(&mut w2.value.borrow_mut(), -bound2, bound2);

    let b0 = ModelVar::create(model, 16, 1, ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD)?;
    let b1 = ModelVar::create(model, 16, 1, ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD)?;
    let b2 = ModelVar::create(model, 10, 1, ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD)?;

    let z0_a = mv_matmul(model, &w0, &input, ModelVarFlags::NONE)?;
    let z0_b = mv_add(model, &z0_a, &b0, ModelVarFlags::NONE)?;
    let a0 = mv_relu(model, &z0_b, ModelVarFlags::NONE)?;

    let z1_a = mv_matmul(model, &w1, &a0, ModelVarFlags::NONE)?;
    let z1_b = mv_add(model, &z1_a, &b1, ModelVarFlags::NONE)?;
    let z1_c = mv_relu(model, &z1_b, ModelVarFlags::NONE)?;
    let a1 = mv_add(model, &z1_c, &a0, ModelVarFlags::NONE)?;

    let z2_a = mv_matmul(model, &w2, &a1, ModelVarFlags::NONE)?;
    let z2_b = mv_add(model, &z2_a, &b2, ModelVarFlags::NONE)?;
    let output = mv_softmax(model, &z2_b, ModelVarFlags::OUTPUT)?;

    let desired_output = ModelVar::create(model, 10, 1, ModelVarFlags::DESIRED_OUTPUT)?;
    let _cost = mv_cross_entropy(model, &desired_output, &output, ModelVarFlags::COST)?;

    Ok(())
}

fn model_train(
    model: &mut ModelContext,
    training_desc: &ModelTrainingDesc,
) -> Result<(), Box<dyn std::error::Error>> {
    let train_images = &training_desc.train_images;
    let train_labels = &training_desc.train_labels;
    let test_images = &training_desc.test_images;
    let test_labels = &training_desc.test_labels;

    let num_examples = train_images.len();
    let input_size = train_images[0].rows * train_images[0].cols;
    let output_size = train_labels[0].rows * train_labels[0].cols;
    let num_tests = test_images.len();

    let num_batches = num_examples / training_desc.batch_size;

    // Find the correct indices in the cost program
    let mut prog_input_idx = None;
    let mut prog_desired_output_idx = None;
    let mut prog_output_idx = None;
    let mut prog_cost_idx = None;
    
    for (i, var) in model.cost_program.vars.iter().enumerate() {
        if var.flags.contains(ModelVarFlags::INPUT) {
            prog_input_idx = Some(i);
        }
        if var.flags.contains(ModelVarFlags::DESIRED_OUTPUT) {
            prog_desired_output_idx = Some(i);
        }
        if var.flags.contains(ModelVarFlags::OUTPUT) {
            prog_output_idx = Some(i);
        }
        if var.flags.contains(ModelVarFlags::COST) {
            prog_cost_idx = Some(i);
        }
    }
    
    let prog_input_idx = prog_input_idx.expect("INPUT variable not found in cost program");
    let prog_desired_output_idx = prog_desired_output_idx.expect("DESIRED_OUTPUT variable not found in cost program");
    let prog_output_idx = prog_output_idx.expect("OUTPUT variable not found in cost program");
    let prog_cost_idx = prog_cost_idx.expect("COST variable not found in cost program");

    let mut training_order: Vec<usize> = (0..num_examples).collect();

    for epoch in 0..training_desc.epochs {
        // Shuffle training order
        let mut rng = rand::rng();
        for _ in 0..num_examples {
            let a = rng.random_range(0..num_examples);
            let b = rng.random_range(0..num_examples);
            training_order.swap(a, b);
        }

        for batch in 0..num_batches {
            // Clear parameter gradients
            for i in 0..model.cost_program.size {
                let cur = &mut model.cost_program.vars[i];
                if cur.flags.contains(ModelVarFlags::PARAMETER) {
                    mat_clear(&mut cur.gradient.borrow_mut());
                }
            }

            let mut avg_cost = 0.0f32;
            for i in 0..training_desc.batch_size {
                let order_index = batch * training_desc.batch_size + i;
                let index = training_order[order_index];

                // Copy input image
                mat_copy(
                    &mut model.cost_program.vars[prog_input_idx].value.borrow_mut(),
                    &train_images[index]
                )?;

                // Copy desired output label
                mat_copy(
                    &mut model.cost_program.vars[prog_desired_output_idx].value.borrow_mut(),
                    &train_labels[index]
                )?;

                model_prog_compute(&mut model.cost_program);
                model_prog_compute_grads(&mut model.cost_program);

                avg_cost += mat_sum(&model.cost_program.vars[prog_cost_idx].value.borrow());
            }
            avg_cost /= training_desc.batch_size as f32;

            // Update parameters
            for i in 0..model.cost_program.size {
                let cur = &mut model.cost_program.vars[i];
                
                if !cur.flags.contains(ModelVarFlags::PARAMETER) {
                    continue;
                }

                let mut grad = cur.gradient.borrow_mut();
                let mut val = cur.value.borrow_mut();

                mat_scale(
                    &mut grad,
                    training_desc.learning_rate / training_desc.batch_size as f32
                );
                
                // cur.value = cur.value - cur.gradient
                for j in 0..val.data.len() {
                    val.data[j] -= grad.data[j];
                }
            }

            print!(
                "\rEpoch {:2} / {:2}, Batch {:4} / {:4}, Average Cost: {:.4}",
                epoch + 1, training_desc.epochs,
                batch + 1, num_batches, avg_cost
            );
            std::io::stdout().flush()?;
        }
        println!();

        // Test accuracy
        let mut num_correct = 0;
        let mut avg_cost = 0.0f32;
        for i in 0..num_tests {
            mat_copy(
                &mut model.cost_program.vars[prog_input_idx].value.borrow_mut(),
                &test_images[i]
            )?;

            mat_copy(
                &mut model.cost_program.vars[prog_desired_output_idx].value.borrow_mut(),
                &test_labels[i]
            )?;

            model_prog_compute(&mut model.cost_program);

            avg_cost += mat_sum(&model.cost_program.vars[prog_cost_idx].value.borrow());
            
            if mat_argmax(&model.cost_program.vars[prog_output_idx].value.borrow()) ==
               mat_argmax(&model.cost_program.vars[prog_desired_output_idx].value.borrow()) {
                num_correct += 1;
            }
        }

        avg_cost /= num_tests as f32;
        println!(
            "Test Completed. Accuracy: {:5} / {:5} ({:.1}%), Average Cost: {:.4}",
            num_correct, num_tests, 
            num_correct as f32 / num_tests as f32 * 100.0,
            avg_cost
        );
    }

    Ok(())
}



fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MNIST dataset...");
    
    // Load MNIST data
    let train_data = mnist::MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .base_path("data/")  // Ensure this directory exists
        .finalize();
    
    let train_images = train_data.trn_img;
    let train_labels = train_data.trn_lbl;
    let test_images = train_data.tst_img;
    let test_labels = train_data.tst_lbl;
    
    println!("Dataset loaded.");
    println!("Preparing data...");
    
    // Convert training data
    let mut train_image_matrices = Vec::new();
    let mut train_label_matrices = Vec::new();
    
    for i in 0..train_labels.len() {
        let mut img_matrix = Matrix::new(784, 1);
        for j in 0..784 {
            img_matrix.data[j] = train_images[i * 784 + j] as f32 / 255.0;
        }
        train_image_matrices.push(img_matrix);
        
        let mut label_matrix = Matrix::new(10, 1);
        label_matrix.data[train_labels[i] as usize] = 1.0;
        train_label_matrices.push(label_matrix);
    }
    
    // Convert test data
    let mut test_image_matrices = Vec::new();
    let mut test_label_matrices = Vec::new();
    
    for i in 0..test_labels.len() {
        let mut img_matrix = Matrix::new(784, 1);
        for j in 0..784 {
            img_matrix.data[j] = test_images[i * 784 + j] as f32 / 255.0;
        }
        test_image_matrices.push(img_matrix);
        
        let mut label_matrix = Matrix::new(10, 1);
        label_matrix.data[test_labels[i] as usize] = 1.0;
        test_label_matrices.push(label_matrix);
    }
    
    println!("Creating model...");
    let mut model = ModelContext::create();
    create_mnist_model(&mut model)?;
    modele_compile(&mut model);
    
    println!("Starting training...\n");
    let training_desc = ModelTrainingDesc {
        train_images: train_image_matrices,
        train_labels: train_label_matrices,
        test_images: test_image_matrices,
        test_labels: test_label_matrices,
        epochs: 5,
        batch_size: 32,
        learning_rate: 0.01,
    };
    
    model_train(&mut model, &training_desc)?;
    
    println!("\nTraining complete!");
    Ok(())
}
