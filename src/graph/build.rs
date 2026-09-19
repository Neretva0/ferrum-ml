use std::cell::RefCell;
use std::rc::Rc;

use crate::matrix::{Matrix, MatrixError};

use super::{MODEL_VAR_MAX_INPUTS, ModelContext, ModelVar, ModelVarFlags, ModelVarOp};

impl ModelVar {
    pub fn create(
        model: &mut ModelContext,
        rows: usize,
        cols: usize,
        flags: ModelVarFlags,
        op: ModelVarOp,
        inputs: [Option<usize>; MODEL_VAR_MAX_INPUTS],
    ) -> Result<Self, MatrixError> {
        let grad = if flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            Matrix::new(rows, cols)
        } else {
            Matrix::new(0, 0)
        };

        let out = ModelVar {
            index: model.num_vars,
            flags,
            op,
            value: Rc::new(RefCell::new(Matrix::new(rows, cols))),
            gradient: Rc::new(RefCell::new(grad)),
            inputs,
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

    pub fn mv_unary_impl(
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
        let out = Self::create(model, rows, cols, flags, op, [Some(input.index), None])?;
        Ok(out)
    }

    pub fn mv_binary_impl(
        model: &mut ModelContext,
        a: &ModelVar,
        b: &ModelVar,
        rows: usize,
        cols: usize,
        mut flags: ModelVarFlags,
        op: ModelVarOp,
    ) -> Result<Self, MatrixError> {
        if a.flags.contains(ModelVarFlags::REQUIRES_GRAD)
            || b.flags.contains(ModelVarFlags::REQUIRES_GRAD)
        {
            flags |= ModelVarFlags::REQUIRES_GRAD;
        }
        let out = Self::create(model, rows, cols, flags, op, [Some(a.index), Some(b.index)])?;
        Ok(out)
    }
}

pub fn mv_relu(
    model: &mut ModelContext,
    input: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    let rows = input.value.borrow().rows;
    let cols = input.value.borrow().cols;
    ModelVar::mv_unary_impl(model, input, rows, cols, flags, ModelVarOp::Relu)
}

pub fn mv_softmax(
    model: &mut ModelContext,
    input: &ModelVar,
    flags: ModelVarFlags,
) -> Result<ModelVar, MatrixError> {
    let rows = input.value.borrow().rows;
    let cols = input.value.borrow().cols;
    ModelVar::mv_unary_impl(model, input, rows, cols, flags, ModelVarOp::Softmax)
}

pub fn mv_add(
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
    ModelVar::mv_binary_impl(model, a, b, a_rows, a_cols, flags, ModelVarOp::Add)
}

pub fn mv_sub(
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
    ModelVar::mv_binary_impl(model, a, b, a_rows, a_cols, flags, ModelVarOp::Sub)
}

pub fn mv_cross_entropy(
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
    ModelVar::mv_binary_impl(model, p, q, p_rows, p_cols, flags, ModelVarOp::CrossEntropy)
}

pub fn mv_matmul(
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

    ModelVar::mv_binary_impl(model, a, b, a_rows, b_cols, flags, ModelVarOp::MatMul)
}

pub fn mv_unary_impl(
    model: &mut ModelContext,
    input: &ModelVar,
    rows: usize,
    cols: usize,
    flags: ModelVarFlags,
    op: ModelVarOp,
) -> Result<ModelVar, MatrixError> {
    ModelVar::mv_unary_impl(model, input, rows, cols, flags, op)
}

pub fn mv_binary_impl(
    model: &mut ModelContext,
    a: &ModelVar,
    b: &ModelVar,
    rows: usize,
    cols: usize,
    flags: ModelVarFlags,
    op: ModelVarOp,
) -> Result<ModelVar, MatrixError> {
    ModelVar::mv_binary_impl(model, a, b, rows, cols, flags, op)
}
