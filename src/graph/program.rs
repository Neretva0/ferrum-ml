use crate::matmul::mat_mul;
use crate::matrix::{MatrixError, mat_add, mat_clear, mat_fill};

use super::{ModelContext, ModelProgram, ModelVar, ModelVarFlags, ModelVarOp};

impl ModelProgram {
    pub fn find_by_flag(&self, flag: ModelVarFlags) -> Option<usize> {
        self.vars.iter().position(|var| var.flags.contains(flag))
    }
}

impl ModelContext {
    pub fn create_program(&self, all_vars: &[ModelVar], out_var_idx: usize) -> ModelProgram {
        let mut program_indices = Vec::new();
        let mut visited = vec![false; self.num_vars];

        fn visit(
            cur_idx: usize,
            all_vars: &[ModelVar],
            visited: &mut Vec<bool>,
            program_indices: &mut Vec<usize>,
        ) {
            if cur_idx >= visited.len() || visited[cur_idx] {
                return;
            }

            let cur_var = &all_vars[cur_idx];
            visited[cur_idx] = true;

            for input_opt in cur_var.inputs.iter() {
                if let Some(input_idx) = input_opt {
                    visit(*input_idx, all_vars, visited, program_indices);
                }
            }

            program_indices.push(cur_idx);
        }

        visit(out_var_idx, all_vars, &mut visited, &mut program_indices);

        let mut index_map = vec![None; self.num_vars];
        for (new_idx, &old_idx) in program_indices.iter().enumerate() {
            index_map[old_idx] = Some(new_idx);
        }

        let mut vars: Vec<ModelVar> = program_indices
            .iter()
            .map(|&idx| all_vars[idx].clone())
            .collect();

        for var in vars.iter_mut() {
            for input_opt in var.inputs.iter_mut() {
                if let Some(old_input_idx) = input_opt {
                    *input_opt = index_map[*old_input_idx];
                }
            }
        }

        let size = vars.len();
        ModelProgram { vars, size }
    }
}

pub fn model_prog_compute(prog: &mut ModelProgram) -> Result<(), MatrixError> {
    for i in 0..prog.size {
        let op = prog.vars[i].op;
        let a_idx = prog.vars[i].inputs[0];
        let b_idx = prog.vars[i].inputs[1];

        let mut out_val = prog.vars[i].value.borrow_mut();

        match op {
            ModelVarOp::Add => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                let b = prog.vars[b_idx.unwrap()].value.borrow();
                mat_add(&mut out_val, &a, &b)?;
            }
            ModelVarOp::Sub => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                let b = prog.vars[b_idx.unwrap()].value.borrow();
                crate::matrix::mat_sub(&mut out_val, &a, &b)?;
            }
            ModelVarOp::MatMul => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                let b = prog.vars[b_idx.unwrap()].value.borrow();
                mat_mul(&mut out_val, &a, &b, true, false, false)?;
            }
            ModelVarOp::Relu => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                crate::ops::mat_relu(&mut out_val, &a)?;
            }
            ModelVarOp::Softmax => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                crate::ops::mat_softmax(&mut out_val, &a)?;
            }
            ModelVarOp::CrossEntropy => {
                let a = prog.vars[a_idx.unwrap()].value.borrow();
                let b = prog.vars[b_idx.unwrap()].value.borrow();
                crate::ops::mat_cross_entropy(&mut out_val, &a, &b)?;
            }
            _ => {}
        }
    }

    Ok(())
}

pub fn model_prog_compute_grads(prog: &mut ModelProgram) -> Result<(), MatrixError> {
    for i in 0..prog.size {
        let cur = &mut prog.vars[i];
        if !cur.flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            continue;
        }
        if cur.flags.contains(ModelVarFlags::PARAMETER) {
            continue;
        }
        mat_clear(&mut cur.gradient.borrow_mut());
    }

    let cost_index = prog
        .vars
        .iter()
        .position(|var| var.flags.contains(ModelVarFlags::COST))
        .ok_or_else(|| MatrixError::DimensionMismatch {
            context: "model_prog_compute_grads: no COST variable found in program".to_string(),
            expected: (0, 0),
            got: (prog.size, prog.size),
        })?;
    mat_fill(&mut prog.vars[cost_index].gradient.borrow_mut(), 1.0);

    for i in (0..prog.size).rev() {
        let num_inputs = prog.vars[i].op.num_inputs();
        let op = prog.vars[i].op;

        if !prog.vars[i].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            continue;
        }

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
                        let b_requires_grad = prog.vars[b_idx]
                            .flags
                            .contains(ModelVarFlags::REQUIRES_GRAD);
                        if !a_requires_grad && !b_requires_grad {
                            continue;
                        }
                    }
                }
            }
        }

        let cur_gradient = prog.vars[i].gradient.borrow();
        let a_idx = prog.vars[i].inputs[0];
        let b_idx = prog.vars[i].inputs[1];

        match op {
            ModelVarOp::Add => {
                if let Some(idx) = a_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let mut a_grad = prog.vars[idx].gradient.borrow_mut();
                        crate::matrix::mat_add_assign(&mut a_grad, &cur_gradient)?;
                    }
                }
                if let Some(idx) = b_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let mut b_grad = prog.vars[idx].gradient.borrow_mut();
                        crate::matrix::mat_add_assign(&mut b_grad, &cur_gradient)?;
                    }
                }
            }
            ModelVarOp::Sub => {
                if let Some(idx) = a_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let mut a_grad = prog.vars[idx].gradient.borrow_mut();
                        crate::matrix::mat_add_assign(&mut a_grad, &cur_gradient)?;
                    }
                }
                if let Some(idx) = b_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let mut b_grad = prog.vars[idx].gradient.borrow_mut();
                        for (b, c) in b_grad.data.iter_mut().zip(cur_gradient.data.iter()) {
                            *b -= c;
                        }
                    }
                }
            }
            ModelVarOp::MatMul => {
                if let (Some(a_idx), Some(b_idx)) = (a_idx, b_idx) {
                    let b_value = prog.vars[b_idx].value.borrow();
                    let a_value = prog.vars[a_idx].value.borrow();

                    if prog.vars[a_idx]
                        .flags
                        .contains(ModelVarFlags::REQUIRES_GRAD)
                    {
                        mat_mul(
                            &mut prog.vars[a_idx].gradient.borrow_mut(),
                            &cur_gradient,
                            &b_value,
                            false,
                            false,
                            true,
                        )?;
                    }
                    if prog.vars[b_idx]
                        .flags
                        .contains(ModelVarFlags::REQUIRES_GRAD)
                    {
                        mat_mul(
                            &mut prog.vars[b_idx].gradient.borrow_mut(),
                            &a_value,
                            &cur_gradient,
                            false,
                            true,
                            false,
                        )?;
                    }
                }
            }
            ModelVarOp::CrossEntropy => {
                if let (Some(p_idx), Some(q_idx)) = (a_idx, b_idx) {
                    let p_value = prog.vars[p_idx].value.borrow();
                    let q_value = prog.vars[q_idx].value.borrow();

                    crate::ops::mat_cross_entropy_add_grad(
                        &mut prog.vars[p_idx].gradient.borrow_mut(),
                        &mut prog.vars[q_idx].gradient.borrow_mut(),
                        &p_value,
                        &q_value,
                        &cur_gradient,
                    )?;
                }
            }
            ModelVarOp::Relu => {
                if let Some(idx) = a_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let a_value = prog.vars[idx].value.borrow();
                        crate::ops::mat_relu_add_grad(
                            &mut prog.vars[idx].gradient.borrow_mut(),
                            &a_value,
                            &cur_gradient,
                        )?;
                    }
                }
            }
            ModelVarOp::Softmax => {
                if let Some(idx) = a_idx {
                    if prog.vars[idx].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                        let softmax_output = prog.vars[i].value.borrow();
                        crate::ops::mat_softmax_add_grad(
                            &mut prog.vars[idx].gradient.borrow_mut(),
                            &softmax_output,
                            &cur_gradient,
                        )?;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

pub fn model_compile(model: &mut ModelContext) {
    model.forward_program = model.create_program(&model.all_vars, model.output_idx);
    model.cost_program = model.create_program(&model.all_vars, model.cost_idx);
}

pub fn model_feedforward(model: &mut ModelContext) -> Result<(), MatrixError> {
    model_prog_compute(&mut model.forward_program)?;
    Ok(())
}
