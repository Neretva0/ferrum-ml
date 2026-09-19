pub mod build;
pub mod program;

use std::cell::RefCell;
use std::rc::Rc;

use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct ModelVarFlags: u32 {
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
pub enum ModelVarOp {
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

pub const MODEL_VAR_MAX_INPUTS: usize = 2;

impl ModelVarOp {
    #[inline]
    /// IMPORTANT: This is based on the enum-ordering trick using UnaryStart/BinaryStart.
    /// Do not reorder those enum variants: the logic below relies on their relative ordering.
    pub fn num_inputs(self) -> usize {
        use ModelVarOp::*;

        match self {
            op if (op as u32) < UnaryStart as u32 => 0,
            op if (op as u32) < BinaryStart as u32 => 1,
            _ => 2,
        }
    }
}

#[derive(Clone)]
pub struct ModelVar {
    pub index: usize,
    pub flags: ModelVarFlags,
    pub op: ModelVarOp,
    pub value: Rc<RefCell<crate::matrix::Matrix>>,
    pub gradient: Rc<RefCell<crate::matrix::Matrix>>,
    pub inputs: [Option<usize>; MODEL_VAR_MAX_INPUTS],
}

#[derive(Clone)]
pub struct ModelProgram {
    pub vars: Vec<ModelVar>,
    pub size: usize,
}

#[derive(Clone)]
pub struct ModelContext {
    pub num_vars: usize,
    pub input_idx: usize,
    pub output_idx: usize,
    pub desired_output_idx: usize,
    pub cost_idx: usize,

    pub all_vars: Vec<ModelVar>,

    pub forward_program: ModelProgram,
    pub cost_program: ModelProgram,
}

impl ModelContext {
    pub fn create() -> Self {
        ModelContext {
            num_vars: 0,
            input_idx: 0,
            output_idx: 0,
            desired_output_idx: 0,
            cost_idx: 0,
            all_vars: Vec::new(),
            forward_program: ModelProgram {
                vars: Vec::new(),
                size: 0,
            },
            cost_program: ModelProgram {
                vars: Vec::new(),
                size: 0,
            },
        }
    }
}
