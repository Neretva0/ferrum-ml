# ML Library - Rust

A lightweight, modular machine learning library written in pure Rust featuring automatic differentiation and neural network training capabilities. Inspired by MagicalBat, which implements a minimal ML framework in C.

## Overview

This library provides a complete computational graph framework with automatic differentiation, supporting the construction and training of neural networks. Built from scratch without relying on heavy ML frameworks, it includes:

- Custom 2D matrix operations optimized for neural network workloads
- Automatic differentiation via reverse-mode backpropagation
- Flexible computational graph construction with topological compilation
- Mini-batch stochastic gradient descent (SGD) training pipeline
- Modular architecture with clean separation of concerns
- End-to-end MNIST digit classification example

## Features

- Pure Rust implementation with minimal dependencies
- Modular codebase partitioned into dedicated modules (`matrix`, `matmul`, `ops`, `graph`, `training`, `mnist_model`)
- Shared buffer semantics with `Rc<RefCell<Matrix>>` for zero-copy graph program execution
- Automatic gradient computation across all operations
- Topological sorting with graph-to-program index remapping
- Common activation functions (ReLU, Softmax) and Cross-Entropy loss with full analytical backward gradients
- Mini-batch SGD optimizer with dataset shuffling
- MNIST example achieving ~95% test accuracy on handwritten digit recognition

## Project Structure

The codebase is organized into focused, decoupled modules:

```text
src/
├── main.rs          # Application entrypoint running the MNIST training pipeline
├── matrix.rs        # Matrix struct, MatrixError, and basic element-wise & reduction ops
├── matmul.rs        # Optimized matrix multiplication kernels with transpose variations
├── ops.rs           # Activation functions & loss (ReLU, Softmax, CrossEntropy) + backward gradients
├── graph/           # Computational graph & automatic differentiation engine
│   ├── mod.rs       # Core types: ModelVar, ModelVarFlags, ModelVarOp, ModelProgram, ModelContext
│   ├── build.rs     # Graph construction helpers (mv_add, mv_matmul, mv_relu, mv_softmax, etc.)
│   └── program.rs   # Topological compilation (model_compile), forward & backward passes
├── training.rs      # Mini-batch SGD training loop and evaluation (ModelTrainingDesc, model_train)
└── mnist_model.rs   # MNIST model architecture (MLP + residual) and dataset preprocessing
```

---

## Core Modules & API Reference

### 1. Matrix (`src/matrix.rs`)

Provides the fundamental data structures and elementary operations.

#### Types
- **`Matrix`**: Represents a 2D matrix storing values in a row-major `Vec<f32>`:
  - `rows`: Number of rows
  - `cols`: Number of columns
  - `data`: Flat `Vec<f32>` storage
  - `Matrix::new(rows, cols)`: Creates a zero-initialized matrix
  - `Matrix::idx(&self, row, col)`: Converts 2D coordinates into a 1D index
- **`MatrixError`**: Custom error enum reporting dimension mismatches with contextual descriptions and `(expected, got)` shape tuples.

#### Operations
- `mat_copy(dst, src)`: Copies contents between identically sized matrices
- `mat_clear(mat)`: Zeroes out all elements in-place
- `mat_fill(mat, x)`: Fills all elements with scalar `x`
- `mat_fill_random(mat, lower, upper)`: Uniform random initialization within `[lower, upper)`
- `mat_scale(mat, x)`: In-place scalar multiplication
- `mat_sum(mat) -> f32`: Computes the sum of all elements
- `mat_argmax(mat) -> usize`: Finds the index of the maximum element
- `mat_add(out, a, b)`: Element-wise addition ($out = a + b$)
- `mat_sub(out, a, b)`: Element-wise subtraction ($out = a - b$)
- `mat_add_assign(a, b)`: In-place addition ($a \mathrel{+}= b$)

---

### 2. Matrix Multiplication (`src/matmul.rs`)

High-performance matrix multiplication kernels supporting arbitrary transpose combinations without allocating intermediate transposed matrices.

#### Low-level Kernels
- **`mat_mul_nn(out, a, b)`**: Standard multiplication ($A \times B$)
- **`mat_mul_nt(out, a, b)`**: Transposed second operand ($A \times B^T$)
- **`mat_mul_tn(out, a, b)`**: Transposed first operand ($A^T \times B$)
- **`mat_mul_tt(out, a, b)`**: Both operands transposed ($A^T \times B^T$)

#### Dispatcher
- **`mat_mul(out, a, b, zero_out, transpose_a, transpose_b) -> Result<(), MatrixError>`**: Validates inner and output dimensions, clears `out` if `zero_out` is true, and dispatches to the corresponding kernel.

---

### 3. Activation Functions & Loss (`src/ops.rs`)

Implements forward and analytical backward gradient passes for nonlinear activations and cost functions.

#### Forward Passes
- **`mat_relu(out, input)`**: Element-wise ReLU activation ($\max(x, 0)$)
- **`mat_softmax(out, input)`**: Numerically normalized softmax probability distribution
- **`mat_cross_entropy(out, p, q)`**: Multi-class cross-entropy loss between distribution $p$ and target $q$ with $\epsilon$-clamping ($10^{-10}$) to avoid $\ln(0)$

#### Backward Gradients
- **`mat_relu_add_grad(out, input, grad)`**: Accumulates gradient through the ReLU subgradient ($out \mathrel{+}= grad \times \mathbb{I}_{x > 0}$)
- **`mat_softmax_add_grad(out, softmax_out, grad)`**: Computes the exact Jacobian-vector product for softmax and accumulates into `out`
- **`mat_cross_entropy_add_grad(p_grad, q_grad, p, q, grad)`**: Accumulates analytical loss gradients with respect to both probability inputs and target inputs

---

### 4. Computational Graph (`src/graph/`)

The core automatic differentiation and computation graph engine, divided into types, graph-building primitives, and program execution.

#### Core Types (`src/graph/mod.rs`)
- **`ModelVar`**: A variable node in the graph:
  - `index`: Global graph variable identifier
  - `flags`: Variable properties (`ModelVarFlags`)
  - `op`: Generating operation (`ModelVarOp`)
  - `value`: Shared matrix buffer wrapped in `Rc<RefCell<Matrix>>`
  - `gradient`: Shared gradient buffer wrapped in `Rc<RefCell<Matrix>>`
  - `inputs`: Dependency variable indices `[Option<usize>; 2]`
- **`ModelVarFlags`**: Bitflags representing node roles:
  - `NONE`: Intermediate node without special flags
  - `REQUIRES_GRAD`: Node requires gradient calculation during backward pass
  - `PARAMETER`: Trainable weight or bias parameter
  - `INPUT`: Primary input placeholder (e.g., feature matrix)
  - `OUTPUT`: Primary inference output placeholder
  - `DESIRED_OUTPUT`: Supervised ground truth / target labels
  - `COST`: Loss / objective function output
- **`ModelVarOp`**: Operation types (`Null`, `Create`, `Relu`, `Softmax`, `Add`, `Sub`, `MatMul`, `CrossEntropy`). Exposes `num_inputs()` for arity checks.
- **`ModelProgram`**: Topologically sorted, flattened execution sequence of variables. Provides `find_by_flag(flag)` for lookup.
- **`ModelContext`**: Master context tracking all graph variables, key tensor indices, and compiled programs (`forward_program`, `cost_program`).

#### Graph Construction (`src/graph/build.rs`)
- **`ModelVar::create(...)`**: Low-level node factory allocating buffers and tracking graph indices
- **`mv_add(model, a, b, flags)`**: Addition operation node
- **`mv_sub(model, a, b, flags)`**: Subtraction operation node
- **`mv_matmul(model, a, b, flags)`**: Matrix multiplication operation node
- **`mv_relu(model, input, flags)`**: ReLU activation node
- **`mv_softmax(model, input, flags)`**: Softmax activation node
- **`mv_cross_entropy(model, p, q, flags)`**: Cross-entropy cost node

#### Program Compilation & Execution (`src/graph/program.rs`)
- **`ModelContext::create_program(&self, all_vars, out_var_idx) -> ModelProgram`**: Performs depth-first topological sorting from the target node, resolving dependencies and remapping global indices into compact program-local indices.
- **`model_compile(model)`**: Compiles both `forward_program` (from output node) and `cost_program` (from cost node).
- **`model_prog_compute(prog)`**: Evaluates the program forward pass in topological order.
- **`model_prog_compute_grads(prog)`**: Reverse-mode automatic differentiation through the compiled program. Clears intermediate gradients, seeds cost gradient to 1.0, and backpropagates through operations in reverse topological order.
- **`model_feedforward(model)`**: Convenience helper running the compiled `forward_program`.

---

### 5. Training Pipeline (`src/training.rs`)

Implements mini-batch stochastic gradient descent (SGD) and evaluation.

- **`ModelTrainingDesc`**:
  - `train_images`, `train_labels`: Training dataset matrices
  - `test_images`, `test_labels`: Validation/test dataset matrices
  - `epochs`: Number of full passes over the dataset
  - `batch_size`: Mini-batch size
  - `learning_rate`: Gradient descent step size ($\alpha$)
- **`model_train(model, training_desc)`**:
  1. Shuffles training sample indices before each epoch using `rand`.
  2. For each mini-batch:
     - Zeroes out parameter gradients.
     - Copies inputs and targets into the compiled cost program.
     - Runs forward pass (`model_prog_compute`) and backward pass (`model_prog_compute_grads`).
     - Scales accumulated parameter gradients by $\frac{\alpha}{\text{batch\_size}}$ and performs gradient descent update ($W \mathrel{-}= \frac{\alpha}{B} \nabla W$).
  3. Evaluates test set accuracy and cost after each epoch.

---

### 6. MNIST Neural Network (`src/mnist_model.rs`)

End-to-end multi-layer perceptron (MLP) for digit classification.

- **`prepare_mnist_data(...)`**: Converts raw byte slices into normalized column vectors ($784 \times 1$ inputs scaled to $[0.0, 1.0]$ and $10 \times 1$ one-hot label vectors).
- **`create_mnist_model(model)`**: Builds a 3-layer neural network with a residual connection:
  - **Input**: $784 \times 1$
  - **Layer 0**: Linear($784 \to 16$) + ReLU
  - **Layer 1**: Linear($16 \to 16$) + ReLU + Residual skip connection ($a_1 = \text{ReLU}(W_1 a_0 + b_1) + a_0$)
  - **Layer 2**: Linear($16 \to 10$) + Softmax
  - **Weights**: Xavier/Glorot uniform initialization ($\pm\sqrt{6 / (d_{in} + d_{out})}$)
  - **Biases**: Initialized to zeros
  - **Loss**: Cross-entropy cost node against one-hot target

---

## Example Usage

### 1. Matrix Math

```rust
use ml_lib_rust::matrix::{Matrix, mat_fill};
use ml_lib_rust::matmul::mat_mul;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Matrix::new(2, 3);
    let mut b = Matrix::new(3, 2);
    let mut out = Matrix::new(2, 2);

    mat_fill(&mut a, 1.0);
    mat_fill(&mut b, 2.0);

    // out = A * B (clearing out first, no transposition)
    mat_mul(&mut out, &a, &b, true, false, false)?;
    assert_eq!(out.data, vec![6.0, 6.0, 6.0, 6.0]);
    Ok(())
}
```

### 2. Neural Network Construction & Training

```rust
use ml_lib_rust::graph::ModelContext;
use ml_lib_rust::graph::program::model_compile;
use ml_lib_rust::mnist_model::{create_mnist_model, prepare_mnist_data};
use ml_lib_rust::training::{ModelTrainingDesc, model_train};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize model context and build computation graph
    let mut model = ModelContext::create();
    create_mnist_model(&mut model)?;

    // 2. Compile forward and cost execution programs (topological sort)
    model_compile(&mut model);

    // 3. Prepare data and train
    // let (train_imgs, train_lbls, test_imgs, test_lbls) = prepare_mnist_data(...);
    // let desc = ModelTrainingDesc {
    //     train_images: train_imgs,
    //     train_labels: train_lbls,
    //     test_images: test_imgs,
    //     test_labels: test_lbls,
    //     epochs: 5,
    //     batch_size: 32,
    //     learning_rate: 0.01,
    // };
    // model_train(&mut model, &desc)?;

    Ok(())
}
```

---

## Quick Start

### 1. Prerequisites
Ensure you have the Rust toolchain (2024 edition supported) installed:
```bash
rustc --version
cargo --version
```

### 2. Download MNIST Dataset
Place the uncompressed binary IDX files inside a `data/` directory at the project root:
```bash
mkdir -p data && cd data
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

### 3. Run Training
```bash
cargo run --release
```

Sample output:
```text
Loading MNIST dataset...
Dataset loaded.
Preparing data...
Creating model...
Starting training...

Epoch  1 /  5, Batch 1562 / 1562, Average Cost: 0.3210
Test Completed. Accuracy:  9280 / 10000 (92.8%), Average Cost: 0.2814
...
Epoch  5 /  5, Batch 1562 / 1562, Average Cost: 0.1542
Test Completed. Accuracy:  9512 / 10000 (95.1%), Average Cost: 0.1705

Training complete!
```

---

## Key Architecture & Design Principles

- **Shared Ownership via `Rc<RefCell<Matrix>>`**: Graph variables hold reference-counted pointers to matrices. When execution programs (`ModelProgram`) are compiled from the graph, they reference the same underlying matrix buffers. This eliminates memory copying between the graph, execution programs, and outside callers.
- **Topological Index Remapping**: The user constructs nodes with global graph identifiers. During compilation (`ModelContext::create_program`), a depth-first traversal orders only required dependency nodes and remaps their input edges into a dense, contiguous array for sequential cache-friendly iteration.
- **Reverse-Mode Automatic Differentiation**: Gradients are seeded at the cost node ($\frac{\partial C}{\partial C} = 1.0$) and propagated backwards in reverse topological order. Parameters accumulate gradients across mini-batch samples before step updates.
