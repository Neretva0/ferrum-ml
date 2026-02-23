# ML Library - Rust

A lightweight machine learning library written in pure Rust featuring automatic differentiation and neural network training capabilities. Inspired by MagicalBat, which implements a minimal ML framework in C.

## Overview

This library provides a complete computational graph framework with automatic differentiation, supporting the construction and training of neural networks. Built from scratch without relying on heavy ML frameworks, it includes:

- Custom matrix operations optimized for neural networks
- Automatic differentiation via reverse-mode backpropagation
- Flexible computational graph construction
- Training loop with mini-batch gradient descent
- MNIST digit classification example

## Features

- ✅ Pure Rust implementation with minimal dependencies
- ✅ Automatic gradient computation for all operations
- ✅ Topological sorting for efficient computation
- ✅ Support for common activation functions (ReLU, Softmax)
- ✅ Cross-entropy loss for classification tasks
- ✅ Mini-batch stochastic gradient descent
- ✅ MNIST example achieving good accuracy

## Core Components

### Error Handling

**`MatrixError`**  
Custom error type for matrix operations that tracks dimension mismatches with detailed context about what operation failed and the expected vs. actual dimensions.

### Data Structures

**`Matrix`**  
Represents a 2D matrix using:
- `rows`: number of rows
- `cols`: number of columns
- `data`: flat vector of `f32` values (row-major order)

Includes helper method `idx()` for converting 2D coordinates to flat array indices.

**`ModelVar`**  
Represents a variable node in the computation graph:
- `index`: unique identifier in the model context
- `flags`: bitflags indicating variable properties
- `op`: operation that created this variable
- `value`: current matrix value
- `gradient`: accumulated gradient (if requires_grad is set)
- `inputs`: indices of input variables (up to 2)

**`ModelProgram`**  
Topologically sorted list of `ModelVar` operations for efficient forward/backward passes:
- `vars`: ordered variables for computation
- `size`: number of variables in the program

**`ModelContext`**  
Central context managing the computation graph:
- Tracks all variables created
- Maintains references to special variables (input, output, desired output, cost)
- Compiles forward and cost computation programs

## Basic Matrix Operations

- **`mat_copy()`** - Copy data from source matrix to destination
- **`mat_clear()`** - Fill matrix with zeros
- **`mat_fill()`** - Fill matrix with a specific scalar value
- **`mat_fill_random()`** - Fill matrix with random values
- **`mat_scale()`** - Multiply all elements by a scalar
- **`mat_sum()`** - Sum all elements in matrix
- **`mat_argmax()`** - Find index of maximum element
- **`mat_add()`** - Element-wise addition of two matrices
- **`mat_sub()`** - Element-wise subtraction of two matrices

## Matrix Multiplication

Supports flexible matrix multiplication with transpose options:

- **`mat_mul_nn()`** - Standard multiplication (A × B)
- **`mat_mul_nt()`** - A × B^T (B transposed)
- **`mat_mul_tn()`** - A^T × B (A transposed)
- **`mat_mul_tt()`** - A^T × B^T (both transposed)

Main function **`mat_mul()`** handles validation and dispatches to the appropriate variant based on transpose flags.

## Activation Functions & Loss

**Forward Operations:**
- **`mat_relu()`** - Rectified Linear Unit: max(x, 0)
- **`mat_softmax()`** - Softmax activation, normalizes to probability distribution
- **`mat_cross_entropy()`** - Cross-entropy loss between predicted and target distributions

**Gradient Operations:**
- **`mat_relu_add_grad()`** - Accumulate gradients through ReLU
- **`mat_softmax_add_grad()`** - Accumulate gradients through Softmax using Jacobian
- **`mat_cross_entropy_add_grad()`** - Accumulate gradients through cross-entropy loss

## Computational Graph API

**Variable Flags:**
- `REQUIRES_GRAD` - Variable requires gradient computation
- `PARAMETER` - Trainable parameter
- `INPUT` - Network input
- `OUTPUT` - Network output
- `DESIRED_OUTPUT` - Target/label for training
- `COST` - Loss function output

**Operations:**
- `Create` - Variable creation
- `Add`, `Sub` - Element-wise arithmetic
- `MatMul` - Matrix multiplication
- `Relu` - ReLU activation
- `Softmax` - Softmax activation
- `CrossEntropy` - Cross-entropy loss

**Graph Construction Functions:**
- **`mv_relu()`** - Create ReLU operation node
- **`mv_softmax()`** - Create Softmax operation node
- **`mv_add()`** - Create addition operation node
- **`mv_sub()`** - Create subtraction operation node
- **`mv_matmul()`** - Create matrix multiplication node
- **`mv_cross_entropy()`** - Create cross-entropy loss node

## Compilation & Execution

**`ModelContext::create_program()`**  
Performs topological sort on computation graph to create efficient execution programs. Visits dependencies recursively to ensure operations execute in correct order.

**`model_prog_compute()`**  
Executes forward pass through a program, computing all variable values in topological order.

**`model_prog_compute_grads()`**  
Executes backward pass, computing gradients via reverse-mode automatic differentiation. Handles all operation types including Add, Sub, MatMul, ReLU, Softmax, and CrossEntropy.

**`modele_compile()`**  
Compiles both forward and cost computation programs from the graph.

**`model_feedforward()`**  
Runs forward computation program to produce outputs.

## Training

**`ModelTrainingDesc`**  
Training configuration structure:
- Training and test datasets (images and labels)
- `epochs`: number of training epochs
- `batch_size`: mini-batch size
- `learning_rate`: learning rate for gradient descent

**`model_train()`**  
Full training loop implementation:
1. Shuffles training data each epoch
2. Processes mini-batches
3. Computes forward pass and loss
4. Computes gradients via backpropagation
5. Updates parameters using gradient descent
6. Reports training progress and test accuracy

## Example: MNIST Neural Network

**`create_mnist_model()`**  
Constructs a 3-layer neural network for MNIST digit classification:

**Architecture:**
- Input: 784 dimensions (28×28 flattened image)
- Hidden Layer 1: 16 neurons with ReLU activation
- Hidden Layer 2: 16 neurons with ReLU and residual connection
- Output Layer: 10 neurons with Softmax activation
- Loss: Cross-entropy

**Layers:**
1. **Layer 0**: Linear(784 → 16) + ReLU
2. **Layer 1**: Linear(16 → 16) + ReLU + Residual
3. **Layer 2**: Linear(16 → 10) + Softmax

All weight matrices initialized with random values. Biases initialized to zero.

## Example Usage

```rust
// Basic matrix multiplication
let mut a = Matrix::new(2, 3);
let mut b = Matrix::new(3, 2);
let mut out = Matrix::new(2, 2);

mat_fill(&mut a, 1.0);
mat_fill(&mut b, 2.0);
mat_mul(&mut out, &a, &b, true, false, false)?;

// Create and train a neural network
let mut model = ModelContext::create();
create_mnist_model(&mut model)?;
modele_compile(&mut model);

let training_desc = ModelTrainingDesc {
    train_images: /* training images */,
    train_labels: /* training labels */,
    test_images: /* test images */,
    test_labels: /* test labels */,
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.01,
};

model_train(&mut model, &training_desc)?;
```

## Error Handling

All operations that can fail return `Result<T, MatrixError>` or `Result<T, Box<dyn std::error::Error>>` for comprehensive error checking with detailed context about dimension mismatches and other failures.

## Dependencies

- `rand` - Random number generation for weight initialization and data shuffling
- `bitflags` - Efficient bitflag implementation for variable flags
- `mnist` - MNIST dataset loader (for the example)

## Quick Start

1. Clone the repository:
```bash
git clone <repo-url>
cd ml-lib-rust
```

2. Download MNIST data:
```bash
mkdir -p data
cd data
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

3. Run the MNIST training example:
```bash
cargo run --release
```

## Architecture

The library is built around a computational graph where each operation creates new nodes. The graph is compiled into efficient execution programs through topological sorting, with automatic index remapping for correctness.

### Key Implementation Details

- **Index Remapping**: When creating execution programs, variable indices are remapped from the global graph to program-local indices to ensure correct dependency resolution.
- **Gradient Accumulation**: Gradients flow backward through the graph, accumulating at parameter nodes.
- **Clone-on-Store**: Variables are cloned when added to the graph to maintain proper ownership semantics.
