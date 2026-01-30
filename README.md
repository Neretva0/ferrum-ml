# ML Library - Rust

A simple matrix computation library written in Rust for machine learning operations.

## Overview

This library provides basic matrix operations and neural network primitives, designed with performance and flexibility in mind.

## Core Components

### `MatrixError`
Custom error type for matrix operations that tracks dimension mismatches with context about what operation failed and the expected vs. actual dimensions.

### `Matrix` Struct
Represents a 2D matrix using:
- `rows`: number of rows
- `cols`: number of columns
- `data`: flat vector of `f32` values (row-major order)

Includes helper method `idx()` for converting 2D coordinates to flat array indices.

## Basic Operations

- **`mat_copy()`** - Copy data from source matrix to destination
- **`mat_clear()`** - Fill matrix with zeros
- **`mat_fill()`** - Fill matrix with a specific scalar value
- **`mat_scale()`** - Multiply all elements by a scalar
- **`mat_sum()`** - Sum all elements in matrix
- **`mat_add()`** - Element-wise addition of two matrices
- **`mat_sub()`** - Element-wise subtraction of two matrices

## Matrix Multiplication

Supports flexible matrix multiplication with transpose options:

- **`mat_mul_nn()`** - Standard multiplication (A × B)
- **`mat_mul_nt()`** - A × B^T (B transposed)
- **`mat_mul_tn()`** - A^T × B (A transposed)
- **`mat_mul_tt()`** - A^T × B^T (both transposed)

Main function **`mat_mul()`** handles validation and dispatches to the appropriate variant based on transpose flags.

## Activation Functions

- **`mat_relu()`** - Rectified Linear Unit activation (element-wise max(x, 0))
- **`mat_softmax()`** - Softmax activation, normalizes input to a probability distribution
- **`mat_cross_entropy()`** - Cross-entropy loss function between predicted probabilities and target distribution

## Neural Network Components (In Progress)

- **`ModelVarFlags`** - Bitflags for variable properties (REQUIRES_GRAD, PARAMETER, INPUT, OUTPUT, DESIRED_OUTPUT, COST)
- **`ModelVarOp`** - Enum representing computation operations (Relu, Softmax, Add, Sub, MatMul, CrossEntropy, etc.)
- **`ModelVar`** - Represents a variable in the computation graph with value, gradient, and operation info
- **`ModelContext`** - Context structure for managing computation graph variables and I/O

## Example

The `main()` function demonstrates basic usage:
- Creates two matrices (2×3 and 3×2)
- Fills them with values
- Performs matrix multiplication
- Prints the result

## Error Handling

All operations that can fail return `Result<(), MatrixError>` for comprehensive error checking with detailed context about dimension mismatches.
