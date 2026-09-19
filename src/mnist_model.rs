use crate::graph::build::{mv_add, mv_cross_entropy, mv_matmul, mv_relu, mv_softmax};
use crate::graph::{ModelContext, ModelVar, ModelVarFlags};
use crate::matrix::{Matrix, MatrixError, mat_fill_random};

pub fn mnist_image_to_matrix(image: &[u8]) -> Matrix {
    let mut img_matrix = Matrix::new(784, 1);
    for j in 0..784 {
        img_matrix.data[j] = image[j] as f32 / 255.0;
    }
    img_matrix
}

pub fn mnist_label_to_matrix(label: u8) -> Matrix {
    let mut label_matrix = Matrix::new(10, 1);
    label_matrix.data[label as usize] = 1.0;
    label_matrix
}

pub fn prepare_mnist_data(
    train_images: &[u8],
    train_labels: &[u8],
    test_images: &[u8],
    test_labels: &[u8],
) -> (Vec<Matrix>, Vec<Matrix>, Vec<Matrix>, Vec<Matrix>) {
    let mut train_image_matrices = Vec::new();
    let mut train_label_matrices = Vec::new();

    for i in 0..train_labels.len() {
        train_image_matrices.push(mnist_image_to_matrix(&train_images[i * 784..(i + 1) * 784]));
        train_label_matrices.push(mnist_label_to_matrix(train_labels[i]));
    }

    let mut test_image_matrices = Vec::new();
    let mut test_label_matrices = Vec::new();

    for i in 0..test_labels.len() {
        test_image_matrices.push(mnist_image_to_matrix(&test_images[i * 784..(i + 1) * 784]));
        test_label_matrices.push(mnist_label_to_matrix(test_labels[i]));
    }

    (
        train_image_matrices,
        train_label_matrices,
        test_image_matrices,
        test_label_matrices,
    )
}

pub fn create_mnist_model(model: &mut ModelContext) -> Result<(), MatrixError> {
    let input = ModelVar::create(
        model,
        784,
        1,
        ModelVarFlags::INPUT,
        crate::graph::ModelVarOp::Null,
        [None, None],
    )?;

    let w0 = ModelVar::create(
        model,
        16,
        784,
        ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD,
        crate::graph::ModelVarOp::Null,
        [None, None],
    )?;
    let w1 = ModelVar::create(
        model,
        16,
        16,
        ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD,
        crate::graph::ModelVarOp::Null,
        [None, None],
    )?;
    let w2 = ModelVar::create(
        model,
        10,
        16,
        ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD,
        crate::graph::ModelVarOp::Null,
        [None, None],
    )?;

    let bound0 = (6.0f32 / (784 + 16) as f32).sqrt();
    let bound1 = (6.0f32 / (16 + 16) as f32).sqrt();
    let bound2 = (6.0f32 / (16 + 10) as f32).sqrt();
    mat_fill_random(&mut w0.value.borrow_mut(), -bound0, bound0);
    mat_fill_random(&mut w1.value.borrow_mut(), -bound1, bound1);
    mat_fill_random(&mut w2.value.borrow_mut(), -bound2, bound2);

    let b0 = ModelVar::create(
        model,
        16,
        1,
        ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD,
        crate::graph::ModelVarOp::Null,
        [None, None],
    )?;
    let b1 = ModelVar::create(
        model,
        16,
        1,
        ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD,
        crate::graph::ModelVarOp::Null,
        [None, None],
    )?;
    let b2 = ModelVar::create(
        model,
        10,
        1,
        ModelVarFlags::PARAMETER | ModelVarFlags::REQUIRES_GRAD,
        crate::graph::ModelVarOp::Null,
        [None, None],
    )?;

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

    let desired_output = ModelVar::create(
        model,
        10,
        1,
        ModelVarFlags::DESIRED_OUTPUT,
        crate::graph::ModelVarOp::Null,
        [None, None],
    )?;
    let _cost = mv_cross_entropy(model, &desired_output, &output, ModelVarFlags::COST)?;

    Ok(())
}
