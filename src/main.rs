mod graph;
mod matmul;
mod matrix;
mod mnist_model;
mod ops;
mod training;

use crate::graph::ModelContext;
use crate::graph::program::model_compile;
use crate::mnist_model::{create_mnist_model, prepare_mnist_data};
use crate::training::{ModelTrainingDesc, model_train};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MNIST dataset...");

    let train_data = mnist::MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .base_path("data/")
        .finalize();

    let (train_image_matrices, train_label_matrices, test_image_matrices, test_label_matrices) =
        prepare_mnist_data(
            &train_data.trn_img,
            &train_data.trn_lbl,
            &train_data.tst_img,
            &train_data.tst_lbl,
        );

    println!("Dataset loaded.");
    println!("Preparing data...");

    println!("Creating model...");
    let mut model = ModelContext::create();
    create_mnist_model(&mut model)?;
    model_compile(&mut model);

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
