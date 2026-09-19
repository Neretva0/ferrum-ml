use std::error::Error;
use std::io::Write;

use rand::Rng;

use crate::graph::program::{model_prog_compute, model_prog_compute_grads};
use crate::graph::{ModelContext, ModelVarFlags};
use crate::matrix::{Matrix, mat_argmax, mat_clear, mat_copy, mat_scale, mat_sum};

pub struct ModelTrainingDesc {
    pub train_images: Vec<Matrix>,
    pub train_labels: Vec<Matrix>,
    pub test_images: Vec<Matrix>,
    pub test_labels: Vec<Matrix>,

    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
}

pub fn model_train(
    model: &mut ModelContext,
    training_desc: &ModelTrainingDesc,
) -> Result<(), Box<dyn Error>> {
    let train_images = &training_desc.train_images;
    let train_labels = &training_desc.train_labels;
    let test_images = &training_desc.test_images;
    let test_labels = &training_desc.test_labels;

    let num_examples = train_images.len();
    let num_tests = test_images.len();

    let num_batches = num_examples / training_desc.batch_size;

    let prog_input_idx = model
        .cost_program
        .find_by_flag(ModelVarFlags::INPUT)
        .expect("INPUT variable not found in cost program");
    let prog_desired_output_idx = model
        .cost_program
        .find_by_flag(ModelVarFlags::DESIRED_OUTPUT)
        .expect("DESIRED_OUTPUT variable not found in cost program");
    let prog_output_idx = model
        .cost_program
        .find_by_flag(ModelVarFlags::OUTPUT)
        .expect("OUTPUT variable not found in cost program");
    let prog_cost_idx = model
        .cost_program
        .find_by_flag(ModelVarFlags::COST)
        .expect("COST variable not found in cost program");

    let mut training_order: Vec<usize> = (0..num_examples).collect();

    for epoch in 0..training_desc.epochs {
        let mut rng = rand::rng();
        for _ in 0..num_examples {
            let a = rng.random_range(0..num_examples);
            let b = rng.random_range(0..num_examples);
            training_order.swap(a, b);
        }

        for batch in 0..num_batches {
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

                mat_copy(
                    &mut model.cost_program.vars[prog_input_idx].value.borrow_mut(),
                    &train_images[index],
                )?;

                mat_copy(
                    &mut model.cost_program.vars[prog_desired_output_idx]
                        .value
                        .borrow_mut(),
                    &train_labels[index],
                )?;

                model_prog_compute(&mut model.cost_program)?;
                model_prog_compute_grads(&mut model.cost_program)?;

                avg_cost += mat_sum(&model.cost_program.vars[prog_cost_idx].value.borrow());
            }
            avg_cost /= training_desc.batch_size as f32;

            for i in 0..model.cost_program.size {
                let cur = &mut model.cost_program.vars[i];

                if !cur.flags.contains(ModelVarFlags::PARAMETER) {
                    continue;
                }

                let mut grad = cur.gradient.borrow_mut();
                let mut val = cur.value.borrow_mut();

                mat_scale(
                    &mut grad,
                    training_desc.learning_rate / training_desc.batch_size as f32,
                );

                for j in 0..val.data.len() {
                    val.data[j] -= grad.data[j];
                }
            }

            print!(
                "\rEpoch {:2} / {:2}, Batch {:4} / {:4}, Average Cost: {:.4}",
                epoch + 1,
                training_desc.epochs,
                batch + 1,
                num_batches,
                avg_cost,
            );
            std::io::stdout().flush()?;
        }
        println!();

        let mut num_correct = 0;
        let mut avg_cost = 0.0f32;
        for i in 0..num_tests {
            mat_copy(
                &mut model.cost_program.vars[prog_input_idx].value.borrow_mut(),
                &test_images[i],
            )?;

            mat_copy(
                &mut model.cost_program.vars[prog_desired_output_idx]
                    .value
                    .borrow_mut(),
                &test_labels[i],
            )?;

            model_prog_compute(&mut model.cost_program)?;

            avg_cost += mat_sum(&model.cost_program.vars[prog_cost_idx].value.borrow());

            if mat_argmax(&model.cost_program.vars[prog_output_idx].value.borrow())
                == mat_argmax(
                    &model.cost_program.vars[prog_desired_output_idx]
                        .value
                        .borrow(),
                )
            {
                num_correct += 1;
            }
        }

        avg_cost /= num_tests as f32;
        println!(
            "Test Completed. Accuracy: {:5} / {:5} ({:.1}%), Average Cost: {:.4}",
            num_correct,
            num_tests,
            num_correct as f32 / num_tests as f32 * 100.0,
            avg_cost,
        );
    }

    Ok(())
}
