
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

// ---------------------- Basic operations ----------------------

fn mat_copy(dst: &mut Matrix, src: &Matrix) -> bool {
    if dst.rows != src.rows || dst.cols != src.cols {
        return false;
    }
    dst.data.copy_from_slice(&src.data);
    true
}

fn mat_clear(mat: &mut Matrix) {
    mat.data.fill(0.0);
}

fn mat_fill(mat: &mut Matrix, x: f32) {
    mat.data.fill(x);
}

fn mat_scale(mat: &mut Matrix, x: f32) {
    for v in &mut mat.data {
        *v *= x;
    }
}

fn mat_sum(mat: &Matrix) -> f32 {
    mat.data.iter().copied().sum()
}

fn mat_add(out: &mut Matrix, a: &Matrix, b: &Matrix) -> bool {
    if a.rows != b.rows || a.cols != b.cols { return false; }
    if out.rows != a.rows || out.cols != a.cols { return false; }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }
    true
}

fn mat_sub(out: &mut Matrix, a: &Matrix, b: &Matrix) -> bool {
    if a.rows != b.rows || a.cols != b.cols { return false; }
    if out.rows != a.rows || out.cols != a.cols { return false; }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] - b.data[i];
    }
    true
}

// ---------------------- Matrix multiplication ----------------------

fn mat_mul_nn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.rows {
        for k in 0..a.cols {
            for j in 0..out.cols {
                out.data[out.idx(i,j)] += a.data[a.idx(i,k)] * b.data[b.idx(k,j)];
            }
        }
    }
}

fn mat_mul_nt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.rows {
        for j in 0..out.cols {
            for k in 0..a.cols {
                out.data[out.idx(i,j)] += a.data[a.idx(i,k)] * b.data[b.idx(j,k)];
            }
        }
    }
}

fn mat_mul_tn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for k in 0..a.rows {
        for i in 0..out.rows {
            for j in 0..out.cols {
                out.data[out.idx(i,j)] += a.data[a.idx(k,i)] * b.data[b.idx(k,j)];
            }
        }
    }
}

fn mat_mul_tt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for k in 0..a.cols {
        for i in 0..out.rows {
            for j in 0..out.cols {
                out.data[out.idx(i,j)] += a.data[a.idx(k,i)] * b.data[b.idx(j,k)];
            }
        }
    }
}

fn mat_mul(
    out: &mut Matrix, a: &Matrix, b: &Matrix, 
    zero_out: bool, transpose_a: bool, transpose_b: bool
) -> bool {
    let a_rows = if transpose_a { a.cols } else { a.rows };
    let a_cols = if transpose_a { a.rows } else { a.cols };
    let b_rows = if transpose_b { b.cols } else { b.rows };
    let b_cols = if transpose_b { b.rows } else { b.cols };

    if a_cols != b_rows { return false; }
    if out.rows != a_rows || out.cols != b_cols { return false; }

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

    true
}

// ---------------------- ReLU ----------------------

fn mat_relu(out: &mut Matrix, input: &Matrix) -> bool {
    if out.rows != input.rows || out.cols != input.cols { return false; }

    for i in 0..out.data.len() {
        out.data[i] = input.data[i].max(0.0);
    }

    true
}

// ---------------------- Example ----------------------

fn main() {
    let mut a = Matrix::new(2, 3);
    let mut b = Matrix::new(3, 2);
    let mut out = Matrix::new(2, 2);

    mat_fill(&mut a, 1.0);
    mat_fill(&mut b, 2.0);

    mat_mul(&mut out, &a, &b, true, false, false);

    println!("Result: {:?}", out);
}

