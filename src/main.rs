struct Matrix {
    rows: u32,
    cols: u32,
    data: Vec<f32>,
}

fn mat_create(rows: u32, cols: u32) -> Matrix {
    Matrix {
        rows,
        cols,
        data: vec![0.0; (rows * cols) as usize],
    }
}

fn mat_copy(dst: &mut Matrix, src: &Matrix) -> bool {
    if dst.rows != src.rows || dst.cols != src.cols {
        return false;
    }

    dst.data.copy_from_slice(&src.data);
    true
}

fn mat_clear(m: &mut Matrix) {
    m.data.fill(0.0);
}

fn mat_fill(mat: &mut Matrix, x: f32){
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
    if a.rows != b.rows || a.cols != b.cols {
        return false;
    }
    
    if out.rows != a.rows || out.cols != a.cols{
        return false;
    }

    let size = (out.rows * out.cols) as usize;

    for i in 0..size {
        out.data[i] = a.data[i] + b.data[i];
    }
    false
}

fn mat_sub(out: &mut Matrix, a: &Matrix, b: &Matrix) -> bool {
    if a.rows != b.rows || a.cols != b.cols {
        return false;
    }
    
    if out.rows != a.rows || out.cols != a.cols{
        return false;
    }

    let size = (out.rows * out.cols) as usize;

    for i in 0..size {
        out.data[i] = a.data[i] - b.data[i];
    }
    false
}
fn main() {
    println!("Hello, world!");
}
