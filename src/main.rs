struct Matrix{
    rows: u32,
    cols: u32,
    data: *mut f32,
}


fn mat_create(rows: u32, cols: u32) -> Box<Matrix> {
    Box::new(Matrix {
        rows,
        cols,
        data: vec![0.0; (rows * cols) as usize],
    })
}


fn main() {
    println!("Hello, world!");
}
