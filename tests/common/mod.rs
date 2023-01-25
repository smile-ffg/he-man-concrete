use bincode;
use std::fs::File;
use std::io::BufReader;

pub fn load_vec(path: &str) -> Vec<f64> {
    let rdr = BufReader::new(File::open(path).unwrap());
    bincode::deserialize_from(rdr).unwrap()
}

pub fn assert_close_abs(a: &[f64], b: &[f64], absolute_epsilon: f64) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (*x - *y).abs() < absolute_epsilon,
            "{} and {} not within eps={}",
            *x,
            *y,
            absolute_epsilon
        );
    }
}
