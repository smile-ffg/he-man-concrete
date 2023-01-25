use crate::cfg;
use crate::io;
use crate::key;

use concrete::*;

pub fn encrypt(secret_key_path: &str, input_path: &str, output_path: &str) {
    let secret_key = key::SecretKey::load(secret_key_path);
    let input = io::load_npy(input_path);

    let encoder = Encoder::new(secret_key.params.input_min, secret_key.params.input_max, cfg::PRECISION, cfg::WEIGHT_PRECISION + 1).unwrap();
    let output = input.encrypt(&encoder, &secret_key);
    io::save_serialized(output_path, output);
}
