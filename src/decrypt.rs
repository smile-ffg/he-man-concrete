use crate::io;
use crate::key;
use crate::tensor::EncryptedTensor;

pub fn decrypt(secret_key_path: &str, input_path: &str, output_path: &str) {
    let secret_key = key::SecretKey::load(secret_key_path);
    let input: EncryptedTensor = io::load_serialized(input_path);

    let output = input.decrypt(&secret_key);
    io::save_npy(output_path, output);
}
