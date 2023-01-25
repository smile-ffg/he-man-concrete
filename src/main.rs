use clap::{Parser, Subcommand};

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod autopad;
#[macro_use]
mod broadcast;
mod cfg;
mod decrypt;
mod encrypt;
mod execute;
mod inference;
mod io;
mod key;
mod onnxengine;
mod operators;
#[macro_use]
mod multiindex;
mod tensor;
mod tensorcover;
mod calibrate;

#[derive(Subcommand)]
enum Command {
    /// Generates KeyParams and calibrated model based on calibration data
    Keyparams {
        /// Input path of the model to be calibrated
        model_input_path: String,
        /// Path to the zip/npz containing the calibration data
        calibration_data_path: String,
        /// Path to store the generated KeyParams file
        keyparams_output_path: String,
        /// Path to store the calibrated Model file
        model_output_path: String,
    },

    /// Generates HE keys
    Keygen {
        /// Path to keyparams 
        keyparams_input_path: String,
        /// Path of the folder to store the generated keys
        key_output_path: String,
    },

    /// Encrypts model input
    Encrypt {
        /// Path of the folder containing the keys to be used for encryption
        secret_key_path: String,
        /// Path of the file containing the plaintext input
        plaintext_input_path: String,
        /// Path of the file to store the ciphertext output
        ciphertext_output_path: String,
    },

    /// Applies a model to an encrypted input
    Inference {
        /// Path of the folder containing the evaluation keys to be used for
        /// the model application
        evaluation_key_path: String,
        /// Path of the model file that is applied
        model_path: String,
        /// Path of the file containing the ciphertext input
        ciphertext_input_path: String,
        /// Path of the file to store the ciphertext output
        ciphertext_output_path: String,
    },

    /// Decrypts model output
    Decrypt {
        /// Path of the folder containing the keys to be used for decryption
        secret_key_path: String,
        /// Path of the file containing the ciphertext input
        ciphertext_input_path: String,
        /// Path of the file to store the plaintext output
        plaintext_output_path: String,
    },
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None, propagate_version = true)]
struct Args {
    /// Command to be executed
    #[clap(subcommand)]
    command: Command,
}

fn main() {
    let args = Args::parse();
    match &args.command {
        Command::Keyparams { model_input_path, calibration_data_path, keyparams_output_path, model_output_path } => {
            calibrate::calibrate(model_input_path, calibration_data_path, keyparams_output_path, model_output_path);
        }
        Command::Keygen { keyparams_input_path, key_output_path } => {
            let params = io::load_serialized(keyparams_input_path);
            let keys = key::KeyChain::generate(params);
            keys.save(key_output_path);
        }
        Command::Encrypt {
            secret_key_path,
            plaintext_input_path,
            ciphertext_output_path,
        } => encrypt::encrypt(
            secret_key_path,
            plaintext_input_path,
            ciphertext_output_path,
        ),
        Command::Inference {
            evaluation_key_path,
            model_path,
            ciphertext_input_path,
            ciphertext_output_path,
        } => inference::inference(
            evaluation_key_path,
            model_path,
            ciphertext_input_path,
            ciphertext_output_path,
        ),
        Command::Decrypt {
            secret_key_path,
            ciphertext_input_path,
            plaintext_output_path,
        } => decrypt::decrypt(
            secret_key_path,
            ciphertext_input_path,
            plaintext_output_path,
        ),
    }
}
