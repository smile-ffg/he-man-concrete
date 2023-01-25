use crate::onnxengine::OnnxEngine;
use crate::tensor::*;
use crate::io;
use crate::key;

pub fn calibrate(model_input_path: &str, calibration_data_path: &str, keyparams_output_path: &str, model_output_path: &str) {
    let mut engine = OnnxEngine::from_path(model_input_path);
    let calibration_data = io::load_npz(calibration_data_path);
    let mut input_min: f64 = 0.;
    let mut input_max: f64 = 0.;

    for input in calibration_data {
        input_min = input.get_values().into_iter().fold(input_min, |acc, x| acc.min(*x));
        input_max = input.get_values().into_iter().fold(input_max, |acc, x| acc.max(*x));
        engine.calibrate(vec![input]);
    }
    
    io::save_serialized(keyparams_output_path, key::KeyParams{input_min, input_max});
    engine.save_to_path(model_output_path);
}
