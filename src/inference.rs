use crate::io;
use crate::key;
use crate::onnxengine::*;
use crate::tensor::*;

pub fn inference(evaluation_key_path: &str, model_path: &str, input_path: &str, output_path: &str) {
    let ev_key = key::EvaluationKey::load(evaluation_key_path);
    let mut engine = OnnxEngine::from_path(model_path);

    let input: EncryptedTensor = io::load_serialized(input_path);
    let mut output =
        EncryptedTensor::peel(engine.run(vec![input], Some(&ev_key)).pop().unwrap()).unwrap();

    // perform remaining bootstraps
    output.evaluate_bootstrap(1, &ev_key);

    io::save_serialized(output_path, output);
}
