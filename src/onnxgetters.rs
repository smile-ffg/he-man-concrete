use crate::onnx;
use crate::tensorcover::*;

use std::collections::HashMap;

pub fn get_positional_input(
    node: &onnx::NodeProto,
    namespace: &HashMap<String, TensorCover>,
    pos: usize,
) -> Option<TensorCover> {
    if node.input.len() > pos && node.input[pos] != "" {
        // attempt to retrieve from namespace
        Some(
            namespace
                .get(&node.input[pos])
                .unwrap_or_else(|| panic!("Input {} not found in namespace", node.input[pos]))
                .clone(),
        )
    } else {
        // optional input not present
        None
    }
}

pub fn get_named_attribute<'a>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> Option<&'a onnx::AttributeProto> {
    node.attribute.iter().find(|&x| x.name == name)
}

pub fn get_metadatum(metadata: &Vec<onnx::StringStringEntryProto>, key: &str) -> Option<f64> {
    match metadata.iter().find(|x| x.key == key) {
        Some(x) => Some(x.value.parse::<f64>().unwrap()),
        None => None,
    }
}

pub fn set_metadatum(metadata: &mut Vec<onnx::StringStringEntryProto>, datum: f64, key: &str) {
    let index = metadata.iter().position(|x| x.key == key);

    match index {
        Some(ind) => metadata[ind].value = datum.to_string(),
        None => metadata.push(onnx::StringStringEntryProto {
            key: key.to_string(),
            value: datum.to_string(),
        }),
    }
}
