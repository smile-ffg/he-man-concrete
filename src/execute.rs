use crate::autopad;
use crate::key;
use crate::onnx;
use crate::operators::elementwise::ElementwiseOperator;
use crate::operators::*;
use crate::tensor::*;
use crate::tensorcover::*;
use std::collections::HashMap;

fn get_positional_input(
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

fn get_named_attribute<'a>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> Option<&'a onnx::AttributeProto> {
    node.attribute.iter().find(|&x| x.name == name)
}

fn get_metadatum(metadata: &Vec<onnx::StringStringEntryProto>, key: &str) -> Option<f64> {
    match metadata.iter().find(|x| x.key == key) {
        Some(x) => Some(x.value.parse::<f64>().unwrap()),
        None => None,
    }
}

pub fn execute_add(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    ev_key: Option<&key::EvaluationKey>,
) {
    let a = get_positional_input(node, namespace, 0).unwrap();
    let b = get_positional_input(node, namespace, 1).unwrap();
    let output = add::add(a, b, ev_key);
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_average_pool(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    ev_key: Option<&key::EvaluationKey>,
) {
    let x = get_positional_input(node, namespace, 0).unwrap();
    let auto_pad = match get_named_attribute(node, "auto_pad") {
        Some(attr) => match std::str::from_utf8(&attr.s).unwrap() {
            "NOTSET" => autopad::AutoPad::NotSet,
            "SAME_UPPER" => autopad::AutoPad::SameUpper,
            "SAME_LOWER" => autopad::AutoPad::SameLower,
            "VALID" => autopad::AutoPad::Valid,
            _ => panic!(
                "Averagepool with unrecognized autopad {}",
                std::str::from_utf8(&attr.s).unwrap()
            ),
        },
        None => autopad::AutoPad::NotSet,
    };
    let ceil_mode = match get_named_attribute(node, "ceil_mode") {
        Some(attr) => Some(attr.i),
        None => None,
    };
    let count_include_pad = match get_named_attribute(node, "count_include_pad") {
        Some(attr) => Some(attr.i),
        None => None,
    };
    let kernel_shape = match get_named_attribute(node, "kernel_shape") {
        Some(attr) => attr.ints.clone(),
        None => panic!("Missing required attribute kernel_shape"),
    };
    let pads = match get_named_attribute(node, "pads") {
        Some(attr) => Some(attr.ints.clone()),
        None => None,
    };
    let strides = match get_named_attribute(node, "strides") {
        Some(attr) => Some(attr.ints.clone()),
        None => None,
    };
    let output = averagepool::average_pool(
        x,
        auto_pad,
        ceil_mode,
        count_include_pad,
        kernel_shape,
        pads,
        strides,
        ev_key,
    );
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_clip(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let mut input = get_positional_input(node, namespace, 0).unwrap();
    let min = match get_positional_input(node, namespace, 1) {
        Some(x) => ClearTensor::peel(x).unwrap().get_values()[0],
        None => f64::NEG_INFINITY,
    };
    let max = match get_positional_input(node, namespace, 2) {
        Some(x) => ClearTensor::peel(x).unwrap().get_values()[0],
        None => f64::INFINITY,
    };
    input.apply_elementwise_operator(ElementwiseOperator::Clip { min: min, max: max });
    namespace.insert(node.output[0].clone(), input);
}

pub fn execute_constant(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let output = match node.attribute[0].name.as_str() {
        "value" => constant::constant(TensorCover::from_proto(
            &node.attribute[0].clone().t.unwrap(),
        )),
        "value_float" => constant::constant(node.attribute[0].f as f64),
        "value_floats" => constant::constant(
            node.attribute[0]
                .floats
                .iter()
                .map(|x| *x as f64)
                .collect::<Vec<f64>>(),
        ),
        "value_int" => constant::constant(node.attribute[0].i),
        "value_ints" => constant::constant(node.attribute[0].ints.clone()),
        _ => panic!("Constant of unsupported type {}", node.attribute[0].name),
    };
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_conv(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    metadata: &Vec<onnx::StringStringEntryProto>,
    ev_key: Option<&key::EvaluationKey>,
) {
    let x = get_positional_input(node, namespace, 0).unwrap();
    let w = get_positional_input(node, namespace, 1).unwrap();
    let b = get_positional_input(node, namespace, 2);
    let auto_pad = match get_named_attribute(node, "auto_pad") {
        Some(attr) => match std::str::from_utf8(&attr.s).unwrap() {
            "NOTSET" => autopad::AutoPad::NotSet,
            "SAME_UPPER" => autopad::AutoPad::SameUpper,
            "SAME_LOWER" => autopad::AutoPad::SameLower,
            "VALID" => autopad::AutoPad::Valid,
            _ => panic!(
                "Conv with unrecognized autopad {}",
                std::str::from_utf8(&attr.s).unwrap()
            ),
        },
        None => autopad::AutoPad::NotSet,
    };
    let dilations = match get_named_attribute(node, "dilations") {
        Some(attr) => Some(attr.ints.clone()),
        None => None,
    };
    let group = match get_named_attribute(node, "group") {
        Some(attr) => Some(attr.i),
        None => None,
    };
    let shape = match get_named_attribute(node, "kernel_shape") {
        Some(attr) => Some(attr.ints.clone()),
        None => None,
    };
    let pads = match get_named_attribute(node, "pads") {
        Some(attr) => Some(attr.ints.clone()),
        None => None,
    };
    let strides = match get_named_attribute(node, "strides") {
        Some(attr) => Some(attr.ints.clone()),
        None => None,
    };
    let scale_hint = get_metadatum(metadata, &format!("{}_scale", node.name));
    let output = conv::conv(
        x, w, b, auto_pad, dilations, group, shape, pads, strides, scale_hint, ev_key,
    );
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_flatten(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let input = get_positional_input(node, namespace, 0).unwrap();
    let axis = match get_named_attribute(node, "axis") {
        Some(attr) => attr.i,
        None => 1,
    };
    let output = flatten::flatten(input, axis);
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_gemm(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    metadata: &Vec<onnx::StringStringEntryProto>,
    ev_key: Option<&key::EvaluationKey>,
) {
    let a = get_positional_input(node, namespace, 0).unwrap();
    let b = get_positional_input(node, namespace, 1).unwrap();
    let c = get_positional_input(node, namespace, 2);
    let alpha = match get_named_attribute(node, "alpha") {
        Some(attr) => attr.f as f64,
        None => 1.,
    };
    let beta = match get_named_attribute(node, "beta") {
        Some(attr) => attr.f as f64,
        None => 1.,
    };
    let trans_a = match get_named_attribute(node, "transA") {
        Some(attr) => attr.i,
        None => 0,
    };
    let trans_b = match get_named_attribute(node, "transB") {
        Some(attr) => attr.i,
        None => 0,
    };
    let scale_hint = get_metadatum(metadata, &format!("{}_scale", node.name));
    let output = gemm::gemm(a, b, c, alpha, beta, trans_a, trans_b, scale_hint, ev_key);
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_matmul(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    metadata: &Vec<onnx::StringStringEntryProto>,
    ev_key: Option<&key::EvaluationKey>,
) {
    let a = get_positional_input(node, namespace, 0).unwrap();
    let b = get_positional_input(node, namespace, 1).unwrap();
    let scale_hint = get_metadatum(metadata, &format!("{}_scale", node.name));
    let output = matmul::matmul(a, b, scale_hint, ev_key);
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_mul(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    ev_key: Option<&key::EvaluationKey>,
) {
    if node.input[0] == node.input[1] {
        // take shortcut via elementwise square
        let mut a = get_positional_input(node, namespace, 0).unwrap();

        let max = match &a {
            TensorCover::Clear(c) => c
                .get_values()
                .iter()
                .fold(f64::NEG_INFINITY, |x, &y| x.max(y)),
            TensorCover::Encrypted(e) => e
                .get_values()
                .iter()
                .map(|x| x.encoder.get_min().abs().max(x.encoder.get_max().abs()))
                .fold(f64::NEG_INFINITY, |x, y| x.max(y)),
            _ => panic!("Multiplication of unsupported tensor data types"),
        };

        a.apply_elementwise_operator(ElementwiseOperator::Square {
            upper_bound: max * max,
        });

        namespace.insert(node.output[0].clone(), a);
    } else {
        let a = get_positional_input(node, namespace, 0).unwrap();
        let b = get_positional_input(node, namespace, 1).unwrap();
        let output = mul::mul(a, b, ev_key);
        namespace.insert(node.output[0].clone(), output);
    }
}

pub fn execute_pad(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    ev_key: Option<&key::EvaluationKey>,
) {
    let input = get_positional_input(node, namespace, 0).unwrap();
    let pads = IntegerTensor::peel(get_positional_input(node, namespace, 1).unwrap()).unwrap();
    let constant = match get_positional_input(node, namespace, 2) {
        Some(x) => ClearTensor::peel(x).unwrap().get_values()[0],
        None => 0.,
    };
    let mode = match get_named_attribute(&node, "mode") {
        Some(attr) => match std::str::from_utf8(&attr.s).unwrap() {
            "constant" => pad::PadMode::Constant,
            "reflect" => pad::PadMode::Reflect,
            "edge" => pad::PadMode::Edge,
            _ => panic!(
                "Pad with unrecognized mode {}",
                std::str::from_utf8(&attr.s).unwrap()
            ),
        },
        None => pad::PadMode::Constant,
    };
    let output = pad::pad(input, pads, mode, constant, ev_key);
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_relu(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let mut input = get_positional_input(node, namespace, 0).unwrap();
    let max_hint = get_metadatum(metadata, &format!("{}_max", node.name));
    let max_hint = match max_hint {
        Some(hint) => hint,
        None => match &input {
            TensorCover::Clear(c) => c.get_values().iter().fold(1.0_f64, |acc, x| acc.max(*x)),
            TensorCover::Encrypted(e) => e
                .get_values()
                .iter()
                .fold(1.0_f64, |acc, x| acc.max(x.encoder.get_max())),
            _ => panic!("Relu with unsupported tensor type"),
        },
    };
    input.apply_elementwise_operator(ElementwiseOperator::Relu { max_hint });
    namespace.insert(node.output[0].clone(), input);
}

pub fn execute_reshape(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let input = get_positional_input(node, namespace, 0).unwrap();
    let shape = IntegerTensor::peel(get_positional_input(node, namespace, 1).unwrap()).unwrap();
    let allow_zero = match get_named_attribute(&node, "allowzero") {
        Some(attr) => attr.i,
        None => 0,
    };
    let output = reshape::reshape(input, shape, allow_zero);
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_sigmoid(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let mut input = get_positional_input(node, namespace, 0).unwrap();
    input.apply_elementwise_operator(ElementwiseOperator::Sigmoid);
    namespace.insert(node.output[0].clone(), input);
}

pub fn execute_tanh(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let mut input = get_positional_input(node, namespace, 0).unwrap();
    input.apply_elementwise_operator(ElementwiseOperator::Tanh);
    namespace.insert(node.output[0].clone(), input);
}

pub fn execute_transpose(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let input = get_positional_input(node, namespace, 0).unwrap();
    let axes_perm = match get_named_attribute(&node, "perm") {
        Some(attr) => attr.ints.clone(),
        None => (0..input.get_shape().len() as i64).rev().collect(),
    };
    let output = transpose::transpose(input, axes_perm);
    namespace.insert(node.output[0].clone(), output);
}

pub fn execute_unsqueeze(
    node: &onnx::NodeProto,
    namespace: &mut HashMap<String, TensorCover>,
    _metadata: &Vec<onnx::StringStringEntryProto>,
    _ev_key: Option<&key::EvaluationKey>,
) {
    let input = get_positional_input(node, namespace, 0).unwrap();
    let axes = IntegerTensor::peel(get_positional_input(node, namespace, 1).unwrap()).unwrap();
    let output = unsqueeze::unsqueeze(input, axes);
    namespace.insert(node.output[0].clone(), output);
}
