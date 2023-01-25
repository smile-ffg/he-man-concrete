use crate::execute::*;
use crate::key;
use crate::onnx;
use crate::tensor::*;
use crate::tensorcover::*;

use prost::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Write};
// use std::time::Instant;

pub struct OnnxEngine {
    model: onnx::ModelProto,
    namespace: HashMap<String, TensorCover>,
}

impl OnnxEngine {
    pub fn from_path(model_path: &str) -> OnnxEngine {
        let mut model_file = File::open(model_path).unwrap();
        let mut model_buffer = Vec::new();
        let _message_size = model_file.read_to_end(&mut model_buffer).unwrap();
        let mut cursor: Cursor<Vec<u8>> = Cursor::new(model_buffer);

        OnnxEngine {
            model: prost::Message::decode(&mut cursor).unwrap(),
            namespace: HashMap::new(),
        }
    }

    pub fn save_to_path(self, model_path: &str) {
        let mut model_file = File::create(model_path).unwrap();

        let mut buffer: Vec<u8> = Vec::with_capacity(self.model.encoded_len());
        self.model.encode(&mut buffer).unwrap();

        model_file.write_all(&buffer).unwrap();
    }

    pub fn run<T: TensorCoverable>(
        &mut self,
        inputs: Vec<T>,
        ev_key: Option<&key::EvaluationKey>,
    ) -> Vec<TensorCover> {
        // clear namespace
        self.reset();

        // check number of inputs
        let model_inputs = &self.model.graph.as_ref().unwrap().input;
        if inputs.len() != model_inputs.len() {
            panic!(
                "Model expects {} inputs, got {}",
                model_inputs.len(),
                inputs.len()
            );
        }

        // wrap inputs in TensorCovers
        let covered_inputs: Vec<TensorCover> = inputs.into_iter().map(|x| x.cover()).collect();

        // add inputs to the namespace
        for (input, m_input) in covered_inputs.iter().zip(model_inputs.iter()) {
            self.namespace.insert(m_input.name.clone(), input.clone());
        }

        // add initializer values to the namespace
        let model_inits = &self.model.graph.as_ref().unwrap().initializer;
        for m_init in model_inits {
            self.namespace
                .insert(m_init.name.clone(), TensorCover::from_proto(m_init));
        }

        // sequentially execute graph nodes
        let model_nodes = &self.model.graph.as_ref().unwrap().node;
        for i in 0..model_nodes.len() {
            self.execute_node(i, ev_key);
        }

        // read output values from the namespace
        let model_outputs = &self.model.graph.as_ref().unwrap().output;
        let mut outputs = Vec::with_capacity(model_outputs.len());
        for m_output in model_outputs {
            outputs.push(self.namespace.get(&m_output.name).unwrap().clone());
        }

        outputs
    }

    pub fn calibrate(&mut self, inputs: Vec<ClearTensor>) {
        // populate namespace by running input
        self.run(inputs, None);

        // sequentially calibrate graph nodes
        let model_nodes = &self.model.graph.as_ref().unwrap().node;
        for i in 0..model_nodes.len() {
            self.calibrate_node(i);
        }
    }

    fn reset(&mut self) {
        self.namespace.clear();
    }

    fn execute_node(&mut self, node_index: usize, ev_key: Option<&key::EvaluationKey>) {
        let node = &self.model.graph.as_ref().unwrap().node[node_index];
        // println!("  {}", node.op_type.as_str());
        // let start_time = Instant::now();

        match node.op_type.as_str() {
            "Add" => execute_add(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "AveragePool" => execute_average_pool(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Clip" => execute_clip(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Constant" => execute_constant(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Conv" => execute_conv(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Flatten" => execute_flatten(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Gemm" => execute_gemm(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "MatMul" => execute_matmul(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Mul" => execute_mul(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Pad" => execute_pad(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Relu" => execute_relu(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Reshape" => execute_reshape(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Sigmoid" => execute_sigmoid(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Tanh" => execute_tanh(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Transpose" => execute_transpose(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            "Unsqueeze" => execute_unsqueeze(
                node,
                &mut self.namespace,
                &self.model.metadata_props,
                ev_key,
            ),
            _ => panic!("Unsupported operator {}", node.op_type),
        };
        // println!("      {} seconds", start_time.elapsed().as_secs_f64());
    }

    fn get_metadata_index(&self, key: &str) -> Option<usize> {
        self.model.metadata_props.iter().position(|x| x.key == key)
    }

    fn calibrate_node(&mut self, node_index: usize) {
        let node = &self.model.graph.as_ref().unwrap().node[node_index];

        match node.op_type.as_str() {
            "Conv" | "Gemm" | "MatMul" => {
                let key = format!("{}_scale", node.name);
                let mut scale = match self.get_metadata_index(&key) {
                    Some(index) => self.model.metadata_props[index]
                        .value
                        .parse::<f64>()
                        .unwrap(),
                    None => 0.,
                };

                let out = ClearTensor::peel(self.namespace.get(&node.output[0]).unwrap().clone())
                    .unwrap();

                scale = out
                    .get_values()
                    .iter()
                    .fold(scale, |acc, x| acc.max(x.abs()));

                match self.get_metadata_index(&key) {
                    Some(index) => self.model.metadata_props[index].value = scale.to_string(),
                    None => self
                        .model
                        .metadata_props
                        .push(onnx::StringStringEntryProto {
                            key: key,
                            value: scale.to_string(),
                        }),
                }
            }
            "Relu" => {
                let key = format!("{}_max", node.name);
                let mut max = match self.get_metadata_index(&key) {
                    Some(index) => self.model.metadata_props[index]
                        .value
                        .parse::<f64>()
                        .unwrap(),
                    None => 1.,
                };

                let out = ClearTensor::peel(self.namespace.get(&node.output[0]).unwrap().clone())
                    .unwrap();

                max = out.get_values().iter().fold(max, |acc, x| acc.max(x.abs()));

                match self.get_metadata_index(&key) {
                    Some(index) => self.model.metadata_props[index].value = max.to_string(),
                    None => self
                        .model
                        .metadata_props
                        .push(onnx::StringStringEntryProto {
                            key: key,
                            value: max.to_string(),
                        }),
                }
            }
            _ => (),
        }
    }
}
