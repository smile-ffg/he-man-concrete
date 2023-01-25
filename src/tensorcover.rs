use crate::onnx;
use crate::onnx::tensor_proto::DataType;
use crate::operators::elementwise::ElementwiseOperator;
use crate::tensor::*;

use permutation::Permutation;

#[derive(Clone)]
pub enum TensorCover {
    Clear(ClearTensor),
    Encrypted(EncryptedTensor),
    Integer(IntegerTensor),
}

impl TensorCover {
    pub fn from_proto(proto: &onnx::TensorProto) -> TensorCover {
        let shape: Vec<usize> = proto.dims.iter().map(|x| *x as usize).collect();

        // hackish way to match i32 data_type field to DataType enum
        let data_type: DataType = unsafe { std::mem::transmute(proto.data_type) };

        match data_type {
            DataType::Float => {
                let values = if proto.raw_data.len() > 0 {
                    // convert raw data to float
                    let mut idx = 0;
                    let mut f_values = Vec::with_capacity(proto.raw_data.len() / 4);
                    while idx < proto.raw_data.len() {
                        f_values.push(f32::from_le_bytes(
                            proto.raw_data[idx..idx + 4].try_into().unwrap(),
                        ) as f64);
                        idx += 4;
                    }
                    f_values
                } else {
                    proto.float_data.iter().map(|x| *x as f64).collect()
                };
                ClearTensor::new(shape, values).cover()
            }
            DataType::Int64 => {
                let values = if proto.raw_data.len() > 0 {
                    // raw data to long int
                    let mut idx = 0;
                    let mut i_values = Vec::with_capacity(proto.raw_data.len() / 8);
                    while idx < proto.raw_data.len() {
                        i_values.push(i64::from_le_bytes(
                            proto.raw_data[idx..idx + 8].try_into().unwrap(),
                        ));
                        idx += 8;
                    }
                    i_values
                } else {
                    proto.int64_data.clone()
                };
                IntegerTensor::new(shape, values).cover()
            }
            _ => panic!("Unsupported tensor data type: {}", proto.data_type),
        }
    }

    pub fn permute_values(&mut self, perm: &mut Permutation) {
        match self {
            TensorCover::Clear(c) => c.permute_values(perm),
            TensorCover::Encrypted(e) => e.permute_values(perm),
            TensorCover::Integer(i) => i.permute_values(perm),
        };
    }

    pub fn apply_elementwise_operator(&mut self, op: ElementwiseOperator) {
        match self {
            TensorCover::Clear(c) => c.apply_elementwise_operator(op),
            TensorCover::Encrypted(e) => e.apply_elementwise_operator(op),
            TensorCover::Integer(i) => i.apply_elementwise_operator(op),
        };
    }

    pub fn get_rank(&self) -> usize {
        match &self {
            TensorCover::Encrypted(e) => e.get_rank(),
            TensorCover::Clear(c) => c.get_rank(),
            TensorCover::Integer(i) => i.get_rank(),
        }
    }

    pub fn get_size(&self) -> usize {
        match &self {
            TensorCover::Encrypted(e) => e.get_size(),
            TensorCover::Clear(c) => c.get_size(),
            TensorCover::Integer(i) => i.get_size(),
        }
    }

    pub fn get_shape(&self) -> &Vec<usize> {
        match &self {
            TensorCover::Encrypted(e) => e.get_shape(),
            TensorCover::Clear(c) => c.get_shape(),
            TensorCover::Integer(i) => i.get_shape(),
        }
    }

    pub fn set_shape(&mut self, new_shape: Vec<usize>) {
        match self {
            TensorCover::Encrypted(e) => e.set_shape(new_shape),
            TensorCover::Clear(c) => c.set_shape(new_shape),
            TensorCover::Integer(i) => i.set_shape(new_shape),
        }
    }
}

pub trait TensorCoverable {
    fn cover(self) -> TensorCover;
}

impl TensorCoverable for TensorCover {
    fn cover(self) -> TensorCover {
        self
    }
}

impl TensorCoverable for ClearTensor {
    fn cover(self) -> TensorCover {
        TensorCover::Clear(self)
    }
}

impl TensorCoverable for EncryptedTensor {
    fn cover(self) -> TensorCover {
        TensorCover::Encrypted(self)
    }
}

impl TensorCoverable for IntegerTensor {
    fn cover(self) -> TensorCover {
        TensorCover::Integer(self)
    }
}

impl TensorCoverable for f64 {
    fn cover(self) -> TensorCover {
        ClearTensor::new(vec![], vec![self]).cover()
    }
}

impl TensorCoverable for Vec<f64> {
    fn cover(self) -> TensorCover {
        ClearTensor::new(vec![self.len()], self).cover()
    }
}

impl TensorCoverable for i64 {
    fn cover(self) -> TensorCover {
        IntegerTensor::new(vec![], vec![self]).cover()
    }
}

impl TensorCoverable for Vec<i64> {
    fn cover(self) -> TensorCover {
        IntegerTensor::new(vec![self.len()], self).cover()
    }
}
