use crate::cfg;
use crate::key;
use crate::operators::elementwise::ElementwiseOperator;
use crate::tensorcover::*;

use concrete::*;
use permutation::Permutation;
use serde::{Deserialize, Serialize};

pub trait TensorLike {
    type ValueType;

    fn new(shape: Vec<usize>, values: Vec<Self::ValueType>) -> Self
    where
        Self: Sized;
    fn empty(shape: Vec<usize>) -> Self
    where
        Self: Sized;
    fn peel(a: TensorCover) -> Option<Self>
    where
        Self: Sized;

    fn get_rank(&self) -> usize;
    fn get_size(&self) -> usize;
    fn get_shape(&self) -> &Vec<usize>;
    fn set_shape(&mut self, new_shape: Vec<usize>);
    fn get_values(&self) -> &Vec<Self::ValueType>;
    fn set(&mut self, i: usize, v: Self::ValueType);
    fn push(&mut self, v: Self::ValueType);

    fn permute_values(&mut self, perm: &mut Permutation);
    fn apply_elementwise_operator(&mut self, op: ElementwiseOperator);
}

#[derive(Clone)]
pub struct ClearTensor {
    shape: Vec<usize>,
    values: Vec<f64>,
}

impl TensorLike for ClearTensor {
    type ValueType = f64;

    fn new(shape: Vec<usize>, values: Vec<Self::ValueType>) -> ClearTensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "Tensor shape {:?} does not match data length {}",
            shape,
            values.len()
        );

        ClearTensor {
            shape: shape,
            values: values,
        }
    }

    fn empty(shape: Vec<usize>) -> ClearTensor {
        let data_len = shape.iter().product::<usize>();

        ClearTensor {
            shape: shape,
            values: Vec::with_capacity(data_len),
        }
    }

    fn peel(a: TensorCover) -> Option<ClearTensor> {
        if let TensorCover::Clear(c) = a {
            Some(c)
        } else {
            None
        }
    }

    fn get_rank(&self) -> usize {
        self.shape.len()
    }

    fn get_size(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }

    fn set_shape(&mut self, new_shape: Vec<usize>) {
        self.shape = new_shape;
    }

    fn get_values(&self) -> &Vec<Self::ValueType> {
        &self.values
    }

    fn set(&mut self, i: usize, v: Self::ValueType) {
        self.values[i] = v;
    }

    fn push(&mut self, v: Self::ValueType) {
        self.values.push(v);
    }

    fn permute_values(&mut self, perm: &mut Permutation) {
        perm.apply_slice_in_place(&mut self.values);
    }

    fn apply_elementwise_operator(&mut self, op: ElementwiseOperator) {
        self.values = self.values.iter().map(|x| op.apply(*x)).collect();
    }
}

impl ClearTensor {
    pub fn encrypt(self, encoder: &Encoder, sk: &key::SecretKey) -> EncryptedTensor {
        let enc_values: Vec<LWE> = self
            .values
            .iter()
            .map(|x| LWE::encode_encrypt(&sk.secret_key_in, *x, encoder).unwrap())
            .collect();
        EncryptedTensor::new(self.shape, enc_values)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedTensor {
    shape: Vec<usize>,
    values: Vec<LWE>,
    bootstrap_stack: Vec<ElementwiseOperator>,
}

impl TensorLike for EncryptedTensor {
    type ValueType = LWE;

    fn new(shape: Vec<usize>, values: Vec<Self::ValueType>) -> EncryptedTensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "Tensor shape {:?} does not match data length {}",
            shape,
            values.len()
        );

        EncryptedTensor {
            shape: shape,
            values: values,
            bootstrap_stack: Vec::new(),
        }
    }

    fn empty(shape: Vec<usize>) -> EncryptedTensor {
        let data_len = shape.iter().product::<usize>();

        EncryptedTensor {
            shape: shape,
            values: Vec::with_capacity(data_len),
            bootstrap_stack: Vec::new(),
        }
    }

    fn peel(a: TensorCover) -> Option<EncryptedTensor> {
        if let TensorCover::Encrypted(e) = a {
            Some(e)
        } else {
            None
        }
    }

    fn get_rank(&self) -> usize {
        self.shape.len()
    }

    fn get_size(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }

    fn set_shape(&mut self, new_shape: Vec<usize>) {
        self.shape = new_shape;
    }

    fn get_values(&self) -> &Vec<Self::ValueType> {
        &self.values
    }

    fn set(&mut self, i: usize, v: Self::ValueType) {
        self.values[i] = v;
    }

    fn push(&mut self, v: Self::ValueType) {
        self.values.push(v);
    }

    fn permute_values(&mut self, perm: &mut Permutation) {
        perm.apply_slice_in_place(&mut self.values);
    }

    fn apply_elementwise_operator(&mut self, op: ElementwiseOperator) {
        self.bootstrap_stack.push(op);
    }
}

impl EncryptedTensor {
    pub fn decrypt(self, secret_key: &key::SecretKey) -> ClearTensor {
        // choose appropriate decryption key
        let decrypt_key = if self.values[0].dimension == cfg::LWE_DIMENSION {
            &secret_key.secret_key_in
        } else {
            &secret_key.secret_key_out
        };

        let dec_values: Vec<f64> = self
            .values
            .iter()
            .map(|x| x.decrypt_decode(decrypt_key).unwrap())
            .collect();
        ClearTensor::new(self.shape, dec_values)
    }

    pub fn evaluate_bootstrap(&mut self, padding: usize, eval_key: &key::EvaluationKey) {
        if self.bootstrap_stack.len() == 0 && self.values[0].encoder.nb_bit_padding >= padding {
            return;
        }

        // println!(
        //     "      {} bootstraps, from p={} to p={}, {} ops",
        //     self.get_shape().iter().product::<usize>(),
        //     self.values[0].encoder.nb_bit_padding,
        //     padding,
        //     self.bootstrap_stack.len()
        // );

        // build bootstrap function from stack
        let mut f: Box<dyn Fn(f64) -> f64> = Box::new(|x: f64| x);
        for j in 0..self.bootstrap_stack.len() {
            f = self.bootstrap_stack[j].compose(f);
        }

        // keyswitch to sk_in if necessary
        if self.values[0].dimension != cfg::LWE_DIMENSION {
            self.values = self
                .values
                .iter()
                .map(|x| x.keyswitch(&eval_key.key_switching_key).unwrap())
                .collect();
        }

        for i in 0..self.values.len() {
            // get interval bounds from stack
            let mut bounds = (
                self.values[i].encoder.get_min(),
                self.values[i].encoder.get_max(),
            );
            for j in 0..self.bootstrap_stack.len() {
                bounds = self.bootstrap_stack[j].transform_interval(bounds.0, bounds.1);
            }

            let encoder = Encoder::new(
                bounds.0 * cfg::BOOTSTRAP_INTERVAL_SCALING,
                bounds.1 * cfg::BOOTSTRAP_INTERVAL_SCALING,
                cfg::PRECISION,
                padding,
            )
            .unwrap();

            // bootstrap
            self.values[i] = self.values[i]
                .bootstrap_with_function(&eval_key.bootstrapping_key, &f, &encoder)
                .unwrap();
        }

        // reset stack
        self.bootstrap_stack.clear();
    }

    pub fn get_delta(&self) -> f64 {
        let mut bounds = (
            self.values[0].encoder.get_min(),
            self.values[0].encoder.get_max(),
        );
        for j in 0..self.bootstrap_stack.len() {
            bounds = self.bootstrap_stack[j].transform_interval(bounds.0, bounds.1);
        }

        bounds.1 - bounds.0
    }
}

#[derive(Clone)]
pub struct IntegerTensor {
    shape: Vec<usize>,
    values: Vec<i64>,
}

impl TensorLike for IntegerTensor {
    type ValueType = i64;

    fn new(shape: Vec<usize>, values: Vec<Self::ValueType>) -> IntegerTensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "Tensor shape {:?} does not match data length {}",
            shape,
            values.len()
        );

        IntegerTensor {
            shape: shape,
            values: values,
        }
    }

    fn empty(shape: Vec<usize>) -> IntegerTensor {
        let data_len = shape.iter().product::<usize>();

        IntegerTensor {
            shape: shape,
            values: Vec::with_capacity(data_len),
        }
    }

    fn peel(a: TensorCover) -> Option<IntegerTensor> {
        if let TensorCover::Integer(i) = a {
            Some(i)
        } else {
            None
        }
    }

    fn get_rank(&self) -> usize {
        self.shape.len()
    }

    fn get_size(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }

    fn set_shape(&mut self, new_shape: Vec<usize>) {
        self.shape = new_shape;
    }

    fn get_values(&self) -> &Vec<Self::ValueType> {
        &self.values
    }

    fn set(&mut self, i: usize, v: Self::ValueType) {
        self.values[i] = v;
    }

    fn push(&mut self, v: Self::ValueType) {
        self.values.push(v);
    }

    fn permute_values(&mut self, perm: &mut Permutation) {
        perm.apply_slice_in_place(&mut self.values);
    }

    fn apply_elementwise_operator(&mut self, _op: ElementwiseOperator) {
        panic!("Application of elementwise operator on unsupported tensor type")
    }
}
