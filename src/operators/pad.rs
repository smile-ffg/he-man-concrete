use crate::cfg;
use crate::iterate_multi_index;
use crate::key;
use crate::multiindex::flatten_multi_index;
use crate::tensor::*;
use crate::tensorcover::*;

use concrete::*;
use itertools::izip;

#[derive(PartialEq)]
pub enum PadMode {
    Constant,
    Reflect,
    Edge,
}

pub fn pad(
    mut input: TensorCover,
    pads: IntegerTensor,
    mode: PadMode,
    constant: f64,
    ev_key: Option<&key::EvaluationKey>,
) -> TensorCover {
    let old_shape = input.get_shape();
    let rank = input.get_rank();

    // check if pad dimensions fits rank
    if pads.get_size() != 2 * rank {
        panic!(
            "Pad shape {:?} with incompatible pads {:?}",
            old_shape,
            pads.get_values()
        );
    }

    // check if negative pad dimensions are too small
    let front_slice = &pads.get_values()[..rank];
    let back_slice = &pads.get_values()[rank..];
    for (front, back, dim) in izip!(front_slice, back_slice, old_shape) {
        if (*front).min(0) + (*back).min(0) < -(*dim as i64) {
            panic!(
                "Negative pad {:?} larger than dimension in shape {:?}",
                pads.get_values(),
                old_shape
            );
        }
    }

    // blow up shape by pad dimensions
    let mut new_shape = old_shape.clone();
    new_shape = new_shape
        .iter()
        .zip(pads.get_values()[..rank].iter())
        .map(|(s, p)| (*s as i64 + *p) as usize)
        .collect();
    new_shape = new_shape
        .iter()
        .zip(pads.get_values()[rank..].iter())
        .map(|(s, p)| (*s as i64 + *p) as usize)
        .collect();

    // disambiguate input type
    match &mut input {
        TensorCover::Clear(c) => {
            let mut result = ClearTensor::empty(new_shape.clone());
            iterate_multi_index!(pad_generic, new_shape, result, &c, &pads, &mode, &constant)
                .cover()
        }
        TensorCover::Encrypted(e) => {
            e.evaluate_bootstrap(cfg::WEIGHT_PRECISION + 1, ev_key.unwrap());

            if mode == PadMode::Constant && constant != 0. {
                panic!("Padding encrypted tensor with non-zero constant padding unsupported");
            }

            let mut result = EncryptedTensor::empty(new_shape.clone());
            let mut enc_constant = LWE::zero(e.get_values()[0].dimension).unwrap();
            enc_constant.encoder = e.get_values()[0].encoder.clone();
            enc_constant.encoder.o = 0.;
            enc_constant.variance = e.get_values()[0].variance;
            iterate_multi_index!(
                pad_generic,
                new_shape,
                result,
                &e,
                &pads,
                &mode,
                &enc_constant
            )
            .cover()
        }
        _ => panic!("Padding of unsupported tensor type"),
    }
}

fn pad_generic<T: TensorLike>(
    multi_index: &Vec<usize>,
    mut result: T,
    input: &T,
    pads: &IntegerTensor,
    mode: &PadMode,
    constant: &T::ValueType,
) -> T
where
    <T as TensorLike>::ValueType: Clone,
{
    // determine if this is a padded index
    for (i, ind) in multi_index.iter().enumerate() {
        if (*ind as i64) < pads.get_values()[i]
            || (*ind as i64) >= pads.get_values()[i] + input.get_shape()[i] as i64
        {
            // disambiguate padding mode
            match mode {
                PadMode::Constant => result.push(constant.clone()),
                PadMode::Reflect => {
                    let reflected_ind = flatten_multi_index(
                        &izip!(multi_index, pads.get_values(), input.get_shape())
                            .map(|(i, p, s)| (*i as i64 - *p).rem_euclid(*s as i64) as usize)
                            .collect(),
                        input.get_shape(),
                    );
                    result.push(input.get_values()[reflected_ind].clone());
                }
                PadMode::Edge => {
                    let edge_ind = flatten_multi_index(
                        &izip!(multi_index, pads.get_values(), input.get_shape())
                            .map(|(i, p, s)| (*i as i64 - *p).max(0).min(*s as i64 - 1) as usize)
                            .collect(),
                        input.get_shape(),
                    );
                    result.push(input.get_values()[edge_ind].clone());
                }
            }
            return result;
        }
    }

    // otherwise just push input value
    let ind = flatten_multi_index(
        &multi_index
            .iter()
            .zip(pads.get_values().iter())
            .map(|(i, p)| (*i as i64 - *p) as usize)
            .collect(),
        input.get_shape(),
    );
    result.push(input.get_values()[ind].clone());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_constant() {
        let input = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let pads = IntegerTensor::new(vec![4], vec![1, 2, 0, 1]);
        let output = ClearTensor::peel(pad(input, pads, PadMode::Constant, 0.5, None)).unwrap();
        assert_eq!(output.get_shape(), &vec![4, 5]);
        assert_eq!(
            output.get_values(),
            &vec![
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 2., 0.5, 0.5, 0.5, 3., 4., 0.5, 0.5, 0.5,
                5., 6., 0.5
            ]
        );
    }

    #[test]
    fn test_pad_reflect() {
        let input = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let pads = IntegerTensor::new(vec![4], vec![1, 2, 0, 1]);
        let output = ClearTensor::peel(pad(input, pads, PadMode::Reflect, 0., None)).unwrap();
        assert_eq!(output.get_shape(), &vec![4, 5]);
        assert_eq!(
            output.get_values(),
            &vec![5., 6., 5., 6., 5., 1., 2., 1., 2., 1., 3., 4., 3., 4., 3., 5., 6., 5., 6., 5.]
        );
    }

    #[test]
    fn test_pad_edge() {
        let input = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let pads = IntegerTensor::new(vec![4], vec![1, 2, 0, 1]);
        let output = ClearTensor::peel(pad(input, pads, PadMode::Edge, 0., None)).unwrap();
        assert_eq!(output.get_shape(), &vec![4, 5]);
        assert_eq!(
            output.get_values(),
            &vec![1., 1., 1., 2., 2., 1., 1., 1., 2., 2., 3., 3., 3., 4., 4., 5., 5., 5., 6., 6.]
        );
    }

    #[test]
    fn test_pad_negative() {
        let input = ClearTensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).cover();
        let pads = IntegerTensor::new(vec![4], vec![-1, 1, 0, -2]);
        let output = ClearTensor::peel(pad(input, pads, PadMode::Constant, -0.5, None)).unwrap();
        assert_eq!(output.get_shape(), &vec![1, 2]);
        assert_eq!(output.get_values(), &vec![-0.5, 4.]);
    }

    #[test]
    #[should_panic]
    fn test_pad_incompatible() {
        let input = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let pads = IntegerTensor::new(vec![6], vec![1, 2, 0, 1, 3, 4]);
        pad(input, pads, PadMode::Constant, 0., None);
    }

    #[test]
    #[should_panic]
    fn test_pad_too_negative() {
        let input = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let pads = IntegerTensor::new(vec![4], vec![2, -1, -4, 1]);
        pad(input, pads, PadMode::Constant, 0., None);
    }
}
