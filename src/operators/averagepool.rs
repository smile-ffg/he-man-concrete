use crate::autopad;
use crate::cfg;
use crate::iterate_multi_index;
use crate::key;
use crate::multiindex::flatten_multi_index;
use crate::operators::pad::*;
use crate::tensor::*;
use crate::tensorcover::*;

use concrete::*;
use itertools::izip;

struct AveragePoolParams {
    _ceil_mode: usize,
    count_include_pad: usize,
    kernel_shape: Vec<usize>,
    strides: Vec<usize>,
}

pub fn average_pool(
    x: TensorCover,
    auto_pad: autopad::AutoPad,
    ceil_mode: Option<i64>,
    count_include_pad: Option<i64>,
    kernel_shape: Vec<i64>,
    pads: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    ev_key: Option<&key::EvaluationKey>,
) -> TensorCover {
    // TODO respect effect of strides on non-divisible input shape and ceil_mode

    // unwrap attributes
    let p = AveragePoolParams {
        _ceil_mode: match ceil_mode {
            Some(x) => x as usize,
            None => 1,
        },
        count_include_pad: match count_include_pad {
            Some(x) => x as usize,
            None => 1,
        },
        kernel_shape: kernel_shape.iter().map(|x| *x as usize).collect(),
        strides: match strides {
            Some(x) => x.into_iter().map(|y| y as usize).collect(),
            None => vec![1; x.get_rank() - 2],
        },
    };

    // bootstrap encrypted inputs
    let x = if let TensorCover::Encrypted(mut x_e) = x {
        x_e.evaluate_bootstrap(cfg::WEIGHT_PRECISION + 1, ev_key.unwrap());
        x_e.cover()
    } else {
        x
    };

    // add padding
    let pad_tensor = autopad::get_auto_pad(auto_pad, pads, &p.kernel_shape);
    let x = pad(x, pad_tensor.clone(), PadMode::Constant, 0., ev_key);

    // construct result tensor
    let result_shape: Vec<usize> = izip!(x.get_shape()[2..].iter(), &p.kernel_shape, &p.strides)
        .map(|(in_dim, kernel_dim, stride)| (in_dim - kernel_dim) / stride + 1)
        .collect();
    let mut result = match &x {
        TensorCover::Clear(_x_c) => {
            ClearTensor::empty([&[x.get_shape()[0], x.get_shape()[1]], &result_shape[..]].concat())
                .cover()
        }
        TensorCover::Encrypted(_x_e) => EncryptedTensor::empty(
            [&[x.get_shape()[0], x.get_shape()[1]], &result_shape[..]].concat(),
        )
        .cover(),
        _ => panic!("AveragePool of unsupported tensor data types"),
    };

    // pool
    for batch_ind in 0..x.get_shape()[0] {
        for channel_ind in 0..x.get_shape()[1] {
            result = iterate_multi_index!(
                average_pool_inner,
                result_shape,
                result,
                &x,
                &batch_ind,
                &channel_ind,
                &p,
                &pad_tensor
            );
        }
    }

    result
}

fn average_pool_inner(
    outer_multi_index: &Vec<usize>,
    result: TensorCover,
    x: &TensorCover,
    batch_ind: &usize,
    channel_ind: &usize,
    p: &AveragePoolParams,
    pad_tensor: &IntegerTensor,
) -> TensorCover {
    // determine divisor for averaging
    let div: usize = match p.count_include_pad {
        0 => p.kernel_shape.iter().product(),
        _ => {
            let kernel_rank = p.kernel_shape.len();

            // subtract overlap of kernel with padding from divisor
            let overlap: Vec<i64> = izip!(
                outer_multi_index,
                x.get_shape()[2..].iter(),
                &p.strides,
                &p.kernel_shape,
                pad_tensor.get_values()[2..2 + kernel_rank].iter(),
                pad_tensor.get_values()[4 + kernel_rank..4 + kernel_rank * 2].iter()
            )
            .map(|(ind, dim, s, k, p1, p2)| {
                (p1 - (ind * s) as i64).max(0)
                    + ((ind * s + k - 1) as i64 - (*dim as i64 - p2 - 1)).max(0)
            })
            .collect();

            p.kernel_shape
                .iter()
                .zip(overlap.iter())
                .map(|(k, o)| k - *o as usize)
                .product()
        }
    };

    // disambiguate tensor types
    match &x {
        TensorCover::Clear(x_c) => {
            let mut result_c = ClearTensor::peel(result).unwrap();
            // initialise sum value to zero
            let mut val = 0.;

            // iterate sum
            val = iterate_multi_index!(
                average_pool_op_clear,
                &p.kernel_shape,
                val,
                &x_c,
                outer_multi_index,
                batch_ind,
                channel_ind,
                p
            );
            result_c.push(val / div as f64);
            result_c.cover()
        }
        TensorCover::Encrypted(x_e) => {
            let mut result_e = EncryptedTensor::peel(result).unwrap();

            let multiplier = 1. / div as f64;

            // initialise sum value to throwaway
            let mut val = LWE::zero(x_e.get_values()[0].dimension).unwrap();
            let mut lower_bound_sum = 0.;

            // iterate sum
            val = iterate_multi_index!(
                average_pool_op_encrypted,
                &p.kernel_shape,
                val,
                &x_e,
                outer_multi_index,
                batch_ind,
                channel_ind,
                p,
                &multiplier,
                &mut lower_bound_sum
            );

            result_e.push(val);
            result_e.cover()
        }
        _ => unreachable!(),
    }
}

fn flatten_inner_index(
    inner_multi_index: &Vec<usize>,
    outer_multi_index: &Vec<usize>,
    x_shape: &Vec<usize>,
    batch_ind: usize,
    channel_ind: usize,
    p: &AveragePoolParams,
) -> usize {
    let x_multi_index = [
        &[batch_ind, channel_ind][..],
        &izip!(inner_multi_index, outer_multi_index, &p.strides)
            .map(|(kernel_ind, result_ind, stride)| kernel_ind + result_ind * stride)
            .collect::<Vec<usize>>()[..],
    ]
    .concat();

    let x_ind = flatten_multi_index(&x_multi_index, x_shape);
    x_ind
}

fn average_pool_op_clear(
    inner_multi_index: &Vec<usize>,
    mut val: f64,
    x: &ClearTensor,
    outer_multi_index: &Vec<usize>,
    batch_ind: &usize,
    channel_ind: &usize,
    p: &AveragePoolParams,
) -> f64 {
    // get correct indices into tensors based on convolution parameters
    let x_ind = flatten_inner_index(
        inner_multi_index,
        outer_multi_index,
        x.get_shape(),
        *batch_ind,
        *channel_ind,
        p,
    );

    // perform average pool operation
    val += x.get_values()[x_ind];
    val
}

fn average_pool_op_encrypted(
    inner_multi_index: &Vec<usize>,
    mut val: LWE,
    x: &EncryptedTensor,
    outer_multi_index: &Vec<usize>,
    batch_ind: &usize,
    channel_ind: &usize,
    p: &AveragePoolParams,
    multiplier: &f64,
    lower_bound_sum: &mut f64,
) -> LWE {
    // get correct indices into tensors based on convolution parameters
    let x_ind = flatten_inner_index(
        inner_multi_index,
        outer_multi_index,
        x.get_shape(),
        *batch_ind,
        *channel_ind,
        p,
    );
    let inner_ind = flatten_multi_index(inner_multi_index, &p.kernel_shape);

    // perform average pool operation
    let term = x.get_values()[x_ind]
        .mul_constant_with_padding(*multiplier, 1., cfg::WEIGHT_PRECISION)
        .unwrap();
    *lower_bound_sum += term.encoder.get_min();

    if inner_ind == 0 {
        val = term;
    } else {
        match val.add_with_new_min_inplace(&term, *lower_bound_sum / (inner_ind + 1) as f64) {
            Ok(_) => (),
            Err(error) => match error {
                concrete::error::CryptoAPIError::NoNoiseInCiphertext { .. } => (),
                _ => panic!("Error when adding ciphertexts: {}", error),
            },
        }
    }

    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_averagepool() {
        let x = ClearTensor::new(
            vec![1, 1, 5, 5],
            vec![
                4., 4., 8., 8., 12., 4., 8., 12., 8., 4., 20., 16., 12., 8., 4., -4., -8., -4.,
                -8., -4., 4., 8., 12., 16., 20.,
            ],
        )
        .cover();

        let output = ClearTensor::peel(average_pool(
            x,
            autopad::AutoPad::Valid,
            None,
            None,
            vec![2, 2],
            None,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(output.get_shape(), &vec![1, 1, 4, 4]);
        assert_eq!(
            output.get_values(),
            &vec![5., 8., 9., 8., 12., 12., 10., 6., 6., 4., 2., 0., 0., 2., 4., 6.]
        );
    }
}
