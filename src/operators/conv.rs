use crate::autopad;
use crate::cfg;
use crate::iterate_multi_index;
use crate::key;
use crate::multiindex::flatten_multi_index;
use crate::operators::add::add;
use crate::operators::pad::*;
use crate::operators::unsqueeze::unsqueeze;
use crate::tensor::*;
use crate::tensorcover::*;

use concrete::*;
use itertools::izip;

struct ConvParams {
    dilations: Vec<usize>,
    group: usize,
    kernel_shape: Vec<usize>,
    strides: Vec<usize>,
}

pub fn conv(
    x: TensorCover,
    w: TensorCover,
    b: Option<TensorCover>,
    auto_pad: autopad::AutoPad,
    dilations: Option<Vec<i64>>,
    group: Option<i64>,
    kernel_shape: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    scale_hint: Option<f64>,
    ev_key: Option<&key::EvaluationKey>,
) -> TensorCover {
    // unwrap attributes
    let p = ConvParams {
        dilations: match dilations {
            Some(x) => x.into_iter().map(|y| y as usize).collect(),
            None => vec![1; x.get_rank() - 2],
        },
        group: match group {
            Some(x) => x as usize,
            None => 1,
        },
        kernel_shape: match kernel_shape {
            Some(x) => x.into_iter().map(|y| y as usize).collect(),
            None => w.get_shape()[2..].to_vec(),
        },
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
    let pads = autopad::get_auto_pad(auto_pad, pads, &p.kernel_shape);
    let x = pad(x, pads, PadMode::Constant, 0., ev_key);

    // construct result tensor
    let result_shape: Vec<usize> = izip!(
        x.get_shape()[2..].iter(),
        &p.kernel_shape,
        &p.dilations,
        &p.strides
    )
    .map(|(in_dim, kernel_dim, dilation, stride)| {
        (in_dim - kernel_dim - (dilation - 1) * (kernel_dim - 1)) / stride + 1
    })
    .collect();
    let mut result = match (&x, &w) {
        (TensorCover::Clear(_x_c), TensorCover::Clear(_w_c)) => {
            ClearTensor::empty([&[x.get_shape()[0], w.get_shape()[0]], &result_shape[..]].concat())
                .cover()
        }
        (TensorCover::Encrypted(_x_e), TensorCover::Clear(_w_c)) => EncryptedTensor::empty(
            [&[x.get_shape()[0], w.get_shape()[0]], &result_shape[..]].concat(),
        )
        .cover(),
        _ => panic!("Convolution of unsupported tensor data types"),
    };

    // convolve
    for batch_ind in 0..x.get_shape()[0] {
        for feature_ind in 0..w.get_shape()[0] {
            result = iterate_multi_index!(
                conv_inner,
                result_shape,
                result,
                &x,
                &w,
                &batch_ind,
                &feature_ind,
                &p,
                &scale_hint
            );
        }
    }

    // add bias
    result = match b {
        Some(b_t) => {
            // broadcast bias across channels
            let b_t = unsqueeze(
                b_t,
                IntegerTensor::new(
                    vec![p.kernel_shape.len()],
                    (1..p.kernel_shape.len() as i64 + 1).collect(),
                ),
            );
            add(result, b_t, ev_key)
        }
        None => result,
    };

    result
}

fn get_encrypted_scale(w: &ClearTensor) -> f64 {
    let filter_len: usize = w.get_shape()[2..].iter().product();
    let filter_count: usize = w.get_shape()[..2].iter().product();
    let mut scale: f64 = 0.;

    for i in 0..filter_count {
        scale = scale.max(
            w.get_values()[i * filter_len..(i + 1) * filter_len]
                .iter()
                .map(|x| x.abs())
                .sum(),
        );
    }

    scale
}

fn conv_inner(
    outer_multi_index: &Vec<usize>,
    result: TensorCover,
    x: &TensorCover,
    w: &TensorCover,
    batch_ind: &usize,
    feature_ind: &usize,
    p: &ConvParams,
    scale_hint: &Option<f64>,
) -> TensorCover {
    // disambiguate tensors types
    match (&x, &w) {
        (TensorCover::Clear(x_c), TensorCover::Clear(w_c)) => {
            let mut result_c = ClearTensor::peel(result).unwrap();
            // initialise sum value to zero
            let mut val = 0.;

            // iterate sum
            val = iterate_multi_index!(
                conv_op_clear,
                &w.get_shape()[1..],
                val,
                &x_c,
                &w_c,
                outer_multi_index,
                batch_ind,
                feature_ind,
                p
            );
            result_c.push(val);
            result_c.cover()
        }
        (TensorCover::Encrypted(x_e), TensorCover::Clear(w_c)) => {
            let mut result_e = EncryptedTensor::peel(result).unwrap();
            // get multiplicative scale of operation
            let scale = match scale_hint {
                Some(x) => {
                    let scale = x * cfg::SCALE_HINT_SCALING / x_e.get_values()[0].encoder.delta;
                    w_c.get_values()
                        .iter()
                        .fold(scale, |acc, x| acc.max(x.abs()))
                }
                None => get_encrypted_scale(&w_c),
            };

            // initialise sum value to throwaway
            let mut val = LWE::zero(x_e.get_values()[0].dimension).unwrap();
            let mut min_numerator = 0.;
            let mut min_denominator = 0.;

            // iterate sum
            val = iterate_multi_index!(
                conv_op_encrypted_x,
                &w.get_shape()[1..],
                val,
                &x_e,
                &w_c,
                outer_multi_index,
                batch_ind,
                feature_ind,
                p,
                &scale,
                &mut min_numerator,
                &mut min_denominator
            );
            result_e.push(val);
            result_e.cover()
        }
        _ => unreachable!(),
    }
}

fn conv_flatten_inner_indices(
    inner_multi_index: &Vec<usize>,
    outer_multi_index: &Vec<usize>,
    x_shape: &Vec<usize>,
    w_shape: &Vec<usize>,
    batch_ind: usize,
    feature_ind: usize,
    p: &ConvParams,
) -> (usize, usize) {
    let x_multi_index = [
        &[
            batch_ind,
            inner_multi_index[0] + (feature_ind % p.group) * x_shape[1] / p.group,
        ][..],
        &izip!(
            inner_multi_index[1..].iter(),
            outer_multi_index,
            &p.dilations,
            &p.strides
        )
        .map(|(dim, r, d, s)| dim * d + r * s)
        .collect::<Vec<usize>>()[..],
    ]
    .concat();
    let w_multi_index = [&[feature_ind][..], &inner_multi_index[..]].concat();

    let x_ind = flatten_multi_index(&x_multi_index, x_shape);
    let w_ind = flatten_multi_index(&w_multi_index, w_shape);

    (x_ind, w_ind)
}

fn conv_op_clear(
    inner_multi_index: &Vec<usize>,
    mut val: f64,
    x: &ClearTensor,
    w: &ClearTensor,
    outer_multi_index: &Vec<usize>,
    batch_ind: &usize,
    feature_ind: &usize,
    p: &ConvParams,
) -> f64 {
    // get correct indices into tensors based on convolution parameters
    let (x_ind, w_ind) = conv_flatten_inner_indices(
        inner_multi_index,
        outer_multi_index,
        x.get_shape(),
        w.get_shape(),
        *batch_ind,
        *feature_ind,
        p,
    );

    // perform convolution operation
    val += x.get_values()[x_ind] * w.get_values()[w_ind];
    val
}

fn conv_op_encrypted_x(
    inner_multi_index: &Vec<usize>,
    mut val: LWE,
    x: &EncryptedTensor,
    w: &ClearTensor,
    outer_multi_index: &Vec<usize>,
    batch_ind: &usize,
    feature_ind: &usize,
    p: &ConvParams,
    scale: &f64,
    min_numerator: &mut f64,
    min_denominator: &mut f64,
) -> LWE {
    // get correct indices into tensors based on convolution parameters
    let (x_ind, w_ind) = conv_flatten_inner_indices(
        inner_multi_index,
        outer_multi_index,
        x.get_shape(),
        w.get_shape(),
        *batch_ind,
        *feature_ind,
        p,
    );
    let inner_ind = flatten_multi_index(inner_multi_index, &w.get_shape()[1..].to_vec());

    // perform convolution operation
    let term = x.get_values()[x_ind]
        .mul_constant_with_padding(w.get_values()[w_ind], *scale, cfg::WEIGHT_PRECISION)
        .unwrap();
    *min_numerator += term.encoder.get_min() * w.get_values()[w_ind].abs();
    *min_denominator += w.get_values()[w_ind].abs();

    if inner_ind == 0 {
        val = term;
    } else {
        match val.add_with_new_min_inplace(&term, *min_numerator / *min_denominator) {
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
    fn test_conv() {
        let x = ClearTensor::new(vec![2, 2, 3, 3], (1..37).map(|x| x as f64).collect()).cover();
        let w = ClearTensor::new(
            vec![3, 2, 2, 2],
            vec![
                1., -1., 1., -1., 2., -1., -1., 2., -3., -1., 1., 3., -1., 1., -1., 1., -2., 1.,
                1., -2., 3., 1., -1., -3.,
            ],
        )
        .cover();
        let b = ClearTensor::new(vec![3], vec![1., 0., -1.]).cover();
        let output = ClearTensor::peel(conv(
            x,
            w,
            Some(b),
            autopad::AutoPad::Valid,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(output.get_shape(), &vec![2, 3, 2, 2]);
        assert_eq!(
            output.get_values(),
            &vec![
                23., 25., 29., 31., 16., 16., 16., 16., -21., -23., -27., -29., 59., 61., 65., 67.,
                16., 16., 16., 16., -57., -59., -63., -65.
            ]
        );
    }

    #[test]
    fn test_conv_pad() {
        let x = ClearTensor::new(vec![1, 1, 3, 3], (1..10).map(|x| x as f64).collect()).cover();
        let w = ClearTensor::new(vec![1, 1, 2, 2], vec![1., -2., 3., -4.]).cover();
        let pads = vec![1, -1, 0, 1];
        let output = ClearTensor::peel(conv(
            x,
            w,
            None,
            autopad::AutoPad::NotSet,
            None,
            None,
            None,
            Some(pads),
            None,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(output.get_shape(), &vec![1, 1, 3, 2]);
        assert_eq!(output.get_values(), &vec![-6., 9., -13., 21., -19., 33.]);
    }

    #[test]
    fn test_conv_autopad() {
        let x = ClearTensor::new(vec![1, 1, 3, 3], (1..10).map(|x| x as f64).collect()).cover();
        let w = ClearTensor::new(vec![1, 1, 2, 3], vec![1., -2., 3., -3., 2., -1.]).cover();
        let output1 = ClearTensor::peel(conv(
            x.clone(),
            w.clone(),
            None,
            autopad::AutoPad::SameUpper,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(output1.get_shape(), &vec![1, 1, 3, 3]);
        assert_eq!(
            output1.get_values(),
            &vec![7., -2., -7., 13., -2., -13., 10., 18., -10.]
        );

        let output2 = ClearTensor::peel(conv(
            x,
            w,
            None,
            autopad::AutoPad::SameLower,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(output2.get_shape(), &vec![1, 1, 3, 3]);
        assert_eq!(
            output2.get_values(),
            &vec![0., -2., 0., 7., -2., -7., 13., -2., -13.]
        );
    }

    #[test]
    fn test_conv_dilations() {
        let x = ClearTensor::new(vec![1, 1, 3, 4], (1..13).map(|x| x as f64).collect()).cover();
        let w = ClearTensor::new(vec![1, 1, 2, 2], vec![1., -2., -3., 4.]).cover();
        let output = ClearTensor::peel(conv(
            x,
            w,
            None,
            autopad::AutoPad::Valid,
            Some(vec![1, 2]),
            None,
            None,
            None,
            None,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(output.get_shape(), &vec![1, 1, 2, 2]);
        assert_eq!(output.get_values(), &vec![8., 8., 8., 8.]);
    }

    #[test]
    fn test_conv_group() {
        let x = ClearTensor::new(vec![1, 2, 3, 3], (1..19).map(|x| x as f64).collect()).cover();
        let w =
            ClearTensor::new(vec![2, 1, 2, 2], vec![1., -1., -1., 1., 2., -1., 0., -1.]).cover();
        let output = ClearTensor::peel(conv(
            x,
            w,
            None,
            autopad::AutoPad::Valid,
            None,
            Some(2),
            None,
            None,
            None,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(output.get_shape(), &vec![1, 2, 2, 2]);
        assert_eq!(
            output.get_values(),
            &vec![0., 0., 0., 0., -5., -5., -5., -5.]
        );
    }

    #[test]
    fn test_conv_strides() {
        let x = ClearTensor::new(vec![1, 1, 4, 3], (1..13).map(|x| x as f64).collect()).cover();
        let w = ClearTensor::new(vec![1, 1, 2, 2], vec![4., 3., 2., 1.]).cover();
        let output = ClearTensor::peel(conv(
            x,
            w,
            None,
            autopad::AutoPad::Valid,
            None,
            None,
            None,
            None,
            Some(vec![2, 1]),
            None,
            None,
        ))
        .unwrap();
        assert_eq!(output.get_shape(), &vec![1, 1, 2, 2]);
        assert_eq!(output.get_values(), &vec![23., 33., 83., 93.]);
    }
}
