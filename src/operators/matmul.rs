use crate::broadcast;
use crate::cfg;
use crate::key;
use crate::operators::transpose::transpose;
use crate::tensor::*;
use crate::tensorcover::*;

pub fn matmul(
    mut a: TensorCover,
    mut b: TensorCover,
    scale_hint: Option<f64>,
    ev_key: Option<&key::EvaluationKey>,
) -> TensorCover {
    let mut rank_a = a.get_rank();
    let mut rank_b = b.get_rank();

    // check if inputs are scalar
    if rank_a == 0 || rank_b == 0 {
        panic!("Matrix multiplication with scalar argument");
    }

    // extend 1D inputs
    let mut is_prepended = false;
    if rank_a == 1 {
        a.set_shape(vec![1, a.get_shape()[0]]);
        is_prepended = true;
        rank_a += 1
    }
    let mut is_appended = false;
    if rank_b == 1 {
        b.set_shape(vec![b.get_shape()[0], 1]);
        is_appended = true;
        rank_b += 1;
    }

    // check if matrix dimensions are compatible
    let m_a = a.get_shape()[rank_a - 2];
    let n_a = a.get_shape()[rank_a - 1];
    let m_b = b.get_shape()[rank_b - 2];
    let n_b = b.get_shape()[rank_b - 1];
    if n_a != m_b {
        panic!(
            "Matrix multiplication with incompatible shapes ({},{}) and ({},{})",
            m_a, n_a, m_b, n_b
        );
    }

    // disambiguate input tensor types
    let mut result = match (&mut a, &mut b) {
        (TensorCover::Clear(a_c), TensorCover::Clear(b_c)) => {
            let result = build_broadcast_result!(&a_c, &b_c, ClearTensor, 2, [m_a, n_b]);
            broadcast!(
                matmul_clear,
                &a_c,
                ClearTensor,
                &b_c,
                ClearTensor,
                result,
                ClearTensor,
                2
            )
            .cover()
        }
        (TensorCover::Clear(a_c), TensorCover::Encrypted(b_e)) => {
            b_e.evaluate_bootstrap(cfg::WEIGHT_PRECISION + 1, ev_key.unwrap());

            let scale = match scale_hint {
                Some(x) => {
                    let scale = x * cfg::SCALE_HINT_SCALING / b_e.get_values()[0].encoder.delta;
                    a_c.get_values()
                        .iter()
                        .fold(scale, |acc, x| acc.max(x.abs()))
                }
                None => get_mixed_scale(&a_c),
            };
            let result = build_broadcast_result!(&a_c, &b_e, EncryptedTensor, 2, [m_a, n_b]);
            broadcast!(
                matmul_mixed,
                &a_c,
                ClearTensor,
                &b_e,
                EncryptedTensor,
                result,
                EncryptedTensor,
                2,
                &scale,
                f64
            )
            .cover()
        }
        (TensorCover::Encrypted(_a_e), TensorCover::Clear(_b_c)) => {
            // calculate product transposed in the last two dimensions
            let mut perm_a: Vec<i64> = (0..rank_a as i64).collect();
            let mut perm_b: Vec<i64> = (0..rank_b as i64).collect();
            perm_a.swap(rank_a - 2, rank_a - 1);
            perm_b.swap(rank_b - 2, rank_b - 1);

            let mut a_t = EncryptedTensor::peel(transpose(a, perm_a)).unwrap();
            let b_t = ClearTensor::peel(transpose(b, perm_b)).unwrap();

            a_t.evaluate_bootstrap(cfg::WEIGHT_PRECISION + 1, ev_key.unwrap());

            let scale = match scale_hint {
                Some(x) => {
                    let scale = x * cfg::SCALE_HINT_SCALING / a_t.get_values()[0].encoder.delta;
                    b_t.get_values()
                        .iter()
                        .fold(scale, |acc, x| acc.max(x.abs()))
                }
                None => get_mixed_scale(&b_t),
            };
            let result = build_broadcast_result!(&b_t, &a_t, EncryptedTensor, 2, [n_b, m_a]);
            let out = broadcast!(
                matmul_mixed,
                &b_t,
                ClearTensor,
                &a_t,
                EncryptedTensor,
                result,
                EncryptedTensor,
                2,
                &scale,
                f64
            )
            .cover();

            // transpose result back
            let rank_out = rank_a.max(rank_b);
            let mut perm_out: Vec<i64> = (0..rank_out as i64).collect();
            perm_out.swap(rank_out - 2, rank_out - 1);
            transpose(out, perm_out)
        }
        _ => panic!("Matrix multiplication of unsupported tensor data types"),
    };

    // remove extensions if one or both inputs were 1D
    let mut final_shape = result.get_shape().clone();
    if is_prepended {
        final_shape.remove(final_shape.len() - 2);
    }
    if is_appended {
        final_shape.pop();
    }
    result.set_shape(final_shape);

    result
}

fn get_mixed_scale(a: &ClearTensor) -> f64 {
    /* return interval scale factor from matmul with this tensor */

    let num_rows = a.get_shape()[0..a.get_rank() - 1].iter().product();
    let num_cols = a.get_shape()[a.get_rank() - 1];
    let mut scale: f64 = 0.;
    // find max sum of absolute row values
    for i in 0..num_rows {
        scale = scale.max(
            a.get_values()[i * num_cols..(i + 1) * num_cols]
                .iter()
                .map(|x| x.abs())
                .sum(),
        );
    }

    scale
}

fn matmul_clear(
    a: &ClearTensor,
    b: &ClearTensor,
    a_off: usize,
    b_off: usize,
    mut result: ClearTensor,
) -> ClearTensor {
    /* matmul between two clear matrices */

    let m = a.get_shape()[a.get_rank() - 2];
    let n = b.get_shape()[b.get_rank() - 1];
    let l = a.get_shape()[a.get_rank() - 1];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.;

            for k in 0..l {
                let a_idx = k + i * l + a_off;
                let b_idx = j + k * n + b_off;

                sum += a.get_values()[a_idx] * b.get_values()[b_idx];
            }
            result.push(sum);
        }
    }

    result
}

fn matmul_mixed(
    a: &ClearTensor,
    b: &EncryptedTensor,
    a_off: usize,
    b_off: usize,
    mut result: EncryptedTensor,
    scale: &f64,
) -> EncryptedTensor {
    /* matmul between clear matrix a and encrypted matrix b */

    let m = a.get_shape()[a.get_rank() - 2];
    let n = b.get_shape()[b.get_rank() - 1];
    let l = a.get_shape()[a.get_rank() - 1];

    for i in 0..m {
        for j in 0..n {
            let a_idx = i * l + a_off;
            let b_idx = j + b_off;

            let mut sum = b.get_values()[b_idx]
                .mul_constant_with_padding(a.get_values()[a_idx], *scale, cfg::WEIGHT_PRECISION)
                .unwrap();
            let mut min_numerator = sum.encoder.get_min() * a.get_values()[a_idx].abs();
            let mut min_denominator = a.get_values()[a_idx].abs();

            for k in 1..l {
                let a_idx = k + i * l + a_off;
                let b_idx = j + k * n + b_off;

                let term = b.get_values()[b_idx]
                    .mul_constant_with_padding(a.get_values()[a_idx], *scale, cfg::WEIGHT_PRECISION)
                    .unwrap();
                min_numerator += term.encoder.get_min() * a.get_values()[a_idx].abs();
                min_denominator += a.get_values()[a_idx].abs();
                match sum.add_with_new_min_inplace(&term, min_numerator / min_denominator) {
                    Ok(_) => (),
                    Err(error) => match error {
                        concrete::error::CryptoAPIError::NoNoiseInCiphertext { .. } => (),
                        _ => panic!("Error when adding ciphertexts: {}", error),
                    },
                }
            }

            result.push(sum);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let b = ClearTensor::new(vec![2, 4], vec![1., 2., 4., 3., 2., 3., 3., 2.]).cover();
        let output = ClearTensor::peel(matmul(a, b, None, None)).unwrap();
        assert_eq!(output.get_shape(), &vec![3, 4]);
        assert_eq!(
            output.get_values(),
            &vec![5., 8., 10., 7., 11., 18., 24., 17., 17., 28., 38., 27.]
        );
    }

    #[test]
    fn test_matmul_1d() {
        let a_1 = ClearTensor::new(vec![2], vec![6., 9.]).cover();
        let b_1 = ClearTensor::new(vec![2, 3], vec![1., 2., 3., -1., -2., -3.]).cover();
        let output_1 = ClearTensor::peel(matmul(a_1, b_1, None, None)).unwrap();
        assert_eq!(output_1.get_shape(), &vec![3]);
        assert_eq!(output_1.get_values(), &vec![-3., -6., -9.]);

        let a_2 = ClearTensor::new(vec![3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).cover();
        let b_2 = ClearTensor::new(vec![3], vec![1., 0., -1.]).cover();
        let output_2 = ClearTensor::peel(matmul(a_2, b_2, None, None)).unwrap();
        assert_eq!(output_2.get_shape(), &vec![3]);
        assert_eq!(output_2.get_values(), &vec![-2., -2., -2.]);

        let a_3 = ClearTensor::new(vec![3], vec![1., 2., 3.]).cover();
        let b_3 = ClearTensor::new(vec![3], vec![4., -5., 6.]).cover();
        let output_3 = ClearTensor::peel(matmul(a_3, b_3, None, None)).unwrap();
        assert_eq!(output_3.get_shape(), &Vec::<usize>::new());
        assert_eq!(output_3.get_values(), &vec![12.]);
    }

    #[test]
    #[should_panic]
    fn test_matmul_scalar() {
        let a = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let b = ClearTensor::new(vec![], vec![1.]).cover();
        matmul(a, b, None, None);
    }

    #[test]
    #[should_panic]
    fn test_matmul_incompatible_shapes() {
        let a = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let b = ClearTensor::new(vec![3, 3], vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]).cover();
        matmul(a, b, None, None);
    }
}
