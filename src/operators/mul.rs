use crate::broadcast;
use crate::cfg;
use crate::key::EvaluationKey;
use crate::operators::elementwise::ElementwiseOperator;
use crate::tensor::*;
use crate::tensorcover::*;

pub fn mul(mut a: TensorCover, mut b: TensorCover, ev_key: Option<&EvaluationKey>) -> TensorCover {
    // disambiguate input tensor types
    match (&mut a, &mut b) {
        (TensorCover::Clear(a_c), TensorCover::Clear(b_c)) => {
            let result = build_broadcast_result!(&a_c, &b_c, ClearTensor);
            broadcast!(
                mul_clear,
                &a_c,
                ClearTensor,
                &b_c,
                ClearTensor,
                result,
                ClearTensor
            )
            .cover()
        }
        (TensorCover::Clear(a_c), TensorCover::Encrypted(b_e)) => {
            b_e.evaluate_bootstrap(cfg::WEIGHT_PRECISION + 1, ev_key.unwrap());

            let scale = a_c
                .get_values()
                .iter()
                .fold(f64::NEG_INFINITY, |x, &y| x.max(y));
            let result = build_broadcast_result!(&a_c, &b_e, EncryptedTensor);
            broadcast!(
                mul_mixed,
                &a_c,
                ClearTensor,
                &b_e,
                EncryptedTensor,
                result,
                EncryptedTensor,
                &scale,
                f64
            )
            .cover()
        }
        (TensorCover::Encrypted(a_e), TensorCover::Clear(b_c)) => {
            a_e.evaluate_bootstrap(cfg::WEIGHT_PRECISION + 1, ev_key.unwrap());

            let scale = b_c
                .get_values()
                .iter()
                .fold(f64::NEG_INFINITY, |x, &y| x.max(y));
            let result = build_broadcast_result!(&b_c, &a_e, EncryptedTensor);
            broadcast!(
                mul_mixed,
                &b_c,
                ClearTensor,
                &a_e,
                EncryptedTensor,
                result,
                EncryptedTensor,
                &scale,
                f64
            )
            .cover()
        }
        (TensorCover::Encrypted(a_e), TensorCover::Encrypted(b_e)) => {
            a_e.evaluate_bootstrap(3, ev_key.unwrap());
            b_e.evaluate_bootstrap(3, ev_key.unwrap());

            // get delta post-multiplication
            let delta = broadcast!(
                get_encrypted_delta,
                &a_e,
                EncryptedTensor,
                &b_e,
                EncryptedTensor,
                0.,
                f64
            );

            // perform mul
            let result = build_broadcast_result!(&a_e, &b_e, EncryptedTensor);
            let mut out = broadcast!(
                mul_encrypted,
                &a_e,
                EncryptedTensor,
                &b_e,
                EncryptedTensor,
                result,
                EncryptedTensor,
                ev_key.unwrap(),
                EvaluationKey
            )
            .cover();

            // rescale intervals to delta
            out.apply_elementwise_operator(ElementwiseOperator::SetDelta { delta: delta });

            out
        }
        _ => panic!("Multiplication of unsupported tensor data types"),
    }
}

fn get_encrypted_delta(
    a: &EncryptedTensor,
    b: &EncryptedTensor,
    a_off: usize,
    b_off: usize,
    result: f64,
) -> f64 {
    /* get maximum delta from encrypted mul */

    // check all possible interval bounds from [u, v] * [x, y]
    let u = a.get_values()[a_off].encoder.get_min();
    let v = a.get_values()[a_off].encoder.get_max();
    let x = b.get_values()[b_off].encoder.get_min();
    let y = b.get_values()[b_off].encoder.get_max();

    let lower = (u * x).min(u * y).min(v * x).min(v * y);
    let upper = (u * x).max(u * y).max(v * x).max(v * y);

    result.max(upper - lower)
}

fn mul_clear(
    a: &ClearTensor,
    b: &ClearTensor,
    a_off: usize,
    b_off: usize,
    mut result: ClearTensor,
) -> ClearTensor {
    /* mul clear values */

    result.push(a.get_values()[a_off] * b.get_values()[b_off]);
    result
}

fn mul_mixed(
    a: &ClearTensor,
    b: &EncryptedTensor,
    a_off: usize,
    b_off: usize,
    mut result: EncryptedTensor,
    scale: &f64,
) -> EncryptedTensor {
    /* mul clear and encrypted values */

    result.push(
        b.get_values()[b_off]
            .mul_constant_with_padding(a.get_values()[a_off], *scale, cfg::WEIGHT_PRECISION)
            .unwrap(),
    );
    result
}

fn mul_encrypted(
    a: &EncryptedTensor,
    b: &EncryptedTensor,
    a_off: usize,
    b_off: usize,
    mut result: EncryptedTensor,
    ev_key: &EvaluationKey,
) -> EncryptedTensor {
    /* mul encrypted values */

    result.push(
        a.get_values()[a_off]
            .mul_from_bootstrap(&b.get_values()[b_off], &ev_key.bootstrapping_key)
            .unwrap(),
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul() {
        let a = ClearTensor::new(
            vec![2, 3, 2],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        )
        .cover();
        let b = ClearTensor::new(
            vec![2, 3, 2],
            vec![1., 1., 2., 2., 3., 3., 1., 1., 2., 2., 3., 3.],
        )
        .cover();
        let output = ClearTensor::peel(mul(a, b, None)).unwrap();
        assert_eq!(
            output.get_values(),
            &vec![1., 2., 6., 8., 15., 18., 7., 8., 18., 20., 33., 36.]
        );
    }
}
