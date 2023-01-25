use crate::broadcast;
use crate::key;
use crate::operators::elementwise::ElementwiseOperator;
use crate::tensor::*;
use crate::tensorcover::*;

pub fn add(
    mut a: TensorCover,
    mut b: TensorCover,
    ev_key: Option<&key::EvaluationKey>,
) -> TensorCover {
    // disambiguate input tensor types
    match (&mut a, &mut b) {
        (TensorCover::Clear(a_c), TensorCover::Clear(b_c)) => {
            let result = build_broadcast_result!(&a_c, &b_c, ClearTensor);
            broadcast!(
                add_clear,
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
            b_e.evaluate_bootstrap(1, ev_key.unwrap());

            // get shift correction
            let mut shift = Shift {
                lower: 0.,
                upper: 0.,
            };
            shift = broadcast!(
                get_mixed_shift,
                &a_c,
                ClearTensor,
                &b_e,
                EncryptedTensor,
                shift,
                Shift
            );

            // add tensors
            let result = build_broadcast_result!(&a_c, &b_e, EncryptedTensor);
            let raw_out = broadcast!(
                add_mixed,
                &a_c,
                ClearTensor,
                &b_e,
                EncryptedTensor,
                result,
                EncryptedTensor
            );
            let new_delta = raw_out.get_delta();
            let mut out = raw_out.cover();

            // apply shift correction
            out.apply_elementwise_operator(ElementwiseOperator::ShiftInterval {
                lower_shift: shift.lower,
                upper_shift: shift.upper,
            });

            // apply delta correction
            out.apply_elementwise_operator(ElementwiseOperator::SetDelta { delta: new_delta });

            out
        }
        (TensorCover::Encrypted(a_e), TensorCover::Clear(b_c)) => {
            a_e.evaluate_bootstrap(1, ev_key.unwrap());

            // get shift correction
            let mut shift = Shift {
                lower: 0.,
                upper: 0.,
            };
            shift = broadcast!(
                get_mixed_shift,
                &b_c,
                ClearTensor,
                &a_e,
                EncryptedTensor,
                shift,
                Shift
            );

            //add tensors
            let result = build_broadcast_result!(&b_c, &a_e, EncryptedTensor);
            let raw_out = broadcast!(
                add_mixed,
                &b_c,
                ClearTensor,
                &a_e,
                EncryptedTensor,
                result,
                EncryptedTensor
            );
            let new_delta = raw_out.get_delta();
            let mut out = raw_out.cover();

            // apply shift correction
            out.apply_elementwise_operator(ElementwiseOperator::ShiftInterval {
                lower_shift: shift.lower,
                upper_shift: shift.upper,
            });

            // apply delta correction
            out.apply_elementwise_operator(ElementwiseOperator::SetDelta { delta: new_delta });

            out
        }
        (TensorCover::Encrypted(a_e), TensorCover::Encrypted(b_e)) => {
            // align deltas
            let a_delta = a_e.get_delta();
            let b_delta = b_e.get_delta();
            a_e.apply_elementwise_operator(ElementwiseOperator::SetDelta {
                delta: (a_delta + b_delta),
            });
            b_e.apply_elementwise_operator(ElementwiseOperator::SetDelta {
                delta: (a_delta + b_delta),
            });

            a_e.evaluate_bootstrap(1, ev_key.unwrap());
            b_e.evaluate_bootstrap(1, ev_key.unwrap());

            // add tensors
            let result = build_broadcast_result!(&a_e, &b_e, EncryptedTensor);
            broadcast!(
                add_encrypted,
                &a_e,
                EncryptedTensor,
                &b_e,
                EncryptedTensor,
                result,
                EncryptedTensor
            )
            .cover()
        }
        _ => panic!("Addition of unsupported tensor data types"),
    }
}

struct Shift {
    pub lower: f64,
    pub upper: f64,
}

fn get_mixed_shift(
    a: &ClearTensor,
    b: &EncryptedTensor,
    a_off: usize,
    b_off: usize,
    mut result: Shift,
) -> Shift {
    /* get interval shift correction when adding clear and encrypted tensor */

    result.lower = result
        .lower
        .min(-(a.get_values()[a_off] + b.get_values()[b_off].encoder.get_min()));
    result.upper = result
        .upper
        .max(-(a.get_values()[a_off] + b.get_values()[b_off].encoder.get_max()));

    result
}

fn add_clear(
    a: &ClearTensor,
    b: &ClearTensor,
    a_off: usize,
    b_off: usize,
    mut result: ClearTensor,
) -> ClearTensor {
    /* add clear values */

    result.push(a.get_values()[a_off] + b.get_values()[b_off]);
    result
}

fn add_mixed(
    a: &ClearTensor,
    b: &EncryptedTensor,
    a_off: usize,
    b_off: usize,
    mut result: EncryptedTensor,
) -> EncryptedTensor {
    /* add clear and encrypted values */

    let add = b.get_values()[b_off]
        .add_constant_dynamic_encoder(a.get_values()[a_off])
        .unwrap();

    result.push(add);

    result
}

fn add_encrypted(
    a: &EncryptedTensor,
    b: &EncryptedTensor,
    a_off: usize,
    b_off: usize,
    mut result: EncryptedTensor,
) -> EncryptedTensor {
    /* add encrypted values */

    let add = a.get_values()[a_off]
        .add_with_new_min(
            &b.get_values()[b_off],
            0.5 * (a.get_values()[a_off].encoder.get_min()
                + b.get_values()[b_off].encoder.get_min()),
        )
        .unwrap();

    result.push(add);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
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
        let output = ClearTensor::peel(add(a, b, None)).unwrap();
        assert_eq!(
            output.get_values(),
            &vec![2., 3., 5., 6., 8., 9., 8., 9., 11., 12., 14., 15.]
        );
    }
}
