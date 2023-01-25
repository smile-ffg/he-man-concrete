use crate::tensor::*;

pub enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

pub fn get_auto_pad(
    auto_pad: AutoPad,
    pads: Option<Vec<i64>>,
    kernel_shape: &Vec<usize>,
) -> IntegerTensor {
    /* perform auto padding as defined for onnx operators Conv and AveragePool */

    // retrieve default pad value
    let mut pads = match pads {
        Some(x) => x,
        None => vec![0; kernel_shape.len() + 2],
    };

    match auto_pad {
        AutoPad::NotSet => {
            // add zero paddings for batch size and channel
            let pads_end = pads.split_off(kernel_shape.len());
            pads = [&[0, 0][..], &pads[..], &[0, 0][..], &pads_end[..]].concat();
            IntegerTensor::new(vec![pads.len()], pads)
        }
        AutoPad::SameUpper | AutoPad::SameLower => {
            // find auto padding
            let total_pads: Vec<i64> = kernel_shape.iter().map(|x| (x - 1) as i64).collect();
            let remainder: Vec<i64> = total_pads.iter().map(|x| x % 2).collect();
            let half_floored: Vec<i64> = total_pads.iter().map(|x| x / 2).collect();
            let half_with_remainder: Vec<i64> = half_floored
                .iter()
                .zip(remainder.iter())
                .map(|(d, m)| d + m)
                .collect::<Vec<i64>>();

            // add uneven padding remainder either on top or bottom
            let auto_pads = if let AutoPad::SameUpper = auto_pad {
                [
                    &[0, 0][..],
                    &half_floored[..],
                    &[0, 0][..],
                    &half_with_remainder[..],
                ]
                .concat()
            } else {
                [
                    &[0, 0][..],
                    &half_with_remainder[..],
                    &[0, 0][..],
                    &half_floored[..],
                ]
                .concat()
            };
            IntegerTensor::new(vec![auto_pads.len()], auto_pads)
        }
        AutoPad::Valid => {
            let pads_len = (kernel_shape.len() + 2) * 2;
            IntegerTensor::new(vec![pads_len], vec![0; pads_len])
        }
    }
}
