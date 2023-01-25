use crate::tensorcover::*;

pub fn flatten(mut input: TensorCover, axis: i64) -> TensorCover {
    let mut left = input.get_shape().clone();
    let right = left.split_off(axis as usize);

    input.set_shape(vec![left.iter().product(), right.iter().product()]);

    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;

    #[test]
    fn test_flatten() {
        let input = ClearTensor::empty(vec![2, 3, 2, 4, 2]).cover();
        let output = ClearTensor::peel(flatten(input, 2)).unwrap();
        assert_eq!(output.get_shape(), &vec![6, 16]);
    }

    #[test]
    fn test_flatten_zero() {
        let input = ClearTensor::empty(vec![9, 3, 1]).cover();
        let output = ClearTensor::peel(flatten(input, 0)).unwrap();
        assert_eq!(output.get_shape(), &vec![1, 27]);
    }
}
