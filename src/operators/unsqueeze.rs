use crate::tensor::*;
use crate::tensorcover::*;

pub fn unsqueeze(mut input: TensorCover, axes: IntegerTensor) -> TensorCover {
    //TODO ensure axes values lie in correct range, and have no duplicates

    let new_rank = (input.get_rank() + axes.get_shape()[0]) as i64;

    // convert negative axes entries to positive
    let mut pos_values: Vec<usize> = axes
        .get_values()
        .clone()
        .into_iter()
        .map(|x| ((x + new_rank) % new_rank) as usize)
        .collect();
    pos_values.sort();

    // insert new dimension for each entry in axes
    let mut new_shape = input.get_shape().clone();
    for i in 0..axes.get_shape()[0] {
        new_shape.insert(pos_values[i], 1);
    }
    input.set_shape(new_shape);

    input
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsqueeze_positive_axes() {
        let input = ClearTensor::empty(vec![2, 3, 2]).cover();
        let axes = IntegerTensor::new(vec![4], vec![0, 5, 3, 2]);
        let output = ClearTensor::peel(unsqueeze(input, axes)).unwrap();
        assert_eq!(output.get_shape(), &vec![1, 2, 1, 1, 3, 1, 2]);
    }

    #[test]
    fn test_unsqueeze_negative_axes() {
        let input = ClearTensor::empty(vec![2, 3, 2]).cover();
        let axes = IntegerTensor::new(vec![4], vec![-5, -2, -1, -6]);
        let output = ClearTensor::peel(unsqueeze(input, axes)).unwrap();
        assert_eq!(output.get_shape(), &vec![2, 1, 1, 3, 2, 1, 1]);
    }

    #[test]
    fn test_unsqueeze_mixed_axes() {
        let input = ClearTensor::empty(vec![2, 3, 2]).cover();
        let axes = IntegerTensor::new(vec![4], vec![2, -2, 1, -3]);
        let output = ClearTensor::peel(unsqueeze(input, axes)).unwrap();
        assert_eq!(output.get_shape(), &vec![2, 1, 1, 3, 1, 1, 2]);
    }
}
