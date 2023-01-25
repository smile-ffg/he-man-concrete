use crate::iterate_multi_index;
use crate::tensorcover::*;

use permutation::Permutation;

pub fn transpose(mut input: TensorCover, permutation: Vec<i64>) -> TensorCover {
    /* apply general transposition to the input, given as a permutation of axes */

    let old_shape = input.get_shape().clone();
    let data_len = old_shape.iter().product();

    // apply permutation to shape
    let permutation: Vec<usize> = permutation.iter().map(|x| *x as usize).collect();
    let shape_permutation = Permutation::oneline(permutation.clone()).inverse();
    let new_shape = shape_permutation.apply_slice(input.get_shape());

    // build permutation for values
    let mut new_indices: Vec<usize> = Vec::with_capacity(data_len);
    new_indices = iterate_multi_index!(
        transform_index,
        new_shape,
        new_indices,
        &permutation,
        &old_shape
    );

    // permute values
    let mut value_permutation = Permutation::oneline(new_indices).inverse();
    input.permute_values(&mut value_permutation);
    input.set_shape(new_shape);

    input
}

fn transform_index(
    multi_index: &Vec<usize>,
    mut new_indices: Vec<usize>,
    permutation: &Vec<usize>,
    old_shape: &Vec<usize>,
) -> Vec<usize> {
    /* calculate 1d index in transposed shape, from multi index of original shape */

    let mut new_ind = 0;
    for (i, dim) in multi_index.iter().enumerate() {
        let axis_ind = permutation[i];
        let multiplier: usize = old_shape[axis_ind + 1..].iter().product();
        new_ind += dim * multiplier;
    }
    new_indices.push(new_ind);
    new_indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;

    #[test]
    fn test_transpose_partial() {
        let input = ClearTensor::new(
            vec![2, 3, 2],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        )
        .cover();
        let permutation = vec![0, 2, 1];
        let output = ClearTensor::peel(transpose(input, permutation)).unwrap();
        assert_eq!(output.get_shape(), &vec![2, 2, 3]);
        assert_eq!(
            output.get_values(),
            &vec![1., 3., 5., 2., 4., 6., 7., 9., 11., 8., 10., 12.]
        );
    }

    #[test]
    fn test_transpose_full() {
        let input = ClearTensor::new(
            vec![2, 3, 2],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        )
        .cover();
        let permutation = vec![1, 2, 0];
        let output = ClearTensor::peel(transpose(input, permutation)).unwrap();
        assert_eq!(output.get_shape(), &vec![3, 2, 2]);
        assert_eq!(
            output.get_values(),
            &vec![1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12.]
        );
    }
}
