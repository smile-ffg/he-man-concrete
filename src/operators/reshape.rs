use crate::tensor::*;
use crate::tensorcover::*;

pub fn reshape(mut data: TensorCover, shape: IntegerTensor, allow_zero: i64) -> TensorCover {
    let mut new_shape = shape.get_values().clone();

    // copy zero values from original shape
    if allow_zero == 0 {
        for i in 0..new_shape.len() {
            if new_shape[i] == 0 {
                new_shape[i] = data.get_shape()[i] as i64;
            }
        }
    }

    // determine number of -1 in new shape
    let determinate: Vec<i64> = new_shape.clone().into_iter().filter(|&x| x != -1).collect();
    let num_indeterminate = new_shape.len() - determinate.len();
    if num_indeterminate > 1 {
        panic!("Reshape with more than one indeterminate dimension");
    }

    let new_size = determinate.iter().product::<i64>() as usize;
    let old_size = data.get_size();

    // find indeterminate dimension
    let indeterminate = if new_size == 0 {
        if old_size != 0 {
            panic!(
                "Reshape with incompatible shapes {:?} and {:?}",
                data.get_shape(),
                shape.get_values()
            );
        }
        1
    } else {
        old_size / new_size
    };

    // check compatibility of shapes
    if new_size * indeterminate != old_size {
        panic!(
            "Reshape with incompatible shapes {:?} and {:?}",
            data.get_shape(),
            shape.get_values()
        );
    }

    data.set_shape(
        new_shape
            .iter()
            .map(|&x| if x == -1 { indeterminate } else { x as usize })
            .collect(),
    );

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape() {
        let data = ClearTensor::new(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).cover();
        let shape = IntegerTensor::new(vec![3], vec![6, 2, 2]);
        let output = ClearTensor::peel(reshape(data, shape, 0)).unwrap();
        assert_eq!(output.get_shape(), &vec![6, 2, 2]);
    }

    #[test]
    fn test_reshape_indeterminate() {
        let data = ClearTensor::new(vec![3, 2, 6, 5], (0..180).map(|x| x as f64).collect()).cover();
        let shape = IntegerTensor::new(vec![3], vec![2, -1, 2]);
        let output = ClearTensor::peel(reshape(data, shape, 0)).unwrap();
        assert_eq!(output.get_shape(), &vec![2, 45, 2]);
    }

    #[test]
    fn test_reshape_zero() {
        let data1 =
            ClearTensor::new(vec![7, 4, 4, 2], (0..224).map(|x| x as f64).collect()).cover();
        let shape1 = IntegerTensor::new(vec![4], vec![0, 1, 0, -1]);
        let output1 = ClearTensor::peel(reshape(data1, shape1, 0)).unwrap();
        assert_eq!(output1.get_shape(), &vec![7, 1, 4, 8]);

        let data2 = ClearTensor::new(vec![3, 0, 5, 2], vec![]).cover();
        let shape2 = IntegerTensor::new(vec![4], vec![0, 1, 0, -1]);
        let output2 = ClearTensor::peel(reshape(data2, shape2, 1)).unwrap();
        assert_eq!(output2.get_shape(), &vec![0, 1, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn test_reshape_multiple_indeterminate() {
        let data = ClearTensor::new(vec![3, 2, 6, 5], (0..180).map(|x| x as f64).collect()).cover();
        let shape = IntegerTensor::new(vec![4], vec![2, -1, 2, -1]);
        reshape(data, shape, 0);
    }

    #[test]
    #[should_panic]
    fn test_reshape_incompatible_shape() {
        let data = ClearTensor::new(vec![3, 2, 6, 5], (0..180).map(|x| x as f64).collect()).cover();
        let shape = IntegerTensor::new(vec![3], vec![3, 2, 4, 5]);
        reshape(data, shape, 0);
    }

    #[test]
    #[should_panic]
    fn test_reshape_incompatible_indeterminate() {
        let data = ClearTensor::new(vec![3, 2, 6, 5], (0..180).map(|x| x as f64).collect()).cover();
        let shape = IntegerTensor::new(vec![3], vec![3, -1, 5, 5]);
        reshape(data, shape, 0);
    }
}
