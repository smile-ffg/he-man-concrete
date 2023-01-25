/*
    Macros to allow binary operations on tensors to use numpy style broadcasting

    Tensor operation must be a callable with signature
        f(a: &T, b: &U, a_off: usize, b_off: usize, mut result: V, [opt: &OptType]) -> V
    where
        a,b are inputs of type T,U
        result is the output of type V which f must modify
        a_off,b_off are offsets from which input values must be accessed in f
        T,U are from {ClearTensor, EncryptedTensor, IntegerTensor}
        one further optional borrow opt of type OptType may be accepted

    The macro is invoked via
        broadcast!(f, &a, T, &b, U, result, V, op_rank [, &opt, OptType])
        broadcast!(f, &a, T, &b, U, result, V [, &opt, OptType])
    where
        op_rank is the number of ranks the operation acts upon
            (e.g. 0 for scalar operations, 2 for matmul)
        op_rank can be omitted for op_rank=0

    Tensors which are result of some broadcast operation can be pre-built with the macro
        build_broadcast_result!(&a, &b, V, op_rank, shape_extension)
        build_broadcast_result!(&a, &b, V)
    where
        shape extension is a slice containing the last op_rank axis dimensions of the result
        op_rank and shape_extension can be omitted for op_rank=0
*/

macro_rules! optional_arg {
    ($opt:expr, $inner_opt:expr) => {
        $inner_opt
    };
    () => {};
}

#[macro_export]
macro_rules! build_broadcast_result {
    ($a:expr, $b:expr, $result_type:ident, $op_rank:expr, $shape_extension:expr) => {{
        // build empty result tensor
        let a_rank = $a.get_rank();
        let b_rank = $b.get_rank();
        let mut result_shape: Vec<usize> = $a.get_shape()[..a_rank - $op_rank]
            .iter()
            .rev()
            .zip($b.get_shape()[..b_rank - $op_rank].iter().rev())
            .map(|(x, y)| *x.max(y))
            .collect();

        if a_rank > b_rank {
            let a_prefix: Vec<&usize> = $a.get_shape()[..a_rank - b_rank].iter().rev().collect();
            result_shape.extend(a_prefix);
        } else if b_rank > a_rank {
            let b_prefix: Vec<&usize> = $b.get_shape()[..b_rank - a_rank].iter().rev().collect();
            result_shape.extend(b_prefix);
        }

        result_shape.reverse();
        result_shape.extend(&$shape_extension);

        $result_type::empty(result_shape)
    }};

    ($a:expr, $b:expr, $result_type:ident) => {
        build_broadcast_result!($a, $b, $result_type, 0, [])
    };
}

#[macro_export]
macro_rules! broadcast {
    ($f:expr,
    $a:expr,
    $a_type:ident,
    $b:expr,
    $b_type:ident,
    $result:expr,
    $result_type:ident,
    $op_rank:expr
    $(, $opt:expr, $opt_type:ident)?) => {{
        fn recursive_broadcast(
            a: &$a_type,
            b: &$b_type,
            a_off: usize,
            b_off: usize,
            n: usize,
            mut result: $result_type
            $(,opt: &$opt_type)?
        ) -> $result_type {
            let max_rank = a.get_rank().max(b.get_rank());
            if n == max_rank - $op_rank {
                // call operation
                return $f(a, b, a_off, b_off, result $(, optional_arg!($opt, opt))?);
            } else {
                // recursive iteration over tensor dims
                let a_rank_diff = max_rank - a.get_rank();
                let b_rank_diff = max_rank - b.get_rank();

                // apply numpy broadcastin rules
                let a_len = if n < a_rank_diff {
                    1
                } else {
                    a.get_shape()[n - a_rank_diff]
                };

                let b_len = if n < b_rank_diff {
                    1
                } else {
                    b.get_shape()[n - b_rank_diff]
                };

                let len = a_len.max(b_len);
                for i in 0..len {
                    let new_a_off = if a_len == 1 {
                        a_off
                    } else {
                        a_off + i * a.get_shape()[n - a_rank_diff + 1..].iter().product::<usize>()
                    };

                    let new_b_off = if b_len == 1 {
                        b_off
                    } else {
                        b_off + i * b.get_shape()[n - b_rank_diff + 1..].iter().product::<usize>()
                    };

                    result = recursive_broadcast(
                        a,
                        b,
                        new_a_off,
                        new_b_off,
                        n + 1,
                        result
                        $(, optional_arg!($opt, opt))?
                    );
                }

                result
            }
        }

        // check broadcast compatibility
        let a_rank = $a.get_rank();
        let b_rank = $b.get_rank();
        if !$a.get_shape()[..a_rank - $op_rank]
            .iter().rev()
            .zip($b.get_shape()[..b_rank - $op_rank].iter().rev())
            .all(|(x,y)| *x == *y || *x == 1 || *y == 1) {
                panic!("Shapes {:?} and {:?} cannot be broadcast together", $a.get_shape(), $b.get_shape());
            }

        // start recursion
        recursive_broadcast(
            &$a,
            &$b,
            0,
            0,
            0,
            $result
            $(, $opt)?
        )
    }};

    ($f:expr,
    $a:expr,
    $a_type:ident,
    $b:expr,
    $b_type:ident,
    $result:expr,
    $result_type:ident
    $(, $opt:expr, $opt_type:ident)?) => {
        broadcast!($f, $a, $a_type, $b, $b_type, $result, $result_type, 0 $(, $opt, $opt_type)?)
    };
}

#[cfg(test)]
mod tests {
    use crate::tensor::*;

    #[test]
    fn test_build_broadcast_result() {
        let a = ClearTensor::empty(vec![2, 3, 1]);
        let b = ClearTensor::empty(vec![1, 3]);
        let output = build_broadcast_result!(a, b, ClearTensor);
        assert_eq!(output.get_shape(), &vec![2, 3, 3]);
    }

    #[test]
    fn test_build_broadcast_result_op_rank() {
        let a = ClearTensor::empty(vec![4, 1, 7]);
        let b = ClearTensor::empty(vec![2, 4, 3, 7]);
        let output = build_broadcast_result!(a, b, ClearTensor, 2, [6, 8]);
        assert_eq!(output.get_shape(), &vec![2, 4, 6, 8]);
    }

    fn perform_max(
        a: &ClearTensor,
        b: &ClearTensor,
        a_off: usize,
        b_off: usize,
        mut result: ClearTensor,
    ) -> ClearTensor {
        result.push(a.get_values()[a_off].max(b.get_values()[b_off]));

        result
    }

    #[test]
    fn test_broadcast_unidirectional() {
        let a = ClearTensor::new(
            vec![2, 3, 2],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        let b = ClearTensor::new(vec![3, 1], vec![1., 12., 6.]);
        let result = ClearTensor::empty(vec![2, 3, 2]);
        let output = broadcast!(
            perform_max,
            a,
            ClearTensor,
            b,
            ClearTensor,
            result,
            ClearTensor
        );
        assert_eq!(
            output.get_values(),
            &vec![1., 2., 12., 12., 6., 6., 7., 8., 12., 12., 11., 12.]
        );
    }

    #[test]
    fn test_broadcast_multidirectional() {
        let a = ClearTensor::new(vec![2, 1, 3], vec![1., 2., 4., 1., 3., 9.]);
        let b = ClearTensor::new(vec![3, 1], vec![8., 5., 2.]);
        let result = ClearTensor::empty(vec![2, 3, 3]);
        let output = broadcast!(
            perform_max,
            a,
            ClearTensor,
            b,
            ClearTensor,
            result,
            ClearTensor
        );
        assert_eq!(
            output.get_values(),
            &vec![8., 8., 8., 5., 5., 5., 2., 2., 4., 8., 8., 9., 5., 5., 9., 2., 3., 9.]
        );
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_shape() {
        let a = ClearTensor::new(vec![2, 3], vec![1., 2., 4., 1., 3., 9.]);
        let b = ClearTensor::new(vec![2], vec![10., 20.]);
        let result = ClearTensor::empty(vec![2, 3]);
        broadcast!(
            perform_max,
            a,
            ClearTensor,
            b,
            ClearTensor,
            result,
            ClearTensor
        );
    }

    fn perform_thresholded_negation(
        a: &ClearTensor,
        b: &ClearTensor,
        a_off: usize,
        b_off: usize,
        mut result: ClearTensor,
        threshold: &f64,
    ) -> ClearTensor {
        if b.get_values()[b_off] > *threshold {
            result.push(-a.get_values()[a_off]);
        } else {
            result.push(a.get_values()[a_off]);
        }

        result
    }

    #[test]
    fn test_broadcast_optional() {
        let a = ClearTensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = ClearTensor::new(vec![1, 3], vec![2., 5., 3.]);

        let result_1 = ClearTensor::empty(vec![2, 3]);
        let output_1 = broadcast!(
            perform_thresholded_negation,
            a,
            ClearTensor,
            b,
            ClearTensor,
            result_1,
            ClearTensor,
            &2.5,
            f64
        );
        assert_eq!(output_1.get_values(), &vec![1., -2., -3., 4., -5., -6.]);

        let result_2 = ClearTensor::empty(vec![2, 3]);
        let output_2 = broadcast!(
            perform_thresholded_negation,
            a,
            ClearTensor,
            b,
            ClearTensor,
            result_2,
            ClearTensor,
            &3.5,
            f64
        );
        assert_eq!(output_2.get_values(), &vec![1., -2., 3., 4., -5., 6.]);
    }

    fn perform_matrix_inner_product(
        a: &ClearTensor,
        b: &ClearTensor,
        a_off: usize,
        b_off: usize,
        mut result: ClearTensor,
    ) -> ClearTensor {
        let nm = a.get_shape()[a.get_rank() - 2] * a.get_shape()[a.get_rank() - 1];
        let frobenius = a.get_values()[a_off..a_off + nm]
            .iter()
            .zip(b.get_values()[b_off..b_off + nm].iter())
            .map(|(x, y)| x * y)
            .sum();

        result.push(frobenius);

        result
    }

    #[test]
    fn test_broadcast_op_rank() {
        let a = ClearTensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = ClearTensor::new(
            vec![3, 2, 3],
            vec![
                1., 2., 3., 4., 5., 6., -1., -2., -3., -4., -5., -6., 1., -2., 3., -4., 5., -6.,
            ],
        );
        let result = ClearTensor::empty(vec![3]);
        let output = broadcast!(
            perform_matrix_inner_product,
            a,
            ClearTensor,
            b,
            ClearTensor,
            result,
            ClearTensor,
            2
        );
        assert_eq!(output.get_values(), &vec![91., -91., -21.]);
    }
}
