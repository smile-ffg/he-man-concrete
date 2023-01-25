use crate::key;
use crate::operators::add::add;
use crate::operators::elementwise::ElementwiseOperator;
use crate::operators::matmul::matmul;
use crate::operators::transpose::transpose;
use crate::tensorcover::*;

pub fn gemm(
    mut a: TensorCover,
    mut b: TensorCover,
    c: Option<TensorCover>,
    alpha: f64,
    beta: f64,
    trans_a: i64,
    trans_b: i64,
    scale_hint: Option<f64>,
    ev_key: Option<&key::EvaluationKey>,
) -> TensorCover {
    let rank_a = a.get_rank();
    let rank_b = b.get_rank();

    // check if inputs are matrices
    if rank_a != 2 || rank_b != 2 {
        panic!(
            "Gemm with tensors of unsupported rank {} and {}",
            rank_a, rank_b
        );
    }

    // apply scalar multipliers
    if alpha != 1. {
        a.apply_elementwise_operator(ElementwiseOperator::ScalarMul { scalar: alpha });
    }
    if beta != 1. {
        b.apply_elementwise_operator(ElementwiseOperator::ScalarMul { scalar: beta });
    }

    // apply transpositions
    if trans_a != 0 {
        a = transpose(a, vec![1, 0]);
    }
    if trans_b != 0 {
        b = transpose(b, vec![1, 0]);
    }

    // check if matrix dimensions are compatible
    let m_a = a.get_shape()[0];
    let n_a = a.get_shape()[1];
    let m_b = b.get_shape()[0];
    let n_b = b.get_shape()[1];
    if n_a != m_b {
        panic!(
            "Gemm with incompatible shapes ({},{}) and ({},{})",
            m_a, n_a, m_b, n_b
        );
    }

    // matrix multiplication
    let mut result = matmul(a, b, scale_hint, ev_key);

    // optional addition
    match c {
        Some(c) => {
            //TODO should only support unidirectional broadcast, currently multidirectional
            result = add(result, c, ev_key);
        }
        None => (),
    };

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;

    #[test]
    fn test_gemm() {
        let a = ClearTensor::new(vec![2, 2], vec![1., 2., 3., 4.]).cover();
        let b = ClearTensor::new(vec![2, 3], vec![1., 2., -1., -2., 2., 1.]).cover();
        let output = ClearTensor::peel(gemm(a, b, None, 1., 1., 0, 0, None, None)).unwrap();
        assert_eq!(output.get_shape(), &vec![2, 3]);
        assert_eq!(output.get_values(), &vec![-3., 6., 1., -5., 14., 1.]);
    }

    #[test]
    fn test_gemm_c() {
        let a = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let b = ClearTensor::new(vec![2, 2], vec![0., 1., 1., 0.]).cover();
        let c = ClearTensor::new(vec![], vec![5.]).cover();
        let output = ClearTensor::peel(gemm(a, b, Some(c), 1., 1., 0, 0, None, None)).unwrap();
        assert_eq!(output.get_shape(), &vec![3, 2]);
        assert_eq!(output.get_values(), &vec![7., 6., 9., 8., 11., 10.]);
    }

    #[test]
    fn test_gemm_scalar_factor() {
        let a = ClearTensor::new(vec![1, 5], vec![1., 2., 3., 4., 5.]).cover();
        let b = ClearTensor::new(vec![5, 1], vec![1., -1., 2., -3., 5.]).cover();
        let output = ClearTensor::peel(gemm(a, b, None, 2., -1., 0, 0, None, None)).unwrap();
        assert_eq!(output.get_shape(), &vec![1, 1]);
        assert_eq!(output.get_values(), &vec![-36.]);
    }

    #[test]
    fn test_gemm_transpose() {
        let a = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let b = ClearTensor::new(vec![1, 3], vec![-3., -5., 1.]).cover();
        let output = ClearTensor::peel(gemm(a, b, None, 1., 1., 1, 1, None, None)).unwrap();
        assert_eq!(output.get_shape(), &vec![2, 1]);
        assert_eq!(output.get_values(), &vec![-13., -20.]);
    }

    #[test]
    #[should_panic]
    fn test_gemm_non_matrix() {
        let a = ClearTensor::new(
            vec![12],
            vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6.],
        )
        .cover();
        let b = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        gemm(a, b, None, 1., 1., 0, 0, None, None);
    }

    #[test]
    #[should_panic]
    fn test_gemm_incompatible_shapes() {
        let a = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).cover();
        let b = ClearTensor::new(vec![3, 3], vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]).cover();
        gemm(a, b, None, 1., 1., 0, 0, None, None);
    }
}
