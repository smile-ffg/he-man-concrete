use serde::{Deserialize, Serialize};
use tool::compose;

/*
    Currently implemented elementwise ONNX operators:
        Clip
        ReLU
        Sigmoid
        Tanh
*/

#[derive(Clone, Deserialize, Serialize)]
pub enum ElementwiseOperator {
    // ONNX specification operators
    Clip { min: f64, max: f64 },
    Relu { max_hint: f64 },
    Sigmoid,
    Tanh,

    // internal operators
    ScalarMul { scalar: f64 },
    SetDelta { delta: f64 },
    ShiftInterval { lower_shift: f64, upper_shift: f64 },
    Square { upper_bound: f64 },
}

impl ElementwiseOperator {
    pub fn apply(&self, input: f64) -> f64 {
        match self {
            Self::Clip { min, max } => input.max(*min).min(*max),
            Self::Relu { .. } => input.max(0.),
            Self::Sigmoid => 1. / (1. + (-input).exp()),
            Self::Tanh => input.tanh(),
            Self::ScalarMul { scalar } => input * *scalar,
            Self::SetDelta { .. } => input,
            Self::ShiftInterval { .. } => input,
            Self::Square { .. } => input * input,
        }
    }

    pub fn compose(&self, f: Box<dyn Fn(f64) -> f64>) -> Box<dyn Fn(f64) -> f64> {
        match self {
            Self::Clip { min, max } => {
                let min_t = *min;
                let max_t = *max;
                Box::new(compose(move |x| x.max(min_t).min(max_t), f))
            }
            Self::Relu { .. } => Box::new(compose(|x| x.max(0.), f)),
            Self::Sigmoid => Box::new(compose(|x| 1. / (1. + (-x).exp()), f)),
            Self::Tanh => Box::new(compose(|x| x.tanh(), f)),
            Self::ScalarMul { scalar } => {
                let scalar_t = *scalar;
                Box::new(compose(move |x| x * scalar_t, f))
            }
            Self::SetDelta { .. } => f,
            Self::ShiftInterval { .. } => f,
            Self::Square { .. } => Box::new(compose(|x| x * x, f)),
        }
    }

    pub fn transform_interval(&self, lower: f64, upper: f64) -> (f64, f64) {
        match self {
            Self::Clip { min, max } => (*min, *max),
            Self::Relu { max_hint } => (0., *max_hint),
            Self::Sigmoid => (0., 2.),
            Self::Tanh => (-2., 2.),
            Self::ScalarMul { scalar } => {
                if *scalar > 0. {
                    (lower * *scalar, upper * *scalar)
                } else {
                    (upper * *scalar, lower * *scalar)
                }
            }
            Self::SetDelta { delta } => (
                lower * delta / (upper - lower),
                upper * delta / (upper - lower),
            ),
            Self::ShiftInterval {
                lower_shift,
                upper_shift,
            } => (lower + lower_shift, upper + upper_shift),
            Self::Square { upper_bound } => (0., *upper_bound),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;

    #[test]
    fn test_clip() {
        let op = ElementwiseOperator::Clip { min: 2.5, max: 5.5 };
        assert_eq!(op.apply(-3.), 2.5);
        assert_eq!(op.apply(3.5), 3.5);
        assert_eq!(op.apply(14.), 5.5);
        assert_eq!(op.transform_interval(-3., 55.), (2.5, 5.5));
        let mut tensor = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]);
        tensor.apply_elementwise_operator(op);
        assert_eq!(tensor.get_values(), &vec![2.5, 2.5, 3., 4., 5., 5.5]);
    }

    #[test]
    fn test_relu() {
        let op = ElementwiseOperator::Relu { max_hint: 5. };
        assert_eq!(op.apply(-3.), 0.);
        assert_eq!(op.apply(3.5), 3.5);
        assert_eq!(op.apply(14.), 14.);
        assert_eq!(op.transform_interval(-3., 55.), (0., 5.));
        let mut tensor = ClearTensor::new(vec![3, 2], vec![-1., -2., -3., 3., 2., 1.]);
        tensor.apply_elementwise_operator(op);
        assert_eq!(tensor.get_values(), &vec![0., 0., 0., 3., 2., 1.]);
    }

    #[test]
    fn test_sigmoid() {
        let op = ElementwiseOperator::Sigmoid;
        assert_eq!(op.apply(-3.), 1. / (1. + (3.0_f64).exp()));
        assert_eq!(op.apply(3.5), 1. / (1. + (-3.5_f64).exp()));
        assert_eq!(op.apply(14.), 1. / (1. + (-14.0_f64).exp()));
        assert_eq!(op.transform_interval(-3., 55.), (0., 2.));
        let mut tensor = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]);
        tensor.apply_elementwise_operator(op);
        assert_eq!(
            tensor.get_values(),
            &vec![
                1. / (1. + (-1.0_f64).exp()),
                1. / (1. + (-2.0_f64).exp()),
                1. / (1. + (-3.0_f64).exp()),
                1. / (1. + (-4.0_f64).exp()),
                1. / (1. + (-5.0_f64).exp()),
                1. / (1. + (-6.0_f64).exp())
            ]
        );
    }

    #[test]
    fn test_tanh() {
        let op = ElementwiseOperator::Tanh;
        assert_eq!(op.apply(-3.), (-3.0_f64).tanh());
        assert_eq!(op.apply(3.5), (3.5_f64).tanh());
        assert_eq!(op.apply(14.), (14.0_f64).tanh());
        assert_eq!(op.transform_interval(-3., 55.), (-2., 2.));
        let mut tensor = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]);
        tensor.apply_elementwise_operator(op);
        assert_eq!(
            tensor.get_values(),
            &vec![
                (1.0_f64).tanh(),
                (2.0_f64).tanh(),
                (3.0_f64).tanh(),
                (4.0_f64).tanh(),
                (5.0_f64).tanh(),
                (6.0_f64).tanh()
            ]
        );
    }

    #[test]
    fn test_scalar_mul() {
        let op_1 = ElementwiseOperator::ScalarMul { scalar: -2. };
        assert_eq!(op_1.apply(-2.), 4.);
        assert_eq!(op_1.apply(5.), -10.);
        assert_eq!(op_1.transform_interval(-5., 4.), (-8., 10.));
        let mut tensor = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]);
        tensor.apply_elementwise_operator(op_1);
        assert_eq!(tensor.get_values(), &vec![-2., -4., -6., -8., -10., -12.]);

        let op_2 = ElementwiseOperator::ScalarMul { scalar: 3. };
        assert_eq!(op_2.transform_interval(1., 6.), (3., 18.));
    }

    #[test]
    fn test_set_delta() {
        let op = ElementwiseOperator::SetDelta { delta: 4. };
        assert_eq!(op.transform_interval(0., 2.), (0., 4.));
        assert_eq!(op.transform_interval(-9., 3.), (-3., 1.));
        assert_eq!(op.transform_interval(-4., 0.), (-4., 0.));
    }

    #[test]
    fn test_shift_interval() {
        let op = ElementwiseOperator::ShiftInterval {
            lower_shift: 2.,
            upper_shift: -1.,
        };
        assert_eq!(op.transform_interval(0., 4.), (2., 3.));
        assert_eq!(op.transform_interval(-4., 5.), (-2., 4.));
    }

    #[test]
    fn test_square() {
        let op = ElementwiseOperator::Square { upper_bound: 9. };
        assert_eq!(op.apply(-2.), 4.);
        assert_eq!(op.apply(5.), 25.);
        assert_eq!(op.transform_interval(-5., 4.), (0., 9.));
        let mut tensor = ClearTensor::new(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]);
        tensor.apply_elementwise_operator(op);
        assert_eq!(tensor.get_values(), &vec![1., 4., 9., 16., 25., 36.]);
    }

    #[test]
    fn test_composition() {
        let op1 = ElementwiseOperator::Square { upper_bound: 9. };
        let op2 = ElementwiseOperator::ShiftInterval {
            lower_shift: -2.,
            upper_shift: 5.,
        };
        let op3 = ElementwiseOperator::Tanh;

        let mut f: Box<dyn Fn(f64) -> f64> = Box::new(|x: f64| x);
        f = op1.compose(f);
        f = op2.compose(f);
        f = op3.compose(f);

        let mut bounds = (-1., 5.);
        bounds = op1.transform_interval(bounds.0, bounds.1);
        bounds = op2.transform_interval(bounds.0, bounds.1);
        bounds = op3.transform_interval(bounds.0, bounds.1);

        assert_eq!(f(5.), (25.0_f64).tanh());
        assert_eq!(bounds, (-2., 2.));
    }
}
