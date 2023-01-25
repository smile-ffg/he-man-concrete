use crate::tensorcover::*;

pub fn constant<T: TensorCoverable>(input: T) -> TensorCover {
    input.cover()
}
