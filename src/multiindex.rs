/*
    Macro to iterate multi-index over a shape and call function to modify a result object
*/

pub fn flatten_multi_index(multi_index: &Vec<usize>, shape: &Vec<usize>) -> usize {
    let mut ind: usize = 0;
    for (i, dim) in multi_index.iter().enumerate() {
        ind += dim * shape[i + 1..].iter().product::<usize>();
    }
    ind
}

#[macro_export]
macro_rules! iterate_multi_index {
    ($f:expr,
    $shape:expr,
    $result:expr
    $(, $opt:expr)*) => {{
        let mut multi_index = Vec::with_capacity($shape.len());
        for i in 0..$shape.iter().product() {
            let mut index = i;
            for dim in $shape.iter().rev() {
                let k = index % dim;
                multi_index.push(k);
                index = index / dim;
            }

            multi_index.reverse();
            $result = $f(&multi_index, $result $(, $opt)*);
            multi_index.clear();
        }
        $result
    }};
}
