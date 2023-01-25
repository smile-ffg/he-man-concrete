use crate::tensor::*;

use npyz::WriterBuilder;
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub fn load_npy(path: &str) -> ClearTensor {
    let bytes = std::fs::read(path).unwrap();

    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    convert_npy(npy)
}

pub fn save_npy(path: &str, tensor: ClearTensor) {
    let mut out_file = File::create(path).unwrap();
    let out_shape: Vec<u64> = tensor.get_shape().iter().map(|dim| *dim as u64).collect();
    let mut wtr = {
        npyz::WriteOptions::new()
            .dtype(npyz::DType::Plain("<f8".parse::<npyz::TypeStr>().unwrap()))
            .shape(out_shape.as_slice())
            .writer(&mut out_file)
            .begin_nd()
            .unwrap()
    };

    wtr.extend(tensor.get_values()).unwrap();
    wtr.finish().unwrap();
}

pub fn load_npz(path: &str) -> Vec<ClearTensor> {
    let mut npz = npyz::npz::NpzArchive::open(path).unwrap();

    let names: Vec<String> = npz
        .array_names()
        .into_iter()
        .map(|x| x.to_owned())
        .collect();
    names
        .into_iter()
        .map(|name| convert_npy(npz.by_name(&name).unwrap().unwrap()))
        .collect()
}

pub fn load_serialized<T>(path: &str) -> T
where
    T: Serialize + DeserializeOwned,
{
    let rdr = BufReader::new(File::open(path).unwrap());
    bincode::deserialize_from(rdr).unwrap()
}

pub fn save_serialized<T>(path: &str, data: T)
where
    T: Serialize + DeserializeOwned,
{
    let mut wtr = BufWriter::new(File::create(path).unwrap());
    bincode::serialize_into(&mut wtr, &data).unwrap();
}

fn convert_npy<R>(npy: npyz::NpyFile<R>) -> ClearTensor
where
    R: std::io::Read,
{
    match npy.dtype() {
        npyz::DType::Plain(s) => match &s.to_string()[1..] {
            "f4" => convert_typed_npy::<R, f32>(npy),
            "f8" => convert_typed_npy::<R, f64>(npy),
            "i4" => convert_typed_npy::<R, i32>(npy),
            "i8" => convert_typed_npy::<R, i64>(npy),
            "I4" => convert_typed_npy::<R, u32>(npy),
            "I8" => convert_typed_npy::<R, u64>(npy),
            _ => panic!("Unsupported scalar dtype {}", s),
        },
        _ => panic!("Non-scalar dtypes are not supported"),
    }
}

fn convert_typed_npy<R, T>(npy: npyz::NpyFile<R>) -> ClearTensor
where
    R: std::io::Read,
    T: npyz::Deserialize + num::ToPrimitive,
{
    ClearTensor::new(
        npy.shape().iter().map(|x| *x as usize).collect(),
        npy.data::<T>()
            .unwrap()
            .map(|x| {
                x.unwrap()
                    .to_f64()
                    .unwrap_or_else(|| panic!("Input not representable as 64-bit floating point"))
            })
            .collect(),
    )
}
