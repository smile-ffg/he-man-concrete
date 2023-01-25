use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(
        &["src/onnx/onnx.proto3", "src/onnx/onnx-operators.proto3"],
        &["src/"],
    )
    .unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=./src/onnx/onnx.proto3");
    println!("cargo:rerun-if-changed=./src/onnx/onnx-operators.proto3");
    Ok(())
}
