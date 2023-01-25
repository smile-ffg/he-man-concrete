mod common;

use assert_cmd::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

#[test]
fn test_inference() {
    // create temporary directory
    let tmp_dir = TempDir::new().unwrap();

    // run key generation
    let mut keygen_cmd = Command::cargo_bin("he-man-concrete").unwrap();
    keygen_cmd
        .args(["keygen", tmp_dir.path().to_str().unwrap()])
        .assert()
        .success();

    let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    model_dir.push("tests/models");
    let model_paths = fs::read_dir(model_dir).unwrap();

    for path in model_paths {
        // run inference with model
        let mut inference_cmd = Command::cargo_bin("he-man-concrete").unwrap();
        inference_cmd
            .args([
                "inference",
                tmp_dir.path().to_str().unwrap(),
                path.unwrap().path().to_str().unwrap(),
                ".",
                ".",
            ])
            .assert()
            .success();

        // load output
        let clear_out = common::load_vec(tmp_dir.path().join("clear_out.bin").to_str().unwrap());
        let enc_out = common::load_vec(tmp_dir.path().join("enc_out.bin").to_str().unwrap());

        common::assert_close_abs(&clear_out, &enc_out, 0.08);
    }

    // clean up
    tmp_dir.close().unwrap();
}
