//! Turns the two MobileSAM ONNX files into burn code and weights.
//!
//! `scripts/fetch.sh` downloads them into `models/`.

use std::path::{Path, PathBuf};

use burn_onnx::{LoadStrategy, ModelGen};

const MODELS: [&str; 2] = ["mobile_sam_image_encoder", "sam_mask_decoder_lowres"];

fn main() {
    for name in MODELS {
        let path = format!("models/{name}.onnx");
        println!("cargo::rerun-if-changed={path}");
        if !Path::new(&path).exists() {
            println!("cargo::error={path} is missing. Run scripts/fetch.sh first.");
            return;
        }
        // The weights go inside the binary, so it runs on a phone with no
        // build folder next to it.
        ModelGen::new()
            .input(&path)
            .out_dir("model/")
            .load_strategy(LoadStrategy::Embedded)
            .run_from_script();
        if std::env::var_os("CARGO_FEATURE_WGPU").is_some()
            && std::env::var_os("CARGO_FEATURE_IOS").is_none()
        {
            narrow_shape_ints(name);
        }
    }
}

/// ONNX keeps shapes as 64-bit integers, and burn-onnx puts those on the
/// device as `I64` tensors. Many phone GPUs cannot run 64-bit integers in
/// WGSL, so the portable build keeps them in 32 bits. Shapes here are far
/// below 2^31.
fn narrow_shape_ints(name: &str) {
    let Some(out) = std::env::var_os("OUT_DIR") else {
        println!("cargo::error=OUT_DIR is not set");
        return;
    };
    let file: PathBuf = Path::new(&out).join("model").join(format!("{name}.rs"));
    let narrowed = std::fs::read_to_string(&file)
        .map(|code| code.replace("burn::tensor::DType::I64", "burn::tensor::DType::I32"))
        .and_then(|code| std::fs::write(&file, code));
    if let Err(error) = narrowed {
        println!("cargo::error={}: {error}", file.display());
    }
}
