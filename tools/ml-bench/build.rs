//! Turns the two MobileSAM ONNX files into burn code and weights.
//!
//! `scripts/fetch.sh` downloads them into `models/`.

use std::path::Path;

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
    }
}
