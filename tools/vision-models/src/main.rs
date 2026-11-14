//! Writes burn code for each ONNX model into `crates/vision/src/generated`,
//! and its weights (`.bpk`) into the folder given as the second argument.
//! Names after that pick the models; with none, the selection models.
//!
//! ```text
//! cargo run --release -- ../ml-bench/models ../../../local-files/models/vision
//! cargo run --release -- ONNX_DIR ../../../local-files/models/detect \
//!     face_detection_yunet_2023mar person_detection_mediapipe_2023mar \
//!     palm_detection_mediapipe_2023feb
//! ```

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn_onnx::{LoadStrategy, ModelGen};

/// The ONNX files the vision crate runs, by the name its code uses.
const MODELS: [&str; 5] = [
    "mobile_sam_image_encoder",
    "sam_mask_decoder_lowres",
    "face_detection_yunet_2023mar",
    "person_detection_mediapipe_2023mar",
    "palm_detection_mediapipe_2023feb",
];

/// The models converted when no names are given.
const SELECTION: [&str; 2] = ["mobile_sam_image_encoder", "sam_mask_decoder_lowres"];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [onnx_dir, weights_dir, names @ ..] = args.as_slice() else {
        return fail("usage: vision-models ONNX_DIR WEIGHTS_DIR [NAME...]");
    };
    let names: Vec<&str> = if names.is_empty() {
        SELECTION.to_vec()
    } else {
        names.iter().map(String::as_str).collect()
    };
    if let Some(name) = names.iter().find(|name| !MODELS.contains(name)) {
        return fail(&format!("{name} is not one of {MODELS:?}"));
    }
    let code_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/vision/src/generated");
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/generated");
    for name in names {
        let onnx = Path::new(onnx_dir).join(format!("{name}.onnx"));
        if !onnx.is_file() {
            return fail(&format!(
                "{} is missing; run ../ml-bench/scripts/fetch.sh",
                onnx.display()
            ));
        }
        ModelGen::new()
            .input(&onnx.to_string_lossy())
            .out_dir(&scratch.to_string_lossy())
            // The vision crate loads the weights itself, so a bad file is an error and not a panic.
            .load_strategy(LoadStrategy::None)
            .run_from_cli();
        let moves = [
            (
                scratch.join(format!("{name}.rs")),
                code_dir.join(format!("{name}.rs")),
            ),
            (
                scratch.join(format!("{name}.bpk")),
                PathBuf::from(weights_dir).join(format!("{name}.bpk")),
            ),
        ];
        for (from, to) in moves {
            if let Some(parent) = to.parent()
                && let Err(error) = std::fs::create_dir_all(parent)
            {
                return fail(&format!("{}: {error}", parent.display()));
            }
            if let Err(error) = std::fs::copy(&from, &to) {
                return fail(&format!("{} to {}: {error}", from.display(), to.display()));
            }
        }
    }
    ExitCode::SUCCESS
}

fn fail(message: &str) -> ExitCode {
    // A standalone tool with no logger; the message is its output.
    let _ = std::io::Write::write_all(
        &mut std::io::stderr(),
        format!("vision-models: {message}\n").as_bytes(),
    );
    ExitCode::FAILURE
}
