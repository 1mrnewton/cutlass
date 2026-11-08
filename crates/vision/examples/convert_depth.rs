//! Converts Depth Anything V2 Small's `model.safetensors` into the `.bpk` that
//! [`cutlass_vision::DepthModel::load`] reads.
//!
//! ```text
//! cargo run -p cutlass-vision --example convert_depth -- IN.safetensors OUT_DIR
//! ```

use std::path::Path;
use std::process::ExitCode;

use cutlass_vision::{DEPTH_WEIGHTS, convert_depth_anything};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [from, dir] = args.as_slice() else {
        return say(
            "usage: convert_depth IN.safetensors OUT_DIR",
            ExitCode::FAILURE,
        );
    };
    let dir = Path::new(dir);
    if let Err(error) = std::fs::create_dir_all(dir) {
        return say(&format!("{}: {error}", dir.display()), ExitCode::FAILURE);
    }
    let to = dir.join(DEPTH_WEIGHTS);
    match convert_depth_anything(Path::new(from), &to) {
        Ok(()) => say(&format!("wrote {}", to.display()), ExitCode::SUCCESS),
        Err(error) => say(&error.to_string(), ExitCode::FAILURE),
    }
}

/// A standalone tool with no logger; the message is its output.
fn say(message: &str, code: ExitCode) -> ExitCode {
    let _ = std::io::Write::write_all(&mut std::io::stderr(), format!("{message}\n").as_bytes());
    code
}
