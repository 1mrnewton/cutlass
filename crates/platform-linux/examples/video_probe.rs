//! Prints the hardware video this machine offers: the NVIDIA GPU with what
//! NVDEC decodes and NVENC encodes there, and each VA-API render node with
//! its driver, profiles, and entrypoints.
//!
//! ```text
//! cargo run -p cutlass-platform-linux --example video_probe
//! ```
//!
//! `CUTLASS_VIDEO_BACKEND` narrows it the same way it narrows the app.

#![allow(clippy::print_stdout)]

fn main() {
    for line in cutlass_platform_linux::probe_video() {
        println!("{line}");
    }
}
