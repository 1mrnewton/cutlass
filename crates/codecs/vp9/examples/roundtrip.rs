//! Decode the 32×32 profile-0 keyframe in `fixtures/vp9_red32_q_0.bin`.

#![allow(clippy::print_stdout)]

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecId, CodecParams, Packet};
use cutlass_vp9::Vp9Decoder;

const FIXTURE: &[u8] = include_bytes!("../fixtures/vp9_red32_q_0.bin");

fn main() -> Result<(), String> {
    let params = CodecParams::video(CodecId::Vp9, 32, 32);
    let mut decoder = Vp9Decoder::open(&params).map_err(|err| err.to_string())?;
    let packet = Packet::video(FIXTURE.to_vec(), 0, 1, 30, true);
    let shown = decoder.push(&packet).map_err(|err| err.to_string())?;
    let Some(output) = shown.first() else {
        return Err("decoder returned no picture".to_string());
    };
    let width = output.frame.width;
    let height = output.frame.height;
    let count = shown.len();
    println!("{width}x{height} frames {count}");
    Ok(())
}
