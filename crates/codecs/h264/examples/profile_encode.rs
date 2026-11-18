//! Temporary: encode the bench's 30 testsrc2 frames in a loop for a profiler.
#![allow(clippy::print_stdout)]

use cutlass_decode::{CodecId, CodecParams, Yuv420p};
use cutlass_encode::{VideoEncoder, VideoInput};
use cutlass_h264::H264Encoder;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const Y_LEN: usize = 1280 * 720;
const C_LEN: usize = Y_LEN / 4;
const FRAME_LEN: usize = Y_LEN + C_LEN * 2;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("pass the yuv path")?;
    let loops: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let raw = std::fs::read(path).map_err(|e| e.to_string())?;
    let frames = raw.len() / FRAME_LEN;
    for _ in 0..loops {
        let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
        params.nal_length_size = 0;
        params.bitrate = 2_000_000;
        let mut encoder = H264Encoder::open(&params).map_err(|e| e.to_string())?;
        let started = std::time::Instant::now();
        let mut bytes = 0usize;
        let mut stream = Vec::new();
        for index in 0..frames {
            let plane = &raw[index * FRAME_LEN..(index + 1) * FRAME_LEN];
            let picture = Yuv420p::try_new(
                WIDTH,
                HEIGHT,
                plane[..Y_LEN].to_vec(),
                plane[Y_LEN..Y_LEN + C_LEN].to_vec(),
                plane[Y_LEN + C_LEN..].to_vec(),
            )
            .map_err(|e| e.to_string())?;
            let input = VideoInput {
                picture,
                pts: index as i64,
                duration: 1,
                timescale: 30,
                force_keyframe: index == 0,
            };
            for p in encoder.push(&input).map_err(|e| e.to_string())? {
                bytes += p.data.len();
                stream.extend_from_slice(&p.data);
            }
        }
        for p in encoder.drain().map_err(|e| e.to_string())? {
            bytes += p.data.len();
            stream.extend_from_slice(&p.data);
        }
        if let Some(out) = std::env::args().nth(3) {
            std::fs::write(out, &stream).map_err(|e| e.to_string())?;
        }
        println!(
            "{frames} frames, {bytes} bytes, {:.2} s",
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
