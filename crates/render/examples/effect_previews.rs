//! Write a short, low-rate preview video of every bundled effect recipe.
//!
//! A small scene is drawn here, each recipe runs over it on the GPU, and the
//! frames are encoded as H.264 into an MP4 per effect, named by the recipe
//! id. The phone picker loops these files. Run it again after a recipe or a
//! block changes:
//!
//! ```sh
//! cargo run -p cutlass-render --release --example effect_previews -- apps/ios/ios/EffectPreviews
//! ```

use std::error::Error;
use std::io::Cursor;
use std::path::PathBuf;

use cutlass_decode::{CodecId, CodecParams};
use cutlass_effects::{Recipe, bundled};
use cutlass_encode::{VideoEncoder, VideoInput, Yuv420p};
use cutlass_h264::H264Encoder;
use cutlass_isobmff::{
    CodecId as BoxCodec, ColorTags, IsoBmffMux, Mux, MuxOptions, Packet, StreamParams, Timescale,
};
use cutlass_model::Micros;
use cutlass_render::{Layer, Placement, Renderer, Rgba};

/// Square, a multiple of 16, about the size of a picker tile at 3x.
const SIDE: u32 = 192;
/// Frames a second: enough to read the motion, small enough to ship 200.
const RATE: u32 = 12;
const SECONDS: u32 = 2;
const BITRATE: u32 = 350_000;

type Failure = Box<dyn Error>;

fn main() -> Result<(), Failure> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: effect_previews <output folder>")?;
    std::fs::create_dir_all(&out)?;
    let renderer = Renderer::headless()?;
    let scene = scene(SIDE, SIDE);
    for recipe in bundled() {
        let bytes = preview(&renderer, &scene, recipe)?;
        std::fs::write(out.join(format!("{}.mp4", recipe.id)), bytes)?;
    }
    Ok(())
}

/// One recipe's MP4.
fn preview(renderer: &Renderer, scene: &[u8], recipe: &Recipe) -> Result<Vec<u8>, Failure> {
    let canvas = renderer.canvas(SIDE, SIDE)?;
    let mut params = CodecParams::video(CodecId::H264, SIDE, SIDE);
    params.bitrate = BITRATE;
    let mut encoder = H264Encoder::open(&params)?;
    let mut packets = Vec::new();
    for index in 0..RATE * SECONDS {
        let time = Micros(i64::from(index) * 1_000_000 / i64::from(RATE));
        let steps = recipe.steps_at(time);
        let picture = Rgba {
            width: SIDE,
            height: SIDE,
            bytes: scene,
        };
        renderer.draw(
            &canvas,
            &[
                Layer::Rgba(picture, Placement::canvas(SIDE, SIDE)),
                Layer::Effect(&steps),
            ],
        )?;
        let rgba = renderer.read(&canvas)?;
        packets.extend(encoder.push(&VideoInput {
            picture: yuv(SIDE, SIDE, &rgba)?,
            pts: i64::from(index),
            duration: 1,
            timescale: RATE,
            force_keyframe: index == 0,
        })?);
    }
    packets.extend(encoder.drain()?);

    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let mut stream = StreamParams::video(BoxCodec::H264, Timescale(RATE), SIDE, SIDE);
    stream.extradata = encoder.extradata().to_vec();
    stream.color = Some(ColorTags {
        primaries: 1,
        transfer: 1,
        matrix: 1,
        full_range: false,
    });
    let track = mux.add_stream(&stream)?;
    for (dts, packet) in packets.into_iter().enumerate() {
        mux.write_packet(&Packet {
            track_id: track,
            payload: packet.data,
            dts: i64::try_from(dts)?,
            pts: packet.pts,
            duration: u32::try_from(packet.duration)?,
            timescale: Timescale(RATE),
            keyframe: packet.keyframe,
        })?;
    }
    Ok(mux.finish_into()?.into_inner())
}

/// Premultiplied, opaque RGBA as limited-range BT.709 4:2:0.
fn yuv(width: u32, height: u32, rgba: &[u8]) -> Result<Yuv420p, Failure> {
    let (w, h) = (width as usize, height as usize);
    let rgb = |x: usize, y: usize| {
        let at = (y * w + x) * 4;
        [
            f32::from(rgba[at]),
            f32::from(rgba[at + 1]),
            f32::from(rgba[at + 2]),
        ]
    };
    let luma = |[r, g, b]: [f32; 3]| 16.0 + (0.2126 * r + 0.7152 * g + 0.0722 * b) * 219.0 / 255.0;
    let mut y_plane = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            y_plane.push(luma(rgb(x, y)).round().clamp(0.0, 255.0) as u8);
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut u_plane = Vec::with_capacity(cw * ch);
    let mut v_plane = Vec::with_capacity(cw * ch);
    for y in 0..ch {
        for x in 0..cw {
            let mut sum = [0.0f32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let pixel = rgb(x * 2 + dx, y * 2 + dy);
                for (total, value) in sum.iter_mut().zip(pixel) {
                    *total += value / 4.0;
                }
            }
            let [r, g, b] = sum;
            let y_value = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let u = 128.0 + (b - y_value) / 1.8556 * 224.0 / 255.0;
            let v = 128.0 + (r - y_value) / 1.5748 * 224.0 / 255.0;
            u_plane.push(u.round().clamp(0.0, 255.0) as u8);
            v_plane.push(v.round().clamp(0.0, 255.0) as u8);
        }
    }
    Ok(Yuv420p::try_new(width, height, y_plane, u_plane, v_plane)?)
}

/// A sunset over two hills, with a row of coloured blocks so a split or a
/// shift shows at a glance. Opaque, so premultiplied is the same as plain.
fn scene(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((width * height * 4) as usize);
    let (w, h) = (width as f32, height as f32);
    let sun = (w * 0.62, h * 0.42, w * 0.16);
    let blocks: [[f32; 3]; 4] = [
        [235.0, 64.0, 80.0],
        [250.0, 200.0, 60.0],
        [70.0, 200.0, 140.0],
        [70.0, 120.0, 240.0],
    ];
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let t = fy / h;
            // Sky: deep blue at the top to warm orange at the horizon.
            let mut color = [40.0 + 215.0 * t, 50.0 + 110.0 * t, 120.0 - 40.0 * t];
            let to_sun = ((fx - sun.0).powi(2) + (fy - sun.1).powi(2)).sqrt();
            if to_sun < sun.2 {
                color = [255.0, 236.0, 170.0];
            } else if to_sun < sun.2 * 1.6 {
                let glow = 1.0 - (to_sun - sun.2) / (sun.2 * 0.6);
                for (channel, target) in color.iter_mut().zip([255.0, 210.0, 140.0]) {
                    *channel += (target - *channel) * glow * 0.5;
                }
            }
            let far = h * (0.62 + 0.06 * (fx / w * 5.0).sin());
            let near = h * (0.74 + 0.05 * (fx / w * 3.0 + 1.0).cos());
            if fy > near {
                color = [34.0, 46.0, 52.0];
            } else if fy > far {
                color = [86.0, 70.0, 110.0];
            }
            let band = (h * 0.86, h * 0.95);
            if fy > band.0 && fy < band.1 {
                let slot = ((fx / w) * 4.0).floor().clamp(0.0, 3.0) as usize;
                let inset = (fx / w * 4.0).fract();
                if inset > 0.12 && inset < 0.88 {
                    color = blocks[slot];
                }
            }
            for channel in color {
                bytes.push(channel.round().clamp(0.0, 255.0) as u8);
            }
            bytes.push(255);
        }
    }
    bytes
}
