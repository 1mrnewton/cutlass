//! Write a short, low-rate preview video of every bundled effect recipe.
//!
//! Each recipe runs on the GPU over the middle square of the scene its card
//! shows on the desktop, and the frames are encoded as H.264 into an MP4
//! per effect, named by the recipe id. The phone picker loops these files. Run it again after a recipe or a
//! block changes:
//!
//! ```sh
//! cargo run -p cutlass-render --release --example effect_previews -- apps/ios/ios/EffectPreviews
//! ```

use std::error::Error;
use std::io::Cursor;
use std::path::PathBuf;

use cutlass_decode::{CodecId, CodecParams};
use cutlass_effects::{Recipe, bundled, preview_scene};
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
    for recipe in bundled() {
        let scene = preview_scene(recipe.preview_scene()).ok_or("a recipe names no scene")?;
        let scene = middle_square(scene.width, scene.height, &scene.rgba, SIDE)?;
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

/// The middle `side` by `side` pixels of a `width` by `height` picture.
fn middle_square(width: u32, height: u32, rgba: &[u8], side: u32) -> Result<Vec<u8>, Failure> {
    if width < side || height < side {
        return Err("a scene is smaller than a preview".into());
    }
    let (left, top) = ((width - side) / 2, (height - side) / 2);
    let mut square = Vec::with_capacity((side * side * 4) as usize);
    for y in top..top + side {
        let start = ((y * width + left) * 4) as usize;
        square.extend_from_slice(&rgba[start..start + (side * 4) as usize]);
    }
    Ok(square)
}
