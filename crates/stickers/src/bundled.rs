//! Stickers from other sets, shipped as their own files: a PNG picture or
//! a Lottie animation. This reads them for the shelf: their size, their
//! length, and a small straight RGBA poster.

use cutlass_decode::{BitDepth, CodecId, CodecParams, ColorType, Decoder, Packet};
use cutlass_lottie::Animation;

/// The picture's width and height, from its header.
pub(crate) fn png_size(png: &[u8]) -> Result<(u32, u32), String> {
    if png.len() < 24 || &png[..8] != b"\x89PNG\r\n\x1a\n" || &png[12..16] != b"IHDR" {
        return Err("the sticker is not a PNG".to_owned());
    }
    let read = |at: usize| u32::from_be_bytes([png[at], png[at + 1], png[at + 2], png[at + 3]]);
    Ok((read(16), read(20)))
}

/// The picture as straight 8-bit RGBA. The bundled pictures are all 8-bit
/// RGBA; any other layout is refused.
pub(crate) fn png_rgba(png: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let mut decoder = cutlass_png::PngDecoder::open(&CodecParams::video(CodecId::Png, 0, 0))
        .map_err(|error| error.to_string())?;
    let pictures = decoder
        .decode_picture(&Packet::video(png.to_vec(), 0, 0, 0, true))
        .map_err(|error| error.to_string())?;
    let Some(picture) = pictures.into_iter().next() else {
        return Err("the PNG held no picture".to_owned());
    };
    let image = picture.image;
    if image.color != ColorType::Rgba || image.depth != BitDepth::Eight {
        return Err("the sticker is not an 8-bit RGBA picture".to_owned());
    }
    let expected = image.width as usize * image.height as usize * 4;
    if image.samples.len() != expected {
        return Err("the PNG's samples do not fill the picture".to_owned());
    }
    let rgba = image
        .samples
        .iter()
        .map(|sample| u8::try_from(*sample).unwrap_or(u8::MAX))
        .collect();
    Ok((image.width, image.height, rgba))
}

/// A square straight RGBA picture `from` pixels wide, `side` pixels wide.
/// Each pixel averages the ones under it, weighted by alpha so the clear
/// edge does not darken the color.
pub(crate) fn resample(rgba: &[u8], from: u32, side: u32) -> Vec<u8> {
    let mut out = vec![0_u8; side as usize * side as usize * 4];
    if from == 0 || side == 0 {
        return out;
    }
    let span = |index: u32| {
        let start = (u64::from(index) * u64::from(from) / u64::from(side)) as u32;
        let end = (u64::from(index + 1) * u64::from(from) / u64::from(side)) as u32;
        (start, end.max(start + 1).min(from))
    };
    for y in 0..side {
        let (top, bottom) = span(y);
        for x in 0..side {
            let (left, right) = span(x);
            let mut sum = [0_u64; 4];
            let mut count = 0_u64;
            for sy in top..bottom {
                for sx in left..right {
                    let at = (sy as usize * from as usize + sx as usize) * 4;
                    let Some(pixel) = rgba.get(at..at + 4) else {
                        continue;
                    };
                    let alpha = u64::from(pixel[3]);
                    for channel in 0..3 {
                        sum[channel] += u64::from(pixel[channel]) * alpha;
                    }
                    sum[3] += alpha;
                    count += 1;
                }
            }
            let at = (y as usize * side as usize + x as usize) * 4;
            if count == 0 || sum[3] == 0 {
                continue;
            }
            for channel in 0..3 {
                out[at + channel] = ((sum[channel] + sum[3] / 2) / sum[3]).min(255) as u8;
            }
            out[at + 3] = ((sum[3] + count / 2) / count).min(255) as u8;
        }
    }
    out
}

pub(crate) fn animation(json: &[u8]) -> Result<Animation, String> {
    Animation::parse(json).map_err(|error| error.to_string())
}

/// The frame the file marks `rest`, its still pose, else the first frame,
/// drawn `side` pixels square, straight RGBA.
pub(crate) fn lottie_poster(json: &[u8], side: u32) -> Result<Vec<u8>, String> {
    let animation = animation(json)?;
    let frame = rest_frame(json)
        .filter(|frame| (animation.in_point()..animation.out_point()).contains(frame))
        .unwrap_or(animation.in_point());
    frame_rgba(&animation, frame, side)
}

/// The first marker named `rest`, as Noto Animated Emoji marks its still
/// pose.
fn rest_frame(json: &[u8]) -> Option<f64> {
    let root: serde_json::Value = serde_json::from_slice(json).ok()?;
    root.get("markers")?
        .as_array()?
        .iter()
        .find(|marker| marker.get("cm").and_then(serde_json::Value::as_str) == Some("rest"))?
        .get("tm")?
        .as_f64()
}

/// Frame `frame` drawn `side` pixels square, straight RGBA.
pub(crate) fn frame_rgba(animation: &Animation, frame: f64, side: u32) -> Result<Vec<u8>, String> {
    let scene = animation
        .scene(frame, side, side)
        .map_err(|error| error.to_string())?;
    let mut rgba = cutlass_lottie::rasterize(&scene);
    for pixel in rgba.as_chunks_mut::<4>().0 {
        let alpha = u32::from(pixel[3]);
        if alpha == 0 {
            continue;
        }
        for channel in &mut pixel[..3] {
            *channel = ((u32::from(*channel) * 255 + alpha / 2) / alpha).min(255) as u8;
        }
    }
    Ok(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_size_needs_a_png_header() {
        assert!(png_size(b"not a picture at all, not at all").is_err());
        assert!(png_size(&[]).is_err());
        assert!(png_rgba(b"\x89PNG\r\n\x1a\n").is_err());
    }

    #[test]
    fn resampling_averages_by_alpha_and_keeps_clear_clear() {
        // Two by two: red opaque, clear black, clear black, red opaque.
        let rgba = [
            255, 0, 0, 255, 0, 0, 0, 0, //
            0, 0, 0, 0, 255, 0, 0, 255,
        ];
        assert_eq!(resample(&rgba, 2, 1), [255, 0, 0, 128]);
        assert_eq!(resample(&rgba, 2, 2), rgba);
        assert_eq!(resample(&[0; 16], 2, 1), [0; 4]);
        // Upscaling repeats the nearest pixel.
        assert_eq!(
            &resample(&rgba, 2, 4)[..8],
            [255, 0, 0, 255, 255, 0, 0, 255]
        );
    }

    #[test]
    fn the_poster_is_the_rest_frame_else_the_first() {
        assert_eq!(
            rest_frame(br#"{"markers":[{"tm":4,"cm":"go"},{"tm":7,"cm":"rest"}]}"#),
            Some(7.0)
        );
        assert_eq!(rest_frame(br#"{"markers":[]}"#), None);
        assert_eq!(rest_frame(b"[1]"), None);
        // A square that moves right: marked at frame 15, the poster is
        // frame 15; unmarked, it is the first frame.
        let square = include_str!("../../lottie/tests/fixtures/square.json");
        let marked = square.replacen('{', r#"{"markers":[{"tm":15,"cm":"rest","dr":0}],"#, 1);
        let animation = animation(square.as_bytes()).unwrap();
        let first = frame_rgba(&animation, animation.in_point(), 16).unwrap();
        let rest = frame_rgba(&animation, 15.0, 16).unwrap();
        assert_ne!(first, rest);
        assert_eq!(lottie_poster(marked.as_bytes(), 16).unwrap(), rest);
        assert_eq!(lottie_poster(square.as_bytes(), 16).unwrap(), first);
        // A mark past the end is not used.
        let late = square.replacen('{', r#"{"markers":[{"tm":99,"cm":"rest","dr":0}],"#, 1);
        assert_eq!(lottie_poster(late.as_bytes(), 16).unwrap(), first);
    }

    #[test]
    fn a_broken_animation_is_an_error() {
        assert!(lottie_poster(b"{}", 8).is_err());
        assert!(lottie_poster(b"not json", 8).is_err());
    }
}
