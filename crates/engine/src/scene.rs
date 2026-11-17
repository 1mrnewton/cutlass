//! Where a picture changes enough to be another scene.
//!
//! The score is the mean absolute difference of a small luma grid, and the
//! change in that difference, on a 0–100 scale. A cut is a score of at least
//! [`SCENE_SCORE`]. The file is read forward from the start of the range,
//! sought from sample to sample when they are far apart. Samples are at
//! least a fifth of a second apart, and there are at most [`MAX_LOOKS`] of
//! them. A still, or a clip with no change at that step, returns no cuts.
//! The luma grid reads the first plane of any planar picture: Y, or G for
//! GBR, at 8, 10 or 12 bits.

use std::borrow::Cow;
use std::path::Path;

use cutlass_codec::access::{BitDepth, ColorType, PixelFormat};
use cutlass_decode::StillImage;
use cutlass_model::{Micros, TimeRange};

use crate::error::EngineError;
use crate::reader::{DecodedFrame, ReaderFactory};

/// Cells in the luma grid. The edge is 32.
pub const SCENE_CELLS: usize = 32 * 32;

/// A score at or above this, on a 0–100 scale, is a scene change.
pub const SCENE_SCORE: f64 = 10.0;

/// Samples one file may decode while looking for cuts.
const MAX_LOOKS: usize = 240;

/// Closest spacing of those samples, in microseconds.
const MIN_GAP: i64 = 200_000;

const EDGE: u32 = 32;

/// Why the picture could not be scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneError {
    /// The path could not be opened.
    Missing,
    /// The file has no picture this scan can read.
    NoPicture,
    /// A sample could not be decoded.
    Decode,
}

/// Source times inside `range` where the picture changes.
///
/// The time is the first sampled frame of the new scene. The start of
/// `range` is not a cut. An empty list is a picture that did not change.
pub fn scene_cuts(
    factory: &ReaderFactory,
    path: &Path,
    range: &TimeRange,
) -> Result<Vec<Micros>, SceneError> {
    if range.duration.0 <= 0 || range.start.0 < 0 {
        return Err(SceneError::Decode);
    }
    let mut media = factory.open(path).map_err(scene_error)?;
    let frames = media
        .pictures_across(range, sample_gap(range), MAX_LOOKS)
        .map_err(scene_error)?;
    let mut grids = Vec::with_capacity(frames.len());
    let mut scored = Vec::with_capacity(frames.len());
    let saw_frame = !frames.is_empty();
    for held in frames {
        let Some(grid) = luma_grid(&held.frame) else {
            continue;
        };
        if scored.last().copied() == Some(held.start.0) {
            continue;
        }
        scored.push(held.start.0);
        grids.push(grid);
    }
    if grids.is_empty() {
        return Err(if saw_frame {
            SceneError::Decode
        } else {
            SceneError::NoPicture
        });
    }
    Ok(cut_times(&scored, &grids).into_iter().map(Micros).collect())
}

/// Times in `times` whose grid differs from the previous one by a scene cut.
///
/// `times` and `grids` are paired in order. A short list stops the pair.
/// The first grid is the reference and is not a cut.
///
/// The score is the change, the lesser of the frame difference and how far
/// it moved from the last one, so steady motion is not a cut at every
/// sample. Right after a cut that would hide a second cut one sample on,
/// whose difference matches the first: so there a difference of twice
/// [`SCENE_SCORE`] is a cut on its own. A fast pan stays well under that.
#[must_use]
pub fn cut_times(times: &[i64], grids: &[[u8; SCENE_CELLS]]) -> Vec<i64> {
    let mut cuts = Vec::new();
    let mut previous_mafd = 0.0;
    let mut previous: Option<&[u8; SCENE_CELLS]> = None;
    let mut after_cut = false;
    for (time, grid) in times.iter().zip(grids) {
        let Some(before) = previous else {
            previous = Some(grid);
            continue;
        };
        let sad = sad(before, grid);
        let count = SCENE_CELLS as f64;
        let mafd = sad as f64 * 100.0 / count / 256.0;
        let difference = (mafd - previous_mafd).abs();
        let score = mafd.min(difference).clamp(0.0, 100.0);
        previous_mafd = mafd;
        let cut = score >= SCENE_SCORE || (after_cut && mafd >= 2.0 * SCENE_SCORE);
        if cut {
            cuts.push(*time);
        }
        after_cut = cut;
        previous = Some(grid);
    }
    cuts
}

fn sample_gap(range: &TimeRange) -> i64 {
    let spread = range.duration.0 / i64::try_from(MAX_LOOKS).unwrap_or(i64::MAX);
    spread.max(MIN_GAP)
}

fn scene_error(error: EngineError) -> SceneError {
    match error {
        EngineError::MissingFile { .. }
        | EngineError::UnknownContainer
        | EngineError::Io { .. }
        | EngineError::Read { .. }
        | EngineError::Truncated { .. }
        | EngineError::Hostile { .. } => SceneError::Missing,
        EngineError::NoDecoder { .. }
        | EngineError::NotMedia
        | EngineError::NotReady
        | EngineError::ColorMiss { .. }
        | EngineError::UnknownCodec { .. } => SceneError::NoPicture,
        _ => SceneError::Decode,
    }
}

fn luma_grid(frame: &DecodedFrame) -> Option<[u8; SCENE_CELLS]> {
    match frame.on_cpu().ok()?.as_ref() {
        DecodedFrame::Coded {
            width,
            height,
            pixel,
            bytes,
            ..
        } => {
            let plane = first_plane(*pixel, *width, *height, bytes)?;
            downsample(&plane, *width, *height)
        }
        DecodedFrame::Still(image) => still_grid(image),
        DecodedFrame::Platform { .. } | DecodedFrame::Surface { .. } => None,
    }
}

/// The first plane of a coded picture as 8-bit samples: Y, or G for GBR,
/// or the luma of packed RGBA. `None` for a layout this scan does not read.
fn first_plane(pixel: PixelFormat, width: u32, height: u32, bytes: &[u8]) -> Option<Cow<'_, [u8]>> {
    let samples = usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)?;
    let shift = match pixel {
        PixelFormat::Yuv420p
        | PixelFormat::Yuv422p
        | PixelFormat::Yuv440p
        | PixelFormat::Yuv444p
        | PixelFormat::Gbrp
        | PixelFormat::Gray8 => return bytes.get(..samples).map(Cow::Borrowed),
        PixelFormat::Rgba8 => {
            let rgba = bytes.get(..samples.checked_mul(4)?)?;
            return rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| {
                    rec601(
                        u16::from(pixel[0]),
                        u16::from(pixel[1]),
                        u16::from(pixel[2]),
                    )
                })
                .collect::<Option<Vec<u8>>>()
                .map(Cow::Owned);
        }
        PixelFormat::Yuv420p10
        | PixelFormat::Yuv422p10
        | PixelFormat::Yuv440p10
        | PixelFormat::Yuv444p10
        | PixelFormat::Gbrp10
        | PixelFormat::Gray10 => 2,
        PixelFormat::Yuv420p12
        | PixelFormat::Yuv422p12
        | PixelFormat::Yuv440p12
        | PixelFormat::Yuv444p12
        | PixelFormat::Gbrp12 => 4,
        PixelFormat::None => return None,
    };
    let wide = bytes.get(..samples.checked_mul(2)?)?;
    Some(Cow::Owned(
        wide.as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let sample = u16::from_le_bytes(*pair) >> shift;
                u8::try_from(sample).unwrap_or(u8::MAX)
            })
            .collect(),
    ))
}

fn still_grid(image: &StillImage) -> Option<[u8; SCENE_CELLS]> {
    if image.depth != BitDepth::Eight {
        return None;
    }
    let channels = match image.color {
        ColorType::Gray => 1,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::GrayAlpha | ColorType::Indexed => return None,
    };
    let pixels = usize::try_from(image.width)
        .ok()?
        .checked_mul(usize::try_from(image.height).ok()?)?;
    if image.samples.len() != pixels.checked_mul(channels)? {
        return None;
    }
    let mut plane = Vec::with_capacity(pixels);
    for pixel in image.samples.chunks(channels) {
        let luma = match channels {
            1 => u8::try_from(pixel[0]).ok()?,
            _ => rec601(pixel[0], pixel[1], pixel[2])?,
        };
        plane.push(luma);
    }
    downsample(&plane, image.width, image.height)
}

fn rec601(red: u16, green: u16, blue: u16) -> Option<u8> {
    let red = u32::from(u8::try_from(red).ok()?);
    let green = u32::from(u8::try_from(green).ok()?);
    let blue = u32::from(u8::try_from(blue).ok()?);
    Some(u8::try_from((red * 77 + green * 150 + blue * 29) >> 8).unwrap_or(255))
}

fn downsample(plane: &[u8], width: u32, height: u32) -> Option<[u8; SCENE_CELLS]> {
    if width == 0 || height == 0 {
        return None;
    }
    let stride = usize::try_from(width).ok()?;
    if plane.len() < stride.checked_mul(usize::try_from(height).ok()?)? {
        return None;
    }
    let mut grid = [0_u8; SCENE_CELLS];
    for gy in 0..EDGE {
        for gx in 0..EDGE {
            let x0 = gx * width / EDGE;
            let x1 = ((gx + 1) * width / EDGE).max(x0 + 1).min(width);
            let y0 = gy * height / EDGE;
            let y1 = ((gy + 1) * height / EDGE).max(y0 + 1).min(height);
            let mut sum = 0_u64;
            let mut count = 0_u64;
            for y in y0..y1 {
                let row = usize::try_from(y).ok()?.checked_mul(stride)?;
                for x in x0..x1 {
                    let index = row.checked_add(usize::try_from(x).ok()?)?;
                    sum += u64::from(*plane.get(index)?);
                    count += 1;
                }
            }
            if count == 0 {
                return None;
            }
            let cell = usize::try_from(gy * EDGE + gx).ok()?;
            grid[cell] = u8::try_from(sum / count).unwrap_or(255);
        }
    }
    Some(grid)
}

fn sad(left: &[u8; SCENE_CELLS], right: &[u8; SCENE_CELLS]) -> u64 {
    left.iter()
        .zip(right)
        .map(|(before, after)| u64::from(before.abs_diff(*after)))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(value: u8) -> [u8; SCENE_CELLS] {
        [value; SCENE_CELLS]
    }

    #[test]
    fn a_hard_change_is_a_cut_and_a_steady_picture_is_not() {
        let steady = [fill(0), fill(0), fill(1), fill(2)];
        assert!(cut_times(&[0, 10, 20, 30], &steady).is_empty());

        let cut = [fill(0), fill(0), fill(255)];
        assert_eq!(cut_times(&[0, 200_000, 400_000], &cut), vec![400_000]);
        assert!(cut_times(&[0], &[fill(255)]).is_empty());
    }

    #[test]
    fn a_cut_one_sample_after_a_cut_is_found_and_a_pan_is_cut_once() {
        // A one-sample shot between two others: both of its edges are cuts.
        let flash = [fill(0), fill(0), fill(255), fill(0), fill(0)];
        assert_eq!(cut_times(&[0, 1, 2, 3, 4], &flash), vec![2, 3]);
        // A steady fast pan: luma moves 38 levels a sample, which is a
        // change at its start and then not.
        let pan = [fill(0), fill(0), fill(38), fill(76), fill(114)];
        assert_eq!(cut_times(&[0, 1, 2, 3, 4], &pan), vec![2]);
    }

    #[test]
    fn the_grid_reads_the_first_plane_at_any_depth_and_layout() {
        // 32x32 at a level of 100 (8-bit), as 8-bit 4:2:2, 10-bit 4:2:0,
        // 12-bit 4:4:4, gray, and packed RGBA.
        let (width, height) = (32_u32, 32_u32);
        let samples = (width * height) as usize;
        let mut eight = vec![100_u8; samples];
        eight.extend(vec![128_u8; samples]);
        let wide = |level: u16| -> Vec<u8> {
            let mut bytes = Vec::new();
            for _ in 0..samples * 3 {
                bytes.extend(level.to_le_bytes());
            }
            bytes
        };
        let rgba: Vec<u8> = (0..samples).flat_map(|_| [100, 100, 100, 255]).collect();
        for (pixel, bytes) in [
            (PixelFormat::Yuv422p, eight.clone()),
            (PixelFormat::Gray8, eight.clone()),
            (PixelFormat::Yuv420p10, wide(400)),
            (PixelFormat::Yuv444p12, wide(1_600)),
            (PixelFormat::Gray10, wide(400)),
            (PixelFormat::Rgba8, rgba.clone()),
        ] {
            let frame = DecodedFrame::Coded {
                width,
                height,
                pixel,
                primaries: cutlass_codec::access::ColorPrimaries::Bt709,
                transfer: cutlass_codec::access::Transfer::Bt709,
                matrix: cutlass_codec::access::ColorMatrix::Bt709,
                range: cutlass_codec::access::ColorRange::Limited,
                bytes: std::sync::Arc::new(bytes),
            };
            let grid = luma_grid(&frame).unwrap_or_else(|| panic!("{pixel:?}"));
            assert!(
                grid.iter().all(|cell| (99..=100).contains(cell)),
                "{pixel:?}"
            );
        }
        assert!(first_plane(PixelFormat::Yuv420p10, width, height, &[0; 4]).is_none());
    }

    #[test]
    fn a_missing_file_is_not_a_scene_cut() {
        let error = scene_cuts(
            &ReaderFactory::new(),
            Path::new("no-such-picture"),
            &TimeRange::new(0, 1_000_000),
        )
        .expect_err("missing");
        assert_eq!(error, SceneError::Missing);
        let error = scene_cuts(
            &ReaderFactory::new(),
            Path::new("no-such-picture"),
            &TimeRange::new(0, 0),
        )
        .expect_err("range");
        assert_eq!(error, SceneError::Decode);
    }
}
