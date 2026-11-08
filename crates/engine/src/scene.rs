//! Where a picture changes enough to be another scene.
//!
//! The score is the mean absolute difference of a small luma grid, and the
//! change in that difference, on a 0–100 scale. A cut is a score of at least
//! [`SCENE_SCORE`]. The file is read forward from the start of the range.
//! Samples are at least a fifth of a second apart, and there are at most
//! [`MAX_LOOKS`] of them. A still, or a clip with no change at that step,
//! returns no cuts.

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
#[must_use]
pub fn cut_times(times: &[i64], grids: &[[u8; SCENE_CELLS]]) -> Vec<i64> {
    let mut cuts = Vec::new();
    let mut previous_mafd = 0.0;
    let mut previous: Option<&[u8; SCENE_CELLS]> = None;
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
        if score >= SCENE_SCORE {
            cuts.push(*time);
        }
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
        } if *pixel == PixelFormat::Yuv420p => {
            let y_len = usize::try_from(*width)
                .ok()?
                .checked_mul(usize::try_from(*height).ok()?)?;
            let plane = bytes.get(..y_len)?;
            downsample(plane, *width, *height)
        }
        DecodedFrame::Still(image) => still_grid(image),
        DecodedFrame::Coded { .. }
        | DecodedFrame::Platform { .. }
        | DecodedFrame::Surface { .. } => None,
    }
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
