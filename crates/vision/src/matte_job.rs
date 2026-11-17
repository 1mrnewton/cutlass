//! The Remove BG job: read a clip's stretch frame by frame, find its
//! subject, and write a grey matte video into the project (white kept,
//! black removed), where eval cuts the clip with it.
//!
//! Auto removal ([`matte_clip`]) asks a [`Matter`] about each frame on its
//! own. A model that sees one frame at a time wavers on soft edges from
//! frame to frame, so a pixel that barely changed is eased toward the last
//! frame's value; one that changed a lot (the subject moved) takes the new
//! value at once.
//!
//! Custom removal ([`follow_clip`]) starts a [`Follower`] on the first
//! picked frame and runs it to the end of the stretch, correcting it on
//! each later pick, then runs it again from the first pick back to the
//! start. The backward run reads the file forward in short chunks, since a
//! file decodes forward from its key frames, and keeps its mattes packed
//! until they are written in order.
//!
//! The matte is the frame's size, its short side at most [`MATTE_SIDE`].
//! The job needs no session and writes no draft, so a shell runs it off the
//! main thread. A stopped or failed job leaves no matte behind.

use std::path::{Path, PathBuf};

use cutlass_effects::{Brush, Pick};
use cutlass_eval::{EnhanceError, EnhanceFrames, MatteAsk, ReaderFactory};
use cutlass_export::{ExportError, PictureMovie, video_bitrate, write_pictures};
use cutlass_model::FrameRate;
use cutlass_store::matte_file;
use image::imageops::{self, FilterType};
use image::{GrayImage, ImageBuffer, Luma};

use crate::enhance_job::rgb_of;
use crate::{Click, EnhanceJobError, Follower, Image, Matte, Matter, Prompt, VisionError};

/// The longest short side a matte video has: enough for hair at 1080p,
/// and a 4K clip's matte is stretched.
pub const MATTE_SIDE: u32 = 1080;

/// A pixel whose matte moved less than this from the last frame is eased.
const STEADY: u8 = 48;

/// How many frames the backward run reads at once.
const CHUNK: usize = 16;

/// The most a smart brush's clicks are apart, in brush radii.
const SMART_STEP: f64 = 2.0;

/// Make `ask`'s matte video in the project at `project` with `matter`, for
/// Auto removal, and return its path. `progress` hears the share done after
/// each frame, and stops the job by answering `false`.
pub fn matte_clip(
    readers: &ReaderFactory,
    encoders: &cutlass_encode::EncodeFactory,
    matter: &mut dyn Matter,
    ask: &MatteAsk,
    project: &Path,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<PathBuf, EnhanceJobError> {
    let mut frames = open_frames(readers, ask)?;
    let count = frames.len();
    let mut last: Option<Vec<u8>> = None;
    let mut make = |k: usize| -> Result<Matte, EnhanceJobError> {
        let (rgb, width, height) = read_rgb(&mut frames, k)?;
        let image = Image {
            width,
            height,
            rgb: &rgb,
        };
        let mut matte = matter.matte(&image).map_err(EnhanceJobError::Vision)?;
        check_size(&matte, width, height, k)?;
        if let Some(last) = &last
            && last.len() == matte.alpha.len()
        {
            ease(last, &mut matte.alpha);
        }
        last = Some(matte.alpha.clone());
        Ok(matte)
    };
    let total = count as f32;
    write_mattes(encoders, ask, project, count, &mut make, &mut |done| {
        progress(done as f32 / total)
    })
}

/// Make `ask`'s matte video for Custom removal with `follower`, from the
/// picks in `ask.matting`, and return its path. `progress` hears the share
/// done, and stops the job by answering `false`.
pub fn follow_clip(
    readers: &ReaderFactory,
    encoders: &cutlass_encode::EncodeFactory,
    follower: &mut dyn Follower,
    ask: &MatteAsk,
    project: &Path,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<PathBuf, EnhanceJobError> {
    let mut frames = open_frames(readers, ask)?;
    let count = frames.len();
    let picks = picked_frames(&ask.matting.picks, ask, count);
    let Some(&(first, _)) = picks.first() else {
        return Err(EnhanceJobError::Clip(EnhanceError::Clip {
            reason: "Custom removal has nothing picked yet".into(),
        }));
    };
    // The backward run and the forward run each read the frames they
    // cover once: `first + 1` frames back, then `count - first` forward.
    let total = (first + 1 + count - first) as f32;
    let mut heard = 0_usize;
    let mut tell = |progress: &mut dyn FnMut(f32) -> bool| -> bool {
        heard += 1;
        progress(heard as f32 / total)
    };

    // From the first pick back to the start, read forward a chunk at a
    // time and followed backward.
    let mut before: Vec<Packed> = Vec::with_capacity(first);
    follower.reset();
    let mut end = first + 1;
    while end > 0 {
        let start = end.saturating_sub(CHUNK);
        let mut chunk = Vec::with_capacity(end - start);
        for k in start..end {
            chunk.push(read_rgb(&mut frames, k)?);
        }
        for (offset, (rgb, width, height)) in chunk.iter().enumerate().rev() {
            let k = start + offset;
            let image = Image {
                width: *width,
                height: *height,
                rgb,
            };
            let prompt = (k == first).then(|| &picks[0].1);
            let matte = prompted_step(follower, &image, prompt, k)?;
            if k < first {
                before.push(Packed::pack(&matte.alpha));
            }
            if !tell(progress) {
                return Err(EnhanceJobError::Stopped);
            }
        }
        end = start;
    }
    before.reverse();

    // From the first pick to the end, written as it goes.
    follower.reset();
    let mut size = None;
    let mut make = |k: usize| -> Result<Matte, EnhanceJobError> {
        let (rgb, width, height) = read_rgb(&mut frames, k)?;
        size = Some((width, height));
        if k < first {
            let alpha = before.get(k).map(Packed::unpack).ok_or_else(|| lost(k))?;
            return Matte::new(width, height, alpha).ok_or_else(|| lost(k));
        }
        let image = Image {
            width,
            height,
            rgb: &rgb,
        };
        let prompt = picks
            .iter()
            .find(|(at, _)| *at == k)
            .map(|(_, prompt)| prompt);
        prompted_step(follower, &image, prompt, k)
    };
    write_mattes(encoders, ask, project, count, &mut make, &mut |done| {
        // The frames before the first pick were counted on the way back.
        if done as usize <= first {
            true
        } else {
            tell(progress)
        }
    })
}

/// One step of `follower` on frame `k`, with `prompt` when the frame was
/// picked. A pick's plain brush strokes paint the answer exactly, and the
/// painted answer is given back so the follower carries it on.
fn prompted_step(
    follower: &mut dyn Follower,
    image: &Image<'_>,
    prompt: Option<&PickPrompt>,
    k: usize,
) -> Result<Matte, EnhanceJobError> {
    let vision = EnhanceJobError::Vision;
    let Some(prompt) = prompt else {
        let matte = follower.step(image, None).map_err(vision)?;
        check_size(&matte, image.width, image.height, k)?;
        return Ok(matte);
    };
    let clicks = prompt.clicks(image.width, image.height);
    let painted = prompt.painted();
    if painted.is_empty() {
        let matte = follower
            .step(image, Some(&Prompt { clicks, mask: None }))
            .map_err(vision)?;
        check_size(&matte, image.width, image.height, k)?;
        return Ok(matte);
    }
    let mut base = if clicks.is_empty() {
        Matte::new(
            image.width,
            image.height,
            vec![0; image.width as usize * image.height as usize],
        )
        .ok_or_else(|| lost(k))?
    } else {
        let matte = follower
            .step(image, Some(&Prompt { clicks, mask: None }))
            .map_err(vision)?;
        check_size(&matte, image.width, image.height, k)?;
        matte
    };
    paint(&mut base, &painted);
    let answer = follower
        .step(
            image,
            Some(&Prompt {
                clicks: Vec::new(),
                mask: Some(base.clone()),
            }),
        )
        .map_err(vision)?;
    check_size(&answer, image.width, image.height, k)?;
    // The user's paint is the truth on the frame they painted.
    Ok(base)
}

/// A pick, ready for the follower: the frame's clicks and strokes, in
/// unit coordinates until the frame's size is known.
struct PickPrompt {
    pick: Pick,
}

impl PickPrompt {
    /// The pick's clicks, and a smart brush's clicks along its path, in a
    /// `width` by `height` frame's pixels.
    fn clicks(&self, width: u32, height: u32) -> Vec<Click> {
        let (w, h) = (f64::from(width), f64::from(height));
        let short = w.min(h);
        let at = |x: f64, y: f64, keep: bool| {
            #[allow(clippy::cast_possible_truncation)]
            Click {
                x: (x * w).clamp(0.0, w) as f32,
                y: (y * h).clamp(0.0, h) as f32,
                keep,
            }
        };
        let mut clicks: Vec<Click> = self
            .pick
            .clicks
            .iter()
            .map(|click| at(click.x, click.y, click.keep))
            .collect();
        for brush in self.pick.brushes.iter().filter(|brush| brush.smart) {
            let step = (brush.radius * short * SMART_STEP).max(1.0);
            let mut walked = f64::INFINITY;
            let mut previous: Option<[f64; 2]> = None;
            for point in &brush.points {
                if let Some(p) = previous {
                    walked += ((point[0] - p[0]) * w).hypot((point[1] - p[1]) * h);
                }
                // A hair under a step is a step: sums of floats drift.
                if walked >= step - 1e-6 {
                    clicks.push(at(point[0], point[1], brush.keep));
                    walked = 0.0;
                }
                previous = Some(*point);
            }
        }
        clicks.truncate(crate::MAX_CLICKS);
        clicks
    }

    /// The plain brush strokes, which paint exactly.
    fn painted(&self) -> Vec<&Brush> {
        self.pick
            .brushes
            .iter()
            .filter(|brush| !brush.smart)
            .collect()
    }
}

/// Paint `strokes` onto `matte`: a keeping stroke sets what it covers to
/// 255, an erasing one to 0, each a round brush along its path.
fn paint(matte: &mut Matte, strokes: &[&Brush]) {
    let (w, h) = (f64::from(matte.width), f64::from(matte.height));
    let short = w.min(h);
    let (width, height) = (matte.width as usize, matte.height as usize);
    for brush in strokes {
        let radius = (brush.radius * short).max(0.5);
        let value = if brush.keep { 255 } else { 0 };
        let points: Vec<[f64; 2]> = brush.points.iter().map(|[x, y]| [x * w, y * h]).collect();
        let segments: Vec<([f64; 2], [f64; 2])> = if points.len() == 1 {
            vec![(points[0], points[0])]
        } else {
            points.windows(2).map(|pair| (pair[0], pair[1])).collect()
        };
        for (a, b) in segments {
            let (x0, x1) = (a[0].min(b[0]) - radius, a[0].max(b[0]) + radius);
            let (y0, y1) = (a[1].min(b[1]) - radius, a[1].max(b[1]) + radius);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let (xs, xe, ys, ye) = (
                x0.floor().max(0.0) as usize,
                (x1.ceil().max(0.0) as usize).min(width),
                y0.floor().max(0.0) as usize,
                (y1.ceil().max(0.0) as usize).min(height),
            );
            for y in ys..ye {
                for x in xs..xe {
                    let p = [x as f64 + 0.5, y as f64 + 0.5];
                    if distance_to_segment(p, a, b) <= radius {
                        matte.alpha[y * width + x] = value;
                    }
                }
            }
        }
    }
}

fn distance_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length = dx * dx + dy * dy;
    let t = if length > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
}

/// Each pick's frame index in a stretch of `count` frames, nearest its
/// source time, in order; two picks on one frame keep the later.
fn picked_frames(picks: &[Pick], ask: &MatteAsk, count: usize) -> Vec<(usize, PickPrompt)> {
    let mut out: Vec<(usize, PickPrompt)> = Vec::new();
    if count == 0 {
        return out;
    }
    for pick in picks {
        let since = (pick.frame.0 - ask.range.start.0).max(0) as f64 / 1e6;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = ((since * ask.rate).round() as usize).min(count - 1);
        let prompt = PickPrompt { pick: pick.clone() };
        match out.last_mut() {
            Some((last, kept)) if *last == k => *kept = prompt,
            _ => out.push((k, prompt)),
        }
    }
    out
}

fn open_frames(readers: &ReaderFactory, ask: &MatteAsk) -> Result<EnhanceFrames, EnhanceJobError> {
    let frames = EnhanceFrames::open(readers, &ask.frames()).map_err(|error| {
        EnhanceJobError::Clip(match error {
            EnhanceError::Clip { reason } => EnhanceError::Clip {
                reason: reason.replace("enhanced", "cut out"),
            },
            other => other,
        })
    })?;
    if frames.is_empty() {
        return Err(EnhanceJobError::Clip(EnhanceError::Clip {
            reason: "the clip has no frames to cut out".into(),
        }));
    }
    Ok(frames)
}

fn read_rgb(frames: &mut EnhanceFrames, k: usize) -> Result<(Vec<u8>, u32, u32), EnhanceJobError> {
    let picture = frames.frame(k).map_err(EnhanceJobError::Clip)?;
    Ok((rgb_of(picture.rgba()), picture.width(), picture.height()))
}

fn check_size(matte: &Matte, width: u32, height: u32, k: usize) -> Result<(), EnhanceJobError> {
    if (matte.width, matte.height) == (width, height)
        && matte.alpha.len() == width as usize * height as usize
    {
        return Ok(());
    }
    Err(EnhanceJobError::Vision(VisionError::Model {
        message: format!(
            "frame {k}'s matte is {}x{}, not {width}x{height}",
            matte.width, matte.height
        ),
    }))
}

/// Ease `alpha` toward `last` where it barely moved.
fn ease(last: &[u8], alpha: &mut [u8]) {
    for (now, was) in alpha.iter_mut().zip(last) {
        if now.abs_diff(*was) < STEADY {
            #[allow(clippy::cast_possible_truncation)]
            let mid = (u16::from(*now) + u16::from(*was)).div_ceil(2) as u8;
            *now = mid;
        }
    }
}

/// The size a `width` by `height` frame's matte video is: its short side
/// at most [`MATTE_SIDE`], both sides even.
#[must_use]
pub fn matte_size(width: u32, height: u32) -> (u32, u32) {
    let short = width.min(height).max(1);
    let (w, h) = if short > MATTE_SIDE {
        let scale = f64::from(MATTE_SIDE) / f64::from(short);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let scaled = (
            (f64::from(width) * scale).round() as u32,
            (f64::from(height) * scale).round() as u32,
        );
        scaled
    } else {
        (width, height)
    };
    ((w & !1).max(2), (h & !1).max(2))
}

/// Write `count` mattes from `make` as `ask`'s matte video.
fn write_mattes(
    encoders: &cutlass_encode::EncodeFactory,
    ask: &MatteAsk,
    project: &Path,
    count: usize,
    make: &mut dyn FnMut(usize) -> Result<Matte, EnhanceJobError>,
    progress: &mut dyn FnMut(u64) -> bool,
) -> Result<PathBuf, EnhanceJobError> {
    let rate = FrameRate::from_stored_fps(ask.rate).ok_or_else(|| {
        EnhanceJobError::Clip(EnhanceError::Clip {
            reason: format!("{} frames a second is not a frame rate", ask.rate),
        })
    })?;
    let first = make(0)?;
    if first.width < 2 || first.height < 2 {
        return Err(EnhanceJobError::Clip(EnhanceError::Frame {
            reason: format!("a {}x{} frame is too small", first.width, first.height),
        }));
    }
    let (width, height) = matte_size(first.width, first.height);
    let movie = PictureMovie {
        width,
        height,
        frame_rate: rate,
        // Twice a picture's rate: soft edges are what the matte is for.
        bitrate: video_bitrate(width.min(height)).saturating_mul(2),
        frames: count as u64,
    };
    let file = matte_file(project, &ask.key(), &ask.range)
        .map_err(|error| EnhanceJobError::Store(error.to_string()))?;
    let mut first = Some(first);
    let mut failed = None;
    let written = write_pictures(
        encoders,
        &movie,
        &file,
        &mut |index| {
            let made = usize::try_from(index)
                .map_err(|_| lost(usize::MAX))
                .and_then(|k| match first.take() {
                    Some(matte) if k == 0 => Ok(matte),
                    kept => {
                        first = kept;
                        make(k)
                    }
                })
                .and_then(|matte| grey_rgba(&matte, (width, height)));
            made.map_err(|error| {
                let message = error.to_string();
                failed = Some(error);
                ExportError::Picture(message)
            })
        },
        &mut |done, _| progress(done),
    );
    match written {
        Ok(()) => Ok(file),
        Err(ExportError::Cancelled) => {
            let _ = std::fs::remove_file(&file);
            Err(EnhanceJobError::Stopped)
        }
        Err(error) => {
            let _ = std::fs::remove_file(&file);
            Err(failed.unwrap_or(EnhanceJobError::Movie(error)))
        }
    }
}

/// `matte` at `size` as grey RGBA.
fn grey_rgba(matte: &Matte, size: (u32, u32)) -> Result<Vec<u8>, EnhanceJobError> {
    let alpha = if (matte.width, matte.height) == size {
        matte.alpha.clone()
    } else {
        let image: GrayImage =
            ImageBuffer::<Luma<u8>, _>::from_raw(matte.width, matte.height, matte.alpha.clone())
                .ok_or_else(|| lost(0))?;
        imageops::resize(&image, size.0, size.1, FilterType::Triangle).into_raw()
    };
    Ok(alpha.iter().flat_map(|a| [*a, *a, *a, 255]).collect())
}

fn lost(k: usize) -> EnhanceJobError {
    EnhanceJobError::Clip(EnhanceError::Frame {
        reason: format!("the clip has no frame {k}"),
    })
}

/// A matte packed as runs of one value: most of a matte is all kept or all
/// removed, so a long backward run fits in memory.
struct Packed(Vec<(u32, u8)>);

impl Packed {
    fn pack(alpha: &[u8]) -> Self {
        let mut runs: Vec<(u32, u8)> = Vec::new();
        for value in alpha {
            match runs.last_mut() {
                Some((count, last)) if *last == *value && *count < u32::MAX => *count += 1,
                _ => runs.push((1, *value)),
            }
        }
        Self(runs)
    }

    fn unpack(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (count, value) in &self.0 {
            out.resize(out.len() + *count as usize, *value);
        }
        out
    }
}

#[cfg(test)]
mod tests;
