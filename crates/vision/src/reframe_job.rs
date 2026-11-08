//! The Auto reframe job: read a clip's whole file a few times a second and
//! find where its subject is in each sample.
//!
//! The subject is what stands out nearest. Depth Anything guesses how near
//! each pixel is; each row's typical depth is taken away, so the floor
//! rising toward the camera and the sky falling away do not count, and
//! what is left is what sticks out of its surroundings toward the camera:
//! a person, a face, a thing held up. Its weighted middle, leaning a
//! little toward the frame's middle, is the subject, and how much stuck out
//! says how sure that is.
//!
//! The job needs no session and writes no draft; the shell keeps what it
//! finds with `cutlass_store::save_subject_path`.

use cutlass_eval::{EnhanceError, EnhanceFrames, ReaderFactory, ReframeAsk};
use cutlass_motion::Subject;
use cutlass_store::SubjectPath;

use crate::enhance_job::rgb_of;
use crate::relight_job::tails;
use crate::{DepthMap, DepthModel, EnhanceJobError, Image};

/// The short side each sample is read at: the subject needs little detail.
pub const REFRAME_SIDE: u32 = 252;

/// How far from the frame's middle, as a share of the frame, the lean
/// toward the middle halves.
const MIDDLE_LEAN: f32 = 0.35;

/// The share of the frame that must stick out for a sample to count fully.
const SURE_AT: f32 = 0.02;

/// Find where the subject is in each of `ask`'s samples with `model`.
/// `progress` hears the share done after each sample, and stops the job
/// by answering `false`.
pub fn find_subjects(
    readers: &ReaderFactory,
    model: &DepthModel,
    ask: &ReframeAsk,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<SubjectPath, EnhanceJobError> {
    let mut frames = EnhanceFrames::open(readers, &ask.frames()).map_err(|error| {
        EnhanceJobError::Clip(match error {
            EnhanceError::Clip { reason } => EnhanceError::Clip {
                reason: reason.replace("enhanced", "reframed"),
            },
            other => other,
        })
    })?;
    if frames.is_empty() {
        return Err(EnhanceJobError::Clip(EnhanceError::Clip {
            reason: "the clip has no frames to reframe".into(),
        }));
    }
    let count = frames.len();
    let mut size = (0, 0);
    let mut subjects = Vec::with_capacity(count);
    for k in 0..count {
        let picture = frames.frame(k).map_err(EnhanceJobError::Clip)?;
        if k == 0 {
            size = (picture.width(), picture.height());
        }
        let rgb = rgb_of(picture.rgba());
        let image = Image {
            width: picture.width(),
            height: picture.height(),
            rgb: &rgb,
        };
        let map = model
            .estimate_at(&image, REFRAME_SIDE)
            .map_err(EnhanceJobError::Vision)?;
        subjects.push(subject_of(&map));
        if !progress((k + 1) as f32 / count as f32) {
            return Err(EnhanceJobError::Stopped);
        }
    }
    Ok(SubjectPath {
        width: size.0,
        height: size.1,
        subjects,
    })
}

/// Where the subject is in one frame's depth, nearer larger.
#[must_use]
pub fn subject_of(map: &DepthMap) -> Subject {
    let (w, h) = (map.width as usize, map.height as usize);
    if w == 0 || h == 0 || map.values.len() < w * h {
        return Subject::NONE;
    }
    let (lo, hi) = tails(&map.values[..w * h]);
    let spread = hi - lo;
    let near = |v: f32| {
        if v.is_finite() {
            ((v - lo) / spread).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let (mut sum_x, mut sum_y, mut total, mut stuck) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f32);
    let mut row = vec![0.0_f32; w];
    for y in 0..h {
        for (x, slot) in row.iter_mut().enumerate() {
            *slot = near(map.values[y * w + x]);
        }
        let mut sorted = row.clone();
        let middle = w / 2;
        let typical = *sorted.select_nth_unstable_by(middle, f32::total_cmp).1;
        let fy = (y as f32 + 0.5) / h as f32;
        for (x, &n) in row.iter().enumerate() {
            let out = (n - typical).max(0.0);
            if out <= 0.0 {
                continue;
            }
            stuck += out;
            let fx = (x as f32 + 0.5) / w as f32;
            let off = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)) / (2.0 * MIDDLE_LEAN * MIDDLE_LEAN);
            let lean = 0.5 + 0.5 * (-off).exp();
            let weight = f64::from((out * n * lean).powi(2));
            sum_x += weight * f64::from(fx);
            sum_y += weight * f64::from(fy);
            total += weight;
        }
    }
    if total <= 1e-12 {
        return Subject::NONE;
    }
    Subject {
        x: sum_x / total,
        y: sum_y / total,
        weight: f64::from((stuck / (w * h) as f32 / SURE_AT).clamp(0.0, 1.0)),
    }
}

#[cfg(test)]
mod tests;
