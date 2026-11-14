//! Remove flicker: even out a clip's brightness from frame to frame.
//!
//! Flicker is the whole picture getting brighter and darker from one frame
//! to the next: a timelapse whose exposure jumps, or lights pulsing at a
//! rate the camera's shutter beats against. It needs no model. Each
//! frame's mean brightness is measured, and the frame is scaled so it
//! matches the mean of the frames around it.
//!
//! The frames around it are [`REACH`] each side, the same number on both,
//! so a fade or any steady change in brightness is a straight line through
//! the window and is left as it is. Near the clip's ends the window
//! narrows to stay even. A cut to a much brighter or darker shot is not
//! flicker: the window stops at it, so the frames beside a cut are not
//! pulled toward the other shot.
//!
//! What this does not take out is flicker that differs across the frame,
//! such as the dark bands a rolling shutter draws under mains lighting.
//!
//! [`Deflicker`] works on the clip's frames in order, reading
//! `2 × REACH` ahead to see the cuts after a frame, and keeps those frames
//! so each is decoded once.

use std::collections::VecDeque;

/// How many frames each side of a frame its brightness is evened with.
pub const DEFLICKER_REACH: usize = 3;

const REACH: usize = DEFLICKER_REACH;

/// Two runs of [`REACH`] frames whose mean brightness differs by more than
/// this ratio meet at a cut. Flicker that alternates frame by frame moves
/// a run's mean by a third of its swing, so swings up to about a third
/// are still taken out.
const CUT: f32 = 1.25;

/// The furthest one frame is scaled, either way.
const MAX_GAIN: f32 = 2.0;

/// A frame darker than this mean, out of 255, is left as it is: black has
/// no brightness to even out.
const DARK: f32 = 2.0;

/// The weights of red, green and blue in a pixel's brightness (Rec. 709),
/// out of 256.
const WEIGHTS: [u64; 3] = [54, 183, 19];

/// One clip's frames, evened out as they are read in order.
pub struct Deflicker {
    count: usize,
    /// The mean brightness of each frame read so far.
    means: Vec<f32>,
    /// The frames read ahead of the one asked for, with their index.
    ahead: VecDeque<(usize, Vec<u8>)>,
}

impl Deflicker {
    /// For a clip of `count` frames.
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            count,
            means: Vec::new(),
            ahead: VecDeque::new(),
        }
    }

    /// Frame `index` of the clip, RGB, evened out. `read` reads one of the
    /// clip's frames as RGB; each is read once when they are asked for in
    /// order, and a frame asked for again is read again.
    pub fn frame<E>(
        &mut self,
        index: usize,
        read: &mut dyn FnMut(usize) -> Result<Vec<u8>, E>,
    ) -> Result<Vec<u8>, E> {
        let last = self.count.saturating_sub(1);
        let wanted = index.saturating_add(2 * REACH).min(last).max(index);
        while self.means.len() <= wanted {
            let next = self.means.len();
            let rgb = read(next)?;
            self.means.push(mean_brightness(&rgb));
            self.ahead.push_back((next, rgb));
        }
        while self.ahead.front().is_some_and(|(at, _)| *at < index) {
            self.ahead.pop_front();
        }
        let mut rgb = match self.ahead.front() {
            Some((at, _)) if *at == index => self.ahead.pop_front().map(|(_, rgb)| rgb),
            _ => None,
        }
        .map_or_else(|| read(index), Ok)?;
        apply(&mut rgb, self.gain(index));
        Ok(rgb)
    }

    /// What frame `index` is scaled by: the mean of the frames around it
    /// over its own.
    fn gain(&self, index: usize) -> f32 {
        let means = &self.means;
        let Some(&own) = means.get(index) else {
            return 1.0;
        };
        if own < DARK {
            return 1.0;
        }
        let last = means.len().saturating_sub(1);
        let mut reach = REACH.min(index).min(last.saturating_sub(index));
        // Stop at the nearest cut on either side, evenly.
        for step in 1..=reach {
            if self.cut_after(index - step) || self.cut_after(index + step - 1) {
                reach = step - 1;
                break;
            }
        }
        if reach == 0 {
            return 1.0;
        }
        let window = &means[index - reach..=index + reach];
        let mean = window.iter().sum::<f32>() / window.len() as f32;
        (mean / own).clamp(1.0 / MAX_GAIN, MAX_GAIN)
    }

    /// Whether the run of frames ending at `index` and the run starting
    /// after it differ enough to be two shots.
    fn cut_after(&self, index: usize) -> bool {
        let before = &self.means[index.saturating_sub(REACH - 1)..=index];
        let after = &self.means[index + 1..(index + 1 + REACH).min(self.means.len())];
        if after.is_empty() {
            return false;
        }
        let a = before.iter().sum::<f32>() / before.len() as f32;
        let b = after.iter().sum::<f32>() / after.len() as f32;
        let (low, high) = if a < b { (a, b) } else { (b, a) };
        high > low.max(DARK) * CUT
    }
}

/// The mean brightness of an RGB frame, 0 to 255, from every other pixel.
#[must_use]
pub fn mean_brightness(rgb: &[u8]) -> f32 {
    let (sum, count) =
        rgb.as_chunks::<3>()
            .0
            .iter()
            .step_by(2)
            .fold((0_u64, 0_u64), |(sum, count), p| {
                let y = u64::from(p[0]) * WEIGHTS[0]
                    + u64::from(p[1]) * WEIGHTS[1]
                    + u64::from(p[2]) * WEIGHTS[2];
                (sum + y, count + 1)
            });
    if count == 0 {
        return 0.0;
    }
    sum as f32 / (count as f32 * 256.0)
}

/// Scale every byte of `rgb` by `gain`.
fn apply(rgb: &mut [u8], gain: f32) {
    if (gain - 1.0).abs() < 1e-4 {
        return;
    }
    let table: Vec<u8> = (0..=255_u8)
        .map(|byte| (f32::from(byte) * gain).round().clamp(0.0, 255.0) as u8)
        .collect();
    for byte in rgb {
        *byte = table[usize::from(*byte)];
    }
}

#[cfg(test)]
mod tests;
