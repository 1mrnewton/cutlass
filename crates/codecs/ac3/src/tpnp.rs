//! Transient pre-noise processing (Annex E 3.7).
//!
//! The encoder names the transient and how far back to copy. The decoder
//! cross-fades a stretch of earlier PCM over the pre-noise, then cross-fades
//! back to the original audio just before the transient.

use crate::frame::Transient;

/// Cross-fade into the copied audio, in samples.
const TC1: usize = 256;
/// Cross-fade back to the original audio, in samples.
const TC2: usize = 128;
/// Samples of previous output kept as the copy source.
pub(crate) const HISTORY: usize = 1536;

/// Replace the pre-noise in `pcm` using `history` for samples before it.
///
/// `history` is oldest first. A sample that is no longer in either buffer
/// is left alone.
pub(crate) fn apply(pcm: &mut [f32], history: &[f32], transient: Transient) {
    let transloc = usize::from(transient.location) * 4;
    let translen = usize::from(transient.length);
    let pn = (transloc % 256) + 256;
    let tot = pn + translen + TC1;
    let start = transloc as isize - tot as isize;
    let synth_start = transloc as isize - (2 * TC1 + 2 * pn) as isize;
    let read = |index: isize| -> f32 {
        if index >= 0 {
            return pcm.get(index as usize).copied().unwrap_or(0.0);
        }
        let at = history.len() as isize + index;
        if at < 0 {
            0.0
        } else {
            history.get(at as usize).copied().unwrap_or(0.0)
        }
    };
    // Snapshot first. The copy can overlap the region it overwrites.
    let synth: Vec<f32> = (0..tot)
        .map(|samp| read(synth_start + samp as isize))
        .collect();
    for (samp, &copied) in synth.iter().enumerate() {
        let index = start + samp as isize;
        if index < 0 || index >= pcm.len() as isize {
            continue;
        }
        let at = index as usize;
        pcm[at] = if samp < TC1 {
            let into = fade_in(samp, TC1);
            pcm[at] * (1.0 - into) + copied * into
        } else if samp + TC2 < tot {
            copied
        } else {
            let into = fade_in(samp + TC2 - tot, TC2);
            pcm[at] * into + copied * (1.0 - into)
        };
    }
}

/// Hanning half: 0 at the first sample, 1 at the last.
fn fade_in(index: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    let angle = std::f32::consts::PI * index as f32 / (len - 1) as f32;
    0.5 * (1.0 - angle.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_middle_of_the_window_is_a_copy_of_earlier_audio() {
        let mut pcm: Vec<f32> = (0..1536).map(|n| n as f32).collect();
        let original = pcm.clone();
        apply(
            &mut pcm,
            &[],
            Transient {
                location: 256,
                length: 64,
            },
        );
        // Transient at sample 1024. The pure copy is samples 704..896,
        // taken from samples 256..448, which the window does not cover.
        assert_eq!(&pcm[704..896], &original[256..448]);
        assert_eq!(pcm[447], original[447]);
        assert_eq!(pcm[1024], original[1024]);
        assert!((pcm[448] - original[448]).abs() < 1.0e-5);
    }

    #[test]
    fn a_source_before_this_frame_is_read_from_the_history() {
        let history: Vec<f32> = (0..HISTORY).map(|n| (n as f32) + 10_000.0).collect();
        let mut pcm = vec![1.0f32; 512];
        apply(
            &mut pcm,
            &history,
            Transient {
                location: 100,
                length: 64,
            },
        );
        // Transient at sample 400. The pure copy begins at the start of
        // this frame and comes from history, not from the constant 1.
        assert!(pcm[0] > 100.0, "{}", pcm[0]);
        assert_eq!(pcm[400], 1.0);
    }
}
