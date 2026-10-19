//! Export-side audio: collect every audible clip and mix it, streamed, into
//! interleaved stereo `f32` blocks for the encoder's AAC track.
//!
//! Decodes straight from the original source files (same rule as export video:
//! no cache, no proxies). The mix policy is the MVP subset of the desktop
//! mixer — sum overlapping spans, apply per-sample volume + fades, clamp to
//! `[-1, 1]`, silence where nothing is audible. It is fail-loud: a source that
//! can't be opened or read aborts the export.
//!
//! The export loop drives [`ExportAudioMixer::mix_into`] with monotonically
//! advancing positions (one block per video frame), so each span's reader seeks
//! once at its in-point and then streams sequentially.
//!
//! Deferred (these clips export *silent* in this pass): retimed/denoised clips,
//! which the desktop pipeline renders through varispeed/RNNoise — skipped in
//! [`ExportAudioMixer::for_project`] until that DSP lands in the mobile build.

use std::path::PathBuf;

use cutlass_core::{AudioReader, DecodeError};
use cutlass_models::{Param, Project, audio_gain_at};

/// Export audio sample rate: the broadcast/web standard for video files.
pub const EXPORT_AUDIO_RATE: u32 = 48_000;
/// Export channel count (interleaved stereo).
pub const EXPORT_AUDIO_CHANNELS: u16 = 2;

const CHANNELS: usize = EXPORT_AUDIO_CHANNELS as usize;

/// One audible clip resolved to output sample frames at [`EXPORT_AUDIO_RATE`].
struct Span {
    path: PathBuf,
    /// Timeline placement in output sample frames.
    start: i64,
    end: i64,
    /// Source position (output sample frames) of the span's first sample.
    source_start: i64,
    /// Clip gain envelope, ticks rebased to clip-relative output sample frames.
    volume: Param<f32>,
    /// Fade ramp lengths in output sample frames, anchored at the span edges.
    fade_in: i64,
    fade_out: i64,
    /// Opened on first overlap, dropped with the mixer.
    reader: Option<Box<dyn AudioReader>>,
    /// Source ran out before the span's out-point: the rest pads as silence.
    exhausted: bool,
}

/// Streamed mixer over every audible span of a project's timeline.
pub struct ExportAudioMixer {
    spans: Vec<Span>,
    scratch: Vec<f32>,
}

impl ExportAudioMixer {
    /// Audible spans: clips on unmuted lanes whose media carries an audio
    /// stream. CapCut-style, a video clip keeps its own sound, so video lanes
    /// are audible too; only a clip detached to a linked audio lane defers its
    /// audio there. `None` when the timeline is silent, so callers can skip the
    /// audio track entirely.
    ///
    /// Retimed/denoised clips are skipped (their audio is deferred), so they
    /// export silent even though their video still retimes.
    pub fn for_project(project: &Project) -> Option<Self> {
        let timeline = project.timeline();
        let fps = timeline.frame_rate;
        let mut spans = Vec::new();
        for track in timeline.tracks_ordered() {
            if track.muted {
                continue;
            }
            for clip in track.clips_ordered() {
                if clip.is_silent() {
                    continue;
                }
                // Deferred: varispeed/denoise DSP isn't in the mobile build, so
                // these clips contribute no audio span yet.
                if clip.is_retimed() || clip.denoise {
                    continue;
                }
                if !timeline.carries_own_audio(clip.id) {
                    continue;
                }
                let Some(media_id) = clip.media() else {
                    continue;
                };
                let Some(media) = project.media(media_id) else {
                    continue;
                };
                if !media.has_audio {
                    continue;
                }
                let Some(source) = clip.source_range() else {
                    continue;
                };
                spans.push(Span {
                    path: media.path().to_path_buf(),
                    start: ticks_to_samples(clip.timeline.start.value, fps.num, fps.den),
                    end: ticks_to_samples(clip.timeline.end_tick(), fps.num, fps.den),
                    source_start: ticks_to_samples(
                        source.start.value,
                        source.start.rate.num,
                        source.start.rate.den,
                    ),
                    volume: clip
                        .volume
                        .map_ticks(|tick| ticks_to_samples(tick, fps.num, fps.den)),
                    fade_in: ticks_to_samples(clip.fade_in, fps.num, fps.den),
                    fade_out: ticks_to_samples(clip.fade_out, fps.num, fps.den),
                    reader: None,
                    exhausted: false,
                });
            }
        }
        if spans.is_empty() {
            None
        } else {
            Some(Self {
                spans,
                scratch: Vec::new(),
            })
        }
    }

    /// Mix every span overlapping `[pos, pos + out.len()/2)` into `out`
    /// (interleaved stereo; cleared to silence first).
    pub fn mix_into(&mut self, pos: i64, out: &mut [f32]) -> Result<(), DecodeError> {
        out.fill(0.0);
        let block_frames = (out.len() / CHANNELS) as i64;
        let block_end = pos + block_frames;

        for span in &mut self.spans {
            if span.start >= block_end || span.end <= pos || span.exhausted {
                continue;
            }
            let s = span.start.max(pos);
            let e = span.end.min(block_end);

            let reader = match &mut span.reader {
                Some(reader) => reader,
                None => {
                    let reader = cutlass_decoder::open_audio_reader(
                        &span.path,
                        EXPORT_AUDIO_RATE,
                        EXPORT_AUDIO_CHANNELS,
                    )
                    .map_err(|err| audio_err("open audio source", &span.path, err))?;
                    span.reader.insert(reader)
                }
            };

            let src_from = span.source_start + (s - span.start);
            reader
                .seek_to_frame(src_from)
                .map_err(|err| audio_err("seek audio source", &span.path, err))?;
            // A stream that starts after the requested point leaves a lead gap;
            // shift the mix-in to keep the rest aligned.
            let lead = reader
                .position()
                .map_or(0, |p| (p - src_from).clamp(0, e - s));

            let want = ((e - s) - lead) as usize;
            if want == 0 {
                continue;
            }
            self.scratch.resize(want * CHANNELS, 0.0);
            let got = reader
                .read(&mut self.scratch[..want * CHANNELS])
                .map_err(|err| audio_err("decode audio source", &span.path, err))?;
            if got < want {
                span.exhausted = true;
            }

            let offset = ((s - pos + lead) as usize) * CHANNELS;
            let unity =
                span.volume.constant() == Some(1.0) && span.fade_in == 0 && span.fade_out == 0;
            if unity {
                for (dst, src) in out[offset..]
                    .iter_mut()
                    .zip(&self.scratch[..got * CHANNELS])
                {
                    *dst += *src;
                }
            } else {
                // Per-sample-frame gain so volume automation and fades stay
                // smooth, not block-stepped.
                let span_len = span.end - span.start;
                let first = s + lead - span.start;
                for frame in 0..got {
                    let gain = audio_gain_at(
                        first + frame as i64,
                        span_len,
                        &span.volume,
                        span.fade_in,
                        span.fade_out,
                    );
                    for ch in 0..CHANNELS {
                        out[offset + frame * CHANNELS + ch] +=
                            self.scratch[frame * CHANNELS + ch] * gain;
                    }
                }
            }
        }

        for sample in out.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
        }
        Ok(())
    }
}

fn audio_err(what: &str, path: &std::path::Path, err: impl std::fmt::Display) -> DecodeError {
    DecodeError::Decode(format!("{what} {}: {err}", path.display()))
}

/// `value` ticks at `num/den` fps → sample frames at the export rate (exact
/// i128, floored) — the same conversion the desktop mixer uses.
fn ticks_to_samples(value: i64, num: i32, den: i32) -> i64 {
    if num <= 0 || den <= 0 {
        return 0;
    }
    let frames =
        i128::from(value) * i128::from(den) * i128::from(EXPORT_AUDIO_RATE) / i128::from(num);
    frames.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// Sample-frame boundary of output video frame `n` at `out_num/out_den` fps:
/// the export loop pushes audio block `[boundary(n), boundary(n+1))` after video
/// frame `n`, so audio and video cover identical wall-clock spans.
pub fn sample_boundary(n: i64, out_num: i32, out_den: i32) -> i64 {
    ticks_to_samples(n, out_num, out_den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::Rational;

    #[test]
    fn ticks_to_samples_is_exact_for_common_rates() {
        assert_eq!(ticks_to_samples(24, 24, 1), 48_000);
        assert_eq!(ticks_to_samples(1, 24, 1), 2_000);
        assert_eq!(ticks_to_samples(30_000, 30_000, 1_001), 1_001 * 48_000);
    }

    #[test]
    fn sample_boundaries_partition_the_stream() {
        assert_eq!(sample_boundary(0, 24, 1), 0);
        assert_eq!(sample_boundary(1, 24, 1), 2_000);
        assert_eq!(sample_boundary(48, 24, 1), 96_000);
        let mut prev = 0;
        for n in 1..=100 {
            let b = sample_boundary(n, 30_000, 1_001);
            assert!(b > prev, "boundaries advance");
            prev = b;
        }
    }

    #[test]
    fn silent_project_has_no_mixer() {
        let project = Project::new("test", Rational::FPS_24);
        assert!(ExportAudioMixer::for_project(&project).is_none());
    }
}
