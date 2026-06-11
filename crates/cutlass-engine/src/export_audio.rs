//! Export-side audio: collect every audible clip and mix it, streamed, into
//! interleaved stereo f32 blocks for the encoder's AAC track.
//!
//! Decodes straight from the original source files (same rule as export
//! video: no cache, no proxies). The mix policy mirrors the preview mixer —
//! sum overlapping spans, clamp to [-1, 1], silence where nothing is audible
//! — but unlike preview it is fail-loud: a source that cannot be opened or
//! read aborts the export, because a deliverable with silently missing audio
//! is worse than an error.
//!
//! The export loop drives [`ExportAudioMixer::mix_into`] with monotonically
//! advancing positions (one block per video frame), so each span's reader
//! seeks once at its in-point and then streams sequentially.

use std::path::PathBuf;

use cutlass_decoder::{AUDIO_CHANNELS, AudioReader};
use cutlass_models::{Project, TrackKind};

use crate::error::EngineError;

/// Export audio sample rate: the broadcast/web standard for video files.
pub const EXPORT_AUDIO_RATE: u32 = 48_000;

/// One audible clip resolved to output sample frames at [`EXPORT_AUDIO_RATE`].
struct Span {
    path: PathBuf,
    /// Timeline placement in output sample frames.
    start: i64,
    end: i64,
    /// Source position (in output sample frames) of the span's first sample.
    source_start: i64,
    /// Opened on first overlap, dropped with the mixer.
    reader: Option<AudioReader>,
    /// Source ran out before the span's out-point: the rest pads as silence.
    exhausted: bool,
}

/// Streamed mixer over every audible span of a project's timeline.
pub struct ExportAudioMixer {
    spans: Vec<Span>,
    scratch: Vec<f32>,
}

impl ExportAudioMixer {
    /// Audible spans: clips on unmuted audio lanes whose media carries an
    /// audio stream (video lanes contribute no sound — linkage lands audio
    /// companions on audio lanes). `None` when the timeline is silent, so
    /// callers can skip the audio track entirely.
    pub fn for_project(project: &Project) -> Option<Self> {
        let timeline = project.timeline();
        let fps = timeline.frame_rate;
        let mut spans = Vec::new();
        for track in timeline.tracks_ordered() {
            if track.kind != TrackKind::Audio || track.muted {
                continue;
            }
            for clip in track.clips_ordered() {
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
    pub fn mix_into(&mut self, pos: i64, out: &mut [f32]) -> Result<(), EngineError> {
        out.fill(0.0);
        let block_frames = (out.len() / AUDIO_CHANNELS) as i64;
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
                    let reader =
                        AudioReader::open(&span.path, EXPORT_AUDIO_RATE).map_err(|err| {
                            audio_err("open audio source", &span.path, err)
                        })?;
                    span.reader.insert(reader)
                }
            };

            let src_from = span.source_start + (s - span.start);
            reader
                .seek_to_frame(src_from)
                .map_err(|err| audio_err("seek audio source", &span.path, err))?;
            // A stream that starts after the requested point leaves a lead
            // gap; shift the mix-in to keep the rest aligned (same handling
            // as the preview mixer).
            let lead = reader
                .position()
                .map_or(0, |p| (p - src_from).clamp(0, e - s));

            let want = ((e - s) - lead) as usize;
            if want == 0 {
                continue;
            }
            self.scratch.resize(want * AUDIO_CHANNELS, 0.0);
            let got = reader
                .read(&mut self.scratch[..want * AUDIO_CHANNELS])
                .map_err(|err| audio_err("decode audio source", &span.path, err))?;
            if got < want {
                span.exhausted = true;
            }

            let offset = ((s - pos + lead) as usize) * AUDIO_CHANNELS;
            for (dst, src) in out[offset..]
                .iter_mut()
                .zip(&self.scratch[..got * AUDIO_CHANNELS])
            {
                *dst += *src;
            }
        }

        for sample in out.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
        }
        Ok(())
    }
}

fn audio_err(what: &str, path: &std::path::Path, err: impl std::fmt::Display) -> EngineError {
    EngineError::Export(format!("{what} {}: {err}", path.display()))
}

/// `value` ticks at `num/den` fps → sample frames at the export rate
/// (exact i128, floored) — the same conversion the preview mixer uses.
fn ticks_to_samples(value: i64, num: i32, den: i32) -> i64 {
    if num <= 0 || den <= 0 {
        return 0;
    }
    let frames = i128::from(value) * i128::from(den) * i128::from(EXPORT_AUDIO_RATE)
        / i128::from(num);
    frames.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// Sample-frame boundary of output video frame `n` at `out_num/out_den` fps:
/// the export loop pushes audio block `[boundary(n), boundary(n+1))` after
/// video frame `n`, so audio and video cover identical wall-clock spans.
pub fn sample_boundary(n: i64, out_num: i32, out_den: i32) -> i64 {
    ticks_to_samples(n, out_num, out_den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::{MediaSource, Rational, RationalTime, TimeRange};

    #[test]
    fn ticks_to_samples_is_exact_for_common_rates() {
        // 24 ticks (1s at 24fps) = 48000 sample frames.
        assert_eq!(ticks_to_samples(24, 24, 1), 48_000);
        // One tick at 24fps = 2000 frames.
        assert_eq!(ticks_to_samples(1, 24, 1), 2_000);
        // NTSC: 30000 ticks at 30000/1001 = 1001s.
        assert_eq!(ticks_to_samples(30_000, 30_000, 1_001), 1_001 * 48_000);
    }

    #[test]
    fn sample_boundaries_partition_the_stream() {
        // 24fps: every video frame owns exactly 2000 samples.
        assert_eq!(sample_boundary(0, 24, 1), 0);
        assert_eq!(sample_boundary(1, 24, 1), 2_000);
        assert_eq!(sample_boundary(48, 24, 1), 96_000);
        // NTSC 30000/1001: boundaries are floored but cover every sample.
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

    #[test]
    fn muted_lanes_and_silent_media_are_skipped() {
        let mut project = Project::new("test", Rational::FPS_24);
        // Media without an audio stream on an audio lane: still silent.
        let silent = project.add_media(MediaSource::new(
            "/tmp/silent.mp4",
            640,
            480,
            Rational::FPS_24,
            100,
            false,
        ));
        let lane = project.add_track(TrackKind::Audio, "A1");
        project
            .add_clip(
                lane,
                silent,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();
        assert!(ExportAudioMixer::for_project(&project).is_none());
    }
}
