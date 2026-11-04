//! The mixed slice queued at one playhead.
//!
//! The bytes are interleaved stereo `f32` at 48 kHz. A mono mix is duplicated.
//! A missing file is silence. The draft is not written.

use std::path::Path;

use cutlass_eval::{ClipAsk, ClockKind, Heard, MixWindow, ReaderFactory, Snapshot, audio_slice};
use cutlass_model::{Draft, Micros};

use crate::apple::{AudioOutput, FRAME_BYTES, MAX_SLICE_BYTES, RATE};

/// Frames queued when play starts, before the next clock step arrives.
pub const PLAY_FRAMES: u32 = 1_024;

/// How many preview frames `elapsed_us` covers, capped at one queue slice.
#[must_use]
pub fn preview_frames(elapsed_us: i64) -> u32 {
    if elapsed_us <= 0 {
        return 0;
    }
    let count = i128::from(elapsed_us).saturating_mul(i128::from(RATE)) / 1_000_000;
    let count = u32::try_from(count).unwrap_or(max_frames());
    count.min(max_frames())
}

/// One mix to queue. `heard` keeps the open files for `revision`.
pub struct PreviewMix<'a> {
    pub draft: &'a Draft,
    pub revision: u64,
    pub time: Micros,
    pub frames: u32,
    pub readers: &'a ReaderFactory,
    pub heard: &'a mut Heard,
    pub root: Option<&'a Path>,
}

/// Queue the mix at `mix.time`. A silent host is left alone. Zero frames
/// leaves the current slice in place. A failed mix queues silence so the
/// picture can still be shown.
pub fn queue_mix(output: &mut AudioOutput, mix: PreviewMix<'_>) {
    let _ = queue_mix_peak(output, mix);
}

/// [`queue_mix`], and the left and right peaks of what was queued, or `None`
/// when nothing was.
pub(crate) fn queue_mix_peak(output: &mut AudioOutput, mix: PreviewMix<'_>) -> Option<[f32; 2]> {
    if output.gap().is_some() {
        return None;
    }
    let frames = mix.frames.min(max_frames());
    if frames == 0 {
        return None;
    }
    let bytes = mixed_bytes(mix, frames);
    output.queue(&bytes);
    Some(crate::meter::window_peak(&bytes))
}

/// `frames` of mixed stereo at `mix.time`. A failed mix is silence of the
/// same length.
pub(crate) fn window_bytes(mix: PreviewMix<'_>, frames: u32) -> Vec<u8> {
    let mut bytes = mixed_bytes(mix, frames);
    let want = usize::try_from(frames).unwrap_or(0) * FRAME_BYTES;
    bytes.resize(want, 0);
    bytes
}

fn max_frames() -> u32 {
    u32::try_from(MAX_SLICE_BYTES / FRAME_BYTES).unwrap_or(0)
}

fn mixed_bytes(mix: PreviewMix<'_>, frames: u32) -> Vec<u8> {
    let mixed = audio_slice(
        &Snapshot::new(mix.draft),
        mix.time,
        MixWindow {
            rate: RATE,
            limiter: 0,
            frames,
        },
        ClockKind::Playback,
        |segment, time, count, rate| {
            mix.heard.clip(ClipAsk {
                readers: mix.readers,
                draft: mix.draft,
                revision: mix.revision,
                segment,
                time,
                frames: count,
                rate,
                root: mix.root,
            })
        },
    );
    match mixed {
        Ok(mixed) => stereo_bytes(&mixed),
        Err(_) => Vec::new(),
    }
}

fn stereo_bytes(mixed: &cutlass_eval::MixResult) -> Vec<u8> {
    let pcm = &mixed.pcm;
    let Some(samples) = pcm.as_f32() else {
        return Vec::new();
    };
    let channels = pcm.layout.len();
    if channels == 0 {
        return Vec::new();
    }
    let frames = samples.len() / channels;
    let mut out = Vec::new();
    for frame in 0..frames {
        if out.len() + FRAME_BYTES > MAX_SLICE_BYTES {
            break;
        }
        let base = frame.saturating_mul(channels);
        let left = samples.get(base).copied().unwrap_or(0.0);
        let right = if channels == 1 {
            left
        } else {
            samples.get(base.saturating_add(1)).copied().unwrap_or(0.0)
        };
        out.extend_from_slice(&left.to_le_bytes());
        out.extend_from_slice(&right.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::Path;

    use cutlass_container::{Demux, OpenError};
    use cutlass_model::{
        Draft, MaterialAudio, MaterialId, Micros, Segment, TimeRange, Track, TrackType,
    };
    use cutlass_session::Session;

    use super::{PreviewMix, preview_frames, queue_mix};
    use crate::apple::{AudioHost, AudioOutput};
    use crate::playhead::Player;

    fn open_wav(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
        let file = File::open(path).map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?;
        let length = file
            .metadata()
            .map_err(|error| OpenError::Source {
                message: error.to_string(),
            })?
            .len();
        Ok(Box::new(cutlass_riff::WavDemux::open(file, length)?))
    }

    fn readers() -> cutlass_eval::ReaderFactory {
        let mut readers = cutlass_eval::ReaderFactory::new();
        readers.register_container(cutlass_riff::probe, open_wav);
        cutlass_pcm::register_decoders(readers.decoders_mut());
        readers
    }

    fn write_tone(path: &Path) {
        let payload = [0xFF_u8, 0x7F, 0xFF, 0x7F, 0xFF, 0x7F, 0xFF, 0x7F];
        let data_bytes = u32::try_from(payload.len()).expect("wav size");
        let rate = 48_000_u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(path, bytes).expect("wav");
    }

    fn draft(path: &Path) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.duration = Micros(1_000_000);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: path.display().to_string(),
            duration: Micros(1_000_000),
            ..MaterialAudio::default()
        });
        let mut track = Track::main("audio");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("clip", "tone", TimeRange::new(0, 1_000_000));
        segment.volume = 0.5;
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        track.segments.push(segment);
        draft.tracks = vec![track];
        draft
    }

    fn samples(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    #[test]
    fn play_queues_the_mixed_slice_and_a_silent_host_does_not() {
        let path = std::env::temp_dir().join(format!("cutlass-preview-{}", std::process::id()));
        write_tone(&path);
        let draft = draft(&path);
        let before = draft.clone();
        let session = Session::open(draft);
        let readers = readers();
        let mut heard = cutlass_eval::Heard::new();
        let mut output = AudioOutput::open(AudioHost::Software);
        queue_mix(
            &mut output,
            PreviewMix {
                draft: session.draft(),
                revision: session.revision(),
                time: Micros(0),
                frames: 4,
                readers: &readers,
                heard: &mut heard,
                root: None,
            },
        );
        assert_eq!(samples(output.queued()), vec![0.5; 8]);
        assert_eq!(session.draft(), &before);

        let mut silent = AudioOutput::open(AudioHost::Linux);
        queue_mix(
            &mut silent,
            PreviewMix {
                draft: session.draft(),
                revision: session.revision(),
                time: Micros(0),
                frames: 4,
                readers: &readers,
                heard: &mut heard,
                root: None,
            },
        );
        assert!(silent.queued().is_empty());
        assert!(silent.gap().is_some());

        let mut player = Player::new();
        player.set_readers(readers);
        player.play(&session).expect("play");
        assert_eq!(samples(player.output().queued()), vec![0.5; 8]);
        assert_eq!(player.output().queued().len(), 4 * 8);
        assert_eq!(preview_frames(1_000_000), 8_192);
        assert_eq!(preview_frames(0), 0);
        let _ = std::fs::remove_file(&path);
    }
}
