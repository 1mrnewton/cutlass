use std::fs::File;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use cutlass_model::TimeRange;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as MediaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::units::{Time, TimeBase, Timestamp};

use crate::prepare::{Pcm, PcmSource, PrepareError};

/// A [`PcmSource`] that reads one media file.
///
/// This is the reader until `container` and `decode` can return PCM for a
/// range. Callers keep using [`PcmSource`] either way.
#[derive(Clone, Debug)]
pub struct FileSource {
    path: PathBuf,
}

impl FileSource {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl PcmSource for FileSource {
    fn read(&mut self, range: &TimeRange) -> Result<Pcm, PrepareError> {
        if range.duration.0 <= 0 || range.start.0 < 0 {
            return Err(PrepareError::Range);
        }
        if range.start.0 == 0 {
            return decode_file(&self.path, range, false);
        }
        // A seek that this file cannot do is reported as `Decode`. Read from
        // the start in that case. A second failure is the file itself.
        match decode_file(&self.path, range, true) {
            Err(PrepareError::Decode) => decode_file(&self.path, range, false),
            other => other,
        }
    }
}

fn decode_file(path: &Path, range: &TimeRange, seek: bool) -> Result<Pcm, PrepareError> {
    let file = File::open(path).map_err(|_| PrepareError::Open)?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let mut reader = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|_| PrepareError::Decode)?;
    let track = reader
        .default_track(TrackType::Audio)
        .ok_or(PrepareError::NoAudio)?;
    let track_id = track.id;
    let time_base = track.time_base;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or(PrepareError::NoAudio)?;
    let sample_rate = params.sample_rate.ok_or(PrepareError::SampleRate)?;
    if sample_rate == 0 {
        return Err(PrepareError::SampleRate);
    }
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|_| PrepareError::Decode)?;

    let start_sample = us_to_sample(range.start.0, sample_rate)?;
    let end_us = range
        .start
        .0
        .checked_add(range.duration.0)
        .ok_or(PrepareError::Range)?;
    let end_sample = us_to_sample(end_us, sample_rate)?;
    if seek && start_sample > 0 {
        let time = Time::from_micros(range.start.0);
        reader
            .seek(
                SeekMode::Accurate,
                SeekTo::Time {
                    time,
                    track_id: Some(track_id),
                },
            )
            .map_err(|_| PrepareError::Decode)?;
        decoder.reset();
    }

    let mut mono = Vec::new();
    if end_sample > start_sample {
        let count = usize::try_from(end_sample - start_sample).unwrap_or(mono.capacity());
        mono.reserve(count);
    }
    let mut interleaved = Vec::new();
    let mut media_end = 0_u64;
    loop {
        let packet = match reader.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(MediaError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(_) => return Err(PrepareError::Decode),
        };
        if packet.track_id != track_id {
            continue;
        }
        let origin = frame_origin(packet.pts, time_base, sample_rate)?;
        let audio = decoder.decode(&packet).map_err(|_| PrepareError::Decode)?;
        if origin >= end_sample {
            break;
        }
        push_frames(
            &audio,
            origin,
            start_sample,
            end_sample,
            &mut interleaved,
            &mut mono,
        )?;
        let frames = u64::try_from(audio.frames()).map_err(|_| PrepareError::TooLong)?;
        media_end = media_end.max(origin.saturating_add(frames));
    }
    if mono.is_empty() && media_end > 0 && start_sample >= media_end {
        return Err(PrepareError::Range);
    }
    Ok(Pcm {
        sample_rate,
        samples: mono,
    })
}

fn push_frames(
    audio: &GenericAudioBufferRef<'_>,
    origin: u64,
    start_sample: u64,
    end_sample: u64,
    interleaved: &mut Vec<f32>,
    mono: &mut Vec<f32>,
) -> Result<(), PrepareError> {
    let frames = audio.frames();
    let width = audio.samples_interleaved();
    if frames == 0 || width == 0 {
        return Ok(());
    }
    if !width.is_multiple_of(frames) {
        return Err(PrepareError::Decode);
    }
    let channels = width / frames;
    interleaved.resize(width, 0.0);
    audio.copy_to_slice_interleaved(&mut interleaved[..]);
    for frame in 0..frames {
        let index = origin.saturating_add(u64::try_from(frame).map_err(|_| PrepareError::TooLong)?);
        if index < start_sample || index >= end_sample {
            continue;
        }
        let offset = frame * channels;
        let mut sum = 0.0_f32;
        for sample in &interleaved[offset..offset + channels] {
            if !sample.is_finite() {
                return Err(PrepareError::Samples);
            }
            sum += sample;
        }
        mono.push(sum / channels as f32);
    }
    Ok(())
}

fn frame_origin(
    pts: Timestamp,
    time_base: Option<TimeBase>,
    rate: u32,
) -> Result<u64, PrepareError> {
    let tick = pts.get();
    if tick <= 0 {
        return Ok(0);
    }
    let tick = u128::from(u64::try_from(tick).map_err(|_| PrepareError::TooLong)?);
    let Some(time_base) = time_base else {
        return u64::try_from(tick).map_err(|_| PrepareError::TooLong);
    };
    let sample = tick
        .checked_mul(u128::from(time_base.numer.get()))
        .and_then(|value| value.checked_mul(u128::from(rate)))
        .ok_or(PrepareError::TooLong)?
        / u128::from(time_base.denom.get());
    u64::try_from(sample).map_err(|_| PrepareError::TooLong)
}

fn us_to_sample(us: i64, rate: u32) -> Result<u64, PrepareError> {
    if us < 0 {
        return Err(PrepareError::Range);
    }
    let sample = u128::from(u64::try_from(us).map_err(|_| PrepareError::Range)?)
        .checked_mul(u128::from(rate))
        .ok_or(PrepareError::TooLong)?
        / 1_000_000;
    u64::try_from(sample).map_err(|_| PrepareError::TooLong)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempWav {
        path: PathBuf,
    }

    impl TempWav {
        fn write(name: &str, rate: u32, channels: u16, samples: &[i16]) -> Self {
            let path = std::env::temp_dir().join(format!("cutlass-transcription-{name}.wav"));
            let data_bytes = samples.len() * 2;
            let block = channels * 2;
            let mut bytes = Vec::with_capacity(44 + data_bytes);
            bytes.extend_from_slice(b"RIFF");
            bytes.extend_from_slice(&(36 + data_bytes as u32).to_le_bytes());
            bytes.extend_from_slice(b"WAVEfmt ");
            bytes.extend_from_slice(&16_u32.to_le_bytes());
            bytes.extend_from_slice(&1_u16.to_le_bytes());
            bytes.extend_from_slice(&channels.to_le_bytes());
            bytes.extend_from_slice(&rate.to_le_bytes());
            bytes.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
            bytes.extend_from_slice(&block.to_le_bytes());
            bytes.extend_from_slice(&16_u16.to_le_bytes());
            bytes.extend_from_slice(b"data");
            bytes.extend_from_slice(&(data_bytes as u32).to_le_bytes());
            for sample in samples {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            std::fs::write(&path, bytes).unwrap();
            Self { path }
        }
    }

    impl Drop for TempWav {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn a_wav_round_trips_the_sample_and_a_later_range() {
        let wav = TempWav::write(
            "tone",
            8_000,
            1,
            &[0, 0, 0, 0, 16_384, 16_384, 16_384, 16_384],
        );
        let mut source = FileSource::new(&wav.path);
        let full = source.read(&TimeRange::new(0, 1_000)).unwrap();
        assert_eq!(full.sample_rate, 8_000);
        assert_eq!(full.samples.len(), 8);
        assert!((full.samples[4] - 0.5).abs() < 1e-3);

        let later = source.read(&TimeRange::new(500, 500)).unwrap();
        assert_eq!(later.samples.len(), 4);
        assert!(
            later
                .samples
                .iter()
                .all(|sample| (sample - 0.5).abs() < 1e-3)
        );
    }

    #[test]
    fn stereo_is_downmixed_to_mono() {
        let wav = TempWav::write("stereo", 8_000, 2, &[16_384, 0]);
        let mut source = FileSource::new(&wav.path);
        let pcm = source.read(&TimeRange::new(0, 125)).unwrap();
        assert_eq!(pcm.samples.len(), 1);
        assert!((pcm.samples[0] - 0.25).abs() < 1e-3);
    }

    #[test]
    fn a_missing_file_is_refused() {
        let mut source =
            FileSource::new(std::env::temp_dir().join("cutlass-transcription-missing.wav"));
        let error = source.read(&TimeRange::new(0, 1_000)).unwrap_err();
        assert_eq!(error, PrepareError::Open);
    }

    #[test]
    fn a_truncated_wav_is_refused() {
        let path = std::env::temp_dir().join("cutlass-transcription-truncated.wav");
        std::fs::write(&path, b"RIFF").unwrap();
        let mut source = FileSource::new(&path);
        let error = source.read(&TimeRange::new(0, 1_000)).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(error, PrepareError::Decode);
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        let wav = TempWav::write("short", 8_000, 1, &[0, 0, 0, 0]);
        let mut source = FileSource::new(&wav.path);
        let error = source.read(&TimeRange::new(2_000, 1_000)).unwrap_err();
        assert_eq!(error, PrepareError::Range);
    }

    #[test]
    fn an_empty_range_is_refused_before_the_file_is_opened() {
        let mut source =
            FileSource::new(std::env::temp_dir().join("cutlass-transcription-unused.wav"));
        let error = source.read(&TimeRange::new(0, 0)).unwrap_err();
        assert_eq!(error, PrepareError::Range);
    }

    #[test]
    fn a_local_media_file_yields_pcm_for_the_opening() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        if !root.is_dir() {
            return;
        }
        let Some(path) = local_audio(&root) else {
            return;
        };
        let mut source = FileSource::new(path);
        let pcm = source.read(&TimeRange::new(0, 200_000)).unwrap();
        assert!(pcm.sample_rate > 0);
        assert!(!pcm.samples.is_empty());
        assert!(pcm.samples.iter().all(|sample| sample.is_finite()));
    }

    fn audio_rank(path: &Path) -> u8 {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "wav" | "mp3" | "m4a" | "flac" | "ogg" | "aac" | "aiff" | "caf" => 0,
            _ => 1,
        }
    }

    fn local_audio(root: &Path) -> Option<PathBuf> {
        let allowed = [
            "wav", "mp3", "m4a", "flac", "ogg", "aac", "aiff", "caf", "webm", "mp4",
        ];
        let entries = std::fs::read_dir(root).ok()?;
        let mut files = Vec::new();
        for entry in entries {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
                continue;
            };
            let extension = extension.to_ascii_lowercase();
            if allowed.contains(&extension.as_str()) {
                files.push(path);
            }
        }
        files.sort_by(|left, right| {
            audio_rank(left)
                .cmp(&audio_rank(right))
                .then(left.cmp(right))
        });
        files.into_iter().next()
    }
}
