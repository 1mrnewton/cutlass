//! A clip's sound for the jobs that listen to it: captions, hearing a
//! stretch, and Sync.
//!
//! The file is read as the player reads it, through the engine. A sample
//! asked for at a time is the sample heard at that time: a decoder's
//! start-up samples are gone and a file's edit list is followed, so a word
//! or a match is found where it plays.

use std::path::{Path, PathBuf};

use cutlass_engine::{EngineError, Media, ReaderFactory};
use cutlass_model::{Micros, TimeRange};
use cutlass_transcription::{Pcm, PcmSource, PrepareError};

/// The most sample frames asked of the file in one read: the engine's own
/// limit is 480,000, ten seconds at 48 kHz.
const WINDOW: u32 = 240_000;

/// A [`PcmSource`] that reads one media file through the engine.
pub(super) struct MediaSource {
    readers: ReaderFactory,
    path: PathBuf,
    /// The open file, kept between reads so a read that starts where the
    /// last one ended goes on without a seek.
    media: Option<Media>,
}

impl MediaSource {
    pub(super) fn new(readers: &ReaderFactory, path: impl Into<PathBuf>) -> Self {
        Self {
            readers: readers.clone(),
            path: path.into(),
            media: None,
        }
    }
}

impl PcmSource for MediaSource {
    fn read(&mut self, range: &TimeRange) -> Result<Pcm, PrepareError> {
        if range.duration.0 <= 0 || range.start.0 < 0 {
            return Err(PrepareError::Range);
        }
        let end = range
            .start
            .0
            .checked_add(range.duration.0)
            .ok_or(PrepareError::Range)?;
        let mut media = match self.media.take() {
            Some(media) => media,
            None => open(&self.readers, &self.path)?,
        };
        let pcm = read_range(&mut media, range.start.0, end)?;
        // A file that failed is opened again by the next read.
        self.media = Some(media);
        Ok(pcm)
    }
}

fn open(readers: &ReaderFactory, path: &Path) -> Result<Media, PrepareError> {
    readers.open(path).map_err(|error| match error {
        EngineError::MissingFile { .. } | EngineError::Io { .. } => PrepareError::Open,
        // A file with nothing in it to play, such as a text file.
        EngineError::NotMedia => PrepareError::NoAudio,
        _ => PrepareError::Decode,
    })
}

/// The sample nearest `micros` at `rate`, which is the one the engine
/// starts a read on.
fn sample_in(micros: i64, rate: u32) -> u64 {
    let micros = u64::try_from(micros).unwrap_or(0);
    let sample = (u128::from(micros) * u128::from(rate) + 500_000) / 1_000_000;
    u64::try_from(sample).unwrap_or(u64::MAX)
}

/// The time sample `sample` starts at `rate`, rounded up so it names that
/// sample and not the one before it.
fn micros_of(sample: u64, rate: u32) -> Result<i64, PrepareError> {
    let micros = (u128::from(sample) * 1_000_000).div_ceil(u128::from(rate.max(1)));
    i64::try_from(micros).map_err(|_| PrepareError::TooLong)
}

/// The sound of `media` from `start` to `end`, in microseconds, as one
/// channel at the rate the file decodes to. Sound that ends before `end`
/// comes back short.
fn read_range(media: &mut Media, start: i64, end: i64) -> Result<Pcm, PrepareError> {
    let has_sound = media
        .info()
        .streams
        .iter()
        .any(|stream| stream.sample_rate.is_some());
    // The rate the decoder runs at, known once it has answered: it can
    // differ from the rate the file states, as HE-AAC's does.
    let mut rate = 0_u32;
    let mut first = 0_u64;
    let mut total = u64::from(WINDOW);
    let mut read = 0_u64;
    let mut samples: Vec<f32> = Vec::new();
    while read < total {
        let frames = u32::try_from(total - read).unwrap_or(WINDOW).min(WINDOW);
        // Each read starts on the sample the last one ended on, which the
        // engine continues without a seek.
        let at = if rate == 0 {
            start
        } else {
            micros_of(first + read, rate)?
        };
        let window = match media.audio_from(Micros(at), frames) {
            Ok(window) => window,
            // No sound stream, or one no decoder here opens.
            Err(EngineError::NotMedia) if has_sound => return Err(PrepareError::Decode),
            Err(EngineError::NotMedia) => return Err(PrepareError::NoAudio),
            // Nothing decodes from here on: the sound has ended.
            Err(EngineError::NotReady) => break,
            Err(_) => return Err(PrepareError::Decode),
        };
        if window.rate == 0 {
            return Err(PrepareError::SampleRate);
        }
        let channels = window.layout.len();
        let Some(heard) = window.as_f32().filter(|_| channels > 0) else {
            return Err(PrepareError::Decode);
        };
        if rate == 0 {
            rate = window.rate;
            first = sample_in(start, rate);
            total = sample_in(end, rate).saturating_sub(first);
            samples.reserve(usize::try_from(total).unwrap_or(0));
        } else if rate != window.rate {
            return Err(PrepareError::Decode);
        }
        let got = heard.len() / channels;
        samples.extend(
            heard
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
        // The first read is asked for before the rate, and so the length,
        // is known.
        samples.truncate(usize::try_from(total).unwrap_or(usize::MAX));
        read = read.saturating_add(got as u64);
        if got < frames as usize {
            break;
        }
    }
    if samples.is_empty() {
        // The range starts past the end of the sound.
        return Err(PrepareError::Range);
    }
    Ok(Pcm {
        sample_rate: rate,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cutlass_codec::{CodecId, ContainerId};
    use cutlass_encode::EncodeFactory;
    use cutlass_engine::ReaderFactory;
    use cutlass_eval::NoFonts;
    use cutlass_export::{AudioExport, ExportRequest, MovieSources, write_sound};
    use cutlass_model::{
        Draft, FrameRate, MaterialAudio, MaterialId, Micros, Segment, TimeRange, Track, TrackType,
    };
    use cutlass_transcription::{PcmSource, PrepareError};

    use super::MediaSource;
    use crate::shell::tests::Temp;

    const RATE: u32 = 48_000;

    /// Where the clicks are, in samples: half a second, one and a half,
    /// and twelve and a half, which is past one read of the file.
    const CLICKS: [usize; 3] = [24_000, 72_000, 600_000];

    /// Fourteen seconds of silence with a click at each of [`CLICKS`]: a
    /// short burst that starts at full height, so its first sample is the
    /// loudest and names its time to the sample.
    fn clicks() -> Vec<f32> {
        let mut sound = vec![0.0_f32; 14 * RATE as usize];
        for at in CLICKS {
            for (index, sample) in sound.iter_mut().skip(at).take(96).enumerate() {
                let time = index as f32 / RATE as f32;
                let tone = (std::f32::consts::TAU * 3_000.0 * time).cos();
                *sample = 0.9 * tone * (-time * 2_000.0).exp();
            }
        }
        sound
    }

    fn write_wav(path: &Path, channels: u16, samples: &[f32]) {
        let data = (samples.len() * 2) as u32;
        let block = channels * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&RATE.to_le_bytes());
        bytes.extend_from_slice(&(RATE * u32::from(block)).to_le_bytes());
        bytes.extend_from_slice(&block.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&((sample.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// Readers with the Rust sound decoders only, so the test hears the
    /// same samples on every platform.
    fn readers() -> ReaderFactory {
        let mut decoders = cutlass_decode::DecodeFactory::new();
        cutlass_pcm::register_decoders(&mut decoders);
        cutlass_aac::register_decoders(&mut decoders);
        cutlass_mp3::register_decoders(&mut decoders);
        cutlass_opus::register_decoders(&mut decoders);
        crate::readers::readers_with(decoders)
    }

    /// `clicks.wav` in `dir`, written again as `codec` in `name` by the
    /// Rust encoder, as an export writes it.
    fn encoded(dir: &Path, codec: CodecId, container: ContainerId, name: &str) -> PathBuf {
        let wav = dir.join("clicks.wav");
        if !wav.is_file() {
            write_wav(&wav, 1, &clicks());
        }
        let mut draft = Draft::empty("clicks");
        draft.duration = Micros(14_000_000);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("clicks"),
            path: "clicks.wav".to_owned(),
            duration: Micros(14_000_000),
            ..MaterialAudio::default()
        });
        let mut track = Track::main("clicks");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("clicks", "clicks", TimeRange::new(0, 14_000_000));
        segment.source_range = Some(TimeRange::new(0, 14_000_000));
        track.segments.push(segment);
        draft.tracks.push(track);

        let mut encoders = EncodeFactory::new();
        cutlass_aac::register_encoders(&mut encoders);
        cutlass_mp3::register_encoders(&mut encoders);
        cutlass_opus::register_encoders(&mut encoders);
        let readers = readers();
        let sources = MovieSources {
            draft: &draft,
            root: Some(dir),
            readers: &readers,
            fonts: &NoFonts,
            renderer: None,
            encoders: &encoders,
        };
        let request = ExportRequest {
            range: TimeRange::new(0, 14_000_000),
            short_side: 0,
            codec,
            bitrate: 0,
            frame_rate: FrameRate::FPS_30,
            container,
            audio: Some(AudioExport {
                codec,
                bitrate: 192_000,
                sample_rate: RATE,
                channels: 2,
                limiter: 0,
            }),
            sidecar: None,
            late_by: 0,
            constant_rate: false,
            prores: cutlass_export::ProresProfile::Standard,
        };
        let path = dir.join(name);
        write_sound(&sources, &request, &path, &mut |_, _| true).expect("export");
        path
    }

    /// The loudest sample of `samples`.
    fn loudest(samples: &[f32]) -> usize {
        let mut best = 0;
        for (index, sample) in samples.iter().enumerate() {
            if sample.abs() > samples[best].abs() {
                best = index;
            }
        }
        best
    }

    /// How many samples late each click is heard through `source`: the
    /// first from the start of the file, the second after a seek to one
    /// second, the third in one read of the whole file.
    fn late(source: &mut dyn PcmSource) -> [i64; 3] {
        let opening = source.read(&TimeRange::new(0, 1_000_000)).expect("opening");
        assert_eq!(opening.sample_rate, RATE);
        assert_eq!(opening.samples.len(), RATE as usize);
        let seeked = source
            .read(&TimeRange::new(1_000_000, 1_000_000))
            .expect("seek");
        assert_eq!(seeked.samples.len(), RATE as usize);
        let whole = source.read(&TimeRange::new(0, 13_500_000)).expect("whole");
        assert_eq!(whole.samples.len(), 13 * RATE as usize + RATE as usize / 2);
        [
            loudest(&opening.samples) as i64 - CLICKS[0] as i64,
            loudest(&seeked.samples) as i64 + i64::from(RATE) - CLICKS[1] as i64,
            loudest(&whole.samples[12 * RATE as usize..]) as i64 + 12 * i64::from(RATE)
                - CLICKS[2] as i64,
        ]
    }

    /// A click is heard at its own time in each kind of file an export
    /// writes: past AAC's 1,024 start-up samples, which its MP4 edit
    /// skips, past MP3's delay and past Opus's pre-skip. A reader that
    /// plays those samples hears every click that much late.
    #[test]
    fn a_click_is_heard_at_its_own_time_in_aac_mp3_and_opus() {
        let temp = Temp::new();
        let readers = readers();
        for (codec, container, name, slack) in [
            (CodecId::Aac, ContainerId::Isobmff, "cut.m4a", 0),
            (CodecId::Mp3, ContainerId::Mp3, "cut.mp3", 2),
            (CodecId::Opus, ContainerId::Ogg, "cut.opus", 2),
        ] {
            let path = encoded(&temp.path, codec, container, name);
            for late in late(&mut MediaSource::new(&readers, &path)) {
                assert!(late.abs() <= slack, "{codec:?} is {late} samples late");
            }
        }
        // The file the clicks were encoded from, to the sample.
        let wav = temp.path.join("clicks.wav");
        assert_eq!(late(&mut MediaSource::new(&readers, &wav)), [0, 0, 0]);
    }

    /// The sound a job is given is the sound the player is given: the same
    /// samples, from the start and after a seek.
    #[test]
    fn a_range_holds_the_samples_the_engine_plays() {
        let temp = Temp::new();
        let readers = readers();
        let path = encoded(&temp.path, CodecId::Aac, ContainerId::Isobmff, "cut.m4a");
        let mut source = MediaSource::new(&readers, &path);
        for start in [0_i64, 1_490_000] {
            let heard = source.read(&TimeRange::new(start, 100_000)).unwrap();
            let played = readers
                .open(&path)
                .unwrap()
                .audio_from(Micros(start), 4_800)
                .unwrap();
            let played: Vec<f32> = played
                .as_f32()
                .unwrap()
                .chunks_exact(played.layout.len())
                .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                .collect();
            assert_eq!(heard.samples, played, "from {start}");
            assert!(heard.samples.iter().any(|sample| sample.abs() > 0.1) == (start > 0));
        }
    }

    /// Reads that follow one another join to the sample, and two channels
    /// come back as their mean.
    #[test]
    fn pieces_join_and_two_channels_are_one() {
        let temp = Temp::new();
        let path = temp.path.join("ramp.wav");
        // Left climbs, right is silent: the mean is half the left.
        let ramp: Vec<f32> = (0..RATE as usize * 12)
            .flat_map(|index| [(index % 30_000) as f32 / 32_767.0, 0.0])
            .collect();
        write_wav(&path, 2, &ramp);
        let mut source = MediaSource::new(&readers(), &path);
        let mut joined = Vec::new();
        for piece in 0..4_i64 {
            let pcm = source
                .read(&TimeRange::new(piece * 3_000_000, 3_000_000))
                .unwrap();
            assert_eq!(pcm.sample_rate, RATE);
            joined.extend(pcm.samples);
        }
        assert_eq!(joined.len(), RATE as usize * 12);
        for (index, sample) in joined.iter().enumerate() {
            let expected = (index % 30_000) as f32 / 32_767.0 / 2.0;
            assert!((sample - expected).abs() < 1e-4, "sample {index}");
        }
    }

    #[test]
    fn sound_that_ends_early_comes_back_short_and_a_range_past_it_is_refused() {
        let temp = Temp::new();
        let path = temp.path.join("short.wav");
        write_wav(&path, 1, &vec![0.25; RATE as usize]);
        let mut source = MediaSource::new(&readers(), &path);
        let tail = source.read(&TimeRange::new(500_000, 2_000_000)).unwrap();
        assert_eq!(tail.samples.len(), RATE as usize / 2);
        let past = source.read(&TimeRange::new(5_000_000, 1_000_000));
        assert_eq!(past.unwrap_err(), PrepareError::Range);
        // The reader still reads after a refusal.
        let again = source.read(&TimeRange::new(0, 250_000)).unwrap();
        assert_eq!(again.samples.len(), RATE as usize / 4);
    }

    #[test]
    fn a_file_that_cannot_be_heard_names_why() {
        let temp = Temp::new();
        let readers = readers();
        let whole = TimeRange::new(0, 1_000_000);
        let read = |path: &Path, range: &TimeRange| {
            MediaSource::new(&readers, path).read(range).unwrap_err()
        };
        // An empty range is refused before the file is opened.
        let missing = temp.path.join("missing.wav");
        assert_eq!(read(&missing, &TimeRange::new(0, 0)), PrepareError::Range);
        assert_eq!(read(&missing, &TimeRange::new(-1, 10)), PrepareError::Range);
        assert_eq!(read(&missing, &whole), PrepareError::Open);
        let noise = temp.path.join("noise.wav");
        std::fs::write(&noise, b"not sound").unwrap();
        assert_eq!(read(&noise, &whole), PrepareError::Decode);
        let cut = temp.path.join("cut.wav");
        std::fs::write(&cut, b"RIFF").unwrap();
        assert_eq!(read(&cut, &whole), PrepareError::Decode);
        // A picture has no sound in it.
        let picture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/engine/fixtures/pattern.png");
        let mut decoders = cutlass_decode::DecodeFactory::new();
        cutlass_png::register_decoders(&mut decoders);
        let stills = crate::readers::readers_with(decoders);
        let error = MediaSource::new(&stills, &picture)
            .read(&whole)
            .unwrap_err();
        assert_eq!(error, PrepareError::NoAudio);
    }
}
