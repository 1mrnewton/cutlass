//! Open a media file with the container crates this desktop already links.
//!
//! A signature that names a family stays with that family. A refusal is the
//! family's error. An empty container crate is named as unavailable.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use cutlass_model::MetaType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeError {
    message: String,
}

impl ProbeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProbeError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Probed {
    Video {
        meta: MetaType,
        width: i32,
        height: i32,
        duration: i64,
    },
    Audio {
        duration: i64,
    },
}

pub fn probe(path: &Path) -> Result<Probed, ProbeError> {
    let header = read_prefix(path, 4_096)?;
    if header.is_empty() {
        return Err(ProbeError::new("the file is empty"));
    }
    if let Some(probed) = still(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = wave(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = movie(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = mpeg_audio(path, &header)? {
        return Ok(probed);
    }
    if let Some(reason) = unavailable(&header) {
        return Err(ProbeError::new(reason));
    }
    Err(ProbeError::new(
        "the file is not a media container this desktop can open",
    ))
}

fn still(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_still::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let input =
        cutlass_container::Input::random(file, length, cutlass_container::Limits::standard());
    let demux = cutlass_still::StillDemux::open(input)
        .map_err(|error| ProbeError::new(format!("still refused the file: {error}")))?;
    let streams = cutlass_container::Demux::streams(&demux);
    let Some(stream) = streams.first() else {
        return Err(ProbeError::new("still opened a file with no picture"));
    };
    let Some(size) = stream.params().size() else {
        return Err(ProbeError::new("still opened a file with no frame size"));
    };
    let width =
        i32::try_from(size.width()).map_err(|_| ProbeError::new("the frame is too wide"))?;
    let height =
        i32::try_from(size.height()).map_err(|_| ProbeError::new("the frame is too tall"))?;
    Ok(Some(Probed::Video {
        meta: still_meta(header),
        width,
        height,
        duration: cutlass_model::MaterialVideo::MAPPING_DURATION.0,
    }))
}

fn still_meta(header: &[u8]) -> MetaType {
    if header.starts_with(b"GIF") {
        MetaType::Gif
    } else if header.starts_with(&[0x89, b'P', b'N', b'G'])
        || (header.len() >= 12 && &header[8..12] == b"WEBP")
    {
        MetaType::Image
    } else {
        MetaType::Photo
    }
}

fn wave(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_riff::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let demux = cutlass_riff::WavDemux::open(file, length)
        .map_err(|error| ProbeError::new(format!("wave refused the file: {error}")))?;
    let Some(samples) = demux.duration_samples() else {
        return Err(ProbeError::new("wave opened a file with no duration"));
    };
    let rate = demux.format().sample_rate();
    let duration = samples_to_micros(samples, rate)?;
    Ok(Some(Probed::Audio { duration }))
}

fn movie(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if header.get(4..8) != Some(b"ftyp") {
        return Ok(None);
    }
    let mut source = cutlass_isobmff::FileSource::open(path)
        .map_err(|error| ProbeError::new(format!("isobmff refused the file: {error}")))?;
    let movie = cutlass_isobmff::IsoBmff::open(&mut source)
        .map_err(|error| ProbeError::new(format!("isobmff refused the file: {error}")))?;
    let streams = cutlass_isobmff::Demux::streams(&movie);
    let video = streams
        .iter()
        .find(|stream| matches!(stream.params.kind, cutlass_isobmff::MediaKind::Video(_)));
    let audio = streams
        .iter()
        .find(|stream| matches!(stream.params.kind, cutlass_isobmff::MediaKind::Audio(_)));
    if let Some(stream) = video {
        let cutlass_isobmff::MediaKind::Video(picture) = stream.params.kind else {
            return Err(ProbeError::new("isobmff video track has no picture"));
        };
        let duration =
            ticks_to_micros(stream.params.duration_ticks, stream.params.timescale.get())?;
        let width =
            i32::try_from(picture.width).map_err(|_| ProbeError::new("the frame is too wide"))?;
        let height =
            i32::try_from(picture.height).map_err(|_| ProbeError::new("the frame is too tall"))?;
        return Ok(Some(Probed::Video {
            meta: MetaType::Video,
            width,
            height,
            duration,
        }));
    }
    if let Some(stream) = audio {
        let duration =
            ticks_to_micros(stream.params.duration_ticks, stream.params.timescale.get())?;
        return Ok(Some(Probed::Audio { duration }));
    }
    Err(ProbeError::new(
        "isobmff opened a file with no video or audio",
    ))
}

fn mpeg_audio(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    match cutlass_mp3_container::probe(header) {
        cutlass_mp3_container::Probe::Matched => {}
        cutlass_mp3_container::Probe::NeedMore { .. } | cutlass_mp3_container::Probe::Reject => {
            return adts(path, header);
        }
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let demux = cutlass_mp3_container::Mp3Demux::open(file)
        .map_err(|error| ProbeError::new(format!("mp3 refused the file: {error}")))?;
    let Some(samples) = cutlass_mp3_container::Demux::duration(&demux) else {
        return Err(ProbeError::new("mp3 opened a file with no duration"));
    };
    if samples <= 0 {
        return Err(ProbeError::new("mp3 opened a file with no duration"));
    }
    let streams = cutlass_container::Demux::streams(&demux);
    let Some(rate) = streams
        .first()
        .and_then(|stream| stream.params().sample_rate())
    else {
        return Err(ProbeError::new("mp3 opened a file with no sample rate"));
    };
    let duration = samples_to_micros(u64::try_from(samples).unwrap_or(0), rate)?;
    Ok(Some(Probed::Audio { duration }))
}

fn adts(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    let Some((first, second)) = header.first().zip(header.get(1)) else {
        return Ok(None);
    };
    if *first != 0xFF || second & 0xF0 != 0xF0 {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut demux = cutlass_adts::AdtsFile::open(file)
        .map_err(|error| ProbeError::new(format!("aac refused the file: {error}")))?;
    let streams = cutlass_container::Demux::streams(&demux);
    let Some(rate) = streams
        .first()
        .and_then(|stream| stream.params().sample_rate())
    else {
        return Err(ProbeError::new("aac opened a file with no sample rate"));
    };
    let mut samples = 0i64;
    loop {
        match cutlass_container::Demux::read_packet(&mut demux) {
            Ok(Some(packet)) => {
                samples = samples.saturating_add(packet.timestamp().duration().unwrap_or(0));
            }
            Ok(None) => break,
            Err(error) => return Err(ProbeError::new(format!("aac refused the file: {error}"))),
        }
    }
    if samples <= 0 {
        return Err(ProbeError::new("aac opened a file with no duration"));
    }
    let duration = samples_to_micros(u64::try_from(samples).unwrap_or(0), rate)?;
    Ok(Some(Probed::Audio { duration }))
}

fn unavailable(header: &[u8]) -> Option<&'static str> {
    if header.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Some("matroska is not available");
    }
    if header.starts_with(b"OggS") {
        return Some("ogg is not available");
    }
    if header.starts_with(b"fLaC") {
        return Some("flac is not available");
    }
    if header.starts_with(b"caff") {
        return Some("caf is not available");
    }
    if header.starts_with(b"FORM") && header.get(8..12) == Some(b"AIFF") {
        return Some("aiff is not available");
    }
    if header.starts_with(b"RIFF") && header.get(8..12) == Some(b"AVI ") {
        return Some("avi is not available");
    }
    if header.first() == Some(&0x47) && header.get(188) == Some(&0x47) {
        return Some("mpegts is not available");
    }
    None
}

fn ticks_to_micros(ticks: u64, timescale: u32) -> Result<i64, ProbeError> {
    if timescale == 0 {
        return Err(ProbeError::new("the track timescale is zero"));
    }
    if ticks == 0 {
        return Err(ProbeError::new("the file has no duration"));
    }
    let micros = i128::from(ticks) * 1_000_000 / i128::from(timescale);
    let micros = if micros == 0 { 1 } else { micros };
    i64::try_from(micros).map_err(|_| ProbeError::new("the duration does not fit"))
}

fn samples_to_micros(samples: u64, rate: u32) -> Result<i64, ProbeError> {
    ticks_to_micros(samples, rate)
}

fn read_prefix(path: &Path, limit: usize) -> Result<Vec<u8>, ProbeError> {
    let mut file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut buffer = vec![0u8; limit];
    let read = file
        .read(&mut buffer)
        .map_err(|error| io_error(path, error))?;
    buffer.truncate(read);
    Ok(buffer)
}

fn io_error(path: &Path, error: std::io::Error) -> ProbeError {
    ProbeError::new(format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{Probed, probe};
    use cutlass_model::MetaType;
    use std::path::PathBuf;

    #[test]
    fn a_png_fixture_is_a_still_on_the_video_lane_clock() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        let probed = probe(&path).expect("png");
        match probed {
            Probed::Video {
                meta,
                width,
                height,
                duration,
            } => {
                assert_eq!(meta, MetaType::Image);
                assert!(width > 0);
                assert!(height > 0);
                assert_eq!(duration, cutlass_model::MaterialVideo::MAPPING_DURATION.0);
            }
            Probed::Audio { .. } => panic!("png opened as audio"),
        }
    }

    #[test]
    fn a_text_file_is_refused_by_name() {
        let path = std::env::temp_dir().join(format!("cutlass-probe-{}", std::process::id()));
        std::fs::write(&path, b"not media").unwrap();
        let error = probe(&path).expect_err("text");
        assert!(error.message().contains("not a media container"));
        std::fs::remove_file(&path).unwrap();
    }
}
