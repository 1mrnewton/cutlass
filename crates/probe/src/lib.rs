//! What a media file is before import: a picture's upright size, a movie's
//! length, a sound's length.
//!
//! The desktop and the phones hand a path here and import what comes back,
//! so a file reads the same on every platform. A signature that names a
//! family stays with that family. A refusal is the family's error. An empty
//! container crate is named as unavailable.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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

/// What [`probe`] read. Sizes are upright: a movie's display turn and a
/// photo's EXIF orientation are applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Probed {
    /// A movie, or a still on the video lane's clock. `has_audio` says
    /// whether a movie also carries a sound track; a still has none.
    Video {
        meta: MetaType,
        width: i32,
        height: i32,
        duration: i64,
        has_audio: bool,
    },
    Audio {
        duration: i64,
    },
}

/// Read the file at `path` without copying or decoding it.
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
    if let Some(probed) = aiff(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = caf(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = flac(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = ogg(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = transport_stream(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = matroska(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = movie(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = avi(path, &header)? {
        return Ok(probed);
    }
    if let Some(probed) = mpeg_audio(path, &header)? {
        return Ok(probed);
    }
    if let Some(reason) = unavailable(&header) {
        return Err(ProbeError::new(reason));
    }
    Err(ProbeError::new(
        "the file is not a media container Cutlass can open",
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
    let (width, height) = if turned_jpeg(path, header) {
        (size.height(), size.width())
    } else {
        (size.width(), size.height())
    };
    let width = i32::try_from(width).map_err(|_| ProbeError::new("the frame is too wide"))?;
    let height = i32::try_from(height).map_err(|_| ProbeError::new("the frame is too tall"))?;
    Ok(Some(Probed::Video {
        meta: still_meta(header),
        width,
        height,
        duration: cutlass_model::MaterialVideo::MAPPING_DURATION.0,
        has_audio: false,
    }))
}

/// How much of a JPEG is read for its EXIF block and frame header. The
/// header comes before the scan, after APP segments of at most 64 KiB each.
const JPEG_HEADER_BYTES: usize = 1 << 20;

/// A JPEG whose EXIF orientation is 5 through 8 is shown a quarter turn
/// from how it is stored, so its edges swap. Anything unreadable is upright.
fn turned_jpeg(path: &Path, header: &[u8]) -> bool {
    if !header.starts_with(&[0xff, 0xd8]) {
        return false;
    }
    let Ok(bytes) = read_prefix(path, JPEG_HEADER_BYTES) else {
        return false;
    };
    cutlass_jpeg::read_jpeg_info(&bytes)
        .ok()
        .and_then(|info| info.orientation)
        .is_some_and(|orientation| (5..=8).contains(&orientation))
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

fn aiff(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_aiff::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let demux = cutlass_aiff::AiffDemux::open(file, length)
        .map_err(|error| ProbeError::new(format!("aiff refused the file: {error}")))?;
    let duration = samples_to_micros(demux.duration_samples(), demux.sample_rate())?;
    Ok(Some(Probed::Audio { duration }))
}

/// The frames of every packet, priming and remainder included, as the
/// last packet's end counts AAC priming in an MP4 and the pre-skip in Ogg.
/// A file whose data runs to its end is measured from the bytes there.
fn caf(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_caf::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let demux = cutlass_caf::CafDemux::open(file, length)
        .map_err(|error| ProbeError::new(format!("caf refused the file: {error}")))?;
    let frames = demux
        .total_frames()
        .ok_or_else(|| ProbeError::new("caf has no frame count"))?;
    let duration = samples_to_micros(frames, demux.sample_rate())?;
    Ok(Some(Probed::Audio { duration }))
}

/// The STREAMINFO sample count, or the frames counted when it is unknown.
fn flac(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_flac_container::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let mut demux = cutlass_flac_container::FlacDemux::open(std::io::BufReader::new(file), length)
        .map_err(|error| ProbeError::new(format!("flac refused the file: {error}")))?;
    let info = *demux.stream_info();
    let mut samples = info.total_samples;
    if samples == 0 {
        loop {
            match cutlass_container::Demux::read_packet(&mut demux) {
                Ok(Some(packet)) => {
                    let span = packet.timestamp().duration().unwrap_or(0);
                    samples = samples.saturating_add(u64::try_from(span).unwrap_or(0));
                }
                Ok(None) => break,
                Err(error) => {
                    return Err(ProbeError::new(format!("flac refused the file: {error}")));
                }
            }
        }
    }
    let duration = samples_to_micros(samples, info.sample_rate)?;
    Ok(Some(Probed::Audio { duration }))
}

/// Ogg Opus or Ogg Vorbis: the first edit's length, which leaves out an
/// Opus pre-skip, a Vorbis stream's late start, and the end trim. Other
/// Ogg mappings (FLAC, Theora) are refused.
fn ogg(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_ogg::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let demux = cutlass_ogg::OggDemux::open(std::io::BufReader::new(file), length)
        .map_err(|error| ProbeError::new(format!("ogg refused the file: {error}")))?;
    let streams = cutlass_container::Demux::streams(&demux);
    let Some(stream) = streams.iter().find(|stream| {
        matches!(
            stream.params().codec(),
            cutlass_codec::StreamCodec::Known(
                cutlass_codec::CodecId::Opus | cutlass_codec::CodecId::Vorbis
            )
        )
    }) else {
        return Err(ProbeError::new("only Opus and Vorbis in Ogg are available"));
    };
    let samples = stream.edits().first().map_or(0, |edit| edit.duration());
    let duration = ticks_to_micros(
        u64::try_from(samples).unwrap_or(0),
        stream.timescale().ticks_per_second(),
    )?;
    Ok(Some(Probed::Audio { duration }))
}

/// MPEG-TS (`.ts`, `.m2ts`, `.mts`): the chosen program's first video
/// size, and its length from the earliest timestamp to the end of the last
/// frame, found without reading the middle of the file.
fn transport_stream(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_mpegts::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let demux = cutlass_mpegts::TsDemux::open(std::io::BufReader::new(file), length)
        .map_err(|error| ProbeError::new(format!("mpegts refused the file: {error}")))?;
    let ticks = demux
        .end()
        .map_or(0, |end| u64::try_from(end - demux.start()).unwrap_or(0));
    let duration = ticks_to_micros(ticks, cutlass_mpegts::TIMESCALE)?;
    let streams = cutlass_container::Demux::streams(&demux);
    let sounds = streams
        .iter()
        .any(|stream| stream.params().sample_rate().is_some());
    if let Some(size) = streams.iter().find_map(|stream| stream.params().size()) {
        let width =
            i32::try_from(size.width()).map_err(|_| ProbeError::new("the frame is too wide"))?;
        let height =
            i32::try_from(size.height()).map_err(|_| ProbeError::new("the frame is too tall"))?;
        return Ok(Some(Probed::Video {
            meta: MetaType::Video,
            width,
            height,
            duration,
            has_audio: sounds,
        }));
    }
    if sounds {
        return Ok(Some(Probed::Audio { duration }));
    }
    Err(ProbeError::new(
        "mpegts opened a file with no video or audio",
    ))
}

/// A Matroska or WebM file: the first video track's size, or the first
/// audio track, and the Segment's Duration. A live recording that states
/// no Duration is read to its last packet.
fn matroska(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_matroska::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let mut demux = cutlass_matroska::MatroskaDemux::open(std::io::BufReader::new(file), length)
        .map_err(|error| ProbeError::new(format!("matroska refused the file: {error}")))?;
    let timescale = demux.clock().ticks_per_second;
    let ticks = match demux.duration_ticks().filter(|ticks| *ticks > 0) {
        Some(ticks) => ticks,
        None => {
            let mut end = 0_i64;
            while let Some(packet) = cutlass_container::Demux::read_packet(&mut demux)
                .map_err(|error| ProbeError::new(format!("matroska refused the file: {error}")))?
            {
                let time = packet.timestamp();
                if let Some(pts) = time.presentation() {
                    end = end.max(pts.saturating_add(time.duration().unwrap_or(0)));
                }
            }
            end
        }
    };
    let duration = ticks_to_micros(u64::try_from(ticks).unwrap_or(0), timescale)?;
    let streams = cutlass_container::Demux::streams(&demux);
    let sounds = streams
        .iter()
        .any(|stream| stream.params().sample_rate().is_some());
    if let Some(size) = streams.iter().find_map(|stream| stream.params().size()) {
        let width =
            i32::try_from(size.width()).map_err(|_| ProbeError::new("the frame is too wide"))?;
        let height =
            i32::try_from(size.height()).map_err(|_| ProbeError::new("the frame is too tall"))?;
        return Ok(Some(Probed::Video {
            meta: MetaType::Video,
            width,
            height,
            duration,
            has_audio: sounds,
        }));
    }
    if sounds {
        return Ok(Some(Probed::Audio { duration }));
    }
    Err(ProbeError::new(
        "matroska opened a file with no video or audio",
    ))
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
        let (width, height) = match stream.display {
            cutlass_container::DisplayQuarter::Clockwise90
            | cutlass_container::DisplayQuarter::Clockwise270 => (picture.height, picture.width),
            cutlass_container::DisplayQuarter::Zero | cutlass_container::DisplayQuarter::Half => {
                (picture.width, picture.height)
            }
        };
        let width = i32::try_from(width).map_err(|_| ProbeError::new("the frame is too wide"))?;
        let height = i32::try_from(height).map_err(|_| ProbeError::new("the frame is too tall"))?;
        return Ok(Some(Probed::Video {
            meta: MetaType::Video,
            width,
            height,
            duration,
            has_audio: audio.is_some(),
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

/// An AVI's first video stream, or its first audio stream, and how long
/// that stream's chunks last.
fn avi(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    if !matches!(
        cutlass_avi::probe(header),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return Ok(None);
    }
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let demux = cutlass_avi::AviDemux::open(file, length)
        .map_err(|error| ProbeError::new(format!("avi refused the file: {error}")))?;
    let streams = cutlass_container::Demux::streams(&demux);
    let length_of = |stream: &cutlass_container::Stream| {
        let ticks = demux.duration_ticks(stream.id().get()).unwrap_or(0);
        ticks_to_micros(
            u64::try_from(ticks).unwrap_or(0),
            stream.timescale().ticks_per_second(),
        )
    };
    let sound = streams.iter().find(|stream| {
        stream.params().sample_rate().is_some() && !stream.params().codec().is_unknown()
    });
    if let Some(stream) = streams
        .iter()
        .find(|stream| stream.params().size().is_some() && !stream.params().codec().is_unknown())
    {
        let Some(size) = stream.params().size() else {
            return Err(ProbeError::new("avi video stream has no frame size"));
        };
        let width =
            i32::try_from(size.width()).map_err(|_| ProbeError::new("the frame is too wide"))?;
        let height =
            i32::try_from(size.height()).map_err(|_| ProbeError::new("the frame is too tall"))?;
        return Ok(Some(Probed::Video {
            meta: MetaType::Video,
            width,
            height,
            duration: length_of(stream)?,
            has_audio: sound.is_some(),
        }));
    }
    if let Some(stream) = sound {
        return Ok(Some(Probed::Audio {
            duration: length_of(stream)?,
        }));
    }
    Err(ProbeError::new(
        "avi opened a file with no video or audio Cutlass reads",
    ))
}

fn mpeg_audio(path: &Path, header: &[u8]) -> Result<Option<Probed>, ProbeError> {
    // Two ADTS frames, after an ID3v2 tag or not, are raw AAC. An ID3v2 tag
    // alone also matches MP3, so ADTS is asked first. A tag can end past
    // the header window, or close enough to its end that two frames do not
    // fit: a signed download carries 17 kB of tag and cover art carries
    // more. ADTS is then asked about the bytes that follow the tag.
    let tag = match cutlass_mp3_container::id3v2_tag_bytes(header) {
        Ok(Some(tag)) => Some(tag),
        // Not a tag, or one past the MP3 reader's limit, which that reader
        // refuses below. Nothing is read for either.
        Ok(None) | Err(_) => None,
    };
    let after_tag = match tag {
        Some(tag) => Some(read_window(path, tag, header.len())?),
        None => None,
    };
    if matches!(
        cutlass_adts::probe(after_tag.as_deref().unwrap_or(header)),
        cutlass_container::ProbeResult::Accept { .. }
    ) {
        return adts(path);
    }
    match cutlass_mp3_container::probe(header) {
        cutlass_mp3_container::Probe::Matched => {}
        // The tag is longer than the window. It still names this family:
        // the reader skips it, and refuses a file whose tag runs past the
        // end or has no Layer III frame behind it.
        cutlass_mp3_container::Probe::NeedMore { .. } if tag.is_some() => {}
        cutlass_mp3_container::Probe::NeedMore { .. } | cutlass_mp3_container::Probe::Reject => {
            return Ok(None);
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

/// A raw AAC stream (ADTS, usually `.aac`). Its length is every frame's
/// 1024 samples, counted by walking the frame headers: the file stores none.
fn adts(path: &Path) -> Result<Option<Probed>, ProbeError> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let demux = cutlass_adts::AdtsSeekFile::open(file, length)
        .map_err(|error| ProbeError::new(format!("adts refused the file: {error}")))?;
    let Some(rate) = cutlass_container::Demux::streams(&demux)
        .first()
        .and_then(|stream| stream.params().sample_rate())
    else {
        return Err(ProbeError::new("adts opened a file with no sample rate"));
    };
    let duration = samples_to_micros(demux.samples(), rate)?;
    Ok(Some(Probed::Audio { duration }))
}

fn unavailable(header: &[u8]) -> Option<&'static str> {
    if header.starts_with(b"OggS") {
        // A capture pattern with a version other than 0.
        return Some("ogg is not available");
    }
    if header.starts_with(b"caff") {
        return Some("caf is not available");
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

/// Up to `limit` bytes of the file from `start`. Empty when the file ends
/// at or before `start`.
fn read_window(path: &Path, start: u64, limit: usize) -> Result<Vec<u8>, ProbeError> {
    let mut file = File::open(path).map_err(|error| io_error(path, error))?;
    file.seek(SeekFrom::Start(start))
        .map_err(|error| io_error(path, error))?;
    let mut buffer = Vec::with_capacity(limit);
    file.take(u64::try_from(limit).unwrap_or(u64::MAX))
        .read_to_end(&mut buffer)
        .map_err(|error| io_error(path, error))?;
    Ok(buffer)
}

fn io_error(path: &Path, error: std::io::Error) -> ProbeError {
    ProbeError::new(format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{Probed, probe};
    use cutlass_model::MetaType;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Temp(PathBuf);

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-probe-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&path).expect("temp");
            Self(path)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).expect("write");
            path
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../engine/fixtures")
            .join(name)
    }

    fn picture(path: &Path) -> (MetaType, i32, i32, i64) {
        match probe(path).expect("probe") {
            Probed::Video {
                meta,
                width,
                height,
                duration,
                ..
            } => (meta, width, height, duration),
            Probed::Audio { .. } => panic!("{} opened as audio", path.display()),
        }
    }

    /// Whether the picture at `path` probed with a sound track.
    fn sounds(path: &Path) -> bool {
        match probe(path).expect("probe") {
            Probed::Video { has_audio, .. } => has_audio,
            Probed::Audio { .. } => panic!("{} opened as audio", path.display()),
        }
    }

    #[test]
    fn an_avi_movie_probes_as_video_with_its_length() {
        use cutlass_codec::{
            CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
        };
        use cutlass_container::{Mux, NewStream, TrackDuration, TrackId};
        let mut mux = cutlass_avi::AviMux::new(std::io::Cursor::new(Vec::new()));
        mux.add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::Mjpeg),
                FrameSize::try_new(64, 36).expect("size"),
                Vec::new(),
            ),
            timescale: Timescale::new(25),
            language: None,
        })
        .expect("stream");
        for frame in 0..50 {
            let stamp = Timestamp::try_new(Some(frame), Some(frame), Some(1)).expect("time");
            let packet = Packet::try_new(
                0,
                vec![0xFF, 0xD8, 0xFF, 0xD9],
                stamp,
                Timescale::new(25),
                true,
            )
            .expect("packet");
            mux.write_packet(&packet).expect("write");
        }
        mux.finish(&[TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(25),
            ticks: 50,
        }])
        .expect("finish");
        let bytes = mux.into_writer().expect("writer").into_inner();
        let temp = Temp::new();
        let path = temp.write("clip.avi", &bytes);
        assert_eq!(picture(&path), (MetaType::Video, 64, 36, 2_000_000));
        assert!(!sounds(&path), "a picture-only AVI has no sound");
    }

    /// One second of 20 ms Opus packets after a 312-sample pre-skip, in Ogg.
    fn ogg_opus() -> Vec<u8> {
        use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
        use cutlass_container::{Mux, NewStream, TrackDuration};
        let mut head = b"OpusHead".to_vec();
        head.extend([1, 2, 0x38, 0x01, 0x80, 0xBB, 0, 0, 0, 0, 0]);
        let mut mux = cutlass_ogg::OggMux::new(Vec::new());
        let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Opus), 48_000, 2, head)
            .expect("params");
        let track = mux
            .add_stream(NewStream {
                params,
                timescale: Timescale::new(48_000),
                language: None,
            })
            .expect("stream");
        let mut ticks = 0;
        while ticks < 48_000 + 312 {
            let duration = 960.min(48_000 + 312 - ticks);
            let time = Timestamp::try_new(Some(ticks), Some(ticks), Some(duration)).expect("time");
            // CELT, fullband, 20 ms, one frame: a TOC and a silent frame.
            let packet = Packet::try_new(
                0,
                vec![0xFC, 0xFF, 0xFE],
                time,
                Timescale::new(48_000),
                true,
            )
            .expect("packet");
            mux.write_packet(&packet).expect("write");
            ticks += duration;
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(48_000),
            ticks,
        }])
        .expect("finish");
        mux.into_inner()
    }

    #[test]
    fn an_ogg_opus_file_is_audio_as_long_as_its_samples_after_the_pre_skip() {
        let temp = Temp::new();
        let path = temp.0.join("voice.opus");
        std::fs::write(&path, ogg_opus()).expect("write");
        assert_eq!(
            probe(&path).expect("opus"),
            Probed::Audio {
                duration: 1_000_000
            }
        );
        // A damaged head page leaves no Opus stream, and the file is
        // refused as Ogg rather than taken for another container.
        let mut damaged = ogg_opus();
        let at = damaged
            .windows(8)
            .position(|w| w == b"OpusHead")
            .expect("head");
        damaged[at..at + 8].copy_from_slice(b"Garbage!");
        std::fs::write(&path, &damaged).expect("write");
        let error = probe(&path).expect_err("damaged");
        assert!(error.to_string().contains("ogg"), "{error}");
    }

    fn ogg_fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../containers/ogg/tests/fixtures")
            .join(name)
    }

    /// libvorbis files muxed by GStreamer: as long as ffprobe reads them,
    /// 22,050 samples at 44.1 kHz and 4,000 at 8 kHz.
    #[test]
    fn an_ogg_vorbis_file_is_audio_as_long_as_its_samples() {
        assert_eq!(
            probe(&ogg_fixture("stereo_44100_q4.ogg")).expect("vorbis"),
            Probed::Audio { duration: 500_000 }
        );
        assert_eq!(
            probe(&ogg_fixture("sweep_8000_mono_q0.ogg")).expect("vorbis"),
            Probed::Audio { duration: 500_000 }
        );
        // A damaged identification header leaves no Vorbis stream.
        let temp = Temp::new();
        let mut damaged = std::fs::read(ogg_fixture("stereo_44100_q4.ogg")).expect("read");
        let at = damaged
            .windows(7)
            .position(|w| w == b"\x01vorbis")
            .expect("head");
        damaged[at + 1..at + 7].copy_from_slice(b"Garbag");
        let path = temp.write("damaged.ogg", &damaged);
        let error = probe(&path).expect_err("damaged");
        assert!(error.to_string().contains("ogg"), "{error}");
    }

    /// A WebM written by the Matroska muxer: two seconds of 320x240 VP9
    /// and a second of Opus. The Segment's Duration is the later end.
    #[test]
    fn a_webm_file_is_video_as_long_as_its_segment() {
        use cutlass_codec::{
            CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
        };
        use cutlass_container::{Mux, NewStream, TrackDuration};
        let mut mux = cutlass_matroska::MatroskaMux::webm(std::io::Cursor::new(Vec::new()));
        let size = FrameSize::try_new(320, 240).expect("size");
        let picture = mux
            .add_stream(NewStream {
                params: StreamParams::video(StreamCodec::Known(CodecId::Vp9), size, Vec::new()),
                timescale: Timescale::new(1000),
                language: None,
            })
            .expect("video");
        for index in 0..50 {
            let time = Timestamp::try_new(None, Some(index * 40), Some(40)).expect("time");
            let packet = Packet::try_new(1, vec![1; 8], time, Timescale::new(1000), index == 0)
                .expect("packet");
            mux.write_packet(&packet).expect("write");
        }
        mux.finish(&[TrackDuration {
            track: picture,
            timescale: Timescale::new(1000),
            ticks: 2000,
        }])
        .expect("finish");
        let bytes = mux.into_writer().expect("writer").into_inner();
        let temp = Temp::new();
        let path = temp.write("clip.webm", &bytes);
        assert_eq!(
            probe(&path).expect("webm"),
            Probed::Video {
                meta: MetaType::Video,
                width: 320,
                height: 240,
                duration: 2_000_000,
                has_audio: false,
            }
        );
        // A cut file still names the family in its error.
        let cut = temp.write("cut.webm", &bytes[..40]);
        let error = probe(&cut).expect_err("cut");
        assert!(error.to_string().contains("matroska"), "{error}");
    }

    #[test]
    fn a_png_fixture_is_a_still_on_the_video_lane_clock() {
        let (meta, width, height, duration) = picture(&fixture("pattern.png"));
        assert_eq!(meta, MetaType::Image);
        assert!(width > 0 && height > 0);
        assert_eq!(duration, cutlass_model::MaterialVideo::MAPPING_DURATION.0);
        assert!(!sounds(&fixture("pattern.png")), "a still has no sound");
    }

    /// `jpeg` with an EXIF block carrying `orientation` just after SOI.
    fn with_exif(jpeg: &[u8], orientation: u16) -> Vec<u8> {
        let mut tiff = vec![b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0];
        tiff.extend(0x0112u16.to_le_bytes());
        tiff.extend(3u16.to_le_bytes());
        tiff.extend(1u32.to_le_bytes());
        tiff.extend(orientation.to_le_bytes());
        tiff.extend([0, 0, 0, 0, 0, 0]);
        let mut body = b"Exif\0\0".to_vec();
        body.extend(tiff);
        let mut bytes = jpeg[..2].to_vec();
        bytes.extend([0xff, 0xe1]);
        bytes.extend(
            u16::try_from(body.len() + 2)
                .expect("segment")
                .to_be_bytes(),
        );
        bytes.extend(body);
        bytes.extend_from_slice(&jpeg[2..]);
        bytes
    }

    /// The 16x16 fixture with its frame header saying 16 wide and 8 tall.
    /// Probing reads the header and nothing past it.
    fn wide_jpeg() -> Vec<u8> {
        let mut bytes = std::fs::read(fixture("pattern.jpg")).expect("jpeg");
        let sof = bytes
            .windows(2)
            .position(|marker| marker[0] == 0xff && matches!(marker[1], 0xc0..=0xc2))
            .expect("frame header");
        bytes[sof + 5..sof + 7].copy_from_slice(&8u16.to_be_bytes());
        bytes
    }

    #[test]
    fn a_photo_turned_by_exif_reports_its_upright_size() {
        let temp = Temp::new();
        let stored = wide_jpeg();
        let (meta, width, height, _) = picture(&temp.write("wide.jpg", &stored));
        assert_eq!(meta, MetaType::Photo);
        assert_eq!((width, height), (16, 8));
        for orientation in 5..=8 {
            let turned = temp.write("turned.jpg", &with_exif(&stored, orientation));
            let (_, w, h, _) = picture(&turned);
            assert_eq!((w, h), (8, 16), "orientation {orientation}");
        }
        for orientation in [1, 2, 3, 4, 0, 9, 0xffff] {
            let kept = temp.write("kept.jpg", &with_exif(&stored, orientation));
            let (_, w, h, _) = picture(&kept);
            assert_eq!((w, h), (width, height), "orientation {orientation}");
        }
    }

    #[test]
    fn a_broken_exif_block_leaves_the_photo_as_stored() {
        let temp = Temp::new();
        let stored = wide_jpeg();
        let mut broken = stored[..2].to_vec();
        broken.extend([0xff, 0xe1, 0x00, 0x0a]);
        broken.extend(b"Exif\0\0II");
        broken.extend_from_slice(&stored[2..]);
        let path = temp.write("broken.jpg", &broken);
        let (_, w, h, _) = picture(&path);
        assert_eq!((w, h), (16, 8));
    }

    /// A one-second 16x8 movie, its track matrix set to `matrix`.
    fn movie(matrix: [i32; 9]) -> Vec<u8> {
        movie_with(matrix, false)
    }

    /// A one-second 16x8 movie, its track matrix set to `matrix`, with a
    /// second of AAC beside the picture when `sound` is set.
    fn movie_with(matrix: [i32; 9], sound: bool) -> Vec<u8> {
        use cutlass_isobmff::{CodecId, Mux, MuxOptions, Packet, StreamParams, Timescale};
        let mut params = StreamParams::video(CodecId::H264, Timescale(30), 16, 8);
        params.extradata = vec![1, 66, 0, 10];
        let mut mux = cutlass_isobmff::IsoBmffMux::new(
            std::io::Cursor::new(Vec::new()),
            MuxOptions::default(),
        );
        let track = mux.add_stream(&params).expect("stream");
        if sound {
            let mut audio = StreamParams::audio(
                CodecId::Aac,
                Timescale(48_000),
                48_000,
                2,
                16,
                Default::default(),
            );
            audio.extradata = vec![0x12, 0x10];
            let audio_track = mux.add_stream(&audio).expect("audio stream");
            for frame in 0..47 {
                mux.write_packet(&Packet {
                    track_id: audio_track,
                    payload: vec![0x21, 0x10],
                    dts: frame * 1024,
                    pts: frame * 1024,
                    duration: 1024,
                    timescale: Timescale(48_000),
                    keyframe: true,
                })
                .expect("audio packet");
            }
        }
        for frame in 0..30 {
            mux.write_packet(&Packet {
                track_id: track,
                payload: vec![1, 2, 3],
                dts: frame,
                pts: frame,
                duration: 1,
                timescale: Timescale(30),
                keyframe: frame == 0,
            })
            .expect("packet");
        }
        let mut bytes = mux.finish_into().expect("finish").into_inner();
        let at = bytes
            .windows(4)
            .position(|window| window == b"tkhd")
            .expect("tkhd")
            - 4;
        let offset = if bytes[at + 8] == 1 { 60 } else { 48 };
        for (cell, value) in matrix.iter().enumerate() {
            let start = at + offset + cell * 4;
            bytes[start..start + 4].copy_from_slice(&value.to_be_bytes());
        }
        bytes
    }

    #[test]
    fn a_movie_reports_its_length_and_its_upright_size() {
        let temp = Temp::new();
        let one = 1 << 16;
        let full = 1 << 30;
        let cases = [
            ([one, 0, 0, 0, one, 0, 0, 0, full], (16, 8)),
            ([0, one, 0, -one, 0, 0, 0, 0, full], (8, 16)),
            ([-one, 0, 0, 0, -one, 0, 0, 0, full], (16, 8)),
            ([0, -one, 0, one, 0, 0, 0, 0, full], (8, 16)),
        ];
        for (matrix, size) in cases {
            let path = temp.write("movie.mp4", &movie(matrix));
            let (meta, width, height, duration) = picture(&path);
            assert_eq!(meta, MetaType::Video);
            assert_eq!((width, height), size, "{matrix:?}");
            assert_eq!(duration, 1_000_000);
            assert!(!sounds(&path), "a picture-only movie has no sound");
        }
    }

    #[test]
    fn a_movie_with_a_sound_track_says_so() {
        let temp = Temp::new();
        let one = 1 << 16;
        let upright = [one, 0, 0, 0, one, 0, 0, 0, 1 << 30];
        let path = temp.write("talk.mp4", &movie_with(upright, true));
        assert_eq!(picture(&path), (MetaType::Video, 16, 8, 1_000_000));
        assert!(sounds(&path), "the AAC track is the movie's sound");
    }

    #[test]
    fn a_caf_file_is_audio_on_the_sample_clock() {
        use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
        use cutlass_container::{Mux, NewStream, TrackDuration, TrackId};

        let temp = Temp::new();
        let mut mux = cutlass_caf::CafMux::new(std::io::Cursor::new(Vec::new()));
        let params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 8_000, 1, vec![1, 1, 0])
                .expect("params");
        mux.add_stream(NewStream {
            params,
            timescale: Timescale::new(8_000),
            language: None,
        })
        .expect("stream");
        let packet = Packet::try_new(
            0,
            vec![0; 8_000],
            Timestamp::try_new(Some(0), Some(0), Some(4_000)).expect("time"),
            Timescale::new(8_000),
            true,
        )
        .expect("packet");
        mux.write_packet(&packet).expect("write");
        mux.finish(&[TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(8_000),
            ticks: 4_000,
        }])
        .expect("finish");
        let path = temp.write("sound.caf", &mux.into_inner().into_inner());
        assert_eq!(
            probe(&path).expect("caf"),
            Probed::Audio { duration: 500_000 }
        );
    }

    #[test]
    fn an_aiff_file_is_audio_on_the_sample_clock() {
        let temp = Temp::new();
        let path = temp.write("sound.aiff", &aiff_s16(&[0x40, 0x00]));
        assert_eq!(probe(&path).expect("aiff"), Probed::Audio { duration: 125 });
    }

    #[test]
    fn a_flac_file_is_audio_with_or_without_a_sample_count() {
        let temp = Temp::new();
        let mut bytes = flac_mono(4_000);
        let path = temp.write("sound.flac", &bytes);
        assert_eq!(
            probe(&path).expect("flac"),
            Probed::Audio { duration: 500_000 }
        );
        // Zero the STREAMINFO sample count: the frames are counted instead.
        bytes[21] &= 0xF0;
        bytes[22..26].fill(0);
        let path = temp.write("unknown.flac", &bytes);
        assert_eq!(
            probe(&path).expect("flac"),
            Probed::Audio { duration: 500_000 }
        );
    }

    fn flac_mono(samples: usize) -> Vec<u8> {
        use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
        use cutlass_container::{Mux, NewStream, TrackDuration, TrackId};
        let source: Vec<i32> = (0..samples).map(|i| (i as i32 % 200) * 50).collect();
        let settings = cutlass_flac::Settings::level(5).expect("level");
        let mut encoder = cutlass_flac::FlacEncoder::new(8_000, 1, 16, settings).expect("encoder");
        let mut frames = encoder.encode(&source).expect("encode");
        frames.extend(encoder.finish().expect("finish"));
        let mut mux = cutlass_flac_container::FlacMux::new(std::io::Cursor::new(Vec::new()));
        let params = StreamParams::try_audio(
            StreamCodec::Known(CodecId::Flac),
            8_000,
            1,
            encoder.stream_info().to_bytes().to_vec(),
        )
        .expect("params");
        mux.add_stream(NewStream {
            params,
            timescale: Timescale::new(8_000),
            language: None,
        })
        .expect("stream");
        for frame in frames {
            let pts = i64::try_from(frame.first_sample).expect("pts");
            let packet = Packet::try_new(
                0,
                frame.data,
                Timestamp::try_new(Some(pts), Some(pts), Some(i64::from(frame.samples)))
                    .expect("time"),
                Timescale::new(8_000),
                true,
            )
            .expect("packet");
            mux.write_packet(&packet).expect("write");
        }
        mux.finish(&[TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(8_000),
            ticks: i64::try_from(samples).expect("ticks"),
        }])
        .expect("finish");
        mux.into_inner().into_inner()
    }

    fn aiff_s16(frame: &[u8]) -> Vec<u8> {
        use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
        use cutlass_container::{Mux, NewStream, TrackDuration, TrackId};
        let mut mux = cutlass_aiff::AiffMux::new(std::io::Cursor::new(Vec::new()));
        let params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 8_000, 1, vec![1, 1, 0])
                .expect("params");
        mux.add_stream(NewStream {
            params,
            timescale: Timescale::new(8_000),
            language: None,
        })
        .expect("stream");
        let packet = Packet::try_new(
            0,
            frame.to_vec(),
            Timestamp::try_new(Some(0), Some(0), Some(1)).expect("time"),
            Timescale::new(8_000),
            true,
        )
        .expect("packet");
        mux.write_packet(&packet).expect("write");
        mux.finish(&[TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(8_000),
            ticks: 1,
        }])
        .expect("finish");
        mux.into_inner().into_inner()
    }

    #[test]
    fn a_text_file_an_empty_file_and_a_missing_file_are_refused() {
        let temp = Temp::new();
        let text = temp.write("notes", b"not media");
        let error = probe(&text).expect_err("text");
        assert!(error.message().contains("not a media container"), "{error}");
        let empty = temp.write("empty.mp4", b"");
        assert_eq!(
            probe(&empty).expect_err("empty").message(),
            "the file is empty"
        );
        assert!(probe(&temp.0.join("missing.mp4")).is_err());
        assert!(probe(&temp.0).is_err());
    }

    /// FFmpeg's H.264 and AAC in a transport stream: the picture size, and
    /// the time from the first AAC frame (PTS 131280) to the end of the last
    /// (PTS 240720 plus 1920), which is 16 ms past ffprobe's estimate.
    #[test]
    fn a_transport_stream_reports_its_picture_and_its_length() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../containers/mpegts/tests/fixtures/h264_aac.ts");
        assert_eq!(picture(&path), (MetaType::Video, 96, 64, 1_237_333));
        assert!(sounds(&path), "the fixture carries AAC beside its H.264");
        let temp = Temp::new();
        let bytes = std::fs::read(&path).expect("fixture");
        let cut = temp.write("cut.ts", &bytes[..1000]);
        assert!(probe(&cut).is_err());
    }

    #[test]
    fn a_cut_movie_or_photo_is_refused_without_a_panic() {
        let temp = Temp::new();
        let whole = movie([1 << 16, 0, 0, 0, 1 << 16, 0, 0, 0, 1 << 30]);
        for end in [8, 64, whole.len() / 2] {
            let cut = temp.write("cut.mp4", &whole[..end]);
            assert!(probe(&cut).is_err(), "{end} bytes opened");
        }
        let stored = std::fs::read(fixture("pattern.jpg")).expect("jpeg");
        let cut = temp.write("cut.jpg", &with_exif(&stored, 6)[..40]);
        assert!(probe(&cut).is_err());
        for name in ["truncated.jpg", "hostile.jpg", "unknown.bin"] {
            let _ = probe(&fixture(name));
        }
    }

    /// Frames of AAC-LC, 48 kHz stereo, each a 7-byte ADTS header with no
    /// CRC and one byte of payload.
    fn adts_frames(count: usize) -> Vec<u8> {
        let frame = [0xFF, 0xF1, 0x4C, 0x80, 0x01, 0x1F, 0xFC, 0x00];
        frame.repeat(count)
    }

    #[test]
    fn a_raw_aac_file_is_audio_as_long_as_its_frames() {
        let temp = Temp::new();
        let aac = temp.write("sound.aac", &adts_frames(48));
        // 48 frames of 1024 samples at 48 kHz.
        assert_eq!(
            probe(&aac).expect("raw aac"),
            Probed::Audio {
                duration: 1_024_000
            }
        );
    }

    #[test]
    fn a_raw_aac_file_behind_an_id3_tag_is_still_aac() {
        let temp = Temp::new();
        let mut bytes = b"ID3\x04\x00\x00\x00\x00\x00\x10".to_vec();
        bytes.extend([0u8; 16]);
        bytes.extend(adts_frames(3));
        let aac = temp.write("tagged.aac", &bytes);
        assert_eq!(
            probe(&aac).expect("tagged aac"),
            Probed::Audio { duration: 64_000 }
        );
    }

    /// An ID3v2.4 tag of `total` bytes, header included, laid out like the
    /// one FFmpeg's writer puts on a signed download: a short `TSSE` text
    /// frame, then one `GEOB` object, the C2PA manifest there, that fills
    /// the rest.
    fn id3_tag(total: usize) -> Vec<u8> {
        let synchsafe = |value: usize| {
            [
                u8::try_from((value >> 21) & 0x7F).expect("seven bits"),
                u8::try_from((value >> 14) & 0x7F).expect("seven bits"),
                u8::try_from((value >> 7) & 0x7F).expect("seven bits"),
                u8::try_from(value & 0x7F).expect("seven bits"),
            ]
        };
        let encoder = b"\x03Lavf63.1.102";
        let object = total - 10 - 10 - encoder.len() - 10;
        let mut tag = b"ID3\x04\x00\x00".to_vec();
        tag.extend(synchsafe(total - 10));
        tag.extend(b"TSSE");
        tag.extend(synchsafe(encoder.len()));
        tag.extend([0, 0]);
        tag.extend(encoder);
        tag.extend(b"GEOB");
        tag.extend(synchsafe(object));
        tag.extend([0, 0]);
        // Bytes that are neither an MPEG sync nor an ADTS one.
        tag.extend((0..object).map(|index| b'a' + u8::try_from(index % 26).expect("a letter")));
        assert_eq!(tag.len(), total);
        tag
    }

    /// Layer III at 48 kHz stereo with no tag, from the MP3 decoder's tests.
    fn layer3() -> Vec<u8> {
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../codecs/mp3/tests/fixtures/clicks_48000_v2.mp3"),
        )
        .expect("mp3 fixture")
    }

    /// The tag ElevenLabs puts on a download is 17,160 bytes, four times the
    /// header window. The sound behind it is as long as it is with no tag.
    #[test]
    fn an_mp3_behind_a_tag_longer_than_the_header_window_is_audio() {
        let temp = Temp::new();
        let bare = probe(&temp.write("bare.mp3", &layer3())).expect("bare mp3");
        // 22 frames of 1152 samples at 48 kHz.
        assert_eq!(bare, Probed::Audio { duration: 528_000 });
        // The window's own length, one byte past it, the real tag, and a
        // tag the size of cover art.
        for total in [4_096, 4_097, 17_160, 600_000] {
            let mut bytes = id3_tag(total);
            bytes.extend(layer3());
            let tagged = temp.write("tagged.mp3", &bytes);
            assert_eq!(
                probe(&tagged).expect("tagged mp3"),
                bare,
                "a {total} byte tag"
            );
        }
    }

    #[test]
    fn a_raw_aac_file_behind_a_tag_longer_than_the_header_window_is_still_aac() {
        let temp = Temp::new();
        // 4,090 leaves the window six bytes of the first frame.
        for total in [4_090, 4_096, 17_160] {
            let mut bytes = id3_tag(total);
            bytes.extend(adts_frames(3));
            let aac = temp.write("tagged.aac", &bytes);
            assert_eq!(
                probe(&aac).expect("tagged aac"),
                Probed::Audio { duration: 64_000 },
                "a {total} byte tag"
            );
        }
    }

    /// A tag that says it is longer than the file, a file cut inside its
    /// tag, a tag with nothing or one frame header after it, a tag over
    /// text, and a tag past the reader's limit are each the MP3 reader's
    /// refusal. None of them opens.
    #[test]
    fn a_long_tag_that_lies_or_is_cut_is_refused_by_the_mp3_reader() {
        let temp = Temp::new();
        let mut whole = id3_tag(17_160);
        whole.extend(layer3());

        let mut lying = whole.clone();
        // 0x7F 0x7F 0x7F is 2,097,151 bytes, under the reader's 16 MiB limit.
        lying[6..10].copy_from_slice(&[0, 0x7F, 0x7F, 0x7F]);
        let mut huge = whole.clone();
        // 0x7F four times is 256 MiB, past the limit: nothing is read for it.
        huge[6..10].copy_from_slice(&[0x7F; 4]);
        let mut text = id3_tag(17_160);
        text.extend(b"no sound follows this tag ".repeat(400));
        let cases: [(&[u8], &str); 6] = [
            (&lying, "mp3 refused the file: truncated id3"),
            (&whole[..9_000], "mp3 refused the file: truncated id3"),
            (&whole[..17_160], "mp3 refused the file: no frames"),
            (&whole[..17_164], "mp3 refused the file: truncated frame"),
            (&text, "mp3 refused the file: no frames"),
            (&huge, "mp3 refused the file: limit exceeded: id3"),
        ];
        for (bytes, refusal) in cases {
            let error = probe(&temp.write("bad.mp3", bytes)).expect_err(refusal);
            assert_eq!(error.message(), refusal);
        }
    }
}
