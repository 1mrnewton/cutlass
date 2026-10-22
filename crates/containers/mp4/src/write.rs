//! An MP4 file. ISO/IEC 14496-12. The index follows the samples.

use std::num::NonZeroU32;

use cutlass_core::{Packet, VideoCodec, VideoStreamConfig};

use crate::Error;

const DPI_72: u32 = 0x0048_0000;
const IDENTITY: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

/// An MP4 file to write. The movie box follows the media data.
#[derive(Debug)]
pub struct MovieFile {
    timescale: NonZeroU32,
    tracks: Vec<Track>,
}

impl MovieFile {
    /// `timescale` is the movie header clock. Track durations must be a whole
    /// number of these ticks.
    pub fn new(timescale: NonZeroU32) -> Self {
        Self {
            timescale,
            tracks: Vec::new(),
        }
    }

    /// Add an H.264 or HEVC track. Samples are in decode order.
    pub fn add_video(
        &mut self,
        timescale: NonZeroU32,
        config: VideoStreamConfig,
        samples: Vec<Packet>,
    ) -> Result<(), Error> {
        let (entry, config_box) = visual_boxes(config.codec)?;
        if config.width == 0 || config.height == 0 {
            return Err(Error::ZeroDimension);
        }
        let width = u16::try_from(config.width)
            .map_err(|_| Error::Inconsistent("the video size does not fit"))?;
        let height = u16::try_from(config.height)
            .map_err(|_| Error::Inconsistent("the video size does not fit"))?;
        let (samples, media_duration) = timed(timescale, samples)?;
        let movie_duration = movie_ticks(media_duration, timescale, self.timescale)?;
        self.tracks.push(Track {
            timescale,
            media_duration,
            movie_duration,
            kind: TrackKind::Video {
                entry,
                config_box,
                width,
                height,
                config: config.config_record,
            },
            samples,
        });
        Ok(())
    }

    /// The file bytes. `ftyp`, then `mdat`, then `moov`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        if self.tracks.is_empty() {
            return Err(Error::Inconsistent("the movie has no tracks"));
        }
        let mut payload = Vec::new();
        let mut placed = Vec::with_capacity(self.tracks.len());
        for track in &self.tracks {
            if track.samples.is_empty() {
                placed.push(None);
                continue;
            }
            let at = u64::try_from(payload.len()).map_err(|_| Error::TooLarge)?;
            placed.push(Some(at));
            for sample in &track.samples {
                payload.extend_from_slice(&sample.data);
            }
        }
        let mut buf = Buf::default();
        write_ftyp(&mut buf)?;
        let data_at = write_mdat(&mut buf, &payload)?;
        write_moov(&mut buf, self, &placed, data_at)?;
        Ok(buf.bytes)
    }
}

#[derive(Debug)]
struct Track {
    timescale: NonZeroU32,
    media_duration: u64,
    movie_duration: u64,
    kind: TrackKind,
    samples: Vec<Sample>,
}

#[derive(Debug)]
enum TrackKind {
    Video {
        entry: [u8; 4],
        config_box: [u8; 4],
        width: u16,
        height: u16,
        config: Vec<u8>,
    },
}

#[derive(Debug)]
struct Sample {
    data: Vec<u8>,
    duration: u32,
    offset: i32,
    keyframe: bool,
}

fn visual_boxes(codec: VideoCodec) -> Result<([u8; 4], [u8; 4]), Error> {
    match codec {
        VideoCodec::H264 => Ok((*b"avc1", *b"avcC")),
        VideoCodec::Hevc => Ok((*b"hvc1", *b"hvcC")),
        VideoCodec::Vp8
        | VideoCodec::Vp9
        | VideoCodec::Av1
        | VideoCodec::Mpeg2
        | VideoCodec::Mpeg4Part2
        | VideoCodec::Mjpeg
        | VideoCodec::ProRes => Err(Error::Inconsistent("the video codec is not written")),
    }
}

fn timed(timescale: NonZeroU32, samples: Vec<Packet>) -> Result<(Vec<Sample>, u64), Error> {
    let _count = u32::try_from(samples.len()).map_err(|_| Error::TooLarge)?;
    let mut stored = Vec::with_capacity(samples.len());
    let mut next = 0i64;
    for sample in samples {
        let dts = sample
            .dts
            .ok_or(Error::Inconsistent("a sample has no decode time"))?;
        let duration = sample
            .duration
            .ok_or(Error::Inconsistent("a sample has no duration"))?;
        if dts.timescale != timescale
            || sample.pts.timescale != timescale
            || duration.timescale != timescale
        {
            return Err(Error::Inconsistent(
                "the sample timescale does not match the track",
            ));
        }
        if dts.value != next {
            return Err(Error::Inconsistent(if next == 0 {
                "decode time does not start at zero"
            } else {
                "decode time is not contiguous"
            }));
        }
        if duration.value < 0 {
            return Err(Error::Inconsistent("a sample duration is negative"));
        }
        let duration = u32::try_from(duration.value)
            .map_err(|_| Error::Inconsistent("a sample duration does not fit"))?;
        let offset = sample
            .pts
            .value
            .checked_sub(dts.value)
            .ok_or(Error::TooLarge)?;
        let offset = i32::try_from(offset)
            .map_err(|_| Error::Inconsistent("the composition offset does not fit"))?;
        let _size = u32::try_from(sample.data.len()).map_err(|_| Error::TooLarge)?;
        next = next
            .checked_add(i64::from(duration))
            .ok_or(Error::TooLarge)?;
        stored.push(Sample {
            data: sample.data,
            duration,
            offset,
            keyframe: sample.keyframe,
        });
    }
    let media_duration = u64::try_from(next).map_err(|_| Error::TooLarge)?;
    Ok((stored, media_duration))
}

fn movie_ticks(media_duration: u64, media: NonZeroU32, movie: NonZeroU32) -> Result<u64, Error> {
    if media == movie {
        return Ok(media_duration);
    }
    let media = u64::from(media.get());
    let movie = u64::from(movie.get());
    let scaled = media_duration.checked_mul(movie).ok_or(Error::TooLarge)?;
    if scaled % media != 0 {
        return Err(Error::Inconsistent(
            "the track duration does not fit the movie timescale",
        ));
    }
    Ok(scaled / media)
}

fn write_ftyp(buf: &mut Buf) -> Result<(), Error> {
    let start = buf.open(b"ftyp");
    buf.push_fourcc(b"isom");
    buf.push_u32(0);
    buf.push_fourcc(b"isom");
    buf.push_fourcc(b"iso2");
    buf.push_fourcc(b"mp41");
    buf.close(start)
}

fn write_mdat(buf: &mut Buf, payload: &[u8]) -> Result<u64, Error> {
    let payload_len = u64::try_from(payload.len()).map_err(|_| Error::TooLarge)?;
    let start = u64::try_from(buf.bytes.len()).map_err(|_| Error::TooLarge)?;
    let small = payload_len.checked_add(8).ok_or(Error::TooLarge)?;
    if let Ok(size) = u32::try_from(small) {
        buf.push_u32(size);
        buf.push_fourcc(b"mdat");
        buf.push_bytes(payload);
        start.checked_add(8).ok_or(Error::TooLarge)
    } else {
        let size = payload_len.checked_add(16).ok_or(Error::TooLarge)?;
        buf.push_u32(1);
        buf.push_fourcc(b"mdat");
        buf.push_u64(size);
        buf.push_bytes(payload);
        start.checked_add(16).ok_or(Error::TooLarge)
    }
}

fn write_moov(
    buf: &mut Buf,
    movie: &MovieFile,
    placed: &[Option<u64>],
    data_at: u64,
) -> Result<(), Error> {
    let start = buf.open(b"moov");
    let duration = movie
        .tracks
        .iter()
        .map(|track| track.movie_duration)
        .max()
        .expect("the movie has a track");
    write_mvhd(buf, movie.timescale.get(), duration, movie.tracks.len())?;
    for (index, track) in movie.tracks.iter().enumerate() {
        let id = u32::try_from(index)
            .map_err(|_| Error::TooLarge)?
            .checked_add(1)
            .ok_or(Error::TooLarge)?;
        let relative = placed.get(index).ok_or(Error::TooLarge)?;
        let offset = match *relative {
            Some(relative) => Some(data_at.checked_add(relative).ok_or(Error::TooLarge)?),
            None => None,
        };
        write_trak(buf, id, track, offset)?;
    }
    buf.close(start)
}

fn write_mvhd(buf: &mut Buf, timescale: u32, duration: u64, tracks: usize) -> Result<(), Error> {
    let version = wide(duration);
    let start = buf.full(b"mvhd", version, 0);
    push_time(buf, version);
    buf.push_u32(timescale);
    push_duration(buf, version, duration);
    buf.push_u32(0x0001_0000);
    buf.push_u16(0x0100);
    buf.push_u16(0);
    buf.push_u32(0);
    buf.push_u32(0);
    for value in IDENTITY {
        buf.push_u32(value);
    }
    buf.push_zeros(24);
    let next = u32::try_from(tracks)
        .map_err(|_| Error::TooLarge)?
        .checked_add(1)
        .ok_or(Error::TooLarge)?;
    buf.push_u32(next);
    buf.close(start)
}

fn write_trak(buf: &mut Buf, id: u32, track: &Track, offset: Option<u64>) -> Result<(), Error> {
    let start = buf.open(b"trak");
    write_tkhd(buf, id, track)?;
    write_mdia(buf, track, offset)?;
    buf.close(start)
}

fn write_tkhd(buf: &mut Buf, id: u32, track: &Track) -> Result<(), Error> {
    let version = wide(track.movie_duration);
    let start = buf.full(b"tkhd", version, 1);
    push_time(buf, version);
    buf.push_u32(id);
    buf.push_u32(0);
    push_duration(buf, version, track.movie_duration);
    buf.push_zeros(8);
    buf.push_u16(0);
    buf.push_u16(0);
    let (volume, width, height) = match track.kind {
        TrackKind::Video { width, height, .. } => (0, u32::from(width), u32::from(height)),
    };
    buf.push_u16(volume);
    buf.push_u16(0);
    for value in IDENTITY {
        buf.push_u32(value);
    }
    buf.push_u32(fixed_16_16(width)?);
    buf.push_u32(fixed_16_16(height)?);
    buf.close(start)
}

fn write_mdia(buf: &mut Buf, track: &Track, offset: Option<u64>) -> Result<(), Error> {
    let start = buf.open(b"mdia");
    let version = wide(track.media_duration);
    let header = buf.full(b"mdhd", version, 0);
    push_time(buf, version);
    buf.push_u32(track.timescale.get());
    push_duration(buf, version, track.media_duration);
    buf.push_u16(0x55C4);
    buf.push_u16(0);
    buf.close(header)?;
    match track.kind {
        TrackKind::Video { .. } => {
            write_hdlr(buf, b"vide", "VideoHandler")?;
            write_video_minf(buf, track, offset)?;
        }
    }
    buf.close(start)
}

fn write_hdlr(buf: &mut Buf, kind: &[u8; 4], name: &str) -> Result<(), Error> {
    let start = buf.full(b"hdlr", 0, 0);
    buf.push_u32(0);
    buf.push_fourcc(kind);
    buf.push_zeros(12);
    buf.push_bytes(name.as_bytes());
    buf.push_u8(0);
    buf.close(start)
}

fn write_video_minf(buf: &mut Buf, track: &Track, offset: Option<u64>) -> Result<(), Error> {
    let TrackKind::Video {
        entry,
        config_box,
        width,
        height,
        config,
    } = &track.kind;
    let start = buf.open(b"minf");
    let vmhd = buf.full(b"vmhd", 0, 1);
    buf.push_zeros(8);
    buf.close(vmhd)?;
    write_dinf(buf)?;
    let stbl = buf.open(b"stbl");
    write_stsd_visual(buf, entry, config_box, *width, *height, config)?;
    write_sample_tables(buf, &track.samples, offset)?;
    buf.close(stbl)?;
    buf.close(start)
}

fn write_dinf(buf: &mut Buf) -> Result<(), Error> {
    let dinf = buf.open(b"dinf");
    let dref = buf.full(b"dref", 0, 0);
    buf.push_u32(1);
    let url = buf.full(b"url ", 0, 1);
    buf.close(url)?;
    buf.close(dref)?;
    buf.close(dinf)
}

fn write_stsd_visual(
    buf: &mut Buf,
    entry: &[u8; 4],
    config_box: &[u8; 4],
    width: u16,
    height: u16,
    config: &[u8],
) -> Result<(), Error> {
    let stsd = buf.full(b"stsd", 0, 0);
    buf.push_u32(1);
    let sample = buf.open(entry);
    buf.push_zeros(6);
    buf.push_u16(1);
    buf.push_zeros(16);
    buf.push_u16(width);
    buf.push_u16(height);
    buf.push_u32(DPI_72);
    buf.push_u32(DPI_72);
    buf.push_u32(0);
    buf.push_u16(1);
    buf.push_zeros(32);
    buf.push_u16(0x0018);
    buf.push_i16(-1);
    let config_at = buf.open(config_box);
    buf.push_bytes(config);
    buf.close(config_at)?;
    buf.close(sample)?;
    buf.close(stsd)
}

fn write_sample_tables(
    buf: &mut Buf,
    samples: &[Sample],
    offset: Option<u64>,
) -> Result<(), Error> {
    write_stts(buf, samples)?;
    write_ctts(buf, samples)?;
    write_stsc(buf, samples)?;
    write_stsz(buf, samples)?;
    write_chunks(buf, offset)?;
    write_stss(buf, samples)
}

fn write_stts(buf: &mut Buf, samples: &[Sample]) -> Result<(), Error> {
    let start = buf.full(b"stts", 0, 0);
    let runs = runs(samples.iter().map(|sample| sample.duration));
    push_runs(buf, &runs, |buf, duration| {
        buf.push_u32(duration);
        Ok(())
    })?;
    buf.close(start)
}

fn write_ctts(buf: &mut Buf, samples: &[Sample]) -> Result<(), Error> {
    if samples.iter().all(|sample| sample.offset == 0) {
        return Ok(());
    }
    let version = u8::from(samples.iter().any(|sample| sample.offset < 0));
    let start = buf.full(b"ctts", version, 0);
    let runs = runs(samples.iter().map(|sample| sample.offset));
    push_runs(buf, &runs, |buf, offset| {
        if version == 0 {
            let offset = u32::try_from(offset).map_err(|_| Error::TooLarge)?;
            buf.push_u32(offset);
        } else {
            buf.push_i32(offset);
        }
        Ok(())
    })?;
    buf.close(start)
}

fn write_stsc(buf: &mut Buf, samples: &[Sample]) -> Result<(), Error> {
    let start = buf.full(b"stsc", 0, 0);
    if samples.is_empty() {
        buf.push_u32(0);
    } else {
        let count = u32::try_from(samples.len()).map_err(|_| Error::TooLarge)?;
        buf.push_u32(1);
        buf.push_u32(1);
        buf.push_u32(count);
        buf.push_u32(1);
    }
    buf.close(start)
}

fn write_stsz(buf: &mut Buf, samples: &[Sample]) -> Result<(), Error> {
    let start = buf.full(b"stsz", 0, 0);
    buf.push_u32(0);
    buf.push_u32(u32::try_from(samples.len()).map_err(|_| Error::TooLarge)?);
    for sample in samples {
        buf.push_u32(u32::try_from(sample.data.len()).map_err(|_| Error::TooLarge)?);
    }
    buf.close(start)
}

fn write_chunks(buf: &mut Buf, offset: Option<u64>) -> Result<(), Error> {
    let Some(offset) = offset else {
        let start = buf.full(b"stco", 0, 0);
        buf.push_u32(0);
        return buf.close(start);
    };
    if let Ok(offset) = u32::try_from(offset) {
        let start = buf.full(b"stco", 0, 0);
        buf.push_u32(1);
        buf.push_u32(offset);
        buf.close(start)
    } else {
        let start = buf.full(b"co64", 0, 0);
        buf.push_u32(1);
        buf.push_u64(offset);
        buf.close(start)
    }
}

fn write_stss(buf: &mut Buf, samples: &[Sample]) -> Result<(), Error> {
    if samples.iter().all(|sample| sample.keyframe) {
        return Ok(());
    }
    let start = buf.full(b"stss", 0, 0);
    let mut numbers = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        if sample.keyframe {
            let number = u32::try_from(index)
                .map_err(|_| Error::TooLarge)?
                .checked_add(1)
                .ok_or(Error::TooLarge)?;
            numbers.push(number);
        }
    }
    buf.push_u32(u32::try_from(numbers.len()).map_err(|_| Error::TooLarge)?);
    for number in numbers {
        buf.push_u32(number);
    }
    buf.close(start)
}

fn push_runs<T: Copy>(
    buf: &mut Buf,
    runs: &[(u32, T)],
    push_value: impl Fn(&mut Buf, T) -> Result<(), Error>,
) -> Result<(), Error> {
    buf.push_u32(u32::try_from(runs.len()).map_err(|_| Error::TooLarge)?);
    for (count, value) in runs {
        buf.push_u32(*count);
        push_value(buf, *value)?;
    }
    Ok(())
}

fn runs<T: Copy + PartialEq>(values: impl Iterator<Item = T>) -> Vec<(u32, T)> {
    let mut out: Vec<(u32, T)> = Vec::new();
    for value in values {
        if let Some((count, previous)) = out.last_mut() {
            if *previous == value {
                *count = count
                    .checked_add(1)
                    .expect("a run is no longer than the sample count");
                continue;
            }
        }
        out.push((1, value));
    }
    out
}

fn wide(duration: u64) -> u8 {
    u8::from(duration > u64::from(u32::MAX))
}

fn push_time(buf: &mut Buf, version: u8) {
    if version == 0 {
        buf.push_u32(0);
        buf.push_u32(0);
    } else {
        buf.push_u64(0);
        buf.push_u64(0);
    }
}

fn push_duration(buf: &mut Buf, version: u8, duration: u64) {
    if version == 0 {
        let duration = u32::try_from(duration).expect("a version 0 duration fits in u32");
        buf.push_u32(duration);
    } else {
        buf.push_u64(duration);
    }
}

fn fixed_16_16(value: u32) -> Result<u32, Error> {
    value.checked_shl(16).ok_or(Error::TooLarge)
}

#[derive(Default)]
struct Buf {
    bytes: Vec<u8>,
}

impl Buf {
    fn open(&mut self, kind: &[u8; 4]) -> usize {
        let start = self.bytes.len();
        self.push_u32(0);
        self.push_fourcc(kind);
        start
    }

    fn full(&mut self, kind: &[u8; 4], version: u8, flags: u32) -> usize {
        let start = self.open(kind);
        let flag_bytes = flags.to_be_bytes();
        self.push_u8(version);
        self.push_u8(flag_bytes[1]);
        self.push_u8(flag_bytes[2]);
        self.push_u8(flag_bytes[3]);
        start
    }

    fn close(&mut self, start: usize) -> Result<(), Error> {
        let size = self.bytes.len().checked_sub(start).ok_or(Error::TooLarge)?;
        let size = u32::try_from(size).map_err(|_| Error::TooLarge)?;
        let slot = self
            .bytes
            .get_mut(start..start + 4)
            .expect("the box header was reserved");
        slot.copy_from_slice(&size.to_be_bytes());
        Ok(())
    }

    fn push_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn push_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_i16(&mut self, value: i16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_i32(&mut self, value: i32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_fourcc(&mut self, kind: &[u8; 4]) {
        self.bytes.extend_from_slice(kind);
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn push_zeros(&mut self, count: usize) {
        let len = self
            .bytes
            .len()
            .checked_add(count)
            .expect("the file fits in memory");
        self.bytes.resize(len, 0);
    }
}
