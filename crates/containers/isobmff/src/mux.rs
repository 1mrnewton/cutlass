//! Writes an ISO base media file.
//!
//! The default layout is `ftyp`, `mdat`, then `moov`. `moov` uses 64-bit chunk
//! offsets. Fragmented output is a separate mode: `ftyp`, `moov` with `mvex`,
//! then `moof`/`mdat` pairs, then `mfra`. A caller that drops the muxer before
//! `finish` does not get a completed index.
//!
//! Packet decode times on a track start at 0 and advance by each sample's
//! duration. Presentation time may differ; that difference is the composition
//! offset.

use crate::limits::{
    DEFAULT_SAMPLES_PER_FRAGMENT, MAX_BUFFERED_MUX_BYTES, MAX_PACKET_BYTES, MAX_TRACKS,
};
use crate::movie::pack_language;
use cutlass_container::{CodecId, Edit, Error, MediaKind, Mux, Packet, SampleFormat, StreamParams};
use std::io::{Seek, SeekFrom, Write};

/// Brand written in `ftyp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FileBrand {
    #[default]
    Mp4,
    Mov,
    M4a,
    M4v,
    ThreeGp,
}

/// Where the index goes, and whether fragments are written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MuxOptions {
    pub brand: FileBrand,
    /// When true, `mdat` is written first and `moov` last. Ignored for fragments.
    pub moov_at_end: bool,
    pub fragmented: bool,
    pub samples_per_fragment: u32,
}

impl Default for MuxOptions {
    fn default() -> Self {
        Self {
            brand: FileBrand::Mp4,
            moov_at_end: true,
            fragmented: false,
            samples_per_fragment: DEFAULT_SAMPLES_PER_FRAGMENT,
        }
    }
}

struct TrackOut {
    params: StreamParams,
    edits: Vec<Edit>,
    next_dts: i64,
}

struct Held {
    track: usize,
    dts: i64,
    pts: i64,
    duration: u32,
    size: u32,
    offset: u64,
    keyframe: bool,
}

struct Pending {
    track: usize,
    dts: i64,
    pts: i64,
    duration: u32,
    keyframe: bool,
    payload: Vec<u8>,
}

struct TfraRow {
    track: usize,
    time: u64,
    moof_offset: u64,
    sample_number: u32,
}

/// Muxes packets into an ISOBMFF file.
pub struct IsoBmffMux<W: Write + Seek> {
    writer: Option<W>,
    options: MuxOptions,
    tracks: Vec<TrackOut>,
    held: Vec<Held>,
    pending: Vec<Pending>,
    buffered: Vec<u8>,
    cursor: u64,
    mdat_at: Option<u64>,
    started: bool,
    finished: bool,
    sequence: u32,
    tfra: Vec<TfraRow>,
}

impl<W: Write + Seek> IsoBmffMux<W> {
    #[must_use]
    pub fn new(writer: W, options: MuxOptions) -> Self {
        Self {
            writer: Some(writer),
            options,
            tracks: Vec::new(),
            held: Vec::new(),
            pending: Vec::new(),
            buffered: Vec::new(),
            cursor: 0,
            mdat_at: None,
            started: false,
            finished: false,
            sequence: 1,
            tfra: Vec::new(),
        }
    }

    pub fn set_edits(&mut self, track_id: u32, edits: Vec<Edit>) -> Result<(), Error> {
        let Some(track) = self
            .tracks
            .iter_mut()
            .find(|track| track.params.track_id == track_id)
        else {
            return Err(Error::Missing { what: "track" });
        };
        track.edits = edits;
        Ok(())
    }

    pub fn finish_into(mut self) -> Result<W, Error> {
        self.finish()?;
        self.writer.take().ok_or(Error::Incomplete)
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(Error::Incomplete);
        };
        writer.write_all(bytes).map_err(|err| Error::Io {
            at: self.cursor,
            kind: err.kind(),
        })?;
        self.cursor = self
            .cursor
            .checked_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX))
            .ok_or(Error::Limit {
                what: "output",
                value: self.cursor,
            })?;
        Ok(())
    }

    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(Error::Incomplete);
        };
        writer.seek(SeekFrom::Start(at)).map_err(|err| Error::Io {
            at,
            kind: err.kind(),
        })?;
        writer.write_all(bytes).map_err(|err| Error::Io {
            at,
            kind: err.kind(),
        })?;
        writer
            .seek(SeekFrom::Start(self.cursor))
            .map_err(|err| Error::Io {
                at: self.cursor,
                kind: err.kind(),
            })?;
        Ok(())
    }

    fn start_progressive(&mut self) -> Result<(), Error> {
        if self.started {
            return Ok(());
        }
        let ftyp = ftyp_bytes(self.options.brand, self.options.fragmented);
        self.write_bytes(&ftyp)?;
        if self.options.fragmented {
            let moov = init_moov(&self.tracks)?;
            self.write_bytes(&moov)?;
        } else if self.options.moov_at_end {
            self.mdat_at = Some(self.cursor);
            self.write_bytes(&[0, 0, 0, 1, b'm', b'd', b'a', b't'])?;
            self.write_bytes(&[0, 0, 0, 0, 0, 0, 0, 0])?;
        }
        self.started = true;
        Ok(())
    }
}

impl<W: Write + Seek> Mux for IsoBmffMux<W> {
    fn add_stream(&mut self, params: &StreamParams) -> Result<u32, Error> {
        if self.finished || self.started {
            return Err(Error::Incomplete);
        }
        if self.tracks.len() >= MAX_TRACKS {
            return Err(Error::Limit {
                what: "tracks",
                value: u64::try_from(self.tracks.len()).unwrap_or(u64::MAX),
            });
        }
        if params.timescale.get() == 0 {
            return Err(Error::Unsupported {
                what: "media timescale 0",
            });
        }
        let sample_entry = entry_code(params)?;
        let mut stored = params.clone();
        if stored.track_id == 0 {
            stored.track_id = u32::try_from(self.tracks.len() + 1).unwrap_or(1);
        }
        if self
            .tracks
            .iter()
            .any(|track| track.params.track_id == stored.track_id)
        {
            return Err(Error::Unsupported {
                what: "duplicate track id",
            });
        }
        stored.sample_entry = sample_entry;
        let id = stored.track_id;
        self.tracks.push(TrackOut {
            params: stored,
            edits: Vec::new(),
            next_dts: 0,
        });
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), Error> {
        if self.finished {
            return Err(Error::Incomplete);
        }
        let Some(track) = self
            .tracks
            .iter()
            .position(|track| track.params.track_id == packet.track_id)
        else {
            return Err(Error::Missing { what: "track" });
        };
        let timescale = self.tracks[track].params.timescale;
        if packet.timescale != timescale {
            return Err(Error::Unsupported {
                what: "packet timescale",
            });
        }
        let size = u32::try_from(packet.payload.len()).map_err(|_| Error::Limit {
            what: "packet",
            value: u64::try_from(packet.payload.len()).unwrap_or(u64::MAX),
        })?;
        if u64::from(size) > MAX_PACKET_BYTES {
            return Err(Error::Limit {
                what: "packet",
                value: u64::from(size),
            });
        }
        if packet.dts != self.tracks[track].next_dts {
            return Err(Error::Unsupported {
                what: "decode time must start at 0 and follow durations",
            });
        }
        let next_dts = packet
            .dts
            .checked_add(i64::from(packet.duration))
            .ok_or(Error::Limit {
                what: "decode time",
                value: packet.dts.unsigned_abs(),
            })?;
        if self.options.fragmented {
            let kept = self.pending.len();
            self.pending.push(Pending {
                track,
                dts: packet.dts,
                pts: packet.pts,
                duration: packet.duration,
                keyframe: packet.keyframe,
                payload: packet.payload.clone(),
            });
            let limit = u32::max(self.options.samples_per_fragment, 1);
            if self.pending.len() >= usize::try_from(limit).unwrap_or(usize::MAX)
                && let Err(err) = self.flush_fragment()
            {
                self.pending.truncate(kept);
                return Err(err);
            }
            self.tracks[track].next_dts = next_dts;
            return Ok(());
        }
        if self.options.moov_at_end {
            self.start_progressive()?;
            let offset = self.cursor;
            self.write_bytes(&packet.payload)?;
            self.held.push(Held {
                track,
                dts: packet.dts,
                pts: packet.pts,
                duration: packet.duration,
                size,
                offset,
                keyframe: packet.keyframe,
            });
        } else {
            let next = self
                .buffered
                .len()
                .checked_add(packet.payload.len())
                .ok_or(Error::Limit {
                    what: "buffered mux",
                    value: MAX_BUFFERED_MUX_BYTES,
                })?;
            if u64::try_from(next).unwrap_or(u64::MAX) > MAX_BUFFERED_MUX_BYTES {
                return Err(Error::Limit {
                    what: "buffered mux",
                    value: u64::try_from(next).unwrap_or(u64::MAX),
                });
            }
            let offset = u64::try_from(self.buffered.len()).unwrap_or(0);
            self.buffered.extend_from_slice(&packet.payload);
            self.held.push(Held {
                track,
                dts: packet.dts,
                pts: packet.pts,
                duration: packet.duration,
                size,
                offset,
                keyframe: packet.keyframe,
            });
        }
        self.tracks[track].next_dts = next_dts;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Error> {
        if self.finished {
            return Ok(());
        }
        if self.options.fragmented {
            self.start_progressive()?;
            self.flush_fragment()?;
            let mfra = mfra_bytes(&self.tracks, &self.tfra)?;
            self.write_bytes(&mfra)?;
        } else if self.options.moov_at_end {
            self.start_progressive()?;
            let Some(mdat_at) = self.mdat_at else {
                return Err(Error::Incomplete);
            };
            let size = self.cursor.saturating_sub(mdat_at);
            self.patch(mdat_at + 8, &size.to_be_bytes())?;
            let moov = movie_bytes(&self.tracks, &self.held, |offset| offset)?;
            self.write_bytes(&moov)?;
        } else {
            let ftyp = ftyp_bytes(self.options.brand, false);
            let probe = movie_bytes(&self.tracks, &self.held, |_| 0)?;
            let payload_at = u64::try_from(ftyp.len() + probe.len()).unwrap_or(0) + 16;
            let moov = movie_bytes(&self.tracks, &self.held, |relative| {
                payload_at.saturating_add(relative)
            })?;
            if moov.len() != probe.len() {
                return Err(Error::Unsupported {
                    what: "moov size changed while placing it first",
                });
            }
            self.write_bytes(&ftyp)?;
            self.write_bytes(&moov)?;
            let mut header = Vec::new();
            header.extend_from_slice(&[0, 0, 0, 1, b'm', b'd', b'a', b't']);
            let mdat_size = 16u64.saturating_add(u64::try_from(self.buffered.len()).unwrap_or(0));
            header.extend_from_slice(&mdat_size.to_be_bytes());
            self.write_bytes(&header)?;
            let buffered = std::mem::take(&mut self.buffered);
            self.write_bytes(&buffered)?;
        }
        self.finished = true;
        Ok(())
    }
}

impl<W: Write + Seek> IsoBmffMux<W> {
    fn flush_fragment(&mut self) -> Result<(), Error> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.start_progressive()?;
        let moof_at = self.cursor;
        let pending = std::mem::take(&mut self.pending);
        let tfra_len = self.tfra.len();
        for (index, sample) in pending.iter().enumerate() {
            if sample.keyframe {
                self.tfra.push(TfraRow {
                    track: sample.track,
                    time: u64::try_from(sample.dts).unwrap_or(0),
                    moof_offset: moof_at,
                    sample_number: u32::try_from(index + 1).unwrap_or(1),
                });
            }
        }
        let built = match fragment_bytes(&self.tracks, &pending, self.sequence) {
            Ok(built) => built,
            Err(err) => {
                self.tfra.truncate(tfra_len);
                self.pending = pending;
                return Err(err);
            }
        };
        let (moof, mdat) = built;
        if let Err(err) = self.write_bytes(&moof) {
            self.tfra.truncate(tfra_len);
            self.pending = pending;
            return Err(err);
        }
        // `moof` is already on the output. Restoring the samples would write a
        // second fragment for bytes the index already names.
        self.write_bytes(&mdat)?;
        self.sequence = self.sequence.saturating_add(1);
        Ok(())
    }
}

fn entry_code(params: &StreamParams) -> Result<[u8; 4], Error> {
    if params.sample_entry != [0; 4] {
        return Ok(params.sample_entry);
    }
    let code = match params.codec {
        CodecId::H264 => *b"avc1",
        CodecId::Hevc => *b"hvc1",
        CodecId::Av1 => *b"av01",
        CodecId::Vp8 => *b"vp08",
        CodecId::Vp9 => *b"vp09",
        CodecId::Mpeg4 => *b"mp4v",
        CodecId::Mpeg2 => *b"mp2v",
        CodecId::Mjpeg => *b"jpeg",
        CodecId::ProRes => *b"apcn",
        CodecId::Aac | CodecId::Mp3 => *b"mp4a",
        CodecId::Opus => *b"Opus",
        CodecId::Flac => *b"fLaC",
        CodecId::Alac => *b"alac",
        CodecId::Ac3 => *b"ac-3",
        CodecId::Pcm => match audio_format(params) {
            SampleFormat::S16Le | SampleFormat::S24Le | SampleFormat::S32Le | SampleFormat::U8 => {
                *b"sowt"
            }
            SampleFormat::F32Le => *b"fl32",
            SampleFormat::F64Le => *b"fl64",
            _ => *b"twos",
        },
        CodecId::Unknown(code) => code,
        _ => {
            return Err(Error::CodecRejected {
                sample_entry: [0, 0, 0, 0],
            });
        }
    };
    Ok(code)
}

fn audio_format(params: &StreamParams) -> SampleFormat {
    match params.kind {
        MediaKind::Audio(audio) => audio.format,
        _ => SampleFormat::Unknown,
    }
}

fn ftyp_bytes(brand: FileBrand, fragmented: bool) -> Vec<u8> {
    let (major, compatible): ([u8; 4], &[&[u8; 4]]) = if fragmented {
        (*b"iso5", &[b"iso5", b"iso6", b"mp41"])
    } else {
        match brand {
            FileBrand::Mp4 => (*b"isom", &[b"isom", b"iso2", b"mp41"]),
            FileBrand::Mov => (*b"qt  ", &[b"qt  "]),
            FileBrand::M4a => (*b"M4A ", &[b"M4A ", b"mp42", b"isom"]),
            FileBrand::M4v => (*b"M4V ", &[b"M4V ", b"mp42", b"isom"]),
            FileBrand::ThreeGp => (*b"3gp4", &[b"3gp4", b"isom"]),
        }
    };
    let mut payload = Vec::new();
    payload.extend_from_slice(&major);
    payload.extend_from_slice(&0u32.to_be_bytes());
    for brand in compatible {
        payload.extend_from_slice(*brand);
    }
    box_bytes(b"ftyp", &payload)
}

fn movie_bytes(
    tracks: &[TrackOut],
    held: &[Held],
    map_offset: impl Fn(u64) -> u64,
) -> Result<Vec<u8>, Error> {
    let movie_scale = movie_timescale(tracks);
    let mut body = Vec::new();
    let duration = movie_duration(tracks, held, movie_scale)?;
    body.extend(mvhd(movie_scale, duration, tracks.len())?);
    for (index, track) in tracks.iter().enumerate() {
        let samples: Vec<&Held> = held.iter().filter(|sample| sample.track == index).collect();
        body.extend(trak(track, &samples, movie_scale, &map_offset)?);
    }
    Ok(box_bytes(b"moov", &body))
}

fn init_moov(tracks: &[TrackOut]) -> Result<Vec<u8>, Error> {
    let movie_scale = movie_timescale(tracks);
    let mut body = Vec::new();
    body.extend(mvhd(movie_scale, 0, tracks.len())?);
    for track in tracks {
        body.extend(trak(track, &[], movie_scale, &|offset| offset)?);
    }
    let mut mvex = Vec::new();
    for track in tracks {
        mvex.extend(trex(track));
    }
    body.extend(box_bytes(b"mvex", &mvex));
    Ok(box_bytes(b"moov", &body))
}

fn movie_timescale(tracks: &[TrackOut]) -> u32 {
    let mut scale = tracks
        .first()
        .map(|track| track.params.timescale.get())
        .unwrap_or(1000);
    for track in tracks.iter().skip(1) {
        scale = lcm(scale, track.params.timescale.get()).unwrap_or(90_000);
    }
    if scale == 0 { 1000 } else { scale }
}

fn lcm(left: u32, right: u32) -> Option<u32> {
    if left == 0 || right == 0 {
        return None;
    }
    let gcd = gcd(left, right);
    let value = u64::from(left) / u64::from(gcd) * u64::from(right);
    u32::try_from(value).ok()
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}

fn movie_duration(tracks: &[TrackOut], held: &[Held], movie_scale: u32) -> Result<u64, Error> {
    let mut duration = 0u64;
    for (index, track) in tracks.iter().enumerate() {
        let ticks = track_duration(held, index)?;
        let scaled = scale_u(ticks, track.params.timescale.get(), movie_scale);
        duration = duration.max(scaled);
    }
    Ok(duration)
}

fn track_duration(held: &[Held], track: usize) -> Result<u64, Error> {
    let mut expect = 0i64;
    let mut total = 0u64;
    for sample in held.iter().filter(|sample| sample.track == track) {
        if sample.dts != expect {
            return Err(Error::Unsupported {
                what: "decode time must start at 0 and follow durations",
            });
        }
        expect = expect
            .checked_add(i64::from(sample.duration))
            .ok_or(Error::Limit {
                what: "decode time",
                value: total,
            })?;
        total = total.saturating_add(u64::from(sample.duration));
    }
    Ok(total)
}

fn scale_u(ticks: u64, from: u32, to: u32) -> u64 {
    if from == 0 {
        return 0;
    }
    let value = u128::from(ticks) * u128::from(to) / u128::from(from);
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn trak(
    track: &TrackOut,
    samples: &[&Held],
    movie_scale: u32,
    map_offset: &impl Fn(u64) -> u64,
) -> Result<Vec<u8>, Error> {
    let media_duration = samples.iter().fold(0u64, |sum, sample| {
        sum.saturating_add(u64::from(sample.duration))
    });
    let movie_duration = scale_u(media_duration, track.params.timescale.get(), movie_scale);
    let mut body = tkhd(track, movie_duration);
    if !track.edits.is_empty() {
        body.extend(elst(&track.edits));
    }
    let mut mdia = Vec::new();
    mdia.extend(mdhd(track, media_duration));
    mdia.extend(hdlr(track));
    mdia.extend(minf(track, samples, map_offset)?);
    body.extend(box_bytes(b"mdia", &mdia));
    Ok(box_bytes(b"trak", &body))
}

fn tkhd(track: &TrackOut, duration: u64) -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 3];
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&track.params.track_id.to_be_bytes());
    payload.extend_from_slice(&[0, 0, 0, 0]);
    payload.extend_from_slice(&u32::try_from(duration).unwrap_or(u32::MAX).to_be_bytes());
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&[0, 0]);
    payload.extend_from_slice(&[0, 0]);
    let volume: u16 = if matches!(track.params.kind, MediaKind::Audio(_)) {
        0x0100
    } else {
        0
    };
    payload.extend_from_slice(&volume.to_be_bytes());
    payload.extend_from_slice(&[0, 0]);
    payload.extend_from_slice(&matrix());
    let (width, height) = match track.params.kind {
        MediaKind::Video(video) => (video.width, video.height),
        _ => (0, 0),
    };
    payload.extend_from_slice(&fixed_16_16(width).to_be_bytes());
    payload.extend_from_slice(&fixed_16_16(height).to_be_bytes());
    box_bytes(b"tkhd", &payload)
}

fn elst(edits: &[Edit]) -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&u32::try_from(edits.len()).unwrap_or(0).to_be_bytes());
    for edit in edits {
        let duration = u32::try_from(edit.segment_duration).unwrap_or(u32::MAX);
        payload.extend_from_slice(&duration.to_be_bytes());
        let media = edit.media_time.unwrap_or(-1);
        let media = i32::try_from(media).unwrap_or(if media < 0 { i32::MIN } else { i32::MAX });
        payload.extend_from_slice(&media.to_be_bytes());
        payload.extend_from_slice(&edit.media_rate.to_be_bytes());
    }
    box_bytes(b"edts", &box_bytes(b"elst", &payload))
}

fn mdhd(track: &TrackOut, duration: u64) -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&track.params.timescale.get().to_be_bytes());
    payload.extend_from_slice(&u32::try_from(duration).unwrap_or(u32::MAX).to_be_bytes());
    payload.extend_from_slice(&pack_language(track.params.language).to_be_bytes());
    payload.extend_from_slice(&[0, 0]);
    box_bytes(b"mdhd", &payload)
}

fn hdlr(track: &TrackOut) -> Vec<u8> {
    let (kind, name): ([u8; 4], &[u8]) = match track.params.kind {
        MediaKind::Video(_) => (*b"vide", b"Video\0"),
        MediaKind::Audio(_) => (*b"soun", b"Sound\0"),
        MediaKind::Data => (*b"meta", b"Data\0"),
    };
    let mut payload = vec![0, 0, 0, 0, 0, 0, 0, 0];
    payload.extend_from_slice(&kind);
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(name);
    box_bytes(b"hdlr", &payload)
}

fn minf(
    track: &TrackOut,
    samples: &[&Held],
    map_offset: &impl Fn(u64) -> u64,
) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    body.extend(match track.params.kind {
        MediaKind::Video(_) => box_bytes(b"vmhd", &[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        MediaKind::Audio(_) => box_bytes(b"smhd", &[0, 0, 0, 0, 0, 0, 0, 0]),
        MediaKind::Data => box_bytes(b"nmhd", &[0, 0, 0, 0]),
    });
    let url = box_bytes(b"url ", &[0, 0, 0, 1]);
    let mut dref = vec![0, 0, 0, 0, 0, 0, 0, 1];
    dref.extend(url);
    body.extend(box_bytes(b"dinf", &box_bytes(b"dref", &dref)));
    body.extend(stbl(track, samples, map_offset)?);
    Ok(box_bytes(b"minf", &body))
}

fn stbl(
    track: &TrackOut,
    samples: &[&Held],
    map_offset: &impl Fn(u64) -> u64,
) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    body.extend(stsd(track)?);
    body.extend(stts(samples));
    if samples.iter().any(|sample| sample.pts != sample.dts) {
        body.extend(ctts(samples)?);
    }
    body.extend(stsc_and_co64(samples, map_offset));
    body.extend(stsz(samples));
    if samples.iter().any(|sample| !sample.keyframe) {
        body.extend(stss(samples));
    }
    Ok(box_bytes(b"stbl", &body))
}

fn stsd(track: &TrackOut) -> Result<Vec<u8>, Error> {
    let mut entry = Vec::new();
    let code = track.params.sample_entry;
    match track.params.kind {
        MediaKind::Video(video) => {
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            entry.extend_from_slice(&u16::try_from(video.width).unwrap_or(u16::MAX).to_be_bytes());
            entry.extend_from_slice(
                &u16::try_from(video.height)
                    .unwrap_or(u16::MAX)
                    .to_be_bytes(),
            );
            entry.extend_from_slice(&0x00480000u32.to_be_bytes());
            entry.extend_from_slice(&0x00480000u32.to_be_bytes());
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 1]);
            entry.extend_from_slice(&[0; 32]);
            entry.extend_from_slice(&0x0018u16.to_be_bytes());
            entry.extend_from_slice(&(-1i16).to_be_bytes());
            entry.extend(config_boxes(track)?);
        }
        MediaKind::Audio(audio) => {
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
            entry.extend_from_slice(&audio.channels.to_be_bytes());
            entry.extend_from_slice(&audio.bits_per_sample.to_be_bytes());
            entry.extend_from_slice(&[0, 0, 0, 0]);
            let rate = u32::from(u16::try_from(audio.sample_rate).unwrap_or(u16::MAX)) << 16;
            entry.extend_from_slice(&rate.to_be_bytes());
            entry.extend(config_boxes(track)?);
        }
        MediaKind::Data => {
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
            entry.extend(&track.params.extradata);
        }
    }
    let mut payload = vec![0, 0, 0, 0, 0, 0, 0, 1];
    payload.extend(box_bytes(&code, &entry));
    Ok(box_bytes(b"stsd", &payload))
}

fn config_boxes(track: &TrackOut) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    let code = track.params.sample_entry;
    if matches!(&code, b"avc1" | b"avc3") {
        out.extend(box_bytes(b"avcC", &track.params.extradata));
    } else if matches!(&code, b"hvc1" | b"hev1" | b"dvh1" | b"dvhe") {
        out.extend(box_bytes(b"hvcC", &track.params.extradata));
    } else if &code == b"av01" {
        out.extend(box_bytes(b"av1C", &track.params.extradata));
    } else if matches!(&code, b"vp08" | b"vp09") {
        out.extend(box_bytes(b"vpcC", &track.params.extradata));
    } else if &code == b"Opus" {
        out.extend(box_bytes(b"dOps", &track.params.extradata));
    } else if &code == b"alac" {
        out.extend(box_bytes(b"alac", &track.params.extradata));
    } else if matches!(&code, b"ac-3" | b"sac3") {
        out.extend(box_bytes(b"dac3", &track.params.extradata));
    } else if &code == b"ec-3" {
        out.extend(box_bytes(b"dec3", &track.params.extradata));
    } else if &code == b"fLaC" {
        out.extend(box_bytes(b"dfLa", &track.params.extradata));
    } else if matches!(&code, b"mp4a" | b"mp4v") && !track.params.extradata.is_empty() {
        out.extend(box_bytes(b"esds", &esds(&track.params)));
    }
    if let Some(color) = track.params.color {
        let mut colr = b"nclx".to_vec();
        colr.extend_from_slice(&color.primaries.to_be_bytes());
        colr.extend_from_slice(&color.transfer.to_be_bytes());
        colr.extend_from_slice(&color.matrix.to_be_bytes());
        colr.push(if color.full_range { 0x80 } else { 0 });
        out.extend(box_bytes(b"colr", &colr));
    }
    Ok(out)
}

fn esds(params: &StreamParams) -> Vec<u8> {
    let (oti, stream_type): (u8, u8) = match params.codec {
        CodecId::Mp3 => (0x6B, 0x15),
        CodecId::Mpeg4 => (0x20, 0x11),
        _ => (0x40, 0x15),
    };
    let specific = descriptor(0x05, &params.extradata);
    let mut decoder = Vec::new();
    decoder.push(oti);
    decoder.push(stream_type);
    decoder.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    decoder.extend(specific);
    let mut es = Vec::new();
    es.extend_from_slice(&[0, 0, 0]);
    es.extend(descriptor(0x04, &decoder));
    es.extend(descriptor(0x06, &[0x02]));
    let mut payload = vec![0, 0, 0, 0];
    payload.extend(descriptor(0x03, &es));
    payload
}

fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let mut size = body.len();
    let mut bytes = Vec::new();
    for _ in 0..4 {
        let mut piece = u8::try_from(size & 0x7f).unwrap_or(0);
        size >>= 7;
        if size > 0 {
            piece |= 0x80;
        }
        bytes.push(piece);
        if size == 0 {
            break;
        }
    }
    out.extend(bytes);
    out.extend_from_slice(body);
    out
}

fn stts(samples: &[&Held]) -> Vec<u8> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for sample in samples {
        if let Some(last) = runs.last_mut()
            && last.1 == sample.duration
            && last.0 < u32::MAX
        {
            last.0 += 1;
            continue;
        }
        runs.push((1, sample.duration));
    }
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&u32::try_from(runs.len()).unwrap_or(0).to_be_bytes());
    for (count, delta) in runs {
        payload.extend_from_slice(&count.to_be_bytes());
        payload.extend_from_slice(&delta.to_be_bytes());
    }
    box_bytes(b"stts", &payload)
}

fn ctts(samples: &[&Held]) -> Result<Vec<u8>, Error> {
    let mut negative = false;
    let mut offsets = Vec::new();
    for sample in samples {
        let offset = sample.pts.checked_sub(sample.dts).ok_or(Error::Limit {
            what: "composition offset",
            value: 0,
        })?;
        if offset < 0 {
            negative = true;
        }
        if i32::try_from(offset).is_err() {
            return Err(Error::Limit {
                what: "composition offset",
                value: offset.unsigned_abs(),
            });
        }
        offsets.push(offset);
    }
    let mut runs: Vec<(u32, i64)> = Vec::new();
    for offset in offsets {
        if let Some(last) = runs.last_mut()
            && last.1 == offset
            && last.0 < u32::MAX
        {
            last.0 += 1;
            continue;
        }
        runs.push((1, offset));
    }
    let mut payload = vec![u8::from(negative), 0, 0, 0];
    payload.extend_from_slice(&u32::try_from(runs.len()).unwrap_or(0).to_be_bytes());
    for (count, offset) in runs {
        payload.extend_from_slice(&count.to_be_bytes());
        let stored = i32::try_from(offset).unwrap_or(0);
        if negative {
            payload.extend_from_slice(&stored.to_be_bytes());
        } else {
            payload.extend_from_slice(&u32::try_from(stored).unwrap_or(0).to_be_bytes());
        }
    }
    Ok(box_bytes(b"ctts", &payload))
}

/// One sample per chunk. Interleaved tracks then keep an exact file offset
/// without a second pass over chunk runs.
fn stsc_and_co64(samples: &[&Held], map_offset: &impl Fn(u64) -> u64) -> Vec<u8> {
    let mut stsc = vec![0, 0, 0, 0];
    if samples.is_empty() {
        stsc.extend_from_slice(&0u32.to_be_bytes());
    } else {
        stsc.extend_from_slice(&1u32.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes());
    }
    let mut co64 = vec![0, 0, 0, 0];
    co64.extend_from_slice(&u32::try_from(samples.len()).unwrap_or(0).to_be_bytes());
    for sample in samples {
        co64.extend_from_slice(&map_offset(sample.offset).to_be_bytes());
    }
    let mut out = box_bytes(b"stsc", &stsc);
    out.extend(box_bytes(b"co64", &co64));
    out
}

fn stsz(samples: &[&Held]) -> Vec<u8> {
    let constant = samples
        .first()
        .is_some_and(|first| samples.iter().all(|sample| sample.size == first.size));
    let mut payload = vec![0, 0, 0, 0];
    if constant && !samples.is_empty() {
        payload.extend_from_slice(&samples[0].size.to_be_bytes());
        payload.extend_from_slice(&u32::try_from(samples.len()).unwrap_or(0).to_be_bytes());
    } else {
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&u32::try_from(samples.len()).unwrap_or(0).to_be_bytes());
        for sample in samples {
            payload.extend_from_slice(&sample.size.to_be_bytes());
        }
    }
    box_bytes(b"stsz", &payload)
}

fn stss(samples: &[&Held]) -> Vec<u8> {
    let keys: Vec<u32> = samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| sample.keyframe)
        .filter_map(|(index, _)| u32::try_from(index + 1).ok())
        .collect();
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&u32::try_from(keys.len()).unwrap_or(0).to_be_bytes());
    for key in keys {
        payload.extend_from_slice(&key.to_be_bytes());
    }
    box_bytes(b"stss", &payload)
}

fn trex(track: &TrackOut) -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&track.params.track_id.to_be_bytes());
    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    box_bytes(b"trex", &payload)
}

fn fragment_bytes(
    tracks: &[TrackOut],
    pending: &[Pending],
    sequence: u32,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let mut runs: Vec<Run<'_>> = Vec::new();
    for sample in pending {
        if sample.track >= tracks.len() {
            return Err(Error::Missing { what: "track" });
        }
        if let Some(run) = runs.last_mut()
            && run.track == sample.track
        {
            run.samples.push(sample);
        } else {
            runs.push(Run {
                track: sample.track,
                local: 0,
                samples: vec![sample],
            });
        }
    }
    let mut mdat_payload = Vec::new();
    for run in &mut runs {
        run.local = u64::try_from(mdat_payload.len()).unwrap_or(0);
        for sample in &run.samples {
            mdat_payload.extend_from_slice(&sample.payload);
        }
    }
    let mdat = {
        let mut header = Vec::new();
        let size = 8u64.saturating_add(u64::try_from(mdat_payload.len()).unwrap_or(0));
        if size > u64::from(u32::MAX) {
            header.extend_from_slice(&[0, 0, 0, 1, b'm', b'd', b'a', b't']);
            header.extend_from_slice(
                &(16u64.saturating_add(u64::try_from(mdat_payload.len()).unwrap_or(0)))
                    .to_be_bytes(),
            );
        } else {
            header.extend_from_slice(&u32::try_from(size).unwrap_or(u32::MAX).to_be_bytes());
            header.extend_from_slice(b"mdat");
        }
        header.extend(mdat_payload);
        header
    };
    // Render once with zero offsets, then patch using the known moof length.
    let draft = render_moof(tracks, &runs, sequence, 0, 8)?;
    let header_len = if mdat.len() >= 16 && mdat[3] == 1 {
        16u64
    } else {
        8u64
    };
    let moof_len = u64::try_from(draft.len()).unwrap_or(0);
    let moof = render_moof(tracks, &runs, sequence, moof_len, header_len)?;
    Ok((moof, mdat))
}

struct Run<'a> {
    track: usize,
    local: u64,
    samples: Vec<&'a Pending>,
}

fn render_moof(
    tracks: &[TrackOut],
    runs: &[Run<'_>],
    sequence: u32,
    moof_len: u64,
    mdat_header: u64,
) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    let mut mfhd = vec![0, 0, 0, 0];
    mfhd.extend_from_slice(&sequence.to_be_bytes());
    body.extend(box_bytes(b"mfhd", &mfhd));
    for run in runs {
        let Some(meta) = tracks.get(run.track) else {
            continue;
        };
        let data_offset = i32::try_from(
            moof_len
                .saturating_add(mdat_header)
                .saturating_add(run.local),
        )
        .map_err(|_| Error::Limit {
            what: "fragment offset",
            value: moof_len,
        })?;
        body.extend(traf(meta, &run.samples, data_offset)?);
    }
    Ok(box_bytes(b"moof", &body))
}

fn traf(track: &TrackOut, samples: &[&Pending], data_offset: i32) -> Result<Vec<u8>, Error> {
    let mut tfhd = vec![0, 0x02, 0x00, 0x00];
    tfhd.extend_from_slice(&track.params.track_id.to_be_bytes());
    let base = samples.first().map(|sample| sample.dts).unwrap_or(0);
    let mut tfdt = vec![1, 0, 0, 0];
    tfdt.extend_from_slice(&u64::try_from(base).unwrap_or(0).to_be_bytes());
    // Version 1 so the composition offset is signed.
    // Flags: data-offset, duration, size, sample flags, composition offset.
    let mut trun = vec![1, 0, 0x0f, 0x01];
    trun.extend_from_slice(&u32::try_from(samples.len()).unwrap_or(0).to_be_bytes());
    trun.extend_from_slice(&data_offset.to_be_bytes());
    for sample in samples {
        trun.extend_from_slice(&sample.duration.to_be_bytes());
        let size = u32::try_from(sample.payload.len()).unwrap_or(0);
        trun.extend_from_slice(&size.to_be_bytes());
        let flags: u32 = if sample.keyframe {
            0x0200_0000
        } else {
            0x0101_0000
        };
        trun.extend_from_slice(&flags.to_be_bytes());
        let cts = i32::try_from(sample.pts.saturating_sub(sample.dts)).unwrap_or(0);
        trun.extend_from_slice(&cts.to_be_bytes());
    }
    let mut body = box_bytes(b"tfhd", &tfhd);
    body.extend(box_bytes(b"tfdt", &tfdt));
    body.extend(box_bytes(b"trun", &trun));
    Ok(box_bytes(b"traf", &body))
}

fn mfra_bytes(tracks: &[TrackOut], rows: &[TfraRow]) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    for (index, track) in tracks.iter().enumerate() {
        let mine: Vec<&TfraRow> = rows.iter().filter(|row| row.track == index).collect();
        if mine.is_empty() {
            continue;
        }
        let mut tfra = vec![1, 0, 0, 0];
        tfra.extend_from_slice(&track.params.track_id.to_be_bytes());
        tfra.extend_from_slice(&0u32.to_be_bytes());
        tfra.extend_from_slice(&u32::try_from(mine.len()).unwrap_or(0).to_be_bytes());
        for row in mine {
            tfra.extend_from_slice(&row.time.to_be_bytes());
            tfra.extend_from_slice(&row.moof_offset.to_be_bytes());
            tfra.push(1);
            tfra.push(1);
            tfra.push(u8::try_from(row.sample_number).unwrap_or(1));
        }
        body.extend(box_bytes(b"tfra", &tfra));
    }
    let mfra_size = u32::try_from(body.len() + 8 + 16).unwrap_or(u32::MAX);
    let mut mfro = vec![0, 0, 0, 0];
    mfro.extend_from_slice(&mfra_size.to_be_bytes());
    body.extend(box_bytes(b"mfro", &mfro));
    Ok(box_bytes(b"mfra", &body))
}

fn mvhd(timescale: u32, duration: u64, tracks: usize) -> Result<Vec<u8>, Error> {
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&timescale.to_be_bytes());
    payload.extend_from_slice(&u32::try_from(duration).unwrap_or(u32::MAX).to_be_bytes());
    payload.extend_from_slice(&0x00010000u32.to_be_bytes());
    payload.extend_from_slice(&0x0100u16.to_be_bytes());
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&matrix());
    payload.extend_from_slice(&[0; 24]);
    let next = u32::try_from(tracks + 1).unwrap_or(1);
    payload.extend_from_slice(&next.to_be_bytes());
    Ok(box_bytes(b"mvhd", &payload))
}

fn matrix() -> Vec<u8> {
    let mut out = Vec::new();
    for value in [0x00010000u32, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000] {
        out.extend_from_slice(&value.to_be_bytes());
    }
    out
}

fn fixed_16_16(value: u32) -> u32 {
    u32::from(u16::try_from(value).unwrap_or(u16::MAX)) << 16
}

fn box_bytes(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = 8u32.saturating_add(u32::try_from(payload.len()).unwrap_or(0));
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}
