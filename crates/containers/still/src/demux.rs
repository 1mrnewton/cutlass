//! Open a still file into packets.
//!
//! The file is read up to the byte cap after the signature matches. A movie
//! is refused from the first bytes and is not buffered. Parse failures
//! return [`OpenError`] and no demuxer, so there is no partial packet list.

use crate::fail::{Fail, as_u64};
use crate::frame::{Frame, FrameControl, Sequence, StillKind};
use crate::gif::parse as parse_gif;
use crate::jpeg::parse as parse_jpeg;
use crate::limits::{MAX_FRAMES, file_cap};
use crate::png::parse as parse_png;
use crate::probe::probe;
use crate::webp::parse as parse_webp;
use cutlass_codec::{
    ContainerId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{
    Access, Demux, Gap, IndexEntry, IndexPlace, Input, Landing, OpenError, ReadError, SeekError,
    SeekLanding, SeekTo, Stream, TrackId, Trim, seek_with_index,
};
use std::io::{Read, Seek};

/// One open JPEG, PNG, WebP, or GIF file.
#[derive(Debug)]
pub struct StillDemux<R> {
    input: Input<R>,
    kind: StillKind,
    animated: bool,
    loop_count: Option<u16>,
    end: Option<i64>,
    streams: Vec<Stream>,
    gaps: Vec<Gap>,
    file: Vec<u8>,
    frames: Vec<OwnedFrame>,
    index: Vec<IndexEntry>,
    cursor: usize,
    cancelled: bool,
}

#[derive(Debug)]
struct OwnedFrame {
    start: usize,
    end: usize,
    pts: i64,
    duration: i64,
    timed: bool,
    keyframe: bool,
    disposal: u8,
    blend: Option<u8>,
}

impl<R: Read + Seek> StillDemux<R> {
    /// Reads and parses `input`.
    ///
    /// On failure the error is the whole result. No stream list and no
    /// packet list are published.
    pub fn open(mut input: Input<R>) -> Result<Self, OpenError> {
        let file = read_file(&mut input)?;
        let sequence = parse_file(&file).map_err(Fail::open)?;
        if sequence.frames.is_empty()
            || sequence.frames.len() > usize::try_from(MAX_FRAMES).unwrap_or(usize::MAX)
        {
            return Err(OpenError::HostileSize {
                declared: as_u64(sequence.frames.len()),
                limit: u64::from(MAX_FRAMES),
            });
        }
        let end = sequence.exclusive_end().map_err(Fail::open)?;
        let timescale = Timescale::new(sequence.kind.clock(sequence.animated));
        let size =
            FrameSize::try_new(sequence.width, sequence.height).map_err(|_| OpenError::Source {
                message: "still frame is empty".to_owned(),
            })?;
        let params =
            StreamParams::video(StreamCodec::Known(sequence.kind.codec()), size, Vec::new());
        let stream = Stream::new(
            TrackId::new(0),
            params,
            timescale,
            None,
            Vec::new(),
            Trim::default(),
        );
        let frames = sequence
            .frames
            .into_iter()
            .map(OwnedFrame::from)
            .collect::<Vec<_>>();
        let index = index_of(&frames);
        Ok(Self {
            input,
            kind: sequence.kind,
            animated: sequence.animated,
            loop_count: sequence.loop_count,
            end,
            streams: vec![stream],
            gaps: Vec::new(),
            file,
            frames,
            index,
            cursor: 0,
            cancelled: false,
        })
    }

    #[must_use]
    pub const fn kind(&self) -> StillKind {
        self.kind
    }

    /// WebP set the animation flag, or the GIF has more than one image.
    ///
    /// A one-frame GIF still uses the GIF timescale. A still WebP does not.
    #[must_use]
    pub const fn animated(&self) -> bool {
        self.animated
    }

    /// Netscape or `ANIM` loop count.
    ///
    /// `None` means the file did not store one. `Some(0)` means repeat
    /// forever. The packets are still one pass.
    #[must_use]
    pub const fn loop_count(&self) -> Option<u16> {
        self.loop_count
    }

    /// Exclusive end of the frame clock, in that clock's ticks.
    ///
    /// `None` when the file has no duration.
    #[must_use]
    pub const fn media_end(&self) -> Option<i64> {
        self.end
    }

    #[must_use]
    pub fn frame(&self, index: usize) -> Option<FrameControl> {
        self.frames.get(index).map(OwnedFrame::control)
    }

    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

impl<R: Read + Seek> Demux for StillDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Still
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn gaps(&self) -> &[Gap] {
        &self.gaps
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        let Some(frame) = self.frames.get(self.cursor) else {
            return Ok(None);
        };
        let Some(bytes) = self.file.get(frame.start..frame.end) else {
            return Err(ReadError::Source {
                message: "frame is outside the file".to_owned(),
            });
        };
        let Some(timescale) = self.streams.first().map(Stream::timescale) else {
            return Err(ReadError::Source {
                message: "still has no stream".to_owned(),
            });
        };
        let timestamp = if frame.timed {
            Timestamp::try_new(Some(frame.pts), Some(frame.pts), Some(frame.duration)).map_err(
                |_| ReadError::Source {
                    message: "frame duration is negative".to_owned(),
                },
            )?
        } else {
            Timestamp::missing()
        };
        let packet = Packet::try_new(0, bytes.to_vec(), timestamp, timescale, frame.keyframe)
            .map_err(|_| ReadError::Source {
                message: "frame clock does not match the stream".to_owned(),
            })?;
        self.cursor += 1;
        Ok(Some(packet))
    }

    fn seek(&mut self, mut to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if to.track != 0 {
            return Err(ReadError::Seek(SeekError::UnknownTrack));
        }
        if self.input.access() == Access::Forward {
            return Err(ReadError::Seek(SeekError::ForwardOnly));
        }
        if to.end.is_none() {
            to.end = self.end;
        }
        let landing = seek_with_index(&self.index, to).map_err(ReadError::Seek)?;
        match landing.landing {
            Landing::End => {
                self.cursor = self.frames.len();
                Ok(SeekLanding {
                    index_used: true,
                    payload_scans: 0,
                    ..landing
                })
            }
            Landing::Packet { ordinal, .. } => {
                self.cursor = usize::try_from(ordinal).map_err(|_| ReadError::Source {
                    message: "seek ordinal does not fit".to_owned(),
                })?;
                Ok(SeekLanding {
                    index_used: true,
                    payload_scans: 0,
                    ..landing
                })
            }
        }
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.input.consumed()
    }
}

impl From<Frame> for OwnedFrame {
    fn from(frame: Frame) -> Self {
        Self {
            start: frame.start,
            end: frame.end,
            pts: frame.pts,
            duration: frame.duration,
            timed: frame.timed,
            keyframe: frame.keyframe,
            disposal: frame.disposal,
            blend: frame.blend,
        }
    }
}

impl OwnedFrame {
    fn control(&self) -> FrameControl {
        FrameControl {
            disposal: self.disposal,
            webp_blend: self.blend,
        }
    }
}

fn index_of(frames: &[OwnedFrame]) -> Vec<IndexEntry> {
    frames
        .iter()
        .enumerate()
        .map(|(ordinal, frame)| IndexEntry {
            track: 0,
            time: if frame.timed { frame.pts } else { 0 },
            ordinal: u32::try_from(ordinal).unwrap_or(u32::MAX),
            keyframe: frame.keyframe,
            byte_position: as_u64(frame.start),
        })
        .collect()
}

fn parse_file(file: &[u8]) -> Result<Sequence, Fail> {
    if file.is_empty() {
        return Err(Fail::Truncated {
            expected: 2,
            got: 0,
        });
    }
    match file[0] {
        0xFF => parse_jpeg(file),
        0x89 => parse_png(file),
        b'G' => parse_gif(file),
        b'R' => parse_webp(file),
        _ => Err(Fail::Unrecognized),
    }
}

fn read_file<R: Read + Seek>(input: &mut Input<R>) -> Result<Vec<u8>, OpenError> {
    let cap = file_cap(input.limits());
    let prefix_len = match input.length() {
        Some(0) => {
            return Err(OpenError::Truncated {
                expected: 2,
                got: 0,
            });
        }
        Some(length) => usize::try_from(length.min(12)).unwrap_or(12),
        None => 12,
    };
    let mut prefix = read_up_to(input, prefix_len)?;
    match probe(&prefix) {
        cutlass_container::ProbeResult::NeedBytes { need, .. } => {
            return Err(OpenError::Truncated {
                expected: u64::try_from(need).unwrap_or(u64::MAX),
                got: as_u64(prefix.len()),
            });
        }
        cutlass_container::ProbeResult::Refuse { .. } => {
            return Err(OpenError::Unrecognized {
                family: ContainerId::Still,
            });
        }
        cutlass_container::ProbeResult::Accept { .. } => {}
    }
    if let Some(length) = input.length() {
        if length > cap {
            return Err(OpenError::HostileSize {
                declared: length,
                limit: cap,
            });
        }
        let have = as_u64(prefix.len());
        if length > have {
            let rest = input
                .read_limited(length - have, cap)
                .map_err(OpenError::from_header)?;
            prefix.extend(rest);
        }
        return Ok(prefix);
    }
    read_until_end(input, prefix, cap)
}

fn read_up_to<R: Read + Seek>(input: &mut Input<R>, len: usize) -> Result<Vec<u8>, OpenError> {
    let mut buf = vec![0_u8; len];
    let mut filled = 0usize;
    while filled < len {
        let amount = input
            .read(&mut buf[filled..])
            .map_err(OpenError::from_header)?;
        if amount == 0 {
            buf.truncate(filled);
            return Ok(buf);
        }
        filled += amount;
    }
    Ok(buf)
}

fn read_until_end<R: Read + Seek>(
    input: &mut Input<R>,
    mut buf: Vec<u8>,
    cap: u64,
) -> Result<Vec<u8>, OpenError> {
    let mut tmp = [0_u8; 8192];
    loop {
        let amount = input.read(&mut tmp).map_err(OpenError::from_header)?;
        if amount == 0 {
            return Ok(buf);
        }
        let next = as_u64(buf.len()).saturating_add(as_u64(amount));
        if next > cap {
            return Err(OpenError::HostileSize {
                declared: next,
                limit: cap,
            });
        }
        buf.extend_from_slice(&tmp[..amount]);
    }
}

/// Index place for a still file: the seek table is built while opening.
/// The file does not store a separate index.
pub(crate) const fn index_place() -> IndexPlace {
    IndexPlace::Absent
}
