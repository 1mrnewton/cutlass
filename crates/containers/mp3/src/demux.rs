//! MP3 demuxer.
//!
//! One Layer III frame is one packet. The packet bytes are the frame,
//! including its header, which is what a Layer III decoder consumes.
//! Presentation time is the sample count at the frame's sample rate.
//! A Xing, Info, or VBRI frame is an index, not a packet.

use std::io::{BufReader, Read, Seek, SeekFrom};

use cutlass_codec::{CodecId, CodecParams, ContainerId, Packet};
use cutlass_container::{ContainerError, Demux, SeekLanding, Stream, Trim};

use crate::header::{Header, MpegVersion, compatible, parse_header, side_info_len};
use crate::id3::{ape_tag_bytes, id3v2_tag_bytes, is_id3v1, is_id3v2};
use crate::limits::{MAX_FRAME_BYTES, MAX_LEADING_ID3, MAX_RESYNC_BYTES, SEEK_WINDOW_BYTES};
use crate::tag::{Vbri, Xing, is_tag_frame, read_vbri, read_xing, xing_offset};

/// How a positive seek chooses a byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexKind {
    XingToc,
    VbriToc,
    ConstantBitrate,
    AverageBytes,
    /// Sequential reads work. A positive seek returns [`ContainerError::SeekUnavailable`].
    None,
}

#[derive(Debug)]
enum Index {
    Xing {
        toc: [u8; 100],
        bytes: u64,
        origin: u64,
    },
    Vbri {
        starts: Vec<u64>,
        frames_per_entry: u64,
        origin: u64,
    },
    Constant {
        stride: Option<u32>,
        bitrate_kbps: u32,
    },
    Average {
        audio_bytes: u64,
        frames: u64,
    },
    None,
}

struct Hit {
    offset: usize,
    header: Header,
    size: u32,
    free_unpadded: Option<u32>,
}

/// A Layer III elementary stream.
#[derive(Debug)]
pub struct Mp3Demux<R: Read + Seek> {
    reader: BufReader<R>,
    reader_at: u64,
    audio_end: u64,
    audio_start: u64,
    pos: u64,
    frame_index: u64,
    streams: Vec<Stream>,
    version: MpegVersion,
    sample_rate: u32,
    channels: u16,
    samples_per_frame: u32,
    free_unpadded: Option<u32>,
    index: Index,
    index_kind: IndexKind,
    audio_frames: Option<u64>,
    duration_samples: Option<i64>,
    duration_exact: bool,
    reservoir_preroll: u32,
    cancelled: bool,
}

impl<R: Read + Seek> Mp3Demux<R> {
    /// Open an MP3 file. The reader is positioned by this demuxer from then on.
    pub fn open(reader: R) -> Result<Self, ContainerError> {
        let mut reader = BufReader::with_capacity(32 * 1024, reader);
        let file_len = reader.seek(SeekFrom::End(0))?;
        let mut audio_end = file_len;
        if file_len >= 128 {
            let mut tail = [0u8; 128];
            reader.seek(SeekFrom::Start(file_len - 128))?;
            reader.read_exact(&mut tail)?;
            if is_id3v1(&tail) {
                audio_end = file_len - 128;
            }
        }
        audio_end = skip_ape(&mut reader, audio_end)?;

        let mut pos = 0u64;
        for _ in 0..MAX_LEADING_ID3 {
            if pos + 10 > audio_end {
                break;
            }
            let mut header = [0u8; 10];
            reader.seek(SeekFrom::Start(pos))?;
            reader.read_exact(&mut header)?;
            if !is_id3v2(&header) {
                break;
            }
            let Some(len) = id3v2_tag_bytes(&header)? else {
                break;
            };
            let end = pos
                .checked_add(len)
                .ok_or(ContainerError::Limit { what: "id3" })?;
            if end > audio_end {
                return Err(ContainerError::Truncated { what: "id3" });
            }
            pos = end;
        }

        if pos >= audio_end {
            return Err(ContainerError::NoFrames);
        }

        let available = audio_end - pos;
        let cap = MAX_RESYNC_BYTES + u64::from(MAX_FRAME_BYTES);
        let file_ended = available <= cap;
        let read_len = usize::try_from(available.min(cap))
            .map_err(|_| ContainerError::Limit { what: "resync" })?;
        let mut window = vec![0u8; read_len];
        reader.seek(SeekFrom::Start(pos))?;
        reader.read_exact(&mut window)?;
        let hunt_end = if file_ended {
            window.len()
        } else {
            usize::try_from(MAX_RESYNC_BYTES).unwrap_or(window.len())
        };
        let hit = find_first(&window, hunt_end.min(window.len()), file_ended)?;
        let Some(hit) = hit else {
            if file_ended {
                return Err(ContainerError::NoFrames);
            }
            return Err(ContainerError::Malformed { what: "sync" });
        };
        let offset =
            u64::try_from(hit.offset).map_err(|_| ContainerError::Limit { what: "frame" })?;
        let frame_pos = pos + offset;
        let frame_end = frame_pos + u64::from(hit.size);
        if frame_end > audio_end {
            return Err(ContainerError::Truncated { what: "frame" });
        }
        let frame = if hit.offset + hit.size as usize <= window.len() {
            window[hit.offset..hit.offset + hit.size as usize].to_vec()
        } else {
            let mut body = vec![0u8; hit.size as usize];
            reader.seek(SeekFrom::Start(frame_pos))?;
            reader.read_exact(&mut body)?;
            body
        };

        let xing = read_xing(&frame, &hit.header);
        let vbri = if xing.is_none() {
            read_vbri(&frame, &hit.header)
        } else {
            None
        };
        let tagged = xing.is_some() || vbri.is_some();
        let audio_start = if tagged { frame_end } else { frame_pos };
        if audio_start > audio_end {
            return Err(ContainerError::Truncated { what: "frame" });
        }

        let mut demux = Self {
            reader,
            reader_at: u64::MAX,
            audio_end,
            audio_start,
            pos: audio_start,
            frame_index: 0,
            streams: Vec::new(),
            version: hit.header.version,
            sample_rate: hit.header.sample_rate,
            channels: hit.header.channels,
            samples_per_frame: hit.header.samples_per_frame,
            free_unpadded: if tagged { None } else { hit.free_unpadded },
            index: Index::None,
            index_kind: IndexKind::None,
            audio_frames: None,
            duration_samples: None,
            duration_exact: false,
            reservoir_preroll: preroll_frames(hit.header.version),
            cancelled: false,
        };
        demux.finish_open(hit.header, frame_pos, &frame, xing, vbri)?;
        Ok(demux)
    }

    #[must_use]
    pub fn index_kind(&self) -> IndexKind {
        self.index_kind
    }

    fn finish_open(
        &mut self,
        header: Header,
        tag_pos: u64,
        frame: &[u8],
        xing: Option<Xing>,
        vbri: Option<Vbri>,
    ) -> Result<(), ContainerError> {
        let _ = (header, frame);
        let first = self.peek_frame(self.audio_start)?;
        let Some((first_header, first_len, first_free)) = first else {
            return Err(ContainerError::NoFrames);
        };
        if first_header.layer != 3 {
            return Err(ContainerError::Unsupported { what: "layer" });
        }
        self.version = first_header.version;
        self.sample_rate = first_header.sample_rate;
        self.channels = first_header.channels;
        self.samples_per_frame = first_header.samples_per_frame;
        self.free_unpadded = first_free;
        self.reservoir_preroll = preroll_frames(self.version);

        let second_at = self.audio_start + u64::from(first_len);
        let second = self.peek_frame(second_at)?;
        let mut variable = false;
        let mut stride = Some(first_len);
        if let Some((next, next_len, _)) = &second {
            if !compatible(&first_header, next) {
                return Err(ContainerError::Malformed {
                    what: "sample rate",
                });
            }
            if next.bitrate_kbps != first_header.bitrate_kbps {
                variable = true;
            }
            if next_len.abs_diff(first_len) > 1 {
                variable = true;
                stride = None;
            } else if *next_len != first_len {
                stride = None;
            }
        }

        let mut frames = None;
        let mut exact = false;
        let mut trim = None;
        if let Some(xing) = xing {
            let trusted = trust_xing(tag_pos, &xing, self.audio_end);
            if trusted {
                frames = xing.frames.map(u64::from);
                exact = frames.is_some();
                if let (Some(delay), Some(padding)) = (xing.encoder_delay, xing.trailing_padding) {
                    trim = Some(Trim {
                        encoder_delay: delay,
                        trailing_padding: padding,
                    });
                }
                // An Info tag is constant bitrate. The frame schedule is a
                // better index than the TOC, which is only a percent hint.
                if !xing.cbr
                    && let Some(toc) = xing.toc
                    && let Some(bytes) = xing.bytes
                {
                    self.index = Index::Xing {
                        toc,
                        bytes: u64::from(bytes),
                        origin: tag_pos,
                    };
                    self.index_kind = IndexKind::XingToc;
                }
                if matches!(self.index, Index::None)
                    && let (Some(bytes), Some(count)) = (xing.bytes, xing.frames)
                {
                    let tag_len = frame_len_at(tag_pos, self.audio_start);
                    let bytes = u64::from(bytes);
                    if bytes > tag_len && count > 0 {
                        self.index = Index::Average {
                            audio_bytes: bytes - tag_len,
                            frames: u64::from(count),
                        };
                        self.index_kind = IndexKind::AverageBytes;
                    }
                }
            }
        } else if let Some(vbri) = vbri {
            frames = Some(u64::from(vbri.frames));
            exact = true;
            if vbri.frames_per_entry > 0 && !vbri.starts.is_empty() {
                self.index = Index::Vbri {
                    starts: vbri.starts,
                    frames_per_entry: u64::from(vbri.frames_per_entry),
                    origin: tag_pos,
                };
                self.index_kind = IndexKind::VbriToc;
            } else if vbri.frames > 0 {
                let tag_len = frame_len_at(tag_pos, self.audio_start);
                let bytes = u64::from(vbri.bytes);
                if bytes > tag_len {
                    self.index = Index::Average {
                        audio_bytes: bytes - tag_len,
                        frames: u64::from(vbri.frames),
                    };
                    self.index_kind = IndexKind::AverageBytes;
                }
            }
        }

        if matches!(self.index, Index::None) && !variable && first_header.bitrate_kbps > 0 {
            self.index = Index::Constant {
                stride,
                bitrate_kbps: first_header.bitrate_kbps,
            };
            self.index_kind = IndexKind::ConstantBitrate;
        } else if matches!(self.index, Index::None) && self.free_unpadded.is_some() {
            self.index = Index::Constant {
                stride: Some(first_len),
                bitrate_kbps: 0,
            };
            self.index_kind = IndexKind::ConstantBitrate;
        }

        self.audio_frames = frames;
        self.duration_samples = if let Some(count) = frames {
            Some(scale_samples(count, self.samples_per_frame)?)
        } else if first_header.bitrate_kbps > 0 {
            estimate_samples(
                self.audio_end - self.audio_start,
                first_header.bitrate_kbps,
                self.sample_rate,
            )
        } else {
            None
        };
        self.duration_exact = exact;
        let bitrate = stream_bitrate(
            first_header.bitrate_kbps,
            self.duration_samples,
            self.audio_end - self.audio_start,
        );
        let mut params = CodecParams::audio(CodecId::Mp3, self.sample_rate, self.channels);
        params.bitrate = bitrate;
        if let Some(trim) = trim {
            params.encoder_delay = trim.encoder_delay;
            params.trailing_padding = trim.trailing_padding;
        }
        self.streams.push(Stream {
            index: 0,
            timescale: self.sample_rate,
            trim,
            reservoir_preroll: self.reservoir_preroll,
            params,
        });
        self.pos = self.audio_start;
        self.frame_index = 0;
        Ok(())
    }

    fn peek_frame(
        &mut self,
        pos: u64,
    ) -> Result<Option<(Header, u32, Option<u32>)>, ContainerError> {
        if pos >= self.audio_end || self.audio_end - pos < 4 {
            return Ok(None);
        }
        let mut hdr = [0u8; 4];
        self.read_at(pos, &mut hdr)?;
        let Some(header) = parse_header(u32::from_be_bytes(hdr)) else {
            return Ok(None);
        };
        if header.layer != 3 {
            return Ok(None);
        }
        let (size, free_unpadded) = if let Some(size) = header.frame_bytes {
            (size, None)
        } else if let Some(unpadded) = self.free_unpadded {
            (unpadded + header.padding, Some(unpadded))
        } else if let Some((size, unpadded)) = self.learn_free(pos, &header)? {
            (size, Some(unpadded))
        } else {
            return Ok(None);
        };
        if pos + u64::from(size) > self.audio_end {
            return Err(ContainerError::Truncated { what: "frame" });
        }
        Ok(Some((header, size, free_unpadded)))
    }

    fn read_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<(), ContainerError> {
        if self.reader_at != pos {
            self.reader.seek(SeekFrom::Start(pos))?;
            self.reader_at = pos;
        }
        self.reader.read_exact(buf)?;
        self.reader_at = self
            .reader_at
            .checked_add(u64::try_from(buf.len()).unwrap_or(0))
            .ok_or(ContainerError::Limit { what: "read" })?;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<(Header, Vec<u8>)>, ContainerError> {
        if self.pos >= self.audio_end {
            return Ok(None);
        }
        let remaining = self.audio_end - self.pos;
        if remaining < 4 {
            self.pos = self.audio_end;
            return Ok(None);
        }
        let mut hdr = [0u8; 4];
        self.read_at(self.pos, &mut hdr)?;
        if let Some(header) = parse_header(u32::from_be_bytes(hdr))
            && self.matches(&header)
            && let Some(size) = self.size_of(&header)
        {
            if u64::from(size) > remaining {
                return Err(ContainerError::Truncated { what: "frame" });
            }
            return Ok(Some(self.take(header, size)?));
        }
        self.resync()
    }

    fn take(&mut self, header: Header, size: u32) -> Result<(Header, Vec<u8>), ContainerError> {
        let mut data = vec![0u8; size as usize];
        self.read_at(self.pos, &mut data)?;
        self.pos += u64::from(size);
        Ok((header, data))
    }

    fn resync(&mut self) -> Result<Option<(Header, Vec<u8>)>, ContainerError> {
        let start = self.pos;
        let end = self
            .audio_end
            .min(start.saturating_add(MAX_RESYNC_BYTES + u64::from(MAX_FRAME_BYTES)));
        let len =
            usize::try_from(end - start).map_err(|_| ContainerError::Limit { what: "resync" })?;
        if len < 4 {
            self.pos = self.audio_end;
            return Ok(None);
        }
        let mut window = vec![0u8; len];
        self.read_at(start, &mut window)?;
        let file_ended = end == self.audio_end;
        let hunt = if file_ended {
            window.len()
        } else {
            usize::try_from(MAX_RESYNC_BYTES)
                .unwrap_or(window.len())
                .min(window.len())
        };
        match find_first(&window, hunt, file_ended) {
            Ok(Some(hit)) => {
                if !self.matches(&hit.header) {
                    return Err(ContainerError::Malformed { what: "sync" });
                }
                let at = start + u64::try_from(hit.offset).unwrap_or(0);
                self.pos = at;
                self.take(hit.header, hit.size).map(Some)
            }
            Ok(None) => {
                if file_ended {
                    self.pos = self.audio_end;
                    Ok(None)
                } else {
                    Err(ContainerError::Malformed { what: "sync" })
                }
            }
            Err(ContainerError::Truncated { .. }) if file_ended => {
                Err(ContainerError::Truncated { what: "frame" })
            }
            Err(err) => Err(err),
        }
    }

    fn matches(&self, header: &Header) -> bool {
        header.layer == 3
            && header.version == self.version
            && header.sample_rate == self.sample_rate
            && header.channels == self.channels
            && (header.bitrate_kbps == 0) == self.free_unpadded.is_some()
    }

    /// The first free-format frame has no size in the header. The stride is
    /// the distance to the next compatible free-format sync.
    fn learn_free(
        &mut self,
        pos: u64,
        header: &Header,
    ) -> Result<Option<(u32, u32)>, ContainerError> {
        let cap = usize::try_from(MAX_FRAME_BYTES)
            .unwrap_or(0)
            .saturating_add(4);
        let available = usize::try_from(self.audio_end.saturating_sub(pos)).unwrap_or(cap);
        let len = available.min(cap);
        if len < 8 {
            return Ok(None);
        }
        let mut buf = vec![0u8; len];
        self.read_at(pos, &mut buf)?;
        Ok(free_stride(&buf, 0, header))
    }

    fn size_of(&self, header: &Header) -> Option<u32> {
        if let Some(unpadded) = self.free_unpadded {
            let size = unpadded + header.padding;
            if (4..=MAX_FRAME_BYTES).contains(&size) {
                Some(size)
            } else {
                None
            }
        } else {
            header.frame_bytes.filter(|size| *size <= MAX_FRAME_BYTES)
        }
    }

    fn seek_byte(&self, target: u64, time: i64) -> Result<u64, ContainerError> {
        if target == 0 {
            return Ok(self.audio_start);
        }
        let byte = match &self.index {
            Index::None => return Err(ContainerError::SeekUnavailable),
            Index::Constant {
                stride,
                bitrate_kbps,
            } => {
                if let Some(stride) = stride {
                    self.audio_start
                        .checked_add(
                            target
                                .checked_mul(u64::from(*stride))
                                .ok_or(ContainerError::Limit { what: "seek" })?,
                        )
                        .ok_or(ContainerError::Limit { what: "seek" })?
                } else if *bitrate_kbps > 0 {
                    let bytes = target
                        .saturating_mul(u64::from(self.samples_per_frame))
                        .saturating_mul(u64::from(*bitrate_kbps) * 1000)
                        / u64::from(self.sample_rate)
                        / 8;
                    self.audio_start.saturating_add(bytes)
                } else {
                    return Err(ContainerError::SeekUnavailable);
                }
            }
            Index::Average {
                audio_bytes,
                frames,
            } => {
                if *frames == 0 {
                    return Err(ContainerError::SeekUnavailable);
                }
                self.audio_start + target.saturating_mul(*audio_bytes) / *frames
            }
            Index::Xing { toc, bytes, origin } => {
                let duration = self
                    .duration_samples
                    .ok_or(ContainerError::SeekUnavailable)?;
                origin.saturating_add(xing_offset(toc, *bytes, time, duration))
            }
            Index::Vbri {
                starts,
                frames_per_entry,
                origin,
            } => {
                if *frames_per_entry == 0 || starts.is_empty() {
                    return Err(ContainerError::SeekUnavailable);
                }
                let index = usize::try_from(target / *frames_per_entry).unwrap_or(starts.len() - 1);
                let index = index.min(starts.len() - 1);
                let mut pos = origin.saturating_add(starts[index]);
                if index + 1 < starts.len() {
                    let span = starts[index + 1].saturating_sub(starts[index]);
                    let within = target % *frames_per_entry;
                    pos = pos.saturating_add(within * span / *frames_per_entry);
                }
                pos
            }
        };
        Ok(byte.min(self.audio_end))
    }

    fn sync_near(&mut self, byte: u64) -> Result<u64, ContainerError> {
        if byte <= self.audio_start {
            return Ok(self.audio_start);
        }
        if byte >= self.audio_end {
            return Ok(self.audio_end);
        }
        let start = byte.saturating_sub(SEEK_WINDOW_BYTES).max(self.audio_start);
        let end = byte.saturating_add(SEEK_WINDOW_BYTES).min(self.audio_end);
        let len =
            usize::try_from(end - start).map_err(|_| ContainerError::Limit { what: "seek" })?;
        let mut window = vec![0u8; len];
        self.read_at(start, &mut window)?;
        let mut best_before: Option<u64> = None;
        let mut index = 0usize;
        while index + 4 <= window.len() {
            let raw = u32::from_be_bytes([
                window[index],
                window[index + 1],
                window[index + 2],
                window[index + 3],
            ]);
            let Some(header) = parse_header(raw) else {
                index += 1;
                continue;
            };
            if !self.matches(&header) {
                index += 1;
                continue;
            }
            let Some(size) = self.size_of(&header) else {
                index += 1;
                continue;
            };
            let size_us = usize::try_from(size).unwrap_or(0);
            let abs = start + u64::try_from(index).unwrap_or(0);
            if abs + u64::from(size) > self.audio_end {
                index += 1;
                continue;
            }
            if abs <= byte {
                best_before = Some(abs);
            } else if best_before.is_none() {
                return Ok(abs);
            } else {
                break;
            }
            if size_us > 1 {
                index += size_us;
            } else {
                index += 1;
            }
        }
        best_before.ok_or(ContainerError::Malformed { what: "seek sync" })
    }
}

impl<R: Read + Seek> Demux for Mp3Demux<R> {
    fn container_id(&self) -> ContainerId {
        ContainerId::Mp3
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn duration(&self) -> Option<i64> {
        self.duration_samples
    }

    fn duration_exact(&self) -> bool {
        self.duration_exact
    }

    fn next_packet(&mut self) -> Result<Option<Packet>, ContainerError> {
        if self.cancelled {
            return Err(ContainerError::Cancelled);
        }
        for _ in 0..8 {
            let Some((header, data)) = self.next_frame()? else {
                return Ok(None);
            };
            if is_tag_frame(&data, &header) {
                continue;
            }
            let pts = scale_samples(self.frame_index, self.samples_per_frame)?;
            let duration = i64::from(self.samples_per_frame);
            self.frame_index = self
                .frame_index
                .checked_add(1)
                .ok_or(ContainerError::Limit { what: "frames" })?;
            return Ok(Some(Packet {
                stream: 0,
                data,
                pts,
                duration,
                timescale: self.sample_rate,
                keyframe: true,
            }));
        }
        Err(ContainerError::Limit { what: "tag frames" })
    }

    fn seek(&mut self, time: i64) -> Result<SeekLanding, ContainerError> {
        if self.cancelled {
            return Err(ContainerError::Cancelled);
        }
        let time = time.max(0);
        let spf = i64::from(self.samples_per_frame);
        if let Some(duration) = self.duration_samples
            && time >= duration
        {
            self.pos = self.audio_end;
            self.frame_index = self.audio_frames.unwrap_or(0);
            return Ok(SeekLanding {
                pts: duration,
                indexed: self.index_kind != IndexKind::None,
                preroll_frames: self.reservoir_preroll,
            });
        }
        let target = u64::try_from(time / spf).unwrap_or(0);
        if let Some(frames) = self.audio_frames
            && target >= frames
        {
            self.pos = self.audio_end;
            self.frame_index = frames;
            let pts = self.duration_samples.unwrap_or(time);
            return Ok(SeekLanding {
                pts,
                indexed: self.index_kind != IndexKind::None,
                preroll_frames: self.reservoir_preroll,
            });
        }
        let rough = self.seek_byte(target, time)?;
        let landed = self.sync_near(rough)?;
        if landed >= self.audio_end {
            self.pos = self.audio_end;
            let pts = self.duration_samples.unwrap_or(time);
            return Ok(SeekLanding {
                pts,
                indexed: self.index_kind != IndexKind::None,
                preroll_frames: self.reservoir_preroll,
            });
        }
        self.pos = landed;
        self.frame_index = target;
        Ok(SeekLanding {
            pts: time / spf * spf,
            indexed: self.index_kind != IndexKind::None,
            preroll_frames: self.reservoir_preroll,
        })
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }
}

fn preroll_frames(version: MpegVersion) -> u32 {
    // MPEG-1 `main_data_begin` is 9 bits. The smallest Layer III frame there
    // is 96 bytes, so nine frames cover the reservoir. MPEG-2 and MPEG-2.5
    // use 8 bits and a 24-byte minimum frame, so twelve frames cover it.
    match version {
        MpegVersion::Mpeg1 => 9,
        MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 12,
    }
}

fn trust_xing(tag_pos: u64, xing: &Xing, audio_end: u64) -> bool {
    let Some(bytes) = xing.bytes else {
        return xing.frames.is_some() || xing.toc.is_some();
    };
    let claimed = tag_pos.saturating_add(u64::from(bytes));
    if claimed > audio_end {
        let over = claimed - audio_end;
        if audio_end == 0 || over > audio_end / 5 {
            return false;
        }
    }
    true
}

fn frame_len_at(start: u64, end: u64) -> u64 {
    end.saturating_sub(start)
}

fn scale_samples(frames: u64, spf: u32) -> Result<i64, ContainerError> {
    let samples = frames
        .checked_mul(u64::from(spf))
        .ok_or(ContainerError::Limit { what: "timestamp" })?;
    i64::try_from(samples).map_err(|_| ContainerError::Limit { what: "timestamp" })
}

fn estimate_samples(audio_len: u64, bitrate_kbps: u32, sample_rate: u32) -> Option<i64> {
    let bits = audio_len
        .checked_mul(8)?
        .checked_mul(u64::from(sample_rate))?;
    let per_second = u64::from(bitrate_kbps).checked_mul(1000)?;
    if per_second == 0 {
        return None;
    }
    i64::try_from(bits / per_second).ok()
}

fn stream_bitrate(kbps: u32, duration: Option<i64>, audio_len: u64) -> u32 {
    if kbps > 0 {
        return kbps.saturating_mul(1000);
    }
    let Some(duration) = duration else {
        return 0;
    };
    if duration <= 0 {
        return 0;
    }
    let bits = audio_len.saturating_mul(8);
    u32::try_from(bits / duration as u64).unwrap_or(0)
}

fn skip_ape<R: Read + Seek>(reader: &mut R, audio_end: u64) -> Result<u64, ContainerError> {
    if audio_end < 32 {
        return Ok(audio_end);
    }
    let mut footer = [0u8; 32];
    reader.seek(SeekFrom::Start(audio_end - 32))?;
    reader.read_exact(&mut footer)?;
    let Some(span) = ape_tag_bytes(&footer)? else {
        return Ok(audio_end);
    };
    if span > audio_end {
        return Err(ContainerError::Truncated { what: "ape" });
    }
    Ok(audio_end - span)
}

fn find_first(
    buf: &[u8],
    hunt_end: usize,
    file_ended: bool,
) -> Result<Option<Hit>, ContainerError> {
    let mut index = 0usize;
    let mut saw_truncated = false;
    let limit = hunt_end.min(buf.len());
    while index + 4 <= limit {
        let raw = u32::from_be_bytes([buf[index], buf[index + 1], buf[index + 2], buf[index + 3]]);
        let Some(header) = parse_header(raw) else {
            index += 1;
            continue;
        };
        if header.layer != 3 {
            if let Some(size) = header.frame_bytes
                && confirmed(buf, index, &header, size, file_ended)
            {
                return Err(ContainerError::Unsupported { what: "layer" });
            }
            index += 1;
            continue;
        }
        if let Some(size) = header.frame_bytes {
            if index as u64 + u64::from(size) > buf.len() as u64 {
                if file_ended {
                    saw_truncated = true;
                }
                index += 1;
                continue;
            }
            if confirmed(buf, index, &header, size, file_ended) {
                return Ok(Some(Hit {
                    offset: index,
                    header,
                    size,
                    free_unpadded: None,
                }));
            }
            index += 1;
            continue;
        }
        if let Some((size, unpadded)) = free_stride(buf, index, &header) {
            return Ok(Some(Hit {
                offset: index,
                header,
                size,
                free_unpadded: Some(unpadded),
            }));
        }
        if file_ended && index == 0 {
            return Err(ContainerError::Unsupported {
                what: "free format",
            });
        }
        index += 1;
    }
    if saw_truncated && file_ended {
        return Err(ContainerError::Truncated { what: "frame" });
    }
    Ok(None)
}

fn confirmed(buf: &[u8], index: usize, header: &Header, size: u32, file_ended: bool) -> bool {
    let Some(size_us) = usize::try_from(size).ok() else {
        return false;
    };
    let Some(next) = index.checked_add(size_us) else {
        return false;
    };
    if next > buf.len() {
        return false;
    }
    if next + 4 > buf.len() {
        return file_ended;
    }
    let raw = u32::from_be_bytes([buf[next], buf[next + 1], buf[next + 2], buf[next + 3]]);
    parse_header(raw).is_some_and(|n| compatible(header, &n))
}

fn free_stride(buf: &[u8], index: usize, header: &Header) -> Option<(u32, u32)> {
    let min = 4 + side_info_len(header);
    let max = usize::try_from(MAX_FRAME_BYTES)
        .ok()?
        .min(buf.len().saturating_sub(index));
    if max < min {
        return None;
    }
    let mut dist = min;
    while dist <= max && index + dist + 4 <= buf.len() {
        let at = index + dist;
        let raw = u32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]);
        if let Some(next) = parse_header(raw)
            && compatible(header, &next)
            && next.bitrate_kbps == 0
        {
            let size = u32::try_from(dist).ok()?;
            let unpadded = size.checked_sub(header.padding)?;
            if unpadded >= 4 {
                return Some((size, unpadded));
            }
        }
        dist += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use super::{IndexKind, Mp3Demux};
    use crate::header::{MpegVersion, frame_index, synthetic_frame};
    use cutlass_container::{ContainerError, Demux};

    fn open_bytes(bytes: &[u8]) -> Result<Mp3Demux<Cursor<Vec<u8>>>, ContainerError> {
        Mp3Demux::open(Cursor::new(bytes.to_vec()))
    }

    fn collect(bytes: &[u8]) -> Result<Vec<cutlass_codec::Packet>, ContainerError> {
        let mut demux = open_bytes(bytes)?;
        let mut out = Vec::new();
        while let Some(packet) = demux.next_packet()? {
            out.push(packet);
        }
        Ok(out)
    }

    #[test]
    fn two_frames_keep_their_bytes_and_sample_clock() {
        let mut file = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 1));
        let packets = collect(&file).expect("demux");
        assert_eq!(packets.len(), 2);
        assert_eq!(frame_index(&packets[0].data), 0);
        assert_eq!(frame_index(&packets[1].data), 1);
        assert_eq!(packets[0].data.len(), 384);
        assert_eq!(packets[0].pts, 0);
        assert_eq!(packets[1].pts, 1152);
        assert_eq!(packets[0].duration, 1152);
        assert_eq!(packets[0].timescale, 48_000);
        assert_ne!(packets[0].timescale, 1_000_000);
        assert!(packets[0].keyframe);
        let demux = open_bytes(&file).expect("open");
        assert_eq!(demux.streams()[0].params.channels, 2);
        assert_eq!(demux.streams()[0].params.codec, cutlass_codec::CodecId::Mp3);
        assert!(demux.streams()[0].params.extra.is_empty());
        assert_eq!(demux.streams()[0].reservoir_preroll, 9);
        assert_eq!(demux.index_kind(), IndexKind::ConstantBitrate);
    }

    #[test]
    fn id3v2_and_id3v1_are_not_audio_packets() {
        let mut file = vec![b'I', b'D', b'3', 4, 0, 0, 0, 0, 0, 4, 1, 2, 3, 4];
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 7));
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 8));
        let mut tail = vec![0u8; 128];
        tail[0] = b'T';
        tail[1] = b'A';
        tail[2] = b'G';
        file.extend(tail);
        let packets = collect(&file).expect("demux");
        assert_eq!(packets.len(), 2);
        assert_eq!(frame_index(&packets[0].data), 7);
        assert!(packets.iter().all(|p| !p.data.starts_with(b"TAG")));
        assert!(packets.iter().all(|p| !p.data.starts_with(b"ID3")));
    }

    #[test]
    fn a_declared_id3_larger_than_the_limit_allocates_nothing() {
        let mut header = [0u8; 10];
        header[0] = b'I';
        header[1] = b'D';
        header[2] = b'3';
        header[3] = 4;
        header[6] = 0x0F;
        header[7] = 0x7F;
        header[8] = 0x7F;
        header[9] = 0x7F;
        let err = open_bytes(&header).expect_err("limit");
        assert_eq!(err, ContainerError::Limit { what: "id3" });
    }

    #[test]
    fn empty_png_and_layer2_are_distinct_errors() {
        assert_eq!(
            open_bytes(&[]).expect_err("empty"),
            ContainerError::NoFrames
        );
        assert_eq!(
            open_bytes(b"ftyp").expect_err("movie"),
            ContainerError::NoFrames
        );
        let (raw, _) =
            crate::header::encode_layer3(MpegVersion::Mpeg1, 128, 48_000, false, 2).expect("base");
        // Force the layer field to Layer II and place a second header one frame later.
        let layer2 = (raw & !(0b11 << 17)) | (0b10 << 17);
        let header = crate::header::parse_header(layer2).expect("layer2");
        assert_eq!(header.layer, 2);
        let size = usize::try_from(header.frame_bytes.expect("size")).expect("size");
        let mut file = vec![0u8; size * 2];
        file[..4].copy_from_slice(&layer2.to_be_bytes());
        file[size..size + 4].copy_from_slice(&layer2.to_be_bytes());
        assert_eq!(
            open_bytes(&file).expect_err("layer"),
            ContainerError::Unsupported { what: "layer" }
        );
    }

    #[test]
    fn truncated_frame_is_not_a_short_packet() {
        let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
        let err = open_bytes(&frame[..frame.len() - 10]).expect_err("short");
        assert_eq!(err, ContainerError::Truncated { what: "frame" });
    }

    #[test]
    fn vbr_without_an_index_still_demuxes_and_refuses_the_seek() {
        let mut file = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 96, 48_000, 2, 1));
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 160, 48_000, 2, 2));
        let mut demux = open_bytes(&file).expect("open");
        assert_eq!(demux.index_kind(), IndexKind::None);
        assert_eq!(
            demux.seek(1152).expect_err("seek"),
            ContainerError::SeekUnavailable
        );
        let first = demux.next_packet().expect("pkt").expect("some");
        assert_eq!(frame_index(&first.data), 0);
        assert_eq!(first.pts, 0);
        demux.seek(0).expect("start");
        assert_eq!(
            frame_index(&demux.next_packet().expect("again").expect("some").data),
            0
        );
    }

    #[test]
    fn seek_lands_on_the_frame_that_contains_the_sample() {
        let mut file = Vec::new();
        for index in 0..1000u16 {
            file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, index));
        }
        let mut demux = Mp3Demux::open(Count::new(Cursor::new(file))).expect("open");
        demux.reader.get_mut().arm();
        let landing = demux.seek(800 * 1152).expect("seek");
        assert_eq!(landing.pts, 800 * 1152);
        assert!(landing.indexed);
        assert_eq!(landing.preroll_frames, 9);
        let packet = demux.next_packet().expect("pkt").expect("some");
        assert_eq!(frame_index(&packet.data), 800);
        assert_eq!(packet.pts, 800 * 1152);
        let counter = demux.reader.get_mut();
        assert!(counter.min_read > 100_000, "min {}", counter.min_read);
        assert!(counter.bytes < 128 * 1024, "bytes {}", counter.bytes);
        assert!(demux.seek(-5).expect("before").pts == 0);
        let past = demux.seek(1000 * 1152).expect("past");
        assert!(demux.next_packet().expect("end").is_none());
        assert!(past.pts >= 1000 * 1152 || past.pts == demux.duration().unwrap_or(past.pts));
    }

    #[test]
    fn xing_frame_is_skipped_and_gapless_is_metadata() {
        let audio = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 3);
        let xing = xing_frame(2, (audio.len() * 3) as u32, 576, 100);
        let mut file = xing;
        file.extend(audio.clone());
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 4));
        let mut demux = open_bytes(&file).expect("open");
        assert_eq!(demux.index_kind(), IndexKind::XingToc);
        assert!(demux.duration_exact());
        assert_eq!(demux.duration(), Some(2 * 1152));
        let trim = demux.streams()[0].trim.expect("trim");
        assert_eq!(trim.encoder_delay, 576);
        assert_eq!(trim.trailing_padding, 100);
        let packets = [
            demux.next_packet().expect("a").expect("some"),
            demux.next_packet().expect("b").expect("some"),
        ];
        assert!(demux.next_packet().expect("end").is_none());
        assert_eq!(frame_index(&packets[0].data), 3);
        assert_eq!(frame_index(&packets[1].data), 4);
        assert_eq!(packets[0].pts, 0);
        assert_eq!(packets[1].pts, 1152);
    }

    #[test]
    fn free_format_stride_is_the_distance_to_the_next_sync() {
        let (raw, size) =
            crate::header::encode_layer3(MpegVersion::Mpeg1, 128, 48_000, false, 2).expect("hdr");
        let free = raw & !0x0000_F000;
        let mut frame = vec![0u8; size as usize];
        frame[..4].copy_from_slice(&free.to_be_bytes());
        frame[5] = 0;
        frame[6] = 1;
        let mut file = frame.clone();
        frame[6] = 2;
        file.extend(frame);
        let packets = collect(&file).expect("demux");
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].data.len(), size as usize);
        assert_eq!(packets[0].pts, 0);
        assert_eq!(packets[1].pts, 1152);
        assert_eq!(
            open_bytes(&file).expect("open").index_kind(),
            IndexKind::ConstantBitrate
        );
    }

    #[test]
    fn one_byte_reads_and_cancel_and_mpeg2_preroll() {
        let mut file = synthetic_frame(MpegVersion::Mpeg2, 64, 22_050, 1, 0);
        file.extend(synthetic_frame(MpegVersion::Mpeg2, 64, 22_050, 1, 1));
        let mut demux = Mp3Demux::open(OneByte {
            inner: Cursor::new(file),
        })
        .expect("open");
        assert_eq!(demux.streams()[0].params.sample_rate, 22_050);
        assert_eq!(demux.streams()[0].params.channels, 1);
        assert_eq!(demux.streams()[0].reservoir_preroll, 12);
        assert_eq!(demux.streams()[0].timescale, 22_050);
        let packet = demux.next_packet().expect("pkt").expect("some");
        assert_eq!(packet.duration, 576);
        assert_eq!(packet.timescale, 22_050);
        demux.cancel();
        assert_eq!(demux.next_packet(), Err(ContainerError::Cancelled));
    }

    #[test]
    fn vbri_seek_uses_the_table() {
        let audio = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
        let stride = audio.len();
        let mut frames = Vec::new();
        for index in 0..4u16 {
            frames.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, index));
        }
        let span = u16::try_from(stride * 2).expect("span");
        let step = u16::try_from(stride).expect("step");
        let bytes = u32::try_from(stride * 5).expect("bytes");
        let tag = vbri_frame(4, bytes, &[span, step, step, step]);
        let mut file = tag;
        file.extend(frames);
        let mut demux = open_bytes(&file).expect("open");
        assert_eq!(demux.index_kind(), IndexKind::VbriToc);
        assert_eq!(demux.duration(), Some(4 * 1152));
        let landing = demux.seek(2 * 1152).expect("seek");
        assert_eq!(landing.pts, 2 * 1152);
        assert_eq!(
            frame_index(&demux.next_packet().expect("pkt").expect("some").data),
            2
        );
    }

    #[test]
    fn ape_trailer_is_not_an_audio_packet() {
        let mut file = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 1);
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 2));
        let mut footer = [0u8; 32];
        footer[..8].copy_from_slice(b"APETAGEX");
        footer[8..12].copy_from_slice(&2000u32.to_le_bytes());
        footer[12..16].copy_from_slice(&32u32.to_le_bytes());
        file.extend_from_slice(&footer);
        let packets = collect(&file).expect("packets");
        assert_eq!(packets.len(), 2);
        assert_eq!(frame_index(&packets[0].data), 1);
        assert_eq!(frame_index(&packets[1].data), 2);
    }

    #[test]
    fn mpeg25_uses_the_file_sample_rate_and_a_twelve_frame_reservoir() {
        let mut file = synthetic_frame(MpegVersion::Mpeg25, 32, 11_025, 2, 7);
        file.extend(synthetic_frame(MpegVersion::Mpeg25, 32, 11_025, 2, 8));
        let mut demux = open_bytes(&file).expect("open");
        assert_eq!(demux.streams()[0].timescale, 11_025);
        assert_eq!(demux.streams()[0].reservoir_preroll, 12);
        let packet = demux.next_packet().expect("pkt").expect("some");
        assert_eq!(packet.duration, 576);
        assert_eq!(packet.timescale, 11_025);
        assert_eq!(packet.pts, 0);
        assert!(packet.keyframe);
        assert_eq!(frame_index(&packet.data), 7);
    }

    #[test]
    fn bytes_before_the_first_sync_are_not_a_packet() {
        let mut file = vec![0u8; 40];
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 4));
        file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 5));
        let packets = collect(&file).expect("packets");
        assert_eq!(packets.len(), 2);
        assert_eq!(frame_index(&packets[0].data), 4);
        assert_eq!(packets[0].pts, 0);
        assert_eq!(packets[0].timescale, 48_000);
    }

    #[test]
    fn repeated_seek_is_stable_and_a_refused_seek_keeps_the_cursor() {
        let mut file = Vec::new();
        for index in 0..4u16 {
            file.extend(synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, index));
        }
        let mut demux = open_bytes(&file).expect("open");
        let landing = demux.seek(2 * 1152).expect("seek");
        assert!(landing.indexed);
        assert_eq!(landing.pts, 2 * 1152);
        let first = demux.next_packet().expect("pkt").expect("some");
        demux.seek(2 * 1152).expect("again");
        let second = demux.next_packet().expect("pkt").expect("some");
        assert_eq!(first.data, second.data);
        let past = demux.seek(100 * 1152).expect("past");
        assert_eq!(past.pts, 4 * 1152);
        assert!(demux.next_packet().expect("end").is_none());
        let start = demux.seek(-10).expect("start");
        assert_eq!(start.pts, 0);
        assert_eq!(
            frame_index(&demux.next_packet().expect("pkt").expect("some").data),
            0
        );

        let mut vbr = Vec::new();
        for (index, kbps) in [128u32, 96, 160].into_iter().enumerate() {
            let index = u16::try_from(index).expect("index");
            vbr.extend(synthetic_frame(MpegVersion::Mpeg1, kbps, 48_000, 2, index));
        }
        let mut demux = open_bytes(&vbr).expect("vbr");
        let opened = demux.next_packet().expect("pkt").expect("some");
        assert_eq!(frame_index(&opened.data), 0);
        assert_eq!(demux.seek(1152), Err(ContainerError::SeekUnavailable));
        assert_eq!(
            frame_index(&demux.next_packet().expect("pkt").expect("some").data),
            1
        );
    }

    fn xing_frame(frames: u32, bytes: u32, delay: u32, padding: u32) -> Vec<u8> {
        let (raw, size) =
            crate::header::encode_layer3(MpegVersion::Mpeg1, 128, 48_000, false, 2).expect("hdr");
        let mut frame = vec![0u8; size as usize];
        frame[..4].copy_from_slice(&raw.to_be_bytes());
        let at = 4 + crate::header::side_info_len(&crate::header::parse_header(raw).expect("h"));
        frame[at..at + 4].copy_from_slice(b"Xing");
        frame[at + 4..at + 8].copy_from_slice(&0x0Fu32.to_be_bytes());
        frame[at + 8..at + 12].copy_from_slice(&frames.to_be_bytes());
        frame[at + 12..at + 16].copy_from_slice(&bytes.to_be_bytes());
        let audio_bytes = u64::from(bytes) - size as u64;
        for percent in 0..100u32 {
            let pos = u64::from(percent) * u64::from(bytes) / 100;
            let _ = audio_bytes;
            let toc = (256 * pos / u64::from(bytes)).min(255);
            frame[at + 16 + percent as usize] = u8::try_from(toc).unwrap_or(255);
        }
        frame[at + 120..at + 127].copy_from_slice(b"Cutlass");
        let packed = (delay << 12) | (padding & 0x0FFF);
        let be = packed.to_be_bytes();
        frame[at + 141] = be[1];
        frame[at + 142] = be[2];
        frame[at + 143] = be[3];
        frame
    }

    fn vbri_frame(frames: u32, bytes: u32, entries: &[u16]) -> Vec<u8> {
        let (raw, size) =
            crate::header::encode_layer3(MpegVersion::Mpeg1, 128, 48_000, false, 2).expect("hdr");
        let mut frame = vec![0u8; size as usize];
        frame[..4].copy_from_slice(&raw.to_be_bytes());
        frame[36..40].copy_from_slice(b"VBRI");
        frame[40..42].copy_from_slice(&1u16.to_be_bytes());
        frame[46..50].copy_from_slice(&bytes.to_be_bytes());
        frame[50..54].copy_from_slice(&frames.to_be_bytes());
        let count = u16::try_from(entries.len()).unwrap_or(0);
        frame[54..56].copy_from_slice(&count.to_be_bytes());
        frame[56..58].copy_from_slice(&1u16.to_be_bytes());
        frame[58..60].copy_from_slice(&2u16.to_be_bytes());
        frame[60..62].copy_from_slice(&1u16.to_be_bytes());
        for (index, entry) in entries.iter().enumerate() {
            let at = 62 + index * 2;
            frame[at..at + 2].copy_from_slice(&entry.to_be_bytes());
        }
        frame
    }

    struct Count<T> {
        inner: T,
        pos: u64,
        armed: bool,
        min_read: u64,
        bytes: u64,
    }

    impl<T> Count<T> {
        fn new(inner: T) -> Self {
            Self {
                inner,
                pos: 0,
                armed: false,
                min_read: u64::MAX,
                bytes: 0,
            }
        }

        fn arm(&mut self) {
            self.armed = true;
            self.min_read = u64::MAX;
            self.bytes = 0;
        }
    }

    impl<T: Read> Read for Count<T> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.armed {
                self.min_read = self.min_read.min(self.pos);
            }
            let n = self.inner.read(buf)?;
            if self.armed {
                self.bytes += u64::try_from(n).unwrap_or(0);
            }
            self.pos += u64::try_from(n).unwrap_or(0);
            Ok(n)
        }
    }

    impl<T: Seek> Seek for Count<T> {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.pos = self.inner.seek(pos)?;
            Ok(self.pos)
        }
    }

    struct OneByte<T> {
        inner: T,
    }

    impl<T: Read> Read for OneByte<T> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            self.inner.read(&mut buf[..1])
        }
    }

    impl<T: Seek> Seek for OneByte<T> {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(pos)
        }
    }
}
