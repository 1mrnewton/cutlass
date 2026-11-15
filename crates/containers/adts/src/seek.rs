//! A raw AAC file that can seek.
//!
//! ADTS stores no index and no duration. [`AdtsSeekFile::open`] walks every
//! frame header once, without copying a payload, and keeps one index row per
//! frame: its byte position and the sample its first raw block starts at.
//! A seek lands on the frame that holds the asked sample and the reader
//! starts again from that frame's first byte.
//!
//! The walk accepts and refuses frames the same way [`AdtsDemuxer`] does, so
//! the index and the packets agree. Two things differ from the forward
//! reader. An ID3v2 tag at the start of the file is skipped by its declared
//! size instead of byte by byte. Bytes after the last whole frame (a cut-off
//! frame, an ID3v1 or APE tag) are left out of the stream instead of ending
//! it with a truncation error.

use std::io::{Read, Seek};

use cutlass_codec::access::{ChannelLayout, CodecError, CodecParams};
use cutlass_codec::{CodecId, ContainerId, Packet};
use cutlass_container::{
    IndexEntry, Input, Limits, OpenError, ProbeResult, ReadError, SeekError, SeekLanding, SeekTo,
    Stream, seek_with_index,
};

use crate::shared::{map_streams, read_codec, read_source, to_shared};
use crate::{
    AdtsDemuxer, Demuxer, Header, MAX_RESYNC, Sniff, StreamInfo, layout_of, named_profile, rate_of,
    sniff,
};

/// Bytes read per pull, while indexing and while reading packets.
const CHUNK: usize = 64 * 1024;
/// An ID3v2 header: `ID3`, version, revision, flags, and a synchsafe size.
const ID3_HEADER: usize = 10;
/// The flag that says a 10-byte footer follows the tag.
const ID3_FOOTER: u8 = 0x10;
/// Raw blocks are 1024 samples each.
const BLOCK: u64 = 1024;

/// Raw AAC (ADTS) on a reader that can seek.
pub struct AdtsSeekFile<R> {
    demux: AdtsDemuxer,
    input: Input<R>,
    shared: Vec<Stream>,
    index: Vec<IndexEntry>,
    /// Byte just past the last whole frame.
    end: u64,
    /// Next byte the reader takes.
    position: u64,
    /// Samples in the whole stream.
    samples: u64,
    cancelled: bool,
    ended: bool,
}

impl<R: Read + Seek> AdtsSeekFile<R> {
    /// Indexes every frame of a `length`-byte file and leaves the reader at
    /// the first one.
    ///
    /// A file with no whole frame is [`OpenError::Unrecognized`]. A frame
    /// the forward reader would refuse (another profile, a configuration
    /// change, junk past the resync limit) refuses the file here.
    pub fn open(reader: R, length: u64) -> Result<Self, OpenError> {
        let limits = Limits::standard();
        let mut input = Input::random(reader, length, limits);
        let start = id3v2_end(&mut input, length)?;
        input.seek(start).map_err(OpenError::from_header)?;
        let frames = walk(&mut input, start, &limits)?;
        let shared = map_streams(std::slice::from_ref(&frames.info))?;
        input.seek(frames.first).map_err(OpenError::from_header)?;
        Ok(Self {
            demux: AdtsDemuxer::new(),
            input,
            shared,
            index: frames.index,
            end: frames.end,
            position: frames.first,
            samples: frames.samples,
            cancelled: false,
            ended: false,
        })
    }
}

impl<R> AdtsSeekFile<R> {
    /// Samples in the stream, on the sample-rate clock: 1024 per raw block.
    #[must_use]
    pub const fn samples(&self) -> u64 {
        self.samples
    }

    /// Frames in the index, one per ADTS frame.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.index.len()
    }
}

impl<R: Read + Seek> AdtsSeekFile<R> {
    fn pull(&mut self) -> Result<(), ReadError> {
        if self.position >= self.end {
            self.ended = true;
            return self.demux.finish_bytes().map_err(read_codec);
        }
        let left = usize::try_from(self.end - self.position).unwrap_or(CHUNK);
        let mut buf = vec![0u8; left.min(CHUNK)];
        let n = self.input.read(&mut buf).map_err(read_source)?;
        if n == 0 {
            // The file is shorter than when it was indexed.
            self.ended = true;
            return self.demux.finish_bytes().map_err(read_codec);
        }
        self.position = self.position.saturating_add(u64::try_from(n).unwrap_or(0));
        self.demux.push_bytes(&buf[..n]).map_err(read_codec)
    }
}

impl<R: Read + Seek + Send> cutlass_container::Demux for AdtsSeekFile<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Adts
    }

    fn streams(&self) -> &[Stream] {
        &self.shared
    }

    fn gaps(&self) -> &[cutlass_container::Gap] {
        &[]
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        loop {
            match self.demux.next_packet() {
                Ok(Some(unit)) => return to_shared(unit).map(Some),
                Ok(None) => return Ok(None),
                Err(CodecError::NeedMoreData) if !self.ended => self.pull()?,
                Err(error) => return Err(read_codec(error)),
            }
        }
    }

    /// Lands on the frame holding `to.time`. Every frame is a random-access
    /// point, so the landing has no preroll of its own; the caller primes
    /// the decoder by asking for an earlier time.
    fn seek(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if to.track != 0 {
            return Err(ReadError::Seek(SeekError::UnknownTrack));
        }
        let total = i64::try_from(self.samples).unwrap_or(i64::MAX);
        let end = to.end.map_or(total, |end| end.min(total));
        let landing = seek_with_index(
            &self.index,
            SeekTo {
                track: 0,
                time: to.time,
                end: Some(end),
            },
        )
        .map_err(ReadError::Seek)?;
        match landing.landing {
            cutlass_container::Landing::Packet {
                time,
                byte_position,
                ..
            } => {
                self.input.seek(byte_position).map_err(read_source)?;
                self.position = byte_position;
                self.demux.restart_at(u64::try_from(time).unwrap_or(0));
            }
            cutlass_container::Landing::End => {
                self.position = self.end;
                self.demux.restart_at(self.samples);
            }
        }
        self.ended = false;
        Ok(landing)
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.input.consumed()
    }
}

/// Accepts a valid ADTS header at the start of `header`, or after an ID3v2
/// tag that ends inside it, when the frame after it is also ADTS with the
/// same configuration. A file that ends inside the window after one frame
/// is accepted on that frame alone.
#[must_use]
pub fn probe(header: &[u8]) -> ProbeResult {
    let examined = header.len();
    let start = match id3v2_size(header) {
        Id3::Absent => 0,
        Id3::NeedBytes => {
            return ProbeResult::NeedBytes {
                have: examined,
                need: ID3_HEADER,
            };
        }
        Id3::Tag(size) => match usize::try_from(size) {
            Ok(size) if size < examined => size,
            _ => return ProbeResult::Refuse { examined },
        },
    };
    let body = &header[start..];
    let first = match frame_at(body) {
        Ok(Some(first)) => first,
        Ok(None) => {
            return ProbeResult::NeedBytes {
                have: examined,
                need: start + 7,
            };
        }
        Err(()) => return ProbeResult::Refuse { examined },
    };
    let next = first.frame_len;
    if next == body.len() {
        return ProbeResult::Accept {
            family: ContainerId::Adts,
            examined,
        };
    }
    match frame_at(body.get(next..).unwrap_or(&[])) {
        Ok(Some(second)) if same_stream(&first, &second) => ProbeResult::Accept {
            family: ContainerId::Adts,
            examined,
        },
        Ok(Some(_)) | Err(()) => ProbeResult::Refuse { examined },
        Ok(None) => ProbeResult::NeedBytes {
            have: examined,
            need: start + next + 7,
        },
    }
}

/// A header this crate plays: AAC-LC, a rate in the table, a channel
/// configuration from 1 to 7. `Ok(None)` wants more bytes.
fn frame_at(data: &[u8]) -> Result<Option<Header>, ()> {
    match sniff(data) {
        Sniff::NeedMore => Ok(None),
        Sniff::Skip => Err(()),
        Sniff::Header(header) => {
            if header.profile != 1
                || rate_of(header.rate_index, header.mpeg2).is_err()
                || layout_of(header.channels).is_err()
            {
                Err(())
            } else {
                Ok(Some(header))
            }
        }
    }
}

fn same_stream(first: &Header, second: &Header) -> bool {
    first.mpeg2 == second.mpeg2
        && first.profile == second.profile
        && first.rate_index == second.rate_index
        && first.channels == second.channels
}

enum Id3 {
    Absent,
    NeedBytes,
    /// The whole tag, header and footer included.
    Tag(u64),
}

/// The size of an ID3v2 tag at the start of `data`. A header with a bad
/// version or a size byte with its top bit set is not a tag.
fn id3v2_size(data: &[u8]) -> Id3 {
    if !data.starts_with(b"ID3") {
        if data.len() < 3 && b"ID3".starts_with(data) {
            return Id3::NeedBytes;
        }
        return Id3::Absent;
    }
    let Some(head) = data.get(..ID3_HEADER) else {
        return Id3::NeedBytes;
    };
    if head[3] == 0xFF || head[4] == 0xFF || head[6..10].iter().any(|byte| byte & 0x80 != 0) {
        return Id3::Absent;
    }
    let size = head[6..10]
        .iter()
        .fold(0u64, |size, byte| (size << 7) | u64::from(*byte));
    let footer = if head[5] & ID3_FOOTER != 0 { 10 } else { 0 };
    Id3::Tag(size + 10 + footer)
}

/// Where the first frame may start: after an ID3v2 tag, or at zero.
fn id3v2_end<R: Read + Seek>(input: &mut Input<R>, length: u64) -> Result<u64, OpenError> {
    if length < 3 {
        return Ok(0);
    }
    let mut head = [0u8; ID3_HEADER];
    let take = usize::try_from(length.min(10)).unwrap_or(ID3_HEADER);
    input
        .read_exact(&mut head[..take])
        .map_err(OpenError::from_header)?;
    match id3v2_size(&head[..take]) {
        Id3::Tag(size) if size < length => Ok(size),
        Id3::Tag(_) => Err(OpenError::Unrecognized {
            family: ContainerId::Adts,
        }),
        Id3::Absent | Id3::NeedBytes => Ok(0),
    }
}

/// What one walk over the file found.
struct Frames {
    index: Vec<IndexEntry>,
    first: u64,
    end: u64,
    samples: u64,
    info: StreamInfo,
}

/// Walks every frame from `start` to the end of the file.
fn walk<R: Read>(input: &mut Input<R>, start: u64, limits: &Limits) -> Result<Frames, OpenError> {
    let mut window: Vec<u8> = Vec::with_capacity(CHUNK * 2);
    let mut base = start;
    let mut at = 0usize;
    let mut eof = false;
    let mut skipped = 0usize;
    let mut index = Vec::new();
    let mut samples = 0u64;
    let mut ordinal = 0u32;
    let mut config: Option<(u32, ChannelLayout, bool)> = None;
    let mut end = start;
    loop {
        let header = match sniff(&window[at..]) {
            Sniff::NeedMore => None,
            Sniff::Skip => {
                at += 1;
                skipped += 1;
                if skipped > MAX_RESYNC {
                    return Err(codec(CodecError::Limit { what: "resync" }));
                }
                continue;
            }
            Sniff::Header(header) => Some(header),
        };
        let whole = header
            .as_ref()
            .is_some_and(|header| window.len() - at >= header.frame_len);
        let Some(header) = header.filter(|_| whole) else {
            if eof {
                break;
            }
            window.drain(..at);
            base = base.saturating_add(u64::try_from(at).unwrap_or(0));
            at = 0;
            let filled = window.len();
            window.resize(filled + CHUNK, 0);
            let n = input
                .read(&mut window[filled..])
                .map_err(OpenError::from_header)?;
            window.truncate(filled + n);
            eof = n == 0;
            continue;
        };
        let (rate, layout) = check(&header, config.as_ref())?;
        config = Some((rate, layout, header.mpeg2));
        if u64::try_from(index.len()).unwrap_or(u64::MAX) >= u64::from(limits.max_index_entries) {
            return Err(OpenError::HostileSize {
                declared: u64::try_from(index.len())
                    .unwrap_or(u64::MAX)
                    .saturating_add(1),
                limit: u64::from(limits.max_index_entries),
            });
        }
        let position = base.saturating_add(u64::try_from(at).unwrap_or(0));
        index.push(IndexEntry {
            track: 0,
            time: i64::try_from(samples).unwrap_or(i64::MAX),
            ordinal,
            keyframe: true,
            byte_position: position,
        });
        let blocks = u32::from(header.blocks) + 1;
        ordinal = ordinal.saturating_add(blocks);
        samples = samples.saturating_add(BLOCK * u64::from(blocks));
        at += header.frame_len;
        end = position.saturating_add(u64::try_from(header.frame_len).unwrap_or(0));
        skipped = 0;
    }
    let (Some((rate, layout, _)), Some(first)) = (config, index.first()) else {
        return Err(OpenError::Unrecognized {
            family: ContainerId::Adts,
        });
    };
    let first = first.byte_position;
    Ok(Frames {
        index,
        first,
        end,
        samples,
        info: StreamInfo {
            index: 0,
            timescale: rate,
            params: CodecParams::audio(CodecId::Aac, rate, layout),
        },
    })
}

/// The checks [`AdtsDemuxer`] makes before it accepts a frame.
fn check(
    header: &Header,
    config: Option<&(u32, ChannelLayout, bool)>,
) -> Result<(u32, ChannelLayout), OpenError> {
    if header.blocks != 0 && header.protection_absent {
        return Err(codec(CodecError::Unsupported {
            what: "unpositioned raw data blocks",
        }));
    }
    if header.profile != 1 {
        return Err(codec(named_profile(header.profile)));
    }
    let rate = rate_of(header.rate_index, header.mpeg2).map_err(codec)?;
    let layout = layout_of(header.channels).map_err(codec)?;
    if let Some((previous_rate, previous_layout, mpeg2)) = config
        && (*previous_rate != rate || *previous_layout != layout || *mpeg2 != header.mpeg2)
    {
        return Err(codec(CodecError::Unsupported {
            what: "configuration change",
        }));
    }
    Ok((rate, layout))
}

fn codec(error: CodecError) -> OpenError {
    OpenError::Source {
        message: error.to_string(),
    }
}

impl AdtsDemuxer {
    /// Drops buffered bytes and packets and numbers the next raw block from
    /// `samples`, for a reader that has moved to the first byte of a frame.
    /// The stream and its configuration stay.
    fn restart_at(&mut self, samples: u64) {
        self.buf.clear();
        self.ready.clear();
        self.ended = false;
        self.skipped = 0;
        self.samples = samples;
    }
}

#[cfg(test)]
mod tests {
    use super::{AdtsSeekFile, probe};
    use crate::{AdtsMuxer, Muxer};
    use cutlass_codec::access::{ChannelLayout, CodecParams, Packet as Unit, SideData};
    use cutlass_codec::{CodecId, ContainerId, Packet};
    use cutlass_container::{Demux, Landing, OpenError, ProbeResult, ReadError, SeekTo};
    use std::io::Cursor;

    const RATE: u32 = 48_000;

    /// `count` frames. Frame `n` carries `n + 1` bytes of the value `n`, so a
    /// packet names the frame it came from.
    fn frames(count: u8, layout: ChannelLayout) -> Vec<u8> {
        let mut mux = AdtsMuxer::new();
        mux.add_stream(CodecParams::audio(CodecId::Aac, RATE, layout))
            .unwrap();
        for n in 0..count {
            let data = vec![n; usize::from(n) + 1];
            mux.write_packet(&Unit::audio(data, i64::from(n) * 1024, 1024, RATE))
                .unwrap();
        }
        mux.finish().unwrap()
    }

    fn open(bytes: Vec<u8>) -> Result<AdtsSeekFile<Cursor<Vec<u8>>>, OpenError> {
        let length = u64::try_from(bytes.len()).unwrap();
        AdtsSeekFile::open(Cursor::new(bytes), length)
    }

    fn read_all(file: &mut AdtsSeekFile<Cursor<Vec<u8>>>) -> Vec<Packet> {
        let mut out = Vec::new();
        while let Some(packet) = file.read_packet().unwrap() {
            out.push(packet);
        }
        out
    }

    fn at(time: i64) -> SeekTo {
        SeekTo {
            track: 0,
            time,
            end: None,
        }
    }

    fn id3v2(body: usize) -> Vec<u8> {
        let size = u32::try_from(body).unwrap();
        let mut tag = b"ID3\x04\x00\x00".to_vec();
        for shift in [21, 14, 7, 0] {
            tag.push(u8::try_from((size >> shift) & 0x7F).unwrap());
        }
        // A false sync inside the tag body.
        let mut payload = vec![0u8; body];
        payload[..2].copy_from_slice(&[0xFF, 0xF1]);
        tag.extend(payload);
        tag
    }

    #[test]
    fn opening_indexes_every_frame_and_reads_them_in_order() {
        let mut file = open(frames(40, ChannelLayout::mono())).unwrap();
        assert_eq!(file.frames(), 40);
        assert_eq!(file.samples(), 40 * 1024);
        assert_eq!(file.family(), ContainerId::Adts);
        assert_eq!(file.streams()[0].timescale().ticks_per_second(), RATE);
        let packets = read_all(&mut file);
        assert_eq!(packets.len(), 40);
        for (n, packet) in packets.iter().enumerate() {
            let n = u8::try_from(n).unwrap();
            assert_eq!(packet.bytes(), vec![n; usize::from(n) + 1].as_slice());
            assert_eq!(packet.timestamp().presentation(), Some(i64::from(n) * 1024));
        }
    }

    #[test]
    fn a_seek_lands_on_the_frame_that_holds_the_time() {
        let mut file = open(frames(40, ChannelLayout::stereo())).unwrap();
        let landing = file.seek(at(17 * 1024 + 300)).unwrap();
        assert!(landing.index_used);
        assert_eq!(landing.payload_scans, 0);
        assert!(matches!(
            landing.landing,
            Landing::Packet { time, .. } if time == 17 * 1024
        ));
        let next = file.read_packet().unwrap().unwrap();
        assert_eq!(next.bytes(), vec![17; 18].as_slice());
        assert_eq!(next.timestamp().presentation(), Some(17 * 1024));
        // Backwards, after reading on.
        let _ = file.read_packet().unwrap();
        file.seek(at(2 * 1024)).unwrap();
        let next = file.read_packet().unwrap().unwrap();
        assert_eq!(next.bytes(), vec![2; 3].as_slice());
        assert_eq!(next.timestamp().presentation(), Some(2 * 1024));
        assert_eq!(read_all(&mut file).len(), 37);
    }

    #[test]
    fn a_seek_reads_only_from_the_landing() {
        let mut file = open(frames(200, ChannelLayout::mono())).unwrap();
        let opened = file.consumed();
        let landing = file.seek(at(150 * 1024)).unwrap();
        // A binary search, not a walk.
        assert!(landing.entries_visited <= 10, "{landing:?}");
        let _ = file.read_packet().unwrap().unwrap();
        let tail: u64 = (150..200u64).map(|n| 7 + n + 1).sum();
        assert!(file.consumed() - opened <= tail);
    }

    #[test]
    fn a_seek_past_the_end_reads_nothing_and_the_start_reads_again() {
        let mut file = open(frames(10, ChannelLayout::mono())).unwrap();
        let landing = file.seek(at(10 * 1024)).unwrap();
        assert_eq!(landing.landing, Landing::End);
        assert!(file.read_packet().unwrap().is_none());
        file.seek(at(-5)).unwrap();
        assert_eq!(read_all(&mut file).len(), 10);
    }

    #[test]
    fn a_seek_on_another_track_is_refused() {
        let mut file = open(frames(4, ChannelLayout::mono())).unwrap();
        let error = file
            .seek(SeekTo {
                track: 1,
                time: 0,
                end: None,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            ReadError::Seek(cutlass_container::SeekError::UnknownTrack)
        ));
    }

    #[test]
    fn a_leading_id3v2_tag_is_skipped_whole() {
        let mut bytes = id3v2(300);
        bytes.extend(frames(5, ChannelLayout::mono()));
        let mut file = open(bytes.clone()).unwrap();
        assert_eq!(file.frames(), 5);
        assert_eq!(read_all(&mut file).len(), 5);
        assert!(matches!(probe(&bytes), ProbeResult::Accept { .. }));
    }

    #[test]
    fn bytes_after_the_last_whole_frame_are_left_out() {
        let whole = frames(6, ChannelLayout::mono());
        let mut cut = whole.clone();
        cut.extend(&frames(7, ChannelLayout::mono())[whole.len()..whole.len() + 4]);
        let mut file = open(cut).unwrap();
        assert_eq!(file.frames(), 6);
        assert_eq!(read_all(&mut file).len(), 6);

        let mut tagged = whole;
        tagged.extend(b"TAG");
        tagged.extend([0u8; 125]);
        let mut file = open(tagged).unwrap();
        assert_eq!(file.samples(), 6 * 1024);
        assert_eq!(read_all(&mut file).len(), 6);
    }

    #[test]
    fn a_configuration_change_refuses_the_file() {
        let mut bytes = frames(3, ChannelLayout::mono());
        bytes.extend(frames(3, ChannelLayout::stereo()));
        let error = open(bytes).err().unwrap();
        assert!(
            matches!(&error, OpenError::Source { message } if message.contains("configuration")),
            "{error:?}"
        );
    }

    #[test]
    fn a_file_with_no_frame_is_not_adts() {
        for bytes in [Vec::new(), vec![0u8; 5000], id3v2(20)] {
            let error = open(bytes).err().unwrap();
            assert!(
                matches!(
                    error,
                    OpenError::Unrecognized {
                        family: ContainerId::Adts
                    }
                ),
                "{error:?}"
            );
        }
    }

    #[test]
    fn a_tag_longer_than_the_file_is_refused() {
        let mut bytes = id3v2(10);
        bytes[9] = 0x7F;
        assert!(open(bytes).is_err());
    }

    #[test]
    fn several_raw_blocks_in_a_frame_are_counted() {
        let mut mux = AdtsMuxer::new();
        mux.add_stream(CodecParams::audio(
            CodecId::Aac,
            RATE,
            ChannelLayout::mono(),
        ))
        .unwrap();
        let blocks: Vec<Unit> = (0..3)
            .map(|n| {
                let mut unit = Unit::audio(vec![n + 1; 4], i64::from(n) * 1024, 1024, RATE);
                unit.side.push(SideData::Protection {
                    crc: u16::from(n),
                    header: None,
                });
                unit
            })
            .collect();
        mux.write_blocks(&blocks).unwrap();
        mux.write_packet(&Unit::audio(vec![9; 4], 3 * 1024, 1024, RATE))
            .unwrap();
        let mut file = open(mux.finish().unwrap()).unwrap();
        assert_eq!(file.frames(), 2);
        assert_eq!(file.samples(), 4 * 1024);
        file.seek(at(3 * 1024 + 1)).unwrap();
        let next = file.read_packet().unwrap().unwrap();
        assert_eq!(next.bytes(), &[9; 4]);
        assert_eq!(next.timestamp().presentation(), Some(3 * 1024));
    }

    #[test]
    fn a_cancelled_file_refuses_reads_and_seeks() {
        let mut file = open(frames(3, ChannelLayout::mono())).unwrap();
        file.cancel();
        assert!(matches!(file.read_packet(), Err(ReadError::Cancelled)));
        assert!(matches!(file.seek(at(0)), Err(ReadError::Cancelled)));
    }

    #[test]
    fn the_probe_wants_two_matching_frames() {
        let two = frames(2, ChannelLayout::mono());
        assert_eq!(
            probe(&two),
            ProbeResult::Accept {
                family: ContainerId::Adts,
                examined: two.len()
            }
        );
        let one = frames(1, ChannelLayout::mono());
        assert!(matches!(probe(&one), ProbeResult::Accept { .. }));
        // The first frame is 8 bytes; the second has started but its header has not.
        assert!(matches!(probe(&two[..12]), ProbeResult::NeedBytes { .. }));
        assert!(matches!(probe(&two[..5]), ProbeResult::NeedBytes { .. }));
        assert!(matches!(probe(b"ID"), ProbeResult::NeedBytes { .. }));
        // A sync pattern with junk after the first frame.
        let mut junk = frames(1, ChannelLayout::mono());
        junk.extend([0x12u8; 32]);
        assert!(matches!(probe(&junk), ProbeResult::Refuse { .. }));
        // An MP3 frame header: layer bits are not zero.
        assert!(matches!(
            probe(&[0xFF, 0xFB, 0x90, 0x64, 0, 0, 0, 0]),
            ProbeResult::Refuse { .. }
        ));
        assert!(matches!(
            probe(b"RIFF\0\0\0\0WAVE"),
            ProbeResult::Refuse { .. }
        ));
    }
}
