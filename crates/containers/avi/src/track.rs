//! One stream's clock and its chunk table.
//!
//! The file timescale is `dwRate / g` ticks per second and one AVI unit is
//! `dwScale / g` ticks, where `g` is the greatest common divisor of the two.
//! That is the stream's own `dwScale/dwRate` clock with nothing rounded, and
//! it is the time base FFmpeg reports for the same stream.
//!
//! How many units one chunk covers:
//!
//! - Video, and any stream with `dwSampleSize` zero and no `nBlockAlign`:
//!   one unit per chunk. An empty chunk is a dropped frame: it takes its
//!   unit, is not returned as a packet, and lengthens the video packet
//!   before it, which stays on screen through it. (FFmpeg's demuxer gives
//!   that packet one unit; its generic layer later guesses a duration from
//!   the frame rate.)
//! - Audio with `dwSampleSize` zero (variable bitrate): `ceil(len /
//!   nBlockAlign)` units, as DirectShow, FFmpeg, and GStreamer count it.
//! - Audio with `dwSampleSize` set (constant bitrate, PCM): the chunk's
//!   bytes divided by the sample size, counted over the whole stream so a
//!   partial block carries into the next chunk.
//!
//! Three header fixes follow FFmpeg, so the same file gets the same times:
//! a `dwSampleSize` that differs from a nonzero `nBlockAlign` is replaced
//! by `nBlockAlign`; AAC with `nBlockAlign` and `dwSampleSize` both 1024 or
//! 4096, and MP3 with both 1152, are read as variable bitrate; and an AAC or
//! FLAC `nBlockAlign` of four or less counts one unit per chunk.
//! `dwStart` is added to every time. A start above one hour is ignored, as
//! FFmpeg ignores it.

use cutlass_codec::CodecId;
use cutlass_container::{IndexEntry, OpenError};

use crate::format::StreamKind;
use crate::header::{MainHeader, StreamHeader};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Clock {
    pub timescale: u32,
    /// Ticks in one AVI unit.
    pub unit: i64,
    /// Ticks before the first chunk.
    pub start: i64,
    /// Bytes per unit for constant-bitrate audio; zero otherwise.
    pub sample_size: u32,
    /// `nBlockAlign` for the variable-bitrate chunk duration; zero for one unit per chunk.
    pub block_align: u32,
    pub video: bool,
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let rest = a % b;
        a = b;
        b = rest;
    }
    a
}

impl Clock {
    pub(crate) fn new(
        header: &StreamHeader,
        main: &MainHeader,
        kind: StreamKind,
        codec: Option<CodecId>,
        block_align: u32,
    ) -> Self {
        let (mut scale, mut rate) = (u64::from(header.scale), u64::from(header.rate));
        if scale == 0 || rate == 0 {
            if main.micro_sec_per_frame > 0 {
                scale = u64::from(main.micro_sec_per_frame);
                rate = 1_000_000;
            } else {
                scale = 1;
                rate = 25;
            }
        }
        let common = gcd(scale, rate).max(1);
        let timescale = u32::try_from(rate / common).unwrap_or(u32::MAX);
        let unit = i64::try_from(scale / common).unwrap_or(i64::MAX);
        let video = kind == StreamKind::Video;
        let mut sample_size = if video { 0 } else { header.sample_size };
        let mut align = if kind == StreamKind::Audio {
            block_align
        } else {
            0
        };
        if kind == StreamKind::Audio {
            if sample_size != 0 && align != 0 && sample_size != align {
                sample_size = align;
            }
            if matches!(codec, Some(CodecId::Aac | CodecId::Flac)) && align <= 4 {
                align = 0;
            }
            let vbr_tagged = matches!(
                (codec, align, sample_size),
                (Some(CodecId::Aac), 1024, 1024)
                    | (Some(CodecId::Aac), 4096, 4096)
                    | (Some(CodecId::Mp3), 1152, 1152)
            );
            if vbr_tagged {
                sample_size = 0;
            }
        }
        let start_units = u64::from(header.start);
        let hour = 3_600u64.saturating_mul(rate) / scale.max(1);
        let start_units = if start_units > hour { 0 } else { start_units };
        let start = i64::try_from(start_units)
            .ok()
            .and_then(|units| units.checked_mul(unit))
            .unwrap_or(0);
        Self {
            timescale,
            unit,
            start,
            sample_size,
            block_align: align,
            video,
        }
    }
}

/// One indexed chunk beyond what [`IndexEntry`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub size: u32,
    /// Duration in ticks.
    pub ticks: u32,
}

pub(crate) struct Track {
    pub number: u32,
    pub kind: StreamKind,
    pub codec: Option<CodecId>,
    pub clock: Clock,
    pub intra: bool,
    pub presentation: bool,
    pub super_index: Option<Vec<u8>>,
    pub entries: Vec<IndexEntry>,
    pub spans: Vec<Span>,
    bytes: u64,
    units: u64,
    last_position: Option<u64>,
}

impl Track {
    pub(crate) fn new(
        number: u32,
        kind: StreamKind,
        codec: Option<CodecId>,
        clock: Clock,
        intra: bool,
        presentation: bool,
        super_index: Option<Vec<u8>>,
    ) -> Self {
        Self {
            number,
            kind,
            codec,
            clock,
            intra,
            presentation,
            super_index,
            entries: Vec::new(),
            spans: Vec::new(),
            bytes: 0,
            units: 0,
            last_position: None,
        }
    }

    /// Forgets a partly built table before another index source is tried.
    pub(crate) fn reset(&mut self) {
        self.entries = Vec::new();
        self.spans = Vec::new();
        self.bytes = 0;
        self.units = 0;
        self.last_position = None;
    }

    /// Adds the chunk whose payload starts at `data` and is `size` bytes.
    ///
    /// `budget` is the entry count left across every track. A chunk at the
    /// same position as the one before it is listed twice in some indexes
    /// and is taken once.
    pub(crate) fn push(
        &mut self,
        data: u64,
        size: u32,
        key: bool,
        budget: &mut u64,
    ) -> Result<(), OpenError> {
        if self.last_position == Some(data) {
            return Ok(());
        }
        if size > 0 && *budget == 0 {
            return Err(OpenError::HostileSize {
                declared: u64::try_from(self.entries.len()).unwrap_or(u64::MAX) + 1,
                limit: u64::try_from(self.entries.len()).unwrap_or(u64::MAX),
            });
        }
        self.last_position = Some(data);
        let (before, after) = if self.clock.sample_size > 0 {
            let width = u64::from(self.clock.sample_size);
            let before = self.bytes / width;
            self.bytes = self.bytes.saturating_add(u64::from(size));
            (before, self.bytes / width)
        } else {
            let span = if self.clock.video || self.clock.block_align == 0 {
                1
            } else {
                u64::from(size).div_ceil(u64::from(self.clock.block_align))
            };
            let before = self.units;
            self.units = self.units.saturating_add(span);
            (before, self.units)
        };
        if size == 0 {
            if self.kind == StreamKind::Video && !self.spans.is_empty() {
                let more = self.ticks(after)?.saturating_sub(self.ticks(before)?);
                if let Some(span) = self.spans.last_mut() {
                    span.ticks = span
                        .ticks
                        .saturating_add(u32::try_from(more).unwrap_or(u32::MAX));
                }
            }
            return Ok(());
        }
        *budget -= 1;
        let time = self.ticks(before)?;
        let ticks = self.ticks(after)?.saturating_sub(time);
        let ordinal = u32::try_from(self.entries.len()).map_err(|_| too_long())?;
        self.entries.push(IndexEntry {
            track: self.number,
            time,
            ordinal,
            keyframe: key || self.intra || self.kind != StreamKind::Video,
            byte_position: data,
        });
        self.spans.push(Span {
            size,
            ticks: u32::try_from(ticks).unwrap_or(u32::MAX),
        });
        Ok(())
    }

    fn ticks(&self, units: u64) -> Result<i64, OpenError> {
        i64::try_from(units)
            .ok()
            .and_then(|units| units.checked_mul(self.clock.unit))
            .and_then(|ticks| ticks.checked_add(self.clock.start))
            .ok_or_else(too_long)
    }

    /// A video index with no keyframe flag at all marks its first chunk,
    /// as FFmpeg does, so a seek can still land.
    pub(crate) fn finish(&mut self) {
        if !self.entries.iter().any(|entry| entry.keyframe)
            && let Some(first) = self.entries.first_mut()
        {
            first.keyframe = true;
        }
    }

    /// The end of the last chunk, in ticks.
    pub(crate) fn end(&self) -> Option<i64> {
        let last = self.entries.last()?;
        let span = self.spans.last()?;
        last.time.checked_add(i64::from(span.ticks))
    }
}

fn too_long() -> OpenError {
    OpenError::HostileSize {
        declared: u64::MAX,
        limit: u64::try_from(i64::MAX).unwrap_or(u64::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::{Clock, Track};
    use crate::format::StreamKind;
    use crate::header::{MainHeader, StreamHeader};
    use cutlass_codec::CodecId;

    fn header(scale: u32, rate: u32, sample_size: u32) -> StreamHeader {
        StreamHeader {
            scale,
            rate,
            sample_size,
            ..StreamHeader::default()
        }
    }

    fn track(clock: Clock, kind: StreamKind) -> Track {
        Track::new(0, kind, None, clock, false, false, None)
    }

    #[test]
    fn ntsc_video_keeps_its_rational_clock() {
        let main = MainHeader::default();
        let clock = Clock::new(&header(1001, 30_000, 0), &main, StreamKind::Video, None, 0);
        assert_eq!((clock.timescale, clock.unit), (30_000, 1001));
        let mut video = track(clock, StreamKind::Video);
        let mut budget = 10;
        video.push(100, 5, true, &mut budget).unwrap();
        video.push(200, 0, false, &mut budget).unwrap();
        video.push(300, 5, false, &mut budget).unwrap();
        assert_eq!(
            video.entries[1].time, 2002,
            "an empty chunk is a dropped frame"
        );
        assert_eq!(video.spans[1].ticks, 1001);
        assert_eq!(
            video.spans[0].ticks, 2002,
            "the frame before stays on screen"
        );
        assert_eq!(budget, 8);
    }

    #[test]
    fn pcm_counts_blocks_and_carries_a_partial_one() {
        let main = MainHeader::default();
        let clock = Clock::new(
            &header(4, 176_400, 4),
            &main,
            StreamKind::Audio,
            Some(CodecId::Pcm),
            4,
        );
        assert_eq!(
            (clock.timescale, clock.unit, clock.sample_size),
            (44_100, 1, 4)
        );
        let mut audio = track(clock, StreamKind::Audio);
        let mut budget = 10;
        audio.push(0, 6, false, &mut budget).unwrap();
        audio.push(10, 6, false, &mut budget).unwrap();
        assert_eq!(
            (
                audio.entries[1].time,
                audio.spans[0].ticks,
                audio.spans[1].ticks
            ),
            (1, 1, 2)
        );
        assert!(audio.entries.iter().all(|entry| entry.keyframe));
    }

    #[test]
    fn variable_bitrate_audio_counts_ceil_of_len_over_block_align() {
        let main = MainHeader::default();
        let clock = Clock::new(
            &header(1152, 44_100, 0),
            &main,
            StreamKind::Audio,
            Some(CodecId::Mp3),
            1152,
        );
        assert_eq!((clock.timescale, clock.unit), (1225, 32));
        let mut audio = track(clock, StreamKind::Audio);
        let mut budget = 10;
        audio.push(0, 418, false, &mut budget).unwrap();
        audio.push(500, 1200, false, &mut budget).unwrap();
        audio.push(2000, 418, false, &mut budget).unwrap();
        assert_eq!(audio.entries[2].time, 96);
    }

    #[test]
    fn ffmpegs_header_fixes_apply() {
        let main = MainHeader::default();
        let tagged = Clock::new(
            &header(1024, 48_000, 1024),
            &main,
            StreamKind::Audio,
            Some(CodecId::Aac),
            1024,
        );
        assert_eq!(tagged.sample_size, 0);
        let small = Clock::new(
            &header(1024, 48_000, 0),
            &main,
            StreamKind::Audio,
            Some(CodecId::Aac),
            4,
        );
        assert_eq!(small.block_align, 0);
        let mismatch = Clock::new(
            &header(1, 8_000, 2),
            &main,
            StreamKind::Audio,
            Some(CodecId::Pcm),
            1,
        );
        assert_eq!(mismatch.sample_size, 1);
        let zero = Clock::new(
            &header(0, 0, 0),
            &MainHeader {
                micro_sec_per_frame: 40_000,
                ..main
            },
            StreamKind::Video,
            None,
            0,
        );
        assert_eq!((zero.timescale, zero.unit), (25, 1));
        let start = Clock::new(
            &StreamHeader {
                start: 3,
                ..header(1, 25, 0)
            },
            &main,
            StreamKind::Video,
            None,
            0,
        );
        assert_eq!(start.start, 3);
        let crazy = Clock::new(
            &StreamHeader {
                start: 25 * 3_601,
                ..header(1, 25, 0)
            },
            &main,
            StreamKind::Video,
            None,
            0,
        );
        assert_eq!(crazy.start, 0);
    }

    #[test]
    fn the_entry_budget_is_shared_and_enforced() {
        let main = MainHeader::default();
        let clock = Clock::new(&header(1, 25, 0), &main, StreamKind::Video, None, 0);
        let mut video = track(clock, StreamKind::Video);
        let mut budget = 1;
        video.push(8, 1, true, &mut budget).unwrap();
        assert!(video.push(16, 1, true, &mut budget).is_err());
        video.push(8, 1, true, &mut 0).unwrap();
    }

    #[test]
    fn a_video_index_without_keyframes_marks_the_first() {
        let main = MainHeader::default();
        let clock = Clock::new(&header(1, 25, 0), &main, StreamKind::Video, None, 0);
        let mut video = track(clock, StreamKind::Video);
        let mut budget = 4;
        video.push(8, 1, false, &mut budget).unwrap();
        video.push(16, 1, false, &mut budget).unwrap();
        video.finish();
        assert!(video.entries[0].keyframe && !video.entries[1].keyframe);
        assert_eq!(video.end(), Some(2));
    }
}
