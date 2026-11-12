//! Access units of a start-code video stream (H.264, HEVC, MPEG-2).
//!
//! The stream is read as one byte sequence across PES packets. A unit is
//! the bytes from one `00 00 01` to the next. Each unit is classified from
//! its first bytes; an access unit ends where the first unit of the next
//! one starts. A unit that opens an access unit (an access unit delimiter,
//! a parameter set, SEI, a sequence or GOP header) only marks a candidate
//! boundary: the next picture unit decides whether it is a new frame or the
//! second field of the frame in progress, which joins it.
//!
//! The common case, one access unit per PES packet starting with a start
//! code, moves the PES payload into the packet without a copy. Other
//! layouts append and split.

use super::bits::{find_start_code, starts_with_start_code};
use super::{Facts, FrameFault, Pes, Unit};

/// How a unit relates to the access unit in progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// The head is too short to decide.
    NeedMore,
    /// Opens the next access unit when it follows a picture. `whole` asks
    /// for the complete unit (a parameter set).
    Opens { key: bool, whole: bool },
    /// A unit of a coded picture. `first` is set on the first unit of a
    /// picture; `pairs` when that picture is the second field of the one in
    /// progress.
    Picture { first: bool, pairs: bool, key: bool },
    /// Belongs to the access unit in progress.
    Inside { whole: bool },
}

impl Class {
    const fn wants_whole(self) -> bool {
        matches!(
            self,
            Self::Opens { whole: true, .. } | Self::Inside { whole: true }
        )
    }
}

/// One codec's unit syntax.
pub trait Syntax: Default {
    /// Bytes after the start code a unit is classified from.
    const HEAD: usize;
    /// Classifies the unit whose bytes after the start code begin with
    /// `head`, at most [`Self::HEAD`] of them. `complete` means `head` is
    /// the whole unit; when it is not, a head too short to decide on is
    /// [`Class::NeedMore`], and nothing is changed by that call.
    fn classify(&mut self, head: &[u8], complete: bool) -> Class;
    /// The whole unit after its start code, of a unit classified as
    /// wanting it.
    fn whole(&mut self, unit: &[u8]);
    fn facts(&self) -> Option<Facts>;
    /// Forgets the picture in progress. Parameter sets stay.
    fn reset(&mut self);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Mark {
    at: i64,
    pts: Option<i64>,
    dts: Option<i64>,
    position: u64,
}

pub struct VideoFramer<S: Syntax> {
    syntax: S,
    cur: Vec<u8>,
    marks: Vec<Mark>,
    /// `cur` starts at a start code; until then bytes are dropped.
    aligned: bool,
    /// Start-code position of the last unit found; its end is not known.
    last: Option<usize>,
    last_classified: bool,
    last_whole: bool,
    /// The start-code search resumes here.
    scan: usize,
    /// Where the next access unit starts, when an opening unit followed a
    /// picture.
    candidate: Option<usize>,
    /// An access unit to emit, found while classifying.
    pending: Option<usize>,
    has_picture: bool,
    key_cur: bool,
    key_next: bool,
    /// The key flag of the access unit after the one being emitted.
    key_after: Option<bool>,
    last_position: u64,
    max_unit: usize,
    /// Bytes dropped: before the first start code, or of an oversized unit.
    pub skipped: u64,
    /// Access units moved out of a PES payload without a copy.
    pub moved: u64,
}

impl<S: Syntax> VideoFramer<S> {
    #[must_use]
    pub fn new(max_unit: usize) -> Self {
        Self {
            syntax: S::default(),
            cur: Vec::new(),
            marks: Vec::new(),
            aligned: false,
            last: None,
            last_classified: false,
            last_whole: false,
            scan: 0,
            candidate: None,
            pending: None,
            has_picture: false,
            key_cur: false,
            key_next: false,
            key_after: None,
            last_position: 0,
            max_unit,
            skipped: 0,
            moved: 0,
        }
    }

    pub fn facts(&self) -> Option<Facts> {
        self.syntax.facts()
    }

    pub fn buffered(&self) -> usize {
        self.cur.len()
    }

    pub fn reset(&mut self) {
        self.cur = Vec::new();
        self.marks.clear();
        self.aligned = false;
        self.last = None;
        self.last_classified = false;
        self.last_whole = false;
        self.scan = 0;
        self.candidate = None;
        self.pending = None;
        self.has_picture = false;
        self.key_cur = false;
        self.key_next = false;
        self.key_after = None;
        self.syntax.reset();
    }

    pub fn push(&mut self, pes: Pes, out: &mut Vec<Unit>) -> Result<(), FrameFault> {
        let incoming = pes.payload;
        let starts = starts_with_start_code(&incoming);
        if starts && !self.cur.is_empty() {
            self.close_last(out);
        }
        self.marks.push(Mark {
            at: self.cur.len() as i64,
            pts: pes.pts,
            dts: pes.dts,
            position: pes.position,
        });
        if self.cur.is_empty() {
            self.cur = incoming;
            self.last = None;
            self.last_classified = false;
            self.last_whole = false;
            self.scan = 0;
        } else if starts {
            self.fast(incoming, out);
        } else {
            self.cur.extend_from_slice(&incoming);
        }
        self.scan_units(out);
        if self.cur.len() > self.max_unit {
            let declared = self.cur.len() as u64;
            self.skipped = self.skipped.saturating_add(declared);
            self.reset();
            return Err(FrameFault::Oversize {
                declared,
                limit: self.max_unit as u64,
            });
        }
        Ok(())
    }

    pub fn flush(&mut self, out: &mut Vec<Unit>) {
        self.scan_units(out);
        if !self.cur.is_empty() {
            self.close_last(out);
        }
        if self.has_picture {
            let end = self.cur.len();
            self.emit(end, out);
        } else {
            self.skipped = self.skipped.saturating_add(self.cur.len() as u64);
        }
        self.reset();
    }

    /// Ends the last unit of `cur` at the end of `cur`: the next payload
    /// starts with a start code, or the stream ended.
    fn close_last(&mut self, out: &mut Vec<Unit>) {
        let Some(start) = self.last.take() else {
            return;
        };
        let end = self.cur.len();
        self.finish_unit(start, end);
        if let Some(at) = self.pending.take() {
            self.emit(at, out);
        }
        self.last_classified = false;
        self.last_whole = false;
        self.scan = self.cur.len();
    }

    /// Classifies the units at the head of `incoming` up to its first
    /// picture unit. When that picture starts a new access unit exactly at
    /// the start of `incoming`, the access unit in `cur` is emitted as it
    /// is and `incoming` becomes `cur` without a copy.
    fn fast(&mut self, incoming: Vec<u8>, out: &mut Vec<Unit>) {
        let base = self.cur.len();
        // `starts_with_start_code` held, so this is 0 or 1.
        let mut unit = find_start_code(&incoming, 0).unwrap_or(0);
        loop {
            let next = find_start_code(&incoming, unit + 3);
            let end = next.unwrap_or(incoming.len());
            let head = &incoming[(unit + 3).min(end)..end.min(unit + 3 + S::HEAD)];
            let class = self.syntax.classify(head, next.is_some());
            if class == Class::NeedMore {
                self.adopt_tail(base, unit, (false, false), &incoming);
                return;
            }
            let whole = class.wants_whole();
            if whole && next.is_some() {
                self.syntax.whole(&incoming[unit + 3..end]);
            }
            let open_whole = whole && next.is_none();
            let split = base + split_point(&incoming, unit);
            let picture = matches!(class, Class::Picture { .. });
            match self.apply(class, split) {
                Some(at) if at == base => {
                    self.emit(base, out);
                    self.cur = incoming;
                    self.moved += 1;
                    self.last = Some(unit);
                    self.last_classified = true;
                    self.last_whole = open_whole;
                    self.scan = unit + 3;
                    return;
                }
                Some(at) => {
                    self.cur.extend_from_slice(&incoming);
                    self.emit(at, out);
                    let start = base + unit - at;
                    self.last = Some(start);
                    self.last_classified = true;
                    self.last_whole = open_whole;
                    self.scan = start + 3;
                    return;
                }
                None => {}
            }
            let Some(after) = next.filter(|_| !picture) else {
                self.adopt_tail(base, unit, (true, open_whole), &incoming);
                return;
            };
            unit = after;
        }
    }

    /// Appends `incoming`, whose unit at `unit` is the last one found, with
    /// its (classified, whole) state.
    fn adopt_tail(&mut self, base: usize, unit: usize, state: (bool, bool), incoming: &[u8]) {
        self.cur.extend_from_slice(incoming);
        self.last = Some(base + unit);
        self.last_classified = state.0;
        self.last_whole = state.1;
        self.scan = base + unit + 3;
    }

    /// Walks the units of `cur` from `scan`.
    fn scan_units(&mut self, out: &mut Vec<Unit>) {
        loop {
            let found = find_start_code(&self.cur, self.scan);
            let Some(start) = self.last else {
                let Some(at) = found else {
                    if !self.aligned {
                        // `00 00 00` may still be the start of `00 00 00 01`.
                        let drop = self.cur.len().saturating_sub(3);
                        self.drop_front(drop);
                    }
                    self.scan = self.cur.len().saturating_sub(2).max(self.scan);
                    return;
                };
                let at = if self.aligned {
                    at
                } else {
                    let from = split_point(&self.cur, at);
                    self.drop_front(from);
                    self.aligned = true;
                    at - from
                };
                self.last = Some(at);
                self.last_classified = false;
                self.last_whole = false;
                self.scan = at + 3;
                continue;
            };
            if let Some(next) = found {
                self.finish_unit(start, next);
                let next = match self.pending.take() {
                    Some(at) => {
                        self.emit(at, out);
                        next - at
                    }
                    None => next,
                };
                self.last = Some(next);
                self.last_classified = false;
                self.last_whole = false;
                self.scan = next + 3;
            } else {
                if !self.last_classified {
                    let end = self.cur.len();
                    self.classify_at(start, end, false);
                    if let Some(at) = self.pending.take() {
                        self.emit(at, out);
                    }
                }
                self.scan = self.cur.len().saturating_sub(2).max(self.scan);
                return;
            }
        }
    }

    fn drop_front(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        self.cur.drain(..count);
        self.skipped = self.skipped.saturating_add(count as u64);
        for mark in &mut self.marks {
            mark.at -= count as i64;
        }
        self.scan = self.scan.saturating_sub(count);
    }

    /// The unit `[start, end)` of `cur` is complete.
    fn finish_unit(&mut self, start: usize, end: usize) {
        if !self.last_classified {
            self.classify_at(start, end, true);
        }
        if self.last_whole {
            let unit = self.cur[(start + 3).min(end)..end].to_vec();
            self.syntax.whole(&unit);
        }
        self.last_classified = true;
        self.last_whole = false;
    }

    fn classify_at(&mut self, start: usize, end: usize, complete: bool) {
        let from = (start + 3).min(end);
        let head = &self.cur[from..end.min(start + 3 + S::HEAD)];
        let mut class = self.syntax.classify(head, complete);
        if class == Class::NeedMore {
            if !complete {
                return;
            }
            // A complete unit that still cannot be read stays in the access
            // unit in progress.
            class = Class::Inside { whole: false };
        }
        self.last_classified = true;
        self.last_whole = class.wants_whole();
        // A PES packet that starts at this start code keeps its leading
        // zero byte, if any, in the packet before.
        let split = if self.marks.iter().any(|mark| mark.at == start as i64) {
            start
        } else {
            split_point(&self.cur, start)
        };
        self.pending = self.apply(class, split);
    }

    /// Updates the access-unit state for one unit. Returns where an access
    /// unit ends, when this unit ends one.
    fn apply(&mut self, class: Class, split: usize) -> Option<usize> {
        match class {
            Class::NeedMore | Class::Inside { .. } => None,
            Class::Opens { key, .. } => {
                if self.has_picture && self.candidate.is_none() {
                    self.candidate = Some(split);
                }
                if self.candidate.is_some() {
                    self.key_next |= key;
                } else {
                    self.key_cur |= key;
                }
                None
            }
            Class::Picture { first, pairs, key } => {
                if !self.has_picture {
                    self.has_picture = true;
                    self.key_cur |= self.key_next | key;
                    self.key_next = false;
                    self.candidate = None;
                    return None;
                }
                let new_frame = first && !pairs;
                if let Some(at) = self.candidate.take() {
                    if new_frame {
                        self.key_after = Some(self.key_next | key);
                        self.key_next = false;
                        return Some(at);
                    }
                    // The second field: what opened it joins the frame.
                    self.key_cur |= self.key_next | key;
                    self.key_next = false;
                    return None;
                }
                if new_frame {
                    self.key_after = Some(key);
                    return Some(split);
                }
                self.key_cur |= key;
                None
            }
        }
    }

    /// Emits `cur[..at]` as one access unit and keeps the rest.
    fn emit(&mut self, at: usize, out: &mut Vec<Unit>) {
        let bytes = if at >= self.cur.len() {
            std::mem::take(&mut self.cur)
        } else {
            let tail = self.cur.split_off(at);
            std::mem::replace(&mut self.cur, tail)
        };
        let shift = bytes.len();
        // The access unit takes the timestamp of the last PES packet that
        // started at or before it, unless an earlier unit took it.
        let taken = self.marks.iter().rev().find(|mark| mark.at <= 0).copied();
        self.marks.retain(|mark| mark.at > 0);
        for mark in &mut self.marks {
            mark.at -= shift as i64;
        }
        if let Some(mark) = taken {
            self.last_position = mark.position;
        }
        let keyframe = self.key_cur;
        self.key_cur = self.key_after.take().unwrap_or(false);
        self.candidate = None;
        if let Some(last) = self.last.as_mut() {
            *last = last.saturating_sub(shift);
        }
        self.scan = self.scan.saturating_sub(shift);
        if bytes.is_empty() {
            return;
        }
        let (pts, dts) = match taken {
            Some(mark) => (mark.pts, mark.dts.or(mark.pts)),
            None => (None, None),
        };
        out.push(Unit {
            bytes,
            pts,
            dts,
            duration: None,
            keyframe,
            position: self.last_position,
        });
    }
}

/// Where an access unit starting with the unit at `start` begins: one byte
/// earlier when the start code has a leading zero byte.
fn split_point(bytes: &[u8], start: usize) -> usize {
    if start > 0 && bytes[start - 1] == 0 {
        start - 1
    } else {
        start
    }
}

#[cfg(test)]
mod tests {
    use super::VideoFramer;
    use crate::es::h264::H264;
    use crate::es::{Pes, Unit};

    /// Writes bits MSB first, with Exp-Golomb codes.
    #[derive(Default)]
    struct Writer {
        bytes: Vec<u8>,
        bits: usize,
    }

    impl Writer {
        fn bit(&mut self, bit: bool) {
            if self.bits.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if bit {
                *self.bytes.last_mut().unwrap() |= 0x80 >> (self.bits % 8);
            }
            self.bits += 1;
        }
        fn bits(&mut self, value: u32, count: u32) {
            for shift in (0..count).rev() {
                self.bit(value >> shift & 1 == 1);
            }
        }
        fn ue(&mut self, value: u32) {
            let code = value + 1;
            let len = 32 - code.leading_zeros();
            self.bits(0, len - 1);
            self.bits(code, len);
        }
        fn finish(mut self) -> Vec<u8> {
            self.bit(true);
            self.bytes
        }
    }

    fn nal(header: u8, body: &[u8]) -> Vec<u8> {
        let mut out = vec![0, 0, 0, 1, header];
        out.extend_from_slice(body);
        out
    }

    /// A baseline SPS for 64x64; `frames_only` false allows field pictures.
    fn sps(frames_only: bool) -> Vec<u8> {
        let mut w = Writer::default();
        w.bits(66, 8);
        w.bits(0, 8);
        w.bits(30, 8);
        w.ue(0); // sps id
        w.ue(0); // log2_max_frame_num - 4
        w.ue(2); // poc type
        w.ue(1); // max refs
        w.bit(false);
        w.ue(3); // width in mbs - 1
        w.ue(if frames_only { 3 } else { 1 });
        w.bit(frames_only);
        if !frames_only {
            w.bit(false);
        }
        w.bit(true);
        w.bit(false);
        w.bit(false);
        nal(0x67, &w.finish())
    }

    fn pps() -> Vec<u8> {
        let mut w = Writer::default();
        w.ue(0);
        w.ue(0);
        w.bits(0, 6);
        nal(0x68, &w.finish())
    }

    /// A slice: `field` is `Some(bottom)` for a field picture.
    fn slice(
        idr: bool,
        first_mb: u32,
        frame_num: u32,
        field: Option<bool>,
        interlaced: bool,
    ) -> Vec<u8> {
        let mut w = Writer::default();
        w.ue(first_mb);
        w.ue(if idr { 7 } else { 5 });
        w.ue(0);
        w.bits(frame_num, 4);
        if interlaced {
            w.bit(field.is_some());
            if let Some(bottom) = field {
                w.bit(bottom);
            }
        }
        w.bits(0xABCD, 16);
        nal(if idr { 0x65 } else { 0x41 }, &w.finish())
    }

    fn aud() -> Vec<u8> {
        nal(0x09, &[0xF0])
    }

    /// Ten progressive access units: an IDR with its parameter sets, then
    /// P pictures, some of two slices.
    fn progressive() -> Vec<(Vec<u8>, bool)> {
        (0..10)
            .map(|index| {
                let mut au = aud();
                let idr = index == 0 || index == 6;
                if idr {
                    au.extend(sps(true));
                    au.extend(pps());
                }
                au.extend(slice(idr, 0, index, None, false));
                if index % 3 == 1 {
                    au.extend(slice(idr, 8, index, None, false));
                }
                (au, idr)
            })
            .collect()
    }

    fn run(pes: Vec<Pes>) -> (Vec<Unit>, u64) {
        let mut framer = VideoFramer::<H264>::new(1 << 20);
        let mut out = Vec::new();
        for packet in pes {
            framer.push(packet, &mut out).expect("push");
        }
        framer.flush(&mut out);
        (out, framer.moved)
    }

    #[test]
    fn one_access_unit_per_pes_packet_moves_without_a_copy() {
        let aus = progressive();
        let pes = aus
            .iter()
            .enumerate()
            .map(|(index, (au, _))| Pes {
                payload: au.clone(),
                pts: Some(index as i64 * 3600),
                dts: None,
                position: index as u64 * 188,
            })
            .collect();
        let (out, moved) = run(pes);
        assert_eq!(out.len(), aus.len());
        for (index, (unit, (au, key))) in out.iter().zip(&aus).enumerate() {
            assert_eq!(&unit.bytes, au, "unit {index}");
            assert_eq!(unit.keyframe, *key, "unit {index}");
            assert_eq!(unit.pts, Some(index as i64 * 3600));
            assert_eq!(unit.dts, unit.pts);
            assert_eq!(unit.position, index as u64 * 188);
        }
        assert_eq!(moved, aus.len() as u64 - 1);
    }

    #[test]
    fn any_pes_split_gives_the_same_access_units() {
        let aus = progressive();
        let stream: Vec<u8> = aus.iter().flat_map(|(au, _)| au.clone()).collect();
        let starts: Vec<usize> = aus
            .iter()
            .scan(0, |at, (au, _)| {
                let start = *at;
                *at += au.len();
                Some(start)
            })
            .collect();
        let mut state = 0x9E37_79B9_u32;
        for _ in 0..300 {
            let mut cuts = vec![0];
            let mut at = 0;
            while at < stream.len() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                at = (at + 1 + (state % 60) as usize).min(stream.len());
                // A packet starting between the leading zero and `00 00 01`
                // leaves the zero with the unit before; keep the test exact.
                if starts.contains(&(at - 1)) && at < stream.len() {
                    at += 1;
                }
                cuts.push(at);
            }
            // Only the first access unit starting in a PES packet has a
            // timestamp.
            let mut expected = vec![None; aus.len()];
            let pes: Vec<Pes> = cuts
                .windows(2)
                .map(|pair| {
                    // H.222.0: the PTS of the first access unit starting in
                    // the packet.
                    let first = starts.iter().position(|&s| s >= pair[0] && s < pair[1]);
                    if let Some(index) = first {
                        expected[index] = Some(index as i64 * 3600);
                    }
                    Pes {
                        payload: stream[pair[0]..pair[1]].to_vec(),
                        pts: first.map(|index| index as i64 * 3600),
                        dts: None,
                        position: pair[0] as u64,
                    }
                })
                .collect();
            let (out, _) = run(pes);
            assert_eq!(out.len(), aus.len(), "cuts {cuts:?}");
            for (index, (unit, (au, key))) in out.iter().zip(&aus).enumerate() {
                assert_eq!(&unit.bytes, au, "unit {index}, cuts {cuts:?}");
                assert_eq!(unit.keyframe, *key, "unit {index}, cuts {cuts:?}");
                assert_eq!(unit.pts, expected[index], "unit {index}, cuts {cuts:?}");
            }
        }
    }

    #[test]
    fn two_fields_of_a_frame_are_one_access_unit() {
        let mut stream = Vec::new();
        stream.extend(sps(false));
        stream.extend(pps());
        let mut aus = Vec::new();
        for frame in 0..4 {
            let mut au = aud();
            au.extend(slice(frame == 0, 0, frame, Some(false), true));
            au.extend(aud());
            au.extend(slice(false, 0, frame, Some(true), true));
            aus.push(au);
        }
        let mut first = stream.clone();
        first.extend(aus[0].clone());
        let mut pes = vec![Pes {
            payload: first.clone(),
            pts: Some(0),
            ..Pes::default()
        }];
        for (index, au) in aus.iter().enumerate().skip(1) {
            pes.push(Pes {
                payload: au.clone(),
                pts: Some(index as i64 * 3600),
                ..Pes::default()
            });
        }
        let (out, _) = run(pes);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].bytes, first);
        assert!(out[0].keyframe);
        for (unit, au) in out.iter().zip(&aus).skip(1) {
            assert_eq!(&unit.bytes, au);
            assert!(!unit.keyframe);
        }
    }

    #[test]
    fn bytes_before_the_first_start_code_are_skipped() {
        let aus = progressive();
        let mut payload = vec![0xEE; 7];
        payload.extend(aus[0].0.clone());
        let mut framer = VideoFramer::<H264>::new(1 << 20);
        let mut out = Vec::new();
        framer
            .push(
                Pes {
                    payload,
                    pts: Some(5),
                    ..Pes::default()
                },
                &mut out,
            )
            .unwrap();
        framer.flush(&mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].bytes, aus[0].0);
        assert_eq!(out[0].pts, Some(5));
        assert_eq!(framer.skipped, 7);
    }

    #[test]
    fn an_endless_access_unit_is_dropped_at_the_cap() {
        let mut framer = VideoFramer::<H264>::new(4096);
        let mut out = Vec::new();
        let mut first = aud();
        first.extend(slice(true, 0, 0, None, false));
        framer
            .push(
                Pes {
                    payload: first,
                    ..Pes::default()
                },
                &mut out,
            )
            .unwrap();
        let mut fault = None;
        for _ in 0..100 {
            if let Err(error) = framer.push(
                Pes {
                    payload: vec![0x55; 1000],
                    ..Pes::default()
                },
                &mut out,
            ) {
                fault = Some(error);
                break;
            }
        }
        assert!(fault.is_some());
        assert!(framer.buffered() <= 4096);
    }
}
