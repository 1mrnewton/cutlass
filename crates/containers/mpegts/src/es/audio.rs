//! Audio frames found by their sync words, timed from the PES timestamp
//! and the samples each frame holds.
//!
//! A frame may straddle two PES packets: its start is kept until the rest
//! arrives. Bytes that are not a frame are skipped one sync candidate at a
//! time and counted. A frame's time is the PTS of the PES packet it starts
//! in when that packet has one, otherwise the time after the frame before.

use super::{Facts, FrameFault, MAX_UNITS_PER_PES, Pes, Unit};

/// What a sync-word syntax found at the start of a byte slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Found {
    /// More bytes are needed to read the header.
    NeedMore,
    /// No frame starts at this byte.
    NotSync,
    Frame(Frame),
}

/// One frame's span and what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Bytes from the sync word to the next frame.
    pub len: usize,
    /// The payload handed on, as offsets inside the frame.
    pub payload: (usize, usize),
    pub samples: u32,
    pub rate: u32,
    /// The frame belongs to the access unit before it (an E-AC-3 dependent
    /// substream, or a second independent substream).
    pub continues: bool,
}

/// A frame syntax: ADTS, AC-3, MPEG audio, LOAS.
pub trait Syntax {
    /// Bytes a header needs.
    const HEADER: usize;
    /// The longest frame, so a broken length cannot hold bytes forever.
    const MAX_FRAME: usize;
    /// The first byte of every sync word.
    const SYNC_BYTE: u8;
    /// Reads the frame at the start of `bytes`.
    fn frame(&mut self, bytes: &[u8]) -> Found;
    /// Notes the first frame's parameters. `frame` is the whole frame.
    fn learn(&mut self, frame: &[u8]);
    fn facts(&self) -> Option<Facts>;
    /// Turns a frame's payload into the unit's bytes. The default copies it.
    fn payload(&mut self, frame: &[u8], payload: (usize, usize)) -> Option<Vec<u8>> {
        frame.get(payload.0..payload.1).map(<[u8]>::to_vec)
    }
}

/// Times frames on the 90 kHz clock without drift: the time of a frame is
/// the last PES timestamp plus the samples since, scaled once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Clock {
    base: Option<i64>,
    rate: u32,
    samples: u64,
}

impl Clock {
    pub fn rebase(&mut self, pts: i64) {
        self.base = Some(pts);
        self.samples = 0;
    }

    pub fn forget(&mut self) {
        *self = Self::default();
    }

    fn offset(samples: u64, rate: u32) -> i64 {
        if rate == 0 {
            return 0;
        }
        let ticks = u128::from(samples) * 90_000 / u128::from(rate);
        i64::try_from(ticks).unwrap_or(i64::MAX)
    }

    /// The time of a frame of `samples` at `rate`, and its length, both in
    /// 90 kHz ticks. The clock then moves past it.
    pub fn advance(&mut self, samples: u32, rate: u32) -> (Option<i64>, Option<i64>) {
        if rate != self.rate {
            if let Some(base) = self.base {
                self.base = Some(base.saturating_add(Self::offset(self.samples, self.rate)));
            }
            self.samples = 0;
            self.rate = rate;
        }
        let start = Self::offset(self.samples, rate);
        self.samples = self.samples.saturating_add(u64::from(samples));
        let end = Self::offset(self.samples, rate);
        let duration = (rate != 0).then(|| end.saturating_sub(start));
        (self.base.map(|base| base.saturating_add(start)), duration)
    }
}

pub struct AudioFramer<S: Syntax> {
    syntax: S,
    carry: Vec<u8>,
    clock: Clock,
    learned: bool,
    /// Bytes skipped looking for a sync word.
    pub skipped: u64,
}

impl<S: Syntax> AudioFramer<S> {
    pub const fn new(syntax: S) -> Self {
        Self {
            syntax,
            carry: Vec::new(),
            clock: Clock {
                base: None,
                rate: 0,
                samples: 0,
            },
            learned: false,
            skipped: 0,
        }
    }

    pub fn reset(&mut self) {
        self.carry.clear();
        self.clock.forget();
    }

    pub fn buffered(&self) -> usize {
        self.carry.len()
    }

    pub fn facts(&self) -> Option<Facts> {
        self.syntax.facts()
    }

    pub fn flush(&mut self, _out: &mut Vec<Unit>) {
        // A partial frame at the end is not a frame.
        self.skipped = self.skipped.saturating_add(self.carry.len() as u64);
        self.carry.clear();
    }

    pub fn push(&mut self, pes: Pes, out: &mut Vec<Unit>) -> Result<(), FrameFault> {
        let fresh_from = self.carry.len();
        let data = if self.carry.is_empty() {
            pes.payload
        } else {
            let mut joined = std::mem::take(&mut self.carry);
            joined.extend_from_slice(&pes.payload);
            joined
        };
        let mut pending_pts = pes.pts;
        let mut at = 0;
        let mut open: Option<Unit> = None;
        let mut units = 0_usize;
        let mut fault = Ok(());
        while data.len() - at >= S::HEADER {
            let frame = match self.syntax.frame(&data[at..]) {
                Found::NeedMore => break,
                Found::NotSync => {
                    let next = data[at + 1..]
                        .iter()
                        .position(|&byte| byte == S::SYNC_BYTE)
                        .map_or(data.len(), |offset| at + 1 + offset);
                    self.skipped = self.skipped.saturating_add((next - at) as u64);
                    at = next;
                    continue;
                }
                Found::Frame(frame) => frame,
            };
            let Some(end) = at.checked_add(frame.len) else {
                break;
            };
            if end > data.len() {
                break;
            }
            if !self.learned {
                self.syntax.learn(&data[at..end]);
                self.learned = true;
            }
            if at >= fresh_from
                && let Some(pts) = pending_pts.take()
            {
                self.clock.rebase(pts);
            }
            let (pts, duration) = self.clock.advance(frame.samples, frame.rate);
            let joins = frame.continues && open.is_some();
            if joins {
                if let Some(unit) = open.as_mut() {
                    unit.bytes.extend_from_slice(&data[at..end]);
                    unit.duration = match (unit.duration, duration) {
                        (Some(a), Some(b)) if frame.samples > 0 => Some(a.saturating_add(b)),
                        (kept, _) => kept,
                    };
                }
            } else {
                if let Some(done) = open.take() {
                    out.push(done);
                }
                units += 1;
                if units > MAX_UNITS_PER_PES {
                    fault = Err(FrameFault::TooManyUnits);
                    at = data.len();
                    break;
                }
                let Some(bytes) = self.syntax.payload(&data[at..end], frame.payload) else {
                    self.skipped = self.skipped.saturating_add(frame.len as u64);
                    at = end;
                    continue;
                };
                open = Some(Unit {
                    bytes,
                    pts,
                    dts: pts,
                    duration,
                    keyframe: true,
                    position: pes.position,
                });
            }
            at = end;
        }
        if let Some(done) = open.take() {
            out.push(done);
        }
        let rest = &data[at..];
        if rest.len() > S::MAX_FRAME {
            self.skipped = self.skipped.saturating_add(rest.len() as u64);
        } else {
            self.carry.extend_from_slice(rest);
        }
        // A timestamp whose PES packet started no frame is dropped, as the
        // specification says it should not be there.
        fault
    }
}

#[cfg(test)]
mod tests {
    use super::Clock;

    #[test]
    fn the_clock_does_not_drift_at_44100() {
        let mut clock = Clock::default();
        clock.rebase(1000);
        let mut total = 0;
        let mut last = None;
        for _ in 0..44_100 {
            let (pts, duration) = clock.advance(1024, 44_100);
            total += duration.unwrap();
            last = pts;
        }
        // 44100 frames of 1024 samples are exactly 1024 seconds.
        assert_eq!(total, 1024 * 90_000);
        assert_eq!(last, Some(1000 + 1024 * 90_000 - 2090));
    }

    #[test]
    fn a_clock_without_a_timestamp_still_gives_lengths() {
        let mut clock = Clock::default();
        assert_eq!(clock.advance(1152, 48_000), (None, Some(2160)));
    }
}
