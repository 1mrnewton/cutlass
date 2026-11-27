use cutlass_container::DisplayQuarter;

use super::StreamParams;

/// One edit from an edit list.
///
/// `segment_duration` is in the movie header timescale. `media_time` is in the
/// track media timescale. `None` is an empty edit (a dwell). The rate is the
/// 16.16 value stored in the file. Packet timestamps are not rewritten from this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edit {
    pub segment_duration: u64,
    pub media_time: Option<i64>,
    pub media_rate: u32,
}

/// The priming and padding Apple's `iTunSMPB` note states, in media ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gapless {
    /// Ticks before the first valid sample.
    pub priming: u64,
    /// Ticks after the last valid sample.
    pub remainder: u64,
    /// Valid ticks between them.
    pub valid: u64,
}

impl Gapless {
    /// The note when its three counts cover exactly `total` media ticks and
    /// trim something. A note that disagrees with the sample table is stale.
    pub fn fitting(self, total: u64) -> Option<Self> {
        let sum = self
            .priming
            .checked_add(self.valid)?
            .checked_add(self.remainder)?;
        (sum == total && (self.priming > 0 || self.remainder > 0)).then_some(self)
    }
}

/// A track the demuxer can read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    pub params: StreamParams,
    pub edits: Vec<Edit>,
    /// The priming and padding an `iTunSMPB` note states for a sound track
    /// with no edit list. `None` when the track has edits, which decide.
    pub gapless: Option<Gapless>,
    /// Clockwise quarter turn. [`DisplayQuarter::Zero`] when the matrix is not one.
    pub display: DisplayQuarter,
}
