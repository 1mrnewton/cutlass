use super::Error;
use super::source::Source;
use super::stream::Stream;
use super::{Packet, Timescale};

/// Where a seek landed, in the track's file timescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekAt {
    pub track_id: u32,
    /// Decode time of the sample the next read returns.
    pub dts: i64,
    pub keyframe: bool,
    /// The landing used a sample table or a fragment index.
    pub used_index: bool,
    /// Sample distance from this sync sample to the sample that contains the
    /// requested time. Zero when the sync sample itself contains that time.
    pub preroll: u32,
}

/// Reads packets from a container.
///
/// `seek` moves the read cursor. A failed seek leaves the cursor where it was.
/// Packets before the landing sample are not returned.
pub trait Demux {
    fn movie_timescale(&self) -> Timescale;
    fn streams(&self) -> &[Stream];
    fn read_packet(&mut self, source: &mut dyn Source) -> Result<Option<Packet>, Error>;
    fn seek(&mut self, source: &mut dyn Source, track_id: u32, ticks: i64)
    -> Result<SeekAt, Error>;
}
