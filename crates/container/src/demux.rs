//! The demux trait.
//!
//! A demuxer owns its input, publishes streams only after open succeeds,
//! and returns one access unit per read, in decode order. Timestamps stay
//! in the file timescale.

use crate::error::ReadError;
use crate::seek::{SeekLanding, SeekTo};
use crate::stream::{Gap, Stream, TrackId};
use cutlass_codec::{ContainerId, Packet};

/// One open file: streams out, then packets out.
///
/// The trait is object-safe so `engine` can hold `Box<dyn Demux>`.
/// Open stays on the concrete type because it consumes the input.
pub trait Demux: Send {
    fn family(&self) -> ContainerId;

    /// Streams published by a successful open.
    fn streams(&self) -> &[Stream];

    /// Gaps in the file timescale. They are not applied to packet timestamps.
    fn gaps(&self) -> &[Gap];

    /// The next access unit in decode order, or [`None`] at the end.
    ///
    /// One call returns one packet. There is no internal queue.
    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError>;

    /// Moves the next read to `to` in the file timescale.
    ///
    /// On failure the read position does not change. Packets from before
    /// the landing are not returned after a successful seek.
    fn seek(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError>;

    /// The next read or seek returns [`ReadError::Cancelled`].
    fn cancel(&mut self);

    /// Bytes read from the owned input. A seek does not increase this.
    fn consumed(&self) -> u64;

    /// The latest presentation time plus duration of any packet of
    /// `track`, in the file timescale, when the file's index holds it
    /// without reading a packet. It is what reading every packet of the
    /// track would find.
    ///
    /// [`None`] when the container has no such index, the track has no
    /// samples, or the index is not complete at open, as in a fragmented
    /// file: a caller that needs the end then reads the packets.
    fn indexed_end(&self, track: TrackId) -> Option<i64> {
        let _ = track;
        None
    }
}
