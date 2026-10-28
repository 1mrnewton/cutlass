use cutlass_model::{Micros, Segment, SegmentId, TrackId};

/// Which edge of a segment a trim moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimEdge {
    /// The timeline start. The end stays put.
    Start,
    /// The timeline end. The start stays put.
    End,
}

/// One change. A combo is a slice of these.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// Insert `segment` on `track_id`, ordered by its target start.
    Place {
        track_id: TrackId,
        segment: Box<Segment>,
    },
    /// Keep the duration. `start` is the new target start. `track_id` may be the same track.
    Move {
        segment_id: SegmentId,
        track_id: TrackId,
        start: Micros,
    },
    /// Move one target edge. A source range maps through the segment speed.
    Trim {
        segment_id: SegmentId,
        edge: TrimEdge,
        time: Micros,
    },
    /// Turn one track-attribute bit on or off. Mute and solo are not gains.
    SetTrackAttribute {
        track_id: TrackId,
        bit: i32,
        on: bool,
    },
    /// Set a constant speed. Target duration becomes the source duration mapped by `speed`.
    SetConstantSpeed { segment_id: SegmentId, speed: f64 },
}
