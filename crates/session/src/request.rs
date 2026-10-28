use cutlass_model::{MaterialId, Micros, Segment, SegmentId, TrackId, WordColumns};

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
    /// Move one target edge. At speed 1, a source range moves with that edge.
    Trim {
        segment_id: SegmentId,
        edge: TrimEdge,
        time: Micros,
    },
    /// Replace the recognized words and the current words on one text material.
    ///
    /// Word times are milliseconds. The caption string becomes those words
    /// joined by spaces. Both word arrays receive this result.
    SetWords {
        material_id: MaterialId,
        words: WordColumns,
    },
}
