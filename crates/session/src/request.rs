use cutlass_model::{
    CommonKeyframeGroup, Crop, FreezeInfo, MaterialAudio, MaterialId, MaterialTransition,
    MaterialVideo, Micros, Segment, SegmentId, SpeedPoint, TrackId, TrackType, WordColumns,
};

/// A media record an import places beside its segment.
#[derive(Clone, Debug, PartialEq)]
pub enum ImportedMedium {
    /// A picture, including a still stored as a video material.
    Video(Box<MaterialVideo>),
    /// A sound. Its track type is supplied by the import, not by lane category.
    Audio(Box<MaterialAudio>),
}

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
    /// Replace the recognized words and the current words on one text material.
    ///
    /// Word times are milliseconds. The caption string becomes those words
    /// joined by spaces. Both word arrays receive this result.
    SetWords {
        material_id: MaterialId,
        words: WordColumns,
    },
    /// Turn one track-attribute bit on or off. Mute and solo are not gains.
    SetTrackAttribute {
        track_id: TrackId,
        bit: i32,
        on: bool,
    },
    /// Set a constant speed. Target duration becomes the source duration mapped by `speed`.
    SetConstantSpeed { segment_id: SegmentId, speed: f64 },
    /// Store a curve. `y` is speed and is linear in `x`. Target duration is the integral.
    SetCurveSpeed {
        segment_id: SegmentId,
        speed_id: MaterialId,
        points: Vec<SpeedPoint>,
    },
    /// Store a blend on the outgoing segment. Target ranges still only touch.
    SetTransition {
        segment_id: SegmentId,
        transition: Box<MaterialTransition>,
    },
    /// Split `segment_id` at `at`. The time must sit outside every blend window on the track.
    Split {
        segment_id: SegmentId,
        at: Micros,
        right_id: SegmentId,
    },
    /// Store a linkage pair. `owner` is the segment the request names.
    Link {
        owner: SegmentId,
        dependent: SegmentId,
    },
    /// Remove one segment. An owner takes its dependents with it when follow is on.
    Remove { segment_id: SegmentId },
    /// Store the reverse flag and the path on the video material. This does not seek.
    SetReverse {
        segment_id: SegmentId,
        reverse: bool,
        path: String,
    },
    /// Store a freeze mapping on the video material. This does not seek.
    SetFreeze {
        segment_id: SegmentId,
        freeze: Option<FreezeInfo>,
    },
    /// Set the segment gain. This is not mute, solo, or a fade.
    SetVolume { segment_id: SegmentId, volume: f64 },
    /// Store fade-in and fade-out durations. The segment volume stays.
    SetFade {
        segment_id: SegmentId,
        fade_id: MaterialId,
        fade_in: Micros,
        fade_out: Micros,
    },
    /// Store the crop quad. Ratio and scale stay.
    SetCrop { segment_id: SegmentId, crop: Crop },
    /// Replace the text material's string. Word columns stay as stored.
    SetTextContent {
        segment_id: SegmentId,
        content: String,
    },
    /// Replace the segment's keyframe groups.
    SetKeyframes {
        segment_id: SegmentId,
        keyframes: Vec<CommonKeyframeGroup>,
    },
    /// Add a material and place its segment. A missing track of `kind` is created.
    Import {
        track_id: TrackId,
        kind: TrackType,
        medium: ImportedMedium,
        segment: Box<Segment>,
    },
}
