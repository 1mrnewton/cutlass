use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::enums::{TrackAttribute, TrackFlag, TrackType};
use crate::id::TrackId;
use crate::segment::Segment;
use crate::time::RangeError;

/// One lane. The file order is the row order. Paint order is `render_index` on each segment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub id: TrackId,
    #[serde(rename = "type")]
    pub kind: TrackType,
    #[serde(default)]
    pub flag: TrackFlag,
    #[serde(default)]
    pub attribute: TrackAttribute,
    #[serde(default)]
    pub name: String,
    #[serde(default = "default_true")]
    pub is_default_name: bool,
    #[serde(default)]
    pub segments: Vec<Segment>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

fn default_true() -> bool {
    true
}

impl Track {
    /// The main video lane. Flag 0 on a video track. It stays when it has no segments.
    /// A mixed track with flag 0 is also main; this builder still makes a video track.
    pub fn main(id: impl Into<TrackId>) -> Self {
        Self {
            id: id.into(),
            kind: TrackType::Video,
            flag: TrackFlag::None,
            attribute: TrackAttribute::default(),
            name: String::new(),
            is_default_name: true,
            segments: Vec::new(),
            rest: Map::new(),
        }
    }

    /// Video or mixed, flag 0. That is the main track.
    #[must_use]
    pub fn is_main(&self) -> bool {
        (self.kind == TrackType::Video || self.kind == TrackType::Mixed)
            && self.flag == TrackFlag::None
    }

    /// Video or mixed with any flag other than 0.
    /// Flag 2 is the ordinary overlay. Flags 1 and 3 through 6 count as well.
    #[must_use]
    pub fn is_sub_video(&self) -> bool {
        (self.kind == TrackType::Video || self.kind == TrackType::Mixed)
            && self.flag != TrackFlag::None
    }

    /// Positive spans overlap when one end is past the next start.
    /// Touching at an endpoint is a cut. A zero or negative duration is refused
    /// before that test.
    pub fn segments_overlap(&self) -> Result<bool, RangeError> {
        for segment in &self.segments {
            if segment.target.duration.0 <= 0 {
                return Err(RangeError::NonPositiveDuration);
            }
        }
        let mut spans: Vec<_> = self
            .segments
            .iter()
            .map(|segment| segment.target.clone())
            .collect();
        spans.sort_by_key(|span| span.start);
        for pair in spans.windows(2) {
            if pair[0].overlaps(&pair[1])? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::Segment;
    use crate::time::TimeRange;

    #[test]
    fn the_main_track_is_video_flag_zero() {
        let main = Track::main("t");
        assert!(main.is_main());
        assert!(!main.is_sub_video());
        let json = serde_json::to_value(&main).unwrap();
        assert_eq!(json["type"], "video");
        assert_eq!(json["flag"], 0);
        let back: Track = serde_json::from_value(json).unwrap();
        assert!(back.is_main());
    }

    #[test]
    fn flag_two_is_an_overlay() {
        let track: Track = serde_json::from_str(
            r#"{"id":"o","type":"video","flag":2,"attribute":0,"name":"","is_default_name":true,"segments":[]}"#,
        )
        .unwrap();
        assert!(track.is_sub_video());
        assert!(!track.is_main());
    }

    #[test]
    fn main_and_sub_video_follow_the_native_predicates() {
        let mixed_main: Track =
            serde_json::from_str(r#"{"id":"m","type":"mixed","flag":0,"segments":[]}"#).unwrap();
        assert!(mixed_main.is_main());
        assert!(!mixed_main.is_sub_video());

        let subtitle: Track =
            serde_json::from_str(r#"{"id":"s","type":"video","flag":1,"segments":[]}"#).unwrap();
        assert!(subtitle.is_sub_video());
        assert!(!subtitle.is_main());

        let mixed_overlay: Track =
            serde_json::from_str(r#"{"id":"o","type":"mixed","flag":2,"segments":[]}"#).unwrap();
        assert!(mixed_overlay.is_sub_video());

        let audio: Track =
            serde_json::from_str(r#"{"id":"a","type":"audio","flag":2,"segments":[]}"#).unwrap();
        assert!(!audio.is_main());
        assert!(!audio.is_sub_video());
    }

    #[test]
    fn a_shared_endpoint_is_not_an_overlap() {
        let mut track = Track::main("t");
        track.segments.push(segment("a", 0, 100));
        track.segments.push(segment("b", 100, 50));
        assert_eq!(track.segments_overlap(), Ok(false));
        track.segments[1].target = TimeRange::new(50, 50);
        assert_eq!(track.segments_overlap(), Ok(true));
    }

    #[test]
    fn a_zero_duration_is_refused_before_the_overlap_test() {
        let mut track = Track::main("t");
        track.segments.push(segment("a", 0, 100));
        track.segments.push(segment("b", 50, 0));
        assert_eq!(
            track.segments_overlap(),
            Err(RangeError::NonPositiveDuration)
        );
    }

    fn segment(id: &str, start: i64, duration: i64) -> Segment {
        Segment::new(id, "m", TimeRange::new(start, duration))
    }
}
