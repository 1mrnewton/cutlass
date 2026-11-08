use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;
use crate::time::Micros;

fn one() -> f64 {
    1.0
}

fn free_crop() -> String {
    "free".to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialVideo {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub width: i32,
    #[serde(default)]
    pub height: i32,
    #[serde(default)]
    pub duration: Micros,
    #[serde(default)]
    pub has_audio: bool,
    #[serde(default = "free_crop")]
    pub crop_ratio: String,
    #[serde(default = "one")]
    pub crop_scale: f64,
    /// Eight coordinates. Absent and `Default` are zero, not a full-frame quad.
    #[serde(default)]
    pub crop: Crop,
    #[serde(default)]
    pub reverse_path: String,
    #[serde(default)]
    pub reverse_intensifies_path: String,
    /// Held source frame. Absent and null stay unset.
    #[serde(default)]
    pub freeze: Option<FreezeInfo>,
    /// Steadying. Absent and null are off.
    #[serde(default)]
    pub stable: Option<VideoStable>,
    /// Algorithms run over the clip's frames, such as Enhance quality.
    /// Absent and null are none.
    #[serde(default)]
    pub video_algorithm: Option<VideoAlgorithm>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// The reference editor's `video_algorithm` block: a list of algorithms,
/// each named by its `type`.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct VideoAlgorithm {
    #[serde(default)]
    pub algorithms: Vec<Algorithm>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// One algorithm over a clip's frames. The reference editor stores only
/// its `type`; the level is Cutlass's own key. The result is found by the
/// clip's file and level in the project folder, not stored here.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Algorithm {
    #[serde(default, rename = "type")]
    pub kind: String,
    /// For Enhance quality, [`Algorithm::HD`] or [`Algorithm::UHD`]; for
    /// Reduce image noise, [`Algorithm::WEAK`] or [`Algorithm::STRONG`]; 0
    /// or below is off.
    #[serde(default)]
    pub level: i64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Algorithm {
    /// The reference editor's name for Enhance quality.
    pub const QUALITY_ENHANCE: &'static str = "QualityEnhance";
    /// Enhanced to 1080 on the short side.
    pub const HD: i64 = 1;
    /// Enhanced to 2160 on the short side.
    pub const UHD: i64 = 2;
    /// The reference editor's name for Reduce image noise.
    pub const NOISE_REDUCTION: &'static str = "NoiseReduction";
    /// Light noise taken out.
    pub const WEAK: i64 = 1;
    /// Heavy noise taken out.
    pub const STRONG: i64 = 2;

    /// Whether the algorithm runs.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.level > 0
    }
}

impl MaterialVideo {
    /// The clip's Enhance quality, when it has one that is on.
    #[must_use]
    pub fn enhance(&self) -> Option<&Algorithm> {
        self.algorithm(Algorithm::QUALITY_ENHANCE)
    }

    /// The clip's Reduce image noise, when it has one that is on.
    #[must_use]
    pub fn denoise(&self) -> Option<&Algorithm> {
        self.algorithm(Algorithm::NOISE_REDUCTION)
    }

    /// The clip's algorithm named `kind`, when it has one that is on.
    #[must_use]
    pub fn algorithm(&self, kind: &str) -> Option<&Algorithm> {
        self.video_algorithm
            .as_ref()?
            .algorithms
            .iter()
            .find(|algorithm| algorithm.kind == kind && algorithm.is_on())
    }
}

/// How a clip is steadied: the reference editor's `stable` block, a level
/// and the file of corrections.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct VideoStable {
    /// 1 cuts least, 2 is the recommended level, 3 is the most stable.
    /// 0 or below is off.
    #[serde(default, rename = "stable_level")]
    pub level: i64,
    /// The measured camera path in the project folder, once there is one.
    #[serde(default)]
    pub matrix_path: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl VideoStable {
    /// Least cut away.
    pub const MINIMUM_CUT: i64 = 1;
    pub const RECOMMENDED: i64 = 2;
    pub const MOST_STABLE: i64 = 3;

    /// Whether the clip is steadied.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.level > 0
    }
}

/// A crop quad. Each corner is an `x` and a `y`. Zero is the empty value.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Crop {
    #[serde(default)]
    pub upper_left_x: f64,
    #[serde(default)]
    pub upper_left_y: f64,
    #[serde(default)]
    pub upper_right_x: f64,
    #[serde(default)]
    pub upper_right_y: f64,
    #[serde(default)]
    pub lower_left_x: f64,
    #[serde(default)]
    pub lower_left_y: f64,
    #[serde(default)]
    pub lower_right_x: f64,
    #[serde(default)]
    pub lower_right_y: f64,
}

/// The source frame a freeze holds across the segment's target range.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct FreezeInfo {
    #[serde(default)]
    pub source_material_id: String,
    #[serde(default)]
    pub source_material_path: String,
    #[serde(default)]
    pub reverse_material_path: String,
    #[serde(default)]
    pub timestamp: Micros,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl MaterialVideo {
    /// Resource-mapping fallback for `duration`. An absent key stays zero.
    pub const MAPPING_DURATION: Micros = Micros(3_000_000);

    pub fn new(id: impl Into<MaterialId>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }
}

impl Default for MaterialVideo {
    fn default() -> Self {
        Self {
            id: MaterialId::default(),
            kind: String::new(),
            path: String::new(),
            width: 0,
            height: 0,
            duration: Micros(0),
            has_audio: false,
            crop_ratio: free_crop(),
            crop_scale: 1.0,
            crop: Crop::default(),
            reverse_path: String::new(),
            reverse_intensifies_path: String::new(),
            freeze: None,
            stable: None,
            video_algorithm: None,
            rest: Map::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_matches_a_missing_object_and_not_the_mapping_duration() {
        let video = MaterialVideo::default();
        let parsed: MaterialVideo = serde_json::from_str("{}").unwrap();
        assert_eq!(video, parsed);
        assert_eq!(video.crop_ratio, "free");
        assert_eq!(video.crop_scale, 1.0);
        assert_eq!(video.crop, Crop::default());
        assert_eq!(video.crop.upper_left_x, 0.0);
        assert_eq!(video.duration, Micros(0));
        assert_ne!(video.duration, MaterialVideo::MAPPING_DURATION);
        assert_eq!(MaterialVideo::new("clip").id.as_str(), "clip");
        assert_eq!(MaterialVideo::new("clip").duration, Micros(0));
        assert!(video.freeze.is_none());
        let cleared: MaterialVideo = serde_json::from_str(r#"{"freeze":null}"#).unwrap();
        assert!(cleared.freeze.is_none());
        let held: MaterialVideo = serde_json::from_str(
            r#"{"freeze":{"source_material_id":"src","source_material_path":"clip","reverse_material_path":"","timestamp":1500}}"#,
        )
        .unwrap();
        let freeze = held.freeze.unwrap();
        assert_eq!(freeze.source_material_id, "src");
        assert_eq!(freeze.timestamp, Micros(1500));
    }

    #[test]
    fn a_stable_block_reads_and_writes_its_level_and_keeps_unknown_keys() {
        let video: MaterialVideo = serde_json::from_str(
            r#"{"stable":{"stable_level":2,"matrix_path":"Resources/video_stable/a.json","time_range":{"start":0,"duration":5}}}"#,
        )
        .unwrap();
        let stable = video.stable.clone().unwrap();
        assert_eq!(stable.level, VideoStable::RECOMMENDED);
        assert!(stable.is_on());
        assert_eq!(stable.matrix_path, "Resources/video_stable/a.json");
        assert!(stable.rest.contains_key("time_range"));
        let back: MaterialVideo =
            serde_json::from_str(&serde_json::to_string(&video).unwrap()).unwrap();
        assert_eq!(back, video);
        let off: MaterialVideo = serde_json::from_str(r#"{"stable":{"stable_level":0}}"#).unwrap();
        assert!(!off.stable.unwrap().is_on());
        let null: MaterialVideo = serde_json::from_str(r#"{"stable":null}"#).unwrap();
        assert!(null.stable.is_none());
    }

    #[test]
    fn an_enhance_reads_its_level_and_keeps_unknown_keys() {
        let video: MaterialVideo = serde_json::from_str(
            r#"{"video_algorithm":{"algorithms":[{"type":"Deflicker","level":1},{"type":"QualityEnhance","level":2,"model_id":"x"}],"gameplay":null}}"#,
        )
        .unwrap();
        let enhance = video.enhance().unwrap();
        assert_eq!(enhance.level, Algorithm::UHD);
        assert!(enhance.rest.contains_key("model_id"));
        assert!(
            video
                .video_algorithm
                .as_ref()
                .unwrap()
                .rest
                .contains_key("gameplay")
        );
        let back: MaterialVideo =
            serde_json::from_str(&serde_json::to_string(&video).unwrap()).unwrap();
        assert_eq!(back, video);
        // The reference editor's own block, with only a type, is off.
        let theirs: MaterialVideo = serde_json::from_str(
            r#"{"video_algorithm":{"algorithms":[{"type":"QualityEnhance"}]}}"#,
        )
        .unwrap();
        assert!(theirs.enhance().is_none());
        for none in [
            "{}",
            r#"{"video_algorithm":null}"#,
            r#"{"video_algorithm":{}}"#,
        ] {
            let video: MaterialVideo = serde_json::from_str(none).unwrap();
            assert!(video.enhance().is_none(), "{none}");
        }
    }

    #[test]
    fn a_noise_reduction_reads_apart_from_an_enhance() {
        let video: MaterialVideo = serde_json::from_str(
            r#"{"video_algorithm":{"algorithms":[{"type":"QualityEnhance","level":1},{"type":"NoiseReduction","level":2}]}}"#,
        )
        .unwrap();
        assert_eq!(video.denoise().unwrap().level, Algorithm::STRONG);
        assert_eq!(video.enhance().unwrap().level, Algorithm::HD);
        let only: MaterialVideo = serde_json::from_str(
            r#"{"video_algorithm":{"algorithms":[{"type":"NoiseReduction","level":0},{"type":"QualityEnhance","level":1}]}}"#,
        )
        .unwrap();
        assert!(only.denoise().is_none());
        assert!(only.algorithm("Deflicker").is_none());
    }
}
