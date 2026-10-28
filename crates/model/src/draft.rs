use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::enums::CanvasRatio;
use crate::id::DraftId;
use crate::materials::Materials;
use crate::time::{FrameRate, Micros, TimecodeLabel};
use crate::track::Track;

fn default_fps() -> f64 {
    30.0
}

fn video_draft() -> String {
    "video".to_owned()
}

fn original() -> String {
    "original".to_owned()
}

fn width_hd() -> i32 {
    1920
}

fn height_hd() -> i32 {
    1080
}

/// `canvas_config`. A missing size is filled as 1920 by 1080. `background` was null on disk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasConfig {
    #[serde(default = "original")]
    pub ratio: String,
    #[serde(default = "width_hd")]
    pub width: i32,
    #[serde(default = "height_hd")]
    pub height: i32,
    #[serde(default)]
    pub background: Option<String>,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for CanvasConfig {
    fn default() -> Self {
        Self {
            ratio: original(),
            width: width_hd(),
            height: height_hd(),
            background: None,
            rest: Map::new(),
        }
    }
}

impl CanvasConfig {
    #[must_use]
    pub fn ratio_kind(&self) -> CanvasRatio {
        CanvasRatio::parse(&self.ratio)
    }
}

/// Switches that change the edit. The rest of `config` is kept and not interpreted.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct DraftConfig {
    /// The magnetic main track. Off leaves gaps between main clips.
    #[serde(default)]
    pub maintrack_adsorb: bool,
    #[serde(default)]
    pub video_mute: bool,
    #[serde(default)]
    pub subtitle_sync: bool,
    #[serde(default)]
    pub lyrics_sync: bool,
    #[serde(default)]
    pub voice_change_sync: bool,
    #[serde(default)]
    pub use_float_render: bool,
    #[serde(default)]
    pub hdr_vivid: bool,
    #[serde(default)]
    pub material_save_mode: i32,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// One cut. `draft_info.json` is this object.
///
/// Time is microseconds. Ids are UUID strings. A segment points at materials by id.
/// Keys this model does not name stay in `rest`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Draft {
    pub id: DraftId,
    pub version: i64,
    #[serde(default)]
    pub new_version: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub duration: Micros,
    #[serde(default)]
    pub create_time: i64,
    #[serde(default)]
    pub update_time: i64,
    #[serde(default = "default_fps")]
    pub fps: f64,
    #[serde(default)]
    pub is_drop_frame_timecode: bool,
    /// An int. `-1` unknown, `0` none, `2` SDR 2020, `67` HLG, `2084` PQ.
    /// A missing value is 0. Other ints stay as stored.
    #[serde(default)]
    pub color_space: i32,
    #[serde(default)]
    pub canvas_config: CanvasConfig,
    #[serde(default)]
    pub config: DraftConfig,
    #[serde(default)]
    pub tracks: Vec<Track>,
    #[serde(default)]
    pub materials: Materials,
    #[serde(default)]
    pub render_index_track_mode_on: bool,
    #[serde(default)]
    pub free_render_index_mode_on: bool,
    #[serde(default)]
    pub mixed_track_mode_on: bool,
    #[serde(default = "video_draft")]
    pub draft_type: String,
    #[serde(default)]
    pub source: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Draft {
    /// 1920×1080, original ratio, 30 fps, magnet on, and one empty main track.
    ///
    /// `version` is 360000 and `new_version` is 189.0.0.
    pub fn empty(id: impl Into<DraftId>) -> Self {
        Self {
            id: id.into(),
            version: 360_000,
            new_version: "189.0.0".to_owned(),
            name: String::new(),
            duration: Micros(0),
            create_time: 0,
            update_time: 0,
            fps: 30.0,
            is_drop_frame_timecode: false,
            color_space: 0,
            canvas_config: CanvasConfig::default(),
            config: DraftConfig {
                maintrack_adsorb: true,
                ..DraftConfig::default()
            },
            tracks: vec![Track::main("")],
            materials: Materials::default(),
            render_index_track_mode_on: false,
            free_render_index_mode_on: false,
            mixed_track_mode_on: false,
            draft_type: video_draft(),
            source: "default".to_owned(),
            rest: Map::new(),
        }
    }

    #[must_use]
    pub fn main_track(&self) -> Option<&Track> {
        self.tracks.iter().find(|track| track.is_main())
    }

    /// The rational rate for the stored `fps` number. Drop frame is a label, not this rate.
    #[must_use]
    pub fn frame_rate(&self) -> Option<FrameRate> {
        FrameRate::from_stored_fps(self.fps)
    }

    #[must_use]
    pub fn timecode(&self, time: Micros) -> Option<TimecodeLabel> {
        self.frame_rate()?.label(time, self.is_drop_frame_timecode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::MaterialRef;
    use crate::time::FrameRate;

    #[test]
    fn an_empty_cut_has_one_main_track() {
        let draft = Draft::empty("D");
        assert_eq!(draft.version, 360_000);
        assert_eq!(draft.canvas_config.ratio_kind(), CanvasRatio::Original);
        assert_eq!(draft.canvas_config.width, 1920);
        assert!(draft.config.maintrack_adsorb);
        assert_eq!(draft.tracks.len(), 1);
        assert!(draft.main_track().unwrap().is_main());
        let back: Draft = serde_json::from_str(&serde_json::to_string(&draft).unwrap()).unwrap();
        assert_eq!(back, draft);
    }

    #[test]
    fn unnamed_root_keys_survive() {
        let draft: Draft = serde_json::from_str(
            r#"{"id":"D","version":360000,"new_version":"189.0.0","fps":30.0,"color_space":-1,"draft_type":"video","canvas_config":{"ratio":"original","width":1080,"height":1920,"background":null},"config":{"maintrack_adsorb":false,"export_range":null},"tracks":[],"materials":{"texts":[{"id":"t","type":"text","content":"hi"}]},"time_marks":null}"#,
        )
        .unwrap();
        assert_eq!(draft.color_space, -1);
        assert!(!draft.config.maintrack_adsorb);
        assert!(draft.config.rest["export_range"].is_null());
        assert!(draft.rest["time_marks"].is_null());
        assert_eq!(draft.canvas_config.height, 1920);
        assert!(matches!(
            draft.materials.get("t"),
            Some(MaterialRef::Text(text)) if text.content == "hi"
        ));
    }

    #[test]
    fn canvas_keeps_unnamed_keys() {
        let canvas: CanvasConfig = serde_json::from_str(
            r#"{"ratio":"original","width":1920,"height":1080,"background":null,"future_canvas":1}"#,
        )
        .unwrap();
        assert_eq!(canvas.rest["future_canvas"], 1);
        assert_eq!(serde_json::to_value(&canvas).unwrap()["future_canvas"], 1);
    }

    #[test]
    fn the_stored_fps_is_the_frame_rate_and_drop_frame_is_a_label() {
        let draft = Draft::empty("D");
        assert_eq!(draft.frame_rate(), Some(FrameRate::FPS_30));
        assert_eq!(draft.timecode(Micros(0)).unwrap().format(), "00:00:00:00");
        assert_eq!(
            draft.timecode(Micros(1_000_000)).unwrap().format(),
            "00:00:01:00"
        );
        let mut drop = draft;
        drop.fps = 29.97;
        drop.is_drop_frame_timecode = true;
        assert_eq!(drop.frame_rate(), Some(FrameRate::FPS_29_97));
        assert_eq!(drop.timecode(Micros(0)).unwrap().format(), "00:00:00;00");
    }
}
