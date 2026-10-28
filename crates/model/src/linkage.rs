use serde_json::Value;

use crate::enums::{TrackFlag, TrackType};
use crate::track::Track;

/// Follow or cut switches for the kinds a dependent can sit on.
///
/// A missing key stays true. An explicit false stays false.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindSwitches {
    pub overlay: bool,
    pub text: bool,
    pub sticker: bool,
    pub effect: bool,
    pub filter: bool,
    pub adjust: bool,
    pub audio: bool,
}

impl KindSwitches {
    pub const ALL: Self = Self {
        overlay: true,
        text: true,
        sticker: true,
        effect: true,
        filter: true,
        adjust: true,
        audio: true,
    };

    /// Whether a dependent on `track` follows this set of switches.
    ///
    /// Overlay is a video or mixed track whose flag is not 0. Captions are
    /// text. Sound effects and text to speech are audio.
    #[must_use]
    pub fn allows(self, track: &Track) -> bool {
        match track.kind {
            TrackType::Video | TrackType::Mixed => track.flag != TrackFlag::None && self.overlay,
            TrackType::Text | TrackType::TextToVideoText => self.text,
            TrackType::Sticker => self.sticker,
            TrackType::Effect | TrackType::PluginEffect => self.effect,
            TrackType::Filter => self.filter,
            TrackType::Adjust => self.adjust,
            TrackType::Audio | TrackType::TextToVideoTextAudio => self.audio,
            TrackType::None | TrackType::Composition | TrackType::Adcube | TrackType::Other(_) => {
                false
            }
        }
    }
}

impl Default for KindSwitches {
    fn default() -> Self {
        Self::ALL
    }
}

/// The linkage master switch, plus follow and cut per kind.
///
/// A new draft sets the master switch and both kind sets true.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkageSettings {
    pub enabled: bool,
    pub follow: KindSwitches,
    pub cut: KindSwitches,
}

impl Default for LinkageSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            follow: KindSwitches::ALL,
            cut: KindSwitches::ALL,
        }
    }
}

/// Read switches from `draft_biz_config.json`.
///
/// `linkage_enabled` lives under `timeline_settings` and the timeline id.
/// A missing master switch is false when the draft has no linkage pairs,
/// and true when it has some. Missing follow and cut keys stay true.
#[must_use]
pub fn linkage_from_biz(
    biz: Option<&Value>,
    timeline_id: &str,
    has_linkage_pairs: bool,
) -> LinkageSettings {
    let Some(timeline) = biz
        .and_then(|value| value.get("timeline_settings"))
        .and_then(|settings| settings.get(timeline_id))
    else {
        return LinkageSettings {
            enabled: has_linkage_pairs,
            ..LinkageSettings::default()
        };
    };
    let enabled = match timeline.get("linkage_enabled") {
        Some(Value::Bool(value)) => *value,
        _ => has_linkage_pairs,
    };
    LinkageSettings {
        enabled,
        follow: kind_switches(timeline.get("follow")),
        cut: kind_switches(timeline.get("cut")),
    }
}

fn kind_switches(value: Option<&Value>) -> KindSwitches {
    let Some(object) = value.and_then(Value::as_object) else {
        return KindSwitches::ALL;
    };
    KindSwitches {
        overlay: flag(object, "overlay"),
        text: flag(object, "text"),
        sticker: flag(object, "sticker"),
        effect: flag(object, "effect"),
        filter: flag(object, "filter"),
        adjust: flag(object, "adjust"),
        audio: flag(object, "audio"),
    }
}

fn flag(object: &serde_json::Map<String, Value>, key: &str) -> bool {
    match object.get(key) {
        Some(Value::Bool(value)) => *value,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_missing_master_switch_follows_whether_pairs_exist() {
        let with_pairs = linkage_from_biz(None, "tl-main", true);
        assert!(with_pairs.enabled);
        assert!(with_pairs.follow.overlay);
        assert!(with_pairs.cut.audio);
        let without = linkage_from_biz(None, "tl-main", false);
        assert!(!without.enabled);
        assert!(without.follow.text);
        let explicit = linkage_from_biz(
            Some(&json!({
                "timeline_settings": {
                    "tl-main": {
                        "linkage_enabled": false,
                        "follow": { "audio": false },
                        "cut": { "text": false }
                    }
                }
            })),
            "tl-main",
            true,
        );
        assert!(!explicit.enabled);
        assert!(!explicit.follow.audio);
        assert!(explicit.follow.overlay);
        assert!(!explicit.cut.text);
        assert!(explicit.cut.audio);
    }
}
