use serde::{Deserialize, Serialize};

/// Track kind. Value 0 and every value above 13 are `none`.
///
/// An unknown string is kept so a save can write it back.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum TrackType {
    #[default]
    None,
    Video,
    Audio,
    Sticker,
    Text,
    Effect,
    Filter,
    Adjust,
    TextToVideoText,
    TextToVideoTextAudio,
    Composition,
    PluginEffect,
    Adcube,
    Mixed,
    Other(String),
}

impl TrackType {
    /// Paint-order floor. The segment can still store its own index.
    /// Text and stickers share 14000, which is the index a text segment stores.
    #[must_use]
    pub fn render_floor(self) -> Option<i32> {
        match self {
            Self::Adjust => Some(7_000),
            Self::Filter => Some(10_000),
            Self::Effect => Some(11_000),
            Self::Text | Self::Sticker => Some(14_000),
            Self::PluginEffect => Some(20_000),
            Self::None
            | Self::Video
            | Self::Audio
            | Self::TextToVideoText
            | Self::TextToVideoTextAudio
            | Self::Composition
            | Self::Adcube
            | Self::Mixed
            | Self::Other(_) => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Sticker => "sticker",
            Self::Text => "text",
            Self::Effect => "effect",
            Self::Filter => "filter",
            Self::Adjust => "adjust",
            Self::TextToVideoText => "text_to_video_text",
            Self::TextToVideoTextAudio => "text_to_video_text_audio",
            Self::Composition => "composition",
            Self::PluginEffect => "plugin_effect",
            Self::Adcube => "adcube",
            Self::Mixed => "mixed",
            Self::Other(name) => name,
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "none" => Self::None,
            "video" => Self::Video,
            "audio" => Self::Audio,
            "sticker" => Self::Sticker,
            "text" => Self::Text,
            "effect" => Self::Effect,
            "filter" => Self::Filter,
            "adjust" => Self::Adjust,
            "text_to_video_text" => Self::TextToVideoText,
            "text_to_video_text_audio" => Self::TextToVideoTextAudio,
            "composition" => Self::Composition,
            "plugin_effect" => Self::PluginEffect,
            "adcube" => Self::Adcube,
            "mixed" => Self::Mixed,
            _ => return None,
        })
    }
}

impl Serialize for TrackType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for TrackType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::parse(&value).unwrap_or(Self::Other(value)))
    }
}

/// Track flag. Values 1 through 6 are kept. Every other int becomes 0.
///
/// On a video or mixed track, 0 is the main track. Any other flag is a sub-video track.
/// 2 is the ordinary overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum TrackFlag {
    #[default]
    None = 0,
    TextSubtitle = 1,
    SubVideo = 2,
    TextLyric = 3,
    VideoSubtitle = 4,
    AudioSubtitle = 5,
    AiScriptSubtitle = 6,
}

impl TrackFlag {
    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        match value {
            1 => Self::TextSubtitle,
            2 => Self::SubVideo,
            3 => Self::TextLyric,
            4 => Self::VideoSubtitle,
            5 => Self::AudioSubtitle,
            6 => Self::AiScriptSubtitle,
            _ => Self::None,
        }
    }

    #[must_use]
    pub fn value(self) -> i32 {
        self as i32
    }
}

impl Serialize for TrackFlag {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_i32(self.value())
    }
}

impl<'de> Deserialize<'de> for TrackFlag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::from_i64(i64::deserialize(deserializer)?))
    }
}

/// Track attribute. Mute, hide, lock, and solo are bits, not keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrackAttribute(pub i32);

impl TrackAttribute {
    pub const MUTE: i32 = 1;
    pub const HIDE: i32 = 2;
    pub const LOCK: i32 = 4;
    pub const SOLO: i32 = 8;
    pub const MUTE_BY_SOLO: i32 = 16;

    #[must_use]
    pub fn contains(self, bit: i32) -> bool {
        self.0 & bit == bit
    }
}

/// Speed mode. Any int other than 1 is a constant speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum SpeedMode {
    #[default]
    Normal = 0,
    Curve = 1,
}

impl SpeedMode {
    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        if value == 1 {
            Self::Curve
        } else {
            Self::Normal
        }
    }

    #[must_use]
    pub fn value(self) -> i32 {
        self as i32
    }
}

impl Serialize for SpeedMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_i32(self.value())
    }
}

impl<'de> Deserialize<'de> for SpeedMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::from_i64(i64::deserialize(deserializer)?))
    }
}

/// Where an effect segment applies. 0 is the main video, 1 is one overlay, 2 is every video.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ApplyType {
    MainVideo = 0,
    SubVideo = 1,
    AllVideo = 2,
}

impl ApplyType {
    #[must_use]
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::MainVideo),
            1 => Some(Self::SubVideo),
            2 => Some(Self::AllVideo),
            _ => None,
        }
    }
}

/// Keyframe curve. The strings are the ones written in `curveType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum KeyframeCurve {
    #[default]
    Line,
    CurveOut,
    CurveIn,
    CurveInOut,
    FreeCurveIn,
    FreeCurveOut,
    FreeCurveInOut,
    #[serde(rename = "none")]
    None,
}

impl KeyframeCurve {
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "Line" => Self::Line,
            "CurveOut" => Self::CurveOut,
            "CurveIn" => Self::CurveIn,
            "CurveInOut" => Self::CurveInOut,
            "FreeCurveIn" => Self::FreeCurveIn,
            "FreeCurveOut" => Self::FreeCurveOut,
            "FreeCurveInOut" => Self::FreeCurveInOut,
            "none" => Self::None,
            _ => return None,
        })
    }
}

/// Animation kind. Out of range is an empty string, which does not parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnimType {
    #[default]
    #[serde(rename = "none")]
    None,
    In,
    Out,
    Loop,
    Caption,
    Group,
}

impl AnimType {
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "none" => Self::None,
            "in" => Self::In,
            "out" => Self::Out,
            "loop" => Self::Loop,
            "caption" => Self::Caption,
            "group" => Self::Group,
            _ => return None,
        })
    }
}

/// Canvas ratio. Out of range reads as `original`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum CanvasRatio {
    #[default]
    Original,
    Custom,
    Ratio16To9,
    Ratio9To16,
    Ratio4To3,
    Ratio3To4,
    Ratio1To1,
    Ratio2To1,
    Ratio235To1,
    Ratio185To1,
    Ratio1125To2436,
    Ratio4To5,
    Ratio21To9,
    Ratio128To27,
    Ratio1To2,
}

impl CanvasRatio {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Custom => "custom",
            Self::Ratio16To9 => "16:9",
            Self::Ratio9To16 => "9:16",
            Self::Ratio4To3 => "4:3",
            Self::Ratio3To4 => "3:4",
            Self::Ratio1To1 => "1:1",
            Self::Ratio2To1 => "2:1",
            Self::Ratio235To1 => "2.35:1",
            Self::Ratio185To1 => "1.85:1",
            Self::Ratio1125To2436 => "1.125:2.436",
            Self::Ratio4To5 => "4:5",
            Self::Ratio21To9 => "21:9",
            Self::Ratio128To27 => "128:27",
            Self::Ratio1To2 => "1:2",
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value {
            "custom" => Self::Custom,
            "16:9" => Self::Ratio16To9,
            "9:16" => Self::Ratio9To16,
            "4:3" => Self::Ratio4To3,
            "3:4" => Self::Ratio3To4,
            "1:1" => Self::Ratio1To1,
            "2:1" => Self::Ratio2To1,
            "2.35:1" => Self::Ratio235To1,
            "1.85:1" => Self::Ratio185To1,
            "1.125:2.436" => Self::Ratio1125To2436,
            "4:5" => Self::Ratio4To5,
            "21:9" => Self::Ratio21To9,
            "128:27" => Self::Ratio128To27,
            "1:2" => Self::Ratio1To2,
            _ => Self::Original,
        }
    }
}

/// Crop ratio. This is not the canvas list: crop has `free` and no `custom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum CropRatio {
    #[default]
    Free,
    Ratio16To9,
    Ratio9To16,
    Ratio4To3,
    Ratio3To4,
    Ratio1To1,
    Ratio2To1,
    Ratio235To1,
    Ratio185To1,
    Ratio1125To2436,
    Original,
    Ratio1To2,
}

impl CropRatio {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Ratio16To9 => "16:9",
            Self::Ratio9To16 => "9:16",
            Self::Ratio4To3 => "4:3",
            Self::Ratio3To4 => "3:4",
            Self::Ratio1To1 => "1:1",
            Self::Ratio2To1 => "2:1",
            Self::Ratio235To1 => "2.35:1",
            Self::Ratio185To1 => "1.85:1",
            Self::Ratio1125To2436 => "1.125:2.436",
            Self::Original => "original",
            Self::Ratio1To2 => "1:2",
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value {
            "16:9" => Self::Ratio16To9,
            "9:16" => Self::Ratio9To16,
            "4:3" => Self::Ratio4To3,
            "3:4" => Self::Ratio3To4,
            "1:1" => Self::Ratio1To1,
            "2:1" => Self::Ratio2To1,
            "2.35:1" => Self::Ratio235To1,
            "1.85:1" => Self::Ratio185To1,
            "1.125:2.436" => Self::Ratio1125To2436,
            "original" => Self::Original,
            "1:2" => Self::Ratio1To2,
            _ => Self::Free,
        }
    }
}

/// Where a segment was made. An unknown string is kept.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum SegmentSource {
    #[default]
    Normal,
    SmartEditQuickClip,
    SmartEditAudio,
    SmartEditText,
    ImageEditorSticker,
    IntelligentAutoCaption,
    IntelligentCaptionOpt,
    AiVoice,
    ImageEditorStickerReplace,
    QualityCompare,
    ImageEditorOverlay,
    ImageEditorCoverImage,
    ImageEditorCoverFrame,
    AiVideoExpand,
    ImageEditorTemp,
    CanvasEditCopy,
    CanvasEditSmartSplit,
    CanvasEditBoxSplit,
    TransparentCanvas,
    Other(String),
}

impl SegmentSource {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Normal => "segmentsourcenormal",
            Self::SmartEditQuickClip => "segmentsourcesmarteditquickclip",
            Self::SmartEditAudio => "segmentsourcesmarteditaudio",
            Self::SmartEditText => "segmentsourcesmartedittext",
            Self::ImageEditorSticker => "segmentsourceimageeditorsticker",
            Self::IntelligentAutoCaption => "segmentsourceintelligentfunctionrecommendautocaption",
            Self::IntelligentCaptionOpt => "segmentsourceintelligentfunctionrecommendcaptionopt",
            Self::AiVoice => "segmentsourceaivoice",
            Self::ImageEditorStickerReplace => "segmentsourceimageeditorstickerreplace",
            Self::QualityCompare => "segmentsourcequalitycompare",
            Self::ImageEditorOverlay => "segmentsourceimageeditoroverlay",
            Self::ImageEditorCoverImage => "segmentsourceimageEditorcoverImage",
            Self::ImageEditorCoverFrame => "segmentsourceimageEditorcoverframe",
            Self::AiVideoExpand => "SegmentSourceAIVideoExpand",
            Self::ImageEditorTemp => "SegmentSourceImageEditorTemp",
            Self::CanvasEditCopy => "SegmentSourceCanvasEditCopy",
            Self::CanvasEditSmartSplit => "SegmentSourceCanvasEditSmartSplit",
            Self::CanvasEditBoxSplit => "SegmentSourceCanvasEditBoxSplit",
            Self::TransparentCanvas => "SegmentSourceTransparentCanvas",
            Self::Other(name) => name,
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "segmentsourcenormal" => Self::Normal,
            "segmentsourcesmarteditquickclip" => Self::SmartEditQuickClip,
            "segmentsourcesmarteditaudio" => Self::SmartEditAudio,
            "segmentsourcesmartedittext" => Self::SmartEditText,
            "segmentsourceimageeditorsticker" => Self::ImageEditorSticker,
            "segmentsourceintelligentfunctionrecommendautocaption" => Self::IntelligentAutoCaption,
            "segmentsourceintelligentfunctionrecommendcaptionopt" => Self::IntelligentCaptionOpt,
            "segmentsourceaivoice" => Self::AiVoice,
            "segmentsourceimageeditorstickerreplace" => Self::ImageEditorStickerReplace,
            "segmentsourcequalitycompare" => Self::QualityCompare,
            "segmentsourceimageeditoroverlay" => Self::ImageEditorOverlay,
            "segmentsourceimageEditorcoverImage" => Self::ImageEditorCoverImage,
            "segmentsourceimageEditorcoverframe" => Self::ImageEditorCoverFrame,
            "SegmentSourceAIVideoExpand" => Self::AiVideoExpand,
            "SegmentSourceImageEditorTemp" => Self::ImageEditorTemp,
            "SegmentSourceCanvasEditCopy" => Self::CanvasEditCopy,
            "SegmentSourceCanvasEditSmartSplit" => Self::CanvasEditSmartSplit,
            "SegmentSourceCanvasEditBoxSplit" => Self::CanvasEditBoxSplit,
            "SegmentSourceTransparentCanvas" => Self::TransparentCanvas,
            _ => return None,
        })
    }
}

impl Serialize for SegmentSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SegmentSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::parse(&value).unwrap_or(Self::Other(value)))
    }
}

/// HDR type values the preview filter keeps. Any other stored int is left as stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum HdrType {
    Unknown = -1,
    None = 0,
    Sdr2020 = 2,
    Hlg = 67,
    Pq = 2084,
}

impl HdrType {
    #[must_use]
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            -1 => Some(Self::Unknown),
            0 => Some(Self::None),
            2 => Some(Self::Sdr2020),
            67 => Some(Self::Hlg),
            2084 => Some(Self::Pq),
            _ => None,
        }
    }
}

/// HDR display mode on `hdr_settings.mode`. Values outside 0 through 11 are not renamed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum HdrDisplayMode {
    None = 0,
    Default = 1,
    Sdr2Hlg = 2,
    Sdr2Pq = 3,
    Hlg2Sdr = 4,
    Hlg2Pq = 5,
    Pq2Sdr = 6,
    Pq2Hlg = 7,
    Sdr2020ToHlg = 8,
    Sdr2020ToPq = 9,
    Hlg2Sdr2020 = 10,
    Pq2Sdr2020 = 11,
}

impl HdrDisplayMode {
    #[must_use]
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::Default),
            2 => Some(Self::Sdr2Hlg),
            3 => Some(Self::Sdr2Pq),
            4 => Some(Self::Hlg2Sdr),
            5 => Some(Self::Hlg2Pq),
            6 => Some(Self::Pq2Sdr),
            7 => Some(Self::Pq2Hlg),
            8 => Some(Self::Sdr2020ToHlg),
            9 => Some(Self::Sdr2020ToPq),
            10 => Some(Self::Hlg2Sdr2020),
            11 => Some(Self::Pq2Sdr2020),
            _ => None,
        }
    }
}

/// Render index mode. The draft also stores this as two bools on the root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderIndexMode {
    #[default]
    Default,
    SyncTrack,
    Free,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_flag_keeps_one_through_six() {
        assert_eq!(TrackFlag::from_i64(2), TrackFlag::SubVideo);
        assert_eq!(TrackFlag::from_i64(99), TrackFlag::None);
        let flag: TrackFlag = serde_json::from_str("2").unwrap();
        assert_eq!(flag, TrackFlag::SubVideo);
        assert_eq!(serde_json::to_string(&flag).unwrap(), "2");
    }

    #[test]
    fn names_round_trip() {
        assert_eq!(CanvasRatio::parse("9:16"), CanvasRatio::Ratio9To16);
        assert_eq!(CanvasRatio::parse("nope").as_str(), "original");
        assert_eq!(CropRatio::parse("free").as_str(), "free");
        assert_eq!(CropRatio::parse("custom"), CropRatio::Free);
        assert_eq!(KeyframeCurve::parse("Line"), Some(KeyframeCurve::Line));
        assert_eq!(AnimType::parse("in"), Some(AnimType::In));
        assert_eq!(AnimType::parse(""), None);
        assert_eq!(HdrType::from_i32(-1), Some(HdrType::Unknown));
        assert_eq!(HdrType::from_i32(1), None);
        assert_eq!(HdrDisplayMode::from_i32(1), Some(HdrDisplayMode::Default));
        assert_eq!(ApplyType::from_i32(2), Some(ApplyType::AllVideo));
        let mode = serde_json::to_value(RenderIndexMode::SyncTrack).unwrap();
        assert_eq!(mode, "sync_track");
        let kind = serde_json::to_value(TrackType::PluginEffect).unwrap();
        assert_eq!(kind, "plugin_effect");
        assert_eq!(TrackType::Text.render_floor(), Some(14_000));
        assert_eq!(TrackType::Video.render_floor(), None);
        let future: TrackType = serde_json::from_str("\"future_lane\"").unwrap();
        assert_eq!(future, TrackType::Other("future_lane".to_owned()));
        assert_eq!(serde_json::to_value(&future).unwrap(), "future_lane");
        let source: SegmentSource = serde_json::from_str("\"SegmentSourceLater\"").unwrap();
        assert_eq!(serde_json::to_value(&source).unwrap(), "SegmentSourceLater");
    }
}
