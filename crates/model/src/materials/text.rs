use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::caption::{CaptionTemplateInfo, Kept, SubtitleKeywords, keywords_in_ranges};
use super::words::WordColumns;
use crate::id::MaterialId;

fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialText {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub base_content: String,
    #[serde(default)]
    pub text_color: String,
    #[serde(default = "one")]
    pub text_alpha: f64,
    #[serde(default)]
    pub font_size: f64,
    #[serde(default)]
    pub font_name: String,
    #[serde(default)]
    pub font_path: String,
    /// Where a face that varies is set along its axes, as `wght=650,wdth=80`:
    /// each axis by its four-letter tag and a value in the axis's own
    /// units. Empty leaves the face as its file, or its named style, has
    /// it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub font_variations: String,
    #[serde(default)]
    pub font_title: String,
    #[serde(default)]
    pub alignment: i32,
    #[serde(default)]
    pub typesetting: i32,
    #[serde(default)]
    pub line_spacing: f64,
    #[serde(default)]
    pub letter_spacing: f64,
    #[serde(default)]
    pub line_max_width: f64,
    #[serde(default)]
    pub background_color: String,
    #[serde(default = "one")]
    pub background_alpha: f64,
    #[serde(default)]
    pub background_style: i32,
    #[serde(default)]
    pub background_round_radius: f64,
    #[serde(default)]
    pub background_width: f64,
    #[serde(default)]
    pub background_height: f64,
    #[serde(default)]
    pub border_color: String,
    #[serde(default = "one")]
    pub border_alpha: f64,
    #[serde(default)]
    pub border_width: f64,
    #[serde(default)]
    pub has_shadow: bool,
    #[serde(default)]
    pub shadow_color: String,
    #[serde(default)]
    pub shadow_alpha: f64,
    #[serde(default)]
    pub shadow_angle: f64,
    #[serde(default)]
    pub shadow_distance: f64,
    #[serde(default)]
    pub shadow_smoothing: f64,
    #[serde(default = "one")]
    pub global_alpha: f64,
    #[serde(default)]
    pub bold_width: f64,
    #[serde(default)]
    pub italic_degree: i32,
    #[serde(default)]
    pub underline: bool,
    #[serde(default)]
    pub is_rich_text: bool,
    #[serde(default)]
    pub text_size: i32,
    #[serde(default)]
    pub check_flag: i32,
    #[serde(default)]
    pub words: WordColumns,
    #[serde(default)]
    pub current_words: WordColumns,
    #[serde(default)]
    pub text_to_audio_ids: Vec<String>,
    #[serde(default)]
    pub fixed_width: f64,
    #[serde(default)]
    pub fixed_height: f64,
    #[serde(default)]
    pub group_id: String,
    #[serde(default)]
    pub sub_type: i32,
    #[serde(default)]
    pub recognize_type: i32,
    /// The speech job that made these words.
    #[serde(default, skip_serializing_if = "Kept::is_absent")]
    pub recognize_task_id: Kept<String>,
    /// The spoken language the words were recognized in.
    #[serde(default, skip_serializing_if = "Kept::is_absent")]
    pub language: Kept<String>,
    /// The words in another language, as a line.
    #[serde(default, skip_serializing_if = "Kept::is_absent")]
    pub translate_text: Kept<String>,
    /// The words in another language, with their times.
    #[serde(default, skip_serializing_if = "Kept::is_absent")]
    pub translate_words: Kept<WordColumns>,
    /// Marked spans of `content`. A draft from the reference editor marks
    /// keywords here; [`Self::adopt_subtitle_keywords`] turns them into
    /// `caption_keywords`.
    #[serde(default, skip_serializing_if = "Kept::is_absent")]
    pub subtitle_keywords: Kept<SubtitleKeywords>,
    /// The caption template this caption was styled with.
    #[serde(default, skip_serializing_if = "Kept::is_absent")]
    pub caption_template_info: Kept<CaptionTemplateInfo>,
    /// The [`Bubble`] the background is shaped as, by its key; empty is a
    /// plain box. The app's own: the reference editor does not read it.
    #[serde(
        default,
        rename = "cutlass_bubble",
        skip_serializing_if = "String::is_empty"
    )]
    pub bubble: String,
    /// The caption style its words are drawn in, by key; empty is plain.
    /// The app's own, like `bubble`.
    #[serde(
        default,
        rename = "cutlass_caption_style",
        skip_serializing_if = "String::is_empty"
    )]
    pub caption_style: String,
    /// Words the caption marks, by index into its word columns: drawn in
    /// the style's spoken look while they show. The app's own.
    #[serde(
        default,
        rename = "cutlass_caption_keywords",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub caption_keywords: Vec<usize>,
    /// Milliseconds each letter of a word starts after the one before it,
    /// so a word's look runs across its letters; 0 changes them together.
    /// The app's own.
    #[serde(
        default,
        rename = "cutlass_caption_stagger_ms",
        skip_serializing_if = "is_zero"
    )]
    pub caption_stagger_ms: i64,
    /// The background is a box around each line, each as wide as its own
    /// letters, rather than one box around them all. The app's own.
    #[serde(
        default,
        rename = "cutlass_background_lines",
        skip_serializing_if = "is_false"
    )]
    pub background_lines: bool,
    /// How far the background box sits from the letters, to the right, in
    /// ems. The app's own.
    #[serde(
        default,
        rename = "cutlass_background_offset_x",
        skip_serializing_if = "is_nothing"
    )]
    pub background_offset_x: f64,
    /// How far the background box sits from the letters, up, in ems. The
    /// app's own.
    #[serde(
        default,
        rename = "cutlass_background_offset_y",
        skip_serializing_if = "is_nothing"
    )]
    pub background_offset_y: f64,
    /// The lines bend along an arc. The app's own.
    #[serde(default, rename = "cutlass_curve_on", skip_serializing_if = "is_false")]
    pub curve_on: bool,
    /// How far the lines bend: the degrees of arc the title's width takes,
    /// up to [`MAX_CURVE`] either way. Above zero they arch up in the
    /// middle, below zero they dip. Kept while the curve is off. The app's
    /// own.
    #[serde(default, rename = "cutlass_curve", skip_serializing_if = "is_nothing")]
    pub curve: f64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// The most a title's lines bend, in degrees of arc either way: a whole
/// circle.
pub const MAX_CURVE: f64 = 360.0;

/// The most axes a title sets a face along.
pub const MAX_FONT_AXES: usize = 16;

impl MaterialText {
    /// The axes [`Self::font_variations`] names, each with its value, in
    /// the order written. A part that is not a tag of four printable
    /// letters and a number is passed over, and so is anything past
    /// [`MAX_FONT_AXES`].
    #[must_use]
    pub fn font_axes(&self) -> Vec<([u8; 4], f32)> {
        read_font_axes(&self.font_variations)
    }

    /// Resource-mapping fallbacks. An absent key stays at 1, and so does [`Default`].
    pub const MAPPING_TEXT_ALPHA: f64 = 0.0;
    pub const MAPPING_BACKGROUND_ALPHA: f64 = 0.0;
    pub const MAPPING_BORDER_ALPHA: f64 = 0.0;
    pub const MAPPING_GLOBAL_ALPHA: f64 = 0.0;

    pub fn new(id: impl Into<MaterialId>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// Mark the words `subtitle_keywords` spans, when this caption marks none
    /// of its own yet. A caption that already has `caption_keywords` keeps
    /// them. `subtitle_keywords` itself is left as read.
    pub fn adopt_subtitle_keywords(&mut self) {
        if !self.caption_keywords.is_empty() {
            return;
        }
        let Some(ranges) = self
            .subtitle_keywords
            .get()
            .and_then(|keywords| keywords.range.get())
        else {
            return;
        };
        self.caption_keywords = keywords_in_ranges(&self.content, &self.words, ranges);
    }
}

/// Serde's skip test for a switch that is off.
#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_false(value: &bool) -> bool {
    !*value
}

/// Serde's skip test for an amount of nothing.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_nothing(value: &f64) -> bool {
    *value == 0.0
}

/// Serde's skip test for a zero stagger.
#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_zero(value: &i64) -> bool {
    *value == 0
}

impl Default for MaterialText {
    fn default() -> Self {
        Self {
            id: MaterialId::default(),
            kind: String::new(),
            name: String::new(),
            content: String::new(),
            base_content: String::new(),
            text_color: String::new(),
            text_alpha: 1.0,
            font_size: 0.0,
            font_name: String::new(),
            font_path: String::new(),
            font_variations: String::new(),
            font_title: String::new(),
            alignment: 0,
            typesetting: 0,
            line_spacing: 0.0,
            letter_spacing: 0.0,
            line_max_width: 0.0,
            background_color: String::new(),
            background_alpha: 1.0,
            background_style: 0,
            background_round_radius: 0.0,
            background_width: 0.0,
            background_height: 0.0,
            border_color: String::new(),
            border_alpha: 1.0,
            border_width: 0.0,
            has_shadow: false,
            shadow_color: String::new(),
            shadow_alpha: 0.0,
            shadow_angle: 0.0,
            shadow_distance: 0.0,
            shadow_smoothing: 0.0,
            global_alpha: 1.0,
            bold_width: 0.0,
            italic_degree: 0,
            underline: false,
            is_rich_text: false,
            text_size: 0,
            check_flag: 0,
            words: WordColumns::default(),
            current_words: WordColumns::default(),
            text_to_audio_ids: Vec::new(),
            fixed_width: 0.0,
            fixed_height: 0.0,
            group_id: String::new(),
            sub_type: 0,
            recognize_type: 0,
            recognize_task_id: Kept::Absent,
            language: Kept::Absent,
            translate_text: Kept::Absent,
            translate_words: Kept::Absent,
            subtitle_keywords: Kept::Absent,
            caption_template_info: Kept::Absent,
            bubble: String::new(),
            caption_style: String::new(),
            caption_keywords: Vec::new(),
            caption_stagger_ms: 0,
            background_lines: false,
            background_offset_x: 0.0,
            background_offset_y: 0.0,
            curve_on: false,
            curve: 0.0,
            rest: Map::new(),
        }
    }
}

/// The keys a caption's style may have, as `cutlass_captions` draws them.
/// How much a bold letter's edge grows when its face has no bold of its
/// own, as a share of the text size.
pub const TEXT_BOLD_WIDTH: f64 = 0.04;
/// How many degrees an italic letter leans when its face has no italic.
pub const TEXT_ITALIC_DEGREE: i32 = 12;

pub const CAPTION_STYLE_KEYS: [&str; 10] = [
    "classic",
    "karaoke",
    "highlight",
    "pop",
    "word-by-word",
    "one-word",
    "neon",
    "bold-box",
    "fade",
    "rise",
];

/// A shape a title's background takes instead of a plain box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bubble {
    /// A box with fully round ends.
    Pill,
    /// A round box with a tail down to the left, as someone speaks.
    Speech,
    /// A cloud with two small puffs trailing down to the left.
    Thought,
    /// A spiky star around the letters.
    Burst,
    /// A strip with notched ends.
    Banner,
}

impl Bubble {
    /// Every bubble, in the order the panel shows them.
    pub const ALL: [Self; 5] = [
        Self::Pill,
        Self::Speech,
        Self::Thought,
        Self::Burst,
        Self::Banner,
    ];

    /// The key a title stores it by.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::Pill => "pill",
            Self::Speech => "speech",
            Self::Thought => "thought",
            Self::Burst => "burst",
            Self::Banner => "banner",
        }
    }

    /// The name the panel shows.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pill => "Pill",
            Self::Speech => "Speech",
            Self::Thought => "Thought",
            Self::Burst => "Burst",
            Self::Banner => "Banner",
        }
    }

    /// The bubble stored as `key`, if it is one.
    #[must_use]
    pub fn of(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|bubble| bubble.key() == key)
    }
}

/// The axes a string such as `wght=650,wdth=80` names, each with its
/// value. A part that does not read is passed over.
#[must_use]
pub fn read_font_axes(written: &str) -> Vec<([u8; 4], f32)> {
    written
        .split(',')
        .filter_map(|part| {
            let (tag, value) = part.split_once('=')?;
            let tag = <[u8; 4]>::try_from(tag.trim().as_bytes()).ok()?;
            let value: f32 = value.trim().parse().ok()?;
            (tag.iter().all(|byte| (0x20..0x7F).contains(byte)) && value.is_finite())
                .then_some((tag, value))
        })
        .take(MAX_FONT_AXES)
        .collect()
}

/// `axes` written as [`MaterialText::font_variations`] keeps them: each
/// value to a hundredth, which is finer than any axis is drawn.
#[must_use]
pub fn write_font_axes(axes: &[([u8; 4], f32)]) -> String {
    axes.iter()
        .filter(|(tag, value)| {
            tag.iter().all(|byte| (0x20..0x7F).contains(byte)) && value.is_finite()
        })
        .take(MAX_FONT_AXES)
        .map(|(tag, value)| {
            let rounded = (f64::from(*value) * 100.0).round() / 100.0;
            format!("{}={rounded}", String::from_utf8_lossy(tag))
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_alphas_stay_opaque() {
        let text = MaterialText::default();
        let parsed: MaterialText = serde_json::from_str("{}").unwrap();
        assert_eq!(text, parsed);
        assert_eq!(text.text_alpha, 1.0);
        assert_eq!(text.background_alpha, 1.0);
        assert_eq!(text.border_alpha, 1.0);
        assert_eq!(text.global_alpha, 1.0);
        assert_ne!(text.text_alpha, MaterialText::MAPPING_TEXT_ALPHA);
        assert_ne!(
            text.background_alpha,
            MaterialText::MAPPING_BACKGROUND_ALPHA
        );
        assert_ne!(text.border_alpha, MaterialText::MAPPING_BORDER_ALPHA);
        assert_ne!(text.global_alpha, MaterialText::MAPPING_GLOBAL_ALPHA);
    }

    #[test]
    fn a_bubble_is_kept_by_its_key_and_left_out_when_there_is_none() {
        let mut text = MaterialText::new("t");
        assert!(!serde_json::to_string(&text).unwrap().contains("bubble"));
        text.bubble = Bubble::Speech.key().to_owned();
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains(r#""cutlass_bubble":"speech""#));
        let back: MaterialText = serde_json::from_str(&json).unwrap();
        assert_eq!(Bubble::of(&back.bubble), Some(Bubble::Speech));
        assert!(back.rest.is_empty());
        assert_eq!(Bubble::of("cloud"), None);
    }

    #[test]
    fn caption_speech_fields_are_named_and_written_back_as_read() {
        let raw = serde_json::json!({
            "id": "t",
            "type": "subtitle",
            "content": "so very good",
            "language": "en",
            "recognize_task_id": "task-1",
            "translate_text": null,
            "translate_words": {"start_time": [0], "end_time": [100], "text": ["tres"], "tag": 1},
            "subtitle_keywords": {"range": [{"location": 3, "length": 4, "source_type": 2}], "x": true},
            "caption_template_info": {"resource_id": "7", "resource_name": "Bold", "path": "/p", "is_new": false},
            "words": {"start_time": [0, 100, 200], "end_time": [100, 200, 300], "text": ["so", "very", "good"]}
        });
        let text: MaterialText = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(text.language.get().map(String::as_str), Some("en"));
        assert_eq!(
            text.recognize_task_id.get().map(String::as_str),
            Some("task-1")
        );
        assert_eq!(text.translate_text, Kept::Null);
        assert_eq!(text.translate_words.get().unwrap().text, ["tres"]);
        let ranges = text.subtitle_keywords.get().unwrap().range.get().unwrap();
        assert_eq!((ranges[0].location, ranges[0].length), (3, 4));
        for key in [
            "language",
            "recognize_task_id",
            "translate_text",
            "translate_words",
            "subtitle_keywords",
            "caption_template_info",
        ] {
            assert!(!text.rest.contains_key(key), "{key}");
        }
        let back = serde_json::to_value(&text).unwrap();
        for key in [
            "language",
            "recognize_task_id",
            "translate_text",
            "translate_words",
            "subtitle_keywords",
            "caption_template_info",
        ] {
            assert_eq!(back[key], raw[key], "{key}");
        }

        // A draft without them writes none of them.
        let plain = serde_json::to_value(MaterialText::new("p")).unwrap();
        for key in [
            "language",
            "recognize_task_id",
            "translate_text",
            "translate_words",
            "subtitle_keywords",
            "caption_template_info",
        ] {
            assert!(plain.get(key).is_none(), "{key}");
        }
    }

    #[test]
    fn subtitle_keywords_become_caption_keywords_unless_some_are_marked() {
        let mut text: MaterialText = serde_json::from_value(serde_json::json!({
            "content": "so very good",
            "subtitle_keywords": {"range": [{"location": 3, "length": 4}]},
            "words": {"start_time": [0, 100, 200], "end_time": [100, 200, 300], "text": ["so", "very", "good"]}
        }))
        .unwrap();
        text.adopt_subtitle_keywords();
        assert_eq!(text.caption_keywords, [1]);
        text.caption_keywords = vec![2];
        text.adopt_subtitle_keywords();
        assert_eq!(text.caption_keywords, [2]);

        let mut none = MaterialText::new("n");
        none.adopt_subtitle_keywords();
        assert!(none.caption_keywords.is_empty());
        none.subtitle_keywords = Kept::Null;
        none.adopt_subtitle_keywords();
        assert!(none.caption_keywords.is_empty());
    }

    #[test]
    fn a_titles_axes_are_written_and_read_back() {
        use super::{MAX_FONT_AXES, read_font_axes, write_font_axes};
        let mut text = MaterialText::new("title");
        assert!(text.font_axes().is_empty());
        text.font_variations = "wght=650,wdth=80".to_owned();
        assert_eq!(text.font_axes(), [(*b"wght", 650.0), (*b"wdth", 80.0)]);
        // Written to a hundredth, and read back the same.
        let written = write_font_axes(&[(*b"wght", 650.0), (*b"slnt", -7.456), (*b"GRAD", 0.5)]);
        assert_eq!(written, "wght=650,slnt=-7.46,GRAD=0.5");
        assert_eq!(
            read_font_axes(&written),
            [(*b"wght", 650.0), (*b"slnt", -7.46), (*b"GRAD", 0.5)]
        );
        // What does not read is passed over: a tag of the wrong length, a
        // value that is no number or no finite one, a part with no sign.
        assert_eq!(
            read_font_axes(" wght = 700 ,wd=3,wdth=x,opsz=inf,slnt,=4,ital=1"),
            [(*b"wght", 700.0), (*b"ital", 1.0)]
        );
        assert_eq!(
            write_font_axes(&[(*b"wght", f32::NAN), ([0, 1, 2, 3], 4.0)]),
            ""
        );
        assert!(read_font_axes("").is_empty());
        // No more axes than a face is ever set along, and a long string
        // is read in time.
        let many = "wght=1,".repeat(100_000);
        let started = std::time::Instant::now();
        assert_eq!(read_font_axes(&many).len(), MAX_FONT_AXES);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        let axes = vec![(*b"wght", 1.0); 100];
        assert_eq!(write_font_axes(&axes).matches('=').count(), MAX_FONT_AXES);
        // The field is left out of a draft that does not use it, and a
        // draft without it reads.
        let plain = MaterialText::new("title");
        let stored = serde_json::to_string(&plain).expect("stored");
        assert!(!stored.contains("font_variations"));
        let read: MaterialText = serde_json::from_str(&stored).expect("read");
        assert!(read.font_variations.is_empty());
        let stored = serde_json::to_string(&text).expect("stored");
        assert!(stored.contains(r#""font_variations":"wght=650,wdth=80""#));
    }
}
