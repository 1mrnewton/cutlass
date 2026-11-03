//! Which panel the inspector opens for the selected clip, what that panel
//! reads, and the edits only the inspector makes. Each edit is one session
//! request, so each is one undo step.

use cutlass_model::{
    Draft, MaterialEffect, MaterialId, MaterialRef, MaterialText, Micros, Segment, SegmentId,
    Track, TrackFlag, TrackType, WordColumns, segment_bloom,
};
use cutlass_session::{Request, TrimEdge};

use super::{Shell, ShellError};

/// The panel a clip opens in the inspector, from its material and its row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Panel {
    /// A picture on the main track. `still` is a photo or a color clip.
    Main {
        still: bool,
    },
    /// A picture on an overlay row.
    Overlay {
        still: bool,
    },
    Audio,
    Text,
    /// A text clip with recognized words, or one on a caption or lyric row.
    Caption,
    Effect,
    Filter,
    Adjust,
    Sticker,
    /// A clip this build has no panel for.
    Other,
}

impl Panel {
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Self::Main { still: false } => "Video",
            Self::Main { still: true } => "Photo",
            Self::Overlay { still: false } => "Overlay video",
            Self::Overlay { still: true } => "Overlay photo",
            Self::Audio => "Audio",
            Self::Text => "Text",
            Self::Caption => "Caption",
            Self::Effect => "Effect",
            Self::Filter => "Filter",
            Self::Adjust => "Adjustment",
            Self::Sticker => "Sticker",
            Self::Other => "Clip",
        }
    }

    /// The clip is drawn as a picture, so it can be placed on the canvas.
    #[must_use]
    pub fn placed(self) -> bool {
        matches!(
            self,
            Self::Main { .. } | Self::Overlay { .. } | Self::Text | Self::Caption | Self::Sticker
        )
    }

    /// The clip carries sound: a video, or a sound on its own.
    #[must_use]
    pub fn heard(self) -> bool {
        matches!(
            self,
            Self::Main { still: false } | Self::Overlay { still: false } | Self::Audio
        )
    }
}

/// How a title's outline, shadow, box, and glow look. A part with an empty
/// color is off; its numbers are what it returns with.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextLook {
    pub outline_color: String,
    /// Ems.
    pub outline_width: f64,
    pub shadow_color: String,
    /// Hundredths of an em.
    pub shadow_distance: f64,
    /// 0 to 1.
    pub shadow_blur: f64,
    pub background_color: String,
    /// 0 to 1.
    pub background_alpha: f64,
    /// 0 square to 1 round.
    pub background_round: f64,
    pub glow_color: String,
    /// 0 to 1.
    pub glow_strength: f64,
    /// 0 to 1.
    pub glow_range: f64,
}

/// What a part takes when its color is first picked and its numbers are
/// still zero. These are the phone's Styles starting values.
pub const OUTLINE_START: f64 = 0.06;
pub const SHADOW_DISTANCE_START: f64 = 8.0;
pub const SHADOW_BLUR_START: f64 = 0.3;
pub const BACKGROUND_ALPHA_START: f64 = 1.0;
pub const BACKGROUND_ROUND_START: f64 = 0.3;
pub const GLOW_STRENGTH_START: f64 = 0.6;
pub const GLOW_RANGE_START: f64 = 0.5;
/// Where a new shadow falls: down and to the right.
const SHADOW_ANGLE: f64 = -45.0;

/// The panel for `segment` on `track`. The material decides first; the row
/// decides for materials that only say what they are through their row.
#[must_use]
pub fn panel_of(draft: &Draft, track: &Track, segment: &Segment) -> Panel {
    let picture = |still: bool| {
        if track.is_main() {
            Panel::Main { still }
        } else {
            Panel::Overlay { still }
        }
    };
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => picture(video.kind == "photo"),
        Some(MaterialRef::Draft(_)) => picture(false),
        Some(MaterialRef::Color(_)) => picture(true),
        Some(MaterialRef::Audio(_)) => Panel::Audio,
        Some(MaterialRef::Text(text)) if caption(track, text) => Panel::Caption,
        Some(MaterialRef::Text(_)) => Panel::Text,
        Some(MaterialRef::VideoEffect(_) | MaterialRef::Plugin(_)) => Panel::Effect,
        _ => match track.kind {
            TrackType::Audio | TrackType::TextToVideoTextAudio => Panel::Audio,
            TrackType::Text | TrackType::TextToVideoText => Panel::Text,
            TrackType::Sticker => Panel::Sticker,
            TrackType::Effect | TrackType::PluginEffect => Panel::Effect,
            TrackType::Filter => Panel::Filter,
            TrackType::Adjust => Panel::Adjust,
            _ if draft
                .materials
                .stickers
                .iter()
                .any(|item| item.id == segment.material_id) =>
            {
                Panel::Sticker
            }
            _ => Panel::Other,
        },
    }
}

fn caption(track: &Track, text: &MaterialText) -> bool {
    matches!(
        track.flag,
        TrackFlag::TextSubtitle
            | TrackFlag::TextLyric
            | TrackFlag::VideoSubtitle
            | TrackFlag::AudioSubtitle
            | TrackFlag::AiScriptSubtitle
    ) || matches!(text.kind.as_str(), "subtitle" | "lyrics")
        || !text.words.text.is_empty()
}

fn text_look(draft: &Draft, segment: &Segment) -> Option<TextLook> {
    let Some(MaterialRef::Text(text)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    let on = |on: bool, color: &str| if on { color.to_owned() } else { String::new() };
    // No glow reads as no amounts, so a first glow color takes the starts.
    let (glow_color, glow_strength, glow_range) = segment_bloom(&draft.materials, segment)
        .and_then(MaterialEffect::bloom)
        .map_or((String::new(), 0.0, 0.0), |glow| {
            (glow.color, glow.strength, glow.range)
        });
    Some(TextLook {
        outline_color: text.border_color.clone(),
        outline_width: text.border_width,
        shadow_color: on(text.has_shadow, &text.shadow_color),
        shadow_distance: text.shadow_distance,
        shadow_blur: text.shadow_smoothing,
        background_color: on(text.background_style != 0, &text.background_color),
        background_alpha: text.background_alpha,
        background_round: text.background_round_radius,
        glow_color,
        glow_strength,
        glow_range,
    })
}

/// `value`, or `start` when a part is being turned on with nothing set.
fn starting(color: &str, value: f64, start: f64) -> f64 {
    if !color.is_empty() && value == 0.0 {
        start
    } else {
        value
    }
}

impl Shell {
    /// The panel the selected clip opens.
    #[must_use]
    pub fn panel(&self) -> Option<Panel> {
        let draft = self.draft()?;
        let id = self.selected()?;
        draft.tracks.iter().find_map(|track| {
            track
                .segments
                .iter()
                .find(|segment| &segment.id == id)
                .map(|segment| panel_of(draft, track, segment))
        })
    }

    /// The selected title's outline, shadow, box, and glow.
    #[must_use]
    pub fn text_look(&self) -> Option<TextLook> {
        text_look(self.draft()?, self.selected_segment()?)
    }

    /// The selected effect, filter, or adjustment's stored name.
    #[must_use]
    pub fn effect_name(&self) -> Option<String> {
        let draft = self.draft()?;
        let segment = self.selected_segment()?;
        let name = match draft.materials.get(segment.material_id.as_str())? {
            MaterialRef::VideoEffect(effect) => effect.name.clone(),
            MaterialRef::Effect(effect) => effect.name.clone(),
            MaterialRef::Plugin(effect) => effect.plugin_id.clone(),
            MaterialRef::Other(raw) => raw
                .fields
                .get("name")
                .and_then(|name| name.as_str())
                .unwrap_or_default()
                .to_owned(),
            _ => String::new(),
        };
        Some(if name.is_empty() {
            segment.material_id.as_str().to_owned()
        } else {
            name
        })
    }

    /// The selected effect, filter, or adjustment's strength from 0 to 1. A
    /// filter with none stored reads as the halfway default.
    #[must_use]
    pub fn effect_strength(&self) -> Option<f64> {
        let draft = self.draft()?;
        let segment = self.selected_segment()?;
        match draft.materials.get(segment.material_id.as_str())? {
            MaterialRef::VideoEffect(effect) => Some(effect.value),
            MaterialRef::Effect(effect) if effect.kind != MaterialEffect::BLOOM => {
                Some(effect.value.unwrap_or(MaterialEffect::MAPPING_VALUE))
            }
            _ => None,
        }
    }

    /// Store the selected effect, filter, or adjustment's strength. A split
    /// clip's halves get their own once one changes. One undo restores it.
    pub fn set_effect_strength(&mut self, value: f64) -> Result<(), ShellError> {
        let segment_id = self.selected_or_refuse()?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("effect"));
        self.apply(Request::SetEffectStrength {
            segment_id,
            material_id,
            value,
        })
    }

    /// Turn the selected picture to `degrees`, clockwise on the preview.
    /// One undo restores the previous turn.
    pub fn set_rotation(&mut self, degrees: f64) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a turn belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetRotation {
            segment_id,
            degrees,
        })
    }

    /// Start the selected clip at `seconds` on its own row. The length
    /// stays. One undo restores the previous start.
    pub fn set_start_seconds(&mut self, seconds: f64) -> Result<(), ShellError> {
        self.set_range_start(seconds_micros(seconds, "the start")?)
    }

    /// Make the selected clip `seconds` long by moving its end. One undo
    /// restores the previous length.
    pub fn set_duration_seconds(&mut self, seconds: f64) -> Result<(), ShellError> {
        let length = seconds_micros(seconds, "the length")?;
        if length.0 <= 0 {
            return Err(ShellError::Unavailable(
                "the length must be more than zero".to_owned(),
            ));
        }
        let segment_id = self.selected_or_refuse()?;
        self.refuse_locked(&segment_id)?;
        let start = self
            .selected_segment()
            .map(|segment| segment.target.start)
            .unwrap_or_default();
        self.apply(Request::Trim {
            segment_id,
            edge: TrimEdge::End,
            time: Micros(start.0.saturating_add(length.0)),
        })
    }

    /// Ring the selected title's letters in `color`, `width` ems wide. An
    /// empty color takes the outline off. One undo restores it.
    pub fn set_text_outline(&mut self, color: &str, width: f64) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetTextOutline {
            segment_id,
            color: color.to_owned(),
            width: starting(color, width, OUTLINE_START),
        })
    }

    /// Cast the selected title's shadow in `color`, `distance` hundredths of
    /// an em away and blurred by `blur`. The angle stays, and a first shadow
    /// falls down and right. One undo restores it.
    pub fn set_text_shadow(
        &mut self,
        color: &str,
        distance: f64,
        blur: f64,
    ) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let angle = self
            .selected_segment()
            .and_then(|segment| {
                let draft = self.draft()?;
                match draft.materials.get(segment.material_id.as_str())? {
                    MaterialRef::Text(text) if text.has_shadow && text.shadow_angle.is_finite() => {
                        Some(text.shadow_angle)
                    }
                    _ => None,
                }
            })
            .unwrap_or(SHADOW_ANGLE);
        self.apply(Request::SetTextShadow {
            segment_id,
            color: color.to_owned(),
            distance: starting(color, distance, SHADOW_DISTANCE_START),
            angle,
            blur: starting(color, blur, SHADOW_BLUR_START),
        })
    }

    /// Fill a box behind the selected title in `color` at `alpha`, its
    /// corners rounded by `round`. An empty color takes it off. One undo
    /// restores it.
    pub fn set_text_background(
        &mut self,
        color: &str,
        alpha: f64,
        round: f64,
    ) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetTextBackground {
            segment_id,
            color: color.to_owned(),
            alpha: starting(color, alpha, BACKGROUND_ALPHA_START),
            round: starting(color, round, BACKGROUND_ROUND_START),
        })
    }

    /// Make the selected title glow in `color`. A split clip's halves glow
    /// on their own once one changes. One undo restores it.
    pub fn set_text_glow(
        &mut self,
        color: &str,
        strength: f64,
        range: f64,
    ) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let glow_id = MaterialId::new(self.fresh("glow"));
        self.apply(Request::SetTextGlow {
            segment_id,
            glow_id,
            color: color.to_owned(),
            strength: starting(color, strength, GLOW_STRENGTH_START),
            range: starting(color, range, GLOW_RANGE_START),
        })
    }

    /// Replace word `index` of the selected caption. The word keeps its
    /// times, and the caption reads the words joined by spaces. One undo
    /// restores the word.
    pub fn set_caption_word(&mut self, index: usize, word: &str) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let word = word.trim();
        if word.is_empty() {
            return Err(ShellError::Unavailable(
                "a word needs at least one letter".to_owned(),
            ));
        }
        let (material_id, mut words) = self.caption_words(&segment_id)?;
        let Some(slot) = words.text.get_mut(index) else {
            return Err(ShellError::Unavailable(
                "the caption has no such word".to_owned(),
            ));
        };
        if slot == word {
            return Ok(());
        }
        word.clone_into(slot);
        self.apply(Request::SetWords { material_id, words })
    }

    fn caption_words(
        &self,
        segment_id: &SegmentId,
    ) -> Result<(MaterialId, WordColumns), ShellError> {
        let draft = self.session_draft()?;
        let found = super::placed(draft, segment_id).and_then(|(segment, _)| {
            match draft.materials.get(segment.material_id.as_str())? {
                MaterialRef::Text(text) => Some((text.id.clone(), text.words.clone())),
                _ => None,
            }
        });
        found.ok_or_else(|| ShellError::Unavailable("words belong on a caption".to_owned()))
    }

    fn selected_or_refuse(&self) -> Result<SegmentId, ShellError> {
        self.selected().cloned().ok_or_else(|| {
            if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            }
        })
    }
}

fn seconds_micros(seconds: f64, what: &str) -> Result<Micros, ShellError> {
    let micros = (seconds * 1_000_000.0).round();
    if !micros.is_finite() || micros < 0.0 || micros > 1e15 {
        return Err(ShellError::Unavailable(format!(
            "{what} must be zero or more seconds"
        )));
    }
    // Checked against a bound far inside the i64 range above.
    #[expect(clippy::cast_possible_truncation, reason = "bounded above")]
    Ok(Micros(micros as i64))
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialColorConfig, MaterialEffect, MaterialText, MaterialVideo,
        MaterialVideoEffect, Micros, RawMaterial, Segment, SegmentId, TimeRange, Track,
        TrackAttribute, TrackFlag, TrackType,
    };
    use cutlass_store::Folder;

    use super::super::tests::{Temp, shell};
    use super::{Panel, TextLook};
    use crate::shell::Shell;

    fn row(id: &str, kind: TrackType, flag: TrackFlag, segment: &str, material: &str) -> Track {
        let mut track = Track::main(id);
        track.kind = kind;
        track.flag = flag;
        track.segments.push(Segment::new(
            segment,
            material,
            TimeRange::new(1_000_000, 2_000_000),
        ));
        track
    }

    /// One clip of each kind, each on its own row from one to three seconds.
    fn every_kind(temp: &Temp) -> Shell {
        let mut draft = Draft::empty("draft");
        draft.materials.videos.push(MaterialVideo::new("movie"));
        let mut photo = MaterialVideo::new("photo");
        photo.kind = "photo".to_owned();
        draft.materials.videos.push(photo);
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "color".into(),
            solid_color: "0xFF336699".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.materials.audios.push(MaterialAudio {
            id: "song".into(),
            ..MaterialAudio::default()
        });
        let mut title = MaterialText::new("title");
        title.content = "Title".to_owned();
        draft.materials.texts.push(title);
        let mut caption = MaterialText::new("caption");
        caption.content = "one two".to_owned();
        caption.words.text = vec!["one".to_owned(), "two".to_owned()];
        caption.words.start_time = vec![0, 500];
        caption.words.end_time = vec![400, 900];
        draft.materials.texts.push(caption);
        let mut lyric = MaterialText::new("lyric");
        lyric.content = "la".to_owned();
        draft.materials.texts.push(lyric);
        draft.materials.video_effects.push(MaterialVideoEffect {
            id: "shake".into(),
            name: "Shake".to_owned(),
            effect_id: "recipe/shake".to_owned(),
            value: 1.0,
            ..MaterialVideoEffect::default()
        });
        let mut filter = MaterialEffect::new("warm");
        filter.kind = "filter".to_owned();
        filter.name = "Warm".to_owned();
        draft.materials.effects.push(filter);
        let mut adjust = MaterialEffect::new("bright");
        adjust.kind = "adjust".to_owned();
        draft.materials.effects.push(adjust);
        draft.materials.stickers.push(RawMaterial {
            id: "star".into(),
            kind: "sticker".to_owned(),
            ..RawMaterial::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "movie", TimeRange::new(0, 4_000_000)));
        draft.tracks.push(row(
            "o1",
            TrackType::Video,
            TrackFlag::SubVideo,
            "overlay",
            "movie",
        ));
        draft.tracks.push(row(
            "o2",
            TrackType::Video,
            TrackFlag::SubVideo,
            "still",
            "photo",
        ));
        draft.tracks.push(row(
            "o3",
            TrackType::Video,
            TrackFlag::SubVideo,
            "solid",
            "color",
        ));
        draft
            .tracks
            .push(row("a", TrackType::Audio, TrackFlag::None, "sound", "song"));
        draft
            .tracks
            .push(row("t", TrackType::Text, TrackFlag::None, "words", "title"));
        draft.tracks.push(row(
            "c",
            TrackType::Text,
            TrackFlag::None,
            "subtitle",
            "caption",
        ));
        draft.tracks.push(row(
            "l",
            TrackType::Text,
            TrackFlag::TextLyric,
            "song-line",
            "lyric",
        ));
        draft
            .tracks
            .push(row("e", TrackType::Effect, TrackFlag::None, "fx", "shake"));
        draft
            .tracks
            .push(row("f", TrackType::Filter, TrackFlag::None, "look", "warm"));
        draft.tracks.push(row(
            "j",
            TrackType::Adjust,
            TrackFlag::None,
            "tone",
            "bright",
        ));
        draft.tracks.push(row(
            "s",
            TrackType::Sticker,
            TrackFlag::None,
            "sticker",
            "star",
        ));
        draft.tracks.push(row(
            "x",
            TrackType::Composition,
            TrackFlag::None,
            "odd",
            "nowhere",
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop
    }

    fn pick(desktop: &mut Shell, id: &str) {
        desktop.select(Some(SegmentId::new(id)));
    }

    fn lock(desktop: &mut Shell, track: &str) {
        let id = desktop
            .draft()
            .unwrap()
            .tracks
            .iter()
            .find(|row| row.id.as_str() == track)
            .unwrap()
            .id
            .clone();
        desktop.toggle_attribute(id, TrackAttribute::LOCK).unwrap();
    }

    #[test]
    fn each_kind_of_clip_opens_its_own_panel() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        assert_eq!(desktop.panel(), None);
        for (id, panel) in [
            ("main", Panel::Main { still: false }),
            ("overlay", Panel::Overlay { still: false }),
            ("still", Panel::Overlay { still: true }),
            ("solid", Panel::Overlay { still: true }),
            ("sound", Panel::Audio),
            ("words", Panel::Text),
            ("subtitle", Panel::Caption),
            ("song-line", Panel::Caption),
            ("fx", Panel::Effect),
            ("look", Panel::Filter),
            ("tone", Panel::Adjust),
            ("sticker", Panel::Sticker),
            ("odd", Panel::Other),
        ] {
            pick(&mut desktop, id);
            assert_eq!(desktop.panel(), Some(panel), "{id}");
        }
        assert!(Panel::Main { still: false }.heard());
        assert!(!Panel::Overlay { still: true }.heard());
        assert!(Panel::Sticker.placed());
        assert!(!Panel::Filter.placed());
        pick(&mut desktop, "fx");
        assert_eq!(desktop.effect_name().as_deref(), Some("Shake"));
        pick(&mut desktop, "look");
        assert_eq!(desktop.effect_name().as_deref(), Some("Warm"));
        pick(&mut desktop, "tone");
        assert_eq!(desktop.effect_name().as_deref(), Some("bright"));
        pick(&mut desktop, "sound");
        assert_eq!(desktop.effect_name().as_deref(), Some("song"));
    }

    #[test]
    fn timing_moves_and_trims_any_clip_as_one_step_each() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        for id in ["overlay", "sound", "words", "fx", "look", "tone", "sticker"] {
            pick(&mut desktop, id);
            let before = desktop.draft().unwrap().clone();
            desktop.set_start_seconds(0.5).unwrap();
            let inspection = desktop.inspection().unwrap();
            assert_eq!(inspection.start, Micros(500_000), "{id}");
            assert_eq!(inspection.duration, Micros(2_000_000), "{id}");
            desktop.set_duration_seconds(1.25).unwrap();
            let inspection = desktop.inspection().unwrap();
            assert_eq!(inspection.start, Micros(500_000), "{id}");
            assert_eq!(inspection.duration, Micros(1_250_000), "{id}");
            desktop.undo().unwrap();
            assert_eq!(desktop.inspection().unwrap().duration, Micros(2_000_000));
            desktop.undo().unwrap();
            assert_eq!(desktop.draft().unwrap(), &before, "{id}");
        }
        pick(&mut desktop, "fx");
        let before = desktop.draft().unwrap().clone();
        for refused in [
            desktop.set_duration_seconds(0.0),
            desktop.set_duration_seconds(-1.0),
            desktop.set_duration_seconds(f64::NAN),
            desktop.set_start_seconds(-0.5),
            desktop.set_start_seconds(f64::INFINITY),
        ] {
            assert!(refused.is_err());
        }
        assert_eq!(desktop.draft().unwrap(), &before);
        lock(&mut desktop, "e");
        assert_eq!(
            desktop.set_start_seconds(0.0).unwrap_err().to_string(),
            "the track is locked"
        );
        assert_eq!(
            desktop.set_duration_seconds(1.0).unwrap_err().to_string(),
            "the track is locked"
        );
    }

    #[test]
    fn a_picture_or_sticker_turns_through_one_request_and_a_sound_does_not() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        for id in ["main", "overlay", "words", "sticker"] {
            pick(&mut desktop, id);
            desktop.set_rotation(30.0).unwrap();
            assert_eq!(desktop.inspection().unwrap().rotation, 30.0, "{id}");
            desktop.set_position(0.25, -0.5).unwrap();
            desktop.set_opacity(0.5).unwrap();
            desktop.undo().unwrap();
            desktop.undo().unwrap();
            let inspection = desktop.inspection().unwrap();
            assert_eq!(inspection.rotation, 30.0, "{id}");
            assert_eq!(inspection.opacity, 1.0, "{id}");
            desktop.undo().unwrap();
            assert_eq!(desktop.inspection().unwrap().rotation, 0.0, "{id}");
        }
        for id in ["sound", "fx", "look"] {
            pick(&mut desktop, id);
            let before = desktop.draft().unwrap().clone();
            assert_eq!(
                desktop.set_rotation(30.0).unwrap_err().to_string(),
                "a turn belongs on a picture"
            );
            assert_eq!(desktop.draft().unwrap(), &before);
        }
        pick(&mut desktop, "main");
        assert!(desktop.set_rotation(f64::NAN).is_err());
        lock(&mut desktop, "s");
        pick(&mut desktop, "sticker");
        assert_eq!(
            desktop.set_rotation(90.0).unwrap_err().to_string(),
            "the track is locked"
        );
    }

    #[test]
    fn an_effect_or_filter_strength_is_one_step_and_a_title_has_none() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "fx");
        assert_eq!(desktop.effect_strength(), Some(1.0));
        let before = desktop.draft().unwrap().clone();
        desktop.set_effect_strength(0.4).unwrap();
        assert_eq!(desktop.effect_strength(), Some(0.4));
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap(), &before);

        pick(&mut desktop, "look");
        assert_eq!(desktop.effect_strength(), Some(0.5));
        desktop.set_effect_strength(0.9).unwrap();
        assert_eq!(desktop.effect_strength(), Some(0.9));
        assert_eq!(
            desktop.set_effect_strength(1.5).unwrap_err().to_string(),
            "the strength must be from zero to one"
        );
        assert_eq!(desktop.effect_strength(), Some(0.9));

        pick(&mut desktop, "words");
        assert_eq!(desktop.effect_strength(), None);
        assert_eq!(
            desktop.set_effect_strength(0.5).unwrap_err().to_string(),
            "a strength belongs on an effect, filter, or adjustment"
        );
        lock(&mut desktop, "f");
        pick(&mut desktop, "look");
        assert_eq!(
            desktop.set_effect_strength(0.1).unwrap_err().to_string(),
            "the track is locked"
        );
    }

    #[test]
    fn a_sound_panel_sets_volume_fade_and_speed_one_step_each() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "sound");
        let before = desktop.draft().unwrap().clone();
        desktop.set_volume(0.5).unwrap();
        desktop.set_fade_in(0.25).unwrap();
        desktop.set_fade_out(0.5).unwrap();
        desktop.set_speed(2.0).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.volume, 0.5);
        assert_eq!(
            (inspection.fade_in, inspection.fade_out),
            (250_000, 500_000)
        );
        assert_eq!(inspection.speed, 2.0);
        for _ in 0..4 {
            desktop.undo().unwrap();
        }
        assert_eq!(desktop.draft().unwrap(), &before);
        pick(&mut desktop, "words");
        assert!(desktop.set_fade_in(1.0).is_err());
    }

    #[test]
    fn a_title_takes_outline_shadow_box_and_glow_with_starting_amounts() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "words");
        let before = desktop.draft().unwrap().clone();
        assert_eq!(
            desktop.text_look(),
            Some(TextLook {
                background_alpha: 1.0,
                ..TextLook::default()
            })
        );

        desktop.set_text_outline("#FF0000", 0.0).unwrap();
        desktop.set_text_shadow("#000000", 0.0, 0.0).unwrap();
        desktop.set_text_background("#FFFFFF", 0.0, 0.0).unwrap();
        desktop.set_text_glow("#00FF00", 0.0, 0.0).unwrap();
        let look = desktop.text_look().unwrap();
        assert_eq!(look.outline_color, "#FF0000");
        assert_eq!(look.outline_width, super::OUTLINE_START);
        assert_eq!(look.shadow_color, "#000000");
        assert_eq!(look.shadow_distance, super::SHADOW_DISTANCE_START);
        assert_eq!(look.shadow_blur, super::SHADOW_BLUR_START);
        assert_eq!(look.background_color, "#FFFFFF");
        assert_eq!(look.background_alpha, super::BACKGROUND_ALPHA_START);
        assert_eq!(look.background_round, super::BACKGROUND_ROUND_START);
        assert_eq!(look.glow_color, "#00FF00");
        assert_eq!(look.glow_strength, super::GLOW_STRENGTH_START);
        assert_eq!(look.glow_range, super::GLOW_RANGE_START);

        desktop.set_text_shadow("#000000", 20.0, 0.5).unwrap();
        let text = desktop.draft().unwrap().materials.texts[0].clone();
        assert_eq!(text.shadow_angle, -45.0);
        assert_eq!(text.shadow_distance, 20.0);
        desktop.set_text_outline("", 0.1).unwrap();
        let look = desktop.text_look().unwrap();
        assert_eq!(look.outline_color, "");
        assert_eq!(look.outline_width, 0.1);

        for _ in 0..6 {
            desktop.undo().unwrap();
        }
        assert_eq!(desktop.draft().unwrap(), &before);

        let refused = desktop.set_text_outline("#FF0000", 1.0);
        assert_eq!(desktop.draft().unwrap(), &before);
        assert!(refused.is_err());
        assert!(desktop.set_text_glow("#00FF00", 2.0, 0.5).is_err());
        assert!(desktop.set_text_background("nope", 1.0, 0.0).is_err());
        assert_eq!(desktop.draft().unwrap(), &before);

        pick(&mut desktop, "main");
        assert_eq!(desktop.text_look(), None);
        assert_eq!(
            desktop
                .set_text_outline("#FF0000", 0.1)
                .unwrap_err()
                .to_string(),
            "a text style belongs on text"
        );
        lock(&mut desktop, "t");
        pick(&mut desktop, "words");
        assert_eq!(
            desktop
                .set_text_glow("#00FF00", 0.5, 0.5)
                .unwrap_err()
                .to_string(),
            "the track is locked"
        );
    }

    #[test]
    fn a_caption_word_keeps_its_time_and_is_one_step() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "subtitle");
        let before = desktop.draft().unwrap().clone();
        desktop.set_caption_word(1, "  too ").unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.words.text, ["one", "too"]);
        assert_eq!(inspection.words.start_time, [0, 500]);
        assert_eq!(inspection.words.end_time, [400, 900]);
        assert_eq!(inspection.text.as_deref(), Some("one too"));
        let revision = desktop.revision();
        desktop.set_caption_word(1, "too").unwrap();
        assert_eq!(desktop.revision(), revision);
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap(), &before);

        for (index, word) in [(1, ""), (1, "   "), (2, "three")] {
            assert!(desktop.set_caption_word(index, word).is_err());
            assert_eq!(desktop.draft().unwrap(), &before);
        }
        pick(&mut desktop, "sound");
        assert!(desktop.set_caption_word(0, "x").is_err());
        lock(&mut desktop, "c");
        pick(&mut desktop, "subtitle");
        assert_eq!(
            desktop.set_caption_word(0, "won").unwrap_err().to_string(),
            "the track is locked"
        );
    }
}
