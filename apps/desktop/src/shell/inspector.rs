//! The inspector's edits on the selected clip. The model each shell shares
//! (panels, text look, decibels, letter case) is in `cutlass_editor`. Each
//! edit is one session request, so each is one undo step.

use cutlass_editor::{
    BACKGROUND_ALPHA_START, BACKGROUND_ROUND_START, GLOW_RANGE_START, GLOW_STRENGTH_START,
    OUTLINE_START, SHADOW_BLUR_START, SHADOW_DISTANCE_START, panel_of, starting, text_look,
};
pub use cutlass_editor::{LetterCase, Panel, SILENT_DB, TextLook, gain_of_db, volume_db};
use cutlass_model::{MaterialEffect, MaterialId, MaterialRef, Micros, SegmentId, WordColumns};
use cutlass_session::Request;

use super::{Shell, ShellError};

/// A setting keyed apart from the clip's pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyedSetting {
    Mask,
    Volume,
    TextColor,
}

impl KeyedSetting {
    /// The key names the setting is stored under.
    #[must_use]
    pub fn names(self) -> &'static [&'static str] {
        match self {
            Self::Mask => &cutlass_model::MASK_KEYS,
            Self::Volume => &[cutlass_model::VOLUME_KEY],
            Self::TextColor => &cutlass_model::TEXT_COLOR_KEYS,
        }
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
        let request = self.keyed_request(&segment_id, |pose| pose.rotation = degrees);
        self.apply(request.unwrap_or(Request::SetRotation {
            segment_id,
            degrees,
        }))
    }

    /// The timeline times of clip `segment_id`'s keys that lie on it.
    #[must_use]
    pub fn clip_key_times(&self, segment_id: &SegmentId) -> Vec<Micros> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        let Some(segment) = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id == *segment_id)
        else {
            return Vec::new();
        };
        let start = segment.target.start.0;
        cutlass_model::key_offsets(segment)
            .into_iter()
            .filter(|offset| (0..=segment.target.duration.0).contains(&offset.0))
            .map(|offset| Micros(start.saturating_add(offset.0)))
            .collect()
    }

    /// Move clip `segment_id`'s keys at timeline time `from` to the frame
    /// nearest `to`, kept on the clip. One undo step.
    pub fn move_clip_key(
        &mut self,
        segment_id: &SegmentId,
        from: Micros,
        to: Micros,
    ) -> Result<(), ShellError> {
        self.refuse_locked(segment_id)?;
        let draft = self.session_draft()?;
        let segment = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id == *segment_id)
            .ok_or_else(|| ShellError::Unavailable("the clip is not on the timeline".to_owned()))?;
        let (start, length) = (segment.target.start.0, segment.target.duration.0);
        let rate = draft
            .frame_rate()
            .unwrap_or(cutlass_model::FrameRate::FPS_30);
        let snapped = rate
            .frame_index(Micros(to.0.max(0)))
            .and_then(|index| {
                let below = rate.frame_start(index)?;
                let above = rate.frame_start(index + 1)?;
                Some(if to.0 - below.0 <= above.0 - to.0 {
                    below
                } else {
                    above
                })
            })
            .unwrap_or(to);
        let offset = Micros((snapped.0 - start).clamp(0, length));
        if offset.0 == from.0 - start {
            return Ok(());
        }
        let request =
            cutlass_editor::move_key(draft, segment_id, Micros(from.0 - start), offset)
                .ok_or_else(|| ShellError::Unavailable("a key is already there".to_owned()))?;
        self.apply(request)
    }

    /// The selected picture or title's keyframes, read at the playhead.
    #[must_use]
    pub fn keys(&self) -> Option<cutlass_editor::ClipKeys> {
        let segment_id = self.selected()?;
        let draft = self.draft()?;
        let sizes = super::FaceSizes {
            draft,
            fonts: &*self.face,
        };
        cutlass_editor::clip_keys(draft, segment_id, self.playhead, &sizes)
    }

    /// The selected clip's keys on `setting`, and the one under the
    /// playhead.
    pub fn setting_keys(&self, setting: KeyedSetting) -> Option<cutlass_editor::ClipKeys> {
        let segment_id = self.selected()?;
        let draft = self.draft()?;
        cutlass_editor::value_keys(draft, segment_id, self.playhead, setting.names())
    }

    /// Whether the selected clip has keys on `setting`.
    pub fn setting_keyed(&self, setting: KeyedSetting) -> bool {
        self.setting_keys(setting)
            .is_some_and(|keys| !keys.times.is_empty())
    }

    /// Remove the selected clip's `setting` key under the playhead, or key
    /// `setting` there as it is now. One undo step.
    pub fn toggle_setting_key(&mut self, setting: KeyedSetting) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        let keys = self.setting_keys(setting).ok_or(ShellError::NoProject)?;
        let offset = keys.offset_at(self.playhead).ok_or_else(|| {
            ShellError::Unavailable("move the playhead onto the clip to key it".to_owned())
        })?;
        if keys.here.is_some() {
            return self.apply(Request::UnkeyValues {
                segment_id,
                offset,
                properties: setting
                    .names()
                    .iter()
                    .map(|&name| name.to_owned())
                    .collect(),
            });
        }
        let values = self.setting_values(setting).ok_or_else(|| {
            ShellError::Unavailable(match setting {
                KeyedSetting::Mask => "pick a mask shape first".to_owned(),
                KeyedSetting::Volume => "the clip has no sound to key".to_owned(),
                KeyedSetting::TextColor => "a color key belongs on text".to_owned(),
            })
        })?;
        self.apply(Request::KeyValues {
            segment_id,
            offset,
            values,
        })
    }

    /// Move the playhead to the selected clip's `setting` key before it,
    /// or after it when `forward`.
    pub fn step_setting_key(&mut self, setting: KeyedSetting, forward: bool) {
        let target = self.setting_keys(setting).and_then(|keys| {
            if forward {
                keys.after(self.playhead)
            } else {
                keys.before(self.playhead)
            }
        });
        if let Some(time) = target {
            self.seek(time);
        }
    }

    /// The selected clip's `setting` as it is at the playhead, as key
    /// values.
    fn setting_values(&self, setting: KeyedSetting) -> Option<Vec<(String, f64)>> {
        let inspection = self.inspection()?;
        let values: Vec<f64> = match setting {
            KeyedSetting::Mask => {
                let mask = inspection.mask?;
                [
                    mask.x,
                    mask.y,
                    mask.width,
                    mask.height,
                    mask.rotation,
                    mask.feather,
                    mask.round,
                ]
                .map(f64::from)
                .to_vec()
            }
            KeyedSetting::Volume => vec![inspection.volume],
            KeyedSetting::TextColor => {
                if inspection.text_color.is_empty() && self.text_look().is_none() {
                    return None;
                }
                unit_rgb(&inspection.text_color).to_vec()
            }
        };
        Some(
            setting
                .names()
                .iter()
                .map(|&name| name.to_owned())
                .zip(values)
                .collect(),
        )
    }

    /// The request that keys `setting` at the playhead as `values`, when
    /// the selected clip has keys on it and the playhead is on the clip.
    pub(super) fn key_setting_here(
        &self,
        setting: KeyedSetting,
        values: Vec<f64>,
    ) -> Option<Request> {
        let keys = self
            .setting_keys(setting)
            .filter(|keys| !keys.times.is_empty())?;
        let offset = keys.offset_at(self.playhead)?;
        Some(Request::KeyValues {
            segment_id: self.selected()?.clone(),
            offset,
            values: setting
                .names()
                .iter()
                .map(|&name| name.to_owned())
                .zip(values)
                .collect(),
        })
    }

    /// Remove the selected clip's key under the playhead, or key its pose
    /// and opacity there. One undo step.
    pub fn toggle_keyframe(&mut self) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("keyframes belong on a picture or title")?;
        self.refuse_locked(&segment_id)?;
        let keys = self.keys().ok_or_else(|| {
            ShellError::Unavailable("keyframes belong on a picture or title".to_owned())
        })?;
        let request = keys.toggle(&segment_id, self.playhead).ok_or_else(|| {
            ShellError::Unavailable("move the playhead onto the clip to key it".to_owned())
        })?;
        self.apply(request)
    }

    /// Move the playhead to the selected clip's key before it, or after it
    /// when `forward`. Nothing moves past the first or last key.
    pub fn step_keyframe(&mut self, forward: bool) {
        let target = self.keys().and_then(|keys| {
            if forward {
                keys.after(self.playhead)
            } else {
                keys.before(self.playhead)
            }
        });
        if let Some(time) = target {
            self.seek(time);
        }
    }

    /// Ease the span of keys the playhead is in along the Graphs curve
    /// `graph`, or make it straight with `None`. One undo step.
    pub fn ease_keyframes(&mut self, graph: Option<String>) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("keyframes belong on a picture or title")?;
        self.refuse_locked(&segment_id)?;
        let request = self
            .keys()
            .and_then(|keys| keys.ease(&segment_id, graph))
            .ok_or_else(|| {
                ShellError::Unavailable(
                    "move the playhead between two keyframes to ease them".to_owned(),
                )
            })?;
        self.apply(request)
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
    /// an em away toward `angle` degrees and blurred by `blur`. An empty
    /// color takes it off. One undo restores it.
    pub fn set_text_shadow(
        &mut self,
        color: &str,
        distance: f64,
        angle: f64,
        blur: f64,
    ) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
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

    /// Shape the selected title's background as the bubble keyed `bubble`.
    /// An empty key takes the bubble off with the background it stands
    /// on. One undo restores it.
    pub fn set_text_bubble(&mut self, bubble: &str) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let mut requests = vec![Request::SetTextBubble {
            segment_id: segment_id.clone(),
            bubble: bubble.to_owned(),
        }];
        if bubble.is_empty() {
            requests.push(Request::SetTextBackground {
                segment_id,
                color: String::new(),
                alpha: 1.0,
                round: 0.0,
            });
        }
        self.apply_text_requests(requests)
    }

    /// Give the selected title the look on the Effects page with `id`, or
    /// take every part off with `"none"`. One undo restores it.
    pub fn set_text_effect(&mut self, id: &str) -> Result<(), ShellError> {
        let effect = std::iter::once(&cutlass_editor::NO_TEXT_EFFECT)
            .chain(cutlass_editor::TEXT_EFFECTS)
            .find(|effect| effect.id == id)
            .ok_or_else(|| ShellError::Unavailable(format!("no text effect is called {id}")))?;
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        // With "Apply to all main captions" every caption on the row takes
        // it, each with a glow of its own.
        let clips: Vec<SegmentId> = match self.draft() {
            Some(draft)
                if self.all_captions
                    && cutlass_editor::main_caption_lane(draft, &segment_id).is_some() =>
            {
                cutlass_editor::caption_rows(draft, &segment_id)
                    .into_iter()
                    .map(|row| row.segment_id)
                    .collect()
            }
            _ => vec![segment_id],
        };
        let face = self.face.name().to_owned();
        let mut requests = Vec::new();
        for clip in &clips {
            let glow_id = MaterialId::new(self.fresh("glow"));
            requests.extend(cutlass_editor::text_effect_requests(
                effect, clip, glow_id, &face,
            ));
        }
        self.apply_requests(&requests)
    }

    /// Give the selected title the bubble on the Bubble page with `id`:
    /// its shape, box and letters. One undo restores it.
    pub fn set_bubble_look(&mut self, id: &str) -> Result<(), ShellError> {
        let look = cutlass_editor::BUBBLE_LOOKS
            .iter()
            .find(|look| look.id == id)
            .ok_or_else(|| ShellError::Unavailable(format!("no bubble is called {id}")))?;
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let face = self.face.name().to_owned();
        self.apply_text_requests(cutlass_editor::bubble_look_requests(
            look,
            &segment_id,
            &face,
        ))
    }

    /// The picture on the tile for `tile`: a short text styled as it, in
    /// the shell's face. `None` when it does not draw.
    #[must_use]
    pub fn tile_picture(&self, tile: cutlass_editor::CatalogTile<'_>) -> Option<super::Picture> {
        let (session, time) = cutlass_editor::tile_session(tile, self.face.name());
        let mut device = cutlass_renderer_software::SoftwareDevice::new();
        let picture = cutlass_playback::FrameCache::new()
            .frame(
                &session,
                &mut device,
                time,
                self.preview_readers(),
                None,
                &*self.face,
            )
            .ok()?;
        Some(super::Picture {
            width: picture.width,
            height: picture.height,
            rgba: picture.rgba,
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

    /// The selected title's words without any rich styling.
    #[must_use]
    pub fn text_words(&self) -> Option<String> {
        let draft = self.draft()?;
        let segment = self.selected_segment()?;
        match draft.materials.get(segment.material_id.as_str())? {
            MaterialRef::Text(text) => Some(text.words().unwrap_or_default().into_owned()),
            _ => None,
        }
    }

    /// The faces a title can be set in, by the name the draft stores.
    #[must_use]
    pub fn text_faces(&self) -> Vec<String> {
        self.face.names()
    }

    /// Set the selected title in `face`. The same face is not an undo step.
    pub fn set_text_face(&mut self, face: &str) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        if !self.face.names().iter().any(|name| name == face) {
            return Err(ShellError::Unavailable(
                "that face is not installed".to_owned(),
            ));
        }
        let current = super::text_font(self.session_draft()?, &segment_id);
        if current == face {
            return Ok(());
        }
        self.apply(Request::SetTextFont {
            segment_id,
            face: face.to_owned(),
        })
    }

    /// Align the selected title left (0), center (1), or right (2), or stand
    /// it in vertical columns at the top (3), middle (4), or bottom (5). A
    /// title with no line width takes the canvas width, and a column the
    /// canvas height. One undo restores it.
    pub fn set_text_align(&mut self, alignment: i32) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let line_width = f64::from(
            self.draft()
                .map(|draft| {
                    if alignment >= 3 {
                        draft.canvas_config.height
                    } else {
                        draft.canvas_config.width
                    }
                })
                .unwrap_or(0),
        );
        let face = self.face.name().to_owned();
        self.apply(Request::SetTextAlign {
            segment_id,
            alignment,
            face,
            line_width,
        })
    }

    /// Turn bold, italic and underline on or off over the selected text.
    /// One undo step; with "Apply to all main captions" on, every caption on
    /// the row takes it. The pattern the text already has is not a step.
    pub fn set_text_pattern(
        &mut self,
        pattern: cutlass_session::TextPattern,
    ) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        if self.inspection().map(|seen| seen.text_pattern) == Some(pattern) {
            return Ok(());
        }
        let face = self.face.name().to_owned();
        self.apply(Request::SetTextPattern {
            segment_id,
            bold: pattern.bold,
            italic: pattern.italic,
            underline: pattern.underline,
            face,
        })
    }

    /// Rewrite the selected title's words in `case`. A rich title keeps its
    /// ranges. Words already in that case are not an undo step.
    pub fn set_text_case(&mut self, case: LetterCase) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let words = self.text_words().unwrap_or_default();
        let content = case.apply(&words);
        if content == words {
            return Ok(());
        }
        self.apply(Request::SetTextContent {
            segment_id,
            content,
        })
    }

    /// Put the selected picture back in the middle at its fitted size,
    /// upright, as one undo step.
    pub fn reset_placement(&mut self) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a placement belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        self.apply_requests(&[
            Request::SetPosition {
                segment_id: segment_id.clone(),
                x: 0.0,
                y: 0.0,
            },
            Request::SetScale {
                segment_id: segment_id.clone(),
                scale: 1.0,
            },
            Request::SetRotation {
                segment_id,
                degrees: 0.0,
            },
        ])
    }

    /// Put the selected clip's sound back to its recorded level with no
    /// fades, as one undo step.
    pub fn reset_sound(&mut self) -> Result<(), ShellError> {
        let segment_id = self.sound_segment()?;
        self.refuse_locked(&segment_id)?;
        let fade_id = MaterialId::new(self.fresh("fade"));
        self.apply_requests(&[
            Request::SetVolume {
                segment_id: segment_id.clone(),
                volume: 1.0,
            },
            Request::SetFade {
                segment_id,
                fade_id,
                fade_in: Micros(0),
                fade_out: Micros(0),
            },
        ])
    }

    /// Set the selected clip's gain in decibels. [`SILENT_DB`] and below is
    /// silence. One undo restores it.
    pub fn set_volume_db(&mut self, db: f64) -> Result<(), ShellError> {
        self.set_volume(gain_of_db(db))
    }

    /// Draw the selected caption in the style keyed `style`, or plain with
    /// an empty key. One undo restores it.
    pub fn set_caption_style(&mut self, style: &str) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetCaptionStyle {
            segment_id,
            style: style.to_owned(),
        })
    }

    /// Move the selected picture clip up or down among the clips it
    /// overlaps. Needs Free layer on. One undo restores it.
    pub fn arrange(&mut self, step: cutlass_session::Arrange) -> Result<(), ShellError> {
        let segment_id = self.selected_or_refuse()?;
        self.refuse_locked(&segment_id)?;
        if !self.draft().is_some_and(|draft| draft.config.free_layer) {
            return Err(ShellError::Unavailable(
                "turn on Free layer in the project settings to arrange clips".to_owned(),
            ));
        }
        self.apply(Request::Arrange { segment_id, step })
    }

    /// Mark word `index` of the selected caption as a keyword, drawn in
    /// its style's spoken look, or take the mark off. One undo restores it.
    pub fn set_caption_keyword(&mut self, index: usize, on: bool) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetCaptionKeyword {
            segment_id,
            word: index,
            on,
        })
    }

    /// Start each letter of the selected caption's words `millis` after
    /// the one before. One undo restores it.
    pub fn set_caption_stagger(&mut self, millis: i64) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetCaptionStagger { segment_id, millis })
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
        let (material_id, words) = self.caption_words(&segment_id)?;
        let Some(slot) = words.text.get(index) else {
            return Err(ShellError::Unavailable(
                "the caption has no such word".to_owned(),
            ));
        };
        if slot == word {
            return Ok(());
        }
        // Words typed apart share the word's time.
        let words = cutlass_editor::retype_word(&words, index, word).ok_or_else(|| {
            ShellError::Unavailable("a line break is not a word to type over".to_owned())
        })?;
        self.apply(Request::SetWords { material_id, words })
    }

    /// Join word `index` of the selected caption with the next, over both
    /// their times. One undo step.
    pub fn join_caption_words(&mut self, index: usize) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let (material_id, words) = self.caption_words(&segment_id)?;
        let words = cutlass_editor::join_words(&words, index).ok_or_else(|| {
            ShellError::Unavailable("there is no next word on the line to join".to_owned())
        })?;
        self.apply(Request::SetWords { material_id, words })
    }

    /// Start a new caption line at word `index`, or join it back to the
    /// line before. One undo step.
    pub fn toggle_caption_break(&mut self, index: usize) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let (material_id, words) = self.caption_words(&segment_id)?;
        let words = cutlass_editor::toggle_line_break(&words, index).ok_or_else(|| {
            ShellError::Unavailable("the first word already starts a line".to_owned())
        })?;
        self.apply(Request::SetWords { material_id, words })
    }

    /// Every caption on the selected caption's lane, in timeline order.
    #[must_use]
    pub fn caption_rows(&self) -> Vec<cutlass_editor::CaptionRow> {
        match (self.draft(), self.selected()) {
            (Some(draft), Some(segment_id)) => cutlass_editor::caption_rows(draft, segment_id),
            _ => Vec::new(),
        }
    }

    /// Make caption `segment_id` say `text`. Its words are rematched, so
    /// the ones that stay keep their times. One undo restores it.
    pub fn set_caption_line(
        &mut self,
        segment_id: &SegmentId,
        text: &str,
    ) -> Result<(), ShellError> {
        self.refuse_locked(segment_id)?;
        let draft = self.session_draft()?;
        let request = cutlass_editor::caption_text_request(draft, segment_id, text.trim())
            .map_err(caption_miss)?;
        match request {
            Some(request) => self.apply(request),
            None => Ok(()),
        }
    }

    /// Add a caption after `segment_id`, looking like it: in the gap after
    /// it, or in the second half of its time. One undo step.
    pub fn add_caption_after(&mut self, segment_id: &SegmentId) -> Result<(), ShellError> {
        self.refuse_locked(segment_id)?;
        let mut ids = Vec::new();
        for prefix in ["caption", "caption"] {
            ids.push(self.fresh(prefix));
        }
        let mut ids = ids.into_iter();
        let draft = self.session_draft()?;
        let requests = cutlass_editor::add_after_requests(draft, segment_id, &mut |prefix| {
            ids.next().unwrap_or_else(|| prefix.to_owned())
        })
        .map_err(caption_miss)?;
        self.apply_requests(&requests)
    }

    /// The selected caption's paragraphs: its pages, or its lines.
    #[must_use]
    pub fn caption_paragraphs(&self) -> Vec<cutlass_editor::Paragraph> {
        match (self.draft(), self.selected()) {
            (Some(draft), Some(segment_id)) => {
                cutlass_editor::caption_paragraphs(draft, segment_id)
            }
            _ => Vec::new(),
        }
    }

    /// Make paragraph `index` of the selected caption say `text`, its own
    /// words rematched so the ones that stay keep their times. One undo
    /// restores it.
    pub fn set_caption_paragraph(&mut self, index: usize, text: &str) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        match cutlass_editor::paragraph_request(draft, &segment_id, index, text)
            .map_err(caption_miss)?
        {
            Some(request) => self.apply(request),
            None => Ok(()),
        }
    }

    /// The template the selected caption was styled with, by its id.
    #[must_use]
    pub fn caption_template(&self) -> Option<String> {
        match (self.draft(), self.selected()) {
            (Some(draft), Some(segment_id)) => {
                cutlass_editor::caption_template_of(draft, segment_id)
            }
            _ => None,
        }
    }

    /// Style the selected caption as template `id`: its style, caption
    /// animation, stagger and look. While "Apply to all main captions" is
    /// on, every caption on its subtitle row takes it. One undo step.
    pub fn set_caption_template(&mut self, id: &str) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let template = cutlass_editor::caption_template(id).ok_or_else(|| {
            ShellError::Unavailable("there is no such caption template".to_owned())
        })?;
        let draft = self.session_draft()?;
        if cutlass_editor::caption_rows(draft, &segment_id).is_empty() {
            return Err(caption_miss(cutlass_editor::CaptionListMiss::NotCaption));
        }
        let requests = cutlass_editor::caption_template_requests(
            draft,
            &segment_id,
            template,
            self.all_captions(),
        );
        self.apply_requests(&requests)
    }

    /// Take the selected caption's template off: the plain style, no
    /// stagger and no caption animation, on every main caption while
    /// "Apply to all main captions" is on. One undo restores it.
    pub fn clear_caption_template(&mut self) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        if cutlass_editor::caption_rows(draft, &segment_id).is_empty() {
            return Err(caption_miss(cutlass_editor::CaptionListMiss::NotCaption));
        }
        let requests =
            cutlass_editor::plain_caption_requests(draft, &segment_id, self.all_captions());
        self.apply_requests(&requests)
    }

    /// Replace `find` with `with`, case ignored, in every caption on the
    /// selected caption's lane. Each changed caption keeps the times of the
    /// words that stay. One undo step; how many were replaced.
    pub fn replace_in_captions(&mut self, find: &str, with: &str) -> Result<usize, ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        if find.trim().is_empty() {
            return Err(ShellError::Unavailable("type the words to find".to_owned()));
        }
        let draft = self.session_draft()?;
        let replaced = cutlass_editor::replace_in_captions(draft, &segment_id, find, with.trim())
            .map_err(|miss| match miss {
            cutlass_editor::CaptionListMiss::Empty => ShellError::Unavailable(
                "that would leave a caption with no words; delete it instead".to_owned(),
            ),
            miss => caption_miss(miss),
        })?;
        if replaced.count == 0 {
            return Err(ShellError::Unavailable("no caption says that".to_owned()));
        }
        if !replaced.requests.is_empty() {
            self.apply_requests(&replaced.requests)?;
        }
        Ok(replaced.count)
    }

    /// Take caption `segment_id` off the timeline. One undo restores it.
    pub fn remove_caption(&mut self, segment_id: &SegmentId) -> Result<(), ShellError> {
        self.refuse_locked(segment_id)?;
        self.apply(cutlass_editor::remove_caption_request(segment_id))
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

fn caption_miss(miss: cutlass_editor::CaptionListMiss) -> ShellError {
    use cutlass_editor::CaptionListMiss as Miss;
    ShellError::Unavailable(
        match miss {
            Miss::NotCaption => "that clip is not a caption",
            Miss::Empty => "a caption needs a word; delete it instead",
            Miss::TooLong => "that line is too long to match its words",
            Miss::NoRoom => "there is no room after this caption",
        }
        .to_owned(),
    )
}

/// The red, green and blue of `#rrggbb`, each from 0 to 1, when it reads.
pub(super) fn hex_rgb(color: &str) -> Option<[f64; 3]> {
    let body = color.trim().trim_start_matches('#');
    if body.len() != 6 || !body.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    Some(unit_rgb(body))
}

/// `#RRGGBB` for red, green and blue from 0 to 1.
pub(super) fn rgb_hex(rgb: [f64; 3]) -> String {
    let byte = |unit: f64| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let byte = (unit.clamp(0.0, 1.0) * 255.0).round() as u8;
        byte
    };
    format!(
        "#{:02X}{:02X}{:02X}",
        byte(rgb[0]),
        byte(rgb[1]),
        byte(rgb[2])
    )
}

/// The red, green and blue of `#rrggbb`, each from 0 to 1; white when it
/// does not read.
fn unit_rgb(color: &str) -> [f64; 3] {
    let body = color.trim().trim_start_matches('#');
    let channel = |at: usize| {
        body.get(at..at + 2)
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
            .map_or(1.0, |byte| f64::from(byte) / 255.0)
    };
    [channel(0), channel(2), channel(4)]
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialColorConfig, MaterialEffect, MaterialText, MaterialVideo,
        MaterialVideoEffect, RawMaterial, Segment, SegmentId, TimeRange, Track, TrackAttribute,
        TrackFlag, TrackType,
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

    /// The video clips in drawing order, bottom first.
    fn stack(desktop: &Shell) -> Vec<String> {
        let draft = desktop.draft().unwrap();
        let mut keys = Vec::new();
        for (track, row) in draft.tracks.iter().enumerate() {
            if !matches!(row.kind, TrackType::Video) {
                continue;
            }
            for (at, segment) in row.segments.iter().enumerate() {
                keys.push(((segment.render_index, track, at), segment.id.to_string()));
            }
        }
        keys.sort();
        keys.into_iter().map(|(_, id)| id).collect()
    }

    #[test]
    fn arrange_needs_free_layer_and_moves_a_clip_in_one_step() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        assert_eq!(stack(&desktop), ["main", "overlay", "still", "solid"]);
        pick(&mut desktop, "overlay");
        assert!(
            desktop.arrange(cutlass_session::Arrange::Front).is_err(),
            "off, the tracks give the order"
        );
        let mut settings = desktop.project_settings().unwrap();
        settings.free_layer = true;
        desktop.save_project_settings(&settings).unwrap();
        desktop.arrange(cutlass_session::Arrange::Front).unwrap();
        assert_eq!(stack(&desktop), ["main", "still", "solid", "overlay"]);
        pick(&mut desktop, "main");
        desktop.arrange(cutlass_session::Arrange::Forward).unwrap();
        assert_eq!(stack(&desktop), ["still", "main", "solid", "overlay"]);
        desktop.undo().unwrap();
        assert_eq!(stack(&desktop), ["main", "still", "solid", "overlay"]);
        settings.free_layer = false;
        desktop.save_project_settings(&settings).unwrap();
        assert_eq!(stack(&desktop), ["main", "overlay", "still", "solid"]);
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
    fn a_placement_reset_is_one_step_and_a_sound_reset_is_one_step() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "overlay");
        let before = desktop.draft().unwrap().clone();
        desktop.set_position(0.5, 0.25).unwrap();
        desktop.set_scale(2.0).unwrap();
        desktop.set_rotation(45.0).unwrap();
        let moved = desktop.draft().unwrap().clone();
        desktop.reset_placement().unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!((inspection.transform_x, inspection.transform_y), (0.0, 0.0));
        assert_eq!((inspection.scale_x, inspection.rotation), (1.0, 0.0));
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap(), &moved);
        for _ in 0..3 {
            desktop.undo().unwrap();
        }
        assert_eq!(desktop.draft().unwrap(), &before);

        pick(&mut desktop, "sound");
        desktop.set_volume_db(-6.0).unwrap();
        desktop.set_fade_in(0.5).unwrap();
        let faded = desktop.draft().unwrap().clone();
        desktop.reset_sound().unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.volume, 1.0);
        assert_eq!((inspection.fade_in, inspection.fade_out), (0, 0));
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap(), &faded);

        pick(&mut desktop, "words");
        assert!(desktop.reset_sound().is_err());
        pick(&mut desktop, "sound");
        assert!(desktop.reset_placement().is_err());
        lock(&mut desktop, "o1");
        pick(&mut desktop, "overlay");
        assert_eq!(
            desktop.reset_placement().unwrap_err().to_string(),
            "the track is locked"
        );
    }

    #[test]
    fn volume_reads_and_writes_in_decibels_with_a_silent_floor() {
        assert_eq!(super::volume_db(1.0), 0.0);
        assert!((super::volume_db(0.5) + 6.0206).abs() < 1e-3);
        assert_eq!(super::volume_db(0.0), super::SILENT_DB);
        assert_eq!(super::volume_db(f64::NAN), super::SILENT_DB);
        assert_eq!(super::volume_db(1e-9), super::SILENT_DB);
        assert_eq!(super::gain_of_db(super::SILENT_DB), 0.0);
        assert_eq!(super::gain_of_db(-90.0), 0.0);
        assert!((super::gain_of_db(12.0) - 3.981).abs() < 1e-3);
        for db in [-48.0, -6.0, 0.0, 6.0, 12.0] {
            assert!((super::volume_db(super::gain_of_db(db)) - db).abs() < 1e-9);
        }
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "sound");
        desktop.set_volume_db(super::SILENT_DB).unwrap();
        assert_eq!(desktop.inspection().unwrap().volume, 0.0);
        desktop.set_volume_db(6.0).unwrap();
        assert!((desktop.inspection().unwrap().volume - 1.9953).abs() < 1e-3);
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().volume, 0.0);
    }

    #[test]
    fn a_keyed_volume_is_set_where_the_playhead_is() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "sound");
        assert!(!desktop.setting_keyed(super::KeyedSetting::Volume));
        desktop.seek(cutlass_model::Micros(1_000_000));
        desktop
            .toggle_setting_key(super::KeyedSetting::Volume)
            .unwrap();
        desktop.seek(cutlass_model::Micros(3_000_000));
        // With keys, a change keys the volume at the playhead.
        desktop.set_volume(0.5).unwrap();
        let keys = desktop.setting_keys(super::KeyedSetting::Volume).unwrap();
        assert_eq!(
            keys.times,
            [
                cutlass_model::Micros(1_000_000),
                cutlass_model::Micros(3_000_000)
            ]
        );
        desktop.seek(cutlass_model::Micros(2_000_000));
        assert!((desktop.inspection().unwrap().volume - 0.75).abs() < 1e-9);
        // The pose's keys are apart.
        assert!(desktop.keys().is_none_or(|keys| keys.times.is_empty()));
        desktop.seek(cutlass_model::Micros(3_000_000));
        desktop
            .toggle_setting_key(super::KeyedSetting::Volume)
            .unwrap();
        let keys = desktop.setting_keys(super::KeyedSetting::Volume).unwrap();
        assert_eq!(keys.times, [cutlass_model::Micros(1_000_000)]);
        desktop.step_setting_key(super::KeyedSetting::Volume, false);
        assert_eq!(desktop.playhead(), cutlass_model::Micros(1_000_000));
    }

    #[test]
    fn a_keyed_mask_moves_and_goes_with_its_keys() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "main");
        let circle = cutlass_effects::Mask::new(cutlass_effects::MaskShape::Circle);
        desktop.set_mask(Some(circle)).unwrap();
        desktop.seek(cutlass_model::Micros(0));
        desktop
            .toggle_setting_key(super::KeyedSetting::Mask)
            .unwrap();
        desktop.seek(cutlass_model::Micros(2_000_000));
        desktop
            .set_mask(Some(cutlass_effects::Mask { x: 0.5, ..circle }))
            .unwrap();
        desktop.seek(cutlass_model::Micros(1_000_000));
        let mask = desktop.inspection().unwrap().mask.unwrap();
        assert!((mask.x - 0.25).abs() < 1e-6, "{mask:?}");
        // One undo step takes the last key off, and the mask is back.
        desktop.undo().unwrap();
        let mask = desktop.inspection().unwrap().mask.unwrap();
        assert!(mask.x.abs() < 1e-6, "{mask:?}");
        desktop.set_mask(None).unwrap();
        assert!(!desktop.setting_keyed(super::KeyedSetting::Mask));
        assert!(desktop.inspection().unwrap().mask.is_none());
    }

    #[test]
    fn a_title_pattern_shows_in_the_inspector_and_undoes_in_one_step() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "words");
        let off = cutlass_session::TextPattern::default();
        assert_eq!(desktop.inspection().unwrap().text_pattern, off);
        let all = cutlass_session::TextPattern {
            bold: true,
            italic: true,
            underline: true,
        };
        desktop.set_text_pattern(all).unwrap();
        assert_eq!(desktop.inspection().unwrap().text_pattern, all);
        let revision = desktop.revision();
        desktop.set_text_pattern(all).unwrap();
        assert_eq!(desktop.revision(), revision);
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().text_pattern, off);
        pick(&mut desktop, "main");
        assert!(desktop.set_text_pattern(all).is_err());
    }

    #[test]
    fn a_title_face_alignment_and_case_are_one_step_each() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        let sample: &[u8] = include_bytes!("../../../../crates/font/fixtures/sample.ttf");
        desktop.use_faces(vec![("A.ttf", sample.to_vec()), ("B.ttf", sample.to_vec())]);
        assert_eq!(desktop.text_faces(), ["A.ttf", "B.ttf"]);
        pick(&mut desktop, "words");
        let before = desktop.draft().unwrap().clone();
        assert_eq!(desktop.text_words().as_deref(), Some("Title"));

        desktop.set_text_face("B.ttf").unwrap();
        assert_eq!(desktop.inspection().unwrap().font, "B.ttf");
        let revision = desktop.revision();
        desktop.set_text_face("B.ttf").unwrap();
        assert_eq!(desktop.revision(), revision);
        assert_eq!(
            desktop.set_text_face("C.ttf").unwrap_err().to_string(),
            "that face is not installed"
        );

        desktop.set_text_align(2).unwrap();
        assert_eq!(desktop.inspection().unwrap().text_align, 2);
        assert!(desktop.set_text_align(7).is_err());

        desktop.set_text_case(super::LetterCase::Upper).unwrap();
        assert_eq!(desktop.text_words().as_deref(), Some("TITLE"));
        let revision = desktop.revision();
        desktop.set_text_case(super::LetterCase::Upper).unwrap();
        assert_eq!(desktop.revision(), revision);
        desktop.set_text_case(super::LetterCase::Lower).unwrap();
        assert_eq!(desktop.text_words().as_deref(), Some("title"));

        for _ in 0..4 {
            desktop.undo().unwrap();
        }
        assert_eq!(desktop.draft().unwrap(), &before);
        pick(&mut desktop, "main");
        assert_eq!(desktop.text_words(), None);
        assert!(desktop.set_text_face("A.ttf").is_err());
        assert!(desktop.set_text_case(super::LetterCase::Title).is_err());
    }

    #[test]
    fn title_case_raises_each_word_and_keeps_the_gaps() {
        use super::LetterCase;
        assert_eq!(
            LetterCase::Title.apply("hELLO  wide\nworld"),
            "Hello  Wide\nWorld"
        );
        assert_eq!(LetterCase::Title.apply("one\ttwo"), "One\tTwo");
        assert_eq!(LetterCase::Title.apply(""), "");
        assert_eq!(LetterCase::Upper.apply("straße"), "STRASSE");
        assert_eq!(LetterCase::Lower.apply("ÉCOLE"), "école");
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
                shadow_angle: cutlass_editor::SHADOW_ANGLE,
                ..TextLook::default()
            })
        );

        desktop.set_text_outline("#FF0000", 0.0).unwrap();
        desktop
            .set_text_shadow("#000000", 0.0, cutlass_editor::SHADOW_ANGLE, 0.0)
            .unwrap();
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

        desktop
            .set_text_shadow("#000000", 20.0, -45.0, 0.5)
            .unwrap();
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

        // A bubble shapes a background it brings; None takes both off in
        // one step.
        desktop.set_text_bubble("thought").unwrap();
        let look = desktop.text_look().unwrap();
        assert_eq!(
            (look.bubble.as_str(), look.background_color.as_str()),
            ("thought", "#FFFFFF")
        );
        desktop.set_text_bubble("").unwrap();
        let look = desktop.text_look().unwrap();
        assert_eq!(
            (look.bubble.as_str(), look.background_color.as_str()),
            ("", "")
        );
        desktop.undo().unwrap();
        assert_eq!(desktop.text_look().unwrap().bubble, "thought");
        assert!(desktop.set_text_bubble("cloud").is_err());

        // An effect sets every part in one step and reads back as itself.
        desktop.set_text_effect("pop").unwrap();
        let look = desktop.text_look().unwrap();
        assert_eq!(
            cutlass_editor::text_effect_of(&look).map(|effect| effect.id),
            Some("pop")
        );
        desktop.undo().unwrap();
        assert_ne!(
            cutlass_editor::text_effect_of(&desktop.text_look().unwrap()).map(|effect| effect.id),
            Some("pop")
        );
        assert!(desktop.set_text_effect("sparkle").is_err());

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
    fn the_caption_list_edits_adds_and_deletes_lines_one_step_each() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "subtitle");
        let before = desktop.draft().unwrap().clone();
        let rows = desktop.caption_rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].text, "one two");
        let id = rows[0].segment_id.clone();

        desktop.set_caption_line(&id, " one three ").unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.words.text, ["one", "three"]);
        assert_eq!(inspection.words.start_time[0], 0);
        assert_eq!(inspection.words.end_time[1], 900);
        assert!(desktop.set_caption_line(&id, "   ").is_err());

        desktop.add_caption_after(&id).unwrap();
        let rows = desktop.caption_rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].text, cutlass_editor::NEW_CAPTION_TEXT);
        let added = rows[1].segment_id.clone();
        desktop.remove_caption(&added).unwrap();
        assert_eq!(desktop.caption_rows().len(), 1);

        for _ in 0..3 {
            desktop.undo().unwrap();
        }
        assert_eq!(desktop.draft().unwrap(), &before);

        pick(&mut desktop, "main");
        assert!(desktop.caption_rows().is_empty());
    }

    #[test]
    fn a_caption_paragraph_edit_keeps_its_word_times() {
        let temp = Temp::new();
        let mut desktop = every_kind(&temp);
        pick(&mut desktop, "subtitle");
        let paragraphs = desktop.caption_paragraphs();
        assert_eq!(paragraphs.len(), 1);
        assert_eq!(paragraphs[0].text, "one two");
        desktop.set_caption_paragraph(0, "one too").unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.words.text, ["one", "too"]);
        assert_eq!(inspection.words.start_time, [0, 500]);
        assert!(desktop.set_caption_paragraph(4, "x").is_err());
        desktop.undo().unwrap();
        assert_eq!(desktop.caption_paragraphs()[0].text, "one two");
    }

    /// Two captions on a subtitle row and a title on a plain text row.
    fn subtitle_row(temp: &Temp) -> Shell {
        let mut draft = Draft::empty("D");
        for (id, words) in [("first", "one"), ("second", "two"), ("title", "hi")] {
            let mut text = MaterialText::new(id);
            text.kind = if id == "title" { "text" } else { "subtitle" }.to_owned();
            text.content = words.to_owned();
            text.font_size = 15.0;
            draft.materials.texts.push(text);
        }
        let mut subs = row(
            "subs",
            TrackType::Text,
            TrackFlag::TextSubtitle,
            "first",
            "first",
        );
        subs.segments.push(Segment::new(
            "second",
            "second",
            TimeRange::new(2_000_000, 1_000_000),
        ));
        draft.tracks.push(subs);
        draft
            .tracks
            .push(row("t", TrackType::Text, TrackFlag::None, "title", "title"));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop
    }

    fn sizes(desktop: &Shell) -> Vec<f64> {
        let draft = desktop.draft().unwrap();
        ["first", "second", "title"]
            .iter()
            .map(|id| {
                draft
                    .materials
                    .texts
                    .iter()
                    .find(|text| text.id.as_str() == *id)
                    .unwrap()
                    .font_size
            })
            .collect()
    }

    #[test]
    fn a_main_caption_style_change_reaches_its_row_only_while_apply_to_all_is_on() {
        let temp = Temp::new();
        let mut desktop = subtitle_row(&temp);
        pick(&mut desktop, "first");
        assert!(desktop.all_captions());
        assert!(desktop.on_main_captions());
        desktop.set_text_size(40.0).unwrap();
        assert_eq!(sizes(&desktop), [40.0, 40.0, 15.0]);
        desktop.undo().unwrap();
        assert_eq!(sizes(&desktop), [15.0, 15.0, 15.0]);

        desktop.set_all_captions(false);
        desktop.set_text_size(30.0).unwrap();
        assert_eq!(sizes(&desktop), [30.0, 15.0, 15.0]);

        desktop.set_all_captions(true);
        pick(&mut desktop, "title");
        assert!(!desktop.on_main_captions());
        desktop.set_text_size(20.0).unwrap();
        assert_eq!(sizes(&desktop), [30.0, 15.0, 20.0]);
    }

    #[test]
    fn a_main_caption_takes_a_pattern_preset_and_bubble_with_its_row() {
        let temp = Temp::new();
        let mut desktop = subtitle_row(&temp);
        pick(&mut desktop, "first");
        let field = |desktop: &Shell, read: fn(&MaterialText) -> String| -> Vec<String> {
            let draft = desktop.draft().unwrap();
            ["first", "second", "title"]
                .iter()
                .map(|id| {
                    read(
                        draft
                            .materials
                            .texts
                            .iter()
                            .find(|text| text.id.as_str() == *id)
                            .unwrap(),
                    )
                })
                .collect()
        };
        desktop
            .set_text_pattern(cutlass_session::TextPattern {
                underline: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            field(&desktop, |text| text.underline.to_string()),
            ["true", "true", "false"]
        );
        let paper = cutlass_editor::TEXT_PRESETS
            .iter()
            .position(|preset| preset.name == "Paper box")
            .unwrap();
        desktop.set_text_preset(paper).unwrap();
        assert_eq!(
            field(&desktop, |text| text.background_color.clone()),
            ["#FFFFFF", "#FFFFFF", ""]
        );
        desktop.set_text_bubble("speech").unwrap();
        assert_eq!(
            field(&desktop, |text| text.bubble.clone()),
            ["speech", "speech", ""]
        );
        desktop.undo().unwrap();
        assert_eq!(field(&desktop, |text| text.bubble.clone()), ["", "", ""]);
    }

    #[test]
    fn a_main_caption_takes_an_effect_and_a_bubble_look_with_its_row() {
        let temp = Temp::new();
        let mut desktop = subtitle_row(&temp);
        pick(&mut desktop, "first");
        // Each clip's look, by its material: the first two are the row's.
        let looks = |desktop: &Shell| -> Vec<TextLook> {
            let draft = desktop.draft().unwrap();
            ["first", "second", "title"]
                .iter()
                .map(|id| {
                    let segment = draft
                        .tracks
                        .iter()
                        .flat_map(|track| &track.segments)
                        .find(|segment| segment.material_id.as_str() == *id)
                        .unwrap();
                    cutlass_editor::text_look(draft, segment).unwrap()
                })
                .collect()
        };
        desktop.set_text_effect("gold-glow").unwrap();
        let worn = looks(&desktop);
        assert_eq!(worn[0].glow_color, "#FFB800");
        assert_eq!(worn[1].glow_color, "#FFB800");
        assert!(worn[2].glow_color.is_empty());
        // Each caption glows with a glow of its own.
        let draft = desktop.draft().unwrap();
        let glows: std::collections::HashSet<_> = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .flat_map(|segment| &segment.extra_material_refs)
            .filter(|id| id.as_str().starts_with("glow"))
            .collect();
        assert_eq!(glows.len(), 2);
        desktop.set_bubble_look("burst-red").unwrap();
        let worn = looks(&desktop);
        for look in &worn[..2] {
            assert_eq!(
                cutlass_editor::bubble_look_of(look).map(|b| b.id),
                Some("burst-red")
            );
        }
        assert!(worn[2].bubble.is_empty());
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        assert!(
            looks(&desktop)
                .iter()
                .all(|look| look.glow_color.is_empty())
        );
        // With the box off, only the picked caption takes it.
        desktop.set_all_captions(false);
        desktop.set_text_effect("neon-blue").unwrap();
        let worn = looks(&desktop);
        assert_eq!(worn[0].glow_color, "#00C2FF");
        assert!(worn[1].glow_color.is_empty());
    }

    #[test]
    fn each_catalog_tile_draws_its_own_picture() {
        let temp = Temp::new();
        let desktop = subtitle_row(&temp);
        let mut seen = std::collections::HashSet::new();
        let tiles = cutlass_editor::CAPTION_TEMPLATES
            .iter()
            .take(3)
            .map(cutlass_editor::CatalogTile::Template)
            .chain(
                cutlass_editor::TEXT_EFFECTS
                    .iter()
                    .take(3)
                    .map(cutlass_editor::CatalogTile::Effect),
            )
            .chain(
                cutlass_editor::BUBBLE_LOOKS
                    .iter()
                    .take(3)
                    .map(cutlass_editor::CatalogTile::Bubble),
            );
        for tile in tiles {
            let picture = desktop.tile_picture(tile).unwrap();
            let side = u32::try_from(cutlass_editor::TILE_SIDE).unwrap();
            assert_eq!((picture.width, picture.height), (side, side));
            assert!(
                picture
                    .rgba
                    .chunks(4)
                    .any(|pixel| pixel[3] > 0 && pixel[..3] != [0, 0, 0]),
                "{tile:?} draws"
            );
            assert!(seen.insert(picture.rgba), "{tile:?} draws its own");
        }
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

        // A word typed as two splits its time; joined again it is one.
        desktop.set_caption_word(1, "to far").unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.words.text, ["one", "to", "far"]);
        assert_eq!(inspection.words.end_time, [400, 660, 900]);
        desktop.join_caption_words(1).unwrap();
        assert_eq!(desktop.inspection().unwrap().words.text, ["one", "to far"]);
        desktop.toggle_caption_break(1).unwrap();
        let broken = desktop.inspection().unwrap();
        assert_eq!(broken.words.text, ["one", "\n", "to far"]);
        assert_eq!(broken.text.as_deref(), Some("one\nto far"));
        assert!(desktop.toggle_caption_break(0).is_err());
        desktop.set_caption_style("karaoke").unwrap();
        assert_eq!(desktop.inspection().unwrap().caption_style, "karaoke");
        assert!(desktop.set_caption_style("loud").is_err());
        desktop.set_caption_keyword(2, true).unwrap();
        desktop.set_caption_stagger(40).unwrap();
        let marked = desktop.inspection().unwrap();
        assert_eq!(marked.caption_keywords, [2]);
        assert_eq!(marked.caption_stagger_ms, 40);
        assert!(desktop.set_caption_keyword(9, true).is_err());
        assert!(desktop.set_caption_stagger(5_000).is_err());
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        for _ in 0..4 {
            desktop.undo().unwrap();
        }
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

    #[test]
    fn a_caption_template_styles_the_row_while_apply_to_all_is_on() {
        let temp = Temp::new();
        let mut desktop = subtitle_row(&temp);
        pick(&mut desktop, "first");
        let before = desktop.draft().unwrap().clone();
        desktop.set_caption_template("cutlass-caption-pop").unwrap();
        assert_eq!(
            desktop.caption_template().as_deref(),
            Some("cutlass-caption-pop")
        );
        let styles = |desktop: &Shell| -> Vec<String> {
            ["first", "second", "title"]
                .iter()
                .map(|id| {
                    let draft = desktop.draft().unwrap();
                    let text = draft
                        .materials
                        .texts
                        .iter()
                        .find(|t| t.id.as_str() == *id)
                        .unwrap();
                    text.caption_style.clone()
                })
                .collect()
        };
        assert_eq!(styles(&desktop), ["pop", "pop", ""]);
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap(), &before);

        desktop.set_all_captions(false);
        desktop
            .set_caption_template("cutlass-caption-neon")
            .unwrap();
        assert_eq!(styles(&desktop), ["neon", "", ""]);

        assert!(desktop.set_caption_template("nope").is_err());
        pick(&mut desktop, "title");
        assert!(desktop.set_caption_template("cutlass-caption-pop").is_err());
        assert_eq!(desktop.caption_template(), None);
    }

    #[test]
    fn find_and_replace_reaches_every_caption_on_the_row_as_one_step() {
        let temp = Temp::new();
        let mut desktop = subtitle_row(&temp);
        pick(&mut desktop, "first");
        let before = desktop.draft().unwrap().clone();
        let lines = |desktop: &Shell| -> Vec<String> {
            desktop
                .caption_rows()
                .into_iter()
                .map(|row| row.text)
                .collect()
        };
        assert_eq!(desktop.replace_in_captions("O", "1").unwrap(), 2);
        assert_eq!(lines(&desktop), ["1ne", "tw1"]);
        let title = desktop
            .draft()
            .unwrap()
            .materials
            .texts
            .iter()
            .find(|t| t.id.as_str() == "title")
            .unwrap()
            .content
            .clone();
        assert_eq!(title, "hi");
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap(), &before);

        let revision = desktop.revision();
        assert!(desktop.replace_in_captions("zebra", "x").is_err());
        assert!(desktop.replace_in_captions(" ", "x").is_err());
        assert!(desktop.replace_in_captions("one", "").is_err());
        assert_eq!(desktop.revision(), revision);
        pick(&mut desktop, "title");
        assert!(desktop.replace_in_captions("hi", "yo").is_err());
    }
}
