//! Caption group mutations: the model-side chokepoint every platform shares.
//!
//! Cues are ordinary text clips, so trim / move / split / delete need nothing
//! here. What does belong here is everything that treats a *group* as one
//! thing: creating a batch of cues atomically, writing a style through to its
//! members, re-timing a line's word table when its text changes, and keeping
//! cue indices dense.

use crate::caption::{
    CaptionCueSpec, CaptionGroup, CaptionGroupSpec, CaptionHighlight, CaptionLayout, CaptionStyle,
    CaptionStyleScope, CaptionWord, MAX_CAPTION_CUES, caption_template_spec,
};
use crate::clip::{Clip, ClipSource, Generator};
use crate::error::ModelError;
use crate::ids::{CaptionGroupId, ClipId, TrackId};
use crate::look::AnimationRef;
use crate::param::Param;
use crate::time::TimeRange;
use crate::track::TrackKind;

use super::Project;

#[cfg(test)]
mod tests;

impl Project {
    /// Create a caption group and place one cue clip per spec, atomically.
    ///
    /// Everything is validated before anything is inserted — including that the
    /// cues are ascending, non-overlapping, and clear of the lane's existing
    /// clips — so a rejected batch leaves the timeline and the id allocators
    /// untouched. Returns the new group id and its cue clip ids in order.
    pub fn add_caption_group(
        &mut self,
        spec: &CaptionGroupSpec,
        cues: &[CaptionCueSpec],
    ) -> Result<(CaptionGroupId, Vec<ClipId>), ModelError> {
        if cues.is_empty() {
            return Err(ModelError::InvalidParam(
                "a caption group needs at least one cue".into(),
            ));
        }
        if cues.len() > MAX_CAPTION_CUES {
            return Err(ModelError::InvalidParam(format!(
                "a caption group holds at most {MAX_CAPTION_CUES} cues"
            )));
        }
        self.require_text_track(spec.track)?;
        self.validate_cue_placements(spec.track, cues)?;

        // Validation is complete: allocate ids and insert.
        let group = spec.resolve()?;
        let group_id = group.id;
        let style = group.style.clone();
        self.timeline.add_caption_group(group)?;

        let mut placed = Vec::with_capacity(cues.len());
        for (index, cue) in cues.iter().enumerate() {
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            let mut clip = Clip::generated(
                Generator::Text {
                    content: cue.text.clone(),
                    style: style.text.clone(),
                },
                cue.timeline,
            );
            apply_caption_style(&mut clip, &style);
            clip.caption = Some(cue.cue(group_id, index));
            match self.timeline.add_clip(spec.track, clip) {
                Ok(id) => placed.push(id),
                Err(e) => {
                    // Pre-validated, so this is a bug rather than user error —
                    // but never leave a half-placed group behind.
                    for id in placed {
                        self.timeline.remove_clip(id);
                    }
                    self.timeline.remove_caption_group(group_id);
                    return Err(e);
                }
            }
        }
        Ok((group_id, placed))
    }

    /// Remove a caption group and every cue clip it owns, returning both for
    /// undo capture.
    pub fn remove_caption_group(
        &mut self,
        group: CaptionGroupId,
    ) -> Result<(CaptionGroup, Vec<Clip>), ModelError> {
        let cue_ids = self.timeline.caption_cue_ids(group);
        let removed_group = self
            .timeline
            .remove_caption_group(group)
            .ok_or(ModelError::UnknownCaptionGroup(group))?;
        let cues = cue_ids
            .into_iter()
            .filter_map(|id| self.timeline.remove_clip(id))
            .collect();
        Ok((removed_group, cues))
    }

    /// Re-insert a removed group and its cue clips (undo of
    /// [`remove_caption_group`](Self::remove_caption_group)), keeping every id.
    pub fn restore_caption_group(
        &mut self,
        group: CaptionGroup,
        cues: Vec<Clip>,
    ) -> Result<CaptionGroupId, ModelError> {
        let track = group.track;
        let id = self.timeline.add_caption_group(group)?;
        for clip in cues {
            if let Err(e) = self.timeline.add_clip(track, clip) {
                self.timeline.remove_caption_group(id);
                return Err(e);
            }
        }
        Ok(id)
    }

    /// Replace a group's shared style and write it through to its cues (CapCut
    /// "Apply to all").
    ///
    /// With [`CaptionStyleScope::KeepOverrides`], cues the user styled
    /// individually keep their look; with [`CaptionStyleScope::All`] every cue
    /// is rewritten and its override flag cleared. Writing through replaces each
    /// cue's text style, layer styles, animation slots, and transform
    /// position/scale — rotation, opacity, anchor, and their keyframes survive.
    pub fn set_caption_group_style(
        &mut self,
        group: CaptionGroupId,
        style: CaptionStyle,
        scope: CaptionStyleScope,
    ) -> Result<(), ModelError> {
        style.validate()?;
        let cue_ids = self.timeline.caption_cue_ids(group);
        {
            let target = self
                .timeline
                .caption_group_mut(group)
                .ok_or(ModelError::UnknownCaptionGroup(group))?;
            target.style = style.clone();
            // The style no longer matches the template it came from.
            target.template = None;
        }
        for clip_id in cue_ids {
            let Some(clip) = self.timeline.clip_mut(clip_id) else {
                continue;
            };
            let overridden = clip.caption.as_ref().is_some_and(|cue| cue.style_override);
            if overridden && scope == CaptionStyleScope::KeepOverrides {
                continue;
            }
            if let ClipSource::Generated(Generator::Text { style: text, .. }) = &mut clip.content {
                *text = style.text.clone();
            }
            apply_caption_style(clip, &style);
            if let Some(cue) = clip.caption.as_mut() {
                cue.style_override = false;
            }
        }
        Ok(())
    }

    /// Apply a caption template: its style, layout, and highlight in one shot,
    /// written through to every cue.
    pub fn set_caption_group_template(
        &mut self,
        group: CaptionGroupId,
        template: &str,
    ) -> Result<(), ModelError> {
        let spec = caption_template_spec(template).ok_or_else(|| {
            ModelError::InvalidParam(format!("unknown caption template '{template}'"))
        })?;
        self.set_caption_group_style(group, spec.style(), CaptionStyleScope::All)?;
        self.set_caption_group_layout(group, spec.layout())?;
        self.set_caption_highlight(group, spec.highlight())?;
        let target = self
            .timeline
            .caption_group_mut(group)
            .ok_or(ModelError::UnknownCaptionGroup(group))?;
        target.template = Some(spec.id.to_owned());
        Ok(())
    }

    /// Replace a group's segmentation rules.
    ///
    /// Re-splitting existing cues against new rules needs the segmenter (see
    /// `cutlass-captions`) and lands as a fresh batch; what happens here is the
    /// part the model owns — storing the rules and moving the cues into the new
    /// safe area.
    pub fn set_caption_group_layout(
        &mut self,
        group: CaptionGroupId,
        layout: CaptionLayout,
    ) -> Result<(), ModelError> {
        layout.validate()?;
        let cue_ids = self.timeline.caption_cue_ids(group);
        let position = {
            let target = self
                .timeline
                .caption_group_mut(group)
                .ok_or(ModelError::UnknownCaptionGroup(group))?;
            let moved =
                (target.layout.safe_area_bottom - layout.safe_area_bottom).abs() > f32::EPSILON;
            target.layout = layout;
            if !moved {
                return Ok(());
            }
            target.style.position = [target.style.position[0], layout.position_y()];
            target.style.position
        };
        for clip_id in cue_ids {
            if let Some(clip) = self.timeline.clip_mut(clip_id) {
                clip.transform.position.set_constant(position);
            }
        }
        Ok(())
    }

    /// Set (or clear) how a group highlights words during playback.
    pub fn set_caption_highlight(
        &mut self,
        group: CaptionGroupId,
        highlight: Option<CaptionHighlight>,
    ) -> Result<(), ModelError> {
        if let Some(highlight) = &highlight {
            highlight.validate()?;
        }
        self.timeline
            .caption_group_mut(group)
            .ok_or(ModelError::UnknownCaptionGroup(group))?
            .highlight = highlight;
        Ok(())
    }

    /// Rename a caption group (its label in the caption list).
    pub fn set_caption_group_label(
        &mut self,
        group: CaptionGroupId,
        label: String,
    ) -> Result<(), ModelError> {
        let target = self
            .timeline
            .caption_group_mut(group)
            .ok_or(ModelError::UnknownCaptionGroup(group))?;
        let before = std::mem::replace(&mut target.label, label);
        if let Err(e) = target.validate() {
            target.label = before;
            return Err(e);
        }
        Ok(())
    }

    /// Edit one cue's text (and optionally its word timings and speaker).
    ///
    /// `words: None` re-derives timings from the old ones by proportional
    /// remap, so correcting a typo keeps karaoke roughly in sync; `Some` sets
    /// them explicitly. Either way the result is validated against the new text,
    /// and the cue is flagged [`CaptionCue::text_edited`] so re-running
    /// recognition will not silently overwrite the correction.
    pub fn set_caption_cue(
        &mut self,
        clip_id: ClipId,
        text: String,
        words: Option<Vec<CaptionWord>>,
        speaker: Option<String>,
    ) -> Result<(), ModelError> {
        if text.trim().is_empty() {
            return Err(ModelError::InvalidParam("a caption cue needs text".into()));
        }
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let cue = clip
            .caption
            .as_ref()
            .ok_or(ModelError::NotACaptionCue(clip_id))?;

        let words = match words {
            Some(words) => words,
            None => retime_words(&cue.words, &text),
        };
        let mut updated = cue.clone();
        updated.words = words;
        updated.speaker = speaker;
        updated.text_edited = true;
        updated.validate(&text)?;

        let ClipSource::Generated(Generator::Text { content, .. }) = &mut clip.content else {
            return Err(ModelError::NotACaptionCue(clip_id));
        };
        *content = text;
        clip.caption = Some(updated);
        Ok(())
    }

    /// Split a caption cue at an absolute timeline tick, partitioning its word
    /// timings and text between the halves.
    ///
    /// Words entirely before the cut stay with the left half; the rest move to
    /// the right, with their times and byte ranges rebased. Without word
    /// timings the text goes to the left half and the right half repeats it, so
    /// the user can retype — matching what CapCut leaves behind.
    pub fn split_caption_cue(
        &mut self,
        clip_id: ClipId,
        at: crate::time::RationalTime,
    ) -> Result<ClipId, ModelError> {
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let cue = clip
            .caption
            .as_ref()
            .ok_or(ModelError::NotACaptionCue(clip_id))?
            .clone();
        let group = cue.group;
        let text = clip
            .text_content()
            .ok_or(ModelError::NotACaptionCue(clip_id))?
            .to_owned();
        let start = clip.timeline.start.value;
        let rate = clip.timeline.start.rate;
        let cut_ms = ticks_to_ms(at.value.saturating_sub(start), rate);

        let (left_words, mut right_words) = split_words(&cue.words, cut_ms);
        let (left_text, right_text) = split_text(&text, &left_words, &mut right_words);

        // A general split refuses to re-time a clip's look animation, which
        // would rule out almost every animated cue: a one-line karaoke cue is
        // shorter than the half-second entrance window, so cutting it always
        // moves that window. A cue is not a shot, though — every cue carries
        // the whole group's look, the same rule a restyle writes through — so
        // the slots come off for the cut and go back on both halves, each line
        // playing the look in full.
        let looks = {
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .expect("clip existence checked above");
            LookSlots::take(clip)
        };
        let split = self.split_clip(clip_id, at);
        // Restored before the error is propagated, so a refused split leaves
        // the cue exactly as it was.
        if let Some(clip) = self.timeline.clip_mut(clip_id) {
            looks.restore(clip);
        }
        let right_id = split?;
        if let Some(clip) = self.timeline.clip_mut(right_id) {
            looks.restore(clip);
        }

        if let Some(clip) = self.timeline.clip_mut(clip_id) {
            set_cue_text(clip, left_text, left_words);
        }
        if let Some(clip) = self.timeline.clip_mut(right_id) {
            set_cue_text(clip, right_text, right_words);
            // `split_clip` clones the source clip, so the caption metadata
            // rode along; only the index needs renumbering below.
        }
        self.timeline.reindex_caption_group(group);
        Ok(right_id)
    }

    /// Merge consecutive caption cues of one group into the first, joining
    /// their text with spaces and concatenating their word timings.
    ///
    /// The surviving cue spans from the earliest start to the latest end, so
    /// merging two lines separated by a gap holds the joined line across it.
    pub fn merge_caption_cues(&mut self, clips: &[ClipId]) -> Result<ClipId, ModelError> {
        if clips.len() < 2 {
            return Err(ModelError::InvalidParam(
                "merging captions needs at least two cues".into(),
            ));
        }
        // Resolve the group from the first cue, then require every clip to
        // share it — merging across groups has no meaning.
        let first = *clips.first().expect("checked above");
        let group = self
            .timeline
            .clip(first)
            .and_then(|clip| clip.caption_group())
            .ok_or(ModelError::NotACaptionCue(first))?;

        let mut ordered: Vec<(i64, ClipId)> = Vec::with_capacity(clips.len());
        for &clip_id in clips {
            let clip = self
                .timeline
                .clip(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            if clip.caption_group() != Some(group) {
                return Err(ModelError::InvalidParam(
                    "every merged cue must belong to the same caption group".into(),
                ));
            }
            ordered.push((clip.timeline.start.value, clip_id));
        }
        ordered.sort_unstable();
        ordered.dedup_by_key(|(_, id)| *id);

        let target_id = ordered[0].1;
        let rate = self
            .timeline
            .clip(target_id)
            .expect("checked above")
            .timeline
            .start
            .rate;
        let base_start = ordered[0].0;
        let mut end_tick = base_start;
        let mut text = String::new();
        let mut words: Vec<CaptionWord> = Vec::new();

        for &(start, clip_id) in &ordered {
            let clip = self.timeline.clip(clip_id).expect("checked above");
            let piece = clip.text_content().unwrap_or_default().to_owned();
            let cue_words = clip
                .caption
                .as_ref()
                .map(|cue| cue.words.clone())
                .unwrap_or_default();
            end_tick = end_tick.max(clip.timeline.end_tick());

            if !text.is_empty() && !piece.is_empty() {
                text.push(' ');
            }
            let byte_offset = u32::try_from(text.len()).unwrap_or(u32::MAX);
            let ms_offset = ticks_to_ms(start - base_start, rate);
            words.extend(cue_words.into_iter().map(|word| CaptionWord {
                start_ms: word.start_ms.saturating_add(ms_offset),
                end_ms: word.end_ms.saturating_add(ms_offset),
                range: word.range.start.saturating_add(byte_offset)
                    ..word.range.end.saturating_add(byte_offset),
            }));
            text.push_str(&piece);
        }

        // Free the span before growing the survivor into it.
        for &(_, clip_id) in &ordered[1..] {
            self.timeline.remove_clip(clip_id);
        }
        let merged = TimeRange::at_rate(base_start, (end_tick - base_start).max(1), rate);
        let clip = self
            .timeline
            .clip_mut(target_id)
            .ok_or(ModelError::UnknownClip(target_id))?;
        clip.timeline = merged;
        set_cue_text(clip, text, words);
        self.timeline.reindex_caption_group(group);
        Ok(target_id)
    }

    /// Detach cues from their group, leaving them as ordinary text clips. The
    /// group is dropped once it has no members.
    pub fn ungroup_caption_cues(&mut self, clips: &[ClipId]) -> Result<(), ModelError> {
        for &clip_id in clips {
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            if clip.caption.is_none() {
                return Err(ModelError::NotACaptionCue(clip_id));
            }
            clip.caption = None;
        }
        self.timeline.prune_empty_caption_groups();
        Ok(())
    }

    /// `Ok` iff `track` exists and is a text lane.
    fn require_text_track(&self, track: TrackId) -> Result<(), ModelError> {
        let kind = self
            .timeline
            .track(track)
            .ok_or(ModelError::UnknownTrack(track))?
            .kind;
        if kind != TrackKind::Text {
            return Err(ModelError::IncompatibleTrackKind { track, kind });
        }
        Ok(())
    }

    /// Check a whole batch of cue placements before any of it is inserted:
    /// each spec is self-consistent, the batch is ascending and
    /// non-overlapping, and nothing collides with the lane's existing clips.
    fn validate_cue_placements(
        &self,
        track: TrackId,
        cues: &[CaptionCueSpec],
    ) -> Result<(), ModelError> {
        let rate = self.timeline.frame_rate;
        let lane = self
            .timeline
            .track(track)
            .ok_or(ModelError::UnknownTrack(track))?;
        let mut previous_end = i64::MIN;
        for cue in cues {
            cue.validate()?;
            crate::time::check_same_rate(cue.timeline.start.rate, rate)?;
            crate::time::check_same_rate(cue.timeline.duration.rate, rate)?;
            if cue.timeline.start.value < previous_end {
                return Err(ModelError::InvalidParam(
                    "caption cues must be ascending and non-overlapping".into(),
                ));
            }
            previous_end = cue.timeline.end_tick();
            if lane.has_overlap(cue.timeline, None)? {
                return Err(ModelError::Overlap(track));
            }
        }
        Ok(())
    }
}

/// Write a caption group's shared look onto one cue clip: layer styles,
/// animation slots, and the transform's position/scale. Rotation, opacity,
/// anchor, and any keyframes on them are left alone — a restyle is not a
/// transform reset.
fn apply_caption_style(clip: &mut Clip, style: &CaptionStyle) {
    clip.styles = style.styles.clone();
    clip.animation_in = style.animation_in.clone();
    clip.animation_out = style.animation_out.clone();
    clip.animation_combo = style.animation_combo.clone();
    clip.transform.position = Param::Constant(style.position);
    clip.transform.scale = Param::Constant(crate::clip::Scale2::uniform(style.scale));
}

/// Replace a cue clip's text and word table together, so the byte ranges can
/// never describe stale text.
fn set_cue_text(clip: &mut Clip, text: String, words: Vec<CaptionWord>) {
    if let ClipSource::Generated(Generator::Text { content, .. }) = &mut clip.content {
        *content = text;
    }
    if let Some(cue) = clip.caption.as_mut() {
        cue.words = words;
    }
}

/// Milliseconds for a tick count at `rate`, clamped to a sane range.
fn ticks_to_ms(ticks: i64, rate: crate::time::Rational) -> u32 {
    let ms = (ticks.max(0) as f64 * rate.seconds_per_unit() * 1000.0).round();
    if ms.is_finite() {
        ms.clamp(0.0, f64::from(u32::MAX)) as u32
    } else {
        0
    }
}

/// Remap word timings onto edited text.
///
/// Byte ranges cannot survive an arbitrary edit, so this re-splits the new text
/// on whitespace and re-attaches timings. Fixing a typo leaves the word count
/// alone, and that case keeps the original per-word timings exactly; adding or
/// removing words falls back to spreading the cue's overall span evenly, which
/// keeps the highlight monotonic and roughly in sync. A cue with no prior
/// timings stays untimed rather than inventing any.
fn retime_words(previous: &[CaptionWord], text: &str) -> Vec<CaptionWord> {
    let (Some(first), Some(last)) = (previous.first(), previous.last()) else {
        return Vec::new();
    };
    let pieces: Vec<(usize, &str)> = text
        .split_whitespace()
        .map(|piece| {
            let offset = piece.as_ptr() as usize - text.as_ptr() as usize;
            (offset, piece)
        })
        .collect();
    if pieces.is_empty() {
        return Vec::new();
    }
    let span_start = first.start_ms;
    let span = last.end_ms.saturating_sub(span_start);
    let count = pieces.len() as u32;
    pieces
        .iter()
        .enumerate()
        .map(|(index, (offset, piece))| {
            let (start_ms, end_ms) = if pieces.len() == previous.len() {
                (previous[index].start_ms, previous[index].end_ms)
            } else {
                let step = |i: u32| {
                    span_start + (u64::from(span) * u64::from(i) / u64::from(count)) as u32
                };
                (step(index as u32), step(index as u32 + 1))
            };
            CaptionWord {
                start_ms,
                end_ms,
                range: u32::try_from(*offset).unwrap_or(u32::MAX)
                    ..u32::try_from(offset + piece.len()).unwrap_or(u32::MAX),
            }
        })
        .collect()
}

/// A cue's look-animation slots, lifted off it for the duration of a split.
struct LookSlots {
    animation_in: Option<AnimationRef>,
    animation_out: Option<AnimationRef>,
    animation_combo: Option<AnimationRef>,
}

impl LookSlots {
    fn take(clip: &mut Clip) -> Self {
        Self {
            animation_in: clip.animation_in.take(),
            animation_out: clip.animation_out.take(),
            animation_combo: clip.animation_combo.take(),
        }
    }

    fn restore(&self, clip: &mut Clip) {
        clip.animation_in = self.animation_in.clone();
        clip.animation_out = self.animation_out.clone();
        clip.animation_combo = self.animation_combo.clone();
    }
}

/// Partition word timings at `cut_ms`: words that end at or before the cut stay
/// left, the rest move right with their times rebased to the new cue start.
fn split_words(words: &[CaptionWord], cut_ms: u32) -> (Vec<CaptionWord>, Vec<CaptionWord>) {
    let split = words.partition_point(|word| word.end_ms <= cut_ms);
    let left = words[..split].to_vec();
    let right = words[split..]
        .iter()
        .map(|word| CaptionWord {
            start_ms: word.start_ms.saturating_sub(cut_ms),
            end_ms: word.end_ms.saturating_sub(cut_ms),
            range: word.range.clone(),
        })
        .collect();
    (left, right)
}

/// Split cue text to match a word partition, rebasing the right half's byte
/// ranges onto its own shorter string. With no word timings both halves keep
/// the whole text — the user retypes one of them.
fn split_text(
    text: &str,
    left_words: &[CaptionWord],
    right_words: &mut [CaptionWord],
) -> (String, String) {
    let (Some(last_left), Some(first_right)) = (left_words.last(), right_words.first()) else {
        return (text.to_owned(), text.to_owned());
    };
    let cut = usize::min(
        first_right.range.start as usize,
        text.len().max(last_left.range.end as usize),
    );
    if !text.is_char_boundary(cut) {
        return (text.to_owned(), text.to_owned());
    }
    let left = text[..cut].trim_end().to_owned();
    let right = text[cut..].trim_start().to_owned();
    // Rebase the right half's ranges onto `right`, accounting for the
    // whitespace `trim_start` removed.
    let trimmed = cut + (text[cut..].len() - text[cut..].trim_start().len());
    let shift = u32::try_from(trimmed).unwrap_or(u32::MAX);
    for word in right_words.iter_mut() {
        word.range = word.range.start.saturating_sub(shift)..word.range.end.saturating_sub(shift);
    }
    (left, right)
}
