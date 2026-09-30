//! Caption edits (captions): create a whole group of cues in one command,
//! restyle a group, and edit / split / merge individual cues.
//!
//! Every caption edit shares one inverse shape — [`RestoreCaptionGroupAction`],
//! a snapshot of the group and its cue clips. That is deliberate: a restyle, a
//! reflow, a split, and a merge can each add, remove, or reshape any subset of
//! the group's cues, so a snapshot is both the simplest exact inverse and the
//! only one that stays correct as caption editing grows. Cue ids are preserved,
//! so selection and deeper history entries keep resolving across undo/redo.

use cutlass_models::{
    CaptionCue, CaptionCueSpec, CaptionGroup, CaptionGroupId, CaptionGroupSpec, CaptionHighlight,
    CaptionLayout, CaptionStyle, CaptionStyleScope, CaptionWord, Clip, ClipId, ModelError,
    RationalTime,
};

use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Create a caption group and all its cue clips. The inverse removes both.
pub fn add(
    ctx: &mut ApplyContext<'_>,
    spec: &CaptionGroupSpec,
    cues: &[CaptionCueSpec],
) -> Result<(CaptionGroupId, Box<dyn EditAction>), EngineError> {
    let (group, _) = ctx.project.add_caption_group(spec, cues)?;
    Ok((group, Box::new(RemoveCaptionGroupAction { group })))
}

/// Replace a group's shared style, writing it through to its cues.
pub fn set_style(
    ctx: &mut ApplyContext<'_>,
    group: CaptionGroupId,
    style: CaptionStyle,
    scope: CaptionStyleScope,
) -> Result<Box<dyn EditAction>, EngineError> {
    let undo = snapshot(ctx, group)?;
    ctx.project.set_caption_group_style(group, style, scope)?;
    Ok(Box::new(undo))
}

/// Replace a group's segmentation rules.
pub fn set_layout(
    ctx: &mut ApplyContext<'_>,
    group: CaptionGroupId,
    layout: CaptionLayout,
) -> Result<Box<dyn EditAction>, EngineError> {
    let undo = snapshot(ctx, group)?;
    ctx.project.set_caption_group_layout(group, layout)?;
    Ok(Box::new(undo))
}

/// Apply a caption template's style, layout, and highlight.
pub fn set_template(
    ctx: &mut ApplyContext<'_>,
    group: CaptionGroupId,
    template: &str,
) -> Result<Box<dyn EditAction>, EngineError> {
    let undo = snapshot(ctx, group)?;
    ctx.project.set_caption_group_template(group, template)?;
    Ok(Box::new(undo))
}

/// Rename a group.
pub fn set_label(
    ctx: &mut ApplyContext<'_>,
    group: CaptionGroupId,
    label: String,
) -> Result<Box<dyn EditAction>, EngineError> {
    let undo = snapshot(ctx, group)?;
    ctx.project.set_caption_group_label(group, label)?;
    Ok(Box::new(undo))
}

/// Set (or clear) a group's word highlighting.
pub fn set_highlight(
    ctx: &mut ApplyContext<'_>,
    group: CaptionGroupId,
    highlight: Option<CaptionHighlight>,
) -> Result<Box<dyn EditAction>, EngineError> {
    let undo = snapshot(ctx, group)?;
    ctx.project.set_caption_highlight(group, highlight)?;
    Ok(Box::new(undo))
}

/// Edit one cue's text, word timings, and speaker.
pub fn set_cue(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    text: String,
    words: Option<Vec<CaptionWord>>,
    speaker: Option<String>,
) -> Result<Box<dyn EditAction>, EngineError> {
    let undo = snapshot(ctx, group_of(ctx, clip)?)?;
    ctx.project.set_caption_cue(clip, text, words, speaker)?;
    Ok(Box::new(undo))
}

/// Split a cue, partitioning its text and word timings. Returns the new
/// right-hand cue's clip id.
pub fn split_cue(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    at: RationalTime,
) -> Result<(ClipId, Box<dyn EditAction>), EngineError> {
    let undo = snapshot(ctx, group_of(ctx, clip)?)?;
    let right = ctx.project.split_caption_cue(clip, at)?;
    Ok((right, Box::new(undo)))
}

/// Merge cues into the earliest of them. Returns the surviving clip id.
pub fn merge_cues(
    ctx: &mut ApplyContext<'_>,
    clips: &[ClipId],
) -> Result<(ClipId, Box<dyn EditAction>), EngineError> {
    let first = clips.first().copied().ok_or(ModelError::InvalidParam(
        "merging captions needs at least two cues".into(),
    ))?;
    let undo = snapshot(ctx, group_of(ctx, first)?)?;
    let merged = ctx.project.merge_caption_cues(clips)?;
    Ok((merged, Box::new(undo)))
}

/// Detach cues from their group, leaving plain text clips. The inverse
/// re-attaches the metadata in place (the clips never move), so it cannot use
/// the snapshot restore.
pub fn ungroup(
    ctx: &mut ApplyContext<'_>,
    clips: &[ClipId],
) -> Result<Box<dyn EditAction>, EngineError> {
    let mut groups: Vec<CaptionGroup> = Vec::new();
    let mut cues: Vec<(ClipId, CaptionCue)> = Vec::with_capacity(clips.len());
    for &clip_id in clips {
        let clip = ctx
            .project
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let cue = clip
            .caption
            .clone()
            .ok_or(ModelError::NotACaptionCue(clip_id))?;
        if !groups.iter().any(|group| group.id == cue.group) {
            let group = ctx
                .project
                .timeline()
                .caption_group(cue.group)
                .cloned()
                .ok_or(ModelError::UnknownCaptionGroup(cue.group))?;
            groups.push(group);
        }
        cues.push((clip_id, cue));
    }
    ctx.project.ungroup_caption_cues(clips)?;
    Ok(Box::new(RegroupCaptionCuesAction { groups, cues }))
}

/// Remove a caption group and its cues; the inverse restores both.
pub struct RemoveCaptionGroupAction {
    pub group: CaptionGroupId,
}

impl EditAction for RemoveCaptionGroupAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        let (group, cues) = ctx.project.remove_caption_group(self.group)?;
        Ok(Box::new(RestoreCaptionGroupAction { group, cues }))
    }
}

/// Restore a caption group and its cue clips exactly as captured, replacing
/// whatever the group holds now.
pub struct RestoreCaptionGroupAction {
    group: CaptionGroup,
    cues: Vec<Clip>,
}

impl EditAction for RestoreCaptionGroupAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        let id = self.group.id;
        // Capture the current state first, so this action oscillates.
        let redo = if ctx.project.timeline().caption_group(id).is_some() {
            let (group, cues) = ctx.project.remove_caption_group(id)?;
            RestoreCaptionGroupAction { group, cues }
        } else {
            // Undo of a remove: there is nothing to capture, and the redo is
            // the remove itself.
            ctx.project.restore_caption_group(self.group, self.cues)?;
            return Ok(Box::new(RemoveCaptionGroupAction { group: id }));
        };
        ctx.project.restore_caption_group(self.group, self.cues)?;
        Ok(Box::new(redo))
    }
}

/// Re-attach caption metadata to clips that were ungrouped, restoring their
/// groups if the last member leaving dropped them.
struct RegroupCaptionCuesAction {
    groups: Vec<CaptionGroup>,
    cues: Vec<(ClipId, CaptionCue)>,
}

impl EditAction for RegroupCaptionCuesAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        let clips: Vec<ClipId> = self.cues.iter().map(|(id, _)| *id).collect();
        let timeline = ctx.project.timeline_mut();
        for group in self.groups {
            if timeline.caption_group(group.id).is_none() {
                timeline.add_caption_group(group)?;
            }
        }
        for (clip_id, cue) in self.cues {
            let group = cue.group;
            timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?
                .caption = Some(cue);
            timeline.reindex_caption_group(group);
        }
        Ok(Box::new(UngroupCaptionCuesAction { clips }))
    }
}

/// Redo of an ungroup.
struct UngroupCaptionCuesAction {
    clips: Vec<ClipId>,
}

impl EditAction for UngroupCaptionCuesAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        ungroup(ctx, &self.clips)
    }
}

/// The group `clip` is a cue of, or an error naming why it is not one.
fn group_of(ctx: &ApplyContext<'_>, clip: ClipId) -> Result<CaptionGroupId, EngineError> {
    ctx.project
        .clip(clip)
        .ok_or(ModelError::UnknownClip(clip))?
        .caption_group()
        .ok_or_else(|| ModelError::NotACaptionCue(clip).into())
}

/// Capture a group and its cue clips as an undo action.
fn snapshot(
    ctx: &ApplyContext<'_>,
    group: CaptionGroupId,
) -> Result<RestoreCaptionGroupAction, EngineError> {
    let timeline = ctx.project.timeline();
    let captured = timeline
        .caption_group(group)
        .cloned()
        .ok_or(ModelError::UnknownCaptionGroup(group))?;
    let cues = timeline.caption_cues(group).into_iter().cloned().collect();
    Ok(RestoreCaptionGroupAction {
        group: captured,
        cues,
    })
}
