use super::*;

/// Look up a caption group, listing the groups that do exist on failure.
pub(super) fn caption_group_ref(project: &Project, raw: u64) -> Result<&CaptionGroup, Rejection> {
    project
        .timeline()
        .caption_group(CaptionGroupId::from_raw(raw))
        .ok_or_else(|| {
            let existing = project
                .timeline()
                .caption_groups_ordered()
                .iter()
                .map(|group| group.id.raw())
                .collect();
            Rejection::new(format!(
                "caption group {raw} does not exist; caption groups: {}",
                list_ids(existing)
            ))
        })
}

/// The cue metadata of a caption clip, rejecting ordinary titles by name.
pub(super) fn caption_cue_ref(project: &Project, raw: u64) -> Result<&Clip, Rejection> {
    let clip = clip_ref(project, raw)?;
    if clip.caption.is_none() {
        return Err(Rejection::new(format!(
            "clip {raw} is not a caption cue; caption tools only work on clips in a \
             caption group (see describe_project). Use set_generator to edit a plain title"
        )));
    }
    Ok(clip)
}

pub(super) fn caption_template_ids() -> String {
    cutlass_models::caption_template_catalog()
        .iter()
        .map(|spec| spec.id)
        .collect::<Vec<_>>()
        .join(", ")
}

fn require_caption_template(id: &str) -> Result<(), Rejection> {
    if cutlass_models::caption_template_spec(id).is_none() {
        return Err(Rejection::new(format!(
            "unknown caption template '{id}'; available templates: {}",
            caption_template_ids()
        )));
    }
    Ok(())
}

/// Lower `add_captions` into the group + cue specs the engine places atomically.
pub(super) fn add_captions(
    project: &Project,
    args: &crate::wire::AddCaptions,
) -> Result<EditCommand, Rejection> {
    let track = track_ref(project, args.track)?;
    if track.kind != TrackKind::Text {
        return Err(Rejection::new(format!(
            "track {} is a {} lane; captions need a text track — call \
             add_track with kind \"text\" first",
            args.track,
            kind_name(track.kind),
        )));
    }
    if args.cues.is_empty() {
        return Err(Rejection::new(
            "add_captions needs at least one cue".to_string(),
        ));
    }
    if args.cues.len() > MAX_WIRE_CAPTION_CUES {
        return Err(Rejection::new(format!(
            "add_captions accepts at most {MAX_WIRE_CAPTION_CUES} cues per call (got {}); \
             split a long script across several calls",
            args.cues.len()
        )));
    }
    if let Some(template) = &args.template {
        require_caption_template(template)?;
    }

    let mut cues = Vec::with_capacity(args.cues.len());
    let mut previous_end = i64::MIN;
    for (index, cue) in args.cues.iter().enumerate() {
        if cue.text.trim().is_empty() {
            return Err(Rejection::new(format!(
                "caption cue {index} has no text; every line needs something to show"
            )));
        }
        let timeline = timeline_range(project, cue.start, cue.duration)?;
        if timeline.start.value < previous_end {
            return Err(Rejection::new(format!(
                "caption cue {index} starts at {:.3}s, before the previous line ends at \
                 {:.3}s; cues must be in order and must not overlap",
                cue.start,
                ticks_to_seconds(previous_end, timeline_rate(project)),
            )));
        }
        previous_end = timeline.end_tick();
        cues.push(CaptionCueSpec::new(cue.text.clone(), timeline));
    }

    let label = args
        .label
        .clone()
        .filter(|label| !label.trim().is_empty())
        .unwrap_or_else(|| "Captions".to_string());
    Ok(EditCommand::AddCaptionGroup {
        group: Box::new(CaptionGroupSpec {
            track: track.id,
            label,
            source: CaptionSource::Manual,
            template: args.template.clone(),
            style: None,
            layout: None,
            highlight: None,
        }),
        cues,
    })
}

/// Patch a group's shared style, keeping every field the call omitted.
pub(super) fn set_caption_style(
    project: &Project,
    args: &crate::wire::SetCaptionStyle,
) -> Result<EditCommand, Rejection> {
    let group = caption_group_ref(project, args.group)?;
    let mut style = group.style.clone();

    if let Some(font) = &args.font {
        style.text.font = font.clone();
    }
    if let Some(size) = args.size {
        style.text.size = Param::Constant(finite(size, "size")? as f32);
    }
    if let Some(fill) = args.fill {
        style.text.fill = Param::Constant(fill);
    }
    if let Some(bold) = args.bold {
        style.text.bold = bold;
    }
    if let Some(italic) = args.italic {
        style.text.italic = italic;
    }
    if let Some(uppercase) = args.uppercase {
        style.text.case = if uppercase {
            cutlass_models::TextCase::Upper
        } else {
            cutlass_models::TextCase::Normal
        };
    }
    if let Some(y) = args.position_y {
        let y = finite(y, "position_y")?;
        if !(-0.5..=0.5).contains(&y) {
            return Err(Rejection::new(format!(
                "position_y must be between -0.5 (top edge) and 0.5 (bottom edge), got {y}"
            )));
        }
        style.position[1] = y as f32;
    }
    if let Some(scale) = args.scale {
        style.scale = finite(scale, "scale")? as f32;
    }
    style
        .validate()
        .map_err(|e| Rejection::new(format!("invalid caption style: {e}")))?;

    Ok(EditCommand::SetCaptionGroupStyle {
        group: group.id,
        style: Box::new(style),
        scope: if args.keep_overrides.unwrap_or(false) {
            CaptionStyleScope::KeepOverrides
        } else {
            CaptionStyleScope::All
        },
    })
}

/// Patch a group's segmentation rules, keeping every field the call omitted.
pub(super) fn set_caption_layout(
    project: &Project,
    args: &crate::wire::SetCaptionLayout,
) -> Result<EditCommand, Rejection> {
    let group = caption_group_ref(project, args.group)?;
    let mut layout = group.layout;

    if let Some(chars) = args.max_chars_per_line {
        layout.max_chars_per_line = u16::try_from(chars).unwrap_or(u16::MAX);
    }
    if let Some(lines) = args.max_lines {
        layout.max_lines = u8::try_from(lines).unwrap_or(u8::MAX);
    }
    if let Some(seconds) = args.min_duration {
        layout.min_duration_ms = milliseconds(seconds, "min_duration")?;
    }
    if let Some(seconds) = args.max_duration {
        layout.max_duration_ms = milliseconds(seconds, "max_duration")?;
    }
    if let Some(seconds) = args.min_gap {
        layout.min_gap_ms = milliseconds(seconds, "min_gap")?;
    }
    if let Some(safe_area) = args.safe_area_bottom {
        layout.safe_area_bottom = finite(safe_area, "safe_area_bottom")? as f32;
    }
    layout
        .validate()
        .map_err(|e| Rejection::new(format!("invalid caption layout: {e}")))?;

    Ok(EditCommand::SetCaptionGroupLayout {
        group: group.id,
        layout,
    })
}

/// Lower a highlight request; `mode: off` clears the group's highlight.
pub(super) fn set_caption_highlight(
    project: &Project,
    args: &crate::wire::SetCaptionHighlight,
) -> Result<EditCommand, Rejection> {
    let group = caption_group_ref(project, args.group)?;
    let mode = match args.mode {
        WireCaptionHighlightMode::Off => {
            return Ok(EditCommand::SetCaptionHighlight {
                group: group.id,
                highlight: None,
            });
        }
        WireCaptionHighlightMode::Word => CaptionHighlightMode::Word,
        WireCaptionHighlightMode::Line => CaptionHighlightMode::Line,
    };
    // Patch whatever the group (or its template) already had, so turning a
    // highlight from word to line does not reset its colors.
    let mut highlight = group.highlight.clone().unwrap_or_default();
    highlight.mode = mode;
    if let Some(fill) = args.fill {
        highlight.fill = fill;
    }
    if let Some(plate) = args.plate {
        highlight.plate = (plate[3] > 0).then_some(plate);
    }
    if let Some(radius) = args.plate_radius {
        highlight.plate_radius = finite(radius, "plate_radius")? as f32;
    }
    if let Some(scale) = args.scale {
        highlight.scale = finite(scale, "scale")? as f32;
    }
    highlight
        .validate()
        .map_err(|e| Rejection::new(format!("invalid caption highlight: {e}")))?;

    Ok(EditCommand::SetCaptionHighlight {
        group: group.id,
        highlight: Some(highlight),
    })
}

/// Lower a cue text edit, keeping the cue's speaker unless one is given.
pub(super) fn set_caption_text(
    project: &Project,
    args: &crate::wire::SetCaptionText,
) -> Result<EditCommand, Rejection> {
    let clip = caption_cue_ref(project, args.clip)?;
    if args.text.trim().is_empty() {
        return Err(Rejection::new(format!(
            "caption cue {} needs text; remove the line with remove_clip instead",
            args.clip
        )));
    }
    let speaker = args
        .speaker
        .clone()
        .or_else(|| clip.caption.as_ref().and_then(|cue| cue.speaker.clone()));
    Ok(EditCommand::SetCaptionCue {
        clip: clip.id,
        text: args.text.clone(),
        // `None` remaps the existing word timings onto the new text.
        words: None,
        speaker,
    })
}

/// Lower a merge, requiring two or more cues of one group.
pub(super) fn merge_captions(
    project: &Project,
    args: &crate::wire::MergeCaptions,
) -> Result<EditCommand, Rejection> {
    if args.clips.len() < 2 {
        return Err(Rejection::new(
            "merge_captions needs at least two cue clips".to_string(),
        ));
    }
    if args.clips.len() > MAX_MULTI_CLIP_REFS {
        return Err(Rejection::new(format!(
            "merge_captions accepts at most {MAX_MULTI_CLIP_REFS} cue clips (got {})",
            args.clips.len()
        )));
    }
    let mut clips = Vec::with_capacity(args.clips.len());
    let mut group = None;
    for &raw in &args.clips {
        let clip = caption_cue_ref(project, raw)?;
        let clip_group = clip.caption_group();
        match group {
            None => group = clip_group,
            Some(first) if clip_group != Some(first) => {
                return Err(Rejection::new(format!(
                    "cue {raw} belongs to caption group {} but the first cue belongs to \
                     group {}; merging across groups has no meaning",
                    clip_group.map_or(0, |g| g.raw()),
                    first.raw(),
                )));
            }
            Some(_) => {}
        }
        if clips.contains(&clip.id) {
            return Err(Rejection::new(format!(
                "merge_captions lists cue {raw} twice; list each cue once"
            )));
        }
        clips.push(clip.id);
    }
    Ok(EditCommand::MergeCaptionCues { clips })
}

fn finite(value: f64, what: &str) -> Result<f64, Rejection> {
    if !value.is_finite() {
        return Err(Rejection::new(format!("{what} must be a finite number")));
    }
    Ok(value)
}

/// Seconds → whole milliseconds, the unit caption layout rules use.
fn milliseconds(seconds: f64, what: &str) -> Result<u32, Rejection> {
    let ms = finite(seconds, what)? * 1000.0;
    if !(0.0..=f64::from(u32::MAX)).contains(&ms) {
        return Err(Rejection::new(format!(
            "{what} of {seconds}s is out of range"
        )));
    }
    Ok(ms.round() as u32)
}
