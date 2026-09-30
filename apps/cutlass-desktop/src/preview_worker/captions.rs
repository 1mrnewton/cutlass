//! Caption edits: create, restyle, retime, and re-segment caption groups.
//!
//! Every entry point here is one undoable engine edit (or one history group),
//! then a projection republish, matching the rest of the worker. The caption
//! commands do the heavy lifting — the work in this module is resolving raw UI
//! ids onto the engine, and turning a subtitle file into cue specs.

use super::*;

use cutlass_captions::{
    ImportOptions, Placement, overflow_cuts, parse_subtitles, place_subtitles, wrap,
};
use cutlass_models::{
    CaptionCueSpec, CaptionGroupId, CaptionGroupSpec, CaptionHighlight, CaptionHighlightMode,
    CaptionLayout, CaptionSource, CaptionStyle, CaptionStyleScope, MediaId, TextStyle,
};

/// Default hold for a hand-added caption line, in seconds — long enough to type
/// into, short enough to sit between two others.
const MANUAL_CUE_SECONDS: i64 = 3;

/// Cues produced by the transcription job, ready to place.
///
/// Boxed inside [`CaptionOp`]: it is by far the largest payload the caption ops
/// carry, and every other worker message would otherwise pay for its size.
#[derive(Debug, Clone)]
pub struct TranscribedCaptions {
    /// Segmented cues, already placed on the timeline's clock.
    pub cues: Vec<CaptionCueSpec>,
    pub label: String,
    pub template: String,
    /// The layout the job segmented with, so the group's later reflows match.
    pub layout: CaptionLayout,
    /// Pool id of the transcribed asset, for the group's provenance.
    pub media: String,
    pub language: Option<String>,
    /// Recognizer model id, so a re-run can be compared against this pass.
    pub model: String,
}

/// One caption edit from the UI. Bundled into a single [`WorkerMsg`] variant
/// rather than a dozen: they all resolve raw ids, apply one caption command,
/// and republish, so the routing table gains one arm instead of twelve.
#[derive(Debug, Clone)]
pub enum CaptionOp {
    /// Create a one-cue group at `tick` on `track` (a fresh text lane when
    /// `track` names none), styled by `template`.
    AddGroup {
        track: String,
        tick: i64,
        text: String,
        template: String,
    },
    /// Import a subtitle file as a new caption group starting at `tick`.
    /// Failures land in `CaptionBackend.import-error`.
    ImportFile {
        path: PathBuf,
        tick: i64,
        template: String,
    },
    /// Place the cues a transcription job produced (see `crate::auto_captions`).
    /// Segmentation already happened off-thread; this only resolves the lane and
    /// applies one `AddCaptionGroup`.
    AddTranscribed {
        captions: Box<TranscribedCaptions>,
        /// How many cues landed, or why none did. The job waits for this so the
        /// dialog reports what actually happened instead of assuming the edit
        /// was accepted.
        reply: Sender<Result<usize, String>>,
    },
    RemoveGroup {
        group: String,
    },
    Ungroup {
        group: String,
    },
    SetLabel {
        group: String,
        label: String,
    },
    SetTemplate {
        group: String,
        template: String,
    },
    /// Write one cue's look through to the group (CapCut "Apply to all").
    ApplyStyle {
        group: String,
        style: Box<TextStyle>,
        keep_overrides: bool,
    },
    SetLayout {
        group: String,
        max_chars_per_line: u16,
        max_lines: u8,
        safe_area_bottom: f32,
        /// Re-flow every cue onto the new rules: fresh line breaks, and a cut
        /// for any cue that no longer fits the line budget.
        reflow: bool,
    },
    SetHighlight {
        group: String,
        mode: CaptionHighlightMode,
        fill: [u8; 4],
        plate: Option<[u8; 4]>,
        scale: f32,
    },
    SetCueText {
        clip: String,
        text: String,
    },
    SplitCue {
        clip: String,
        tick: i64,
    },
    MergeWithNext {
        clip: String,
    },
}

pub(super) fn caption_op(engine: &mut Engine, op: CaptionOp, ui: &UiSink) {
    match op {
        CaptionOp::AddGroup {
            track,
            tick,
            text,
            template,
        } => add_manual_group(engine, &track, tick, text, &template, ui),
        CaptionOp::ImportFile {
            path,
            tick,
            template,
        } => import_subtitles(engine, &path, tick, &template, ui),
        CaptionOp::AddTranscribed { captions, reply } => {
            let outcome = add_transcribed(engine, *captions, ui);
            let _ = reply.send(outcome);
        }
        CaptionOp::RemoveGroup { group } => with_group(engine, &group, ui, |group| {
            EditCommand::RemoveCaptionGroup { group }
        }),
        CaptionOp::Ungroup { group } => ungroup(engine, &group, ui),
        CaptionOp::SetLabel { group, label } => with_group(engine, &group, ui, |group| {
            EditCommand::SetCaptionGroupLabel { group, label }
        }),
        CaptionOp::SetTemplate { group, template } => set_template(engine, &group, template, ui),
        CaptionOp::ApplyStyle {
            group,
            style,
            keep_overrides,
        } => apply_style(engine, &group, *style, keep_overrides, ui),
        CaptionOp::SetLayout {
            group,
            max_chars_per_line,
            max_lines,
            safe_area_bottom,
            reflow,
        } => set_layout(
            engine,
            &group,
            max_chars_per_line,
            max_lines,
            safe_area_bottom,
            reflow,
            ui,
        ),
        CaptionOp::SetHighlight {
            group,
            mode,
            fill,
            plate,
            scale,
        } => set_highlight(engine, &group, mode, fill, plate, scale, ui),
        CaptionOp::SetCueText { clip, text } => set_cue_text(engine, &clip, text, ui),
        CaptionOp::SplitCue { clip, tick } => split_cue(engine, &clip, tick, ui),
        CaptionOp::MergeWithNext { clip } => merge_with_next(engine, &clip, ui),
    }
}

/// Apply one caption command built from a resolved group id.
fn with_group(
    engine: &mut Engine,
    group: &str,
    ui: &UiSink,
    command: impl FnOnce(CaptionGroupId) -> EditCommand,
) {
    let Some(group_id) = caption_group_id(engine, group) else {
        error!(group, "caption edit ignored: unknown caption group");
        return;
    };
    apply_and_publish(engine, command(group_id), ui);
}

fn apply_and_publish(engine: &mut Engine, command: EditCommand, ui: &UiSink) {
    match engine.apply(Command::Edit(command)) {
        Ok(_) => publish_projection(engine, ui),
        Err(e) => error!("caption edit failed: {e}"),
    }
}

/// The group named by a raw id from the projection, when it exists.
fn caption_group_id(engine: &Engine, group: &str) -> Option<CaptionGroupId> {
    let id = CaptionGroupId::from_raw(parse_raw_id(group)?);
    engine
        .project()
        .timeline()
        .caption_group(id)
        .is_some()
        .then_some(id)
}

/// Hand-added captions: one cue at the playhead on the text lane, in the
/// picked template's look. Creating the lane (when there is none) rides the
/// same history group, so one undo removes the whole gesture.
fn add_manual_group(
    engine: &mut Engine,
    track: &str,
    tick: i64,
    text: String,
    template: &str,
    ui: &UiSink,
) {
    let rate = engine.project().timeline().frame_rate;
    let duration = (i64::from(rate.num) * MANUAL_CUE_SECONDS / i64::from(rate.den.max(1))).max(1);
    let desired = tick.max(0);

    engine.begin_group();
    let (track_id, start) = match lane_of_kind(engine, track, TrackKind::Text) {
        Some(lane) => {
            let lane_track = engine
                .project()
                .timeline()
                .track(lane)
                .expect("lane_of_kind returned an existing track");
            (lane, first_fit_start(lane_track, desired, duration))
        }
        // A text lane is inserted at the top of the visual stack; the model's
        // zone rules clamp it into the text band regardless of the row asked
        // for, so the row here is only a hint.
        None => match create_track(engine, TrackKind::Text, 0) {
            Ok(id) => (id, desired),
            Err(e) => {
                error!("add caption failed creating text track: {e}");
                engine.rollback_group();
                return;
            }
        },
    };

    let mut spec = CaptionGroupSpec::manual(track_id, "Captions");
    if !template.is_empty() {
        spec = spec.with_template(template);
    }
    let cue = CaptionCueSpec::new(
        if text.trim().is_empty() {
            "Caption".to_owned()
        } else {
            text
        },
        TimeRange::at_rate(start, duration, rate),
    );
    match engine.apply(Command::Edit(EditCommand::AddCaptionGroup {
        group: Box::new(spec),
        cues: vec![cue],
    })) {
        Ok(_) => {
            engine.commit_group();
            info!(%track_id, start, "added manual caption group");
            publish_projection(engine, ui);
        }
        Err(e) => {
            error!(%track_id, start, "add caption group failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
        }
    }
}

/// Import an `.srt` / `.vtt` file as a caption group whose first cue lands at
/// `tick`. Read, parse, and placement failures surface as session errors (the
/// same dialog a failed open uses) rather than failing silently — an import is
/// an explicit user gesture, and a parse error names the offending line.
fn import_subtitles(engine: &mut Engine, path: &Path, tick: i64, template: &str, ui: &UiSink) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => {
            error!(?path, "subtitle import failed reading file: {e}");
            publish_session_error(ui, format!("Could not read {}: {e}", file_name(path)));
            return;
        }
    };
    let (format, cues) = match parse_subtitles(&text) {
        Ok(parsed) => parsed,
        Err(e) => {
            error!(?path, "subtitle import failed parsing: {e}");
            publish_session_error(ui, format!("{}: {e}", file_name(path)));
            return;
        }
    };
    if cues.is_empty() {
        publish_session_error(ui, format!("{} has no caption cues", file_name(path)));
        return;
    }

    let rate = engine.project().timeline().frame_rate;
    let layout = template_layout(template);
    let mut options = ImportOptions::new(Placement::new(rate, tick.max(0))).with_layout(layout);
    // A subtitle file's own line breaks are someone's editorial choice; keep
    // them. Word timings are estimated so an imported group can still drive
    // the karaoke highlight.
    options.estimate_words = true;

    let specs = match place_subtitles(&cues, &options) {
        Ok(specs) => specs,
        Err(e) => {
            error!(?path, "subtitle import failed placing cues: {e}");
            publish_session_error(ui, format!("{}: {e}", file_name(path)));
            return;
        }
    };

    engine.begin_group();
    let track_id = match free_text_lane(engine, &specs) {
        Some(lane) => lane,
        None => match create_track(engine, TrackKind::Text, 0) {
            Ok(id) => id,
            Err(e) => {
                error!("subtitle import failed creating text track: {e}");
                engine.rollback_group();
                publish_session_error(ui, format!("Could not create a text lane: {e}"));
                return;
            }
        },
    };

    let mut spec = CaptionGroupSpec {
        track: track_id,
        label: subtitle_label(path),
        source: CaptionSource::Imported { format },
        template: None,
        style: None,
        layout: Some(layout),
        highlight: None,
    };
    if !template.is_empty() {
        spec = spec.with_template(template);
    }
    let count = specs.len();
    match engine.apply(Command::Edit(EditCommand::AddCaptionGroup {
        group: Box::new(spec),
        cues: specs,
    })) {
        Ok(_) => {
            engine.commit_group();
            info!(?path, count, "imported subtitles as a caption group");
            publish_projection(engine, ui);
        }
        Err(e) => {
            error!(?path, "subtitle import rejected: {e}");
            engine.rollback_group();
            publish_session_error(ui, format!("{}: {e}", file_name(path)));
            publish_projection(engine, ui);
        }
    }
}

/// Place transcribed cues as one auto-sourced caption group.
///
/// The job segmented against the same timeline rate, so nothing is retimed
/// here; what remains is finding the text lane (creating one inside the history
/// group when there is none) and recording which asset and model produced the
/// lines, so a later re-run can be compared against this pass.
///
/// Returns the number of cues placed, so the transcription dialog reports the
/// engine's answer rather than its own optimism.
fn add_transcribed(
    engine: &mut Engine,
    transcribed: TranscribedCaptions,
    ui: &UiSink,
) -> Result<usize, String> {
    let Some(media) = parse_raw_id(&transcribed.media).map(MediaId::from_raw) else {
        error!(
            media = transcribed.media,
            "auto captions ignored: unparsable media id"
        );
        return Err("The transcribed clip is no longer on the timeline".to_owned());
    };
    if engine.project().media(media).is_none() {
        return Err("The transcribed clip's media is no longer in this project".to_owned());
    }

    engine.begin_group();
    let track = match free_text_lane(engine, &transcribed.cues) {
        Some(lane) => lane,
        None => match create_track(engine, TrackKind::Text, 0) {
            Ok(id) => id,
            Err(e) => {
                error!("auto captions failed creating text track: {e}");
                engine.rollback_group();
                return Err(format!("Could not create a text lane: {e}"));
            }
        },
    };

    let mut spec = CaptionGroupSpec {
        track,
        label: transcribed.label,
        source: CaptionSource::Auto {
            media,
            language: transcribed.language,
            model: transcribed.model,
        },
        template: None,
        style: None,
        layout: Some(transcribed.layout),
        highlight: None,
    };
    if !transcribed.template.is_empty() {
        spec = spec.with_template(transcribed.template);
    }

    let count = transcribed.cues.len();
    match engine.apply(Command::Edit(EditCommand::AddCaptionGroup {
        group: Box::new(spec),
        cues: transcribed.cues,
    })) {
        Ok(_) => {
            engine.commit_group();
            info!(%media, count, "placed transcribed captions");
            publish_projection(engine, ui);
            Ok(count)
        }
        Err(e) => {
            error!(%media, count, "auto captions rejected: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            Err(format!("Could not add the captions: {e}"))
        }
    }
}

/// The lowest text lane with room for every cue.
///
/// A caption batch lands at times the transcript (or subtitle file) fixed, so
/// unlike a manual cue it cannot slide past a blocker. Reusing the first text
/// lane blindly means a second pass — captions for a newly imported clip that
/// starts where the first one did — is rejected wholesale for overlap after
/// the transcription has already run. Falling through to the next free lane,
/// and to a new one when every lane is busy, matches how a drop finds a home.
fn free_text_lane(engine: &Engine, cues: &[CaptionCueSpec]) -> Option<TrackId> {
    let mut spans: Vec<(i64, i64)> = cues
        .iter()
        .map(|cue| (cue.timeline.start.value, cue.timeline.end_tick()))
        .collect();
    spans.sort_unstable();
    engine
        .project()
        .timeline()
        .tracks_ordered()
        .find(|track| track.kind == TrackKind::Text && spans_free(track, &spans))
        .map(|track| track.id)
}

/// Whether every span in `spans` (sorted by start) clears every clip on
/// `track`. Both sides are in start order, so one merge walk answers it in
/// O(cues + clips) rather than scanning the lane once per cue.
fn spans_free(track: &Track, spans: &[(i64, i64)]) -> bool {
    let clips = track.clips_ordered();
    let mut next = 0;
    for &(start, end) in spans {
        while clips
            .get(next)
            .is_some_and(|clip| clip.timeline.end_tick() <= start)
        {
            next += 1;
        }
        match clips.get(next) {
            Some(clip) if clip.timeline.start.value < end => return false,
            Some(_) => {}
            None => return true,
        }
    }
    true
}

/// The named template's segmentation rules, or the defaults.
fn template_layout(template: &str) -> CaptionLayout {
    cutlass_models::caption_template_spec(template)
        .map_or_else(CaptionLayout::default, |spec| spec.layout())
}

fn subtitle_label(path: &Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or_else(|| "Imported captions".to_owned())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

/// Detach every cue of a group, leaving ordinary text clips. Cue ids survive,
/// so selection and keyframes are untouched.
fn ungroup(engine: &mut Engine, group: &str, ui: &UiSink) {
    let Some(group_id) = caption_group_id(engine, group) else {
        error!(group, "ungroup ignored: unknown caption group");
        return;
    };
    let clips = engine.project().timeline().caption_cue_ids(group_id);
    if clips.is_empty() {
        error!(group, "ungroup ignored: group has no cues");
        return;
    }
    apply_and_publish(engine, EditCommand::UngroupCaptions { clips }, ui);
}

/// Write a cue's typography through to the whole group. The group keeps its
/// other style facets (placement, layer styles, animations) — only the text
/// style the inspector edits is replaced, so "Apply to all" can't silently
/// discard a template's glow or entrance.
fn apply_style(
    engine: &mut Engine,
    group: &str,
    text: TextStyle,
    keep_overrides: bool,
    ui: &UiSink,
) {
    let Some(group_id) = caption_group_id(engine, group) else {
        error!(group, "caption restyle ignored: unknown caption group");
        return;
    };
    let current = engine
        .project()
        .timeline()
        .caption_group(group_id)
        .expect("caption_group_id returned an existing group")
        .style
        .clone();
    let style = CaptionStyle { text, ..current };
    let scope = if keep_overrides {
        CaptionStyleScope::KeepOverrides
    } else {
        CaptionStyleScope::All
    };
    apply_and_publish(
        engine,
        EditCommand::SetCaptionGroupStyle {
            group: group_id,
            style: Box::new(style),
            scope,
        },
        ui,
    );
}

/// New segmentation rules, optionally re-flowing the existing cues onto them.
#[allow(clippy::too_many_arguments)]
fn set_layout(
    engine: &mut Engine,
    group: &str,
    max_chars_per_line: u16,
    max_lines: u8,
    safe_area_bottom: f32,
    reflow: bool,
    ui: &UiSink,
) {
    let Some(group_id) = caption_group_id(engine, group) else {
        error!(group, "caption layout ignored: unknown caption group");
        return;
    };
    let layout = CaptionLayout {
        max_chars_per_line,
        max_lines,
        safe_area_bottom,
        ..engine
            .project()
            .timeline()
            .caption_group(group_id)
            .expect("caption_group_id returned an existing group")
            .layout
    };

    // One history entry for the whole gesture: the rules and every re-flowed
    // line undo together.
    engine.begin_group();
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetCaptionGroupLayout {
        group: group_id,
        layout,
    })) {
        error!(group, "caption layout failed: {e}");
        engine.rollback_group();
        return;
    }

    if reflow {
        reflow_cues(engine, group_id, layout);
    }
    engine.commit_group();
    publish_projection(engine, ui);
}

/// Switch a group to a catalog template and re-flow its cues onto the line
/// rules that template carries. A look is its typography *and* its line
/// budget — Outline's single 20-character line is the look — so leaving the
/// cues wrapped for the previous template would only half-apply it.
fn set_template(engine: &mut Engine, group: &str, template: String, ui: &UiSink) {
    let Some(group_id) = caption_group_id(engine, group) else {
        error!(group, "caption template ignored: unknown caption group");
        return;
    };
    engine.begin_group();
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetCaptionGroupTemplate {
        group: group_id,
        template,
    })) {
        error!(group, "caption template failed: {e}");
        engine.rollback_group();
        return;
    }
    let layout = engine
        .project()
        .timeline()
        .caption_group(group_id)
        .expect("caption_group_id returned an existing group")
        .layout;
    reflow_cues(engine, group_id, layout);
    engine.commit_group();
    publish_projection(engine, ui);
}

/// Re-flow a group's cues onto `layout`: cut every cue the line budget no
/// longer holds, then wrap what is left to the character limit.
///
/// The two limits only mean anything together — a one-line budget at 16
/// characters is a request for short cues, not for a cue that quietly grows a
/// fifth line — so a shrunken budget re-times the group rather than being
/// ignored. Cutting comes first so the new pieces are wrapped too, and a cue's
/// cuts are applied back to front: every offset was measured against the cue
/// as it stands, and splitting from the end leaves the earlier offsets, and
/// the clip they address, untouched. Cues are only ever cut, never rejoined:
/// a widened budget re-wraps the lines it has rather than guessing which
/// neighbors used to be one line.
///
/// Assumes the caller is inside a history group.
fn reflow_cues(engine: &mut Engine, group: CaptionGroupId, layout: CaptionLayout) {
    let rate = engine.project().timeline().frame_rate;
    let placement = Placement::at_rate(rate);
    let cuts: Vec<(ClipId, Vec<i64>)> = engine
        .project()
        .timeline()
        .caption_cues(group)
        .into_iter()
        .filter_map(|clip| {
            let text = clip.text_content()?;
            let words = clip
                .caption
                .as_ref()
                .map(|cue| cue.words.as_slice())
                .unwrap_or_default();
            let start = clip.timeline.start.value;
            let end = clip.timeline.end_tick();
            let duration_ms = placement.ms(clip.timeline.duration.value);
            let mut ticks: Vec<i64> = overflow_cuts(text, words, duration_ms, &layout)
                .into_iter()
                .map(|ms| start + placement.ticks(ms))
                .filter(|tick| *tick > start && *tick < end)
                .collect();
            // Two cuts inside one frame round to the same tick; the second
            // would only ask to split on a boundary that no longer exists.
            ticks.dedup();
            (!ticks.is_empty()).then_some((clip.id, ticks))
        })
        .collect();
    for (clip, ticks) in cuts {
        for tick in ticks.into_iter().rev() {
            if let Err(e) = engine.apply(Command::Edit(EditCommand::SplitCaptionCue {
                clip,
                at: RationalTime::new(tick, rate),
            })) {
                warn!(%clip, tick, "caption reflow could not split an overlong cue: {e}");
            }
        }
    }

    let rewrapped: Vec<(ClipId, String)> = engine
        .project()
        .timeline()
        .caption_cues(group)
        .into_iter()
        .filter_map(|clip| {
            let text = clip.text_content()?;
            let wrapped = wrap(text, layout.max_chars_per_line);
            (wrapped != text).then_some((clip.id, wrapped))
        })
        .collect();
    for (clip, text) in rewrapped {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetCaptionCue {
            clip,
            text,
            words: None,
            speaker: None,
        })) {
            warn!(%clip, "caption reflow skipped a cue: {e}");
        }
    }
}

fn set_highlight(
    engine: &mut Engine,
    group: &str,
    mode: CaptionHighlightMode,
    fill: [u8; 4],
    plate: Option<[u8; 4]>,
    scale: f32,
    ui: &UiSink,
) {
    let Some(group_id) = caption_group_id(engine, group) else {
        error!(group, "caption highlight ignored: unknown caption group");
        return;
    };
    // Off clears the whole block rather than storing a disabled one, so saves
    // of plain captions stay free of highlight fields.
    let highlight = (mode != CaptionHighlightMode::Off).then(|| {
        let current = engine
            .project()
            .timeline()
            .caption_group(group_id)
            .and_then(|g| g.highlight.clone())
            .unwrap_or_default();
        CaptionHighlight {
            mode,
            fill,
            plate,
            scale,
            ..current
        }
    });
    apply_and_publish(
        engine,
        EditCommand::SetCaptionHighlight {
            group: group_id,
            highlight,
        },
        ui,
    );
}

/// Edit one cue's text. Unlike a plain title edit this keeps the cue's word
/// timings, remapped onto the new text.
fn set_cue_text(engine: &mut Engine, clip: &str, text: String, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "caption cue edit ignored: unparsable clip id");
        return;
    };
    apply_and_publish(
        engine,
        EditCommand::SetCaptionCue {
            clip: clip_id,
            text,
            words: None,
            speaker: None,
        },
        ui,
    );
}

fn split_cue(engine: &mut Engine, clip: &str, tick: i64, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "caption split ignored: unparsable clip id");
        return;
    };
    let rate = engine.project().timeline().frame_rate;
    apply_and_publish(
        engine,
        EditCommand::SplitCaptionCue {
            clip: clip_id,
            at: RationalTime::new(tick, rate),
        },
        ui,
    );
}

/// Merge a cue with the next cue of its group. The neighbor is resolved here
/// (the UI only knows the row it clicked) by cue order, which the model keeps
/// in timeline order.
fn merge_with_next(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "caption merge ignored: unparsable clip id");
        return;
    };
    let timeline = engine.project().timeline();
    let Some(cue) = timeline.clip(clip_id).and_then(|c| c.caption.as_ref()) else {
        error!(clip, "caption merge ignored: clip is not a caption cue");
        return;
    };
    let group = cue.group;
    let index = cue.index;
    let Some(next) = timeline
        .caption_cues(group)
        .into_iter()
        .filter(|c| c.caption.as_ref().is_some_and(|cue| cue.index > index))
        .min_by_key(|c| c.caption.as_ref().map_or(u32::MAX, |cue| cue.index))
        .map(|c| c.id)
    else {
        warn!(clip, "caption merge skipped: last cue in its group");
        return;
    };
    apply_and_publish(
        engine,
        EditCommand::MergeCaptionCues {
            clips: vec![clip_id, next],
        },
        ui,
    );
}

#[cfg(test)]
mod tests;
