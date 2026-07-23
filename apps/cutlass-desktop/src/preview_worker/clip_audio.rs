use super::*;

/// Set a clip's audio mix (CapCut volume + fades, M1). A video clip carries
/// its own sound, so the edit lands on the clicked clip; only when its audio
/// was detached to a linked audio lane does it follow to the audible half
/// there. One history group; the republish re-snapshots the playback mixer, so
/// the change is audible within a block.
pub(super) fn set_clip_audio_and_publish(
    engine: &mut Engine,
    clip: &str,
    volume: Option<f32>,
    fade_in_s: f32,
    fade_out_s: f32,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-audio ignored: unparsable clip id");
        return;
    };
    // CapCut keeps a video's sound on its own clip, so volume/fades land on the
    // clicked clip when it carries its own audio (a video drop, or an audio
    // lane). Only when its sound was detached to a linked audio lane does the
    // edit follow to the audible half there.
    let targets: Vec<ClipId> = if engine.project().timeline().carries_own_audio(clip_id) {
        vec![clip_id]
    } else {
        link_group_ids(engine, clip_id)
            .into_iter()
            .filter(|id| engine.project().timeline().carries_own_audio(*id))
            .collect()
    };
    if targets.is_empty() {
        warn!(%clip_id, "set-clip-audio ignored: no audible clip to adjust");
        return;
    }

    let tl_rate = engine.project().timeline().frame_rate;
    let to_ticks = |seconds: f32| {
        let ticks = (f64::from(seconds) * f64::from(tl_rate.num) / f64::from(tl_rate.den)).round();
        RationalTime::new(ticks.max(0.0) as i64, tl_rate)
    };
    let (fade_in, fade_out) = (to_ticks(fade_in_s), to_ticks(fade_out_s));

    // Flat volume commits clear a live Volume override so the next mix block
    // never flashes the stale drag gain (mirrors set_param_constant).
    if volume.is_some() {
        for target in &targets {
            engine.clear_param_override(*target, ClipParam::Volume);
            ui.audio.clear_param_override(*target, ClipParam::Volume);
        }
    }

    engine.begin_group();
    for target in &targets {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipAudio {
            clip: *target,
            volume,
            fade_in,
            fade_out,
        })) {
            error!(clip_id = %target, "set clip audio failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, ?volume, fade_in_s, fade_out_s, clips = targets.len(), "set clip audio");
    publish_projection(engine, ui);
}

/// Duck a music clip under the voice lanes (M8 Phase 4). Gathers every clip on
/// a voice-tagged (`duck_source`) audio lane that overlaps the selected music
/// clip and lowers `DuckLanes` onto it — the engine writes the dip as ordinary
/// M8 volume keyframes, so the result is one undoable edit, audible on the next
/// mixer snapshot and editable through the volume envelope afterwards. The
/// defaults mirror the decoder's broadcast-typical ducker (and the agent
/// `duck` tool); the linear speech-band threshold stays an internal detail.
pub(super) fn duck_under_voice_and_publish(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(music_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "duck-under-voice ignored: unparsable clip id");
        return;
    };

    // Resolve the overlapping voice clips against an immutable view, never
    // ducking a clip under its own lane.
    let voice: Vec<ClipId> = {
        let project = engine.project();
        let timeline = project.timeline();
        let Some(music) = project.clip(music_id) else {
            warn!(%music_id, "duck-under-voice ignored: unknown clip");
            return;
        };
        let music_track = timeline.track_of(music_id);
        let music_range = music.timeline;
        timeline
            .tracks_ordered()
            .filter(|track| {
                track.kind == TrackKind::Audio && track.duck_source && Some(track.id) != music_track
            })
            .flat_map(|track| track.clips_ordered())
            .filter(|c| c.timeline.overlaps(music_range).unwrap_or(false))
            .map(|c| c.id)
            .collect()
    };
    if voice.is_empty() {
        warn!(%music_id, "duck-under-voice: no voice-lane clips overlap the selected music");
        return;
    }

    match engine.apply(Command::Edit(EditCommand::DuckLanes {
        voice,
        music: vec![music_id],
        // Mirror `DuckSettings::default()` / the agent `duck` tool defaults.
        threshold: 0.025,
        amount: 0.66,
        attack: 0.08,
        release: 0.32,
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            info!(%music_id, "ducked music under voice");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(%music_id, "unexpected duck-under-voice outcome: {other:?}"),
        Err(e) => error!(%music_id, "duck under voice failed: {e}"),
    }
}

/// Detect beat markers on a media clip (CapCut "Beat", M8 Phase 6): the engine
/// decodes the clip's audio, runs onset/tempo analysis, and stores the beat
/// grid (source ticks) so the timeline magnet can snap clip edges to it. One
/// undoable history entry. A rejection (generated clip / no audio) just logs —
/// the inspector only offers the button on media clips with sound, so it would
/// be a stale-projection race.
pub(super) fn detect_beats_and_publish(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "detect-beats ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::DetectBeats { clip: clip_id })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            info!(%clip_id, "detected beats");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(%clip_id, "unexpected detect-beats outcome: {other:?}"),
        Err(e) => error!(%clip_id, "detect beats failed: {e}"),
    }
}

/// Clear a clip's detected beat markers (M8 Phase 6). One undoable entry.
pub(super) fn clear_beats_and_publish(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "clear-beats ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::ClearBeats { clip: clip_id })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            info!(%clip_id, "cleared beats");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(%clip_id, "unexpected clear-beats outcome: {other:?}"),
        Err(e) => error!(%clip_id, "clear beats failed: {e}"),
    }
}
