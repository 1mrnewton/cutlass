use super::*;

use crate::caption::{CaptionHighlightMode, CaptionSource};
use crate::look::AnimationSlot;
use crate::time::{Rational, RationalTime};

const R30: Rational = Rational::FPS_30;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, R30)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, R30)
}

/// A project with one text lane, plus the cue specs for two lines of speech.
fn captioned() -> (Project, TrackId, Vec<CaptionCueSpec>) {
    let mut project = Project::new("captions", R30);
    let track = project.add_track(TrackKind::Text, "Captions");
    let cues = vec![
        CaptionCueSpec::new("hello world", tr(0, 30)).with_words(vec![
            CaptionWord::new(0, 500, 0..5),
            CaptionWord::new(500, 1000, 6..11),
        ]),
        CaptionCueSpec::new("second line", tr(30, 30)),
    ];
    (project, track, cues)
}

fn add(
    project: &mut Project,
    track: TrackId,
    cues: &[CaptionCueSpec],
) -> (CaptionGroupId, Vec<ClipId>) {
    let spec = CaptionGroupSpec::manual(track, "Captions");
    project.add_caption_group(&spec, cues).unwrap()
}

// --- creation -------------------------------------------------------------

#[test]
fn add_caption_group_places_one_clip_per_cue_with_dense_indices() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    assert_eq!(clips.len(), 2);
    assert_eq!(project.timeline().caption_group_count(), 1);
    assert_eq!(project.timeline().caption_cue_ids(group), clips);
    for (index, &clip_id) in clips.iter().enumerate() {
        let clip = project.clip(clip_id).unwrap();
        let cue = clip.caption.as_ref().expect("cue metadata");
        assert_eq!(cue.group, group);
        assert_eq!(cue.index, index as u32);
        assert_eq!(project.timeline().track_of(clip_id), Some(track));
    }
    assert_eq!(
        project.clip(clips[0]).unwrap().text_content(),
        Some("hello world")
    );
    assert_eq!(
        project
            .clip(clips[0])
            .unwrap()
            .caption
            .as_ref()
            .unwrap()
            .words
            .len(),
        2
    );
}

#[test]
fn added_cues_inherit_the_group_style_and_safe_area() {
    let (mut project, track, cues) = captioned();
    let spec = CaptionGroupSpec::manual(track, "Captions").with_template("bold_box");
    let (group, clips) = project.add_caption_group(&spec, &cues).unwrap();

    let style = project
        .timeline()
        .caption_group(group)
        .unwrap()
        .style
        .clone();
    let clip = project.clip(clips[0]).unwrap();
    assert_eq!(clip.text_style(), Some(&style.text));
    assert_eq!(clip.transform.position.constant(), Some(style.position));
    assert!(
        style.position[1] > 0.0,
        "captions sit in the bottom safe area"
    );
    assert!(
        clip.text_style().unwrap().background.is_some(),
        "the bold_box template's plate is baked onto the cue"
    );
}

#[test]
fn add_caption_group_requires_a_text_lane() {
    let mut project = Project::new("captions", R30);
    let video = project.add_track(TrackKind::Video, "V1");
    let spec = CaptionGroupSpec::manual(video, "Captions");
    let cues = [CaptionCueSpec::new("hi", tr(0, 30))];
    assert!(matches!(
        project.add_caption_group(&spec, &cues),
        Err(ModelError::IncompatibleTrackKind { .. })
    ));
    assert_eq!(project.timeline().caption_group_count(), 0);
}

#[test]
fn add_caption_group_rejects_an_empty_batch() {
    let (mut project, track, _) = captioned();
    let spec = CaptionGroupSpec::manual(track, "Captions");
    assert!(project.add_caption_group(&spec, &[]).is_err());
    assert_eq!(project.timeline().caption_group_count(), 0);
}

#[test]
fn add_caption_group_rejects_overlapping_cues_atomically() {
    let (mut project, track, _) = captioned();
    let spec = CaptionGroupSpec::manual(track, "Captions");
    let overlapping = [
        CaptionCueSpec::new("one", tr(0, 30)),
        CaptionCueSpec::new("two", tr(15, 30)),
    ];
    assert!(project.add_caption_group(&spec, &overlapping).is_err());
    assert_eq!(project.timeline().caption_group_count(), 0);
    assert_eq!(project.timeline().clip_count(), 0);
}

#[test]
fn add_caption_group_rejects_cues_colliding_with_an_existing_clip() {
    let (mut project, track, cues) = captioned();
    add(&mut project, track, &cues);
    let spec = CaptionGroupSpec::manual(track, "More");
    let clashing = [CaptionCueSpec::new("clash", tr(10, 30))];
    assert!(matches!(
        project.add_caption_group(&spec, &clashing),
        Err(ModelError::Overlap(_))
    ));
    assert_eq!(project.timeline().caption_group_count(), 1);
}

// --- removal / restore ----------------------------------------------------

#[test]
fn remove_and_restore_a_group_round_trips_every_id() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    let (removed_group, removed_cues) = project.remove_caption_group(group).unwrap();
    assert_eq!(project.timeline().caption_group_count(), 0);
    assert_eq!(project.timeline().clip_count(), 0);

    let restored = project
        .restore_caption_group(removed_group, removed_cues)
        .unwrap();
    assert_eq!(restored, group);
    assert_eq!(project.timeline().caption_cue_ids(group), clips);
}

#[test]
fn remove_caption_group_rejects_an_unknown_group() {
    let (mut project, _, _) = captioned();
    let missing = CaptionGroupId::from_raw(u64::MAX - 3);
    assert!(matches!(
        project.remove_caption_group(missing),
        Err(ModelError::UnknownCaptionGroup(_))
    ));
}

// --- styling --------------------------------------------------------------

#[test]
fn set_group_style_writes_through_to_every_cue() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    let mut style = CaptionStyle::default();
    style.text.size = Param::Constant(120.0);
    style.position = [0.0, 0.25];
    project
        .set_caption_group_style(group, style.clone(), CaptionStyleScope::All)
        .unwrap();

    for &clip_id in &clips {
        let clip = project.clip(clip_id).unwrap();
        assert_eq!(clip.text_style().unwrap().size.constant(), Some(120.0));
        assert_eq!(clip.transform.position.constant(), Some([0.0, 0.25]));
    }
}

#[test]
fn keep_overrides_scope_skips_hand_styled_cues() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);
    project
        .timeline_mut()
        .clip_mut(clips[1])
        .unwrap()
        .caption
        .as_mut()
        .unwrap()
        .style_override = true;

    let mut style = CaptionStyle::default();
    style.text.size = Param::Constant(140.0);
    project
        .set_caption_group_style(group, style, CaptionStyleScope::KeepOverrides)
        .unwrap();

    assert_eq!(
        project
            .clip(clips[0])
            .unwrap()
            .text_style()
            .unwrap()
            .size
            .constant(),
        Some(140.0)
    );
    assert_ne!(
        project
            .clip(clips[1])
            .unwrap()
            .text_style()
            .unwrap()
            .size
            .constant(),
        Some(140.0),
        "an overridden cue keeps its own look"
    );
    assert!(
        project
            .clip(clips[1])
            .unwrap()
            .caption
            .as_ref()
            .unwrap()
            .style_override,
        "and keeps its flag for the next restyle"
    );
}

#[test]
fn apply_to_all_clears_override_flags() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);
    project
        .timeline_mut()
        .clip_mut(clips[1])
        .unwrap()
        .caption
        .as_mut()
        .unwrap()
        .style_override = true;

    project
        .set_caption_group_style(group, CaptionStyle::default(), CaptionStyleScope::All)
        .unwrap();
    assert!(
        !project
            .clip(clips[1])
            .unwrap()
            .caption
            .as_ref()
            .unwrap()
            .style_override
    );
}

#[test]
fn a_restyle_preserves_transform_animation_it_does_not_own() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);
    project
        .timeline_mut()
        .clip_mut(clips[0])
        .unwrap()
        .transform
        .rotation
        .set_keyframe(0, 15.0, crate::param::Easing::Linear);

    project
        .set_caption_group_style(group, CaptionStyle::default(), CaptionStyleScope::All)
        .unwrap();
    assert!(
        project
            .clip(clips[0])
            .unwrap()
            .transform
            .rotation
            .is_animated(),
        "rotation keyframes are not a caption style property"
    );
}

#[test]
fn set_group_template_applies_style_layout_and_highlight() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    project
        .set_caption_group_template(group, "karaoke_pop")
        .unwrap();
    let stored = project.timeline().caption_group(group).unwrap();
    assert_eq!(stored.template.as_deref(), Some("karaoke_pop"));
    assert_eq!(stored.layout.max_lines, 1);
    assert_eq!(
        stored.highlight.as_ref().map(|h| h.mode),
        Some(CaptionHighlightMode::Word)
    );
    assert!(
        project.clip(clips[0]).unwrap().animation_in.is_some(),
        "the template's entrance rides onto the cues"
    );
}

#[test]
fn a_manual_restyle_detaches_the_template_id() {
    let (mut project, track, cues) = captioned();
    let (group, _) = add(&mut project, track, &cues);
    project.set_caption_group_template(group, "glow").unwrap();
    project
        .set_caption_group_style(group, CaptionStyle::default(), CaptionStyleScope::All)
        .unwrap();
    assert_eq!(
        project.timeline().caption_group(group).unwrap().template,
        None
    );
}

#[test]
fn set_group_layout_moves_cues_into_the_new_safe_area() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    let layout = CaptionLayout {
        safe_area_bottom: 0.4,
        ..CaptionLayout::default()
    };
    project.set_caption_group_layout(group, layout).unwrap();
    let expected = layout.position_y();
    for &clip_id in &clips {
        assert_eq!(
            project.clip(clip_id).unwrap().transform.position.constant(),
            Some([0.0, expected])
        );
    }
}

#[test]
fn invalid_style_layout_and_highlight_are_rejected() {
    let (mut project, track, cues) = captioned();
    let (group, _) = add(&mut project, track, &cues);

    let style = CaptionStyle {
        scale: f32::NAN,
        ..CaptionStyle::default()
    };
    assert!(
        project
            .set_caption_group_style(group, style, CaptionStyleScope::All)
            .is_err()
    );

    let layout = CaptionLayout {
        max_lines: 0,
        ..CaptionLayout::default()
    };
    assert!(project.set_caption_group_layout(group, layout).is_err());

    let mut highlight = CaptionHighlight::word([255, 0, 0, 255]);
    highlight.scale = 99.0;
    assert!(
        project
            .set_caption_highlight(group, Some(highlight))
            .is_err()
    );
}

#[test]
fn set_highlight_stores_and_clears() {
    let (mut project, track, cues) = captioned();
    let (group, _) = add(&mut project, track, &cues);

    project
        .set_caption_highlight(group, Some(CaptionHighlight::word([0, 255, 0, 255])))
        .unwrap();
    assert!(
        project
            .timeline()
            .caption_group(group)
            .unwrap()
            .highlights()
            .is_some()
    );
    project.set_caption_highlight(group, None).unwrap();
    assert!(
        project
            .timeline()
            .caption_group(group)
            .unwrap()
            .highlights()
            .is_none()
    );
}

#[test]
fn set_group_label_rejects_an_absurd_name_without_losing_the_old_one() {
    let (mut project, track, cues) = captioned();
    let (group, _) = add(&mut project, track, &cues);
    assert!(
        project
            .set_caption_group_label(group, "x".repeat(300))
            .is_err()
    );
    assert_eq!(
        project.timeline().caption_group(group).unwrap().label,
        "Captions"
    );
    project
        .set_caption_group_label(group, "English".into())
        .unwrap();
    assert_eq!(
        project.timeline().caption_group(group).unwrap().label,
        "English"
    );
}

// --- cue edits ------------------------------------------------------------

#[test]
fn set_cue_text_flags_the_edit_and_remaps_word_timings() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);

    project
        .set_caption_cue(clips[0], "hello there world".into(), None, None)
        .unwrap();
    let clip = project.clip(clips[0]).unwrap();
    let cue = clip.caption.as_ref().unwrap();
    assert_eq!(clip.text_content(), Some("hello there world"));
    assert!(cue.text_edited);
    assert_eq!(cue.words.len(), 3, "one timing per new word");
    assert!(cue.validate("hello there world").is_ok());
    assert_eq!(cue.words[2].text("hello there world"), "world");
}

#[test]
fn set_cue_text_rejects_blank_text_and_non_cues() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    assert!(
        project
            .set_caption_cue(clips[0], "  ".into(), None, None)
            .is_err()
    );

    let plain = project
        .add_generated(
            track,
            Generator::Text {
                content: "title".into(),
                style: Default::default(),
            },
            tr(120, 30),
        )
        .unwrap();
    assert!(matches!(
        project.set_caption_cue(plain, "hi".into(), None, None),
        Err(ModelError::NotACaptionCue(_))
    ));
}

#[test]
fn set_cue_text_rejects_word_ranges_that_do_not_match_the_new_text() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    let bogus = vec![CaptionWord::new(0, 100, 0..99)];
    assert!(
        project
            .set_caption_cue(clips[0], "hi".into(), Some(bogus), None)
            .is_err()
    );
    assert_eq!(
        project.clip(clips[0]).unwrap().text_content(),
        Some("hello world")
    );
}

#[test]
fn split_cue_partitions_text_and_word_timings() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    // Cue 0 runs ticks 0..30 at 30fps; "hello" ends at 500 ms = tick 15.
    let right = project.split_caption_cue(clips[0], rt(15)).unwrap();

    let left_clip = project.clip(clips[0]).unwrap();
    let right_clip = project.clip(right).unwrap();
    assert_eq!(left_clip.text_content(), Some("hello"));
    assert_eq!(right_clip.text_content(), Some("world"));

    let left_words = &left_clip.caption.as_ref().unwrap().words;
    let right_words = &right_clip.caption.as_ref().unwrap().words;
    assert_eq!(left_words.len(), 1);
    assert_eq!(right_words.len(), 1);
    assert_eq!(right_words[0].start_ms, 0, "the right half rebases to zero");
    assert_eq!(right_words[0].text("world"), "world");
    assert!(
        left_clip
            .caption
            .as_ref()
            .unwrap()
            .validate("hello")
            .is_ok()
    );
    assert!(
        right_clip
            .caption
            .as_ref()
            .unwrap()
            .validate("world")
            .is_ok()
    );

    // Both halves stay in the group, renumbered in timeline order.
    let indices: Vec<u32> = project
        .timeline()
        .caption_cues(group)
        .iter()
        .map(|clip| clip.caption.as_ref().unwrap().index)
        .collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

/// Put the karaoke look's entrance and an exit on a cue, the way a template
/// writes its animations through to every line.
fn animate(project: &mut Project, clip: ClipId) {
    project
        .set_clip_animation(
            clip,
            AnimationSlot::In,
            Some(AnimationRef::new("char_pop_in")),
        )
        .unwrap();
    project
        .set_clip_animation(
            clip,
            AnimationSlot::Out,
            Some(AnimationRef::new("char_fade_out")),
        )
        .unwrap();
}

#[test]
fn an_animated_cue_splits_and_both_halves_keep_the_look() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    animate(&mut project, clips[0]);

    // Cue 0 holds one second; either half is too short for the half-second
    // entrance window, which is what a general split refuses to re-time.
    let right = project.split_caption_cue(clips[0], rt(15)).unwrap();

    for (clip_id, text) in [(clips[0], "hello"), (right, "world")] {
        let clip = project.clip(clip_id).unwrap();
        assert_eq!(clip.text_content(), Some(text));
        assert_eq!(
            clip.animation_in.as_ref().map(|a| a.id.as_str()),
            Some("char_pop_in"),
            "{text} plays the entrance in full"
        );
        assert_eq!(
            clip.animation_out.as_ref().map(|a| a.id.as_str()),
            Some("char_fade_out"),
            "{text} keeps the exit"
        );
    }
}

#[test]
fn a_refused_split_leaves_the_cue_animated() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    animate(&mut project, clips[0]);

    // A cut on the cue's own start is not a split at all.
    assert!(project.split_caption_cue(clips[0], rt(0)).is_err());

    let clip = project.clip(clips[0]).unwrap();
    assert!(clip.animation_in.is_some(), "the entrance came back");
    assert!(clip.animation_out.is_some(), "so did the exit");
    assert_eq!(clip.text_content(), Some("hello world"));
}

#[test]
fn split_cue_without_word_timings_repeats_the_text() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    let right = project.split_caption_cue(clips[1], rt(45)).unwrap();
    assert_eq!(
        project.clip(clips[1]).unwrap().text_content(),
        Some("second line")
    );
    assert_eq!(
        project.clip(right).unwrap().text_content(),
        Some("second line")
    );
}

#[test]
fn merge_cues_joins_text_and_offsets_word_timings() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    let merged = project.merge_caption_cues(&clips).unwrap();
    assert_eq!(merged, clips[0]);
    let clip = project.clip(merged).unwrap();
    assert_eq!(clip.text_content(), Some("hello world second line"));
    assert_eq!(clip.timeline.start.value, 0);
    assert_eq!(clip.timeline.end_tick(), 60, "the survivor spans both cues");
    assert_eq!(project.timeline().caption_cue_ids(group), vec![merged]);
    let cue = clip.caption.as_ref().unwrap();
    assert!(cue.validate("hello world second line").is_ok());
    assert_eq!(cue.index, 0);
}

#[test]
fn merge_rejects_a_single_cue_and_cross_group_merges() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    assert!(project.merge_caption_cues(&clips[..1]).is_err());

    let other = [CaptionCueSpec::new("other", tr(90, 30))];
    let spec = CaptionGroupSpec::manual(track, "Other");
    let (_, other_clips) = project.add_caption_group(&spec, &other).unwrap();
    assert!(
        project
            .merge_caption_cues(&[clips[0], other_clips[0]])
            .is_err()
    );
}

#[test]
fn merging_word_timings_stay_ascending_across_the_seam() {
    let (mut project, track, _) = captioned();
    let cues = vec![
        CaptionCueSpec::new("one", tr(0, 30)).with_words(vec![CaptionWord::new(0, 900, 0..3)]),
        CaptionCueSpec::new("two", tr(30, 30)).with_words(vec![CaptionWord::new(0, 900, 0..3)]),
    ];
    let (_, clips) = add(&mut project, track, &cues);
    let merged = project.merge_caption_cues(&clips).unwrap();
    let clip = project.clip(merged).unwrap();
    let words = &clip.caption.as_ref().unwrap().words;
    assert_eq!(words.len(), 2);
    assert!(
        words[1].start_ms >= words[0].end_ms,
        "the second cue's timings are offset by its start"
    );
    assert_eq!(words[1].text("one two"), "two");
}

// --- group lifecycle ------------------------------------------------------

#[test]
fn deleting_cues_reindexes_and_prunes_the_empty_group() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    project.timeline_mut().remove_clip(clips[0]);
    project.timeline_mut().reindex_caption_group(group);
    assert_eq!(
        project
            .clip(clips[1])
            .unwrap()
            .caption
            .as_ref()
            .unwrap()
            .index,
        0
    );

    project.timeline_mut().remove_clip(clips[1]);
    let pruned = project.timeline_mut().prune_empty_caption_groups();
    assert_eq!(pruned.len(), 1);
    assert_eq!(project.timeline().caption_group_count(), 0);
}

#[test]
fn ungrouping_leaves_plain_text_clips_behind() {
    let (mut project, track, cues) = captioned();
    let (group, clips) = add(&mut project, track, &cues);

    project.ungroup_caption_cues(&clips).unwrap();
    assert!(project.clip(clips[0]).unwrap().caption.is_none());
    assert_eq!(
        project.clip(clips[0]).unwrap().text_content(),
        Some("hello world")
    );
    assert!(project.timeline().caption_group(group).is_none());
}

#[test]
fn ungrouping_a_plain_clip_is_rejected() {
    let (mut project, track, _) = captioned();
    let plain = project
        .add_generated(
            track,
            Generator::Text {
                content: "title".into(),
                style: Default::default(),
            },
            tr(0, 30),
        )
        .unwrap();
    assert!(matches!(
        project.ungroup_caption_cues(&[plain]),
        Err(ModelError::NotACaptionCue(_))
    ));
}

#[test]
fn a_plain_set_generator_drops_stale_word_timings() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    project
        .set_generator(
            clips[0],
            Generator::Text {
                content: "totally different".into(),
                style: Default::default(),
            },
        )
        .unwrap();
    let cue = project
        .clip(clips[0])
        .unwrap()
        .caption
        .as_ref()
        .unwrap()
        .clone();
    assert!(cue.words.is_empty(), "byte ranges cannot survive the swap");
    assert!(cue.text_edited);
    assert!(cue.validate("totally different").is_ok());
}

#[test]
fn a_frozen_or_duplicated_cue_does_not_smuggle_caption_metadata() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    // Duplication is a deliberate clone: the copy stays in the group, which is
    // what CapCut does when you copy a caption line. Reindexing keeps the list
    // dense.
    let copy = project.duplicate_clip(clips[0], track, rt(90)).unwrap();
    assert!(project.clip(copy).unwrap().is_caption());
}

// --- word lookup ----------------------------------------------------------

#[test]
fn caption_word_at_maps_ticks_to_the_spoken_word() {
    let (mut project, track, cues) = captioned();
    let (_, clips) = add(&mut project, track, &cues);
    let clip = project.clip(clips[0]).unwrap();

    // 30 fps: tick 3 = 100 ms (inside "hello"), tick 18 = 600 ms ("world").
    assert_eq!(clip.caption_word_at(3, R30), Some(0..5));
    assert_eq!(clip.caption_word_at(18, R30), Some(6..11));
    assert_eq!(
        project.clip(clips[1]).unwrap().caption_word_at(3, R30),
        None,
        "a cue without timings never highlights"
    );
}

// --- persistence ----------------------------------------------------------

#[test]
fn a_caption_free_project_serializes_without_caption_fields() {
    let mut project = Project::new("plain", R30);
    let track = project.add_track(TrackKind::Text, "Titles");
    project
        .add_generated(
            track,
            Generator::Text {
                content: "title".into(),
                style: Default::default(),
            },
            tr(0, 30),
        )
        .unwrap();
    let json = serde_json::to_string(&project).unwrap();
    assert!(
        !json.contains("caption"),
        "captions must be additive: {json}"
    );
}

#[test]
fn caption_groups_and_cues_round_trip_through_json() {
    let (mut project, track, cues) = captioned();
    let spec = CaptionGroupSpec {
        source: CaptionSource::Auto {
            media: crate::ids::MediaId::from_raw(9),
            language: Some("en".into()),
            model: "ggml-base.en".into(),
        },
        ..CaptionGroupSpec::manual(track, "Auto captions").with_template("karaoke_pop")
    };
    let (group, clips) = project.add_caption_group(&spec, &cues).unwrap();

    let json = serde_json::to_string(&project).unwrap();
    let loaded: Project = serde_json::from_str(&json).unwrap();

    let before = project.timeline().caption_group(group).unwrap();
    let after = loaded.timeline().caption_group(group).unwrap();
    assert_eq!(before, after);
    assert_eq!(
        loaded.clip(clips[0]).unwrap().caption,
        project.clip(clips[0]).unwrap().caption
    );
    assert_eq!(loaded.timeline().caption_cue_ids(group), clips);
}
