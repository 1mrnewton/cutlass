//! A clip's mark in a project template: a slot, editable text, or the
//! music the user may swap.

use cutlass_model::{Draft, MaterialRef, SegmentId, TemplateMark};

use super::{EditError, ensure_unlocked, locate};

/// Mark `segment_id` with `mark`, or take its mark off with `None`.
///
/// A slot is a photo or video clip, and one that takes only a video (or
/// only a photo) must hold one now, so the template previews as it will
/// play. Editable text is a text clip. Music is a sound clip, and a draft
/// has one: marking another moves it there.
pub fn set_template_mark(
    draft: &mut Draft,
    segment_id: &SegmentId,
    mark: Option<TemplateMark>,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = &draft.tracks[track_index].segments[segment_index].material_id;
    let material = draft.materials.get(material_id.as_str());
    let fits = match (mark, material) {
        (None, _) => true,
        (Some(TemplateMark::Slot { takes, .. }), Some(MaterialRef::Video(video))) => {
            takes.accepts(video.is_still())
        }
        (Some(TemplateMark::Text), Some(MaterialRef::Text(_)))
        | (Some(TemplateMark::Music), Some(MaterialRef::Audio(_))) => true,
        _ => false,
    };
    if !fits {
        return Err(EditError::TemplateMark);
    }
    if mark == Some(TemplateMark::Music) {
        for segment in draft
            .tracks
            .iter_mut()
            .flat_map(|track| &mut track.segments)
        {
            if segment.template_mark == Some(TemplateMark::Music) {
                segment.template_mark = None;
            }
        }
    }
    draft.tracks[track_index].segments[segment_index].template_mark = mark;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialText, MaterialVideo, MetaType, Segment, SegmentId, SlotTakes,
        TemplateMark, TimeRange, Track, TrackAttribute, TrackFlag, TrackType,
    };

    use super::set_template_mark;
    use crate::edit::EditError;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        let mut video = MaterialVideo::new("v");
        video.kind = MetaType::Video.as_str().to_owned();
        let mut photo = MaterialVideo::new("p");
        photo.kind = MetaType::Photo.as_str().to_owned();
        draft.materials.videos.extend([video, photo]);
        draft.materials.texts.push(MaterialText::new("t"));
        draft.materials.audios.push(MaterialAudio {
            id: "a".into(),
            ..MaterialAudio::default()
        });
        draft.tracks[0].segments.extend([
            Segment::new("video", "v", TimeRange::new(0, 1_000_000)),
            Segment::new("photo", "p", TimeRange::new(1_000_000, 1_000_000)),
        ]);
        let mut text = Track {
            kind: TrackType::Text,
            flag: TrackFlag::None,
            ..Track::main("text")
        };
        text.segments
            .push(Segment::new("words", "t", TimeRange::new(0, 1_000_000)));
        let mut audio = Track {
            kind: TrackType::Audio,
            ..Track::main("sound")
        };
        audio.segments.extend([
            Segment::new("song", "a", TimeRange::new(0, 1_000_000)),
            Segment::new("song2", "a", TimeRange::new(1_000_000, 1_000_000)),
        ]);
        draft.tracks.extend([text, audio]);
        draft
    }

    const fn slot() -> TemplateMark {
        TemplateMark::slot(SlotTakes::Either)
    }

    fn mark_of(draft: &Draft, id: &str) -> Option<TemplateMark> {
        draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == id)
            .and_then(|segment| segment.template_mark)
    }

    #[test]
    fn each_mark_lands_on_its_kind_of_clip_and_comes_off() {
        let mut draft = draft();
        for (id, mark) in [
            ("video", slot()),
            ("photo", slot()),
            ("words", TemplateMark::Text),
            ("song", TemplateMark::Music),
        ] {
            set_template_mark(&mut draft, &SegmentId::new(id), Some(mark)).unwrap();
            assert_eq!(mark_of(&draft, id), Some(mark));
        }
        set_template_mark(&mut draft, &SegmentId::new("video"), None).unwrap();
        assert_eq!(mark_of(&draft, "video"), None);
    }

    #[test]
    fn a_mark_on_the_wrong_kind_of_clip_is_refused_and_leaves_the_draft() {
        let before = draft();
        let wrong = [
            ("words", slot()),
            ("song", slot()),
            ("video", TemplateMark::Text),
            ("words", TemplateMark::Music),
            ("photo", TemplateMark::slot(SlotTakes::Video)),
            ("video", TemplateMark::slot(SlotTakes::Photo)),
        ];
        for (id, mark) in wrong {
            let mut draft = before.clone();
            assert_eq!(
                set_template_mark(&mut draft, &SegmentId::new(id), Some(mark)),
                Err(EditError::TemplateMark),
                "{id} {mark:?}"
            );
            assert_eq!(draft, before);
        }
    }

    #[test]
    fn a_draft_has_one_music_and_marking_another_moves_it() {
        let mut draft = draft();
        set_template_mark(
            &mut draft,
            &SegmentId::new("song"),
            Some(TemplateMark::Music),
        )
        .unwrap();
        set_template_mark(
            &mut draft,
            &SegmentId::new("song2"),
            Some(TemplateMark::Music),
        )
        .unwrap();
        assert_eq!(mark_of(&draft, "song"), None);
        assert_eq!(mark_of(&draft, "song2"), Some(TemplateMark::Music));
    }

    #[test]
    fn a_missing_clip_and_a_locked_row_are_refused() {
        let mut draft = draft();
        assert_eq!(
            set_template_mark(&mut draft, &SegmentId::new("gone"), Some(slot())),
            Err(EditError::MissingSegment)
        );
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(
            set_template_mark(&mut draft, &SegmentId::new("video"), Some(slot())),
            Err(EditError::LockedTrack)
        );
    }

    #[test]
    fn a_mark_is_one_undo_step_and_the_request_reads_from_json() {
        let request: crate::request::Request = serde_json::from_value(serde_json::json!({
            "op": "set_template_mark",
            "segment_id": "video",
            "mark": {"role": "slot", "takes": "video"}
        }))
        .unwrap();
        let mut session = crate::session::Session::open(draft());
        session.apply(0, &[request]).unwrap();
        assert_eq!(
            mark_of(session.draft(), "video"),
            Some(TemplateMark::slot(SlotTakes::Video))
        );
        session.undo().unwrap();
        assert_eq!(mark_of(session.draft(), "video"), None);
    }

    #[test]
    fn a_template_clip_is_not_ungrouped() {
        use crate::request::{ImportedMedium, Request};
        let mut inside = Draft::empty("in");
        inside.tracks[0]
            .segments
            .push(Segment::new("x", "v", TimeRange::new(0, 1_000_000)));
        let mut session = crate::session::Session::open(Draft::empty("D"));
        let track_id = session.draft().tracks[0].id.clone();
        session
            .apply(
                0,
                &[
                    Request::Import {
                        medium: ImportedMedium::Template(Box::new(cutlass_model::MaterialDraft {
                            id: "tpl".into(),
                            draft: Some(Box::new(inside)),
                            ..cutlass_model::MaterialDraft::default()
                        })),
                    },
                    Request::Place {
                        track_id,
                        segment: Box::new(Segment::new(
                            "clip",
                            "tpl",
                            TimeRange::new(0, 1_000_000),
                        )),
                    },
                ],
            )
            .unwrap();
        let before = session.draft().clone();
        let refused = session.apply(
            1,
            &[Request::Ungroup {
                segment_id: SegmentId::new("clip"),
            }],
        );
        assert_eq!(refused.unwrap_err(), EditError::ClosedTemplate);
        assert_eq!(session.draft(), &before);
    }
}
