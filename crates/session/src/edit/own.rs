//! A clip's own copy of a material before an edit changes it.
//!
//! Split hands both halves the same materials, and a draft may point two
//! clips at one material. An edit that changes a material in place first
//! takes the clip's own copy, so the other clips keep theirs.

use cutlass_model::{Draft, MaterialId};

use super::EditError;

/// How many segments point at `id`, as their material or an extra one.
pub(super) fn users(draft: &Draft, id: &MaterialId) -> usize {
    draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .map(|segment| {
            usize::from(&segment.material_id == id)
                + segment
                    .extra_material_refs
                    .iter()
                    .filter(|kept| *kept == id)
                    .count()
        })
        .sum()
}

/// The id the segment at `track_index`, `segment_index` may change for
/// `id`: `id` itself when no other segment points at it, else a copy made
/// for this segment and pointed at in `id`'s place. An empty `id` is
/// returned as it is.
pub(super) fn own(
    draft: &mut Draft,
    track_index: usize,
    segment_index: usize,
    id: &MaterialId,
) -> Result<MaterialId, EditError> {
    if id.is_empty() || users(draft, id) <= 1 {
        return Ok(id.clone());
    }
    let segment_id = draft.tracks[track_index].segments[segment_index].id.clone();
    let base = format!("{}:{}", segment_id.as_str(), id.as_str());
    let mut copy = MaterialId::new(base.clone());
    let mut n = 2;
    while draft.materials.get(copy.as_str()).is_some() {
        copy = MaterialId::new(format!("{base}:{n}"));
        n += 1;
    }
    // A material the pools do not hold gets no copy; the edit then makes
    // one under the new id, so it still lands on this segment alone.
    if draft.materials.get(id.as_str()).is_some() && !draft.materials.copy_as(id.as_str(), &copy) {
        return Err(EditError::DuplicateMaterial);
    }
    let segment = &mut draft.tracks[track_index].segments[segment_index];
    if &segment.material_id == id {
        segment.material_id = copy.clone();
    }
    for kept in &mut segment.extra_material_refs {
        if kept == id {
            *kept = copy.clone();
        }
    }
    Ok(copy)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudioFade, MaterialSpeed, MaterialText, MaterialVideo, Micros, Segment,
        TimeRange,
    };

    use crate::request::Request;
    use crate::session::Session;

    /// A main clip "a" on video "v" with fade "f" and speed "s", and a
    /// title "t" on text "x", each split at half a second.
    fn split_pair() -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = "main".into();
        draft.materials.videos.push(MaterialVideo {
            id: "v".into(),
            ..MaterialVideo::default()
        });
        draft.materials.audio_fades.push(MaterialAudioFade {
            id: "f".into(),
            fade_type: 2,
            ..MaterialAudioFade::default()
        });
        draft.materials.speeds.push(MaterialSpeed {
            id: "s".into(),
            speed: 1.0,
            ..MaterialSpeed::default()
        });
        draft.materials.texts.push(MaterialText {
            id: "x".into(),
            content: "hi".into(),
            ..MaterialText::default()
        });
        let mut session = Session::open(draft);
        let mut clip = Segment::new("a", "v", TimeRange::new(0, 1_000_000));
        clip.source_range = Some(TimeRange::new(0, 1_000_000));
        clip.extra_material_refs = vec!["f".into(), "s".into()];
        let revision = session
            .apply(
                0,
                &[Request::Place {
                    track_id: "main".into(),
                    segment: Box::new(clip),
                }],
            )
            .unwrap();
        let revision = session
            .apply(
                revision,
                &[Request::InsertTrack {
                    track_id: "words".into(),
                    kind: cutlass_model::TrackType::Text,
                }],
            )
            .unwrap();
        let revision = session
            .apply(
                revision,
                &[Request::Place {
                    track_id: "words".into(),
                    segment: Box::new(Segment::new("t", "x", TimeRange::new(0, 1_000_000))),
                }],
            )
            .unwrap();
        session
            .apply(
                revision,
                &[
                    Request::Split {
                        segment_id: "a".into(),
                        at: Micros(500_000),
                        right_id: "b".into(),
                    },
                    Request::Split {
                        segment_id: "t".into(),
                        at: Micros(500_000),
                        right_id: "u".into(),
                    },
                ],
            )
            .unwrap();
        session
    }

    fn segment<'a>(session: &'a Session, id: &str) -> &'a Segment {
        session
            .draft()
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == id)
            .unwrap()
    }

    #[test]
    fn an_edit_on_one_half_of_a_split_leaves_the_other_half_alone() {
        let mut session = split_pair();
        let revision = session.revision();
        session
            .apply(
                revision,
                &[
                    Request::SetFade {
                        segment_id: "a".into(),
                        fade_id: "unused".into(),
                        fade_in: Micros(200_000),
                        fade_out: Micros(0),
                    },
                    Request::SetCrop {
                        segment_id: "a".into(),
                        crop: cutlass_model::Crop::default(),
                    },
                    Request::SetConstantSpeed {
                        segment_id: "a".into(),
                        speed: 2.0,
                    },
                    Request::SetTextContent {
                        segment_id: "t".into(),
                        content: "left".into(),
                    },
                    Request::SetTextSize {
                        segment_id: "t".into(),
                        size: 40.0,
                        face: String::new(),
                    },
                ],
            )
            .unwrap();
        let draft = session.draft();
        let (left, right) = (segment(&session, "a"), segment(&session, "b"));
        assert_eq!(right.material_id.as_str(), "v");
        assert_ne!(left.material_id, right.material_id);
        let ids = |segment: &Segment| -> Vec<String> {
            segment
                .extra_material_refs
                .iter()
                .map(|id| id.as_str().to_owned())
                .collect()
        };
        assert_eq!(ids(right), ["f", "s"]);
        assert!(!ids(left).iter().any(|id| id == "f" || id == "s"));
        let fade = |id: &cutlass_model::MaterialId| {
            draft
                .materials
                .audio_fades
                .iter()
                .find(|fade| &fade.id == id)
                .unwrap()
                .fade_in_duration
        };
        assert_eq!(fade(&"f".into()), Micros(0));
        assert_eq!(fade(&left.extra_material_refs[0]), Micros(200_000));
        let speed = |id: &str| {
            draft
                .materials
                .speeds
                .iter()
                .find(|speed| speed.id.as_str() == id)
                .unwrap()
                .speed
        };
        assert_eq!(speed("s"), 1.0);
        assert_eq!(speed(left.extra_material_refs[1].as_str()), 2.0);

        let text = |id: &cutlass_model::MaterialId| {
            draft
                .materials
                .texts
                .iter()
                .find(|text| &text.id == id)
                .unwrap()
        };
        let (title, rest) = (segment(&session, "t"), segment(&session, "u"));
        assert_eq!(text(&rest.material_id).content, "hi");
        assert_eq!(text(&title.material_id).content, "left");
        assert_eq!(text(&title.material_id).font_size, 40.0);
        assert_eq!(text(&rest.material_id).font_size, 0.0);

        // The left half now owns its copies, so a second edit changes them
        // in place rather than copying again.
        let count = draft.materials.audio_fades.len();
        let revision = session.revision();
        session
            .apply(
                revision,
                &[Request::SetFade {
                    segment_id: "a".into(),
                    fade_id: "unused".into(),
                    fade_in: Micros(300_000),
                    fade_out: Micros(0),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.audio_fades.len(), count);
    }
}
