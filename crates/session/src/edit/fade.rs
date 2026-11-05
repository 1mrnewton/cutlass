use cutlass_model::{Draft, MaterialAudioFade, MaterialId, MaterialRef, Micros, SegmentId};

use super::locate;
use super::{EditError, ensure_unlocked};

/// A straight ramp. Curve 0 skips the durations, so a fade that would
/// otherwise stay silent uses this curve. Curves 1 through 4 stay as stored.
const LINEAR: i32 = 2;

/// Store fade durations. The segment volume stays as it was.
pub fn set_fade(
    draft: &mut Draft,
    segment_id: &SegmentId,
    fade_id: &MaterialId,
    fade_in: Micros,
    fade_out: Micros,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    if fade_in.0 < 0 || fade_out.0 < 0 {
        return Err(EditError::Fade);
    }
    let existing = draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|material_id| {
            matches!(
                draft.materials.get(material_id.as_str()),
                Some(MaterialRef::Fade(_))
            )
        })
        .cloned();
    if let Some(existing_id) = existing {
        let existing_id = super::own::own(draft, track_index, segment_index, &existing_id)?;
        let Some(fade) = draft
            .materials
            .audio_fades
            .iter_mut()
            .find(|fade| fade.id == existing_id)
        else {
            return Err(EditError::MissingMaterial);
        };
        fade.fade_in_duration = fade_in;
        fade.fade_out_duration = fade_out;
        if !(1..=4).contains(&fade.fade_type) {
            fade.fade_type = LINEAR;
        }
        return Ok(());
    }
    if fade_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(fade_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.audio_fades.push(MaterialAudioFade {
        id: fade_id.clone(),
        fade_type: LINEAR,
        fade_in_duration: fade_in,
        fade_out_duration: fade_out,
        ..MaterialAudioFade::default()
    });
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(fade_id.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudioFade, MaterialId, Micros, Segment, TimeRange, TrackAttribute, TrackId,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    #[test]
    fn a_fade_is_a_straight_ramp_and_a_negative_duration_is_refused() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let track = draft.tracks[0].id.clone();
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track,
                    segment: Box::new(Segment::new("a", "media", TimeRange::new(0, 1_000_000))),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetFade {
                    segment_id: "a".into(),
                    fade_id: MaterialId::new("fade"),
                    fade_in: Micros(100_000),
                    fade_out: Micros(0),
                }],
            )
            .unwrap();
        let fade = &session.draft().materials.audio_fades[0];
        assert_eq!(fade.fade_type, 2);
        assert_eq!(fade.fade_in_duration, Micros(100_000));
        assert_eq!(fade.fade_out_duration, Micros(0));
        assert_eq!(session.draft().tracks[0].segments[0].volume, 1.0);

        session
            .apply(
                2,
                &[Request::SetFade {
                    segment_id: "a".into(),
                    fade_id: MaterialId::new("other"),
                    fade_in: Micros(100_000),
                    fade_out: Micros(50_000),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.audio_fades.len(), 1);
        assert_eq!(
            session.draft().materials.audio_fades[0].fade_out_duration,
            Micros(50_000)
        );
        assert_eq!(session.draft().materials.audio_fades[0].id.as_str(), "fade");

        let error = session
            .apply(
                3,
                &[Request::SetFade {
                    segment_id: "a".into(),
                    fade_id: MaterialId::new("other"),
                    fade_in: Micros(-1),
                    fade_out: Micros(0),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Fade);
        assert_eq!(
            session.draft().materials.audio_fades[0].fade_in_duration,
            Micros(100_000)
        );
        assert_eq!(
            session.draft().materials.audio_fades[0].fade_out_duration,
            Micros(50_000)
        );
    }

    #[test]
    fn a_stored_curve_stays_and_a_locked_lane_is_refused() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.materials.audio_fades.push(MaterialAudioFade {
            id: MaterialId::new("fade"),
            fade_type: 1,
            fade_in_duration: Micros(1_000),
            fade_out_duration: Micros(2_000),
            ..MaterialAudioFade::default()
        });
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.extra_material_refs.push(MaterialId::new("fade"));
        draft.tracks[0].segments.push(segment);
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[Request::SetFade {
                    segment_id: "a".into(),
                    fade_id: MaterialId::new("unused"),
                    fade_in: Micros(10_000),
                    fade_out: Micros(20_000),
                }],
            )
            .unwrap();
        let fade = &session.draft().materials.audio_fades[0];
        assert_eq!(fade.fade_type, 1);
        assert_eq!(fade.fade_in_duration, Micros(10_000));
        assert_eq!(session.draft().materials.audio_fades.len(), 1);

        session
            .apply(
                1,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("main"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(
                2,
                &[Request::SetFade {
                    segment_id: "a".into(),
                    fade_id: MaterialId::new("unused"),
                    fade_in: Micros(0),
                    fade_out: Micros(0),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(
            session.draft().materials.audio_fades[0].fade_in_duration,
            Micros(10_000)
        );
    }
}
