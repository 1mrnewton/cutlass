use cutlass_model::{Draft, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Set the segment gain. Mute and solo are track bits, not this number.
pub fn set_volume(draft: &mut Draft, segment_id: &SegmentId, volume: f64) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments[segment_index].volume = volume;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        CommonKeyframe, CommonKeyframeGroup, Crop, Draft, MaterialId, MaterialText, Micros,
        Segment, TimeRange, TrackId, WordColumns,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn open() -> (Session, TrackId) {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        (Session::open(draft), TrackId::new("main"))
    }

    fn place(track: &TrackId) -> Request {
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.volume = 1.0;
        Request::Place {
            track_id: track.clone(),
            segment: Box::new(segment),
        }
    }

    #[test]
    fn volume_fade_crop_text_and_keyframes_change_only_those_fields() {
        let (mut session, track) = open();
        session.apply(0, &[place(&track)]).unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[
                    Request::SetVolume {
                        segment_id: "a".into(),
                        volume: 0.25,
                    },
                    Request::SetVolume {
                        segment_id: "missing".into(),
                        volume: 0.0,
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::MissingSegment);
        assert_eq!(session.draft(), &before);

        session
            .apply(
                1,
                &[Request::SetVolume {
                    segment_id: "a".into(),
                    volume: 0.25,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.volume, 0.25);
        assert_eq!(segment.last_nonzero_volume, 1.0);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert!(session.draft().materials.audio_fades.is_empty());

        session
            .apply(
                2,
                &[Request::SetFade {
                    segment_id: "a".into(),
                    fade_id: MaterialId::new("fade"),
                    fade_in: Micros(100_000),
                    fade_out: Micros(200_000),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments[0].volume, 0.25);
        assert_eq!(
            session.draft().materials.audio_fades[0].fade_in_duration,
            Micros(100_000)
        );
        assert_eq!(
            session.draft().materials.audio_fades[0].fade_out_duration,
            Micros(200_000)
        );

        session
            .apply(
                3,
                &[Request::SetCrop {
                    segment_id: "a".into(),
                    crop: Crop {
                        upper_left_x: 0.1,
                        upper_right_x: 0.9,
                        lower_left_y: 1.0,
                        lower_right_y: 1.0,
                        ..Crop::default()
                    },
                }],
            )
            .unwrap();
        let video = &session.draft().materials.videos[0];
        assert_eq!(video.crop.upper_left_x, 0.1);
        assert_eq!(video.crop.upper_left_y, 0.0);
        assert_eq!(video.crop_ratio, "free");
        assert_eq!(video.crop_scale, 1.0);
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(0, 1_000_000)
        );

        session
            .apply(
                4,
                &[Request::SetKeyframes {
                    segment_id: "a".into(),
                    keyframes: vec![CommonKeyframeGroup {
                        property_type: "alpha".to_owned(),
                        keyframe_list: vec![CommonKeyframe {
                            time_offset: Micros(0),
                            values: vec![0.5],
                            curve_type: "Line".to_owned(),
                            ..CommonKeyframe::default()
                        }],
                        ..CommonKeyframeGroup::default()
                    }],
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments[0].volume, 0.25);
        assert_eq!(
            session.draft().tracks[0].segments[0].common_keyframes[0].property_type,
            "alpha"
        );
        assert_eq!(
            session.draft().tracks[0].segments[0].common_keyframes[0].keyframe_list[0].values,
            vec![0.5]
        );
    }

    #[test]
    fn text_content_does_not_rewrite_word_columns() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("media"),
            content: "old".to_owned(),
            words: WordColumns {
                text: vec!["hi".to_owned()],
                start_time: vec![0],
                end_time: vec![100],
                ..WordColumns::default()
            },
            ..MaterialText::default()
        });
        let mut session = Session::open(draft);
        session.apply(0, &[place(&TrackId::new("main"))]).unwrap();
        let words = session.draft().materials.texts[0].words.clone();
        session
            .apply(
                1,
                &[Request::SetTextContent {
                    segment_id: "a".into(),
                    content: "new".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.texts[0].content, "new");
        assert_eq!(session.draft().materials.texts[0].words, words);
        assert_eq!(session.draft().tracks[0].segments[0].volume, 1.0);
    }
}
