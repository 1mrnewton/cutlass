use cutlass_model::{Draft, MetaType, TrackFlag, TrackId, TrackType};

/// A row drawn for a track. The draft does not store one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lane {
    pub track_id: TrackId,
    pub kind: TrackType,
    pub flag: TrackFlag,
    /// Video or mixed, flag 0.
    pub main: bool,
    /// Index in the draft's track list. File order is the row order.
    pub track_index: usize,
    /// Index among the rows that are drawn. Zero is the top.
    pub row: usize,
}

/// Rows for the tracks that stay on screen.
///
/// An ordinary track with no segments is left out. The main lane stays when
/// it holds nothing. The draft's track list is not changed.
#[must_use]
pub fn lanes(draft: &Draft) -> Vec<Lane> {
    let mut rows = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.segments.is_empty() && !track.is_main() {
            continue;
        }
        let row = rows.len();
        rows.push(Lane {
            track_id: track.id.clone(),
            kind: track.kind.clone(),
            flag: track.flag,
            main: track.is_main(),
            track_index,
            row,
        });
    }
    rows
}

/// Where a material lands.
///
/// Video, photo, gif, image, and combination land on video. Text, subtitles,
/// lyrics, and text templates land on text. A sticker lands on sticker. An
/// adjustment lands on adjust. Audio, a filter, and a video effect return
/// [`None`]: their insert path already stored the track type.
#[must_use]
pub fn lane_category(meta: MetaType) -> Option<TrackType> {
    Some(match meta {
        MetaType::Video
        | MetaType::Photo
        | MetaType::Gif
        | MetaType::Image
        | MetaType::Combination => TrackType::Video,
        MetaType::Text
        | MetaType::Subtitle
        | MetaType::Lyrics
        | MetaType::TextTemplate
        | MetaType::TextPreset
        | MetaType::CaptionTemplate
        | MetaType::TextTemplateSubtitle
        | MetaType::TextTemplateLyrics => TrackType::Text,
        MetaType::Sticker => TrackType::Sticker,
        MetaType::Adjust => TrackType::Adjust,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MetaType, Segment, TimeRange, Track, TrackFlag, TrackType};

    use super::*;

    fn segment(id: &str) -> Segment {
        Segment::new(id, "media", TimeRange::new(0, 1_000_000))
    }

    #[test]
    fn an_empty_timeline_keeps_its_main_lane() {
        let draft = Draft::empty("D");
        let rows = lanes(&draft);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].main);
        assert_eq!(rows[0].kind, TrackType::Video);
        assert_eq!(rows[0].flag, TrackFlag::None);
        assert_eq!(rows[0].row, 0);
        assert!(draft.tracks[0].segments.is_empty());
        assert_eq!(draft.tracks.len(), 1);
    }

    #[test]
    fn ordinary_empty_lanes_are_removed_and_the_draft_is_left_alone() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = "main".into();
        let mut audio = Track::main("audio");
        audio.kind = TrackType::Audio;
        let mut text = Track::main("text");
        text.kind = TrackType::Text;
        text.segments.push(segment("caption"));
        let mut overlay = Track::main("overlay");
        overlay.flag = TrackFlag::SubVideo;
        draft.tracks.push(audio);
        draft.tracks.push(text);
        draft.tracks.push(overlay);
        let before = draft.clone();

        let rows = lanes(&draft);

        assert_eq!(draft, before);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].track_id.as_str(), "main");
        assert!(rows[0].main);
        assert!(rows[0].track_id.as_str() != "audio");
        assert_eq!(rows[1].track_id.as_str(), "text");
        assert_eq!(rows[1].kind, TrackType::Text);
        assert!(!rows[1].main);
        assert_eq!(rows[1].track_index, 2);
        assert_eq!(rows[1].row, 1);
    }

    #[test]
    fn a_filled_audio_filter_or_effect_keeps_its_stored_type() {
        let mut draft = Draft::empty("D");
        for (id, kind) in [
            ("audio", TrackType::Audio),
            ("filter", TrackType::Filter),
            ("effect", TrackType::Effect),
        ] {
            let mut track = Track::main(id);
            track.kind = kind;
            track.segments.push(segment(id));
            draft.tracks.push(track);
        }
        let rows = lanes(&draft);
        assert_eq!(
            rows.iter()
                .skip(1)
                .map(|lane| lane.kind.clone())
                .collect::<Vec<_>>(),
            vec![TrackType::Audio, TrackType::Filter, TrackType::Effect]
        );
        assert_eq!(lane_category(MetaType::Music), None);
        assert_eq!(lane_category(MetaType::Sound), None);
        assert_eq!(lane_category(MetaType::Record), None);
        assert_eq!(lane_category(MetaType::Filter), None);
        assert_eq!(lane_category(MetaType::VideoEffect), None);
        assert_eq!(lane_category(MetaType::Cover), None);
    }

    #[test]
    fn materials_land_on_the_named_categories() {
        for meta in [
            MetaType::Video,
            MetaType::Photo,
            MetaType::Gif,
            MetaType::Image,
            MetaType::Combination,
        ] {
            assert_eq!(lane_category(meta), Some(TrackType::Video));
        }
        for meta in [
            MetaType::Text,
            MetaType::Subtitle,
            MetaType::Lyrics,
            MetaType::TextTemplate,
            MetaType::TextPreset,
            MetaType::CaptionTemplate,
            MetaType::TextTemplateSubtitle,
            MetaType::TextTemplateLyrics,
        ] {
            assert_eq!(lane_category(meta), Some(TrackType::Text));
        }
        assert_eq!(lane_category(MetaType::Sticker), Some(TrackType::Sticker));
        assert_eq!(lane_category(MetaType::Adjust), Some(TrackType::Adjust));
    }

    #[test]
    fn an_empty_mixed_main_lane_stays_and_an_empty_overlay_does_not() {
        let mut draft = Draft::empty("D");
        draft.tracks.clear();
        let mut mixed = Track::main("mixed");
        mixed.kind = TrackType::Mixed;
        let mut overlay = Track::main("overlay");
        overlay.flag = TrackFlag::SubVideo;
        draft.tracks.push(mixed);
        draft.tracks.push(overlay);
        let rows = lanes(&draft);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].main);
        assert_eq!(rows[0].kind, TrackType::Mixed);
    }

    #[test]
    fn many_empty_ordinary_tracks_add_no_rows() {
        let mut draft = Draft::empty("D");
        for index in 0..64 {
            let mut track = Track::main(format!("audio-{index}"));
            track.kind = TrackType::Audio;
            draft.tracks.push(track);
        }
        assert_eq!(lanes(&draft).len(), 1);
    }
}
