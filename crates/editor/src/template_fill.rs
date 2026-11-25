//! Fill a project template's slots with the person's own media.
//!
//! The fill is one list of requests, so it lands as one undo step: each
//! picked file is imported, played in its slot in place of the sample,
//! and sized so it covers what the sample covered. The slot keeps its
//! place, length, speed, effects, animations and transitions. Editable
//! texts take new words and the music may be swapped too. Every shell
//! fills a template through here.

use std::fmt;

use cutlass_model::{
    Draft, MaterialAudio, MaterialRef, MaterialVideo, MetaType, Micros, SegmentId, TemplateMark,
    template_slots,
};
use cutlass_session::{ImportedMedium, Request};

/// What a person picked to fill a template.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateFill {
    /// A photo or video for a slot, keyed by the slot's clip. The
    /// material is not on the draft yet: the fill imports it. Its id must
    /// be fresh.
    pub media: Vec<(SegmentId, MaterialVideo)>,
    /// New words for an editable text.
    pub texts: Vec<(SegmentId, String)>,
    /// A sound in place of the marked music. Its id must be fresh.
    pub music: Option<MaterialAudio>,
}

/// Why a pick does not fit its template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FillError {
    /// The clip is not a slot, an editable text, or the music of this draft.
    NotMarked { segment_id: SegmentId },
    /// The slot takes only a video and got a photo, or the other way round.
    Takes { segment_id: SegmentId },
    /// The video is shorter than the part of it the slot plays.
    TooShort {
        segment_id: SegmentId,
        needs: Micros,
        has: Micros,
    },
    /// The same slot or text was picked twice.
    Twice { segment_id: SegmentId },
}

impl fmt::Display for FillError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMarked { .. } => formatter.write_str("that clip is not part of the template"),
            Self::Takes { .. } => formatter.write_str("that slot takes another kind of media"),
            Self::TooShort { needs, has, .. } => write!(
                formatter,
                "that video is {:.1} s; the slot needs {:.1} s",
                seconds(*has),
                seconds(*needs)
            ),
            Self::Twice { .. } => formatter.write_str("that slot was picked twice"),
        }
    }
}

impl std::error::Error for FillError {}

#[allow(clippy::cast_precision_loss)]
fn seconds(time: Micros) -> f64 {
    time.0 as f64 / 1_000_000.0
}

/// Whether `material` can go in the slot `segment_id` of `draft`. The
/// fill screen greys out what this refuses.
pub fn fits_slot(
    draft: &Draft,
    segment_id: &SegmentId,
    material: &MaterialVideo,
) -> Result<(), FillError> {
    let slot = template_slots(draft)
        .into_iter()
        .find(|slot| &slot.segment_id == segment_id)
        .ok_or_else(|| FillError::NotMarked {
            segment_id: segment_id.clone(),
        })?;
    let photo = material.kind == MetaType::Photo.as_str();
    if !slot.takes.accepts(photo) {
        return Err(FillError::Takes {
            segment_id: segment_id.clone(),
        });
    }
    if !photo && material.duration < slot.needs {
        return Err(FillError::TooShort {
            segment_id: segment_id.clone(),
            needs: slot.needs,
            has: material.duration,
        });
    }
    Ok(())
}

/// The requests that fill `draft` with `fill`, as one undo step. Nothing
/// is built when any pick does not fit. A slot left out keeps its sample.
pub fn template_fill_requests(
    draft: &Draft,
    fill: &TemplateFill,
) -> Result<Vec<Request>, FillError> {
    let mut requests = Vec::new();
    let mut seen: Vec<SegmentId> = Vec::new();
    for (segment_id, material) in &fill.media {
        first_time(&mut seen, segment_id)?;
        fits_slot(draft, segment_id, material)?;
        requests.push(Request::Import {
            medium: ImportedMedium::Video(Box::new(material.clone())),
        });
        requests.push(Request::ReplaceMedia {
            segment_id: segment_id.clone(),
            material_id: material.id.clone(),
        });
        if let Some(scale) = cover_scale(draft, segment_id, material) {
            requests.push(scale);
        }
    }
    for (segment_id, words) in &fill.texts {
        first_time(&mut seen, segment_id)?;
        if mark_of(draft, segment_id) != Some(TemplateMark::Text) {
            return Err(FillError::NotMarked {
                segment_id: segment_id.clone(),
            });
        }
        requests.push(Request::SetTextContent {
            segment_id: segment_id.clone(),
            content: words.clone(),
        });
    }
    if let Some(music) = &fill.music {
        let segment_id =
            cutlass_model::template_music(draft).ok_or_else(|| FillError::NotMarked {
                segment_id: SegmentId::default(),
            })?;
        requests.push(Request::Import {
            medium: ImportedMedium::Audio(Box::new(music.clone())),
        });
        requests.push(Request::ReplaceMedia {
            segment_id,
            material_id: music.id.clone(),
        });
    }
    Ok(requests)
}

fn first_time(seen: &mut Vec<SegmentId>, segment_id: &SegmentId) -> Result<(), FillError> {
    if seen.contains(segment_id) {
        return Err(FillError::Twice {
            segment_id: segment_id.clone(),
        });
    }
    seen.push(segment_id.clone());
    Ok(())
}

fn mark_of(draft: &Draft, segment_id: &SegmentId) -> Option<TemplateMark> {
    draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
        .and_then(|segment| segment.template_mark)
}

/// The scale that makes `material` cover what the slot's sample covered.
///
/// A picture is first fitted inside the canvas, keeping its shape, and
/// scale 1 is that fitted picture. A new picture of another shape at the
/// same scale would leave bars where the sample had none, so its scale
/// grows until it covers the sample's box. `None` when the shapes match
/// or a size is unknown. Keyframed scale is left as it is.
fn cover_scale(draft: &Draft, segment_id: &SegmentId, material: &MaterialVideo) -> Option<Request> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)?;
    let Some(MaterialRef::Video(sample)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    let canvas = (
        f64::from(draft.canvas_config.width),
        f64::from(draft.canvas_config.height),
    );
    let old = (f64::from(sample.width), f64::from(sample.height));
    let new = (f64::from(material.width), f64::from(material.height));
    let factor = cover_factor(canvas, old, new)?;
    if (factor - 1.0).abs() < 1e-9 {
        return None;
    }
    Some(Request::SetScaleAxes {
        segment_id: segment_id.clone(),
        x: segment.clip.scale.x * factor,
        y: segment.clip.scale.y * factor,
    })
}

/// How much bigger a `new` picture must be drawn than an `old` one, both
/// fitted inside `canvas`, to cover the old one's box.
fn cover_factor(canvas: (f64, f64), old: (f64, f64), new: (f64, f64)) -> Option<f64> {
    let sizes = [canvas.0, canvas.1, old.0, old.1, new.0, new.1];
    if sizes.iter().any(|size| !size.is_finite() || *size <= 0.0) {
        return None;
    }
    let fit = |(width, height): (f64, f64)| (canvas.0 / width).min(canvas.1 / height);
    let (old_fit, new_fit) = (fit(old), fit(new));
    let factor = ((old.0 * old_fit) / (new.0 * new_fit)).max((old.1 * old_fit) / (new.1 * new_fit));
    factor.is_finite().then_some(factor)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialText, MaterialVideo, MetaType, Micros, Segment, SegmentId,
        SlotTakes, TemplateMark, TimeRange,
    };
    use cutlass_session::{Request, Session};

    use super::{FillError, TemplateFill, cover_factor, fits_slot, template_fill_requests};

    fn video(id: &str, kind: MetaType, size: (i32, i32), length: i64) -> MaterialVideo {
        let mut material = MaterialVideo::new(id);
        material.kind = kind.as_str().to_owned();
        material.width = size.0;
        material.height = size.1;
        material.duration = Micros(length);
        material.path = format!("Resources/{id}.mp4");
        material
    }

    /// A 1080 by 1920 template: a 9:16 video slot at 2x speed playing
    /// 4 s of its sample, a photo-only slot, an editable title and music.
    fn template() -> Draft {
        let mut draft = Draft::empty("T");
        draft.canvas_config.width = 1080;
        draft.canvas_config.height = 1920;
        draft.materials.videos.extend([
            video("sample", MetaType::Video, (1080, 1920), 10_000_000),
            video("still", MetaType::Photo, (1080, 1920), 0),
        ]);
        draft.materials.texts.push(MaterialText::new("words"));
        draft.materials.audios.push(MaterialAudio {
            id: "song".into(),
            duration: Micros(10_000_000),
            ..MaterialAudio::default()
        });
        let mut first = Segment::new("one", "sample", TimeRange::new(0, 2_000_000));
        first.speed = 2.0;
        first.source_range = Some(TimeRange::new(0, 4_000_000));
        first.template_mark = Some(TemplateMark::Slot {
            takes: SlotTakes::Either,
        });
        let mut second = Segment::new("two", "still", TimeRange::new(2_000_000, 2_000_000));
        second.source_range = Some(TimeRange::new(0, 2_000_000));
        second.template_mark = Some(TemplateMark::Slot {
            takes: SlotTakes::Photo,
        });
        draft.tracks[0].segments.extend([first, second]);
        let mut title = Segment::new("title", "words", TimeRange::new(0, 4_000_000));
        title.template_mark = Some(TemplateMark::Text);
        let mut music = Segment::new("music", "song", TimeRange::new(0, 4_000_000));
        music.source_range = Some(TimeRange::new(0, 4_000_000));
        music.template_mark = Some(TemplateMark::Music);
        let mut text_track = cutlass_model::Track::main("text");
        text_track.kind = cutlass_model::TrackType::Text;
        text_track.flag = cutlass_model::TrackFlag::None;
        text_track.segments.push(title);
        let mut sound = cutlass_model::Track::main("sound");
        sound.kind = cutlass_model::TrackType::Audio;
        sound.segments.push(music);
        draft.tracks.extend([text_track, sound]);
        draft
    }

    fn segment<'a>(draft: &'a Draft, id: &str) -> &'a Segment {
        draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == id)
            .unwrap()
    }

    #[test]
    fn a_fill_lands_as_one_undo_step_and_keeps_each_slot() {
        let draft = template();
        let fill = TemplateFill {
            media: vec![
                (
                    SegmentId::new("one"),
                    video("mine", MetaType::Video, (1080, 1920), 6_000_000),
                ),
                (
                    SegmentId::new("two"),
                    video("photo", MetaType::Photo, (1080, 1920), 0),
                ),
            ],
            texts: vec![(SegmentId::new("title"), "Our trip".to_owned())],
            music: Some(MaterialAudio {
                id: "my-song".into(),
                duration: Micros(30_000_000),
                ..MaterialAudio::default()
            }),
        };
        let requests = template_fill_requests(&draft, &fill).unwrap();
        let mut session = Session::open(draft.clone());
        session.apply(0, &requests).unwrap();
        let filled = session.draft();
        let one = segment(filled, "one");
        assert_eq!(one.material_id.as_str(), "mine");
        assert_eq!(one.target, TimeRange::new(0, 2_000_000));
        assert_eq!(one.speed, 2.0);
        assert_eq!(segment(filled, "two").material_id.as_str(), "photo");
        assert_eq!(segment(filled, "music").material_id.as_str(), "my-song");
        session.undo().unwrap();
        assert_eq!(session.draft(), &draft);
    }

    #[test]
    fn a_short_video_or_the_wrong_kind_is_refused_before_anything_is_built() {
        let draft = template();
        let short = video("short", MetaType::Video, (1080, 1920), 3_000_000);
        assert_eq!(
            fits_slot(&draft, &SegmentId::new("one"), &short),
            Err(FillError::TooShort {
                segment_id: SegmentId::new("one"),
                needs: Micros(4_000_000),
                has: Micros(3_000_000),
            })
        );
        let clip = video("clip", MetaType::Video, (1080, 1920), 9_000_000);
        assert_eq!(
            fits_slot(&draft, &SegmentId::new("two"), &clip),
            Err(FillError::Takes {
                segment_id: SegmentId::new("two")
            })
        );
        // A photo is always long enough.
        let photo = video("p", MetaType::Photo, (100, 100), 0);
        assert_eq!(fits_slot(&draft, &SegmentId::new("one"), &photo), Ok(()));
        let fill = TemplateFill {
            media: vec![
                (SegmentId::new("one"), photo),
                (SegmentId::new("two"), clip),
            ],
            ..TemplateFill::default()
        };
        assert!(matches!(
            template_fill_requests(&draft, &fill),
            Err(FillError::Takes { .. })
        ));
    }

    #[test]
    fn a_clip_that_is_not_marked_or_is_picked_twice_is_refused() {
        let draft = template();
        let photo = video("p", MetaType::Photo, (1080, 1920), 0);
        for (id, error) in [
            ("title", "not marked"),
            ("music", "not marked"),
            ("gone", "not marked"),
        ] {
            assert!(
                matches!(
                    fits_slot(&draft, &SegmentId::new(id), &photo),
                    Err(FillError::NotMarked { .. })
                ),
                "{id} {error}"
            );
        }
        let twice = TemplateFill {
            media: vec![
                (SegmentId::new("two"), photo.clone()),
                (SegmentId::new("two"), photo),
            ],
            ..TemplateFill::default()
        };
        assert!(matches!(
            template_fill_requests(&draft, &twice),
            Err(FillError::Twice { .. })
        ));
        let words = TemplateFill {
            texts: vec![(SegmentId::new("one"), "no".to_owned())],
            ..TemplateFill::default()
        };
        assert!(matches!(
            template_fill_requests(&draft, &words),
            Err(FillError::NotMarked { .. })
        ));
    }

    #[test]
    fn a_wide_photo_in_a_tall_slot_is_scaled_to_cover_it() {
        let draft = template();
        let fill = TemplateFill {
            media: vec![(
                SegmentId::new("two"),
                video("wide", MetaType::Photo, (1920, 1080), 0),
            )],
            ..TemplateFill::default()
        };
        let requests = template_fill_requests(&draft, &fill).unwrap();
        let Some(Request::SetScaleAxes { x, y, .. }) = requests.last() else {
            panic!("no scale in {requests:?}");
        };
        // Fitted inside 1080x1920 the wide photo is 1080x607.5; it must
        // grow to 1920/607.5 to fill the height the tall sample filled.
        let expected = 1920.0 / 607.5;
        assert!((x - expected).abs() < 1e-9 && (y - expected).abs() < 1e-9);
        let mut session = Session::open(draft);
        session.apply(0, &requests).unwrap();
        assert!((segment(session.draft(), "two").clip.scale.x - expected).abs() < 1e-9);
    }

    #[test]
    fn the_cover_factor_is_one_for_the_same_shape_and_none_for_no_size() {
        let canvas = (1080.0, 1920.0);
        assert_eq!(
            cover_factor(canvas, (540.0, 960.0), (1080.0, 1920.0)),
            Some(1.0)
        );
        assert_eq!(cover_factor(canvas, (0.0, 960.0), (1080.0, 1920.0)), None);
        assert_eq!(
            cover_factor(canvas, (1080.0, 1920.0), (f64::NAN, 1.0)),
            None
        );
        // A tall picture already covers a square sample's box.
        assert_eq!(
            cover_factor(canvas, (1000.0, 1000.0), (1080.0, 1920.0)),
            Some(1.0)
        );
        // A square picture in a tall sample's place grows to its height.
        let square = cover_factor(canvas, (1080.0, 1920.0), (1000.0, 1000.0)).unwrap();
        assert!((square - 1920.0 / 1080.0).abs() < 1e-9);
    }
}
