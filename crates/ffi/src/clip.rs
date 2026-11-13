//! One-tap clip edits the phone's clip bars offer: mirror, turn, reverse,
//! crop, fade, freeze, extract audio and beats. Each follows the desktop's
//! rule for the same button, and each is one undo step.

use std::collections::HashSet;
use std::path::Path;

use cutlass_model::{Crop, Draft, MaterialId, MaterialRef, Segment, SegmentId, TrackType};
use cutlass_session::{Micros, Request};

use crate::{FfiError, PhoneHost, media};

/// A clip's fade in and fade out, in microseconds. Zero is no fade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhoneFade {
    pub fade_in: Micros,
    pub fade_out: Micros,
}

/// What the clip bar can do to one clip, so a shell can dim the rest.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhoneClipAbilities {
    /// A picture: mirror and turn.
    pub turn: bool,
    /// A picture or a sound: reverse and fade.
    pub reverse: bool,
    /// A video or a composition: crop.
    pub crop: bool,
    /// A video that is not a still: freeze and extract audio.
    pub moving: bool,
    /// On an audio row: beats.
    pub beats: bool,
    pub mirrored: bool,
    pub reversed: bool,
    pub cropped: bool,
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

fn placed<'a>(draft: &'a Draft, id: &SegmentId) -> Result<(&'a Segment, &'a TrackType), FfiError> {
    draft
        .tracks
        .iter()
        .find_map(|track| {
            track
                .segments
                .iter()
                .find(|segment| &segment.id == id)
                .map(|segment| (segment, &track.kind))
        })
        .ok_or_else(|| refused("the clip is not on the timeline"))
}

fn stored_crop(draft: &Draft, segment: &Segment) -> Option<Crop> {
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => Some(video.crop),
        Some(MaterialRef::Draft(compound)) => Some(compound.crop),
        _ => None,
    }
}

/// The middle half of the picture, as the desktop's Crop button keeps.
/// Coordinates are fractions, Y downward.
fn center_crop() -> Crop {
    Crop {
        upper_left_x: 0.25,
        upper_left_y: 0.25,
        upper_right_x: 0.75,
        upper_right_y: 0.25,
        lower_left_x: 0.25,
        lower_left_y: 0.75,
        lower_right_x: 0.75,
        lower_right_y: 0.75,
    }
}

/// One clockwise quarter turn. The stored angle turns the other way.
fn clockwise_quarter(degrees: f64) -> f64 {
    let current = if degrees.is_finite() { degrees } else { 0.0 };
    (current - 90.0).rem_euclid(360.0)
}

fn fade_of(draft: &Draft, segment: &Segment) -> PhoneFade {
    for id in &segment.extra_material_refs {
        if let Some(MaterialRef::Fade(fade)) = draft.materials.get(id.as_str()) {
            return PhoneFade {
                fade_in: fade.fade_in_duration,
                fade_out: fade.fade_out_duration,
            };
        }
    }
    PhoneFade {
        fade_in: Micros(0),
        fade_out: Micros(0),
    }
}

fn reversible(draft: &Draft, segment: &Segment) -> bool {
    matches!(
        draft.materials.get(segment.material_id.as_str()),
        Some(MaterialRef::Video(_) | MaterialRef::Audio(_))
    )
}

fn moving(draft: &Draft, segment: &Segment) -> bool {
    matches!(
        draft.materials.get(segment.material_id.as_str()),
        Some(MaterialRef::Video(video)) if video.kind != "photo" && video.freeze.is_none()
    )
}

/// Every id the draft uses, clips' extra materials included, so a new fade
/// or beat id is unused.
fn taken(draft: &Draft) -> HashSet<String> {
    let mut taken = media::ids(draft);
    for track in &draft.tracks {
        for segment in &track.segments {
            for id in &segment.extra_material_refs {
                taken.insert(id.as_str().to_owned());
            }
        }
    }
    taken
}

fn fresher(draft: &Draft) -> impl FnMut(&str) -> String {
    let mut taken = taken(draft);
    move |prefix: &str| {
        let id = media::fresh(&taken, prefix);
        taken.insert(id.clone());
        id
    }
}

/// `seconds` as a fade length. Below zero, or not a number, is refused.
fn fade_length(micros: i64) -> Result<Micros, FfiError> {
    if micros < 0 {
        return Err(refused("a fade cannot be negative"));
    }
    Ok(Micros(micros))
}

/// Where a beat dropped at `at` on the timeline sits in the clip's own
/// file, so a later move carries it.
fn beat_source(segment: &Segment, at: Micros) -> Result<i64, FfiError> {
    cutlass_model::source_time_at(segment, at)
        .ok_or_else(|| refused("move the playhead onto the clip"))
}

/// A beat this close to the playhead, on the clip's own time, is the one a
/// second tap takes away.
const BEAT_REACH: i64 = 100_000;

impl PhoneHost {
    /// What the clip bar can do to `segment_id`.
    pub fn clip_abilities(&self, segment_id: &SegmentId) -> Result<PhoneClipAbilities, FfiError> {
        let draft = self.draft();
        let (segment, kind) = placed(draft, segment_id)?;
        let turn = cutlass_editor::can_turn(draft, segment, kind);
        let crop = stored_crop(draft, segment);
        Ok(PhoneClipAbilities {
            turn,
            reverse: reversible(draft, segment),
            crop: crop.is_some(),
            moving: moving(draft, segment),
            beats: cutlass_model::track_holds_beats(kind),
            mirrored: turn && segment.clip.flip.horizontal,
            reversed: segment.reverse,
            cropped: crop.is_some_and(|crop| crop != Crop::default()),
        })
    }

    /// Flip a picture left to right. A second tap restores it.
    pub fn toggle_mirror(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let draft = self.draft();
        let (segment, kind) = placed(draft, segment_id)?;
        if !cutlass_editor::can_turn(draft, segment, kind) {
            return Err(refused("a mirror belongs on a picture"));
        }
        let horizontal = !segment.clip.flip.horizontal;
        self.apply(
            self.revision(),
            &[Request::SetMirror {
                segment_id: segment_id.clone(),
                horizontal,
            }],
        )?;
        Ok(())
    }

    /// Turn a picture clockwise by a quarter. Four taps bring it upright.
    pub fn turn(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let draft = self.draft();
        let (segment, kind) = placed(draft, segment_id)?;
        if !cutlass_editor::can_turn(draft, segment, kind) {
            return Err(refused("a rotate belongs on a picture"));
        }
        let degrees = clockwise_quarter(segment.clip.rotation);
        self.apply(
            self.revision(),
            &[Request::SetRotation {
                segment_id: segment_id.clone(),
                degrees,
            }],
        )?;
        Ok(())
    }

    /// Play a picture or a sound backward through its own file. A second
    /// tap plays it forward.
    pub fn toggle_reverse(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let draft = self.draft();
        let (segment, _) = placed(draft, segment_id)?;
        if !reversible(draft, segment) {
            return Err(refused("reverse belongs on a picture or a sound"));
        }
        let reverse = !segment.reverse;
        self.apply(
            self.revision(),
            &[Request::SetReverse {
                segment_id: segment_id.clone(),
                reverse,
                path: String::new(),
            }],
        )?;
        Ok(())
    }

    /// Keep the middle half of a picture. A second tap restores the whole
    /// frame. The crop is stored on the video, so its other clips keep it too.
    pub fn toggle_crop(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let draft = self.draft();
        let (segment, _) = placed(draft, segment_id)?;
        let current =
            stored_crop(draft, segment).ok_or_else(|| refused("a crop belongs on a picture"))?;
        let crop = if current == Crop::default() {
            center_crop()
        } else {
            Crop::default()
        };
        self.apply(
            self.revision(),
            &[Request::SetCrop {
                segment_id: segment_id.clone(),
                crop,
            }],
        )?;
        Ok(())
    }

    /// A picture's or a sound's fade in and fade out.
    pub fn fade(&self, segment_id: &SegmentId) -> Result<PhoneFade, FfiError> {
        let draft = self.draft();
        let (segment, _) = placed(draft, segment_id)?;
        Ok(fade_of(draft, segment))
    }

    /// Fade a picture or a sound in over `fade_in` and out over `fade_out`.
    /// The same fade again is not a step.
    pub fn set_fade(
        &mut self,
        segment_id: &SegmentId,
        fade_in: Micros,
        fade_out: Micros,
    ) -> Result<(), FfiError> {
        let fade_in = fade_length(fade_in.0)?;
        let fade_out = fade_length(fade_out.0)?;
        let draft = self.draft();
        let (segment, _) = placed(draft, segment_id)?;
        if !reversible(draft, segment) {
            return Err(refused("a fade belongs on a picture or a sound"));
        }
        if fade_of(draft, segment) == (PhoneFade { fade_in, fade_out }) {
            return Ok(());
        }
        let fade_id = MaterialId::new(fresher(draft)("fade"));
        self.apply(
            self.revision(),
            &[Request::SetFade {
                segment_id: segment_id.clone(),
                fade_id,
                fade_in,
                fade_out,
            }],
        )?;
        Ok(())
    }

    /// Place a video's sound on its own clip on an audio row, lined up with
    /// it, and silence the video. Returns the new sound clip.
    pub fn extract_audio(&mut self, segment_id: &SegmentId) -> Result<SegmentId, FfiError> {
        let draft = self.draft();
        let (segment, _) = placed(draft, segment_id)?;
        let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str())
        else {
            return Err(refused("an extract belongs on video"));
        };
        if video.kind == "photo" {
            return Err(refused("the clip has no sound"));
        }
        let stored = if segment.reverse && !video.reverse_path.is_empty() {
            &video.reverse_path
        } else {
            &video.path
        };
        let file = self.resolve(Path::new(stored))?;
        let heard = self
            .preview
            .readers()
            .carries_sound(&file)
            .map_err(|_| refused("the clip's sound could not be read"))?;
        if !heard {
            return Err(refused("the clip has no sound"));
        }
        let request = cutlass_editor::extract_request(draft, segment_id, &mut fresher(draft))
            .ok_or_else(|| refused("the clip is not on the timeline"))?;
        let Request::ExtractAudio {
            segment_id: sound, ..
        } = &request
        else {
            return Err(refused("the clip is not on the timeline"));
        };
        let sound = sound.clone();
        self.apply(self.revision(), &[request])?;
        Ok(sound)
    }

    /// Hold the video's frame under `at` as a still, cut in at the playhead
    /// (or at the clip's edge when the playhead is close to it). Returns
    /// the still's clip.
    pub fn freeze(&mut self, segment_id: &SegmentId, at: Micros) -> Result<SegmentId, FfiError> {
        let point = cutlass_eval::freeze_point(self.draft(), segment_id, at).map_err(|miss| {
            refused(match miss {
                cutlass_eval::FreezeMiss::UnknownSegment => "the clip is not on the timeline",
                cutlass_eval::FreezeMiss::NotVideo => "a freeze belongs on video",
                cutlass_eval::FreezeMiss::NoFrame => "the clip has no frame to hold",
                cutlass_eval::FreezeMiss::Outside => "move the playhead onto the clip",
                cutlass_eval::FreezeMiss::AlreadyHeld => "the clip is already a still",
            })
        })?;
        let unreadable = || refused("the clip's frame could not be read");
        let source = self.resolve(Path::new(&point.freeze.source_material_path))?;
        let image = cutlass_eval::freeze_image(
            self.preview.readers(),
            &crate::encoders(),
            &source,
            point.freeze.timestamp,
        )
        .ok_or_else(unreadable)?;
        let name = format!(
            "freeze_{}_{}.png",
            segment_id
                .as_str()
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                })
                .collect::<String>(),
            point.freeze.timestamp.0
        );
        let (stored, written) = media::write_in(self.root(), &name, &image.png)?;
        let size = |side: u32| i32::try_from(side).unwrap_or(i32::MAX);
        let still = cutlass_editor::FreezeImage {
            path: stored,
            width: size(image.width),
            height: size(image.height),
        };
        let requests = cutlass_editor::freeze_requests(
            self.draft(),
            segment_id,
            point.at,
            point.freeze,
            still,
            cutlass_eval::FREEZE_DURATION,
            &mut fresher(self.draft()),
        );
        let Some(requests) = requests else {
            let _ = std::fs::remove_file(&written);
            return Err(refused("the clip is not on the timeline"));
        };
        let placed = requests.iter().rev().find_map(|request| match request {
            Request::Place { segment, .. } => Some(segment.id.clone()),
            _ => None,
        });
        if let Err(error) = self.apply(self.revision(), &requests) {
            let _ = std::fs::remove_file(&written);
            return Err(error);
        }
        placed.ok_or_else(|| refused("the still was not placed"))
    }

    /// Drop a beat on a sound at the playhead `at`, or take away the beat
    /// already there. Returns whether a beat is there now.
    pub fn toggle_beat(&mut self, segment_id: &SegmentId, at: Micros) -> Result<bool, FfiError> {
        let draft = self.draft();
        let (segment, kind) = placed(draft, segment_id)?;
        if !cutlass_model::track_holds_beats(kind) {
            return Err(refused("a beat belongs on audio"));
        }
        let source = beat_source(segment, at)?;
        let near = cutlass_model::visible_beats(draft)
            .into_iter()
            .filter(|beat| &beat.segment_id == segment_id)
            .find(|beat| (beat.source - source).abs() <= BEAT_REACH);
        if let Some(beat) = near {
            self.apply(
                self.revision(),
                &[Request::RemoveBeat {
                    segment_id: segment_id.clone(),
                    source: Micros(beat.source),
                }],
            )?;
            return Ok(false);
        }
        let material_id = MaterialId::new(fresher(draft)("beats"));
        self.apply(
            self.revision(),
            &[Request::AddBeat {
                segment_id: segment_id.clone(),
                source: Micros(source),
                material_id,
            }],
        )?;
        Ok(true)
    }

    /// Put the whole of a picture from the library on an overlay row at
    /// `at`: the first free one, or a new one above the others. Returns the
    /// new clip.
    pub fn add_overlay(&mut self, material_id: &str, at: Micros) -> Result<SegmentId, FfiError> {
        let draft = self.draft();
        let (kind, duration) = cutlass_editor::material_span(draft, material_id)
            .ok_or_else(|| refused("the media is not in the library"))?;
        if kind != TrackType::Video {
            return Err(refused("an overlay is a picture"));
        }
        if duration.0 <= 0 {
            return Err(refused("the media has no duration"));
        }
        let at = Micros(at.0.max(0));
        let (track_id, segment_id, _) = media::room(draft, TrackType::Video, at, duration);
        let mut requests = Vec::new();
        let main = draft
            .tracks
            .iter()
            .find(|track| track.is_main())
            .map(|track| &track.id);
        let open = draft.tracks.iter().any(|track| track.id == track_id);
        if !open {
            requests.push(Request::InsertTrack {
                track_id: track_id.clone(),
                kind: TrackType::Video,
            });
        }
        if open && main == Some(&track_id) {
            return Err(refused("the overlay found no row"));
        }
        let mut segment = Segment::new(
            segment_id.clone(),
            material_id,
            cutlass_model::TimeRange::new(at.0, duration.0),
        );
        segment.source_range = Some(cutlass_model::TimeRange::new(0, duration.0));
        requests.push(Request::Place {
            track_id,
            segment: Box::new(segment),
        });
        self.apply(self.revision(), &requests)?;
        Ok(segment_id)
    }

    /// Move an overlay picture onto the main row where it starts. The main
    /// row makes room for it.
    pub fn to_main(&mut self, segment_id: &SegmentId) -> Result<(), FfiError> {
        let draft = self.draft();
        let (segment, kind) = placed(draft, segment_id)?;
        let main = draft
            .tracks
            .iter()
            .find(|track| track.is_main())
            .ok_or_else(|| refused("the project has no main row"))?;
        if *kind != TrackType::Video || main.segments.iter().any(|other| &other.id == segment_id) {
            return Err(refused("only an overlay picture goes to main"));
        }
        let request = Request::Move {
            segment_id: segment_id.clone(),
            track_id: main.id.clone(),
            start: segment.target.start,
        };
        self.apply(self.revision(), &[request])?;
        Ok(())
    }

    /// The beats on `segment_id`, as timeline times.
    pub fn beats(&self, segment_id: &SegmentId) -> Vec<Micros> {
        cutlass_model::visible_beats(self.draft())
            .into_iter()
            .filter(|beat| &beat.segment_id == segment_id)
            .map(|beat| beat.time)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quarter_turn_goes_clockwise_and_four_come_back() {
        let mut degrees = 0.0;
        for expected in [270.0, 180.0, 90.0, 0.0] {
            degrees = clockwise_quarter(degrees);
            assert_eq!(degrees, expected);
        }
        assert_eq!(clockwise_quarter(f64::NAN), 270.0);
    }

    #[test]
    fn a_negative_fade_is_refused() {
        assert!(fade_length(-1).is_err());
        assert_eq!(fade_length(0).unwrap(), Micros(0));
    }

    #[test]
    fn the_center_crop_keeps_the_middle_half() {
        let crop = center_crop();
        assert_ne!(crop, Crop::default());
        assert_eq!(crop.lower_right_x - crop.upper_left_x, 0.5);
        assert_eq!(crop.lower_right_y - crop.upper_left_y, 0.5);
    }
}
