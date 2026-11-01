mod beat;
mod crop;
mod extract;
mod fade;
mod freeze;
mod import;
mod keyframe;
mod linkage;
mod magnet;
mod mark;
mod mirror;
mod move_segment;
mod place;
mod placement;
mod reverse;
mod rotate;
mod speed;
mod split;
mod text;
mod track;
mod transition;
mod trim;
mod volume;
mod words;

use cutlass_model::{Draft, LinkageSettings, Micros, RangeError, TrackAttribute};

use crate::request::Request;

pub use move_segment::move_segment;
pub use place::place;
pub use trim::trim;
pub use words::set_words;

/// Why one request did not change the draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditError {
    /// The caller read a different revision than the open draft.
    RevisionMismatch {
        expected: u64,
        actual: u64,
    },
    MissingTrack,
    MissingSegment,
    /// The segment id is empty or already on the draft.
    DuplicateSegment,
    /// The track attribute has the lock bit set.
    LockedTrack,
    /// Two positive spans on one track overlap. Touching does not.
    Overlap,
    /// The constant-speed mapping refused the duration or the speed.
    Speed(cutlass_model::ConstantSpeedError),
    /// Word columns disagree, or a word ends before it starts.
    WordShape,
    Range(RangeError),
    EmptyUndo,
    EmptyRedo,
    /// `begin` or `record` was called while a transaction was already open.
    HistoryOpen,
    /// `commit`, `amend`, or `rollback` needs the transaction `begin` or `record` opened.
    HistoryClosed,
    /// `amend` folds into a recorded step, and none exists yet.
    NothingToAmend,
    /// The bit is not mute, hide, lock, solo, or muted-by-solo.
    UnknownAttribute,
    MissingMaterial,
    /// The material id is already in a pool.
    DuplicateMaterial,
    /// The split time falls inside a blend window.
    TransitionWindow,
    /// The dependent already has an owner, or it is an owner itself.
    AlreadyLinked,
    /// A marker is already stored at that time.
    DuplicateMark,
    /// A beat is already stored at that source time.
    DuplicateBeat,
    /// A beat can be stored only on an audio segment.
    NotAudio,
    /// Extract audio applies to a video clip.
    NotVideo,
    /// The picture has no sound to place.
    NoSound,
    /// The picture's sound is already its own clip.
    AlreadyExtracted,
    /// A track with this id is already on the draft, or this type cannot be a lane.
    TrackKind,
    /// The rotation is not a finite number of degrees.
    Angle,
    /// Position is not a finite number.
    Position,
    /// Scale is not finite and greater than zero.
    Scale,
    /// A fade duration is negative.
    Fade,
}

pub fn apply_request(draft: &mut Draft, request: &Request) -> Result<(), EditError> {
    match request {
        Request::Place { track_id, segment } => place(draft, track_id, (**segment).clone()),
        Request::Move {
            segment_id,
            track_id,
            start,
        } => move_segment(draft, segment_id, track_id, *start),
        Request::Trim {
            segment_id,
            edge,
            time,
        } => trim(draft, segment_id, *edge, *time),
        Request::SetWords { material_id, words } => set_words(draft, material_id, words.clone()),
        Request::AddCaption {
            source_id,
            track_id,
            segment_id,
            material_id,
            words,
        } => words::add_caption(
            draft,
            source_id,
            track_id,
            segment_id,
            material_id,
            words.clone(),
        ),
        Request::ExtractAudio {
            source_id,
            track_id,
            segment_id,
            material_id,
        } => extract::extract_audio(draft, source_id, track_id, segment_id, material_id),
        Request::SetTrackAttribute { track_id, bit, on } => {
            track::set_track_attribute(draft, track_id, *bit, *on)
        }
        Request::SetConstantSpeed { segment_id, speed } => {
            speed::set_constant_speed(draft, segment_id, *speed)
        }
        Request::SetCurveSpeed {
            segment_id,
            speed_id,
            points,
        } => speed::set_curve_speed(draft, segment_id, speed_id, points),
        Request::SetTransition {
            segment_id,
            transition,
        } => transition::set_transition(draft, segment_id, (**transition).clone()),
        Request::Split {
            segment_id,
            at,
            right_id,
        } => split::split(draft, segment_id, *at, right_id),
        Request::Link { owner, dependent } => linkage::link(draft, owner, dependent),
        Request::Remove { segment_id } => linkage::remove(draft, segment_id),
        Request::SetReverse {
            segment_id,
            reverse,
            path,
        } => reverse::set_reverse(draft, segment_id, *reverse, path),
        Request::SetFreeze { segment_id, freeze } => {
            freeze::set_freeze(draft, segment_id, freeze.clone())
        }
        Request::SetMirror {
            segment_id,
            horizontal,
        } => mirror::set_mirror(draft, segment_id, *horizontal),
        Request::SetRotation {
            segment_id,
            degrees,
        } => rotate::set_rotation(draft, segment_id, *degrees),
        Request::SetPosition { segment_id, x, y } => {
            placement::set_position(draft, segment_id, *x, *y)
        }
        Request::SetScale { segment_id, scale } => placement::set_scale(draft, segment_id, *scale),
        Request::SetVolume { segment_id, volume } => volume::set_volume(draft, segment_id, *volume),
        Request::SetFade {
            segment_id,
            fade_id,
            fade_in,
            fade_out,
        } => fade::set_fade(draft, segment_id, fade_id, *fade_in, *fade_out),
        Request::SetCrop { segment_id, crop } => crop::set_crop(draft, segment_id, *crop),
        Request::SetTextContent {
            segment_id,
            content,
        } => text::set_text(draft, segment_id, content),
        Request::SetKeyframes {
            segment_id,
            keyframes,
        } => keyframe::set_keyframes(draft, segment_id, keyframes),
        Request::Import { medium } => import::import(draft, medium.clone()),
        Request::InsertTrack { track_id, kind } => track::insert_track(draft, track_id, kind),
        Request::AddTimeMark {
            segment_id,
            at,
            color,
            material_id,
        } => mark::add_time_mark(draft, segment_id.as_ref(), *at, color, material_id),
        Request::RemoveTimeMark { segment_id, start } => {
            mark::remove_time_mark(draft, segment_id.as_ref(), *start)
        }
        Request::AddBeat {
            segment_id,
            source,
            material_id,
        } => beat::add_beat(draft, segment_id, *source, material_id),
        Request::RemoveBeat { segment_id, source } => beat::remove_beat(draft, segment_id, *source),
        Request::SetMagnet { on } => magnet::set_magnet(draft, *on),
    }
}

pub fn apply_all(
    draft: &mut Draft,
    requests: &[Request],
    linkage: &LinkageSettings,
) -> Result<(), EditError> {
    for request in requests {
        let before = draft.clone();
        apply_request(draft, request)?;
        linkage::after_request(draft, &before, linkage)?;
        magnet::pack(draft, linkage)?;
    }
    sync_duration(draft);
    Ok(())
}

/// Stored duration becomes the latest target end. Playback reads that field.
pub fn sync_duration(draft: &mut Draft) {
    let mut end = 0_i64;
    for track in &draft.tracks {
        for segment in &track.segments {
            end = end.max(segment.target.end().0);
        }
    }
    draft.duration = Micros(end);
}

pub fn ensure_unlocked(attribute: TrackAttribute) -> Result<(), EditError> {
    if attribute.contains(TrackAttribute::LOCK) {
        Err(EditError::LockedTrack)
    } else {
        Ok(())
    }
}
