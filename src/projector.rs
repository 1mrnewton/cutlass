//! Slint view-model projection of the canonical Rust `Project`.
//!
//! The full domain → Slint conversion only runs **once**, at startup
//! (see [`Projector::new`]). After that the projector keeps id-keyed
//! handles to each track's clip `VecModel` plus an inner `clip_id → row`
//! map, so a single-clip mutation costs:
//!
//!   - 1 `HashMap` lookup by `&str` (no `String` allocation on the
//!     hot path — `HashMap<String, _>::get` accepts `&str` via `Borrow`),
//!   - 1 inner `HashMap` lookup by `&str`,
//!   - 1 `VecModel::row_data` clone of a `SlintClip` (refcounted
//!     `SharedString`s; no heap allocation in the steady state),
//!   - 1 `VecModel::set_row_data` write,
//!
//! and triggers a repaint of exactly that clip — no re-allocation of
//! any `VecModel`, no walk over unrelated tracks/clips. This is the
//! shape the command layer and the agent both feed mutations through.
//!
//! Structural mutations (insert lane, transfer clip across lanes) shift
//! a small number of row indices alongside the `VecModel` writes; see
//! [`Projector::transfer_clip`] and [`Projector::insert_track_with_clip`].

use std::collections::HashMap;
use std::rc::Rc;

use slint::{Model, ModelRc, SharedString, VecModel};

use crate::models::{
    Clip, Color, Project, Rational, RationalTime, TimeRange, Track, TrackKind,
};
use crate::{
    Clip as SlintClip, Project as SlintProject, Rational as SlintRational,
    RationalTime as SlintRationalTime, Sequence as SlintSequence, TimeRange as SlintTimeRange,
    Track as SlintTrack, TrackKind as SlintTrackKind,
};

/// Per-track Slint-side state needed for O(1) targeted updates.
struct TrackProjection {
    /// The `VecModel` backing this track's `clips` array in Slint.
    /// We keep a strong `Rc` so we can mutate rows in place after the
    /// projection has been handed to the UI (the `ModelRc` inside
    /// `SlintTrack.clips` shares the same allocation).
    clips: Rc<VecModel<SlintClip>>,
    /// `clip_id → row` in the `clips` model above. Patched whenever a
    /// command inserts / removes / reorders clips on this track.
    clip_row: HashMap<String, usize>,
}

/// Owns the Slint-facing `Project` plus the indices needed to push
/// targeted updates into it without re-projecting from scratch.
pub struct Projector {
    project: SlintProject,
    /// `Rc` to the same `VecModel` that lives inside `project.sequence.tracks`.
    /// Held separately so structural lane changes (insert / remove) only
    /// touch this allocation and never re-issue `set_project`.
    tracks_model: Rc<VecModel<SlintTrack>>,
    /// Per-track projection state, keyed by track id.
    tracks: HashMap<String, TrackProjection>,
}

impl Projector {
    /// Build the initial projection of `project` and the index tables
    /// that subsequent commands will use to patch it surgically.
    pub fn new(project: &Project) -> Self {
        let sequence = &project.sequence;

        let mut tracks_index: HashMap<String, TrackProjection> =
            HashMap::with_capacity(sequence.track_order.len());
        let mut slint_tracks: Vec<SlintTrack> = Vec::with_capacity(sequence.track_order.len());

        for track_id in &sequence.track_order {
            let Some(track) = sequence.tracks.get(track_id) else {
                continue;
            };

            let (slint_track, projection) = make_track_projection(track);
            slint_tracks.push(slint_track);
            tracks_index.insert(track.id.clone(), projection);
        }

        let tracks_model = Rc::new(VecModel::from(slint_tracks));

        let slint_project = SlintProject {
            id: SharedString::from(project.id.as_str()),
            title: SharedString::from(project.title.as_str()),
            sequence: SlintSequence {
                id: SharedString::from(sequence.id.as_str()),
                name: SharedString::from(sequence.name.as_str()),
                fps: rational_to_slint(&sequence.fps),
                drop_frame: sequence.drop_frame,
                tracks: ModelRc::from(tracks_model.clone()),
                width: sequence.width,
                height: sequence.height,
            },
        };

        Self {
            project: slint_project,
            tracks_model,
            tracks: tracks_index,
        }
    }

    /// The Slint-facing `Project`. Cheap to `clone` — every field is
    /// either `Copy`, a refcounted `SharedString`, or a `ModelRc`
    /// (an `Rc` under the hood), so the clone the caller does to hand
    /// it to `EditorStore::set_project` is just a handful of refcount
    /// bumps.
    #[inline]
    pub fn slint_project(&self) -> &SlintProject {
        &self.project
    }

    /// Patch a single clip's `timeline_start.value` in the Slint
    /// projection. Returns `false` if the (track, clip) pair isn't
    /// indexed — caller decides whether that's an error or a no-op.
    ///
    /// Touches exactly one `VecModel` row; Slint repaints just that
    /// clip. The clip's rate is preserved (a move never changes
    /// rate), matching the contract of `Command::MoveClip`.
    pub fn move_clip(&self, track_id: &str, clip_id: &str, new_start_value: i32) -> bool {
        let Some(track) = self.tracks.get(track_id) else {
            return false;
        };
        let Some(&row) = track.clip_row.get(clip_id) else {
            return false;
        };
        let Some(mut clip) = track.clips.row_data(row) else {
            return false;
        };
        clip.timeline_start.value = new_start_value;
        track.clips.set_row_data(row, clip);
        true
    }

    /// Move a clip from one lane to another at `new_start_value`. The
    /// row is removed from the source lane and pushed onto the end of
    /// the destination lane; intra-lane order is otherwise preserved.
    /// (Z-order within a lane doesn't matter since clips can't overlap
    /// — the command layer guarantees that.)
    pub fn transfer_clip(
        &mut self,
        source_track_id: &str,
        target_track_id: &str,
        clip_id: &str,
        new_start_value: i32,
    ) -> bool {
        // Read+remove from source. Done in a borrow scope so we can
        // re-borrow `self.tracks` mutably for the target lane after.
        let mut clip = {
            let Some(source) = self.tracks.get_mut(source_track_id) else {
                return false;
            };
            let Some(&row) = source.clip_row.get(clip_id) else {
                return false;
            };
            let Some(clip) = source.clips.row_data(row) else {
                return false;
            };
            source.clips.remove(row);
            source.clip_row.remove(clip_id);
            // Every clip after the removed row shifted up by one.
            for (_, r) in source.clip_row.iter_mut() {
                if *r > row {
                    *r -= 1;
                }
            }
            clip
        };

        clip.timeline_start.value = new_start_value;

        let Some(target) = self.tracks.get_mut(target_track_id) else {
            return false;
        };
        let row = target.clips.row_count();
        target.clips.push(clip);
        target.clip_row.insert(clip_id.to_owned(), row);
        true
    }

    /// Insert a freshly minted lane into the tracks model at
    /// `insert_at_index`, then move `clip_id` from `source_track_id`
    /// into it at `new_start_value`. Returns `false` if anything looks
    /// stale — same contract as the other patch methods.
    ///
    /// Caller owns the (track id, name, kind, color) tuple so the
    /// policy layer fully decides identity / placement; the projector
    /// only has to mirror it into Slint.
    pub fn insert_track_with_clip(
        &mut self,
        new_track_id: &str,
        new_track_name: &str,
        new_track_kind: TrackKind,
        new_track_color: Color,
        insert_at_index: usize,
        source_track_id: &str,
        clip_id: &str,
        new_start_value: i32,
    ) -> bool {
        // Pull the clip off the source lane first; if that fails we
        // don't want to leave a half-built track around. This mirrors
        // the domain order in `command::apply` so the two stay in
        // lock-step.
        let mut clip = {
            let Some(source) = self.tracks.get_mut(source_track_id) else {
                return false;
            };
            let Some(&row) = source.clip_row.get(clip_id) else {
                return false;
            };
            let Some(clip) = source.clips.row_data(row) else {
                return false;
            };
            source.clips.remove(row);
            source.clip_row.remove(clip_id);
            for (_, r) in source.clip_row.iter_mut() {
                if *r > row {
                    *r -= 1;
                }
            }
            clip
        };

        clip.timeline_start.value = new_start_value;

        // Build the empty Slint track + its projection, then put the
        // clip in. Doing the clip insert *through* the projection
        // (rather than building a 1-element `VecModel`) keeps the
        // `clip_row` map and the `VecModel` strictly in sync via the
        // same code path.
        let new_track = Track {
            id: new_track_id.to_owned(),
            name: new_track_name.to_owned(),
            kind: new_track_kind,
            color: new_track_color,
            clip_order: Vec::new(),
            clips: HashMap::new(),
        };
        let (slint_track, mut projection) = make_track_projection(&new_track);

        let row = projection.clips.row_count();
        projection.clips.push(clip);
        projection.clip_row.insert(clip_id.to_owned(), row);

        // `VecModel::insert` on the tracks model + index insert. Slint
        // sees one new row and renders it; existing track rows aren't
        // touched.
        self.tracks_model.insert(insert_at_index, slint_track);
        self.tracks.insert(new_track_id.to_owned(), projection);
        true
    }
}

/// Build a Slint track row + its per-track projection from a domain
/// `Track`. Shared by initial projection and structural inserts so the
/// two paths can't drift.
fn make_track_projection(track: &Track) -> (SlintTrack, TrackProjection) {
    let mut slint_clips: Vec<SlintClip> = Vec::with_capacity(track.clip_order.len());
    let mut clip_row: HashMap<String, usize> = HashMap::with_capacity(track.clip_order.len());

    for (row, clip_id) in track.clip_order.iter().enumerate() {
        let Some(clip) = track.clips.get(clip_id) else {
            continue;
        };
        slint_clips.push(clip_to_slint(clip));
        clip_row.insert(clip_id.clone(), row);
    }

    let clips_model = Rc::new(VecModel::from(slint_clips));
    let slint_track = SlintTrack {
        id: SharedString::from(track.id.as_str()),
        name: SharedString::from(track.name.as_str()),
        kind: track_kind_to_slint(track.kind),
        color: color_to_slint(track.color),
        clips: ModelRc::from(clips_model.clone()),
    };
    (
        slint_track,
        TrackProjection {
            clips: clips_model,
            clip_row,
        },
    )
}

#[inline]
fn clip_to_slint(clip: &Clip) -> SlintClip {
    SlintClip {
        id: SharedString::from(clip.id.as_str()),
        name: SharedString::from(clip.name.as_str()),
        timeline_start: rational_time_to_slint(&clip.timeline_start),
        source_range: time_range_to_slint(&clip.source_range),
    }
}

#[inline]
fn rational_to_slint(r: &Rational) -> SlintRational {
    SlintRational {
        num: r.num,
        den: r.den,
    }
}

#[inline]
fn rational_time_to_slint(rt: &RationalTime) -> SlintRationalTime {
    SlintRationalTime {
        value: rt.value,
        rate: rational_to_slint(&rt.rate),
    }
}

#[inline]
fn time_range_to_slint(range: &TimeRange) -> SlintTimeRange {
    SlintTimeRange {
        start: rational_time_to_slint(&range.start),
        duration: rational_time_to_slint(&range.duration),
    }
}

#[inline]
pub(crate) fn track_kind_to_slint(kind: TrackKind) -> SlintTrackKind {
    match kind {
        TrackKind::Video => SlintTrackKind::Video,
        TrackKind::Audio => SlintTrackKind::Audio,
    }
}

#[inline]
fn color_to_slint(c: Color) -> slint::Color {
    // `slint::Color::from_argb_u8` takes ARGB, not RGBA — easy to
    // get backwards. Our domain `Color` stores RGBA so we reorder
    // explicitly here in one place.
    slint::Color::from_argb_u8(c.a, c.r, c.g, c.b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::sample_project;

    #[test]
    fn projection_preserves_track_and_clip_order() {
        let domain = sample_project();
        let projector = Projector::new(&domain);
        let slint = projector.slint_project();

        assert_eq!(
            slint.sequence.tracks.row_count(),
            domain.sequence.track_order.len()
        );

        for (i, track_id) in domain.sequence.track_order.iter().enumerate() {
            let track = domain.sequence.tracks.get(track_id).unwrap();
            let slint_track = slint.sequence.tracks.row_data(i).unwrap();
            assert_eq!(slint_track.id, track.id);
            assert_eq!(slint_track.clips.row_count(), track.clip_order.len());

            for (j, clip_id) in track.clip_order.iter().enumerate() {
                let clip = track.clips.get(clip_id).unwrap();
                let slint_clip = slint_track.clips.row_data(j).unwrap();
                assert_eq!(slint_clip.id, clip.id);
                assert_eq!(slint_clip.timeline_start.value, clip.timeline_start.value);
            }
        }
    }

    #[test]
    fn move_clip_patches_only_the_targeted_row() {
        let domain = sample_project();
        let projector = Projector::new(&domain);

        // Capture the state of every other clip so we can assert
        // nothing else moved.
        let before: Vec<(usize, usize, SlintClip)> = {
            let slint = projector.slint_project();
            let mut out = Vec::new();
            for ti in 0..slint.sequence.tracks.row_count() {
                let t = slint.sequence.tracks.row_data(ti).unwrap();
                for ci in 0..t.clips.row_count() {
                    out.push((ti, ci, t.clips.row_data(ci).unwrap()));
                }
            }
            out
        };

        // Move "Clip 1" on track "1" from 10 → 42.
        assert!(projector.move_clip("1", "1", 42));

        let slint = projector.slint_project();
        for (ti, ci, prev) in before {
            let now = slint
                .sequence
                .tracks
                .row_data(ti)
                .unwrap()
                .clips
                .row_data(ci)
                .unwrap();
            if prev.id == "1" {
                assert_eq!(now.timeline_start.value, 42, "target clip not updated");
                assert_eq!(now.timeline_start.rate.num, prev.timeline_start.rate.num);
                assert_eq!(now.timeline_start.rate.den, prev.timeline_start.rate.den);
                assert_eq!(
                    now.source_range.duration.value,
                    prev.source_range.duration.value
                );
            } else {
                assert_eq!(
                    now.timeline_start.value, prev.timeline_start.value,
                    "unrelated clip {:?} was modified",
                    prev.id
                );
            }
        }
    }

    #[test]
    fn move_clip_returns_false_for_unknown_ids() {
        let domain = sample_project();
        let projector = Projector::new(&domain);

        assert!(!projector.move_clip("does-not-exist", "1", 0));
        assert!(!projector.move_clip("1", "does-not-exist", 0));
    }

    #[test]
    fn transfer_clip_moves_row_and_updates_indices() {
        let domain = sample_project();
        let mut projector = Projector::new(&domain);

        // Move "Clip 1" from V1 → V2 at start=500.
        assert!(projector.transfer_clip("1", "2", "1", 500));

        let slint = projector.slint_project();

        // Source V1 (row 0) should be empty.
        let v1 = slint.sequence.tracks.row_data(0).unwrap();
        assert_eq!(v1.clips.row_count(), 0);

        // Target V2 (row 1) should now end with the transferred clip.
        let v2 = slint.sequence.tracks.row_data(1).unwrap();
        assert_eq!(v2.clips.row_count(), 3);
        let last = v2.clips.row_data(v2.clips.row_count() - 1).unwrap();
        assert_eq!(last.id, "1");
        assert_eq!(last.timeline_start.value, 500);
    }

    #[test]
    fn insert_track_with_clip_creates_new_lane_at_index() {
        let domain = sample_project();
        let initial_count = domain.sequence.track_order.len();
        let mut projector = Projector::new(&domain);

        // Spawn a new video lane just above V2 (row index 1) and put
        // Clip 1 there, colored with a sentinel so we can check the
        // color path end-to-end.
        let new_color = Color::rgb(0x11, 0x22, 0x33);
        assert!(projector.insert_track_with_clip(
            "new1",
            "V?",
            TrackKind::Video,
            new_color,
            1,
            "1",
            "1",
            777,
        ));

        let slint = projector.slint_project();
        assert_eq!(slint.sequence.tracks.row_count(), initial_count + 1);

        let new_track = slint.sequence.tracks.row_data(1).unwrap();
        assert_eq!(new_track.id, "new1");
        assert_eq!(new_track.kind, SlintTrackKind::Video);
        assert_eq!(new_track.clips.row_count(), 1);
        // Color round-trips through the projector.
        assert_eq!(new_track.color.red(), 0x11);
        assert_eq!(new_track.color.green(), 0x22);
        assert_eq!(new_track.color.blue(), 0x33);

        let inserted = new_track.clips.row_data(0).unwrap();
        assert_eq!(inserted.id, "1");
        assert_eq!(inserted.timeline_start.value, 777);

        // Original source lane (now row 0) lost its clip.
        let v1 = slint.sequence.tracks.row_data(0).unwrap();
        assert_eq!(v1.clips.row_count(), 0);
    }
}
