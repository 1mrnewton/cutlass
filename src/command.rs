//! Structured edit commands.
//!
//! Every domain mutation funnels through this enum — UI gestures and
//! the AI agent both speak the same vocabulary. The `#[serde(tag = "kind")]`
//! representation gives a tagged-JSON wire format that the agent can
//! emit directly:
//!
//! ```json
//! {
//!   "kind": "move_clip",
//!   "source_track_id": "1",
//!   "clip_id": "1",
//!   "target_track_id": "2",
//!   "new_start_value": 42
//! }
//! ```
//!
//! ## Move-clip semantics
//!
//! A single `MoveClip` command encodes the full intent of a drag:
//! "place `clip_id` on `target_track_id` starting at
//! `new_start_value` ticks". The command layer turns that into one of
//! three concrete edits, picked by inspecting current sequence state:
//!
//!   1. **Same lane, no collision** — the clip's `timeline_start` is
//!      updated in place. Produces [`Effect::ClipMoved`].
//!   2. **Cross-lane, no collision, same kind** — the clip is removed
//!      from `source_track_id` and appended onto `target_track_id`.
//!      Produces [`Effect::ClipTransferred`].
//!   3. **Collision on the target lane** — a fresh lane is minted of
//!      the source clip's kind and inserted just above
//!      `target_track_id` in `track_order`; the clip lands there.
//!      Produces [`Effect::ClipTransferredToNewTrack`].
//!
//! "Collision" means strict integer overlap on the target lane,
//! `[a.start, a.end) ∩ [b.start, b.end) != ∅`. A 1-tick gap is fine.
//!
//! Cross-kind drops (e.g. a video clip dropped on an audio lane) are
//! rejected with [`CommandError::TrackKindMismatch`]. The gesture
//! layer already filters these out of the target list; the check here
//! is a defensive guard for agent-emitted commands.
//!
//! ## Invariants preserved
//!
//! * Audio lanes always trail video lanes in `Sequence::track_order`
//!   — new lanes inherit the source clip's kind and are inserted at
//!   the target index, which already sits inside its kind's block.
//! * `Track::clip_order` and `Track::clips` are kept in lock-step.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::models::{Color, Project, TrackKind};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Command {
    /// Place `clip_id` (currently on `source_track_id`) onto
    /// `target_track_id` starting at `new_start_value` ticks.
    ///
    /// `target_track_id` may equal `source_track_id` (intra-lane
    /// reposition). When they differ, this is a transfer; collisions
    /// auto-spawn a new lane (see module docs).
    ///
    /// Rate is intentionally **not** part of the command — a move
    /// never changes the clip's authoring rate, and inheriting it
    /// from the existing `timeline_start.rate` removes the only
    /// rounding step the hot drag path would otherwise need.
    MoveClip {
        source_track_id: String,
        clip_id: String,
        target_track_id: String,
        new_start_value: i32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandError {
    UnknownTrack {
        track_id: String,
    },
    UnknownClip {
        track_id: String,
        clip_id: String,
    },
    /// Drop target's kind doesn't match the clip's source lane (e.g.
    /// video clip → audio lane). The gesture layer already prevents
    /// this; commands from the agent that violate it are no-ops.
    TrackKindMismatch {
        source_track_id: String,
        target_track_id: String,
    },
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommandError::UnknownTrack { track_id } => {
                write!(f, "unknown track {track_id:?}")
            }
            CommandError::UnknownClip { track_id, clip_id } => {
                write!(f, "unknown clip {clip_id:?} in track {track_id:?}")
            }
            CommandError::TrackKindMismatch {
                source_track_id,
                target_track_id,
            } => {
                write!(
                    f,
                    "cannot move clip from {source_track_id:?} (one kind) to \
                     {target_track_id:?} (different kind)",
                )
            }
        }
    }
}

impl std::error::Error for CommandError {}

/// Apply `command` to `project` in place. Pure function over plain
/// Rust structs — no Slint dependency, no I/O — so this is where
/// command-level invariants are unit-tested.
///
/// Returns `Ok(Effect)` describing what changed so the projector
/// can update only the affected rows (instead of re-walking the
/// whole tree).
pub fn apply(project: &mut Project, command: &Command) -> Result<Effect, CommandError> {
    match command {
        Command::MoveClip {
            source_track_id,
            clip_id,
            target_track_id,
            new_start_value,
        } => apply_move_clip(project, source_track_id, clip_id, target_track_id, *new_start_value),
    }
}

/// What changed in the project as a result of applying a command.
///
/// Keeping this distinct from `Command` lets the projector update
/// only the surfaces that need it — a structural drop produces an
/// effect that mentions multiple rows without inflating the command
/// vocabulary the agent has to learn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// In-place reposition on `track_id` — no structural change.
    ClipMoved {
        track_id: String,
        clip_id: String,
        new_start_value: i32,
    },
    /// Clip removed from `source_track_id` and appended onto
    /// `target_track_id` at `new_start_value`. Lane row counts shift;
    /// existing rows on other lanes are untouched.
    ClipTransferred {
        source_track_id: String,
        target_track_id: String,
        clip_id: String,
        new_start_value: i32,
    },
    /// A fresh lane was inserted at `insert_at_index` in the sequence's
    /// `track_order` (and in the Slint tracks model) and the clip was
    /// moved there. The lane's kind matches the moving clip's source
    /// kind, so the audio-below-video invariant is preserved.
    ClipTransferredToNewTrack {
        source_track_id: String,
        new_track_id: String,
        new_track_name: String,
        new_track_kind: TrackKind,
        new_track_color: Color,
        insert_at_index: usize,
        clip_id: String,
        new_start_value: i32,
    },
}

fn apply_move_clip(
    project: &mut Project,
    source_track_id: &str,
    clip_id: &str,
    target_track_id: &str,
    new_start_value: i32,
) -> Result<Effect, CommandError> {
    // ---- 1. Validate the source clip exists --------------------------
    let (clip_duration, source_kind) = {
        let source = project
            .sequence
            .tracks
            .get(source_track_id)
            .ok_or_else(|| CommandError::UnknownTrack {
                track_id: source_track_id.to_owned(),
            })?;
        let clip = source.clips.get(clip_id).ok_or_else(|| CommandError::UnknownClip {
            track_id: source_track_id.to_owned(),
            clip_id: clip_id.to_owned(),
        })?;
        (clip.source_range.duration.value, source.kind)
    };

    // ---- 2. Fast path: intra-lane reposition -------------------------
    if source_track_id == target_track_id {
        let track = project
            .sequence
            .tracks
            .get_mut(source_track_id)
            .expect("checked above");
        // Collision check excludes the clip being moved.
        if has_overlap_excluding(track, clip_id, new_start_value, clip_duration) {
            // Same-lane collision → spawn a fresh lane above source.
            return spawn_lane_and_transfer(
                project,
                source_track_id,
                source_kind,
                clip_id,
                target_track_id,
                new_start_value,
            );
        }
        // No collision: just patch the value.
        let clip = track.clips.get_mut(clip_id).expect("checked above");
        clip.timeline_start.value = new_start_value;
        return Ok(Effect::ClipMoved {
            track_id: source_track_id.to_owned(),
            clip_id: clip_id.to_owned(),
            new_start_value,
        });
    }

    // ---- 3. Cross-lane: validate target lane -------------------------
    let target = project
        .sequence
        .tracks
        .get(target_track_id)
        .ok_or_else(|| CommandError::UnknownTrack {
            track_id: target_track_id.to_owned(),
        })?;

    if target.kind != source_kind {
        return Err(CommandError::TrackKindMismatch {
            source_track_id: source_track_id.to_owned(),
            target_track_id: target_track_id.to_owned(),
        });
    }

    // Cross-lane collision check looks at every clip on the target —
    // there's no "self" to exclude because the clip lives on the
    // source lane until this command commits.
    let collides = has_overlap_excluding(target, "", new_start_value, clip_duration);

    if collides {
        return spawn_lane_and_transfer(
            project,
            source_track_id,
            source_kind,
            clip_id,
            target_track_id,
            new_start_value,
        );
    }

    // ---- 4. Cross-lane, no collision: simple transfer ----------------
    let mut clip = {
        let source = project
            .sequence
            .tracks
            .get_mut(source_track_id)
            .expect("checked above");
        source.clip_order.retain(|id| id != clip_id);
        source.clips.remove(clip_id).expect("checked above")
    };
    clip.timeline_start.value = new_start_value;

    let target = project
        .sequence
        .tracks
        .get_mut(target_track_id)
        .expect("checked above");
    target.clip_order.push(clip_id.to_owned());
    target.clips.insert(clip_id.to_owned(), clip);

    Ok(Effect::ClipTransferred {
        source_track_id: source_track_id.to_owned(),
        target_track_id: target_track_id.to_owned(),
        clip_id: clip_id.to_owned(),
        new_start_value,
    })
}

/// Mint a fresh lane of `new_kind`, insert it into `track_order` just
/// above `above_track_id`, transfer `clip_id` from `source_track_id`
/// into it at `new_start_value`. The "just above" placement keeps the
/// new lane inside the source kind's contiguous block of lanes — so
/// the audio-below-video invariant is automatic.
fn spawn_lane_and_transfer(
    project: &mut Project,
    source_track_id: &str,
    new_kind: TrackKind,
    clip_id: &str,
    above_track_id: &str,
    new_start_value: i32,
) -> Result<Effect, CommandError> {
    let seq = &mut project.sequence;

    let insert_at_index = seq
        .track_order
        .iter()
        .position(|id| id == above_track_id)
        .ok_or_else(|| CommandError::UnknownTrack {
            track_id: above_track_id.to_owned(),
        })?;

    let new_track_id = format!("auto-{}", seq.next_track_id);
    seq.next_track_id += 1;
    let kind_count = count_kind(seq, new_kind);
    let new_track_name = match new_kind {
        TrackKind::Video => format!("V{}", kind_count + 1),
        TrackKind::Audio => format!("A{}", kind_count + 1),
    };
    // Pick the next color in this kind's palette so the new lane is
    // visually distinct from the existing lanes of the same kind
    // (until the palette runs out — then it cycles, which is
    // intentional: better than running out into noisy hues).
    let new_track_color = new_kind.palette_color(kind_count);

    // Lift the clip off the source lane.
    let mut clip = {
        let source = seq.tracks.get_mut(source_track_id).ok_or_else(|| {
            CommandError::UnknownTrack {
                track_id: source_track_id.to_owned(),
            }
        })?;
        source.clip_order.retain(|id| id != clip_id);
        source.clips.remove(clip_id).ok_or_else(|| CommandError::UnknownClip {
            track_id: source_track_id.to_owned(),
            clip_id: clip_id.to_owned(),
        })?
    };
    clip.timeline_start.value = new_start_value;

    // Build the new lane with the clip already on it.
    let mut clips = std::collections::HashMap::with_capacity(1);
    clips.insert(clip_id.to_owned(), clip);
    let new_track = crate::models::Track {
        id: new_track_id.clone(),
        name: new_track_name.clone(),
        kind: new_kind,
        color: new_track_color,
        clip_order: vec![clip_id.to_owned()],
        clips,
    };

    seq.track_order.insert(insert_at_index, new_track_id.clone());
    seq.tracks.insert(new_track_id.clone(), new_track);

    Ok(Effect::ClipTransferredToNewTrack {
        source_track_id: source_track_id.to_owned(),
        new_track_id,
        new_track_name,
        new_track_kind: new_kind,
        new_track_color,
        insert_at_index,
        clip_id: clip_id.to_owned(),
        new_start_value,
    })
}

/// Strict overlap test (no touching allowed isn't quite right — clips
/// sharing exactly one tick at the boundary `a.end == b.start` are
/// considered NOT colliding, matching the half-open `[start, end)`
/// interval convention used everywhere else in the codebase).
#[inline]
fn has_overlap_excluding(
    track: &crate::models::Track,
    exclude_clip_id: &str,
    new_start: i32,
    new_duration: i32,
) -> bool {
    let new_end = new_start.saturating_add(new_duration);
    for (id, c) in &track.clips {
        if id == exclude_clip_id {
            continue;
        }
        let other_start = c.timeline_start.value;
        let other_end = other_start.saturating_add(c.source_range.duration.value);
        // Half-open overlap: max(a_start, b_start) < min(a_end, b_end).
        if new_start.max(other_start) < new_end.min(other_end) {
            return true;
        }
    }
    false
}

#[inline]
fn count_kind(seq: &crate::models::Sequence, kind: TrackKind) -> usize {
    seq.tracks.values().filter(|t| t.kind == kind).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::sample_project;

    // ---- intra-lane, no collision -------------------------------------

    #[test]
    fn move_clip_intra_lane_updates_only_timeline_start_value() {
        let mut p = sample_project();

        let target_before = p
            .sequence
            .tracks
            .get("1")
            .unwrap()
            .clips
            .get("1")
            .unwrap()
            .clone();
        let neighbour_before = p
            .sequence
            .tracks
            .get("2")
            .unwrap()
            .clips
            .get("2")
            .unwrap()
            .clone();

        let effect = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "1".into(),
                clip_id: "1".into(),
                target_track_id: "1".into(),
                new_start_value: 999,
            },
        )
        .unwrap();

        assert_eq!(
            effect,
            Effect::ClipMoved {
                track_id: "1".into(),
                clip_id: "1".into(),
                new_start_value: 999,
            }
        );

        let after = p.sequence.tracks.get("1").unwrap().clips.get("1").unwrap();
        assert_eq!(after.timeline_start.value, 999);
        assert_eq!(after.timeline_start.rate, target_before.timeline_start.rate);
        assert_eq!(after.source_range, target_before.source_range);
        assert_eq!(after.name, target_before.name);
        assert_eq!(after.id, target_before.id);

        let neighbour_after = p.sequence.tracks.get("2").unwrap().clips.get("2").unwrap();
        assert_eq!(neighbour_after, &neighbour_before);
    }

    #[test]
    fn move_clip_accepts_negative_start() {
        let mut p = sample_project();
        apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "1".into(),
                clip_id: "1".into(),
                target_track_id: "1".into(),
                new_start_value: -50,
            },
        )
        .unwrap();
        assert_eq!(
            p.sequence
                .tracks
                .get("1")
                .unwrap()
                .clips
                .get("1")
                .unwrap()
                .timeline_start
                .value,
            -50
        );
    }

    // ---- intra-lane collision → new lane spawn -----------------------

    #[test]
    fn spawned_lane_picks_color_from_source_kind_palette() {
        let mut p = sample_project();

        // Force a spawn — drop Clip 3 onto Clip 2 within V2.
        let effect = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "2".into(),
                clip_id: "3".into(),
                target_track_id: "2".into(),
                new_start_value: 10,
            },
        )
        .unwrap();

        match effect {
            Effect::ClipTransferredToNewTrack {
                new_track_color,
                new_track_kind,
                ..
            } => {
                assert_eq!(new_track_kind, TrackKind::Video);
                // The new lane's color comes from the video palette
                // (not the audio palette, not the theme default,
                // not zero/black).
                let video_palette = TrackKind::Video.default_palette();
                assert!(
                    video_palette.contains(&new_track_color),
                    "spawned video lane color {new_track_color:?} not in video palette",
                );
            }
            other => panic!("expected spawn, got {other:?}"),
        }
    }

    #[test]
    fn intra_lane_drop_on_neighbour_spawns_new_lane_above() {
        let mut p = sample_project();
        let initial_v_count = count_kind(&p.sequence, TrackKind::Video);
        let initial_track_order = p.sequence.track_order.clone();

        // V2 has Clip 2 at [0, 80) and Clip 3 at [120, 180). Drop Clip
        // 3 onto Clip 2 by trying to start it at 10 — overlaps Clip 2.
        let effect = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "2".into(),
                clip_id: "3".into(),
                target_track_id: "2".into(),
                new_start_value: 10,
            },
        )
        .unwrap();

        let (new_id, insert_at) = match effect {
            Effect::ClipTransferredToNewTrack {
                ref new_track_id,
                insert_at_index,
                new_track_kind,
                ..
            } => {
                assert_eq!(new_track_kind, TrackKind::Video);
                (new_track_id.clone(), insert_at_index)
            }
            other => panic!("unexpected effect: {other:?}"),
        };

        // New lane lives at the index of V2 in the OLD order — V2
        // shifted down by one.
        let old_v2_pos = initial_track_order
            .iter()
            .position(|id| id == "2")
            .unwrap();
        assert_eq!(insert_at, old_v2_pos);
        assert_eq!(p.sequence.track_order[insert_at], new_id);
        assert_eq!(p.sequence.track_order[insert_at + 1], "2");

        // Clip 3 is now on the new lane at value 10, off V2.
        let new_lane = p.sequence.tracks.get(&new_id).unwrap();
        assert_eq!(new_lane.clip_order, vec!["3".to_string()]);
        assert_eq!(new_lane.clips.get("3").unwrap().timeline_start.value, 10);
        assert!(p.sequence.tracks.get("2").unwrap().clips.get("3").is_none());

        // Video count grew by 1, audio count unchanged → invariant holds.
        assert_eq!(count_kind(&p.sequence, TrackKind::Video), initial_v_count + 1);
        assert_track_order_invariant(&p);
    }

    #[test]
    fn boundary_touch_is_not_a_collision() {
        let mut p = sample_project();

        // V2 Clip 2 occupies [0, 80). Move Clip 3 (duration 60) so it
        // starts exactly at 80 — a 1-tick boundary, NOT an overlap.
        let effect = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "2".into(),
                clip_id: "3".into(),
                target_track_id: "2".into(),
                new_start_value: 80,
            },
        )
        .unwrap();
        assert!(matches!(effect, Effect::ClipMoved { .. }));
        assert_eq!(
            p.sequence
                .tracks
                .get("2")
                .unwrap()
                .clips
                .get("3")
                .unwrap()
                .timeline_start
                .value,
            80
        );
    }

    #[test]
    fn one_tick_overlap_is_a_collision() {
        let mut p = sample_project();

        // Same setup as above; one tick earlier than the boundary
        // would have Clip 3's [79, 139) overlap Clip 2's [0, 80) by 1.
        let effect = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "2".into(),
                clip_id: "3".into(),
                target_track_id: "2".into(),
                new_start_value: 79,
            },
        )
        .unwrap();
        assert!(matches!(
            effect,
            Effect::ClipTransferredToNewTrack { .. }
        ));
    }

    // ---- cross-lane, no collision ------------------------------------

    #[test]
    fn cross_lane_same_kind_no_collision_transfers() {
        let mut p = sample_project();

        // Move Clip 4 from V3 → V1 at start=500. V1 only has Clip 1 at
        // [10, 110), so [500, 590) is clear.
        let effect = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "3".into(),
                clip_id: "4".into(),
                target_track_id: "1".into(),
                new_start_value: 500,
            },
        )
        .unwrap();
        assert_eq!(
            effect,
            Effect::ClipTransferred {
                source_track_id: "3".into(),
                target_track_id: "1".into(),
                clip_id: "4".into(),
                new_start_value: 500,
            }
        );

        assert!(p.sequence.tracks.get("3").unwrap().clips.get("4").is_none());
        let on_v1 = p.sequence.tracks.get("1").unwrap().clips.get("4").unwrap();
        assert_eq!(on_v1.timeline_start.value, 500);
    }

    // ---- cross-lane collision → new lane spawn -----------------------

    #[test]
    fn cross_lane_collision_spawns_new_lane_above_target() {
        let mut p = sample_project();

        // V1 has Clip 1 at [10, 110). Drop Clip 4 (duration 90) from
        // V3 onto V1 at start=50 — [50, 140) overlaps [10, 110).
        let effect = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "3".into(),
                clip_id: "4".into(),
                target_track_id: "1".into(),
                new_start_value: 50,
            },
        )
        .unwrap();
        match effect {
            Effect::ClipTransferredToNewTrack {
                insert_at_index,
                new_track_kind,
                ..
            } => {
                assert_eq!(insert_at_index, 0); // V1 was at index 0.
                assert_eq!(new_track_kind, TrackKind::Video);
            }
            other => panic!("unexpected effect: {other:?}"),
        }
        assert_track_order_invariant(&p);
    }

    // ---- kind mismatch rejection ------------------------------------

    #[test]
    fn dropping_video_clip_on_audio_lane_is_rejected() {
        let mut p = sample_project();
        let err = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "1".into(),
                clip_id: "1".into(),
                target_track_id: "4".into(), // A1
                new_start_value: 0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, CommandError::TrackKindMismatch { .. }));
    }

    // ---- unknown-id error paths -------------------------------------

    #[test]
    fn move_clip_rejects_unknown_source_track() {
        let mut p = sample_project();
        let err = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "no-such".into(),
                clip_id: "1".into(),
                target_track_id: "1".into(),
                new_start_value: 0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, CommandError::UnknownTrack { ref track_id } if track_id == "no-such"));
    }

    #[test]
    fn move_clip_rejects_unknown_clip() {
        let mut p = sample_project();
        let err = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "1".into(),
                clip_id: "no-such".into(),
                target_track_id: "1".into(),
                new_start_value: 0,
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CommandError::UnknownClip { ref clip_id, .. } if clip_id == "no-such"
        ));
    }

    #[test]
    fn move_clip_rejects_unknown_target_track() {
        let mut p = sample_project();
        let err = apply(
            &mut p,
            &Command::MoveClip {
                source_track_id: "1".into(),
                clip_id: "1".into(),
                target_track_id: "no-such".into(),
                new_start_value: 0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, CommandError::UnknownTrack { ref track_id } if track_id == "no-such"));
    }

    // ---- wire format --------------------------------------------------

    #[test]
    fn command_serde_round_trips_through_json() {
        let cmd = Command::MoveClip {
            source_track_id: "trk".into(),
            clip_id: "clp".into(),
            target_track_id: "trk".into(),
            new_start_value: 7,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"move_clip","source_track_id":"trk","clip_id":"clp","target_track_id":"trk","new_start_value":7}"#
        );
        let parsed: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, cmd);
    }

    // ---- helpers -----------------------------------------------------

    fn assert_track_order_invariant(p: &Project) {
        let mut seen_audio = false;
        for id in &p.sequence.track_order {
            let kind = p.sequence.tracks.get(id).unwrap().kind;
            match kind {
                TrackKind::Video => assert!(
                    !seen_audio,
                    "video lane {id:?} appeared after an audio lane — invariant broken"
                ),
                TrackKind::Audio => seen_audio = true,
            }
        }
    }
}
