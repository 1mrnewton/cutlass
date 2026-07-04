//! Multi-selection and group-drag helpers operating on the Slint view model.
//!
//! Selection is a set of clip ids (engine clip ids are globally unique) plus
//! a primary anchor used by the inspector and single-clip ops. The helpers
//! here resolve selection gestures (click, shift-click, marquee) — expanding
//! link groups while the Link toggle is on — and the group drag: one uniform
//! (dx, row delta) for the whole set, validated against everything outside
//! it (CapCut's reject policy: no valid landing ⇒ release commits nothing).

use slint::{Model, ModelRc, SharedString, VecModel};
use std::rc::Rc;

use crate::{ClipMove, GroupDragResolution, GroupGhost, SelectionUpdate, Sequence, TrackKind};

/// Flat view of one placed clip, with its lane row (top-first) and lane
/// metadata — gathered once per resolution, O(total clips).
struct PlacedClip {
    row: i32,
    track_id: SharedString,
    kind: TrackKind,
    locked: bool,
    color: slint::Color,
    clip_id: SharedString,
    link_id: SharedString,
    start: i32,
    duration: i32,
}

impl PlacedClip {
    fn end(&self) -> i32 {
        self.start.saturating_add(self.duration)
    }
}

fn placed_clips(sequence: &Sequence) -> Vec<PlacedClip> {
    let mut clips = Vec::new();
    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        for idx in 0..track.clips.row_count() {
            let Some(clip) = track.clips.row_data(idx) else {
                continue;
            };
            clips.push(PlacedClip {
                row: row as i32,
                track_id: track.id.clone(),
                kind: track.kind,
                locked: track.locked,
                color: track.color,
                clip_id: clip.id.clone(),
                link_id: clip.link_id.clone(),
                start: clip.timeline_start.value,
                duration: clip.source_range.duration.value,
            });
        }
    }
    clips
}

fn ids_model(ids: Vec<SharedString>) -> ModelRc<SharedString> {
    ModelRc::from(Rc::new(VecModel::from(ids)))
}

/// Membership test for the selection set (drives per-clip highlights).
pub fn selection_contains(ids: &ModelRc<SharedString>, id: &str) -> bool {
    (0..ids.row_count()).any(|i| ids.row_data(i).is_some_and(|v| v == id))
}

/// The clip plus — while linkage is on — every clip sharing its link group,
/// ordered (row, start) so visuals and primaries are deterministic.
fn link_group(clips: &[PlacedClip], clip_id: &str, link_enabled: bool) -> Vec<SharedString> {
    let link = clips
        .iter()
        .find(|c| c.clip_id == clip_id)
        .map(|c| c.link_id.clone())
        .unwrap_or_default();
    if !link_enabled || link.is_empty() {
        return vec![SharedString::from(clip_id)];
    }
    let mut members: Vec<&PlacedClip> = clips.iter().filter(|c| c.link_id == link).collect();
    members.sort_by_key(|c| (c.row, c.start));
    members.iter().map(|c| c.clip_id.clone()).collect()
}

/// Plain click: select the clip (and its link group).
pub fn select_clip(
    sequence: &Sequence,
    track_id: &str,
    clip_id: &str,
    link_enabled: bool,
) -> SelectionUpdate {
    let clips = placed_clips(sequence);
    SelectionUpdate {
        ids: ids_model(link_group(&clips, clip_id, link_enabled)),
        primary_track_id: track_id.into(),
        primary_clip_id: clip_id.into(),
    }
}

/// Shift-click: toggle the clip's link group in the selection. Deselecting
/// the primary re-anchors it on the first remaining member (row/start order).
pub fn toggle_clip(
    sequence: &Sequence,
    current: &ModelRc<SharedString>,
    track_id: &str,
    clip_id: &str,
    link_enabled: bool,
) -> SelectionUpdate {
    let clips = placed_clips(sequence);
    let group = link_group(&clips, clip_id, link_enabled);
    let was_selected = selection_contains(current, clip_id);

    let mut ids: Vec<SharedString> = (0..current.row_count())
        .filter_map(|i| current.row_data(i))
        .filter(|id| !group.iter().any(|g| g == id))
        .collect();
    if !was_selected {
        ids.extend(group.iter().cloned());
    }

    let (primary_track, primary_clip) = if !was_selected {
        (SharedString::from(track_id), SharedString::from(clip_id))
    } else {
        // First remaining selected clip in (row, start) order, if any.
        clips
            .iter()
            .filter(|c| ids.contains(&c.clip_id))
            .min_by_key(|c| (c.row, c.start))
            .map(|c| (c.track_id.clone(), c.clip_id.clone()))
            .unwrap_or_default()
    };

    SelectionUpdate {
        ids: ids_model(ids),
        primary_track_id: primary_track,
        primary_clip_id: primary_clip,
    }
}

/// Marquee: every clip on an unlocked lane intersecting the rect
/// `[tick0, tick1] × [row0, row1]` (rows in fractional lane units,
/// top-first), expanded to whole link groups while linkage is on.
pub fn resolve_marquee(
    sequence: &Sequence,
    tick0: i32,
    tick1: i32,
    row0: f32,
    row1: f32,
    link_enabled: bool,
) -> SelectionUpdate {
    let clips = placed_clips(sequence);
    let hit = |c: &PlacedClip| {
        !c.locked
            && (c.row as f32) < row1
            && (c.row as f32 + 1.0) > row0
            && c.start < tick1
            && c.end() > tick0
    };

    let mut ids: Vec<SharedString> = Vec::new();
    let mut push_unique = |id: SharedString| {
        if !ids.contains(&id) {
            ids.push(id);
        }
    };
    let mut ordered: Vec<&PlacedClip> = clips.iter().filter(|c| hit(c)).collect();
    ordered.sort_by_key(|c| (c.row, c.start));
    let primary = ordered
        .first()
        .map(|c| (c.track_id.clone(), c.clip_id.clone()))
        .unwrap_or_default();
    for clip in ordered {
        for member in link_group(&clips, clip.clip_id.as_str(), link_enabled) {
            push_unique(member);
        }
    }

    SelectionUpdate {
        ids: ids_model(ids),
        primary_track_id: primary.0,
        primary_clip_id: primary.1,
    }
}

/// Whether any selected clip carries a link-group id — gates the toolbar's
/// Unlink button.
pub fn selection_has_link(sequence: &Sequence, ids: &ModelRc<SharedString>) -> bool {
    placed_clips(sequence)
        .iter()
        .any(|c| !c.link_id.is_empty() && selection_contains(ids, c.clip_id.as_str()))
}

/// Reconcile the selection with a freshly published projection (undo/redo,
/// agent edits — the tracked debt from the timeline roadmap): ids whose
/// clips no longer exist are dropped (set order preserved), and the primary
/// anchor follows its clip's current lane — or re-anchors on the first
/// surviving member in (row, start) order when its own clip vanished.
pub fn prune_selection(
    sequence: &Sequence,
    ids: &ModelRc<SharedString>,
    primary_clip_id: &str,
) -> SelectionUpdate {
    let clips = placed_clips(sequence);
    let kept: Vec<SharedString> = (0..ids.row_count())
        .filter_map(|i| ids.row_data(i))
        .filter(|id| clips.iter().any(|c| c.clip_id == id))
        .collect();

    let anchor = |c: &PlacedClip| (c.track_id.clone(), c.clip_id.clone());
    let primary = clips
        .iter()
        .find(|c| c.clip_id == primary_clip_id && kept.contains(&c.clip_id))
        .map(anchor)
        .or_else(|| {
            clips
                .iter()
                .filter(|c| kept.contains(&c.clip_id))
                .min_by_key(|c| (c.row, c.start))
                .map(anchor)
        })
        .unwrap_or_default();

    SelectionUpdate {
        ids: ids_model(kept),
        primary_track_id: primary.0,
        primary_clip_id: primary.1,
    }
}

/// Press-time rectangles of the selected clips, for the floating copies that
/// follow the cursor during a group drag.
pub fn group_floaters(sequence: &Sequence, ids: &ModelRc<SharedString>) -> ModelRc<GroupGhost> {
    let clips = placed_clips(sequence);
    let mut selected: Vec<&PlacedClip> = clips
        .iter()
        .filter(|c| selection_contains(ids, c.clip_id.as_str()))
        .collect();
    selected.sort_by_key(|c| (c.row, c.start));
    let ghosts: Vec<GroupGhost> = selected
        .iter()
        .map(|c| GroupGhost {
            row: c.row,
            start_tick: c.start,
            duration_ticks: c.duration,
            color: c.color,
        })
        .collect();
    ModelRc::from(Rc::new(VecModel::from(ghosts)))
}

/// Resolve a multi-selection drag: one uniform dx (clamped to tick 0,
/// magneted on the grabbed clip's edges) and one row delta (the hovered row
/// minus the grabbed clip's row; forced to 0 when the set spans multiple
/// lane kinds — audio partners stay in the audio zone, CapCut-style).
///
/// Every member must land on an unlocked lane of its own kind with no
/// overlap against clips outside the set (members can't collide with each
/// other: a uniform shift preserves their relative placement). If the
/// snapped or row-shifted variants don't fit, progressively weaker variants
/// are tried (raw dx, then horizontal-only); when nothing fits the
/// resolution is invalid and release commits nothing.
#[allow(clippy::too_many_arguments)]
pub fn resolve_group_drag(
    sequence: &Sequence,
    ids: &ModelRc<SharedString>,
    anchor_track_id: &str,
    anchor_clip_id: &str,
    dx_ticks: i32,
    hover_row: i32,
    playhead_tick: i32,
    snap_threshold_ticks: i32,
) -> GroupDragResolution {
    let clips = placed_clips(sequence);
    let track_count = sequence.tracks.row_count() as i32;

    let selected: Vec<&PlacedClip> = clips
        .iter()
        .filter(|c| selection_contains(ids, c.clip_id.as_str()))
        .collect();
    let Some(anchor) = selected
        .iter()
        .find(|c| c.track_id == anchor_track_id && c.clip_id == anchor_clip_id)
    else {
        return invalid_resolution();
    };

    let uniform_kind = selected.iter().all(|c| c.kind == selected[0].kind);
    let row_delta = if uniform_kind {
        hover_row - anchor.row
    } else {
        0
    };

    // The set can't cross tick 0.
    let min_start = selected.iter().map(|c| c.start).min().unwrap_or(0);
    let dx_min = -min_start;
    let dx_raw = dx_ticks.max(dx_min);

    // Magnet on the grabbed clip's edges against everything outside the set,
    // the playhead, and tick 0 (same candidates as single drags).
    let snap = snap_dx(
        &clips,
        ids,
        anchor.start,
        anchor.end(),
        dx_raw,
        snap_threshold_ticks,
        playhead_tick,
    );
    let (dx_snapped, snap_line) = match snap {
        Some((dx, line)) if dx >= dx_min => (Some(dx), line),
        _ => (None, 0),
    };

    // Preference order: snapped at the hovered rows, raw at the hovered
    // rows, then horizontal-only (vertical intent yields before the magnet).
    let mut candidates: Vec<(i32, i32, bool)> = Vec::with_capacity(4);
    if let Some(dx) = dx_snapped {
        candidates.push((dx, row_delta, true));
    }
    candidates.push((dx_raw, row_delta, false));
    if row_delta != 0 {
        if let Some(dx) = dx_snapped {
            candidates.push((dx, 0, true));
        }
        candidates.push((dx_raw, 0, false));
    }

    for (dx, rd, snapped) in candidates {
        if !placement_fits(&clips, ids, &selected, sequence, track_count, dx, rd) {
            continue;
        }
        let mut landed: Vec<&PlacedClip> = selected.clone();
        landed.sort_by_key(|c| (c.row + rd, c.start));
        let moves: Vec<ClipMove> = landed
            .iter()
            .map(|c| ClipMove {
                clip_id: c.clip_id.clone(),
                track_id: track_id_at(sequence, c.row + rd),
                start_tick: c.start + dx,
            })
            .collect();
        let ghosts: Vec<GroupGhost> = landed
            .iter()
            .map(|c| GroupGhost {
                row: c.row + rd,
                start_tick: c.start + dx,
                duration_ticks: c.duration,
                color: track_color_at(sequence, c.row + rd).unwrap_or(c.color),
            })
            .collect();
        return GroupDragResolution {
            valid: true,
            is_noop: dx == 0 && rd == 0,
            resolved_dx_ticks: dx,
            row_delta: rd,
            has_snap: snapped,
            snap_line_tick: if snapped { snap_line } else { 0 },
            moves: ModelRc::from(Rc::new(VecModel::from(moves))),
            ghosts: ModelRc::from(Rc::new(VecModel::from(ghosts))),
        };
    }

    invalid_resolution()
}

/// Whether every selected clip, shifted by (`dx`, `rd`), lands on an
/// unlocked lane of its kind without overlapping any clip outside the set.
fn placement_fits(
    clips: &[PlacedClip],
    ids: &ModelRc<SharedString>,
    selected: &[&PlacedClip],
    sequence: &Sequence,
    track_count: i32,
    dx: i32,
    rd: i32,
) -> bool {
    for member in selected {
        let target_row = member.row + rd;
        if !(0..track_count).contains(&target_row) {
            return false;
        }
        let Some(track) = sequence.tracks.row_data(target_row as usize) else {
            return false;
        };
        if track.kind != member.kind || track.locked {
            return false;
        }
        let start = member.start + dx;
        let end = start.saturating_add(member.duration);
        let collides = clips.iter().any(|other| {
            other.row == target_row
                && !selection_contains(ids, other.clip_id.as_str())
                && start < other.end()
                && other.start < end
        });
        if collides {
            return false;
        }
    }
    true
}

/// Best magnet candidate for the anchor's shifted edges: clip edges outside
/// the selection, the playhead, and tick 0. Returns `(snapped dx, guide
/// tick)` when a candidate sits within the threshold.
fn snap_dx(
    clips: &[PlacedClip],
    ids: &ModelRc<SharedString>,
    anchor_start: i32,
    anchor_end: i32,
    dx: i32,
    threshold: i32,
    playhead_tick: i32,
) -> Option<(i32, i32)> {
    if threshold <= 0 {
        return None;
    }
    let lead = anchor_start.saturating_add(dx);
    let trail = anchor_end.saturating_add(dx);
    let mut best: Option<(i32, i32, i32)> = None; // (distance, dx, line)

    let mut consider = |candidate: i32| {
        let d_lead = (candidate - lead).abs();
        if d_lead <= threshold && best.is_none_or(|(d, _, _)| d_lead < d) {
            best = Some((d_lead, dx + (candidate - lead), candidate));
        }
        let d_trail = (candidate - trail).abs();
        if d_trail <= threshold && best.is_none_or(|(d, _, _)| d_trail < d) {
            best = Some((d_trail, dx + (candidate - trail), candidate));
        }
    };

    consider(0);
    consider(playhead_tick);
    for clip in clips {
        if selection_contains(ids, clip.clip_id.as_str()) {
            continue;
        }
        consider(clip.start);
        consider(clip.end());
    }

    best.map(|(_, snapped_dx, line)| (snapped_dx, line))
}

fn track_id_at(sequence: &Sequence, row: i32) -> SharedString {
    sequence
        .tracks
        .row_data(row.max(0) as usize)
        .map(|t| t.id.clone())
        .unwrap_or_default()
}

fn track_color_at(sequence: &Sequence, row: i32) -> Option<slint::Color> {
    sequence
        .tracks
        .row_data(row.max(0) as usize)
        .map(|t| t.color)
}

fn invalid_resolution() -> GroupDragResolution {
    GroupDragResolution {
        valid: false,
        is_noop: true,
        resolved_dx_ticks: 0,
        row_delta: 0,
        has_snap: false,
        snap_line_tick: 0,
        moves: ModelRc::from(Rc::new(VecModel::<ClipMove>::default())),
        ghosts: ModelRc::from(Rc::new(VecModel::<GroupGhost>::default())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Clip, Rational, RationalTime, TimeRange, Track};

    fn rt(value: i32) -> RationalTime {
        RationalTime {
            value,
            rate: Rational { num: 24, den: 1 },
        }
    }

    fn clip(id: &str, start: i32, dur: i32) -> Clip {
        linked_clip(id, start, dur, "")
    }

    fn linked_clip(id: &str, start: i32, dur: i32, link: &str) -> Clip {
        Clip {
            id: SharedString::from(id),
            name: SharedString::from(id),
            timeline_start: rt(start),
            source_range: TimeRange {
                start: rt(0),
                duration: rt(dur),
            },
            link_id: SharedString::from(link),
            ..Default::default()
        }
    }

    fn track(id: &str, kind: TrackKind, clips: Vec<Clip>) -> Track {
        Track {
            id: SharedString::from(id),
            name: SharedString::from(id),
            kind,
            color: slint::Color::from_rgb_u8(0x4A, 0x6F, 0xA5),
            clips: ModelRc::from(Rc::new(VecModel::from(clips))),
            enabled: true,
            muted: false,
            locked: false,
            duck_source: false,
            transitions: ModelRc::default(),
        }
    }

    fn sequence(tracks: Vec<Track>) -> Sequence {
        Sequence {
            id: SharedString::from("1"),
            name: SharedString::from("Sequence 1"),
            fps: Rational { num: 24, den: 1 },
            drop_frame: false,
            tracks: ModelRc::from(Rc::new(VecModel::from(tracks))),
            markers: Default::default(),
            width: 1920.0,
            height: 1080.0,
            aspect_index: 0,
            background: Default::default(),
        }
    }

    fn ids(values: &[&str]) -> ModelRc<SharedString> {
        ids_model(values.iter().map(|v| SharedString::from(*v)).collect())
    }

    fn ids_vec(model: &ModelRc<SharedString>) -> Vec<String> {
        (0..model.row_count())
            .filter_map(|i| model.row_data(i))
            .map(|s| s.to_string())
            .collect()
    }

    /// Rows: 0 = V2 (video, clips A [0,50) B [100,40)), 1 = V1 (video,
    /// C [0,80)), 2 = A1 (audio, D [0,50)). A and D share link "L".
    fn sample() -> Sequence {
        sequence(vec![
            track(
                "2",
                TrackKind::Video,
                vec![linked_clip("A", 0, 50, "L"), clip("B", 100, 40)],
            ),
            track("1", TrackKind::Video, vec![clip("C", 0, 80)]),
            track("9", TrackKind::Audio, vec![linked_clip("D", 0, 50, "L")]),
        ])
    }

    // --- selection ----------------------------------------------------------

    #[test]
    fn contains_finds_member() {
        let set = ids(&["A", "D"]);
        assert!(selection_contains(&set, "A"));
        assert!(!selection_contains(&set, "B"));
    }

    #[test]
    fn select_expands_link_group_when_linkage_on() {
        let seq = sample();
        let upd = select_clip(&seq, "2", "A", true);
        assert_eq!(ids_vec(&upd.ids), vec!["A", "D"]);
        assert_eq!(upd.primary_clip_id, "A");
        assert_eq!(upd.primary_track_id, "2");

        let off = select_clip(&seq, "2", "A", false);
        assert_eq!(ids_vec(&off.ids), vec!["A"]);
    }

    #[test]
    fn toggle_adds_then_removes_link_group() {
        let seq = sample();
        let added = toggle_clip(&seq, &ids(&["B"]), "2", "A", true);
        assert_eq!(ids_vec(&added.ids), vec!["B", "A", "D"]);
        assert_eq!(added.primary_clip_id, "A");

        let removed = toggle_clip(&seq, &added.ids, "2", "A", true);
        assert_eq!(ids_vec(&removed.ids), vec!["B"]);
        // Primary re-anchors on the first remaining clip.
        assert_eq!(removed.primary_clip_id, "B");
        assert_eq!(removed.primary_track_id, "2");
    }

    #[test]
    fn toggle_last_clip_clears_selection() {
        let seq = sample();
        let upd = toggle_clip(&seq, &ids(&["B"]), "2", "B", true);
        assert!(ids_vec(&upd.ids).is_empty());
        assert_eq!(upd.primary_clip_id, "");
    }

    #[test]
    fn marquee_selects_intersecting_clips_and_links() {
        let seq = sample();
        // Rect over rows 0..1.5, ticks 10..60: hits A (row 0) and C (row 1);
        // A pulls in its linked partner D (group joins right after A).
        let upd = resolve_marquee(&seq, 10, 60, -0.2, 1.5, true);
        assert_eq!(ids_vec(&upd.ids), vec!["A", "D", "C"]);
        assert_eq!(upd.primary_clip_id, "A");

        // Linkage off: only the intersecting clips.
        let off = resolve_marquee(&seq, 10, 60, -0.2, 1.5, false);
        assert_eq!(ids_vec(&off.ids), vec!["A", "C"]);
    }

    #[test]
    fn marquee_skips_locked_lanes_and_misses() {
        let mut locked = track("2", TrackKind::Video, vec![clip("A", 0, 50)]);
        locked.locked = true;
        let seq = sequence(vec![
            locked,
            track("1", TrackKind::Video, vec![clip("C", 0, 80)]),
        ]);
        let upd = resolve_marquee(&seq, 0, 100, -0.5, 2.0, true);
        assert_eq!(ids_vec(&upd.ids), vec!["C"]);

        let miss = resolve_marquee(&seq, 200, 300, -0.5, 2.0, true);
        assert!(ids_vec(&miss.ids).is_empty());
    }

    #[test]
    fn has_link_spots_linked_members_only() {
        let seq = sample();
        // A carries link "L"; B is unlinked.
        assert!(selection_has_link(&seq, &ids(&["A", "B"])));
        assert!(!selection_has_link(&seq, &ids(&["B"])));
        assert!(!selection_has_link(&seq, &ids(&[])));
    }

    // --- prune (projection republish reconciliation) -------------------------

    #[test]
    fn prune_keeps_selection_when_clips_survive() {
        let seq = sample();
        let upd = prune_selection(&seq, &ids(&["A", "D"]), "A");
        assert_eq!(ids_vec(&upd.ids), vec!["A", "D"]);
        assert_eq!(upd.primary_clip_id, "A");
        assert_eq!(upd.primary_track_id, "2");
    }

    #[test]
    fn prune_drops_vanished_ids_and_reanchors_primary() {
        let seq = sample();
        // "X" was deleted by the history step; it was also the primary, so
        // the anchor moves to the first surviving member in (row, start)
        // order — A on row 0.
        let upd = prune_selection(&seq, &ids(&["X", "D", "A"]), "X");
        assert_eq!(ids_vec(&upd.ids), vec!["D", "A"]);
        assert_eq!(upd.primary_clip_id, "A");
        assert_eq!(upd.primary_track_id, "2");
    }

    #[test]
    fn prune_clears_when_nothing_survives() {
        let seq = sample();
        let upd = prune_selection(&seq, &ids(&["X", "Y"]), "X");
        assert!(ids_vec(&upd.ids).is_empty());
        assert_eq!(upd.primary_clip_id, "");
        assert_eq!(upd.primary_track_id, "");
    }

    #[test]
    fn prune_remaps_primary_track_after_cross_lane_move() {
        // Undo of a cross-lane move: the clip id survives but now lives on a
        // different lane than the stored anchor track — follow the clip.
        let seq = sequence(vec![
            track("2", TrackKind::Video, vec![]),
            track("1", TrackKind::Video, vec![clip("A", 0, 50)]),
        ]);
        let upd = prune_selection(&seq, &ids(&["A"]), "A");
        assert_eq!(ids_vec(&upd.ids), vec!["A"]);
        assert_eq!(upd.primary_clip_id, "A");
        assert_eq!(upd.primary_track_id, "1");
    }

    // --- group drag -----------------------------------------------------------

    #[test]
    fn group_drag_moves_set_horizontally() {
        let seq = sample();
        // A (video row 0) + D (audio row 2): mixed kinds ⇒ row delta forced 0
        // even though the cursor wandered a row down.
        let r = resolve_group_drag(&seq, &ids(&["A", "D"]), "2", "A", 200, 1, 0, 0);
        assert!(r.valid && !r.is_noop);
        assert_eq!(r.resolved_dx_ticks, 200);
        assert_eq!(r.row_delta, 0);
        let moves = ids_moves(&r);
        assert!(moves.contains(&("A".into(), "2".into(), 200)));
        assert!(moves.contains(&("D".into(), "9".into(), 200)));
    }

    #[test]
    fn group_drag_clamps_set_to_tick_zero() {
        let seq = sample();
        // B starts at 100, A at 0 — dragging left by 150 clamps at dx = 0
        // (A pins the set).
        let r = resolve_group_drag(&seq, &ids(&["A", "B"]), "2", "B", -150, 0, 0, 0);
        assert!(r.valid);
        assert_eq!(r.resolved_dx_ticks, 0);
    }

    #[test]
    fn group_drag_conflict_rejects() {
        let seq = sample();
        // A+D dragged right by 90: A [90,140) would overlap B [100,140) on
        // its own lane — no fallback fits (audio partner pins the rows).
        let r = resolve_group_drag(&seq, &ids(&["A", "D"]), "2", "A", 90, 0, 0, 0);
        assert!(!r.valid);
    }

    #[test]
    fn group_drag_uniform_kind_changes_rows() {
        // Two stacked video lanes plus an empty third: dragging the pair on
        // rows 0+1 down one row lands on rows 1+2.
        let seq = sequence(vec![
            track("3", TrackKind::Video, vec![clip("A", 0, 50)]),
            track("2", TrackKind::Video, vec![clip("B", 0, 50)]),
            track("1", TrackKind::Video, vec![]),
        ]);
        let r = resolve_group_drag(&seq, &ids(&["A", "B"]), "3", "A", 0, 1, 0, 0);
        assert!(r.valid);
        assert_eq!(r.row_delta, 1);
        let moves = ids_moves(&r);
        assert!(moves.contains(&("A".into(), "2".into(), 0)));
        assert!(moves.contains(&("B".into(), "1".into(), 0)));
    }

    #[test]
    fn group_drag_falls_back_to_horizontal_when_rows_blocked() {
        // Set on the only video lane; hovering the audio row below can't
        // re-lane the set, so the move stays horizontal.
        let seq = sample();
        let r = resolve_group_drag(&seq, &ids(&["C"]), "1", "C", 200, 2, 0, 0);
        assert!(r.valid);
        assert_eq!(r.row_delta, 0);
        assert_eq!(r.resolved_dx_ticks, 200);
    }

    #[test]
    fn group_drag_snaps_anchor_edges() {
        let seq = sample();
        // Dragging A+D right so A's start sits at 137: B's start (100) is
        // out of reach but B's end (140) magnets A's start within 5 ticks…
        // 140 collides with nothing (A [140,190), B ends at 140).
        let r = resolve_group_drag(&seq, &ids(&["A", "D"]), "2", "A", 137, 0, 0, 5);
        assert!(r.valid);
        assert!(r.has_snap);
        assert_eq!(r.resolved_dx_ticks, 140);
        assert_eq!(r.snap_line_tick, 140);
    }

    #[test]
    fn group_drag_unmoved_is_noop() {
        let seq = sample();
        let r = resolve_group_drag(&seq, &ids(&["A", "D"]), "2", "A", 0, 0, 0, 0);
        assert!(r.valid && r.is_noop);
    }

    #[test]
    fn group_drag_unknown_anchor_is_invalid() {
        let seq = sample();
        let r = resolve_group_drag(&seq, &ids(&["A"]), "2", "404", 0, 0, 0, 0);
        assert!(!r.valid);
    }

    #[test]
    fn floaters_report_selected_rects() {
        let seq = sample();
        let floaters = group_floaters(&seq, &ids(&["A", "D"]));
        assert_eq!(floaters.row_count(), 2);
        let first = floaters.row_data(0).unwrap();
        assert_eq!(
            (first.row, first.start_tick, first.duration_ticks),
            (0, 0, 50)
        );
        let second = floaters.row_data(1).unwrap();
        assert_eq!(second.row, 2);
    }

    fn ids_moves(r: &GroupDragResolution) -> Vec<(String, String, i32)> {
        (0..r.moves.row_count())
            .filter_map(|i| r.moves.row_data(i))
            .map(|m| (m.clip_id.to_string(), m.track_id.to_string(), m.start_tick))
            .collect()
    }
}
