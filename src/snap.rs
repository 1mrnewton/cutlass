//! Drag snap targets — CapCut-style.
//!
//! While the user drags a clip, the gesture layer calls
//! [`compute_drag_snap`] every frame to convert the raw cursor-derived
//! `new_start_value` into either the same value (no snap) or the
//! nearest snap target inside the `snap_threshold_ticks` window.
//!
//! Snap targets are:
//!
//!   * sequence origin (`0`),
//!   * every other clip's start (`timeline_start.value`),
//!   * every other clip's end   (`timeline_start.value + source_range.duration.value`).
//!
//! Both edges of the dragged clip are considered: when the dragged
//! clip's *trailing* edge is within threshold of a candidate, we snap
//! by aligning that trailing edge to the candidate
//! (`snapped_start = candidate - duration`). This is what makes
//! "drop a clip flush against the end of another clip" feel instant.
//!
//! Hot path: one linear pass over all clips, no allocation. O(n)
//! where n is the total clip count — fine for the tens-of-thousands
//! regime that even very heavy timelines hit. We can switch to a
//! sorted-edges structure (bsearch over `~2n` edges) the moment
//! profiling shows this matters.

use crate::models::{Project, TrackKind};

/// Result of one snap probe. `has_snap == false` means the cursor was
/// outside every target's threshold; callers should use the original
/// `cursor_start_value` unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapResult {
    pub has_snap: bool,
    /// `new_start_value` to use after snapping. When `has_snap` is
    /// false, this echoes the input `cursor_start_value` so callers
    /// can write it unconditionally.
    pub snapped_start_value: i32,
    /// The timeline tick where the snap guide line should be drawn,
    /// in sequence ticks. Either the dragged clip's leading edge
    /// (`snapped_start_value`) or its trailing edge
    /// (`snapped_start_value + clip_duration`) depending on which
    /// edge actually snapped. Undefined when `has_snap == false`.
    pub snap_line_tick: i32,
}

impl SnapResult {
    #[inline]
    fn none(cursor: i32) -> Self {
        Self {
            has_snap: false,
            snapped_start_value: cursor,
            snap_line_tick: cursor,
        }
    }
}

/// Walk every clip on every lane and pick the snap target nearest to
/// the dragged clip's leading or trailing edge, within
/// `snap_threshold_ticks`. Ignores the clip identified by
/// (`dragging_source_track_id`, `dragging_clip_id`) so a dragging
/// clip never snaps to itself.
///
/// `cursor_start_value` is the dragged clip's *unsnapped* leading
/// edge in sequence ticks; `clip_duration_ticks` is its width. Both
/// are expressed at the clip's own rate, which (per the move-clip
/// command contract) is also the sequence rate today.
pub fn compute_drag_snap(
    project: &Project,
    dragging_source_track_id: &str,
    dragging_clip_id: &str,
    cursor_start_value: i32,
    clip_duration_ticks: i32,
    snap_threshold_ticks: i32,
) -> SnapResult {
    if snap_threshold_ticks <= 0 {
        return SnapResult::none(cursor_start_value);
    }

    let cursor_end = cursor_start_value.saturating_add(clip_duration_ticks);

    // Best candidate so far. Stored as
    //   (abs distance, snapped_start_value, snap_line_tick)
    // so the inner loop can compare distances and only commit when
    // it strictly improves.
    let mut best: Option<(i32, i32, i32)> = None;

    let mut consider = |candidate: i32| {
        // Snap the LEADING edge to `candidate`.
        let d_leading = (candidate - cursor_start_value).abs();
        if d_leading <= snap_threshold_ticks
            && best.map_or(true, |(d, _, _)| d_leading < d)
        {
            best = Some((d_leading, candidate, candidate));
        }
        // Snap the TRAILING edge to `candidate` — i.e. align the
        // dragged clip's right edge with `candidate`.
        let d_trailing = (candidate - cursor_end).abs();
        if d_trailing <= snap_threshold_ticks
            && best.map_or(true, |(d, _, _)| d_trailing < d)
        {
            let snapped_start = candidate.saturating_sub(clip_duration_ticks);
            best = Some((d_trailing, snapped_start, candidate));
        }
    };

    // Sequence origin.
    consider(0);

    // Every clip edge, except the dragging clip itself.
    for (track_id, track) in &project.sequence.tracks {
        for (clip_id, clip) in &track.clips {
            if track_id == dragging_source_track_id && clip_id == dragging_clip_id {
                continue;
            }
            let s = clip.timeline_start.value;
            let e = s.saturating_add(clip.source_range.duration.value);
            consider(s);
            consider(e);
        }
    }

    match best {
        None => SnapResult::none(cursor_start_value),
        Some((_, snapped_start, line)) => SnapResult {
            has_snap: true,
            snapped_start_value: snapped_start,
            snap_line_tick: line,
        },
    }
}

/// Resolved drop target for an in-flight clip drag. Returned by
/// [`resolve_drag_target`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTarget {
    /// Id of the lane the cursor currently sits over, after clamping
    /// to the source kind's contiguous block. Empty string when the
    /// source lane is unknown — callers should treat that as "no
    /// target" and fall through to the source lane on release.
    pub track_id: String,
    /// Signed lane offset from source to resolved target. The
    /// gesture layer uses this to clamp the dragged clip's visual y
    /// to valid space (so dragging a video clip past the
    /// video / audio boundary doesn't visually drift into audio land).
    pub clamped_offset: i32,
}

/// Map "source lane + vertical lane offset" to a [`ResolvedTarget`],
/// clamped to the source kind's contiguous block of lanes so
/// cross-kind drops are impossible at the gesture level (the command
/// layer rejects them defensively too, but filtering here means the
/// lane highlight never lights up an audio lane while dragging a
/// video clip).
///
/// `lane_offset` is signed: negative means "above the source lane in
/// iteration order" (smaller index), positive means below. A drag
/// that stays on the source lane resolves to the source lane itself
/// with `clamped_offset == 0`.
pub fn resolve_drag_target(
    project: &Project,
    source_track_id: &str,
    lane_offset: i32,
) -> ResolvedTarget {
    let order = &project.sequence.track_order;
    let Some(source_idx) = order.iter().position(|id| id == source_track_id) else {
        return ResolvedTarget {
            track_id: String::new(),
            clamped_offset: 0,
        };
    };
    let Some(source_track) = project.sequence.tracks.get(source_track_id) else {
        return ResolvedTarget {
            track_id: String::new(),
            clamped_offset: 0,
        };
    };
    let source_kind = source_track.kind;

    // Walk `track_order` once to find the contiguous [first, last]
    // index range of `source_kind`. Cheap: two hashmap lookups per
    // lane, and lane counts are tiny (tens).
    let mut first: i32 = source_idx as i32;
    let mut last: i32 = source_idx as i32;
    for (i, id) in order.iter().enumerate() {
        let i_signed = i as i32;
        let kind = project
            .sequence
            .tracks
            .get(id)
            .map(|t| t.kind)
            .unwrap_or(TrackKind::Video);
        if kind == source_kind {
            if i_signed < first {
                first = i_signed;
            }
            if i_signed > last {
                last = i_signed;
            }
        }
    }

    let raw = (source_idx as i32).saturating_add(lane_offset);
    let clamped_idx = raw.clamp(first, last);
    let clamped_offset = clamped_idx - source_idx as i32;
    ResolvedTarget {
        track_id: order
            .get(clamped_idx as usize)
            .cloned()
            .unwrap_or_default(),
        clamped_offset,
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::*;
    use crate::models::sample_project;

    #[test]
    fn zero_offset_returns_source_lane() {
        let p = sample_project();
        let r = resolve_drag_target(&p, "2", 0);
        assert_eq!(r.track_id, "2");
        assert_eq!(r.clamped_offset, 0);
    }

    #[test]
    fn positive_offset_moves_down_within_video_block() {
        let p = sample_project();
        // V1 (idx 0) → +1 lane = V2 (idx 1).
        let r = resolve_drag_target(&p, "1", 1);
        assert_eq!(r.track_id, "2");
        assert_eq!(r.clamped_offset, 1);
    }

    #[test]
    fn negative_offset_moves_up_within_video_block() {
        let p = sample_project();
        // V3 (idx 2) → -2 lanes = V1 (idx 0).
        let r = resolve_drag_target(&p, "3", -2);
        assert_eq!(r.track_id, "1");
        assert_eq!(r.clamped_offset, -2);
    }

    #[test]
    fn offset_past_video_block_clamps_to_last_video_lane() {
        let p = sample_project();
        // Sample has 3 video lanes (V1, V2, V3 at indices 0..2) and 2
        // audio lanes (A1, A2). Dragging V1 down by +5 would naively
        // land on an audio lane — must clamp to V3.
        let r = resolve_drag_target(&p, "1", 5);
        assert_eq!(r.track_id, "3");
        // Source V1 = 0, V3 = 2, clamped offset is 2 not the raw 5.
        assert_eq!(r.clamped_offset, 2);
    }

    #[test]
    fn offset_past_audio_block_clamps_to_first_audio_lane() {
        let p = sample_project();
        // A1 (idx 3) → -10 must clamp to A1 itself (A1 is the first
        // audio lane; can't go higher and stay audio).
        let r = resolve_drag_target(&p, "4", -10);
        assert_eq!(r.track_id, "4");
        assert_eq!(r.clamped_offset, 0);
    }

    #[test]
    fn unknown_source_returns_empty() {
        let p = sample_project();
        let r = resolve_drag_target(&p, "no-such", 0);
        assert_eq!(r.track_id, "");
        assert_eq!(r.clamped_offset, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::sample_project;

    #[test]
    fn snaps_to_zero_origin_when_dragging_near_start() {
        let p = sample_project();
        let r = compute_drag_snap(&p, "1", "1", 3, 100, 5);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 0);
        assert_eq!(r.snap_line_tick, 0);
    }

    #[test]
    fn snaps_to_neighbour_start_within_threshold() {
        let p = sample_project();
        // V2 Clip 3 starts at 120. Drag Clip 1 (duration 100) so its
        // leading edge is at 117 — within 5 ticks of 120 → snap.
        let r = compute_drag_snap(&p, "1", "1", 117, 100, 5);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 120);
        assert_eq!(r.snap_line_tick, 120);
    }

    #[test]
    fn snaps_trailing_edge_to_neighbour_start() {
        let p = sample_project();
        // V2 Clip 3 starts at 120. Drag Clip 1 (duration 100) so its
        // *trailing* edge lands near 120 → leading edge at 22.
        let r = compute_drag_snap(&p, "1", "1", 22, 100, 5);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 20); // 120 - 100
        assert_eq!(r.snap_line_tick, 120);
    }

    #[test]
    fn no_snap_outside_threshold() {
        // Edges across the sample project at threshold=5 around the
        // cursor value: 0, 10, 50, 80, 110, 120, 140, 150, 180, 195, 200.
        // 300 is at least 95 ticks away from every edge — well outside
        // any reasonable threshold.
        let p = sample_project();
        let r = compute_drag_snap(&p, "1", "1", 300, 50, 5);
        assert!(!r.has_snap, "expected no snap, got {r:?}");
        assert_eq!(r.snapped_start_value, 300);
    }

    #[test]
    fn dragged_clip_does_not_snap_to_itself() {
        let p = sample_project();
        // Clip 1 lives at 10 on V1. If self-snap leaked, cursor=12
        // would snap back to 10 instead of falling through.
        let r = compute_drag_snap(&p, "1", "1", 12, 100, 5);
        // Could legitimately snap to "0" (distance 12 > threshold 5)
        // or some other edge. Importantly: NOT to 10.
        if r.has_snap {
            assert_ne!(r.snapped_start_value, 10);
        }
    }

    #[test]
    fn picks_closer_of_two_candidates_in_range() {
        let p = sample_project();
        // V2 has Clip 2 ending at 80 and Clip 3 starting at 120. Drag
        // Clip 1 (duration 100) with leading edge at 78 — distance 2
        // to "80" via leading-edge snap, distance 22 to "120" via
        // trailing-edge snap. Should pick 80.
        let r = compute_drag_snap(&p, "1", "1", 78, 100, 5);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 80);
    }

    #[test]
    fn zero_threshold_disables_snap() {
        let p = sample_project();
        let r = compute_drag_snap(&p, "1", "1", 0, 100, 0);
        assert!(!r.has_snap);
        assert_eq!(r.snapped_start_value, 0);
    }
}
