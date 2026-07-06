//! Drag placement helpers operating on the Slint view model.
//!
//! While the user drags (a clip, or a library tile), the gesture layer calls
//! these every frame; the same resolution is used for the live ghost and for
//! the commit on release, so the preview can never disagree with the drop.
//!
//! Policy follows CapCut:
//! - magnet snapping pulls the dragged edges to clip edges on *all* lanes,
//!   the playhead, and tick 0, with a vertical guide line at the snap point;
//! - lanes only accept their own kind; hovering a foreign-kind lane, empty
//!   space, or a conflicting (overlapping) span resolves to a *new lane*
//!   inserted at the hovered row;
//! - with the main-track magnet on, the bottom video lane stays gapless:
//!   hovering it resolves to an *insertion* between clips (caret instead of
//!   a free landing ghost); the commit shifts later clips right.

use slint::{Model, ModelRc, VecModel};
use std::rc::Rc;

use crate::{
    ClipDragResolution, ClipTrimResolution, GroupGhost, LibraryDropResolution, Sequence,
    SnapResult, TrackKind,
};

pub fn compute_drag_snap(
    sequence: &Sequence,
    dragging_source_track_id: &str,
    dragging_clip_id: &str,
    cursor_start_value: i32,
    clip_duration_ticks: i32,
    snap_threshold_ticks: i32,
    playhead_tick: i32,
) -> SnapResult {
    if snap_threshold_ticks <= 0 {
        return SnapResult::none(cursor_start_value);
    }

    let cursor_end = cursor_start_value.saturating_add(clip_duration_ticks);
    let mut best: Option<(i32, i32, i32)> = None;

    // A clip shorter than two thresholds has overlapping edge zones: both
    // edges pinning means a candidate (say the playhead) holds the clip
    // continuously across a span wider than the clip itself, with no free
    // placement anywhere near it — a tiny clip feels stuck. Snap such clips
    // by their leading edge only. (Duration 0 — the trim resolvers'
    // single-point snap — is unaffected: both edges coincide.)
    let consider_trailing = clip_duration_ticks >= 2 * snap_threshold_ticks;

    let mut consider = |candidate: i32| {
        let d_leading = (candidate - cursor_start_value).abs();
        if d_leading <= snap_threshold_ticks && best.is_none_or(|(d, _, _)| d_leading < d) {
            best = Some((d_leading, candidate, candidate));
        }
        if !consider_trailing {
            return;
        }
        let d_trailing = (candidate - cursor_end).abs();
        if d_trailing <= snap_threshold_ticks && best.is_none_or(|(d, _, _)| d_trailing < d) {
            let snapped_start = candidate.saturating_sub(clip_duration_ticks);
            best = Some((d_trailing, snapped_start, candidate));
        }
    };

    consider(0);
    consider(playhead_tick);

    for track_idx in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(track_idx) else {
            continue;
        };
        for clip_idx in 0..track.clips.row_count() {
            let Some(clip) = track.clips.row_data(clip_idx) else {
                continue;
            };
            if track.id == dragging_source_track_id && clip.id == dragging_clip_id {
                continue;
            }
            let s = clip.timeline_start.value;
            let e = s.saturating_add(clip.source_range.duration.value);
            consider(s);
            consider(e);
            // Beat markers (M8 Phase 6): the magnet pulls clip edges onto the
            // music's beats, already published as absolute sequence ticks.
            for beat_idx in 0..clip.beats.row_count() {
                if let Some(beat) = clip.beats.row_data(beat_idx) {
                    consider(beat);
                }
            }
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

/// Resolve where a clip being dragged by (`dx_ticks`, `hover_row`) would land.
///
/// `hover_row` is the lane-list row under the dragged clip's vertical center
/// (top-first; may be out of range, meaning above/below the existing lanes).
/// With `main_magnet` on, hovering the main lane (bottom video lane) with a
/// video clip resolves to an insertion between clips instead of a free spot.
/// O(total clips) per call — evaluated per drag frame, fine at editing scale.
#[allow(clippy::too_many_arguments)]
pub fn resolve_clip_drag(
    sequence: &Sequence,
    source_track_id: &str,
    dragging_clip_id: &str,
    dx_ticks: i32,
    hover_row: i32,
    playhead_tick: i32,
    snap_threshold_ticks: i32,
    main_magnet: bool,
) -> ClipDragResolution {
    let track_count = sequence.tracks.row_count() as i32;
    let Some((source_kind, orig_start, duration)) =
        find_dragged_clip(sequence, source_track_id, dragging_clip_id)
    else {
        return ClipDragResolution::invalid();
    };

    let desired = (orig_start.saturating_add(dx_ticks)).max(0);

    // Main-track magnet: the bottom video lane is gapless, so a video clip
    // hovering it lands *between* clips (the commit shifts later clips right
    // to open the hole). The edge magnet is irrelevant here — position is
    // quantized to clip boundaries, picked by the unsnapped left edge.
    if main_magnet
        && source_kind == TrackKind::Video
        && main_video_row(sequence) == Some(hover_row)
        && let Some(track) = sequence.tracks.row_data(hover_row as usize)
        && !track.locked
    {
        let exclude = (track.id == source_track_id).then_some(dragging_clip_id);
        let ins = resolve_insertion(&track, exclude, desired);
        return ClipDragResolution {
            valid: true,
            is_new_lane: false,
            target_track_id: track.id.clone(),
            target_row: hover_row,
            resolved_start: ins.commit_tick,
            duration_ticks: duration,
            has_snap: false,
            snap_line_tick: 0,
            is_noop: ins.noop,
            is_insert: true,
            caret_tick: ins.display_tick,
        };
    }

    let snap = compute_drag_snap(
        sequence,
        source_track_id,
        dragging_clip_id,
        desired,
        duration,
        snap_threshold_ticks,
        playhead_tick,
    );
    let snapped = snap.snapped_start_value.max(0);

    // Hovered lane accepts the clip only when kinds match and it isn't locked.
    let hover_track = (0..track_count)
        .contains(&hover_row)
        .then(|| sequence.tracks.row_data(hover_row as usize))
        .flatten()
        .filter(|t| t.kind == source_kind && !t.locked);

    if let Some(track) = hover_track {
        let exclude = (track.id == source_track_id).then_some(dragging_clip_id);
        if span_is_free(&track, snapped, duration, exclude) {
            return ClipDragResolution {
                valid: true,
                is_new_lane: false,
                target_track_id: track.id.clone(),
                target_row: hover_row,
                resolved_start: snapped,
                duration_ticks: duration,
                has_snap: snap.has_snap,
                snap_line_tick: snap.snap_line_tick,
                is_noop: track.id == source_track_id && snapped == orig_start,
                is_insert: false,
                caret_tick: 0,
            };
        }
        // The snap pulled us into a conflict the raw position doesn't have —
        // prefer landing free without the magnet over forcing a new lane.
        if snap.has_snap && snapped != desired && span_is_free(&track, desired, duration, exclude) {
            return ClipDragResolution {
                valid: true,
                is_new_lane: false,
                target_track_id: track.id.clone(),
                target_row: hover_row,
                resolved_start: desired,
                duration_ticks: duration,
                has_snap: false,
                snap_line_tick: 0,
                is_noop: track.id == source_track_id && desired == orig_start,
                is_insert: false,
                caret_tick: 0,
            };
        }
    }

    // Foreign kind, out of range, or conflicting span: a new lane is inserted
    // at the hovered row (clamped to just above / just below the stack).
    ClipDragResolution {
        valid: true,
        is_new_lane: true,
        target_track_id: Default::default(),
        target_row: hover_row.clamp(0, track_count),
        resolved_start: snapped,
        duration_ticks: duration,
        has_snap: snap.has_snap,
        snap_line_tick: snap.snap_line_tick,
        is_noop: false,
        is_insert: false,
        caret_tick: 0,
    }
}

/// Resolve where a library tile dropped at (`cursor_tick`, `drop_row`) lands.
///
/// Freeform: a `lane_kind` lane under the cursor (or empty ⇒ the worker
/// creates one at `drop_row`), position magneted to clip edges / playhead /
/// tick 0. Media tiles target video lanes; generated tiles (text titles,
/// solids, shapes) target their own kind. With `main_magnet` on, dropping a
/// *video* tile on the main lane resolves to an insertion between clips (the
/// worker ripple-inserts, shifting later clips right); generated lanes are
/// never the main track, so they always land freeform.
#[allow(clippy::too_many_arguments)]
pub fn resolve_library_drop(
    sequence: &Sequence,
    lane_kind: TrackKind,
    duration_ticks: i32,
    cursor_tick: i32,
    drop_row: i32,
    playhead_tick: i32,
    snap_threshold_ticks: i32,
    main_magnet: bool,
) -> LibraryDropResolution {
    let track_count = sequence.tracks.row_count() as i32;
    let row_track = (0..track_count)
        .contains(&drop_row)
        .then(|| sequence.tracks.row_data(drop_row as usize))
        .flatten()
        .filter(|t| t.kind == lane_kind && !t.locked);

    if lane_kind == TrackKind::Video
        && main_magnet
        && main_video_row(sequence) == Some(drop_row)
        && let Some(track) = &row_track
    {
        let ins = resolve_insertion(track, None, cursor_tick.max(0));
        return LibraryDropResolution {
            target_track_id: track.id.clone(),
            target_row: drop_row,
            resolved_start: ins.commit_tick,
            is_insert: true,
            caret_tick: ins.display_tick,
            has_snap: false,
            snap_line_tick: 0,
        };
    }

    // Magnet against clip edges, the playhead, and tick 0. Empty drag ids ⇒
    // every placed clip is a snap candidate.
    let snap = compute_drag_snap(
        sequence,
        "",
        "",
        cursor_tick,
        duration_ticks,
        snap_threshold_ticks,
        playhead_tick,
    );
    // A snap pulled below tick 0 is clamped away (no guide there).
    let resolved_start = snap.snapped_start_value.max(0);
    let has_snap = snap.has_snap && resolved_start == snap.snapped_start_value;
    LibraryDropResolution {
        target_track_id: row_track.map(|t| t.id.clone()).unwrap_or_default(),
        target_row: drop_row,
        resolved_start,
        is_insert: false,
        caret_tick: 0,
        has_snap,
        snap_line_tick: if has_snap { snap.snap_line_tick } else { 0 },
    }
}

/// Lane-list row of the main track: the *bottom* video lane. Rows are
/// top-first, so that's the last video row; `None` without any video lane.
fn main_video_row(sequence: &Sequence) -> Option<i32> {
    let mut main = None;
    for idx in 0..sequence.tracks.row_count() {
        if sequence
            .tracks
            .row_data(idx)
            .is_some_and(|t| t.kind == TrackKind::Video)
        {
            main = Some(idx as i32);
        }
    }
    main
}

/// An insertion slot on the (gapless) main lane.
struct Insertion {
    /// Caret position in the lane's *current* visual space.
    display_tick: i32,
    /// Commit position handed to the worker. For reorders (`exclude` set)
    /// this is in the post-close arrangement — the worker closes the dragged
    /// clip's own gap before opening the new hole, shifting every later
    /// boundary left by the clip's duration.
    commit_tick: i32,
    /// The slot is exactly where the excluded clip already sits.
    noop: bool,
}

/// Pick the insertion slot for content whose left edge sits at `desired`:
/// before the first clip whose midpoint lies right of it (CapCut feel —
/// crossing a clip's middle flips the caret to its other side), else after
/// the last clip.
fn resolve_insertion(track: &crate::Track, exclude: Option<&str>, desired: i32) -> Insertion {
    // (start, end) spans of every clip except the excluded one. The
    // projection publishes clips in start order, but sort defensively —
    // lanes hold tens of clips, this runs once per drag frame.
    let mut spans: Vec<(i32, i32)> = Vec::new();
    let mut excluded: Option<(i32, i32)> = None; // (start, duration)
    for idx in 0..track.clips.row_count() {
        let Some(clip) = track.clips.row_data(idx) else {
            continue;
        };
        let start = clip.timeline_start.value;
        let dur = clip.source_range.duration.value;
        if exclude.is_some_and(|id| clip.id == id) {
            excluded = Some((start, dur));
        } else {
            spans.push((start, start.saturating_add(dur)));
        }
    }
    spans.sort_unstable();

    let index = spans
        .iter()
        .position(|(s, e)| desired < s + (e - s) / 2)
        .unwrap_or(spans.len());

    let (noop, ex_start, ex_dur) = match excluded {
        Some((start, dur)) => {
            let orig_index = spans.iter().filter(|(s, _)| *s < start).count();
            (index == orig_index, start, dur)
        }
        None => (false, 0, 0),
    };

    let display_tick = if noop {
        ex_start
    } else if index < spans.len() {
        spans[index].0
    } else {
        // After the last clip: the lane's current content end (the excluded
        // clip can't be last here, or the slot would be its own ⇒ noop).
        spans.iter().map(|(_, e)| *e).max().unwrap_or(0)
    };

    // Reorders commit against the closed arrangement: every boundary right
    // of the excluded clip's old start moves left by its duration.
    let closed = |tick: i32, anchor: i32| -> i32 {
        if excluded.is_some() && anchor > ex_start {
            tick - ex_dur
        } else {
            tick
        }
    };
    let commit_tick = if index < spans.len() {
        closed(spans[index].0, spans[index].0)
    } else {
        spans.iter().map(|(s, e)| closed(*e, *s)).max().unwrap_or(0)
    };

    Insertion {
        display_tick,
        commit_tick,
        noop,
    }
}

/// Resolve an edge-trim drag of `dx_ticks` (`trim_head` ⇔ the left edge).
///
/// The dragged edge magnets to clip edges / the playhead / tick 0, then
/// clamps to (in CapCut order of feel): the neighboring clips on the lane,
/// the source media headroom (`head-room-ticks` / `tail-room-ticks` from the
/// projection), tick 0, and a 1-tick minimum duration. A snap that the clamp
/// rejects is dropped (no guide line) rather than shown lying.
///
/// With `link_enabled`, the same edge delta applies to every clip in the
/// dragged clip's link group on release, so the clamp is the *intersection*
/// of every member's allowed delta range (each member's own neighbors,
/// headroom, and 1-tick minimum). The partners' post-trim extents come back
/// as `ghosts` for their stretch previews.
///
/// With `main_magnet` on, a trim on the main lane (bottom video track) skips
/// neighbor clamping — downstream clips ripple on commit instead.
#[allow(clippy::too_many_arguments)]
pub fn resolve_clip_trim(
    sequence: &Sequence,
    track_id: &str,
    clip_id: &str,
    trim_head: bool,
    dx_ticks: i32,
    playhead_tick: i32,
    snap_threshold_ticks: i32,
    link_enabled: bool,
    main_magnet: bool,
) -> ClipTrimResolution {
    let Some(ctx) = trim_context(sequence, track_id, clip_id) else {
        return ClipTrimResolution::invalid();
    };
    let old_end = ctx.start.saturating_add(ctx.duration);
    let old_edge = if trim_head { ctx.start } else { old_end };

    let partners = linked_partners(sequence, clip_id, link_enabled);
    let ripple = main_magnet && is_main_track(sequence, track_id);

    // Allowed edge-delta range: the dragged clip's bounds intersected with
    // every linked partner's. Delta 0 is inside each member's range, so the
    // intersection is never empty.
    let (mut delta_lo, mut delta_hi) = edge_delta_bounds(&ctx, trim_head, ripple);
    for (row, _, partner) in &partners {
        let partner_ripple = main_magnet
            && sequence
                .tracks
                .row_data(*row as usize)
                .is_some_and(|t| is_main_track(sequence, t.id.as_str()));
        let (lo, hi) = edge_delta_bounds(partner, trim_head, partner_ripple);
        delta_lo = delta_lo.max(lo);
        delta_hi = delta_hi.min(hi);
    }

    // Magnet the dragged edge alone: duration 0 turns the (start, end) pair
    // snap into a single-point snap.
    let desired = old_edge.saturating_add(dx_ticks);
    let snap = compute_drag_snap(
        sequence,
        track_id,
        clip_id,
        desired,
        0,
        snap_threshold_ticks,
        playhead_tick,
    );

    let delta = (snap.snapped_start_value - old_edge).clamp(delta_lo, delta_hi);
    let edge = old_edge + delta;
    let has_snap = snap.has_snap && edge == snap.snapped_start_value;

    let (new_start, new_duration) = if trim_head {
        (edge, old_end - edge)
    } else {
        (ctx.start, edge - ctx.start)
    };

    let ghosts: Vec<GroupGhost> = partners
        .iter()
        .map(|(row, color, p)| GroupGhost {
            row: *row,
            start_tick: if trim_head { p.start + delta } else { p.start },
            duration_ticks: if trim_head {
                p.duration - delta
            } else {
                p.duration + delta
            },
            color: *color,
        })
        .collect();

    ClipTrimResolution {
        valid: true,
        new_start,
        new_duration,
        has_snap,
        snap_line_tick: if has_snap { snap.snap_line_tick } else { 0 },
        is_noop: new_start == ctx.start && new_duration == ctx.duration,
        ghosts: ModelRc::from(Rc::new(VecModel::from(ghosts))),
    }
}

/// Allowed range for the dragged edge's delta on one clip: its lane
/// neighbors, source headroom, tick 0, and the 1-tick minimum, expressed
/// relative to the edge's current position. Always contains 0. On the main
/// lane with the magnet on, neighbor clamps are skipped (ripple on commit).
fn edge_delta_bounds(ctx: &TrimContext, trim_head: bool, ripple: bool) -> (i32, i32) {
    let end = ctx.start.saturating_add(ctx.duration);
    if trim_head {
        let lo = if ripple {
            ctx.start.saturating_sub(ctx.head_room).max(0)
        } else {
            ctx.prev_end
                .max(ctx.start.saturating_sub(ctx.head_room))
                .max(0)
        };
        (lo - ctx.start, (end - 1) - ctx.start)
    } else {
        let hi = if ripple {
            end.saturating_add(ctx.tail_room)
        } else {
            ctx.next_start.min(end.saturating_add(ctx.tail_room))
        };
        ((ctx.start + 1) - end, hi - end)
    }
}

fn is_main_track(sequence: &Sequence, track_id: &str) -> bool {
    main_video_row(sequence)
        .and_then(|row| sequence.tracks.row_data(row as usize))
        .is_some_and(|t| t.id == track_id)
}

/// `(lane row, lane color, trim context)` of every other clip in the dragged
/// clip's link group; empty while linkage is off or the clip is unlinked.
fn linked_partners(
    sequence: &Sequence,
    clip_id: &str,
    link_enabled: bool,
) -> Vec<(i32, slint::Color, TrimContext)> {
    if !link_enabled {
        return Vec::new();
    }
    let mut link = None;
    for idx in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(idx) else {
            continue;
        };
        for clip_idx in 0..track.clips.row_count() {
            let Some(clip) = track.clips.row_data(clip_idx) else {
                continue;
            };
            if clip.id == clip_id && !clip.link_id.is_empty() {
                link = Some(clip.link_id.clone());
            }
        }
    }
    let Some(link) = link else {
        return Vec::new();
    };

    let mut partners = Vec::new();
    for idx in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(idx) else {
            continue;
        };
        for clip_idx in 0..track.clips.row_count() {
            let Some(clip) = track.clips.row_data(clip_idx) else {
                continue;
            };
            if clip.link_id == link
                && clip.id != clip_id
                && let Some(ctx) = trim_context(sequence, track.id.as_str(), clip.id.as_str())
            {
                partners.push((idx as i32, track.color, ctx));
            }
        }
    }
    partners
}

/// The trimmed clip's placement, source headroom, and lane neighbors.
struct TrimContext {
    start: i32,
    duration: i32,
    head_room: i32,
    tail_room: i32,
    /// End of the nearest clip left of this one on the lane (0 if none).
    prev_end: i32,
    /// Start of the nearest clip right of this one (huge sentinel if none).
    next_start: i32,
}

fn trim_context(sequence: &Sequence, track_id: &str, clip_id: &str) -> Option<TrimContext> {
    let track = (0..sequence.tracks.row_count())
        .filter_map(|i| sequence.tracks.row_data(i))
        .find(|t| t.id == track_id)?;
    let clip = (0..track.clips.row_count())
        .filter_map(|i| track.clips.row_data(i))
        .find(|c| c.id == clip_id)?;

    let start = clip.timeline_start.value;
    let duration = clip.source_range.duration.value;
    let end = start.saturating_add(duration);

    let mut prev_end = 0;
    let mut next_start = i32::MAX / 2;
    for idx in 0..track.clips.row_count() {
        let Some(other) = track.clips.row_data(idx) else {
            continue;
        };
        if other.id == clip_id {
            continue;
        }
        let s = other.timeline_start.value;
        let e = s.saturating_add(other.source_range.duration.value);
        // Lane clips never overlap, so every other clip is fully on one side.
        if e <= start {
            prev_end = prev_end.max(e);
        }
        if s >= end {
            next_start = next_start.min(s);
        }
    }

    Some(TrimContext {
        start,
        duration,
        head_room: clip.head_room_ticks.max(0),
        tail_room: clip.tail_room_ticks.max(0),
        prev_end,
        next_start,
    })
}

/// `(track kind, timeline start, duration ticks)` of the dragged clip.
fn find_dragged_clip(
    sequence: &Sequence,
    source_track_id: &str,
    clip_id: &str,
) -> Option<(TrackKind, i32, i32)> {
    for track_idx in 0..sequence.tracks.row_count() {
        let track = sequence.tracks.row_data(track_idx)?;
        if track.id != source_track_id {
            continue;
        }
        for clip_idx in 0..track.clips.row_count() {
            let clip = track.clips.row_data(clip_idx)?;
            if clip.id == clip_id {
                return Some((
                    track.kind,
                    clip.timeline_start.value,
                    clip.source_range.duration.value,
                ));
            }
        }
    }
    None
}

/// Whether `[start, start + duration)` overlaps no clip on `track`
/// (excluding the dragged clip itself when moving within its own lane).
fn span_is_free(track: &crate::Track, start: i32, duration: i32, exclude: Option<&str>) -> bool {
    let end = start.saturating_add(duration);
    for clip_idx in 0..track.clips.row_count() {
        let Some(clip) = track.clips.row_data(clip_idx) else {
            continue;
        };
        if exclude.is_some_and(|id| clip.id == id) {
            continue;
        }
        let s = clip.timeline_start.value;
        let e = s.saturating_add(clip.source_range.duration.value);
        if start < e && s < end {
            return false;
        }
    }
    true
}

impl SnapResult {
    fn none(cursor: i32) -> Self {
        Self {
            has_snap: false,
            snapped_start_value: cursor,
            snap_line_tick: cursor,
        }
    }
}

impl ClipDragResolution {
    fn invalid() -> Self {
        Self {
            valid: false,
            is_new_lane: false,
            target_track_id: Default::default(),
            target_row: 0,
            resolved_start: 0,
            duration_ticks: 0,
            has_snap: false,
            snap_line_tick: 0,
            is_noop: true,
            is_insert: false,
            caret_tick: 0,
        }
    }
}

impl ClipTrimResolution {
    fn invalid() -> Self {
        Self {
            valid: false,
            new_start: 0,
            new_duration: 0,
            has_snap: false,
            snap_line_tick: 0,
            is_noop: true,
            ghosts: ModelRc::from(Rc::new(VecModel::<GroupGhost>::default())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Clip, Rational, RationalTime, Sequence, TimeRange, Track, TrackKind};
    use slint::{ModelRc, SharedString, VecModel};
    use std::rc::Rc;

    fn rt(value: i32) -> RationalTime {
        RationalTime {
            value,
            rate: Rational { num: 24, den: 1 },
        }
    }

    /// A clip with effectively unbounded source headroom on both edges.
    fn clip(id: &str, start: i32, dur: i32) -> Clip {
        clip_with_rooms(id, start, dur, 1_000_000, 1_000_000)
    }

    fn clip_with_rooms(id: &str, start: i32, dur: i32, head_room: i32, tail_room: i32) -> Clip {
        Clip {
            id: SharedString::from(id),
            name: SharedString::from(id),
            timeline_start: rt(start),
            source_range: TimeRange {
                start: rt(0),
                duration: rt(dur),
            },
            head_room_ticks: head_room,
            tail_room_ticks: tail_room,
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

    /// Rows (top-first): 0 = V2 video, 1 = V1 video, 2 = A1 audio.
    fn sample_sequence() -> Sequence {
        sequence(vec![
            track(
                "2",
                TrackKind::Video,
                vec![clip("2", 0, 80), clip("3", 200, 60)],
            ),
            track("1", TrackKind::Video, vec![clip("1", 10, 100)]),
            track("9", TrackKind::Audio, vec![]),
        ])
    }

    // --- compute_drag_snap --------------------------------------------------

    #[test]
    fn snaps_to_zero_origin_when_dragging_near_start() {
        let seq = sample_sequence();
        let r = compute_drag_snap(&seq, "1", "1", 3, 100, 5, 0);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 0);
    }

    #[test]
    fn snaps_to_playhead() {
        let seq = sample_sequence();
        let r = compute_drag_snap(&seq, "1", "1", 202, 100, 5, 200);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 200);
        assert_eq!(r.snap_line_tick, 200);
    }

    #[test]
    fn zero_threshold_disables_snapping() {
        let seq = sample_sequence();
        let r = compute_drag_snap(&seq, "1", "1", 3, 100, 0, 0);
        assert!(!r.has_snap);
        assert_eq!(r.snapped_start_value, 3);
    }

    #[test]
    fn snaps_to_beat_marker() {
        // An audio lane whose clip carries beat markers at ticks 100 and 200.
        let mut beaten = clip("a", 0, 300);
        beaten.beats = ModelRc::from(Rc::new(VecModel::from(vec![100i32, 200])));
        let seq = sequence(vec![
            track("1", TrackKind::Video, vec![clip("1", 0, 50)]),
            track("9", TrackKind::Audio, vec![beaten]),
        ]);
        // Drag the video clip's leading edge to 197 — within 5 of beat 200.
        let r = compute_drag_snap(&seq, "1", "1", 197, 50, 5, 0);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 200);
        assert_eq!(r.snap_line_tick, 200);

        // Trailing edge snaps too: a clip of duration 50 dragged so its end
        // lands near beat 100 snaps the end to 100 (start 50).
        let r = compute_drag_snap(&seq, "1", "1", 48, 50, 5, 0);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 50);
        assert_eq!(r.snap_line_tick, 100);
    }

    #[test]
    fn short_clips_snap_by_leading_edge_only() {
        let seq = sample_sequence();

        // Duration 6 < 2×5: the trailing edge stops magneting. Dragged so
        // its *end* (196) is within threshold of the playhead (200), the
        // clip stays where the cursor put it instead of double-pinning.
        let r = compute_drag_snap(&seq, "1", "1", 190, 6, 5, 200);
        assert!(!r.has_snap);
        assert_eq!(r.snapped_start_value, 190);

        // Its leading edge still magnets normally.
        let r = compute_drag_snap(&seq, "1", "1", 197, 6, 5, 200);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 200);
        assert_eq!(r.snap_line_tick, 200);

        // Control: at duration 12 ≥ 2×5 the trailing edge snaps as before.
        let r = compute_drag_snap(&seq, "1", "1", 190, 12, 5, 200);
        assert!(r.has_snap);
        assert_eq!(r.snapped_start_value, 188);
        assert_eq!(r.snap_line_tick, 200);
    }

    // --- resolve_clip_drag --------------------------------------------------

    #[test]
    fn free_spot_on_same_lane_lands_there() {
        let seq = sample_sequence();
        // Clip "1" (row 1, start 10, dur 100) dragged right by 200 ticks.
        let r = resolve_clip_drag(&seq, "1", "1", 200, 1, 0, 5, false);
        assert!(r.valid && !r.is_new_lane);
        assert_eq!(r.target_track_id, "1");
        assert_eq!(r.resolved_start, 210);
        assert!(!r.is_noop);
    }

    #[test]
    fn unmoved_drag_is_noop() {
        let seq = sample_sequence();
        let r = resolve_clip_drag(&seq, "1", "1", 0, 1, 0, 0, false);
        assert!(r.valid && !r.is_new_lane && r.is_noop);
    }

    #[test]
    fn conflict_on_hovered_lane_creates_new_lane_at_row() {
        let seq = sample_sequence();
        // Drag clip "1" up onto row 0 at a position overlapping clip "2" [0,80).
        let r = resolve_clip_drag(&seq, "1", "1", 20, 0, 0, 0, false);
        assert!(r.valid && r.is_new_lane);
        assert_eq!(r.target_row, 0);
        assert_eq!(r.resolved_start, 30);
    }

    #[test]
    fn edge_snap_resolves_conflict_into_abutment() {
        let seq = sample_sequence();
        // Dragging clip "1" so its start sits at 78 — overlaps "2" [0,80) by 2
        // ticks, but the magnet pulls it to abut at 80, which is free.
        let r = resolve_clip_drag(&seq, "1", "1", 68, 0, 0, 5, false);
        assert!(r.valid && !r.is_new_lane);
        assert_eq!(r.target_track_id, "2");
        assert_eq!(r.resolved_start, 80);
        assert!(r.has_snap);
    }

    #[test]
    fn foreign_kind_lane_creates_new_lane() {
        let seq = sample_sequence();
        // Video clip hovered over the audio lane (row 2).
        let r = resolve_clip_drag(&seq, "1", "1", 200, 2, 0, 0, false);
        assert!(r.valid && r.is_new_lane);
        assert_eq!(r.target_row, 2);
    }

    #[test]
    fn rows_outside_stack_clamp_to_top_and_bottom_insertion() {
        let seq = sample_sequence();
        let above = resolve_clip_drag(&seq, "1", "1", 200, -3, 0, 0, false);
        assert!(above.is_new_lane);
        assert_eq!(above.target_row, 0);

        let below = resolve_clip_drag(&seq, "1", "1", 200, 9, 0, 0, false);
        assert!(below.is_new_lane);
        assert_eq!(below.target_row, 3);
    }

    #[test]
    fn unknown_ids_resolve_invalid() {
        let seq = sample_sequence();
        let r = resolve_clip_drag(&seq, "1", "404", 0, 1, 0, 0, false);
        assert!(!r.valid);
    }

    #[test]
    fn locked_lane_is_skipped_into_new_lane() {
        // Row 0 is a locked video lane; dragging clip "1" up onto it must not
        // land there (it resolves to a new lane inserted at the row instead).
        let mut locked = track("2", TrackKind::Video, vec![]);
        locked.locked = true;
        let seq = sequence(vec![
            locked,
            track("1", TrackKind::Video, vec![clip("1", 10, 100)]),
        ]);
        let r = resolve_clip_drag(&seq, "1", "1", 0, 0, 0, 0, false);
        assert!(r.valid && r.is_new_lane);
        assert_eq!(r.target_row, 0);
    }

    #[test]
    fn locked_video_lane_rejects_library_drop() {
        let mut locked = track("2", TrackKind::Video, vec![]);
        locked.locked = true;
        let seq = sequence(vec![locked]);
        // No unlocked video lane under the cursor → empty target (worker makes
        // a new lane), and no insertion even with the magnet on.
        let r = resolve_library_drop(&seq, TrackKind::Video, 48, 10, 0, 0, 5, true);
        assert_eq!(r.target_track_id, "");
        assert!(!r.is_insert);
    }

    // --- main-track magnet (insertion) --------------------------------------
    // sample_sequence rows: 0 = V2 (overlay video), 1 = V1 (main: bottom
    // video lane, clip "1" [10,110)), 2 = A1 (audio).

    #[test]
    fn magnet_hover_on_main_lane_resolves_to_insertion() {
        let seq = sample_sequence();
        // Video clip "2" (row 0, [0,80)) onto the main lane; left edge 5 is
        // left of clip "1"'s midpoint (60) → caret before it.
        let r = resolve_clip_drag(&seq, "2", "2", 5, 1, 0, 5, true);
        assert!(r.valid && r.is_insert && !r.is_new_lane);
        assert_eq!(r.target_track_id, "1");
        assert_eq!(r.caret_tick, 10);
        assert_eq!(r.resolved_start, 10);
        assert!(!r.has_snap && !r.is_noop);
    }

    #[test]
    fn magnet_insert_after_last_clip_lands_at_content_end() {
        let seq = sample_sequence();
        let r = resolve_clip_drag(&seq, "2", "2", 500, 1, 0, 0, true);
        assert!(r.is_insert);
        assert_eq!(r.caret_tick, 110);
        assert_eq!(r.resolved_start, 110);
    }

    #[test]
    fn magnet_reorder_to_own_slot_is_noop() {
        let seq = sample_sequence();
        let r = resolve_clip_drag(&seq, "1", "1", 0, 1, 0, 0, true);
        assert!(r.is_insert && r.is_noop);
        assert_eq!(r.caret_tick, 10);
    }

    #[test]
    fn magnet_reorder_commits_in_post_close_space() {
        // Single (= main) video lane, packed: A [0,50) B [50,80) C [80,120).
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![clip("A", 0, 50), clip("B", 50, 30), clip("C", 80, 40)],
        )]);
        // A dragged right past C's midpoint (100): insert after C.
        let r = resolve_clip_drag(&seq, "1", "A", 105, 0, 0, 0, true);
        assert!(r.is_insert && !r.is_noop);
        // Caret renders at the current content end…
        assert_eq!(r.caret_tick, 120);
        // …but commits against the closed arrangement (A's 50 ticks gone).
        assert_eq!(r.resolved_start, 70);
    }

    #[test]
    fn magnet_off_or_overlay_lane_stays_freeform() {
        let seq = sample_sequence();
        let off = resolve_clip_drag(&seq, "2", "2", 130, 1, 0, 0, false);
        assert!(!off.is_insert && !off.is_new_lane);
        assert_eq!(off.resolved_start, 130);

        // Magnet on, but hovering the *overlay* video lane (row 0): freeform.
        let overlay = resolve_clip_drag(&seq, "1", "1", 300, 0, 0, 0, true);
        assert!(!overlay.is_insert && !overlay.is_new_lane);
        assert_eq!(overlay.target_track_id, "2");
    }

    // --- resolve_library_drop -----------------------------------------------

    #[test]
    fn magnet_library_drop_on_main_lane_inserts_at_boundary() {
        let seq = sample_sequence();
        let before = resolve_library_drop(&seq, TrackKind::Video, 48, 30, 1, 0, 5, true);
        assert!(before.is_insert);
        assert_eq!(before.target_track_id, "1");
        assert_eq!(before.resolved_start, 10);
        assert_eq!(before.caret_tick, 10);

        let after = resolve_library_drop(&seq, TrackKind::Video, 48, 90, 1, 0, 5, true);
        assert!(after.is_insert);
        assert_eq!(after.resolved_start, 110);
    }

    #[test]
    fn library_drop_freeform_snaps_and_targets_video_lane() {
        let seq = sample_sequence();
        // Overlay video lane (row 0): cursor 78 magnets to clip "2"'s end.
        let r = resolve_library_drop(&seq, TrackKind::Video, 48, 78, 0, 0, 5, false);
        assert!(!r.is_insert);
        assert_eq!(r.target_track_id, "2");
        assert_eq!(r.resolved_start, 80);

        // Audio row: no video target — the worker creates a lane at the row.
        let foreign = resolve_library_drop(&seq, TrackKind::Video, 48, 100, 2, 0, 0, true);
        assert_eq!(foreign.target_track_id, "");
        assert_eq!(foreign.target_row, 2);
        assert!(!foreign.is_insert);
    }

    #[test]
    fn generated_drop_targets_matching_kind_lane() {
        // Rows: 0 = text lane "T", 1 = video lane "1".
        let seq = sequence(vec![
            track("T", TrackKind::Text, vec![clip("t1", 0, 72)]),
            track("1", TrackKind::Video, vec![clip("1", 0, 100)]),
        ]);
        // A text tile dropped on the text lane lands there (first-fit handled
        // by the worker; resolver just targets the lane).
        let on_text = resolve_library_drop(&seq, TrackKind::Text, 72, 100, 0, 0, 5, false);
        assert_eq!(on_text.target_track_id, "T");
        assert!(!on_text.is_insert);

        // A text tile dropped on the *video* row finds no text lane there →
        // empty target, worker creates a text lane at the row.
        let on_video = resolve_library_drop(&seq, TrackKind::Text, 72, 100, 1, 0, 5, false);
        assert_eq!(on_video.target_track_id, "");
        assert_eq!(on_video.target_row, 1);

        // Main magnet never turns a generated drop into an insertion.
        let magnet = resolve_library_drop(&seq, TrackKind::Sticker, 120, 50, 1, 0, 5, true);
        assert!(!magnet.is_insert);
    }

    #[test]
    fn library_drop_exposes_snap_guide() {
        let seq = sample_sequence();
        // Overlay video lane (row 0): cursor 78 magnets to clip "2"'s end (80).
        let r = resolve_library_drop(&seq, TrackKind::Video, 48, 78, 0, 0, 5, false);
        assert!(r.has_snap);
        assert_eq!(r.snap_line_tick, 80);

        // Out of threshold: no snap, no guide.
        let no = resolve_library_drop(&seq, TrackKind::Video, 48, 130, 0, 0, 5, false);
        assert!(!no.has_snap);
        assert_eq!(no.snap_line_tick, 0);
    }

    // --- resolve_clip_trim ----------------------------------------------------

    #[test]
    fn tail_trim_extends_until_next_clip() {
        let seq = sample_sequence();
        // Clip "2" [0,80) on track "2"; clip "3" starts at 200 on the same lane.
        let r = resolve_clip_trim(&seq, "2", "2", false, 300, 0, 0, false, false);
        assert!(r.valid);
        assert_eq!(r.new_start, 0);
        assert_eq!(r.new_duration, 200);
        assert!(!r.is_noop);
    }

    #[test]
    fn head_trim_extends_until_previous_clip() {
        let seq = sample_sequence();
        // Clip "3" [200,260) on track "2"; clip "2" ends at 80 on the same lane.
        let r = resolve_clip_trim(&seq, "2", "3", true, -500, 0, 0, false, false);
        assert!(r.valid);
        assert_eq!(r.new_start, 80);
        assert_eq!(r.new_duration, 180);
    }

    #[test]
    fn head_trim_clamps_to_source_headroom() {
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![clip_with_rooms("1", 10, 100, 5, 1_000_000)],
        )]);
        let r = resolve_clip_trim(&seq, "1", "1", true, -50, 0, 0, false, false);
        assert!(r.valid);
        assert_eq!(r.new_start, 5);
        assert_eq!(r.new_duration, 105);
    }

    #[test]
    fn tail_trim_clamps_to_source_tailroom() {
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![clip_with_rooms("1", 10, 100, 0, 7)],
        )]);
        let r = resolve_clip_trim(&seq, "1", "1", false, 50, 0, 0, false, false);
        assert!(r.valid);
        assert_eq!(r.new_start, 10);
        assert_eq!(r.new_duration, 107);
    }

    #[test]
    fn trim_never_collapses_below_one_tick() {
        let seq = sample_sequence();
        let head = resolve_clip_trim(&seq, "2", "2", true, 1_000, 0, 0, false, false);
        assert_eq!(head.new_start, 79);
        assert_eq!(head.new_duration, 1);

        let tail = resolve_clip_trim(&seq, "2", "2", false, -1_000, 0, 0, false, false);
        assert_eq!(tail.new_start, 0);
        assert_eq!(tail.new_duration, 1);
    }

    #[test]
    fn trimmed_edge_snaps_to_playhead() {
        let seq = sample_sequence();
        // Clip "1" [10,110) on track "1": tail dragged to 148, playhead at 150.
        let r = resolve_clip_trim(&seq, "1", "1", false, 38, 150, 5, false, false);
        assert!(r.valid && r.has_snap);
        assert_eq!(r.new_duration, 140);
        assert_eq!(r.snap_line_tick, 150);
    }

    #[test]
    fn snap_beyond_clamp_is_dropped() {
        // Tail room of 2 caps the end at 102, but clip "B" at 105 is a magnet
        // candidate within threshold — the clamp wins and the guide hides.
        let seq = sequence(vec![
            track(
                "1",
                TrackKind::Video,
                vec![clip_with_rooms("A", 0, 100, 0, 2)],
            ),
            track("2", TrackKind::Video, vec![clip("B", 105, 50)]),
        ]);
        let r = resolve_clip_trim(&seq, "1", "A", false, 4, 0, 5, false, false);
        assert!(r.valid && !r.has_snap);
        assert_eq!(r.new_duration, 102);
    }

    fn linked(mut clip: Clip, link: &str) -> Clip {
        clip.link_id = SharedString::from(link);
        clip
    }

    #[test]
    fn linked_tail_trim_clamps_to_partner_and_reports_ghost() {
        // Video member has plenty of tail room; the audio partner runs out
        // after 5 ticks — the link intersects the clamps.
        let seq = sequence(vec![
            track(
                "1",
                TrackKind::Video,
                vec![linked(clip_with_rooms("V", 0, 100, 0, 1_000), "L")],
            ),
            track(
                "9",
                TrackKind::Audio,
                vec![linked(clip_with_rooms("A", 0, 100, 0, 5), "L")],
            ),
        ]);
        let r = resolve_clip_trim(&seq, "1", "V", false, 50, 0, 0, true, false);
        assert!(r.valid);
        assert_eq!(r.new_duration, 105);
        assert_eq!(r.ghosts.row_count(), 1);
        let g = r.ghosts.row_data(0).unwrap();
        assert_eq!((g.row, g.start_tick, g.duration_ticks), (1, 0, 105));

        // Linkage off: the dragged edge only honors its own headroom, and
        // no partner ghosts are offered.
        let off = resolve_clip_trim(&seq, "1", "V", false, 50, 0, 0, false, false);
        assert_eq!(off.new_duration, 150);
        assert_eq!(off.ghosts.row_count(), 0);
    }

    #[test]
    fn linked_head_trim_respects_partner_neighbor() {
        // The audio partner has a neighbor ending at 20 on its lane; the
        // video head extension clamps there even though its own lane is clear.
        let seq = sequence(vec![
            track("1", TrackKind::Video, vec![linked(clip("V", 30, 50), "L")]),
            track(
                "9",
                TrackKind::Audio,
                vec![clip("X", 0, 20), linked(clip("A", 30, 50), "L")],
            ),
        ]);
        let r = resolve_clip_trim(&seq, "1", "V", true, -30, 0, 0, true, false);
        assert!(r.valid);
        assert_eq!(r.new_start, 20);
        assert_eq!(r.new_duration, 60);
        let g = r.ghosts.row_data(0).unwrap();
        assert_eq!((g.start_tick, g.duration_ticks), (20, 60));
    }

    #[test]
    fn unmoved_trim_is_noop() {
        let seq = sample_sequence();
        let r = resolve_clip_trim(&seq, "1", "1", true, 0, 0, 0, false, false);
        assert!(r.valid && r.is_noop);
    }

    #[test]
    fn unknown_trim_ids_resolve_invalid() {
        let seq = sample_sequence();
        let r = resolve_clip_trim(&seq, "1", "404", true, 0, 0, 0, false, false);
        assert!(!r.valid);
    }

    // --- main-track magnet (ripple trim resolver) ---------------------------

    #[test]
    fn magnet_tail_trim_extends_past_next_clip() {
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![clip_with_rooms("A", 0, 50, 0, 1_000_000), clip("B", 50, 30)],
        )]);
        let free = resolve_clip_trim(&seq, "1", "A", false, 80, 0, 0, false, false);
        assert_eq!(free.new_duration, 50);
        // Ripple mode: the next clip no longer clamps the dragged edge —
        // the full 80-tick drag lands (B shifts right on commit).
        let magnet = resolve_clip_trim(&seq, "1", "A", false, 80, 0, 0, false, true);
        assert_eq!(magnet.new_duration, 130);
    }

    #[test]
    fn magnet_head_trim_extends_into_headroom_not_neighbor() {
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![
                clip("A", 0, 50),
                clip_with_rooms("B", 50, 30, 100, 1_000_000),
            ],
        )]);
        let free = resolve_clip_trim(&seq, "1", "B", true, -100, 0, 0, false, false);
        assert_eq!(free.new_start, 50);
        let magnet = resolve_clip_trim(&seq, "1", "B", true, -100, 0, 0, false, true);
        assert_eq!(magnet.new_start, 0);
        assert_eq!(magnet.new_duration, 80);
    }
}
