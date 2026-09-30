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
        pinned: false,
        is_main: false,
        transitions: ModelRc::default(),
    }
}

fn sequence(mut tracks: Vec<Track>) -> Sequence {
    // The model designates the bottom video lane as the main track and
    // the projection publishes the flag; mirror that here (rows are
    // top-first, so the bottom lane is the last video entry).
    if let Some(main) = tracks.iter_mut().rev().find(|t| t.kind == TrackKind::Video) {
        main.is_main = true;
    }
    Sequence {
        id: SharedString::from("1"),
        name: SharedString::from("Sequence 1"),
        fps: Rational { num: 24, den: 1 },
        drop_frame: false,
        tracks: ModelRc::from(Rc::new(VecModel::from(tracks))),
        markers: Default::default(),
        caption_groups: Default::default(),
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

// --- resolve_transition_junction -----------------------------------------

#[test]
fn transition_junction_rejects_canvas_pass_and_audio_lanes() {
    // Abutting pairs on every lane; only frame-compositing lanes
    // (video/sticker/text) may take a transition at the junction.
    let pair = |a: &str, b: &str| vec![clip(a, 0, 40), clip(b, 40, 40)];
    let seq = sequence(vec![
        track("e", TrackKind::Effect, pair("e1", "e2")),
        track("s", TrackKind::Sticker, pair("s1", "s2")),
        track("v", TrackKind::Video, pair("v1", "v2")),
        track("a", TrackKind::Audio, pair("a1", "a2")),
    ]);

    assert!(!resolve_transition_junction(&seq, 40, 0, 5).valid);
    assert!(resolve_transition_junction(&seq, 40, 1, 5).valid);
    assert!(resolve_transition_junction(&seq, 40, 2, 5).valid);
    assert!(!resolve_transition_junction(&seq, 40, 3, 5).valid);
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
