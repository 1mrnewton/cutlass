use super::*;
use crate::clip::{Clip, Generator};
use crate::time::TimeRange;
use crate::track::{Track, TrackKind};

const R24: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, R24)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, R24)
}

fn generated_clip(start: i64, duration: i64) -> Clip {
    Clip::generated(Generator::Adjustment, tr(start, duration))
}

fn timeline_with_track() -> (Timeline, TrackId) {
    let mut timeline = Timeline::new(R24);
    let track = timeline.add_track(Track::new(TrackKind::Adjustment, "FX"));
    (timeline, track)
}

// --- Timeline::new ----------------------------------------------------

#[test]
fn new_starts_empty_at_frame_rate() {
    let timeline = Timeline::new(R24);
    assert_eq!(timeline.frame_rate, R24);
    assert_eq!(timeline.track_count(), 0);
    assert_eq!(timeline.clip_count(), 0);
    assert!(timeline.order().is_empty());
    assert_eq!(timeline.duration(), rt(0));
}

// --- tracks -----------------------------------------------------------

#[test]
fn add_track_appends_visual_to_top_and_floors_audio() {
    let mut timeline = Timeline::new(R24);
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
    let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));

    // Visual lanes stack bottom→top in insert order; the audio lane sinks
    // below them (index 0 = bottom of the stack / bottom of the UI).
    assert_eq!(timeline.order(), &[a1, v1, v2]);
    assert_eq!(timeline.track_count(), 3);
    assert_eq!(timeline.track(v1).unwrap().name, "V1");
    assert_eq!(timeline.track(a1).unwrap().kind, TrackKind::Audio);
}

#[test]
fn audio_always_sinks_below_video_regardless_of_add_order() {
    // Interleave kinds and confirm the CapCut zones: audio at the bottom,
    // the main video lane above it, overlay lanes next, text on top —
    // with relative order preserved inside each zone.
    let mut timeline = Timeline::new(R24);
    let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let a2 = timeline.add_track(Track::new(TrackKind::Audio, "A2"));
    let t1 = timeline.add_track(Track::new(TrackKind::Text, "T1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));

    assert_eq!(timeline.order(), &[a1, a2, v1, v2, t1]);
    assert_eq!(timeline.main_track(), Some(v1));
}

#[test]
fn insert_track_audio_sinks_and_visual_clamps_above_main() {
    let mut timeline = Timeline::new(R24);
    let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));

    // Requesting a video track at the very bottom (index 0) lifts it
    // above the audio block *and* the main lane: nothing but audio may
    // sit below the main track.
    let v0 = timeline.insert_track(Track::new(TrackKind::Video, "V0"), 0);
    assert_eq!(timeline.order(), &[a1, v1, v0]);
    assert_eq!(timeline.main_track(), Some(v1), "main status is sticky");

    // Requesting an audio track at the top sinks back into the audio block.
    let a2 = timeline.insert_track(Track::new(TrackKind::Audio, "A2"), 99);
    assert_eq!(timeline.order(), &[a1, a2, v1, v0]);
}

#[test]
fn move_track_reorders_within_zone_only() {
    let mut timeline = Timeline::new(Rational::new(24, 1));
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
    let v3 = timeline.add_track(Track::new(TrackKind::Video, "V3"));
    let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));
    assert_eq!(timeline.order(), &[a1, v1, v2, v3]);

    // Overlay lanes reorder freely within the overlay zone.
    timeline.move_track(v2, 3).unwrap();
    assert_eq!(timeline.order(), &[a1, v1, v3, v2]);

    // The main lane snaps back: it can't rise above overlays…
    timeline.move_track(v1, 3).unwrap();
    assert_eq!(timeline.order(), &[a1, v1, v3, v2]);
    // …and no overlay can sink below it.
    timeline.move_track(v3, 0).unwrap();
    assert_eq!(timeline.order(), &[a1, v1, v3, v2]);
}

#[test]
fn insert_track_places_at_order_index_and_clamps() {
    let mut timeline = Timeline::new(R24);
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));

    // V1 is the main lane, so a bottom insert clamps to just above it.
    let above_main = timeline.insert_track(Track::new(TrackKind::Video, "V3"), 0);
    let middle = timeline.insert_track(Track::new(TrackKind::Video, "V4"), 2);
    let top = timeline.insert_track(Track::new(TrackKind::Video, "V5"), 99);

    assert_eq!(timeline.order(), &[v1, above_main, middle, v2, top]);
    assert_eq!(timeline.track_count(), 5);
}

#[test]
fn first_video_track_becomes_main_and_persists() {
    let mut timeline = Timeline::new(R24);
    assert_eq!(timeline.main_track(), None);

    // Non-video lanes never claim main status.
    timeline.add_track(Track::new(TrackKind::Text, "T1"));
    timeline.add_track(Track::new(TrackKind::Audio, "A1"));
    assert_eq!(timeline.main_track(), None);

    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    assert_eq!(timeline.main_track(), Some(v1));
    assert!(timeline.track(v1).unwrap().main);

    // Later video lanes are overlays.
    let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
    assert_eq!(timeline.main_track(), Some(v1));
    assert!(!timeline.track(v2).unwrap().main);
}

#[test]
fn removing_main_promotes_next_bottom_video() {
    let mut timeline = Timeline::new(R24);
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
    let v3 = timeline.add_track(Track::new(TrackKind::Video, "V3"));

    timeline.remove_track(v1);
    assert_eq!(timeline.main_track(), Some(v2));
    assert_eq!(timeline.order(), &[v2, v3]);

    timeline.remove_track(v2);
    assert_eq!(timeline.main_track(), Some(v3));

    timeline.remove_track(v3);
    assert_eq!(timeline.main_track(), None);
}

#[test]
fn text_lanes_stay_on_top_and_only_audio_below_main() {
    let mut timeline = Timeline::new(R24);
    let t1 = timeline.add_track(Track::new(TrackKind::Text, "T1"));
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let s1 = timeline.add_track(Track::new(TrackKind::Sticker, "ST1"));
    let adj = timeline.add_track(Track::new(TrackKind::Adjustment, "ADJ1"));
    let t2 = timeline.add_track(Track::new(TrackKind::Text, "T2"));
    let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));

    // Zones bottom→top: audio, main video, overlays, text.
    assert_eq!(timeline.order(), &[a1, v1, s1, adj, t1, t2]);

    // A text lane moved to the bottom snaps back above the overlays.
    timeline.move_track(t1, 0).unwrap();
    assert_eq!(timeline.order(), &[a1, v1, s1, adj, t1, t2]);

    // An adjustment lane can never sink below the main track.
    timeline.move_track(adj, 0).unwrap();
    assert_eq!(timeline.order(), &[a1, v1, adj, s1, t1, t2]);
}

#[test]
fn normalize_lanes_derives_main_and_zones_for_legacy_files() {
    // Simulate a pre-main-flag file: build the raw state without going
    // through add_track (serde would produce exactly this shape).
    let mut timeline = Timeline::new(R24);
    let t1 = timeline.add_track(Track::new(TrackKind::Text, "T1"));
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
    // Strip the flags to fake the legacy load.
    timeline.track_mut(v1).unwrap().main = false;
    timeline.track_mut(v2).unwrap().main = false;

    timeline.normalize_lanes();
    assert_eq!(timeline.main_track(), Some(v1), "bottom video lane wins");
    assert_eq!(timeline.order(), &[v1, v2, t1]);

    // A corrupt file flagging a non-video lane is repaired.
    timeline.track_mut(t1).unwrap().main = true;
    timeline.track_mut(v1).unwrap().main = false;
    timeline.normalize_lanes();
    assert!(!timeline.track(t1).unwrap().main);
    assert_eq!(timeline.main_track(), Some(v1));
}

#[test]
fn main_flag_serde_round_trips_and_defaults_false() {
    let mut timeline = Timeline::new(R24);
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));

    let json = serde_json::to_value(&timeline).unwrap();
    let back: Timeline = serde_json::from_value(json).unwrap();
    assert_eq!(back.main_track(), Some(v1));

    // Pre-main-flag track JSON loads as not-main (normalize_lanes
    // re-derives on file load).
    let legacy = serde_json::json!({
        "id": 1,
        "kind": "Video",
        "name": "V1",
        "enabled": true,
        "muted": false,
        "clips": []
    });
    let track: Track = serde_json::from_value(legacy).unwrap();
    assert!(!track.main);
}

#[test]
fn tracks_ordered_yields_bottom_to_top() {
    let mut timeline = Timeline::new(R24);
    timeline.add_track(Track::new(TrackKind::Video, "bottom"));
    timeline.add_track(Track::new(TrackKind::Video, "top"));
    let names: Vec<&str> = timeline.tracks_ordered().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["bottom", "top"]);
}

#[test]
fn track_mut_can_toggle_enabled() {
    let mut timeline = Timeline::new(R24);
    let id = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    timeline.track_mut(id).unwrap().enabled = false;
    assert!(!timeline.track(id).unwrap().enabled);
}

#[test]
fn restore_track_reinserts_stack_position_and_clip_index() {
    let (mut timeline, track_id) = timeline_with_track();
    let clip_id = timeline
        .add_clip(track_id, generated_clip(0, 10))
        .expect("clip");
    let track = timeline.remove_track(track_id).expect("remove");
    assert_eq!(timeline.track_count(), 0);
    assert!(timeline.clip(clip_id).is_none());

    timeline.restore_track(track, 0).expect("restore");
    assert_eq!(timeline.track_count(), 1);
    assert_eq!(timeline.track_of(clip_id), Some(track_id));
}

#[test]
fn restore_track_keeps_audio_below_video() {
    // Undo of removing a video lane must not slip it under the audio floor.
    let mut timeline = Timeline::new(R24);
    let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));
    let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
    assert_eq!(timeline.order(), &[a1, v1, v2]);

    let order_index = timeline.order().iter().position(|&t| t == v1).unwrap();
    let removed = timeline.remove_track(v1).expect("remove");
    assert_eq!(timeline.order(), &[a1, v2]);

    timeline
        .restore_track(removed, order_index)
        .expect("restore");
    assert_eq!(timeline.order(), &[a1, v1, v2]);
}

#[test]
fn remove_track_purges_clips_from_index() {
    let (mut timeline, track) = timeline_with_track();
    let clip = timeline.add_clip(track, generated_clip(0, 50)).unwrap();

    let removed = timeline.remove_track(track).unwrap();
    assert_eq!(removed.len(), 1);
    assert_eq!(timeline.track_count(), 0);
    assert_eq!(timeline.clip_count(), 0);
    assert!(timeline.clip(clip).is_none());
    assert!(timeline.track_of(clip).is_none());
}

#[test]
fn remove_unknown_track_returns_none() {
    let mut timeline = Timeline::new(R24);
    assert!(timeline.remove_track(TrackId::from_raw(99)).is_none());
}

// --- add_clip / clip index --------------------------------------------

#[test]
fn add_clip_registers_in_track_and_index() {
    let (mut timeline, track) = timeline_with_track();
    let clip = generated_clip(10, 40);
    let clip_id = clip.id;

    let returned = timeline.add_clip(track, clip).unwrap();
    assert_eq!(returned, clip_id);
    assert_eq!(timeline.clip_count(), 1);
    assert_eq!(timeline.track_of(clip_id), Some(track));
    assert_eq!(timeline.clip(clip_id).unwrap().timeline, tr(10, 40));
    assert_eq!(timeline.track(track).unwrap().len(), 1);
}

#[test]
fn add_clip_unknown_track_errors() {
    let (mut timeline, _) = timeline_with_track();
    let missing = TrackId::from_raw(404);
    assert_eq!(
        timeline.add_clip(missing, generated_clip(0, 10)),
        Err(ModelError::UnknownTrack(missing))
    );
}

#[test]
fn add_clip_rejects_overlap_on_same_track() {
    let (mut timeline, track) = timeline_with_track();
    timeline.add_clip(track, generated_clip(0, 50)).unwrap();
    assert_eq!(
        timeline.add_clip(track, generated_clip(25, 50)),
        Err(ModelError::Overlap(track))
    );
}

#[test]
fn add_clip_rejects_duplicate_id_without_mutating() {
    // Defense in depth for the id-collision bug: a clip whose id already
    // lives on the timeline must be rejected, never silently overwrite the
    // existing clip (which would drop it from its track/index).
    let mut timeline = Timeline::new(R24);
    let v1 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX2"));
    let clip = generated_clip(0, 50);
    let id = clip.id;
    let mut dup = generated_clip(200, 10);
    dup.id = id;

    timeline.add_clip(v1, clip).unwrap();
    assert_eq!(
        timeline.add_clip(v2, dup),
        Err(ModelError::DuplicateClip(id))
    );
    // The original is untouched: still on v1, still 50 ticks at start 0.
    assert_eq!(timeline.clip_count(), 1);
    assert_eq!(timeline.track_of(id), Some(v1));
    assert_eq!(timeline.clip(id).unwrap().timeline, tr(0, 50));
    assert!(timeline.track(v2).unwrap().is_empty());
}

#[test]
fn add_clip_allows_same_range_on_different_tracks() {
    let mut timeline = Timeline::new(R24);
    let v1 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX2"));

    let c1 = timeline.add_clip(v1, generated_clip(0, 50)).unwrap();
    let c2 = timeline.add_clip(v2, generated_clip(0, 50)).unwrap();
    assert_ne!(c1, c2);
    assert_eq!(timeline.clip_count(), 2);
}

#[test]
fn add_clip_allows_adjacent_non_overlapping_clips() {
    let (mut timeline, track) = timeline_with_track();
    timeline.add_clip(track, generated_clip(0, 50)).unwrap();
    let second = timeline.add_clip(track, generated_clip(50, 50)).unwrap();
    assert_eq!(timeline.clip_count(), 2);
    assert_eq!(timeline.clip(second).unwrap().start().value, 50);
}

#[test]
fn carries_own_audio_follows_lane_and_detach() {
    let mut timeline = Timeline::new(R24);
    let v = timeline.add_track(Track::new(TrackKind::Video, "V1"));
    let a = timeline.add_track(Track::new(TrackKind::Audio, "A1"));
    let fx = timeline.add_track(Track::new(TrackKind::Adjustment, "FX1"));

    let media = crate::ids::MediaId::from_raw(1);
    let media_clip = |start, dur| Clip::from_media(media, tr(start, dur), tr(start, dur));

    // CapCut keeps a video's sound on the clip, so a video clip carries its
    // own audio; audio lanes always do; non-AV lanes never do.
    let vc = timeline.add_clip(v, media_clip(0, 24)).unwrap();
    let ac = timeline.add_clip(a, media_clip(0, 24)).unwrap();
    let fxc = timeline.add_clip(fx, generated_clip(0, 24)).unwrap();
    assert!(timeline.carries_own_audio(vc));
    assert!(timeline.carries_own_audio(ac));
    assert!(!timeline.carries_own_audio(fxc));

    // "Separate audio": link the video clip to a clip on an audio lane and
    // the video half defers its sound there (silent on its own lane).
    let companion = timeline.add_clip(a, media_clip(48, 24)).unwrap();
    let link = crate::ids::LinkId::next();
    timeline.clip_mut(vc).unwrap().link = Some(link);
    timeline.clip_mut(companion).unwrap().link = Some(link);
    assert!(
        !timeline.carries_own_audio(vc),
        "detached video half is silent"
    );
    assert!(
        timeline.carries_own_audio(companion),
        "the audio-lane partner sounds"
    );
}

// --- remove_clip / lookup ---------------------------------------------

#[test]
fn remove_clip_returns_clip_and_clears_index() {
    let (mut timeline, track) = timeline_with_track();
    let id = timeline.add_clip(track, generated_clip(0, 30)).unwrap();

    let removed = timeline.remove_clip(id).unwrap();
    assert_eq!(removed.id, id);
    assert_eq!(timeline.clip_count(), 0);
    assert!(timeline.clip(id).is_none());
    assert!(timeline.track_of(id).is_none());
    assert!(timeline.track(track).unwrap().is_empty());
}

#[test]
fn remove_clip_unknown_returns_none() {
    let (mut timeline, _) = timeline_with_track();
    assert!(timeline.remove_clip(ClipId::from_raw(77)).is_none());
}

#[test]
fn clip_mut_updates_timeline_range() {
    let (mut timeline, track) = timeline_with_track();
    let id = timeline.add_clip(track, generated_clip(0, 50)).unwrap();

    timeline.clip_mut(id).unwrap().timeline = tr(10, 40);
    assert_eq!(timeline.clip(id).unwrap().timeline, tr(10, 40));
}

#[test]
fn clip_lookup_finds_across_tracks() {
    let mut timeline = Timeline::new(R24);
    let v1 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX2"));
    let on_v2 = timeline.add_clip(v2, generated_clip(100, 20)).unwrap();
    timeline.add_clip(v1, generated_clip(0, 10)).unwrap();

    assert_eq!(timeline.track_of(on_v2), Some(v2));
    assert_eq!(timeline.clip(on_v2).unwrap().start().value, 100);
}

// --- duration ---------------------------------------------------------

#[test]
fn duration_empty_timeline_is_zero() {
    let timeline = Timeline::new(R24);
    assert_eq!(timeline.duration(), rt(0));
}

#[test]
fn duration_is_max_end_across_tracks() {
    let mut timeline = Timeline::new(R24);
    let v1 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX1"));
    let v2 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX2"));
    timeline.add_clip(v1, generated_clip(0, 100)).unwrap();
    timeline.add_clip(v2, generated_clip(50, 200)).unwrap(); // ends at 250

    assert_eq!(timeline.duration().value, 250);
    assert_eq!(timeline.duration().rate, R24);
}

#[test]
fn duration_ignores_gap_between_clips_on_same_track() {
    let (mut timeline, track) = timeline_with_track();
    timeline.add_clip(track, generated_clip(0, 50)).unwrap();
    timeline.add_clip(track, generated_clip(100, 30)).unwrap(); // ends 130

    assert_eq!(timeline.duration().value, 130);
}

// --- markers ------------------------------------------------------------

fn marker_at(tick: i64) -> Marker {
    Marker::new(rt(tick), "", MarkerColor::Teal)
}

#[test]
fn add_marker_keeps_tick_order_and_resamples() {
    let mut timeline = Timeline::new(R24);
    let late = timeline.add_marker(marker_at(100)).unwrap();
    let early = timeline.add_marker(marker_at(10)).unwrap();
    // 2 s at 48 ticks/s resamples to 24 ticks at the 24 fps timeline.
    let resampled = timeline
        .add_marker(Marker::new(
            RationalTime::new(96, Rational::new(48, 1)),
            "beat",
            MarkerColor::Red,
        ))
        .unwrap();

    let order: Vec<MarkerId> = timeline.markers().iter().map(|m| m.id).collect();
    assert_eq!(order, [early, resampled, late]);
    let beat = timeline.marker(resampled).unwrap();
    assert_eq!(beat.tick, rt(48));
    assert_eq!(beat.name, "beat");
    assert_eq!(beat.color, MarkerColor::Red);
}

#[test]
fn add_marker_rejects_negative_tick_and_duplicate_id() {
    let mut timeline = Timeline::new(R24);
    assert_eq!(
        timeline.add_marker(marker_at(-1)),
        Err(ModelError::InvalidRange)
    );
    let marker = marker_at(5);
    let dup = marker.clone();
    timeline.add_marker(marker).unwrap();
    assert_eq!(timeline.add_marker(dup), Err(ModelError::InvalidRange));
    assert_eq!(timeline.marker_count(), 1);
}

#[test]
fn remove_marker_returns_snapshot_for_undo() {
    let mut timeline = Timeline::new(R24);
    let id = timeline
        .add_marker(Marker::new(rt(7), "drop", MarkerColor::Pink))
        .unwrap();

    let removed = timeline.remove_marker(id).unwrap();
    assert_eq!(removed.name, "drop");
    assert_eq!(timeline.marker_count(), 0);
    assert!(timeline.remove_marker(id).is_none());

    // Restoring the snapshot keeps the same id (undo of remove).
    timeline.add_marker(removed).unwrap();
    assert_eq!(timeline.marker(id).unwrap().tick, rt(7));
}

#[test]
fn set_marker_moves_renames_recolors_and_resorts() {
    let mut timeline = Timeline::new(R24);
    let a = timeline.add_marker(marker_at(10)).unwrap();
    let b = timeline.add_marker(marker_at(20)).unwrap();

    timeline
        .set_marker(a, rt(30), "outro".into(), MarkerColor::Green)
        .unwrap();
    let order: Vec<MarkerId> = timeline.markers().iter().map(|m| m.id).collect();
    assert_eq!(order, [b, a], "tick change re-sorts");
    let moved = timeline.marker(a).unwrap();
    assert_eq!((moved.tick, moved.name.as_str()), (rt(30), "outro"));
    assert_eq!(moved.color, MarkerColor::Green);

    assert_eq!(
        timeline.set_marker(
            MarkerId::from_raw(999),
            rt(0),
            String::new(),
            MarkerColor::Teal
        ),
        Err(ModelError::UnknownMarker(MarkerId::from_raw(999)))
    );
    // A rejected move leaves the marker untouched.
    assert_eq!(
        timeline.set_marker(a, rt(-5), String::new(), MarkerColor::Teal),
        Err(ModelError::InvalidRange)
    );
    assert_eq!(timeline.marker(a).unwrap().tick, rt(30));
    assert_eq!(timeline.marker(a).unwrap().name, "outro");
}

#[test]
fn markers_serialize_only_when_present() {
    let mut timeline = Timeline::new(R24);
    let empty = serde_json::to_value(&timeline).unwrap();
    assert!(
        empty.get("markers").is_none(),
        "marker-free timelines serialize without the field"
    );
    // Pre-marker files (no `markers` key) deserialize to an empty list.
    let loaded: Timeline = serde_json::from_value(empty).unwrap();
    assert_eq!(loaded.marker_count(), 0);

    timeline
        .add_marker(Marker::new(rt(12), "intro", MarkerColor::Blue))
        .unwrap();
    let json = serde_json::to_value(&timeline).unwrap();
    assert_eq!(json["markers"][0]["name"], "intro");
    assert_eq!(json["markers"][0]["color"], "blue");
    let back: Timeline = serde_json::from_value(json).unwrap();
    assert_eq!(back.markers(), timeline.markers());
}

#[test]
fn marker_color_cycles_through_the_palette() {
    assert_eq!(MarkerColor::cycle(0), MarkerColor::Teal);
    assert_eq!(MarkerColor::cycle(7), MarkerColor::Green);
    assert_eq!(MarkerColor::cycle(8), MarkerColor::Teal);
    for color in MarkerColor::ALL {
        assert_eq!(color.rgba()[3], 0xFF, "palette colors are opaque");
        assert!(color.name().is_some_and(|n| !n.is_empty()));
    }
}

#[test]
fn marker_color_rgba_serde_keeps_named_presets_and_round_trips() {
    assert_eq!(
        serde_json::to_value(MarkerColor::Red).unwrap(),
        serde_json::json!("red")
    );
    let custom = MarkerColor::custom(0x12, 0x34, 0x56);
    let json = serde_json::to_value(custom).unwrap();
    assert_eq!(
        json,
        serde_json::json!({ "rgba": [0x12, 0x34, 0x56, 0xFF] })
    );
    let back: MarkerColor = serde_json::from_value(json).unwrap();
    assert_eq!(back, custom);
    // Old project files with named colors still load.
    let legacy: MarkerColor = serde_json::from_value(serde_json::json!("teal")).unwrap();
    assert_eq!(legacy, MarkerColor::Teal);
    assert_eq!(
        MarkerColor::parse_token("#abc"),
        Some(MarkerColor::custom(0xAA, 0xBB, 0xCC))
    );
    assert_eq!(custom.token(), "#123456");
}

// --- canvas -------------------------------------------------------------

#[test]
fn canvas_defaults_to_auto_black_and_round_trips() {
    let mut timeline = Timeline::new(R24);
    assert!(timeline.canvas().is_default());
    assert_eq!(timeline.canvas().aspect, CanvasAspect::Auto);
    assert_eq!(timeline.canvas().background, [0, 0, 0]);

    // Default settings serialize without the field; pre-canvas files
    // (no `canvas` key) deserialize to the default.
    let json = serde_json::to_value(&timeline).unwrap();
    assert!(json.get("canvas").is_none());
    let loaded: Timeline = serde_json::from_value(json).unwrap();
    assert!(loaded.canvas().is_default());

    timeline.set_canvas(CanvasSettings {
        aspect: CanvasAspect::Tall9x16,
        background: [20, 30, 40],
    });
    let json = serde_json::to_value(&timeline).unwrap();
    assert_eq!(json["canvas"]["aspect"], "9:16");
    assert_eq!(
        json["canvas"]["background"],
        serde_json::json!([20, 30, 40])
    );
    let back: Timeline = serde_json::from_value(json).unwrap();
    assert_eq!(back.canvas(), timeline.canvas());
}

#[test]
fn canvas_partial_fields_serialize_independently() {
    let mut timeline = Timeline::new(R24);
    // Only the aspect set: the black background stays off the wire.
    timeline.set_canvas(CanvasSettings {
        aspect: CanvasAspect::Square1x1,
        background: [0, 0, 0],
    });
    let json = serde_json::to_value(&timeline).unwrap();
    assert_eq!(json["canvas"]["aspect"], "1:1");
    assert!(json["canvas"].get("background").is_none());

    // Only the background set: auto aspect stays off the wire.
    timeline.set_canvas(CanvasSettings {
        aspect: CanvasAspect::Auto,
        background: [255, 255, 255],
    });
    let json = serde_json::to_value(&timeline).unwrap();
    assert!(json["canvas"].get("aspect").is_none());
    assert_eq!(
        json["canvas"]["background"],
        serde_json::json!([255, 255, 255])
    );
    let back: Timeline = serde_json::from_value(json).unwrap();
    assert_eq!(back.canvas().background, [255, 255, 255]);
    assert_eq!(back.canvas().aspect, CanvasAspect::Auto);
}

#[test]
fn canvas_aspect_names_round_trip() {
    for aspect in CanvasAspect::ALL {
        assert_eq!(CanvasAspect::from_name(aspect.name()), Some(aspect));
        match aspect.ratio() {
            None => assert_eq!(aspect, CanvasAspect::Auto),
            Some((w, h)) => {
                assert!(w > 0 && h > 0);
                assert_eq!(aspect.name(), format!("{w}:{h}"));
            }
        }
    }
    assert_eq!(CanvasAspect::from_name("4:3"), None);
}

// --- Clone ------------------------------------------------------------

#[test]
fn clone_is_independent_snapshot() {
    let (mut timeline, track) = timeline_with_track();
    let clip = timeline.add_clip(track, generated_clip(0, 50)).unwrap();

    let mut cloned = timeline.clone();
    cloned.remove_clip(clip);
    assert_eq!(cloned.clip_count(), 0);
    assert_eq!(timeline.clip_count(), 1);
}
