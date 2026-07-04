//! Playback transport tick math — the pure callback behind
//! `ui/lib/transport-backend.slint` (same resolver pattern as `snap.rs`).
//!
//! The Slint-side clock stores an anchor (tick + wall-clock ms) when play
//! starts and asks Rust for the current tick on every timer step, so the
//! rational-rate math runs in exact i64 instead of Slint floats. The tick
//! is always *computed* from elapsed time — never incremented per rendered
//! frame — which is what lets slow decode drop frames without slowing the
//! playhead (see docs/playback-roadmap.md).

/// Playhead tick after `now_ms - anchor_ms` milliseconds of playback
/// starting at `anchor_tick`, at `fps_num / fps_den` ticks per second,
/// scaled by `speed_num / speed_den` (JKL shuttle — playback roadmap
/// Phase 4; 1/1 is plain playback).
///
/// Negative `speed_num` runs the playhead in reverse; mid-frame time
/// truncates toward the anchor in both directions, so the first frame
/// boundary is one frame-duration away forward *and* backward. Elapsed
/// time clamps at 0 (a stale anchor can never rewind the playhead); a
/// zero/invalid speed or rate returns the anchor unchanged.
pub fn playback_tick_scaled(
    anchor_tick: i32,
    anchor_ms: i32,
    now_ms: i32,
    fps_num: i32,
    fps_den: i32,
    speed_num: i32,
    speed_den: i32,
) -> i32 {
    if fps_num <= 0 || fps_den <= 0 || speed_num == 0 || speed_den <= 0 {
        return anchor_tick;
    }
    let elapsed_ms = i128::from((i64::from(now_ms) - i64::from(anchor_ms)).max(0));
    let ticks = elapsed_ms * i128::from(fps_num) * i128::from(speed_num)
        / (i128::from(fps_den) * i128::from(speed_den) * 1000);
    (i128::from(anchor_tick) + ticks).clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1x path, as the Slint clock calls it during plain playback.
    fn playback_tick(
        anchor_tick: i32,
        anchor_ms: i32,
        now_ms: i32,
        fps_num: i32,
        fps_den: i32,
    ) -> i32 {
        playback_tick_scaled(anchor_tick, anchor_ms, now_ms, fps_num, fps_den, 1, 1)
    }

    #[test]
    fn integer_rate_advances_exactly() {
        assert_eq!(playback_tick(0, 0, 1000, 24, 1), 24);
        assert_eq!(playback_tick(10, 0, 2000, 24, 1), 58);
        assert_eq!(playback_tick(0, 500, 1500, 60, 1), 60);
    }

    #[test]
    fn ntsc_rate_is_exact() {
        // 30000/1001 fps: 1001ms is exactly 30 frames.
        assert_eq!(playback_tick(0, 0, 1001, 30000, 1001), 30);
        // One hour of NTSC: 3_600_000 ms · 30000 / (1001 · 1000) = 107_892.1… → floor.
        assert_eq!(playback_tick(0, 0, 3_600_000, 30000, 1001), 107_892);
    }

    #[test]
    fn mid_frame_floors() {
        // 24fps ⇒ 41.67ms per frame.
        assert_eq!(playback_tick(0, 0, 41, 24, 1), 0);
        assert_eq!(playback_tick(0, 0, 42, 24, 1), 1);
    }

    #[test]
    fn stale_anchor_never_rewinds() {
        assert_eq!(playback_tick(100, 5000, 4000, 24, 1), 100);
    }

    #[test]
    fn invalid_rate_returns_anchor() {
        assert_eq!(playback_tick(7, 0, 1000, 0, 1), 7);
        assert_eq!(playback_tick(7, 0, 1000, 24, 0), 7);
        assert_eq!(playback_tick(7, 0, 1000, -24, 1), 7);
    }

    #[test]
    fn scaled_speeds_multiply_the_rate() {
        // 2x: one second of 24fps covers 48 ticks.
        assert_eq!(playback_tick_scaled(0, 0, 1000, 24, 1, 2, 1), 48);
        // 8x cap case.
        assert_eq!(playback_tick_scaled(0, 0, 1000, 24, 1, 8, 1), 192);
        // Fractional speed (half): 12 ticks per second.
        assert_eq!(playback_tick_scaled(0, 0, 1000, 24, 1, 1, 2), 12);
    }

    #[test]
    fn reverse_speed_walks_backward() {
        assert_eq!(playback_tick_scaled(100, 0, 1000, 24, 1, -1, 1), 76);
        assert_eq!(playback_tick_scaled(100, 0, 1000, 24, 1, -2, 1), 52);
        // Mid-frame truncates toward the anchor: 41ms at -1x is no step yet.
        assert_eq!(playback_tick_scaled(100, 0, 41, 24, 1, -1, 1), 100);
        assert_eq!(playback_tick_scaled(100, 0, 42, 24, 1, -1, 1), 99);
    }

    #[test]
    fn zero_speed_returns_anchor() {
        assert_eq!(playback_tick_scaled(7, 0, 1000, 24, 1, 0, 1), 7);
        assert_eq!(playback_tick_scaled(7, 0, 1000, 24, 1, 1, 0), 7);
    }
}
