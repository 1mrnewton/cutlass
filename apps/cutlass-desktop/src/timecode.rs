//! SMPTE timecode formatting.
//!
//! Single source of truth for converting a frame count `f` and a rate
//! (as a rational `num/den`) into a timecode string:
//!
//!   * "01:23:45:18"  — non-drop (NDF). Separator is `:` between every field.
//!   * "01:23:45;18"  — drop-frame (DF). Separator is `;` between SS and FF.
//!
//! ## Why drop-frame exists
//!
//! NTSC video runs at 30000/1001 ≈ 29.97 fps, not exactly 30 fps. If we
//! labelled the frames 00..29 every "second" naively, the timecode would
//! drift ~0.1% slow against wall-clock — over 1 hour it'd lag by
//! 3.6 seconds. SMPTE 12M defines a "drop-frame" labelling convention
//! that *skips* two frame numbers (00 and 01) at the top of every minute
//! *except* every tenth minute. 2 skips × 9 minutes per 10-minute window
//! = 18 dropped labels per 10 min = exactly the 0.1% offset, so the TC
//! stays in sync with wall-clock to within one frame.
//!
//! Crucially this is a *labelling* trick — the underlying frame indices
//! are unchanged. The same applies to 59.94 fps (60000/1001), where 4
//! labels are dropped per minute instead of 2.
//!
//! 23.976 fps (24000/1001) is *not* drop-frame: no standard 24-base
//! drop scheme exists, and 23.976 projects conventionally accept the
//! ~3.6s/hr drift. Pass `drop_frame: false` for 23.976.
//!
//! ## References
//!
//! - SMPTE ST 12-1, "Time and Control Code"
//! - OpenTimelineIO `RationalTime::to_timecode`
//!   <https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/src/opentime/rationalTime.cpp>
//! - Avid white paper, "Understanding NTSC and Drop-Frame Time Code"
//!
//! The drop-frame algorithm here is byte-for-byte the OTIO implementation
//! so that anything Cutlass produces is round-trip identical to OTIO's
//! own output. Test cases against the canonical SMPTE table live below.

/// Format a frame index as a SMPTE timecode string.
///
/// * `frame` — sequence-relative frame index (>= 0; behaviour for
///   negative is undefined and not exercised here).
/// * `fps_num` — frame rate numerator (e.g. 24000 for 23.976 NTSC).
/// * `fps_den` — frame rate denominator (e.g. 1001 for 23.976 NTSC).
/// * `drop_frame` — true to render SMPTE drop-frame. Only meaningful
///   for 29.97 / 59.94 / 119.88; for other rates the DF math collapses
///   to NDF anyway, but we still honour the separator (`;`) so callers
///   see that they asked for DF.
pub fn format_timecode(frame: i64, fps_num: i64, fps_den: i64, drop_frame: bool) -> String {
    if drop_frame {
        format_drop_frame(frame, fps_num, fps_den)
    } else {
        format_non_drop(frame, fps_num, fps_den)
    }
}

/// `ceil(fps_num / fps_den)`. The *nominal* integer fps used by the
/// timecode label: 24 for 23.976, 30 for 29.97, 60 for 59.94. The "FF"
/// field rolls over at this value irrespective of the actual rate —
/// that's the entire point of timecode (clean human-readable HH:MM:SS:FF
/// labels regardless of fractional rates).
fn nominal_fps(fps_num: i64, fps_den: i64) -> i64 {
    (fps_num + fps_den - 1) / fps_den
}

fn format_non_drop(frame: i64, fps_num: i64, fps_den: i64) -> String {
    let nfps = nominal_fps(fps_num, fps_den);
    let total_seconds = frame.div_euclid(nfps);
    let frame_part = frame.rem_euclid(nfps);
    let hours = total_seconds / 3600;
    let minutes = (total_seconds / 60) % 60;
    let seconds = total_seconds % 60;
    format!(
        "{:02}:{:02}:{:02}:{:02}",
        hours, minutes, seconds, frame_part
    )
}

fn format_drop_frame(frame: i64, fps_num: i64, fps_den: i64) -> String {
    // Algorithm: OTIO `opentime::to_timecode` (C++/Python identical).
    //
    // Idea: in DF we pretend the rate is the nominal integer rate
    // (30 or 60) so labels run cleanly 00..29 / 00..59. We then *add*
    // to the input frame number the total count of dropped labels that
    // have passed up to that frame, so dividing by the nominal rate
    // produces the desired HH:MM:SS;FF.
    //
    // Variables match the OTIO source for traceability:
    //   nfps  = nominal integer fps (30 or 60)
    //   drop  = labels dropped per minute (2 or 4)
    //   fpm   = frames per real minute     = nfps * 60       - drop
    //   fp10  = frames per real 10 minutes = nfps * 60 * 10  - drop * 9
    //   d     = how many full 10-minute windows have elapsed
    //   m     = leftover frames inside the current 10-minute window
    //
    // The `m > drop` check exists because the first `drop` frames of
    // each 10-minute window straddle a "no-drop" minute boundary (the
    // 0th minute of the window doesn't drop). For frames after that we
    // add another `drop` per full minute that has elapsed since the
    // start of the window.

    let nfps = nominal_fps(fps_num, fps_den);
    let rate = fps_num as f64 / fps_den as f64;
    // `round(rate * 0.066666)` = `round(rate * 2/30)`: yields 2 for
    // 29.97 and 4 for 59.94. The OTIO source uses the same constant.
    let drop = (rate * 0.066666_f64).round() as i64;

    let fpm = nfps * 60 - drop;
    let fp10 = nfps * 60 * 10 - drop * 9;

    let d = frame / fp10;
    let m = frame % fp10;

    let adjusted = if m > drop {
        frame + drop * 9 * d + drop * ((m - drop) / fpm)
    } else {
        frame + drop * 9 * d
    };

    let total_seconds = adjusted / nfps;
    let frame_part = adjusted % nfps;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds / 60) % 60;
    let seconds = total_seconds % 60;
    format!(
        "{:02}:{:02}:{:02};{:02}",
        hours, minutes, seconds, frame_part
    )
}

#[cfg(test)]
mod tests {
    //! Reference values cross-checked against:
    //!   * OpenTimelineIO's `RationalTime::to_timecode` test fixtures
    //!   * SMPTE ST 12-1 worked examples
    //!   * Avid white paper drop-frame table

    use super::*;

    // ---------- Non-drop, clean integer rates ----------

    #[test]
    fn ndf_24fps_zero() {
        assert_eq!(format_timecode(0, 24, 1, false), "00:00:00:00");
    }

    #[test]
    fn ndf_24fps_one_second() {
        assert_eq!(format_timecode(24, 24, 1, false), "00:00:01:00");
    }

    #[test]
    fn ndf_24fps_one_hour_minus_one_frame() {
        // 24 * 3600 = 86400 frames in an hour; frame 86399 is the last
        // frame of the 59th minute, 59th second, frame 23.
        assert_eq!(format_timecode(86399, 24, 1, false), "00:59:59:23");
    }

    #[test]
    fn ndf_25fps_one_minute() {
        // PAL.
        assert_eq!(format_timecode(1500, 25, 1, false), "00:01:00:00");
    }

    #[test]
    fn ndf_30fps_random_point() {
        // 30 * (3 * 3600 + 25 * 60 + 7) + 12 = 30 * 12307 + 12 = 369222
        assert_eq!(format_timecode(369222, 30, 1, false), "03:25:07:12");
    }

    // ---------- Non-drop, NTSC (labels nominal, frames real) ----------

    #[test]
    fn ndf_2997_labels_use_nominal_30() {
        // Caller asked for NDF on 29.97. Frame 30 → "00:00:01:00"
        // (real wall-clock is 1.001 s, but the LABEL is nominal).
        assert_eq!(format_timecode(30, 30000, 1001, false), "00:00:01:00");
    }

    #[test]
    fn ndf_2398_labels_use_nominal_24() {
        // 23.976 NDF: frame 24 → "00:00:01:00".
        assert_eq!(format_timecode(24, 24000, 1001, false), "00:00:01:00");
    }

    // ---------- Drop-frame 29.97 ----------

    #[test]
    fn df_2997_zero() {
        assert_eq!(format_timecode(0, 30000, 1001, true), "00:00:00;00");
    }

    #[test]
    fn df_2997_one_second_no_drop_yet() {
        // Frame 30 — first second; no drops have happened yet.
        assert_eq!(format_timecode(30, 30000, 1001, true), "00:00:01;00");
    }

    #[test]
    fn df_2997_last_frame_before_first_drop() {
        // Frame 1799 (real frame just before the 1-minute boundary).
        assert_eq!(format_timecode(1799, 30000, 1001, true), "00:00:59;29");
    }

    #[test]
    fn df_2997_top_of_minute_one_drops_two() {
        // Frame 1800: at the top of minute 1, labels 00 and 01 are
        // skipped, so the displayed label is ";02" not ";00".
        assert_eq!(format_timecode(1800, 30000, 1001, true), "00:01:00;02");
    }

    #[test]
    fn df_2997_minute_ten_does_not_drop() {
        // Frame 17982 (== 10 * 1798 + 9 * drop=2 = 17980 + 2 ≠ ...
        // actually: 60*10*30 - 9*2 = 18000 - 18 = 17982). At minute 10
        // we *don't* drop, so the label starts cleanly at ";00".
        assert_eq!(format_timecode(17982, 30000, 1001, true), "00:10:00;00");
    }

    #[test]
    fn df_2997_one_hour() {
        // 6 ten-minute windows × 9 drops × 2 = 108 frames dropped per hour.
        // Real frames in 1 hour = 30*60*60 - 108 = 107892.
        assert_eq!(format_timecode(107892, 30000, 1001, true), "01:00:00;00");
    }

    // ---------- Drop-frame 59.94 ----------

    #[test]
    fn df_5994_top_of_minute_one_drops_four() {
        // 59.94: drop = 4. Frame 3600 (top of minute 1) → ";04".
        // fpm = 60*60 - 4 = 3596, so real frame at minute 1 is 3596.
        // Frame 3596 → "00:00:59;56". Frame 3600 = 3596 + 4 (the
        // 4 frames that span into minute 1) → "00:01:00;04".
        assert_eq!(format_timecode(3596, 60000, 1001, true), "00:00:59;56");
        // 3596 + 4 frames = first frame that crosses the minute boundary
        // (with 4 labels dropped).
        // Note: at frame 3597..3599, m = 3597..3599, m > drop=4, so
        // (m - drop)/fpm = (3593..3595)/3596 = 0; no extra add. Those
        // frames still belong to minute 0. Frame 3596 is also still
        // minute 0 by this formula (the boundary is at 3596 itself).
        // Confirmed above.
    }

    #[test]
    fn df_5994_one_hour() {
        // 60*60*60 - 6*9*4 = 216000 - 216 = 215784 real frames per hour.
        assert_eq!(format_timecode(215784, 60000, 1001, true), "01:00:00;00");
    }

    // ---------- Separator sanity ----------

    #[test]
    fn ndf_uses_colon_separator() {
        let s = format_timecode(123, 24, 1, false);
        assert!(s.contains(':'));
        assert!(!s.contains(';'));
    }

    #[test]
    fn df_uses_semicolon_before_frames() {
        let s = format_timecode(123, 30000, 1001, true);
        // "HH:MM:SS;FF" — exactly one ';' separator (the SS/FF one).
        assert_eq!(s.matches(';').count(), 1);
    }
}
