//! Timeline ruler. The step table is the CapCut 9.6.0 table in `docs/ruler.md`.

use cutlass_project::{FrameRate, SequenceRate};

const MARK_CAP: usize = 1001;

/// One tick on the ruler. `us` is sequence time. A label is present on a major tick.
#[derive(Debug, PartialEq, Eq)]
pub struct Mark {
    pub us: i64,
    pub major: bool,
    pub label: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Clock { seconds: i64, minors: i64 },
    Frames { frames: i64, minors: i64 },
}

impl Step {
    const fn minors(self) -> i64 {
        match self {
            Self::Clock { minors, .. } | Self::Frames { minors, .. } => minors,
        }
    }

    const fn frames(self) -> bool {
        matches!(self, Self::Frames { .. })
    }
}

struct ClockStep {
    seconds: i64,
    threshold: f64,
}

/// Coarse to fine. The threshold is pixels per microsecond.
const CLOCK: [ClockStep; 18] = [
    ClockStep {
        seconds: 108_000,
        threshold: 2.6e-9,
    },
    ClockStep {
        seconds: 36_000,
        threshold: 8e-9,
    },
    ClockStep {
        seconds: 18_000,
        threshold: 1.6e-8,
    },
    ClockStep {
        seconds: 10_800,
        threshold: 2.6e-8,
    },
    ClockStep {
        seconds: 7_200,
        threshold: 4e-8,
    },
    ClockStep {
        seconds: 3_600,
        threshold: 8e-8,
    },
    ClockStep {
        seconds: 1_800,
        threshold: 1.6e-7,
    },
    ClockStep {
        seconds: 600,
        threshold: 5e-7,
    },
    ClockStep {
        seconds: 300,
        threshold: 1e-6,
    },
    ClockStep {
        seconds: 180,
        threshold: 1.6e-6,
    },
    ClockStep {
        seconds: 120,
        threshold: 2.5e-6,
    },
    ClockStep {
        seconds: 60,
        threshold: 5e-6,
    },
    ClockStep {
        seconds: 30,
        threshold: 1e-5,
    },
    ClockStep {
        seconds: 10,
        threshold: 3e-5,
    },
    ClockStep {
        seconds: 5,
        threshold: 6e-5,
    },
    ClockStep {
        seconds: 3,
        threshold: 1e-4,
    },
    ClockStep {
        seconds: 2,
        threshold: 1.5e-4,
    },
    ClockStep {
        seconds: 1,
        threshold: 3e-4,
    },
];

struct FrameStep {
    frames: i64,
    minors: i64,
    /// Pixels per microsecond, in `FrameRate` order. Zero means that rate has no such step.
    threshold: [f64; 8],
}

const FRAMES: [FrameStep; 12] = [
    FrameStep {
        frames: 30,
        minors: 10,
        threshold: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0006, 0.0006],
    },
    FrameStep {
        frames: 25,
        minors: 5,
        threshold: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0006, 0.0, 0.0],
    },
    FrameStep {
        frames: 20,
        minors: 10,
        threshold: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008, 0.0008],
    },
    FrameStep {
        frames: 15,
        minors: 5,
        threshold: [0.0, 0.0, 0.0, 0.0006, 0.0006, 0.0, 0.00085, 0.00085],
    },
    FrameStep {
        frames: 12,
        minors: 6,
        threshold: [0.0006, 0.0006, 0.0, 0.0, 0.0, 0.0, 0.0009, 0.0],
    },
    FrameStep {
        frames: 10,
        minors: 10,
        threshold: [0.0, 0.0, 0.0, 0.0008, 0.0008, 0.0009, 0.001, 0.001],
    },
    FrameStep {
        frames: 8,
        minors: 4,
        threshold: [0.0008, 0.0008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    FrameStep {
        frames: 6,
        minors: 6,
        threshold: [0.0009, 0.0, 0.0, 0.0009, 0.0, 0.0, 0.0013, 0.0],
    },
    FrameStep {
        frames: 5,
        minors: 5,
        threshold: [0.0, 0.0, 0.005, 0.001, 0.001, 0.005, 0.003, 0.003],
    },
    FrameStep {
        frames: 4,
        minors: 4,
        threshold: [0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.006, 0.0],
    },
    FrameStep {
        frames: 3,
        minors: 3,
        threshold: [0.002, 0.002, 0.0, 0.0013, 0.0013, 0.0, 0.009, 0.009],
    },
    FrameStep {
        frames: 2,
        minors: 2,
        threshold: [0.0048, 0.0048, 0.0, 0.006, 0.006, 0.01, 0.012, 0.012],
    },
];

fn rate_index(rate: FrameRate) -> usize {
    match rate {
        FrameRate::Fps23976 => 0,
        FrameRate::Fps24 => 1,
        FrameRate::Fps25 => 2,
        FrameRate::Fps2997 => 3,
        FrameRate::Fps30 => 4,
        FrameRate::Fps50 => 5,
        FrameRate::Fps5994 => 6,
        FrameRate::Fps60 => 7,
    }
}

fn clock_minors(rate: FrameRate) -> i64 {
    match rate {
        FrameRate::Fps23976 | FrameRate::Fps24 => 12,
        FrameRate::Fps25 => 5,
        FrameRate::Fps2997
        | FrameRate::Fps30
        | FrameRate::Fps50
        | FrameRate::Fps5994
        | FrameRate::Fps60 => 10,
    }
}

fn choose(rate: FrameRate, pixel_size: f64) -> Step {
    for step in CLOCK {
        if step.threshold >= pixel_size {
            return Step::Clock {
                seconds: step.seconds,
                minors: clock_minors(rate),
            };
        }
    }
    let index = rate_index(rate);
    for step in FRAMES {
        let threshold = step.threshold[index];
        if threshold > 0.0 && threshold >= pixel_size {
            return Step::Frames {
                frames: step.frames,
                minors: step.minors,
            };
        }
    }
    if rate == FrameRate::Fps25 {
        Step::Frames {
            frames: 5,
            minors: 5,
        }
    } else {
        Step::Frames {
            frames: 2,
            minors: 2,
        }
    }
}

fn minor_frames(rate: FrameRate, step: Step) -> i64 {
    match step {
        Step::Clock { seconds, minors } => i64::from(rate.nominal_fps()) * seconds / minors,
        Step::Frames { frames, minors } => frames / minors,
    }
}

/// Sequence time of a nominal frame, truncating toward zero.
fn frame_us(frame: i64, rate: FrameRate) -> Option<i64> {
    cutlass_engine::frame_time(frame, rate)
}

/// The frame that contains `us`. Tick times use `frame_us`, so a time that is
/// itself a tick maps back to that tick.
fn frame_at(us: i64, rate: FrameRate) -> i64 {
    cutlass_engine::frame_index(us, rate)
}

/// Sequence time of the frame under `us`, clamped to the cut.
pub fn seek(us: i64, rate: FrameRate, duration_us: i64) -> i64 {
    cutlass_engine::snap(us, rate, duration_us)
}

/// Pixels per microsecond for the opening view.
///
/// A cut with a length fills three quarters of the lanes. An empty cut treats
/// one second as its length and puts that second across a third of the lanes.
pub fn opening_pixel_size(lane_width: f32, duration_us: i64) -> f64 {
    if lane_width <= 0.0 {
        return 0.0;
    }
    let (seconds, fraction) = if duration_us > 0 {
        (duration_us as f64 / 1_000_000.0, 0.75)
    } else {
        (1.0, 1.0 / 3.0)
    };
    f64::from(lane_width) * fraction / seconds / 1_000_000.0
}

/// The finest pixels-per-microsecond in the step table for `rate`.
///
/// That is the two-frame step, except 25 fps, which stops at five frames.
pub fn finest_pixel_size(rate: FrameRate) -> f64 {
    let index = rate_index(rate);
    let mut finest = 0.0;
    for step in FRAMES {
        let threshold = step.threshold[index];
        if threshold > finest {
            finest = threshold;
        }
    }
    finest
}

/// Pixels per microsecond when the cut fills a quarter of the lanes.
pub fn quarter_pixel_size(lane_width: f32, duration_us: i64) -> f64 {
    if lane_width <= 0.0 {
        return 0.0;
    }
    let span = if duration_us > 0 {
        duration_us as f64
    } else {
        1_000_000.0
    };
    f64::from(lane_width) * 0.25 / span
}

/// Ticks across `width` pixels, starting at `origin_us`. The loop stops at 1001 marks.
pub fn marks(rate: SequenceRate, pixel_size: f64, width: f32, origin_us: i64) -> Vec<Mark> {
    if !(pixel_size.is_finite() && pixel_size > 0.0 && width.is_finite() && width > 0.0) {
        return Vec::new();
    }
    let step = choose(rate.rate(), pixel_size);
    let spacing = minor_frames(rate.rate(), step);
    if spacing <= 0 {
        return Vec::new();
    }
    let origin = origin_us.max(0);
    let mut out = Vec::new();
    let mut index = frame_at(origin, rate.rate()) / spacing;
    if index < 0 {
        index = 0;
    }
    while out.len() < MARK_CAP {
        let Some(us) = frame_us(index * spacing, rate.rate()) else {
            break;
        };
        let x = (us - origin) as f64 * pixel_size;
        if x > f64::from(width) {
            break;
        }
        if x >= 0.0 {
            let major = index.rem_euclid(step.minors()) == 0;
            let frame = index * spacing;
            out.push(Mark {
                us,
                major,
                label: major.then(|| label(frame, rate, step)),
            });
        }
        index += 1;
    }
    out
}

fn label(frame: i64, rate: SequenceRate, step: Step) -> String {
    if rate.drop_frame() {
        return drop_frame(frame, i64::from(rate.rate().nominal_fps()));
    }
    let nominal = i64::from(rate.rate().nominal_fps());
    if step.frames() && frame.rem_euclid(nominal) != 0 {
        return format!("{}f", frame.rem_euclid(nominal));
    }
    clock(frame / nominal)
}

fn clock(whole_seconds: i64) -> String {
    let whole = whole_seconds.max(0);
    let secs = whole.rem_euclid(60);
    let mins = whole.div_euclid(60).rem_euclid(60);
    let hours = whole.div_euclid(3600);
    if hours >= 1 {
        format!("{hours:02}:{mins:02}:{secs:02}")
    } else {
        format!("{mins:02}:{secs:02}")
    }
}

/// SMPTE drop frame. 29.97 skips two frame numbers a minute, 59.94 skips four,
/// and every tenth minute skips none.
fn drop_frame(frame: i64, nominal: i64) -> String {
    let drop = 2 * nominal / 30;
    let per_minute = nominal * 60 - drop;
    let per_ten = nominal * 60 * 10 - drop * 9;
    let tens = frame.div_euclid(per_ten);
    let within = frame.rem_euclid(per_ten);
    let extra = if within > drop {
        drop * (within - drop).div_euclid(per_minute)
    } else {
        0
    };
    let adjusted = frame + drop * 9 * tens + extra;
    let frames = adjusted.rem_euclid(nominal);
    let seconds = adjusted.div_euclid(nominal).rem_euclid(60);
    let minutes = adjusted.div_euclid(nominal * 60).rem_euclid(60);
    let hours = adjusted.div_euclid(nominal * 60 * 60);
    format!("{hours:02}:{minutes:02}:{seconds:02};{frames:02}")
}

/// `HH:MM:SS:FF`, or `HH:MM:SS;FF` when drop frame is on.
pub(crate) fn timecode(us: i64, rate: SequenceRate) -> String {
    let frame = frame_at(us, rate.rate());
    let nominal = i64::from(rate.rate().nominal_fps());
    if rate.drop_frame() {
        return drop_frame(frame, nominal);
    }
    let frames = frame.rem_euclid(nominal);
    let seconds = frame.div_euclid(nominal).rem_euclid(60);
    let minutes = frame.div_euclid(nominal * 60).rem_euclid(60);
    let hours = frame.div_euclid(nominal * 60 * 60);
    format!("{hours:02}:{minutes:02}:{seconds:02}:{frames:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence(rate: FrameRate) -> SequenceRate {
        rate.sequence()
    }

    #[test]
    fn one_second_is_the_step_at_its_threshold() {
        let step = choose(FrameRate::Fps30, 3e-4);
        assert_eq!(
            step,
            Step::Clock {
                seconds: 1,
                minors: 10
            }
        );
        assert_eq!(
            choose(FrameRate::Fps24, 3e-4),
            Step::Clock {
                seconds: 1,
                minors: 12
            }
        );
        assert_eq!(
            choose(FrameRate::Fps25, 3e-4),
            Step::Clock {
                seconds: 1,
                minors: 5
            }
        );
        let rate = sequence(FrameRate::Fps30);
        assert_eq!(label(0, rate, step), "00:00");
        assert_eq!(label(30, rate, step), "00:01");
        assert_eq!(label(30 * 3600, rate, step), "01:00:00");
    }

    #[test]
    fn a_frame_step_inside_a_second_shows_the_frame_offset() {
        let step = choose(FrameRate::Fps30, 0.0006);
        assert_eq!(
            step,
            Step::Frames {
                frames: 15,
                minors: 5
            }
        );
        let rate = sequence(FrameRate::Fps30);
        assert_eq!(label(0, rate, step), "00:00");
        assert_eq!(label(15, rate, step), "15f");
        assert_eq!(label(30, rate, step), "00:01");
    }

    #[test]
    fn zooming_past_the_table_stops_at_two_frames_except_at_25() {
        assert_eq!(
            choose(FrameRate::Fps30, 1.0),
            Step::Frames {
                frames: 2,
                minors: 2
            }
        );
        assert_eq!(
            choose(FrameRate::Fps60, 1.0),
            Step::Frames {
                frames: 2,
                minors: 2
            }
        );
        assert_eq!(
            choose(FrameRate::Fps25, 1.0),
            Step::Frames {
                frames: 5,
                minors: 5
            }
        );
    }

    #[test]
    fn drop_frame_skips_frame_numbers_except_on_the_tenth_minute() {
        let step = choose(FrameRate::Fps2997, 3e-4);
        let rate = sequence(FrameRate::Fps2997);
        assert!(rate.drop_frame());
        assert_eq!(label(0, rate, step), "00:00:00;00");
        assert_eq!(label(1799, rate, step), "00:00:59;29");
        assert_eq!(label(1800, rate, step), "00:01:00;02");
        assert_eq!(label(17_982, rate, step), "00:10:00;00");
        let fast = sequence(FrameRate::Fps5994);
        assert_eq!(label(3600, fast, step), "00:01:00;04");
    }

    #[test]
    fn visible_ticks_use_sequence_microseconds() {
        let marks = marks(sequence(FrameRate::Fps30), 3e-4, 300.0, 0);
        assert_eq!(marks[0].label.as_deref(), Some("00:00"));
        let second = marks
            .iter()
            .find(|mark| mark.us == 1_000_000)
            .expect("second");
        assert!(second.major);
        assert_eq!(second.label.as_deref(), Some("00:01"));
        assert!(
            marks
                .iter()
                .any(|mark| mark.us == 100_000 && !mark.major && mark.label.is_none())
        );
    }

    #[test]
    fn an_empty_cut_puts_one_second_across_a_third_of_the_lanes() {
        assert!((opening_pixel_size(900.0, 0) - 3e-4).abs() < 1e-12);
        assert!((opening_pixel_size(1000.0, 10_000_000) - 7.5e-5).abs() < 1e-15);
        assert_eq!(opening_pixel_size(0.0, 0), 0.0);
        assert!(marks(sequence(FrameRate::Fps30), 0.0, 300.0, 0).is_empty());
    }

    #[test]
    fn the_playhead_lands_on_the_frame_tick() {
        assert_eq!(seek(0, FrameRate::Fps30, 10_000_000), 0);
        assert_eq!(seek(33_333, FrameRate::Fps30, 10_000_000), 33_333);
        assert_eq!(seek(66_666, FrameRate::Fps30, 10_000_000), 66_666);
        assert_eq!(seek(1_000_000, FrameRate::Fps30, 10_000_000), 1_000_000);
        assert_eq!(seek(2_000_000, FrameRate::Fps30, 1_000_000), 1_000_000);
        assert_eq!(seek(66_666, FrameRate::Fps30, 0), 0);
        assert_eq!(seek(-1, FrameRate::Fps30, 10_000_000), 0);
    }

    #[test]
    fn the_transport_shows_hours_minutes_seconds_and_frames() {
        let rate = sequence(FrameRate::Fps30);
        assert_eq!(timecode(0, rate), "00:00:00:00");
        assert_eq!(timecode(33_333, rate), "00:00:00:01");
        assert_eq!(timecode(1_000_000, rate), "00:00:01:00");
        assert_eq!(
            timecode(66_733, sequence(FrameRate::Fps2997)),
            "00:00:00;02"
        );
        assert_eq!(
            timecode(60_060_000, sequence(FrameRate::Fps2997)),
            "00:01:00;02"
        );
    }

    #[test]
    fn the_painter_stops_at_the_cap() {
        let marks = marks(sequence(FrameRate::Fps30), 1e-9, 20_000.0, 0);
        assert_eq!(marks.len(), 1001);
    }

    #[test]
    fn zoom_stops_at_the_finest_step_and_a_quarter_of_the_lanes() {
        assert!((finest_pixel_size(FrameRate::Fps30) - 0.006).abs() < 1e-12);
        assert!((finest_pixel_size(FrameRate::Fps25) - 0.005).abs() < 1e-12);
        assert!((quarter_pixel_size(400.0, 1_000_000) - 0.0001).abs() < 1e-12);
        let shifted = marks(sequence(FrameRate::Fps30), 3e-4, 300.0, 1_000_000);
        assert_eq!(shifted.first().map(|mark| mark.us), Some(1_000_000));
    }
}
