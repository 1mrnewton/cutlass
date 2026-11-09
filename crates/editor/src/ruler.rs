//! The timeline ruler: the step table at the draft rate.
//!
//! Pixel size is pixels per microsecond. The steps run coarse to fine, and
//! the ruler keeps the first whose threshold is at least the pixel size.
//! Every step is a whole number of the rate's nominal frames: 24 for
//! 23.976, 30 for 29.97, and 60 for 59.94. A labeled step is split into
//! minor ticks. Labels count nominal frames, so `01:00` at 29.97 sits on
//! frame 1800.

use cutlass_model::{FrameRate, Micros};

/// Clock steps in seconds, coarse to fine, with their thresholds.
const CLOCK_STEPS: [(i64, f64); 18] = [
    (108_000, 2.6e-9),
    (36_000, 8e-9),
    (18_000, 1.6e-8),
    (10_800, 2.6e-8),
    (7_200, 4e-8),
    (3_600, 8e-8),
    (1_800, 1.6e-7),
    (600, 5e-7),
    (300, 1e-6),
    (180, 1.6e-6),
    (120, 2.5e-6),
    (60, 5e-6),
    (30, 1e-5),
    (10, 3e-5),
    (5, 6e-5),
    (3, 1e-4),
    (2, 1.5e-4),
    (1, 3e-4),
];

/// One of the nine project rates.
struct RateSteps {
    fps: f64,
    nominal: i64,
    /// Minor ticks inside a clock step.
    clock_minors: i64,
    /// Frame steps, coarse to fine, with their thresholds.
    frames: &'static [(i64, f64)],
}

const RATES: [RateSteps; 9] = [
    RateSteps {
        fps: 24_000.0 / 1001.0,
        nominal: 24,
        clock_minors: 12,
        frames: &[
            (12, 0.0006),
            (8, 0.0008),
            (6, 0.0009),
            (4, 0.001),
            (3, 0.002),
            (2, 0.0048),
        ],
    },
    RateSteps {
        fps: 24.0,
        nominal: 24,
        clock_minors: 12,
        frames: &[
            (12, 0.0006),
            (8, 0.0008),
            (4, 0.001),
            (3, 0.002),
            (2, 0.0048),
        ],
    },
    RateSteps {
        fps: 25.0,
        nominal: 25,
        clock_minors: 5,
        frames: &[(5, 0.005)],
    },
    RateSteps {
        fps: 30_000.0 / 1001.0,
        nominal: 30,
        clock_minors: 10,
        frames: &[
            (15, 0.0006),
            (10, 0.0008),
            (6, 0.0009),
            (5, 0.001),
            (3, 0.0013),
            (2, 0.006),
        ],
    },
    RateSteps {
        fps: 30.0,
        nominal: 30,
        clock_minors: 10,
        frames: &[
            (15, 0.0006),
            (10, 0.0008),
            (5, 0.001),
            (3, 0.0013),
            (2, 0.006),
        ],
    },
    RateSteps {
        fps: 50.0,
        nominal: 50,
        clock_minors: 10,
        frames: &[(25, 0.0006), (10, 0.0009), (5, 0.005), (2, 0.01)],
    },
    RateSteps {
        fps: 60_000.0 / 1001.0,
        nominal: 60,
        clock_minors: 10,
        frames: &[
            (30, 0.0006),
            (20, 0.0008),
            (15, 0.00085),
            (12, 0.0009),
            (10, 0.001),
            (6, 0.0013),
            (5, 0.003),
            (4, 0.006),
            (3, 0.009),
            (2, 0.012),
        ],
    },
    RateSteps {
        fps: 60.0,
        nominal: 60,
        clock_minors: 10,
        frames: &[
            (30, 0.0006),
            (20, 0.0008),
            (15, 0.00085),
            (10, 0.001),
            (5, 0.003),
            (3, 0.009),
            (2, 0.012),
        ],
    },
    // A frame is half as long as at 60, so each step is twice the frames
    // at the same width, and the finest step zooms in twice as far.
    RateSteps {
        fps: 120.0,
        nominal: 120,
        clock_minors: 10,
        frames: &[
            (60, 0.0006),
            (30, 0.00085),
            (20, 0.001),
            (10, 0.003),
            (6, 0.006),
            (4, 0.009),
            (2, 0.018),
        ],
    },
];

/// Minor ticks inside a frame step of that many frames.
fn frame_minors(frames: i64) -> i64 {
    match frames {
        60 | 30 | 20 | 10 => 10,
        25 | 15 | 5 => 5,
        12 | 6 => 6,
        8 | 4 => 4,
        3 => 3,
        _ => 2,
    }
}

/// The table entry nearest `rate`.
fn steps_for(rate: FrameRate) -> &'static RateSteps {
    let fps = if rate.denominator == 0 {
        30.0
    } else {
        f64::from(rate.numerator) / f64::from(rate.denominator)
    };
    RATES
        .iter()
        .min_by(|a, b| (a.fps - fps).abs().total_cmp(&(b.fps - fps).abs()))
        .unwrap_or(&RATES[4])
}

/// The labeled step the ruler draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RulerStep {
    /// Nominal frames from one label to the next.
    pub frames: i64,
    /// Minor ticks inside the step, the labeled one included.
    pub minors: i64,
    /// A frame step labels a tick inside a second by its frame, such as `15f`.
    pub frame: bool,
}

impl RulerStep {
    /// Nominal frames between two ticks.
    #[must_use]
    pub fn minor_frames(self) -> i64 {
        (self.frames / self.minors).max(1)
    }
}

/// The step for `pixel_size` at `rate`. Past the last threshold the
/// finest step stays.
#[must_use]
pub fn ruler_step(rate: FrameRate, pixel_size: f64) -> RulerStep {
    let table = steps_for(rate);
    let clock = CLOCK_STEPS.iter().map(|&(seconds, threshold)| {
        (
            RulerStep {
                frames: seconds * table.nominal,
                minors: table.clock_minors,
                frame: false,
            },
            threshold,
        )
    });
    let frames = table.frames.iter().map(|&(frames, threshold)| {
        (
            RulerStep {
                frames,
                minors: frame_minors(frames),
                frame: true,
            },
            threshold,
        )
    });
    let mut last = None;
    for (step, threshold) in clock.chain(frames) {
        if threshold >= pixel_size {
            return step;
        }
        last = Some(step);
    }
    last.unwrap_or(RulerStep {
        frames: 2,
        minors: 2,
        frame: true,
    })
}

/// The largest pixel size: the finest step's threshold at `rate`. That is
/// the two-frame step, or five frames at 25.
#[must_use]
pub fn zoom_ceiling(rate: FrameRate) -> f64 {
    let table = steps_for(rate);
    table
        .frames
        .last()
        .map_or(CLOCK_STEPS[CLOCK_STEPS.len() - 1].1, |&(_, threshold)| {
            threshold
        })
}

/// One mark on the ruler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tick {
    pub time: Micros,
    pub major: bool,
    /// The text beside a major tick.
    pub label: Option<String>,
}

/// The most ticks one call returns.
const MAX_TICKS: usize = 2002;

/// The ticks from `from` through `to` at `pixel_size`. Time zero is the
/// first tick. A drop-frame flag on 29.97 or 59.94 labels each major tick
/// `HH:MM:SS;FF`.
#[must_use]
pub fn ruler_ticks(
    rate: FrameRate,
    drop_frame: bool,
    pixel_size: f64,
    from: Micros,
    to: Micros,
) -> Vec<Tick> {
    let mut ticks = Vec::new();
    if !pixel_size.is_finite() || pixel_size <= 0.0 || to.0 < from.0 {
        return ticks;
    }
    let table = steps_for(rate);
    let step = ruler_step(rate, pixel_size);
    let minor = step.minor_frames();
    let drop = drop_frame && rate.dropped_per_minute() > 0;
    let first = rate.frame_index(Micros(from.0.max(0))).unwrap_or(0);
    let mut frame = first - first.rem_euclid(minor);
    while ticks.len() < MAX_TICKS {
        let Some(time) = rate.frame_start(frame) else {
            break;
        };
        if time.0 > to.0 {
            break;
        }
        let major = frame % step.frames == 0;
        let label = major.then(|| {
            if drop {
                rate.label(time, true)
                    .map(|label| label.format())
                    .unwrap_or_default()
            } else if !step.frame || frame % table.nominal == 0 {
                clock_label(frame, table.nominal)
            } else {
                format!("{}f", frame % table.nominal)
            }
        });
        ticks.push(Tick { time, major, label });
        frame += minor;
    }
    ticks
}

/// The transport clock at `time`: whole seconds, `MM:SS` under an hour and
/// `HH:MM:SS` from an hour, so the text keeps its width as time runs. A
/// time before zero reads as zero.
#[must_use]
pub fn clock(time: Micros) -> String {
    clock_text(time.0.max(0) / 1_000_000)
}

/// `MM:SS` under an hour, `HH:MM:SS` from an hour.
fn clock_label(frame: i64, nominal: i64) -> String {
    clock_text(frame / nominal)
}

fn clock_text(whole: i64) -> String {
    let (hours, minutes, seconds) = (whole / 3600, whole / 60 % 60, whole % 60);
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_is_whole_seconds_and_keeps_its_width() {
        assert_eq!(clock(Micros(0)), "00:00");
        assert_eq!(clock(Micros(3_999_999)), "00:03");
        assert_eq!(clock(Micros(17_100_000)), "00:17");
        assert_eq!(clock(Micros(3_599_999_999)), "59:59");
        assert_eq!(clock(Micros(3_600_000_000)), "01:00:00");
        assert_eq!(clock(Micros(-5)), "00:00");
        assert_eq!(
            clock(Micros(i64::MAX)).len(),
            clock(Micros(i64::MAX - 1)).len()
        );
        let widths: std::collections::HashSet<_> = (0..600)
            .map(|tenth| clock(Micros(tenth * 100_000)).len())
            .collect();
        assert_eq!(widths.len(), 1);
    }

    fn labels(ticks: &[Tick]) -> Vec<&str> {
        ticks
            .iter()
            .filter_map(|tick| tick.label.as_deref())
            .collect()
    }

    #[test]
    fn the_first_step_at_or_above_the_zoom_is_drawn() {
        let rate = FrameRate::FPS_30;
        let ten = ruler_step(rate, 2.0e-5);
        assert_eq!((ten.frames, ten.minors, ten.frame), (300, 10, false));
        let second = ruler_step(rate, 3.0e-4);
        assert_eq!(second.frames, 30);
        let fifteen = ruler_step(rate, 5.0e-4);
        assert_eq!(
            (fifteen.frames, fifteen.minors, fifteen.frame),
            (15, 5, true)
        );
        assert_eq!(ruler_step(rate, 1.0).frames, 2);
        assert_eq!(ruler_step(FrameRate::FPS_25, 1.0).frames, 5);
        assert_eq!(ruler_step(rate, 1.0e-12).frames, 108_000 * 30);
    }

    #[test]
    fn the_ceiling_is_the_finest_step() {
        assert_eq!(zoom_ceiling(FrameRate::FPS_30), 0.006);
        assert_eq!(zoom_ceiling(FrameRate::FPS_29_97), 0.006);
        assert_eq!(zoom_ceiling(FrameRate::FPS_24), 0.0048);
        assert_eq!(zoom_ceiling(FrameRate::FPS_25), 0.005);
        assert_eq!(zoom_ceiling(FrameRate::FPS_59_94), 0.012);
    }

    #[test]
    fn a_clock_step_labels_minutes_and_seconds() {
        let ticks = ruler_ticks(
            FrameRate::FPS_30,
            false,
            2.0e-5,
            Micros(0),
            Micros(30_000_000),
        );
        assert_eq!(labels(&ticks), ["00:00", "00:10", "00:20", "00:30"]);
        assert_eq!(ticks.len(), 31);
        assert_eq!(ticks[1].time, Micros(1_000_000));
        assert!(!ticks[1].major);
    }

    #[test]
    fn a_frame_step_names_the_frame_inside_a_second() {
        let ticks = ruler_ticks(
            FrameRate::FPS_24,
            false,
            5.0e-4,
            Micros(0),
            Micros(1_000_000),
        );
        assert_eq!(labels(&ticks), ["00:00", "12f", "00:01"]);
        // A 12-frame step at 24 is a minor tick every 2 frames.
        assert_eq!(ticks[1].time, FrameRate::FPS_24.frame_start(2).unwrap());
    }

    #[test]
    fn an_hour_adds_the_hours() {
        let rate = FrameRate::FPS_30;
        let ticks = ruler_ticks(
            rate,
            false,
            1.0e-6,
            Micros(3_600_000_000),
            Micros(3_600_000_000),
        );
        assert_eq!(labels(&ticks), ["01:00:00"]);
    }

    #[test]
    fn ticks_start_at_the_first_one_in_view() {
        let ticks = ruler_ticks(
            FrameRate::FPS_30,
            false,
            2.0e-5,
            Micros(12_500_000),
            Micros(20_000_000),
        );
        assert_eq!(ticks[0].time, Micros(12_000_000));
        assert_eq!(labels(&ticks), ["00:20"]);
    }

    #[test]
    fn drop_frame_labels_each_major_tick_with_its_timecode() {
        let ticks = ruler_ticks(
            FrameRate::FPS_29_97,
            true,
            2.0e-5,
            Micros(0),
            Micros(10_100_000),
        );
        assert_eq!(labels(&ticks), ["00:00:00;00", "00:00:10;00"]);
        let plain = ruler_ticks(
            FrameRate::FPS_30,
            true,
            2.0e-5,
            Micros(0),
            Micros(10_000_000),
        );
        assert_eq!(labels(&plain), ["00:00", "00:10"]);
    }

    #[test]
    fn a_hundred_and_twenty_counts_its_own_frames() {
        let rate = FrameRate::FPS_120;
        let labels = |pixel_size: f64| -> Vec<(i64, String)> {
            ruler_ticks(rate, false, pixel_size, Micros(0), Micros(1_100_000))
                .into_iter()
                .filter_map(|tick| tick.label.map(|label| (tick.time.0, label)))
                .collect()
        };
        // One second is labeled 00:01, not half a second.
        let seconds = labels(3e-4);
        assert_eq!(seconds[1], (1_000_000, "00:01".to_owned()), "{seconds:?}");
        // A frame step names frames out of 120.
        let frames = labels(0.001);
        assert_eq!(frames[1], (166_667, "20f".to_owned()), "{frames:?}");
        assert!(frames.iter().any(|label| label.1 == "00:01"), "{frames:?}");
        assert!((zoom_ceiling(rate) - 0.018).abs() < 1e-12);
    }
}
