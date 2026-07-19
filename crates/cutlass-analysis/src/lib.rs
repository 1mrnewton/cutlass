//! Deterministic, dependency-free media analysis over borrowed buffers.
//!
//! The crate deliberately stops at decoded audio and video boundaries. Callers
//! provide finite, interleaved [`f32`] PCM or validated row-strided RGBA8 frame
//! views; decoder, background-job, and index adapters can therefore use the same
//! analysis without coupling it to an engine or platform codec.
//!
//! Multi-channel audio is measured by averaging the energy of every sample in
//! a frame. Channels are **not** mixed as signed amplitudes first, so
//! opposite-polarity stereo does not cancel into silence.
//!
//! Window boundaries are converted back to seconds from integer frame indices,
//! rather than by repeatedly adding a floating-point hop. This keeps timestamps
//! deterministic and prevents accumulated clock drift.
//!
//! # Complexity and storage
//!
//! The internal rolling-energy pass visits each participating sample a bounded
//! number of times. Each public analysis is `O(samples + windows)` and never
//! copies the PCM. Auxiliary storage is `O(windows)` (plus returned events), not
//! `O(samples)`. Empty PCM is valid and produces empty analysis output.
//!
//! Shot detection samples a configured bounded grid and fixed-bin histogram.
//! It is linear in frames and sampled pixels, retains two bounded feature
//! buffers rather than full frames, and returns deterministic hard-cut
//! candidates intended for later semantic verification.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::error::Error;
use std::fmt;

pub mod moments;
mod shot_detection;

pub use shot_detection::{
    MAX_SHOT_HISTOGRAM_BINS, MAX_SHOT_SAMPLE_HEIGHT, MAX_SHOT_SAMPLE_WIDTH, Rgba8Frame,
    Rgba8FrameError, ShotBoundary, ShotBoundaryKind, ShotDetectionConfig, ShotDetectionError,
    ShotDetector, ShotScoreWeights, detect_shots,
};

/// A failure to construct a validated analysis configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    field: &'static str,
    reason: &'static str,
}

impl ConfigError {
    fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }

    /// The configuration field that failed validation.
    #[must_use]
    pub const fn field(&self) -> &'static str {
        self.field
    }

    /// A stable, human-readable explanation of the failed constraint.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        self.reason
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid analysis configuration `{}`: {}",
            self.field, self.reason
        )
    }
}

impl Error for ConfigError {}

/// A failure to construct a valid seconds-based [`TimeSpan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeSpanError {
    reason: &'static str,
}

impl TimeSpanError {
    fn new(reason: &'static str) -> Self {
        Self { reason }
    }

    /// A stable, human-readable explanation of the failed constraint.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        self.reason
    }
}

impl fmt::Display for TimeSpanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid time span: {}", self.reason)
    }
}

impl Error for TimeSpanError {}

/// Validation errors for borrowed interleaved PCM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioInputError {
    /// The sample rate was zero.
    ZeroSampleRate,
    /// The channel count was zero.
    ZeroChannels,
    /// The sample count was not divisible by the channel count.
    NonFrameAligned {
        /// Number of interleaved samples in the input.
        sample_count: usize,
        /// Declared number of channels per frame.
        channels: usize,
    },
    /// A sample was NaN or positive/negative infinity.
    NonFiniteSample {
        /// Zero-based sample index of the first invalid value.
        index: usize,
    },
}

impl fmt::Display for AudioInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSampleRate => formatter.write_str("sample rate must be non-zero"),
            Self::ZeroChannels => formatter.write_str("channel count must be non-zero"),
            Self::NonFrameAligned {
                sample_count,
                channels,
            } => write!(
                formatter,
                "{sample_count} interleaved samples do not form complete {channels}-channel frames"
            ),
            Self::NonFiniteSample { index } => {
                write!(formatter, "PCM sample at index {index} is not finite")
            }
        }
    }
}

impl Error for AudioInputError {}

/// A validated half-open time span, measured in seconds.
///
/// The span is `[start_seconds, end_seconds)`. Both boundaries are finite and
/// non-negative, and the end is never before the start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeSpan {
    start_seconds: f64,
    end_seconds: f64,
}

impl TimeSpan {
    /// Constructs a validated half-open span from absolute second values.
    ///
    /// # Errors
    ///
    /// Returns [`TimeSpanError`] for non-finite or negative boundaries, or when
    /// `end_seconds < start_seconds`.
    pub fn new(start_seconds: f64, end_seconds: f64) -> Result<Self, TimeSpanError> {
        if !start_seconds.is_finite() || !end_seconds.is_finite() {
            return Err(TimeSpanError::new("boundaries must be finite"));
        }
        if start_seconds < 0.0 || end_seconds < 0.0 {
            return Err(TimeSpanError::new("boundaries must be non-negative"));
        }
        if end_seconds < start_seconds {
            return Err(TimeSpanError::new(
                "end boundary must not precede start boundary",
            ));
        }

        Ok(Self {
            start_seconds,
            end_seconds,
        })
    }

    fn from_frames(start_frame: usize, end_frame: usize, sample_rate: u32) -> Self {
        Self {
            start_seconds: seconds_at_frame(start_frame, sample_rate),
            end_seconds: seconds_at_frame(end_frame, sample_rate),
        }
    }

    /// The inclusive start boundary in seconds.
    #[must_use]
    pub const fn start_seconds(self) -> f64 {
        self.start_seconds
    }

    /// The exclusive end boundary in seconds.
    #[must_use]
    pub const fn end_seconds(self) -> f64 {
        self.end_seconds
    }

    /// The span duration in seconds.
    #[must_use]
    pub fn duration_seconds(self) -> f64 {
        self.end_seconds - self.start_seconds
    }

    /// Whether the span has zero duration.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.start_seconds == self.end_seconds
    }
}

/// One window in a loudness contour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessPoint {
    /// Exact half-open window boundaries derived from PCM frame indices.
    pub span: TimeSpan,
    /// Window RMS in dBFS, clamped to the configured finite floor and `0 dBFS`.
    pub dbfs: f32,
}

/// A detected region of silence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilenceSpan {
    /// Exact half-open boundaries derived from PCM frame indices.
    pub span: TimeSpan,
}

/// A beat/onset candidate detected from a positive short-time energy change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatEvent {
    /// Event time in seconds, mapped from the end frame of the causal window.
    pub time_seconds: f64,
    /// Positive log-energy increase in decibels, always finite.
    pub strength_db: f32,
}

/// A validated borrowed view of finite, interleaved `f32` PCM.
///
/// Construction performs one validation scan but does not copy the samples.
#[derive(Clone, Copy)]
pub struct InterleavedPcm<'a> {
    samples: &'a [f32],
    sample_rate: u32,
    channels: usize,
}

impl<'a> InterleavedPcm<'a> {
    /// Validates and borrows an interleaved PCM slice.
    ///
    /// An empty slice is valid when `sample_rate` and `channels` are non-zero.
    ///
    /// # Errors
    ///
    /// Rejects a zero sample rate, zero channels, a sample count that does not
    /// contain complete frames, or any NaN/infinite sample.
    pub fn new(
        samples: &'a [f32],
        sample_rate: u32,
        channels: usize,
    ) -> Result<Self, AudioInputError> {
        if sample_rate == 0 {
            return Err(AudioInputError::ZeroSampleRate);
        }
        if channels == 0 {
            return Err(AudioInputError::ZeroChannels);
        }
        if !samples.len().is_multiple_of(channels) {
            return Err(AudioInputError::NonFrameAligned {
                sample_count: samples.len(),
                channels,
            });
        }
        if let Some(index) = samples.iter().position(|sample| !sample.is_finite()) {
            return Err(AudioInputError::NonFiniteSample { index });
        }

        Ok(Self {
            samples,
            sample_rate,
            channels,
        })
    }

    /// The original borrowed interleaved samples.
    #[must_use]
    pub const fn samples(self) -> &'a [f32] {
        self.samples
    }

    /// Frames per second.
    #[must_use]
    pub const fn sample_rate(self) -> u32 {
        self.sample_rate
    }

    /// Number of interleaved channels per frame.
    #[must_use]
    pub const fn channels(self) -> usize {
        self.channels
    }

    /// Number of complete PCM frames.
    #[must_use]
    pub fn frame_count(self) -> usize {
        self.samples.len() / self.channels
    }

    /// Exact input duration as `frame_count / sample_rate` seconds.
    #[must_use]
    pub fn duration_seconds(self) -> f64 {
        seconds_at_frame(self.frame_count(), self.sample_rate)
    }

    /// Whether the PCM contains no frames.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.samples.is_empty()
    }

    /// Computes a deterministic windowed-RMS loudness contour.
    ///
    /// Channel energies are averaged, preventing opposite-polarity cancellation.
    /// The final point may cover a shorter partial window. Empty PCM returns an
    /// empty vector.
    ///
    /// Complexity is `O(samples + windows)` time and `O(windows)` output space.
    #[must_use]
    pub fn loudness_contour(self, config: LoudnessConfig) -> Vec<LoudnessPoint> {
        let windows = energy_windows(self, config.window_seconds, config.hop_seconds);

        windows
            .into_iter()
            .map(|window| LoudnessPoint {
                span: TimeSpan::from_frames(window.start_frame, window.end_frame, self.sample_rate),
                dbfs: dbfs_from_mean_square(window.mean_square, config.silence_floor_dbfs),
            })
            .collect()
    }

    /// Detects silence with entry/exit hysteresis, short-gap merging, and a
    /// minimum final duration.
    ///
    /// A region enters silence at or below `threshold_dbfs` and remains silent
    /// until it rises above `threshold_dbfs + hysteresis_db`. Raw silent regions
    /// separated by at most `merge_gap_seconds` are merged *before* the
    /// minimum-duration filter is applied.
    ///
    /// Complexity is `O(samples + windows)` time and `O(windows)` worst-case
    /// auxiliary/output space.
    #[must_use]
    pub fn detect_silence(self, config: SilenceConfig) -> Vec<SilenceSpan> {
        if self.is_empty() {
            return Vec::new();
        }

        let windows = energy_windows(
            self,
            config.loudness.window_seconds,
            config.loudness.hop_seconds,
        );
        let exit_threshold = config.threshold_dbfs + config.hysteresis_db;
        let mut raw_spans = Vec::new();
        let mut silence_start = None;

        for window in windows {
            let dbfs =
                dbfs_from_mean_square(window.mean_square, config.loudness.silence_floor_dbfs);
            let is_silent = if silence_start.is_some() {
                dbfs <= exit_threshold
            } else {
                dbfs <= config.threshold_dbfs
            };

            match (silence_start, is_silent) {
                (None, true) => silence_start = Some(window.start_frame),
                (Some(start_frame), false) => {
                    if window.start_frame > start_frame {
                        raw_spans.push(FrameSpan {
                            start_frame,
                            end_frame: window.start_frame,
                        });
                    }
                    silence_start = None;
                }
                _ => {}
            }
        }

        if let Some(start_frame) = silence_start {
            raw_spans.push(FrameSpan {
                start_frame,
                end_frame: self.frame_count(),
            });
        }

        let mut merged: Vec<FrameSpan> = Vec::with_capacity(raw_spans.len());
        let maximum_gap_frames = config.merge_gap_seconds * f64::from(self.sample_rate);

        for span in raw_spans {
            if let Some(previous) = merged.last_mut() {
                let gap_frames = span.start_frame.saturating_sub(previous.end_frame);
                if gap_frames as f64 <= maximum_gap_frames {
                    previous.end_frame = span.end_frame;
                    continue;
                }
            }
            merged.push(span);
        }

        let minimum_frames = config.minimum_duration_seconds * f64::from(self.sample_rate);
        merged
            .into_iter()
            .filter(|span| span.duration_frames() as f64 >= minimum_frames)
            .map(|span| SilenceSpan {
                span: TimeSpan::from_frames(span.start_frame, span.end_frame, self.sample_rate),
            })
            .collect()
    }

    /// Detects beat/onset candidates from positive short-time log-energy change.
    ///
    /// Each positive change is compared with the mean plus a configurable
    /// standard-deviation multiple over a preceding local history. A minimum
    /// onset strength rejects numerical/tonal ripple, and the refractory
    /// interval suppresses candidates too close to the last emitted event.
    ///
    /// Complexity is `O(samples + windows)` time and `O(windows)` auxiliary
    /// space. Empty PCM and constant energy return no events.
    #[must_use]
    pub fn detect_beats(self, config: BeatConfig) -> Vec<BeatEvent> {
        if self.is_empty() {
            return Vec::new();
        }

        let mut windows = energy_windows(
            self,
            config.loudness.window_seconds,
            config.loudness.hop_seconds,
        );
        let complete_window_frames = duration_to_frames(
            config.loudness.window_seconds,
            self.sample_rate,
            self.frame_count(),
        );
        let complete_window_count = windows
            .iter()
            .position(|window| window.end_frame - window.start_frame < complete_window_frames)
            .unwrap_or(windows.len());
        windows.truncate(complete_window_count);

        let mut previous_dbfs = windows
            .first()
            .map_or(config.loudness.silence_floor_dbfs, |window| {
                dbfs_from_mean_square(window.mean_square, config.loudness.silence_floor_dbfs)
            });
        let mut fluxes = Vec::with_capacity(windows.len());

        for (index, window) in windows.iter().enumerate() {
            let current_dbfs =
                dbfs_from_mean_square(window.mean_square, config.loudness.silence_floor_dbfs);
            let flux = if index == 0 {
                0.0
            } else {
                (current_dbfs - previous_dbfs).max(0.0)
            };
            fluxes.push(flux);
            previous_dbfs = current_dbfs;
        }

        let hop_frames = duration_to_frames(
            config.loudness.hop_seconds,
            self.sample_rate,
            self.frame_count(),
        );
        let history_frames = duration_to_frames(
            config.adaptive_window_seconds,
            self.sample_rate,
            self.frame_count(),
        );
        let history_limit = history_frames.div_ceil(hop_frames).max(1);
        let refractory_frames = config.refractory_seconds * f64::from(self.sample_rate);

        let mut events = Vec::new();
        let mut history_sum = 0.0_f64;
        let mut history_squared_sum = 0.0_f64;
        let mut last_event_frame = None;

        for (index, (&flux, window)) in fluxes.iter().zip(&windows).enumerate() {
            if index > history_limit {
                let expired = f64::from(fluxes[index - history_limit - 1]);
                history_sum -= expired;
                history_squared_sum -= expired * expired;
            }

            let history_count = index.min(history_limit);
            let adaptive_threshold = if history_count == 0 {
                f64::from(config.minimum_onset_db)
            } else {
                let count = history_count as f64;
                let mean = history_sum / count;
                let variance = (history_squared_sum / count - mean * mean).max(0.0);
                (mean + f64::from(config.threshold_stddevs) * variance.sqrt())
                    .max(f64::from(config.minimum_onset_db))
            };

            let flux_f64 = f64::from(flux);
            if flux_f64 >= adaptive_threshold {
                let event_frame = window.end_frame;
                let outside_refractory = last_event_frame.is_none_or(|previous_frame| {
                    event_frame.saturating_sub(previous_frame) as f64 >= refractory_frames
                });

                if outside_refractory {
                    events.push(BeatEvent {
                        time_seconds: seconds_at_frame(event_frame, self.sample_rate),
                        strength_db: flux,
                    });
                    last_event_frame = Some(event_frame);
                }
            }

            history_sum += flux_f64;
            history_squared_sum += flux_f64 * flux_f64;
        }

        events
    }
}

/// Validated window parameters for RMS loudness analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessConfig {
    window_seconds: f64,
    hop_seconds: f64,
    silence_floor_dbfs: f32,
}

impl LoudnessConfig {
    /// Constructs a loudness configuration.
    ///
    /// `hop_seconds` may equal but must not exceed `window_seconds`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] unless window and hop are finite and positive,
    /// and the silence floor is finite and below `0 dBFS`.
    pub fn new(
        window_seconds: f64,
        hop_seconds: f64,
        silence_floor_dbfs: f32,
    ) -> Result<Self, ConfigError> {
        validate_positive_seconds("window_seconds", window_seconds)?;
        validate_positive_seconds("hop_seconds", hop_seconds)?;
        if hop_seconds > window_seconds {
            return Err(ConfigError::new(
                "hop_seconds",
                "must not exceed window_seconds",
            ));
        }
        if !silence_floor_dbfs.is_finite() || silence_floor_dbfs >= 0.0 {
            return Err(ConfigError::new(
                "silence_floor_dbfs",
                "must be finite and below 0 dBFS",
            ));
        }

        Ok(Self {
            window_seconds,
            hop_seconds,
            silence_floor_dbfs,
        })
    }

    /// Analysis window duration in seconds.
    #[must_use]
    pub const fn window_seconds(self) -> f64 {
        self.window_seconds
    }

    /// Distance between consecutive window starts in seconds.
    #[must_use]
    pub const fn hop_seconds(self) -> f64 {
        self.hop_seconds
    }

    /// Finite dBFS value emitted for mathematical zero and lower values.
    #[must_use]
    pub const fn silence_floor_dbfs(self) -> f32 {
        self.silence_floor_dbfs
    }
}

impl Default for LoudnessConfig {
    fn default() -> Self {
        Self {
            window_seconds: 0.050,
            hop_seconds: 0.025,
            silence_floor_dbfs: -120.0,
        }
    }
}

/// Validated silence detection parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilenceConfig {
    loudness: LoudnessConfig,
    threshold_dbfs: f32,
    minimum_duration_seconds: f64,
    merge_gap_seconds: f64,
    hysteresis_db: f32,
}

impl SilenceConfig {
    /// Constructs a silence configuration around validated loudness windows.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when the entry threshold lies outside the
    /// loudness floor/`0 dBFS` range, durations are negative or non-finite,
    /// hysteresis is negative/non-finite, or the hysteresis exit threshold
    /// would exceed `0 dBFS`.
    pub fn new(
        loudness: LoudnessConfig,
        threshold_dbfs: f32,
        minimum_duration_seconds: f64,
        merge_gap_seconds: f64,
        hysteresis_db: f32,
    ) -> Result<Self, ConfigError> {
        if !threshold_dbfs.is_finite()
            || threshold_dbfs < loudness.silence_floor_dbfs
            || threshold_dbfs >= 0.0
        {
            return Err(ConfigError::new(
                "threshold_dbfs",
                "must be finite, at least the silence floor, and below 0 dBFS",
            ));
        }
        validate_non_negative_seconds("minimum_duration_seconds", minimum_duration_seconds)?;
        validate_non_negative_seconds("merge_gap_seconds", merge_gap_seconds)?;
        if !hysteresis_db.is_finite() || hysteresis_db < 0.0 {
            return Err(ConfigError::new(
                "hysteresis_db",
                "must be finite and non-negative",
            ));
        }
        if threshold_dbfs + hysteresis_db > 0.0 {
            return Err(ConfigError::new(
                "hysteresis_db",
                "threshold_dbfs + hysteresis_db must not exceed 0 dBFS",
            ));
        }

        Ok(Self {
            loudness,
            threshold_dbfs,
            minimum_duration_seconds,
            merge_gap_seconds,
            hysteresis_db,
        })
    }

    /// Windowing and finite-floor parameters.
    #[must_use]
    pub const fn loudness(self) -> LoudnessConfig {
        self.loudness
    }

    /// dBFS level at or below which a silent region begins.
    #[must_use]
    pub const fn threshold_dbfs(self) -> f32 {
        self.threshold_dbfs
    }

    /// Minimum duration retained after short-gap merging.
    #[must_use]
    pub const fn minimum_duration_seconds(self) -> f64 {
        self.minimum_duration_seconds
    }

    /// Maximum non-silent gap merged between silent regions.
    #[must_use]
    pub const fn merge_gap_seconds(self) -> f64 {
        self.merge_gap_seconds
    }

    /// Extra dB above the entry threshold required to leave silence.
    #[must_use]
    pub const fn hysteresis_db(self) -> f32 {
        self.hysteresis_db
    }
}

impl Default for SilenceConfig {
    fn default() -> Self {
        Self {
            loudness: LoudnessConfig::default(),
            threshold_dbfs: -50.0,
            minimum_duration_seconds: 0.200,
            merge_gap_seconds: 0.100,
            hysteresis_db: 3.0,
        }
    }
}

/// Validated short-time energy onset/beat detection parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatConfig {
    loudness: LoudnessConfig,
    adaptive_window_seconds: f64,
    threshold_stddevs: f32,
    minimum_onset_db: f32,
    refractory_seconds: f64,
}

impl BeatConfig {
    /// Constructs a beat/onset configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] unless the adaptive window is finite/positive,
    /// the standard-deviation multiplier is finite/non-negative, the minimum
    /// onset is finite/positive, and the refractory duration is
    /// finite/non-negative.
    pub fn new(
        loudness: LoudnessConfig,
        adaptive_window_seconds: f64,
        threshold_stddevs: f32,
        minimum_onset_db: f32,
        refractory_seconds: f64,
    ) -> Result<Self, ConfigError> {
        validate_positive_seconds("adaptive_window_seconds", adaptive_window_seconds)?;
        if !threshold_stddevs.is_finite() || threshold_stddevs < 0.0 {
            return Err(ConfigError::new(
                "threshold_stddevs",
                "must be finite and non-negative",
            ));
        }
        if !minimum_onset_db.is_finite() || minimum_onset_db <= 0.0 {
            return Err(ConfigError::new(
                "minimum_onset_db",
                "must be finite and positive",
            ));
        }
        validate_non_negative_seconds("refractory_seconds", refractory_seconds)?;

        Ok(Self {
            loudness,
            adaptive_window_seconds,
            threshold_stddevs,
            minimum_onset_db,
            refractory_seconds,
        })
    }

    /// Windowing and finite-floor parameters.
    #[must_use]
    pub const fn loudness(self) -> LoudnessConfig {
        self.loudness
    }

    /// Duration of preceding positive changes used for the local threshold.
    #[must_use]
    pub const fn adaptive_window_seconds(self) -> f64 {
        self.adaptive_window_seconds
    }

    /// Standard-deviation multiple added to the local mean threshold.
    #[must_use]
    pub const fn threshold_stddevs(self) -> f32 {
        self.threshold_stddevs
    }

    /// Smallest positive log-energy change accepted as an onset.
    #[must_use]
    pub const fn minimum_onset_db(self) -> f32 {
        self.minimum_onset_db
    }

    /// Minimum time between emitted events.
    #[must_use]
    pub const fn refractory_seconds(self) -> f64 {
        self.refractory_seconds
    }
}

impl Default for BeatConfig {
    fn default() -> Self {
        Self {
            loudness: LoudnessConfig {
                window_seconds: 0.020,
                hop_seconds: 0.010,
                silence_floor_dbfs: -120.0,
            },
            adaptive_window_seconds: 0.500,
            threshold_stddevs: 1.5,
            minimum_onset_db: 3.0,
            refractory_seconds: 0.120,
        }
    }
}

/// Validates PCM and computes a windowed-RMS loudness contour.
///
/// This convenience API performs `O(samples + windows)` work, borrows rather
/// than copies `samples`, and returns an empty vector for valid empty PCM.
///
/// # Errors
///
/// Returns [`AudioInputError`] when the PCM metadata, frame alignment, or
/// sample values are invalid.
pub fn loudness_contour(
    samples: &[f32],
    sample_rate: u32,
    channels: usize,
    config: LoudnessConfig,
) -> Result<Vec<LoudnessPoint>, AudioInputError> {
    let pcm = InterleavedPcm::new(samples, sample_rate, channels)?;
    Ok(pcm.loudness_contour(config))
}

/// Validates PCM and detects silence with hysteresis and short-gap merging.
///
/// This convenience API performs `O(samples + windows)` work, borrows rather
/// than copies `samples`, and returns an empty vector for valid empty PCM.
///
/// # Errors
///
/// Returns [`AudioInputError`] when the PCM metadata, frame alignment, or
/// sample values are invalid.
pub fn detect_silence(
    samples: &[f32],
    sample_rate: u32,
    channels: usize,
    config: SilenceConfig,
) -> Result<Vec<SilenceSpan>, AudioInputError> {
    let pcm = InterleavedPcm::new(samples, sample_rate, channels)?;
    Ok(pcm.detect_silence(config))
}

/// Validates PCM and detects adaptive short-time-energy beat/onset events.
///
/// This convenience API performs `O(samples + windows)` work, borrows rather
/// than copies `samples`, and returns an empty vector for valid empty PCM.
///
/// # Errors
///
/// Returns [`AudioInputError`] when the PCM metadata, frame alignment, or
/// sample values are invalid.
pub fn detect_beats(
    samples: &[f32],
    sample_rate: u32,
    channels: usize,
    config: BeatConfig,
) -> Result<Vec<BeatEvent>, AudioInputError> {
    let pcm = InterleavedPcm::new(samples, sample_rate, channels)?;
    Ok(pcm.detect_beats(config))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct EnergyWindow {
    start_frame: usize,
    end_frame: usize,
    mean_square: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameSpan {
    start_frame: usize,
    end_frame: usize,
}

impl FrameSpan {
    fn duration_frames(self) -> usize {
        self.end_frame.saturating_sub(self.start_frame)
    }
}

fn validate_positive_seconds(field: &'static str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ConfigError::new(field, "must be finite and positive"));
    }
    Ok(())
}

fn validate_non_negative_seconds(field: &'static str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ConfigError::new(field, "must be finite and non-negative"));
    }
    Ok(())
}

fn seconds_at_frame(frame: usize, sample_rate: u32) -> f64 {
    frame as f64 / f64::from(sample_rate)
}

fn duration_to_frames(seconds: f64, sample_rate: u32, frame_count: usize) -> usize {
    if frame_count == 0 {
        return 0;
    }

    let frames = seconds * f64::from(sample_rate);
    if frames >= frame_count as f64 {
        frame_count
    } else {
        (frames.round() as usize).max(1)
    }
}

fn energy_windows(
    pcm: InterleavedPcm<'_>,
    window_seconds: f64,
    hop_seconds: f64,
) -> Vec<EnergyWindow> {
    let frame_count = pcm.frame_count();
    if frame_count == 0 {
        return Vec::new();
    }

    let window_frames = duration_to_frames(window_seconds, pcm.sample_rate, frame_count);
    let hop_frames = duration_to_frames(hop_seconds, pcm.sample_rate, frame_count);
    let window_count = (frame_count - 1) / hop_frames + 1;
    let mut windows = Vec::with_capacity(window_count);

    let mut start_frame = 0;
    let mut end_frame = window_frames.min(frame_count);
    let mut energy_sum = squared_sample_sum(pcm, start_frame, end_frame);

    loop {
        let sample_count = (end_frame - start_frame) * pcm.channels;
        windows.push(EnergyWindow {
            start_frame,
            end_frame,
            mean_square: energy_sum / sample_count as f64,
        });

        let Some(next_start_frame) = start_frame.checked_add(hop_frames) else {
            break;
        };
        if next_start_frame >= frame_count {
            break;
        }
        let next_end_frame = next_start_frame
            .saturating_add(window_frames)
            .min(frame_count);

        if next_start_frame < end_frame {
            energy_sum -= squared_sample_sum(pcm, start_frame, next_start_frame);
            if next_end_frame > end_frame {
                energy_sum += squared_sample_sum(pcm, end_frame, next_end_frame);
            }
            energy_sum = energy_sum.max(0.0);
        } else {
            energy_sum = squared_sample_sum(pcm, next_start_frame, next_end_frame);
        }

        start_frame = next_start_frame;
        end_frame = next_end_frame;
    }

    windows
}

fn squared_sample_sum(pcm: InterleavedPcm<'_>, start_frame: usize, end_frame: usize) -> f64 {
    let start_sample = start_frame * pcm.channels;
    let end_sample = end_frame * pcm.channels;
    pcm.samples[start_sample..end_sample]
        .iter()
        .map(|&sample| {
            let sample = f64::from(sample);
            sample * sample
        })
        .sum()
}

fn dbfs_from_mean_square(mean_square: f64, silence_floor_dbfs: f32) -> f32 {
    if mean_square <= 0.0 {
        return silence_floor_dbfs;
    }

    let raw_dbfs = 10.0 * mean_square.log10();
    if raw_dbfs.is_finite() {
        raw_dbfs.clamp(f64::from(silence_floor_dbfs), 0.0) as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_energy_matches_naive_overlapping_windows() {
        let samples = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.0, -0.75, 0.75];
        let pcm = InterleavedPcm::new(&samples, 10, 2).expect("valid stereo");
        let windows = energy_windows(pcm, 0.3, 0.1);

        assert_eq!(windows.len(), 5);
        for window in windows {
            let naive_sum = squared_sample_sum(pcm, window.start_frame, window.end_frame);
            let naive_count = (window.end_frame - window.start_frame) * pcm.channels;
            let naive_mean = naive_sum / naive_count as f64;
            assert!((window.mean_square - naive_mean).abs() < 1.0e-12);
        }
    }

    #[test]
    fn window_spans_come_from_frame_indices_without_accumulated_hops() {
        let samples = vec![0.25; 11];
        let contour = loudness_contour(
            &samples,
            10,
            1,
            LoudnessConfig::new(0.3, 0.2, -100.0).expect("valid config"),
        )
        .expect("valid PCM");

        assert_eq!(contour.len(), 6);
        assert_eq!(contour[3].span.start_seconds(), 0.6);
        assert_eq!(contour[3].span.end_seconds(), 0.9);
        assert_eq!(contour[5].span.start_seconds(), 1.0);
        assert_eq!(contour[5].span.end_seconds(), 1.1);
    }

    #[test]
    fn time_span_constructor_enforces_invariants() {
        assert!(TimeSpan::new(0.25, 0.5).is_ok());
        assert!(TimeSpan::new(f64::NAN, 1.0).is_err());
        assert!(TimeSpan::new(-0.1, 1.0).is_err());
        assert!(TimeSpan::new(1.0, 0.5).is_err());
    }
}
