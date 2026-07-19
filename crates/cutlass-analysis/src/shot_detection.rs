use std::borrow::Borrow;
use std::error::Error;
use std::fmt;

use crate::ConfigError;

/// Largest accepted horizontal sampling bound.
///
/// This limit keeps each reusable feature buffer bounded even when callers
/// analyze unusually large decoded frames.
pub const MAX_SHOT_SAMPLE_WIDTH: usize = 512;

/// Largest accepted vertical sampling bound.
///
/// Together with [`MAX_SHOT_SAMPLE_WIDTH`], this limits one frame's sampled
/// feature storage to 262,144 pixels.
pub const MAX_SHOT_SAMPLE_HEIGHT: usize = 512;

/// Largest accepted luminance histogram size.
pub const MAX_SHOT_HISTOGRAM_BINS: usize = 256;

/// Validation errors for a borrowed RGBA8 frame view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rgba8FrameError {
    /// The declared width was zero.
    ZeroWidth,
    /// The declared height was zero.
    ZeroHeight,
    /// Multiplying the width by four bytes per pixel overflowed `usize`.
    RowBytesOverflow {
        /// Declared frame width in pixels.
        width: usize,
    },
    /// The row stride cannot hold one complete visible RGBA8 row.
    StrideTooSmall {
        /// Declared bytes between consecutive row starts.
        row_stride: usize,
        /// Minimum bytes required for one visible row.
        minimum_row_stride: usize,
    },
    /// Computing the byte offset of the final visible pixel overflowed `usize`.
    RequiredLengthOverflow {
        /// Declared frame height in pixels.
        height: usize,
        /// Declared bytes between consecutive row starts.
        row_stride: usize,
        /// Bytes occupied by visible pixels in one row.
        visible_row_bytes: usize,
    },
    /// The borrowed buffer does not reach the final visible pixel.
    BufferTooShort {
        /// Actual borrowed byte length.
        actual_length: usize,
        /// Minimum byte length implied by the dimensions and row stride.
        required_length: usize,
    },
    /// The frame timestamp was NaN or positive/negative infinity.
    NonFiniteTimestamp,
    /// The frame timestamp was finite but negative.
    NegativeTimestamp,
}

impl fmt::Display for Rgba8FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth => formatter.write_str("RGBA8 frame width must be non-zero"),
            Self::ZeroHeight => formatter.write_str("RGBA8 frame height must be non-zero"),
            Self::RowBytesOverflow { width } => {
                write!(
                    formatter,
                    "RGBA8 row byte count overflows for width {width}"
                )
            }
            Self::StrideTooSmall {
                row_stride,
                minimum_row_stride,
            } => write!(
                formatter,
                "RGBA8 row stride {row_stride} is smaller than required {minimum_row_stride}"
            ),
            Self::RequiredLengthOverflow {
                height,
                row_stride,
                visible_row_bytes,
            } => write!(
                formatter,
                "RGBA8 required length overflows for height {height}, stride {row_stride}, and \
                 visible row size {visible_row_bytes}"
            ),
            Self::BufferTooShort {
                actual_length,
                required_length,
            } => write!(
                formatter,
                "RGBA8 buffer has {actual_length} bytes but requires at least {required_length}"
            ),
            Self::NonFiniteTimestamp => formatter.write_str("RGBA8 frame timestamp must be finite"),
            Self::NegativeTimestamp => {
                formatter.write_str("RGBA8 frame timestamp must be non-negative")
            }
        }
    }
}

impl Error for Rgba8FrameError {}

/// A validated borrowed view of one row-strided RGBA8 frame.
///
/// Construction validates metadata and buffer reachability without copying any
/// pixels. Rows may contain padding; only the first `width * 4` bytes of each
/// row participate in detection. RGB and alpha bytes are interpreted in that
/// order, and no color-space conversion beyond fixed integer luma is assumed.
#[derive(Clone, Copy)]
pub struct Rgba8Frame<'a> {
    bytes: &'a [u8],
    width: usize,
    height: usize,
    row_stride: usize,
    visible_row_bytes: usize,
    timestamp_seconds: f64,
}

impl<'a> Rgba8Frame<'a> {
    /// Validates and borrows an RGBA8 frame.
    ///
    /// The minimum reachable length is
    /// `(height - 1) * row_stride + width * 4`; trailing padding after the last
    /// row is therefore optional.
    ///
    /// # Errors
    ///
    /// Returns [`Rgba8FrameError`] for zero dimensions, arithmetic overflow, a
    /// stride shorter than a visible row, an undersized buffer, or a timestamp
    /// that is non-finite or negative.
    pub fn new(
        bytes: &'a [u8],
        width: usize,
        height: usize,
        row_stride: usize,
        timestamp_seconds: f64,
    ) -> Result<Self, Rgba8FrameError> {
        if width == 0 {
            return Err(Rgba8FrameError::ZeroWidth);
        }
        if height == 0 {
            return Err(Rgba8FrameError::ZeroHeight);
        }

        let visible_row_bytes = width
            .checked_mul(4)
            .ok_or(Rgba8FrameError::RowBytesOverflow { width })?;
        if row_stride < visible_row_bytes {
            return Err(Rgba8FrameError::StrideTooSmall {
                row_stride,
                minimum_row_stride: visible_row_bytes,
            });
        }

        let required_length = (height - 1)
            .checked_mul(row_stride)
            .and_then(|row_offset| row_offset.checked_add(visible_row_bytes))
            .ok_or(Rgba8FrameError::RequiredLengthOverflow {
                height,
                row_stride,
                visible_row_bytes,
            })?;
        if bytes.len() < required_length {
            return Err(Rgba8FrameError::BufferTooShort {
                actual_length: bytes.len(),
                required_length,
            });
        }
        if !timestamp_seconds.is_finite() {
            return Err(Rgba8FrameError::NonFiniteTimestamp);
        }
        if timestamp_seconds < 0.0 {
            return Err(Rgba8FrameError::NegativeTimestamp);
        }

        Ok(Self {
            bytes,
            width,
            height,
            row_stride,
            visible_row_bytes,
            timestamp_seconds,
        })
    }

    /// The original borrowed bytes, including row padding and any trailing data.
    #[must_use]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Visible frame width in pixels.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Visible frame height in pixels.
    #[must_use]
    pub const fn height(self) -> usize {
        self.height
    }

    /// Bytes between consecutive row starts.
    #[must_use]
    pub const fn row_stride(self) -> usize {
        self.row_stride
    }

    /// Bytes occupied by visible RGBA pixels in one row.
    #[must_use]
    pub const fn visible_row_bytes(self) -> usize {
        self.visible_row_bytes
    }

    /// Finite, non-negative presentation timestamp in seconds.
    #[must_use]
    pub const fn timestamp_seconds(self) -> f64 {
        self.timestamp_seconds
    }
}

/// Relative weights for the two normalized scene-change components.
///
/// Weights are normalized by their sum when scoring, so they need not add to
/// one. At least one must be positive. The histogram component measures
/// luminance-distribution movement; the pixel component measures sampled RGBA
/// change after compensating RGB channels for the global mean-luma shift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShotScoreWeights {
    histogram_weight: f32,
    pixel_difference_weight: f32,
}

impl ShotScoreWeights {
    /// Constructs validated relative score weights.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when either weight is non-finite or negative, or
    /// when both weights are zero.
    pub fn new(histogram_weight: f32, pixel_difference_weight: f32) -> Result<Self, ConfigError> {
        if !histogram_weight.is_finite() || histogram_weight < 0.0 {
            return Err(ConfigError::new(
                "histogram_weight",
                "must be finite and non-negative",
            ));
        }
        if !pixel_difference_weight.is_finite() || pixel_difference_weight < 0.0 {
            return Err(ConfigError::new(
                "pixel_difference_weight",
                "must be finite and non-negative",
            ));
        }
        if histogram_weight == 0.0 && pixel_difference_weight == 0.0 {
            return Err(ConfigError::new(
                "score_weights",
                "at least one score weight must be positive",
            ));
        }

        Ok(Self {
            histogram_weight,
            pixel_difference_weight,
        })
    }

    /// Relative weight of normalized luminance-histogram distance.
    #[must_use]
    pub const fn histogram_weight(self) -> f32 {
        self.histogram_weight
    }

    /// Relative weight of brightness-compensated sampled RGBA difference.
    #[must_use]
    pub const fn pixel_difference_weight(self) -> f32 {
        self.pixel_difference_weight
    }
}

impl Default for ShotScoreWeights {
    fn default() -> Self {
        Self {
            histogram_weight: 0.4,
            pixel_difference_weight: 0.6,
        }
    }
}

/// Validated deterministic shot-boundary detection parameters.
///
/// Sampling uses at most `sample_width * sample_height` pixels from each frame.
/// The score is a weighted mean of two components in `[0, 1]`:
///
/// - one-dimensional earth-mover distance between luma histograms, which makes
///   gradual global brightness changes proportional to their magnitude; and
/// - sampled RGBA absolute difference after subtracting the global mean-luma
///   shift from RGB deltas, which discounts a pure additive brightness change.
///
/// A transition is a candidate when `score >= hard_cut_threshold`. It is
/// emitted only when the current timestamp is at least
/// `minimum_shot_duration_seconds` after the first frame or last emitted
/// boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShotDetectionConfig {
    sample_width: usize,
    sample_height: usize,
    luma_histogram_bins: usize,
    score_weights: ShotScoreWeights,
    hard_cut_threshold: f32,
    minimum_shot_duration_seconds: f64,
}

impl ShotDetectionConfig {
    /// Constructs a validated shot-detection configuration.
    ///
    /// Sampling dimensions must be in their corresponding public maximum
    /// ranges, histogram bins must be in `2..=256`, the threshold must be in
    /// `(0, 1]`, and the minimum duration must be finite and non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] naming the first invalid field.
    pub fn new(
        sample_width: usize,
        sample_height: usize,
        luma_histogram_bins: usize,
        score_weights: ShotScoreWeights,
        hard_cut_threshold: f32,
        minimum_shot_duration_seconds: f64,
    ) -> Result<Self, ConfigError> {
        if sample_width == 0 || sample_width > MAX_SHOT_SAMPLE_WIDTH {
            return Err(ConfigError::new(
                "sample_width",
                "must be between 1 and MAX_SHOT_SAMPLE_WIDTH",
            ));
        }
        if sample_height == 0 || sample_height > MAX_SHOT_SAMPLE_HEIGHT {
            return Err(ConfigError::new(
                "sample_height",
                "must be between 1 and MAX_SHOT_SAMPLE_HEIGHT",
            ));
        }
        if !(2..=MAX_SHOT_HISTOGRAM_BINS).contains(&luma_histogram_bins) {
            return Err(ConfigError::new(
                "luma_histogram_bins",
                "must be between 2 and MAX_SHOT_HISTOGRAM_BINS",
            ));
        }
        if !hard_cut_threshold.is_finite() || hard_cut_threshold <= 0.0 || hard_cut_threshold > 1.0
        {
            return Err(ConfigError::new(
                "hard_cut_threshold",
                "must be finite, positive, and at most 1",
            ));
        }
        if !minimum_shot_duration_seconds.is_finite() || minimum_shot_duration_seconds < 0.0 {
            return Err(ConfigError::new(
                "minimum_shot_duration_seconds",
                "must be finite and non-negative",
            ));
        }

        Ok(Self {
            sample_width,
            sample_height,
            luma_histogram_bins,
            score_weights,
            hard_cut_threshold,
            minimum_shot_duration_seconds,
        })
    }

    /// Maximum horizontal samples taken from a frame.
    #[must_use]
    pub const fn sample_width(self) -> usize {
        self.sample_width
    }

    /// Maximum vertical samples taken from a frame.
    #[must_use]
    pub const fn sample_height(self) -> usize {
        self.sample_height
    }

    /// Number of fixed luminance histogram bins.
    #[must_use]
    pub const fn luma_histogram_bins(self) -> usize {
        self.luma_histogram_bins
    }

    /// Relative component weights normalized during scoring.
    #[must_use]
    pub const fn score_weights(self) -> ShotScoreWeights {
        self.score_weights
    }

    /// Inclusive score threshold for a hard-cut candidate.
    #[must_use]
    pub const fn hard_cut_threshold(self) -> f32 {
        self.hard_cut_threshold
    }

    /// Minimum timestamp distance from the current shot's sampled start.
    #[must_use]
    pub const fn minimum_shot_duration_seconds(self) -> f64 {
        self.minimum_shot_duration_seconds
    }

    /// Maximum sampled pixels stored for one frame feature.
    #[must_use]
    pub const fn maximum_sampled_pixels(self) -> usize {
        self.sample_width * self.sample_height
    }
}

impl Default for ShotDetectionConfig {
    fn default() -> Self {
        Self {
            sample_width: 64,
            sample_height: 36,
            luma_histogram_bins: 32,
            score_weights: ShotScoreWeights {
                histogram_weight: 0.4,
                pixel_difference_weight: 0.6,
            },
            hard_cut_threshold: 0.38,
            minimum_shot_duration_seconds: 0.5,
        }
    }
}

/// Why a shot boundary candidate was emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotBoundaryKind {
    /// Same-sized neighboring frames exceeded the configured visual threshold.
    VisualChange,
    /// Neighboring frames had different visible dimensions.
    ///
    /// Dimension changes receive score `1.0`, reset the comparison baseline,
    /// and remain subject to minimum-shot-duration suppression.
    DimensionChange,
}

/// A deterministic boundary between two neighboring sampled frames.
///
/// The boundary timestamp is the current frame's timestamp, meaning the current
/// sampled frame is the first frame of the candidate new shot. The previous and
/// current sampled timestamps are retained explicitly. No synthetic boundary is
/// produced for the first frame or after the final frame.
///
/// `score` lies in `[0, 1]` and is a candidate strength, not a calibrated
/// semantic probability. Boundaries follow input order and therefore have
/// nondecreasing timestamps; duplicate input timestamps are accepted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShotBoundary {
    timestamp_seconds: f64,
    score: f32,
    previous_frame_timestamp_seconds: f64,
    current_frame_timestamp_seconds: f64,
    kind: ShotBoundaryKind,
}

impl ShotBoundary {
    /// Timestamp of the first sampled frame in the candidate new shot.
    #[must_use]
    pub const fn timestamp_seconds(self) -> f64 {
        self.timestamp_seconds
    }

    /// Finite normalized candidate strength in `[0, 1]`.
    #[must_use]
    pub const fn score(self) -> f32 {
        self.score
    }

    /// Timestamp of the sampled frame immediately before the transition.
    #[must_use]
    pub const fn previous_frame_timestamp_seconds(self) -> f64 {
        self.previous_frame_timestamp_seconds
    }

    /// Timestamp of the sampled frame immediately after the transition.
    #[must_use]
    pub const fn current_frame_timestamp_seconds(self) -> f64 {
        self.current_frame_timestamp_seconds
    }

    /// Whether visual statistics or a dimension change caused the boundary.
    #[must_use]
    pub const fn kind(self) -> ShotBoundaryKind {
        self.kind
    }
}

/// Sequence validation errors from shot detection.
#[derive(Debug, Clone, PartialEq)]
pub enum ShotDetectionError {
    /// A frame timestamp preceded the last successfully processed timestamp.
    OutOfOrderTimestamp {
        /// Timestamp of the last successfully processed frame.
        previous_timestamp_seconds: f64,
        /// Timestamp of the rejected frame.
        current_timestamp_seconds: f64,
    },
}

impl fmt::Display for ShotDetectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfOrderTimestamp {
                previous_timestamp_seconds,
                current_timestamp_seconds,
            } => write!(
                formatter,
                "frame timestamp {current_timestamp_seconds} precedes previous timestamp \
                 {previous_timestamp_seconds}"
            ),
        }
    }
}

impl Error for ShotDetectionError {}

#[derive(Debug, Clone, Copy)]
struct FrameMetadata {
    width: usize,
    height: usize,
    timestamp_seconds: f64,
}

#[derive(Debug)]
struct FrameFeatures {
    sampled_rgba: Vec<[u8; 4]>,
    luma_histogram: Vec<u32>,
    luma_sum: u64,
}

impl FrameFeatures {
    fn new(maximum_sampled_pixels: usize, histogram_bins: usize) -> Self {
        Self {
            sampled_rgba: Vec::with_capacity(maximum_sampled_pixels),
            luma_histogram: vec![0; histogram_bins],
            luma_sum: 0,
        }
    }

    fn clear(&mut self) {
        self.sampled_rgba.clear();
        self.luma_histogram.fill(0);
        self.luma_sum = 0;
    }
}

/// Reusable deterministic streaming shot detector.
///
/// Each call processes at most `sample_width * sample_height` pixels and all
/// histogram bins. Time is therefore
/// `O(sampled_pixels + luma_histogram_bins)` per frame and
/// `O(frames * (sampled_pixels + luma_histogram_bins))` overall.
///
/// The detector owns two reusable bounded feature buffers and never retains or
/// copies a full input frame. Storage is
/// `O(sample_width * sample_height + luma_histogram_bins)`, independent of
/// source dimensions and frame count. Emitted boundaries are returned
/// causally, one per [`ShotDetector::push_frame`] call at most.
///
/// Duplicate timestamps are accepted and compared normally. Out-of-order
/// timestamps are rejected without changing detector state. A dimension change
/// becomes a score-`1.0` candidate and resets the visual comparison baseline.
///
/// # Limitations
///
/// This is candidate generation, not semantic shot classification. It compares
/// adjacent supplied samples, so sparse sampling can miss short shots and cannot
/// precisely locate a cut between timestamps. The bounded regular grid can
/// alias fine detail; flashes, abrupt exposure changes, fast motion, and
/// dissolves can still produce false positives or negatives. Fixed integer luma
/// treats RGB bytes as encoded values without transfer-function or color-profile
/// correction. Downstream timeline logic or VLM verification should decide
/// whether a candidate is a meaningful editorial boundary.
#[derive(Debug)]
pub struct ShotDetector {
    config: ShotDetectionConfig,
    previous_metadata: Option<FrameMetadata>,
    current_shot_start_seconds: Option<f64>,
    previous_features: FrameFeatures,
    scratch_features: FrameFeatures,
}

impl ShotDetector {
    /// Creates an empty streaming detector with reusable bounded buffers.
    #[must_use]
    pub fn new(config: ShotDetectionConfig) -> Self {
        let maximum_sampled_pixels = config.maximum_sampled_pixels();
        let histogram_bins = config.luma_histogram_bins;
        Self {
            config,
            previous_metadata: None,
            current_shot_start_seconds: None,
            previous_features: FrameFeatures::new(maximum_sampled_pixels, histogram_bins),
            scratch_features: FrameFeatures::new(maximum_sampled_pixels, histogram_bins),
        }
    }

    /// The validated configuration used by this detector.
    #[must_use]
    pub const fn config(&self) -> ShotDetectionConfig {
        self.config
    }

    /// Logical upper bound on sampled pixel records across both reusable buffers.
    ///
    /// This excludes allocator bookkeeping and equals
    /// `2 * config.maximum_sampled_pixels()`.
    #[must_use]
    pub const fn sampled_pixel_storage_limit(&self) -> usize {
        self.config.maximum_sampled_pixels() * 2
    }

    /// Logical upper bound on histogram bins across both reusable buffers.
    #[must_use]
    pub const fn histogram_bin_storage_limit(&self) -> usize {
        self.config.luma_histogram_bins * 2
    }

    /// Number of sampled pixel records currently retained in reusable buffers.
    ///
    /// This never exceeds [`ShotDetector::sampled_pixel_storage_limit`].
    #[must_use]
    pub fn retained_sampled_pixels(&self) -> usize {
        self.previous_features.sampled_rgba.len() + self.scratch_features.sampled_rgba.len()
    }

    /// Clears sequence history while retaining allocated feature buffers.
    pub fn reset(&mut self) {
        self.previous_metadata = None;
        self.current_shot_start_seconds = None;
        self.previous_features.clear();
        self.scratch_features.clear();
    }

    /// Processes one validated frame and returns at most one boundary.
    ///
    /// The first frame establishes a baseline and returns `Ok(None)`. A visual
    /// candidate is emitted when its score is at least the inclusive configured
    /// threshold and the minimum duration has elapsed. A suppressed candidate
    /// is not delayed; later frames are still compared with their immediate
    /// predecessor.
    ///
    /// # Errors
    ///
    /// Returns [`ShotDetectionError::OutOfOrderTimestamp`] when this frame
    /// precedes the last successfully processed frame. The detector is unchanged
    /// in that case.
    pub fn push_frame(
        &mut self,
        frame: Rgba8Frame<'_>,
    ) -> Result<Option<ShotBoundary>, ShotDetectionError> {
        if let Some(previous) = self.previous_metadata
            && frame.timestamp_seconds < previous.timestamp_seconds
        {
            return Err(ShotDetectionError::OutOfOrderTimestamp {
                previous_timestamp_seconds: previous.timestamp_seconds,
                current_timestamp_seconds: frame.timestamp_seconds,
            });
        }

        extract_features(frame, self.config, &mut self.scratch_features);
        let current_metadata = FrameMetadata {
            width: frame.width,
            height: frame.height,
            timestamp_seconds: frame.timestamp_seconds,
        };

        let Some(previous_metadata) = self.previous_metadata else {
            std::mem::swap(&mut self.previous_features, &mut self.scratch_features);
            self.previous_metadata = Some(current_metadata);
            self.current_shot_start_seconds = Some(frame.timestamp_seconds);
            return Ok(None);
        };

        let (score, kind) =
            if frame.width != previous_metadata.width || frame.height != previous_metadata.height {
                (1.0, ShotBoundaryKind::DimensionChange)
            } else {
                (
                    scene_change_score(
                        &self.previous_features,
                        &self.scratch_features,
                        self.config.score_weights,
                    ),
                    ShotBoundaryKind::VisualChange,
                )
            };

        let duration_elapsed = self.current_shot_start_seconds.is_none_or(|start| {
            frame.timestamp_seconds - start >= self.config.minimum_shot_duration_seconds
        });
        let boundary =
            (score >= self.config.hard_cut_threshold && duration_elapsed).then_some(ShotBoundary {
                timestamp_seconds: frame.timestamp_seconds,
                score,
                previous_frame_timestamp_seconds: previous_metadata.timestamp_seconds,
                current_frame_timestamp_seconds: frame.timestamp_seconds,
                kind,
            });

        std::mem::swap(&mut self.previous_features, &mut self.scratch_features);
        self.previous_metadata = Some(current_metadata);
        if boundary.is_some() {
            self.current_shot_start_seconds = Some(frame.timestamp_seconds);
        }

        Ok(boundary)
    }
}

/// Detects shot boundaries over validated borrowed RGBA8 frame views.
///
/// Both owned iterators and borrowed slices are accepted because iterator items
/// may be either `Rgba8Frame` or `&Rgba8Frame`. One-shot output is identical to
/// feeding the same frames to [`ShotDetector`] in order.
///
/// The first input frame only establishes a baseline, and no synthetic final
/// boundary is appended. Empty and one-frame inputs therefore return an empty
/// vector. Output timestamps are finite and nondecreasing.
///
/// Complexity is
/// `O(frames * (sample_width * sample_height + luma_histogram_bins))` time,
/// bounded detector feature storage, and `O(boundaries)` output storage.
///
/// # Errors
///
/// Returns [`ShotDetectionError::OutOfOrderTimestamp`] and discards accumulated
/// one-shot output if any timestamp is lower than its predecessor. Duplicate
/// timestamps are valid.
pub fn detect_shots<'a, I>(
    frames: I,
    config: ShotDetectionConfig,
) -> Result<Vec<ShotBoundary>, ShotDetectionError>
where
    I: IntoIterator,
    I::Item: Borrow<Rgba8Frame<'a>>,
{
    let mut detector = ShotDetector::new(config);
    let mut boundaries = Vec::new();
    for frame in frames {
        if let Some(boundary) = detector.push_frame(*frame.borrow())? {
            boundaries.push(boundary);
        }
    }
    Ok(boundaries)
}

fn extract_features(
    frame: Rgba8Frame<'_>,
    config: ShotDetectionConfig,
    output: &mut FrameFeatures,
) {
    output.clear();
    debug_assert_eq!(output.luma_histogram.len(), config.luma_histogram_bins);

    let sample_columns = frame.width.min(config.sample_width);
    let sample_rows = frame.height.min(config.sample_height);

    for sample_y in 0..sample_rows {
        let source_y = sample_coordinate(sample_y, frame.height, sample_rows);
        let row_start = source_y * frame.row_stride;
        for sample_x in 0..sample_columns {
            let source_x = sample_coordinate(sample_x, frame.width, sample_columns);
            let pixel_start = row_start + source_x * 4;
            let rgba = [
                frame.bytes[pixel_start],
                frame.bytes[pixel_start + 1],
                frame.bytes[pixel_start + 2],
                frame.bytes[pixel_start + 3],
            ];
            let luma = rgb_luma(rgba);
            let histogram_bin =
                usize::from(luma) * config.luma_histogram_bins / (usize::from(u8::MAX) + 1);

            output.sampled_rgba.push(rgba);
            output.luma_histogram[histogram_bin] += 1;
            output.luma_sum += u64::from(luma);
        }
    }
}

fn sample_coordinate(index: usize, source_length: usize, sample_count: usize) -> usize {
    let denominator = sample_count * 2;
    let multiplier = index * 2 + 1;
    let quotient = source_length / denominator;
    let remainder = source_length % denominator;
    quotient * multiplier + (remainder * multiplier) / denominator
}

fn rgb_luma(rgba: [u8; 4]) -> u8 {
    let weighted =
        77 * u32::from(rgba[0]) + 150 * u32::from(rgba[1]) + 29 * u32::from(rgba[2]) + 128;
    (weighted >> 8) as u8
}

fn scene_change_score(
    previous: &FrameFeatures,
    current: &FrameFeatures,
    weights: ShotScoreWeights,
) -> f32 {
    debug_assert_eq!(previous.sampled_rgba.len(), current.sampled_rgba.len());
    debug_assert_eq!(previous.luma_histogram.len(), current.luma_histogram.len());

    let histogram_distance = histogram_earth_mover_distance(previous, current);
    let pixel_difference = compensated_pixel_difference(previous, current);
    let histogram_weight = f64::from(weights.histogram_weight);
    let pixel_weight = f64::from(weights.pixel_difference_weight);
    let weight_sum = histogram_weight + pixel_weight;

    ((histogram_weight * histogram_distance + pixel_weight * pixel_difference) / weight_sum)
        .clamp(0.0, 1.0) as f32
}

fn histogram_earth_mover_distance(previous: &FrameFeatures, current: &FrameFeatures) -> f64 {
    let sample_count = previous.sampled_rgba.len() as u64;
    let interval_count = previous.luma_histogram.len() - 1;
    let mut cumulative_difference = 0_i64;
    let mut transport = 0_u64;

    for index in 0..interval_count {
        cumulative_difference +=
            i64::from(current.luma_histogram[index]) - i64::from(previous.luma_histogram[index]);
        transport += cumulative_difference.unsigned_abs();
    }

    transport as f64 / (sample_count * interval_count as u64) as f64
}

fn compensated_pixel_difference(previous: &FrameFeatures, current: &FrameFeatures) -> f64 {
    let sample_count = previous.sampled_rgba.len();
    let mean_luma_shift =
        (current.luma_sum as f64 - previous.luma_sum as f64) / sample_count as f64;
    let mut absolute_difference = 0.0_f64;

    for (&previous_rgba, &current_rgba) in previous.sampled_rgba.iter().zip(&current.sampled_rgba) {
        for channel in 0..3 {
            let channel_delta =
                f64::from(current_rgba[channel]) - f64::from(previous_rgba[channel]);
            absolute_difference += (channel_delta - mean_luma_shift).abs();
        }
        absolute_difference += (f64::from(current_rgba[3]) - f64::from(previous_rgba[3])).abs();
    }

    let maximum_uncompensated_difference = sample_count as f64 * 4.0 * f64::from(u8::MAX);
    (absolute_difference / maximum_uncompensated_difference).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_sampling_stays_in_bounds_and_hits_cell_centers() {
        assert_eq!(sample_coordinate(0, 8, 4), 1);
        assert_eq!(sample_coordinate(1, 8, 4), 3);
        assert_eq!(sample_coordinate(2, 8, 4), 5);
        assert_eq!(sample_coordinate(3, 8, 4), 7);
        assert_eq!(sample_coordinate(0, 1, 1), 0);
    }

    #[test]
    fn histogram_distance_has_normalized_endpoints() {
        let mut dark = FrameFeatures::new(1, 4);
        dark.sampled_rgba.push([0; 4]);
        dark.luma_histogram[0] = 1;

        let mut bright = FrameFeatures::new(1, 4);
        bright.sampled_rgba.push([255; 4]);
        bright.luma_histogram[3] = 1;

        assert_eq!(histogram_earth_mover_distance(&dark, &dark), 0.0);
        assert_eq!(histogram_earth_mover_distance(&dark, &bright), 1.0);
    }
}
