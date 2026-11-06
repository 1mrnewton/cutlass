//! What each runtime runs: the MobileSAM image encoder once per frame, then
//! the mask decoder once per click.

use std::fmt;

use crate::frame::Frame;

/// The side of the square the encoder sees.
pub const IMAGE_SIZE: usize = 1024;
/// Values in one frame's embeddings: 256 channels on a 64 by 64 grid.
pub const EMBEDDING_LEN: usize = 256 * 64 * 64;
/// The candidate masks SAM offers for a click. Its decoder makes four; the
/// first is for prompts with several points, so a click gets the other three.
pub const MASKS: usize = 3;
/// Each low-res mask is 256 by 256.
pub const MASK_SIDE: usize = 256;

/// One prompt point in the 1024 square, after the frame's resize.
///
/// `label` is 1 for "this object", 0 for "not this", and -1 for SAM's
/// padding point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub label: f32,
}

/// The decoder's answer: a score per candidate, and three 256 by 256 masks
/// of logits, positive inside.
#[derive(Clone, Debug, PartialEq)]
pub struct Masks {
    pub scores: Vec<f32>,
    pub logits: Vec<f32>,
}

impl Masks {
    /// The mask the model scores highest.
    #[must_use]
    pub fn best(&self) -> usize {
        let mut best = 0;
        for (index, score) in self.scores.iter().enumerate() {
            if self.scores.get(best).is_some_and(|top| score > top) {
                best = index;
            }
        }
        best
    }

    /// The logits of one mask.
    #[must_use]
    pub fn mask(&self, index: usize) -> &[f32] {
        let len = MASK_SIDE * MASK_SIDE;
        self.logits
            .get(index * len..(index + 1) * len)
            .unwrap_or_default()
    }
}

/// Why a run failed.
#[derive(Debug)]
pub enum BenchError {
    /// candle refused an operation.
    Candle(candle_core::Error),
    /// A file could not be read or written.
    Io(String),
    /// The input or a result had the wrong size or kind.
    Shape(String),
    /// The command line asked for something this build cannot do.
    Usage(String),
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Candle(error) => write!(f, "candle: {error}"),
            Self::Io(message) | Self::Shape(message) | Self::Usage(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for BenchError {}

impl From<candle_core::Error> for BenchError {
    fn from(error: candle_core::Error) -> Self {
        Self::Candle(error)
    }
}

/// One runtime holding the MobileSAM weights on one device.
pub trait Engine {
    /// The runtime and device, such as `burn wgpu-metal`.
    fn label(&self) -> String;

    /// Runs the image encoder and keeps the embeddings on the device. It
    /// returns once the device has finished.
    fn encode(&mut self, frame: &Frame) -> Result<(), BenchError>;

    /// The embeddings of the last frame, read back to the CPU.
    fn embeddings(&self) -> Result<Vec<f32>, BenchError>;

    /// Runs the decoder on the last frame's embeddings and reads the masks
    /// back to the CPU.
    fn decode(&mut self, points: &[Point]) -> Result<Masks, BenchError>;
}

impl Masks {
    /// Keeps the three click candidates out of the decoder's four masks.
    pub fn from_all_four(scores: &[f32], logits: &[f32]) -> Result<Self, BenchError> {
        let len = MASK_SIDE * MASK_SIDE;
        let all = MASKS + 1;
        if scores.len() != all || logits.len() != all * len {
            return Err(BenchError::Shape(format!(
                "decoder gave {} scores and {} mask values, four masks need {all} and {}",
                scores.len(),
                logits.len(),
                all * len
            )));
        }
        Ok(Self {
            scores: scores.get(1..).unwrap_or_default().to_vec(),
            logits: logits.get(len..).unwrap_or_default().to_vec(),
        })
    }
}
