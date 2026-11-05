//! Resource limits for untrusted animation files.
//!
//! A file past a limit is refused with [`LottieError::Limit`]; a frame past
//! the geometry limits is refused with the same error. Nothing is cut short
//! quietly.

use crate::LottieError;

/// The largest file read, in bytes.
pub const MAX_BYTES: usize = 16 * 1024 * 1024;
/// Layers in the file, every precomposition counted.
pub const MAX_LAYERS: usize = 4_096;
/// Shape items in the file.
pub const MAX_SHAPES: usize = 100_000;
/// Vertices in one path.
pub const MAX_PATH_POINTS: usize = 20_000;
/// Numbers in the file's values, every key counted.
pub const MAX_NUMBERS: usize = 4_000_000;
/// Keys in one property.
pub const MAX_KEYS: usize = 10_000;
/// Groups inside groups, and precompositions inside precompositions.
pub const MAX_DEPTH: usize = 32;
/// Copies one repeater makes.
pub const MAX_COPIES: usize = 1_000;
/// The width and height of the animation.
pub const MAX_EDGE: f64 = 16_384.0;
/// Layers drawn in one frame, every precomposition instance counted.
pub const MAX_FRAME_LAYERS: usize = 20_000;
/// Fills and strokes drawn in one frame.
pub const MAX_FRAME_DRAWS: usize = 50_000;
/// Line edges in one frame.
pub const MAX_FRAME_EDGES: usize = 1 << 20;

/// What a file has used so far.
#[derive(Debug, Default)]
pub(crate) struct Limits {
    layers: usize,
    shapes: usize,
    numbers: usize,
}

impl Limits {
    pub(crate) fn layers(&mut self, count: usize) -> Result<(), LottieError> {
        self.layers = self.layers.saturating_add(count);
        check(self.layers <= MAX_LAYERS, "layers")
    }

    pub(crate) fn shapes(&mut self, count: usize) -> Result<(), LottieError> {
        self.shapes = self.shapes.saturating_add(count);
        check(self.shapes <= MAX_SHAPES, "shape items")
    }

    pub(crate) fn numbers(&mut self, count: usize) -> Result<(), LottieError> {
        self.numbers = self.numbers.saturating_add(count);
        check(self.numbers <= MAX_NUMBERS, "numbers")
    }

    pub(crate) fn keys(&mut self, count: usize) -> Result<(), LottieError> {
        check(count <= MAX_KEYS, "keys in a property")
    }

    pub(crate) fn path_points(&mut self, count: usize) -> Result<(), LottieError> {
        self.numbers(count.saturating_mul(6))?;
        check(count <= MAX_PATH_POINTS, "points in a path")
    }

    pub(crate) fn depth(depth: usize) -> Result<(), LottieError> {
        check(depth <= MAX_DEPTH, "nesting depth")
    }
}

/// What one frame has drawn so far.
#[derive(Debug, Default)]
pub(crate) struct Budget {
    layers: usize,
    draws: usize,
    edges: usize,
}

impl Budget {
    pub(crate) fn layer(&mut self) -> Result<(), LottieError> {
        self.layers += 1;
        check(self.layers <= MAX_FRAME_LAYERS, "layers in a frame")
    }

    pub(crate) fn draw(&mut self) -> Result<(), LottieError> {
        self.draws += 1;
        check(self.draws <= MAX_FRAME_DRAWS, "draws in a frame")
    }

    pub(crate) fn edges(&mut self, count: usize) -> Result<(), LottieError> {
        self.edges = self.edges.saturating_add(count);
        check(self.edges <= MAX_FRAME_EDGES, "edges in a frame")
    }
}

fn check(within: bool, what: &'static str) -> Result<(), LottieError> {
    if within {
        Ok(())
    } else {
        Err(LottieError::Limit(what))
    }
}
