//! A frame as layers a GPU compositor draws, instead of composed pixels.
//!
//! A video clip with no effect is a decoded 4:2:0 picture and the maps that
//! place it, so the GPU converts and places it. Every other layer, and a
//! clip the GPU path does not cover, is premultiplied RGBA from the CPU
//! path. A recipe effect is a layer of its own that the GPU runs over the
//! layers before it. A frame with any other effect mark or a clip filter is
//! composed on the CPU as before, and its recipe effects still run on the
//! GPU over that picture.

use cutlass_compositor::Frame;
use cutlass_effects::Step;
use cutlass_gpu::{Affine, Placement};
use cutlass_picture::CpuBytes;

/// The pixels one layer draws.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuSource {
    /// Packed Y, then U, then V, limited-range BT.709.
    Yuv {
        width: u32,
        height: u32,
        bytes: Vec<u8>,
    },
    /// Premultiplied RGBA.
    Rgba(CpuBytes),
    /// A compound clip's child frame, drawn onto its own canvas first.
    Nested(Box<FrameLayers>),
    /// A recipe effect over every layer before it, with its values at this
    /// frame's time.
    Effect(Vec<Step>),
}

/// One layer and where it lands.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuLayer {
    pub source: GpuSource,
    pub placement: Placement,
    /// A blurred copy of the source drawn over the whole canvas first.
    pub backdrop: Option<GpuBackdrop>,
}

/// A segment canvas of kind `canvas_blur`: the source scaled to cover the
/// canvas through `cover`, box-blurred by `radius` pixels, not faded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBackdrop {
    pub cover: Placement,
    pub radius: u32,
}

impl GpuLayer {
    /// A CPU picture the size of the canvas.
    pub(crate) fn canvas(picture: CpuBytes) -> Self {
        let placement = Placement::canvas(picture.width(), picture.height());
        Self {
            source: GpuSource::Rgba(picture),
            placement,
            backdrop: None,
        }
    }

    /// An effect over everything below it on a `width` by `height` canvas.
    pub(crate) fn effect(steps: Vec<Step>, width: u32, height: u32) -> Self {
        Self {
            source: GpuSource::Effect(steps),
            placement: Placement::canvas(width, height),
            backdrop: None,
        }
    }

    /// Layers drawn on a canvas of their own, then placed over the whole
    /// frame.
    pub(crate) fn nested(child: FrameLayers) -> Self {
        let placement = Placement::canvas(child.width, child.height);
        Self {
            source: GpuSource::Nested(Box::new(child)),
            placement,
            backdrop: None,
        }
    }

    /// One pixel stretched over the whole canvas.
    pub(crate) fn fill(pixel: CpuBytes) -> Self {
        Self {
            source: GpuSource::Rgba(pixel),
            placement: Placement {
                to_picture: Affine([0.0, 0.0, 0.5, 0.0, 0.0, 0.5]),
                ..Placement::canvas(1, 1)
            },
            backdrop: None,
        }
    }
}

/// Every layer of one frame, back to front, over a transparent canvas.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameLayers {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<GpuLayer>,
}

impl FrameLayers {
    /// A frame composed on the CPU, as one layer the size of the canvas.
    #[must_use]
    pub fn pixels(frame: Frame) -> Option<Self> {
        let picture = CpuBytes::new(frame.extent.width, frame.extent.height, frame.rgba).ok()?;
        Some(Self {
            width: picture.width(),
            height: picture.height(),
            layers: vec![GpuLayer::canvas(picture)],
        })
    }
}

/// A frame as GPU layers, or composed pixels when it needs the CPU.
#[derive(Debug)]
pub enum Composed {
    Layers(FrameLayers),
    Frame(Frame),
}
