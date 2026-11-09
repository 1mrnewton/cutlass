//! A frame as layers a GPU compositor draws, instead of composed pixels.
//!
//! A video clip with no effect is a decoded 4:2:0 picture and the maps that
//! place it, so the GPU converts and places it. Every other layer, and a
//! clip the GPU path does not cover, is premultiplied RGBA from the CPU
//! path. A recipe effect is a layer of its own that the GPU runs over the
//! layers before it, and a clip with a blend mode is drawn alone and then
//! mixed with the layers before it. A clip's colour work, mask and
//! selection run over that clip alone, and a look or an adjustment layer's
//! colour work over the layers before it. A relit clip is drawn upright on
//! a canvas of its own, lit there from its depth, then placed. A frame
//! with an effect package
//! is composed on the CPU as before, and its recipe effects still run on
//! the GPU over that picture.

use std::sync::Arc;

use cutlass_compositor::Frame;
use cutlass_decode::Surface;
use cutlass_effects::{BlendMode, CanvasMask, Colour, Lighting, Step, TransitionKind};
use cutlass_gpu::{Affine, Placement, YuvTone};
use cutlass_picture::CpuBytes;

/// The pixels one layer draws.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuSource {
    /// 8-bit 4:2:0, limited-range BT.709, then `tone`, as the CPU
    /// conversion does.
    Yuv {
        width: u32,
        height: u32,
        /// The decoded picture, shared with the clip's cache.
        samples: YuvSamples,
        tone: YuvTone,
    },
    /// Premultiplied RGBA.
    Rgba(CpuBytes),
    /// A compound clip's child frame, drawn onto its own canvas first.
    Nested(Box<FrameLayers>),
    /// A recipe effect over every layer before it, with its values at this
    /// frame's time, shown only inside its mask when it has one.
    Effect(Vec<Step>, Option<CanvasMask>),
    /// Colour work (a chroma key, grades, adjust sliders, a LUT) over every
    /// layer before it, its grain at `seed`, shown only inside its mask
    /// when it has one.
    Colour {
        colour: Box<Colour>,
        seed: u32,
        mask: Option<CanvasMask>,
    },
    /// Every layer before it, kept as much as a clip's mask keeps.
    Cut(CanvasMask),
    /// Every layer before it, kept only where a clip's selection is.
    Select(SelectionCut),
    /// Every layer before it, lit again by `lighting` from `depth`, a grey
    /// picture nearer when lighter, stretched to the canvas. Only on a
    /// relit clip's own upright canvas.
    Relight { depth: CpuBytes, lighting: Lighting },
    /// A clip drawn on a canvas of its own, then mixed with every layer
    /// before it by `mode`.
    Blend {
        mode: BlendMode,
        layers: Box<FrameLayers>,
    },
    /// A Lottie frame as geometry, drawn on a canvas of the scene's size
    /// by the renderer's vector pass, then placed like a picture.
    Vector(Arc<cutlass_lottie::Scene>),
    /// Moment `t` of a built-in transition between two canvas-sized
    /// premultiplied pictures.
    Transition {
        kind: TransitionKind,
        t: f32,
        outgoing: CpuBytes,
        incoming: CpuBytes,
    },
}

/// A clip's selection placed on the canvas.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionCut {
    pub width: u32,
    pub height: u32,
    /// One byte a pixel, nonzero selected, row by row.
    pub alpha: Vec<u8>,
    /// Keep everything but the object.
    pub invert: bool,
    /// A canvas pixel centre to the clip point it shows, -1 to 1 across
    /// the clip with y up, as `(a·x + b·y + c, d·x + e·y + f)`.
    pub frame: [f32; 6],
}

/// Where a 4:2:0 layer's samples are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum YuvSamples {
    /// Packed Y, then U, then V.
    Bytes(Arc<Vec<u8>>),
    /// A platform GPU buffer. A renderer that cannot import it draws the
    /// bytes the surface copies out.
    Surface(Surface),
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
    pub(crate) fn effect(
        steps: Vec<Step>,
        mask: Option<CanvasMask>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            source: GpuSource::Effect(steps, mask),
            placement: Placement::canvas(width, height),
            backdrop: None,
        }
    }

    /// Colour work over everything below it on a `width` by `height`
    /// canvas.
    pub(crate) fn colour(
        colour: Colour,
        seed: u32,
        mask: Option<CanvasMask>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            source: GpuSource::Colour {
                colour: Box::new(colour),
                seed,
                mask,
            },
            placement: Placement::canvas(width, height),
            backdrop: None,
        }
    }

    /// Everything below it, cut by `source` on a `width` by `height`
    /// canvas.
    pub(crate) fn cut(source: GpuSource, width: u32, height: u32) -> Self {
        Self {
            source,
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

    /// `child`, a clip on a canvas of its own, mixed by `mode` with what is
    /// below.
    pub(crate) fn blend(mode: BlendMode, child: FrameLayers) -> Self {
        let placement = Placement::canvas(child.width, child.height);
        Self {
            source: GpuSource::Blend {
                mode,
                layers: Box::new(child),
            },
            placement,
            backdrop: None,
        }
    }

    /// A transition between two canvas-sized pictures.
    pub(crate) fn transition(
        kind: TransitionKind,
        t: f32,
        outgoing: CpuBytes,
        incoming: CpuBytes,
    ) -> Self {
        let placement = Placement::canvas(outgoing.width(), outgoing.height());
        Self {
            source: GpuSource::Transition {
                kind,
                t,
                outgoing,
                incoming,
            },
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
