//! Portrait matting: MODNet on burn, which takes the background off a frame
//! with people in it.
//!
//! A MobileNetV2 reads the frame. A low-resolution branch finds the people
//! from its deepest layer, a high-resolution branch finds their edges from
//! its first layers and the frame itself, and a fusion branch joins the two
//! into a soft matte at the size the frame was read. About 6.5 million
//! weights, Apache-2.0. It was trained on portraits: it is made for people,
//! and leaves most other subjects out.
//!
//! The frame is read with its short side [`PORTRAIT_SIDE`], each side a
//! whole number of 32 pixels, as the model was trained.

use std::path::Path;

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Device, Tensor};
use burn_store::{BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::device::{self, Compute};
use crate::matte_frame::{self, resize};
use crate::mobile_sam::load_weights;
use crate::{Image, Matte, Matter, VisionError};

/// The weights, in the folder passed to [`Portrait::load`].
pub const PORTRAIT_WEIGHTS: &str = "modnet.bpk";

/// The short side the model reads a frame at, in pixels.
pub const PORTRAIT_SIDE: u32 = 512;

/// The model and the size it reads at, as [`Matter::id`] gives it.
pub const PORTRAIT_MODEL: &str = "modnet-512";

/// Each side of the frame as read is a whole number of these, the
/// backbone's deepest stride.
const MULTIPLE: u32 = 32;
/// The longest side a frame is read at, so a long thin frame stays cheap.
const LONGEST: u32 = 4 * PORTRAIT_SIDE;
/// The frame is scaled from 0..1 to -1..1.
const MEAN: [f32; 3] = [0.5; 3];
const SPREAD: [f32; 3] = [0.5; 3];
/// MobileNetV2's blocks: expansion, channels, repeats, first stride.
const BLOCKS: [(usize, usize, usize, usize); 7] = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
];
/// The backbone's last layer, and the squeeze in front of it.
const DEEPEST: usize = 1280;
const SQUEEZE: usize = 4;

/// An inverted residual block. Its batch norms are folded into the
/// convolutions.
#[derive(Module, Debug)]
struct Block {
    expand: Option<Conv2d>,
    depthwise: Conv2d,
    project: Conv2d,
}

#[derive(Module, Debug)]
struct Backbone {
    stem: Conv2d,
    blocks: Vec<Block>,
    last: Conv2d,
}

/// Squeeze and excitation over the deepest layer.
#[derive(Module, Debug)]
struct Excite {
    fc1: Linear,
    fc2: Linear,
}

/// A convolution, then half its channels batch-normed and half
/// instance-normed, then ReLU.
#[derive(Module, Debug)]
struct ConvNorm {
    conv: Conv2d,
    bnorm: BatchNorm,
}

#[derive(Module, Debug)]
struct Low {
    conv_lr16x: ConvNorm,
    conv_lr8x: ConvNorm,
}

#[derive(Module, Debug)]
struct High {
    tohr_enc2x: ConvNorm,
    conv_enc2x: ConvNorm,
    tohr_enc4x: ConvNorm,
    conv_enc4x: ConvNorm,
    conv_hr4x: Vec<ConvNorm>,
    conv_hr2x: Vec<ConvNorm>,
}

#[derive(Module, Debug)]
struct Fusion {
    conv_lr4x: ConvNorm,
    conv_f2x: ConvNorm,
    conv_f: ConvNorm,
    out: Conv2d,
}

/// MODNet, named as `tools/vision-models/scripts/portrait_weights.py`
/// names its weights.
#[derive(Module, Debug)]
struct Modnet {
    backbone: Backbone,
    se: Excite,
    lr: Low,
    hr: High,
    fusion: Fusion,
}

fn conv(input: usize, output: usize, kernel: usize, stride: usize, device: &Device) -> Conv2d {
    let pad = kernel / 2;
    Conv2dConfig::new([input, output], [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .init(device)
}

fn conv_norm(
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    device: &Device,
) -> ConvNorm {
    ConvNorm {
        conv: conv(input, output, kernel, stride, device),
        bnorm: BatchNormConfig::new(output / 2).init(device),
    }
}

impl Modnet {
    fn new(device: &Device) -> Self {
        let mut blocks = Vec::new();
        let mut channels = 32;
        for (expansion, output, repeats, stride) in BLOCKS {
            for repeat in 0..repeats {
                let hidden = channels * expansion;
                let stride = if repeat == 0 { stride } else { 1 };
                blocks.push(Block {
                    expand: (expansion != 1).then(|| conv(channels, hidden, 1, 1, device)),
                    depthwise: Conv2dConfig::new([hidden, hidden], [3, 3])
                        .with_stride([stride, stride])
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                        .with_groups(hidden)
                        .init(device),
                    project: conv(hidden, output, 1, 1, device),
                });
                channels = output;
            }
        }
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        Self {
            backbone: Backbone {
                stem: conv(3, 32, 3, 2, device),
                blocks,
                last: conv(channels, DEEPEST, 1, 1, device),
            },
            se: Excite {
                fc1: linear(DEEPEST, DEEPEST / SQUEEZE),
                fc2: linear(DEEPEST / SQUEEZE, DEEPEST),
            },
            lr: Low {
                conv_lr16x: conv_norm(DEEPEST, 96, 5, 1, device),
                conv_lr8x: conv_norm(96, 32, 5, 1, device),
            },
            hr: High {
                tohr_enc2x: conv_norm(16, 32, 1, 1, device),
                conv_enc2x: conv_norm(32 + 3, 32, 3, 2, device),
                tohr_enc4x: conv_norm(24, 32, 1, 1, device),
                conv_enc4x: conv_norm(64, 64, 3, 1, device),
                conv_hr4x: vec![
                    conv_norm(64 + 32 + 3, 64, 3, 1, device),
                    conv_norm(64, 64, 3, 1, device),
                    conv_norm(64, 32, 3, 1, device),
                ],
                conv_hr2x: vec![
                    conv_norm(64, 64, 3, 1, device),
                    conv_norm(64, 32, 3, 1, device),
                    conv_norm(32, 32, 3, 1, device),
                    conv_norm(32, 32, 3, 1, device),
                ],
            },
            fusion: Fusion {
                conv_lr4x: conv_norm(32, 32, 5, 1, device),
                conv_f2x: conv_norm(64, 32, 3, 1, device),
                conv_f: conv_norm(32 + 3, 16, 3, 1, device),
                out: conv(16, 1, 1, 1, device),
            },
        }
    }
}

/// ReLU6, MobileNetV2's activation.
fn relu6(x: Tensor<4>) -> Tensor<4> {
    x.clamp(0.0, 6.0)
}

/// Each channel of `x` normalized over its own pixels, with no learned
/// scale.
fn instance_norm(x: Tensor<4>) -> Tensor<4> {
    let [batch, channels, h, w] = x.dims();
    let flat = x.reshape([batch, channels, h * w]);
    let (var, mean) = flat.clone().var_mean_bias(2);
    ((flat - mean) / (var + 1e-5).sqrt()).reshape([batch, channels, h, w])
}

impl ConvNorm {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(x);
        let [batch, channels, h, w] = x.dims();
        let half = channels / 2;
        let batched = self
            .bnorm
            .forward(x.clone().slice([0..batch, 0..half, 0..h, 0..w]));
        let instanced = instance_norm(x.slice([0..batch, half..channels, 0..h, 0..w]));
        relu(Tensor::cat(vec![batched, instanced], 1))
    }
}

impl Block {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let hidden = match &self.expand {
            Some(expand) => relu6(expand.forward(x.clone())),
            None => x.clone(),
        };
        let y = self.project.forward(relu6(self.depthwise.forward(hidden)));
        if x.dims() == y.dims() { x + y } else { y }
    }
}

/// `x` at twice its size.
fn up(x: Tensor<4>) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    resize(x, [2 * h, 2 * w])
}

impl Modnet {
    /// The share of each pixel kept, `[1, 1, h, w]`, for a normalized
    /// frame whose sides are whole multiples of 32.
    fn forward(&self, img: Tensor<4>) -> Tensor<4> {
        let [_, _, h, w] = img.dims();
        let backbone = &self.backbone;
        let blocks = |x: Tensor<4>, skip: usize, take: usize| {
            backbone
                .blocks
                .iter()
                .skip(skip)
                .take(take)
                .fold(x, |x, block| block.forward(x))
        };
        // The first block's answer is at half the frame's size, the third's
        // at a quarter; both feed the high-resolution branch.
        let enc2x = blocks(relu6(backbone.stem.forward(img.clone())), 0, 1);
        let enc4x = blocks(enc2x.clone(), 1, 2);
        let x = blocks(enc4x.clone(), 3, usize::MAX);
        let enc32x = relu6(backbone.last.forward(x));
        let [batch, channels, _, _] = enc32x.dims();
        let squeezed = enc32x.clone().mean_dims(&[2, 3]).reshape([batch, channels]);
        let weights = sigmoid(self.se.fc2.forward(relu(self.se.fc1.forward(squeezed))));
        let enc32x = enc32x * weights.reshape([batch, channels, 1, 1]);

        let lr = &self.lr;
        let lr16x = lr.conv_lr16x.forward(up(enc32x));
        let lr8x = lr.conv_lr8x.forward(up(lr16x));

        let hr = &self.hr;
        let img2x = resize(img.clone(), [h / 2, w / 2]);
        let img4x = resize(img.clone(), [h / 4, w / 4]);
        let enc2x = hr.tohr_enc2x.forward(enc2x);
        let hr4x = hr
            .conv_enc2x
            .forward(Tensor::cat(vec![img2x, enc2x.clone()], 1));
        let enc4x = hr.tohr_enc4x.forward(enc4x);
        let mut hr4x = hr.conv_enc4x.forward(Tensor::cat(vec![hr4x, enc4x], 1));
        let lr4x = up(lr8x);
        hr4x = Tensor::cat(vec![hr4x, lr4x.clone(), img4x], 1);
        for layer in &hr.conv_hr4x {
            hr4x = layer.forward(hr4x);
        }
        let mut hr2x = Tensor::cat(vec![up(hr4x), enc2x], 1);
        for layer in &hr.conv_hr2x {
            hr2x = layer.forward(hr2x);
        }

        let fusion = &self.fusion;
        let lr2x = up(fusion.conv_lr4x.forward(lr4x));
        let f2x = fusion.conv_f2x.forward(Tensor::cat(vec![lr2x, hr2x], 1));
        let f = fusion.conv_f.forward(Tensor::cat(vec![up(f2x), img], 1));
        sigmoid(fusion.out.forward(f))
    }
}

/// MODNet, loaded on one device.
pub struct Portrait {
    device: Device,
    model: Modnet,
}

impl Portrait {
    /// Loads [`PORTRAIT_WEIGHTS`] from the folder `weights`.
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let mut model = Modnet::new(&device);
        load_weights(&mut model, &weights.join(PORTRAIT_WEIGHTS))?;
        Ok(Self { device, model })
    }

    /// The model with random weights: the same work as the real one, for
    /// timing and for tests that only check shapes.
    #[cfg(test)]
    pub(crate) fn untrained(compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let model = Modnet::new(&device);
        Ok(Self { device, model })
    }

    /// The size a `width` by `height` frame is read at.
    #[must_use]
    pub fn read_size(width: u32, height: u32) -> (u32, u32) {
        matte_frame::read_size(width, height, PORTRAIT_SIDE, MULTIPLE, LONGEST)
    }

    /// The share of each pixel kept, `[1, 1, h, w]` at the size read.
    fn kept(&self, image: &Image<'_>) -> Result<Tensor<4>, VisionError> {
        let (width, height) = matte_frame::check(image)?;
        let size = Self::read_size(width, height);
        let frame = matte_frame::input(image, size, MEAN, SPREAD, &self.device)?;
        Ok(self.model.forward(frame))
    }
}

impl Matter for Portrait {
    fn id(&self) -> &str {
        PORTRAIT_MODEL
    }

    fn matte(&mut self, image: &Image<'_>) -> Result<Matte, VisionError> {
        let kept = self.kept(image)?;
        matte_frame::to_matte(kept, image.width, image.height)
    }
}

/// Reads the safetensors `tools/vision-models/scripts/portrait_weights.py`
/// makes from MODNet's ONNX file, and writes it to `to` as `.bpk` for
/// [`Portrait::load`]. Run once, offline.
pub fn convert_modnet(from: &Path, to: &Path) -> Result<(), VisionError> {
    let refused = |path: &Path, message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !from.is_file() {
        return Err(refused(from, "missing".into()));
    }
    let device = Device::flex();
    let mut model = Modnet::new(&device);
    let mut store = SafetensorsStore::from_file(from)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true);
    let applied = model
        .load_from(&mut store)
        .map_err(|error| refused(from, error.to_string()))?;
    if !applied.errors.is_empty() || !applied.missing.is_empty() {
        return Err(refused(
            from,
            format!(
                "{:?} failed and {:?} are missing",
                applied.errors, applied.missing
            ),
        ));
    }
    let mut out = BurnpackStore::from_file(to).overwrite(true);
    model
        .save_into(&mut out)
        .map_err(|error| refused(to, error.to_string()))
}

#[cfg(test)]
mod tests;
