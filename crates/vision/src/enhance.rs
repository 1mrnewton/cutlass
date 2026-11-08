//! Enhance quality: Real-ESRGAN's compact model (SRVGG) on burn.
//!
//! The model makes a frame four times larger and repairs blur, noise and
//! compression on the way. It is 34 plain 3 by 3 convolutions with a
//! learned slope (PReLU) between them, then a pixel shuffle, added to the
//! frame grown by nearest neighbour. About 1.2 million weights.
//!
//! On the GPU the model runs in half precision, which Apple GPUs do at
//! about twice the rate; a byte comes out the same or one apart.
//!
//! A frame is read in tiles so memory stays flat at any size. Each tile
//! carries [`REACH`] pixels of its neighbours, as far as one output pixel
//! sees, so the tiled frame is the frame the model would make in one go.

use std::path::Path;

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{PRelu, PReluConfig, PaddingConfig2d, PixelShuffle, PixelShuffleConfig};
use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot, PytorchStore};

use crate::device::{self, Compute};
use crate::mobile_sam::load_weights;
use crate::{Image, VisionError};

/// The weights, in the folder passed to [`Enhancer::load`].
pub const ENHANCE_WEIGHTS: &str = "realesr_general_x4v3.bpk";

/// How many times larger each side comes out.
pub const SCALE: u32 = 4;

/// The longest side a frame to enhance may have. A 4K frame comes out 16K.
pub const ENHANCE_MAX_SIDE: u32 = 4096;

const FEATURES: usize = 64;
/// The convolutions between the first and the last.
const BODY: usize = 32;
/// How far an output pixel sees, in input pixels: one per convolution.
pub(crate) const REACH: u32 = BODY as u32 + 2;
/// The side of a tile's own part. Its neighbours' [`REACH`] comes on top.
/// At 1024 a 720p frame is one tile; smaller tiles cost the GPU more.
pub(crate) const TILE: u32 = 1024;

/// One layer of the chain, in the order the PyTorch file numbers them.
#[derive(Module, Debug)]
enum Step {
    Conv(Conv2d),
    Act(PRelu),
}

/// SRVGG compact, as Real-ESRGAN's `SRVGGNetCompact` with 64 features, 32
/// body convolutions and PReLU. `body.N` matches the PyTorch names.
#[derive(Module, Debug)]
struct Srvgg {
    body: Vec<Step>,
    #[module(skip)]
    shuffle: PixelShuffle,
}

impl Srvgg {
    fn new(device: &Device) -> Self {
        let conv = |from: usize, to: usize| {
            Step::Conv(
                Conv2dConfig::new([from, to], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init(device),
            )
        };
        let act = || {
            Step::Act(
                PReluConfig::new()
                    .with_num_parameters(FEATURES)
                    .init(device),
            )
        };
        let mut body = vec![conv(3, FEATURES), act()];
        for _ in 0..BODY {
            body.push(conv(FEATURES, FEATURES));
            body.push(act());
        }
        let grown = (SCALE * SCALE) as usize;
        body.push(conv(FEATURES, 3 * grown));
        Self {
            body,
            shuffle: PixelShuffleConfig::new(SCALE as usize).init(),
        }
    }

    /// `[1, 3, h, w]` RGB from 0 to 1, to `[1, 3, 4h, 4w]`.
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [n, c, h, w] = x.dims();
        let mut out = x.clone();
        for step in &self.body {
            out = match step {
                Step::Conv(conv) => conv.forward(out),
                Step::Act(act) => act.forward(out),
            };
        }
        // Each colour sixteen times over, which the shuffle spreads into a
        // 4 by 4 block: the frame grown by nearest neighbour.
        let grown = (SCALE * SCALE) as usize;
        let base = x
            .reshape([n, c, 1, h, w])
            .repeat_dim(2, grown)
            .reshape([n, c * grown, h, w]);
        self.shuffle.forward(out + base)
    }
}

/// RGB pixels four times larger each way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Enhanced {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// The enhance model, loaded on one device.
pub struct Enhancer {
    device: Device,
    model: Srvgg,
    precision: FloatDType,
}

impl Enhancer {
    /// Loads [`ENHANCE_WEIGHTS`] from `weights`.
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let mut model = Srvgg::new(&device);
        load_weights(&mut model, &weights.join(ENHANCE_WEIGHTS))?;
        Ok(Self::on(device, model, compute))
    }

    /// The model with random weights: the same work as the real one, for
    /// timing and for tests that only check shapes and tiles.
    #[cfg(test)]
    pub(crate) fn untrained(compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let model = Srvgg::new(&device);
        Ok(Self::on(device, model, compute))
    }

    /// Half precision on the GPU, full on the CPU, whose half is slower.
    fn on(device: Device, model: Srvgg, compute: Compute) -> Self {
        let precision = match compute {
            Compute::Gpu => FloatDType::F16,
            Compute::Cpu => FloatDType::F32,
        };
        Self::at(device, model, precision)
    }

    fn at(device: Device, model: Srvgg, precision: FloatDType) -> Self {
        let model = if precision == FloatDType::F32 {
            model
        } else {
            model.map(&mut Precision(precision))
        };
        Self {
            device,
            model,
            precision,
        }
    }

    /// `image` four times larger each way.
    pub fn enhance(&self, image: &Image<'_>) -> Result<Enhanced, VisionError> {
        let (width, height) = check(image)?;
        let (out_w, out_h) = (width * SCALE, height * SCALE);
        let mut rgb = vec![0_u8; out_w as usize * out_h as usize * 3];
        for top in (0..height).step_by(TILE as usize) {
            for left in (0..width).step_by(TILE as usize) {
                let own = Rect {
                    left,
                    top,
                    width: TILE.min(width - left),
                    height: TILE.min(height - top),
                };
                self.tile(image, own, &mut rgb, out_w)?;
            }
        }
        Ok(Enhanced {
            width: out_w,
            height: out_h,
            rgb,
        })
    }

    /// Runs the model on `own` and its neighbours' reach, and writes the
    /// output of `own` alone into `out`, a frame `out_w` wide.
    fn tile(
        &self,
        image: &Image<'_>,
        own: Rect,
        out: &mut [u8],
        out_w: u32,
    ) -> Result<(), VisionError> {
        let read = own.grown(REACH, image.width, image.height);
        let chw = planes(image, read);
        let shape = [1, 3, read.height as usize, read.width as usize];
        let input =
            Tensor::<4>::from_data(TensorData::new(chw, shape), &self.device).cast(self.precision);
        let output = self.model.forward(input).cast(FloatDType::F32);
        let (read_w, read_h) = (read.width * SCALE, read.height * SCALE);
        if output.dims() != [1, 3, read_h as usize, read_w as usize] {
            return Err(VisionError::Model {
                message: format!("the enhancer returned {:?}", output.dims()),
            });
        }
        let values =
            output
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|error| VisionError::Model {
                    message: format!("{error:?}"),
                })?;
        let plane = read_w as usize * read_h as usize;
        let (skip_x, skip_y) = ((own.left - read.left) * SCALE, (own.top - read.top) * SCALE);
        for y in 0..own.height * SCALE {
            let from_row = (skip_y + y) as usize * read_w as usize + skip_x as usize;
            let to_row =
                ((own.top * SCALE + y) as usize * out_w as usize + (own.left * SCALE) as usize) * 3;
            for x in 0..(own.width * SCALE) as usize {
                for channel in 0..3 {
                    let value = values
                        .get(channel * plane + from_row + x)
                        .copied()
                        .unwrap_or(0.0);
                    if let Some(byte) = out.get_mut(to_row + x * 3 + channel) {
                        *byte = to_byte(value);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Reads a PyTorch `.pth` of `SRVGGNetCompact` (its `params`) and writes
/// the weights as `.bpk` for [`Enhancer::load`]. Run once, offline.
pub fn convert_pytorch(from: &Path, to: &Path) -> Result<(), VisionError> {
    let refused = |path: &Path, message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !from.is_file() {
        return Err(refused(from, "missing".into()));
    }
    let mut model = Srvgg::new(&Device::flex());
    // PyTorch calls PReLU's slopes `weight`; burn calls them `alpha`. The
    // slopes sit at the odd places of the chain.
    let mut store = PytorchStore::from_file(from)
        .with_top_level_key("params")
        .with_key_remapping(r"^body\.(\d*[13579])\.weight$", "body.$1.alpha")
        .map_indices_contiguous(false);
    let applied = model
        .load_from(&mut store)
        .map_err(|error| refused(from, error.to_string()))?;
    if !applied.errors.is_empty() || !applied.missing.is_empty() || !applied.unused.is_empty() {
        return Err(refused(
            from,
            format!(
                "{} tensors failed, {} are missing and {} are not the model's",
                applied.errors.len(),
                applied.missing.len(),
                applied.unused.len()
            ),
        ));
    }
    let mut out = BurnpackStore::from_file(to).overwrite(true);
    model
        .save_into(&mut out)
        .map_err(|error| refused(to, error.to_string()))
}

/// Casts every weight to one precision.
pub(crate) struct Precision(pub(crate) FloatDType);

impl ModuleMapper for Precision {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|tensor| tensor.cast(self.0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub(crate) left: u32,
    pub(crate) top: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Rect {
    /// `self` with `by` more on each side, inside a `width` by `height` frame.
    pub(crate) fn grown(self, by: u32, width: u32, height: u32) -> Self {
        let left = self.left.saturating_sub(by);
        let top = self.top.saturating_sub(by);
        let right = (self.left + self.width + by).min(width);
        let bottom = (self.top + self.height + by).min(height);
        Self {
            left,
            top,
            width: right - left,
            height: bottom - top,
        }
    }
}

/// Refuses a frame the model cannot take, before any work.
fn check(image: &Image<'_>) -> Result<(u32, u32), VisionError> {
    let (width, height) = (image.width, image.height);
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3));
    if width == 0
        || height == 0
        || width > ENHANCE_MAX_SIDE
        || height > ENHANCE_MAX_SIDE
        || expected != Some(image.rgb.len())
    {
        return Err(VisionError::Image {
            width,
            height,
            bytes: image.rgb.len(),
        });
    }
    Ok((width, height))
}

/// The pixels of `rect` as three planes from 0 to 1.
fn planes(image: &Image<'_>, rect: Rect) -> Vec<f32> {
    let plane = rect.width as usize * rect.height as usize;
    let mut chw = vec![0.0_f32; plane * 3];
    for y in 0..rect.height as usize {
        let row = (rect.top as usize + y) * image.width as usize + rect.left as usize;
        for x in 0..rect.width as usize {
            for channel in 0..3 {
                let byte = image.rgb.get((row + x) * 3 + channel).copied().unwrap_or(0);
                if let Some(value) = chw.get_mut(channel * plane + y * rect.width as usize + x) {
                    *value = f32::from(byte) / 255.0;
                }
            }
        }
    }
    chw
}

pub(crate) fn to_byte(value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    // In range after the clamp, so the cast keeps the value.
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests;
