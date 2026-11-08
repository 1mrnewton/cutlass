//! Optical flow: RIFE (v4.26) on burn, which makes the frames between two
//! frames of a clip.
//!
//! RIFE guesses how each pixel moved from one frame to the next (the
//! flow), coarse to fine in five steps, and warps both frames to the moment
//! asked for. A mask then picks, pixel by pixel, which warped frame to
//! trust. A small encoder adds four channels of features to each frame
//! first. About 4.5 million weights.
//!
//! A frame is [`prepare`](Interpolator::prepare)d once: padded to the
//! model's grid and encoded. Any number of in-between frames come from two
//! prepared frames, and each frame serves the pair before it and the pair
//! after, so a slowed clip encodes each frame once.
//!
//! The model runs in full precision on the GPU too. In half, a few pixels
//! at moving edges came out tens of levels from PyTorch, for a sixth less
//! time: a flow summed over five steps, and a sampling position near the
//! edge of a 1280 pixel frame, need more than half precision holds.

use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{PaddingConfig2d, PixelShuffle, PixelShuffleConfig};
use burn::tensor::activation::{leaky_relu, sigmoid};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{
    GridSampleOptions, GridSamplePaddingMode, InterpolateMode, InterpolateOptions,
};
use burn::tensor::{Device, Tensor, TensorData};
use burn_store::pytorch::PytorchReader;
use burn_store::{BurnpackStore, ModuleSnapshot};

use crate::device::{self, Compute};
use crate::enhance::to_byte;
use crate::mobile_sam::load_weights;
use crate::{Image, VisionError};

/// The weights, in the folder passed to [`Interpolator::load`].
pub const INTERPOLATE_WEIGHTS: &str = "rife_v4_26.bpk";

/// The longest side a frame may have.
pub const INTERPOLATE_MAX_SIDE: u32 = 4096;

/// Sides are padded to a multiple of this: the coarsest step works at a
/// sixty-fourth of the frame.
const GRID: u32 = 64;

/// The slope of every leaky ReLU.
const SLOPE: f64 = 0.2;

/// How much smaller each step sees the frame, coarse to fine.
const SCALES: [u32; 5] = [16, 8, 4, 2, 1];

/// Each step's input channels and width.
const BLOCKS: [(usize, usize); 5] = [(15, 192), (28, 128), (28, 96), (28, 64), (28, 32)];

/// Feature channels the encoder adds to a frame.
const FEATURES: usize = 4;

/// A convolution and its learned residual weight, RIFE's `ResConv`.
#[derive(Module, Debug)]
struct ResConv {
    conv: Conv2d,
    beta: Param<Tensor<4>>,
}

impl ResConv {
    fn new(width: usize, device: &Device) -> Self {
        Self {
            conv: conv3(width, width, 1, device),
            beta: Param::from_tensor(Tensor::ones([1, width, 1, 1], device)),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        leaky_relu(self.conv.forward(x.clone()) * self.beta.val() + x, SLOPE)
    }
}

/// One step of the flow, RIFE's `IFBlock`.
#[derive(Module, Debug)]
struct Block {
    down0: Conv2d,
    down1: Conv2d,
    res: [ResConv; 8],
    up: ConvTranspose2d,
    #[module(skip)]
    shuffle: PixelShuffle,
}

/// What a step makes: the flow both ways, the mask and its features.
struct Step {
    flow: Tensor<4>,
    mask: Tensor<4>,
    features: Tensor<4>,
}

impl Block {
    fn new(inputs: usize, width: usize, device: &Device) -> Self {
        Self {
            down0: conv3(inputs, width / 2, 2, device),
            down1: conv3(width / 2, width, 2, device),
            res: std::array::from_fn(|_| ResConv::new(width, device)),
            up: ConvTranspose2dConfig::new([width, 4 * 13], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            shuffle: PixelShuffleConfig::new(2).init(),
        }
    }

    /// `x` at the full padded size, read `scale` times smaller.
    fn forward(&self, x: Tensor<4>, flow: Option<Tensor<4>>, scale: u32) -> Step {
        let [_, _, h, w] = x.dims();
        let small = [h / scale as usize, w / scale as usize];
        let mut x = resize(x, small);
        if let Some(flow) = flow {
            let flow = resize(flow, small) / f64::from(scale);
            x = Tensor::cat(vec![x, flow], 1);
        }
        let mut feat = leaky_relu(
            self.down1.forward(leaky_relu(self.down0.forward(x), SLOPE)),
            SLOPE,
        );
        for res in &self.res {
            feat = res.forward(feat);
        }
        let out = resize(self.shuffle.forward(self.up.forward(feat)), [h, w]);
        Step {
            flow: out.clone().slice([0..1, 0..4]) * f64::from(scale),
            mask: out.clone().slice([0..1, 4..5]),
            features: out.slice([0..1, 5..13]),
        }
    }
}

/// The frame encoder, RIFE's `Head`.
#[derive(Module, Debug)]
struct Head {
    cnn0: Conv2d,
    cnn1: Conv2d,
    cnn2: Conv2d,
    cnn3: ConvTranspose2d,
}

impl Head {
    fn new(device: &Device) -> Self {
        Self {
            cnn0: conv3(3, 16, 2, device),
            cnn1: conv3(16, 16, 1, device),
            cnn2: conv3(16, 16, 1, device),
            cnn3: ConvTranspose2dConfig::new([16, FEATURES], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = leaky_relu(self.cnn0.forward(x.clamp(0.0, 1.0)), SLOPE);
        let x = leaky_relu(self.cnn1.forward(x), SLOPE);
        let x = leaky_relu(self.cnn2.forward(x), SLOPE);
        self.cnn3.forward(x)
    }
}

/// RIFE v4.26's `IFNet` and its encoder.
#[derive(Module, Debug)]
struct Rife {
    encode: Head,
    blocks: [Block; 5],
}

impl Rife {
    fn new(device: &Device) -> Self {
        Self {
            encode: Head::new(device),
            blocks: BLOCKS.map(|(inputs, width)| Block::new(inputs, width, device)),
        }
    }
}

fn conv3(from: usize, to: usize, stride: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([from, to], [3, 3])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

/// Bilinear, sampling at pixel centres, as PyTorch's `interpolate` without
/// `align_corners`. The same size is left as it is.
fn resize(x: Tensor<4>, size: [usize; 2]) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    if [h, w] == size {
        return x;
    }
    interpolate(
        x,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_align_corners(false)
            .with_output_size(size),
    )
}

/// One frame ready for the model: padded, with its features.
pub struct Prepared {
    width: u32,
    height: u32,
    image: Tensor<4>,
    features: Tensor<4>,
    /// Each padded pixel's place from -1 to 1, which a warp moves by the
    /// flow.
    grid: Tensor<4>,
}

impl Prepared {
    /// The frame's own size.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// The optical flow model, loaded on one device.
pub struct Interpolator {
    device: Device,
    model: Rife,
}

impl Interpolator {
    /// Loads [`INTERPOLATE_WEIGHTS`] from `weights`.
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let mut model = Rife::new(&device);
        load_weights(&mut model, &weights.join(INTERPOLATE_WEIGHTS))?;
        Ok(Self { device, model })
    }

    /// The model with random weights: the same work as the real one, for
    /// timing and for tests that only check shapes.
    #[cfg(test)]
    pub(crate) fn untrained(compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let model = Rife::new(&device);
        Ok(Self { device, model })
    }

    /// `image` padded with black to the model's grid and encoded.
    pub fn prepare(&self, image: &Image<'_>) -> Result<Prepared, VisionError> {
        let (width, height) = check(image)?;
        let (pw, ph) = (
            width.next_multiple_of(GRID) as usize,
            height.next_multiple_of(GRID) as usize,
        );
        let plane = pw * ph;
        let mut chw = vec![0.0_f32; plane * 3];
        for y in 0..height as usize {
            for x in 0..width as usize {
                for channel in 0..3 {
                    let byte = image
                        .rgb
                        .get((y * width as usize + x) * 3 + channel)
                        .copied()
                        .unwrap_or(0);
                    if let Some(value) = chw.get_mut(channel * plane + y * pw + x) {
                        *value = f32::from(byte) / 255.0;
                    }
                }
            }
        }
        let image = Tensor::<4>::from_data(TensorData::new(chw, [1, 3, ph, pw]), &self.device);
        let features = self.model.encode.forward(image.clone());
        Ok(Prepared {
            width,
            height,
            image,
            features,
            grid: self.grid(pw, ph),
        })
    }

    /// PyTorch's `linspace(-1, 1)` across and down, as two planes.
    fn grid(&self, pw: usize, ph: usize) -> Tensor<4> {
        let at = |i: usize, n: usize| {
            if n < 2 {
                0.0
            } else {
                -1.0 + 2.0 * i as f32 / (n - 1) as f32
            }
        };
        let mut planes = Vec::with_capacity(2 * pw * ph);
        planes.extend((0..ph).flat_map(|_| (0..pw).map(|x| at(x, pw))));
        planes.extend((0..ph).flat_map(|y| (0..pw).map(move |_| at(y, ph))));
        Tensor::<4>::from_data(TensorData::new(planes, [1, 2, ph, pw]), &self.device)
    }

    /// `input` moved by `flow`, two planes of pixels: each pixel reads the
    /// place the flow points to, bilinearly, the edge held past the edge.
    fn warp(&self, input: Tensor<4>, flow: Tensor<4>, grid: &Tensor<4>) -> Tensor<4> {
        let [_, _, h, w] = flow.dims();
        let across = flow.clone().slice([0..1, 0..1]) / ((w as f64 - 1.0) / 2.0).max(1.0);
        let down = flow.slice([0..1, 1..2]) / ((h as f64 - 1.0) / 2.0).max(1.0);
        // Each plane laid out as `[1, h, w, 1]` and joined on the last
        // side: a new tensor in the order `grid_sample_2d` reads. burn's
        // Metal kernel reads a permuted grid as if it were in order.
        let plane = |at: Tensor<4>| at.reshape([1, h, w, 1]);
        let at = Tensor::cat(
            vec![
                plane(grid.clone().slice([0..1, 0..1]) + across),
                plane(grid.clone().slice([0..1, 1..2]) + down),
            ],
            3,
        );
        input.grid_sample_2d(
            at,
            GridSampleOptions::new(InterpolateMode::Bilinear)
                .with_padding_mode(GridSamplePaddingMode::Border)
                .with_align_corners(true),
        )
    }

    /// The frame `at` of the way from `a` to `b`, 0 to 1, as RGB.
    pub fn between(&self, a: &Prepared, b: &Prepared, at: f32) -> Result<Vec<u8>, VisionError> {
        if a.size() != b.size() {
            return Err(VisionError::Model {
                message: format!(
                    "frames of {}x{} and {}x{} cannot be blended",
                    a.width, a.height, b.width, b.height
                ),
            });
        }
        if !(0.0..=1.0).contains(&at) {
            return Err(VisionError::Model {
                message: format!("{at} is not a moment between two frames"),
            });
        }
        let [_, _, ph, pw] = a.image.dims();
        let time = Tensor::<4>::full([1, 1, ph, pw], at, &self.device);
        let (img0, img1) = (a.image.clone(), b.image.clone());
        let (f0, f1) = (a.features.clone(), b.features.clone());
        let mut warped0 = img0.clone();
        let mut warped1 = img1.clone();
        let mut last: Option<Step> = None;
        for (block, scale) in self.model.blocks.iter().zip(SCALES) {
            let step = match last {
                None => block.forward(
                    Tensor::cat(
                        vec![
                            img0.clone(),
                            img1.clone(),
                            f0.clone(),
                            f1.clone(),
                            time.clone(),
                        ],
                        1,
                    ),
                    None,
                    scale,
                ),
                Some(last) => {
                    let wf0 = self.warp(f0.clone(), last.flow.clone().slice([0..1, 0..2]), &a.grid);
                    let wf1 = self.warp(f1.clone(), last.flow.clone().slice([0..1, 2..4]), &a.grid);
                    let input = Tensor::cat(
                        vec![
                            warped0,
                            warped1,
                            wf0,
                            wf1,
                            time.clone(),
                            last.mask,
                            last.features,
                        ],
                        1,
                    );
                    let step = block.forward(input, Some(last.flow.clone()), scale);
                    Step {
                        flow: last.flow + step.flow,
                        ..step
                    }
                }
            };
            warped0 = self.warp(img0.clone(), step.flow.clone().slice([0..1, 0..2]), &a.grid);
            warped1 = self.warp(img1.clone(), step.flow.clone().slice([0..1, 2..4]), &a.grid);
            last = Some(step);
        }
        let mask = last
            .map(|step| sigmoid(step.mask))
            .ok_or_else(|| VisionError::Model {
                message: "the model has no steps".into(),
            })?;
        let blended = warped0 * mask.clone() + warped1 * (mask.neg() + 1.0);
        let values =
            blended
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|error| VisionError::Model {
                    message: format!("{error:?}"),
                })?;
        let plane = pw * ph;
        let mut rgb = Vec::with_capacity(a.width as usize * a.height as usize * 3);
        for y in 0..a.height as usize {
            for x in 0..a.width as usize {
                for channel in 0..3 {
                    let value = values
                        .get(channel * plane + y * pw + x)
                        .copied()
                        .unwrap_or(0.0);
                    rgb.push(to_byte(value));
                }
            }
        }
        Ok(rgb)
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
        || width > INTERPOLATE_MAX_SIDE
        || height > INTERPOLATE_MAX_SIDE
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

/// Reads RIFE v4.26's PyTorch `flownet.pkl`, keeps what makes frames (not
/// the training-only teacher), and writes it as `.bpk` for
/// [`Interpolator::load`]. Run once, offline.
pub fn convert_rife(from: &Path, to: &Path) -> Result<(), VisionError> {
    let refused = |path: &Path, message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !from.is_file() {
        return Err(refused(from, "missing".into()));
    }
    let reader = PytorchReader::new(from).map_err(|error| refused(from, error.to_string()))?;
    let device = Device::flex();
    let read = |name: &str, shape: &[usize]| -> Result<TensorData, VisionError> {
        let tensor = reader
            .get(&format!("module.{name}"))
            .or_else(|| reader.get(name))
            .ok_or_else(|| refused(from, format!("{name} is missing")))?;
        if tensor.shape() != shape || tensor.dtype() != burn_store::pytorch_reader::DType::F32 {
            return Err(refused(
                from,
                format!(
                    "{name} is {:?} {:?}, not f32 {shape:?}",
                    tensor.dtype(),
                    tensor.shape()
                ),
            ));
        }
        let bytes = tensor
            .read()
            .map_err(|error| refused(from, error.to_string()))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(TensorData::new(values, shape.to_vec()))
    };
    let conv = |conv: &mut Conv2d, name: &str| -> Result<(), VisionError> {
        let shape = conv.weight.dims();
        conv.weight = Param::from_tensor(Tensor::from_data(
            read(&format!("{name}.weight"), &shape)?,
            &device,
        ));
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            read(&format!("{name}.bias"), &[shape[0]])?,
            &device,
        )));
        Ok(())
    };
    let transposed = |conv: &mut ConvTranspose2d, name: &str| -> Result<(), VisionError> {
        let shape = conv.weight.dims();
        conv.weight = Param::from_tensor(Tensor::from_data(
            read(&format!("{name}.weight"), &shape)?,
            &device,
        ));
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            read(&format!("{name}.bias"), &[shape[1]])?,
            &device,
        )));
        Ok(())
    };
    let mut model = Rife::new(&device);
    conv(&mut model.encode.cnn0, "encode.cnn0")?;
    conv(&mut model.encode.cnn1, "encode.cnn1")?;
    conv(&mut model.encode.cnn2, "encode.cnn2")?;
    transposed(&mut model.encode.cnn3, "encode.cnn3")?;
    for (index, block) in model.blocks.iter_mut().enumerate() {
        let name = format!("block{index}");
        conv(&mut block.down0, &format!("{name}.conv0.0.0"))?;
        conv(&mut block.down1, &format!("{name}.conv0.1.0"))?;
        for (j, res) in block.res.iter_mut().enumerate() {
            conv(&mut res.conv, &format!("{name}.convblock.{j}.conv"))?;
            let shape = res.beta.dims();
            res.beta = Param::from_tensor(Tensor::from_data(
                read(&format!("{name}.convblock.{j}.beta"), &shape)?,
                &device,
            ));
        }
        transposed(&mut block.up, &format!("{name}.lastconv.0"))?;
    }
    let mut out = BurnpackStore::from_file(to).overwrite(true);
    model
        .save_into(&mut out)
        .map_err(|error| refused(to, error.to_string()))
}

#[cfg(test)]
mod tests;
