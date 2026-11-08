//! Reduce image noise: FastDVDnet on burn.
//!
//! FastDVDnet denoises a frame from it and the two frames each side of it,
//! so the noise it takes out does not flicker from frame to frame. It is
//! two of the same small U-Net, one after the other: the first cleans each
//! run of three frames, and the second cleans three of those results into
//! the frame. Both are also told how strong the noise is, as a level the
//! caller picks. About 2.5 million weights.
//!
//! The first stage of one frame is the first stage of the next, moved on
//! by one, so a [`DenoiseStream`] keeps the last three and runs each stage
//! once a frame, half the work of running the model fresh.
//!
//! The original model normalises after each convolution (batch norm). The
//! converter folds those into the convolutions, which leaves plain
//! convolutions that run in half precision on the GPU.
//!
//! A frame is read in tiles so memory stays flat at any size. Each stage
//! carries [`REACH`] pixels of its neighbours, so the tiled frame is the
//! frame the model would make in one go.

use std::collections::VecDeque;
use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{PaddingConfig2d, PixelShuffle, PixelShuffleConfig};
use burn::tensor::activation::relu;
use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use burn_store::pytorch::PytorchReader;
use burn_store::{BurnpackStore, ModuleSnapshot};

use crate::device::{self, Compute};
use crate::enhance::{Precision, Rect, to_byte};
use crate::mobile_sam::load_weights;
use crate::{Image, VisionError};

/// The weights, in the folder passed to [`Denoiser::load`].
pub const DENOISE_WEIGHTS: &str = "fastdvdnet_clipped.bpk";

/// The longest side a frame to denoise may have.
pub const DENOISE_MAX_SIDE: u32 = 4096;

/// The noise levels the model learned, as the spread of the noise in 8-bit
/// steps. A level outside them is refused.
pub const MIN_NOISE: f32 = 5.0;
pub const MAX_NOISE: f32 = 55.0;

/// How far an output pixel of one stage sees, in input pixels, rounded up
/// to a multiple of four so tiles keep the model's grid.
pub(crate) const REACH: u32 = 48;
/// The side of a tile's own part. A 1080p frame is one tile.
pub(crate) const TILE: u32 = 2048;
/// Sides go in multiples of this through the model's two halvings.
const GRID: u32 = 4;

/// Frames on each side of the one denoised.
const SIDE_FRAMES: usize = 2;
/// The decoded frames kept: the five a frame is made from.
const HELD: usize = 2 * SIDE_FRAMES + 1;

/// One convolution of a stage and the batch norm folded into it.
struct Layer {
    /// Where PyTorch keeps it, inside `temp1` or `temp2`.
    path: &'static str,
    from: usize,
    to: usize,
    stride: usize,
    groups: usize,
    /// The batch norm after it. A layer with one is followed by a ReLU.
    norm: Option<&'static str>,
}

const fn layer(
    path: &'static str,
    from: usize,
    to: usize,
    stride: usize,
    groups: usize,
    norm: Option<&'static str>,
) -> Layer {
    Layer {
        path,
        from,
        to,
        stride,
        groups,
        norm,
    }
}

/// The convolutions of one stage, in the order it runs them.
const LAYERS: [Layer; 16] = [
    // Three frames, each with the noise level beside it, read apart.
    layer("inc.convblock.0", 12, 90, 1, 3, Some("inc.convblock.1")),
    layer("inc.convblock.3", 90, 32, 1, 1, Some("inc.convblock.4")),
    layer(
        "downc0.convblock.0",
        32,
        64,
        2,
        1,
        Some("downc0.convblock.1"),
    ),
    layer(
        "downc0.convblock.3.convblock.0",
        64,
        64,
        1,
        1,
        Some("downc0.convblock.3.convblock.1"),
    ),
    layer(
        "downc0.convblock.3.convblock.3",
        64,
        64,
        1,
        1,
        Some("downc0.convblock.3.convblock.4"),
    ),
    layer(
        "downc1.convblock.0",
        64,
        128,
        2,
        1,
        Some("downc1.convblock.1"),
    ),
    layer(
        "downc1.convblock.3.convblock.0",
        128,
        128,
        1,
        1,
        Some("downc1.convblock.3.convblock.1"),
    ),
    layer(
        "downc1.convblock.3.convblock.3",
        128,
        128,
        1,
        1,
        Some("downc1.convblock.3.convblock.4"),
    ),
    layer(
        "upc2.convblock.0.convblock.0",
        128,
        128,
        1,
        1,
        Some("upc2.convblock.0.convblock.1"),
    ),
    layer(
        "upc2.convblock.0.convblock.3",
        128,
        128,
        1,
        1,
        Some("upc2.convblock.0.convblock.4"),
    ),
    layer("upc2.convblock.1", 128, 256, 1, 1, None),
    layer(
        "upc1.convblock.0.convblock.0",
        64,
        64,
        1,
        1,
        Some("upc1.convblock.0.convblock.1"),
    ),
    layer(
        "upc1.convblock.0.convblock.3",
        64,
        64,
        1,
        1,
        Some("upc1.convblock.0.convblock.4"),
    ),
    layer("upc1.convblock.1", 64, 128, 1, 1, None),
    layer("outc.convblock.0", 32, 32, 1, 1, Some("outc.convblock.1")),
    layer("outc.convblock.3", 32, 3, 1, 1, None),
];

impl Layer {
    fn init(&self, device: &Device) -> Conv2d {
        Conv2dConfig::new([self.from, self.to], [3, 3])
            .with_stride([self.stride, self.stride])
            .with_groups(self.groups)
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device)
    }
}

/// One stage: FastDVDnet's `DenBlock` with its batch norms folded.
#[derive(Module, Debug)]
struct Stage {
    convs: [Conv2d; 16],
    #[module(skip)]
    shuffle: PixelShuffle,
}

impl Stage {
    fn new(device: &Device) -> Self {
        Self {
            convs: LAYERS.each_ref().map(|layer| layer.init(device)),
            shuffle: PixelShuffleConfig::new(2).init(),
        }
    }

    /// Three `[1, 3, h, w]` frames and the `[1, 1, h, w]` noise level, to
    /// the middle frame cleaned. `h` and `w` are multiples of [`GRID`].
    fn forward(&self, frames: [Tensor<4>; 3], noise: Tensor<4>) -> Tensor<4> {
        let [
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
            c10,
            c11,
            c12,
            c13,
            c14,
            c15,
        ] = &self.convs;
        let on = |conv: &Conv2d, x: Tensor<4>| relu(conv.forward(x));
        let [a, b, c] = frames;
        let middle = b.clone();
        let x = Tensor::cat(vec![a, noise.clone(), b, noise.clone(), c, noise], 1);
        let x0 = on(c1, on(c0, x));
        let x1 = on(c4, on(c3, on(c2, x0.clone())));
        let x2 = on(c7, on(c6, on(c5, x1.clone())));
        let x2 = self.shuffle.forward(c10.forward(on(c9, on(c8, x2))));
        let x1 = self.shuffle.forward(c13.forward(on(c12, on(c11, x1 + x2))));
        let x = c15.forward(on(c14, x0 + x1));
        middle - x
    }
}

/// FastDVDnet: `temp1` and `temp2`, as PyTorch names them.
#[derive(Module, Debug)]
struct FastDvd {
    temp1: Stage,
    temp2: Stage,
}

impl FastDvd {
    fn new(device: &Device) -> Self {
        Self {
            temp1: Stage::new(device),
            temp2: Stage::new(device),
        }
    }
}

/// The denoise model, loaded on one device.
pub struct Denoiser {
    device: Device,
    model: FastDvd,
    precision: FloatDType,
    tile: u32,
}

impl Denoiser {
    /// Loads [`DENOISE_WEIGHTS`] from `weights`.
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let mut model = FastDvd::new(&device);
        load_weights(&mut model, &weights.join(DENOISE_WEIGHTS))?;
        Ok(Self::on(device, model, compute))
    }

    /// The model with random weights: the same work as the real one, for
    /// timing and for tests that only check shapes and tiles.
    #[cfg(test)]
    pub(crate) fn untrained(compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let model = FastDvd::new(&device);
        Ok(Self::on(device, model, compute))
    }

    /// Half precision on the GPU, full on the CPU, whose half is slower.
    fn on(device: Device, model: FastDvd, compute: Compute) -> Self {
        let precision = match compute {
            Compute::Gpu => FloatDType::F16,
            Compute::Cpu => FloatDType::F32,
        };
        Self::at(device, model, precision)
    }

    fn at(device: Device, model: FastDvd, precision: FloatDType) -> Self {
        let model = if precision == FloatDType::F32 {
            model
        } else {
            model.map(&mut Precision(precision))
        };
        Self {
            device,
            model,
            precision,
            tile: TILE,
        }
    }

    /// Denoises a clip of `frames` frames, each `width` by `height`, at
    /// `noise`: from [`MIN_NOISE`] to [`MAX_NOISE`], the spread of the
    /// noise to take out in 8-bit steps.
    pub fn stream(
        &self,
        frames: usize,
        width: u32,
        height: u32,
        noise: f32,
    ) -> Result<DenoiseStream<'_>, VisionError> {
        if width == 0 || height == 0 || width > DENOISE_MAX_SIDE || height > DENOISE_MAX_SIDE {
            return Err(VisionError::Image {
                width,
                height,
                bytes: 0,
            });
        }
        if !(MIN_NOISE..=MAX_NOISE).contains(&noise) {
            return Err(VisionError::Model {
                message: format!(
                    "a noise level of {noise} is outside the {MIN_NOISE} to {MAX_NOISE} the model learned"
                ),
            });
        }
        if frames == 0 {
            return Err(VisionError::Model {
                message: "a clip with no frames has nothing to denoise".into(),
            });
        }
        let mut tiles = Vec::new();
        for top in (0..height).step_by(self.tile as usize) {
            for left in (0..width).step_by(self.tile as usize) {
                let own = Rect {
                    left,
                    top,
                    width: self.tile.min(width - left),
                    height: self.tile.min(height - top),
                };
                tiles.push(Tile::new(own, width, height));
            }
        }
        Ok(DenoiseStream {
            denoiser: self,
            frames,
            width,
            height,
            noise: noise / 255.0,
            next: 0,
            tiles,
            held: VecDeque::with_capacity(HELD + 1),
        })
    }

    fn input(&self, values: Vec<f32>, channels: usize, rect: (u32, u32)) -> Tensor<4> {
        let shape = [1, channels, rect.1 as usize, rect.0 as usize];
        Tensor::<4>::from_data(TensorData::new(values, shape), &self.device).cast(self.precision)
    }

    fn noise(&self, level: f32, size: (u32, u32)) -> Tensor<4> {
        Tensor::<4>::full(
            [1, 1, size.1 as usize, size.0 as usize],
            level,
            &self.device,
        )
        .cast(self.precision)
    }
}

/// Where one tile reads and keeps, and the first stage of its last three
/// runs of frames.
struct Tile {
    /// What it writes.
    own: Rect,
    /// What the first stage reads: `own` and twice the reach around it,
    /// inside the frame.
    read: Rect,
    /// `read` grown to the model's grid at the frame's right and bottom,
    /// by reflection, as FastDVDnet pads a frame.
    padded: (u32, u32),
    /// What the second stage reads, inside `padded`: `own` and the reach
    /// around it.
    keep: Rect,
    /// The first stage of the runs of frames the next frame is made from.
    first: VecDeque<Tensor<4>>,
}

impl Tile {
    fn new(own: Rect, width: u32, height: u32) -> Self {
        let read = own.grown(2 * REACH, width, height);
        let padded = (
            read.width.next_multiple_of(GRID),
            read.height.next_multiple_of(GRID),
        );
        // A side at the frame's edge keeps all it has, padding too; a side
        // inside the frame keeps the reach around `own`, which the first
        // stage got right since it read twice that.
        let keep_side = |own_start: u32, own_len: u32, read_start: u32, frame: u32, padded: u32| {
            let start = (own_start - read_start).saturating_sub(REACH);
            let end = if read_start + padded >= frame {
                padded
            } else {
                own_start + own_len + REACH - read_start
            };
            (start, end - start)
        };
        let (left, kept_w) = keep_side(own.left, own.width, read.left, width, padded.0);
        let (top, kept_h) = keep_side(own.top, own.height, read.top, height, padded.1);
        Self {
            own,
            read,
            padded,
            keep: Rect {
                left,
                top,
                width: kept_w,
                height: kept_h,
            },
            first: VecDeque::with_capacity(4),
        }
    }
}

/// A clip being denoised, frame by frame in order.
pub struct DenoiseStream<'a> {
    denoiser: &'a Denoiser,
    frames: usize,
    width: u32,
    height: u32,
    /// From 0 to 1, as the model reads it.
    noise: f32,
    next: usize,
    tiles: Vec<Tile>,
    /// Decoded frames by their place in the clip, RGB.
    held: VecDeque<(usize, Vec<u8>)>,
}

impl DenoiseStream<'_> {
    /// How many frames the clip has.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames
    }

    /// Whether it has none; never, as a stream is refused without frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }

    /// The next frame denoised, as RGB, or `None` after the last. `source`
    /// gives frame `k` of the clip as RGB; it is asked for each frame about
    /// once, in order, with a few asked again near the clip's ends.
    pub fn next(
        &mut self,
        source: &mut dyn FnMut(usize) -> Result<Vec<u8>, VisionError>,
    ) -> Result<Option<Vec<u8>>, VisionError> {
        let t = self.next;
        if t >= self.frames {
            return Ok(None);
        }
        // Output `t` is made from the first stage of runs `t`, `t + 1` and
        // `t + 2`; the first frame makes all three, the others one.
        let runs = if t == 0 { 0..3 } else { t + 2..t + 3 };
        for run in runs {
            self.first_stage(run, source)?;
        }
        let mut out = vec![0_u8; self.width as usize * self.height as usize * 3];
        for tile in &self.tiles {
            self.denoiser
                .second_stage(tile, self.noise, &mut out, self.width)?;
        }
        self.next += 1;
        Ok(Some(out))
    }

    /// Frame `k` of the window the clip is read through: the clip with two
    /// frames mirrored onto each end, as FastDVDnet reads it.
    fn window(&self, k: usize) -> usize {
        let last = self.frames.saturating_sub(1);
        let at = k.abs_diff(SIDE_FRAMES);
        let at = if k >= SIDE_FRAMES && at > last {
            (2 * last).saturating_sub(at)
        } else {
            at
        };
        at.min(last)
    }

    /// Runs the first stage on window frames `run` to `run + 2` in each tile.
    fn first_stage(
        &mut self,
        run: usize,
        source: &mut dyn FnMut(usize) -> Result<Vec<u8>, VisionError>,
    ) -> Result<(), VisionError> {
        let wanted = [self.window(run), self.window(run + 1), self.window(run + 2)];
        let bytes = self.width as usize * self.height as usize * 3;
        for index in wanted {
            if self.held.iter().any(|(k, _)| *k == index) {
                continue;
            }
            let rgb = source(index)?;
            if rgb.len() != bytes {
                return Err(VisionError::Image {
                    width: self.width,
                    height: self.height,
                    bytes: rgb.len(),
                });
            }
            self.held.push_back((index, rgb));
            while self.held.len() > HELD {
                let Some(old) = self.held.iter().position(|(k, _)| !wanted.contains(k)) else {
                    break;
                };
                self.held.remove(old);
            }
        }
        let frame = |index: usize| {
            self.held
                .iter()
                .find(|(k, _)| *k == index)
                .map(|(_, rgb)| Image {
                    width: self.width,
                    height: self.height,
                    rgb,
                })
                .ok_or_else(|| VisionError::Model {
                    message: format!("frame {index} was not held"),
                })
        };
        let [a, b, c] = wanted;
        let trio = [frame(a)?, frame(b)?, frame(c)?];
        let denoiser = self.denoiser;
        for tile in &mut self.tiles {
            let frames = trio.map(|image| {
                denoiser.input(planes(&image, tile.read, tile.padded), 3, tile.padded)
            });
            let cleaned = denoiser
                .model
                .temp1
                .forward(frames, denoiser.noise(self.noise, tile.padded));
            tile.first.push_back(cleaned);
            if tile.first.len() > 3 {
                tile.first.pop_front();
            }
        }
        Ok(())
    }
}

impl Denoiser {
    /// Runs the second stage on a tile's last three first stages and
    /// writes its `own` part into `out`, a frame `out_w` wide.
    fn second_stage(
        &self,
        tile: &Tile,
        noise: f32,
        out: &mut [u8],
        out_w: u32,
    ) -> Result<(), VisionError> {
        let keep = tile.keep;
        let ranges = [
            0..1,
            0..3,
            keep.top as usize..(keep.top + keep.height) as usize,
            keep.left as usize..(keep.left + keep.width) as usize,
        ];
        let mut runs = tile
            .first
            .iter()
            .map(|run| run.clone().slice(ranges.clone()));
        let (Some(a), Some(b), Some(c)) = (runs.next(), runs.next(), runs.next()) else {
            return Err(VisionError::Model {
                message: "the second stage ran before three first stages".into(),
            });
        };
        let output = self
            .model
            .temp2
            .forward([a, b, c], self.noise(noise, (keep.width, keep.height)))
            .cast(FloatDType::F32);
        let expected = [1, 3, keep.height as usize, keep.width as usize];
        if output.dims() != expected {
            return Err(VisionError::Model {
                message: format!("the denoiser returned {:?}", output.dims()),
            });
        }
        let values =
            output
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|error| VisionError::Model {
                    message: format!("{error:?}"),
                })?;
        let plane = keep.width as usize * keep.height as usize;
        let skip_x = (tile.own.left - tile.read.left - keep.left) as usize;
        let skip_y = (tile.own.top - tile.read.top - keep.top) as usize;
        for y in 0..tile.own.height as usize {
            let from_row = (skip_y + y) * keep.width as usize + skip_x;
            let to_row =
                ((tile.own.top as usize + y) * out_w as usize + tile.own.left as usize) * 3;
            for x in 0..tile.own.width as usize {
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

/// The pixels of `rect` as three planes from 0 to 1, grown to `padded` by
/// mirroring the frame's last rows and columns, as PyTorch's reflect pad.
fn planes(image: &Image<'_>, rect: Rect, padded: (u32, u32)) -> Vec<f32> {
    let (pw, ph) = (padded.0 as usize, padded.1 as usize);
    let (w, h) = (image.width as usize, image.height as usize);
    let mirror = |at: usize, len: usize| {
        if at < len {
            at
        } else {
            (2 * (len - 1)).saturating_sub(at)
        }
    };
    let plane = pw * ph;
    let mut chw = vec![0.0_f32; plane * 3];
    for y in 0..ph {
        let sy = mirror(rect.top as usize + y, h);
        for x in 0..pw {
            let sx = mirror(rect.left as usize + x, w);
            for channel in 0..3 {
                let byte = image
                    .rgb
                    .get((sy * w + sx) * 3 + channel)
                    .copied()
                    .unwrap_or(0);
                if let Some(value) = chw.get_mut(channel * plane + y * pw + x) {
                    *value = f32::from(byte) / 255.0;
                }
            }
        }
    }
    chw
}

/// Reads FastDVDnet's PyTorch `.pth`, folds each batch norm into the
/// convolution before it, and writes the weights as `.bpk` for
/// [`Denoiser::load`]. Run once, offline.
pub fn convert_fastdvdnet(from: &Path, to: &Path) -> Result<(), VisionError> {
    let refused = |path: &Path, message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !from.is_file() {
        return Err(refused(from, "missing".into()));
    }
    let reader = PytorchReader::new(from).map_err(|error| refused(from, error.to_string()))?;
    let read = |name: &str, shape: &[usize]| -> Result<Vec<f32>, VisionError> {
        // Saved from a multi-GPU run, names may start with `module.`.
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
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    };
    let device = Device::flex();
    let mut model = FastDvd::new(&device);
    for (name, stage) in [("temp1", &mut model.temp1), ("temp2", &mut model.temp2)] {
        for (layer, conv) in LAYERS.iter().zip(stage.convs.iter_mut()) {
            let shape = [layer.to, layer.from / layer.groups, 3, 3];
            let mut weight = read(&format!("{name}.{}.weight", layer.path), &shape)?;
            let mut bias = vec![0.0_f32; layer.to];
            if let Some(norm) = layer.norm {
                let part = |what: &str| read(&format!("{name}.{norm}.{what}"), &[layer.to]);
                let (gamma, beta) = (part("weight")?, part("bias")?);
                let (mean, var) = (part("running_mean")?, part("running_var")?);
                let each = weight.len() / layer.to;
                for (out, (weights, bias)) in weight.chunks_mut(each).zip(&mut bias).enumerate() {
                    let (g, b, m, v) = (
                        gamma.get(out).copied().unwrap_or(1.0),
                        beta.get(out).copied().unwrap_or(0.0),
                        mean.get(out).copied().unwrap_or(0.0),
                        var.get(out).copied().unwrap_or(1.0),
                    );
                    // PyTorch's batch norm epsilon.
                    let scale = g / (v + 1e-5).sqrt();
                    weights.iter_mut().for_each(|w| *w *= scale);
                    *bias = b - m * scale;
                }
            }
            conv.weight =
                Param::from_tensor(Tensor::from_data(TensorData::new(weight, shape), &device));
            conv.bias = Some(Param::from_tensor(Tensor::from_data(
                TensorData::new(bias, [layer.to]),
                &device,
            )));
        }
    }
    let mut out = BurnpackStore::from_file(to).overwrite(true);
    model
        .save_into(&mut out)
        .map_err(|error| refused(to, error.to_string()))
}

#[cfg(test)]
mod tests;
