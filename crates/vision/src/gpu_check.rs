//! Whether this machine's GPU gives the vision models' answers.
//!
//! A GPU path can be built and still be wrong on one machine: burn's
//! kernels gave wrong masks on an iPhone, and a phone's Vulkan driver
//! aborted the process (see `tools/ml-bench/README.md`). So before the
//! models use a GPU, [`check_gpu`] runs a small network made of the layers
//! they use (convolutions, a deformable one, attention, norms, resizing,
//! warping by a sampled grid) on
//! the GPU and on the CPU, with the same made-up weights, and compares.
//!
//! A driver that aborts takes the process with it, so a shell runs the
//! check in a process of its own and trusts the GPU only when that process
//! says it passed.

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::conv::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, DeformConv2d, DeformConv2dConfig,
};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::{gelu, relu, sigmoid, softmax};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{
    GridSampleOptions, GridSamplePaddingMode, InterpolateMode, InterpolateOptions,
};
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData};

use crate::VisionError;
use crate::device::{self, Compute};

/// The check's picture: three channels, this many pixels a side.
const SIDE: usize = 32;
/// Channels inside the check's network.
const WIDTH: usize = 8;
/// Tokens a side in its attention.
const TOKENS: usize = 8;
/// Channels of its wide layer: a convolution there sums 576 products, as
/// Enhance quality's do, which is where half floats lose most.
const WIDE: usize = 64;

/// The largest buffer the models ask for, rounded up: BiRefNet and SAM 2
/// at 1024 each ask for 240 MiB. Mesa's software Vulkan (lavapipe) holds
/// 128 MiB.
const LARGEST: usize = 256 << 20;

/// The most a GPU answer in 32-bit floats may differ from the CPU's.
const F32_WORST: f32 = 1e-3;
/// The same in 16-bit floats, which keep about three decimal digits.
const F16_WORST: f32 = 3e-2;

/// What the GPU passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuCheck {
    /// Models may run in 16-bit floats on it, for speed. Only on the Mac:
    /// on Vulkan (lavapipe) Enhance quality in half floats came out 8 levels
    /// from PyTorch where Metal's are 2, though this check's own network
    /// differed alike on both, so a Vulkan GPU keeps 32-bit floats until
    /// half ones are measured on real cards.
    pub half: bool,
}

/// Run the check network on the GPU and the CPU and compare: `Ok` when
/// the GPU opens and agrees in 32-bit floats. A build with no GPU path is
/// [`VisionError::NoGpu`]; a GPU that disagrees is [`VisionError::Model`].
/// A broken driver can abort the process instead of returning: run this in
/// a process of its own.
pub fn check_gpu() -> Result<GpuCheck, VisionError> {
    holds_largest()?;
    let (full, half) = differences()?;
    if full > F32_WORST {
        return Err(VisionError::Model {
            message: format!("the GPU's answers differ from the CPU's by up to {full}"),
        });
    }
    Ok(GpuCheck {
        half: cfg!(target_os = "macos") && half.is_some_and(|half| half <= F16_WORST),
    })
}

/// Whether the GPU holds a buffer as large as the models' largest, and
/// sums it right.
fn holds_largest() -> Result<(), VisionError> {
    let gpu = device::device(Compute::Gpu)?;
    let count = LARGEST / size_of::<f32>();
    let ones = Tensor::<1>::from_data(TensorData::new(vec![1.0_f32; count], [count]), &gpu);
    let total = crate::read_back(ones.sum())
        .and_then(|data| {
            data.try_into_vec::<f32>()
                .map_err(|error| VisionError::Model {
                    message: format!("{error:?}"),
                })
        })
        .map_err(|error| VisionError::Model {
            message: format!(
                "the GPU cannot hold the models' {} MiB buffers: {error}",
                LARGEST >> 20
            ),
        })?;
    // A sum this long in 32-bit floats is exact to well within a part in
    // a thousand.
    let summed = total.first().copied().unwrap_or(0.0);
    if (summed - count as f32).abs() > count as f32 * 1e-3 {
        return Err(VisionError::Model {
            message: format!("the GPU summed {count} ones to {summed}"),
        });
    }
    Ok(())
}

/// How far the GPU's answers are from the CPU's in 32-bit floats, and in
/// 16-bit ones where the GPU has them and they run.
fn differences() -> Result<(f32, Option<f32>), VisionError> {
    let gpu = device::device(Compute::Gpu)?;
    let cpu = device::device(Compute::Cpu)?;
    let want = run(&cpu, FloatDType::F32)?;
    let full = worst(&want, &run(&gpu, FloatDType::F32)?)?;
    let half = if gpu.supports_dtype(DType::F16) {
        run(&gpu, FloatDType::F16)
            .and_then(|halved| worst(&want, &halved))
            .ok()
    } else {
        None
    };
    Ok((full, half))
}

/// The largest difference between `a` and `b`, which must be the same
/// length and finite.
fn worst(a: &[f32], b: &[f32]) -> Result<f32, VisionError> {
    if a.len() != b.len() || a.is_empty() {
        return Err(VisionError::Model {
            message: format!("the check gave {} and {} values", a.len(), b.len()),
        });
    }
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            return Err(VisionError::Model {
                message: "the check gave a value that is not a number".into(),
            });
        }
        worst = worst.max((x - y).abs());
    }
    Ok(worst)
}

#[derive(Module, Debug)]
struct Net {
    stem: Conv2d,
    widen: Conv2d,
    wide: Conv2d,
    narrow: Conv2d,
    offset: Conv2d,
    modulator: Conv2d,
    deform: DeformConv2d,
    norm: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    up: ConvTranspose2d,
}

impl Net {
    fn new(device: &Device) -> Self {
        let same = PaddingConfig2d::Explicit(1, 1, 1, 1);
        let conv = |input, output| {
            Conv2dConfig::new([input, output], [3, 3])
                .with_padding(same.clone())
                .init(device)
        };
        let linear = || LinearConfig::new(WIDTH, WIDTH).init(device);
        Self {
            stem: conv(3, WIDTH),
            widen: conv(WIDTH, WIDE),
            wide: conv(WIDE, WIDE),
            narrow: conv(WIDE, WIDTH),
            offset: conv(WIDTH, 18),
            modulator: conv(WIDTH, 9),
            deform: DeformConv2dConfig::new([WIDTH, WIDTH], [3, 3])
                .with_padding(same.clone())
                .init(device),
            norm: LayerNormConfig::new(WIDTH).init(device),
            query: linear(),
            key: linear(),
            value: linear(),
            up: ConvTranspose2dConfig::new([WIDTH, 4], [2, 2])
                .with_stride([2, 2])
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = relu(self.stem.forward(x));
        let wide = relu(self.wide.forward(relu(self.widen.forward(x))));
        let x = relu(self.narrow.forward(wide));
        let offset = self.offset.forward(x.clone()) * 0.5;
        let modulator = sigmoid(self.modulator.forward(x.clone()));
        let x = self.deform.forward(x, offset, Some(modulator));
        // Channels last for the norm, as a vision transformer has them.
        let x = gelu(self.norm.forward(x.permute([0, 2, 3, 1]))).permute([0, 3, 1, 2]);
        let pooled = interpolate(
            x,
            InterpolateOptions::new(InterpolateMode::Bilinear).with_output_size([TOKENS, TOKENS]),
        );
        let tokens = pooled
            .permute([0, 2, 3, 1])
            .reshape([1, TOKENS * TOKENS, WIDTH]);
        let scores = self
            .query
            .forward(tokens.clone())
            .matmul(self.key.forward(tokens.clone()).swap_dims(1, 2))
            / (WIDTH as f64).sqrt();
        let mixed = softmax(scores, 2).matmul(self.value.forward(tokens));
        let grid = mixed
            .reshape([1, TOKENS, TOKENS, WIDTH])
            .permute([0, 3, 1, 2]);
        let up = sigmoid(self.up.forward(grid));
        // Warp by a grid laid out as Optical flow's is: each plane on its
        // own, joined on the last side.
        let [_, _, h, w] = up.dims();
        let plane = |at: Tensor<4>| (at * 2.0 - 1.0).reshape([1, h, w, 1]);
        let at = Tensor::cat(
            vec![
                plane(up.clone().slice([0..1, 0..1])),
                plane(up.clone().slice([0..1, 1..2])),
            ],
            3,
        );
        let warped = up.grid_sample_2d(
            at,
            GridSampleOptions::new(InterpolateMode::Bilinear)
                .with_padding_mode(GridSamplePaddingMode::Border)
                .with_align_corners(true),
        );
        interpolate(
            warped,
            InterpolateOptions::new(InterpolateMode::Bilinear).with_output_size([SIDE, SIDE]),
        )
    }
}

/// Gives every weight a fixed made-up value, the same on any device, and
/// casts it to `float`.
struct Fill {
    next: u32,
    float: FloatDType,
}

impl ModuleMapper for Fill {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let float = self.float;
        let seed = &mut self.next;
        param.map(|tensor| {
            let shape = tensor.shape();
            let values = made_up(seed, shape.num_elements(), 0.3);
            Tensor::<D>::from_data(TensorData::new(values, shape), &tensor.device()).cast(float)
        })
    }
}

/// `count` values in `-scale..scale` from a small fixed generator, carried
/// on from `seed`.
fn made_up(seed: &mut u32, count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((*seed >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The check network's answer on `device` in `float`, as 32-bit floats.
fn run(device: &Device, float: FloatDType) -> Result<Vec<f32>, VisionError> {
    let net = Net::new(device).map(&mut Fill { next: 1, float });
    let mut seed = 7;
    let picture = made_up(&mut seed, 3 * SIDE * SIDE, 1.0);
    let input =
        Tensor::<4>::from_data(TensorData::new(picture, [1, 3, SIDE, SIDE]), device).cast(float);
    crate::read_back(net.forward(input).cast(FloatDType::F32))?
        .try_into_vec::<f32>()
        .map_err(|error| VisionError::Model {
            message: format!("the check's answer did not read: {error:?}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_check_network_gives_the_same_answer_twice_on_the_cpu() {
        let cpu = device::device(Compute::Cpu).unwrap();
        let first = run(&cpu, FloatDType::F32).unwrap();
        assert_eq!(first.len(), 4 * SIDE * SIDE);
        assert_eq!(worst(&first, &run(&cpu, FloatDType::F32).unwrap()), Ok(0.0));
        assert!(first.iter().any(|value| (value - first[0]).abs() > 1e-3));
    }

    #[test]
    fn answers_of_other_lengths_or_not_numbers_fail() {
        assert!(worst(&[1.0], &[1.0, 2.0]).is_err());
        assert!(worst(&[], &[]).is_err());
        assert!(worst(&[f32::NAN], &[1.0]).is_err());
        assert_eq!(worst(&[1.0, 2.0], &[1.5, 2.0]), Ok(0.5));
    }

    #[test]
    #[ignore = "needs a GPU and its kernel warm-up"]
    fn this_machines_gpu_passes() {
        let (full, half) = differences().unwrap();
        eprintln!("GPU apart from the CPU: {full} in f32, {half:?} in f16");
        let passed = check_gpu().unwrap();
        eprintln!("GPU passed the check; 16-bit floats too: {}", passed.half);
    }
}
