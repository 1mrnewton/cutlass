//! Matting anything: BiRefNet_lite on burn, which takes the background off a
//! frame whatever stands in front of it.
//!
//! A Swin-T transformer reads the frame twice, at its size and at half, and
//! the two readings are joined at each of its four levels. A decoder climbs
//! back from the deepest level, each step looking again at the frame's own
//! pixels, cut into as many tiles as that step has pixels, and sharpening
//! itself with an attention map of its edges. Deformable convolutions let
//! each step look around for the shape. About 44 million weights, MIT.
//!
//! The model was trained on squares: the frame is scaled to a square of
//! [`MATTING_SIDE`] whatever its shape, and the matte scaled back.

use std::path::Path;
use std::sync::Mutex;

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, DeformConv2d, DeformConv2dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::tensor::activation::{gelu, relu, sigmoid, softmax};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::device::{self, Compute};
use crate::matte_frame;
use crate::mobile_sam::load_weights;
use crate::{Image, Matte, Matter, VisionError};

/// The weights, in the folder passed to [`Matting::load`].
pub const MATTING_WEIGHTS: &str = "birefnet_lite.bpk";

/// The side of the square the model reads a frame at, as it was trained.
pub const MATTING_SIDE: u32 = 1024;

/// The model and the size it reads at, as [`Matter::id`] gives it at
/// [`MATTING_SIDE`].
pub const MATTING_MODEL: &str = "birefnet-lite-1024";

/// A side the model can read at is a whole number of these: the backbone
/// halves the frame five times, once more for the half-size reading.
pub const MATTING_MULTIPLE: u32 = 64;

/// ImageNet's mean and spread, which the model was trained on.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const SPREAD: [f32; 3] = [0.229, 0.224, 0.225];
/// Swin-T: the first width, the blocks and heads of each stage, the window.
const EMBED: usize = 96;
const DEPTHS: [usize; 4] = [2, 2, 6, 2];
const HEADS: [usize; 4] = [3, 6, 12, 24];
const WINDOW: usize = 7;
const SHIFT: usize = WINDOW / 2;
/// The decoder's channels at each level, deepest first: each Swin level's
/// width, twice over for the two readings.
const CHANNELS: [usize; 4] = [1536, 768, 384, 192];
/// The decoder blocks' own width, and the deformable branches'.
const INTER: usize = 64;
const BRANCH: usize = 256;
/// The deformable branches' kernels, after the 1x1 one.
const KERNELS: [usize; 3] = [1, 3, 7];
/// The edge attention's width.
const GRADIENT: usize = 16;
/// What masks a token out of another window's attention.
const MASKED: f32 = -100.0;

#[derive(Module, Debug)]
struct PatchEmbed {
    proj: Conv2d,
    norm: LayerNorm,
}

#[derive(Module, Debug)]
struct WindowAttention {
    qkv: Linear,
    proj: Linear,
    relative_position_bias_table: Param<Tensor<2>>,
}

#[derive(Module, Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

#[derive(Module, Debug)]
struct SwinBlock {
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: Mlp,
}

#[derive(Module, Debug)]
struct PatchMerging {
    norm: LayerNorm,
    reduction: Linear,
}

#[derive(Module, Debug)]
struct Stage {
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
}

#[derive(Module, Debug)]
struct Swin {
    patch_embed: PatchEmbed,
    layers: Vec<Stage>,
    norm0: LayerNorm,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

/// A convolution whose taps move, each by an offset and weighed by a
/// modulator it reads from its input.
#[derive(Module, Debug)]
struct Deformable {
    offset_conv: Conv2d,
    modulator_conv: Conv2d,
    regular_conv: DeformConv2d,
}

#[derive(Module, Debug)]
struct Branch {
    atrous_conv: Deformable,
    bn: BatchNorm,
}

#[derive(Module, Debug)]
struct Pool {
    conv: Conv2d,
    bn: BatchNorm,
}

/// Atrous spatial pyramid pooling with deformable branches.
#[derive(Module, Debug)]
struct Aspp {
    aspp1: Branch,
    aspp_deforms: Vec<Branch>,
    global_avg_pool: Pool,
    conv1: Conv2d,
    bn1: BatchNorm,
}

#[derive(Module, Debug)]
struct DecBlock {
    conv_in: Conv2d,
    bn_in: BatchNorm,
    dec_att: Aspp,
    conv_out: Conv2d,
    bn_out: BatchNorm,
}

#[derive(Module, Debug)]
struct SimpleConvs {
    conv1: Conv2d,
    conv_out: Conv2d,
}

#[derive(Module, Debug)]
struct Lateral {
    conv: Conv2d,
}

/// Where a decoder step's edges are, as a gate on its answer.
#[derive(Module, Debug)]
struct Gradient {
    conv: Conv2d,
    bn: BatchNorm,
    attn: Conv2d,
}

#[derive(Module, Debug)]
struct Decoder {
    ipt_blk1: SimpleConvs,
    ipt_blk2: SimpleConvs,
    ipt_blk3: SimpleConvs,
    ipt_blk4: SimpleConvs,
    ipt_blk5: SimpleConvs,
    decoder_block1: DecBlock,
    decoder_block2: DecBlock,
    decoder_block3: DecBlock,
    decoder_block4: DecBlock,
    conv_out1: Conv2d,
    lateral_block2: Lateral,
    lateral_block3: Lateral,
    lateral_block4: Lateral,
    gdt_2: Gradient,
    gdt_3: Gradient,
    gdt_4: Gradient,
}

/// BiRefNet, named as its published weights name it, after the renames in
/// [`convert_birefnet`].
#[derive(Module, Debug)]
struct Net {
    bb: Swin,
    squeeze_module: Vec<DecBlock>,
    decoder: Decoder,
}

fn conv(input: usize, output: usize, kernel: usize, bias: bool, device: &Device) -> Conv2d {
    let pad = kernel / 2;
    Conv2dConfig::new([input, output], [kernel, kernel])
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .with_bias(bias)
        .init(device)
}

fn norm(channels: usize, device: &Device) -> LayerNorm {
    LayerNormConfig::new(channels).init(device)
}

fn batch(channels: usize, device: &Device) -> BatchNorm {
    BatchNormConfig::new(channels).init(device)
}

fn simple(input: usize, output: usize, device: &Device) -> SimpleConvs {
    SimpleConvs {
        conv1: conv(input, INTER, 3, true, device),
        conv_out: conv(INTER, output, 3, true, device),
    }
}

fn branch(kernel: usize, device: &Device) -> Branch {
    let pad = kernel / 2;
    let taps = kernel * kernel;
    Branch {
        atrous_conv: Deformable {
            offset_conv: conv(INTER, 2 * taps, kernel, true, device),
            modulator_conv: conv(INTER, taps, kernel, true, device),
            regular_conv: DeformConv2dConfig::new([INTER, BRANCH], [kernel, kernel])
                .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
                .with_bias(false)
                .init(device),
        },
        bn: batch(BRANCH, device),
    }
}

fn dec_block(input: usize, output: usize, device: &Device) -> DecBlock {
    DecBlock {
        conv_in: conv(input, INTER, 3, true, device),
        bn_in: batch(INTER, device),
        dec_att: Aspp {
            aspp1: branch(1, device),
            aspp_deforms: KERNELS.iter().map(|k| branch(*k, device)).collect(),
            global_avg_pool: Pool {
                conv: conv(INTER, BRANCH, 1, false, device),
                bn: batch(BRANCH, device),
            },
            conv1: conv(BRANCH * (2 + KERNELS.len()), INTER, 1, false, device),
            bn1: batch(INTER, device),
        },
        conv_out: conv(INTER, output, 3, true, device),
        bn_out: batch(output, device),
    }
}

fn gradient(input: usize, device: &Device) -> Gradient {
    Gradient {
        conv: conv(input, GRADIENT, 3, true, device),
        bn: batch(GRADIENT, device),
        attn: conv(GRADIENT, 1, 1, true, device),
    }
}

impl Net {
    fn new(device: &Device) -> Self {
        let linear = |i, o| LinearConfig::new(i, o).init(device);
        let layers = DEPTHS
            .iter()
            .zip(HEADS)
            .enumerate()
            .map(|(level, (depth, heads))| {
                let dim = EMBED << level;
                Stage {
                    blocks: (0..*depth)
                        .map(|_| SwinBlock {
                            norm1: norm(dim, device),
                            attn: WindowAttention {
                                qkv: linear(dim, 3 * dim),
                                proj: linear(dim, dim),
                                relative_position_bias_table: Param::from_tensor(Tensor::zeros(
                                    [(2 * WINDOW - 1) * (2 * WINDOW - 1), heads],
                                    device,
                                )),
                            },
                            norm2: norm(dim, device),
                            mlp: Mlp {
                                fc1: linear(dim, 4 * dim),
                                fc2: linear(4 * dim, dim),
                            },
                        })
                        .collect(),
                    downsample: (level + 1 < DEPTHS.len()).then(|| PatchMerging {
                        norm: norm(4 * dim, device),
                        reduction: LinearConfig::new(4 * dim, 2 * dim)
                            .with_bias(false)
                            .init(device),
                    }),
                }
            })
            .collect();
        let [c0, c1, c2, c3] = CHANNELS;
        // The deepest level also carries the three above it, brought down.
        let deepest = c0 + c1 + c2 + c3;
        Self {
            bb: Swin {
                patch_embed: PatchEmbed {
                    proj: Conv2dConfig::new([3, EMBED], [4, 4])
                        .with_stride([4, 4])
                        .init(device),
                    norm: norm(EMBED, device),
                },
                layers,
                norm0: norm(EMBED, device),
                norm1: norm(EMBED * 2, device),
                norm2: norm(EMBED * 4, device),
                norm3: norm(EMBED * 8, device),
            },
            squeeze_module: vec![dec_block(deepest, c0, device)],
            decoder: Decoder {
                ipt_blk5: simple(3 << 10, c0 / 8, device),
                ipt_blk4: simple(3 << 8, c0 / 8, device),
                ipt_blk3: simple(3 << 6, c1 / 8, device),
                ipt_blk2: simple(3 << 4, c2 / 8, device),
                ipt_blk1: simple(3, c3 / 8, device),
                decoder_block4: dec_block(c0 + c0 / 8, c1, device),
                decoder_block3: dec_block(c1 + c0 / 8, c2, device),
                decoder_block2: dec_block(c2 + c1 / 8, c3, device),
                decoder_block1: dec_block(c3 + c2 / 8, c3 / 2, device),
                conv_out1: conv(c3 / 2 + c3 / 8, 1, 1, true, device),
                lateral_block4: Lateral {
                    conv: conv(c1, c1, 1, true, device),
                },
                lateral_block3: Lateral {
                    conv: conv(c2, c2, 1, true, device),
                },
                lateral_block2: Lateral {
                    conv: conv(c3, c3, 1, true, device),
                },
                gdt_4: gradient(c1, device),
                gdt_3: gradient(c2, device),
                gdt_2: gradient(c3, device),
            },
        }
    }
}

/// `x` resized bilinearly to `size`, corners on corners, as BiRefNet
/// resizes between its levels.
fn resize_corners(x: Tensor<4>, size: [usize; 2]) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    if [h, w] == size {
        return x;
    }
    interpolate(
        x,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_align_corners(true)
            .with_output_size(size),
    )
}

/// For each pair of positions in a window, which row of the bias table
/// holds the bias between them.
fn relative_index() -> Vec<i64> {
    let span = 2 * WINDOW as i64 - 1;
    let mut index = Vec::with_capacity(WINDOW.pow(4));
    for a in 0..WINDOW * WINDOW {
        for b in 0..WINDOW * WINDOW {
            let (ay, ax) = ((a / WINDOW) as i64, (a % WINDOW) as i64);
            let (by, bx) = ((b / WINDOW) as i64, (b % WINDOW) as i64);
            index.push((ay - by + WINDOW as i64 - 1) * span + (ax - bx + WINDOW as i64 - 1));
        }
    }
    index
}

/// The shifted windows' mask for a padded `hp` by `wp` grid: a token sees
/// only the tokens that were its neighbours before the shift.
/// `[windows, 49, 49]`, 0 or [`MASKED`].
fn shift_mask(hp: usize, wp: usize) -> Vec<f32> {
    let region = |at: usize, side: usize| {
        if at < side - WINDOW {
            0
        } else if at < side - SHIFT {
            1
        } else {
            2
        }
    };
    let (rows, cols) = (hp / WINDOW, wp / WINDOW);
    let area = WINDOW * WINDOW;
    let mut mask = Vec::with_capacity(rows * cols * area * area);
    for row in 0..rows {
        for col in 0..cols {
            let label = |i: usize| {
                let (y, x) = (row * WINDOW + i / WINDOW, col * WINDOW + i % WINDOW);
                region(y, hp) * 3 + region(x, wp)
            };
            for a in 0..area {
                for b in 0..area {
                    mask.push(if label(a) == label(b) { 0.0 } else { MASKED });
                }
            }
        }
    }
    mask
}

/// What the Swin blocks share for one grid: the bias index and the mask.
#[derive(Clone)]
struct Grid {
    index: Tensor<1, Int>,
    /// `[windows, 1, 49, 49]`.
    mask: Tensor<4>,
}

impl WindowAttention {
    fn forward(
        &self,
        x: Tensor<3>,
        heads: usize,
        index: &Tensor<1, Int>,
        mask: Option<&Tensor<4>>,
    ) -> Tensor<3> {
        let [windows, tokens, dim] = x.dims();
        let head = dim / heads;
        let qkv = self
            .qkv
            .forward(x)
            .reshape([windows, tokens, 3, heads, head])
            .permute([2, 0, 3, 1, 4]);
        let part = |i: usize| {
            qkv.clone()
                .narrow(0, i, 1)
                .reshape([windows, heads, tokens, head])
        };
        let q = part(0) * (head as f64).powf(-0.5);
        let bias = self
            .relative_position_bias_table
            .val()
            .select(0, index.clone())
            .reshape([tokens, tokens, heads])
            .permute([2, 0, 1])
            .unsqueeze::<4>();
        let mut scores = q.matmul(part(1).swap_dims(2, 3)) + bias;
        if let Some(mask) = mask {
            scores = scores + mask.clone();
        }
        let mixed = softmax(scores, 3)
            .matmul(part(2))
            .swap_dims(1, 2)
            .reshape([windows, tokens, dim]);
        self.proj.forward(mixed)
    }
}

impl SwinBlock {
    /// `x` is `[1, h * w, dim]`.
    fn forward(
        &self,
        x: Tensor<3>,
        (h, w): (usize, usize),
        heads: usize,
        shift: bool,
        grid: &Grid,
    ) -> Tensor<3> {
        let [batch, _, dim] = x.dims();
        let (hp, wp) = (h.next_multiple_of(WINDOW), w.next_multiple_of(WINDOW));
        let y = self
            .norm1
            .forward(x.clone())
            .reshape([batch, h, w, dim])
            .pad([(0, hp - h), (0, wp - w), (0, 0)], PadMode::Constant(0.0));
        let s = SHIFT as i64;
        let y = if shift { y.roll(&[-s, -s], &[1, 2]) } else { y };
        let (rows, cols) = (hp / WINDOW, wp / WINDOW);
        let windows = y
            .reshape([batch, rows, WINDOW, cols, WINDOW, dim])
            .permute([0, 1, 3, 2, 4, 5])
            .reshape([batch * rows * cols, WINDOW * WINDOW, dim]);
        let attended = self
            .attn
            .forward(windows, heads, &grid.index, shift.then_some(&grid.mask));
        let y = attended
            .reshape([batch, rows, cols, WINDOW, WINDOW, dim])
            .permute([0, 1, 3, 2, 4, 5])
            .reshape([batch, hp, wp, dim]);
        let y = if shift { y.roll(&[s, s], &[1, 2]) } else { y };
        let y = y
            .slice([0..batch, 0..h, 0..w, 0..dim])
            .reshape([batch, h * w, dim]);
        let x = x + y;
        let fed = self
            .mlp
            .fc2
            .forward(gelu(self.mlp.fc1.forward(self.norm2.forward(x.clone()))));
        x + fed
    }
}

impl PatchMerging {
    /// `[1, h * w, dim]` to `[1, h/2 * w/2, 2 * dim]`; `h` and `w` are even.
    fn forward(&self, x: Tensor<3>, (h, w): (usize, usize)) -> Tensor<3> {
        let [batch, _, dim] = x.dims();
        // Each 2x2 cell's four tokens side by side, top left, bottom left,
        // top right, bottom right.
        let x = x
            .reshape([batch, h / 2, 2, w / 2, 2, dim])
            .permute([0, 1, 3, 4, 2, 5])
            .reshape([batch, h / 2 * (w / 2), 4 * dim]);
        self.reduction.forward(self.norm.forward(x))
    }
}

impl Swin {
    /// The four levels of `x`, each `[1, channels, h, w]`, a quarter of the
    /// frame's size down to a thirty-second.
    fn forward(&self, x: Tensor<4>, grids: &Grids) -> Result<Vec<Tensor<4>>, VisionError> {
        let x = self.patch_embed.proj.forward(x);
        let [batch, dim, mut h, mut w] = x.dims();
        let mut x = self
            .patch_embed
            .norm
            .forward(x.reshape([batch, dim, h * w]).swap_dims(1, 2));
        let norms = [&self.norm0, &self.norm1, &self.norm2, &self.norm3];
        let mut levels = Vec::with_capacity(norms.len());
        for ((stage, heads), norm) in self.layers.iter().zip(HEADS).zip(norms) {
            let grid = grids.get(h, w, x.device())?;
            for (index, block) in stage.blocks.iter().enumerate() {
                x = block.forward(x, (h, w), heads, index % 2 == 1, &grid);
            }
            let [_, _, dim] = x.dims();
            levels.push(
                norm.forward(x.clone())
                    .reshape([batch, h, w, dim])
                    .permute([0, 3, 1, 2]),
            );
            if let Some(down) = &stage.downsample {
                x = down.forward(x, (h, w));
                (h, w) = (h / 2, w / 2);
            }
        }
        Ok(levels)
    }
}

/// The bias index and shift masks, kept for each grid size read.
struct Grids {
    kept: Mutex<Vec<((usize, usize), Grid)>>,
    float: FloatDType,
}

impl Grids {
    fn get(&self, h: usize, w: usize, device: Device) -> Result<Grid, VisionError> {
        let mut kept = self.kept.lock().map_err(|_| VisionError::Model {
            message: "the window masks were left half made".into(),
        })?;
        if let Some((_, grid)) = kept.iter().find(|(size, _)| *size == (h, w)) {
            return Ok(grid.clone());
        }
        let (hp, wp) = (h.next_multiple_of(WINDOW), w.next_multiple_of(WINDOW));
        let area = WINDOW * WINDOW;
        let windows = (hp / WINDOW) * (wp / WINDOW);
        let index =
            Tensor::<1, Int>::from_data(TensorData::new(relative_index(), [area * area]), &device);
        let mask = Tensor::<4>::from_data(
            TensorData::new(shift_mask(hp, wp), [windows, 1, area, area]),
            &device,
        )
        .cast(self.float);
        let grid = Grid { index, mask };
        kept.push(((h, w), grid.clone()));
        Ok(grid)
    }
}

impl Deformable {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let offset = self.offset_conv.forward(x.clone());
        let modulator = sigmoid(self.modulator_conv.forward(x.clone())).mul_scalar(2.0);
        self.regular_conv.forward(x, offset, Some(modulator))
    }
}

impl Branch {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu(self.bn.forward(self.atrous_conv.forward(x)))
    }
}

impl Aspp {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, channels, h, w] = x.dims();
        let mut parts = vec![self.aspp1.forward(x.clone())];
        parts.extend(self.aspp_deforms.iter().map(|b| b.forward(x.clone())));
        let pooled = x.mean_dims(&[2, 3]).reshape([batch, channels, 1, 1]);
        let pooled = relu(
            self.global_avg_pool
                .bn
                .forward(self.global_avg_pool.conv.forward(pooled)),
        );
        parts.push(pooled.expand([batch, BRANCH, h, w]));
        relu(self.bn1.forward(self.conv1.forward(Tensor::cat(parts, 1))))
    }
}

impl DecBlock {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = relu(self.bn_in.forward(self.conv_in.forward(x)));
        let x = self.dec_att.forward(x);
        self.bn_out.forward(self.conv_out.forward(x))
    }
}

impl SimpleConvs {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.conv_out.forward(self.conv1.forward(x))
    }
}

impl Gradient {
    /// `x` gated by where its edges are.
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let edges = relu(self.bn.forward(self.conv.forward(x.clone())));
        x * sigmoid(self.attn.forward(edges))
    }
}

/// The frame cut into a `grid` by `grid` of tiles laid over each other as
/// channels: channel, then tile row, then tile column.
fn tiles(x: &Tensor<4>, grid: usize) -> Tensor<4> {
    let [batch, channels, h, w] = x.dims();
    if grid == 1 {
        return x.clone();
    }
    x.clone()
        .reshape([batch, channels, grid, h / grid, grid, w / grid])
        .permute([0, 1, 2, 4, 3, 5])
        .reshape([batch, channels * grid * grid, h / grid, w / grid])
}

impl Net {
    /// The logits of what is kept, `[1, 1, side, side]`, for a normalized
    /// square frame whose side is a whole number of [`MATTING_MULTIPLE`].
    fn forward(&self, x: Tensor<4>, grids: &Grids) -> Result<Tensor<4>, VisionError> {
        let [_, _, h, w] = x.dims();
        let full = self.bb.forward(x.clone(), grids)?;
        let half = self
            .bb
            .forward(resize_corners(x.clone(), [h / 2, w / 2]), grids)?;
        let levels: Vec<Tensor<4>> = full
            .into_iter()
            .zip(half)
            .map(|(full, half)| {
                let [_, _, lh, lw] = full.dims();
                Tensor::cat(vec![full, resize_corners(half, [lh, lw])], 1)
            })
            .collect();
        let [x1, x2, x3, x4] =
            <[Tensor<4>; 4]>::try_from(levels).map_err(|_| VisionError::Model {
                message: "the backbone gave other than four levels".into(),
            })?;
        let size = |t: &Tensor<4>| {
            let [_, _, h, w] = t.dims();
            [h, w]
        };
        let deepest = size(&x4);
        let x4 = Tensor::cat(
            vec![
                resize_corners(x1.clone(), deepest),
                resize_corners(x2.clone(), deepest),
                resize_corners(x3.clone(), deepest),
                x4,
            ],
            1,
        );
        let x4 = self
            .squeeze_module
            .iter()
            .fold(x4, |x, block| block.forward(x));

        let d = &self.decoder;
        let grid = |level: [usize; 2]| h / level[0];
        let x4 = Tensor::cat(vec![x4, d.ipt_blk5.forward(tiles(&x, grid(deepest)))], 1);
        let p4 = d.gdt_4.forward(d.decoder_block4.forward(x4));
        let p3 = resize_corners(p4, size(&x3)) + d.lateral_block4.conv.forward(x3.clone());
        let p3 = Tensor::cat(vec![p3, d.ipt_blk4.forward(tiles(&x, grid(size(&x3))))], 1);
        let p3 = d.gdt_3.forward(d.decoder_block3.forward(p3));
        let p2 = resize_corners(p3, size(&x2)) + d.lateral_block3.conv.forward(x2.clone());
        let p2 = Tensor::cat(vec![p2, d.ipt_blk3.forward(tiles(&x, grid(size(&x2))))], 1);
        let p2 = d.gdt_2.forward(d.decoder_block2.forward(p2));
        let p1 = resize_corners(p2, size(&x1)) + d.lateral_block2.conv.forward(x1.clone());
        let p1 = Tensor::cat(vec![p1, d.ipt_blk2.forward(tiles(&x, grid(size(&x1))))], 1);
        let p1 = resize_corners(d.decoder_block1.forward(p1), [h, w]);
        let p1 = Tensor::cat(vec![p1, d.ipt_blk1.forward(x)], 1);
        Ok(d.conv_out1.forward(p1))
    }
}

/// Casts every weight to one float type.
struct CastTo(FloatDType);

impl ModuleMapper for CastTo {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let float = self.0;
        param.map(|tensor| tensor.cast(float))
    }
}

/// BiRefNet_lite, loaded on one device.
pub struct Matting {
    device: Device,
    float: FloatDType,
    model: Net,
    grids: Grids,
    side: u32,
    id: String,
}

impl Matting {
    /// Loads [`MATTING_WEIGHTS`] from the folder `weights`, to read frames
    /// at [`MATTING_SIDE`].
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        Self::load_at(weights, compute, MATTING_SIDE)
    }

    /// [`Self::load`] reading at `side`, a whole number of
    /// [`MATTING_MULTIPLE`] from 256 to 2048: less detail in less time.
    pub fn load_at(weights: &Path, compute: Compute, side: u32) -> Result<Self, VisionError> {
        check_side(side)?;
        let device = device::device(compute)?;
        let mut model = Net::new(&device);
        load_weights(&mut model, &weights.join(MATTING_WEIGHTS))?;
        Ok(Self::on(device, compute, model, side))
    }

    /// The model with random weights: the same work as the real one, for
    /// timing and for tests that only check shapes.
    #[cfg(test)]
    pub(crate) fn untrained(compute: Compute, side: u32) -> Result<Self, VisionError> {
        check_side(side)?;
        let device = device::device(compute)?;
        let model = Net::new(&device);
        Ok(Self::on(device, compute, model, side))
    }

    /// Half precision on a GPU that has it: a quarter faster on a real
    /// 1080p frame (188 ms against 250 ms on an M-series Mac), with mattes
    /// that differ by a few levels at the edges.
    fn on(device: Device, compute: Compute, model: Net, side: u32) -> Self {
        let float = device::quick_float(&device, compute);
        let model = if float == FloatDType::F32 {
            model
        } else {
            model.map(&mut CastTo(float))
        };
        Self {
            device,
            float,
            model,
            grids: Grids {
                kept: Mutex::new(Vec::new()),
                float,
            },
            side,
            id: format!("birefnet-lite-{side}"),
        }
    }

    /// The square side frames are read at.
    #[must_use]
    pub fn side(&self) -> u32 {
        self.side
    }

    /// The share of each pixel kept, `[1, 1, side, side]`.
    fn kept(&self, image: &Image<'_>) -> Result<Tensor<4>, VisionError> {
        matte_frame::check(image)?;
        let frame = matte_frame::input(image, (self.side, self.side), MEAN, SPREAD, &self.device)?
            .cast(self.float);
        let kept = sigmoid(self.model.forward(frame, &self.grids)?);
        Ok(kept.cast(FloatDType::F32))
    }
}

fn check_side(side: u32) -> Result<(), VisionError> {
    if !side.is_multiple_of(MATTING_MULTIPLE) || !(256..=2048).contains(&side) {
        return Err(VisionError::Model {
            message: format!(
                "BiRefNet reads squares of 256 to 2048 pixels in steps of {MATTING_MULTIPLE}, not {side}"
            ),
        });
    }
    Ok(())
}

impl Matter for Matting {
    fn id(&self) -> &str {
        &self.id
    }

    fn matte(&mut self, image: &Image<'_>) -> Result<Matte, VisionError> {
        let kept = self.kept(image)?;
        matte_frame::to_matte(kept, image.width, image.height)
    }
}

/// Reads BiRefNet_lite's `model.safetensors` as published, and writes it to
/// `to` as `.bpk` for [`Matting::load`]. Run once, offline. What only
/// training reads (the side outputs) is left out.
pub fn convert_birefnet(from: &Path, to: &Path) -> Result<(), VisionError> {
    let refused = |path: &Path, message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !from.is_file() {
        return Err(refused(from, "missing".into()));
    }
    let device = Device::flex();
    let mut model = Net::new(&device);
    let mut store = SafetensorsStore::from_file(from)
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"global_avg_pool\.1\.", "global_avg_pool.conv.")
        .with_key_remapping(r"global_avg_pool\.2\.", "global_avg_pool.bn.")
        .with_key_remapping(r"gdt_convs_attn_(\d)\.0\.", "gdt_$1.attn.")
        .with_key_remapping(r"gdt_convs_(\d)\.0\.", "gdt_$1.conv.")
        .with_key_remapping(r"gdt_convs_(\d)\.1\.", "gdt_$1.bn.")
        .with_key_remapping(r"conv_out1\.0\.", "conv_out1.")
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
