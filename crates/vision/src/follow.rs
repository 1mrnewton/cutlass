//! Following what the user picked through a clip: SAM 2.1 hiera tiny on
//! burn, Custom removal's [`Follower`].
//!
//! A Hiera transformer reads each frame, squeezed to a square of
//! [`FOLLOW_SIDE`], into features at three sizes. On a frame the user
//! prompted, clicks become tokens and a painted mask a map, and a two-way
//! transformer between them and the features answers with a mask. On every
//! other frame the features first look back, through four layers of
//! attention, at what the model remembers: the masks it made on the
//! prompted frames and on the six frames before this one, each squeezed by a
//! memory encoder to 64 channels, and a pointer token for the object from
//! each of up to sixteen frames. Then the same decoder answers with no
//! prompt. About 39 million weights, Apache-2.0.
//!
//! One object is followed at a time. The model remembers the frames it was
//! given in order, so each run starts with [`Follower::reset`] and a
//! prompted frame, and frames come one after another as the clip plays.
//!
//! The matte is the model's own soft answer: the sigmoid of its mask
//! logits, scaled from 256 a side to the frame's size.

use std::collections::VecDeque;
use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::tensor::activation::{gelu, relu, sigmoid, softmax};
use burn::tensor::ops::PadMode;
use burn::tensor::{Device, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::device::{self, Compute};
use crate::matte_frame;
use crate::mobile_sam::load_weights;
use crate::{Follower, Image, Matte, Prompt, VisionError};

/// The weights, in the folder passed to [`FollowModel::load`].
pub const FOLLOW_WEIGHTS: &str = "sam2.1_hiera_tiny.bpk";

/// The side of the square the model reads every frame at.
pub const FOLLOW_SIDE: u32 = 1024;

/// The model and the size it reads at, as [`Follower::id`] gives it.
pub const FOLLOW_MODEL: &str = "sam2.1-hiera-tiny-1024";

/// ImageNet's mean and spread, which the model was trained on.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const SPREAD: [f32; 3] = [0.229, 0.224, 0.225];
/// Hiera-T: each stage's width, blocks, heads and window, and the blocks
/// that look at the whole frame instead of a window.
const DIMS: [usize; 4] = [96, 192, 384, 768];
const BLOCKS: [usize; 4] = [1, 2, 7, 2];
const HEADS: [usize; 4] = [1, 2, 4, 8];
const WINDOWS: [usize; 4] = [8, 4, 14, 7];
const GLOBAL: [usize; 3] = [5, 7, 9];
/// The window position embedding's own size, scaled to the first stage.
const BACKGROUND: usize = 7;
/// The width of the features, prompts and decoder.
const HIDDEN: usize = 256;
/// The width of a memory.
const MEM: usize = 64;
/// The deepest level has a token for each this many pixels a side, and
/// the decoder's masks a pixel for each this many.
const TOKEN: usize = 16;
const PIXEL: usize = 4;
/// Masks remembered: the prompted frames, and this many frames less one
/// before the current one.
const MEMORIES: usize = 7;
/// Object pointers looked back over, the current frame's place included.
const POINTERS: usize = 16;
/// What a mask is set to where the model sees no object.
const NO_OBJECT: f32 = -1024.0;
/// A single mask is kept when its area hardly moves as its edge moves by
/// this much, by this share; otherwise the best of the three others.
const STABILITY_DELTA: f32 = 0.05;
const STABILITY: f32 = 0.98;
/// Mask logits for a mask the user gave, and the memory's view of a mask.
const MASK_SCALE: f32 = 20.0;
const MASK_BIAS: f32 = -10.0;
/// The decoder's attention heads, and the most attention scores one step
/// of the memory attention holds at a time.
const DECODER_HEADS: usize = 8;
const MOST_SCORES: usize = 1 << 26;

#[derive(Module, Debug)]
struct PatchEmbed {
    projection: Conv2d,
}

/// Linear layers with an activation between each.
#[derive(Module, Debug)]
struct Mlp {
    proj_in: Linear,
    layers: Vec<Linear>,
    proj_out: Linear,
}

#[derive(Clone, Copy)]
enum Act {
    Relu,
    Gelu,
}

#[derive(Module, Debug)]
struct HieraAttention {
    qkv: Linear,
    proj: Linear,
}

#[derive(Module, Debug)]
struct HieraBlock {
    layer_norm1: LayerNorm,
    attn: HieraAttention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
    /// Widens the residual where a stage starts.
    proj: Option<Linear>,
}

#[derive(Module, Debug)]
struct Backbone {
    patch_embed: PatchEmbed,
    pos_embed: Param<Tensor<4>>,
    pos_embed_window: Param<Tensor<4>>,
    blocks: Vec<HieraBlock>,
}

#[derive(Module, Debug)]
struct Neck {
    convs: Vec<Conv2d>,
}

#[derive(Module, Debug)]
struct VisionEncoder {
    backbone: Backbone,
    neck: Neck,
}

/// Random Fourier features of a point in the square.
#[derive(Module, Debug)]
struct Fourier {
    positional_embedding: Param<Tensor<2>>,
}

#[derive(Module, Debug)]
struct MaskEmbed {
    conv1: Conv2d,
    layer_norm1: LayerNorm,
    conv2: Conv2d,
    layer_norm2: LayerNorm,
    conv3: Conv2d,
}

#[derive(Module, Debug)]
struct PromptEncoder {
    shared_embedding: Fourier,
    mask_embed: MaskEmbed,
    no_mask_embed: Embedding,
    not_a_point_embed: Embedding,
    point_embed: Embedding,
}

#[derive(Module, Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

#[derive(Module, Debug)]
struct TwoWayBlock {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    cross_attn_token_to_image: Attention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
    layer_norm3: LayerNorm,
    layer_norm4: LayerNorm,
    cross_attn_image_to_token: Attention,
}

#[derive(Module, Debug)]
struct TwoWay {
    layers: Vec<TwoWayBlock>,
    final_attn_token_to_image: Attention,
    layer_norm_final_attn: LayerNorm,
}

#[derive(Module, Debug)]
struct MaskDecoder {
    iou_token: Embedding,
    mask_tokens: Embedding,
    obj_score_token: Embedding,
    transformer: TwoWay,
    upscale_conv1: ConvTranspose2d,
    upscale_conv2: ConvTranspose2d,
    upscale_layer_norm: LayerNorm,
    output_hypernetworks_mlps: Vec<Mlp>,
    iou_prediction_head: Mlp,
    pred_obj_score_head: Mlp,
    conv_s0: Conv2d,
    conv_s1: Conv2d,
}

#[derive(Module, Debug)]
struct MemoryLayer {
    self_attn: Attention,
    cross_attn_image: Attention,
    linear1: Linear,
    linear2: Linear,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    layer_norm3: LayerNorm,
}

#[derive(Module, Debug)]
struct MemoryAttention {
    layers: Vec<MemoryLayer>,
    layer_norm: LayerNorm,
}

#[derive(Module, Debug)]
struct DownLayer {
    conv: Conv2d,
    layer_norm: LayerNorm,
}

#[derive(Module, Debug)]
struct MaskDownSampler {
    layers: Vec<DownLayer>,
    final_conv: Conv2d,
}

#[derive(Module, Debug)]
struct CxBlock {
    depthwise_conv: Conv2d,
    layer_norm: LayerNorm,
    pointwise_conv1: Linear,
    pointwise_conv2: Linear,
    scale: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct Fuser {
    layers: Vec<CxBlock>,
}

#[derive(Module, Debug)]
struct MemoryEncoder {
    mask_downsampler: MaskDownSampler,
    feature_projection: Conv2d,
    memory_fuser: Fuser,
    projection: Conv2d,
}

/// SAM 2.1, named as its published weights are.
#[derive(Module, Debug)]
struct Sam2 {
    shared_image_embedding: Fourier,
    vision_encoder: VisionEncoder,
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
    memory_attention: MemoryAttention,
    memory_encoder: MemoryEncoder,
    /// `[MEMORIES, 1, 1, MEM]`: how long ago a memory was made.
    memory_temporal_positional_encoding: Param<Tensor<4>>,
    /// `[1, 1, HIDDEN]`: added to a prompted frame's features.
    no_memory_embedding: Param<Tensor<3>>,
    /// `[1, HIDDEN]`: the pointer of a frame with no object.
    no_object_pointer: Param<Tensor<2>>,
    /// `[1, MEM]`: added to the memory of a frame with no object.
    occlusion_spatial_embedding_parameter: Param<Tensor<2>>,
    temporal_positional_encoding_projection_layer: Linear,
    mask_downsample: Conv2d,
    object_pointer_proj: Mlp,
}

fn conv(input: usize, output: usize, kernel: usize, stride: usize, device: &Device) -> Conv2d {
    let pad = if stride == kernel { 0 } else { kernel / 2 };
    Conv2dConfig::new([input, output], [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .init(device)
}

fn norm(channels: usize, epsilon: f64, device: &Device) -> LayerNorm {
    LayerNormConfig::new(channels)
        .with_epsilon(epsilon)
        .init(device)
}

fn linear(input: usize, output: usize, device: &Device) -> Linear {
    LinearConfig::new(input, output).init(device)
}

fn mlp(input: usize, hidden: usize, output: usize, layers: usize, device: &Device) -> Mlp {
    Mlp {
        proj_in: linear(input, hidden, device),
        layers: (2..layers)
            .map(|_| linear(hidden, hidden, device))
            .collect(),
        proj_out: linear(hidden, output, device),
    }
}

fn embedding(count: usize, device: &Device) -> Embedding {
    EmbeddingConfig::new(count, HIDDEN).init(device)
}

fn zeros<const D: usize>(shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(Tensor::zeros(shape, device))
}

fn attention(kv: usize, internal: usize, device: &Device) -> Attention {
    Attention {
        q_proj: linear(HIDDEN, internal, device),
        k_proj: linear(kv, internal, device),
        v_proj: linear(kv, internal, device),
        o_proj: linear(internal, HIDDEN, device),
    }
}

fn fourier(device: &Device) -> Fourier {
    Fourier {
        positional_embedding: zeros([2, HIDDEN / 2], device),
    }
}

/// Which stage block `index` is in, and whether it starts the stage.
fn stage_of(index: usize) -> (usize, bool) {
    let mut start = 0;
    for (stage, blocks) in BLOCKS.iter().enumerate() {
        if index < start + blocks {
            return (stage, index == start);
        }
        start += blocks;
    }
    (BLOCKS.len() - 1, false)
}

/// The last block of each stage, whose output the neck reads.
fn stage_ends() -> [usize; 4] {
    let mut ends = [0; 4];
    let mut sum = 0;
    for (end, blocks) in ends.iter_mut().zip(BLOCKS) {
        sum += blocks;
        *end = sum - 1;
    }
    ends
}

impl Sam2 {
    fn new(device: &Device) -> Self {
        let blocks = (0..BLOCKS.iter().sum())
            .map(|index| {
                let (stage, first) = stage_of(index);
                let out = DIMS[stage];
                let input = if stage > 0 && first {
                    DIMS[stage - 1]
                } else {
                    out
                };
                HieraBlock {
                    layer_norm1: norm(input, 1e-6, device),
                    attn: HieraAttention {
                        qkv: linear(input, 3 * out, device),
                        proj: linear(out, out, device),
                    },
                    layer_norm2: norm(out, 1e-6, device),
                    mlp: mlp(out, 4 * out, out, 2, device),
                    proj: (input != out).then(|| linear(input, out, device)),
                }
            })
            .collect();
        let backbone = Backbone {
            patch_embed: PatchEmbed {
                projection: Conv2dConfig::new([3, DIMS[0]], [7, 7])
                    .with_stride([4, 4])
                    .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
                    .init(device),
            },
            pos_embed: zeros([1, DIMS[0], BACKGROUND, BACKGROUND], device),
            pos_embed_window: zeros([1, DIMS[0], WINDOWS[0], WINDOWS[0]], device),
            blocks,
        };
        let neck = Neck {
            convs: DIMS
                .iter()
                .rev()
                .map(|dim| conv(*dim, HIDDEN, 1, 1, device))
                .collect(),
        };
        let prompt_encoder = PromptEncoder {
            shared_embedding: fourier(device),
            mask_embed: MaskEmbed {
                conv1: conv(1, 4, 2, 2, device),
                layer_norm1: norm(4, 1e-6, device),
                conv2: conv(4, 16, 2, 2, device),
                layer_norm2: norm(16, 1e-6, device),
                conv3: conv(16, HIDDEN, 1, 1, device),
            },
            no_mask_embed: embedding(1, device),
            not_a_point_embed: embedding(1, device),
            point_embed: embedding(4, device),
        };
        let two_way = |_| TwoWayBlock {
            self_attn: attention(HIDDEN, HIDDEN, device),
            layer_norm1: norm(HIDDEN, 1e-5, device),
            cross_attn_token_to_image: attention(HIDDEN, HIDDEN / 2, device),
            layer_norm2: norm(HIDDEN, 1e-5, device),
            mlp: mlp(HIDDEN, 2048, HIDDEN, 2, device),
            layer_norm3: norm(HIDDEN, 1e-5, device),
            layer_norm4: norm(HIDDEN, 1e-5, device),
            cross_attn_image_to_token: attention(HIDDEN, HIDDEN / 2, device),
        };
        let mask_decoder = MaskDecoder {
            iou_token: embedding(1, device),
            mask_tokens: embedding(4, device),
            obj_score_token: embedding(1, device),
            transformer: TwoWay {
                layers: (0..2).map(two_way).collect(),
                final_attn_token_to_image: attention(HIDDEN, HIDDEN / 2, device),
                layer_norm_final_attn: norm(HIDDEN, 1e-5, device),
            },
            upscale_conv1: ConvTranspose2dConfig::new([HIDDEN, HIDDEN / 4], [2, 2])
                .with_stride([2, 2])
                .init(device),
            upscale_conv2: ConvTranspose2dConfig::new([HIDDEN / 4, HIDDEN / 8], [2, 2])
                .with_stride([2, 2])
                .init(device),
            upscale_layer_norm: norm(HIDDEN / 4, 1e-6, device),
            output_hypernetworks_mlps: (0..4)
                .map(|_| mlp(HIDDEN, HIDDEN, HIDDEN / 8, 3, device))
                .collect(),
            iou_prediction_head: mlp(HIDDEN, HIDDEN, 4, 3, device),
            pred_obj_score_head: mlp(HIDDEN, HIDDEN, 1, 3, device),
            conv_s0: conv(HIDDEN, HIDDEN / 8, 1, 1, device),
            conv_s1: conv(HIDDEN, HIDDEN / 4, 1, 1, device),
        };
        let memory_attention = MemoryAttention {
            layers: (0..4)
                .map(|_| MemoryLayer {
                    self_attn: attention(HIDDEN, HIDDEN, device),
                    cross_attn_image: attention(MEM, HIDDEN, device),
                    linear1: linear(HIDDEN, 2048, device),
                    linear2: linear(2048, HIDDEN, device),
                    layer_norm1: norm(HIDDEN, 1e-5, device),
                    layer_norm2: norm(HIDDEN, 1e-5, device),
                    layer_norm3: norm(HIDDEN, 1e-5, device),
                })
                .collect(),
            layer_norm: norm(HIDDEN, 1e-5, device),
        };
        let memory_encoder = MemoryEncoder {
            mask_downsampler: MaskDownSampler {
                layers: [(1, 4), (4, 16), (16, 64), (64, 256)]
                    .into_iter()
                    .map(|(input, output)| DownLayer {
                        conv: conv(input, output, 3, 2, device),
                        layer_norm: norm(output, 1e-6, device),
                    })
                    .collect(),
                final_conv: conv(HIDDEN, HIDDEN, 1, 1, device),
            },
            feature_projection: conv(HIDDEN, HIDDEN, 1, 1, device),
            memory_fuser: Fuser {
                layers: (0..2)
                    .map(|_| CxBlock {
                        depthwise_conv: Conv2dConfig::new([HIDDEN, HIDDEN], [7, 7])
                            .with_groups(HIDDEN)
                            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
                            .init(device),
                        layer_norm: norm(HIDDEN, 1e-6, device),
                        pointwise_conv1: linear(HIDDEN, 1024, device),
                        pointwise_conv2: linear(1024, HIDDEN, device),
                        scale: Param::from_tensor(Tensor::full([HIDDEN], 1e-6, device)),
                    })
                    .collect(),
            },
            projection: conv(HIDDEN, MEM, 1, 1, device),
        };
        Self {
            shared_image_embedding: fourier(device),
            vision_encoder: VisionEncoder { backbone, neck },
            prompt_encoder,
            mask_decoder,
            memory_attention,
            memory_encoder,
            memory_temporal_positional_encoding: zeros([MEMORIES, 1, 1, MEM], device),
            no_memory_embedding: zeros([1, 1, HIDDEN], device),
            no_object_pointer: zeros([1, HIDDEN], device),
            occlusion_spatial_embedding_parameter: zeros([1, MEM], device),
            temporal_positional_encoding_projection_layer: linear(HIDDEN, MEM, device),
            mask_downsample: Conv2dConfig::new([1, 1], [4, 4])
                .with_stride([4, 4])
                .init(device),
            object_pointer_proj: mlp(HIDDEN, HIDDEN, HIDDEN, 3, device),
        }
    }
}

impl Act {
    fn apply<const D: usize>(self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Relu => relu(x),
            Self::Gelu => gelu(x),
        }
    }
}

impl Mlp {
    fn forward<const D: usize>(&self, x: Tensor<D>, act: Act) -> Tensor<D> {
        let mut x = act.apply(self.proj_in.forward(x));
        for layer in &self.layers {
            x = act.apply(layer.forward(x));
        }
        self.proj_out.forward(x)
    }
}

/// A layer norm over the channels of `[b, c, h, w]`.
fn norm_channels(norm: &LayerNorm, x: Tensor<4>) -> Tensor<4> {
    norm.forward(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
}

/// The largest of each two by two of `[b, h, w, c]`.
fn pool(x: Tensor<4>) -> Tensor<4> {
    let [b, h, w, c] = x.dims();
    x.reshape([b, h / 2, 2, w / 2, 2, c])
        .max_dim(2)
        .max_dim(4)
        .reshape([b, h / 2, w / 2, c])
}

/// `[b, h, w, c]` cut into windows of `size`, padded with zeros to a whole
/// number of them, and the padded size.
fn partition(x: Tensor<4>, size: usize) -> (Tensor<4>, (usize, usize)) {
    let [b, h, w, c] = x.dims();
    let (hp, wp) = (h.div_ceil(size) * size, w.div_ceil(size) * size);
    let x = if (hp, wp) == (h, w) {
        x
    } else {
        x.pad(
            [(0, 0), (0, hp - h), (0, wp - w), (0, 0)],
            PadMode::Constant(0.0),
        )
    };
    let windows = x
        .reshape([b, hp / size, size, wp / size, size, c])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b * (hp / size) * (wp / size), size, size, c]);
    (windows, (hp, wp))
}

/// Windows of `size` put back together as `[b, h, w, c]`.
fn unpartition(
    x: Tensor<4>,
    size: usize,
    (hp, wp): (usize, usize),
    (h, w): (usize, usize),
) -> Tensor<4> {
    let [count, _, _, c] = x.dims();
    let (nh, nw) = (hp / size, wp / size);
    let b = count / (nh * nw);
    let x = x
        .reshape([b, nh, nw, size, size, c])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b, hp, wp, c]);
    if (hp, wp) == (h, w) {
        x
    } else {
        x.narrow(1, 0, h).narrow(2, 0, w)
    }
}

/// Scaled dot-product attention of `[b, heads, n, d]`.
fn attend<const D: usize>(q: Tensor<D>, k: Tensor<D>, v: Tensor<D>) -> Tensor<D> {
    let head = q.dims()[D - 1];
    let scores = q.matmul(k.swap_dims(D - 2, D - 1)) * (head as f64).powf(-0.5);
    softmax(scores, D - 1).matmul(v)
}

impl HieraAttention {
    /// `x` is `[b, h, w, in]`; the queries are pooled by two when `pooled`.
    fn forward(&self, x: Tensor<4>, heads: usize, pooled: bool) -> Tensor<4> {
        let [b, h, w, _] = x.dims();
        let qkv = self.qkv.forward(x);
        let out = qkv.dims()[3] / 3;
        let head = out / heads;
        let qkv = qkv.reshape([b, h * w, 3, heads, head]);
        let part = |i: usize| qkv.clone().narrow(2, i, 1).reshape([b, h * w, heads, head]);
        let (mut q, k, v) = (part(0), part(1), part(2));
        let (mut qh, mut qw) = (h, w);
        if pooled {
            (qh, qw) = (h / 2, w / 2);
            q = pool(q.reshape([b, h, w, out])).reshape([b, qh * qw, heads, head]);
        }
        let mixed = attend(q.swap_dims(1, 2), k.swap_dims(1, 2), v.swap_dims(1, 2))
            .swap_dims(1, 2)
            .reshape([b, qh, qw, out]);
        self.proj.forward(mixed)
    }
}

impl HieraBlock {
    /// `x` is `[1, h, w, c]`.
    fn forward(&self, x: Tensor<4>, index: usize) -> Tensor<4> {
        let (stage, first) = stage_of(index);
        let pooled = stage > 0 && first;
        let window = if GLOBAL.contains(&index) {
            0
        } else if pooled {
            WINDOWS[stage - 1]
        } else {
            WINDOWS[stage]
        };
        let normed = self.layer_norm1.forward(x.clone());
        let residual = match &self.proj {
            Some(proj) => pool(proj.forward(normed.clone())),
            None => x,
        };
        let [_, h, w, _] = normed.dims();
        let attended = if window == 0 {
            self.attn.forward(normed, HEADS[stage], pooled)
        } else {
            let (windows, padded) = partition(normed, window);
            let out = self.attn.forward(windows, HEADS[stage], pooled);
            if pooled {
                let size = window / 2;
                let [_, rh, rw, _] = residual.dims();
                let padded = (rh.div_ceil(size) * size, rw.div_ceil(size) * size);
                unpartition(out, size, padded, (rh, rw))
            } else {
                unpartition(out, window, padded, (h, w))
            }
        };
        let x = residual + attended;
        x.clone() + self.mlp.forward(self.layer_norm2.forward(x), Act::Gelu)
    }
}

/// What the image encoder makes of a frame, channels first: the decoder's
/// two finer levels, already narrowed, and the deepest level as it is.
struct Features {
    s0: Tensor<4>,
    s1: Tensor<4>,
    top: Tensor<4>,
}

/// Everything that depends only on the weights, made once.
struct Fixed {
    /// The side of the square frames are read at.
    side: usize,
    /// `[1, side / 4, side / 4, 96]`: the backbone's position embedding.
    hiera: Tensor<4>,
    /// `[1, tokens, HIDDEN]`: the decoder's image position embedding.
    image: Tensor<3>,
    /// `[1, tokens, HIDDEN]`: the deepest features' sine position embedding.
    vision: Tensor<3>,
    /// `[1, tokens, MEM]`: a memory's sine position embedding.
    memory: Tensor<3>,
    /// `[1, tokens, HIDDEN]`: the rotary embedding's angles.
    cos: Tensor<3>,
    sin: Tensor<3>,
}

impl Sam2 {
    fn features(&self, frame: Tensor<4>, fixed: &Fixed) -> Result<Features, VisionError> {
        let backbone = &self.vision_encoder.backbone;
        let mut x = backbone
            .patch_embed
            .projection
            .forward(frame)
            .permute([0, 2, 3, 1])
            + fixed.hiera.clone();
        let ends = stage_ends();
        let mut stages = Vec::with_capacity(ends.len());
        for (index, block) in backbone.blocks.iter().enumerate() {
            x = block.forward(x, index);
            if ends.contains(&index) {
                stages.push(x.clone());
            }
        }
        let convs = &self.vision_encoder.neck.convs;
        // The neck's convolutions run deepest first.
        let lateral = |level: usize| -> Result<Tensor<4>, VisionError> {
            let conv = convs.get(DIMS.len() - 1 - level);
            let stage = stages.get(level);
            match (conv, stage) {
                (Some(conv), Some(stage)) => Ok(conv.forward(stage.clone().permute([0, 3, 1, 2]))),
                _ => Err(VisionError::Model {
                    message: format!("no stage {level} in the image encoder"),
                }),
            }
        };
        let deepest = lateral(3)?;
        let [b, c, h, w] = deepest.dims();
        let up = deepest
            .reshape([b, c, h, 1, w, 1])
            .expand([b, c, h, 2, w, 2])
            .reshape([b, c, 2 * h, 2 * w]);
        Ok(Features {
            s0: self.mask_decoder.conv_s0.forward(lateral(0)?),
            s1: self.mask_decoder.conv_s1.forward(lateral(1)?),
            top: lateral(2)? + up,
        })
    }
}

impl Fourier {
    /// `[n, 2]` points in the unit square as `[n, HIDDEN]`.
    fn forward(&self, points: Tensor<2>) -> Tensor<2> {
        let angles = (points * 2.0 - 1.0).matmul(self.positional_embedding.val())
            * (2.0 * std::f64::consts::PI);
        Tensor::cat(vec![angles.clone().sin(), angles.cos()], 1)
    }
}

impl Attention {
    fn project(&self, q: Tensor<3>, k: Tensor<3>, v: Tensor<3>) -> [Tensor<3>; 3] {
        [
            self.q_proj.forward(q),
            self.k_proj.forward(k),
            self.v_proj.forward(v),
        ]
    }

    /// `[1, n, c]` queries against `[1, m, c]` keys and values.
    fn forward(&self, q: Tensor<3>, k: Tensor<3>, v: Tensor<3>, heads: usize) -> Tensor<3> {
        let [q, k, v] = self.project(q, k, v);
        let split = |x: Tensor<3>| {
            let [b, n, c] = x.dims();
            x.reshape([b, n, heads, c / heads]).swap_dims(1, 2)
        };
        let [b, n, c] = q.dims();
        let mixed = attend(split(q), split(k), split(v))
            .swap_dims(1, 2)
            .reshape([b, n, c]);
        self.o_proj.forward(mixed)
    }

    /// One head with rotary positions: `[1, tokens, HIDDEN]` queries
    /// against keys of whole frames' tokens, then `pointers` keys left
    /// unturned. The queries are taken a share at a time so the scores
    /// stay bounded however much is remembered.
    fn forward_rotary(
        &self,
        q: Tensor<3>,
        k: Tensor<3>,
        v: Tensor<3>,
        fixed: &Fixed,
        pointers: usize,
    ) -> Tensor<3> {
        let [q, k, v] = self.project(q, k, v);
        let q = rotate(q, &fixed.cos, &fixed.sin);
        let [_, n, _] = q.dims();
        let [_, keys, _] = k.dims();
        let turned = keys - pointers;
        let k = if turned == 0 {
            k
        } else {
            let frames = turned / n.max(1);
            let rotated = rotate(
                k.clone().narrow(1, 0, turned),
                &fixed.cos.clone().repeat_dim(1, frames),
                &fixed.sin.clone().repeat_dim(1, frames),
            );
            if pointers == 0 {
                rotated
            } else {
                Tensor::cat(vec![rotated, k.narrow(1, turned, pointers)], 1)
            }
        };
        let rows = (MOST_SCORES / keys.max(1) / 256 * 256)
            .max(256)
            .min(n.max(1));
        let mut mixed = Vec::with_capacity(n.div_ceil(rows));
        let mut start = 0;
        while start < n {
            let take = rows.min(n - start);
            mixed.push(attend(
                q.clone().narrow(1, start, take),
                k.clone(),
                v.clone(),
            ));
            start += take;
        }
        self.o_proj.forward(Tensor::cat(mixed, 1))
    }
}

/// `x * cos + turned(x) * sin`, each pair of channels `(a, b)` turned to
/// `(-b, a)`.
fn rotate(x: Tensor<3>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<3> {
    let [b, n, c] = x.dims();
    let pairs = x.clone().reshape([b, n, c / 2, 2]);
    let turned = Tensor::cat(
        vec![pairs.clone().narrow(3, 1, 1).neg(), pairs.narrow(3, 0, 1)],
        3,
    )
    .reshape([b, n, c]);
    x * cos.clone() + turned * sin.clone()
}

impl TwoWayBlock {
    fn forward(
        &self,
        queries: Tensor<3>,
        keys: Tensor<3>,
        query_pe: &Tensor<3>,
        key_pe: &Tensor<3>,
        first: bool,
    ) -> (Tensor<3>, Tensor<3>) {
        let heads = DECODER_HEADS;
        let queries = if first {
            self.self_attn
                .forward(queries.clone(), queries.clone(), queries, heads)
        } else {
            let q = queries.clone() + query_pe.clone();
            queries.clone() + self.self_attn.forward(q.clone(), q, queries, heads)
        };
        let queries = self.layer_norm1.forward(queries);
        let attended = self.cross_attn_token_to_image.forward(
            queries.clone() + query_pe.clone(),
            keys.clone() + key_pe.clone(),
            keys.clone(),
            heads,
        );
        let queries = self.layer_norm2.forward(queries + attended);
        let queries = self
            .layer_norm3
            .forward(queries.clone() + self.mlp.forward(queries, Act::Relu));
        let attended = self.cross_attn_image_to_token.forward(
            keys.clone() + key_pe.clone(),
            queries.clone() + query_pe.clone(),
            queries.clone(),
            heads,
        );
        let keys = self.layer_norm4.forward(keys + attended);
        (queries, keys)
    }
}

/// The decoder's answer, before one mask is chosen.
struct Answer {
    /// `[1, 4, side / 4, side / 4]`.
    masks: Tensor<4>,
    /// The predicted overlap of each mask with the object.
    scores: Vec<f32>,
    /// `[1, 4, HIDDEN]`: the token behind each mask.
    tokens: Tensor<3>,
    /// Above 0 when the decoder sees the object.
    object: f32,
}

impl MaskDecoder {
    fn forward(
        &self,
        image: Tensor<4>,
        image_pe: &Tensor<3>,
        sparse: Tensor<3>,
        dense: Tensor<4>,
        features: &Features,
    ) -> Result<Answer, VisionError> {
        let output = Tensor::cat(
            vec![
                self.obj_score_token.weight.val(),
                self.iou_token.weight.val(),
                self.mask_tokens.weight.val(),
            ],
            0,
        )
        .unsqueeze::<3>();
        let tokens = Tensor::cat(vec![output, sparse], 1);
        let [b, c, h, w] = image.dims();
        let src = (image + dense).reshape([b, c, h * w]).swap_dims(1, 2);
        let two_way = &self.transformer;
        let (mut queries, mut keys) = (tokens.clone(), src);
        for (index, layer) in two_way.layers.iter().enumerate() {
            (queries, keys) = layer.forward(queries, keys, &tokens, image_pe, index == 0);
        }
        let attended = two_way.final_attn_token_to_image.forward(
            queries.clone() + tokens,
            keys.clone() + image_pe.clone(),
            keys.clone(),
            DECODER_HEADS,
        );
        let queries = two_way.layer_norm_final_attn.forward(queries + attended);
        let mask_tokens = queries.clone().narrow(1, 2, 4);
        let image = keys.swap_dims(1, 2).reshape([b, c, h, w]);
        let up = self.upscale_conv1.forward(image) + features.s1.clone();
        let up = gelu(norm_channels(&self.upscale_layer_norm, up));
        let up = gelu(self.upscale_conv2.forward(up) + features.s0.clone());
        let [_, channels, uh, uw] = up.dims();
        let hyper: Vec<Tensor<3>> = self
            .output_hypernetworks_mlps
            .iter()
            .enumerate()
            .map(|(i, mlp)| mlp.forward(mask_tokens.clone().narrow(1, i, 1), Act::Relu))
            .collect();
        let masks = Tensor::cat(hyper, 1)
            .matmul(up.reshape([b, channels, uh * uw]))
            .reshape([b, 4, uh, uw]);
        let scores = sigmoid(
            self.iou_prediction_head
                .forward(queries.clone().narrow(1, 1, 1), Act::Relu),
        );
        let object = self
            .pred_obj_score_head
            .forward(queries.narrow(1, 0, 1), Act::Relu);
        Ok(Answer {
            masks,
            scores: matte_frame::values(scores)?,
            tokens: mask_tokens,
            object: first(matte_frame::values(object)?)?,
        })
    }
}

fn first(values: Vec<f32>) -> Result<f32, VisionError> {
    values.first().copied().ok_or_else(|| VisionError::Model {
        message: "an empty answer".into(),
    })
}

/// A point the decoder reads, in the model's square, with its label: 1
/// keep, 0 take out, -1 none.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    x: f32,
    y: f32,
    label: i32,
}

impl PromptEncoder {
    /// The tokens of `points`, after the point that ends every list.
    fn sparse(&self, points: &[Point], side: usize, device: &Device) -> Tensor<3> {
        let none = Point {
            x: 0.0,
            y: 0.0,
            label: -1,
        };
        let mut all = points.to_vec();
        if all.is_empty() {
            all.push(none);
        }
        all.push(none);
        let side = side as f32;
        let coords: Vec<f32> = all
            .iter()
            .flat_map(|p| [(p.x + 0.5) / side, (p.y + 0.5) / side])
            .collect();
        let placed = self.shared_embedding.forward(Tensor::from_data(
            TensorData::new(coords, [all.len(), 2]),
            device,
        ));
        let rows = all
            .iter()
            .enumerate()
            .map(|(i, point)| match usize::try_from(point.label) {
                Ok(label) => {
                    placed.clone().narrow(0, i, 1)
                        + self.point_embed.weight.val().narrow(0, label, 1)
                }
                Err(_) => self.not_a_point_embed.weight.val(),
            })
            .collect();
        Tensor::cat(rows, 0).unsqueeze::<3>()
    }

    /// `[1, HIDDEN, grid, grid]` from a `[1, 1, 4 * grid, 4 * grid]` mask,
    /// or the embedding of no mask.
    fn dense(&self, mask: Option<Tensor<4>>, grid: usize) -> Tensor<4> {
        match mask {
            Some(mask) => {
                let m = &self.mask_embed;
                let x = gelu(norm_channels(&m.layer_norm1, m.conv1.forward(mask)));
                let x = gelu(norm_channels(&m.layer_norm2, m.conv2.forward(x)));
                m.conv3.forward(x)
            }
            None => self
                .no_mask_embed
                .weight
                .val()
                .reshape([1, HIDDEN, 1, 1])
                .expand([1, HIDDEN, grid, grid]),
        }
    }
}

/// One frame's answer.
struct Decoded {
    /// `[1, 1, side / 4, side / 4]` logits.
    low: Tensor<4>,
    /// `[1, 1, side, side]` logits, for the memory.
    high: Tensor<4>,
    /// `[1, HIDDEN]`.
    pointer: Tensor<2>,
    /// Above 0 when the object is in the frame.
    object: f32,
}

/// The index of the largest of `scores` from `from` on, the first of equals.
fn best(scores: &[f32], from: usize) -> usize {
    let mut best = from;
    for (i, score) in scores.iter().enumerate().skip(from) {
        if scores.get(best).is_some_and(|b| score > b) {
            best = i;
        }
    }
    best
}

impl Sam2 {
    /// The mask for `points` and a `[1, 1, side / 4, side / 4]` mask over
    /// `image`, the deepest features as the decoder reads them.
    fn decode(
        &self,
        features: &Features,
        image: Tensor<4>,
        points: &[Point],
        mask: Option<Tensor<4>>,
        multimask: bool,
        fixed: &Fixed,
    ) -> Result<Decoded, VisionError> {
        let device = image.device();
        let sparse = self.prompt_encoder.sparse(points, fixed.side, &device);
        let dense = self.prompt_encoder.dense(mask, fixed.side / TOKEN);
        let answer = self
            .mask_decoder
            .forward(image, &fixed.image, sparse, dense, features)?;
        let pick = if multimask {
            best(&answer.scores, 1)
        } else {
            let single = answer.masks.clone().narrow(1, 0, 1);
            let inside = single.clone().greater_elem(STABILITY_DELTA).float().sum();
            let around = single.greater_elem(-STABILITY_DELTA).float().sum();
            let inside = first(matte_frame::values(inside)?)?;
            let around = first(matte_frame::values(around)?)?;
            let stability = if around > 0.0 { inside / around } else { 1.0 };
            if stability >= STABILITY {
                0
            } else {
                best(&answer.scores, 1)
            }
        };
        let token = if multimask { pick } else { 0 };
        let seen = answer.object > 0.0;
        let low = answer.masks.narrow(1, pick, 1);
        let low = if seen {
            low
        } else {
            Tensor::full(low.dims(), NO_OBJECT, &device)
        };
        let pointer = if seen {
            let token = answer.tokens.narrow(1, token, 1).reshape([1, HIDDEN]);
            self.object_pointer_proj.forward(token, Act::Relu)
        } else {
            self.no_object_pointer.val()
        };
        Ok(Decoded {
            high: matte_frame::resize(low.clone(), [fixed.side, fixed.side]),
            low,
            pointer,
            object: answer.object,
        })
    }

    /// A painted mask taken as the answer, `mask` 0 or 1 at the model's
    /// square: the decoder only gives its pointer.
    fn as_answer(
        &self,
        features: &Features,
        mask: &[f32],
        fixed: &Fixed,
    ) -> Result<Decoded, VisionError> {
        let device = features.top.device();
        let logits: Vec<f32> = mask.iter().map(|m| m * MASK_SCALE + MASK_BIAS).collect();
        let (side, low) = (fixed.side, fixed.side / PIXEL);
        let small = resample(&logits, (side, side), (low, low));
        let square = |values: Vec<f32>, side: usize| {
            Tensor::<4>::from_data(TensorData::new(values, [1, 1, side, side]), &device)
        };
        let down = self.mask_downsample.forward(square(mask.to_vec(), side));
        let decoded = self.decode(features, features.top.clone(), &[], Some(down), true, fixed)?;
        let seen = mask.iter().any(|m| *m > 0.0);
        Ok(Decoded {
            low: square(small, low),
            high: square(logits, side),
            pointer: if seen {
                decoded.pointer
            } else {
                self.no_object_pointer.val()
            },
            object: if seen {
                MASK_SCALE + MASK_BIAS
            } else {
                MASK_BIAS
            },
        })
    }

    /// The deepest features of a frame looking back at `memory`, its
    /// position embedding `memory_pe`, the last `pointers` of its tokens
    /// object pointers.
    fn recall(
        &self,
        top: Tensor<4>,
        memory: Tensor<3>,
        memory_pe: Tensor<3>,
        pointers: usize,
        fixed: &Fixed,
    ) -> Tensor<4> {
        let [b, c, h, w] = top.dims();
        let mut x = top.reshape([b, c, h * w]).swap_dims(1, 2) + fixed.vision.clone() * 0.1;
        let keys = memory.clone() + memory_pe;
        for layer in &self.memory_attention.layers {
            let q = layer.layer_norm1.forward(x.clone());
            x = x + layer
                .self_attn
                .forward_rotary(q.clone(), q.clone(), q, fixed, 0);
            let q = layer.layer_norm2.forward(x.clone());
            x = x + layer.cross_attn_image.forward_rotary(
                q,
                keys.clone(),
                memory.clone(),
                fixed,
                pointers,
            );
            let q = layer.layer_norm3.forward(x.clone());
            x = x.clone() + layer.linear2.forward(relu(layer.linear1.forward(q)));
        }
        self.memory_attention
            .layer_norm
            .forward(x)
            .swap_dims(1, 2)
            .reshape([b, c, h, w])
    }

    /// What is remembered of a frame: `[1, tokens, MEM]`, from its deepest
    /// features and its `[1, 1, side, side]` mask logits, taken as they are
    /// when they came from the user and softened otherwise.
    fn remember(
        &self,
        top: Tensor<4>,
        high: Tensor<4>,
        prompted: bool,
        object: f32,
    ) -> Result<Tensor<3>, VisionError> {
        let encoder = &self.memory_encoder;
        let mask = if prompted {
            high.greater_elem(0.0).float()
        } else {
            sigmoid(high)
        };
        let mut m = mask * MASK_SCALE + MASK_BIAS;
        for layer in &encoder.mask_downsampler.layers {
            m = gelu(norm_channels(&layer.layer_norm, layer.conv.forward(m)));
        }
        let m = encoder.mask_downsampler.final_conv.forward(m);
        let mut x = encoder.feature_projection.forward(top) + m;
        for block in &encoder.memory_fuser.layers {
            let h = norm_channels(&block.layer_norm, block.depthwise_conv.forward(x.clone()))
                .permute([0, 2, 3, 1]);
            let h = block
                .pointwise_conv2
                .forward(gelu(block.pointwise_conv1.forward(h)))
                * block.scale.val().reshape([1, 1, 1, HIDDEN]);
            x = x + h.permute([0, 3, 1, 2]);
        }
        let [b, _, h, w] = x.dims();
        let mut memory = encoder
            .projection
            .forward(x)
            .reshape([b, MEM, h * w])
            .swap_dims(1, 2);
        if object <= 0.0 {
            memory = memory
                + self
                    .occlusion_spatial_embedding_parameter
                    .val()
                    .unsqueeze::<3>();
        }
        // The model keeps its memories in bfloat16.
        let device = memory.device();
        let kept: Vec<f32> = matte_frame::values(memory)?
            .into_iter()
            .map(bfloat16)
            .collect();
        Ok(Tensor::from_data(
            TensorData::new(kept, [b, h * w, MEM]),
            &device,
        ))
    }
}

/// `value` rounded to the nearest bfloat16, ties to even.
fn bfloat16(value: f32) -> f32 {
    if value.is_nan() {
        return value;
    }
    let bits = value.to_bits();
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding) & 0xFFFF_0000)
}

/// `values`, a `from` (width, height) plane, resized to `to` as PyTorch's
/// bilinear `interpolate` does with `antialias=True`: a triangle as wide as
/// the step between output pixels when shrinking.
fn resample(values: &[f32], from: (usize, usize), to: (usize, usize)) -> Vec<f32> {
    let rows = weights(from.0, to.0);
    let columns = weights(from.1, to.1);
    let mut across = vec![0.0_f32; from.1 * to.0];
    for (y, row) in values.chunks_exact(from.0).enumerate() {
        for (x, (start, taps)) in rows.iter().enumerate() {
            let sum: f32 = taps
                .iter()
                .zip(row.iter().skip(*start))
                .map(|(w, v)| w * v)
                .sum();
            if let Some(out) = across.get_mut(y * to.0 + x) {
                *out = sum;
            }
        }
    }
    let mut out = vec![0.0_f32; to.0 * to.1];
    for (y, (start, taps)) in columns.iter().enumerate() {
        for x in 0..to.0 {
            let sum: f32 = taps
                .iter()
                .enumerate()
                .map(|(i, w)| w * across.get((start + i) * to.0 + x).copied().unwrap_or(0.0))
                .sum();
            if let Some(cell) = out.get_mut(y * to.0 + x) {
                *cell = sum;
            }
        }
    }
    out
}

/// For each of `to` outputs along a side of `from`, its first input and the
/// weights from there on.
fn weights(from: usize, to: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = from as f64 / to as f64;
    let support = scale.max(1.0);
    let step = 1.0 / support;
    (0..to)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as i64).min(from as i64) as usize;
            let taps: Vec<f64> = (start..end)
                .map(|j| (1.0 - ((j as f64 - center + 0.5) * step).abs()).max(0.0))
                .collect();
            let total: f64 = taps.iter().sum();
            let taps = taps
                .iter()
                .map(|w| if total > 0.0 { (w / total) as f32 } else { 0.0 })
                .collect();
            (start, taps)
        })
        .collect()
}

/// `[side * side, 2 * feats]` sine position embedding of a `side` square,
/// rows then columns, as SAM 2's: each half the `feats` channels of y then
/// of x, sine and cosine in turn.
fn sine_grid(feats: usize, side: usize) -> Vec<f32> {
    let scale = 2.0 * std::f32::consts::PI;
    let last = side as f32 + 1e-6;
    let dim: Vec<f32> = (0..feats)
        .map(|i| 10000_f32.powf((2 * (i / 2)) as f32 / feats as f32))
        .collect();
    let mut out = Vec::with_capacity(side * side * 2 * feats);
    for y in 0..side {
        for x in 0..side {
            for along in [y, x] {
                let at = (along + 1) as f32 / last * scale;
                for (i, d) in dim.iter().enumerate() {
                    let angle = at / d;
                    out.push(if i % 2 == 0 { angle.sin() } else { angle.cos() });
                }
            }
        }
    }
    out
}

/// The sine embedding of how far back each object pointer is, `HIDDEN`
/// wide: sines of the first half's frequencies, then cosines.
fn pointer_sines(offsets: &[usize]) -> Vec<f32> {
    let half = HIDDEN / 2;
    let most = (POINTERS - 1) as f32;
    let dim: Vec<f32> = (0..half)
        .map(|i| 10000_f32.powf((2 * (i / 2)) as f32 / half as f32))
        .collect();
    let mut out = Vec::with_capacity(offsets.len() * HIDDEN);
    for offset in offsets {
        let at = *offset as f32 / most;
        out.extend(dim.iter().map(|d| (at / d).sin()));
        out.extend(dim.iter().map(|d| (at / d).cos()));
    }
    out
}

/// The rotary embedding's angles for each token of the deepest level: the
/// first half of the channels turn with the column, the second with the
/// row, each frequency for two channels.
fn rotary(grid: usize) -> (Vec<f32>, Vec<f32>) {
    let half = HIDDEN / 2;
    let inverse: Vec<f32> = (0..half / 2)
        .map(|i| 1.0 / 10000_f32.powf((2 * i) as f32 / half as f32))
        .collect();
    let mut cos = Vec::with_capacity(grid * grid * HIDDEN);
    let mut sin = Vec::with_capacity(grid * grid * HIDDEN);
    for row in 0..grid {
        for column in 0..grid {
            for along in [column, row] {
                for frequency in &inverse {
                    let angle = along as f32 * frequency;
                    cos.extend([angle.cos(); 2]);
                    sin.extend([angle.sin(); 2]);
                }
            }
        }
    }
    (cos, sin)
}

/// PyTorch's bicubic weights for a tap `t` past the sample before it.
fn cubic(t: f32) -> [f32; 4] {
    let a = -0.75_f32;
    let near = |x: f32| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let far = |x: f32| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
}

/// `[c, from, from]` planes resized bicubically to `[to, to, c]`, as
/// PyTorch's `interpolate(mode="bicubic")` with half-pixel centres.
fn bicubic(values: &[f32], channels: usize, from: usize, to: usize) -> Vec<f32> {
    let scale = from as f32 / to as f32;
    let taps: Vec<([usize; 4], [f32; 4])> = (0..to)
        .map(|i| {
            let source = scale * (i as f32 + 0.5) - 0.5;
            let floor = source.floor();
            let index = |k: i64| (floor as i64 + k).clamp(0, from as i64 - 1) as usize;
            (
                [index(-1), index(0), index(1), index(2)],
                cubic(source - floor),
            )
        })
        .collect();
    let mut out = vec![0.0_f32; to * to * channels];
    for (c, plane) in values.chunks_exact(from * from).enumerate().take(channels) {
        let at = |y: usize, x: usize| plane.get(y * from + x).copied().unwrap_or(0.0);
        for (y, (rows, wy)) in taps.iter().enumerate() {
            for (x, (columns, wx)) in taps.iter().enumerate() {
                let mut sum = 0.0;
                for (row, weight) in rows.iter().zip(wy) {
                    let across: f32 = columns
                        .iter()
                        .zip(wx)
                        .map(|(column, w)| w * at(*row, *column))
                        .sum();
                    sum += weight * across;
                }
                if let Some(cell) = out.get_mut((y * to + x) * channels + c) {
                    *cell = sum;
                }
            }
        }
    }
    out
}

impl Fixed {
    fn new(model: &Sam2, device: &Device, frame: usize) -> Result<Self, VisionError> {
        let backbone = &model.vision_encoder.backbone;
        let (side, grid) = (frame / PIXEL, frame / TOKEN);
        let tokens = grid * grid;
        let channels = DIMS[0];
        let background = matte_frame::values(backbone.pos_embed.val())?;
        let window = matte_frame::values(backbone.pos_embed_window.val())?;
        let mut hiera = bicubic(&background, channels, BACKGROUND, side);
        let size = WINDOWS[0];
        for (i, cell) in hiera.iter_mut().enumerate() {
            let (c, x, y) = (i % channels, (i / channels) % side, i / channels / side);
            *cell += window
                .get((c * size + y % size) * size + x % size)
                .copied()
                .unwrap_or(0.0);
        }
        let centres: Vec<f32> = (0..tokens)
            .flat_map(|t| {
                let (y, x) = (t / grid, t % grid);
                [
                    (x as f32 + 0.5) / grid as f32,
                    (y as f32 + 0.5) / grid as f32,
                ]
            })
            .collect();
        let image = model
            .shared_image_embedding
            .forward(Tensor::from_data(
                TensorData::new(centres, [tokens, 2]),
                device,
            ))
            .unsqueeze::<3>();
        let table = |values: Vec<f32>, width: usize| {
            Tensor::<3>::from_data(TensorData::new(values, [1, tokens, width]), device)
        };
        let (cos, sin) = rotary(grid);
        Ok(Self {
            side: frame,
            hiera: Tensor::from_data(TensorData::new(hiera, [1, side, side, channels]), device),
            image,
            vision: table(sine_grid(HIDDEN / 2, grid), HIDDEN),
            memory: table(sine_grid(MEM / 2, grid), MEM),
            cos: table(cos, HIDDEN),
            sin: table(sin, HIDDEN),
        })
    }
}

/// What the model keeps of one frame.
struct Memory {
    /// The frame's place in the run.
    index: usize,
    /// `[1, tokens, MEM]`.
    features: Tensor<3>,
    /// `[1, HIDDEN]`.
    pointer: Tensor<2>,
}

/// One frame's answer, as logits.
pub(crate) struct Step {
    /// `[1, 1, side / 4, side / 4]`.
    pub(crate) logits: Tensor<4>,
    /// Above 0 when the object is in the frame.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "only the reference test reads it")
    )]
    pub(crate) object: f32,
}

/// SAM 2.1 hiera tiny, loaded on one device, with its memory of the run.
pub struct FollowModel {
    device: Device,
    model: Sam2,
    fixed: Fixed,
    /// The prompted frames, in order.
    prompted: Vec<Memory>,
    /// The latest frames followed without a prompt, oldest first.
    followed: VecDeque<Memory>,
    /// The next frame's place in the run.
    next: usize,
}

impl FollowModel {
    /// Loads [`FOLLOW_WEIGHTS`] from the folder `weights`.
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let mut model = Sam2::new(&device);
        load_weights(&mut model, &weights.join(FOLLOW_WEIGHTS))?;
        Self::on(device, model, FOLLOW_SIDE as usize)
    }

    /// The model with random weights, reading frames at `side`, a whole
    /// number of 64 pixels: the same work as the real one at 1024, and
    /// quick at a smaller side for tests that only check shapes.
    #[cfg(test)]
    pub(crate) fn untrained(compute: Compute, side: usize) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let model = Sam2::new(&device);
        Self::on(device, model, side)
    }

    fn on(device: Device, model: Sam2, side: usize) -> Result<Self, VisionError> {
        let fixed = Fixed::new(&model, &device, side)?;
        Ok(Self {
            device,
            model,
            fixed,
            prompted: Vec::new(),
            followed: VecDeque::new(),
            next: 0,
        })
    }

    /// The next frame's logits, and what it leaves in the memory.
    pub(crate) fn advance(
        &mut self,
        image: &Image<'_>,
        prompt: Option<&Prompt>,
    ) -> Result<Step, VisionError> {
        let (width, height) = matte_frame::check(image)?;
        match prompt {
            Some(prompt) => prompt.check(width, height)?,
            None if self.prompted.is_empty() => {
                return Err(VisionError::Clicks {
                    reason: "the first frame of a run needs clicks or a painted mask",
                });
            }
            None => {}
        }
        let side = self.fixed.side as u32;
        let frame = matte_frame::input(image, (side, side), MEAN, SPREAD, &self.device)?;
        let features = self.model.features(frame, &self.fixed)?;
        let index = self.next;
        let decoded = match prompt {
            Some(prompt) => self.prompted_frame(&features, prompt, width, height)?,
            None => self.followed_frame(&features, index)?,
        };
        let memory =
            self.model
                .remember(features.top, decoded.high, prompt.is_some(), decoded.object)?;
        let memory = Memory {
            index,
            features: memory,
            pointer: decoded.pointer,
        };
        if prompt.is_some() {
            self.prompted.push(memory);
        } else {
            self.followed.push_back(memory);
            while self.followed.len() > POINTERS - 1 {
                self.followed.pop_front();
            }
        }
        self.next += 1;
        Ok(Step {
            logits: decoded.low,
            object: decoded.object,
        })
    }

    fn prompted_frame(
        &self,
        features: &Features,
        prompt: &Prompt,
        width: u32,
        height: u32,
    ) -> Result<Decoded, VisionError> {
        let side = self.fixed.side;
        let painted = prompt.mask.as_ref().map(|mask| {
            let share: Vec<f32> = mask.alpha.iter().map(|a| f32::from(*a) / 255.0).collect();
            let share = if (mask.width as usize, mask.height as usize) == (side, side) {
                share
            } else {
                resample(
                    &share,
                    (mask.width as usize, mask.height as usize),
                    (side, side),
                )
            };
            share
                .into_iter()
                .map(|v| if v >= 0.5 { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>()
        });
        if prompt.clicks.is_empty() {
            let painted = painted.ok_or(VisionError::Clicks { reason: "none" })?;
            return self.model.as_answer(features, &painted, &self.fixed);
        }
        let (sx, sy) = (side as f32 / width as f32, side as f32 / height as f32);
        let points: Vec<Point> = prompt
            .clicks
            .iter()
            .map(|click| Point {
                x: click.x * sx,
                y: click.y * sy,
                label: i32::from(click.keep),
            })
            .collect();
        // Clicks over a painted mask: the mask goes in as the decoder's
        // mask input, as logits of the user's certainty.
        let dense = painted.map(|painted| {
            let logits: Vec<f32> = painted.iter().map(|m| m * MASK_SCALE + MASK_BIAS).collect();
            Tensor::<4>::from_data(
                TensorData::new(
                    resample(&logits, (side, side), (side / PIXEL, side / PIXEL)),
                    [1, 1, side / PIXEL, side / PIXEL],
                ),
                &self.device,
            )
        });
        let image = features.top.clone()
            + self
                .model
                .no_memory_embedding
                .val()
                .reshape([1, HIDDEN, 1, 1]);
        self.model.decode(
            features,
            image,
            &points,
            dense,
            points.len() <= 1,
            &self.fixed,
        )
    }

    fn followed_frame(&self, features: &Features, index: usize) -> Result<Decoded, VisionError> {
        let temporal = |offset: usize| {
            self.model
                .memory_temporal_positional_encoding
                .val()
                .narrow(0, offset, 1)
                .reshape([1, 1, MEM])
        };
        let mut memories = Vec::new();
        let mut places = Vec::new();
        // Every prompted frame, as if it were just now; then the frames
        // just before this one, furthest first.
        for memory in &self.prompted {
            memories.push(memory.features.clone());
            places.push(self.fixed.memory.clone() + temporal(MEMORIES - 1));
        }
        for back in (1..MEMORIES).rev() {
            let found = self.followed.iter().find(|m| m.index + back == index);
            if let Some(memory) = found {
                memories.push(memory.features.clone());
                places.push(self.fixed.memory.clone() + temporal(back - 1));
            }
        }
        let mut offsets = Vec::new();
        let mut pointers = Vec::new();
        for memory in self.prompted.iter().filter(|m| m.index <= index) {
            offsets.push(index - memory.index);
            pointers.push(memory.pointer.clone());
        }
        for back in 1..POINTERS.min(index + 1) {
            if let Some(memory) = self.followed.iter().find(|m| m.index + back == index) {
                offsets.push(back);
                pointers.push(memory.pointer.clone());
            }
        }
        let count = pointers.len();
        let splits = HIDDEN / MEM;
        if count > 0 {
            let tokens = Tensor::cat(pointers, 0).reshape([1, count * splits, MEM]);
            let sines = Tensor::<2>::from_data(
                TensorData::new(pointer_sines(&offsets), [count, HIDDEN]),
                &self.device,
            );
            let place = self
                .model
                .temporal_positional_encoding_projection_layer
                .forward(sines)
                .reshape([count, 1, MEM])
                .expand([count, splits, MEM])
                .reshape([1, count * splits, MEM]);
            memories.push(tokens);
            places.push(place);
        }
        let image = self.model.recall(
            features.top.clone(),
            Tensor::cat(memories, 1),
            Tensor::cat(places, 1),
            count * splits,
            &self.fixed,
        );
        self.model
            .decode(features, image, &[], None, true, &self.fixed)
    }
}

impl Follower for FollowModel {
    fn id(&self) -> &str {
        FOLLOW_MODEL
    }

    fn reset(&mut self) {
        self.prompted.clear();
        self.followed.clear();
        self.next = 0;
    }

    fn step(&mut self, image: &Image<'_>, prompt: Option<&Prompt>) -> Result<Matte, VisionError> {
        let step = self.advance(image, prompt)?;
        let (width, height) = (image.width, image.height);
        let logits = matte_frame::resize(step.logits, [height as usize, width as usize]);
        matte_frame::to_matte(sigmoid(logits), width, height)
    }
}

/// Reads SAM 2.1 hiera tiny's `model.safetensors` as transformers publishes
/// it, and writes it to `to` as `.bpk` for [`FollowModel::load`]. Run once,
/// offline.
pub fn convert_sam2(from: &Path, to: &Path) -> Result<(), VisionError> {
    let refused = |path: &Path, message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !from.is_file() {
        return Err(refused(from, "missing".into()));
    }
    let device = Device::flex();
    let mut model = Sam2::new(&device);
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
