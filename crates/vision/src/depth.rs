//! Depth for Relight: Depth Anything V2 Small on burn, which guesses how
//! near each pixel of a frame is.
//!
//! A DINOv2 vision transformer (ViT-S/14, twelve layers) reads the frame in
//! 14 pixel patches. A DPT head joins four of its layers back into a
//! picture at the frame's size. The answer is relative inverse depth:
//! larger is nearer, with no unit, so each frame is scaled by its own
//! spread before it is kept. About 25 million weights. The small model is
//! Apache-2.0; the larger ones are not, and are not used.
//!
//! The frame is scaled so its short side is [`DEPTH_SIDE`] and both sides
//! are whole patches. The model runs in full precision: depth is turned
//! into the direction each surface faces, which reads small differences
//! between neighbours.

use std::path::Path;
use std::sync::Mutex;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::{gelu, relu, softmax};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::device::{self, Compute};
use crate::mobile_sam::load_weights;
use crate::{Image, VisionError};

/// The weights, in the folder passed to [`DepthModel::load`].
pub const DEPTH_WEIGHTS: &str = "depth_anything_v2_small.bpk";

/// The short side the model reads a frame at, in pixels.
pub const DEPTH_SIDE: u32 = 518;

/// The longest side a frame may have.
pub const DEPTH_MAX_SIDE: u32 = 8192;

/// The model's patch, in pixels.
const PATCH: usize = 14;
/// The transformer's width, heads and layers.
const WIDTH: usize = 384;
const HEADS: usize = 6;
const LAYERS: usize = 12;
/// The patches a side the position embeddings were learned for (518 / 14).
const LEARNED: usize = 37;
/// The layers the head reads, counted from 1.
const TAPPED: [usize; 4] = [3, 6, 9, 12];
/// Each tapped layer's channels in the head.
const NECK: [usize; 4] = [48, 96, 192, 384];
/// The width the head joins the layers at.
const FUSION: usize = 64;
/// ImageNet's mean and spread, which the model was trained on.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const SPREAD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Module, Debug)]
struct PatchEmbeddings {
    projection: Conv2d,
}

#[derive(Module, Debug)]
struct Embeddings {
    cls_token: Param<Tensor<3>>,
    position_embeddings: Param<Tensor<3>>,
    patch_embeddings: PatchEmbeddings,
}

#[derive(Module, Debug)]
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
}

#[derive(Module, Debug)]
struct AttentionOutput {
    dense: Linear,
}

#[derive(Module, Debug)]
struct Attention {
    attention: SelfAttention,
    output: AttentionOutput,
}

#[derive(Module, Debug)]
struct LayerScale {
    lambda1: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

#[derive(Module, Debug)]
struct Layer {
    norm1: LayerNorm,
    attention: Attention,
    layer_scale1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    layer_scale2: LayerScale,
}

#[derive(Module, Debug)]
struct Encoder {
    layer: Vec<Layer>,
}

#[derive(Module, Debug)]
struct Backbone {
    embeddings: Embeddings,
    encoder: Encoder,
    layernorm: LayerNorm,
}

/// One tapped layer brought to its own size: 4 times the patches, 2
/// times, as is, or half.
#[derive(Module, Debug)]
struct Reassemble {
    projection: Conv2d,
    up: Option<ConvTranspose2d>,
    down: Option<Conv2d>,
}

#[derive(Module, Debug)]
struct ReassembleStage {
    layers: Vec<Reassemble>,
}

#[derive(Module, Debug)]
struct Residual {
    convolution1: Conv2d,
    convolution2: Conv2d,
}

#[derive(Module, Debug)]
struct Fusion {
    projection: Conv2d,
    residual_layer1: Residual,
    residual_layer2: Residual,
}

#[derive(Module, Debug)]
struct FusionStage {
    layers: Vec<Fusion>,
}

#[derive(Module, Debug)]
struct Neck {
    reassemble_stage: ReassembleStage,
    convs: Vec<Conv2d>,
    fusion_stage: FusionStage,
}

#[derive(Module, Debug)]
struct Head {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
}

/// Depth Anything V2, named as its published weights name it.
#[derive(Module, Debug)]
struct DepthAnything {
    backbone: Backbone,
    neck: Neck,
    head: Head,
}

fn linear(input: usize, output: usize, device: &Device) -> Linear {
    LinearConfig::new(input, output).init(device)
}

fn conv(input: usize, output: usize, kernel: usize, bias: bool, device: &Device) -> Conv2d {
    Conv2dConfig::new([input, output], [kernel, kernel])
        .with_padding(PaddingConfig2d::Explicit(
            kernel / 2,
            kernel / 2,
            kernel / 2,
            kernel / 2,
        ))
        .with_bias(bias)
        .init(device)
}

fn norm(device: &Device) -> LayerNorm {
    LayerNormConfig::new(WIDTH).with_epsilon(1e-6).init(device)
}

impl DepthAnything {
    fn new(device: &Device) -> Self {
        let layer = || Layer {
            norm1: norm(device),
            attention: Attention {
                attention: SelfAttention {
                    query: linear(WIDTH, WIDTH, device),
                    key: linear(WIDTH, WIDTH, device),
                    value: linear(WIDTH, WIDTH, device),
                },
                output: AttentionOutput {
                    dense: linear(WIDTH, WIDTH, device),
                },
            },
            layer_scale1: LayerScale {
                lambda1: Param::from_tensor(Tensor::ones([WIDTH], device)),
            },
            norm2: norm(device),
            mlp: Mlp {
                fc1: linear(WIDTH, 4 * WIDTH, device),
                fc2: linear(4 * WIDTH, WIDTH, device),
            },
            layer_scale2: LayerScale {
                lambda1: Param::from_tensor(Tensor::ones([WIDTH], device)),
            },
        };
        let reassemble = |index: usize, channels: usize| Reassemble {
            projection: conv(WIDTH, channels, 1, true, device),
            up: match index {
                0 | 1 => {
                    let factor = 4 >> index;
                    Some(
                        ConvTranspose2dConfig::new([channels, channels], [factor, factor])
                            .with_stride([factor, factor])
                            .init(device),
                    )
                }
                _ => None,
            },
            down: (index == 3).then(|| {
                Conv2dConfig::new([channels, channels], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init(device)
            }),
        };
        let residual = || Residual {
            convolution1: conv(FUSION, FUSION, 3, true, device),
            convolution2: conv(FUSION, FUSION, 3, true, device),
        };
        Self {
            backbone: Backbone {
                embeddings: Embeddings {
                    cls_token: Param::from_tensor(Tensor::zeros([1, 1, WIDTH], device)),
                    position_embeddings: Param::from_tensor(Tensor::zeros(
                        [1, LEARNED * LEARNED + 1, WIDTH],
                        device,
                    )),
                    patch_embeddings: PatchEmbeddings {
                        projection: Conv2dConfig::new([3, WIDTH], [PATCH, PATCH])
                            .with_stride([PATCH, PATCH])
                            .init(device),
                    },
                },
                encoder: Encoder {
                    layer: (0..LAYERS).map(|_| layer()).collect(),
                },
                layernorm: norm(device),
            },
            neck: Neck {
                reassemble_stage: ReassembleStage {
                    layers: NECK
                        .iter()
                        .enumerate()
                        .map(|(index, channels)| reassemble(index, *channels))
                        .collect(),
                },
                convs: NECK
                    .iter()
                    .map(|channels| conv(*channels, FUSION, 3, false, device))
                    .collect(),
                fusion_stage: FusionStage {
                    layers: (0..NECK.len())
                        .map(|_| Fusion {
                            projection: conv(FUSION, FUSION, 1, true, device),
                            residual_layer1: residual(),
                            residual_layer2: residual(),
                        })
                        .collect(),
                },
            },
            head: Head {
                conv1: conv(FUSION, FUSION / 2, 3, true, device),
                conv2: conv(FUSION / 2, 32, 3, true, device),
                conv3: conv(32, 1, 1, true, device),
            },
        }
    }
}

impl Layer {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, tokens, _] = x.dims();
        let head = WIDTH / HEADS;
        let normed = self.norm1.forward(x.clone());
        let split = |t: Tensor<3>| t.reshape([batch, tokens, HEADS, head]).swap_dims(1, 2);
        let own = &self.attention.attention;
        let q = split(own.query.forward(normed.clone()));
        let k = split(own.key.forward(normed.clone()));
        let v = split(own.value.forward(normed));
        let scores = q.matmul(k.swap_dims(2, 3)) / (head as f64).sqrt();
        let mixed = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, tokens, WIDTH]);
        let attended = self.attention.output.dense.forward(mixed);
        let x = x + attended * self.layer_scale1.lambda1.val().reshape([1, 1, WIDTH]);
        let normed = self.norm2.forward(x.clone());
        let fed = self.mlp.fc2.forward(gelu(self.mlp.fc1.forward(normed)));
        x + fed * self.layer_scale2.lambda1.val().reshape([1, 1, WIDTH])
    }
}

impl Residual {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = self.convolution1.forward(relu(x.clone()));
        self.convolution2.forward(relu(y)) + x
    }
}

/// `x` resized bilinearly to `size`, corners on corners.
fn resize(x: Tensor<4>, size: [usize; 2]) -> Tensor<4> {
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

impl Fusion {
    fn forward(
        &self,
        x: Tensor<4>,
        skip: Option<Tensor<4>>,
        size: Option<[usize; 2]>,
    ) -> Tensor<4> {
        let x = match skip {
            Some(skip) => x + self.residual_layer1.forward(skip),
            None => x,
        };
        let x = self.residual_layer2.forward(x);
        let [_, _, h, w] = x.dims();
        let x = resize(x, size.unwrap_or([2 * h, 2 * w]));
        self.projection.forward(x)
    }
}

/// Cubic convolution with A = -0.75, as PyTorch's bicubic weighs.
fn cubic(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let near = |x: f64| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let far = |x: f64| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
}

/// Bicubic resampling of `from` (`side` by `side` rows of `WIDTH`) to `h`
/// by `w`, corners not aligned and the edge held, as PyTorch's
/// `interpolate(size=..., mode="bicubic", align_corners=False)` does.
fn bicubic(from: &[f32], side: usize, h: usize, w: usize) -> Vec<f32> {
    let taps = |out: usize, i: usize| -> ([usize; 4], [f64; 4]) {
        let scale = side as f64 / out as f64;
        let x = (i as f64 + 0.5) * scale - 0.5;
        let base = x.floor();
        let weights = cubic(x - base);
        let at = |k: i64| (base as i64 + k).clamp(0, side as i64 - 1) as usize;
        ([at(-1), at(0), at(1), at(2)], weights)
    };
    let rows: Vec<_> = (0..h).map(|y| taps(h, y)).collect();
    let cols: Vec<_> = (0..w).map(|x| taps(w, x)).collect();
    let mut out = vec![0.0_f32; h * w * WIDTH];
    for (y, (ry, wy)) in rows.iter().enumerate() {
        for (x, (cx, wx)) in cols.iter().enumerate() {
            let cell = &mut out[(y * w + x) * WIDTH..(y * w + x + 1) * WIDTH];
            for (a, row) in ry.iter().enumerate() {
                for (b, col) in cx.iter().enumerate() {
                    let weight = (wy[a] * wx[b]) as f32;
                    let src = &from[(row * side + col) * WIDTH..(row * side + col + 1) * WIDTH];
                    for (sum, value) in cell.iter_mut().zip(src) {
                        *sum += weight * value;
                    }
                }
            }
        }
    }
    out
}

/// One frame's depth: relative inverse depth, larger nearer, at the size
/// the model read the frame.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

/// The depth model, loaded on one device.
pub struct DepthModel {
    device: Device,
    model: DepthAnything,
    /// The position embeddings resampled for the last patch grid read.
    positions: Mutex<Option<([usize; 2], Tensor<3>)>>,
}

impl DepthModel {
    /// Loads [`DEPTH_WEIGHTS`] from the folder `weights`.
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let mut model = DepthAnything::new(&device);
        load_weights(&mut model, &weights.join(DEPTH_WEIGHTS))?;
        Ok(Self::on(device, model))
    }

    /// The model with random weights: the same work as the real one, for
    /// timing and for tests that only check shapes.
    #[cfg(test)]
    pub(crate) fn untrained(compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let model = DepthAnything::new(&device);
        Ok(Self::on(device, model))
    }

    fn on(device: Device, model: DepthAnything) -> Self {
        Self {
            device,
            model,
            positions: Mutex::new(None),
        }
    }

    /// The patch grid a `width` by `height` frame is read at: its short
    /// side [`DEPTH_SIDE`], each side a whole number of patches.
    #[must_use]
    pub fn grid(width: u32, height: u32) -> (u32, u32) {
        Self::grid_at(width, height, DEPTH_SIDE)
    }

    /// [`Self::grid`] with the short side `side` pixels, at least a patch.
    #[must_use]
    pub fn grid_at(width: u32, height: u32, side: u32) -> (u32, u32) {
        let short = width.min(height).max(1);
        let scale = f64::from(side.max(PATCH as u32)) / f64::from(short);
        let patches = |side: u32| ((f64::from(side) * scale / PATCH as f64).round() as u32).max(1);
        (patches(width), patches(height))
    }

    /// The depth of `image`, read with its short side [`DEPTH_SIDE`].
    pub fn estimate(&self, image: &Image<'_>) -> Result<DepthMap, VisionError> {
        self.estimate_at(image, DEPTH_SIDE)
    }

    /// The depth of `image`, read with its short side `side` pixels: less
    /// detail, in less time, about as the square of the side.
    pub fn estimate_at(&self, image: &Image<'_>, side: u32) -> Result<DepthMap, VisionError> {
        let (width, height) = check(image)?;
        let (pw, ph) = Self::grid_at(width, height, side);
        let (pw, ph) = (pw as usize, ph as usize);
        let plane = (width * height) as usize;
        let mut chw = vec![0.0_f32; 3 * plane];
        for (index, pixel) in image.rgb.as_chunks::<3>().0.iter().enumerate() {
            for (c, byte) in pixel.iter().enumerate() {
                chw[c * plane + index] = (f32::from(*byte) / 255.0 - MEAN[c]) / SPREAD[c];
            }
        }
        let frame = Tensor::<4>::from_data(
            TensorData::new(chw, [1, 3, height as usize, width as usize]),
            &self.device,
        );
        let frame = if [height as usize, width as usize] == [ph * PATCH, pw * PATCH] {
            frame
        } else {
            // The frame's own corners stay its corners, as a picture
            // scaled for the screen.
            interpolate(
                frame,
                InterpolateOptions::new(InterpolateMode::Bilinear)
                    .with_align_corners(false)
                    .with_output_size([ph * PATCH, pw * PATCH]),
            )
        };
        let depth = self.forward(frame, ph, pw)?;
        let values = depth
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .map_err(|error| VisionError::Model {
                message: format!("{error:?}"),
            })?;
        Ok(DepthMap {
            width: (pw * PATCH) as u32,
            height: (ph * PATCH) as u32,
            values,
        })
    }

    /// The position embeddings for a `ph` by `pw` patch grid, the class
    /// token's first.
    fn positions(&self, ph: usize, pw: usize) -> Result<Tensor<3>, VisionError> {
        let learned = self.model.backbone.embeddings.position_embeddings.val();
        if [ph, pw] == [LEARNED, LEARNED] {
            return Ok(learned);
        }
        let mut kept = self.positions.lock().map_err(|_| VisionError::Model {
            message: "the position embeddings were left half made".into(),
        })?;
        if let Some((grid, tensor)) = kept.as_ref()
            && *grid == [ph, pw]
        {
            return Ok(tensor.clone());
        }
        let class = learned.clone().slice([0..1, 0..1]);
        let patches = learned
            .slice([0..1, 1..LEARNED * LEARNED + 1])
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .map_err(|error| VisionError::Model {
                message: format!("{error:?}"),
            })?;
        let grown = bicubic(&patches, LEARNED, ph, pw);
        let grown =
            Tensor::<3>::from_data(TensorData::new(grown, [1, ph * pw, WIDTH]), &self.device);
        let tensor = Tensor::cat(vec![class, grown], 1);
        *kept = Some(([ph, pw], tensor.clone()));
        Ok(tensor)
    }

    fn forward(&self, frame: Tensor<4>, ph: usize, pw: usize) -> Result<Tensor<4>, VisionError> {
        let model = &self.model;
        let embeddings = &model.backbone.embeddings;
        let patches = embeddings
            .patch_embeddings
            .projection
            .forward(frame)
            .reshape([1, WIDTH, ph * pw])
            .swap_dims(1, 2);
        let mut x =
            Tensor::cat(vec![embeddings.cls_token.val(), patches], 1) + self.positions(ph, pw)?;
        let mut tapped = Vec::with_capacity(TAPPED.len());
        for (index, layer) in model.backbone.encoder.layer.iter().enumerate() {
            x = layer.forward(x);
            if TAPPED.contains(&(index + 1)) {
                tapped.push(model.backbone.layernorm.forward(x.clone()));
            }
        }
        let features: Vec<Tensor<4>> = tapped
            .into_iter()
            .zip(&model.neck.reassemble_stage.layers)
            .zip(&model.neck.convs)
            .map(|((tokens, reassemble), conv)| {
                let grid = tokens
                    .slice([0..1, 1..ph * pw + 1])
                    .reshape([1, ph, pw, WIDTH])
                    .permute([0, 3, 1, 2]);
                let mut map = reassemble.projection.forward(grid);
                if let Some(up) = &reassemble.up {
                    map = up.forward(map);
                }
                if let Some(down) = &reassemble.down {
                    map = down.forward(map);
                }
                conv.forward(map)
            })
            .collect();
        // Deepest first, each step brought to the next one's size.
        let mut fused: Option<Tensor<4>> = None;
        for (step, fusion) in model.neck.fusion_stage.layers.iter().enumerate() {
            let index = features.len() - 1 - step;
            let size = index
                .checked_sub(1)
                .and_then(|next| features.get(next))
                .map(|next| {
                    let [_, _, h, w] = next.dims();
                    [h, w]
                });
            let feature = features[index].clone();
            fused = Some(match fused {
                None => fusion.forward(feature, None, size),
                Some(fused) => fusion.forward(fused, Some(feature), size),
            });
        }
        let fused = fused.ok_or_else(|| VisionError::Model {
            message: "the head joined no layers".into(),
        })?;
        let head = &model.head;
        let y = resize(head.conv1.forward(fused), [ph * PATCH, pw * PATCH]);
        let y = relu(head.conv2.forward(y));
        Ok(relu(head.conv3.forward(y)))
    }
}

fn check(image: &Image<'_>) -> Result<(u32, u32), VisionError> {
    let (width, height) = (image.width, image.height);
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3));
    if width == 0
        || height == 0
        || width > DEPTH_MAX_SIDE
        || height > DEPTH_MAX_SIDE
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

/// Reads Depth Anything V2 Small's `model.safetensors` as published for
/// `transformers`, and writes it to `to` as `.bpk` for
/// [`DepthModel::load`]. Run once, offline.
pub fn convert_depth_anything(from: &Path, to: &Path) -> Result<(), VisionError> {
    let refused = |path: &Path, message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !from.is_file() {
        return Err(refused(from, "missing".into()));
    }
    let device = Device::flex();
    let mut model = DepthAnything::new(&device);
    let mut store = SafetensorsStore::from_file(from)
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"(reassemble_stage\.layers\.[01])\.resize\.", "$1.up.")
        .with_key_remapping(r"(reassemble_stage\.layers\.3)\.resize\.", "$1.down.")
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
