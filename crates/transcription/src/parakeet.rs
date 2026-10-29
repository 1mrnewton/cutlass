use std::path::Path;

use candle_core::{D, Device, Tensor};
use candle_nn::{
    BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, ModuleT,
};
use safetensors::tensor::SafeTensors;

use crate::features::{self, ENCODER_HOP_MS, N_MELS};
use crate::prepare::Pcm;
use crate::search::{self, Span};
use crate::words::{RecognizedWord, Utterance};
use crate::{ModelError, RecognizeError, Recognizer};

pub(crate) const HIDDEN: usize = 1024;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
pub(crate) const INTERMEDIATE: usize = 4096;
const VOCAB: usize = 1025;
const BLANK: usize = 1024;
pub(crate) const LAYERS: usize = 24;
pub(crate) const SUB_CHANNELS: usize = 256;
const SUB_FLAT: usize = 2560;
const INPUT_SCALE: f64 = 32.0;
pub(crate) const NORM_EPS: f64 = 1.0e-5;
/// Word-piece boundary in the vocabulary.
const BOUNDARY: char = '\u{2581}';

/// FastConformer CTC.
///
/// `open` reads `model.safetensors` and `tokenizer.json` from `dir`. The graph
/// is the 0.6B English checkpoint: 24 layers, 80 mel bins, vocabulary 1025,
/// blank at 1024. Tensors are built on `device`. [`Parakeet::cpu`] uses the CPU.
pub struct Parakeet {
    device: Device,
    subsampling: Subsampling,
    inv_freq: Tensor,
    blocks: Vec<Block>,
    ctc: Conv1d,
    vocab: Vec<String>,
}

pub(crate) struct Subsampling {
    pub(crate) conv: Conv2d,
    pub(crate) depthwise1: Conv2d,
    pub(crate) pointwise1: Conv2d,
    pub(crate) depthwise2: Conv2d,
    pub(crate) pointwise2: Conv2d,
    pub(crate) linear: Linear,
}

pub(crate) struct FeedForward {
    pub(crate) up: Linear,
    pub(crate) down: Linear,
}

pub(crate) struct Attention {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) output: Linear,
    pub(crate) relative: Linear,
    pub(crate) bias_u: Tensor,
    pub(crate) bias_v: Tensor,
}

pub(crate) struct Convolution {
    pub(crate) expand: Conv1d,
    pub(crate) depthwise: Conv1d,
    pub(crate) norm: BatchNorm,
    pub(crate) project: Conv1d,
}

pub(crate) struct Block {
    pub(crate) feed_forward1: FeedForward,
    pub(crate) feed_forward2: FeedForward,
    pub(crate) attention: Attention,
    pub(crate) convolution: Convolution,
    pub(crate) norm_feed_forward1: LayerNorm,
    pub(crate) norm_attention: LayerNorm,
    pub(crate) norm_convolution: LayerNorm,
    pub(crate) norm_feed_forward2: LayerNorm,
    pub(crate) norm_out: LayerNorm,
}

impl Parakeet {
    pub fn cpu(dir: &Path) -> Result<Self, ModelError> {
        Self::open(&Device::Cpu, dir)
    }

    pub fn open(device: &Device, dir: &Path) -> Result<Self, ModelError> {
        let bytes =
            std::fs::read(dir.join("model.safetensors")).map_err(|_| ModelError::Weights)?;
        let file = SafeTensors::deserialize(&bytes).map_err(|_| ModelError::Weights)?;
        let subsampling = load_subsampling(&file, device)?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            blocks.push(load_block(&file, index, device)?);
        }
        let ctc = conv1d(
            &file,
            "ctc_head",
            VOCAB,
            HIDDEN,
            1,
            conv1d_config(0, 1, 1),
            device,
        )?;
        let vocab = load_vocab(&dir.join("tokenizer.json"))?;
        Ok(Self {
            device: device.clone(),
            subsampling,
            inv_freq: inv_freq(device)?,
            blocks,
            ctc,
            vocab,
        })
    }

    fn logits(&self, mel: &Tensor) -> Result<Vec<Vec<f32>>, ModelError> {
        let hidden = subsample(&self.subsampling, mel)?;
        let time = hidden.dim(1).map_err(|_| ModelError::Shape)?;
        if time == 0 {
            return Ok(Vec::new());
        }
        let positions = positions(&self.inv_freq, time)?;
        let mut hidden = hidden;
        for block in &self.blocks {
            hidden = block_forward(block, &hidden, &positions)?;
        }
        let logits = ctc_forward(&self.ctc, &hidden)?;
        logits.to_vec2::<f32>().map_err(|_| ModelError::Device)
    }
}

impl Recognizer for Parakeet {
    fn sample_rate(&self) -> u32 {
        features::SAMPLE_RATE
    }

    fn recognize(&self, pcm: &Pcm) -> Result<Vec<Utterance>, RecognizeError> {
        if pcm.sample_rate != features::SAMPLE_RATE {
            return Err(RecognizeError::SampleRate {
                expected: features::SAMPLE_RATE,
                actual: pcm.sample_rate,
            });
        }
        let mel = features::conformer_mel(&pcm.samples).map_err(|_| RecognizeError::Model)?;
        let frames = mel.len() / N_MELS;
        if frames == 0 {
            return Ok(Vec::new());
        }
        let mel = Tensor::from_vec(mel, (1, frames, N_MELS), &self.device)
            .map_err(|_| RecognizeError::Model)?;
        let rows = self.logits(&mel).map_err(|_| RecognizeError::Model)?;
        let spans = search::collapse(&rows, VOCAB, BLANK).map_err(|_| RecognizeError::Search)?;
        let words = pieces_to_words(&spans, &self.vocab, ENCODER_HOP_MS, false)
            .map_err(|_| RecognizeError::Search)?;
        if words.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![Utterance { words }])
        }
    }
}

pub(crate) fn pieces_to_words(
    spans: &[Span],
    vocab: &[String],
    hop_ms: i64,
    lone_boundary_splits: bool,
) -> Result<Vec<RecognizedWord>, search::SearchError> {
    let mut open: Option<OpenWord> = None;
    let mut words = Vec::new();
    for span in spans {
        let Some(piece) = vocab.get(span.token) else {
            return Err(search::SearchError::Shape);
        };
        if piece.is_empty() || piece == "<unk>" {
            continue;
        }
        let (boundary, body) = match piece.strip_prefix(BOUNDARY) {
            Some(body) => (true, body),
            None => (false, piece.as_str()),
        };
        if body == "<unk>" {
            continue;
        }
        if boundary && (lone_boundary_splits || !body.is_empty()) {
            finish_word(&mut words, open.take(), hop_ms)?;
        }
        if body.is_empty() {
            continue;
        }
        match open.as_mut() {
            Some(word) => {
                word.text.push_str(body);
                word.end = span.end;
            }
            None => {
                open = Some(OpenWord {
                    text: body.to_owned(),
                    start: span.start,
                    end: span.end,
                });
            }
        }
    }
    finish_word(&mut words, open, hop_ms)?;
    Ok(words)
}

struct OpenWord {
    text: String,
    start: usize,
    end: usize,
}

fn finish_word(
    words: &mut Vec<RecognizedWord>,
    open: Option<OpenWord>,
    hop_ms: i64,
) -> Result<(), search::SearchError> {
    let Some(open) = open else {
        return Ok(());
    };
    if open.text.is_empty() {
        return Ok(());
    }
    let (start_ms, end_ms) = crate::timestamp::span_ms(open.start, open.end, hop_ms)?;
    words.push(RecognizedWord {
        text: open.text,
        start_ms,
        end_ms,
    });
    Ok(())
}

pub(crate) fn subsample(layer: &Subsampling, mel: &Tensor) -> Result<Tensor, ModelError> {
    let mut hidden = mel.unsqueeze(1).map_err(|_| ModelError::Device)?;
    hidden = conv2d_relu(&layer.conv, &hidden)?;
    hidden = layer
        .depthwise1
        .forward(&hidden)
        .map_err(|_| ModelError::Device)?;
    hidden = conv2d_relu(&layer.pointwise1, &hidden)?;
    hidden = layer
        .depthwise2
        .forward(&hidden)
        .map_err(|_| ModelError::Device)?;
    hidden = conv2d_relu(&layer.pointwise2, &hidden)?;
    let time = hidden.dim(2).map_err(|_| ModelError::Shape)?;
    let freq = hidden.dim(3).map_err(|_| ModelError::Shape)?;
    let Some(flat) = freq.checked_mul(SUB_CHANNELS) else {
        return Err(ModelError::Shape);
    };
    let hidden = hidden
        .transpose(1, 2)
        .and_then(|hidden| hidden.reshape((1, time, flat)))
        .map_err(|_| ModelError::Device)?;
    let hidden = layer
        .linear
        .forward(&hidden)
        .map_err(|_| ModelError::Device)?;
    hidden
        .affine(INPUT_SCALE, 0.0)
        .map_err(|_| ModelError::Device)
}

fn conv2d_relu(layer: &Conv2d, hidden: &Tensor) -> Result<Tensor, ModelError> {
    layer
        .forward(hidden)
        .and_then(|hidden| hidden.relu())
        .map_err(|_| ModelError::Device)
}

pub(crate) fn block_forward(
    block: &Block,
    hidden: &Tensor,
    positions: &Tensor,
) -> Result<Tensor, ModelError> {
    let ff1 = feed_forward(&block.feed_forward1, &block.norm_feed_forward1, hidden)?;
    let hidden = scaled_add(hidden, &ff1, 0.5)?;
    let normed = block
        .norm_attention
        .forward(&hidden)
        .map_err(|_| ModelError::Device)?;
    let attended = attend(&block.attention, &normed, positions)?;
    let hidden = scaled_add(&hidden, &attended, 1.0)?;
    let normed = block
        .norm_convolution
        .forward(&hidden)
        .map_err(|_| ModelError::Device)?;
    let convolved = convolve(&block.convolution, &normed)?;
    let hidden = scaled_add(&hidden, &convolved, 1.0)?;
    let ff2 = feed_forward(&block.feed_forward2, &block.norm_feed_forward2, &hidden)?;
    let hidden = scaled_add(&hidden, &ff2, 0.5)?;
    block
        .norm_out
        .forward(&hidden)
        .map_err(|_| ModelError::Device)
}

fn feed_forward(
    layer: &FeedForward,
    norm: &LayerNorm,
    hidden: &Tensor,
) -> Result<Tensor, ModelError> {
    let hidden = norm.forward(hidden).map_err(|_| ModelError::Device)?;
    let hidden = layer.up.forward(&hidden).map_err(|_| ModelError::Device)?;
    let hidden = hidden.silu().map_err(|_| ModelError::Device)?;
    layer.down.forward(&hidden).map_err(|_| ModelError::Device)
}

fn scaled_add(residual: &Tensor, delta: &Tensor, scale: f64) -> Result<Tensor, ModelError> {
    let delta = delta.affine(scale, 0.0).map_err(|_| ModelError::Device)?;
    residual
        .broadcast_add(&delta)
        .map_err(|_| ModelError::Device)
}

fn attend(layer: &Attention, hidden: &Tensor, positions: &Tensor) -> Result<Tensor, ModelError> {
    let dims = hidden.dims();
    if dims.len() != 3 || dims[2] != HIDDEN {
        return Err(ModelError::Shape);
    }
    let batch = dims[0];
    let time = dims[1];
    let query = split_heads(
        &layer
            .query
            .forward(hidden)
            .map_err(|_| ModelError::Device)?,
    )?;
    let key = split_heads(&layer.key.forward(hidden).map_err(|_| ModelError::Device)?)?;
    let value = split_heads(
        &layer
            .value
            .forward(hidden)
            .map_err(|_| ModelError::Device)?,
    )?;
    let bias_u = layer
        .bias_u
        .reshape((1, HEADS, 1, HEAD_DIM))
        .map_err(|_| ModelError::Device)?;
    let bias_v = layer
        .bias_v
        .reshape((1, HEADS, 1, HEAD_DIM))
        .map_err(|_| ModelError::Device)?;
    let query_u = query
        .broadcast_add(&bias_u)
        .map_err(|_| ModelError::Device)?;
    let query_v = query
        .broadcast_add(&bias_v)
        .map_err(|_| ModelError::Device)?;
    let relative = layer
        .relative
        .forward(positions)
        .map_err(|_| ModelError::Device)?;
    let relative = relative
        .reshape((batch, (), HEADS, HEAD_DIM))
        .and_then(|relative| relative.permute((0, 2, 3, 1)))
        .map_err(|_| ModelError::Device)?;
    let shift = query_v.matmul(&relative).map_err(|_| ModelError::Device)?;
    let shift = rel_shift(&shift)?;
    if shift.dim(3).map_err(|_| ModelError::Shape)? < time {
        return Err(ModelError::Shape);
    }
    let shift = shift.narrow(3, 0, time).map_err(|_| ModelError::Device)?;
    let scale = (HEAD_DIM as f64).sqrt().recip();
    let shift = shift.affine(scale, 0.0).map_err(|_| ModelError::Device)?;
    let key = key.transpose(2, 3).map_err(|_| ModelError::Device)?;
    let content = query_u
        .matmul(&key)
        .and_then(|content| content.affine(scale, 0.0))
        .map_err(|_| ModelError::Device)?;
    let scores = content
        .broadcast_add(&shift)
        .map_err(|_| ModelError::Device)?;
    let weights = candle_nn::ops::softmax(&scores, D::Minus1).map_err(|_| ModelError::Device)?;
    let mixed = weights.matmul(&value).map_err(|_| ModelError::Device)?;
    let mixed = mixed
        .transpose(1, 2)
        .and_then(|mixed| mixed.reshape((batch, time, HIDDEN)))
        .map_err(|_| ModelError::Device)?;
    layer.output.forward(&mixed).map_err(|_| ModelError::Device)
}

fn split_heads(states: &Tensor) -> Result<Tensor, ModelError> {
    let dims = states.dims();
    if dims.len() != 3 {
        return Err(ModelError::Shape);
    }
    states
        .reshape((dims[0], dims[1], HEADS, HEAD_DIM))
        .and_then(|states| states.transpose(1, 2))
        .map_err(|_| ModelError::Device)
}

/// Shaw relative shift. Pad one column, then drop the wrapped row.
fn rel_shift(scores: &Tensor) -> Result<Tensor, ModelError> {
    let dims = scores.dims();
    if dims.len() != 4 {
        return Err(ModelError::Shape);
    }
    let (batch, heads, queries, positions) = (dims[0], dims[1], dims[2], dims[3]);
    let zero = Tensor::zeros((batch, heads, queries, 1), scores.dtype(), scores.device())
        .map_err(|_| ModelError::Device)?;
    let padded = Tensor::cat(&[&zero, scores], 3).map_err(|_| ModelError::Device)?;
    padded
        .reshape((batch, heads, positions + 1, queries))
        .and_then(|padded| padded.narrow(2, 1, positions))
        .and_then(|shifted| shifted.reshape((batch, heads, queries, positions)))
        .map_err(|_| ModelError::Device)
}

fn convolve(layer: &Convolution, hidden: &Tensor) -> Result<Tensor, ModelError> {
    let hidden = hidden.transpose(1, 2).map_err(|_| ModelError::Device)?;
    let hidden = layer
        .expand
        .forward(&hidden)
        .map_err(|_| ModelError::Device)?;
    let hidden = glu(&hidden)?;
    let hidden = layer
        .depthwise
        .forward(&hidden)
        .map_err(|_| ModelError::Device)?;
    let hidden = layer
        .norm
        .forward_t(&hidden, false)
        .map_err(|_| ModelError::Device)?;
    let hidden = hidden.silu().map_err(|_| ModelError::Device)?;
    let hidden = layer
        .project
        .forward(&hidden)
        .map_err(|_| ModelError::Device)?;
    hidden.transpose(1, 2).map_err(|_| ModelError::Device)
}

fn glu(hidden: &Tensor) -> Result<Tensor, ModelError> {
    let channels = hidden.dim(1).map_err(|_| ModelError::Shape)?;
    if !channels.is_multiple_of(2) {
        return Err(ModelError::Shape);
    }
    let half = channels / 2;
    let left = hidden.narrow(1, 0, half).map_err(|_| ModelError::Device)?;
    let right = hidden
        .narrow(1, half, half)
        .map_err(|_| ModelError::Device)?;
    let gate = candle_nn::ops::sigmoid(&right).map_err(|_| ModelError::Device)?;
    left.broadcast_mul(&gate).map_err(|_| ModelError::Device)
}

pub(crate) fn positions(inv_freq: &Tensor, time: usize) -> Result<Tensor, ModelError> {
    if time == 0 {
        return Err(ModelError::Shape);
    }
    let mut ids = Vec::with_capacity(time * 2 - 1);
    let mut id = time as i64 - 1;
    let end = -(time as i64);
    while id > end {
        ids.push(id as f32);
        id -= 1;
    }
    let ids = Tensor::from_vec(ids, (time * 2 - 1, 1), inv_freq.device())
        .map_err(|_| ModelError::Device)?;
    let freq = inv_freq
        .reshape((1, HIDDEN / 2))
        .map_err(|_| ModelError::Device)?;
    let freqs = ids.broadcast_mul(&freq).map_err(|_| ModelError::Device)?;
    let sin = freqs
        .sin()
        .and_then(|sin| sin.unsqueeze(2))
        .map_err(|_| ModelError::Device)?;
    let cos = freqs
        .cos()
        .and_then(|cos| cos.unsqueeze(2))
        .map_err(|_| ModelError::Device)?;
    Tensor::cat(&[&sin, &cos], 2)
        .and_then(|stacked| stacked.reshape((1, time * 2 - 1, HIDDEN)))
        .map_err(|_| ModelError::Device)
}

pub(crate) fn inv_freq(device: &Device) -> Result<Tensor, ModelError> {
    let mut values = Vec::with_capacity(HIDDEN / 2);
    for index in 0..HIDDEN / 2 {
        let exponent = (2 * index) as f32 / HIDDEN as f32;
        values.push(10_000f32.powf(-exponent));
    }
    Tensor::from_vec(values, HIDDEN / 2, device).map_err(|_| ModelError::Device)
}

fn ctc_forward(layer: &Conv1d, hidden: &Tensor) -> Result<Tensor, ModelError> {
    hidden
        .transpose(1, 2)
        .and_then(|hidden| layer.forward(&hidden))
        .and_then(|hidden| hidden.transpose(1, 2))
        .and_then(|hidden| hidden.squeeze(0))
        .map_err(|_| ModelError::Device)
}

fn load_subsampling(file: &SafeTensors<'_>, device: &Device) -> Result<Subsampling, ModelError> {
    let wide = conv2d_config(1, 2, 1);
    let depth = conv2d_config(1, 2, SUB_CHANNELS);
    Ok(Subsampling {
        conv: conv2d(
            file,
            "encoder.subsampling.layers.0",
            SUB_CHANNELS,
            1,
            3,
            wide,
            device,
        )?,
        depthwise1: conv2d(
            file,
            "encoder.subsampling.layers.2",
            SUB_CHANNELS,
            1,
            3,
            depth,
            device,
        )?,
        pointwise1: conv2d(
            file,
            "encoder.subsampling.layers.3",
            SUB_CHANNELS,
            SUB_CHANNELS,
            1,
            conv2d_config(0, 1, 1),
            device,
        )?,
        depthwise2: conv2d(
            file,
            "encoder.subsampling.layers.5",
            SUB_CHANNELS,
            1,
            3,
            depth,
            device,
        )?,
        pointwise2: conv2d(
            file,
            "encoder.subsampling.layers.6",
            SUB_CHANNELS,
            SUB_CHANNELS,
            1,
            conv2d_config(0, 1, 1),
            device,
        )?,
        linear: linear(
            file,
            "encoder.subsampling.linear",
            HIDDEN,
            SUB_FLAT,
            true,
            device,
        )?,
    })
}

fn load_block(file: &SafeTensors<'_>, index: usize, device: &Device) -> Result<Block, ModelError> {
    let prefix = format!("encoder.layers.{index}");
    Ok(Block {
        feed_forward1: load_feed_forward(file, &format!("{prefix}.feed_forward1"), device)?,
        feed_forward2: load_feed_forward(file, &format!("{prefix}.feed_forward2"), device)?,
        attention: load_attention(file, &format!("{prefix}.self_attn"), device)?,
        convolution: load_convolution(file, &format!("{prefix}.conv"), device)?,
        norm_feed_forward1: layer_norm(file, &format!("{prefix}.norm_feed_forward1"), device)?,
        norm_attention: layer_norm(file, &format!("{prefix}.norm_self_att"), device)?,
        norm_convolution: layer_norm(file, &format!("{prefix}.norm_conv"), device)?,
        norm_feed_forward2: layer_norm(file, &format!("{prefix}.norm_feed_forward2"), device)?,
        norm_out: layer_norm(file, &format!("{prefix}.norm_out"), device)?,
    })
}

fn load_feed_forward(
    file: &SafeTensors<'_>,
    prefix: &str,
    device: &Device,
) -> Result<FeedForward, ModelError> {
    Ok(FeedForward {
        up: linear(
            file,
            &format!("{prefix}.linear1"),
            INTERMEDIATE,
            HIDDEN,
            true,
            device,
        )?,
        down: linear(
            file,
            &format!("{prefix}.linear2"),
            HIDDEN,
            INTERMEDIATE,
            true,
            device,
        )?,
    })
}

fn load_attention(
    file: &SafeTensors<'_>,
    prefix: &str,
    device: &Device,
) -> Result<Attention, ModelError> {
    Ok(Attention {
        query: linear(
            file,
            &format!("{prefix}.q_proj"),
            HIDDEN,
            HIDDEN,
            true,
            device,
        )?,
        key: linear(
            file,
            &format!("{prefix}.k_proj"),
            HIDDEN,
            HIDDEN,
            true,
            device,
        )?,
        value: linear(
            file,
            &format!("{prefix}.v_proj"),
            HIDDEN,
            HIDDEN,
            true,
            device,
        )?,
        output: linear(
            file,
            &format!("{prefix}.o_proj"),
            HIDDEN,
            HIDDEN,
            true,
            device,
        )?,
        relative: linear(
            file,
            &format!("{prefix}.relative_k_proj"),
            HIDDEN,
            HIDDEN,
            false,
            device,
        )?,
        bias_u: f32_tensor(
            file,
            &format!("{prefix}.bias_u"),
            &[HEADS, HEAD_DIM],
            device,
        )?,
        bias_v: f32_tensor(
            file,
            &format!("{prefix}.bias_v"),
            &[HEADS, HEAD_DIM],
            device,
        )?,
    })
}

fn load_convolution(
    file: &SafeTensors<'_>,
    prefix: &str,
    device: &Device,
) -> Result<Convolution, ModelError> {
    Ok(Convolution {
        expand: conv1d(
            file,
            &format!("{prefix}.pointwise_conv1"),
            HIDDEN * 2,
            HIDDEN,
            1,
            conv1d_config(0, 1, 1),
            device,
        )?,
        depthwise: conv1d(
            file,
            &format!("{prefix}.depthwise_conv"),
            HIDDEN,
            1,
            9,
            conv1d_config(4, 1, HIDDEN),
            device,
        )?,
        norm: batch_norm(file, &format!("{prefix}.norm"), device)?,
        project: conv1d(
            file,
            &format!("{prefix}.pointwise_conv2"),
            HIDDEN,
            HIDDEN,
            1,
            conv1d_config(0, 1, 1),
            device,
        )?,
    })
}

fn load_vocab(path: &Path) -> Result<Vec<String>, ModelError> {
    let bytes = std::fs::read(path).map_err(|_| ModelError::Weights)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| ModelError::Weights)?;
    let vocab = value
        .get("model")
        .and_then(|model| model.get("vocab"))
        .and_then(|vocab| vocab.as_object())
        .ok_or(ModelError::Weights)?;
    let mut pieces = vec![String::new(); VOCAB];
    for (piece, id) in vocab {
        let Some(id) = id.as_u64() else {
            return Err(ModelError::Weights);
        };
        let id = usize::try_from(id).map_err(|_| ModelError::Weights)?;
        let Some(slot) = pieces.get_mut(id) else {
            return Err(ModelError::Weights);
        };
        *slot = piece.clone();
    }
    if pieces.iter().take(BLANK).any(String::is_empty) {
        return Err(ModelError::Weights);
    }
    Ok(pieces)
}

fn linear(
    file: &SafeTensors<'_>,
    name: &str,
    rows: usize,
    columns: usize,
    bias: bool,
    device: &Device,
) -> Result<Linear, ModelError> {
    let weight = f32_tensor(file, &format!("{name}.weight"), &[rows, columns], device)?;
    let bias = if bias {
        Some(f32_tensor(file, &format!("{name}.bias"), &[rows], device)?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

fn layer_norm(
    file: &SafeTensors<'_>,
    name: &str,
    device: &Device,
) -> Result<LayerNorm, ModelError> {
    let weight = f32_tensor(file, &format!("{name}.weight"), &[HIDDEN], device)?;
    let bias = f32_tensor(file, &format!("{name}.bias"), &[HIDDEN], device)?;
    Ok(LayerNorm::new(weight, bias, NORM_EPS))
}

fn batch_norm(
    file: &SafeTensors<'_>,
    name: &str,
    device: &Device,
) -> Result<BatchNorm, ModelError> {
    let mean = f32_tensor(file, &format!("{name}.running_mean"), &[HIDDEN], device)?;
    let variance = f32_tensor(file, &format!("{name}.running_var"), &[HIDDEN], device)?;
    let weight = f32_tensor(file, &format!("{name}.weight"), &[HIDDEN], device)?;
    let bias = f32_tensor(file, &format!("{name}.bias"), &[HIDDEN], device)?;
    BatchNorm::new(HIDDEN, mean, variance, weight, bias, NORM_EPS).map_err(|_| ModelError::Device)
}

fn conv1d(
    file: &SafeTensors<'_>,
    name: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    config: Conv1dConfig,
    device: &Device,
) -> Result<Conv1d, ModelError> {
    let weight = f32_tensor(
        file,
        &format!("{name}.weight"),
        &[out_channels, in_channels, kernel],
        device,
    )?;
    let bias = f32_tensor(file, &format!("{name}.bias"), &[out_channels], device)?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn conv2d(
    file: &SafeTensors<'_>,
    name: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    config: Conv2dConfig,
    device: &Device,
) -> Result<Conv2d, ModelError> {
    let weight = f32_tensor(
        file,
        &format!("{name}.weight"),
        &[out_channels, in_channels, kernel, kernel],
        device,
    )?;
    let bias = f32_tensor(file, &format!("{name}.bias"), &[out_channels], device)?;
    Ok(Conv2d::new(weight, Some(bias), config))
}

fn conv1d_config(padding: usize, stride: usize, groups: usize) -> Conv1dConfig {
    Conv1dConfig {
        padding,
        stride,
        dilation: 1,
        groups,
        cudnn_fwd_algo: None,
    }
}

fn conv2d_config(padding: usize, stride: usize, groups: usize) -> Conv2dConfig {
    Conv2dConfig {
        padding,
        stride,
        dilation: 1,
        groups,
        cudnn_fwd_algo: None,
    }
}

fn f32_tensor(
    file: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor, ModelError> {
    let tensor = file.tensor(name).map_err(|_| ModelError::Weights)?;
    if tensor.dtype() != safetensors::Dtype::F32 || tensor.shape() != shape {
        return Err(ModelError::Shape);
    }
    if !tensor.data().len().is_multiple_of(4) {
        return Err(ModelError::Weights);
    }
    let mut values = Vec::with_capacity(tensor.data().len() / 4);
    for chunk in tensor.data().as_chunks::<4>().0 {
        values.push(f32::from_le_bytes(*chunk));
    }
    Tensor::from_vec(values, shape, device).map_err(|_| ModelError::Device)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use cutlass_model::{Draft, MaterialText, TimeRange};
    use cutlass_session::Session;

    use super::*;
    use crate::file::FileSource;
    use crate::job::Job;
    use crate::search::Span;

    #[test]
    fn a_missing_checkpoint_is_refused() {
        let error = Parakeet::cpu(Path::new("no-such-parakeet"));
        assert!(matches!(error, Err(ModelError::Weights)));
    }

    #[test]
    fn pieces_join_on_the_word_boundary() {
        let vocab = vec!["▁go".to_owned(), "ogle".to_owned(), "▁is".to_owned()];
        let spans = [
            Span {
                token: 0,
                start: 0,
                end: 1,
            },
            Span {
                token: 1,
                start: 2,
                end: 2,
            },
            Span {
                token: 2,
                start: 4,
                end: 4,
            },
        ];
        let words = pieces_to_words(&spans, &vocab, 80, false).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "google");
        assert_eq!((words[0].start_ms, words[0].end_ms), (0, 240));
        assert_eq!(words[1].text, "is");
        assert_eq!((words[1].start_ms, words[1].end_ms), (320, 400));
    }

    #[test]
    fn the_gemini_opening_becomes_words() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let model = root.join("parakeet-ctc-0.6b");
        let wav = root.join("GOOGLE IS BACK! (Gemini 3.8 Flash).wav");
        if !model.join("model.safetensors").is_file() || !wav.is_file() {
            return;
        }
        let recognizer = Parakeet::cpu(&model).unwrap();
        let mut draft = Draft::empty("D");
        let material_id = cutlass_model::MaterialId::new("caption");
        draft
            .materials
            .texts
            .push(MaterialText::new(material_id.clone()));
        let mut session = Session::open(draft);
        let job = Job::start(&session);
        let mut source = FileSource::new(&wav);
        job.recognize(
            &mut session,
            &material_id,
            &mut source,
            &TimeRange::new(0, 4_000_000),
            &recognizer,
        )
        .unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(
            text.words.text,
            ["google", "is", "back", "they", "just", "dropped", "gemini"]
        );
        assert_eq!(
            text.words.start_time,
            [160, 960, 1680, 2240, 2560, 2800, 3280]
        );
        assert_eq!(
            text.words.end_time,
            [640, 1040, 1760, 2320, 2640, 3200, 3840]
        );
        assert_eq!(text.current_words.text, text.words.text);
        assert_eq!(text.current_words.start_time, text.words.start_time);
        assert_eq!(text.current_words.end_time, text.words.end_time);
        assert_eq!(text.content, "google is back they just dropped gemini");
        assert_eq!(text.base_content, text.content);
    }
}
