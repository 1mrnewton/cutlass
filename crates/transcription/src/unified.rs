//! Parakeet unified English. FastConformer encoder, RNN-T greedy search.
//!
//! `open` reads `parakeet-unified-en-0.6b.nemo` from `dir`. The archive holds
//! the NeMo checkpoint and the sentence-piece vocabulary. Tensors are built
//! on `device`. [`ParakeetUnified::cpu`] uses the CPU. Audio longer than 30
//! seconds is recognized in chunks so the attention grid stays bounded.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module};

use crate::features::{self, ENCODER_HOP_MS, N_FFT, WINDOW};
use crate::parakeet::{
    self, Attention, Block, Convolution, FeedForward, HIDDEN, INTERMEDIATE, LAYERS, NORM_EPS,
    SUB_CHANNELS, Subsampling,
};
use crate::prepare::Pcm;
use crate::search::Span;
use crate::words::{RecognizedWord, Utterance};
use crate::{ModelError, RecognizeError, Recognizer};

const PRED: usize = 640;
const VOCAB: usize = 1024;
const CLASSES: usize = 1025;
const BLANK: usize = 1024;
const MELS: usize = 128;
const BINS: usize = N_FFT / 2 + 1;
const MAX_SYMBOLS: usize = 10;
const CHUNK_SAMPLES: usize = 16_000 * 30;
const ARCHIVE: &str = "parakeet-unified-en-0.6b.nemo";

/// FastConformer RNN-T.
pub struct ParakeetUnified {
    device: Device,
    window: Vec<f32>,
    filterbank: Vec<f32>,
    subsampling: Subsampling,
    inv_freq: Tensor,
    blocks: Vec<Block>,
    embed: Embedding,
    layer0: Gate,
    layer1: Gate,
    predict: Linear,
    encode: Linear,
    joint: Linear,
    vocab: Vec<String>,
}

struct Gate {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
}

impl ParakeetUnified {
    pub fn cpu(dir: &Path) -> Result<Self, ModelError> {
        Self::open(&Device::Cpu, dir)
    }

    pub fn open(device: &Device, dir: &Path) -> Result<Self, ModelError> {
        let nemo = dir.join(ARCHIVE);
        let yaml = read_text(&nemo, &["./model_config.yaml", "model_config.yaml"])?;
        let vocab = vocabulary(&yaml)?;
        let checkpoint = cached_checkpoint(&nemo)?;
        let weights = candle_core::pickle::PthTensors::new(&checkpoint, None)
            .map_err(|_| ModelError::Weights)?;
        let window = vector(&weights, "preprocessor.featurizer.window", &[WINDOW])?;
        let filterbank = vector(&weights, "preprocessor.featurizer.fb", &[1, MELS, BINS])?;
        Ok(Self {
            device: device.clone(),
            window,
            filterbank,
            subsampling: load_subsampling(&weights, device)?,
            inv_freq: parakeet::inv_freq(device)?,
            blocks: load_blocks(&weights, device)?,
            embed: Embedding::new(
                tensor(
                    &weights,
                    "decoder.prediction.embed.weight",
                    &[CLASSES, PRED],
                    device,
                )?,
                PRED,
            ),
            layer0: load_gate(&weights, 0, device)?,
            layer1: load_gate(&weights, 1, device)?,
            predict: linear(&weights, "joint.pred", PRED, PRED, true, device)?,
            encode: linear(&weights, "joint.enc", PRED, HIDDEN, true, device)?,
            joint: linear(&weights, "joint.joint_net.2", CLASSES, PRED, true, device)?,
            vocab,
        })
    }

    fn recognize_span(&self, samples: &[f32]) -> Result<Vec<RecognizedWord>, RecognizeError> {
        let mel = features::nemo_mel(samples, &self.window, &self.filterbank, MELS)
            .map_err(|_| RecognizeError::Model)?;
        let frames = mel.len() / MELS;
        if frames == 0 {
            return Ok(Vec::new());
        }
        let mel = Tensor::from_vec(mel, (1, frames, MELS), &self.device)
            .map_err(|_| RecognizeError::Model)?;
        let encoded = self.encode_audio(&mel).map_err(|_| RecognizeError::Model)?;
        let time = encoded.dim(0).map_err(|_| RecognizeError::Model)?;
        let (mut hidden, mut state) = self.prime(BLANK).map_err(|_| RecognizeError::Model)?;
        let mut spans = Vec::new();
        for frame in 0..time {
            let acoustic = encoded
                .narrow(0, frame, 1)
                .map_err(|_| RecognizeError::Model)?;
            for _ in 0..MAX_SYMBOLS {
                let token = self.choose(&acoustic, &hidden)?;
                if token == BLANK {
                    break;
                }
                spans.push(Span {
                    token,
                    start: frame,
                    end: frame,
                });
                (hidden, state) = self.step(token, state).map_err(|_| RecognizeError::Model)?;
            }
        }
        parakeet::pieces_to_words(&spans, &self.vocab, ENCODER_HOP_MS, true)
            .map_err(|_| RecognizeError::Search)
    }

    fn encode_audio(&self, mel: &Tensor) -> Result<Tensor, ModelError> {
        let mut hidden = parakeet::subsample(&self.subsampling, mel)?;
        let time = hidden.dim(1).map_err(|_| ModelError::Shape)?;
        if time == 0 {
            return Err(ModelError::Shape);
        }
        let positions = parakeet::positions(&self.inv_freq, time)?;
        for block in &self.blocks {
            hidden = parakeet::block_forward(block, &hidden, &positions)?;
        }
        hidden.squeeze(0).map_err(|_| ModelError::Device)
    }

    fn prime(&self, token: usize) -> Result<(Tensor, LstmState), ModelError> {
        let state = LstmState {
            h0: zeros(&self.device)?,
            c0: zeros(&self.device)?,
            h1: zeros(&self.device)?,
            c1: zeros(&self.device)?,
        };
        self.step(token, state)
    }

    fn step(&self, token: usize, state: LstmState) -> Result<(Tensor, LstmState), ModelError> {
        let token = u32::try_from(token).map_err(|_| ModelError::Shape)?;
        let ids =
            Tensor::from_vec(vec![token], (1,), &self.device).map_err(|_| ModelError::Device)?;
        let embedded = self.embed.forward(&ids).map_err(|_| ModelError::Device)?;
        let (h0, c0) = gate_step(&self.layer0, &embedded, &state.h0, &state.c0)?;
        let (h1, c1) = gate_step(&self.layer1, &h0, &state.h1, &state.c1)?;
        Ok((h1.clone(), LstmState { h0, c0, h1, c1 }))
    }

    fn choose(&self, acoustic: &Tensor, hidden: &Tensor) -> Result<usize, RecognizeError> {
        let acoustic = self
            .encode
            .forward(acoustic)
            .map_err(|_| RecognizeError::Model)?;
        let hidden = self
            .predict
            .forward(hidden)
            .map_err(|_| RecognizeError::Model)?;
        let mixed = acoustic
            .broadcast_add(&hidden)
            .and_then(|mixed| mixed.relu())
            .map_err(|_| RecognizeError::Model)?;
        let logits = self
            .joint
            .forward(&mixed)
            .and_then(|logits| logits.reshape(CLASSES))
            .and_then(|logits| logits.to_vec1::<f32>())
            .map_err(|_| RecognizeError::Model)?;
        if logits.len() != CLASSES {
            return Err(RecognizeError::Model);
        }
        let mut best = 0;
        let mut score = f32::NEG_INFINITY;
        for (index, value) in logits.iter().enumerate() {
            if !value.is_finite() {
                return Err(RecognizeError::Search);
            }
            if *value > score {
                score = *value;
                best = index;
            }
        }
        Ok(best)
    }
}

struct LstmState {
    h0: Tensor,
    c0: Tensor,
    h1: Tensor,
    c1: Tensor,
}

impl Recognizer for ParakeetUnified {
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
        let mut words = Vec::new();
        let mut offset = 0;
        while offset < pcm.samples.len() {
            let end = (offset + CHUNK_SAMPLES).min(pcm.samples.len());
            let mut part = self.recognize_span(&pcm.samples[offset..end])?;
            let shift = i64::try_from(offset / (features::SAMPLE_RATE as usize / 1_000))
                .map_err(|_| RecognizeError::WordTime)?;
            for word in &mut part {
                word.start_ms = word
                    .start_ms
                    .checked_add(shift)
                    .ok_or(RecognizeError::WordTime)?;
                word.end_ms = word
                    .end_ms
                    .checked_add(shift)
                    .ok_or(RecognizeError::WordTime)?;
            }
            words.append(&mut part);
            offset = end;
        }
        if words.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![Utterance { words }])
        }
    }
}

fn zeros(device: &Device) -> Result<Tensor, ModelError> {
    Tensor::zeros((1, PRED), DType::F32, device).map_err(|_| ModelError::Device)
}

fn gate_step(
    layer: &Gate,
    input: &Tensor,
    hidden: &Tensor,
    cell: &Tensor,
) -> Result<(Tensor, Tensor), ModelError> {
    let input_part = input
        .matmul(&layer.w_ih.t().map_err(|_| ModelError::Device)?)
        .and_then(|value| value.broadcast_add(&layer.b_ih))
        .map_err(|_| ModelError::Device)?;
    let hidden_part = hidden
        .matmul(&layer.w_hh.t().map_err(|_| ModelError::Device)?)
        .and_then(|value| value.broadcast_add(&layer.b_hh))
        .map_err(|_| ModelError::Device)?;
    let chunks = input_part
        .broadcast_add(&hidden_part)
        .and_then(|sum| sum.chunk(4, 1))
        .map_err(|_| ModelError::Device)?;
    if chunks.len() != 4 {
        return Err(ModelError::Shape);
    }
    let enter = candle_nn::ops::sigmoid(&chunks[0]).map_err(|_| ModelError::Device)?;
    let forget = candle_nn::ops::sigmoid(&chunks[1]).map_err(|_| ModelError::Device)?;
    let update = chunks[2].tanh().map_err(|_| ModelError::Device)?;
    let leave = candle_nn::ops::sigmoid(&chunks[3]).map_err(|_| ModelError::Device)?;
    let next_cell = forget
        .broadcast_mul(cell)
        .and_then(|kept| {
            enter
                .broadcast_mul(&update)
                .and_then(|added| kept.broadcast_add(&added))
        })
        .map_err(|_| ModelError::Device)?;
    let next_hidden = next_cell
        .tanh()
        .and_then(|activated| leave.broadcast_mul(&activated))
        .map_err(|_| ModelError::Device)?;
    Ok((next_hidden, next_cell))
}

fn load_blocks(
    weights: &candle_core::pickle::PthTensors,
    device: &Device,
) -> Result<Vec<Block>, ModelError> {
    let mut blocks = Vec::with_capacity(LAYERS);
    for index in 0..LAYERS {
        blocks.push(load_block(weights, index, device)?);
    }
    Ok(blocks)
}

fn load_subsampling(
    weights: &candle_core::pickle::PthTensors,
    device: &Device,
) -> Result<Subsampling, ModelError> {
    let wide = conv2d_config(1, 2, 1);
    let depth = conv2d_config(1, 2, SUB_CHANNELS);
    let point = conv2d_config(0, 1, 1);
    Ok(Subsampling {
        conv: conv2d(
            weights,
            "encoder.pre_encode.conv.0",
            SUB_CHANNELS,
            1,
            3,
            wide,
            device,
        )?,
        depthwise1: conv2d(
            weights,
            "encoder.pre_encode.conv.2",
            SUB_CHANNELS,
            1,
            3,
            depth,
            device,
        )?,
        pointwise1: conv2d(
            weights,
            "encoder.pre_encode.conv.3",
            SUB_CHANNELS,
            SUB_CHANNELS,
            1,
            point,
            device,
        )?,
        depthwise2: conv2d(
            weights,
            "encoder.pre_encode.conv.5",
            SUB_CHANNELS,
            1,
            3,
            depth,
            device,
        )?,
        pointwise2: conv2d(
            weights,
            "encoder.pre_encode.conv.6",
            SUB_CHANNELS,
            SUB_CHANNELS,
            1,
            point,
            device,
        )?,
        linear: linear(
            weights,
            "encoder.pre_encode.out",
            HIDDEN,
            SUB_CHANNELS * 16,
            true,
            device,
        )?,
    })
}

fn load_block(
    weights: &candle_core::pickle::PthTensors,
    index: usize,
    device: &Device,
) -> Result<Block, ModelError> {
    let prefix = format!("encoder.layers.{index}");
    Ok(Block {
        feed_forward1: load_feed_forward(weights, &format!("{prefix}.feed_forward1"), device)?,
        feed_forward2: load_feed_forward(weights, &format!("{prefix}.feed_forward2"), device)?,
        attention: load_attention(weights, &format!("{prefix}.self_attn"), device)?,
        convolution: load_convolution(weights, &format!("{prefix}.conv"), device)?,
        norm_feed_forward1: layer_norm(weights, &format!("{prefix}.norm_feed_forward1"), device)?,
        norm_attention: layer_norm(weights, &format!("{prefix}.norm_self_att"), device)?,
        norm_convolution: layer_norm(weights, &format!("{prefix}.norm_conv"), device)?,
        norm_feed_forward2: layer_norm(weights, &format!("{prefix}.norm_feed_forward2"), device)?,
        norm_out: layer_norm(weights, &format!("{prefix}.norm_out"), device)?,
    })
}

fn load_feed_forward(
    weights: &candle_core::pickle::PthTensors,
    prefix: &str,
    device: &Device,
) -> Result<FeedForward, ModelError> {
    Ok(FeedForward {
        up: linear(
            weights,
            &format!("{prefix}.linear1"),
            INTERMEDIATE,
            HIDDEN,
            true,
            device,
        )?,
        down: linear(
            weights,
            &format!("{prefix}.linear2"),
            HIDDEN,
            INTERMEDIATE,
            true,
            device,
        )?,
    })
}

fn load_attention(
    weights: &candle_core::pickle::PthTensors,
    prefix: &str,
    device: &Device,
) -> Result<Attention, ModelError> {
    Ok(Attention {
        query: linear(
            weights,
            &format!("{prefix}.linear_q"),
            HIDDEN,
            HIDDEN,
            true,
            device,
        )?,
        key: linear(
            weights,
            &format!("{prefix}.linear_k"),
            HIDDEN,
            HIDDEN,
            true,
            device,
        )?,
        value: linear(
            weights,
            &format!("{prefix}.linear_v"),
            HIDDEN,
            HIDDEN,
            true,
            device,
        )?,
        output: linear(
            weights,
            &format!("{prefix}.linear_out"),
            HIDDEN,
            HIDDEN,
            true,
            device,
        )?,
        relative: linear(
            weights,
            &format!("{prefix}.linear_pos"),
            HIDDEN,
            HIDDEN,
            false,
            device,
        )?,
        bias_u: tensor(weights, &format!("{prefix}.pos_bias_u"), &[8, 128], device)?,
        bias_v: tensor(weights, &format!("{prefix}.pos_bias_v"), &[8, 128], device)?,
    })
}

fn load_convolution(
    weights: &candle_core::pickle::PthTensors,
    prefix: &str,
    device: &Device,
) -> Result<Convolution, ModelError> {
    Ok(Convolution {
        expand: conv1d(
            weights,
            &format!("{prefix}.pointwise_conv1"),
            HIDDEN * 2,
            HIDDEN,
            1,
            conv1d_config(0, 1, 1),
            device,
        )?,
        depthwise: conv1d(
            weights,
            &format!("{prefix}.depthwise_conv"),
            HIDDEN,
            1,
            9,
            conv1d_config(4, 1, HIDDEN),
            device,
        )?,
        norm: batch_norm(weights, &format!("{prefix}.batch_norm"), device)?,
        project: conv1d(
            weights,
            &format!("{prefix}.pointwise_conv2"),
            HIDDEN,
            HIDDEN,
            1,
            conv1d_config(0, 1, 1),
            device,
        )?,
    })
}

fn load_gate(
    weights: &candle_core::pickle::PthTensors,
    layer: usize,
    device: &Device,
) -> Result<Gate, ModelError> {
    let prefix = "decoder.prediction.dec_rnn.lstm";
    let gates = PRED * 4;
    Ok(Gate {
        w_ih: tensor(
            weights,
            &format!("{prefix}.weight_ih_l{layer}"),
            &[gates, PRED],
            device,
        )?,
        w_hh: tensor(
            weights,
            &format!("{prefix}.weight_hh_l{layer}"),
            &[gates, PRED],
            device,
        )?,
        b_ih: tensor(
            weights,
            &format!("{prefix}.bias_ih_l{layer}"),
            &[gates],
            device,
        )?,
        b_hh: tensor(
            weights,
            &format!("{prefix}.bias_hh_l{layer}"),
            &[gates],
            device,
        )?,
    })
}

fn linear(
    weights: &candle_core::pickle::PthTensors,
    name: &str,
    rows: usize,
    columns: usize,
    bias: bool,
    device: &Device,
) -> Result<Linear, ModelError> {
    let weight = tensor(weights, &format!("{name}.weight"), &[rows, columns], device)?;
    let bias = if bias {
        Some(tensor(weights, &format!("{name}.bias"), &[rows], device)?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

fn layer_norm(
    weights: &candle_core::pickle::PthTensors,
    name: &str,
    device: &Device,
) -> Result<LayerNorm, ModelError> {
    let weight = tensor(weights, &format!("{name}.weight"), &[HIDDEN], device)?;
    let bias = tensor(weights, &format!("{name}.bias"), &[HIDDEN], device)?;
    Ok(LayerNorm::new(weight, bias, NORM_EPS))
}

fn batch_norm(
    weights: &candle_core::pickle::PthTensors,
    name: &str,
    device: &Device,
) -> Result<candle_nn::BatchNorm, ModelError> {
    let mean = tensor(weights, &format!("{name}.running_mean"), &[HIDDEN], device)?;
    let variance = tensor(weights, &format!("{name}.running_var"), &[HIDDEN], device)?;
    let weight = tensor(weights, &format!("{name}.weight"), &[HIDDEN], device)?;
    let bias = tensor(weights, &format!("{name}.bias"), &[HIDDEN], device)?;
    candle_nn::BatchNorm::new(HIDDEN, mean, variance, weight, bias, NORM_EPS)
        .map_err(|_| ModelError::Device)
}

fn conv1d(
    weights: &candle_core::pickle::PthTensors,
    name: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    config: Conv1dConfig,
    device: &Device,
) -> Result<Conv1d, ModelError> {
    let weight = tensor(
        weights,
        &format!("{name}.weight"),
        &[out_channels, in_channels, kernel],
        device,
    )?;
    let bias = tensor(weights, &format!("{name}.bias"), &[out_channels], device)?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn conv2d(
    weights: &candle_core::pickle::PthTensors,
    name: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    config: Conv2dConfig,
    device: &Device,
) -> Result<Conv2d, ModelError> {
    let weight = tensor(
        weights,
        &format!("{name}.weight"),
        &[out_channels, in_channels, kernel, kernel],
        device,
    )?;
    let bias = tensor(weights, &format!("{name}.bias"), &[out_channels], device)?;
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

fn tensor(
    weights: &candle_core::pickle::PthTensors,
    name: &str,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor, ModelError> {
    let tensor = weights
        .get(name)
        .map_err(|_| ModelError::Weights)?
        .ok_or(ModelError::Weights)?;
    if tensor.dtype() != DType::F32 || tensor.dims() != shape {
        return Err(ModelError::Shape);
    }
    tensor.to_device(device).map_err(|_| ModelError::Device)
}

fn vector(
    weights: &candle_core::pickle::PthTensors,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, ModelError> {
    tensor(weights, name, shape, &Device::Cpu)?
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|_| ModelError::Weights)
}

fn vocabulary(yaml: &str) -> Result<Vec<String>, ModelError> {
    let mut lines = yaml.lines();
    if lines.find(|line| *line == "  vocabulary:").is_none() {
        return Err(ModelError::Weights);
    }
    let mut vocab = Vec::with_capacity(VOCAB);
    for line in lines {
        let Some(item) = line.strip_prefix("  - ") else {
            break;
        };
        vocab.push(unquote(item));
    }
    if vocab.len() != VOCAB {
        return Err(ModelError::Shape);
    }
    Ok(vocab)
}

fn unquote(item: &str) -> String {
    item.strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
        .map(|inner| inner.replace("''", "'"))
        .unwrap_or_else(|| item.to_owned())
}

fn cached_checkpoint(nemo: &Path) -> Result<PathBuf, ModelError> {
    let dest = std::env::temp_dir().join("cutlass-parakeet-unified-en-0.6b.ckpt");
    let (mut source, size) = find_member(nemo, &["./model_weights.ckpt", "model_weights.ckpt"])?;
    if dest.is_file() && std::fs::metadata(&dest).is_ok_and(|meta| meta.len() == size) {
        return Ok(dest);
    }
    let partial = std::env::temp_dir().join("cutlass-parakeet-unified-en-0.6b.ckpt.partial");
    write_member(&mut source, size, &partial)?;
    std::fs::rename(&partial, &dest).map_err(|_| ModelError::Weights)?;
    Ok(dest)
}

fn read_text(nemo: &Path, names: &[&str]) -> Result<String, ModelError> {
    let (mut source, size) = find_member(nemo, names)?;
    if size > 2 * 1024 * 1024 {
        return Err(ModelError::Weights);
    }
    let mut bytes = vec![0; usize::try_from(size).map_err(|_| ModelError::Weights)?];
    source
        .read_exact(&mut bytes)
        .map_err(|_| ModelError::Weights)?;
    String::from_utf8(bytes).map_err(|_| ModelError::Weights)
}

fn write_member(source: &mut File, size: u64, dest: &Path) -> Result<(), ModelError> {
    let mut out = File::create(dest).map_err(|_| ModelError::Weights)?;
    let mut left = size;
    let mut buffer = vec![0; 1024 * 1024];
    while left > 0 {
        let take = buffer
            .len()
            .min(usize::try_from(left).map_err(|_| ModelError::Weights)?);
        source
            .read_exact(&mut buffer[..take])
            .map_err(|_| ModelError::Weights)?;
        out.write_all(&buffer[..take])
            .map_err(|_| ModelError::Weights)?;
        left -= u64::try_from(take).map_err(|_| ModelError::Weights)?;
    }
    Ok(())
}

fn find_member(archive: &Path, names: &[&str]) -> Result<(File, u64), ModelError> {
    let mut file = File::open(archive).map_err(|_| ModelError::Weights)?;
    let mut header = [0; 512];
    loop {
        file.read_exact(&mut header)
            .map_err(|_| ModelError::Weights)?;
        if header.iter().all(|byte| *byte == 0) {
            return Err(ModelError::Weights);
        }
        let name = header_name(&header);
        let size = header_size(&header).ok_or(ModelError::Weights)?;
        if names.iter().any(|wanted| name == *wanted) {
            return Ok((file, size));
        }
        let padded = size.div_ceil(512) * 512;
        file.seek(SeekFrom::Current(
            i64::try_from(padded).map_err(|_| ModelError::Weights)?,
        ))
        .map_err(|_| ModelError::Weights)?;
    }
}

fn header_name(header: &[u8]) -> String {
    let bytes = &header[..100];
    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(100);
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn header_size(header: &[u8]) -> Option<u64> {
    let field = &header[124..136];
    let end = field
        .iter()
        .position(|byte| *byte == 0 || *byte == b' ')
        .unwrap_or(field.len());
    let text = std::str::from_utf8(&field[..end]).ok()?.trim();
    if text.is_empty() {
        Some(0)
    } else {
        u64::from_str_radix(text, 8).ok()
    }
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
        let error = ParakeetUnified::cpu(Path::new("no-such-unified"));
        assert!(matches!(error, Err(ModelError::Weights)));
    }

    #[test]
    fn quoted_pieces_keep_their_mark() {
        assert_eq!(unquote("','"), ",");
        assert_eq!(unquote("''''"), "'");
        assert_eq!(unquote("."), ".");
        assert_eq!(unquote("▁go"), "▁go");
    }

    #[test]
    fn a_lone_boundary_starts_the_next_word() {
        let vocab = vec!["▁go".to_owned(), "▁".to_owned(), "3".to_owned()];
        let spans = [
            Span {
                token: 0,
                start: 0,
                end: 0,
            },
            Span {
                token: 1,
                start: 1,
                end: 1,
            },
            Span {
                token: 2,
                start: 1,
                end: 1,
            },
        ];
        let split = parakeet::pieces_to_words(&spans, &vocab, 80, true).unwrap();
        assert_eq!(
            split
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            ["go", "3"]
        );
        let glued = parakeet::pieces_to_words(&spans, &vocab, 80, false).unwrap();
        assert_eq!(glued.len(), 1);
        assert_eq!(glued[0].text, "go3");
    }

    #[test]
    fn the_gemini_opening_keeps_punctuation() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let model = root.join("parakeet-unified-en-0.6b");
        let wav = root.join("GOOGLE IS BACK! (Gemini 3.8 Flash).wav");
        if !model.join(ARCHIVE).is_file() || !wav.is_file() {
            return;
        }
        let recognizer = ParakeetUnified::cpu(&model).unwrap();
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
            [
                "Google", "is", "back.", "They", "just", "dropped", "Gemini."
            ]
        );
        assert_eq!(
            text.words.start_time,
            [640, 1360, 1920, 2720, 2880, 3200, 3680]
        );
        assert_eq!(
            text.words.end_time,
            [800, 1440, 2560, 2880, 2960, 3280, 4080]
        );
        assert_eq!(text.current_words.text, text.words.text);
        assert_eq!(text.current_words.start_time, text.words.start_time);
        assert_eq!(text.current_words.end_time, text.words.end_time);
        assert_eq!(text.content, "Google is back. They just dropped Gemini.");
        assert_eq!(text.base_content, text.content);
    }
}
