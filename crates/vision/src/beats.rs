//! Auto beats: Beat This!'s small model on burn, which hears where the
//! beats of a piece of music fall.
//!
//! The sound is read as a spectrogram: [`FRAMES_A_SECOND`] frames a
//! second, each 128 bands from 30 Hz to 11 kHz on the mel scale. A front
//! end of three blocks halves the bands each time, attending across the
//! bands and then across time. Six transformer layers follow, and a head
//! answers, for every frame, how sure it is that a beat falls there and
//! that the first beat of a bar does. About 2 million weights, in full
//! precision. Code and weights are MIT.
//!
//! The model reads 30 seconds at once. A longer sound is read in stretches
//! that overlap by a few frames, whose edges are dropped, because the
//! model was not taught at the edges of what it reads.
//!
//! A beat is a frame that is the surest within 70 ms each way and more
//! likely than not. A downbeat is found the same way, then moved onto the
//! beat nearest it.
//!
//! The model opens no file: it takes mono samples at [`BEATS_RATE`].

use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::{gelu, sigmoid, softmax};
use burn::tensor::{Device, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use crate::VisionError;
use crate::device::{self, Compute};
use crate::mobile_sam::load_weights;

/// The weights, in the folder passed to [`BeatModel::load`].
pub const BEATS_WEIGHTS: &str = "beat_this_small.bpk";

/// The sample rate the model hears at, in Hz. The sound is mono.
pub const BEATS_RATE: u32 = 22_050;

/// The answers the model gives for a second of sound.
pub const FRAMES_A_SECOND: usize = 50;

/// The shortest sound the model reads, in samples: one window.
pub const BEATS_MIN_SAMPLES: usize = WINDOW;

/// The longest sound the model reads, in samples: two hours.
pub const BEATS_MAX_SAMPLES: usize = BEATS_RATE as usize * 60 * 60 * 2;

/// The samples one spectrogram frame is worked out from.
const WINDOW: usize = 1024;
/// The samples from one frame to the next.
const HOP: usize = 441;
/// The bands of a frame, and the frequencies they span.
const BANDS: usize = 128;
const LOWEST_HZ: f64 = 30.0;
const HIGHEST_HZ: f64 = 11_000.0;
/// What a band's level is scaled by before its logarithm is taken.
const LOUDNESS: f32 = 1000.0;

/// The frames the model reads at once: 30 seconds.
const STRETCH: usize = 1500;
/// The frames dropped at each edge of what it read.
const BORDER: usize = 6;
/// What a frame no stretch answered for holds: surely no beat.
const SILENT: f32 = -1000.0;

/// The channels the stem makes, doubled by each front end block.
const STEM: usize = 32;
/// The front end's blocks.
const BLOCKS: usize = 3;
/// The transformer's width and layers.
const WIDTH: usize = 128;
const LAYERS: usize = 6;
/// The width of one attention head, everywhere in the model.
const HEAD: usize = 32;
/// The most attention scores worked out at once: 64 MB of them.
const MAX_SCORES: usize = 1 << 24;
/// The frames each way a beat must be the surest of.
const REACH: usize = 3;

/// Scales each place's values to the same length, then by a learned gain.
#[derive(Module, Debug)]
struct Rms {
    gamma: Param<Tensor<1>>,
}

impl Rms {
    fn new(dim: usize, device: &Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([dim], device)),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, _, dim] = x.dims();
        let length = x.clone().powi_scalar(2).sum_dim(2).sqrt().clamp_min(1e-12);
        x / length * (dim as f64).sqrt() * self.gamma.val().reshape([1, 1, dim])
    }
}

/// The rotary turn of every place in a row: how attention knows where a
/// place is. Each pair of a head's values is turned by an angle that grows
/// with the place, a different rate for each pair.
struct Turns {
    cos: Tensor<4>,
    sin: Tensor<4>,
}

impl Turns {
    fn new(places: usize, device: &Device) -> Self {
        let mut cos = Vec::with_capacity(places * HEAD);
        let mut sin = Vec::with_capacity(places * HEAD);
        // In single precision throughout, as the model was taught.
        let rates: Vec<f32> = (0..HEAD / 2)
            .map(|pair| 1.0 / 10_000_f32.powf((2 * pair) as f32 / HEAD as f32))
            .collect();
        for place in 0..places {
            for rate in &rates {
                let angle = place as f32 * rate;
                cos.extend([angle.cos(); 2]);
                sin.extend([angle.sin(); 2]);
            }
        }
        let shape = [1, 1, places, HEAD];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
        }
    }

    /// `x` (rows, heads, places, [`HEAD`]) with every place turned.
    fn turn(&self, x: Tensor<4>) -> Tensor<4> {
        let [rows, heads, places, _] = x.dims();
        let pairs = x.clone().reshape([rows, heads, places, HEAD / 2, 2]);
        let first = pairs.clone().narrow(4, 0, 1);
        let second = pairs.narrow(4, 1, 1);
        let crossed =
            Tensor::cat(vec![second.neg(), first], 4).reshape([rows, heads, places, HEAD]);
        x * self.cos.clone() + crossed * self.sin.clone()
    }
}

/// Self-attention along a row, each head's answer scaled by a gate.
#[derive(Module, Debug)]
struct Attention {
    norm: Rms,
    to_qkv: Linear,
    to_gates: Linear,
    to_out: Linear,
}

impl Attention {
    fn new(dim: usize, device: &Device) -> Self {
        Self {
            norm: Rms::new(dim, device),
            to_qkv: LinearConfig::new(dim, 3 * dim)
                .with_bias(false)
                .init(device),
            to_gates: LinearConfig::new(dim, dim / HEAD).init(device),
            to_out: LinearConfig::new(dim, dim).with_bias(false).init(device),
        }
    }

    /// What each place of `x` (rows, places, width) takes from the others
    /// in its row.
    fn forward(&self, x: Tensor<3>, turns: &Turns) -> Tensor<3> {
        let [rows, places, dim] = x.dims();
        let heads = dim / HEAD;
        let x = self.norm.forward(x);
        let qkv = self
            .to_qkv
            .forward(x.clone())
            .reshape([rows, places, 3, heads, HEAD]);
        let part = |index: usize| -> Tensor<4> {
            qkv.clone()
                .slice([0..rows, 0..places, index..index + 1])
                .reshape([rows, places, heads, HEAD])
                .swap_dims(1, 2)
        };
        let q = turns.turn(part(0));
        let k = turns.turn(part(1));
        let v = part(2);
        let scores = q.matmul(k.swap_dims(2, 3)) / (HEAD as f64).sqrt();
        let mixed = softmax(scores, 3).matmul(v);
        let gates = sigmoid(self.to_gates.forward(x))
            .swap_dims(1, 2)
            .reshape([rows, heads, places, 1]);
        let out = (mixed * gates).swap_dims(1, 2).reshape([rows, places, dim]);
        self.to_out.forward(out)
    }

    /// [`Self::forward`], a few rows at a time when all of them at once
    /// would hold more than [`MAX_SCORES`] scores: half a minute of frames
    /// attending to each other, for every band, is hundreds of megabytes.
    fn attend(&self, x: Tensor<3>, turns: &Turns) -> Tensor<3> {
        let [rows, places, dim] = x.dims();
        let a_row = (dim / HEAD) * places * places;
        let group = (MAX_SCORES / a_row.max(1)).max(1);
        if rows <= group {
            return self.forward(x, turns);
        }
        let mut parts = Vec::with_capacity(rows.div_ceil(group));
        let mut from = 0;
        while from < rows {
            let to = (from + group).min(rows);
            parts.push(self.forward(x.clone().narrow(0, from, to - from), turns));
            from = to;
        }
        Tensor::cat(parts, 0)
    }
}

/// A beat logit and a downbeat logit for every frame.
type Logits = (Vec<f32>, Vec<f32>);

#[derive(Module, Debug)]
struct FeedForward {
    norm: Rms,
    inner: Linear,
    out: Linear,
}

impl FeedForward {
    fn new(dim: usize, device: &Device) -> Self {
        Self {
            norm: Rms::new(dim, device),
            inner: LinearConfig::new(dim, 4 * dim).init(device),
            out: LinearConfig::new(4 * dim, dim).init(device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.out
            .forward(gelu(self.inner.forward(self.norm.forward(x))))
    }
}

/// Attention across the bands of each frame, then across the frames of
/// each band.
#[derive(Module, Debug)]
struct Partial {
    attn_f: Attention,
    ff_f: FeedForward,
    attn_t: Attention,
    ff_t: FeedForward,
}

impl Partial {
    fn new(dim: usize, device: &Device) -> Self {
        Self {
            attn_f: Attention::new(dim, device),
            ff_f: FeedForward::new(dim, device),
            attn_t: Attention::new(dim, device),
            ff_t: FeedForward::new(dim, device),
        }
    }

    /// `x` is (1, channels, bands, frames), and so is the answer.
    fn forward(&self, x: Tensor<4>, bands: &Turns, frames: &Turns) -> Tensor<4> {
        let [batch, dim, f, t] = x.dims();
        let x = x.permute([0, 3, 2, 1]).reshape([batch * t, f, dim]);
        let x = x.clone() + self.attn_f.attend(x, bands);
        let x = x.clone() + self.ff_f.forward(x);
        let x = x
            .reshape([batch, t, f, dim])
            .permute([0, 2, 1, 3])
            .reshape([batch * f, t, dim]);
        let x = x.clone() + self.attn_t.attend(x, frames);
        let x = x.clone() + self.ff_t.forward(x);
        x.reshape([batch, f, t, dim]).permute([0, 3, 1, 2])
    }
}

/// A convolution over bands and frames that keeps the frames.
fn conv(input: usize, output: usize, bands: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([input, output], [bands, 3])
        .with_stride([bands, 1])
        .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
        .with_bias(false)
        .init(device)
}

/// One front end block: attention, then half the bands at twice the
/// channels.
#[derive(Module, Debug)]
struct Block {
    partial: Partial,
    conv2d: Conv2d,
    norm: BatchNorm,
}

impl Block {
    fn new(dim: usize, device: &Device) -> Self {
        Self {
            partial: Partial::new(dim, device),
            conv2d: conv(dim, 2 * dim, 2, device),
            norm: BatchNormConfig::new(2 * dim).init(device),
        }
    }

    fn forward(&self, x: Tensor<4>, bands: &Turns, frames: &Turns) -> Tensor<4> {
        let x = self.partial.forward(x, bands, frames);
        gelu(self.norm.forward(self.conv2d.forward(x)))
    }
}

/// Evens out each band's level, then joins every four bands.
#[derive(Module, Debug)]
struct Stem {
    bn1d: BatchNorm,
    conv2d: Conv2d,
    bn2d: BatchNorm,
}

#[derive(Module, Debug)]
struct Frontend {
    stem: Stem,
    blocks: Vec<Block>,
    linear: Linear,
}

#[derive(Module, Debug)]
struct Layer {
    attn: Attention,
    ff: FeedForward,
}

#[derive(Module, Debug)]
struct Transformer {
    layers: Vec<Layer>,
    norm: Rms,
}

#[derive(Module, Debug)]
struct Head {
    linear: Linear,
}

#[derive(Module, Debug)]
struct BeatThis {
    frontend: Frontend,
    transformer: Transformer,
    head: Head,
}

impl BeatThis {
    fn new(device: &Device) -> Self {
        // The bands left after the stem and the blocks: 128 / 4 / 2 / 2 / 2.
        let bands = (BANDS / 4) >> BLOCKS;
        let channels = STEM << BLOCKS;
        Self {
            frontend: Frontend {
                stem: Stem {
                    bn1d: BatchNormConfig::new(BANDS).init(device),
                    conv2d: conv(1, STEM, 4, device),
                    bn2d: BatchNormConfig::new(STEM).init(device),
                },
                blocks: (0..BLOCKS)
                    .map(|block| Block::new(STEM << block, device))
                    .collect(),
                linear: LinearConfig::new(channels * bands, WIDTH).init(device),
            },
            transformer: Transformer {
                layers: (0..LAYERS)
                    .map(|_| Layer {
                        attn: Attention::new(WIDTH, device),
                        ff: FeedForward::new(WIDTH, device),
                    })
                    .collect(),
                norm: Rms::new(WIDTH, device),
            },
            head: Head {
                linear: LinearConfig::new(WIDTH, 2).init(device),
            },
        }
    }

    /// `spect` is (1, frames, [`BANDS`]). The answer is (1, frames, 2):
    /// how sure of a beat that is not a downbeat, and of a downbeat.
    fn forward(&self, spect: Tensor<3>) -> Tensor<3> {
        let device = spect.device();
        let [batch, frames, bands] = spect.dims();
        let over_frames = Turns::new(frames, &device);
        let stem = &self.frontend.stem;
        let x = stem
            .bn1d
            .forward(spect.swap_dims(1, 2))
            .reshape([batch, 1, bands, frames]);
        let mut x = gelu(stem.bn2d.forward(stem.conv2d.forward(x)));
        for block in &self.frontend.blocks {
            let [_, _, bands, _] = x.dims();
            x = block.forward(x, &Turns::new(bands, &device), &over_frames);
        }
        let [_, channels, bands, _] = x.dims();
        let x = x
            .permute([0, 3, 1, 2])
            .reshape([batch, frames, channels * bands]);
        let mut x = self.frontend.linear.forward(x);
        for layer in &self.transformer.layers {
            x = layer.attn.attend(x.clone(), &over_frames) + x;
            x = layer.ff.forward(x.clone()) + x;
        }
        self.head.linear.forward(self.transformer.norm.forward(x))
    }
}

/// Where the beats of a sound fall, in microseconds from its start, in
/// order. Every downbeat, the first beat of a bar, is one of the beats.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Beats {
    pub beats: Vec<i64>,
    pub downbeats: Vec<i64>,
}

/// The beat model, loaded on one device.
pub struct BeatModel {
    device: Device,
    model: BeatThis,
}

impl BeatModel {
    /// Loads [`BEATS_WEIGHTS`] from the folder `weights`.
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let mut model = BeatThis::new(&device);
        load_weights(&mut model, &weights.join(BEATS_WEIGHTS))?;
        Ok(Self { device, model })
    }

    /// The model with random weights: the same work as the real one, for
    /// timing and for tests that only check shapes.
    #[cfg(test)]
    pub(crate) fn untrained(compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let model = BeatThis::new(&device);
        Ok(Self { device, model })
    }

    /// The beats of `samples`: mono, at [`BEATS_RATE`], from
    /// [`BEATS_MIN_SAMPLES`] to [`BEATS_MAX_SAMPLES`] of them.
    ///
    /// `keep_going` hears how far along the model is, from 0 to 1, before
    /// each stretch it reads. When it answers `false` the model stops, and
    /// the answer is `None`.
    pub fn find(
        &self,
        samples: &[f32],
        keep_going: &mut dyn FnMut(f32) -> bool,
    ) -> Result<Option<Beats>, VisionError> {
        let spect = spectrogram(samples)?;
        let Some((beat, downbeat)) = self.answers(&spect, keep_going)? else {
            return Ok(None);
        };
        Ok(Some(beats_of(&beat, &downbeat)))
    }

    /// For every frame of `spect`, how sure the model is of a beat and of
    /// a downbeat, as logits: above zero is more likely than not.
    fn answers(
        &self,
        spect: &[f32],
        keep_going: &mut dyn FnMut(f32) -> bool,
    ) -> Result<Option<Logits>, VisionError> {
        let frames = spect.len() / BANDS;
        let stretches = stretches(frames);
        let mut read = Vec::with_capacity(stretches.len());
        for (index, stretch) in stretches.iter().enumerate() {
            if !keep_going(index as f32 / stretches.len() as f32) {
                return Ok(None);
            }
            let length = stretch.frames();
            let mut padded = vec![0.0_f32; length * BANDS];
            padded[stretch.left * BANDS..(stretch.left + stretch.to - stretch.from) * BANDS]
                .copy_from_slice(&spect[stretch.from * BANDS..stretch.to * BANDS]);
            let input =
                Tensor::<3>::from_data(TensorData::new(padded, [1, length, BANDS]), &self.device);
            let answer = crate::read_back(self.model.forward(input))?
                .convert::<f32>()
                .try_into_vec::<f32>()
                .map_err(|error| VisionError::Model {
                    message: format!("{error:?}"),
                })?;
            if answer.len() != length * 2 {
                return Err(VisionError::Model {
                    message: format!("{} answers for a stretch of {length} frames", answer.len()),
                });
            }
            read.push(answer);
        }
        let mut beat = vec![SILENT; frames];
        let mut downbeat = vec![SILENT; frames];
        // Last first, so where two stretches overlap the earlier one's
        // answers stay.
        for (stretch, answer) in stretches.iter().zip(&read).rev() {
            let pairs = answer.as_chunks::<2>().0;
            let kept = &pairs[BORDER..pairs.len() - BORDER];
            for (offset, [other, down]) in kept.iter().enumerate() {
                let Some(frame) = stretch.first_answer().checked_add(offset) else {
                    break;
                };
                if frame >= frames {
                    break;
                }
                // The head's first answer is every beat that is not a
                // downbeat, so a beat is either.
                beat[frame] = other + down;
                downbeat[frame] = *down;
            }
        }
        Ok(Some((beat, downbeat)))
    }
}

/// One stretch of a spectrogram the model reads at once: frames `from` to
/// `to`, with `left` and `right` frames of silence added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stretch {
    from: usize,
    to: usize,
    left: usize,
    right: usize,
}

impl Stretch {
    fn frames(&self) -> usize {
        self.left + (self.to - self.from) + self.right
    }

    /// The frame of the sound the first kept answer is for.
    fn first_answer(&self) -> usize {
        self.from + BORDER - self.left
    }
}

/// How a spectrogram of `frames` frames is read: in stretches of
/// [`STRETCH`] frames, each [`BORDER`] frames into the one before, so
/// every frame has an answer away from an edge. The first and the last
/// have silence where the sound has no frames. The last stretch is moved
/// back to end with the sound instead of being short.
fn stretches(frames: usize) -> Vec<Stretch> {
    let step = STRETCH - 2 * BORDER;
    // Where each stretch starts, counted from `BORDER` frames before the
    // sound.
    let mut starts: Vec<usize> = (0..frames).step_by(step).collect();
    if frames > step
        && let Some(last) = starts.last_mut()
    {
        *last = frames + 2 * BORDER - STRETCH;
    }
    starts
        .into_iter()
        .map(|start| {
            let end = start + STRETCH;
            let from = start.saturating_sub(BORDER);
            let to = (end - BORDER).min(frames);
            Stretch {
                from,
                to,
                left: BORDER.saturating_sub(start),
                right: (end - BORDER).saturating_sub(frames).min(BORDER),
            }
        })
        .collect()
}

/// The frames where `logits` peak: the highest within [`REACH`] frames
/// each way and above zero. Peaks on neighbouring frames are one peak,
/// between them.
fn peaks(logits: &[f32]) -> Vec<f64> {
    let mut found = Vec::new();
    // The peak being gathered, and how many frames are in it.
    let mut gathered: Option<(f64, f64)> = None;
    for (frame, logit) in logits.iter().enumerate() {
        let from = frame.saturating_sub(REACH);
        let to = (frame + REACH + 1).min(logits.len());
        let highest = logits[from..to].iter().copied().fold(f32::MIN, f32::max);
        if *logit <= 0.0 || *logit != highest {
            continue;
        }
        let frame = frame as f64;
        gathered = Some(match gathered {
            Some((at, count)) if frame - at <= 1.0 => {
                let count = count + 1.0;
                (at + (frame - at) / count, count)
            }
            Some((at, _)) => {
                found.push(at);
                (frame, 1.0)
            }
            None => (frame, 1.0),
        });
    }
    if let Some((at, _)) = gathered {
        found.push(at);
    }
    found
}

/// The beats and downbeats in the model's answers, each downbeat moved
/// onto the beat nearest it.
fn beats_of(beat: &[f32], downbeat: &[f32]) -> Beats {
    let micros = |frame: f64| (frame * (1_000_000.0 / FRAMES_A_SECOND as f64)).round() as i64;
    let beats: Vec<i64> = peaks(beat).into_iter().map(micros).collect();
    let mut downbeats: Vec<i64> = peaks(downbeat)
        .into_iter()
        .map(micros)
        .map(|down| {
            beats
                .iter()
                .copied()
                .min_by_key(|beat| (beat - down).abs())
                .unwrap_or(down)
        })
        .collect();
    downbeats.sort_unstable();
    downbeats.dedup();
    Beats { beats, downbeats }
}

/// Slaney's mel scale: even steps up to 1 kHz, then steps that grow with
/// the frequency.
fn mel_of(hz: f64) -> f64 {
    const STEP: f64 = 200.0 / 3.0;
    if hz < 1000.0 {
        hz / STEP
    } else {
        1000.0 / STEP + (hz / 1000.0).ln() / (6.4_f64.ln() / 27.0)
    }
}

fn hz_of(mel: f64) -> f64 {
    const STEP: f64 = 200.0 / 3.0;
    let knee = 1000.0 / STEP;
    if mel < knee {
        mel * STEP
    } else {
        1000.0 * ((mel - knee) * (6.4_f64.ln() / 27.0)).exp()
    }
}

/// One band of the spectrogram: the first frequency bin it hears, and how
/// much of that bin and each one after it.
struct Band {
    first: usize,
    weights: Vec<f32>,
}

/// The [`BANDS`] bands: triangles evenly spaced on the mel scale from
/// [`LOWEST_HZ`] to [`HIGHEST_HZ`], each rising from the band before's
/// centre and falling to the next one's.
fn bands() -> Vec<Band> {
    let bins = WINDOW / 2 + 1;
    let bin_hz = |bin: usize| f64::from(BEATS_RATE / 2) * bin as f64 / (bins - 1) as f64;
    let (low, high) = (mel_of(LOWEST_HZ), mel_of(HIGHEST_HZ));
    let edge = |index: usize| hz_of(low + (high - low) * index as f64 / (BANDS + 1) as f64);
    (0..BANDS)
        .map(|band| {
            let (left, centre, right) = (edge(band), edge(band + 1), edge(band + 2));
            let weight = |bin: usize| {
                let hz = bin_hz(bin);
                ((hz - left) / (centre - left))
                    .min((right - hz) / (right - centre))
                    .max(0.0)
            };
            let first = (0..bins).find(|bin| weight(*bin) > 0.0).unwrap_or(bins);
            let weights = (first..bins)
                .map(weight)
                .take_while(|weight| *weight > 0.0)
                .map(|weight| weight as f32)
                .collect();
            Band { first, weights }
        })
        .collect()
}

/// The spectrogram of `samples` as the model reads it: a frame every
/// [`HOP`] samples, the first centred on the first sample, each
/// [`BANDS`] values, frame after frame. A frame is the level of each band
/// in a [`WINDOW`] of samples around its centre, `ln(1 + 1000 * level)`.
/// Past either end the sound is taken to run back the way it came.
pub(crate) fn spectrogram(samples: &[f32]) -> Result<Vec<f32>, VisionError> {
    let count = samples.len();
    if !(BEATS_MIN_SAMPLES..=BEATS_MAX_SAMPLES).contains(&count)
        || samples.iter().any(|sample| !sample.is_finite())
    {
        return Err(VisionError::Sound { samples: count });
    }
    let fft = FftPlanner::<f32>::new().plan_fft_forward(WINDOW);
    let window: Vec<f32> = (0..WINDOW)
        .map(|index| {
            let turn = std::f64::consts::TAU * index as f64 / WINDOW as f64;
            (0.5 - 0.5 * turn.cos()) as f32
        })
        .collect();
    let bands = bands();
    let scale = 1.0 / (WINDOW as f32).sqrt();
    let frames = count / HOP + 1;
    let mut spect = Vec::with_capacity(frames * BANDS);
    let mut buffer = vec![Complex::new(0.0_f32, 0.0); WINDOW];
    let mut scratch = vec![Complex::new(0.0_f32, 0.0); fft.get_inplace_scratch_len()];
    let mut levels = vec![0.0_f32; WINDOW / 2 + 1];
    for frame in 0..frames {
        let centre = frame * HOP;
        for (index, (slot, weight)) in buffer.iter_mut().zip(&window).enumerate() {
            // The sample `index - WINDOW / 2` from the centre, mirrored at
            // the ends of the sound.
            let at = centre + index;
            let at = if at < WINDOW / 2 {
                WINDOW / 2 - at
            } else if at - WINDOW / 2 >= count {
                2 * (count - 1) - (at - WINDOW / 2)
            } else {
                at - WINDOW / 2
            };
            *slot = Complex::new(samples[at] * weight, 0.0);
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        for (level, bin) in levels.iter_mut().zip(&buffer) {
            *level = bin.norm() * scale;
        }
        for band in &bands {
            let level: f32 = band
                .weights
                .iter()
                .zip(&levels[band.first..])
                .map(|(weight, level)| weight * level)
                .sum();
            spect.push((LOUDNESS * level).ln_1p());
        }
    }
    Ok(spect)
}

/// Reads Beat This!'s small weights as
/// `tools/vision-models/scripts/beats_weights.py` saves them, and writes
/// them to `to` as `.bpk` for [`BeatModel::load`]. Run once, offline.
pub fn convert_beat_this(from: &Path, to: &Path) -> Result<(), VisionError> {
    let refused = |path: &Path, message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !from.is_file() {
        return Err(refused(from, "missing".into()));
    }
    let device = Device::flex();
    let mut model = BeatThis::new(&device);
    let mut store = SafetensorsStore::from_file(from).with_from_adapter(PyTorchToBurnAdapter);
    let applied = model
        .load_from(&mut store)
        .map_err(|error| refused(from, error.to_string()))?;
    if !applied.errors.is_empty() || !applied.missing.is_empty() || !applied.unused.is_empty() {
        return Err(refused(
            from,
            format!(
                "{:?} failed, {:?} are missing and {:?} are not the model's",
                applied.errors, applied.missing, applied.unused
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
