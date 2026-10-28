use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use safetensors::tensor::SafeTensors;

use crate::features::{self, HOP_MS, N_MELS};
use crate::prepare::Pcm;
use crate::search;
use crate::words::Utterance;
use crate::{RecognizeError, Recognizer};

/// Weights for the two-layer acoustic head.
///
/// `hidden.weight` is `[hidden, n_mels]`. `output.weight` is `[vocab, hidden]`.
/// Vocabulary index 0 is the CTC blank.
#[derive(Clone, Debug, PartialEq)]
pub struct AcousticWeights {
    pub n_mels: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub hidden_weight: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub output_weight: Vec<f32>,
    pub output_bias: Vec<f32>,
}

impl AcousticWeights {
    /// Read the four tensors. Names are `hidden.weight`, `hidden.bias`,
    /// `output.weight`, and `output.bias`, all little-endian `f32`.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, ModelError> {
        let tensors = SafeTensors::deserialize(bytes).map_err(|_| ModelError::Weights)?;
        let hidden_weight = f32_tensor(&tensors, "hidden.weight")?;
        let hidden_bias = f32_tensor(&tensors, "hidden.bias")?;
        let output_weight = f32_tensor(&tensors, "output.weight")?;
        let output_bias = f32_tensor(&tensors, "output.bias")?;
        if hidden_weight.shape.len() != 2 || output_weight.shape.len() != 2 {
            return Err(ModelError::Shape);
        }
        let hidden = hidden_weight.shape[0];
        let n_mels = hidden_weight.shape[1];
        let vocab = output_weight.shape[0];
        if output_weight.shape[1] != hidden
            || hidden_bias.shape != [hidden]
            || output_bias.shape != [vocab]
        {
            return Err(ModelError::Shape);
        }
        Self::new(
            n_mels,
            hidden,
            vocab,
            hidden_weight.values,
            hidden_bias.values,
            output_weight.values,
            output_bias.values,
        )
    }

    pub fn new(
        n_mels: usize,
        hidden: usize,
        vocab: usize,
        hidden_weight: Vec<f32>,
        hidden_bias: Vec<f32>,
        output_weight: Vec<f32>,
        output_bias: Vec<f32>,
    ) -> Result<Self, ModelError> {
        let weights = Self {
            n_mels,
            hidden,
            vocab,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        };
        weights.check()?;
        Ok(weights)
    }

    fn check(&self) -> Result<(), ModelError> {
        if self.n_mels == 0 || self.hidden == 0 || self.vocab == 0 {
            return Err(ModelError::Shape);
        }
        if self.hidden_weight.len() != self.hidden * self.n_mels
            || self.hidden_bias.len() != self.hidden
            || self.output_weight.len() != self.vocab * self.hidden
            || self.output_bias.len() != self.vocab
        {
            return Err(ModelError::Shape);
        }
        let finite = self
            .hidden_weight
            .iter()
            .chain(&self.hidden_bias)
            .chain(&self.output_weight)
            .chain(&self.output_bias)
            .all(|value| value.is_finite());
        if !finite {
            return Err(ModelError::Weights);
        }
        Ok(())
    }
}

struct LoadedTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn f32_tensor(tensors: &SafeTensors<'_>, name: &str) -> Result<LoadedTensor, ModelError> {
    let tensor = tensors.tensor(name).map_err(|_| ModelError::Weights)?;
    if tensor.dtype() != safetensors::Dtype::F32 {
        return Err(ModelError::Weights);
    }
    if !tensor.data().len().is_multiple_of(4) {
        return Err(ModelError::Weights);
    }
    let mut values = Vec::with_capacity(tensor.data().len() / 4);
    for chunk in tensor.data().chunks_exact(4) {
        let Some(bytes) = chunk.try_into().ok() else {
            return Err(ModelError::Weights);
        };
        values.push(f32::from_le_bytes(bytes));
    }
    Ok(LoadedTensor {
        shape: tensor.shape().to_vec(),
        values,
    })
}

/// The acoustic head. Tensors live on `device`.
///
/// [`LocalRecognizer::cpu`] uses [`Device::Cpu`]. [`LocalRecognizer::on_device`]
/// takes the same head on whatever device the caller built, so a later Apple
/// build can pass a Metal device without a second model.
pub struct AcousticModel {
    device: Device,
    hidden: Linear,
    output: Linear,
    n_mels: usize,
    vocab: usize,
}

impl AcousticModel {
    pub fn on_device(device: &Device, weights: &AcousticWeights) -> Result<Self, ModelError> {
        weights.check()?;
        let hidden_weight = Tensor::from_vec(
            weights.hidden_weight.clone(),
            (weights.hidden, weights.n_mels),
            device,
        )
        .map_err(|_| ModelError::Device)?;
        let hidden_bias = Tensor::from_vec(weights.hidden_bias.clone(), weights.hidden, device)
            .map_err(|_| ModelError::Device)?;
        let output_weight = Tensor::from_vec(
            weights.output_weight.clone(),
            (weights.vocab, weights.hidden),
            device,
        )
        .map_err(|_| ModelError::Device)?;
        let output_bias = Tensor::from_vec(weights.output_bias.clone(), weights.vocab, device)
            .map_err(|_| ModelError::Device)?;
        Ok(Self {
            device: device.clone(),
            hidden: Linear::new(hidden_weight, Some(hidden_bias)),
            output: Linear::new(output_weight, Some(output_bias)),
            n_mels: weights.n_mels,
            vocab: weights.vocab,
        })
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Frame logits. `mel` is `[frames, n_mels]` on this model's device.
    pub fn logits(&self, mel: &Tensor) -> Result<Tensor, ModelError> {
        let dims = mel.dims();
        if dims.len() != 2 || dims[1] != self.n_mels {
            return Err(ModelError::Shape);
        }
        let hidden = self
            .hidden
            .forward(mel)
            .and_then(|hidden| hidden.relu())
            .map_err(|_| ModelError::Device)?;
        self.output.forward(&hidden).map_err(|_| ModelError::Device)
    }
}

/// The local recognizer. Speech stays on this path.
pub struct LocalRecognizer {
    model: AcousticModel,
    vocab: Vec<String>,
}

impl LocalRecognizer {
    pub fn cpu(weights: AcousticWeights, vocab: Vec<String>) -> Result<Self, ModelError> {
        Self::on_device(&Device::Cpu, weights, vocab)
    }

    pub fn on_device(
        device: &Device,
        weights: AcousticWeights,
        vocab: Vec<String>,
    ) -> Result<Self, ModelError> {
        if vocab.len() != weights.vocab || weights.n_mels != N_MELS {
            return Err(ModelError::Shape);
        }
        let model = AcousticModel::on_device(device, &weights)?;
        Ok(Self { model, vocab })
    }
}

impl Recognizer for LocalRecognizer {
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
        let frames = features::frame_count(pcm.samples.len());
        if frames == 0 {
            return Ok(Vec::new());
        }
        let mel = features::log_mel(&pcm.samples).map_err(|_| RecognizeError::Model)?;
        let mel = Tensor::from_vec(mel, (frames, N_MELS), self.model.device())
            .map_err(|_| RecognizeError::Model)?;
        let logits = self.model.logits(&mel).map_err(|_| RecognizeError::Model)?;
        let rows = logits.to_vec2::<f32>().map_err(|_| RecognizeError::Model)?;
        let words = search::ctc(&rows, &self.vocab, HOP_MS).map_err(|_| RecognizeError::Search)?;
        if words.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![Utterance { words }])
        }
    }
}

/// Why the acoustic model did not load or run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// A tensor shape does not match the head.
    Shape,
    /// The weight bytes are not the four `f32` tensors.
    Weights,
    /// A tensor could not be built on the device.
    Device,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use candle_core::Device;
    use cutlass_model::{Draft, MaterialText, TimeRange};
    use cutlass_session::Session;
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    use super::*;
    use crate::file::FileSource;
    use crate::job::Job;
    use crate::prepare::Pcm;

    fn tiny() -> AcousticWeights {
        AcousticWeights::new(
            2,
            1,
            2,
            vec![1.0, 0.0],
            vec![0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn the_head_follows_the_first_mel_bin() {
        let model = AcousticModel::on_device(&Device::Cpu, &tiny()).unwrap();
        let loud = Tensor::from_vec(vec![3.0f32, 0.0], (1, 2), model.device()).unwrap();
        let quiet = Tensor::from_vec(vec![-2.0f32, 0.0], (1, 2), model.device()).unwrap();
        let loud = model.logits(&loud).unwrap().to_vec2::<f32>().unwrap();
        let quiet = model.logits(&quiet).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(loud[0], [0.0f32, 3.0]);
        assert_eq!(quiet[0], [0.0f32, 0.0]);
    }

    #[test]
    fn a_bad_shape_is_refused() {
        let error = AcousticWeights::new(
            2,
            1,
            2,
            vec![1.0],
            vec![0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        );
        assert_eq!(error.unwrap_err(), ModelError::Shape);
    }

    #[test]
    fn a_non_finite_weight_is_refused() {
        let error = AcousticWeights::new(
            2,
            1,
            2,
            vec![f32::NAN, 0.0],
            vec![0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        );
        assert_eq!(error.unwrap_err(), ModelError::Weights);
    }

    #[test]
    fn truncated_weights_are_refused() {
        assert_eq!(
            AcousticWeights::from_safetensors(b"nope").unwrap_err(),
            ModelError::Weights
        );
    }

    #[test]
    fn safetensors_round_trip_the_head() {
        let weights = tiny();
        let hidden_w = f32_bytes(&weights.hidden_weight);
        let hidden_b = f32_bytes(&weights.hidden_bias);
        let output_w = f32_bytes(&weights.output_weight);
        let output_b = f32_bytes(&weights.output_bias);
        let bytes = serialize(
            [
                (
                    "hidden.weight",
                    TensorView::new(Dtype::F32, vec![1, 2], &hidden_w).unwrap(),
                ),
                (
                    "hidden.bias",
                    TensorView::new(Dtype::F32, vec![1], &hidden_b).unwrap(),
                ),
                (
                    "output.weight",
                    TensorView::new(Dtype::F32, vec![2, 1], &output_w).unwrap(),
                ),
                (
                    "output.bias",
                    TensorView::new(Dtype::F32, vec![2], &output_b).unwrap(),
                ),
            ],
            None,
        )
        .unwrap();
        let loaded = AcousticWeights::from_safetensors(&bytes).unwrap();
        assert_eq!(loaded, weights);
    }

    #[test]
    fn a_wav_becomes_words_on_the_draft() {
        let wav = TempWav::write(vec![0; 16_000]);
        let mut draft = Draft::empty("D");
        let material_id = cutlass_model::MaterialId::new("caption");
        draft
            .materials
            .texts
            .push(MaterialText::new(material_id.clone()));
        let mut session = Session::open(draft);
        let job = Job::start(&session);
        let weights = AcousticWeights::new(
            N_MELS,
            1,
            2,
            vec![0.0; N_MELS],
            vec![1.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        )
        .unwrap();
        let recognizer =
            LocalRecognizer::cpu(weights, vec![String::new(), "on".to_owned()]).unwrap();
        let mut source = FileSource::new(&wav.path);
        let revision = job
            .recognize(
                &mut session,
                &material_id,
                &mut source,
                &TimeRange::new(0, 1_000_000),
                &recognizer,
            )
            .unwrap();
        assert_eq!(revision, 1);
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.words.text, ["on"]);
        assert_eq!(text.words.start_time, [0]);
        assert_eq!(text.words.end_time, [980]);
        assert_eq!(text.current_words.text, text.words.text);
        assert_eq!(text.content, "on");
        assert_eq!(text.base_content, "on");
    }

    #[test]
    fn the_wrong_rate_is_refused() {
        let recognizer = bias_recognizer(vec![0.0], vec![1.0, 0.0]);
        let error = recognizer
            .recognize(&Pcm {
                sample_rate: 8_000,
                samples: vec![0.0; 400],
            })
            .unwrap_err();
        assert_eq!(
            error,
            RecognizeError::SampleRate {
                expected: 16_000,
                actual: 8_000,
            }
        );
    }

    #[test]
    fn a_non_finite_sample_is_refused() {
        let recognizer = bias_recognizer(vec![0.0], vec![1.0, 0.0]);
        let error = recognizer
            .recognize(&Pcm {
                sample_rate: 16_000,
                samples: vec![f32::NAN],
            })
            .unwrap_err();
        assert_eq!(error, RecognizeError::Model);
    }

    fn bias_recognizer(bias: Vec<f32>, output_weight: Vec<f32>) -> LocalRecognizer {
        LocalRecognizer::cpu(
            AcousticWeights::new(
                N_MELS,
                1,
                2,
                vec![0.0; N_MELS],
                bias,
                output_weight,
                vec![0.0, 0.0],
            )
            .unwrap(),
            vec![String::new(), "on".to_owned()],
        )
        .unwrap()
    }

    #[test]
    fn silence_writes_no_words() {
        let recognizer = bias_recognizer(vec![0.0], vec![1.0, 0.0]);
        let words = recognizer
            .recognize(&Pcm {
                sample_rate: 16_000,
                samples: vec![0.0; 16_000],
            })
            .unwrap();
        assert!(words.is_empty());
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    struct TempWav {
        path: PathBuf,
    }

    impl TempWav {
        fn write(samples: Vec<i16>) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cutlass-transcription-recognizer-{}.wav",
                std::process::id()
            ));
            let data_bytes = samples.len() * 2;
            let mut bytes = Vec::with_capacity(44 + data_bytes);
            bytes.extend_from_slice(b"RIFF");
            bytes.extend_from_slice(&(36 + data_bytes as u32).to_le_bytes());
            bytes.extend_from_slice(b"WAVEfmt ");
            bytes.extend_from_slice(&16_u32.to_le_bytes());
            bytes.extend_from_slice(&1_u16.to_le_bytes());
            bytes.extend_from_slice(&1_u16.to_le_bytes());
            bytes.extend_from_slice(&16_000_u32.to_le_bytes());
            bytes.extend_from_slice(&(16_000_u32 * 2).to_le_bytes());
            bytes.extend_from_slice(&2_u16.to_le_bytes());
            bytes.extend_from_slice(&16_u16.to_le_bytes());
            bytes.extend_from_slice(b"data");
            bytes.extend_from_slice(&(data_bytes as u32).to_le_bytes());
            for sample in samples {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            std::fs::write(&path, bytes).unwrap();
            Self { path }
        }
    }

    impl Drop for TempWav {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
