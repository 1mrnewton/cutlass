//! MobileSAM on burn: click-to-select.
//!
//! The encoder reads a frame resized to 1024 on its longest side and returns
//! 256 by 64 by 64 embeddings. The decoder turns those and the clicks into
//! four low-resolution masks: the first for several clicks, and three
//! candidates for one click.

use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};

use crate::device::{self, Compute};
use crate::frame::{self, Resized};
use crate::generated::mobile_sam_image_encoder as encoder;
use crate::generated::sam_mask_decoder_lowres as decoder;
use crate::mask::{self, LOW_RES, Mask};
use crate::{Click, Image, Selector, VisionError, check_clicks};

/// The encoder's weights, in the folder passed to [`MobileSam::load`].
pub const ENCODER_WEIGHTS: &str = "mobile_sam_image_encoder.bpk";
/// The decoder's weights, in the same folder.
pub const DECODER_WEIGHTS: &str = "sam_mask_decoder_lowres.bpk";

const EMBEDDINGS: [usize; 4] = [1, 256, 64, 64];
/// The decoder's masks: one for several clicks, then three for one click.
const CANDIDATES: usize = 4;

/// MobileSAM, loaded on one device.
pub struct MobileSam {
    device: Device,
    encoder: encoder::Model,
    decoder: decoder::Model,
    frame: Option<Encoded>,
}

/// The last frame the encoder read.
struct Encoded {
    embeddings: Tensor<4>,
    source: (u32, u32),
    resized: (u32, u32),
    scale: f32,
}

impl MobileSam {
    /// Loads [`ENCODER_WEIGHTS`] and [`DECODER_WEIGHTS`] from `weights`.
    pub fn load(weights: &Path, compute: Compute) -> Result<Self, VisionError> {
        let device = device::device(compute)?;
        let mut encoder = encoder::Model::new(&device);
        load_weights(&mut encoder, &weights.join(ENCODER_WEIGHTS))?;
        let mut decoder = decoder::Model::new(&device);
        load_weights(&mut decoder, &weights.join(DECODER_WEIGHTS))?;
        Ok(Self {
            device,
            encoder,
            decoder,
            frame: None,
        })
    }

    /// Runs the encoder on a frame [`frame::resize`] already checked.
    fn encode_resized(&mut self, resized: Resized, source: (u32, u32)) -> Result<(), VisionError> {
        let shape = [resized.height as usize, resized.width as usize, 3];
        let image = Tensor::<3>::from_data(TensorData::new(resized.hwc, shape), &self.device);
        let embeddings = self.encoder.forward(image);
        if embeddings.dims() != EMBEDDINGS {
            return Err(VisionError::Model {
                message: format!("the encoder returned {:?}", embeddings.dims()),
            });
        }
        self.frame = Some(Encoded {
            embeddings,
            source,
            resized: (resized.width, resized.height),
            scale: resized.scale,
        });
        Ok(())
    }

    /// The decoder's four scores and four masks of logits for points in the
    /// resized frame. Each point is `x`, `y`, and 1 to keep or 0 to remove.
    fn low_res(&self, points: &[[f32; 3]]) -> Result<(Vec<f32>, Vec<f32>), VisionError> {
        let frame = self.frame.as_ref().ok_or(VisionError::NoFrame)?;
        // SAM's padding point closes a prompt that has no box.
        let prompt: Vec<[f32; 3]> = points.iter().copied().chain([[0.0, 0.0, -1.0]]).collect();
        let count = prompt.len();
        let coords: Vec<f32> = prompt.iter().flat_map(|[x, y, _]| [*x, *y]).collect();
        let labels: Vec<f32> = prompt.iter().map(|[_, _, label]| *label).collect();
        let coords = Tensor::<3>::from_data(TensorData::new(coords, [1, count, 2]), &self.device);
        let labels = Tensor::<2>::from_data(TensorData::new(labels, [1, count]), &self.device);
        let no_mask = Tensor::<4>::zeros([1, 1, LOW_RES, LOW_RES], &self.device);
        let has_mask = Tensor::<1>::zeros([1], &self.device);
        let (scores, logits) =
            self.decoder
                .forward(frame.embeddings.clone(), coords, labels, no_mask, has_mask);
        let scores = read(scores.into_data())?;
        let logits = read(logits.into_data())?;
        if scores.len() != CANDIDATES || logits.len() != CANDIDATES * LOW_RES * LOW_RES {
            return Err(VisionError::Model {
                message: format!(
                    "the decoder returned {} scores and {} logits",
                    scores.len(),
                    logits.len()
                ),
            });
        }
        Ok((scores, logits))
    }
}

impl Selector for MobileSam {
    fn encode(&mut self, image: &Image<'_>) -> Result<(), VisionError> {
        let resized = frame::resize(image)?;
        self.encode_resized(resized, (image.width, image.height))
    }

    fn select(&self, clicks: &[Click]) -> Result<Mask, VisionError> {
        let frame = self.frame.as_ref().ok_or(VisionError::NoFrame)?;
        check_clicks(clicks, frame.source.0, frame.source.1)?;
        let points: Vec<[f32; 3]> = clicks
            .iter()
            .map(|c| {
                let label = if c.keep { 1.0 } else { 0.0 };
                [c.x * frame.scale, c.y * frame.scale, label]
            })
            .collect();
        let (scores, logits) = self.low_res(&points)?;
        let pick = pick(&scores, clicks.len());
        let len = LOW_RES * LOW_RES;
        let (Some(score), Some(chosen)) =
            (scores.get(pick), logits.get(pick * len..(pick + 1) * len))
        else {
            return Err(VisionError::Model {
                message: "the chosen mask is missing".into(),
            });
        };
        mask::upscale(chosen, frame.resized, frame.source, *score)
    }
}

/// SAM's own rule: one click is ambiguous, so the best of the three
/// candidates wins; several clicks use the single mask made for them.
fn pick(scores: &[f32], clicks: usize) -> usize {
    if clicks > 1 {
        return 0;
    }
    let mut best = 1;
    for (index, score) in scores.iter().enumerate().skip(1) {
        if scores.get(best).is_some_and(|top| score > top) {
            best = index;
        }
    }
    best
}

fn read(data: TensorData) -> Result<Vec<f32>, VisionError> {
    data.try_to_vec::<f32>()
        .map_err(|error| VisionError::Model {
            message: format!("{error:?}"),
        })
}

/// Loads a `.bpk` file into `model`. Every tensor the model has must come
/// from the file, at the shape the model expects.
fn load_weights<M: ModuleSnapshot>(model: &mut M, path: &Path) -> Result<(), VisionError> {
    let refused = |message: String| VisionError::Weights {
        path: path.to_path_buf(),
        message,
    };
    if !path.is_file() {
        return Err(refused("missing".into()));
    }
    let mut store = BurnpackStore::from_file(path);
    let applied = model
        .load_from(&mut store)
        .map_err(|error| refused(error.to_string()))?;
    if !applied.errors.is_empty() || !applied.missing.is_empty() {
        return Err(refused(format!(
            "{} tensors failed and {} are missing",
            applied.errors.len(),
            applied.missing.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use safetensors::SafeTensors;

    use super::*;

    #[test]
    fn one_click_takes_the_best_candidate_and_several_take_the_first_mask() {
        assert_eq!(pick(&[0.99, 0.2, 0.9, 0.5], 1), 2);
        assert_eq!(pick(&[0.1, 0.8, 0.8, 0.5], 1), 1);
        assert_eq!(pick(&[0.1, 0.8, 0.9, 0.95], 2), 0);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cutlass-vision-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_weights_name_the_file() {
        let dir = scratch("missing");
        let Err(VisionError::Weights { path, message }) = MobileSam::load(&dir, Compute::Cpu)
        else {
            panic!("loaded without weights");
        };
        assert_eq!(path, dir.join(ENCODER_WEIGHTS));
        assert_eq!(message, "missing");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_bad_weights_file_is_an_error_and_not_a_panic() {
        let dir = scratch("bad");
        std::fs::write(dir.join(ENCODER_WEIGHTS), b"not a burnpack").unwrap();
        assert!(matches!(
            MobileSam::load(&dir, Compute::Cpu),
            Err(VisionError::Weights { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Set in a tagged run, the tests that need the real weights fail when
    /// they are missing instead of skipping.
    const REQUIRE: &str = "CUTLASS_REQUIRE_VISION_MODEL";

    fn local_models() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/models/vision")
    }

    /// What onnxruntime produced for one frame and one click, from
    /// `tools/ml-bench/scripts/reference.py`.
    struct Reference {
        resized: Resized,
        click: [f32; 2],
        embeddings: Vec<f32>,
        scores: Vec<f32>,
        logits: Vec<f32>,
    }

    fn reference(name: &str) -> Option<Reference> {
        let dir = local_models();
        let file = dir.join(format!("reference_{name}.safetensors"));
        if !dir.join(ENCODER_WEIGHTS).is_file() || !file.is_file() {
            assert!(
                std::env::var_os(REQUIRE).is_none(),
                "{REQUIRE} is set and the MobileSAM weights or {} are missing",
                file.display()
            );
            return None;
        }
        let bytes = std::fs::read(&file).unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let values = |key: &str| -> (Vec<f32>, Vec<usize>) {
            let view = tensors.tensor(key).unwrap();
            let values = view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            (values, view.shape().to_vec())
        };
        let (hwc, shape) = values("resized");
        let (coords, _) = values("point_coords");
        Some(Reference {
            resized: Resized {
                hwc,
                width: u32::try_from(shape[1]).unwrap(),
                height: u32::try_from(shape[0]).unwrap(),
                scale: 1.0,
            },
            click: [coords[0], coords[1]],
            embeddings: values("embeddings").0,
            scores: values("iou").0,
            logits: values("low_res_masks").0,
        })
    }

    fn iou(a: &[f32], b: &[f32]) -> f64 {
        let (mut both, mut either) = (0_u32, 0_u32);
        for (x, y) in a.iter().zip(b) {
            both += u32::from(*x > 0.0 && *y > 0.0);
            either += u32::from(*x > 0.0 || *y > 0.0);
        }
        if either == 0 {
            1.0
        } else {
            f64::from(both) / f64::from(either)
        }
    }

    /// The masks match onnxruntime's, and a click through the public API
    /// selects part of the frame at the frame's size.
    fn matches_onnxruntime(compute: Compute) {
        for name in ["landscape", "portrait"] {
            let Some(reference) = reference(name) else {
                return;
            };
            let mut sam = MobileSam::load(&local_models(), compute).unwrap();
            assert_eq!(
                sam.select(&[Click {
                    x: 1.0,
                    y: 1.0,
                    keep: true
                }])
                .err(),
                Some(VisionError::NoFrame)
            );
            let size = (reference.resized.width, reference.resized.height);
            let rgb: Vec<u8> = reference
                .resized
                .hwc
                .iter()
                .map(|v| {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let byte = v.round().clamp(0.0, 255.0) as u8;
                    byte
                })
                .collect();
            sam.encode_resized(reference.resized, size).unwrap();

            let frame = sam.frame.as_ref().unwrap();
            let embeddings = read(frame.embeddings.clone().into_data()).unwrap();
            let drift = embeddings
                .iter()
                .zip(&reference.embeddings)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(drift < 1e-3, "{name}: embeddings drift {drift}");

            let [x, y] = reference.click;
            let (scores, logits) = sam.low_res(&[[x, y, 1.0]]).unwrap();
            let len = LOW_RES * LOW_RES;
            for (candidate, (score, expected)) in
                scores.iter().zip(&reference.scores).enumerate().skip(1)
            {
                let range = candidate * len..(candidate + 1) * len;
                let overlap = iou(&logits[range.clone()], &reference.logits[range]);
                assert!(
                    overlap > 0.99,
                    "{name}: mask {candidate} overlaps {overlap}"
                );
                let off = (score - expected).abs();
                assert!(off < 1e-3, "{name}: score {candidate} is off by {off}");
            }

            let image = Image {
                width: size.0,
                height: size.1,
                rgb: &rgb,
            };
            sam.encode(&image).unwrap();
            let mask = sam.select(&[Click { x, y, keep: true }]).unwrap();
            assert_eq!((mask.width, mask.height), size);
            let selected = mask.selected();
            assert!(
                selected > 0 && selected < mask.alpha.len(),
                "{name}: {selected}"
            );
            assert!(mask.score > 0.5, "{name}: score {}", mask.score);
        }
    }

    #[test]
    fn the_cpu_matches_onnxruntime() {
        matches_onnxruntime(Compute::Cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs a Metal GPU and about half a minute of kernel warm-up"]
    fn the_mac_gpu_matches_onnxruntime() {
        matches_onnxruntime(Compute::Gpu);
    }
}
