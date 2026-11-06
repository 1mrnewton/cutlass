//! MobileSAM on candle, from the model code in `candle-transformers`.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam::Sam;

use crate::engine::{BenchError, Engine, Masks, Point};
use crate::frame::Frame;

pub struct CandleEngine {
    device: Device,
    name: &'static str,
    sam: Sam,
    embeddings: Option<Tensor>,
    frame_size: (usize, usize),
}

impl CandleEngine {
    /// Loads `mobile_sam-tiny-vitt.safetensors` onto the CPU, or onto the GPU
    /// when `gpu` is set and this build has one for candle.
    pub fn load(weights: &Path, gpu: bool) -> Result<Self, BenchError> {
        let (device, name) = if gpu {
            gpu_device()?
        } else {
            (Device::Cpu, "cpu")
        };
        let bytes = std::fs::read(weights)
            .map_err(|error| BenchError::Io(format!("{}: {error}", weights.display())))?;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device)?;
        let sam = Sam::new_tiny(vb)?;
        Ok(Self {
            device,
            name,
            sam,
            embeddings: None,
            frame_size: (0, 0),
        })
    }
}

#[cfg(feature = "metal")]
fn gpu_device() -> Result<(Device, &'static str), BenchError> {
    Ok((Device::new_metal(0)?, "metal"))
}

#[cfg(not(feature = "metal"))]
fn gpu_device() -> Result<(Device, &'static str), BenchError> {
    Err(BenchError::Usage(
        "candle has no GPU in this build: it runs on Metal (build with --features metal) or CUDA, \
         and has no Vulkan or DirectX backend"
            .into(),
    ))
}

impl Engine for CandleEngine {
    fn label(&self) -> String {
        format!("candle {}", self.name)
    }

    fn encode(&mut self, frame: &Frame) -> Result<(), BenchError> {
        let image = Tensor::from_slice(&frame.hwc, (frame.height, frame.width, 3), &self.device)?
            .permute((2, 0, 1))?
            .contiguous()?;
        let embeddings = self.sam.embeddings(&image)?;
        self.device.synchronize()?;
        self.embeddings = Some(embeddings);
        self.frame_size = (frame.height, frame.width);
        Ok(())
    }

    fn embeddings(&self) -> Result<Vec<f32>, BenchError> {
        let embeddings = self.embeddings.as_ref().ok_or_else(no_frame)?;
        Ok(embeddings.flatten_all()?.to_vec1::<f32>()?)
    }

    fn decode(&mut self, points: &[Point]) -> Result<Masks, BenchError> {
        let embeddings = self.embeddings.as_ref().ok_or_else(no_frame)?;
        let (height, width) = self.frame_size;
        // candle adds SAM's padding point itself, and takes points as parts
        // of the resized frame.
        #[allow(clippy::cast_precision_loss)]
        let prompt: Vec<(f64, f64, bool)> = points
            .iter()
            .filter(|point| point.label >= 0.0)
            .map(|point| {
                (
                    f64::from(point.x) / width as f64,
                    f64::from(point.y) / height as f64,
                    point.label > 0.5,
                )
            })
            .collect();
        let (logits, scores) = self
            .sam
            .forward_for_embeddings(embeddings, height, width, &prompt, true)?;
        Ok(Masks {
            scores: scores.flatten_all()?.to_vec1::<f32>()?,
            logits: logits.flatten_all()?.to_vec1::<f32>()?,
        })
    }
}

fn no_frame() -> BenchError {
    BenchError::Usage("decode ran before encode".into())
}
