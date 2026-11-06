//! MobileSAM on burn, from code `burn-onnx` generated out of the ONNX files.

use burn::tensor::{Device, Tensor, TensorData};

use crate::engine::{BenchError, Engine, MASK_SIDE, Masks, Point};
use crate::frame::Frame;

// The generated code is burn-onnx's, not ours. It unwraps and panics on a bad
// weights file and on shapes it did not expect, which the editor's own crates
// may not do. The bench lets it; a Cutlass crate would have to answer for it.
#[allow(
    clippy::all,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::get_unwrap,
    rust_2018_idioms,
    unused,
    unused_qualifications,
    trivial_casts,
    trivial_numeric_casts
)]
mod encoder {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/mobile_sam_image_encoder.rs"
    ));
}
#[allow(
    clippy::all,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::get_unwrap,
    rust_2018_idioms,
    unused,
    unused_qualifications,
    trivial_casts,
    trivial_numeric_casts
)]
mod decoder {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/sam_mask_decoder_lowres.rs"
    ));
}

pub struct BurnEngine {
    device: Device,
    name: &'static str,
    encoder: encoder::Model,
    decoder: decoder::Model,
    embeddings: Option<Tensor<4>>,
}

impl BurnEngine {
    /// Loads the weights built into the binary onto the CPU (burn's flex
    /// backend), or onto the GPU through wgpu when `gpu` is set.
    pub fn load(gpu: bool) -> Result<Self, BenchError> {
        let (device, name) = if gpu {
            gpu_device()?
        } else {
            (Device::flex(), "flex-cpu")
        };
        let encoder = encoder::Model::from_embedded(&device);
        let decoder = decoder::Model::from_embedded(&device);
        Ok(Self {
            device,
            name,
            encoder,
            decoder,
            embeddings: None,
        })
    }

    fn sync(&self) -> Result<(), BenchError> {
        self.device
            .sync()
            .map_err(|error| BenchError::Shape(format!("burn device sync: {error:?}")))
    }
}

#[cfg(feature = "metal")]
fn gpu_device() -> Result<(Device, &'static str), BenchError> {
    Ok((
        Device::metal(burn::tensor::DeviceKind::DefaultDevice),
        "wgpu-metal",
    ))
}

#[cfg(all(feature = "vulkan", not(feature = "metal")))]
fn gpu_device() -> Result<(Device, &'static str), BenchError> {
    Ok((
        Device::vulkan(burn::tensor::DeviceKind::DefaultDevice),
        "wgpu-vulkan",
    ))
}

#[cfg(not(any(feature = "metal", feature = "vulkan")))]
fn gpu_device() -> Result<(Device, &'static str), BenchError> {
    Err(BenchError::Usage(
        "burn has no GPU in this build: build with --features metal or --features vulkan".into(),
    ))
}

fn to_vec(data: &TensorData) -> Result<Vec<f32>, BenchError> {
    data.as_slice::<f32>()
        .map(<[f32]>::to_vec)
        .map_err(|error| BenchError::Shape(format!("burn tensor read: {error:?}")))
}

impl Engine for BurnEngine {
    fn label(&self) -> String {
        format!("burn {}", self.name)
    }

    fn encode(&mut self, frame: &Frame) -> Result<(), BenchError> {
        let image = Tensor::<3>::from_data(
            TensorData::new(frame.hwc.clone(), [frame.height, frame.width, 3]),
            &self.device,
        );
        let embeddings = self.encoder.forward(image);
        self.embeddings = Some(embeddings);
        self.sync()
    }

    fn embeddings(&self) -> Result<Vec<f32>, BenchError> {
        let embeddings = self
            .embeddings
            .as_ref()
            .ok_or_else(|| BenchError::Usage("decode ran before encode".into()))?;
        to_vec(&embeddings.clone().into_data())
    }

    fn decode(&mut self, points: &[Point]) -> Result<Masks, BenchError> {
        let embeddings = self
            .embeddings
            .clone()
            .ok_or_else(|| BenchError::Usage("decode ran before encode".into()))?;
        // The ONNX decoder takes SAM's padding point from the caller.
        let mut prompt: Vec<Point> = points.iter().copied().filter(|p| p.label >= 0.0).collect();
        prompt.push(Point {
            x: 0.0,
            y: 0.0,
            label: -1.0,
        });
        let coords: Vec<f32> = prompt.iter().flat_map(|p| [p.x, p.y]).collect();
        let labels: Vec<f32> = prompt.iter().map(|p| p.label).collect();
        let count = prompt.len();
        let coords = Tensor::<3>::from_data(TensorData::new(coords, [1, count, 2]), &self.device);
        let labels = Tensor::<2>::from_data(TensorData::new(labels, [1, count]), &self.device);
        let mask_input = Tensor::<4>::zeros([1, 1, MASK_SIDE, MASK_SIDE], &self.device);
        let has_mask = Tensor::<1>::zeros([1], &self.device);
        let (scores, logits) = self
            .decoder
            .forward(embeddings, coords, labels, mask_input, has_mask);
        Masks::from_all_four(&to_vec(&scores.into_data())?, &to_vec(&logits.into_data())?)
    }
}
