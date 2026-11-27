//! Linux hardware video codecs and the ALSA preview queue.
//!
//! H.264, HEVC, and AV1 decode through NVDEC, and H.264 and HEVC encode
//! through NVENC, on an NVIDIA GPU. H.264 and HEVC decode and encode
//! through VA-API on Intel and AMD, and on any other GPU with a VA-API
//! driver. All of them sit behind [`cutlass_decode::Decoder`] and
//! [`cutlass_encode::Encoder`]. The desktop registers them ahead of the
//! Rust codecs. An opener that this machine cannot run returns
//! [`cutlass_decode::CodecError::Unsupported`], and the factory asks the
//! next opener. Off Linux every opener declines.
//!
//! [`OutputQueue`] plays 48 kHz stereo `f32` on the default ALSA device,
//! which on a desktop is usually PipeWire or PulseAudio behind ALSA's plugin.
//! Playback calls this safe queue. This crate does not mix samples.
//!
//! Every system library (`libasound.so.2`, `libva.so.2`, `libcuda.so.1`,
//! `libnvcuvid.so.1`, `libnvidia-encode.so.1`) is loaded when it is first
//! needed, so the build needs no headers, and a system without one of them
//! starts and uses the Rust codecs instead.

#![deny(clippy::undocumented_unsafe_blocks)]

mod slice;
mod video;

#[cfg(target_os = "linux")]
mod alsa;
#[cfg(target_os = "linux")]
mod dl;
#[cfg(not(target_os = "linux"))]
mod off;

#[cfg(target_os = "linux")]
pub use alsa::OutputQueue;
#[cfg(not(target_os = "linux"))]
pub use off::OutputQueue;

pub use slice::QueueError;
/// Lines describing the hardware video this machine offers: the NVIDIA
/// GPU, what NVDEC decodes and NVENC encodes there, and each VA-API render
/// node. For the `video_probe` example and bug reports.
pub use video::probe as probe_video;
pub use video::{
    BACKEND_VAR, register_decoders, register_decoders_with, register_encoders,
    register_encoders_with,
};
