//! FFmpeg backend for the [`cutlass_core::VideoEncoder`] contract.
//!
//! The export-seam mirror of [`cutlass_decoder::FfmpegDecoder`]: the
//! platform-native encoders (VideoToolbox / MediaCodec / Media Foundation) cover
//! mobile + Apple/Windows; this wraps the in-tree [`VideoExport`] so Linux and
//! desktop dev builds can mux an H.264 MP4 through the same trait the future
//! Rust exporter and the Python bindings drive.
//!
//! Frames arrive as core [`VideoFrame`]s; the wrapper feeds RGBA straight to the
//! configured output size and routes planar YUV through `push_yuv420p` (which
//! scales/converts as needed). Padded plane strides are compacted to the tight
//! buffers FFmpeg expects.

use std::path::Path;

use cutlass_core::{
    CpuImage, EncodeError, EncoderConfig, PixelFormat, RationalTime, VideoEncoder, VideoFrame,
};

use crate::EncodeError as FfEncodeError;
use crate::{ExportConfig, VideoExport};

/// A [`cutlass_core::VideoEncoder`] backed by the FFmpeg [`VideoExport`].
pub struct FfmpegEncoder {
    // `Option` so `finish` (which takes `&mut self`) can consume the inner
    // `VideoExport::finish(self)` exactly once.
    inner: Option<VideoExport>,
}

// SAFETY: the wrapped FFmpeg output context and encoder are only ever touched by
// one thread at a time — the trait is `&mut self` and an export worker owns the
// encoder exclusively (never shared as `&self`). Moving the handle to another
// thread is sound; this mirrors the platform backends' `unsafe impl Send`.
unsafe impl Send for FfmpegEncoder {}

impl FfmpegEncoder {
    /// Open an H.264 MP4 encoder at `path` for `config`.
    pub fn open(path: &Path, config: EncoderConfig) -> Result<Self, EncodeError> {
        let export = VideoExport::create(path, to_export_config(config)?).map_err(to_core_err)?;
        Ok(Self {
            inner: Some(export),
        })
    }

    fn export(&mut self) -> Result<&mut VideoExport, EncodeError> {
        self.inner
            .as_mut()
            .ok_or_else(|| EncodeError::Encode("encoder already finished".into()))
    }
}

impl VideoEncoder for FfmpegEncoder {
    fn push(&mut self, frame: &VideoFrame) -> Result<(), EncodeError> {
        let (width, height) = (frame.width(), frame.height());
        let format = frame.format;
        let cpu = frame
            .cpu()
            .ok_or_else(|| EncodeError::unsupported("FFmpeg encoder needs CPU frames"))?;
        let export = self.export()?;

        match format {
            PixelFormat::Rgba8 => {
                let row_bytes = width as usize * 4;
                let rgba = tight_plane(cpu, 0, row_bytes, height as usize)?;
                export.push_rgba(&rgba).map_err(to_core_err)
            }
            PixelFormat::Yuv420p => {
                let y = tight_plane(cpu, 0, width as usize, height as usize)?;
                let chroma_w = width as usize / 2;
                let chroma_h = height as usize / 2;
                let u = tight_plane(cpu, 1, chroma_w, chroma_h)?;
                let v = tight_plane(cpu, 2, chroma_w, chroma_h)?;
                export
                    .push_yuv420p(width, height, &y, &u, &v)
                    .map_err(to_core_err)
            }
            other => Err(EncodeError::unsupported(format!(
                "FFmpeg encoder cannot ingest {other:?} frames"
            ))),
        }
    }

    fn push_audio(&mut self, samples: &[f32], _pts: RationalTime) -> Result<(), EncodeError> {
        // `VideoExport` interleaves audio into the muxer in arrival order; the
        // explicit pts isn't needed (video/audio share the export timeline).
        self.export()?.push_audio(samples).map_err(to_core_err)
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        match self.inner.take() {
            Some(export) => export.finish().map(|_| ()).map_err(to_core_err),
            None => Ok(()),
        }
    }
}

/// Copy plane `index` into a tight `row_bytes × rows` buffer, dropping any
/// stride padding (FFmpeg's `push_*` paths want packed input).
fn tight_plane(
    cpu: &CpuImage,
    index: usize,
    row_bytes: usize,
    rows: usize,
) -> Result<Vec<u8>, EncodeError> {
    let plane = cpu
        .planes
        .get(index)
        .ok_or_else(|| EncodeError::Encode(format!("frame missing plane {index}")))?;
    let need = row_bytes
        .checked_mul(rows)
        .ok_or_else(|| EncodeError::Encode("frame plane size overflow".into()))?;

    if plane.stride == row_bytes {
        if plane.data.len() < need {
            return Err(EncodeError::Encode(format!(
                "plane {index} too small: have {}, need {need}",
                plane.data.len()
            )));
        }
        return Ok(plane.data[..need].to_vec());
    }

    if plane.stride < row_bytes || plane.data.len() < plane.stride * rows {
        return Err(EncodeError::Encode(format!(
            "plane {index} stride {} too small for {row_bytes}×{rows}",
            plane.stride
        )));
    }

    let mut out = Vec::with_capacity(need);
    for row in 0..rows {
        let start = row * plane.stride;
        out.extend_from_slice(&plane.data[start..start + row_bytes]);
    }
    Ok(out)
}

fn to_export_config(config: EncoderConfig) -> Result<ExportConfig, EncodeError> {
    let (width, height) = config.size;
    let audio_rate = match config.audio {
        Some(audio) => {
            if audio.channels != 2 {
                return Err(EncodeError::unsupported(
                    "FFmpeg encoder only supports stereo audio",
                ));
            }
            Some(audio.sample_rate)
        }
        None => None,
    };

    Ok(ExportConfig {
        width,
        height,
        frame_rate_num: config.frame_rate.num,
        frame_rate_den: config.frame_rate.den,
        audio_rate,
        ..ExportConfig::default()
    })
}

fn to_core_err(err: FfEncodeError) -> EncodeError {
    match err {
        FfEncodeError::Open(e) => EncodeError::Start(e.to_string()),
        FfEncodeError::Io(e) => EncodeError::Encode(e.to_string()),
        FfEncodeError::Encode(e) => EncodeError::Encode(e.to_string()),
        FfEncodeError::Unsupported { what } => EncodeError::Unsupported(what),
        FfEncodeError::Decode(e) => EncodeError::Encode(e.to_string()),
    }
}
