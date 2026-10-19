//! Frame buffers for preview and export.
//!
//! The decode→composite currency is [`cutlass_core::VideoFrame`]: the engine
//! pulls frames from the FFmpeg desktop backend (and, on mobile, from the
//! platform-native backends) as `VideoFrame`s and turns them into compositor
//! layers here. GPU conversion is the default path ([`decoded_to_yuv_layer`]);
//! the CPU routines are legacy fallbacks kept for tests and
//! [`ColorConvertPath::LegacyCpu`](crate::ColorConvertPath).

use cutlass_compositor::Yuv420pLayer;
use cutlass_core::{
    ColorSpace, CpuImage, FrameData, PixelFormat, Plane, Rational, RationalTime, Rect, Rotation,
    VideoFrame,
};

use crate::error::EngineError;

/// RGBA8 preview frame (row-major, tightly packed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaFrame {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

impl RgbaFrame {
    pub fn new(width: u32, height: u32, bytes: Vec<u8>) -> Result<Self, EngineError> {
        let expected = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().map(|h| w * h * 4))
            .ok_or_else(|| EngineError::Preview("invalid frame dimensions".into()))?;
        if bytes.len() != expected {
            return Err(EngineError::Preview(format!(
                "rgba buffer is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        Ok(Self {
            width,
            height,
            bytes,
        })
    }
}

/// The CPU planes of a frame, or an error if it is a (zero-copy) GPU surface —
/// the desktop FFmpeg path is always CPU; GPU surfaces arrive only from the
/// mobile backends, which the compositor imports directly (not through here).
fn cpu_planes(frame: &VideoFrame) -> Result<&[Plane], EngineError> {
    frame
        .cpu()
        .map(|image| image.planes.as_slice())
        .ok_or_else(|| EngineError::Preview("expected a CPU frame, got a GPU surface".into()))
}

fn plane(planes: &[Plane], index: usize) -> Result<&Plane, EngineError> {
    planes
        .get(index)
        .ok_or_else(|| EngineError::Preview(format!("frame missing plane {index}")))
}

/// Build a [`Yuv420pLayer`] from a decoded frame for GPU conversion.
pub fn decoded_to_yuv_layer(frame: &VideoFrame) -> Result<Yuv420pLayer, EngineError> {
    let planes = cpu_planes(frame)?;
    let full_range = frame.color.range.is_full();
    let (width, height) = (frame.width(), frame.height());
    match frame.format {
        PixelFormat::Yuv420p => {
            let y = plane(planes, 0)?;
            let u = plane(planes, 1)?;
            let v = plane(planes, 2)?;
            Ok(Yuv420pLayer::new(
                width,
                height,
                y.data.clone(),
                y.stride as u32,
                u.data.clone(),
                u.stride as u32,
                v.data.clone(),
                v.stride as u32,
            )
            .with_full_range(full_range))
        }
        // Hardware decoders (VideoToolbox / NVDEC / MediaCodec) deliver frames
        // as NV12; the compositor's YUV pipeline wants planar U/V, so split the
        // interleaved chroma once here on the way in.
        PixelFormat::Nv12 => {
            let y = plane(planes, 0)?;
            let (u, v) = deinterleave_nv12(frame)?;
            let uv_stride = width / 2;
            Ok(Yuv420pLayer::new(
                width,
                height,
                y.data.clone(),
                y.stride as u32,
                u,
                uv_stride,
                v,
                uv_stride,
            )
            .with_full_range(full_range))
        }
        other => Err(EngineError::Preview(format!(
            "unsupported pixel format for GPU YUV layer: {other:?}"
        ))),
    }
}

/// Split an NV12 frame's interleaved chroma plane into tight planar U and V
/// (each `width/2 × height/2`). Shared by the GPU YUV path and the CPU
/// fallback — both want separate chroma planes, but hardware decode hands back
/// NV12.
fn deinterleave_nv12(frame: &VideoFrame) -> Result<(Vec<u8>, Vec<u8>), EngineError> {
    if frame.format != PixelFormat::Nv12 {
        return Err(EngineError::Preview("expected NV12 frame".into()));
    }
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    if w == 0 || h == 0 || !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        return Err(EngineError::Preview("invalid NV12 dimensions".into()));
    }
    let planes = cpu_planes(frame)?;
    let uv = plane(planes, 1)?;
    let uv_w = w / 2;
    let uv_h = h / 2;
    let mut u = vec![0u8; uv_w * uv_h];
    let mut v = vec![0u8; uv_w * uv_h];
    for row in 0..uv_h {
        let src = row * uv.stride;
        if src + 2 * uv_w > uv.data.len() {
            return Err(EngineError::Preview("NV12 chroma row out of bounds".into()));
        }
        let dst = row * uv_w;
        for col in 0..uv_w {
            u[dst + col] = uv.data[src + 2 * col];
            v[dst + col] = uv.data[src + 2 * col + 1];
        }
    }
    Ok((u, v))
}

/// Repack an NV12 frame as planar YUV420P (tight chroma planes), so the CPU
/// fallback's YUV→RGBA path can consume hardware-decoded frames too.
fn nv12_as_yuv420p(frame: &VideoFrame) -> Result<VideoFrame, EngineError> {
    let (u, v) = deinterleave_nv12(frame)?;
    let planes = cpu_planes(frame)?;
    let y = plane(planes, 0)?;
    let uv_w = (frame.width() / 2) as usize;
    let uv_h = (frame.height() / 2) as usize;
    let yuv = CpuImage::new(vec![
        Plane::new(y.data.clone(), y.stride, frame.height() as usize),
        Plane::new(u, uv_w, uv_h),
        Plane::new(v, uv_w, uv_h),
    ]);
    Ok(VideoFrame::new(
        frame.pts,
        PixelFormat::Yuv420p,
        frame.color,
        frame.coded_size,
        frame.visible,
        frame.rotation,
        FrameData::Cpu(yuv),
    ))
}

/// Legacy CPU YUV/RGBA conversion.
pub fn legacy_decoded_to_rgba(frame: &VideoFrame) -> Result<RgbaFrame, EngineError> {
    decoded_to_rgba_inner(frame)
}

/// Legacy alias used by older tests and [`ColorConvertPath::LegacyCpu`].
#[allow(dead_code)]
pub fn decoded_to_rgba(frame: &VideoFrame) -> Result<RgbaFrame, EngineError> {
    legacy_decoded_to_rgba(frame)
}

fn decoded_to_rgba_inner(frame: &VideoFrame) -> Result<RgbaFrame, EngineError> {
    match frame.format {
        PixelFormat::Rgba8 => rgba_from_rgba8(frame),
        PixelFormat::Yuv420p => yuv420p_to_rgba(frame),
        PixelFormat::Nv12 => yuv420p_to_rgba(&nv12_as_yuv420p(frame)?),
        other => Err(EngineError::Preview(format!(
            "unsupported pixel format for CPU RGBA conversion: {other:?}"
        ))),
    }
}

fn rgba_from_rgba8(frame: &VideoFrame) -> Result<RgbaFrame, EngineError> {
    let planes = cpu_planes(frame)?;
    let plane = plane(planes, 0)?;
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let mut bytes = Vec::with_capacity(w * h * 4);
    for row in 0..h {
        let start = row * plane.stride;
        let end = start + w * 4;
        if end > plane.data.len() {
            return Err(EngineError::Preview("RGBA plane row out of bounds".into()));
        }
        bytes.extend_from_slice(&plane.data[start..end]);
    }
    RgbaFrame::new(frame.width(), frame.height(), bytes)
}

/// Tight-packed YUV420P (Y, then U, then V) for the frame cache.
pub fn pack_yuv420p(frame: &VideoFrame) -> Result<Vec<u8>, EngineError> {
    if frame.format != PixelFormat::Yuv420p {
        return Err(EngineError::Preview(
            "expected YUV420P for cache pack".into(),
        ));
    }
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    if w == 0 || h == 0 || !h.is_multiple_of(2) || !w.is_multiple_of(2) {
        return Err(EngineError::Preview("invalid YUV420P dimensions".into()));
    }
    let planes = cpu_planes(frame)?;
    let y = plane(planes, 0)?;
    let u = plane(planes, 1)?;
    let v = plane(planes, 2)?;
    let mut out = Vec::with_capacity(w * h + (w / 2) * (h / 2) * 2);
    for row in 0..h {
        let start = row * y.stride;
        out.extend_from_slice(&y.data[start..start + w]);
    }
    let uv_h = h / 2;
    let uv_w = w / 2;
    for row in 0..uv_h {
        let start = row * u.stride;
        out.extend_from_slice(&u.data[start..start + uv_w]);
    }
    for row in 0..uv_h {
        let start = row * v.stride;
        out.extend_from_slice(&v.data[start..start + uv_w]);
    }
    Ok(out)
}

/// Rebuild a YUV420P [`VideoFrame`] from cache bytes. The cache stores raw
/// planes only (PTS/colorimetry are re-stamped by the caller from the source),
/// so this carries placeholder time and the BT.709 default.
pub fn unpack_yuv420p(bytes: &[u8], width: u32, height: u32) -> Result<VideoFrame, EngineError> {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let need = y_size + uv_size * 2;
    if bytes.len() != need {
        return Err(EngineError::Preview(format!(
            "packed YUV is {} bytes, expected {need}",
            bytes.len()
        )));
    }
    let image = CpuImage::new(vec![
        Plane::new(bytes[..y_size].to_vec(), w, h),
        Plane::new(bytes[y_size..y_size + uv_size].to_vec(), w / 2, h / 2),
        Plane::new(bytes[y_size + uv_size..].to_vec(), w / 2, h / 2),
    ]);
    Ok(VideoFrame::new(
        RationalTime::new(0, Rational::new(1, 1)),
        PixelFormat::Yuv420p,
        ColorSpace::BT709,
        (width, height),
        Rect::from_size(width, height),
        Rotation::None,
        FrameData::Cpu(image),
    ))
}

fn yuv420p_to_rgba(frame: &VideoFrame) -> Result<RgbaFrame, EngineError> {
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let planes = cpu_planes(frame)?;
    let y_plane = plane(planes, 0)?;
    let u_plane = plane(planes, 1)?;
    let v_plane = plane(planes, 2)?;
    let full_range = frame.color.range.is_full();
    let mut rgba = vec![0u8; w * h * 4];

    for row in 0..h {
        for col in 0..w {
            let y = i32::from(y_plane.data[row * y_plane.stride + col]);
            let uv_row = row / 2;
            let uv_col = col / 2;
            let u = i32::from(u_plane.data[uv_row * u_plane.stride + uv_col]) - 128;
            let v = i32::from(v_plane.data[uv_row * v_plane.stride + uv_col]) - 128;

            // Mirror the GPU shader (yuv_blit.wgsl): full-range YUV (yuvj420p)
            // skips the 16/219 studio rescale.
            let (r, g, b) = if full_range {
                (
                    ((256 * y + 359 * v + 128) >> 8).clamp(0, 255) as u8,
                    ((256 * y - 88 * u - 183 * v + 128) >> 8).clamp(0, 255) as u8,
                    ((256 * y + 454 * u + 128) >> 8).clamp(0, 255) as u8,
                )
            } else {
                (
                    ((298 * (y - 16) + 409 * v + 128) >> 8).clamp(0, 255) as u8,
                    ((298 * (y - 16) - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8,
                    ((298 * (y - 16) + 516 * u + 128) >> 8).clamp(0, 255) as u8,
                )
            };

            let i = (row * w + col) * 4;
            rgba[i] = r;
            rgba[i + 1] = g;
            rgba[i + 2] = b;
            rgba[i + 3] = 255;
        }
    }

    RgbaFrame::new(frame.width(), frame.height(), rgba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_core::ColorRange;

    fn cpu_frame(format: PixelFormat, width: u32, height: u32, planes: Vec<Plane>) -> VideoFrame {
        VideoFrame::new(
            RationalTime::new(0, Rational::new(1, 1)),
            format,
            ColorSpace::BT709,
            (width, height),
            Rect::from_size(width, height),
            Rotation::None,
            FrameData::Cpu(CpuImage::new(planes)),
        )
    }

    fn solid_yuv420p(width: u32, height: u32, y: u8, u: u8, v: u8) -> VideoFrame {
        let w = width as usize;
        let h = height as usize;
        cpu_frame(
            PixelFormat::Yuv420p,
            width,
            height,
            vec![
                Plane::new(vec![y; w * h], w, h),
                Plane::new(vec![u; (w / 2) * (h / 2)], w / 2, h / 2),
                Plane::new(vec![v; (w / 2) * (h / 2)], w / 2, h / 2),
            ],
        )
    }

    /// NV12: full-res Y plane, then one interleaved U,V,U,V,… chroma plane at
    /// quarter resolution (the layout VideoToolbox / NVDEC transfer back).
    fn solid_nv12(width: u32, height: u32, y: u8, u: u8, v: u8) -> VideoFrame {
        let w = width as usize;
        let h = height as usize;
        let mut uv = Vec::with_capacity((w / 2) * (h / 2) * 2);
        for _ in 0..(w / 2) * (h / 2) {
            uv.push(u);
            uv.push(v);
        }
        cpu_frame(
            PixelFormat::Nv12,
            width,
            height,
            vec![
                Plane::new(vec![y; w * h], w, h),
                Plane::new(uv, w, h / 2),
            ],
        )
    }

    fn set_full_range(frame: &mut VideoFrame, full: bool) {
        frame.color.range = if full {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let frame = solid_yuv420p(64, 64, 128, 128, 128);
        let packed = pack_yuv420p(&frame).unwrap();
        let restored = unpack_yuv420p(&packed, 64, 64).unwrap();
        let original = frame.cpu().unwrap();
        let restored_cpu = restored.cpu().unwrap();
        assert_eq!(restored_cpu.planes[0].data, original.planes[0].data);
        assert_eq!(restored_cpu.planes[1].data, original.planes[1].data);
    }

    #[test]
    fn yuv_gray_maps_to_neutral_rgb() {
        let frame = solid_yuv420p(2, 2, 128, 128, 128);
        let rgba = yuv420p_to_rgba(&frame).unwrap();
        assert_eq!(rgba.bytes.len(), 16);
        assert_eq!(rgba.bytes[0], rgba.bytes[1]);
        assert_eq!(rgba.bytes[1], rgba.bytes[2]);
        assert_eq!(rgba.bytes[3], 255);
    }

    #[test]
    fn nv12_deinterleaves_to_separate_chroma_planes() {
        // Distinct U/V so a swapped deinterleave would be caught.
        let frame = solid_nv12(4, 4, 200, 90, 240);
        let (u, v) = deinterleave_nv12(&frame).unwrap();
        assert_eq!(u, vec![90; 4]);
        assert_eq!(v, vec![240; 4]);
    }

    #[test]
    fn nv12_yuv_layer_matches_equivalent_yuv420p() {
        // An NV12 frame and the YUV420P frame with the same samples must build
        // the identical GPU layer — hardware decode is transparent downstream.
        let nv12 = solid_nv12(4, 4, 128, 100, 150);
        let planar = solid_yuv420p(4, 4, 128, 100, 150);
        let from_nv12 = decoded_to_yuv_layer(&nv12).unwrap();
        let from_planar = decoded_to_yuv_layer(&planar).unwrap();
        assert_eq!(from_nv12.tight_y(), from_planar.tight_y());
        assert_eq!(from_nv12.tight_u(), from_planar.tight_u());
        assert_eq!(from_nv12.tight_v(), from_planar.tight_v());
    }

    #[test]
    fn nv12_rgba_fallback_matches_equivalent_yuv420p() {
        let nv12 = solid_nv12(2, 2, 128, 128, 128);
        let planar = solid_yuv420p(2, 2, 128, 128, 128);
        assert_eq!(
            decoded_to_rgba_inner(&nv12).unwrap(),
            decoded_to_rgba_inner(&planar).unwrap()
        );
    }

    #[test]
    fn full_range_flag_rides_into_the_gpu_layer() {
        let mut frame = solid_yuv420p(4, 4, 128, 128, 128);
        set_full_range(&mut frame, true);
        assert!(decoded_to_yuv_layer(&frame).unwrap().full_range);
        let limited = solid_yuv420p(4, 4, 128, 128, 128);
        assert!(!decoded_to_yuv_layer(&limited).unwrap().full_range);
    }

    /// At Y=235 the two ranges diverge: limited treats it as peak white (255),
    /// full range treats it as the literal level (~235). This is exactly the
    /// washed-out/clipped look we'd get if range were ignored.
    #[test]
    fn full_range_luma_skips_the_studio_rescale() {
        let mut full = solid_yuv420p(2, 2, 235, 128, 128);
        set_full_range(&mut full, true);
        let r_full = yuv420p_to_rgba(&full).unwrap().bytes[0];
        assert!(
            (231..=239).contains(&r_full),
            "full-range Y=235 should stay ~235, got {r_full}"
        );

        let limited = solid_yuv420p(2, 2, 235, 128, 128);
        let r_limited = yuv420p_to_rgba(&limited).unwrap().bytes[0];
        assert_eq!(r_limited, 255, "limited Y=235 is peak white");
    }
}
