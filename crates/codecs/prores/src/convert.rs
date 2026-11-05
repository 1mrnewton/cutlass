//! Wide 4:2:2 or 4:4:4 to the 8-bit 4:2:0 the app shows.

use cutlass_decode::{CodecError, PixelFormat, Plane, VideoFrame};

fn sample(plane: &Plane, x: usize, y: usize) -> u32 {
    let at = y * plane.stride + x * 2;
    match plane.samples.get(at..at + 2) {
        Some(pair) => u32::from(u16::from_le_bytes([pair[0], pair[1]])),
        None => 0,
    }
}

/// Packs `frame` as tight 8-bit 4:2:0, Y then U then V. Odd edges grow by
/// one repeated sample to the even size 4:2:0 needs. Returns the size and bytes.
pub fn to_yuv420p(frame: &VideoFrame) -> Result<(u32, u32, Vec<u8>), CodecError> {
    let (depth, horizontal) = match frame.format {
        PixelFormat::Yuv422p10 => (10, true),
        PixelFormat::Yuv422p12 => (12, true),
        PixelFormat::Yuv444p10 => (10, false),
        PixelFormat::Yuv444p12 => (12, false),
        _ => {
            return Err(CodecError::InvalidApi {
                what: "picture format",
            });
        }
    };
    let width = frame.width as usize;
    let height = frame.height as usize;
    let out_w = width.div_ceil(2) * 2;
    let out_h = height.div_ceil(2) * 2;
    let shift = depth - 8;
    let half = 1u32 << (shift - 1);
    let mut bytes = Vec::with_capacity(out_w * out_h * 3 / 2);
    for y in 0..out_h {
        let sy = y.min(height - 1);
        for x in 0..out_w {
            let value = (sample(&frame.y, x.min(width - 1), sy) + half) >> shift;
            bytes.push(value.min(255) as u8);
        }
    }
    let chroma_w = frame.u.width as usize;
    for plane in [&frame.u, &frame.v] {
        for cy in 0..out_h / 2 {
            let r0 = (2 * cy).min(height - 1);
            let r1 = (2 * cy + 1).min(height - 1);
            for cx in 0..out_w / 2 {
                let value = if horizontal {
                    let c = cx.min(chroma_w - 1);
                    (sample(plane, c, r0) + sample(plane, c, r1) + 2 * half) >> (shift + 1)
                } else {
                    let c0 = (2 * cx).min(width - 1);
                    let c1 = (2 * cx + 1).min(width - 1);
                    let sum = sample(plane, c0, r0)
                        + sample(plane, c1, r0)
                        + sample(plane, c0, r1)
                        + sample(plane, c1, r1);
                    (sum + 4 * half) >> (shift + 2)
                };
                bytes.push(value.min(255) as u8);
            }
        }
    }
    Ok((out_w as u32, out_h as u32, bytes))
}

#[cfg(test)]
mod tests {
    use super::to_yuv420p;
    use cutlass_decode::{PixelFormat, Plane, VideoFrame};

    fn plane(width: u32, height: u32, value: u16) -> Plane {
        let samples = (0..width * height)
            .flat_map(|_| value.to_le_bytes())
            .collect();
        Plane::bytes(width, height, 2, samples).expect("plane")
    }

    #[test]
    fn an_odd_422_frame_grows_to_even_420() {
        let frame = VideoFrame::planar(
            PixelFormat::Yuv422p10,
            3,
            3,
            plane(3, 3, 940),
            plane(2, 3, 512),
            plane(2, 3, 64),
        )
        .expect("frame");
        let (w, h, bytes) = to_yuv420p(&frame).expect("convert");
        assert_eq!((w, h), (4, 4));
        assert_eq!(bytes.len(), 16 + 4 + 4);
        assert!(bytes[..16].iter().all(|&b| b == 235));
        assert!(bytes[16..20].iter().all(|&b| b == 128));
        assert!(bytes[20..].iter().all(|&b| b == 16));
    }

    #[test]
    fn a_12_bit_444_frame_averages_chroma() {
        let frame = VideoFrame::planar(
            PixelFormat::Yuv444p12,
            2,
            2,
            plane(2, 2, 256),
            plane(2, 2, 2048),
            plane(2, 2, 4095),
        )
        .expect("frame");
        let (w, h, bytes) = to_yuv420p(&frame).expect("convert");
        assert_eq!((w, h), (2, 2));
        assert_eq!(bytes, vec![16, 16, 16, 16, 128, 255]);
    }
}
