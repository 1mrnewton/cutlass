//! CPU RGBA → I420 / NV12 (YUV 4:2:0) conversion.
//!
//! The renderer hands the export loop packed RGBA, but hardware video encoders
//! take YUV. Apple's VideoToolbox accepts BGRA directly (see the Apple
//! backend); MediaCodec and Media Foundation do not, so the Android backend
//! converts to I420 and the Windows backend to NV12 (same 4:2:0 samples,
//! interleaved UV) here first. Kept platform-independent so it's
//! unit-testable on the host.
//!
//! Limited-range, with the matrix chosen by the caller ([`YuvMatrix`]): the
//! encoder must convert with the same coefficients decoders will use to
//! convert back, or every round-trip shifts colors. Chroma is 2×2
//! box-averaged.

/// RGB↔YUV conversion matrix (limited range). Encoder backends derive it
/// from the config's color tag ([`matrix_for`]) so the coefficients used to
/// build the stream match what it's tagged as — and match the **BT.709**
/// fallback decoders (including this workspace's own decode path) apply to
/// HD H.264 that doesn't signal otherwise. Mismatched matrices shift every
/// round-tripped color visibly (blues drift teal, skin drifts orange).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvMatrix {
    Bt601,
    Bt709,
}

/// The conversion matrix for an [`EncoderConfig`](cutlass_core::EncoderConfig)
/// color tag: an explicit BT.601 request keeps 601; everything else this SDR
/// encode path can be handed (BT.709, unspecified) converts as BT.709 — the
/// same fallback decoders apply, so round-trips can't shift colors.
#[cfg_attr(
    not(any(target_os = "windows", target_os = "android")),
    allow(dead_code)
)]
pub fn matrix_for(color: &cutlass_core::ColorSpace) -> YuvMatrix {
    match color.matrix {
        cutlass_core::MatrixCoefficients::Bt601 => YuvMatrix::Bt601,
        _ => YuvMatrix::Bt709,
    }
}

impl YuvMatrix {
    /// Fixed-point (/256) luma coefficients `[R, G, B]` for limited-range Y.
    fn luma(self) -> [i32; 3] {
        match self {
            // Y = 0.257R + 0.504G + 0.098B (+16).
            YuvMatrix::Bt601 => [66, 129, 25],
            // Y = 0.183R + 0.614G + 0.062B (+16).
            YuvMatrix::Bt709 => [47, 157, 16],
        }
    }

    /// Fixed-point (/256) chroma coefficients `([U; 3], [V; 3])`. Each row
    /// sums to zero so grays stay chroma-neutral exactly.
    fn chroma(self) -> ([i32; 3], [i32; 3]) {
        match self {
            // U = -0.148R - 0.291G + 0.439B; V = 0.439R - 0.368G - 0.071B.
            YuvMatrix::Bt601 => ([-38, -74, 112], [112, -94, -18]),
            // U = -0.115R - 0.385G + 0.500B; V = 0.500R - 0.454G - 0.046B
            // (scaled by 224/255).
            YuvMatrix::Bt709 => ([-26, -87, 113], [112, -102, -10]),
        }
    }
}

/// Convert a packed RGBA image to a tightly-packed I420 buffer (`Y` plane, then
/// `U`, then `V`), using `matrix` limited-range coefficients.
///
/// `stride` is the source row length in bytes (≥ `width * 4`). `width` and
/// `height` are rounded down to even for the 4:2:0 chroma grid. Returns a
/// buffer of `w*h + 2*(w/2)*(h/2)` bytes.
// The Android encoder is the only non-test caller; allow dead code on hosts
// where that backend is cfg'd out (the unit tests below still exercise it).
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub fn rgba_to_i420(
    rgba: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    matrix: YuvMatrix,
) -> Vec<u8> {
    let w = (width & !1) as usize;
    let h = (height & !1) as usize;
    let cw = w / 2;
    let ch = h / 2;
    let y_size = w * h;
    let c_size = cw * ch;
    let mut out = vec![0u8; y_size + 2 * c_size];

    let (y_plane, uv) = out.split_at_mut(y_size);
    let (u_plane, v_plane) = uv.split_at_mut(c_size);

    let [yr, yg, yb] = matrix.luma();
    let ([ur, ug, ub], [vr, vg, vb]) = matrix.chroma();

    let px = |x: usize, y: usize| -> (i32, i32, i32) {
        let base = y * stride + x * 4;
        (
            rgba[base] as i32,
            rgba[base + 1] as i32,
            rgba[base + 2] as i32,
        )
    };

    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, y);
            let luma = (yr * r + yg * g + yb * b + 128) >> 8;
            y_plane[y * w + x] = (luma + 16).clamp(0, 255) as u8;
        }
    }

    for cy in 0..ch {
        for cx in 0..cw {
            // Average the 2×2 RGBA block for one chroma sample.
            let (mut rs, mut gs, mut bs) = (0i32, 0i32, 0i32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (r, g, b) = px(cx * 2 + dx, cy * 2 + dy);
                    rs += r;
                    gs += g;
                    bs += b;
                }
            }
            let (r, g, b) = (rs / 4, gs / 4, bs / 4);
            let u = ((ur * r + ug * g + ub * b + 128) >> 8) + 128;
            let v = ((vr * r + vg * g + vb * b + 128) >> 8) + 128;
            u_plane[cy * cw + cx] = u.clamp(0, 255) as u8;
            v_plane[cy * cw + cx] = v.clamp(0, 255) as u8;
        }
    }

    out
}

/// Convert a packed RGBA image to a tightly-packed NV12 buffer (`Y` plane,
/// then interleaved `UV`) — the input format hardware H.264 encoder MFTs
/// accept universally on Windows.
///
/// Same sampling and coefficients as [`rgba_to_i420`]; only the chroma
/// layout differs. `stride` is the source row length in bytes
/// (≥ `width * 4`). `width` and `height` are rounded down to even for the
/// 4:2:0 chroma grid. Returns a buffer of `w*h + 2*(w/2)*(h/2)` bytes.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
pub fn rgba_to_nv12(
    rgba: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    matrix: YuvMatrix,
) -> Vec<u8> {
    let i420 = rgba_to_i420(rgba, width, height, stride, matrix);
    let w = (width & !1) as usize;
    let h = (height & !1) as usize;
    let y_size = w * h;
    let c_size = (w / 2) * (h / 2);

    let mut out = vec![0u8; y_size + 2 * c_size];
    out[..y_size].copy_from_slice(&i420[..y_size]);
    let (u_plane, v_plane) = i420[y_size..].split_at(c_size);
    let uv = &mut out[y_size..];
    for i in 0..c_size {
        uv[i * 2] = u_plane[i];
        uv[i * 2 + 1] = v_plane[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `width*height` packed RGBA image from a per-pixel closure.
    fn image(width: u32, height: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let mut buf = vec![0u8; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let base = ((y * width + x) * 4) as usize;
                buf[base..base + 4].copy_from_slice(&f(x, y));
            }
        }
        buf
    }

    #[test]
    fn plane_sizes_are_correct_for_420() {
        let rgba = image(4, 4, |_, _| [0, 0, 0, 255]);
        let out = rgba_to_i420(&rgba, 4, 4, 16, YuvMatrix::Bt601);
        // 4*4 luma + 2 * (2*2) chroma = 16 + 8 = 24.
        assert_eq!(out.len(), 24);
    }

    #[test]
    fn solid_black_and_white_map_to_expected_luma() {
        for matrix in [YuvMatrix::Bt601, YuvMatrix::Bt709] {
            let black = image(2, 2, |_, _| [0, 0, 0, 255]);
            let white = image(2, 2, |_, _| [255, 255, 255, 255]);
            let yb = rgba_to_i420(&black, 2, 2, 8, matrix);
            let yw = rgba_to_i420(&white, 2, 2, 8, matrix);
            // Limited-range luma: black ≈ 16, white ≈ 235, in either matrix.
            assert_eq!(yb[0], 16, "{matrix:?} black");
            assert!(
                (233..=235).contains(&yw[0]),
                "{matrix:?} white luma was {}",
                yw[0]
            );
        }
    }

    #[test]
    fn gray_is_chroma_neutral() {
        for matrix in [YuvMatrix::Bt601, YuvMatrix::Bt709] {
            let gray = image(2, 2, |_, _| [128, 128, 128, 255]);
            let out = rgba_to_i420(&gray, 2, 2, 8, matrix);
            // One U and one V sample, both ≈ 128 (neutral chroma) for gray.
            let u = out[4];
            let v = out[5];
            assert!((127..=129).contains(&u), "{matrix:?} U was {u}");
            assert!((127..=129).contains(&v), "{matrix:?} V was {v}");
        }
    }

    #[test]
    fn matrices_produce_distinct_chroma_for_saturated_color() {
        // Pure blue: BT.601 puts more of it in luma than BT.709 does, so the
        // planes must differ — guards against the matrix silently not being
        // applied.
        let blue = image(2, 2, |_, _| [0, 0, 255, 255]);
        let out_601 = rgba_to_i420(&blue, 2, 2, 8, YuvMatrix::Bt601);
        let out_709 = rgba_to_i420(&blue, 2, 2, 8, YuvMatrix::Bt709);
        assert!(out_601[0] > out_709[0], "601 blue luma should exceed 709");
        assert_ne!(out_601, out_709);
    }

    #[test]
    fn bt709_roundtrips_the_mobile_fixture_blue() {
        // The workspace's decode paths convert untagged HD H.264 back with
        // BT.709; encoding [40, 80, 160] with BT.709 must land within the
        // decoder tests' per-channel tolerance after the inverse transform.
        let rgba = image(2, 2, |_, _| [40, 80, 160, 255]);
        let out = rgba_to_i420(&rgba, 2, 2, 8, YuvMatrix::Bt709);
        let (y, u, v) = (
            f64::from(out[0]) - 16.0,
            f64::from(out[4]) - 128.0,
            f64::from(out[5]) - 128.0,
        );
        // Inverse BT.709 limited-range.
        let r = 1.164 * y + 1.793 * v;
        let g = 1.164 * y - 0.213 * u - 0.533 * v;
        let b = 1.164 * y + 2.112 * u;
        assert!((r - 40.0).abs() < 3.0, "R came back as {r}");
        assert!((g - 80.0).abs() < 3.0, "G came back as {g}");
        assert!((b - 160.0).abs() < 3.0, "B came back as {b}");
    }

    #[test]
    fn matrix_for_maps_explicit_601_and_defaults_to_709() {
        use cutlass_core::ColorSpace;
        assert_eq!(matrix_for(&ColorSpace::BT601), YuvMatrix::Bt601);
        assert_eq!(matrix_for(&ColorSpace::BT709), YuvMatrix::Bt709);
        // Unspecified/odd tags fall back to 709, matching decoder behavior.
        assert_eq!(matrix_for(&ColorSpace::SRGB), YuvMatrix::Bt709);
    }

    #[test]
    fn nv12_matches_i420_samples_with_interleaved_chroma() {
        // A red/blue split so U and V differ per sample.
        let rgba = image(4, 4, |x, _| {
            if x < 2 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        });
        let i420 = rgba_to_i420(&rgba, 4, 4, 16, YuvMatrix::Bt601);
        let nv12 = rgba_to_nv12(&rgba, 4, 4, 16, YuvMatrix::Bt601);
        assert_eq!(nv12.len(), i420.len());
        // Identical luma plane.
        assert_eq!(nv12[..16], i420[..16]);
        // Chroma: NV12 interleaves the I420 U and V planes pairwise.
        let (u, v) = i420[16..].split_at(4);
        for i in 0..4 {
            assert_eq!(nv12[16 + i * 2], u[i], "U sample {i}");
            assert_eq!(nv12[16 + i * 2 + 1], v[i], "V sample {i}");
        }
    }

    #[test]
    fn honors_source_stride() {
        // 2×2 image in a buffer padded to a 12-byte stride; padding must be
        // ignored.
        let width = 2;
        let height = 2;
        let stride = 12;
        let mut rgba = vec![7u8; stride * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let base = y * stride + x * 4;
                rgba[base..base + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let out = rgba_to_i420(&rgba, width, height, stride, YuvMatrix::Bt601);
        assert!((233..=235).contains(&out[0]));
    }
}
