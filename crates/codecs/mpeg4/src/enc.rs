//! A Simple-profile intra encoder.
//!
//! One I-VOP, H.263 quantisation, DC prediction, no AC prediction.
//! Predicted pictures, quarter-sample motion, B-VOPs and rate control are
//! not emitted here yet. The coded size is a whole number of macroblocks;
//! a caller with another size is refused rather than cropped.

use std::sync::OnceLock;

use cutlass_decode::CodecError;

use crate::bits::BitWriter;
use crate::tables::{
    CBPY, DC_CHROMA, DC_LUMA, MCBPC_INTRA, TCOEF_ESCAPE, dc_scaler_chroma, dc_scaler_luma, tables,
};

const TIME_RESOLUTION: u32 = 25;

/// One intra picture as an elementary stream: the video object layer,
/// then one I-VOP. `quant` is `vop_quant`, 1 to 31.
pub fn encode_intra(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    quant: u8,
) -> Result<Vec<u8>, CodecError> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(16)
        || !height.is_multiple_of(16)
        || u64::from(width) * u64::from(height) > 8192 * 8192
    {
        return Err(CodecError::InvalidApi { what: "frame size" });
    }
    if !(1..=31).contains(&quant) {
        return Err(CodecError::InvalidApi { what: "quantiser" });
    }
    let (w, h) = (width as usize, height as usize);
    let chroma = (w / 2) * (h / 2);
    if y.len() < w * h || u.len() < chroma || v.len() < chroma {
        return Err(CodecError::InvalidApi {
            what: "plane length",
        });
    }
    let mut out = BitWriter::new();
    start(&mut out, 0x20);
    write_vol(&mut out, width, height);
    start(&mut out, 0xB6);
    write_i_vop(&mut out, y, u, v, w, h, i32::from(quant));
    Ok(out.finish())
}

fn start(out: &mut BitWriter, code: u8) {
    out.bytes(&[0, 0, 1, code]);
}

/// Simple object, 4:2:0, square pixels, low delay, H.263 quantiser.
fn write_vol(out: &mut BitWriter, width: u32, height: u32) {
    out.put(1, 0); // random_accessible_vol
    out.put(8, 1); // simple
    out.put(1, 0); // is_object_layer_identifier
    out.put(4, 1); // square
    out.put(1, 1); // vol_control_parameters
    out.put(2, 1); // 4:2:0
    out.put(1, 1); // low_delay
    out.put(1, 0); // no vbv
    out.put(2, 0); // rectangular
    out.put(1, 1);
    out.put(16, TIME_RESOLUTION);
    out.put(1, 1);
    out.put(1, 0); // fixed_vop_rate
    out.put(1, 1);
    out.put(13, width);
    out.put(1, 1);
    out.put(13, height);
    out.put(1, 1);
    out.put(1, 0); // progressive
    out.put(1, 1); // obmc_disable
    out.put(1, 0); // no sprite
    out.put(1, 0); // not_8_bit
    out.put(1, 0); // h.263 quantiser
    out.put(1, 1); // complexity_estimation_disable
    out.put(1, 1); // resync_marker_disable
    out.put(1, 0); // data_partitioned
    out.put(1, 0); // scalability
    out.stuff();
}

fn write_i_vop(
    out: &mut BitWriter,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    quant: i32,
) {
    let time_bits = (32 - (TIME_RESOLUTION - 1).leading_zeros()).max(1);
    out.put(2, 0); // I
    out.put(1, 0); // modulo_time_base
    out.put(1, 1);
    out.put(time_bits, 0);
    out.put(1, 1);
    out.put(1, 1); // coded
    out.put(3, 0); // intra_dc_vlc_thr: DC stays in the DC VLC
    out.put(5, quant as u32);
    let mbw = width / 16;
    let mbh = height / 16;
    let mut dc = [
        vec![1024i32; mbw * 2 * mbh * 2],
        vec![1024i32; mbw * mbh],
        vec![1024i32; mbw * mbh],
    ];
    for mby in 0..mbh {
        for mbx in 0..mbw {
            write_intra_mb(
                out,
                &Planes {
                    y,
                    u,
                    v,
                    stride: width,
                },
                mbx,
                mby,
                quant,
                &mut dc,
            );
        }
    }
    out.stuff();
}

struct Planes<'a> {
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    stride: usize,
}

fn write_intra_mb(
    out: &mut BitWriter,
    planes: &Planes<'_>,
    mbx: usize,
    mby: usize,
    quant: i32,
    dc: &mut [Vec<i32>; 3],
) {
    let mut coded = [false; 6];
    let mut levels = [[0i32; 64]; 6];
    let stride = planes.stride;
    for k in 0..6 {
        let (plane, x, y, step) = if k < 4 {
            (
                planes.y,
                mbx * 16 + (k & 1) * 8,
                mby * 16 + (k >> 1) * 8,
                stride,
            )
        } else if k == 4 {
            (planes.u, mbx * 8, mby * 8, stride / 2)
        } else {
            (planes.v, mbx * 8, mby * 8, stride / 2)
        };
        let coef = forward(&sample(plane, step, x, y));
        let across = if k < 4 { stride / 8 } else { stride / 16 };
        let (bx, by) = if k < 4 {
            ((2 * mbx + (k & 1)) as isize, (2 * mby + (k >> 1)) as isize)
        } else {
            (mbx as isize, mby as isize)
        };
        let scale = if k < 4 {
            dc_scaler_luma(quant as u32)
        } else {
            dc_scaler_chroma(quant as u32)
        };
        let predicted = predict(&dc[plane_of(k)], across, bx, by);
        let qf = div_round(coef[0], scale).clamp(0, 2047 / scale);
        let diff = qf - (predicted + (scale >> 1)) / scale;
        dc[plane_of(k)][by as usize * across + bx as usize] = qf * scale;
        levels[k][0] = diff;
        for i in 1..64 {
            levels[k][i] = quant_ac(coef[i], quant);
            coded[k] |= levels[k][i] != 0;
        }
    }
    let cbpc = u32::from(coded[4]) << 1 | u32::from(coded[5]);
    let (len, code) = MCBPC_INTRA[cbpc as usize];
    out.put(u32::from(len), u32::from(code));
    out.bit(false); // ac_pred
    let cbpy = u32::from(coded[0]) << 3
        | u32::from(coded[1]) << 2
        | u32::from(coded[2]) << 1
        | u32::from(coded[3]);
    let (len, code) = CBPY[cbpy as usize];
    out.put(u32::from(len), u32::from(code));
    for k in 0..6 {
        write_dc(out, levels[k][0], k >= 4);
        if coded[k] {
            write_ac(out, &levels[k]);
        }
    }
}

fn plane_of(k: usize) -> usize {
    if k < 4 { 0 } else { k - 3 }
}

/// Section 7.4.3.1. A missing neighbour predicts 1024, mid-grey.
fn predict(dc: &[i32], across: usize, bx: isize, by: isize) -> i32 {
    let get = |dx: isize, dy: isize| -> i32 {
        let x = bx + dx;
        let y = by + dy;
        if x < 0 || y < 0 || across == 0 || x as usize >= across {
            return 1024;
        }
        dc.get(y as usize * across + x as usize)
            .copied()
            .unwrap_or(1024)
    };
    let (left, corner, above) = (get(-1, 0), get(-1, -1), get(0, -1));
    if (left - corner).abs() < (corner - above).abs() {
        above
    } else {
        left
    }
}

fn write_dc(out: &mut BitWriter, diff: i32, chroma: bool) {
    let size = if diff == 0 {
        0
    } else {
        32 - diff.unsigned_abs().leading_zeros()
    };
    let size = size.min(12);
    let (len, code) = if chroma {
        DC_CHROMA[size as usize]
    } else {
        DC_LUMA[size as usize]
    };
    out.put(u32::from(len), u32::from(code));
    if size == 0 {
        return;
    }
    // A positive differential is itself. A negative one is
    // `diff + 2^size - 1`, whose top bit is clear.
    let bits = if diff > 0 {
        diff as u32
    } else {
        (diff + (1 << size) - 1) as u32
    };
    out.put(size, bits);
    if size > 8 {
        out.bit(true);
    }
}

fn write_ac(out: &mut BitWriter, levels: &[i32; 64]) {
    let mut coded = Vec::new();
    for pos in 1..64 {
        let level = levels[usize::from(crate::tables::ZIGZAG[pos])];
        if level != 0 {
            coded.push((pos, level));
        }
    }
    let mut cursor = 1usize;
    for (n, &(pos, level)) in coded.iter().enumerate() {
        let run = pos - cursor;
        write_coef(out, n + 1 == coded.len(), run, level);
        cursor = pos + 1;
    }
}

fn write_coef(out: &mut BitWriter, last: bool, run: usize, level: i32) {
    let mag = level.unsigned_abs();
    if let Some((len, code)) = tables().intra.code(last, run, mag) {
        out.put(u32::from(len), u32::from(code));
        out.bit(level < 0);
        return;
    }
    let (elen, ecode) = TCOEF_ESCAPE;
    out.put(u32::from(elen), u32::from(ecode));
    out.bit(true);
    out.bit(true);
    out.bit(last);
    out.put(6, run as u32);
    out.bit(true);
    out.put_signed(12, level);
    out.bit(true);
}

/// H.263 intra AC level whose reconstruction is nearest `coef`.
fn quant_ac(coef: i32, quant: i32) -> i32 {
    if coef == 0 {
        return 0;
    }
    let mul = 2 * quant;
    let add = (quant - 1) | 1;
    let sign = coef.signum();
    let mag = coef.unsigned_abs() as i32;
    if mag * 2 <= mul + add {
        return 0;
    }
    let level = ((mag - add + mul / 2) / mul).clamp(1, 2047);
    sign * level
}

fn div_round(value: i32, divisor: i32) -> i32 {
    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        (value - divisor / 2) / divisor
    }
}

fn sample(plane: &[u8], stride: usize, x: usize, y: usize) -> [i32; 64] {
    let mut out = [0; 64];
    for row in 0..8 {
        for col in 0..8 {
            out[row * 8 + col] = i32::from(plane[(y + row) * stride + x + col]);
        }
    }
    out
}

/// The forward transform paired with [`crate::idct`]: coefficients are
/// the analysis of the same cosine basis the inverse uses.
fn forward(pixels: &[i32; 64]) -> [i32; 64] {
    let b = basis();
    let mut out = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f64;
            for y in 0..8 {
                for x in 0..8 {
                    sum += f64::from(pixels[y * 8 + x]) * b[u][x] * b[v][y];
                }
            }
            out[v * 8 + u] = nearest(sum);
        }
    }
    out
}

fn basis() -> &'static [[f64; 8]; 8] {
    static BASIS: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
    BASIS.get_or_init(|| {
        let mut rows = [[0.0f64; 8]; 8];
        for (k, row) in rows.iter_mut().enumerate() {
            let scale = if k == 0 {
                0.5 / std::f64::consts::SQRT_2
            } else {
                0.5
            };
            for (n, value) in row.iter_mut().enumerate() {
                let angle = (2.0 * n as f64 + 1.0) * k as f64 * std::f64::consts::PI / 16.0;
                *value = scale * angle.cos();
            }
        }
        rows
    })
}

fn nearest(value: f64) -> i32 {
    let rounded = value.round();
    if rounded >= f64::from(i32::MAX) {
        i32::MAX
    } else if rounded <= f64::from(i32::MIN) {
        i32::MIN
    } else {
        rounded as i32
    }
}

#[cfg(test)]
mod tests {
    use super::encode_intra;
    use crate::dec::Decoder;
    use cutlass_decode::CodecError;

    fn decode(bytes: &[u8]) -> Vec<u8> {
        let mut decoder = Decoder::with_max_pixels(8192 * 8192);
        let mut pictures = decoder.decode(bytes).expect("decode").pictures;
        pictures.extend(decoder.drain());
        assert_eq!(pictures.len(), 1);
        pictures.remove(0).bytes
    }

    fn mae(got: &[u8], want: &[u8]) -> f64 {
        let n = got.len().min(want.len()).max(1);
        let sum: u64 = got
            .iter()
            .zip(want)
            .map(|(&a, &b)| u64::from(a.abs_diff(b)))
            .sum();
        sum as f64 / n as f64
    }

    #[test]
    fn a_flat_intra_picture_comes_back() {
        let y = vec![140u8; 48 * 32];
        let c = vec![120u8; 24 * 16];
        let bytes = encode_intra(&y, &c, &c, 48, 32, 4).expect("encode");
        let got = decode(&bytes);
        assert!(
            mae(&got[..48 * 32], &y) < 1.5,
            "{}",
            mae(&got[..48 * 32], &y)
        );
        assert!(
            mae(&got[48 * 32..], &{
                let mut chroma = c.clone();
                chroma.extend_from_slice(&c);
                chroma
            }) < 2.0
        );
    }

    #[test]
    fn a_ramp_intra_picture_stays_close() {
        let mut y = vec![0u8; 64 * 48];
        for row in 0..48 {
            for col in 0..64 {
                y[row * 64 + col] = (col * 3 + row) as u8;
            }
        }
        let mut u = vec![0u8; 32 * 24];
        let mut v = vec![0u8; 32 * 24];
        for (i, (cu, cv)) in u.iter_mut().zip(&mut v).enumerate() {
            *cu = 80 + (i % 40) as u8;
            *cv = 160;
        }
        let bytes = encode_intra(&y, &u, &v, 64, 48, 4).expect("encode");
        let got = decode(&bytes);
        assert!(
            mae(&got[..64 * 48], &y) < 6.0,
            "{}",
            mae(&got[..64 * 48], &y)
        );
    }

    #[test]
    fn ffmpeg_decodes_the_same_intra_picture() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!("ffmpeg is not installed; the intra stream was not checked independently");
            return;
        };
        let mut y = vec![0u8; 48 * 32];
        for (i, sample) in y.iter_mut().enumerate() {
            *sample = (40 + (i * 5) % 180) as u8;
        }
        let u = vec![90u8; 24 * 16];
        let v = vec![150u8; 24 * 16];
        let bytes = encode_intra(&y, &u, &v, 48, 32, 6).expect("encode");
        let ours = decode(&bytes);
        let dir = std::env::temp_dir().join(format!("cutlass-mpeg4-intra-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let stream = dir.join("intra.m4v");
        let yuv = dir.join("intra.yuv");
        std::fs::write(&stream, &bytes).expect("stream");
        let status = std::process::Command::new(&ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-cpuflags",
                "0",
                "-flags",
                "+bitexact",
                "-idct",
                "simple",
                "-i",
            ])
            .arg(&stream)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&yuv)
            .status()
            .expect("spawn");
        assert!(status.success(), "ffmpeg decode");
        let theirs = std::fs::read(&yuv).expect("yuv");
        assert_eq!(ours, theirs);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn ffmpeg_bin() -> Option<std::path::PathBuf> {
        [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ]
        .into_iter()
        .map(std::path::PathBuf::from)
        .find(|path| {
            std::process::Command::new(path)
                .arg("-version")
                .output()
                .is_ok_and(|out| out.status.success())
        })
    }

    #[test]
    fn a_partial_macroblock_size_is_refused() {
        let y = vec![0u8; 16 * 8];
        let c = vec![0u8; 8 * 4];
        let refused = encode_intra(&y, &c, &c, 16, 8, 4);
        assert!(matches!(refused, Err(CodecError::InvalidApi { .. })));
        assert!(encode_intra(&y, &c, &c, 16, 16, 0).is_err());
    }
}
