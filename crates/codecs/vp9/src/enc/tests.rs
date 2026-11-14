use super::{FrameEncoder, FrameParams};
use crate::decode::Decoder;

/// A moving, textured test picture.
fn source(w: usize, h: usize, t: usize) -> [Vec<u8>; 3] {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let (x, yy) = ((c + 2 * t) as f64, (r + t) as f64);
            let v = 128.0 + 60.0 * (x / 7.0).sin() * (yy / 11.0).cos() + 40.0 * ((x + yy) / 23.0).sin();
            y[r * w + c] = v.clamp(0.0, 255.0) as u8;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = (128 + ((c + t) % 32) as i32 - 16) as u8;
            v[r * cw + c] = (128 + ((r * 3 + t) % 40) as i32 - 20) as u8;
        }
    }
    [y, u, v]
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let se: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum();
    let mse = se / a.len() as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

fn round_trip(w: usize, h: usize, frames: usize, qindex: u8) {
    let mut enc = FrameEncoder::new(w as u32, h as u32);
    let mut dec = Decoder::new();
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut bytes = 0;
    for t in 0..frames {
        let src = source(w, h, t);
        enc.load([(&src[0], w), (&src[1], cw), (&src[2], cw)]);
        let coded = enc.encode(FrameParams { key: t == 0, golden: t % 3 == 2, qindex, speed: 5 }).expect("encode");
        bytes += coded.data.len();
        let pic = dec.decode(&coded.data).expect("decode").expect("shown");
        // The decoder's picture is the encoder's reconstruction.
        let rec = enc.last().expect("recon");
        for p in 0..3 {
            let (pw, ph) = if p == 0 { (w, h) } else { (cw, ch) };
            let (data, stride) = rec.buf.plane(p);
            let got = match p {
                0 => &pic.y,
                1 => &pic.u,
                _ => &pic.v,
            };
            for r in 0..ph {
                assert_eq!(&got[r * pw..(r + 1) * pw], &data[r * stride..r * stride + pw], "frame {t} plane {p} row {r}");
            }
        }
        let q = psnr(&src[0], &pic.y);
        assert!(q > 25.0 || qindex > 200, "frame {t}: psnr {q}");
        eprintln!("{w}x{h} q{qindex} frame {t}: {} bytes, psnr {q:.2}", coded.data.len());
    }
    eprintln!("total {bytes}");
}

#[test]
fn round_trips_small() {
    round_trip(64, 64, 3, 60);
}

#[test]
fn round_trips_odd_sizes() {
    round_trip(100, 70, 4, 100);
    round_trip(8, 8, 2, 40);
    round_trip(200, 136, 3, 20);
}

#[test]
fn round_trips_lossless() {
    round_trip(72, 40, 3, 0);
}

#[test]
fn transform_gains() {
    for l in 0..4 {
        eprintln!("gain {l}: {}", super::tx::gain(l));
    }
}
