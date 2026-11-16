use super::{FrameEncoder, FrameParams};
use crate::decode::Decoder;

/// A moving, textured test picture.
fn source(w: usize, h: usize, t: usize) -> [Vec<u8>; 3] {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let (x, yy) = ((c + 2 * t) as f64, (r + t) as f64);
            let v =
                128.0 + 60.0 * (x / 7.0).sin() * (yy / 11.0).cos() + 40.0 * ((x + yy) / 23.0).sin();
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
    let se: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum();
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn round_trip(w: usize, h: usize, frames: usize, qindex: u8) {
    round_trip_threads(w, h, frames, qindex, 4, 5);
}

fn round_trip_threads(w: usize, h: usize, frames: usize, qindex: u8, threads: usize, speed: u8) {
    round_trip_with(w, h, frames, qindex, threads, speed, false);
}

fn round_trip_with(
    w: usize,
    h: usize,
    frames: usize,
    qindex: u8,
    threads: usize,
    speed: u8,
    aq: bool,
) {
    let mut enc = FrameEncoder::new(w as u32, h as u32, threads);
    let mut dec = Decoder::new();
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut bytes = 0;
    for t in 0..frames {
        let src = source(w, h, t);
        enc.load([(&src[0], w), (&src[1], cw), (&src[2], cw)]);
        let coded = enc
            .encode(FrameParams {
                key: t == 0,
                golden: t % 3 == 2,
                hidden: false,
                qindex,
                speed,
                altref_future: false,
                aq,
                second_pass: false,
                full_search: false,
                fast_arf: false,
                uv_delta: 0,
                golden_slot: 1,
                also_refresh: 0,
                noise: 0.0,
            })
            .expect("encode");
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
                assert_eq!(
                    &got[r * pw..(r + 1) * pw],
                    &data[r * stride..r * stride + pw],
                    "frame {t} plane {p} row {r}"
                );
            }
        }
        let q = psnr(&src[0], &pic.y);
        assert!(q > 25.0 || qindex > 200, "frame {t}: psnr {q}");
        eprintln!(
            "{w}x{h} q{qindex} frame {t}: {} bytes, psnr {q:.2}",
            coded.data.len()
        );
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
fn round_trips_with_segments() {
    // Adaptive quantization: a quantizer by segment, the map coded per
    // block, on both search paths.
    round_trip_with(200, 136, 3, 120, 2, 5, true);
    round_trip_with(200, 136, 3, 120, 2, 9, true);
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

#[test]
fn round_trips_tile_columns() {
    // Four tile columns, on four threads, at both search paths.
    round_trip_threads(1100, 40, 3, 90, 4, 5);
    round_trip_threads(1100, 40, 3, 90, 4, 9);
    // Past 4096 samples two columns are the least, on one thread too.
    round_trip_threads(4200, 16, 2, 120, 1, 9);
}

#[test]
fn round_trips_every_speed() {
    for speed in [0, 3, 6] {
        round_trip_threads(136, 72, 3, 100, 2, speed);
    }
}

#[test]
fn row_lanes_code_the_frames_rows_one_after_another_do() {
    // Two tile columns of five superblocks and six superblock rows (the
    // last part of one), at speeds with rectangles, with the full search
    // and on the fast path.
    let (w, h) = (600usize, 328usize);
    let cw = w.div_ceil(2);
    for speed in [2, 5, 8] {
        let run = |lanes: usize| -> Vec<Vec<u8>> {
            let mut enc = FrameEncoder::new(w as u32, h as u32, 4);
            enc.set_row_lanes(lanes);
            (0..5)
                .map(|t| {
                    let src = source(w, h, t);
                    enc.load([(&src[0], w), (&src[1], cw), (&src[2], cw)]);
                    enc.encode(FrameParams {
                        key: t == 0,
                        golden: t == 3,
                        hidden: false,
                        qindex: 100,
                        speed,
                        altref_future: false,
                        aq: false,
                        second_pass: false,
                        full_search: false,
                        fast_arf: false,
                        uv_delta: 0,
                        golden_slot: 1,
                        also_refresh: 0,
                        noise: 0.0,
                    })
                    .expect("encode")
                    .data
                })
                .collect()
        };
        let one = run(1);
        for lanes in [2, 3] {
            assert!(run(lanes) == one, "speed {speed}, {lanes} lanes");
        }
    }
}

/// A detailed, non-repeating luma picture `w`×`h` cut from a larger one
/// at `(x, y)`, with flat chroma.
fn texture(w: usize, h: usize, (x, y): (usize, usize)) -> [Vec<u8>; 3] {
    let at = |c: usize, r: usize| -> u32 {
        let k = (((c + x) / 4) as u32).wrapping_mul(73_856_093)
            ^ (((r + y) / 4) as u32).wrapping_mul(19_349_663);
        40 + (k.wrapping_mul(2_654_435_761) >> 24) % 170
    };
    let mut luma = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            luma[r * w + c] = ((at(c, r) + at(c + 1, r)) / 2) as u8;
        }
    }
    let n = w.div_ceil(2) * h.div_ceil(2);
    [luma, vec![120; n], vec![136; n]]
}

#[test]
fn a_far_jump_is_found_and_searched_from() {
    let (w, h) = (640usize, 320usize);
    let a = texture(w, h, (200, 100));
    let b = texture(w, h, (100, 60));
    // `b`'s blocks are `a`'s 100 to the left and 40 up (on one thread
    // or several).
    for threads in [1, 3] {
        assert_eq!(
            super::arf::jump(&a[0], &b[0], w, h, threads),
            Some((-100, -40))
        );
    }
    // Another picture altogether: no jump.
    let other: Vec<u8> = (0..w * h)
        .map(|i| (30 + (i % w + i / w) / 8) as u8)
        .collect();
    assert_eq!(super::arf::jump(&a[0], &other, w, h, 1), None);
    // Coded after `a`, `b` takes far fewer bits with the jump given.
    let cw = w.div_ceil(2);
    let code = |global: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
        let mut sizes = Vec::new();
        for (t, pic) in [&a, &b].into_iter().enumerate() {
            if t == 1 && global {
                enc.set_global((-100, -40));
            }
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc
                .encode(FrameParams {
                    key: t == 0,
                    golden: false,
                    hidden: false,
                    qindex: 120,
                    speed: 6,
                    altref_future: false,
                    aq: false,
                    second_pass: false,
                    full_search: false,
                    fast_arf: false,
                    uv_delta: 0,
                    golden_slot: 1,
                    also_refresh: 0,
                    noise: 0.0,
                })
                .expect("encode");
            sizes.push(coded.data.len());
        }
        sizes[1]
    };
    let (with, without) = (code(true), code(false));
    assert!(
        with * 2 < without,
        "{with} bytes with the jump, {without} without"
    );
}
