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
                altref_slot: 2,
                last_slot: 0,
                hidden_slot: 2,
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
                        altref_slot: 2,
                        last_slot: 0,
                        hidden_slot: 2,
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
                    altref_slot: 2,
                    last_slot: 0,
                    hidden_slot: 2,
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

/// Codes four pictures of a moving texture with `bd`-bit samples and chroma `ss` subsampled, at
/// speed 5 and quantizer `qindex`: each decodes as the encoder
/// reconstructed it. Returns how many took a loop filter level other than
/// the quantizer's.
fn searched_levels<P: super::EncPixel>(
    ss: (usize, usize),
    bd: u8,
    threads: usize,
    qindex: u8,
) -> usize {
    let (w, h) = (520usize, 136usize);
    let mut enc = FrameEncoder::<P>::with_depth(w as u32, h as u32, threads, ss, bd);
    let mut dec = Decoder::new();
    let (cw, ch) = (w.div_ceil(1 << ss.0), h.div_ceil(1 << ss.1));
    let max = (1i32 << bd) - 1;
    let mut searched = 0;
    for t in 0..4 {
        let plane = |pw: usize, ph: usize, k: usize| -> Vec<P> {
            (0..pw * ph)
                .map(|i| {
                    let (x, y) = ((i % pw + 2 * t + 9 * k) as f64, (i / pw + t) as f64);
                    let v = 128.0
                        + 60.0 * (x / 7.0).sin() * (y / 11.0).cos()
                        + 40.0 * ((x + y) / 23.0).sin();
                    P::new((v as i32).clamp(0, 255) * max / 255)
                })
                .collect()
        };
        let src = [plane(w, h, 0), plane(cw, ch, 1), plane(cw, ch, 2)];
        enc.load([(&src[0], w), (&src[1], cw), (&src[2], cw)]);
        let coded = enc
            .encode(FrameParams {
                key: t == 0,
                golden: false,
                hidden: false,
                qindex,
                speed: 5,
                altref_future: false,
                aq: false,
                second_pass: false,
                full_search: true,
                fast_arf: false,
                uv_delta: 0,
                golden_slot: 1,
                also_refresh: 0,
                altref_slot: 2,
                last_slot: 0,
                hidden_slot: 2,
                noise: 0.0,
            })
            .expect("encode");
        let head = crate::header::parse(dec.persist(), &coded.data)
            .expect("header")
            .header;
        if head.filter_level != super::filter_level(qindex, t == 0, bd) {
            searched += 1;
        }
        let pic = dec.decode(&coded.data).expect("decode").expect("shown");
        let rec = enc.last().expect("recon");
        for p in 0..3 {
            let (pw, ph) = if p == 0 { (w, h) } else { (cw, ch) };
            let (data, stride) = rec.buf.plane(p);
            let bytes = match p {
                0 => &pic.y,
                1 => &pic.u,
                _ => &pic.v,
            };
            let got: Vec<i32> = if bd > 8 {
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| i32::from(u16::from_le_bytes(b)))
                    .collect()
            } else {
                bytes.iter().map(|&b| i32::from(b)).collect()
            };
            for r in 0..ph {
                let want: Vec<i32> = data[r * stride..r * stride + pw]
                    .iter()
                    .map(|v| v.get())
                    .collect();
                assert_eq!(
                    &got[r * pw..(r + 1) * pw],
                    &want[..],
                    "{bd}-bit {ss:?} q{qindex} picture {t} plane {p} row {r}"
                );
            }
        }
    }
    eprintln!("{bd}-bit {ss:?}, {threads} threads, q{qindex}: {searched} of 4 searched");
    searched
}

#[test]
fn a_searched_filter_level_decodes_as_the_encoder_filtered() {
    // 4:2:0 and 4:4:4 (chroma in the search), 4:2:2 and 4:4:0 (luma alone,
    // their chroma filtered from the grid at the level searched), 8 and 10
    // bits, on one tile column and on two (520 samples wide).
    for q in [60, 100] {
        assert!(searched_levels::<u8>((1, 1), 8, 1, q) > 0);
        assert!(searched_levels::<u8>((1, 1), 8, 2, q) > 0);
        assert!(searched_levels::<u16>((1, 1), 10, 2, q) > 0);
        assert!(searched_levels::<u8>((0, 0), 8, 2, q) > 0);
        assert!(searched_levels::<u16>((1, 0), 10, 1, q) > 0);
        assert!(searched_levels::<u8>((0, 1), 8, 1, q) > 0);
    }
}

#[test]
fn a_reference_seed_reaches_past_the_search_range() {
    use crate::dsp::pixel::Pixel;
    // `b`'s blocks are `a`'s 100 to the left and 40 up: past the 64
    // samples the search reaches from its neighbours' vectors.
    let (w, h) = (640usize, 320usize);
    let a = texture(w, h, (200, 100));
    let b = texture(w, h, (100, 60));
    let cw = w.div_ceil(2);
    let code = |speed: u8, seed: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
        let mut dec = Decoder::new();
        let mut out = Vec::new();
        for (t, pic) in [&a, &b].into_iter().enumerate() {
            if t == 1 && seed {
                enc.set_ref_seed(0, (-100, -40));
            }
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc
                .encode(FrameParams {
                    key: t == 0,
                    golden: false,
                    hidden: false,
                    qindex: 120,
                    speed,
                    altref_future: false,
                    aq: false,
                    second_pass: false,
                    full_search: speed <= 5,
                    fast_arf: false,
                    uv_delta: 0,
                    golden_slot: 1,
                    also_refresh: 0,
                    altref_slot: 2,
                    last_slot: 0,
                    hidden_slot: 2,
                    noise: 0.0,
                })
                .expect("encode");
            // Each picture decodes as the encoder reconstructed it.
            let shown = dec.decode(&coded.data).expect("decode").expect("shown");
            let rec = enc.last().expect("recon");
            let (data, stride) = rec.buf.plane(0);
            for r in 0..h {
                let want: Vec<u8> = data[r * stride..r * stride + w]
                    .iter()
                    .map(|v| v.get() as u8)
                    .collect();
                assert_eq!(
                    &shown.y[r * w..(r + 1) * w],
                    &want[..],
                    "speed {speed} row {r}"
                );
            }
            out.push(coded.data);
        }
        out
    };
    for speed in [5, super::tile::REF_SEED_SPEED] {
        let (with, without) = (code(speed, true), code(speed, false));
        assert!(
            with[1].len() * 2 < without[1].len(),
            "speed {speed}: {} bytes with the seed, {} without",
            with[1].len(),
            without[1].len()
        );
    }
    // Past REF_SEED_SPEED the seed is not read.
    let speed = super::tile::REF_SEED_SPEED + 1;
    assert!(code(speed, true) == code(speed, false));
}

/// `pic` with each band of 8 rows moved across: `d` samples, `-d`, then
/// twice, three and four times as far in the next 16 rows.
fn banded(pic: &[Vec<u8>; 3], w: usize, h: usize, d: isize) -> [Vec<u8>; 3] {
    let mut y = pic[0].clone();
    for r in 0..h {
        let s = if (r / 8) % 2 == 0 { d } else { -d } * ((r / 16) % 4 + 1) as isize;
        for c in 0..w {
            let from = (c as isize - s).clamp(0, w as isize - 1) as usize;
            y[r * w + c] = pic[0][r * w + from];
        }
    }
    [y, pic[1].clone(), pic[2].clone()]
}

#[test]
fn halves_after_a_split_take_bands_that_move_apart() {
    use crate::dsp::pixel::Pixel;
    // Bands of 8 rows moving one way and the other: each 16×16 block's
    // quarters pair up across, and its halves across code it in two
    // blocks rather than four (`tile::RECT_SPEED`).
    let (w, h) = (320usize, 192usize);
    let a = texture(w, h, (50, 30));
    let b = banded(&a, w, h, 3);
    let cw = w.div_ceil(2);
    let code = |speed: u8, full_search: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
        let mut dec = Decoder::new();
        let mut out = Vec::new();
        for (t, pic) in [&a, &b].into_iter().enumerate() {
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc
                .encode(FrameParams {
                    key: t == 0,
                    golden: false,
                    hidden: false,
                    qindex: 100,
                    speed,
                    altref_future: false,
                    aq: false,
                    second_pass: false,
                    full_search,
                    fast_arf: false,
                    uv_delta: 0,
                    golden_slot: 1,
                    also_refresh: 0,
                    altref_slot: 2,
                    last_slot: 0,
                    hidden_slot: 2,
                    noise: 0.0,
                })
                .expect("encode");
            // Each picture decodes as the encoder reconstructed it.
            let shown = dec.decode(&coded.data).expect("decode").expect("shown");
            let rec = enc.last().expect("recon");
            let (data, stride) = rec.buf.plane(0);
            for r in 0..h {
                let want: Vec<u8> = data[r * stride..r * stride + w]
                    .iter()
                    .map(|v| v.get() as u8)
                    .collect();
                assert_eq!(
                    &shown.y[r * w..(r + 1) * w],
                    &want[..],
                    "speed {speed} row {r}"
                );
            }
            out.push(coded.data);
        }
        out[1].len()
    };
    // As small as speed 3's, which tries halves at every size (without
    // them, 9% and 12% over it).
    let three = code(3, true);
    for speed in super::tile::RECT_SPEED {
        let n = code(speed, true);
        assert!(
            n * 20 < three * 21,
            "speed {speed}: {n} bytes, speed 3 {three}"
        );
    }
}

/// Frame parameters for the tests below: picture `t` of a stream at a fixed
/// quantizer (`second_pass` for a second pass's).
fn fixed_params(t: usize, qindex: u8, speed: u8, second_pass: bool) -> FrameParams {
    FrameParams {
        key: t == 0,
        golden: false,
        hidden: false,
        qindex,
        speed,
        altref_future: false,
        aq: false,
        second_pass,
        full_search: speed <= 5,
        fast_arf: false,
        uv_delta: 0,
        golden_slot: 1,
        also_refresh: 0,
        altref_slot: 2,
        last_slot: 0,
        hidden_slot: 2,
        noise: 0.0,
    }
}

/// Codes `pics` (4:2:0, `w`×`h`) as a key frame and inter frames, each
/// decoding as the encoder reconstructed it; returns the packets and the
/// last reconstruction's luma.
fn code_checked(
    pics: &[[Vec<u8>; 3]],
    (w, h): (usize, usize),
    qindex: u8,
    speed: u8,
    second_pass: bool,
) -> (Vec<Vec<u8>>, Vec<u8>) {
    use crate::dsp::pixel::Pixel;
    let cw = w.div_ceil(2);
    let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
    let mut dec = Decoder::new();
    let mut out = Vec::new();
    let mut last = Vec::new();
    for (t, pic) in pics.iter().enumerate() {
        enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
        let coded = enc
            .encode(fixed_params(t, qindex, speed, second_pass))
            .expect("encode");
        let shown = dec.decode(&coded.data).expect("decode").expect("shown");
        let rec = enc.last().expect("recon");
        let (data, stride) = rec.buf.plane(0);
        last = (0..h)
            .flat_map(|r| {
                data[r * stride..r * stride + w]
                    .iter()
                    .map(|v| v.get() as u8)
            })
            .collect();
        assert_eq!(shown.y, last, "speed {speed} picture {t}");
        out.push(coded.data);
    }
    (out, last)
}

#[test]
fn a_small_object_in_a_block_that_predicts_whole_is_coded() {
    // The second picture is the first (flat) with a 2×2 square brighter,
    // inside a 64×64 block that otherwise keeps the first picture exactly:
    // summed over the block its error was under the stop of the partition
    // search, and the square went uncoded (its error 9216, as dropped). At
    // a fixed quantizer the stop reads the worst 8×8 too (`Tile::peak_sse`).
    let (w, h) = (128usize, 128usize);
    let a = [
        vec![100u8; w * h],
        vec![128; w * h / 4],
        vec![128; w * h / 4],
    ];
    let mut b = a.clone();
    for r in 36..38 {
        for c in 20..22 {
            b[0][r * w + c] += 48;
        }
    }
    let err = |rec: &[u8]| -> u64 {
        let mut e = 0;
        for r in 36..38 {
            for c in 20..22 {
                let d = i64::from(rec[r * w + c]) - i64::from(b[0][r * w + c]);
                e += (d * d) as u64;
            }
        }
        e
    };
    let dropped = err(&a[0]);
    let (_, rec) = code_checked(&[a.clone(), b.clone()], (w, h), 120, 5, false);
    let coded = err(&rec);
    assert!(
        coded * 4 < dropped,
        "the square's error {coded}, uncoded {dropped}"
    );
}

#[test]
fn a_fixed_quantizer_scales_the_inter_frames_searched_level() {
    // At a fixed quantizer in one pass, an inter frame takes
    // `LF_FIXED_SCALE` of the level the search finds; a key frame, and a
    // second pass's frames, take the level found.
    let (w, h) = (256usize, 128usize);
    let pics: Vec<[Vec<u8>; 3]> = (0..3)
        .map(|t| texture(w, h, (10 + 3 * t, 20 + t)))
        .collect();
    let cw = w.div_ceil(2);
    for second_pass in [false, true] {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 1);
        let mut dec = Decoder::new();
        let mut scaled = 0;
        for (t, pic) in pics.iter().enumerate() {
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc
                .encode(fixed_params(t, 100, 5, second_pass))
                .expect("encode");
            let head = crate::header::parse(dec.persist(), &coded.data)
                .expect("header")
                .header;
            dec.decode(&coded.data).expect("decode").expect("shown");
            let found = enc.lf_found.expect("searched");
            let want = if t > 0 && !second_pass {
                ((f64::from(found) * super::LF_FIXED_SCALE) as u8).max(2)
            } else {
                found
            };
            assert_eq!(
                head.filter_level, want,
                "picture {t}, second pass {second_pass}"
            );
            scaled += usize::from(want != found);
        }
        assert_eq!(scaled > 0, !second_pass);
    }
}

/// The partitions (block sizes) of the key frame `data`.
fn key_block_sizes(data: &[u8]) -> Vec<u8> {
    let parsed = crate::header::parse(&crate::header::Persist::new(), data).expect("header");
    let mut fp =
        crate::dec::FrameParse::new(&parsed, None, &[], [0; 3], Default::default()).expect("parse");
    for r in 0..fp.sb_rows() {
        fp.row(r).expect("row");
    }
    fp.finish()
        .tiles
        .iter()
        .flat_map(|t| t.blocks.iter().map(|b| b.bsize))
        .collect()
}

#[test]
fn a_key_frame_codes_a_smooth_picture_in_64x64_halves() {
    // A smooth picture: at a fixed quantizer, speed 5 tries the halves of
    // a key frame's 64×64 blocks that came close whole
    // (`tile::KEY_HALVES`), and some win; a second pass does not try them.
    let (w, h) = (320usize, 192usize);
    let y: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            (128.0 + 127.0 * (x / 50.0).sin() * (y / 40.0).cos()) as u8
        })
        .collect();
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let pic = [y, vec![128; cw * ch], vec![128; cw * ch]];
    let halves = |second_pass: bool| {
        let (out, _) = code_checked(std::slice::from_ref(&pic), (w, h), 100, 5, second_pass);
        key_block_sizes(&out[0])
            .into_iter()
            .filter(|&b| b == crate::geom::BLOCK_64X32 || b == crate::geom::BLOCK_32X64)
            .count()
    };
    assert!(halves(false) > 0);
    assert_eq!(halves(true), 0);
}

#[test]
fn a_mesh_finds_a_small_textured_square_moved_over_a_flat_picture() {
    // A 6×6 textured square on a flat picture moves by (12, 14) samples:
    // short of the square, every vector leaves the same error, and the
    // diamond stops at zero. At a fixed quantizer in one pass a 16×16
    // block's search ends with a mesh (`tile::MESH_RANGE`), which finds the
    // square's vector: its picture codes in 24 bytes, against 70 without
    // the mesh (as a second pass, which takes none).
    let (w, h) = (128usize, 128usize);
    let flat = [
        vec![90u8; w * h],
        vec![128; w * h / 4],
        vec![128; w * h / 4],
    ];
    let with_square = |r0: usize, c0: usize| {
        let mut p = flat.clone();
        for r in r0..r0 + 6 {
            for c in c0..c0 + 6 {
                p[0][r * w + c] = 40 + (((r - r0) * 7 + (c - c0) * 13) % 11 * 19) as u8;
            }
        }
        p
    };
    let pics = [with_square(42, 40), with_square(54, 54)];
    let (one, _) = code_checked(&pics, (w, h), 60, 5, false);
    let (two, _) = code_checked(&pics, (w, h), 60, 5, true);
    assert!(
        one[1].len() * 2 < two[1].len(),
        "one pass {} bytes, a second pass (no mesh) {}",
        one[1].len(),
        two[1].len()
    );
}

#[test]
fn a_fixed_quantizer_tries_a_second_smaller_transform() {
    // A detailed picture moved, with lone samples brighter every 16 across
    // and down: the residual of 16×16 and larger inter blocks is in specks
    // that small transforms code for less. At a fixed quantizer in one
    // pass, from speed 4, a block whose next smaller transform won tries
    // the one after it too; a second pass (one size smaller only) takes
    // none two sizes under the block's.
    let (w, h) = (128usize, 128usize);
    let a = texture(w, h, (0, 0));
    let mut b = texture(w, h, (4, 0));
    for r in (3..h).step_by(16) {
        for c in (5..w).step_by(16) {
            b[0][r * w + c] = b[0][r * w + c].saturating_add(70);
        }
    }
    let two_under = |second_pass: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
        let mut dec = Decoder::new();
        let cw = w.div_ceil(2);
        let mut n = 0;
        for (t, pic) in [&a, &b].into_iter().enumerate() {
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc
                .encode(fixed_params(t, 60, 5, second_pass))
                .expect("encode");
            dec.decode(&coded.data).expect("decode");
            if t == 1 {
                n = dec
                    .last_blocks()
                    .filter(|k| k.refs[0] != 0)
                    .filter(|k| match k.bsize {
                        crate::geom::BLOCK_16X16 => k.tx == 0,
                        crate::geom::BLOCK_32X32 | crate::geom::BLOCK_64X64 => k.tx <= 1,
                        _ => false,
                    })
                    .count();
            }
        }
        n
    };
    assert!(two_under(false) > 0);
    assert_eq!(two_under(true), 0);
}

#[test]
fn a_key_frame_tries_the_chroma_mode_satd_ranks_first() {
    // Luma in horizontal stripes, chroma in vertical ones: the luma's best
    // mode is H, and the one chroma mode a key frame's best tried after DC
    // was the luma's. At a fixed quantizer in one pass it tries the best by
    // SATD besides DC (`tile::KEY_RD_SPEED`), V; a second pass takes DC or
    // H.
    let (w, h) = (128usize, 128usize);
    let (cw, ch) = (w / 2, h / 2);
    let luma: Vec<u8> = (0..w * h)
        .map(|i| if (i / w) % 8 < 4 { 70 } else { 150 })
        .collect();
    let stripes: Vec<u8> = (0..cw * ch)
        .map(|i| if (i % cw) % 8 < 4 { 90 } else { 170 })
        .collect();
    let pic = [luma, stripes.clone(), stripes];
    let vertical = |second_pass: bool| {
        let (out, _) = code_checked(std::slice::from_ref(&pic), (w, h), 60, 5, second_pass);
        let mut dec = Decoder::new();
        dec.decode(&out[0]).expect("decode");
        dec.last_blocks()
            .filter(|b| b.uv == crate::geom::V || b.uv == crate::geom::TM)
            .count()
    };
    assert!(vertical(false) > 0);
    assert_eq!(vertical(true), 0);
}

#[test]
fn an_alt_ref_searches_altref_however_little_the_frames_before_used_it() {
    // A key frame of one picture, then a golden frame and inter frames of
    // another: ALTREF (the key frame) predicts none of them, so from speed
    // 5 it is left out of the search but every fourth frame. Then a hidden
    // alt-ref of the first picture again: at a fixed quantizer in one pass
    // it searches every reference (`REF_PRUNE_SPEED`), and codes its
    // blocks from ALTREF, wherever the count stood.
    let (w, h) = (128usize, 128usize);
    let cw = w.div_ceil(2);
    let a = texture(w, h, (0, 0));
    let b = texture(w, h, (500, 300));
    for before in 2..=6 {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
        let mut dec = Decoder::new();
        for t in 0..=before {
            let pic = if t == 0 { &a } else { &b };
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc
                .encode(FrameParams {
                    golden: t == 1,
                    ..fixed_params(t, 120, 5, false)
                })
                .expect("encode");
            dec.decode(&coded.data).expect("decode");
        }
        enc.load([(&a[0], w), (&a[1], cw), (&a[2], cw)]);
        let coded = enc
            .encode(FrameParams {
                hidden: true,
                hidden_slot: 3,
                ..fixed_params(before + 1, 120, 5, false)
            })
            .expect("encode");
        dec.decode(&coded.data).expect("decode");
        let (all, altref) = dec.last_blocks().fold((0, 0), |(n, k), blk| {
            (n + 1, k + usize::from(blk.refs[0] == 3))
        });
        assert!(
            altref * 2 > all,
            "after {before} frames: {altref} of {all} blocks on ALTREF"
        );
    }
}

#[test]
fn the_previous_frames_common_vector_starts_the_search() {
    // Three bands: a texture on the left that moves 48 samples a picture
    // from the second picture on (found with the jump given), a static
    // texture in the middle, and a texture on the right that starts to
    // move, by as much, in the third. Its blocks' neighbours and the
    // blocks where they stood took no vector, and 48 is past the diamond
    // and the mesh: at a fixed quantizer in one pass the previous frame's
    // most common vectors on LAST start each search on it
    // (`tile::POP_STARTS`), and the right band takes 48.
    let (w, h) = (384usize, 128usize);
    let cw = w.div_ceil(2);
    let still = texture(w, h, (1000, 700));
    let band = |t: usize, right: usize| -> [Vec<u8>; 3] {
        let left = texture(w, h, (48 * t, 0));
        let moved = texture(w, h, (48 * right, 0));
        let mut pic = still.clone();
        for r in 0..h {
            for c in 0..w {
                pic[0][r * w + c] = if c < 160 {
                    left[0][r * w + c]
                } else if c < 224 {
                    still[0][r * w + c]
                } else {
                    moved[0][r * w + c]
                };
            }
        }
        pic
    };
    let pics = [band(0, 0), band(1, 0), band(2, 1)];
    let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
    let mut dec = Decoder::new();
    for (t, pic) in pics.iter().enumerate() {
        if t == 1 {
            enc.set_global((48, 0));
        }
        enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
        let coded = enc.encode(fixed_params(t, 120, 5, false)).expect("encode");
        dec.decode(&coded.data).expect("decode");
    }
    // The right band's blocks whose match lies inside the picture.
    let (all, moved) = dec
        .last_blocks()
        .filter(|b| b.col >= 28 && b.col < 40)
        .fold((0, 0), |(n, k), b| {
            let v = b.bmv[3][0];
            let on = b.refs[0] == 1 && v.col == 48 * 8 && v.row == 0;
            (n + 1, k + usize::from(on))
        });
    assert!(moved * 4 >= all * 3, "{moved} of {all} blocks moved");
}

#[test]
fn a_drawn_key_frame_tries_halves_at_every_size() {
    // A key frame of drawn pictures (`FrameEncoder::set_drawn`) at a fixed
    // quantizer in one pass tries the halves of its blocks at every size
    // (`tile::Shared::drawn_key`); otherwise, at speed 5, only a 64×64
    // block's. A texture of 4×4 cells in bands of 8 rows, each band's cells
    // alike: halves across follow the bands.
    let (w, h) = (128usize, 128usize);
    let cw = w.div_ceil(2);
    let mut pic = texture(w, h, (0, 0));
    for r in 0..h {
        if (r / 8) % 2 == 1 {
            for c in 0..w {
                pic[0][r * w + c] = 128;
            }
        }
    }
    let rects = |drawn: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
        enc.set_drawn(drawn);
        enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
        let coded = enc.encode(fixed_params(0, 120, 5, false)).expect("encode");
        let mut dec = Decoder::new();
        dec.decode(&coded.data).expect("decode");
        dec.last_blocks()
            .filter(|b| {
                matches!(
                    b.bsize,
                    crate::geom::BLOCK_8X16
                        | crate::geom::BLOCK_16X8
                        | crate::geom::BLOCK_16X32
                        | crate::geom::BLOCK_32X16
                )
            })
            .count()
    };
    assert!(rects(true) > 0);
    assert_eq!(rects(false), 0);
}

#[test]
fn a_drawn_key_frame_codes_8x4_and_4x8_intra_blocks() {
    // A key frame of drawn pictures at a fixed quantizer in one pass
    // (`tile::Tile::drawn_key_rd`) tries an 8×8 block's 8×4 and 4×8
    // halves, each with one mode, besides its four 4×4s; otherwise it
    // never codes them. Its reconstruction is the decoder's.
    use crate::dsp::pixel::Pixel;
    let (w, h) = (128usize, 128usize);
    let cw = w.div_ceil(2);
    let pic = texture(w, h, (0, 0));
    let halves = |drawn: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
        enc.set_drawn(drawn);
        enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
        let coded = enc.encode(fixed_params(0, 120, 5, false)).expect("encode");
        let mut dec = Decoder::new();
        let shown = dec.decode(&coded.data).expect("decode").expect("shown");
        let rec = enc.last().expect("recon");
        let (data, stride) = rec.buf.plane(0);
        let recon: Vec<u8> = (0..h)
            .flat_map(|r| {
                data[r * stride..r * stride + w]
                    .iter()
                    .map(|v| v.get() as u8)
            })
            .collect();
        assert_eq!(shown.y, recon, "drawn {drawn}");
        dec.last_blocks()
            .filter(|b| matches!(b.bsize, crate::geom::BLOCK_8X4 | crate::geom::BLOCK_4X8))
            .count()
    };
    assert!(halves(true) > 0);
    assert_eq!(halves(false), 0);
}

#[test]
fn a_coarse_fixed_quantizer_still_searches_the_filter_level() {
    // From about quantizer 208 the level from the quantizer is 63, which
    // the masks cannot be rebased from: at a fixed quantizer in one pass
    // the level starts from 61 (`LF_SEARCH_CAP`) and is searched, and an
    // inter frame takes `LF_FIXED_SCALE` of what the search finds.
    let (w, h) = (256usize, 128usize);
    let pics: Vec<[Vec<u8>; 3]> = (0..3)
        .map(|t| texture(w, h, (10 + 3 * t, 20 + t)))
        .collect();
    assert_eq!(super::filter_level(230, false, 8), 63);
    let cw = w.div_ceil(2);
    let mut enc = FrameEncoder::new(w as u32, h as u32, 1);
    let mut dec = Decoder::new();
    for (t, pic) in pics.iter().enumerate() {
        enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
        enc.lf_found = None;
        let coded = enc.encode(fixed_params(t, 230, 5, false)).expect("encode");
        let head = crate::header::parse(dec.persist(), &coded.data)
            .expect("header")
            .header;
        dec.decode(&coded.data).expect("decode").expect("shown");
        let found = enc.lf_found.expect("searched");
        if t > 0 {
            assert_eq!(
                head.filter_level,
                ((f64::from(found) * super::LF_FIXED_SCALE) as u8).max(2),
                "picture {t}"
            );
            assert!(head.filter_level < 62, "picture {t}");
        }
    }
}

#[test]
fn a_coarse_fixed_quantizer_codes_vectors_in_quarters() {
    // At a fixed quantizer in one pass, an inter frame from quantizer 200
    // (`HP_MAX_Q`, libvpx's `HIGH_PRECISION_MV_QTHRESH`) does not allow
    // eighths of a sample; under it, and in a second pass, it does.
    let (w, h) = (128usize, 64usize);
    let pics: Vec<[Vec<u8>; 3]> = (0..2).map(|t| texture(w, h, (t, 0))).collect();
    let cw = w.div_ceil(2);
    let hp = |qindex: u8, second_pass: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 1);
        let mut dec = Decoder::new();
        let mut last = None;
        for (t, pic) in pics.iter().enumerate() {
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc
                .encode(fixed_params(t, qindex, 5, second_pass))
                .expect("encode");
            let head = crate::header::parse(dec.persist(), &coded.data)
                .expect("header")
                .header;
            dec.decode(&coded.data).expect("decode").expect("shown");
            last = Some(head.allow_hp);
        }
        last.expect("an inter frame")
    };
    assert!(!hp(200, false));
    assert!(!hp(240, false));
    assert!(hp(196, false));
    assert!(hp(200, true));
}

#[test]
fn a_still_drawn_picture_tries_vectors_under_8x8_where_it_changed() {
    // An inter frame of a drawn picture that stands still
    // (`FrameEncoder::set_still`) tries 4×4, 8×4 and 4×8 blocks with
    // vectors of their own in the 16×16 blocks that changed (at speed 5
    // otherwise only speeds 0 and 1 do). In the second picture every 8×8
    // block of a band has its left half moved one way and its right half
    // the other: halves of 8×8 blocks follow them.
    let (w, h) = (128usize, 64usize);
    let cw = w.div_ceil(2);
    let a = texture(w, h, (0, 0));
    let left = texture(w, h, (2, 0));
    let right = texture(w, h, (6, 0));
    let mut b = a.clone();
    for r in 16..48 {
        for c in 16..112 {
            b[0][r * w + c] = if c % 8 < 4 {
                left[0][r * w + c]
            } else {
                right[0][r * w + c]
            };
        }
    }
    let changed: Vec<bool> = (0..h / 16)
        .flat_map(|br| {
            let (a, b) = (&a, &b);
            (0..w / 16).map(move |bc| {
                (0..16).any(|r| {
                    let o = (br * 16 + r) * w + bc * 16;
                    a[0][o..o + 16] != b[0][o..o + 16]
                })
            })
        })
        .collect();
    let small = |still: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 1);
        let mut dec = Decoder::new();
        let mut n = 0;
        for (t, pic) in [&a, &b].into_iter().enumerate() {
            enc.set_drawn(true);
            enc.set_still((still && t > 0).then(|| changed.clone()));
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc.encode(fixed_params(t, 120, 5, false)).expect("encode");
            dec.decode(&coded.data).expect("decode").expect("shown");
            if t > 0 {
                n = dec
                    .last_blocks()
                    .filter(|b| b.bsize < crate::geom::BLOCK_8X8 && b.refs[0] > 0)
                    .count();
            }
        }
        n
    };
    assert!(small(true) > 0);
    assert_eq!(small(false), 0);
}

#[test]
fn a_still_drawn_picture_tries_halves_over_what_changed_and_what_did_not() {
    // An inter frame of a drawn picture that stands still
    // (`FrameEncoder::set_still`) tries the halves of a 32×32 or 64×64
    // block that covers 16×16 blocks that changed and blocks that did not
    // (at speed 5 otherwise only a key frame's). In the second picture the
    // lower half of each 64×64 block moved and the upper half stands.
    let (w, h) = (128usize, 128usize);
    let cw = w.div_ceil(2);
    let a = texture(w, h, (0, 0));
    let moved = texture(w, h, (3, 0));
    let mut b = a.clone();
    for r in 0..h {
        if r % 64 >= 32 {
            b[0][r * w..r * w + w].copy_from_slice(&moved[0][r * w..r * w + w]);
        }
    }
    let changed: Vec<bool> = (0..h / 16)
        .flat_map(|br| (0..w / 16).map(move |_| br % 4 >= 2))
        .collect();
    let rects = |still: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 1);
        let mut dec = Decoder::new();
        let mut n = 0;
        for (t, pic) in [&a, &b].into_iter().enumerate() {
            enc.set_drawn(true);
            enc.set_still((still && t > 0).then(|| changed.clone()));
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc.encode(fixed_params(t, 120, 5, false)).expect("encode");
            dec.decode(&coded.data).expect("decode").expect("shown");
            if t > 0 {
                n = dec
                    .last_blocks()
                    .filter(|b| {
                        matches!(
                            b.bsize,
                            crate::geom::BLOCK_32X16
                                | crate::geom::BLOCK_16X32
                                | crate::geom::BLOCK_64X32
                                | crate::geom::BLOCK_32X64
                        )
                    })
                    .count();
            }
        }
        n
    };
    assert!(rects(true) > 0);
    assert_eq!(rects(false), 0);
}

#[test]
fn a_drawn_picture_at_a_fixed_quantizer_picks_each_blocks_filter() {
    // Noise, then the same noise through the smooth filter half a sample
    // across: in a drawn picture at a fixed quantizer in one pass at speed
    // 5, the inter frame codes the filter per block (`DRAWN_FILTER_SPEED`),
    // and blocks take the smooth one; otherwise the frame keeps one filter.
    let (w, h) = (128usize, 64usize);
    let cw = w.div_ceil(2);
    let mut k = 0x9e37_79b9u32;
    let mut next = || {
        k ^= k << 13;
        k ^= k >> 17;
        k ^= k << 5;
        (40 + (k >> 24) % 120) as u8
    };
    let first: [Vec<u8>; 3] = [
        (0..w * h).map(|_| next()).collect(),
        (0..cw * h / 2).map(|_| next()).collect(),
        (0..cw * h / 2).map(|_| next()).collect(),
    ];
    // (Luma half a sample, chroma a quarter of its own.)
    let smooth = |p: &[u8], pw: usize, taps: [i32; 8]| -> Vec<u8> {
        (0..p.len())
            .map(|i| {
                let (r, c) = (i / pw, i % pw);
                let mut v = 64;
                for (t, &f) in taps.iter().enumerate() {
                    let x = (c as i32 + t as i32 - 3).clamp(0, pw as i32 - 1) as usize;
                    v += f * i32::from(p[r * pw + x]);
                }
                (v >> 7).clamp(0, 255) as u8
            })
            .collect()
    };
    let second = [
        smooth(&first[0], w, [-1, -4, 14, 55, 55, 14, -4, -1]),
        smooth(&first[1], cw, [-2, -3, 24, 62, 46, 5, -4, 0]),
        smooth(&first[2], cw, [-2, -3, 24, 62, 46, 5, -4, 0]),
    ];
    let noise = |t: u32| {
        if t == 0 {
            first.clone()
        } else {
            second.clone()
        }
    };
    let run = |drawn: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 2);
        let mut dec = Decoder::new();
        let mut out = (false, 0);
        for t in 0..2 {
            let pic = noise(t as u32);
            enc.set_drawn(drawn);
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            let coded = enc.encode(fixed_params(t, 120, 5, false)).expect("encode");
            let head = crate::header::parse(dec.persist(), &coded.data)
                .expect("header")
                .header;
            dec.decode(&coded.data).expect("decode");
            if t == 1 {
                let smooth = dec
                    .last_blocks()
                    .filter(|b| b.refs[0] > 0 && b.interp == crate::geom::SMOOTH)
                    .count();
                out = (head.interp == crate::geom::SWITCHABLE, smooth);
            }
        }
        out
    };
    let (switchable, smooth) = run(true);
    assert!(switchable);
    assert!(smooth > 0, "no block took the smooth filter");
    assert!(!run(false).0);
}

/// Two pictures for the tests of a still drawn picture's changed 16×16
/// blocks: a texture, then the same with each 16×16 block of a band
/// redrawn as noise, its upper eight rows apart from its lower eight;
/// and the map of the blocks that changed.
fn noise_band(w: usize, h: usize) -> ([Vec<u8>; 3], [Vec<u8>; 3], Vec<bool>) {
    let a = texture(w, h, (0, 0));
    let mut b = a.clone();
    let mut k = 0x2545_f491u32;
    for r in 16..48 {
        for c in 0..w {
            k ^= k << 13;
            k ^= k >> 17;
            k ^= k << 5;
            let base = if (r / 8) % 2 == 0 { 60 } else { 150 };
            b[0][r * w + c] = (base + (k >> 24) % 48) as u8;
        }
    }
    let changed = (0..h / 16)
        .flat_map(|br| (0..w / 16).map(move |_| (1..3).contains(&br)))
        .collect();
    (a, b, changed)
}

#[test]
fn a_still_drawn_picture_tries_halves_of_the_16x16_blocks_that_changed() {
    // An inter frame of a drawn picture that stands still
    // (`FrameEncoder::set_still`) tries the halves of the 16×16 blocks
    // that changed before their split (`STILL_HALVES_LEVEL`), with a
    // search of their own; at speed 5 otherwise only after a split that
    // won, from its quarters' choices (`RECT_SPEED`).
    let (w, h) = (128usize, 64usize);
    let cw = w.div_ceil(2);
    let (a, b, changed) = noise_band(w, h);
    let tried = |still: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 1);
        for (t, pic) in [&a, &b].into_iter().enumerate() {
            enc.set_drawn(true);
            enc.set_still((still && t > 0).then(|| changed.clone()));
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            super::tile::INTER_16X16_HALVES.with(|c| c.set(0));
            enc.encode(fixed_params(t, 120, 5, false)).expect("encode");
        }
        super::tile::INTER_16X16_HALVES.with(|c| c.get())
    };
    // (The band's 16×16 blocks: two rows of eight.)
    assert_eq!(tried(true), 16);
    assert_eq!(tried(false), 0);
}

#[test]
fn a_still_drawn_picture_tries_more_intra_modes_where_it_changed() {
    // An inter frame of a drawn picture that stands still
    // (`FrameEncoder::set_still`) tries `STILL_INTRA_CANDS` intra modes in
    // the blocks over 16×16 blocks that changed; at speed 5 otherwise one.
    let (w, h) = (128usize, 64usize);
    let cw = w.div_ceil(2);
    let (a, b, changed) = noise_band(w, h);
    let most = |still: bool| {
        let mut enc = FrameEncoder::new(w as u32, h as u32, 1);
        for (t, pic) in [&a, &b].into_iter().enumerate() {
            enc.set_drawn(true);
            enc.set_still((still && t > 0).then(|| changed.clone()));
            enc.load([(&pic[0], w), (&pic[1], cw), (&pic[2], cw)]);
            super::tile::INTER_INTRA_MOST.with(|c| c.set(0));
            enc.encode(fixed_params(t, 120, 5, false)).expect("encode");
        }
        super::tile::INTER_INTRA_MOST.with(|c| c.get())
    };
    let (on, off) = (most(true), most(false));
    assert_eq!(off, 1);
    assert!(on > off, "{on}");
}
