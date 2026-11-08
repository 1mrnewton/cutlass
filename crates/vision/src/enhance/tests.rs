use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use burn::nn::Initializer;

use super::*;

/// A frame whose pixels all differ: a smooth gradient with a pattern.
fn frame(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let v = |k: u32| ((x * (3 + k) + y * (7 - k) + (x * y) % (11 + k)) % 256) as u8;
                [v(0), v(1), v(2)]
            })
        })
        .collect()
}

fn image(width: u32, height: u32, rgb: &[u8]) -> Image<'_> {
    Image { width, height, rgb }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-enhance-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

impl Enhancer {
    /// `image` read in tiles `tile` pixels wide.
    fn enhance_in(&self, image: &Image<'_>, tile: u32) -> Enhanced {
        let (width, height) = check(image).unwrap();
        let out_w = width * SCALE;
        let mut rgb = vec![0_u8; (out_w * height * SCALE * 3) as usize];
        for top in (0..height).step_by(tile as usize) {
            for left in (0..width).step_by(tile as usize) {
                let own = Rect {
                    left,
                    top,
                    width: tile.min(width - left),
                    height: tile.min(height - top),
                };
                self.tile(image, own, &mut rgb, out_w).unwrap();
            }
        }
        Enhanced {
            width: out_w,
            height: height * SCALE,
            rgb,
        }
    }
}

#[test]
fn a_frame_comes_out_four_times_larger() {
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let rgb = frame(9, 5);
    let out = enhancer.enhance(&image(9, 5, &rgb)).unwrap();
    assert_eq!((out.width, out.height), (36, 20));
    assert_eq!(out.rgb.len(), 36 * 20 * 3);
}

#[test]
fn with_nothing_learned_the_frame_is_grown_by_nearest_neighbour() {
    // The last convolution at zero leaves only the grown frame, which
    // checks the shuffle puts each colour's sixteen copies in its block.
    let mut enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let grown = (SCALE * SCALE) as usize;
    if let Some(last) = enhancer.model.body.last_mut() {
        *last = Step::Conv(
            Conv2dConfig::new([FEATURES, 3 * grown], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_initializer(Initializer::Zeros)
                .init(&enhancer.device),
        );
    }
    let (w, h) = (7, 4);
    let rgb = frame(w, h);
    let out = enhancer.enhance(&image(w, h, &rgb)).unwrap();
    for y in 0..h * SCALE {
        for x in 0..w * SCALE {
            let from = (((y / SCALE) * w + x / SCALE) * 3) as usize;
            let to = ((y * w * SCALE + x) * 3) as usize;
            assert_eq!(&out.rgb[to..to + 3], &rgb[from..from + 3], "({x}, {y})");
        }
    }
}

#[test]
fn half_precision_makes_the_same_frame_a_byte_apart_at_most() {
    let full = Enhancer::untrained(Compute::Cpu).unwrap();
    let half = Enhancer::at(full.device.clone(), full.model.clone(), FloatDType::F16);
    let rgb = frame(20, 12);
    let a = full.enhance(&image(20, 12, &rgb)).unwrap();
    let b = half.enhance(&image(20, 12, &rgb)).unwrap();
    let worst = a.rgb.iter().zip(&b.rgb).map(|(x, y)| x.abs_diff(*y)).max();
    assert!(worst <= Some(1), "{worst:?}");
}

#[test]
fn tiles_make_the_frame_the_model_makes_in_one_go() {
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let (w, h) = (45, 30);
    let rgb = frame(w, h);
    let source = image(w, h, &rgb);
    let whole = enhancer.enhance_in(&source, 64);
    let tiled = enhancer.enhance_in(&source, 8);
    let worst = whole
        .rgb
        .iter()
        .zip(&tiled.rgb)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    // Sums run in another order in another tile, so a byte may round apart.
    assert!(worst <= 1, "tiles differ by {worst}");
}

#[test]
fn a_tile_reads_its_neighbours_reach_and_stops_at_the_edge() {
    let own = Rect {
        left: 512,
        top: 0,
        width: 512,
        height: 512,
    };
    assert_eq!(
        own.grown(REACH, 1280, 720),
        Rect {
            left: 512 - REACH,
            top: 0,
            width: 512 + 2 * REACH,
            height: 512 + REACH,
        }
    );
    let corner = Rect {
        left: 1024,
        top: 512,
        width: 256,
        height: 208,
    };
    assert_eq!(
        corner.grown(REACH, 1280, 720),
        Rect {
            left: 1024 - REACH,
            top: 512 - REACH,
            width: 256 + REACH,
            height: 208 + REACH,
        }
    );
    // Every convolution reads one pixel around, so the reach is one per
    // convolution: the first, the body, and the last.
    let convs = Srvgg::new(&Device::flex())
        .body
        .iter()
        .filter(|step| matches!(step, Step::Conv(_)))
        .count();
    assert_eq!(REACH as usize, convs);
}

#[test]
fn a_frame_the_model_cannot_take_is_refused_before_any_work() {
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let rgb = frame(4, 4);
    let big = ENHANCE_MAX_SIDE + 1;
    for (width, height, bytes) in [
        (0, 4, &rgb[..0]),
        (4, 0, &rgb[..0]),
        (4, 4, &rgb[..47]),
        (4, 4, &rgb[..]),
        (big, 1, &rgb[..]),
        (1, big, &rgb[..]),
        (u32::MAX, u32::MAX, &rgb[..]),
    ] {
        let wrong = (width, height) != (4, 4) || bytes.len() != 48;
        if !wrong {
            continue;
        }
        assert!(
            matches!(
                enhancer.enhance(&image(width, height, bytes)),
                Err(VisionError::Image { .. })
            ),
            "{width}x{height} with {} bytes",
            bytes.len()
        );
    }
    assert!(enhancer.enhance(&image(4, 4, &rgb)).is_ok());
}

#[test]
fn saved_weights_load_back_into_the_same_model() {
    let dir = scratch("round-trip");
    let saved = Enhancer::untrained(Compute::Cpu).unwrap();
    let mut store = BurnpackStore::from_file(dir.join(ENHANCE_WEIGHTS)).overwrite(true);
    saved.model.save_into(&mut store).unwrap();
    let loaded = Enhancer::load(&dir, Compute::Cpu).unwrap();
    let rgb = frame(6, 6);
    assert_eq!(
        saved.enhance(&image(6, 6, &rgb)).unwrap(),
        loaded.enhance(&image(6, 6, &rgb)).unwrap()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_or_damaged_weights_are_an_error_and_not_a_panic() {
    let dir = scratch("bad");
    let Err(VisionError::Weights { path, message }) = Enhancer::load(&dir, Compute::Cpu) else {
        panic!("loaded without weights");
    };
    assert_eq!(path, dir.join(ENHANCE_WEIGHTS));
    assert_eq!(message, "missing");
    std::fs::write(dir.join(ENHANCE_WEIGHTS), b"not a burnpack").unwrap();
    assert!(matches!(
        Enhancer::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    let pth = dir.join("model.pth");
    assert!(matches!(
        convert_pytorch(&pth, &dir.join("out.bpk")),
        Err(VisionError::Weights { .. })
    ));
    std::fs::write(&pth, b"not a pickle").unwrap();
    assert!(matches!(
        convert_pytorch(&pth, &dir.join("out.bpk")),
        Err(VisionError::Weights { .. })
    ));
    assert!(!dir.join("out.bpk").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bytes_round_and_stay_in_range() {
    assert_eq!(to_byte(0.0), 0);
    assert_eq!(to_byte(1.0), 255);
    assert_eq!(to_byte(0.5), 128);
    assert_eq!(to_byte(-3.0), 0);
    assert_eq!(to_byte(7.0), 255);
    assert_eq!(to_byte(f32::NAN), 0);
    assert_eq!(to_byte(f32::INFINITY), 255);
}

/// How long one 720p frame takes, written to stderr: run with
/// `cargo test -p cutlass-vision --release enhance_720p -- --ignored --nocapture`.
fn time_720p(compute: Compute) {
    let enhancer = Enhancer::untrained(compute).unwrap();
    let (w, h) = (1280, 720);
    let rgb = frame(w, h);
    let source = image(w, h, &rgb);
    // The first frame compiles and tunes the GPU kernels.
    let start = Instant::now();
    enhancer.enhance(&source).unwrap();
    let first = start.elapsed();
    let runs = 3;
    let start = Instant::now();
    for _ in 0..runs {
        enhancer.enhance(&source).unwrap();
    }
    let each = start.elapsed() / runs;
    let _ = std::io::stderr().write_all(
        format!("enhance 720p on {compute:?}: first {first:?}, then {each:?} a frame\n").as_bytes(),
    );
}

#[test]
#[ignore = "needs a Metal GPU; a timing run, not a check"]
fn enhance_720p_on_the_gpu() {
    time_720p(Compute::Gpu);
}

#[test]
#[ignore = "a timing run, not a check, and slow on the CPU"]
fn enhance_720p_on_the_cpu() {
    time_720p(Compute::Cpu);
}
