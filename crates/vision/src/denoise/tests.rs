use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::*;

/// Frame `k` of a clip: a pattern every pixel of which differs, moving a
/// little from frame to frame, with some speckle for the model to remove.
fn frame(width: u32, height: u32, k: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let v = |c: u32| {
                    let smooth = (x + k) * (3 + c) + y * (7 - c);
                    let speckle =
                        (x * 31 + y * 17 + k * 13 + c * 7).wrapping_mul(2_654_435_761) >> 28;
                    ((smooth + (x * y) % (11 + c) + speckle) % 256) as u8
                };
                [v(0), v(1), v(2)]
            })
        })
        .collect()
}

fn clip(frames: u32, width: u32, height: u32) -> Vec<Vec<u8>> {
    (0..frames).map(|k| frame(width, height, k)).collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-denoise-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every frame of `clip` through a stream, and how often each was read.
fn run(
    denoiser: &Denoiser,
    clip: &[Vec<u8>],
    width: u32,
    height: u32,
) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut stream = denoiser.stream(clip.len(), width, height, 25.0).unwrap();
    let mut reads = vec![0; clip.len()];
    let mut source = |k: usize| {
        reads[k] += 1;
        Ok(clip[k].clone())
    };
    let mut out = Vec::new();
    while let Some(rgb) = stream.next(&mut source).unwrap() {
        out.push(rgb);
    }
    (out, reads)
}

fn worst(a: &[Vec<u8>], b: &[Vec<u8>]) -> u8 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap()
}

/// A frame that is not one flat colour: random weights can saturate, which
/// would make every comparison pass.
fn assert_varied(rgb: &[u8]) {
    let levels: BTreeSet<u8> = rgb.iter().copied().collect();
    assert!(levels.len() > 32, "only {} levels", levels.len());
}

impl Denoiser {
    /// Frame `t` with nothing kept from the frames before: both stages run
    /// fresh on the whole frame, as FastDVDnet's own loop does.
    fn fresh(&self, clip: &[Vec<u8>], width: u32, height: u32, t: usize) -> Vec<u8> {
        let stream = self.stream(clip.len(), width, height, 25.0).unwrap();
        let padded = (width.next_multiple_of(GRID), height.next_multiple_of(GRID));
        let whole = Rect {
            left: 0,
            top: 0,
            width,
            height,
        };
        let input = |k: usize| {
            let image = Image {
                width,
                height,
                rgb: &clip[stream.window(k)],
            };
            self.input(planes(&image, whole, padded), 3, padded)
        };
        let noise = || self.noise(25.0 / 255.0, padded);
        let first: Vec<_> = (t..t + 3)
            .map(|run| {
                self.model
                    .temp1
                    .forward([input(run), input(run + 1), input(run + 2)], noise())
            })
            .collect();
        let [a, b, c]: [Tensor<4>; 3] = first.try_into().unwrap();
        let out = self
            .model
            .temp2
            .forward([a, b, c], noise())
            .cast(FloatDType::F32);
        let values = out.into_data().try_to_vec::<f32>().unwrap();
        let plane = (padded.0 * padded.1) as usize;
        let mut rgb = Vec::new();
        for y in 0..height as usize {
            for x in 0..width as usize {
                for channel in 0..3 {
                    rgb.push(to_byte(values[channel * plane + y * padded.0 as usize + x]));
                }
            }
        }
        rgb
    }
}

#[test]
fn a_clip_comes_out_frame_for_frame_at_its_size() {
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let (w, h) = (13, 9);
    let (out, _) = run(&denoiser, &clip(4, w, h), w, h);
    assert_eq!(out.len(), 4);
    for rgb in &out {
        assert_eq!(rgb.len(), (w * h * 3) as usize);
    }
}

#[test]
fn the_stream_makes_what_the_model_makes_fresh_each_frame() {
    // Keeping the first stage from frame to frame is only a saving.
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let (w, h) = (22, 14);
    let frames = clip(6, w, h);
    let (out, _) = run(&denoiser, &frames, w, h);
    let fresh: Vec<_> = (0..6).map(|t| denoiser.fresh(&frames, w, h, t)).collect();
    assert_varied(&out[3]);
    assert!(worst(&out, &fresh) <= 1, "{}", worst(&out, &fresh));
}

#[test]
fn tiles_make_the_frame_the_model_makes_in_one_go() {
    let mut denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let (w, h) = (230, 150);
    let frames = clip(3, w, h);
    let (whole, _) = run(&denoiser, &frames, w, h);
    denoiser.tile = 64;
    let (tiled, _) = run(&denoiser, &frames, w, h);
    assert_eq!(denoiser.stream(3, w, h, 25.0).unwrap().tiles.len(), 4 * 3);
    assert_varied(&whole[1]);
    // Sums run in another order in another tile, so a byte may round apart.
    assert!(
        worst(&whole, &tiled) <= 1,
        "tiles differ by {}",
        worst(&whole, &tiled)
    );
}

#[test]
fn the_window_mirrors_two_frames_onto_each_end() {
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let window = |frames: usize| {
        let stream = denoiser.stream(frames, 4, 4, 25.0).unwrap();
        (0..frames + 4)
            .map(|k| stream.window(k))
            .collect::<Vec<_>>()
    };
    assert_eq!(window(6), [2, 1, 0, 1, 2, 3, 4, 5, 4, 3]);
    assert_eq!(window(3), [2, 1, 0, 1, 2, 1, 0]);
    // Too short to mirror: the frames it has, never one past the end.
    assert_eq!(window(2), [1, 1, 0, 1, 0, 0]);
    assert_eq!(window(1), [0, 0, 0, 0, 0]);
}

#[test]
fn each_frame_is_read_about_once() {
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let (out, reads) = run(&denoiser, &clip(8, 8, 8), 8, 8);
    assert_eq!(out.len(), 8);
    assert!(reads.iter().all(|&n| (1..=2).contains(&n)), "{reads:?}");
    assert!(reads.iter().sum::<usize>() <= 8 + 2, "{reads:?}");
    let (out, reads) = run(&denoiser, &clip(1, 8, 8), 8, 8);
    assert_eq!((out.len(), reads), (1, vec![1]));
}

#[test]
fn half_precision_makes_the_same_frames_a_byte_or_two_apart() {
    let full = Denoiser::untrained(Compute::Cpu).unwrap();
    let half = Denoiser::at(full.device.clone(), full.model.clone(), FloatDType::F16);
    let (w, h) = (20, 12);
    let frames = clip(3, w, h);
    let (a, _) = run(&full, &frames, w, h);
    let (b, _) = run(&half, &frames, w, h);
    assert!(worst(&a, &b) <= 2, "{}", worst(&a, &b));
}

#[test]
fn a_clip_the_model_cannot_take_is_refused_before_any_work() {
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let big = DENOISE_MAX_SIDE + 1;
    for (frames, w, h, noise) in [
        (1, 0, 4, 25.0),
        (1, 4, 0, 25.0),
        (1, big, 4, 25.0),
        (1, 4, big, 25.0),
        (0, 4, 4, 25.0),
        (1, 4, 4, MIN_NOISE - 0.5),
        (1, 4, 4, MAX_NOISE + 0.5),
        (1, 4, 4, f32::NAN),
    ] {
        assert!(
            denoiser.stream(frames, w, h, noise).is_err(),
            "{frames} frames {w}x{h} at {noise}"
        );
    }
    assert!(denoiser.stream(1, 4, 4, MIN_NOISE).is_ok());
    assert!(denoiser.stream(1, 4, 4, MAX_NOISE).is_ok());
}

#[test]
fn a_frame_of_the_wrong_size_or_a_failed_read_stops_the_stream() {
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let mut stream = denoiser.stream(3, 4, 4, 25.0).unwrap();
    let short = stream.next(&mut |_| Ok(vec![0; 47]));
    assert!(
        matches!(short, Err(VisionError::Image { bytes: 47, .. })),
        "{short:?}"
    );
    let mut stream = denoiser.stream(3, 4, 4, 25.0).unwrap();
    let failed = stream.next(&mut |_| {
        Err(VisionError::Model {
            message: "unreadable".into(),
        })
    });
    assert!(matches!(failed, Err(VisionError::Model { .. })));
}

#[test]
fn saved_weights_load_back_into_the_same_model() {
    let dir = scratch("round-trip");
    let saved = Denoiser::untrained(Compute::Cpu).unwrap();
    let mut store = BurnpackStore::from_file(dir.join(DENOISE_WEIGHTS)).overwrite(true);
    saved.model.save_into(&mut store).unwrap();
    let loaded = Denoiser::load(&dir, Compute::Cpu).unwrap();
    let frames = clip(3, 6, 6);
    assert_eq!(run(&saved, &frames, 6, 6).0, run(&loaded, &frames, 6, 6).0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_or_damaged_weights_are_an_error_and_not_a_panic() {
    let dir = scratch("bad");
    let Err(VisionError::Weights { path, message }) = Denoiser::load(&dir, Compute::Cpu) else {
        panic!("loaded without weights");
    };
    assert_eq!(path, dir.join(DENOISE_WEIGHTS));
    assert_eq!(message, "missing");
    std::fs::write(dir.join(DENOISE_WEIGHTS), b"not a burnpack").unwrap();
    assert!(matches!(
        Denoiser::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    let pth = dir.join("model.pth");
    assert!(matches!(
        convert_fastdvdnet(&pth, &dir.join("out.bpk")),
        Err(VisionError::Weights { .. })
    ));
    std::fs::write(&pth, b"not a pickle").unwrap();
    assert!(matches!(
        convert_fastdvdnet(&pth, &dir.join("out.bpk")),
        Err(VisionError::Weights { .. })
    ));
    assert!(!dir.join("out.bpk").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn padding_mirrors_the_last_rows_and_columns() {
    let rgb: Vec<u8> = (0..5 * 3 * 3).map(|v| v as u8).collect();
    let image = Image {
        width: 5,
        height: 3,
        rgb: &rgb,
    };
    let whole = Rect {
        left: 0,
        top: 0,
        width: 5,
        height: 3,
    };
    let chw = planes(&image, whole, (8, 4));
    let red = |x: usize, y: usize| (chw[y * 8 + x] * 255.0).round() as u8;
    // Columns 5, 6, 7 mirror 3, 2, 1; row 3 mirrors row 1.
    for (padded, from) in [(5, 3), (6, 2), (7, 1)] {
        assert_eq!(red(padded, 0), rgb[from * 3]);
    }
    assert_eq!(red(0, 3), rgb[5 * 3]);
}

/// How long one 720p frame takes once the stream is going, written to
/// stderr: run with
/// `cargo test -p cutlass-vision --release denoise_720p -- --ignored --nocapture`.
fn time_720p(compute: Compute) {
    let denoiser = Denoiser::untrained(compute).unwrap();
    let (w, h) = (1280, 720);
    let frames = clip(8, w, h);
    let mut stream = denoiser.stream(frames.len(), w, h, 25.0).unwrap();
    let mut source = |k: usize| Ok(frames[k].clone());
    // The first frame runs the first stage three times, and compiles and
    // tunes the GPU kernels.
    let start = Instant::now();
    stream.next(&mut source).unwrap();
    let first = start.elapsed();
    let start = Instant::now();
    let mut runs = 0;
    while stream.next(&mut source).unwrap().is_some() {
        runs += 1;
    }
    let each = start.elapsed() / runs;
    let _ = std::io::stderr().write_all(
        format!("denoise 720p on {compute:?}: first {first:?}, then {each:?} a frame\n").as_bytes(),
    );
}

#[test]
#[ignore = "needs a Metal GPU; a timing run, not a check"]
fn denoise_720p_on_the_gpu() {
    time_720p(Compute::Gpu);
}

#[test]
#[ignore = "a timing run, not a check, and slow on the CPU"]
fn denoise_720p_on_the_cpu() {
    time_720p(Compute::Cpu);
}

/// Set in a tagged run, the tests that need the real weights fail when
/// they are missing instead of skipping.
const REQUIRE: &str = "CUTLASS_REQUIRE_DENOISE_MODEL";

fn local_models() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/models/denoise")
}

/// A noisy clip, its noise level and what FastDVDnet's own code made of
/// it under PyTorch, written by `tools/vision-models/scripts/fetch_denoise.sh`.
struct Reference {
    width: u32,
    height: u32,
    noise: f32,
    noisy: Vec<Vec<u8>>,
    denoised: Vec<Vec<u8>>,
}

fn reference() -> Option<Reference> {
    let dir = local_models();
    let file = dir.join("reference_clip.safetensors");
    if !dir.join(DENOISE_WEIGHTS).is_file() || !file.is_file() {
        assert!(
            std::env::var_os(REQUIRE).is_none(),
            "{REQUIRE} is set and the denoise weights or {} are missing",
            file.display()
        );
        return None;
    }
    let bytes = std::fs::read(&file).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let noisy = tensors.tensor("noisy").unwrap();
    let denoised = tensors.tensor("denoised").unwrap();
    let &[frames, height, width, 3] = noisy.shape() else {
        panic!("noisy is {:?}", noisy.shape());
    };
    assert_eq!(denoised.shape(), noisy.shape());
    let noise = tensors.tensor("noise").unwrap();
    let noise = f32::from_le_bytes(noise.data()[..4].try_into().unwrap());
    let split = |data: &[u8]| {
        data.chunks(width * height * 3)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>()
    };
    let reference = Reference {
        width: width as u32,
        height: height as u32,
        noise,
        noisy: split(noisy.data()),
        denoised: split(denoised.data()),
    };
    assert_eq!(reference.noisy.len(), frames);
    Some(reference)
}

fn worst_against_pytorch(compute: Compute) -> Option<u8> {
    let reference = reference()?;
    let denoiser = Denoiser::load(&local_models(), compute).unwrap();
    let mut stream = denoiser
        .stream(
            reference.noisy.len(),
            reference.width,
            reference.height,
            reference.noise,
        )
        .unwrap();
    let mut got = Vec::new();
    while let Some(rgb) = stream
        .next(&mut |k| Ok(reference.noisy[k].clone()))
        .unwrap()
    {
        got.push(rgb);
    }
    // The model does denoise: it moves the frames well away from the input.
    assert!(worst(&got, &reference.noisy) > 8);
    Some(worst(&got, &reference.denoised))
}

#[test]
fn the_real_weights_make_what_pytorch_makes() {
    if let Some(worst) = worst_against_pytorch(Compute::Cpu) {
        assert!(worst <= 1, "{worst} levels from PyTorch");
    }
}

#[test]
#[ignore = "needs a Metal GPU"]
fn the_real_weights_in_half_precision_stay_close_to_pytorch() {
    if let Some(worst) = worst_against_pytorch(Compute::Gpu) {
        assert!(worst <= 2, "{worst} levels from PyTorch");
    }
}
