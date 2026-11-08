use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::*;

/// A pattern every pixel of which differs, moved `shift` pixels across.
fn frame(width: u32, height: u32, shift: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let x = x + shift;
                let v = |c: u32| ((x * (3 + c) + y * (7 - c) + (x * y) % (11 + c)) % 256) as u8;
                [v(0), v(1), v(2)]
            })
        })
        .collect()
}

fn image(width: u32, height: u32, rgb: &[u8]) -> Image<'_> {
    Image { width, height, rgb }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-rife-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn worst(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap()
}

fn values(tensor: Tensor<4>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().unwrap()
}

#[test]
fn a_frame_between_two_comes_out_at_their_size() {
    let model = Interpolator::untrained(Compute::Cpu).unwrap();
    let (w, h) = (70, 45);
    let (a, b) = (frame(w, h, 0), frame(w, h, 2));
    let a = model.prepare(&image(w, h, &a)).unwrap();
    let b = model.prepare(&image(w, h, &b)).unwrap();
    assert_eq!(a.size(), (70, 45));
    assert_eq!(a.image.dims(), [1, 3, 64, 128]);
    assert_eq!(a.features.dims(), [1, FEATURES, 64, 128]);
    for at in [0.0, 0.25, 1.0] {
        assert_eq!(
            model.between(&a, &b, at).unwrap().len(),
            (w * h * 3) as usize
        );
    }
}

#[test]
fn a_warp_reads_the_pixel_the_flow_points_to_and_holds_the_edge() {
    let model = Interpolator::untrained(Compute::Cpu).unwrap();
    let (w, h) = (6, 4);
    let input: Vec<f32> = (0..w * h).map(|v| v as f32).collect();
    let input = Tensor::<4>::from_data(TensorData::new(input.clone(), [1, 1, h, w]), &model.device);
    let grid = model.grid(w, h);
    let still = Tensor::<4>::zeros([1, 2, h, w], &model.device);
    let same = values(model.warp(input.clone(), still, &grid));
    for (i, v) in same.iter().enumerate() {
        assert!((v - i as f32).abs() < 1e-4, "{i}: {v}");
    }
    // One pixel right and half a pixel down: each pixel reads its right
    // neighbour, half way to the row below, and the last column and row
    // hold the edge.
    let flow = Tensor::cat(
        vec![
            Tensor::<4>::ones([1, 1, h, w], &model.device),
            Tensor::<4>::full([1, 1, h, w], 0.5, &model.device),
        ],
        1,
    );
    let moved = values(model.warp(input, flow, &grid));
    for y in 0..h {
        for x in 0..w {
            let sx = (x + 1).min(w - 1) as f32;
            let sy = (y as f32 + 0.5).min((h - 1) as f32);
            let want = sy * w as f32 + sx;
            let got = moved[y * w + x];
            assert!((got - want).abs() < 1e-3, "({x}, {y}): {got}, not {want}");
        }
    }
}

#[test]
fn a_resize_samples_at_pixel_centres_as_pytorch_does() {
    let device = Device::flex();
    let x = Tensor::<4>::from_data(
        TensorData::new(
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
            [1, 1, 2, 4],
        ),
        &device,
    );
    // Halved, each output pixel sits between two inputs: their mean.
    assert_eq!(values(resize(x.clone(), [1, 2])), [5.0, 9.0]);
    assert_eq!(values(resize(x.clone(), [2, 4])), values(x));
}

#[test]
fn a_moment_outside_the_pair_or_frames_of_two_sizes_are_refused() {
    let model = Interpolator::untrained(Compute::Cpu).unwrap();
    let small = frame(8, 8, 0);
    let wide = frame(16, 8, 0);
    let a = model.prepare(&image(8, 8, &small)).unwrap();
    let b = model.prepare(&image(16, 8, &wide)).unwrap();
    assert!(model.between(&a, &b, 0.5).is_err());
    for at in [-0.1, 1.1, f32::NAN] {
        assert!(model.between(&a, &a, at).is_err(), "{at}");
    }
}

#[test]
fn a_frame_the_model_cannot_take_is_refused_before_any_work() {
    let model = Interpolator::untrained(Compute::Cpu).unwrap();
    let rgb = frame(4, 4, 0);
    let big = INTERPOLATE_MAX_SIDE + 1;
    for (width, height, bytes) in [
        (0, 4, &rgb[..0]),
        (4, 0, &rgb[..0]),
        (4, 4, &rgb[..47]),
        (big, 1, &rgb[..]),
        (1, big, &rgb[..]),
        (u32::MAX, u32::MAX, &rgb[..]),
    ] {
        assert!(
            matches!(
                model.prepare(&image(width, height, bytes)),
                Err(VisionError::Image { .. })
            ),
            "{width}x{height} with {} bytes",
            bytes.len()
        );
    }
    assert!(model.prepare(&image(4, 4, &rgb)).is_ok());
}

#[test]
#[ignore = "needs a Metal GPU"]
fn a_warp_on_the_gpu_reads_what_it_reads_on_the_cpu() {
    // burn's Metal grid sample once read a permuted grid as if it were in
    // order, which moved every pixel.
    let cpu = Interpolator::untrained(Compute::Cpu).unwrap();
    let gpu = Interpolator::untrained(Compute::Gpu).unwrap();
    let (w, h) = (96, 64);
    let input: Vec<f32> = (0..4 * w * h)
        .map(|i| ((i * 37 % 101) as f32) / 101.0)
        .collect();
    let flow: Vec<f32> = (0..2 * w * h)
        .map(|i| ((i * 13 % 29) as f32) - 14.0)
        .collect();
    let run = |model: &Interpolator| {
        let input =
            Tensor::<4>::from_data(TensorData::new(input.clone(), [1, 4, h, w]), &model.device);
        let flow =
            Tensor::<4>::from_data(TensorData::new(flow.clone(), [1, 2, h, w]), &model.device);
        values(model.warp(input, flow, &model.grid(w, h)))
    };
    let (a, b) = (run(&cpu), run(&gpu));
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst < 1e-4, "{worst}");
}

#[test]
fn saved_weights_load_back_into_the_same_model() {
    let dir = scratch("round-trip");
    let saved = Interpolator::untrained(Compute::Cpu).unwrap();
    let mut store = BurnpackStore::from_file(dir.join(INTERPOLATE_WEIGHTS)).overwrite(true);
    saved.model.save_into(&mut store).unwrap();
    let loaded = Interpolator::load(&dir, Compute::Cpu).unwrap();
    let (a, b) = (frame(9, 7, 0), frame(9, 7, 1));
    let run = |model: &Interpolator| {
        let a = model.prepare(&image(9, 7, &a)).unwrap();
        let b = model.prepare(&image(9, 7, &b)).unwrap();
        model.between(&a, &b, 0.5).unwrap()
    };
    assert_eq!(run(&saved), run(&loaded));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_or_damaged_weights_are_an_error_and_not_a_panic() {
    let dir = scratch("bad");
    let Err(VisionError::Weights { path, message }) = Interpolator::load(&dir, Compute::Cpu) else {
        panic!("loaded without weights");
    };
    assert_eq!(path, dir.join(INTERPOLATE_WEIGHTS));
    assert_eq!(message, "missing");
    std::fs::write(dir.join(INTERPOLATE_WEIGHTS), b"not a burnpack").unwrap();
    assert!(matches!(
        Interpolator::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    let pkl = dir.join("flownet.pkl");
    assert!(matches!(
        convert_rife(&pkl, &dir.join("out.bpk")),
        Err(VisionError::Weights { .. })
    ));
    std::fs::write(&pkl, b"not a pickle").unwrap();
    assert!(matches!(
        convert_rife(&pkl, &dir.join("out.bpk")),
        Err(VisionError::Weights { .. })
    ));
    assert!(!dir.join("out.bpk").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// How long one 720p in-between frame takes, and preparing a frame, written
/// to stderr: run with
/// `cargo test -p cutlass-vision --release interpolate_720p -- --ignored --nocapture`.
fn time_720p(compute: Compute) {
    let model = Interpolator::untrained(compute).unwrap();
    let (w, h) = (1280, 720);
    let (a, b) = (frame(w, h, 0), frame(w, h, 4));
    let start = Instant::now();
    let pa = model.prepare(&image(w, h, &a)).unwrap();
    let pb = model.prepare(&image(w, h, &b)).unwrap();
    model.between(&pa, &pb, 0.5).unwrap();
    let first = start.elapsed();
    let runs = 4;
    let start = Instant::now();
    for _ in 0..runs {
        model.prepare(&image(w, h, &b)).unwrap();
    }
    let prepare = start.elapsed() / runs;
    let start = Instant::now();
    for k in 0..runs {
        model.between(&pa, &pb, (k + 1) as f32 / 5.0).unwrap();
    }
    let each = start.elapsed() / runs;
    let _ = std::io::stderr().write_all(
        format!(
            "interpolate 720p on {compute:?}: first {first:?}, then {prepare:?} to prepare a frame and {each:?} a frame between\n"
        )
        .as_bytes(),
    );
}

#[test]
#[ignore = "needs a Metal GPU; a timing run, not a check"]
fn interpolate_720p_on_the_gpu() {
    time_720p(Compute::Gpu);
}

#[test]
#[ignore = "a timing run, not a check, and slow on the CPU"]
fn interpolate_720p_on_the_cpu() {
    time_720p(Compute::Cpu);
}

/// Set in a tagged run, the tests that need the real weights fail when
/// they are missing instead of skipping.
const REQUIRE: &str = "CUTLASS_REQUIRE_INTERPOLATE_MODEL";

fn local_models() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/models/interpolate")
}

/// Two frames, and what RIFE made between them under PyTorch at a quarter
/// and a half, written by `tools/vision-models/scripts/fetch_interpolate.sh`.
struct Reference {
    width: u32,
    height: u32,
    a: Vec<u8>,
    b: Vec<u8>,
    quarter: Vec<u8>,
    half: Vec<u8>,
}

fn reference() -> Option<Reference> {
    let dir = local_models();
    let file = dir.join("reference_pair.safetensors");
    if !dir.join(INTERPOLATE_WEIGHTS).is_file() || !file.is_file() {
        assert!(
            std::env::var_os(REQUIRE).is_none(),
            "{REQUIRE} is set and the interpolate weights or {} are missing",
            file.display()
        );
        return None;
    }
    let bytes = std::fs::read(&file).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let a = tensors.tensor("a").unwrap();
    let &[height, width, 3] = a.shape() else {
        panic!("a is {:?}", a.shape());
    };
    let take = |name: &str| {
        let tensor = tensors.tensor(name).unwrap();
        assert_eq!(tensor.shape(), a.shape(), "{name}");
        tensor.data().to_vec()
    };
    Some(Reference {
        width: width as u32,
        height: height as u32,
        a: take("a"),
        b: take("b"),
        quarter: take("quarter"),
        half: take("half"),
    })
}

fn worst_against_pytorch(compute: Compute) -> Option<u8> {
    let reference = reference()?;
    let (w, h) = (reference.width, reference.height);
    let model = Interpolator::load(&local_models(), compute).unwrap();
    let a = model.prepare(&image(w, h, &reference.a)).unwrap();
    let b = model.prepare(&image(w, h, &reference.b)).unwrap();
    let quarter = model.between(&a, &b, 0.25).unwrap();
    let half = model.between(&a, &b, 0.5).unwrap();
    // The model does move things: the middle frame is neither end.
    assert!(worst(&half, &reference.a) > 8 && worst(&half, &reference.b) > 8);
    Some(worst(&quarter, &reference.quarter).max(worst(&half, &reference.half)))
}

#[test]
fn the_real_weights_make_what_pytorch_makes() {
    if let Some(worst) = worst_against_pytorch(Compute::Cpu) {
        assert!(worst <= 1, "{worst} levels from PyTorch");
    }
}

#[test]
#[ignore = "needs a Metal GPU"]
fn the_real_weights_on_the_gpu_make_what_pytorch_makes() {
    if let Some(worst) = worst_against_pytorch(Compute::Gpu) {
        assert!(worst <= 1, "{worst} levels from PyTorch");
    }
}

#[test]
fn the_real_weights_leave_a_still_frame_still() {
    let Some(_) = reference() else {
        return;
    };
    let model = Interpolator::load(&local_models(), Compute::Cpu).unwrap();
    let rgb = frame(96, 64, 0);
    let a = model.prepare(&image(96, 64, &rgb)).unwrap();
    let still = model.between(&a, &a, 0.5).unwrap();
    // RIFE itself, as PyTorch runs it, finds a little motion in a pattern
    // that repeats every few pixels; on average it stays put.
    let moved: u64 = still
        .iter()
        .zip(&rgb)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    let mean = moved as f64 / rgb.len() as f64;
    assert!(mean < 0.25, "a still frame moved by {mean} on average");
}
