use std::path::PathBuf;
use std::time::Instant;

use super::*;
use crate::MATTE_MAX_SIDE;
use crate::matte_frame::real;

const REQUIRE: &str = "CUTLASS_REQUIRE_PORTRAIT_MODEL";

fn local_models() -> PathBuf {
    real::models("portrait")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-portrait-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn image(width: u32, height: u32, rgb: &[u8]) -> Image<'_> {
    Image { width, height, rgb }
}

/// A frame with something in it: a slope of light and a square.
fn frame(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let inside = x > width / 3 && x < width * 2 / 3 && y > height / 3;
            let shade = (x * 255 / width.max(1)) as u8;
            rgb.extend_from_slice(&if inside {
                [200, 60, 40]
            } else {
                [shade, 120, 255 - shade]
            });
        }
    }
    rgb
}

/// The frame and what onnxruntime made of it, from
/// `tools/vision-models/scripts/fetch_portrait.sh`.
fn reference() -> Option<(Vec<u8>, Vec<f32>)> {
    let dir = local_models();
    let file = dir.join("reference_frame.safetensors");
    if !dir.join(PORTRAIT_WEIGHTS).is_file() || !file.is_file() {
        assert!(
            std::env::var_os(REQUIRE).is_none(),
            "{REQUIRE} is set and the portrait weights or {} are missing",
            file.display()
        );
        return None;
    }
    let bytes = std::fs::read(&file).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let rgb = tensors.tensor("rgb").unwrap();
    assert_eq!(rgb.shape(), [640, 512, 3]);
    let matte = tensors.tensor("matte").unwrap();
    assert_eq!(matte.shape(), [640, 512]);
    let matte = matte
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Some((rgb.data().to_vec(), matte))
}

/// The worst and the mean difference from onnxruntime's share kept.
fn against_onnxruntime(compute: Compute) -> Option<(f32, f32)> {
    let (rgb, expected) = reference()?;
    let model = Portrait::load(&local_models(), compute).unwrap();
    let frame = image(512, 640, &rgb);
    assert_eq!(Portrait::read_size(512, 640), (512, 640));
    let got = matte_frame::values(model.kept(&frame).unwrap()).unwrap();
    assert_eq!(got.len(), expected.len());
    let kept = expected.iter().filter(|v| **v > 0.5).count();
    assert!(
        kept > expected.len() / 10,
        "onnxruntime kept too little: {kept}"
    );
    let errors: Vec<f32> = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let worst = errors.iter().copied().fold(0.0, f32::max);
    let mean = errors.iter().sum::<f32>() / errors.len() as f32;
    Some((worst, mean))
}

#[test]
fn the_cpu_matches_onnxruntime() {
    if let Some((worst, mean)) = against_onnxruntime(Compute::Cpu) {
        real::say(&format!("modnet cpu: worst {worst}, mean {mean}"));
        assert!(worst < 2e-3 && mean < 1e-5, "worst {worst}, mean {mean}");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn the_gpu_matches_onnxruntime() {
    // Trusted as the editor trusts a GPU that passed the check.
    crate::trust_gpu(crate::check_gpu().ok());
    if let Some((worst, mean)) = against_onnxruntime(Compute::Gpu) {
        real::say(&format!("modnet gpu: worst {worst}, mean {mean}"));
        assert!(worst < 2e-3 && mean < 1e-5, "worst {worst}, mean {mean}");
    }
}

#[test]
fn a_matte_comes_back_at_the_frame_size() {
    let mut model = Portrait::untrained(Compute::Cpu).unwrap();
    assert_eq!(model.id(), PORTRAIT_MODEL);
    let rgb = frame(70, 40);
    let matte = model.matte(&image(70, 40, &rgb)).unwrap();
    assert_eq!((matte.width, matte.height), (70, 40));
    assert_eq!(matte.alpha.len(), 70 * 40);
    assert_eq!(Portrait::read_size(70, 40), (896, 512));
    assert_eq!(Portrait::read_size(1920, 1080), (896, 512));
}

#[test]
fn an_empty_wrong_sized_or_huge_frame_is_refused() {
    let mut model = Portrait::untrained(Compute::Cpu).unwrap();
    for (w, h, bytes) in [
        (0, 4, 0),
        (4, 4, 47),
        (4, 4, 49),
        (MATTE_MAX_SIDE + 1, 1, 3 * (MATTE_MAX_SIDE as usize + 1)),
    ] {
        let rgb = vec![0_u8; bytes];
        assert!(
            matches!(
                model.matte(&image(w, h, &rgb)),
                Err(VisionError::Image { .. })
            ),
            "{w}x{h} with {bytes}"
        );
    }
}

#[test]
fn saved_weights_load_back_and_damaged_or_missing_ones_are_refused() {
    let dir = scratch("round-trip");
    let mut saved = Portrait::untrained(Compute::Cpu).unwrap();
    let mut store = BurnpackStore::from_file(dir.join(PORTRAIT_WEIGHTS)).overwrite(true);
    saved.model.save_into(&mut store).unwrap();
    let mut loaded = Portrait::load(&dir, Compute::Cpu).unwrap();
    let rgb = frame(64, 64);
    assert_eq!(
        saved.matte(&image(64, 64, &rgb)).unwrap(),
        loaded.matte(&image(64, 64, &rgb)).unwrap()
    );
    std::fs::write(dir.join(PORTRAIT_WEIGHTS), b"not weights").unwrap();
    assert!(matches!(
        Portrait::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    std::fs::remove_file(dir.join(PORTRAIT_WEIGHTS)).unwrap();
    assert!(matches!(
        Portrait::load(&dir, Compute::Cpu),
        Err(VisionError::Weights { .. })
    ));
    assert!(convert_modnet(&dir.join("none.safetensors"), &dir.join("out.bpk")).is_err());
    std::fs::write(dir.join("bad.safetensors"), b"nope").unwrap();
    assert!(convert_modnet(&dir.join("bad.safetensors"), &dir.join("out.bpk")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

fn time_real_frames(compute: Compute) {
    let Ok(mut model) = Portrait::load(&local_models(), compute) else {
        return;
    };
    for frame in real::frames() {
        let image = image(frame.width, frame.height, &frame.rgb);
        let matte = model.matte(&image).unwrap();
        let runs = 5;
        let started = Instant::now();
        for _ in 0..runs {
            model.matte(&image).unwrap();
        }
        let each = started.elapsed() / runs;
        let (w, h) = Portrait::read_size(frame.width, frame.height);
        real::say(&format!(
            "modnet {} {}x{} read at {w}x{h} on {compute:?}: {each:?} a frame",
            frame.name, frame.width, frame.height
        ));
        real::save(&frame.name, PORTRAIT_MODEL, &matte);
    }
}

#[test]
#[ignore = "timing on real frames; run with --release --ignored --nocapture"]
fn portrait_real_frames_on_the_gpu() {
    time_real_frames(Compute::Gpu);
}

#[test]
#[ignore = "timing on real frames; run with --release --ignored --nocapture"]
fn portrait_real_frames_on_the_cpu() {
    time_real_frames(Compute::Cpu);
}
