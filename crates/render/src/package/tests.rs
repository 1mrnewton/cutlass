use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use cutlass_effects::{
    AnimSpan, EffectSource, Located, PackageError, PackageKind, PackageStore, entries,
};
use cutlass_gpu::Placement;
use cutlass_model::{AnimType, MaterialEffect, Micros, TimeRange};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample};
use cutlass_storage::ResourceId;

use crate::{Layer, Renderer, Rgba};

fn renderer() -> Option<Renderer> {
    let renderer = Renderer::for_test().expect("a GPU is required");
    if renderer.is_none() {
        eprintln!("skipped: no GPU");
    }
    renderer
}

const SIZE: (u32, u32) = (23, 17);

/// Every catalog package, published from a scratch folder.
struct Store {
    folder: PathBuf,
    files: BTreeMap<String, Located>,
}

impl Store {
    fn new() -> Self {
        let folder = std::env::temp_dir().join(format!(
            "cutlass-render-package-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&folder).expect("folder");
        let mut files = BTreeMap::new();
        for row in entries() {
            let path = folder.join(row.id);
            fs::write(&path, b"zzzz").expect("package");
            files.insert(
                row.id.to_owned(),
                Located {
                    path,
                    size: 4,
                    checksum_hex: None,
                },
            );
        }
        Self { folder, files }
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.folder);
    }
}

impl PackageStore for Store {
    fn locate(&self, id: &ResourceId) -> Result<Located, PackageError> {
        self.files
            .get(id.as_str())
            .cloned()
            .ok_or_else(|| PackageError::Missing {
                id: id.as_str().to_owned(),
            })
    }
}

/// A premultiplied picture with every alpha from clear to opaque.
fn picture(seed: u32) -> CpuBytes {
    let bytes = (0..SIZE.0 * SIZE.1)
        .flat_map(|index| {
            let alpha = (index * 37 + seed * 11) % 256;
            let channel = |k: u32| ((index * k + seed * 29) % 256) * alpha / 255;
            [channel(13), channel(71), channel(5), alpha].map(|c| c as u8)
        })
        .collect();
    CpuBytes::new(SIZE.0, SIZE.1, bytes).expect("picture")
}

fn rgba(picture: &CpuBytes) -> Rgba<'_> {
    Rgba {
        width: picture.width(),
        height: picture.height(),
        bytes: picture.rgba(),
    }
}

fn source(
    store: &Store,
    id: &str,
    input: &CpuBytes,
    backdrop: &CpuBytes,
    anim: Option<AnimSpan>,
) -> EffectSource {
    let mut effect = MaterialEffect::new("fx");
    effect.effect_id = id.to_owned();
    EffectSource::open(
        store,
        &effect,
        &Picture::Cpu(input.clone()),
        Some(&Picture::Cpu(backdrop.clone())),
        TimeRange::new(0, 1_000_000),
        anim,
    )
    .expect("source")
}

#[test]
fn every_package_paints_as_its_cpu_stand_in() {
    let Some(renderer) = renderer() else { return };
    let store = Store::new();
    let (input, backdrop) = (picture(1), picture(2));
    let fades = [
        None,
        Some(AnimSpan {
            kind: AnimType::In,
            start: Micros(0),
            duration: Micros(1_000_000),
        }),
        Some(AnimSpan {
            kind: AnimType::Out,
            start: Micros(100_000),
            duration: Micros(700_000),
        }),
    ];
    let mut checked = 0;
    for row in entries() {
        if matches!(row.kind, PackageKind::Transition(_)) {
            continue;
        }
        for anim in fades {
            let label = format!("{} {anim:?}", row.id);
            let source = source(&store, row.id, &input, &backdrop, anim);
            let time = Micros(300_000);
            let Sample::Ready(Picture::Cpu(expected)) = source.sample(time) else {
                panic!("{label}: no CPU sample");
            };
            let pass = source.pass(time).expect("pass");
            let canvas = renderer.canvas(SIZE.0, SIZE.1).expect("canvas");
            let placement = Placement::canvas(SIZE.0, SIZE.1);
            let layers = match pass.blend {
                Some(mode) => vec![
                    Layer::Rgba(rgba(&backdrop), placement),
                    Layer::Blend {
                        mode,
                        layers: vec![Layer::Rgba(rgba(&input), placement), Layer::Package(&pass)],
                    },
                ],
                None => vec![Layer::Rgba(rgba(&input), placement), Layer::Package(&pass)],
            };
            renderer.draw(&canvas, &layers).expect("draw");
            let got = renderer.read(&canvas).expect("read");
            // Paint passes are exact; a blend mixes in floats as the
            // shipped blend modes do.
            let tolerance = if pass.blend.is_some() { 1 } else { 0 };
            for (index, (a, b)) in got.iter().zip(expected.rgba()).enumerate() {
                assert!(
                    a.abs_diff(*b) <= tolerance,
                    "{label}: byte {index} is {a}, the CPU has {b}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked >= 10, "{checked}");
}

#[test]
fn a_pixelated_preview_keeps_blocks_laid_out_on_the_full_canvas() {
    let Some(renderer) = renderer() else { return };
    let store = Store::new();
    let input = picture(3);
    let source = source(&store, cutlass_effects::PIXELATE, &input, &input, None);
    let pass = source.pass(Micros(0)).expect("pass");
    let small = crate::preview_size(SIZE, 0.4);
    let draw = |layers: &[Layer<'_>]| {
        let canvas = renderer.canvas(small.0, small.1).expect("canvas");
        renderer.draw_scaled(&canvas, layers, SIZE).expect("draw");
        renderer.read(&canvas).expect("read")
    };
    let placement = Placement::canvas(SIZE.0, SIZE.1);
    let shrunk = draw(&[Layer::Rgba(rgba(&input), placement)]);
    let got = draw(&[Layer::Rgba(rgba(&input), placement), Layer::Package(&pass)]);
    // Each target pixel takes the shrunk pixel where its full-canvas
    // block starts.
    let scale = (
        small.0 as f32 / SIZE.0 as f32,
        small.1 as f32 / SIZE.1 as f32,
    );
    let block = cutlass_effects::PIXELATE_BLOCK;
    let start = |p: u32, scale: f32, side: u32| {
        let full = ((p as f32 + 0.5) / scale).floor() as u32;
        let origin = full / block * block;
        (((origin as f32 + 0.5) * scale).floor() as u32).min(side - 1)
    };
    let mut moved = false;
    for y in 0..small.1 {
        for x in 0..small.0 {
            let (sx, sy) = (start(x, scale.0, small.0), start(y, scale.1, small.1));
            let at = ((y * small.0 + x) * 4) as usize;
            let from = ((sy * small.0 + sx) * 4) as usize;
            assert_eq!(got[at..at + 4], shrunk[from..from + 4], "pixel {x}, {y}");
            moved |= (sx, sy) != (x, y);
        }
    }
    assert!(moved, "some pixels take another's colour");
}

#[test]
fn a_pass_outside_its_time_or_for_a_transition_is_none() {
    let store = Store::new();
    let input = picture(4);
    let invert = source(&store, cutlass_effects::INVERT, &input, &input, None);
    assert!(invert.pass(Micros(1_000_000)).is_none());
    assert!(invert.pass(Micros(-1)).is_none());
    assert!(invert.pass(Micros(0)).is_some());
    let fade = source(
        &store,
        cutlass_effects::TRANSITION_FADE,
        &input,
        &input,
        None,
    );
    assert!(fade.pass(Micros(0)).is_none());
    let mut effect = MaterialEffect::new("fx");
    effect.effect_id = "not-published".to_owned();
    let absent = EffectSource::open(
        &store,
        &effect,
        &Picture::Cpu(input.clone()),
        None,
        TimeRange::new(0, 1_000_000),
        None,
    )
    .expect("absent");
    assert!(absent.pass(Micros(0)).is_none());
}
