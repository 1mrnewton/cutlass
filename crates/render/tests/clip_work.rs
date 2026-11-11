//! A clip's colour work, mask and selection, and a look or adjustment
//! layer's colour work, draw the same frame on the GPU path as on the CPU.

#![cfg(test)]

use cutlass_effects::{Located, Mask, MaskShape, PackageError, PackageStore, Selection};
use cutlass_eval::{GpuSource, Micros, OpenClips, ReaderFactory, Snapshot};
use cutlass_model::{
    Draft, MaterialColorConfig, MaterialEffect, Segment, SegmentId, TimeRange, Track, TrackType,
};
use cutlass_render::Renderer;
use cutlass_renderer_software::SoftwareDevice;
use cutlass_session::{Request, Session};
use cutlass_storage::ResourceId;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 40;

fn renderer() -> Option<Renderer> {
    let renderer = Renderer::for_test().expect("a GPU is required");
    if renderer.is_none() {
        eprintln!("skipped: no GPU");
    }
    renderer
}

/// Two colour clips on a grey canvas: a blue one under the whole frame and
/// an orange one, shrunk, turned and moved off the middle, over it.
fn scene() -> Draft {
    let mut draft = Draft::empty("draft");
    draft.canvas_config.width = WIDTH as i32;
    draft.canvas_config.height = HEIGHT as i32;
    draft.canvas_config.background = Some("#808080".to_owned());
    for (id, colour) in [("blue", "0xFF2A5FB0"), ("orange", "0xFFE08A2C")] {
        draft.materials.material_colors.push(MaterialColorConfig {
            id: id.into(),
            is_color_clip: true,
            solid_color: colour.to_owned(),
            ..MaterialColorConfig::default()
        });
    }
    draft.tracks[0]
        .segments
        .push(Segment::new("under", "blue", TimeRange::new(0, 1_000_000)));
    let mut over = Segment::new("clip", "orange", TimeRange::new(0, 1_000_000));
    over.clip.scale = cutlass_model::Point::new(0.6, 0.7);
    over.clip.transform = cutlass_model::Point::new(0.2, -0.1);
    over.clip.rotation = 12.0;
    let mut row = Track::main("over");
    row.segments.push(over);
    draft.tracks.push(row);
    draft
}

fn apply(draft: Draft, requests: &[Request]) -> Draft {
    let mut session = Session::open(draft);
    session.apply(0, requests).expect("requests");
    session.draft().clone()
}

fn clip() -> SegmentId {
    SegmentId::new("clip")
}

/// The frame on both paths, with no effect packages.
fn both(renderer: &Renderer, draft: &Draft, time: i64) -> (Vec<u8>, Vec<u8>, Vec<GpuSource>) {
    both_with(renderer, draft, time, &cutlass_effects::MissingPackages)
}

/// The frame on both paths, packages read from `store`. The GPU one must
/// not fall back to the CPU.
fn both_with(
    renderer: &Renderer,
    draft: &Draft,
    time: i64,
    store: &dyn PackageStore,
) -> (Vec<u8>, Vec<u8>, Vec<GpuSource>) {
    let media = ReaderFactory::new();
    let snapshot = Snapshot::with_packages(draft, store);
    let mut device = SoftwareDevice::new();
    let mut clips = OpenClips::new();
    let cpu = cutlass_eval::frame(&snapshot, Micros(time), &mut device, &media, &mut clips, 0)
        .expect("cpu frame");
    let layers = cutlass_eval::layers_in(
        &snapshot,
        Micros(time),
        &mut device,
        &media,
        &mut clips,
        0,
        None,
    )
    .expect("layers");
    let canvas = renderer.canvas(WIDTH, HEIGHT).expect("canvas");
    renderer.draw_frame(&canvas, &layers).expect("draw");
    let sources = layers
        .layers
        .iter()
        .map(|layer| layer.source.clone())
        .collect();
    (renderer.read(&canvas).expect("read"), cpu.rgba, sources)
}

/// Every byte within two of the CPU's and nearly all within one.
fn agree(label: &str, gpu: &[u8], cpu: &[u8]) {
    assert_eq!(gpu.len(), cpu.len(), "{label}");
    let mut near = 0;
    for (index, (a, b)) in gpu.iter().zip(cpu).enumerate() {
        let off = a.abs_diff(*b);
        assert!(
            off <= 2,
            "{label}: byte {index} (pixel {}, {}) is {a} on the GPU and {b} on the CPU",
            (index / 4) as u32 % WIDTH,
            (index / 4) as u32 / WIDTH,
        );
        near += usize::from(off <= 1);
    }
    assert!(
        near * 100 >= gpu.len() * 98,
        "{label}: {near} bytes within one"
    );
}

fn check(renderer: &Renderer, label: &str, draft: &Draft) -> Vec<GpuSource> {
    let plain = both(renderer, &scene(), 0).1;
    let (gpu, cpu, sources) = both(renderer, draft, 40_000);
    assert_ne!(cpu, plain, "{label} changes the frame");
    agree(label, &gpu, &cpu);
    sources
}

fn adjust(key: &str, value: f64, id: &str) -> Request {
    Request::SetAdjust {
        segment_id: clip(),
        material_id: id.into(),
        key: key.to_owned(),
        value,
    }
}

#[test]
fn a_clips_sliders_match_inside_its_colour_mask() {
    let Some(renderer) = renderer() else { return };
    let draft = apply(
        scene(),
        &[
            adjust("exposure", 0.4, "a"),
            adjust("contrast", 0.3, "b"),
            adjust("saturation", -0.4, "c"),
            adjust("clear", 0.6, "d"),
            adjust("sharpen", 0.5, "e"),
            adjust("vignetting", 0.7, "f"),
            adjust("particle", 0.5, "g"),
        ],
    );
    check(&renderer, "sliders", &draft);
    let masked = apply(
        draft,
        &[Request::SetColorMask {
            segment_id: clip(),
            material_id: "where".into(),
            mask: Some(Mask {
                feather: 0.5,
                rotation: 30.0,
                ..Mask::new(MaskShape::Rectangle)
            }),
        }],
    );
    check(&renderer, "sliders in a colour mask", &masked);
}

#[test]
fn a_clips_grade_lut_and_key_match() {
    let Some(renderer) = renderer() else { return };
    let mut cube = String::from("LUT_3D_SIZE 3\n");
    for b in 0..3 {
        for g in 0..3 {
            for r in 0..3 {
                let at = |index: i32| f64::from(index) / 2.0;
                cube.push_str(&format!("{} {} {}\n", at(g), at(b) * 0.8, at(r)));
            }
        }
    }
    let draft = apply(
        scene(),
        &[
            Request::SetCurve {
                segment_id: clip(),
                material_id: "curves".into(),
                channel: "rgb".to_owned(),
                points: vec![[0.0, 0.1], [0.5, 0.6], [1.0, 0.9]],
            },
            Request::SetWheel {
                segment_id: clip(),
                material_id: "wheels".into(),
                wheel: "gain".to_owned(),
                x: 0.3,
                y: -0.4,
                luma: 0.1,
            },
            Request::SetHsl {
                segment_id: clip(),
                material_id: "hsl-orange".into(),
                color: 1,
                hue: 0.6,
                saturation: -0.3,
                lightness: 0.2,
            },
            Request::SetLut {
                segment_id: clip(),
                material_id: "lut".into(),
                name: "turn".to_owned(),
                cube: Some(cube),
                strength: 0.6,
            },
            Request::SetChroma {
                segment_id: SegmentId::new("under"),
                material_id: "key".into(),
                color: "#2a5fb0".to_owned(),
                strength: 0.4,
                shadow: 0.3,
            },
        ],
    );
    check(&renderer, "grade, lut and key", &draft);
}

#[test]
fn a_clips_mask_and_selection_match() {
    let Some(renderer) = renderer() else { return };
    let masked = apply(
        scene(),
        &[Request::SetMask {
            segment_id: clip(),
            material_id: "mask".into(),
            mask: Some(Mask {
                feather: 0.4,
                ..Mask::new(MaskShape::Heart)
            }),
        }],
    );
    check(&renderer, "mask", &masked);
    // A diagonal object, on a frame smaller than the clip shows.
    let (w, h) = (24u32, 16u32);
    let alpha: Vec<u8> = (0..w * h)
        .map(|index| if index % w + index / w > 14 { 255 } else { 0 })
        .collect();
    let selection = Selection::from_alpha(w, h, &alpha, Micros(0)).expect("selection");
    let selected = apply(
        masked,
        &[
            Request::SetSelection {
                segment_id: clip(),
                material_id: "object".into(),
                selection: Some(selection),
            },
            adjust("exposure", -0.5, "dark"),
        ],
    );
    check(&renderer, "colour, mask and selection", &selected);
}

#[test]
fn an_adjustment_layer_matches_over_the_clips_below_it() {
    let Some(renderer) = renderer() else { return };
    let layer = SegmentId::new("layer");
    let draft = apply(
        scene(),
        &[
            Request::AddAdjustment {
                track_id: "look".into(),
                segment_id: layer.clone(),
                material_id: "adj".into(),
                start: Micros(0),
                duration: Micros(1_000_000),
            },
            Request::SetAdjust {
                segment_id: layer.clone(),
                material_id: "warm".into(),
                key: "temperature".to_owned(),
                value: 0.7,
            },
            Request::SetAdjust {
                segment_id: layer.clone(),
                material_id: "fade".into(),
                key: "fade".to_owned(),
                value: 0.4,
            },
            Request::SetColorMask {
                segment_id: layer,
                material_id: "where".into(),
                mask: Some(Mask::new(MaskShape::Circle)),
            },
        ],
    );
    let sources = check(&renderer, "adjustment layer", &draft);
    assert!(
        matches!(
            sources.last(),
            Some(GpuSource::Colour { mask: Some(_), .. })
        ),
        "{sources:?}"
    );
}

#[test]
fn a_bundled_filter_matches_over_the_clips_below_it() {
    let Some(renderer) = renderer() else { return };
    let draft = apply(
        scene(),
        &[Request::AddFilter {
            track_id: "looks".into(),
            segment_id: SegmentId::new("look"),
            material_id: "f".into(),
            start: Micros(0),
            duration: Micros(1_000_000),
            filter_id: "look/bright".to_owned(),
        }],
    );
    check(&renderer, "filter", &draft);
}

/// Every catalog package, published from a scratch folder.
struct Store {
    folder: std::path::PathBuf,
    files: std::collections::BTreeMap<String, Located>,
}

impl Store {
    fn new(label: &str) -> Self {
        let folder =
            std::env::temp_dir().join(format!("cutlass-clip-work-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&folder).expect("folder");
        let mut files = std::collections::BTreeMap::new();
        for row in cutlass_effects::entries() {
            let path = folder.join(row.id);
            std::fs::write(&path, b"zzzz").expect("package");
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
        let _ = std::fs::remove_dir_all(&self.folder);
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

/// An effect segment running `package` over everything below it.
fn global(draft: &mut Draft, id: &str, package: &str, render_index: i32) {
    let mut effect = MaterialEffect::new(id);
    effect.effect_id = package.to_owned();
    draft.materials.effects.push(effect);
    let mut segment = Segment::new(id, id, TimeRange::new(0, 1_000_000));
    segment.render_index = render_index;
    let mut track = Track::main(id);
    track.kind = TrackType::Effect;
    track.segments.push(segment);
    draft.tracks.push(track);
}

/// `package` on the orange clip itself.
fn on_clip(draft: &mut Draft, id: &str, package: &str) {
    let mut effect = MaterialEffect::new(id);
    effect.effect_id = package.to_owned();
    draft.materials.effects.push(effect);
    draft.tracks[1].segments[0]
        .extra_material_refs
        .push(id.into());
}

/// `draft` on both paths, which must agree, and differ from `draft`
/// without its packages.
fn packaged(renderer: &Renderer, store: &Store, label: &str, draft: &Draft) -> Vec<GpuSource> {
    let mut unpackaged = draft.clone();
    unpackaged.materials.effects.clear();
    let plain = both_with(renderer, &unpackaged, 300_000, store).1;
    let (gpu, cpu, sources) = both_with(renderer, draft, 300_000, store);
    assert_ne!(cpu, plain, "{label} changes the frame");
    agree(label, &gpu, &cpu);
    sources
}

#[test]
fn effect_packages_over_the_clips_below_match() {
    let Some(renderer) = renderer() else { return };
    let store = Store::new("global");
    // A feathered edge, so pixelating has steps to keep.
    let edged = apply(
        scene(),
        &[Request::SetMask {
            segment_id: clip(),
            material_id: "mask".into(),
            mask: Some(Mask {
                feather: 0.6,
                ..Mask::new(MaskShape::Heart)
            }),
        }],
    );
    // Contrast doubles the edge's one-level rounding, so it runs on the
    // flat clips, which both paths draw alike.
    for (package, base) in [
        (cutlass_effects::INVERT, &edged),
        (cutlass_effects::CONTRAST, &scene()),
        (cutlass_effects::PIXELATE, &edged),
    ] {
        let mut draft = base.clone();
        global(&mut draft, "fx", package, 11_000);
        let sources = packaged(&renderer, &store, package, &draft);
        assert!(
            matches!(sources.last(), Some(GpuSource::Package(_))),
            "{package}: {sources:?}"
        );
    }
    // Two, with a clip drawn between them.
    let mut draft = scene();
    draft.tracks[1].segments[0].render_index = 12_000;
    global(&mut draft, "first", cutlass_effects::PIXELATE, 11_000);
    global(&mut draft, "second", cutlass_effects::INVERT, 13_000);
    packaged(&renderer, &store, "pixelate under, invert over", &draft);
}

#[test]
fn a_clips_own_package_runs_after_its_colour_and_mask() {
    let Some(renderer) = renderer() else { return };
    let store = Store::new("clip");
    let mut draft = apply(
        scene(),
        &[
            adjust("exposure", 0.4, "a"),
            Request::SetMask {
                segment_id: clip(),
                material_id: "mask".into(),
                mask: Some(Mask {
                    feather: 0.3,
                    ..Mask::new(MaskShape::Circle)
                }),
            },
        ],
    );
    on_clip(&mut draft, "fx", cutlass_effects::INVERT);
    packaged(&renderer, &store, "inverted clip", &draft);
    let mut faded = scene();
    on_clip(&mut faded, "fx", cutlass_effects::ANIM_IN);
    packaged(&renderer, &store, "fading in", &faded);
}

#[test]
fn a_blend_package_mixes_its_clip_with_the_clips_below() {
    let Some(renderer) = renderer() else { return };
    let store = Store::new("blend");
    for row in cutlass_effects::entries() {
        if !matches!(row.kind, cutlass_effects::PackageKind::Blend(_)) {
            continue;
        }
        let mut draft = scene();
        on_clip(&mut draft, "fx", row.id);
        let (gpu, cpu, sources) = both_with(&renderer, &draft, 300_000, &store);
        agree(row.id, &gpu, &cpu);
        assert!(
            sources
                .iter()
                .any(|source| matches!(source, GpuSource::Blend { .. })),
            "{}: {sources:?}",
            row.id
        );
    }
}

#[test]
fn a_package_the_store_does_not_have_leaves_the_picture() {
    let Some(renderer) = renderer() else { return };
    let store = Store::new("none");
    let mut draft = scene();
    global(&mut draft, "fx", "not-published", 11_000);
    on_clip(&mut draft, "mul", "not-published-either");
    let (gpu, cpu, sources) = both_with(&renderer, &draft, 300_000, &store);
    assert_eq!(cpu, both_with(&renderer, &scene(), 300_000, &store).1);
    agree("missing", &gpu, &cpu);
    assert!(
        !sources
            .iter()
            .any(|source| matches!(source, GpuSource::Package(_) | GpuSource::Blend { .. })),
        "{sources:?}"
    );
}
