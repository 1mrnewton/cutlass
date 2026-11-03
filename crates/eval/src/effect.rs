//! Effect packages on one evaluated frame.
//!
//! A drawn segment whose extra material names a package is filtered.
//! A distortion or a text animation replaces that segment's picture.
//! A blend uses the picture already composited below as the backdrop
//! and replaces the frame with the blend. That segment is not drawn again.
//!
//! An effect, video-effect, or plugin segment that is not itself drawn
//! filters the picture below it. A blend on that segment is not applied:
//! the segment has no picture of its own, so the picture below stays.
//!
//! A missing package, a package the store cannot read, or a sample that
//! misses leaves the picture unchanged. The draft is not written.
//! [`crate::Snapshot::new`] has no package store, so those frames skip
//! this pass.

use cutlass_compositor::{ComposeError, Device, Extent, Frame, submit};
use cutlass_effects::{
    EffectSource, PackageKind, PackageStore, animation_package_id, entry, package_id,
    plugin_package_id, segment_package_id, video_package_id,
};
use cutlass_model::{Draft, MaterialRef, Micros, Segment, TrackAttribute};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample, rgba_len};

use crate::EvalError;
use crate::canvas::project_canvas;

pub(crate) struct StackLayer<'a> {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    source: &'a dyn PictureSource,
}

impl<'a> StackLayer<'a> {
    pub(crate) fn new(
        render_index: i32,
        track_index: usize,
        segment_index: usize,
        source: &'a dyn PictureSource,
    ) -> Self {
        Self {
            render_index,
            track_index,
            segment_index,
            source,
        }
    }
}

struct Mark<'a> {
    render_index: i32,
    track_index: usize,
    segment_index: usize,
    segment: &'a Segment,
}

enum ClipFilter {
    None,
    Blend,
    Unary,
}

enum Slot<'a> {
    Borrowed(&'a dyn PictureSource),
    Owned(CpuBytes),
}

struct Held(CpuBytes);

impl PictureSource for Held {
    fn sample(&self, _time: Micros) -> Sample {
        Sample::Ready(Picture::Cpu(self.0.clone()))
    }
}

pub(crate) fn present<D: Device>(
    draft: &Draft,
    store: &dyn PackageStore,
    time: Micros,
    device: &mut D,
    extent: Extent,
    stack: &[StackLayer<'_>],
) -> Result<Frame, EvalError<D::Error>> {
    let marks = global_marks(draft, time, stack);
    let filtered = stack.iter().any(|layer| {
        !matches!(
            clip_filter(draft, layer.track_index, layer.segment_index),
            ClipFilter::None
        )
    });
    let canvas = project_canvas(
        draft.canvas_config.background.as_deref(),
        extent.width,
        extent.height,
    );
    if marks.is_empty() && !filtered {
        let held = canvas.map(Held);
        let mut sources: Vec<&dyn PictureSource> = Vec::new();
        if let Some(held) = held.as_ref() {
            sources.push(held);
        }
        sources.extend(stack.iter().map(|layer| layer.source));
        return submit(device, extent, time, &sources).map_err(EvalError::Compose);
    }

    let mut pass = Pass {
        draft,
        store,
        time,
        device,
        extent,
    };
    let mut base = canvas;
    let mut batch = Vec::new();
    let mut visual_at = 0;
    let mut mark_at = 0;
    loop {
        let visual = stack.get(visual_at);
        let mark = marks.get(mark_at);
        let take_visual = match (visual, mark) {
            (None, None) => break,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(layer), Some(mark)) => {
                (layer.render_index, layer.track_index, layer.segment_index)
                    <= (mark.render_index, mark.track_index, mark.segment_index)
            }
        };
        if take_visual {
            let Some(layer) = visual else {
                break;
            };
            visual_at += 1;
            push_visual(&mut pass, layer, &mut base, &mut batch)?;
        } else {
            let Some(mark) = mark else {
                break;
            };
            mark_at += 1;
            let frame = paint(pass.device, pass.extent, pass.time, base.take(), &batch)?;
            batch.clear();
            let bytes = cpu_of(frame, pass.extent)?;
            base = Some(apply_unary(
                pass.store,
                pass.draft,
                mark.segment,
                bytes,
                pass.extent,
                pass.time,
            ));
        }
    }
    paint(pass.device, pass.extent, pass.time, base, &batch)
}

struct Pass<'a, D: Device> {
    draft: &'a Draft,
    store: &'a dyn PackageStore,
    time: Micros,
    device: &'a mut D,
    extent: Extent,
}

fn push_visual<'a, D: Device>(
    pass: &mut Pass<'_, D>,
    layer: &StackLayer<'a>,
    base: &mut Option<CpuBytes>,
    batch: &mut Vec<Slot<'a>>,
) -> Result<(), EvalError<D::Error>> {
    let filter = clip_filter(pass.draft, layer.track_index, layer.segment_index);
    let Some(segment) = segment(pass.draft, layer.track_index, layer.segment_index) else {
        batch.push(Slot::Borrowed(layer.source));
        return Ok(());
    };
    match filter {
        ClipFilter::None => batch.push(Slot::Borrowed(layer.source)),
        ClipFilter::Unary => {
            if let Some(input) = sample_cpu(layer.source, pass.time) {
                batch.push(Slot::Owned(apply_unary(
                    pass.store,
                    pass.draft,
                    segment,
                    input,
                    pass.extent,
                    pass.time,
                )));
            }
        }
        ClipFilter::Blend => {
            let frame = paint(pass.device, pass.extent, pass.time, base.take(), batch)?;
            batch.clear();
            let backdrop = cpu_of(frame, pass.extent)?;
            let Some(input) = sample_cpu(layer.source, pass.time) else {
                *base = Some(backdrop);
                return Ok(());
            };
            *base = Some(
                match try_blend(
                    pass.store,
                    pass.draft,
                    segment,
                    &input,
                    &backdrop,
                    pass.extent,
                    pass.time,
                ) {
                    Some(blended) => blended,
                    None => {
                        batch.push(Slot::Owned(input));
                        backdrop
                    }
                },
            );
        }
    }
    Ok(())
}

fn paint<D: Device>(
    device: &mut D,
    extent: Extent,
    time: Micros,
    base: Option<CpuBytes>,
    batch: &[Slot<'_>],
) -> Result<Frame, EvalError<D::Error>> {
    if batch.is_empty()
        && let Some(bytes) = base
    {
        return Ok(Frame {
            extent,
            rgba: bytes.into_rgba(),
        });
    }
    let base_held = base.map(Held);
    let mut owned = Vec::new();
    for slot in batch {
        if let Slot::Owned(bytes) = slot {
            owned.push(Held(bytes.clone()));
        }
    }
    let mut refs: Vec<&dyn PictureSource> = Vec::new();
    if let Some(held) = base_held.as_ref() {
        refs.push(held);
    }
    let mut owned_at = 0;
    for slot in batch {
        match slot {
            Slot::Borrowed(source) => refs.push(*source),
            Slot::Owned(_) => {
                let Some(held) = owned.get(owned_at) else {
                    return Err(EvalError::Compose(ComposeError::Extent));
                };
                refs.push(held);
                owned_at += 1;
            }
        }
    }
    submit(device, extent, time, &refs).map_err(EvalError::Compose)
}

fn global_marks<'a>(draft: &'a Draft, time: Micros, stack: &[StackLayer<'_>]) -> Vec<Mark<'a>> {
    let mut marks = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            if !segment.visible || drawn(stack, track_index, segment_index) {
                continue;
            }
            if !covers(segment.target.start.0, segment.target.duration.0, time.0) {
                continue;
            }
            if !is_effect_material(draft, segment) {
                continue;
            }
            let Some(id) = segment_package_id(draft, segment) else {
                continue;
            };
            if matches!(entry(id).map(|row| row.kind), Some(PackageKind::Blend(_))) {
                continue;
            }
            marks.push(Mark {
                render_index: segment.render_index,
                track_index,
                segment_index,
                segment,
            });
        }
    }
    marks.sort_by_key(|mark| (mark.render_index, mark.track_index, mark.segment_index));
    marks
}

fn drawn(stack: &[StackLayer<'_>], track_index: usize, segment_index: usize) -> bool {
    stack
        .iter()
        .any(|layer| layer.track_index == track_index && layer.segment_index == segment_index)
}

fn is_effect_material(draft: &Draft, segment: &Segment) -> bool {
    matches!(
        draft.materials.get(segment.material_id.as_str()),
        Some(MaterialRef::Effect(_) | MaterialRef::VideoEffect(_) | MaterialRef::Plugin(_))
    )
}

fn clip_filter(draft: &Draft, track_index: usize, segment_index: usize) -> ClipFilter {
    let Some(segment) = segment(draft, track_index, segment_index) else {
        return ClipFilter::None;
    };
    let Some(id) = extra_package(draft, segment) else {
        return ClipFilter::None;
    };
    match entry(id).map(|row| row.kind) {
        Some(PackageKind::Blend(_)) => ClipFilter::Blend,
        Some(PackageKind::Transition(_)) => ClipFilter::None,
        Some(PackageKind::Distort(_) | PackageKind::TextAnim(_)) | None => ClipFilter::Unary,
    }
}

fn extra_package<'a>(draft: &'a Draft, segment: &Segment) -> Option<&'a str> {
    segment
        .extra_material_refs
        .iter()
        .find_map(|id| draft.materials.get(id.as_str()).and_then(package_of))
}

fn package_of(material: MaterialRef<'_>) -> Option<&str> {
    match material {
        MaterialRef::Effect(effect) => package_id(effect),
        MaterialRef::VideoEffect(effect) => video_package_id(effect),
        MaterialRef::Plugin(effect) => plugin_package_id(effect),
        MaterialRef::Animations(group) => group.animations.iter().find_map(animation_package_id),
        MaterialRef::Video(_)
        | MaterialRef::Audio(_)
        | MaterialRef::Text(_)
        | MaterialRef::Speed(_)
        | MaterialRef::Fade(_)
        | MaterialRef::Transition(_)
        | MaterialRef::Canvas(_)
        | MaterialRef::Draft(_)
        | MaterialRef::Hsl(_)
        | MaterialRef::Shape(_)
        | MaterialRef::Color(_)
        | MaterialRef::Other(_)
        | MaterialRef::Extra(_) => None,
    }
}

fn segment(draft: &Draft, track_index: usize, segment_index: usize) -> Option<&Segment> {
    draft.tracks.get(track_index)?.segments.get(segment_index)
}

fn apply_unary(
    store: &dyn PackageStore,
    draft: &Draft,
    segment: &Segment,
    input: CpuBytes,
    extent: Extent,
    time: Micros,
) -> CpuBytes {
    let picture = Picture::Cpu(input.clone());
    let Ok(source) = EffectSource::open_segment(store, draft, segment, &picture, None, None) else {
        return input;
    };
    match source.sample(time) {
        Sample::Ready(Picture::Cpu(bytes)) if canvas_bytes(&bytes, extent) => bytes,
        Sample::Ready(_) | Sample::Miss => input,
    }
}

fn try_blend(
    store: &dyn PackageStore,
    draft: &Draft,
    segment: &Segment,
    input: &CpuBytes,
    backdrop: &CpuBytes,
    extent: Extent,
    time: Micros,
) -> Option<CpuBytes> {
    let input = Picture::Cpu(input.clone());
    let backdrop = Picture::Cpu(backdrop.clone());
    let source =
        EffectSource::open_segment(store, draft, segment, &input, Some(&backdrop), None).ok()?;
    match source.sample(time) {
        Sample::Ready(Picture::Cpu(bytes)) if canvas_bytes(&bytes, extent) => Some(bytes),
        Sample::Ready(_) | Sample::Miss => None,
    }
}

fn canvas_bytes(bytes: &CpuBytes, extent: Extent) -> bool {
    bytes.width() == extent.width
        && bytes.height() == extent.height
        && rgba_len(extent.width, extent.height).is_some_and(|len| bytes.rgba().len() == len)
}

fn sample_cpu(source: &dyn PictureSource, time: Micros) -> Option<CpuBytes> {
    match source.sample(time) {
        Sample::Ready(Picture::Cpu(bytes)) => Some(bytes),
        Sample::Ready(Picture::Platform(_)) | Sample::Miss => None,
    }
}

fn cpu_of<E>(frame: Frame, extent: Extent) -> Result<CpuBytes, EvalError<E>> {
    CpuBytes::new(extent.width, extent.height, frame.rgba)
        .map_err(|_| EvalError::Compose(ComposeError::Extent))
}

fn covers(start: i64, duration: i64, time: i64) -> bool {
    if duration <= 0 {
        return false;
    }
    let Some(end) = start.checked_add(duration) else {
        return false;
    };
    time >= start && time < end
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_effects::{INVERT, MULTIPLY, SOURCE_OVER, package::Located};
    use cutlass_engine::ReaderFactory;
    use cutlass_model::{
        Clip, Draft, MaterialColorConfig, MaterialDraft, MaterialEffect, MaterialPluginEffect,
        MaterialVideoEffect, Micros, Segment, TimeRange, Track, TrackAttribute, TrackType,
    };
    use cutlass_renderer_software::SoftwareDevice;
    use cutlass_storage::ResourceId;

    use crate::frame::{Snapshot, frame};
    use crate::media::OpenClips;

    use super::PackageStore;

    const RED: [u8; 4] = [255, 0, 0, 255];
    const CYAN: [u8; 4] = [0, 255, 255, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cutlass-eval-effect-{label}-{}-{n}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("temp");
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    struct Store {
        files: BTreeMap<String, Located>,
    }

    impl Store {
        fn new() -> Self {
            Self {
                files: BTreeMap::new(),
            }
        }

        fn publish(&mut self, temp: &Temp, id: &str, bytes: &[u8], size: u64) {
            let path = temp.path.join(id);
            fs::write(&path, bytes).expect("package");
            self.files.insert(
                id.to_owned(),
                Located {
                    path,
                    size,
                    checksum_hex: None,
                },
            );
        }
    }

    impl PackageStore for Store {
        fn locate(&self, id: &ResourceId) -> Result<Located, cutlass_effects::PackageError> {
            self.files.get(id.as_str()).cloned().ok_or_else(|| {
                cutlass_effects::PackageError::Missing {
                    id: id.as_str().to_owned(),
                }
            })
        }
    }

    fn draw(draft: &Draft, store: &Store, time: i64) -> Vec<u8> {
        let mut device = SoftwareDevice::new();
        let media = ReaderFactory::new();
        let mut clips = OpenClips::new();
        frame(
            &Snapshot::with_packages(draft, store),
            Micros(time),
            &mut device,
            &media,
            &mut clips,
            0,
        )
        .expect("frame")
        .rgba
    }

    fn red_draft() -> Draft {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        add_color(&mut draft, "color", "0xFFFF0000", 0);
        draft
    }

    fn add_color(draft: &mut Draft, id: &str, packed: &str, render_index: i32) {
        draft.materials.material_colors.push(MaterialColorConfig {
            id: id.into(),
            is_color_clip: true,
            solid_color: packed.to_owned(),
            ..MaterialColorConfig::default()
        });
        let mut segment = Segment::new(id, id, TimeRange::new(0, 1_000_000));
        segment.render_index = render_index;
        draft.tracks[0].segments.push(segment);
    }

    fn add_global(draft: &mut Draft, id: &str, package: &str, render_index: i32, duration: i64) {
        let mut effect = MaterialEffect::new(id);
        effect.effect_id = package.to_owned();
        draft.materials.effects.push(effect);
        let mut segment = Segment::new(id, id, TimeRange::new(0, duration));
        segment.render_index = render_index;
        let mut track = Track::main(id);
        track.kind = TrackType::Effect;
        track.segments.push(segment);
        draft.tracks.push(track);
    }

    fn publish_invert(temp: &Temp, store: &mut Store) {
        let len = u64::try_from(b"zzzz".len()).expect("len");
        store.publish(temp, INVERT, b"zzzz", len);
    }

    #[test]
    fn an_effect_segment_inverts_the_picture_below_it() {
        let temp = Temp::new("invert");
        let mut store = Store::new();
        publish_invert(&temp, &mut store);
        let mut draft = red_draft();
        add_global(&mut draft, "fx", INVERT, 11_000, 1_000_000);
        let before = draft.clone();
        assert_eq!(draw(&draft, &store, 0), CYAN);
        assert_eq!(draw(&draft, &store, 1_000_000), [0, 0, 0, 0]);
        assert_eq!(draft, before);

        add_global(&mut draft, "fx2", INVERT, 12_000, 1_000_000);
        assert_eq!(draw(&draft, &store, 0), RED);
    }

    #[test]
    fn a_missing_package_a_hidden_track_and_no_store_leave_the_picture() {
        let temp = Temp::new("miss");
        let mut store = Store::new();
        publish_invert(&temp, &mut store);
        let mut draft = red_draft();
        add_global(&mut draft, "fx", INVERT, 11_000, 500_000);
        assert_eq!(draw(&draft, &store, 500_000), RED);

        let mut hidden = red_draft();
        add_global(&mut hidden, "fx", INVERT, 11_000, 1_000_000);
        hidden.tracks[1].attribute = TrackAttribute(TrackAttribute::HIDE);
        assert_eq!(draw(&hidden, &store, 0), RED);

        let mut invisible = red_draft();
        add_global(&mut invisible, "fx", INVERT, 11_000, 1_000_000);
        invisible.tracks[1].segments[0].visible = false;
        assert_eq!(draw(&invisible, &store, 0), RED);

        let mut missing = red_draft();
        add_global(&mut missing, "fx", INVERT, 11_000, 1_000_000);
        let empty = Store::new();
        let before = missing.clone();
        assert_eq!(draw(&missing, &empty, 0), RED);
        assert_eq!(missing, before);

        let mut device = SoftwareDevice::new();
        let media = ReaderFactory::new();
        let mut clips = OpenClips::new();
        let rgba = frame(
            &Snapshot::new(&missing),
            Micros(0),
            &mut device,
            &media,
            &mut clips,
            0,
        )
        .expect("frame")
        .rgba;
        assert_eq!(rgba, RED);
    }

    #[test]
    fn paint_order_decides_whether_the_effect_sees_the_solid() {
        let temp = Temp::new("order");
        let mut store = Store::new();
        publish_invert(&temp, &mut store);

        let mut below = red_draft();
        below.tracks[0].segments[0].render_index = 10;
        add_global(&mut below, "fx", INVERT, 0, 1_000_000);
        assert_eq!(draw(&below, &store, 0), RED);

        let mut between = red_draft();
        add_color(&mut between, "blue", "0xFF0000FF", 20_000);
        add_global(&mut between, "fx", INVERT, 11_000, 1_000_000);
        assert_eq!(draw(&between, &store, 0), BLUE);
    }

    #[test]
    fn a_clip_effect_filters_that_segment_and_a_blend_uses_the_picture_below() {
        let temp = Temp::new("clip");
        let mut store = Store::new();
        let len = u64::try_from(b"zzzz".len()).expect("len");
        store.publish(&temp, INVERT, b"zzzz", len);
        store.publish(&temp, MULTIPLY, b"zzzz", len);

        let mut inverted = red_draft();
        let mut effect = MaterialEffect::new("fx");
        effect.effect_id = INVERT.to_owned();
        inverted.materials.effects.push(effect);
        inverted.tracks[0].segments[0]
            .extra_material_refs
            .push("fx".into());
        let before = inverted.clone();
        assert_eq!(draw(&inverted, &store, 0), CYAN);
        assert_eq!(inverted, before);

        let mut blended = red_draft();
        blended.materials.material_colors[0].solid_color = "0xFF640000".to_owned();
        add_color(&mut blended, "upper", "0xFFC80000", 1);
        let mut multiply = MaterialEffect::new("mul");
        multiply.effect_id = MULTIPLY.to_owned();
        blended.materials.effects.push(multiply);
        blended.tracks[0].segments[1]
            .extra_material_refs
            .push("mul".into());
        assert_eq!(draw(&blended, &store, 0), [78, 0, 0, 255]);
    }

    #[test]
    fn a_blend_on_an_effect_segment_leaves_the_picture() {
        let temp = Temp::new("blend");
        let mut store = Store::new();
        let len = u64::try_from(b"zzzz".len()).expect("len");
        store.publish(&temp, SOURCE_OVER, b"zzzz", len);
        let mut draft = red_draft();
        draft.materials.material_colors[0].solid_color = "0x80FF0000".to_owned();
        add_global(&mut draft, "fx", SOURCE_OVER, 11_000, 1_000_000);
        assert_eq!(draw(&draft, &store, 0), [255, 0, 0, 128]);
    }

    #[test]
    fn a_video_effect_and_a_plugin_use_the_same_stand_in() {
        let temp = Temp::new("kinds");
        let mut store = Store::new();
        publish_invert(&temp, &mut store);

        let mut video = red_draft();
        let effect = MaterialVideoEffect {
            id: "fx".into(),
            effect_id: INVERT.to_owned(),
            ..MaterialVideoEffect::default()
        };
        video.materials.video_effects.push(effect);
        let mut segment = Segment::new("fx", "fx", TimeRange::new(0, 1_000_000));
        segment.render_index = 11_000;
        let mut track = Track::main("fx");
        track.kind = TrackType::Effect;
        track.segments.push(segment);
        video.tracks.push(track);
        assert_eq!(draw(&video, &store, 0), CYAN);

        let mut plugin = red_draft();
        let effect = MaterialPluginEffect {
            id: "fx".into(),
            plugin_id: INVERT.to_owned(),
            ..MaterialPluginEffect::default()
        };
        plugin.materials.plugin_effects.push(effect);
        let mut segment = Segment::new("fx", "fx", TimeRange::new(0, 1_000_000));
        segment.render_index = 11_000;
        let mut track = Track::main("fx");
        track.kind = TrackType::PluginEffect;
        track.segments.push(segment);
        plugin.tracks.push(track);
        assert_eq!(draw(&plugin, &store, 0), CYAN);
    }

    #[test]
    fn a_bad_package_leaves_the_picture_and_a_nested_effect_is_the_child() {
        let temp = Temp::new("nest");
        let mut store = Store::new();
        store.publish(&temp, INVERT, b"zzzz", 1);
        let mut broken = red_draft();
        add_global(&mut broken, "fx", INVERT, 11_000, 1_000_000);
        let before = broken.clone();
        assert_eq!(draw(&broken, &store, 0), RED);
        assert_eq!(broken, before);

        let mut child = red_draft();
        add_global(&mut child, "fx", INVERT, 11_000, 1_000_000);
        let len = u64::try_from(b"zzzz".len()).expect("len");
        store.publish(&temp, INVERT, b"zzzz", len);
        let mut parent = Draft::empty("parent");
        parent.canvas_config.width = 1;
        parent.canvas_config.height = 1;
        parent.materials.drafts.push(MaterialDraft {
            id: "nest".into(),
            draft: Some(Box::new(child)),
            ..MaterialDraft::default()
        });
        let mut segment = Segment::new("compound", "nest", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        parent.tracks[0].segments.push(segment);
        assert_eq!(draw(&parent, &store, 0), CYAN);

        parent.tracks[0].segments[0].clip = Clip {
            alpha: 0.0,
            ..Clip::default()
        };
        assert_eq!(draw(&parent, &store, 0), [0, 0, 0, 0]);
    }
}
