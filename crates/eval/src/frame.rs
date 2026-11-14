use std::fmt;

use cutlass_compositor::{Device, Extent, Frame};
use cutlass_effects::{MissingPackages, PackageStore};
use cutlass_engine::ReaderFactory;
use cutlass_generated::{PictureSource, SolidColor};
use cutlass_model::{Draft, Micros};
use cutlass_picture::{CpuBytes, Picture, Sample};

use crate::canvas::canvas_rgba;
use crate::effect::{
    RecipeMark, StackLayer, clip_package, clip_passes, colour_marks, present, recipe_marks,
};
use crate::gpu::{FrameLayers, GpuLayer};
use crate::lottie::{LottieLayer, PlacedLottie, lottie_layers, shape_layers};
use crate::media::{OpenClips, Placed, video_layers};
use crate::nest::{NestPass, NestedLayer, NestedPicture, nest_layers};
use crate::sources::{SolidLayer, solid_layers};
use crate::text::{TextLayer, text_layers};

/// Font bytes for a text material's `font_path`.
///
/// Evaluation does not open that path. A face this returns `None` for is not drawn.
pub trait Fonts {
    /// The font file named by `path`, or `None` when this table does not have it.
    fn face(&self, path: &str) -> Option<&[u8]>;
}

/// No font bytes. Text segments are not drawn.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoFonts;

impl Fonts for NoFonts {
    fn face(&self, _path: &str) -> Option<&[u8]> {
        None
    }
}

static NO_FONTS: NoFonts = NoFonts;
static NO_PACKAGES: MissingPackages = MissingPackages;

/// A draft borrowed for evaluation. The draft is not written.
#[derive(Clone, Copy)]
pub struct Snapshot<'a> {
    draft: &'a Draft,
    fonts: &'a dyn Fonts,
    packages: &'a dyn PackageStore,
}

impl<'a> Snapshot<'a> {
    /// No font bytes and no package store. Text and effects are not drawn.
    #[must_use]
    pub fn new(draft: &'a Draft) -> Self {
        Self::prepare(draft, &NO_FONTS, &NO_PACKAGES)
    }

    /// `fonts` is consulted for each text segment. Effects are not drawn.
    #[must_use]
    pub fn with_fonts(draft: &'a Draft, fonts: &'a dyn Fonts) -> Self {
        Self::prepare(draft, fonts, &NO_PACKAGES)
    }

    /// `packages` is consulted for each effect. Text is not drawn.
    #[must_use]
    pub fn with_packages(draft: &'a Draft, packages: &'a dyn PackageStore) -> Self {
        Self::prepare(draft, &NO_FONTS, packages)
    }

    /// `fonts` and `packages` are consulted. The draft is not written.
    #[must_use]
    pub fn with_fonts_and_packages(
        draft: &'a Draft,
        fonts: &'a dyn Fonts,
        packages: &'a dyn PackageStore,
    ) -> Self {
        Self::prepare(draft, fonts, packages)
    }

    fn prepare(draft: &'a Draft, fonts: &'a dyn Fonts, packages: &'a dyn PackageStore) -> Self {
        Self {
            draft,
            fonts,
            packages,
        }
    }

    #[must_use]
    pub fn draft(self) -> &'a Draft {
        self.draft
    }

    pub(crate) fn fonts(self) -> &'a dyn Fonts {
        self.fonts
    }

    pub(crate) fn packages(self) -> &'a dyn PackageStore {
        self.packages
    }
}

/// Why a snapshot did not produce a frame.
#[derive(Debug, PartialEq, Eq)]
pub enum EvalError<E> {
    /// The canvas width or height is not a framebuffer size.
    Canvas,
    /// The compositor or the device refused the frame.
    Compose(cutlass_compositor::ComposeError<E>),
    /// The export clock did not accept the picture.
    Wait,
}

impl<E: fmt::Display> fmt::Display for EvalError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canvas => formatter.write_str("the canvas size is not a frame"),
            Self::Compose(error) => write!(formatter, "{error}"),
            Self::Wait => formatter.write_str("export did not accept the frame"),
        }
    }
}

impl<E: std::error::Error> std::error::Error for EvalError<E> {}

fn canvas_extent(draft: &Draft) -> Option<Extent> {
    let width = u32::try_from(draft.canvas_config.width).ok()?;
    let height = u32::try_from(draft.canvas_config.height).ok()?;
    let extent = Extent { width, height };
    extent.rgba_len()?;
    Some(extent)
}

/// The composited frame at `time`. The snapshot's draft is not changed.
///
/// Solids, text, nested drafts, video, and effect packages share one paint
/// order: `render_index`, then track, then segment. A video the factory
/// cannot open is skipped. A text segment whose face the snapshot does not
/// supply is skipped. An effect whose package the snapshot does not supply
/// is skipped. A transition package fades the pictures that meet at its cut.
/// A missing package stays a hard cut. The project canvas color is under
/// every layer. An open file
/// in `clips` is seeked when `time` changes.
/// `revision` drops those files when the draft changes.
pub fn frame<D: Device>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    device: &mut D,
    media: &ReaderFactory,
    clips: &mut OpenClips,
    revision: u64,
) -> Result<Frame, EvalError<D::Error>> {
    frame_in(snapshot, time, device, media, clips, revision, None)
}

/// The composited picture. A relative media path is joined to `root`.
pub fn frame_in<D: Device>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    device: &mut D,
    media: &ReaderFactory,
    clips: &mut OpenClips,
    revision: u64,
    root: Option<&std::path::Path>,
) -> Result<Frame, EvalError<D::Error>> {
    clips.set_project(root);
    compose(snapshot, time, device, media, clips, revision, &[])
}

/// The frame as GPU layers: the project background, then every layer in
/// order. A clip the GPU path covers is its decoded picture and placement,
/// a solid is one stretched pixel, and the rest are their CPU pictures.
/// Effects, looks, adjustment layers and effect packages are layers of
/// their own over the layers before them, and a clip's own colour work,
/// mask, selection and package run over that clip alone, as [`frame_in`]
/// runs them on the CPU.
pub fn layers_in<D: Device>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    device: &mut D,
    media: &ReaderFactory,
    clips: &mut OpenClips,
    revision: u64,
    root: Option<&std::path::Path>,
) -> Result<FrameLayers, EvalError<D::Error>> {
    clips.set_project(root);
    compose_layers(snapshot, time, device, media, clips, revision, &[])
}

/// [`layers_in`] for a draft under `ancestors`, as a compound's child is.
pub(crate) fn compose_layers<D: Device>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    device: &mut D,
    media: &ReaderFactory,
    clips: &mut OpenClips,
    revision: u64,
    ancestors: &[cutlass_model::DraftId],
) -> Result<FrameLayers, EvalError<D::Error>> {
    let store = snapshot.packages();
    with_layers(
        snapshot,
        time,
        device,
        media,
        clips,
        revision,
        ancestors,
        true,
        |draft, _device, extent, layers| {
            let stack = stack_of(layers);
            let effects = recipe_marks(draft, time, extent, &stack);
            let colours = colour_marks(draft, store, time, extent, &stack);
            Ok(gpu_layers(
                draft, store, time, extent, layers, effects, colours,
            ))
        },
    )
}

/// `layers` as GPU layers, with `effects` (recipes) and `colours` (looks,
/// adjustment layers and effect packages) each over the layers before it,
/// and every clip's own colour work, mask, selection, package and bound
/// recipes over that clip alone.
fn gpu_layers(
    draft: &Draft,
    store: &dyn PackageStore,
    time: Micros,
    extent: Extent,
    layers: &[Ordered<'_>],
    effects: Vec<RecipeMark>,
    colours: Vec<((i32, usize, usize), GpuLayer)>,
) -> FrameLayers {
    let mut out = Vec::new();
    let (bound, effects): (Vec<RecipeMark>, Vec<RecipeMark>) =
        effects.into_iter().partition(|mark| mark.bound.is_some());
    let effect =
        |mark: RecipeMark| GpuLayer::effect(mark.steps, mark.mask, extent.width, extent.height);
    let mut marks: Vec<((i32, usize, usize), GpuLayer)> = effects
        .into_iter()
        .map(|mark| (mark.key, effect(mark)))
        .chain(colours)
        .collect();
    marks.sort_by_key(|(key, _)| *key);
    let mut effects = marks.into_iter().peekable();
    if let Some(pixel) = draft
        .canvas_config
        .background
        .as_deref()
        .and_then(canvas_rgba)
        .and_then(|pixel| CpuBytes::new(1, 1, pixel.to_vec()).ok())
    {
        out.push(GpuLayer::fill(pixel));
    }
    for layer in layers {
        while let Some((_, mark)) = effects.next_if(|(key, _)| *key < layer.key) {
            out.push(mark);
        }
        let gpu = match &layer.paint {
            Paint::Video(placed) => placed.gpu(time),
            Paint::Solid(solid) => match solid.one_pixel(time) {
                Sample::Ready(Picture::Cpu(pixel)) => Some(vec![GpuLayer::fill(pixel)]),
                Sample::Ready(Picture::Platform(_)) | Sample::Miss => Some(Vec::new()),
            },
            Paint::Nest(nested) => nested.gpu(),
            Paint::Text(text) => text.gpu(),
            Paint::Lottie(lottie) => lottie.gpu(),
            Paint::Mixed(mixed) => Some(vec![GpuLayer::transition(
                mixed.moment.kind,
                mixed.moment.t(),
                mixed.outgoing.clone(),
                mixed.incoming.clone(),
            )]),
        };
        let mut drawn = match gpu {
            Some(gpu) => gpu,
            None => match layer.source().sample(time) {
                Sample::Ready(Picture::Cpu(picture)) => vec![GpuLayer::canvas(picture)],
                Sample::Ready(Picture::Platform(_)) | Sample::Miss => Vec::new(),
            },
        };
        let mut own = clip_passes(draft, time, extent, layer.key.1, layer.key.2);
        let package = clip_package(draft, store, time, layer.key.1, layer.key.2);
        let package_blend = package.as_ref().and_then(|pass| pass.blend);
        // A blend package's own pass is its fade, left out when it has none.
        if let Some(pass) = package.filter(|pass| {
            pass.blend.is_none()
                || pass
                    .faded
                    .iter()
                    .enumerate()
                    .any(|(alpha, faded)| usize::from(*faded) != alpha)
        }) {
            own.push(GpuLayer::package(pass, extent.width, extent.height));
        }
        own.extend(
            bound
                .iter()
                .filter(|mark| mark.bound == Some((layer.key.1, layer.key.2)))
                .map(|mark| {
                    effect(RecipeMark {
                        steps: mark.steps.clone(),
                        ..*mark
                    })
                }),
        );
        let mode = draft
            .tracks
            .get(layer.key.1)
            .and_then(|track| track.segments.get(layer.key.2))
            .and_then(|segment| cutlass_effects::segment_blend(draft, segment))
            .or(package_blend);
        if drawn.is_empty() {
            // Nothing of this clip shows at this time.
        } else if let Some(mode) = mode {
            // The clip and its own effects on a canvas of their own, mixed
            // with everything below.
            drawn.extend(own);
            out.push(GpuLayer::blend(
                mode,
                FrameLayers {
                    width: extent.width,
                    height: extent.height,
                    layers: drawn,
                },
            ));
        } else if own.is_empty() {
            out.extend(drawn);
        } else {
            // The clip is drawn on a canvas of its own, its effects run over
            // that canvas only, and the result lands like any other layer.
            drawn.extend(own);
            out.push(GpuLayer::nested(FrameLayers {
                width: extent.width,
                height: extent.height,
                layers: drawn,
            }));
        }
    }
    out.extend(effects.map(|(_, layer)| layer));
    FrameLayers {
        width: extent.width,
        height: extent.height,
        layers: out,
    }
}

pub(crate) fn compose<D: Device>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    device: &mut D,
    media: &ReaderFactory,
    clips: &mut OpenClips,
    revision: u64,
    ancestors: &[cutlass_model::DraftId],
) -> Result<Frame, EvalError<D::Error>> {
    with_layers(
        snapshot,
        time,
        device,
        media,
        clips,
        revision,
        ancestors,
        false,
        |draft, device, extent, layers| {
            let stack = stack_of(layers);
            present(draft, snapshot.packages(), time, device, extent, &stack)
        },
    )
}

/// Build every layer at `time` in order, then hand them to `finish`.
#[allow(clippy::too_many_arguments)]
fn with_layers<D: Device, T>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    device: &mut D,
    media: &ReaderFactory,
    clips: &mut OpenClips,
    revision: u64,
    ancestors: &[cutlass_model::DraftId],
    gpu: bool,
    finish: impl FnOnce(&Draft, &mut D, Extent, &[Ordered<'_>]) -> Result<T, EvalError<D::Error>>,
) -> Result<T, EvalError<D::Error>> {
    let draft = snapshot.draft();
    let extent = canvas_extent(draft).ok_or(EvalError::Canvas)?;
    clips.prepare(draft, revision);
    let mut pass = NestPass {
        media,
        width: extent.width,
        height: extent.height,
        time,
        fonts: snapshot.fonts(),
        packages: snapshot.packages(),
        device,
        clips,
        revision,
        ancestors,
        gpu,
    };
    let nests = nest_layers(draft, &mut pass)?;
    let prepared = crate::transition::prepare(
        draft,
        snapshot.packages(),
        pass.time,
        extent.width,
        extent.height,
        &mut pass,
    )?;
    let mut lotties = lottie_layers(draft, extent.width, extent.height, pass.time, pass.clips);
    // Shapes are vector scenes too, drawn and placed the same way.
    lotties.extend(shape_layers(
        draft,
        extent.width,
        extent.height,
        pass.time,
        pass.clips.shapes(),
    ));
    let mut layers = ordered(
        solid_layers(draft, extent.width, extent.height),
        text_layers(
            draft,
            pass.fonts,
            extent.width,
            extent.height,
            pass.time,
            pass.clips.texts(),
        ),
        video_layers(
            draft,
            pass.media,
            extent.width,
            extent.height,
            pass.time,
            pass.clips,
        ),
        nests,
    );
    // Lottie stickers and shapes join in their place among the other clips.
    add_lotties(&mut layers, lotties);
    apply_transitions(
        draft,
        snapshot.packages(),
        &mut layers,
        &prepared,
        extent.width,
        extent.height,
        pass.time,
    );
    finish(draft, pass.device, extent, &layers)
}

fn stack_of<'a>(layers: &'a [Ordered<'a>]) -> Vec<StackLayer<'a>> {
    layers
        .iter()
        .map(|layer| StackLayer::new(layer.key.0, layer.key.1, layer.key.2, layer.source()))
        .collect()
}

enum Paint<'a> {
    Solid(SolidColor),
    Text(crate::text::PlacedText),
    Video(Placed<'a>),
    Lottie(PlacedLottie),
    Nest(NestedPicture),
    Mixed(Mixed),
}

/// A clip in a transition: both pictures and the moment, mixed on the GPU
/// when the frame is drawn there and on the CPU when it is composed.
struct Mixed {
    moment: crate::transition::Moment,
    outgoing: CpuBytes,
    incoming: CpuBytes,
}

impl PictureSource for Mixed {
    fn sample(&self, _time: Micros) -> Sample {
        let (width, height) = (self.outgoing.width(), self.outgoing.height());
        match self
            .moment
            .mix(self.outgoing.rgba(), self.incoming.rgba(), width, height)
        {
            Some(mixed) => Sample::Ready(Picture::Cpu(mixed)),
            None => Sample::Miss,
        }
    }
}

struct Ordered<'a> {
    key: (i32, usize, usize),
    paint: Paint<'a>,
}

impl Ordered<'_> {
    fn source(&self) -> &dyn PictureSource {
        match &self.paint {
            Paint::Solid(solid) => solid,
            Paint::Text(text) => text,
            Paint::Video(video) => video,
            Paint::Lottie(lottie) => lottie,
            Paint::Nest(nested) => nested,
            Paint::Mixed(mixed) => mixed,
        }
    }
}

fn apply_transitions(
    draft: &Draft,
    store: &dyn PackageStore,
    layers: &mut [Ordered<'_>],
    prepared: &crate::transition::Prepared,
    width: u32,
    height: u32,
    time: Micros,
) {
    for cut in prepared.cuts() {
        let (cover_track, cover_segment) = cut.cover();
        let Some(index) = layers
            .iter()
            .position(|layer| layer.key.1 == cover_track && layer.key.2 == cover_segment)
        else {
            continue;
        };
        let Some(covering) = cpu_at(layers[index].source(), time) else {
            continue;
        };
        let (partner_track, partner_segment, partner_time) = cut.partner();
        let Some(partner) = partner_picture(
            layers,
            prepared,
            partner_track,
            partner_segment,
            partner_time,
        ) else {
            continue;
        };
        let Some(moment) = crate::transition::moment(store, draft, cut) else {
            continue;
        };
        let size = (width, height);
        if (covering.width(), covering.height()) != size
            || (partner.width(), partner.height()) != size
        {
            continue;
        }
        let (outgoing, incoming) = if cut.covers_outgoing() {
            (covering, partner)
        } else {
            (partner, covering)
        };
        layers[index].paint = Paint::Mixed(Mixed {
            moment,
            outgoing,
            incoming,
        });
    }
}

fn partner_picture(
    layers: &[Ordered<'_>],
    prepared: &crate::transition::Prepared,
    track: usize,
    segment: usize,
    time: Micros,
) -> Option<CpuBytes> {
    if let Some(bytes) = prepared.picture(track, segment) {
        return Some(bytes.clone());
    }
    let layer = layers
        .iter()
        .find(|layer| layer.key.1 == track && layer.key.2 == segment)?;
    match &layer.paint {
        Paint::Solid(solid) => cpu_at(solid, time),
        Paint::Lottie(lottie) => cpu_at(lottie, time),
        Paint::Text(_) | Paint::Video(_) | Paint::Nest(_) | Paint::Mixed(_) => None,
    }
}

fn cpu_at(source: &dyn PictureSource, time: Micros) -> Option<CpuBytes> {
    match source.sample(time) {
        Sample::Ready(Picture::Cpu(bytes)) => Some(bytes),
        Sample::Ready(Picture::Platform(_)) | Sample::Miss => None,
    }
}

fn ordered<'a>(
    solids: Vec<SolidLayer>,
    texts: Vec<TextLayer>,
    videos: Vec<crate::media::VideoLayer<'a>>,
    nests: Vec<NestedLayer>,
) -> Vec<Ordered<'a>> {
    let mut layers = Vec::with_capacity(solids.len() + texts.len() + videos.len() + nests.len());
    for solid in solids {
        layers.push(Ordered {
            key: (solid.render_index, solid.track_index, solid.segment_index),
            paint: Paint::Solid(solid.solid),
        });
    }
    for text in texts {
        layers.push(Ordered {
            key: (text.render_index, text.track_index, text.segment_index),
            paint: Paint::Text(text.placed),
        });
    }
    for video in videos {
        layers.push(Ordered {
            key: (video.render_index, video.track_index, video.segment_index),
            paint: Paint::Video(video.placed),
        });
    }
    for nested in nests {
        layers.push(Ordered {
            key: (
                nested.render_index,
                nested.track_index,
                nested.segment_index,
            ),
            paint: Paint::Nest(nested.placed),
        });
    }
    layers.sort_by_key(|layer| layer.key);
    layers
}

fn add_lotties(layers: &mut Vec<Ordered<'_>>, lotties: Vec<LottieLayer>) {
    if lotties.is_empty() {
        return;
    }
    for lottie in lotties {
        layers.push(Ordered {
            key: (
                lottie.render_index,
                lottie.track_index,
                lottie.segment_index,
            ),
            paint: Paint::Lottie(lottie.placed),
        });
    }
    layers.sort_by_key(|layer| layer.key);
}

/// Keeps `frame` for export. A late arrival is still the picture.
///
/// This calls the engine export clock. Playback's late drop is a different clock.
pub fn wait_export_frame<E>(
    requested: Micros,
    ready_at: Micros,
    frame: Frame,
) -> Result<Frame, EvalError<E>> {
    let clock = cutlass_engine::MediaClock::new(cutlass_engine::ClockKind::Export);
    let extent = frame.extent;
    let rgba = clock
        .export_rgba(requested, ready_at, extent.width, extent.height, frame.rgba)
        .map_err(|_| EvalError::Wait)?;
    Ok(Frame { extent, rgba })
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::{Path, PathBuf};

    use cutlass_codec::CodecId;
    use cutlass_container::{Demux, Input, OpenError, Probe, ProbeResult};
    use cutlass_decode::{CodecError, CodecParams, Decoder};
    use cutlass_engine::{DisplayQuarter, ReaderFactory};
    use cutlass_model::{
        Crop, Draft, MaterialCanvas, MaterialColorConfig, MaterialEffect, MaterialVideo, Micros,
        Segment, TimeRange, TrackAttribute,
    };
    use cutlass_picture::CpuBytes;
    use cutlass_png::PngDecoder;
    use cutlass_renderer_software::SoftwareDevice;
    use cutlass_still::{StillDemux, StillProbe, still_limits};

    use super::*;

    fn media() -> ReaderFactory {
        ReaderFactory::new()
    }

    fn draw(draft: &Draft, time: i64, device: &mut SoftwareDevice) -> Vec<u8> {
        let media = media();
        let mut clips = OpenClips::new();
        frame(
            &Snapshot::new(draft),
            Micros(time),
            device,
            &media,
            &mut clips,
            0,
        )
        .unwrap()
        .rgba
    }

    fn color_clip(color: &str, gradient: bool) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "color".into(),
            is_color_clip: true,
            is_gradient: gradient,
            solid_color: color.to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0].segments.push(Segment::new(
            "segment",
            "color",
            TimeRange::new(0, 1_000_000),
        ));
        draft
    }

    fn pixels(draft: &Draft, time: i64) -> Vec<u8> {
        let mut device = SoftwareDevice::new();
        draw(draft, time, &mut device)
    }

    #[test]
    fn the_same_snapshot_at_the_same_microsecond_returns_the_same_pixels() {
        let draft = color_clip("0xFF336699", false);
        let before = draft.clone();
        let mut device = SoftwareDevice::new();
        let media = media();
        let mut clips = OpenClips::new();
        let snapshot = Snapshot::new(&draft);
        let first = frame(&snapshot, Micros(10), &mut device, &media, &mut clips, 0).unwrap();
        let second = frame(&snapshot, Micros(10), &mut device, &media, &mut clips, 0).unwrap();
        assert_eq!(first.rgba, second.rgba);
        assert_eq!(first.rgba, [0x33, 0x66, 0x99, 0xFF].repeat(2));
        assert_eq!(draft, before);
    }

    #[test]
    fn a_time_outside_the_span_a_gradient_and_a_hidden_track_draw_nothing() {
        let draft = color_clip("0xFF336699", false);
        let before = draft.clone();
        assert_eq!(pixels(&draft, 1_000_000), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(pixels(&draft, -1), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(draft, before);

        let gradient = color_clip("0xFF336699", true);
        let before_gradient = gradient.clone();
        assert_eq!(pixels(&gradient, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(gradient, before_gradient);

        let mut hidden = color_clip("0xFF336699", false);
        hidden.tracks[0].attribute = TrackAttribute(TrackAttribute::HIDE);
        let before_hidden = hidden.clone();
        assert_eq!(pixels(&hidden, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(hidden, before_hidden);

        let mut invisible = color_clip("0xFF336699", false);
        invisible.tracks[0].segments[0].visible = false;
        let before_invisible = invisible.clone();
        assert_eq!(pixels(&invisible, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(invisible, before_invisible);
    }

    #[test]
    fn a_blended_clip_mixes_with_the_canvas_on_both_paths() {
        let mut draft = color_clip("0xFF000000", false);
        draft.canvas_config.background = Some("#808080".to_owned());
        // Black over grey hides it; screened, black leaves the grey.
        assert_eq!(pixels(&draft, 0), [0, 0, 0, 255].repeat(2));
        let segment = cutlass_model::SegmentId::new("segment");
        let mut session = cutlass_session::Session::open(draft);
        session
            .apply(
                0,
                &[cutlass_session::Request::SetBlend {
                    segment_id: segment,
                    material_id: "mix".into(),
                    mode: "blend-screen".to_owned(),
                }],
            )
            .unwrap();
        let draft = session.draft().clone();
        assert_eq!(pixels(&draft, 0), [128, 128, 128, 255].repeat(2));
        let mut device = SoftwareDevice::new();
        let mut clips = OpenClips::new();
        let Ok(frame) = layers_in(
            &Snapshot::new(&draft),
            Micros(0),
            &mut device,
            &media(),
            &mut clips,
            0,
            None,
        ) else {
            panic!("a blend stays on the GPU path");
        };
        assert!(
            frame.layers.iter().any(|layer| matches!(
                layer.source,
                crate::GpuSource::Blend {
                    mode: cutlass_effects::BlendMode::Screen,
                    ..
                }
            )),
            "{frame:?}"
        );
    }

    #[test]
    fn an_adjusted_clip_runs_its_sliders_over_itself_on_the_gpu_path() {
        let draft = color_clip("0xFF404040", false);
        assert_eq!(pixels(&draft, 0), [64, 64, 64, 255].repeat(2));
        let mut session = cutlass_session::Session::open(draft);
        session
            .apply(
                0,
                &[cutlass_session::Request::SetAdjust {
                    segment_id: cutlass_model::SegmentId::new("segment"),
                    material_id: "bright".into(),
                    key: "exposure".to_owned(),
                    value: 0.5,
                }],
            )
            .unwrap();
        let draft = session.draft().clone();
        assert_eq!(pixels(&draft, 0), [128, 128, 128, 255].repeat(2));
        let mut device = SoftwareDevice::new();
        let mut clips = OpenClips::new();
        let Ok(frame) = layers_in(
            &Snapshot::new(&draft),
            Micros(0),
            &mut device,
            &media(),
            &mut clips,
            0,
            None,
        ) else {
            panic!("colour work stays on the GPU path");
        };
        // The clip on a canvas of its own, its sliders over it alone.
        let [layer] = frame.layers.as_slice() else {
            panic!("one layer: {frame:?}");
        };
        let crate::GpuSource::Nested(child) = &layer.source else {
            panic!("a nested clip: {layer:?}");
        };
        let Some(crate::GpuSource::Colour { colour, .. }) =
            child.layers.last().map(|layer| &layer.source)
        else {
            panic!("colour last: {child:?}");
        };
        assert_eq!(colour.adjust.get(cutlass_effects::AdjustKey::Exposure), 0.5);
    }

    fn gpu_frame(draft: &Draft) -> FrameLayers {
        let mut device = SoftwareDevice::new();
        let mut clips = OpenClips::new();
        let Ok(frame) = layers_in(
            &Snapshot::new(draft),
            Micros(0),
            &mut device,
            &media(),
            &mut clips,
            0,
            None,
        ) else {
            panic!("the frame stays on the GPU path");
        };
        frame
    }

    #[test]
    fn a_clip_mask_and_selection_cut_the_clip_alone_after_its_colour() {
        let mut draft = color_clip("0xFF404040", false);
        let mask = cutlass_effects::Mask::new(cutlass_effects::MaskShape::Circle);
        let selection =
            cutlass_effects::Selection::from_alpha(2, 1, &[0, 255], Micros(0)).expect("selection");
        let mut add = |id: &str, kind: &str, fields| {
            draft
                .materials
                .common_mask
                .push(cutlass_model::RawMaterial {
                    id: id.into(),
                    kind: kind.to_owned(),
                    fields,
                });
            draft.tracks[0].segments[0]
                .extra_material_refs
                .push(id.into());
        };
        add("cut", cutlass_effects::MASK, mask.to_fields());
        add("object", cutlass_effects::SELECTION, selection.to_fields());
        let frame = gpu_frame(&draft);
        let [layer] = frame.layers.as_slice() else {
            panic!("one layer: {frame:?}");
        };
        let crate::GpuSource::Nested(child) = &layer.source else {
            panic!("a nested clip: {layer:?}");
        };
        let kinds: Vec<&str> = child
            .layers
            .iter()
            .map(|layer| match &layer.source {
                crate::GpuSource::Cut(_) => "cut",
                crate::GpuSource::Select(_) => "select",
                _ => "picture",
            })
            .collect();
        assert_eq!(kinds, ["picture", "cut", "select"]);
        let Some(crate::GpuSource::Select(cut)) = child.layers.last().map(|layer| &layer.source)
        else {
            unreachable!();
        };
        assert_eq!(
            (cut.width, cut.height, cut.alpha.as_slice()),
            (2, 1, &[0, 255][..])
        );
    }

    #[test]
    fn an_adjustment_layer_is_a_colour_layer_over_the_clips_before_it() {
        let draft = color_clip("0xFF404040", false);
        let mut session = cutlass_session::Session::open(draft);
        let layer = cutlass_model::SegmentId::new("layer");
        session
            .apply(
                0,
                &[
                    cutlass_session::Request::AddAdjustment {
                        track_id: "look".into(),
                        segment_id: layer.clone(),
                        material_id: "adj".into(),
                        start: Micros(0),
                        duration: Micros(1_000),
                    },
                    cutlass_session::Request::SetAdjust {
                        segment_id: layer,
                        material_id: "bright".into(),
                        key: "exposure".to_owned(),
                        value: 0.5,
                    },
                ],
            )
            .unwrap();
        let frame = gpu_frame(session.draft());
        let [clip, adjust] = frame.layers.as_slice() else {
            panic!("the clip, then the layer: {frame:?}");
        };
        assert!(matches!(clip.source, crate::GpuSource::Rgba(_)), "{clip:?}");
        assert!(
            matches!(adjust.source, crate::GpuSource::Colour { mask: None, .. }),
            "{adjust:?}"
        );
    }

    #[test]
    fn an_adjustment_layer_runs_over_the_picture_below_it() {
        let draft = color_clip("0xFF404040", false);
        let mut session = cutlass_session::Session::open(draft);
        let layer = cutlass_model::SegmentId::new("layer");
        session
            .apply(
                0,
                &[
                    cutlass_session::Request::AddAdjustment {
                        track_id: "look".into(),
                        segment_id: layer.clone(),
                        material_id: "adj".into(),
                        start: Micros(0),
                        duration: Micros(1_000),
                    },
                    cutlass_session::Request::SetAdjust {
                        segment_id: layer,
                        material_id: "bright".into(),
                        key: "exposure".to_owned(),
                        value: 0.5,
                    },
                ],
            )
            .unwrap();
        let mut draft = session.draft().clone();
        assert_eq!(pixels(&draft, 0), [128, 128, 128, 255].repeat(2));
        let row = draft
            .tracks
            .iter_mut()
            .find(|track| track.id.as_str() == "look")
            .unwrap();
        row.segments[0].visible = false;
        assert_eq!(pixels(&draft, 0), [64, 64, 64, 255].repeat(2));
    }

    #[test]
    fn a_colour_mask_limits_a_clip_and_an_adjustment_layer_to_its_shape() {
        // Turned a quarter, a linear mask keeps the right half.
        let right = cutlass_effects::Mask {
            rotation: 90.0,
            ..cutlass_effects::Mask::new(cutlass_effects::MaskShape::Linear)
        };
        let draft = color_clip("0xFF404040", false);
        let mut session = cutlass_session::Session::open(draft);
        let clip = cutlass_model::SegmentId::new("segment");
        session
            .apply(
                0,
                &[
                    cutlass_session::Request::SetAdjust {
                        segment_id: clip.clone(),
                        material_id: "bright".into(),
                        key: "exposure".to_owned(),
                        value: 0.5,
                    },
                    cutlass_session::Request::SetColorMask {
                        segment_id: clip.clone(),
                        material_id: "where".into(),
                        mask: Some(right),
                    },
                ],
            )
            .unwrap();
        assert_eq!(
            pixels(session.draft(), 0),
            [[64, 64, 64, 255], [128, 128, 128, 255]].concat()
        );
        let layer = cutlass_model::SegmentId::new("layer");
        session
            .apply(
                1,
                &[
                    cutlass_session::Request::SetAdjust {
                        segment_id: clip,
                        material_id: "off".into(),
                        key: "exposure".to_owned(),
                        value: 0.0,
                    },
                    cutlass_session::Request::AddAdjustment {
                        track_id: "look".into(),
                        segment_id: layer.clone(),
                        material_id: "adj".into(),
                        start: Micros(0),
                        duration: Micros(1_000),
                    },
                    cutlass_session::Request::SetAdjust {
                        segment_id: layer.clone(),
                        material_id: "bright2".into(),
                        key: "exposure".to_owned(),
                        value: 0.5,
                    },
                    cutlass_session::Request::SetColorMask {
                        segment_id: layer,
                        material_id: "where2".into(),
                        mask: Some(cutlass_effects::Mask {
                            invert: true,
                            ..right
                        }),
                    },
                ],
            )
            .unwrap();
        assert_eq!(
            pixels(session.draft(), 0),
            [[128, 128, 128, 255], [64, 64, 64, 255]].concat()
        );
    }

    #[test]
    fn a_bundled_filter_runs_its_look_at_its_strength() {
        let draft = color_clip("0xFF404040", false);
        let mut session = cutlass_session::Session::open(draft);
        let clip = cutlass_model::SegmentId::new("bright");
        session
            .apply(
                0,
                &[cutlass_session::Request::AddFilter {
                    track_id: "looks".into(),
                    segment_id: clip.clone(),
                    material_id: "f".into(),
                    start: Micros(0),
                    duration: Micros(1_000),
                    filter_id: "look/bright".to_owned(),
                }],
            )
            .unwrap();
        let full = pixels(session.draft(), 0);
        assert!(full[0] > 64, "{full:?}");
        session
            .apply(
                1,
                &[cutlass_session::Request::SetEffectStrength {
                    segment_id: clip,
                    material_id: "f2".into(),
                    value: 0.0,
                }],
            )
            .unwrap();
        assert_eq!(pixels(session.draft(), 0), [64, 64, 64, 255].repeat(2));
    }

    #[test]
    fn a_graded_clip_runs_its_curve() {
        let draft = color_clip("0xFF404040", false);
        let mut session = cutlass_session::Session::open(draft);
        session
            .apply(
                0,
                &[cutlass_session::Request::SetCurve {
                    segment_id: cutlass_model::SegmentId::new("segment"),
                    material_id: "curve".into(),
                    channel: "rgb".to_owned(),
                    points: vec![[0.0, 1.0], [1.0, 0.0]],
                }],
            )
            .unwrap();
        assert_eq!(pixels(session.draft(), 0), [191, 191, 191, 255].repeat(2));
    }

    #[test]
    fn a_clip_lut_maps_its_colours_at_its_strength() {
        let draft = color_clip("0xFFFF0000", false);
        let mut session = cutlass_session::Session::open(draft);
        let swap = "LUT_3D_SIZE 2\n0 0 0\n0 1 0\n1 0 0\n1 1 0\n0 0 1\n0 1 1\n1 0 1\n1 1 1\n";
        session
            .apply(
                0,
                &[cutlass_session::Request::SetLut {
                    segment_id: cutlass_model::SegmentId::new("segment"),
                    material_id: "lut".into(),
                    name: "Swap".to_owned(),
                    cube: Some(swap.to_owned()),
                    strength: 1.0,
                }],
            )
            .unwrap();
        assert_eq!(pixels(session.draft(), 0), [0, 255, 0, 255].repeat(2));
    }

    #[test]
    fn a_keyed_clip_shows_what_is_below_its_key_colour() {
        let mut draft = color_clip("0xFF00FF00", false);
        draft.canvas_config.background = Some("#808080".to_owned());
        let mut session = cutlass_session::Session::open(draft);
        session
            .apply(
                0,
                &[cutlass_session::Request::SetChroma {
                    segment_id: cutlass_model::SegmentId::new("segment"),
                    material_id: "key".into(),
                    color: "#00ff00".to_owned(),
                    strength: 0.4,
                    shadow: 0.0,
                }],
            )
            .unwrap();
        assert_eq!(pixels(session.draft(), 0), [128, 128, 128, 255].repeat(2));
    }

    #[test]
    fn a_masked_clip_shows_the_canvas_outside_its_mask_and_the_mask_moves_with_it() {
        let mut draft = color_clip("0xFFFF0000", false);
        draft.canvas_config.width = 8;
        draft.canvas_config.height = 8;
        draft.canvas_config.background = Some("#808080".to_owned());
        let mut session = cutlass_session::Session::open(draft);
        session
            .apply(
                0,
                &[cutlass_session::Request::SetMask {
                    segment_id: cutlass_model::SegmentId::new("segment"),
                    material_id: "mask".into(),
                    mask: Some(cutlass_effects::Mask::new(
                        cutlass_effects::MaskShape::Circle,
                    )),
                }],
            )
            .unwrap();
        let at = |frame: &[u8], x: usize, y: usize| {
            let index = (y * 8 + x) * 4;
            [
                frame[index],
                frame[index + 1],
                frame[index + 2],
                frame[index + 3],
            ]
        };
        let frame = pixels(session.draft(), 0);
        assert_eq!(at(&frame, 4, 4), [255, 0, 0, 255]);
        assert_eq!(at(&frame, 0, 0), [128, 128, 128, 255]);
        assert_eq!(at(&frame, 7, 3), [128, 128, 128, 255]);
        assert_eq!(at(&frame, 6, 4), [128, 128, 128, 255]);

        // Moved a quarter of the canvas right, the circle goes with it.
        let mut moved = session.draft().clone();
        moved.tracks[0].segments[0].clip.transform.x = 0.5;
        let frame = pixels(&moved, 0);
        assert_eq!(at(&frame, 6, 4), [255, 0, 0, 255]);
        assert_eq!(at(&frame, 3, 4), [128, 128, 128, 255]);
    }

    #[test]
    fn a_keyed_mask_moves_from_key_to_key() {
        let mut draft = color_clip("0xFFFF0000", false);
        draft.canvas_config.width = 8;
        draft.canvas_config.height = 8;
        draft.canvas_config.background = Some("#808080".to_owned());
        let duration = draft.tracks[0].segments[0].target.duration.0;
        let mut session = cutlass_session::Session::open(draft);
        let x = cutlass_model::MASK_KEYS[0];
        let small = cutlass_effects::Mask {
            width: 0.2,
            height: 0.2,
            ..cutlass_effects::Mask::new(cutlass_effects::MaskShape::Rectangle)
        };
        let key = |offset: i64, value: f64| cutlass_session::Request::KeyValues {
            segment_id: cutlass_model::SegmentId::new("segment"),
            offset: cutlass_model::Micros(offset),
            values: vec![(x.to_owned(), value)],
        };
        session
            .apply(
                0,
                &[
                    cutlass_session::Request::SetMask {
                        segment_id: cutlass_model::SegmentId::new("segment"),
                        material_id: "mask".into(),
                        mask: Some(small),
                    },
                    key(0, -0.5),
                    key(duration, 0.5),
                ],
            )
            .unwrap();
        let red = |frame: &[u8], x: usize| frame[(4 * 8 + x) * 4] == 255;
        // A small square a quarter of the canvas left at the start, in the
        // middle half way, a quarter right at the end.
        let start = pixels(session.draft(), 0);
        assert!(red(&start, 2) && !red(&start, 4) && !red(&start, 6));
        let middle = pixels(session.draft(), duration / 2);
        assert!(!red(&middle, 2) && red(&middle, 4) && !red(&middle, 6));
        let end = pixels(session.draft(), duration - 1);
        assert!(!red(&end, 2) && !red(&end, 4) && red(&end, 6));
    }

    #[test]
    fn a_selected_clip_keeps_the_object_and_shows_the_canvas_around_it() {
        let mut draft = color_clip("0xFFFF0000", false);
        draft.canvas_config.width = 8;
        draft.canvas_config.height = 8;
        draft.canvas_config.background = Some("#808080".to_owned());
        // The object is the left half of an 8 by 8 frame.
        let alpha: Vec<u8> = (0..64).map(|i| if i % 8 < 4 { 255 } else { 0 }).collect();
        let selection =
            cutlass_effects::Selection::from_alpha(8, 8, &alpha, cutlass_model::Micros(0)).unwrap();
        let mut session = cutlass_session::Session::open(draft);
        session
            .apply(
                0,
                &[cutlass_session::Request::SetSelection {
                    segment_id: cutlass_model::SegmentId::new("segment"),
                    material_id: "selection".into(),
                    selection: Some(selection.clone()),
                }],
            )
            .unwrap();
        let at = |frame: &[u8], x: usize, y: usize| {
            let index = (y * 8 + x) * 4;
            [
                frame[index],
                frame[index + 1],
                frame[index + 2],
                frame[index + 3],
            ]
        };
        let frame = pixels(session.draft(), 0);
        for y in 0..8 {
            assert_eq!(at(&frame, 1, y), [255, 0, 0, 255], "row {y}");
            assert_eq!(at(&frame, 6, y), [128, 128, 128, 255], "row {y}");
        }

        // Inverted, the clip keeps everything but the object.
        let mut session = cutlass_session::Session::open(session.draft().clone());
        session
            .apply(
                0,
                &[cutlass_session::Request::SetSelection {
                    segment_id: cutlass_model::SegmentId::new("segment"),
                    material_id: "inverted".into(),
                    selection: Some(cutlass_effects::Selection {
                        invert: true,
                        ..selection
                    }),
                }],
            )
            .unwrap();
        let frame = pixels(session.draft(), 0);
        assert_eq!(at(&frame, 1, 4), [128, 128, 128, 255]);
        assert_eq!(at(&frame, 6, 4), [255, 0, 0, 255]);

        // Taken off, the whole clip shows again.
        session
            .apply(
                session.revision(),
                &[cutlass_session::Request::SetSelection {
                    segment_id: cutlass_model::SegmentId::new("segment"),
                    material_id: "gone".into(),
                    selection: None,
                }],
            )
            .unwrap();
        let frame = pixels(session.draft(), 0);
        assert_eq!(at(&frame, 6, 4), [255, 0, 0, 255]);
        assert_eq!(at(&frame, 1, 4), [255, 0, 0, 255]);
    }

    #[test]
    fn a_compound_clip_draws_what_its_clips_drew_and_ungroups_to_the_same() {
        let mut draft = color_clip("0xFFFF0000", false);
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "green".into(),
            is_color_clip: true,
            solid_color: "0xFF00FF00".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "blue".into(),
            is_color_clip: true,
            solid_color: "0xFF0000FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0].segments.push(Segment::new(
            "second",
            "green",
            TimeRange::new(1_000_000, 1_000_000),
        ));
        let mut over = cutlass_model::Track::main("over");
        over.flag = cutlass_model::TrackFlag::default();
        let mut blue = Segment::new("over", "blue", TimeRange::new(500_000, 1_000_000));
        blue.clip.alpha = 0.5;
        blue.render_index = 1;
        over.segments.push(blue);
        draft.tracks.push(over);
        let times = [200_000, 700_000, 1_200_000, 1_700_000];
        let before: Vec<Vec<u8>> = times.iter().map(|time| pixels(&draft, *time)).collect();
        let mut session = cutlass_session::Session::open(draft);
        session
            .apply(
                0,
                &[cutlass_session::Request::CreateCompound {
                    segment_ids: vec!["segment".into(), "second".into(), "over".into()],
                    segment_id: "nest".into(),
                    material_id: "inside".into(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        let nested: Vec<Vec<u8>> = times
            .iter()
            .map(|time| pixels(session.draft(), *time))
            .collect();
        assert_eq!(nested, before);
        session
            .apply(
                1,
                &[cutlass_session::Request::Ungroup {
                    segment_id: "nest".into(),
                }],
            )
            .unwrap();
        let back: Vec<Vec<u8>> = times
            .iter()
            .map(|time| pixels(session.draft(), *time))
            .collect();
        assert_eq!(back, before);
    }

    #[test]
    fn the_project_canvas_color_is_under_every_layer() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.canvas_config.background = Some("#336699".to_owned());
        let before = draft.clone();
        assert_eq!(pixels(&draft, 0), [0x33, 0x66, 0x99, 0xFF].repeat(2));
        assert_eq!(pixels(&draft, -1), [0x33, 0x66, 0x99, 0xFF].repeat(2));
        assert_eq!(draft, before);

        draft.materials.effects.push(MaterialEffect {
            id: "fx".into(),
            effect_id: "missing".to_owned(),
            ..MaterialEffect::default()
        });
        let mut effect = Segment::new("fx", "fx", TimeRange::new(0, 1_000));
        effect.render_index = 11_000;
        draft.tracks[0].segments.push(effect);
        assert_eq!(pixels(&draft, 0), [0x33, 0x66, 0x99, 0xFF].repeat(2));

        let mut packed = draft.clone();
        packed.canvas_config.background = Some("0xFF336699".to_owned());
        assert_eq!(pixels(&packed, 0), [0x33, 0x66, 0x99, 0xFF].repeat(2));

        let mut unread = draft.clone();
        unread.canvas_config.background = Some("white".to_owned());
        assert_eq!(pixels(&unread, 0), vec![0, 0, 0, 0, 0, 0, 0, 0]);

        let mut blended = color_clip("0x80640000", false);
        blended.canvas_config.background = Some("#0000C8".to_owned());
        assert_eq!(pixels(&blended, 0), [100, 0, 99, 255].repeat(2));
        assert_eq!(pixels(&blended, 1_000_000), [0, 0, 200, 255].repeat(2));
    }

    #[test]
    fn an_empty_draft_and_a_bad_color_leave_a_clear_frame() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        let before = draft.clone();
        assert_eq!(pixels(&draft, 0), vec![0, 0, 0, 0]);
        assert_eq!(draft, before);

        let named = color_clip("white", false);
        let before_named = named.clone();
        assert_eq!(pixels(&named, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(named, before_named);

        let mut not_a_clip = color_clip("0xFF336699", false);
        not_a_clip.materials.material_colors[0].is_color_clip = false;
        let before_clip = not_a_clip.clone();
        assert_eq!(pixels(&not_a_clip, 10), vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(not_a_clip, before_clip);
    }

    #[test]
    fn a_higher_render_index_is_drawn_on_top() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "blue".into(),
            is_color_clip: true,
            solid_color: "0xFF0000FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "red".into(),
            is_color_clip: true,
            solid_color: "0xFFFF0000".to_owned(),
            ..MaterialColorConfig::default()
        });
        let mut under = Segment::new("under", "blue", TimeRange::new(0, 1_000));
        under.render_index = 1;
        let mut over = Segment::new("over", "red", TimeRange::new(0, 1_000));
        over.render_index = 2;
        draft.tracks[0].segments.push(over);
        draft.tracks[0].segments.push(under);
        let before = draft.clone();
        assert_eq!(pixels(&draft, 0), vec![255, 0, 0, 255]);
        assert_eq!(draft, before);
    }

    #[test]
    fn a_movie_the_factory_cannot_open_leaves_the_solid() {
        let mut draft = color_clip("0xFF336699", false);
        let mut video = MaterialVideo::new("movie");
        video.path = "/no/such/movie.mp4".to_owned();
        draft.materials.videos.push(video);
        let mut segment = Segment::new("movie", "movie", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        segment.render_index = 5;
        draft.tracks[0].segments.push(segment);
        let before = draft.clone();
        assert_eq!(
            pixels(&draft, 0),
            vec![0x33, 0x66, 0x99, 0xFF, 0x33, 0x66, 0x99, 0xFF]
        );
        assert_eq!(draft, before);
    }

    #[test]
    fn a_negative_canvas_is_refused_and_the_draft_stays() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = -1;
        let before = draft.clone();
        let mut device = SoftwareDevice::new();
        let media = media();
        let mut clips = OpenClips::new();
        let error = frame(
            &Snapshot::new(&draft),
            Micros(0),
            &mut device,
            &media,
            &mut clips,
            0,
        )
        .unwrap_err();
        assert_eq!(error, EvalError::Canvas);
        assert_eq!(draft, before);
    }

    #[test]
    fn export_keeps_a_frame_that_arrives_after_its_request() {
        use cutlass_compositor::Extent;

        let frame = Frame {
            extent: Extent {
                width: 1,
                height: 1,
            },
            rgba: vec![4, 5, 6, 7],
        };
        let kept = wait_export_frame::<()>(Micros(10), Micros(80), frame).expect("wait");
        assert_eq!(kept.rgba, vec![4, 5, 6, 7]);
        assert_eq!(kept.extent.width, 1);
    }

    fn probe_still(header: &[u8]) -> ProbeResult {
        StillProbe.probe(header)
    }

    fn open_still(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
        let file = File::open(path).map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?;
        let len = file
            .metadata()
            .map_err(|error| OpenError::Source {
                message: error.to_string(),
            })?
            .len();
        Ok(Box::new(StillDemux::open(Input::random(
            file,
            len,
            still_limits(),
        ))?))
    }

    fn open_png(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        Ok(Box::new(PngDecoder::open(params)?))
    }

    fn stills() -> ReaderFactory {
        let mut factory = ReaderFactory::new();
        factory.register_container(probe_still, open_still);
        factory.decoders_mut().register(CodecId::Png, open_png);
        factory
    }

    fn left_half() -> Crop {
        Crop {
            upper_left_x: 0.0,
            upper_left_y: 0.0,
            upper_right_x: 0.5,
            upper_right_y: 0.0,
            lower_left_x: 0.0,
            lower_left_y: 1.0,
            lower_right_x: 0.5,
            lower_right_y: 1.0,
        }
    }

    fn still_frame(path: &Path, width: u32, height: u32, crop: Crop, ratio: &str) -> Vec<u8> {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = i32::try_from(width).expect("width");
        draft.canvas_config.height = i32::try_from(height).expect("height");
        let mut video = MaterialVideo::new("still");
        video.path = path.display().to_string();
        video.crop = crop;
        video.crop_ratio = ratio.to_owned();
        draft.materials.videos.push(video);
        let mut segment = Segment::new("still", "still", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(segment);
        let before = draft.clone();
        let mut device = SoftwareDevice::new();
        let media = stills();
        let mut clips = OpenClips::new();
        let rgba = frame(
            &Snapshot::new(&draft),
            Micros(0),
            &mut device,
            &media,
            &mut clips,
            0,
        )
        .expect("frame")
        .rgba;
        assert_eq!(draft, before);
        rgba
    }

    fn columns(rgba: &[u8], width: usize, start: usize, count: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for row in rgba.chunks(width * 4) {
            let from = start * 4;
            let to = (start + count) * 4;
            out.extend_from_slice(&row[from..to]);
        }
        out
    }

    fn oriented(path: &Path) -> (u32, u32) {
        let info = stills().probe(path).expect("probe");
        let width = info.width.expect("width");
        let height = info.height.expect("height");
        match info.display {
            DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (height, width),
            DisplayQuarter::Zero | DisplayQuarter::Half => (width, height),
        }
    }

    #[test]
    fn a_still_crop_keeps_the_left_half_and_ignores_the_ratio_label() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine/fixtures/pattern.png");
        let (width, height) = oriented(&path);
        assert_eq!((width, height), (16, 16));
        let full = still_frame(&path, width, height, Crop::default(), "free");
        let half = width / 2;
        let left = still_frame(&path, half, height, left_half(), "free");
        let labeled = still_frame(&path, half, height, left_half(), "square");
        assert_eq!(left, labeled);
        assert_eq!(left, columns(&full, width as usize, 0, half as usize));
        assert_ne!(
            columns(&full, width as usize, 0, half as usize),
            columns(&full, width as usize, half as usize, half as usize)
        );
    }

    #[test]
    fn a_missing_still_with_a_crop_stays_clear() {
        let path = Path::new("/no/such/still.png");
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        let mut video = MaterialVideo::new("still");
        video.path = path.display().to_string();
        video.crop = left_half();
        draft.materials.videos.push(video);
        let mut segment = Segment::new("still", "still", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(segment);
        let before = draft.clone();
        let mut device = SoftwareDevice::new();
        let media = stills();
        let mut clips = OpenClips::new();
        let rgba = frame(
            &Snapshot::new(&draft),
            Micros(0),
            &mut device,
            &media,
            &mut clips,
            0,
        )
        .expect("frame")
        .rgba;
        assert_eq!(rgba, vec![0, 0, 0, 0]);
        assert_eq!(draft, before);
    }

    #[test]
    fn a_segment_canvas_fills_the_letterbox_around_the_still() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine/fixtures/pattern.png");
        let picture = still_frame(&path, 16, 16, Crop::default(), "free");
        let covered = crate::canvas::cover(
            &CpuBytes::new(16, 16, picture.clone()).expect("picture"),
            16,
            18,
        )
        .expect("cover");
        let project = framed(&path, Some("#0000FF"), None);
        assert_eq!(row(&project, 16, 0), [0, 0, 255, 255].repeat(16));
        assert_eq!(row(&project, 16, 17), [0, 0, 255, 255].repeat(16));
        assert_eq!(rows(&project, 16, 1, 16), picture);

        let color = framed(
            &path,
            Some("#0000FF"),
            Some(canvas("canvas_color", "#FF0000", 0.0, "")),
        );
        assert_eq!(row(&color, 16, 0), [255, 0, 0, 255].repeat(16));
        assert_eq!(rows(&color, 16, 1, 16), picture);

        let image = framed(
            &path,
            None,
            Some(canvas(
                "canvas_image",
                "#00FF00",
                0.0,
                path.to_str().expect("path"),
            )),
        );
        assert_eq!(row(&image, 16, 0), row(covered.rgba(), 16, 0));
        assert_eq!(rows(&image, 16, 1, 16), picture);

        let missing = framed(
            &path,
            None,
            Some(canvas(
                "canvas_image",
                "#00FF00",
                0.0,
                "/no/such/canvas.png",
            )),
        );
        assert_eq!(row(&missing, 16, 0), [0, 255, 0, 255].repeat(16));

        let blur = framed(&path, None, Some(canvas("canvas_blur", "", 0.0, "")));
        assert_eq!(row(&blur, 16, 0), row(covered.rgba(), 16, 0));
        assert_eq!(rows(&blur, 16, 1, 16), picture);

        // A keyed color fills the letterbox as keyed at the frame's time.
        let [r, g, b] = cutlass_model::CANVAS_COLOR_KEYS;
        let keys = [
            (r, 0, 1.0),
            (r, 1_000_000, 0.0),
            (g, 0, 0.0),
            (b, 0, 0.0),
            (b, 1_000_000, 1.0),
        ];
        let red = Some(canvas("canvas_color", "#00FF00", 0.0, ""));
        let start = framed_at(&path, None, red.clone(), &keys, 0);
        assert_eq!(row(&start, 16, 0), [255, 0, 0, 255].repeat(16));
        let middle = framed_at(&path, None, red, &keys, 500_000);
        assert_eq!(row(&middle, 16, 0), [128, 0, 128, 255].repeat(16));
        assert_eq!(rows(&middle, 16, 1, 16), picture);
    }

    fn canvas(kind: &str, color: &str, blur: f64, image: &str) -> MaterialCanvas {
        MaterialCanvas {
            id: "canvas".into(),
            kind: kind.to_owned(),
            color: color.to_owned(),
            blur,
            image: image.to_owned(),
            ..MaterialCanvas::default()
        }
    }

    fn framed(path: &Path, background: Option<&str>, canvas: Option<MaterialCanvas>) -> Vec<u8> {
        framed_at(path, background, canvas, &[], 0)
    }

    /// [`framed`] with `keys` (name, offset, value) on the still's clip,
    /// drawn at `time`.
    fn framed_at(
        path: &Path,
        background: Option<&str>,
        canvas: Option<MaterialCanvas>,
        keys: &[(&str, i64, f64)],
        time: i64,
    ) -> Vec<u8> {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 16;
        draft.canvas_config.height = 18;
        draft.canvas_config.background = background.map(str::to_owned);
        let mut video = MaterialVideo::new("still");
        video.path = path.display().to_string();
        draft.materials.videos.push(video);
        let mut segment = Segment::new("still", "still", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        for (name, offset, value) in keys {
            cutlass_model::put_key(&mut segment, name, Micros(*offset), *value);
        }
        if let Some(canvas) = canvas {
            segment.extra_material_refs.push(canvas.id.clone());
            draft.materials.canvases.push(canvas);
        }
        draft.tracks[0].segments.push(segment);
        let before = draft.clone();
        let mut device = SoftwareDevice::new();
        let media = stills();
        let mut clips = OpenClips::new();
        let rgba = frame(
            &Snapshot::new(&draft),
            Micros(time),
            &mut device,
            &media,
            &mut clips,
            0,
        )
        .expect("frame")
        .rgba;
        assert_eq!(draft, before);
        rgba
    }

    fn row(rgba: &[u8], width: usize, y: usize) -> &[u8] {
        let stride = width * 4;
        &rgba[y * stride..(y + 1) * stride]
    }

    fn rows(rgba: &[u8], width: usize, y: usize, count: usize) -> Vec<u8> {
        let stride = width * 4;
        rgba[y * stride..(y + count) * stride].to_vec()
    }

    #[test]
    fn a_local_still_crops_to_its_left_half() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/04-motion-graphics.png");
        if !path.is_file() {
            eprintln!("skipped: ../local-files is missing");
            return;
        }
        let (width, height) = oriented(&path);
        assert_eq!(width % 2, 0);
        let half = width / 2;
        let full = still_frame(&path, width, height, Crop::default(), "free");
        let left = still_frame(&path, half, height, left_half(), "free");
        assert_eq!(left, columns(&full, width as usize, 0, half as usize));
    }
}

#[cfg(test)]
mod recipe_tests {
    use cutlass_effects::bundled_recipe;
    use cutlass_engine::ReaderFactory;
    use cutlass_model::{
        Draft, MaterialColorConfig, MaterialVideoEffect, Micros, Segment, TimeRange, Track,
        TrackAttribute, TrackType,
    };
    use cutlass_renderer_software::SoftwareDevice;

    use super::{Snapshot, layers_in};
    use crate::gpu::{FrameLayers, GpuSource};
    use crate::media::OpenClips;

    fn colour(draft: &mut Draft, id: &str, render_index: i32, track: usize) {
        draft.materials.material_colors.push(MaterialColorConfig {
            id: id.into(),
            is_color_clip: true,
            solid_color: "0xFFFF0000".to_owned(),
            ..MaterialColorConfig::default()
        });
        let mut segment = Segment::new(id, id, TimeRange::new(0, 2_000_000));
        segment.render_index = render_index;
        draft.tracks[track].segments.push(segment);
    }

    /// A red clip with `effect_id` on an effect row from 0.5 s for 1 s.
    fn with_effect(effect_id: &str) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 4;
        draft.canvas_config.height = 2;
        colour(&mut draft, "below", 0, 0);
        draft.materials.video_effects.push(MaterialVideoEffect {
            id: "fx".into(),
            effect_id: effect_id.to_owned(),
            ..MaterialVideoEffect::default()
        });
        let mut row = Track::main("effects");
        row.kind = TrackType::Effect;
        let mut segment = Segment::new("fx", "fx", TimeRange::new(500_000, 1_000_000));
        segment.render_index = TrackType::Effect.render_floor().unwrap_or(0);
        row.segments.push(segment);
        draft.tracks.push(row);
        draft
    }

    fn layers(draft: &Draft, time: i64) -> FrameLayers {
        let mut device = SoftwareDevice::new();
        let media = ReaderFactory::new();
        let mut clips = OpenClips::new();
        layers_in(
            &Snapshot::new(draft),
            Micros(time),
            &mut device,
            &media,
            &mut clips,
            0,
            None,
        )
        .expect("layers")
    }

    fn effects(frame: &FrameLayers) -> Vec<usize> {
        frame
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| matches!(layer.source, GpuSource::Effect(..)))
            .map(|(index, _)| index)
            .collect()
    }

    #[test]
    fn a_recipe_effect_is_a_gpu_layer_over_what_is_below_with_its_own_clock() {
        let draft = with_effect("recipe/white-flash");
        let before = draft.clone();
        let frame = layers(&draft, 600_000);
        assert_eq!(effects(&frame), vec![frame.layers.len() - 1]);
        let Some(GpuSource::Effect(steps, _)) = frame.layers.last().map(|layer| &layer.source)
        else {
            panic!("no effect layer");
        };
        let recipe = bundled_recipe("recipe/white-flash").expect("recipe");
        assert_eq!(steps, &recipe.steps_at(Micros(100_000)));
        assert_eq!(draft, before);
    }

    // The one test here that loads a studio folder: the studio's recipes
    // are shared by every test in this process.
    #[test]
    fn a_studio_recipe_is_a_gpu_layer_like_a_bundled_one() {
        let studio =
            std::env::temp_dir().join(format!("cutlass-eval-studio-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&studio);
        let draft = with_effect("recipe/studio-soft");
        // Before the studio holds it, the id names nothing a frame draws.
        assert_eq!(effects(&layers(&draft, 600_000)), Vec::<usize>::new());
        let recipe = cutlass_effects::Recipe::read(
            br#"{"version":2,"id":"studio-soft","name":"Soft","passes":[{"block":"blur","params":{"radius":{"keys":[[0,0],[1,0.05]],"ease":["out"]}}}]}"#,
        )
        .expect("a recipe");
        cutlass_effects::write_studio_recipe(&studio, &recipe).expect("saved");
        cutlass_effects::load_studio(&studio);
        let frame = layers(&draft, 600_000);
        assert_eq!(effects(&frame), vec![frame.layers.len() - 1]);
        let Some(GpuSource::Effect(steps, _)) = frame.layers.last().map(|layer| &layer.source)
        else {
            panic!("no effect layer");
        };
        assert_eq!(steps, &recipe.steps_at(Micros(100_000)));
        cutlass_effects::clear_studio();
        assert_eq!(effects(&layers(&draft, 600_000)), Vec::<usize>::new());
        let _ = std::fs::remove_dir_all(studio);
    }

    #[test]
    fn an_effect_runs_with_the_controls_stored_on_its_clip() {
        let mut draft = with_effect("recipe/photo-drop");
        let effect = draft
            .materials
            .video_effects
            .last_mut()
            .expect("the effect");
        cutlass_effects::store_control(effect, "speed", 2.0, 1.0);
        let frame = layers(&draft, 600_000);
        let Some(GpuSource::Effect(steps, _)) = frame.layers.last().map(|layer| &layer.source)
        else {
            panic!("no effect layer");
        };
        let recipe = bundled_recipe("recipe/photo-drop").expect("recipe");
        let fast = [("speed".to_owned(), 2.0)].into_iter().collect();
        assert_eq!(steps, &recipe.steps_with(Micros(100_000), &fast));
        assert_eq!(steps, &recipe.steps_at(Micros(200_000)), "twice as far in");
    }

    #[test]
    fn an_effect_with_a_mask_carries_it_placed_on_the_canvas() {
        let mut draft = with_effect("recipe/vignette");
        let frame = layers(&draft, 600_000);
        let Some(GpuSource::Effect(_, mask)) = frame.layers.last().map(|layer| &layer.source)
        else {
            panic!("no effect layer");
        };
        assert_eq!(*mask, None);
        let shape = cutlass_effects::Mask::new(cutlass_effects::MaskShape::Circle);
        draft
            .materials
            .common_mask
            .push(cutlass_model::RawMaterial {
                id: "fx-mask".into(),
                kind: cutlass_effects::COLOR_MASK.to_owned(),
                fields: shape.to_fields(),
            });
        let row = draft.tracks.last_mut().expect("the effect row");
        row.segments[0].extra_material_refs.push("fx-mask".into());
        let frame = layers(&draft, 600_000);
        let Some(GpuSource::Effect(_, Some(mask))) = frame.layers.last().map(|layer| &layer.source)
        else {
            panic!("no masked effect layer");
        };
        assert_eq!(mask.mask, shape);
        // The canvas is 4 by 2; its middle is the shape's middle, and a
        // corner is outside a circle half the canvas across.
        assert!(mask.coverage_at(2.0, 1.0) > 0.99);
        assert!(mask.coverage_at(0.0, 0.0) < 0.01);
    }

    #[test]
    fn a_layer_with_a_higher_index_stays_above_the_effect() {
        let mut draft = with_effect("recipe/vignette");
        let top = draft.tracks.len() - 1;
        colour(&mut draft, "above", 20_000, top);
        let frame = layers(&draft, 600_000);
        let at = effects(&frame);
        assert_eq!(at.len(), 1);
        assert_eq!(at[0], frame.layers.len() - 2);
    }

    #[test]
    fn no_effect_layer_outside_its_span_on_a_hidden_row_or_when_unknown() {
        let draft = with_effect("recipe/vignette");
        assert!(effects(&layers(&draft, 100_000)).is_empty());
        assert!(effects(&layers(&draft, 1_500_000)).is_empty());

        let mut hidden = draft.clone();
        if let Some(row) = hidden.tracks.last_mut() {
            row.attribute = TrackAttribute(TrackAttribute::HIDE);
        }
        assert!(effects(&layers(&hidden, 600_000)).is_empty());

        let mut invisible = draft;
        if let Some(row) = invisible.tracks.last_mut() {
            row.segments[0].visible = false;
        }
        assert!(effects(&layers(&invisible, 600_000)).is_empty());
    }

    #[test]
    fn a_bound_recipe_runs_inside_its_clip_alone() {
        let mut draft = with_effect("recipe/vignette");
        let top = draft.tracks.len() - 1;
        colour(&mut draft, "above", 20_000, top);
        if let Some(effect) = draft.tracks[top]
            .segments
            .iter_mut()
            .find(|segment| segment.id.as_str() == "fx")
        {
            effect.bind_to(Some(&"below".into()));
        }
        let frame = layers(&draft, 600_000);
        assert!(effects(&frame).is_empty(), "no effect over the whole frame");
        let nested: Vec<&FrameLayers> = frame
            .layers
            .iter()
            .filter_map(|layer| match &layer.source {
                GpuSource::Nested(child) => Some(child.as_ref()),
                _ => None,
            })
            .collect();
        assert_eq!(nested.len(), 1);
        assert_eq!(effects(nested[0]), vec![nested[0].layers.len() - 1]);
        assert!(nested[0].layers.len() >= 2, "the clip, then its effect");
        assert!(
            !matches!(
                frame.layers.last().map(|layer| &layer.source),
                Some(GpuSource::Nested(_))
            ),
            "the clip above is drawn after, untouched"
        );

        let mut gone = draft.clone();
        gone.tracks[0].segments.clear();
        let frame = layers(&gone, 600_000);
        assert!(effects(&frame).is_empty());
        assert!(
            frame
                .layers
                .iter()
                .all(|layer| !matches!(layer.source, GpuSource::Nested(_)))
        );
    }

    #[test]
    fn an_unknown_recipe_is_not_run() {
        let mut device = SoftwareDevice::new();
        let media = ReaderFactory::new();
        let mut clips = OpenClips::new();
        let draft = with_effect("recipe/no-such-effect");
        let layers = layers_in(
            &Snapshot::new(&draft),
            Micros(600_000),
            &mut device,
            &media,
            &mut clips,
            0,
            None,
        )
        .expect("frame");
        assert!(effects(&layers).is_empty());
    }
}
