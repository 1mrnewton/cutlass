use std::fmt;

use cutlass_compositor::{Device, Extent, Frame};
use cutlass_effects::{MissingPackages, PackageStore};
use cutlass_engine::ReaderFactory;
use cutlass_generated::{PictureSource, SolidColor};
use cutlass_model::{Draft, Micros};

use crate::effect::{StackLayer, present};
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
/// is skipped. The project canvas color is under every layer. An open file
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
    compose(snapshot, time, device, media, clips, revision, &[])
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
    };
    let nests = nest_layers(draft, &mut pass)?;
    let layers = ordered(
        solid_layers(draft, extent.width, extent.height),
        text_layers(draft, pass.fonts, extent.width, extent.height, pass.time),
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
    let stack = stack_of(&layers);
    present(
        draft,
        snapshot.packages(),
        pass.time,
        pass.device,
        extent,
        &stack,
    )
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
    Nest(NestedPicture),
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
            Paint::Nest(nested) => nested,
        }
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
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 16;
        draft.canvas_config.height = 18;
        draft.canvas_config.background = background.map(str::to_owned);
        let mut video = MaterialVideo::new("still");
        video.path = path.display().to_string();
        draft.materials.videos.push(video);
        let mut segment = Segment::new("still", "still", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
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
