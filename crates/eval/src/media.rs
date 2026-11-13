//! Video pictures for one evaluated frame.
//!
//! A segment the factory cannot open is omitted. The other layers still draw.
//! Identity fits the decoded picture inside the canvas, keeping its shape,
//! and centers it, so a smaller picture is scaled up and a larger one down.
//! Scale 1 is that fitted picture. The segment
//! clip then scales, rotates, flips, and moves that picture, and multiplies
//! its opacity. Rotation is degrees counterclockwise with positive Y up.
//! A translation of 1 is half the canvas. A file display quarter turn is
//! clockwise and is applied before that fit. The video material's crop quad
//! is in that oriented picture, from the top left, with Y down. An all-zero
//! quad is the whole picture. `crop_ratio` is not read. `crop_scale` above 1
//! shrinks the quad around its center. Alpha, rotation, position, and scale
//! keyframes are measured from the segment start. A line, an empty curve,
//! `none`, and an unknown curve stay straight and ignore their handles. Any
//! other curve is a cubic through the earlier key's right handle and the
//! later key's left handle. A graph id is not sampled. A segment canvas is
//! drawn behind that fitted picture.
//!
//! [`OpenClips`] keeps a file open for one draft revision. A later time seeks
//! that file. A failed open is not kept, so the next time tries again.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cutlass_codec::access::{ColorMatrix, ColorPrimaries, ColorRange, PixelFormat, Transfer};
use cutlass_color::{Matrix, Signal, SignalCurve};
use cutlass_engine::{
    ClockKind, DecodedFrame, DisplayQuarter, ReaderFactory, VideoSource, decoded_picture,
    open_segment_from, open_segment_in,
};
use cutlass_gpu::{Affine, Placement, ToneCurve, YuvTone};
use cutlass_model::{
    Clip, Crop, Draft, DraftId, MaterialCanvas, MaterialRef, Micros, Segment, SegmentId, TimeRange,
    TrackAttribute, UniformScale,
};
use cutlass_picture::{CpuBytes, Picture, PictureSource, Sample, rgba_len};

use crate::canvas::{Backdrop, cover, image_backdrop, over_backdrop, segment_backdrop};
use crate::gpu::{GpuBackdrop, GpuLayer, GpuSource, YuvSamples};

/// Pictures each open clip decodes ahead of the time asked for.
const READ_AHEAD: usize = 8;

/// Open one clip and start its decode thread. A clip whose thread cannot
/// start still decodes on the caller's thread.
fn open_clip(
    media: &ReaderFactory,
    draft: &Draft,
    id: &SegmentId,
    root: Option<&Path>,
    patient: bool,
) -> Option<VideoSource> {
    // A clip with an enhanced copy draws from it; a copy that will not
    // open leaves the clip its own file.
    let copy = crate::enhance::enhanced_copy(draft, root, id).and_then(|(path, from)| {
        open_segment_from(media, draft, id, ClockKind::Export, 1, path, from).ok()
    });
    let mut source = match copy {
        Some(source) => source,
        None => open_segment_in(media, draft, id, ClockKind::Export, 1, root).ok()?,
    };
    let _ = source.read_ahead(media, READ_AHEAD);
    source.set_patient(patient);
    Some(source)
}

/// Open the depth video that lights clip `id`, when it has one, decoding
/// ahead like the clip.
fn open_depth(
    media: &ReaderFactory,
    draft: &Draft,
    id: &SegmentId,
    root: Option<&Path>,
    patient: bool,
) -> Option<VideoSource> {
    let (path, from) = crate::relight::relight_depth(draft, root, id)?;
    let mut source = open_segment_from(media, draft, id, ClockKind::Export, 1, path, from).ok()?;
    let _ = source.read_ahead(media, READ_AHEAD);
    source.set_patient(patient);
    Some(source)
}

/// A shape drawn once and shown on every frame after: the shape as it was
/// drawn, the density it was asked for, the density it was drawn at, and
/// the picture.
#[derive(Clone)]
pub(crate) struct KeptShape {
    pub(crate) shape: cutlass_model::MaterialShape,
    pub(crate) wanted: f64,
    pub(crate) drawn: f64,
    pub(crate) raster: CpuBytes,
}

/// Open video files for one draft revision.
pub struct OpenClips {
    draft: Option<DraftId>,
    revision: u64,
    bound: bool,
    /// Project folder for a relative media path. An absolute path ignores it.
    root: Option<std::path::PathBuf>,
    sources: HashMap<SegmentId, VideoSource>,
    /// Relit clips' depth videos for this revision; `None` is a clip
    /// whose depth is not made yet.
    relights: HashMap<SegmentId, Option<VideoSource>>,
    nests: HashMap<SegmentId, OpenClips>,
    images: HashMap<String, CpuBytes>,
    /// Title rasters by segment, with the density each was drawn at. A
    /// title does not change within a revision unless its scale is keyed.
    texts: HashMap<SegmentId, (f64, CpuBytes)>,
    /// Shape rasters by segment. A move or another clip's edit leaves a
    /// shape as it was, so these outlive a revision; each is checked
    /// against its shape before it is shown.
    shapes: HashMap<SegmentId, KeptShape>,
    /// Off, a clip whose picture is not decoded yet shows its last one.
    patient: bool,
    /// Lottie files by stored path; `None` is a file that did not read.
    lotties: HashMap<String, Option<Arc<cutlass_lottie::Animation>>>,
    /// Steadied clips' plans, by segment.
    steadies: HashMap<SegmentId, Arc<crate::steady::Steady>>,
    /// Reframed clips' planned windows, by segment.
    reframes: HashMap<SegmentId, Arc<crate::reframe::Reframed>>,
}

impl OpenClips {
    #[must_use]
    pub fn new() -> Self {
        Self {
            draft: None,
            revision: 0,
            bound: false,
            root: None,
            sources: HashMap::new(),
            relights: HashMap::new(),
            nests: HashMap::new(),
            images: HashMap::new(),
            texts: HashMap::new(),
            shapes: HashMap::new(),
            patient: true,
            lotties: HashMap::new(),
            steadies: HashMap::new(),
            reframes: HashMap::new(),
        }
    }

    /// Wait for every clip's picture, as export must, or show a clip's last
    /// picture while the one asked for decodes, as preview does.
    pub fn set_patient(&mut self, patient: bool) {
        self.patient = patient;
        for source in self
            .sources
            .values()
            .chain(self.relights.values().flatten())
        {
            source.set_patient(patient);
        }
        for nest in self.nests.values_mut() {
            nest.set_patient(patient);
        }
    }

    /// Whether any clip showed an earlier picture than the one asked for
    /// since the last call. A caller redraws until this is false.
    pub fn take_stale(&mut self) -> bool {
        let mut stale = false;
        for source in self
            .sources
            .values()
            .chain(self.relights.values().flatten())
        {
            stale |= source.take_stale();
        }
        for nest in self.nests.values_mut() {
            stale |= nest.take_stale();
        }
        stale
    }

    /// Drops every open file.
    pub fn clear(&mut self) {
        self.sources.clear();
        self.relights.clear();
        self.nests.clear();
        self.images.clear();
        self.texts.clear();
        self.shapes.clear();
        self.lotties.clear();
        self.steadies.clear();
        self.reframes.clear();
        self.draft = None;
        self.revision = 0;
        self.bound = false;
    }

    /// The Lottie animation stored at `stored`, read once per revision. A
    /// relative path stays inside the project folder. A file past
    /// [`cutlass_lottie::limits::MAX_BYTES`], unreadable or not an animation
    /// is `None`.
    pub(crate) fn lottie(&mut self, stored: &str) -> Option<Arc<cutlass_lottie::Animation>> {
        if let Some(known) = self.lotties.get(stored) {
            return known.clone();
        }
        let path = if Path::new(stored).is_absolute() {
            Some(std::path::PathBuf::from(stored))
        } else {
            match &self.root {
                Some(root) => cutlass_engine::media_in_project(root, stored),
                None => Some(std::path::PathBuf::from(stored)),
            }
        };
        let read = path.and_then(|path| {
            let size = std::fs::metadata(&path).ok()?.len();
            if size > cutlass_lottie::limits::MAX_BYTES as u64 {
                return None;
            }
            let bytes = std::fs::read(&path).ok()?;
            cutlass_lottie::Animation::parse(&bytes).ok().map(Arc::new)
        });
        self.lotties.insert(stored.to_owned(), read.clone());
        read
    }

    /// The light over `segment` at `time`, opening its depth video the
    /// first time it is lit this revision. `None` when it is not relit.
    fn light(
        &mut self,
        media: &ReaderFactory,
        draft: &Draft,
        segment: &Segment,
        time: Micros,
    ) -> Option<cutlass_effects::Lighting> {
        let lit = crate::relight::light_at(draft, segment, time)?;
        if !self.relights.contains_key(&segment.id) {
            let depth = open_depth(
                media,
                draft,
                &segment.id,
                self.root.as_deref(),
                self.patient,
            );
            self.relights.insert(segment.id.clone(), depth);
        }
        Some(lit)
    }

    /// The depth video opened for clip `id`.
    fn depth(&self, id: &SegmentId) -> Option<&VideoSource> {
        self.relights.get(id).and_then(Option::as_ref)
    }

    /// The project folder, when there is one.
    pub(crate) fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    /// Steadied clips' plans, kept until the clips are dropped.
    pub(crate) fn steadies(&mut self) -> &mut HashMap<SegmentId, Arc<crate::steady::Steady>> {
        &mut self.steadies
    }

    /// Reframed clips' planned windows, kept until the clips are dropped.
    pub(crate) fn reframes(&mut self) -> &mut HashMap<SegmentId, Arc<crate::reframe::Reframed>> {
        &mut self.reframes
    }

    /// Shape rasters, kept across revisions.
    pub(crate) fn shapes(&mut self) -> &mut HashMap<SegmentId, KeptShape> {
        &mut self.shapes
    }

    /// Title rasters kept for this revision.
    pub(crate) fn texts(&mut self) -> &mut HashMap<SegmentId, (f64, CpuBytes)> {
        &mut self.texts
    }

    /// Open files for the compound segment `id`.
    ///
    /// Each compound keeps its own child files. A second segment that stores
    /// its own copy of a nest does not share them.
    pub(crate) fn child(&mut self, id: &SegmentId) -> &mut Self {
        let root = self.root.clone();
        let patient = self.patient;
        let child = self.nests.entry(id.clone()).or_default();
        if child.root.is_none() {
            child.root = root;
        }
        if child.patient != patient {
            child.set_patient(patient);
        }
        child
    }

    /// The folder a relative media path is joined to. `None` opens a relative path from the process directory.
    pub fn set_project(&mut self, root: Option<&Path>) {
        self.root = root.map(Path::to_path_buf);
    }

    /// Open the video clips that start after `time` and within `horizon`,
    /// and start decoding each one's first picture, so playback reaching a
    /// cut finds it ready instead of opening and seeking on that frame.
    pub fn preroll(
        &mut self,
        media: &ReaderFactory,
        draft: &Draft,
        revision: u64,
        time: Micros,
        horizon: Micros,
    ) {
        self.prepare(draft, revision);
        let until = time.0.saturating_add(horizon.0);
        for track in &draft.tracks {
            if track.attribute.contains(TrackAttribute::HIDE) {
                continue;
            }
            for segment in &track.segments {
                let start = segment.target.start.0;
                if !segment.visible
                    || segment.source_range.is_none()
                    || start <= time.0
                    || start > until
                    || !matches!(
                        draft.materials.get(segment.material_id.as_str()),
                        Some(MaterialRef::Video(video)) if !cutlass_lottie::is_lottie_path(&video.path)
                    )
                {
                    continue;
                }
                if !self.sources.contains_key(&segment.id) {
                    let Some(source) = open_clip(
                        media,
                        draft,
                        &segment.id,
                        self.root.as_deref(),
                        self.patient,
                    ) else {
                        continue;
                    };
                    self.sources.insert(segment.id.clone(), source);
                }
                if let Some(source) = self.sources.get(&segment.id) {
                    source.prefetch(segment.target.start);
                }
            }
        }
    }

    pub(crate) fn prepare(&mut self, draft: &Draft, revision: u64) {
        if self.bound && self.draft.as_ref() == Some(&draft.id) && self.revision == revision {
            return;
        }
        self.sources.clear();
        self.relights.clear();
        self.nests.clear();
        self.images.clear();
        self.texts.clear();
        self.lotties.clear();
        self.draft = Some(draft.id.clone());
        self.revision = revision;
        self.bound = true;
    }
}

impl Default for OpenClips {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) struct VideoLayer<'a> {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    pub(crate) placed: Placed<'a>,
}

/// Scale, rotation, flip, translation, and opacity for one picture.
///
/// `scale_x` and `scale_y` are positive. A negative clip scale flips that
/// axis and uses the absolute value. `sin` and `cos` are the counterclockwise
/// rotation, with positive Y up. `dx` is pixels to the right and `dy` is
/// pixels up. One translation unit is half the canvas on that axis.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PictureTransform {
    scale_x: f64,
    scale_y: f64,
    sin: f64,
    cos: f64,
    flip_h: bool,
    flip_v: bool,
    dx: f64,
    dy: f64,
    opacity: u8,
    geometry_identity: bool,
    /// Keep the picture's own size instead of fitting it to the canvas.
    native: bool,
    /// Picture pixels to one drawn pixel of a native picture.
    density: f64,
}

pub(crate) struct Placed<'a> {
    source: &'a VideoSource,
    width: u32,
    height: u32,
    transform: PictureTransform,
    crop: Crop,
    crop_scale: f64,
    backdrop: Backdrop,
    /// The steadying correction at this frame.
    steady: Option<cutlass_motion::Correction>,
    /// The clip's depth video and the light over it at this frame.
    relight: Option<(&'a VideoSource, cutlass_effects::Lighting)>,
}

pub(crate) fn picture_transform(
    clip: &Clip,
    uniform: &UniformScale,
    canvas_w: u32,
    canvas_h: u32,
) -> Option<PictureTransform> {
    let (mut scale_x, mut scale_y) = if uniform.on {
        (uniform.value, uniform.value)
    } else {
        (clip.scale.x, clip.scale.y)
    };
    if !scale_x.is_finite()
        || !scale_y.is_finite()
        || !clip.rotation.is_finite()
        || !clip.transform.x.is_finite()
        || !clip.transform.y.is_finite()
        || !clip.alpha.is_finite()
    {
        return None;
    }
    let mut flip_h = clip.flip.horizontal;
    let mut flip_v = clip.flip.vertical;
    if scale_x < 0.0 {
        flip_h = !flip_h;
        scale_x = -scale_x;
    }
    if scale_y < 0.0 {
        flip_v = !flip_v;
        scale_y = -scale_y;
    }
    let degrees = clip.rotation.rem_euclid(360.0);
    let (sin, cos) = sin_cos(degrees);
    let dx = clip.transform.x * f64::from(canvas_w) / 2.0;
    let dy = clip.transform.y * f64::from(canvas_h) / 2.0;
    let opacity = opacity_byte(clip.alpha)?;
    let geometry_identity = scale_x == 1.0
        && scale_y == 1.0
        && degrees == 0.0
        && !flip_h
        && !flip_v
        && dx == 0.0
        && dy == 0.0;
    Some(PictureTransform {
        scale_x,
        scale_y,
        sin,
        cos,
        flip_h,
        flip_v,
        dx,
        dy,
        opacity,
        geometry_identity,
        native: false,
        density: 1.0,
    })
}

fn sin_cos(degrees: f64) -> (f64, f64) {
    if degrees == 0.0 {
        (0.0, 1.0)
    } else if degrees == 90.0 {
        (1.0, 0.0)
    } else if degrees == 180.0 {
        (0.0, -1.0)
    } else if degrees == 270.0 {
        (-1.0, 0.0)
    } else {
        let radians = degrees.to_radians();
        (radians.sin(), radians.cos())
    }
}

fn opacity_byte(alpha: f64) -> Option<u8> {
    if !alpha.is_finite() {
        return None;
    }
    if alpha <= 0.0 {
        return Some(0);
    }
    if alpha >= 1.0 {
        return Some(255);
    }
    let scaled = (alpha * 255.0).round();
    let mut byte = 0_u8;
    let mut cursor = 0.0_f64;
    while cursor + 0.5 < scaled && byte < 255 {
        byte = byte.saturating_add(1);
        cursor += 1.0;
    }
    Some(byte)
}

/// Visible video segments whose target covers `time`.
///
/// A segment already in `clips` is sampled from that file. A failed open is
/// left out of `clips`.
pub(crate) fn video_layers<'a>(
    draft: &Draft,
    media: &ReaderFactory,
    width: u32,
    height: u32,
    time: Micros,
    clips: &'a mut OpenClips,
) -> Vec<VideoLayer<'a>> {
    let mut chosen = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            if !segment.visible || segment.source_range.is_none() {
                continue;
            }
            if !covers(segment.target.start.0, segment.target.duration.0, time.0) {
                continue;
            }
            let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            // A Lottie sticker is drawn from its geometry, not decoded.
            if cutlass_lottie::is_lottie_path(&video.path) {
                continue;
            }
            let crop = crate::reframe::crop_at(clips, draft, segment, time, (width, height))
                .unwrap_or(video.crop);
            let crop_scale = video.crop_scale;
            let Some((clip, uniform)) = cutlass_model::animated_at(draft, segment, time) else {
                continue;
            };
            let Some(transform) = picture_transform(&clip, &uniform, width, height) else {
                continue;
            };
            if transform.omitted() {
                continue;
            }
            if !clips.sources.contains_key(&segment.id) {
                let Some(source) = open_clip(
                    media,
                    draft,
                    &segment.id,
                    clips.root.as_deref(),
                    clips.patient,
                ) else {
                    continue;
                };
                clips.sources.insert(segment.id.clone(), source);
            }
            let backdrop = backdrop_for(draft, segment, media, clips, width, height, time);
            let steady = crate::steady::steady_at(clips, draft, segment, time);
            let light = clips.light(media, draft, segment, time);
            chosen.push((
                segment.render_index,
                track_index,
                segment_index,
                segment.id.clone(),
                transform,
                (crop, crop_scale, steady, light),
                backdrop,
            ));
        }
    }
    let mut layers = Vec::with_capacity(chosen.len());
    for (
        render_index,
        track_index,
        segment_index,
        id,
        transform,
        (crop, crop_scale, steady, light),
        backdrop,
    ) in chosen
    {
        let Some(source) = clips.sources.get(&id) else {
            continue;
        };
        let relight = clips.depth(&id).zip(light);
        layers.push(VideoLayer {
            render_index,
            track_index,
            segment_index,
            placed: Placed {
                source,
                width,
                height,
                transform,
                crop,
                crop_scale,
                backdrop,
                steady,
                relight,
            },
        });
    }
    layers
}

/// The placed picture of one video segment at `time`.
///
/// `time` must fall inside the target. A failed open is not kept. The draft
/// is not changed.
pub(crate) fn video_picture(
    draft: &Draft,
    segment: &Segment,
    media: &ReaderFactory,
    width: u32,
    height: u32,
    time: Micros,
    clips: &mut OpenClips,
) -> Option<CpuBytes> {
    if !segment.visible || segment.source_range.is_none() {
        return None;
    }
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    let crop =
        crate::reframe::crop_at(clips, draft, segment, time, (width, height)).unwrap_or(video.crop);
    let crop_scale = video.crop_scale;
    let (clip, uniform) = cutlass_model::animated_at(draft, segment, time)?;
    let transform = picture_transform(&clip, &uniform, width, height)?;
    if transform.omitted() {
        return None;
    }
    if !clips.sources.contains_key(&segment.id) {
        let source = open_clip(
            media,
            draft,
            &segment.id,
            clips.root.as_deref(),
            clips.patient,
        )?;
        clips.sources.insert(segment.id.clone(), source);
    }
    let backdrop = backdrop_for(draft, segment, media, clips, width, height, time);
    let steady = crate::steady::steady_at(clips, draft, segment, time);
    let light = clips.light(media, draft, segment, time);
    let source = clips.sources.get(&segment.id)?;
    let relight = clips.depth(&segment.id).zip(light);
    let placed = Placed {
        source,
        width,
        height,
        transform,
        crop,
        crop_scale,
        backdrop,
        steady,
        relight,
    };
    match placed.sample(time) {
        Sample::Ready(Picture::Cpu(bytes)) => Some(bytes),
        Sample::Ready(Picture::Platform(_)) | Sample::Miss => None,
    }
}

pub(crate) fn backdrop_for(
    draft: &Draft,
    segment: &Segment,
    media: &ReaderFactory,
    clips: &mut OpenClips,
    width: u32,
    height: u32,
    time: Micros,
) -> Backdrop {
    let Some(stored) = segment_canvas(draft, segment) else {
        return Backdrop::None;
    };
    // A keyed color fills the letterbox as keyed at `time`.
    let keyed = cutlass_model::hex_color_at(
        segment,
        &draft.keyframe_graph_list,
        &cutlass_model::CANVAS_COLOR_KEYS,
        time,
    )
    .map(|color| MaterialCanvas {
        color,
        ..stored.clone()
    });
    let canvas = keyed.as_ref().unwrap_or(stored);
    if canvas.kind == "canvas_image" {
        let image = canvas_image(canvas, media, clips, width, height);
        return image_backdrop(canvas, image);
    }
    segment_backdrop(canvas, width, height)
}

fn segment_canvas<'a>(draft: &'a Draft, segment: &Segment) -> Option<&'a MaterialCanvas> {
    segment
        .extra_material_refs
        .iter()
        .find_map(|id| match draft.materials.get(id.as_str()) {
            Some(MaterialRef::Canvas(canvas)) => Some(canvas),
            _ => None,
        })
}

fn canvas_image(
    canvas: &MaterialCanvas,
    media: &ReaderFactory,
    clips: &mut OpenClips,
    width: u32,
    height: u32,
) -> Option<CpuBytes> {
    let path = if canvas.image.is_empty() {
        canvas.album_image.as_str()
    } else {
        canvas.image.as_str()
    };
    if path.is_empty() {
        return None;
    }
    let decoded = if let Some(cached) = clips.images.get(path) {
        cached.clone()
    } else {
        let decoded = decode_canvas_image(media, path)?;
        clips.images.insert(path.to_owned(), decoded.clone());
        decoded
    };
    cover(&decoded, width, height)
}

fn decode_canvas_image(media: &ReaderFactory, path: &str) -> Option<CpuBytes> {
    let mut opened = media.open(Path::new(path)).ok()?;
    let display = opened.info().display;
    let held = opened.decode_at(Micros(0), &TimeRange::new(0, 1)).ok()?;
    let Picture::Cpu(bytes) = decoded_picture(&held.frame).ok()? else {
        return None;
    };
    orient(bytes, display)
}

/// The clip at `time`, eased by the draft's `graphs`; see
/// [`cutlass_model::clip_at`].
#[cfg(test)]
pub(crate) fn segment_clip(
    graphs: &[cutlass_model::KeyframeGraph],
    segment: &Segment,
    time: Micros,
) -> Option<(Clip, UniformScale)> {
    cutlass_model::clip_at(segment, graphs, time)
}

fn covers(start: i64, duration: i64, time: i64) -> bool {
    let Some(end) = start.checked_add(duration) else {
        return false;
    };
    time >= start && time < end
}

impl PictureSource for Placed<'_> {
    fn sample(&self, time: Micros) -> Sample {
        let Sample::Ready(picture) = self.source.sample(time) else {
            return Sample::Miss;
        };
        match picture {
            Picture::Cpu(bytes) => {
                let (bytes, display) = self.relit(bytes, time);
                finish_picture(
                    (self.width, self.height),
                    bytes,
                    self.transform,
                    display,
                    (self.crop, self.crop_scale),
                    self.steady.as_ref(),
                    &self.backdrop,
                )
            }
            Picture::Platform(_) => Sample::Miss,
        }
    }
}

impl Placed<'_> {
    /// The decoded `picture` and its display turn, or, when the clip is
    /// relit and its depth at `time` is decoded, the picture turned upright
    /// and lit from that depth, with no turn left.
    fn relit(&self, picture: CpuBytes, time: Micros) -> (CpuBytes, DisplayQuarter) {
        let display = self.source.display();
        let Some((depth, lit)) = &self.relight else {
            return (picture, display);
        };
        let Sample::Ready(Picture::Cpu(near)) = depth.sample(time) else {
            return (picture, display);
        };
        let (Some(upright), Some(near)) = (
            orient(picture.clone(), display),
            orient(near, depth.display()),
        ) else {
            return (picture, display);
        };
        (
            crate::relight::relit(upright, &near, lit),
            DisplayQuarter::Zero,
        )
    }

    /// This clip as GPU layers: a color backdrop, if any, then the decoded
    /// picture with the maps that place it.
    ///
    /// `None` when the GPU path does not reproduce the CPU path: a picture
    /// that is not 8-bit 4:2:0, one whose matrix is not BT.709 or whose
    /// range is full (the GPU matrix is BT.709 studio swing), one whose
    /// tags the color crate does not map, a crop
    /// quad that is not a parallelogram, or a blur or image backdrop (a
    /// relit clip's blur too). The caller then uses the CPU picture. A
    /// relit clip whose depth is decoded is drawn upright on its own
    /// canvas and lit there.
    pub(crate) fn gpu(&self, time: Micros) -> Option<Vec<GpuLayer>> {
        let (backdrop, blur) = match &self.backdrop {
            Backdrop::None => (None, None),
            Backdrop::Color(pixel) => (
                Some(GpuLayer::fill(CpuBytes::new(1, 1, pixel.to_vec()).ok()?)),
                None,
            ),
            Backdrop::Blur(radius) => (None, Some(*radius)),
            Backdrop::Image(_) => return None,
        };
        let (width, height, primaries, transfer, samples) =
            gpu_samples(self.source.decoded_at(time).ok()?)?;
        let tone = yuv_tone(primaries, transfer)?;
        if width % 2 != 0 || height % 2 != 0 {
            return None;
        }
        let display = self.source.display();
        let oriented = match display {
            DisplayQuarter::Zero | DisplayQuarter::Half => (width, height),
            DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (height, width),
        };
        let (to_oriented, picture) = gpu_crop(self.crop, self.crop_scale, oriented)?;
        let to_oriented = match &self.steady {
            Some(correction) => crate::steady::then(
                to_oriented,
                crate::steady::steady_map(correction, oriented)?,
            ),
            None => to_oriented,
        };
        let placement = Placement {
            to_picture: gpu_place(self.transform, picture, (self.width, self.height))?,
            picture,
            to_oriented,
            oriented,
            to_texture: gpu_turn(display, width, height),
            opacity: f64::from(self.transform.opacity) / 255.0,
        };
        let source = GpuSource::Yuv {
            width,
            height,
            samples,
            tone,
        };
        let mut layers: Vec<GpuLayer> = backdrop.into_iter().collect();
        if let Some((depth, lighting)) = self.lit_depth(time) {
            // A blurred backdrop is a copy of the lit picture: the CPU's.
            if blur.is_some() {
                return None;
            }
            layers.push(relit_layer(source, depth, lighting, placement));
            return Some(layers);
        }
        let backdrop = match blur {
            Some(radius) => Some(GpuBackdrop {
                cover: Placement {
                    to_picture: gpu_cover(picture, (self.width, self.height))?,
                    opacity: 1.0,
                    ..placement
                },
                radius,
            }),
            None => None,
        };
        layers.push(GpuLayer {
            source,
            placement,
            backdrop,
        });
        Some(layers)
    }

    /// The clip's depth at `time`, upright, and the light over it, when it
    /// is relit and that depth is decoded: what [`Self::relit`] lights
    /// from. `None` draws the clip unlit, as the CPU path does then.
    fn lit_depth(&self, time: Micros) -> Option<(CpuBytes, cutlass_effects::Lighting)> {
        let (depth, lit) = self.relight.as_ref()?;
        let Sample::Ready(Picture::Cpu(near)) = depth.sample(time) else {
            return None;
        };
        let near = orient(near, depth.display())?;
        if near.width() == 0 || near.height() == 0 {
            return None;
        }
        Some((near, lit.clone()))
    }
}

/// A relit clip as one layer: its decoded picture drawn upright on a
/// canvas of its own oriented size, lit there from `depth`, then that
/// canvas placed as `placement` places the picture, with no turn left.
fn relit_layer(
    source: GpuSource,
    depth: CpuBytes,
    lighting: cutlass_effects::Lighting,
    placement: Placement,
) -> GpuLayer {
    let oriented = placement.oriented;
    let upright = Placement {
        to_picture: Affine::IDENTITY,
        picture: oriented,
        to_oriented: Affine::IDENTITY,
        oriented,
        to_texture: placement.to_texture,
        opacity: 1.0,
    };
    let child = crate::gpu::FrameLayers {
        width: oriented.0,
        height: oriented.1,
        layers: vec![
            GpuLayer {
                source,
                placement: upright,
                backdrop: None,
            },
            GpuLayer {
                source: GpuSource::Relight { depth, lighting },
                placement: Placement::canvas(oriented.0, oriented.1),
                backdrop: None,
            },
        ],
    };
    GpuLayer {
        source: GpuSource::Nested(Box::new(child)),
        placement: Placement {
            to_texture: Affine::IDENTITY,
            ..placement
        },
        backdrop: None,
    }
}

/// The 4:2:0 samples the GPU matrix reproduces, with their size and tags:
/// 8-bit 4:2:0 whose matrix is BT.709 (or unnamed) in studio swing, or a
/// platform surface. `None` for every other picture: the CPU conversion
/// reads its own matrix, range, depth, and chroma layout.
fn gpu_samples(frame: DecodedFrame) -> Option<(u32, u32, ColorPrimaries, Transfer, YuvSamples)> {
    match frame {
        DecodedFrame::Coded {
            width,
            height,
            pixel: PixelFormat::Yuv420p,
            primaries,
            transfer,
            matrix: ColorMatrix::Unspecified | ColorMatrix::Bt709,
            range: ColorRange::Unspecified | ColorRange::Limited | ColorRange::Studio,
            bytes,
        } => Some((width, height, primaries, transfer, YuvSamples::Bytes(bytes))),
        DecodedFrame::Surface {
            width,
            height,
            primaries,
            transfer,
            surface,
        } => Some((
            width,
            height,
            primaries,
            transfer,
            YuvSamples::Surface(surface),
        )),
        DecodedFrame::Coded { .. } | DecodedFrame::Platform { .. } | DecodedFrame::Still(_) => None,
    }
}

/// The step the engine's CPU conversion applies after the BT.709 matrix,
/// for the draft working space it converts to, or `None` when the color
/// crate does not map these tags.
fn yuv_tone(primaries: ColorPrimaries, transfer: Transfer) -> Option<YuvTone> {
    let signal = Signal::for_tags(primaries, transfer, Matrix::Bt709, 0).ok()?;
    let curve = |curve| match curve {
        SignalCurve::Rec709 => ToneCurve::Rec709,
        SignalCurve::Hlg => ToneCurve::Hlg,
        SignalCurve::Pq => ToneCurve::Pq,
    };
    Some(YuvTone {
        source: curve(signal.source),
        dest: curve(signal.dest),
        adapt: signal.adapt,
        nits: signal.nits,
    })
}

/// Oriented-source pixel center to stored pixels, as [`turned_source`].
fn gpu_turn(display: DisplayQuarter, width: u32, height: u32) -> Affine {
    let (w, h) = (f64::from(width), f64::from(height));
    Affine(match display {
        DisplayQuarter::Zero => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        DisplayQuarter::Clockwise90 => [0.0, 1.0, 0.0, -1.0, 0.0, h],
        DisplayQuarter::Half => [-1.0, 0.0, w, 0.0, -1.0, h],
        DisplayQuarter::Clockwise270 => [0.0, -1.0, w, 1.0, 0.0, 0.0],
    })
}

/// Canvas pixel center to picture pixels for a copy that covers the canvas,
/// as [`crate::canvas::cover`]: scaled with left-edge sampling until both
/// edges reach the canvas, then centered.
fn gpu_cover(picture: (u32, u32), canvas: (u32, u32)) -> Option<Affine> {
    if picture == canvas {
        return Some(Affine::IDENTITY);
    }
    let (dw, dh) = crate::canvas::cover_size(picture.0, picture.1, canvas.0, canvas.1)?;
    let x0 = f64::from(dw.checked_sub(canvas.0)? / 2);
    let y0 = f64::from(dh.checked_sub(canvas.1)? / 2);
    let kx = f64::from(picture.0) / f64::from(dw);
    let ky = f64::from(picture.1) / f64::from(dh);
    Some(Affine([kx, 0.0, kx * x0, 0.0, ky, ky * y0]))
}

/// Picture pixel center to oriented-source pixels, and the picture size,
/// as [`apply_crop`]. A crop that is not a parallelogram is not affine.
pub(crate) fn gpu_crop(
    crop: Crop,
    crop_scale: f64,
    oriented: (u32, u32),
) -> Option<(Affine, (u32, u32))> {
    let uncropped = Some((Affine::IDENTITY, oriented));
    if crop_absent(crop) {
        return uncropped;
    }
    let quad = scale_quad(crop, crop_scale);
    if !quad_finite(quad) {
        return uncropped;
    }
    let (src_w, src_h) = oriented;
    let Some(out_w) = round_edge(edge_len(
        quad.upper_left_x,
        quad.upper_left_y,
        quad.upper_right_x,
        quad.upper_right_y,
        src_w,
        src_h,
    )) else {
        return uncropped;
    };
    let Some(out_h) = round_edge(edge_len(
        quad.upper_left_x,
        quad.upper_left_y,
        quad.lower_left_x,
        quad.lower_left_y,
        src_w,
        src_h,
    )) else {
        return uncropped;
    };
    let across = (
        quad.upper_right_x - quad.upper_left_x,
        quad.upper_right_y - quad.upper_left_y,
    );
    let below = (
        quad.lower_right_x - quad.lower_left_x,
        quad.lower_right_y - quad.lower_left_y,
    );
    if (across.0 - below.0).abs() > 1e-9 || (across.1 - below.1).abs() > 1e-9 {
        return None;
    }
    let down = (
        quad.lower_left_x - quad.upper_left_x,
        quad.lower_left_y - quad.upper_left_y,
    );
    let (w, h) = (f64::from(src_w), f64::from(src_h));
    let (ow, oh) = (f64::from(out_w), f64::from(out_h));
    Some((
        Affine([
            w * across.0 / ow,
            w * down.0 / oh,
            w * quad.upper_left_x,
            h * across.1 / ow,
            h * down.1 / oh,
            h * quad.upper_left_y,
        ]),
        (out_w, out_h),
    ))
}

/// Canvas pixel center to placed-picture pixels, as [`place_with`].
pub(crate) fn gpu_place(
    transform: PictureTransform,
    picture: (u32, u32),
    canvas: (u32, u32),
) -> Option<Affine> {
    let (pw, ph) = picture;
    let (cw, ch) = canvas;
    if transform.geometry_identity {
        if pw == cw && ph == ch {
            return Some(Affine::IDENTITY);
        }
        // Stamped at the floored center, each canvas pixel center on the
        // matching point of the picture.
        let (dw, dh) = transform.drawn_size(picture, canvas)?;
        let x0 = f64::from((cw - dw) / 2);
        let y0 = f64::from((ch - dh) / 2);
        let kx = f64::from(pw) / f64::from(dw);
        let ky = f64::from(ph) / f64::from(dh);
        return Some(Affine([kx, 0.0, -kx * x0, 0.0, ky, -ky * y0]));
    }
    let (fw, fh) = transform.drawn_size(picture, canvas)?;
    let (fw, fh) = (f64::from(fw), f64::from(fh));
    let (pw, ph) = (f64::from(pw), f64::from(ph));
    let cx = f64::from(cw) / 2.0 + transform.dx;
    let cy = f64::from(ch) / 2.0 - transform.dy;
    let (sin, cos) = (transform.sin, transform.cos);
    let (sx, sy) = (transform.scale_x, transform.scale_y);
    // lx = (cos*(x-cx) + sin*(cy-y)) / sx ; ly = (-sin*(x-cx) + cos*(cy-y)) / sy
    let lx = [cos / sx, -sin / sx, (-cos * cx + sin * cy) / sx];
    let ly = [-sin / sy, -cos / sy, (sin * cx + cos * cy) / sy];
    // su = lx * pw/fw + pw/2 ; sv = -ly * ph/fh + ph/2
    let mut su = lx.map(|value| value * pw / fw);
    su[2] += pw / 2.0;
    let mut sv = ly.map(|value| -value * ph / fh);
    sv[2] += ph / 2.0;
    if transform.flip_h {
        su = [-su[0], -su[1], pw - su[2]];
    }
    if transform.flip_v {
        sv = [-sv[0], -sv[1], ph - sv[2]];
    }
    Some(Affine([su[0], su[1], su[2], sv[0], sv[1], sv[2]]))
}

/// Canvas pixel centre to the clip's own frame at `time`: -1 to 1 across
/// and up, so a mask follows the clip as it moves, turns and flips. The
/// clip's media size gives its shape; without one it fills the canvas.
pub(crate) fn clip_frame(
    draft: &Draft,
    segment: &Segment,
    time: Micros,
    canvas: (u32, u32),
) -> Option<Affine> {
    let (clip, uniform) = cutlass_model::animated_at(draft, segment, time)?;
    let transform = picture_transform(&clip, &uniform, canvas.0, canvas.1)?;
    let picture = match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => u32::try_from(video.width)
            .ok()
            .zip(u32::try_from(video.height).ok())
            .filter(|&(width, height)| width > 0 && height > 0),
        _ => None,
    }
    .unwrap_or(canvas);
    let [a, b, c, d, e, f] = gpu_place(transform, picture, canvas)?.0;
    let (kx, ky) = (2.0 / f64::from(picture.0), 2.0 / f64::from(picture.1));
    Some(Affine([
        a * kx,
        b * kx,
        c * kx - 1.0,
        -d * ky,
        -e * ky,
        1.0 - f * ky,
    ]))
}

impl PictureTransform {
    /// Keep the picture's own size at scale 1: a title, whose raster is cut
    /// tight around its glyphs. A picture larger than the canvas still
    /// shrinks to fit.
    pub(crate) const fn native(mut self) -> Self {
        self.native = true;
        self
    }

    /// A native picture drawn with `density` pixels to each drawn pixel,
    /// so a title scaled up keeps sharp edges. Below 1 counts as 1.
    pub(crate) fn dense(mut self, density: f64) -> Self {
        self.density = if density.is_finite() {
            density.max(1.0)
        } else {
            1.0
        };
        self
    }

    /// The size the picture is drawn at scale 1. Media fits inside the
    /// canvas, keeping its shape.
    fn drawn_size(self, picture: (u32, u32), canvas: (u32, u32)) -> Option<(u32, u32)> {
        let (pw, ph) = picture;
        if self.native && pw > 0 && ph > 0 {
            let own = |edge: u32| round_edge(f64::from(edge) / self.density).unwrap_or(1);
            let (w, h) = (own(pw), own(ph));
            if w <= canvas.0 && h <= canvas.1 {
                return Some((w, h));
            }
        }
        contain_size(pw, ph, canvas.0, canvas.1)
    }

    /// The clip's opacity, 0 to 255.
    pub(crate) const fn opacity(self) -> u8 {
        self.opacity
    }

    /// Nothing to draw: the picture is fully transparent or scaled to nothing.
    pub(crate) fn omitted(self) -> bool {
        self.opacity == 0 || self.scale_x == 0.0 || self.scale_y == 0.0
    }
}

fn finish_picture(
    canvas: (u32, u32),
    picture: CpuBytes,
    transform: PictureTransform,
    display: DisplayQuarter,
    (crop, crop_scale): (Crop, f64),
    steady: Option<&cutlass_motion::Correction>,
    backdrop: &Backdrop,
) -> Sample {
    let (canvas_w, canvas_h) = canvas;
    let Some(picture) = orient(picture, display) else {
        return Sample::Miss;
    };
    let picture = match steady {
        Some(correction) => {
            let Some(picture) = steadied(&picture, correction, crop, crop_scale) else {
                return Sample::Miss;
            };
            picture
        }
        None => apply_crop(picture, crop, crop_scale),
    };
    let sharp = matches!(backdrop, Backdrop::Blur(_)).then(|| picture.clone());
    let Sample::Ready(Picture::Cpu(placed)) =
        place_with(canvas_w, canvas_h, picture, transform, DisplayQuarter::Zero)
    else {
        return Sample::Miss;
    };
    Sample::Ready(Picture::Cpu(over_backdrop(
        placed,
        sharp.as_ref(),
        backdrop,
    )))
}

/// The oriented `picture` steadied by `correction`, then cropped: one
/// resample through both maps, as the GPU pass draws it. A crop that is not
/// a parallelogram, which the GPU pass does not draw, is cut after.
fn steadied(
    picture: &CpuBytes,
    correction: &cutlass_motion::Correction,
    crop: Crop,
    crop_scale: f64,
) -> Option<CpuBytes> {
    let size = (picture.width(), picture.height());
    let steady = crate::steady::steady_map(correction, size)?;
    match gpu_crop(crop, crop_scale, size) {
        Some((to_oriented, cropped)) => {
            crate::steady::resample(picture, crate::steady::then(to_oriented, steady), cropped)
        }
        None => Some(apply_crop(
            crate::steady::resample(picture, steady, size)?,
            crop,
            crop_scale,
        )),
    }
}

/// Sample `crop` from the oriented picture.
///
/// Coordinates are fractions of the picture. Y grows downward. An all-zero
/// quad, a non-finite corner, or an edge that does not cover a pixel returns
/// `picture`. `crop_scale` above 1 moves each corner toward the quad center.
/// A scale that is not finite or not positive stays 1. A sample that falls
/// outside the picture is clear.
pub(crate) fn apply_crop(picture: CpuBytes, crop: Crop, crop_scale: f64) -> CpuBytes {
    if crop_absent(crop) {
        return picture;
    }
    let quad = scale_quad(crop, crop_scale);
    if !quad_finite(quad) {
        return picture;
    }
    let src_w = picture.width();
    let src_h = picture.height();
    let Some(out_w) = round_edge(edge_len(
        quad.upper_left_x,
        quad.upper_left_y,
        quad.upper_right_x,
        quad.upper_right_y,
        src_w,
        src_h,
    )) else {
        return picture;
    };
    let Some(out_h) = round_edge(edge_len(
        quad.upper_left_x,
        quad.upper_left_y,
        quad.lower_left_x,
        quad.lower_left_y,
        src_w,
        src_h,
    )) else {
        return picture;
    };
    let Some(len) = rgba_len(out_w, out_h) else {
        return picture;
    };
    let mut dst = vec![0_u8; len];
    let built = {
        let src = picture.rgba();
        let mut wrote = true;
        for y in 0..out_h {
            for x in 0..out_w {
                let u = (f64::from(x) + 0.5) / f64::from(out_w);
                let v = (f64::from(y) + 0.5) / f64::from(out_h);
                let nx = bilinear(
                    u,
                    v,
                    quad.upper_left_x,
                    quad.upper_right_x,
                    quad.lower_left_x,
                    quad.lower_right_x,
                );
                let ny = bilinear(
                    u,
                    v,
                    quad.upper_left_y,
                    quad.upper_right_y,
                    quad.lower_left_y,
                    quad.lower_right_y,
                );
                let pixel = sample_norm(src, src_w, src_h, nx, ny);
                let Some(slot) = dst_slot(&mut dst, out_w, x, y) else {
                    wrote = false;
                    break;
                };
                slot.copy_from_slice(&pixel);
            }
            if !wrote {
                break;
            }
        }
        wrote
    };
    if !built {
        return picture;
    }
    CpuBytes::new(out_w, out_h, dst).unwrap_or(picture)
}

fn crop_absent(crop: Crop) -> bool {
    is_zero(crop.upper_left_x)
        && is_zero(crop.upper_left_y)
        && is_zero(crop.upper_right_x)
        && is_zero(crop.upper_right_y)
        && is_zero(crop.lower_left_x)
        && is_zero(crop.lower_left_y)
        && is_zero(crop.lower_right_x)
        && is_zero(crop.lower_right_y)
}

fn is_zero(value: f64) -> bool {
    value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == (-0.0_f64).to_bits()
}

fn scale_quad(crop: Crop, scale: f64) -> Crop {
    if !scale.is_finite() || scale <= 0.0 || scale.to_bits() == 1.0_f64.to_bits() {
        return crop;
    }
    let center_x =
        (crop.upper_left_x + crop.upper_right_x + crop.lower_left_x + crop.lower_right_x) / 4.0;
    let center_y =
        (crop.upper_left_y + crop.upper_right_y + crop.lower_left_y + crop.lower_right_y) / 4.0;
    Crop {
        upper_left_x: toward(crop.upper_left_x, center_x, scale),
        upper_left_y: toward(crop.upper_left_y, center_y, scale),
        upper_right_x: toward(crop.upper_right_x, center_x, scale),
        upper_right_y: toward(crop.upper_right_y, center_y, scale),
        lower_left_x: toward(crop.lower_left_x, center_x, scale),
        lower_left_y: toward(crop.lower_left_y, center_y, scale),
        lower_right_x: toward(crop.lower_right_x, center_x, scale),
        lower_right_y: toward(crop.lower_right_y, center_y, scale),
    }
}

fn toward(value: f64, center: f64, scale: f64) -> f64 {
    center + (value - center) / scale
}

fn quad_finite(crop: Crop) -> bool {
    [
        crop.upper_left_x,
        crop.upper_left_y,
        crop.upper_right_x,
        crop.upper_right_y,
        crop.lower_left_x,
        crop.lower_left_y,
        crop.lower_right_x,
        crop.lower_right_y,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn edge_len(x0: f64, y0: f64, x1: f64, y1: f64, width: u32, height: u32) -> f64 {
    let dx = (x1 - x0) * f64::from(width);
    let dy = (y1 - y0) * f64::from(height);
    dx.mul_add(dx, dy * dy).sqrt()
}

fn round_edge(value: f64) -> Option<u32> {
    if !value.is_finite() || value < 1.0 {
        return None;
    }
    let rounded = value.round();
    if !(1.0..4_294_967_296.0).contains(&rounded) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(rounded as u32)
}

fn bilinear(
    u: f64,
    v: f64,
    upper_left: f64,
    upper_right: f64,
    lower_left: f64,
    lower_right: f64,
) -> f64 {
    let upper = upper_left + (upper_right - upper_left) * u;
    let lower = lower_left + (lower_right - lower_left) * u;
    upper + (lower - upper) * v
}

fn sample_norm(src: &[u8], width: u32, height: u32, nx: f64, ny: f64) -> [u8; 4] {
    let Some(x) = floor_index(nx * f64::from(width), width) else {
        return [0, 0, 0, 0];
    };
    let Some(y) = floor_index(ny * f64::from(height), height) else {
        return [0, 0, 0, 0];
    };
    src_pixel(src, width, x, y).unwrap_or([0, 0, 0, 0])
}

fn dst_slot(dst: &mut [u8], width: u32, x: u32, y: u32) -> Option<&mut [u8]> {
    let stride = usize::try_from(width).ok()?.checked_mul(4)?;
    let index = usize::try_from(y)
        .ok()?
        .checked_mul(stride)?
        .checked_add(usize::try_from(x).ok()?.checked_mul(4)?)?;
    dst.get_mut(index..index + 4)
}

pub(crate) fn place_with(
    canvas_w: u32,
    canvas_h: u32,
    picture: CpuBytes,
    transform: PictureTransform,
    display: DisplayQuarter,
) -> Sample {
    if transform.omitted() {
        return Sample::Miss;
    }
    let Some(picture) = orient(picture, display) else {
        return Sample::Miss;
    };
    if transform.geometry_identity {
        let sample = place(canvas_w, canvas_h, picture, transform);
        return if transform.opacity == 255 {
            sample
        } else {
            fade(sample, transform.opacity)
        };
    }
    sample_transformed(canvas_w, canvas_h, &picture, transform)
}

pub(crate) fn orient(picture: CpuBytes, display: DisplayQuarter) -> Option<CpuBytes> {
    if display == DisplayQuarter::Zero {
        return Some(picture);
    }
    let src_w = picture.width();
    let src_h = picture.height();
    let (dst_w, dst_h) = match display {
        DisplayQuarter::Zero | DisplayQuarter::Half => (src_w, src_h),
        DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (src_h, src_w),
    };
    let len = rgba_len(dst_w, dst_h)?;
    let src = picture.rgba();
    let mut dst = vec![0_u8; len];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let (sx, sy) = turned_source(display, src_w, src_h, x, y)?;
            let from = pixel_at(src_w, sx, sy)?;
            let to = pixel_at(dst_w, x, y)?;
            dst.get_mut(to..to + 4)?
                .copy_from_slice(src.get(from..from + 4)?);
        }
    }
    CpuBytes::new(dst_w, dst_h, dst).ok()
}

fn turned_source(
    display: DisplayQuarter,
    src_w: u32,
    src_h: u32,
    x: u32,
    y: u32,
) -> Option<(u32, u32)> {
    match display {
        DisplayQuarter::Zero => Some((x, y)),
        DisplayQuarter::Clockwise90 => Some((y, src_h.checked_sub(x)?.checked_sub(1)?)),
        DisplayQuarter::Half => Some((
            src_w.checked_sub(x)?.checked_sub(1)?,
            src_h.checked_sub(y)?.checked_sub(1)?,
        )),
        DisplayQuarter::Clockwise270 => Some((src_w.checked_sub(y)?.checked_sub(1)?, x)),
    }
}

fn pixel_at(width: u32, x: u32, y: u32) -> Option<usize> {
    let stride = usize::try_from(width).ok()?.checked_mul(4)?;
    usize::try_from(y)
        .ok()?
        .checked_mul(stride)?
        .checked_add(usize::try_from(x).ok()?.checked_mul(4)?)
}

fn place(canvas_w: u32, canvas_h: u32, picture: CpuBytes, transform: PictureTransform) -> Sample {
    let src_w = picture.width();
    let src_h = picture.height();
    if src_w == canvas_w && src_h == canvas_h {
        return Sample::Ready(Picture::Cpu(picture));
    }
    let Some((draw_w, draw_h)) = transform.drawn_size((src_w, src_h), (canvas_w, canvas_h)) else {
        return Sample::Miss;
    };
    let Some(pixels) = crate::sample::scale(src_w, src_h, picture.rgba(), draw_w, draw_h) else {
        return Sample::Miss;
    };
    let Some(canvas) = stamp(canvas_w, canvas_h, draw_w, draw_h, &pixels) else {
        return Sample::Miss;
    };
    match CpuBytes::new(canvas_w, canvas_h, canvas) {
        Ok(bytes) => Sample::Ready(Picture::Cpu(bytes)),
        Err(_) => Sample::Miss,
    }
}

fn fade(sample: Sample, opacity: u8) -> Sample {
    let Sample::Ready(Picture::Cpu(bytes)) = sample else {
        return sample;
    };
    if opacity == 255 {
        return Sample::Ready(Picture::Cpu(bytes));
    }
    let mut rgba = bytes.rgba().to_vec();
    for pixel in rgba.chunks_mut(4) {
        for channel in pixel {
            *channel = mix_channel(*channel, opacity);
        }
    }
    match CpuBytes::new(bytes.width(), bytes.height(), rgba) {
        Ok(bytes) => Sample::Ready(Picture::Cpu(bytes)),
        Err(_) => Sample::Miss,
    }
}

fn mix_channel(channel: u8, opacity: u8) -> u8 {
    let mixed = u16::from(channel) * u16::from(opacity) + 127;
    u8::try_from(mixed / 255).unwrap_or(0)
}

fn sample_transformed(
    canvas_w: u32,
    canvas_h: u32,
    picture: &CpuBytes,
    transform: PictureTransform,
) -> Sample {
    let src_w = picture.width();
    let src_h = picture.height();
    let Some((fitted_w, fitted_h)) = transform.drawn_size((src_w, src_h), (canvas_w, canvas_h))
    else {
        return Sample::Miss;
    };
    let draw = Draw {
        src_w,
        src_h,
        fitted_w: f64::from(fitted_w),
        fitted_h: f64::from(fitted_h),
        center_x: f64::from(canvas_w) / 2.0 + transform.dx,
        center_y: f64::from(canvas_h) / 2.0 - transform.dy,
        transform,
    };
    if draw.drawn_w() <= 0.0 || draw.drawn_h() <= 0.0 {
        return Sample::Miss;
    }
    let Some(canvas_len) = rgba_len(canvas_w, canvas_h) else {
        return Sample::Miss;
    };
    let mut canvas = vec![0_u8; canvas_len];
    let Some((x0, x1, y0, y1)) = cover_box(&draw, canvas_w, canvas_h) else {
        return Sample::Miss;
    };
    let src = picture.rgba();
    let mut wrote = false;
    for y in y0..y1 {
        for x in x0..x1 {
            let Some((u, v)) = source_point(x, y, &draw) else {
                continue;
            };
            let Some(pixel) = crate::sample::bilinear(src, src_w, src_h, u, v) else {
                continue;
            };
            let Some(dest) = canvas_slot(&mut canvas, canvas_w, x, y) else {
                continue;
            };
            dest[0] = mix_channel(pixel[0], draw.transform.opacity);
            dest[1] = mix_channel(pixel[1], draw.transform.opacity);
            dest[2] = mix_channel(pixel[2], draw.transform.opacity);
            dest[3] = mix_channel(pixel[3], draw.transform.opacity);
            wrote = true;
        }
    }
    if !wrote {
        return Sample::Miss;
    }
    match CpuBytes::new(canvas_w, canvas_h, canvas) {
        Ok(bytes) => Sample::Ready(Picture::Cpu(bytes)),
        Err(_) => Sample::Miss,
    }
}

struct Draw {
    src_w: u32,
    src_h: u32,
    fitted_w: f64,
    fitted_h: f64,
    center_x: f64,
    center_y: f64,
    transform: PictureTransform,
}

impl Draw {
    fn drawn_w(&self) -> f64 {
        self.fitted_w * self.transform.scale_x
    }

    fn drawn_h(&self) -> f64 {
        self.fitted_h * self.transform.scale_y
    }
}

fn cover_box(draw: &Draw, canvas_w: u32, canvas_h: u32) -> Option<(u32, u32, u32, u32)> {
    let half_w = draw.drawn_w() / 2.0;
    let half_h = draw.drawn_h() / 2.0;
    let corners = [
        (half_w, half_h),
        (half_w, -half_h),
        (-half_w, half_h),
        (-half_w, -half_h),
    ];
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let rotated_x = x * draw.transform.cos - y * draw.transform.sin;
        let rotated_y = x * draw.transform.sin + y * draw.transform.cos;
        let canvas_x = draw.center_x + rotated_x;
        let canvas_y = draw.center_y - rotated_y;
        min_x = min_x.min(canvas_x);
        max_x = max_x.max(canvas_x);
        min_y = min_y.min(canvas_y);
        max_y = max_y.max(canvas_y);
    }
    let x0 = axis_bound(min_x.floor(), canvas_w)?;
    let y0 = axis_bound(min_y.floor(), canvas_h)?;
    let x1 = axis_bound(max_x.ceil(), canvas_w)?;
    let y1 = axis_bound(max_y.ceil(), canvas_h)?;
    if x0 >= x1 || y0 >= y1 {
        None
    } else {
        Some((x0, x1, y0, y1))
    }
}

fn axis_bound(value: f64, limit: u32) -> Option<u32> {
    if !value.is_finite() {
        return None;
    }
    if value <= 0.0 {
        return Some(0);
    }
    if value >= f64::from(limit) {
        return Some(limit);
    }
    u32::try_from(value as u64).ok()
}

/// The continuous point of the picture under canvas pixel `(x, y)`, or none
/// outside the drawn box.
fn source_point(x: u32, y: u32, draw: &Draw) -> Option<(f64, f64)> {
    let rel_x = f64::from(x) + 0.5 - draw.center_x;
    let rel_y = f64::from(y) + 0.5 - draw.center_y;
    let up_y = -rel_y;
    let local_x = (rel_x * draw.transform.cos + up_y * draw.transform.sin) / draw.transform.scale_x;
    let local_y =
        (-rel_x * draw.transform.sin + up_y * draw.transform.cos) / draw.transform.scale_y;
    let half_w = draw.fitted_w / 2.0;
    let half_h = draw.fitted_h / 2.0;
    // The covered span is half-open on the right and on the bottom, matching
    // the integer center used when the clip is identity.
    if local_x < -half_w || local_x >= half_w || local_y <= -half_h || local_y > half_h {
        return None;
    }
    let (src_w, src_h) = (f64::from(draw.src_w), f64::from(draw.src_h));
    let mut u = (local_x + half_w) / draw.fitted_w * src_w;
    let mut v = (half_h - local_y) / draw.fitted_h * src_h;
    if draw.transform.flip_h {
        u = src_w - u;
    }
    if draw.transform.flip_v {
        v = src_h - v;
    }
    Some((u, v))
}

fn floor_index(value: f64, limit: u32) -> Option<u32> {
    if !value.is_finite() || value < 0.0 || value >= f64::from(limit) {
        return None;
    }
    u32::try_from(value as u64).ok()
}

fn src_pixel(src: &[u8], src_w: u32, x: u32, y: u32) -> Option<[u8; 4]> {
    let stride = usize::try_from(src_w).ok()?.checked_mul(4)?;
    let index = usize::try_from(y)
        .ok()?
        .checked_mul(stride)?
        .checked_add(usize::try_from(x).ok()?.checked_mul(4)?)?;
    let pixel = src.get(index..index + 4)?;
    Some([pixel[0], pixel[1], pixel[2], pixel[3]])
}

fn canvas_slot(canvas: &mut [u8], canvas_w: u32, x: u32, y: u32) -> Option<&mut [u8]> {
    let stride = usize::try_from(canvas_w).ok()?.checked_mul(4)?;
    let index = usize::try_from(y)
        .ok()?
        .checked_mul(stride)?
        .checked_add(usize::try_from(x).ok()?.checked_mul(4)?)?;
    canvas.get_mut(index..index + 4)
}

fn contain_size(src_w: u32, src_h: u32, canvas_w: u32, canvas_h: u32) -> Option<(u32, u32)> {
    if src_w == 0 || src_h == 0 || canvas_w == 0 || canvas_h == 0 {
        return None;
    }
    let by_width = u64::from(canvas_w).checked_mul(u64::from(src_h))?;
    let by_height = u64::from(canvas_h).checked_mul(u64::from(src_w))?;
    if by_width <= by_height {
        let draw_h = u64::from(src_h)
            .checked_mul(u64::from(canvas_w))?
            .checked_div(u64::from(src_w))?;
        let draw_h = u32::try_from(draw_h).ok()?.clamp(1, canvas_h);
        Some((canvas_w, draw_h))
    } else {
        let draw_w = u64::from(src_w)
            .checked_mul(u64::from(canvas_h))?
            .checked_div(u64::from(src_h))?;
        let draw_w = u32::try_from(draw_w).ok()?.clamp(1, canvas_w);
        Some((draw_w, canvas_h))
    }
}

fn stamp(canvas_w: u32, canvas_h: u32, draw_w: u32, draw_h: u32, pixels: &[u8]) -> Option<Vec<u8>> {
    if draw_w > canvas_w || draw_h > canvas_h || pixels.len() != rgba_len(draw_w, draw_h)? {
        return None;
    }
    let mut canvas = vec![0_u8; rgba_len(canvas_w, canvas_h)?];
    let x0 = canvas_w - draw_w;
    let y0 = canvas_h - draw_h;
    let x0 = x0 / 2;
    let y0 = y0 / 2;
    let canvas_stride = usize::try_from(canvas_w).ok()?.checked_mul(4)?;
    let draw_stride = usize::try_from(draw_w).ok()?.checked_mul(4)?;
    for row in 0..draw_h {
        let y = usize::try_from(y0.checked_add(row)?).ok()?;
        let dest = y
            .checked_mul(canvas_stride)?
            .checked_add(usize::try_from(x0).ok()?.checked_mul(4)?)?;
        let src_at = usize::try_from(row).ok()?.checked_mul(draw_stride)?;
        canvas
            .get_mut(dest..dest + draw_stride)?
            .copy_from_slice(pixels.get(src_at..src_at + draw_stride)?);
    }
    Some(canvas)
}

#[cfg(test)]
mod tests {
    use cutlass_codec::access::{ColorMatrix, ColorPrimaries, ColorRange, PixelFormat, Transfer};
    use cutlass_engine::DecodedFrame;

    fn coded(pixel: PixelFormat, matrix: ColorMatrix, range: ColorRange) -> DecodedFrame {
        DecodedFrame::Coded {
            width: 2,
            height: 2,
            pixel,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Bt709,
            matrix,
            range,
            bytes: vec![0; 6].into(),
        }
    }

    #[test]
    fn the_gpu_takes_only_the_pictures_its_matrix_reproduces() {
        // 8-bit 4:2:0, BT.709 or unnamed, studio swing or unnamed.
        for (matrix, range) in [
            (ColorMatrix::Unspecified, ColorRange::Unspecified),
            (ColorMatrix::Bt709, ColorRange::Limited),
            (ColorMatrix::Bt709, ColorRange::Studio),
        ] {
            assert!(gpu_samples(coded(PixelFormat::Yuv420p, matrix, range)).is_some());
        }
        // Full range, another matrix, another depth or chroma layout: the
        // CPU conversion draws those.
        for (pixel, matrix, range) in [
            (PixelFormat::Yuv420p, ColorMatrix::Bt709, ColorRange::Full),
            (
                PixelFormat::Yuv420p,
                ColorMatrix::Bt601,
                ColorRange::Limited,
            ),
            (
                PixelFormat::Yuv420p,
                ColorMatrix::Bt2020,
                ColorRange::Limited,
            ),
            (
                PixelFormat::Yuv420p10,
                ColorMatrix::Bt709,
                ColorRange::Limited,
            ),
            (
                PixelFormat::Yuv444p,
                ColorMatrix::Bt709,
                ColorRange::Limited,
            ),
            (PixelFormat::Gbrp, ColorMatrix::Identity, ColorRange::Full),
        ] {
            assert!(
                gpu_samples(coded(pixel, matrix, range)).is_none(),
                "{pixel:?}"
            );
        }
    }

    use cutlass_model::{
        CommonKeyframe, CommonKeyframeGroup, Crop, Flip, Point, Segment, TimeRange,
    };

    use super::*;

    fn solid(width: u32, height: u32, pixel: [u8; 4]) -> CpuBytes {
        let mut rgba = Vec::new();
        for _ in 0..width * height {
            rgba.extend(pixel);
        }
        CpuBytes::new(width, height, rgba).unwrap()
    }

    #[test]
    fn a_matching_canvas_keeps_the_decoded_bytes() {
        let picture = solid(2, 1, [9, 8, 7, 6]);
        let Sample::Ready(Picture::Cpu(bytes)) = place(2, 1, picture, identity(2, 1)) else {
            panic!("ready");
        };
        assert_eq!(bytes.rgba(), &[9, 8, 7, 6, 9, 8, 7, 6]);
    }

    #[test]
    fn a_smaller_picture_is_scaled_up_to_fit() {
        let picture = solid(1, 1, [1, 2, 3, 4]);
        let Sample::Ready(Picture::Cpu(bytes)) = place(3, 3, picture, identity(3, 3)) else {
            panic!("ready");
        };
        assert!(bytes.rgba().chunks(4).all(|pixel| pixel == [1, 2, 3, 4]));

        let wide = solid(2, 1, [5, 6, 7, 8]);
        let Sample::Ready(Picture::Cpu(bytes)) = place(4, 4, wide, identity(4, 4)) else {
            panic!("ready");
        };
        let rgba = bytes.rgba();
        // 2x1 fits 4x2, centered on rows 1 and 2.
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
        assert!(rgba[16..48].chunks(4).all(|pixel| pixel == [5, 6, 7, 8]));
        assert_eq!(&rgba[48..52], &[0, 0, 0, 0]);
    }

    #[test]
    fn a_larger_picture_is_scaled_down_to_fit() {
        let mut rgba = Vec::new();
        for pixel in [1_u8, 2, 3, 4, 5, 6, 7, 8] {
            rgba.extend([pixel, 0, 0, 255]);
        }
        let picture = CpuBytes::new(4, 2, rgba).unwrap();
        let Sample::Ready(Picture::Cpu(bytes)) = place(2, 2, picture, identity(2, 2)) else {
            panic!("ready");
        };
        let rgba = bytes.rgba();
        // Each output pixel averages a 2x2 block: 3.5 and 5.5, rounded.
        assert_eq!(&rgba[0..8], &[4, 0, 0, 255, 6, 0, 0, 255]);
        assert_eq!(&rgba[8..16], &[0, 0, 0, 0, 0, 0, 0, 0]);
    }

    fn transform_of(
        clip: Clip,
        uniform: UniformScale,
        width: u32,
        height: u32,
    ) -> PictureTransform {
        picture_transform(&clip, &uniform, width, height).expect("transform")
    }

    fn identity(width: u32, height: u32) -> PictureTransform {
        transform_of(Clip::default(), UniformScale::default(), width, height)
    }

    fn paint(
        width: u32,
        height: u32,
        picture: CpuBytes,
        clip: Clip,
        uniform: UniformScale,
    ) -> Vec<u8> {
        let transform = transform_of(clip, uniform, width, height);
        let Sample::Ready(Picture::Cpu(bytes)) =
            place_with(width, height, picture, transform, DisplayQuarter::Zero)
        else {
            panic!("ready");
        };
        bytes.into_rgba()
    }

    #[test]
    fn a_default_clip_matches_the_untransformed_placement() {
        let picture = solid(1, 1, [1, 2, 3, 4]);
        let clip = Clip::default();
        let uniform = UniformScale::default();
        let transform = transform_of(clip, uniform, 3, 3);
        assert!(transform.geometry_identity);
        let Sample::Ready(Picture::Cpu(bytes)) =
            place_with(3, 3, picture, transform, DisplayQuarter::Zero)
        else {
            panic!("ready");
        };
        assert_eq!(&bytes.rgba()[16..20], &[1, 2, 3, 4]);
    }

    #[test]
    fn opacity_scales_premultiplied_channels_and_zero_draws_nothing() {
        let clip = Clip {
            alpha: 0.5,
            ..Clip::default()
        };
        let rgba = paint(
            1,
            1,
            solid(1, 1, [200, 0, 0, 255]),
            clip,
            UniformScale::default(),
        );
        assert_eq!(rgba, vec![100, 0, 0, 128]);

        let clear = Clip {
            alpha: 0.0,
            ..Clip::default()
        };
        let transform = transform_of(clear, UniformScale::default(), 1, 1);
        assert!(matches!(
            place_with(
                1,
                1,
                solid(1, 1, [1, 2, 3, 4]),
                transform,
                DisplayQuarter::Zero,
            ),
            Sample::Miss
        ));
    }

    #[test]
    fn flip_scale_and_translation_move_the_source_pixels() {
        let flip = Clip {
            flip: Flip {
                horizontal: true,
                ..Flip::default()
            },
            ..Clip::default()
        };
        let row = CpuBytes::new(2, 1, vec![1, 0, 0, 255, 2, 0, 0, 255]).unwrap();
        assert_eq!(
            &paint(2, 1, row, flip, UniformScale::default())[..8],
            &[2, 0, 0, 255, 1, 0, 0, 255]
        );

        let wide = Clip {
            scale: Point::new(3.0, 1.0),
            ..Clip::default()
        };
        let separate = UniformScale {
            on: false,
            ..UniformScale::default()
        };
        let scaled = paint(3, 1, solid(1, 1, [4, 0, 0, 255]), wide, separate);
        assert_eq!(scaled, vec![4, 0, 0, 255, 4, 0, 0, 255, 4, 0, 0, 255]);

        let shift = Clip {
            transform: Point::new(1.0, 0.0),
            ..Clip::default()
        };
        let moved = paint(
            4,
            1,
            solid(1, 1, [9, 0, 0, 255]),
            shift,
            UniformScale::default(),
        );
        assert_eq!(&moved[0..12], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&moved[12..16], &[9, 0, 0, 255]);

        let up = Clip {
            transform: Point::new(0.0, 0.5),
            ..Clip::default()
        };
        let raised = paint(
            1,
            4,
            solid(1, 1, [8, 0, 0, 255]),
            up,
            UniformScale::default(),
        );
        assert_eq!(&raised[0..4], &[8, 0, 0, 255]);
        assert!(raised[4..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn a_quarter_turn_puts_the_right_pixel_on_top() {
        let clip = Clip {
            rotation: 90.0,
            ..Clip::default()
        };
        let picture = CpuBytes::new(3, 1, vec![1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255]).unwrap();
        let rgba = paint(3, 3, picture, clip, UniformScale::default());
        let column = [rgba[4], rgba[16], rgba[28]];
        assert_eq!(column, [3, 2, 1]);
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[8], 0);
    }

    #[test]
    fn uniform_scale_drives_both_axes_and_a_negative_scale_flips() {
        let clip = Clip {
            scale: Point::new(9.0, 1.0),
            ..Clip::default()
        };
        let uniform = UniformScale::default();
        let doubled = UniformScale {
            value: 3.0,
            ..UniformScale::default()
        };
        let wide = paint(3, 1, solid(1, 1, [4, 0, 0, 255]), clip.clone(), doubled);
        assert_eq!(wide, vec![4, 0, 0, 255, 4, 0, 0, 255, 4, 0, 0, 255]);
        let ignored = transform_of(clip, uniform, 3, 1);
        assert_eq!(ignored.scale_x, 1.0);

        let negative = Clip {
            scale: Point::new(-1.0, 1.0),
            ..Clip::default()
        };
        let separate = UniformScale {
            on: false,
            ..UniformScale::default()
        };
        let row = CpuBytes::new(2, 1, vec![1, 0, 0, 255, 2, 0, 0, 255]).unwrap();
        assert_eq!(
            &paint(2, 1, row, negative, separate)[..8],
            &[2, 0, 0, 255, 1, 0, 0, 255]
        );
    }

    #[test]
    fn an_angle_that_is_not_a_quarter_turn_keeps_the_center_pixel() {
        let clip = Clip {
            rotation: 45.0,
            ..Clip::default()
        };
        // A third of the fitted picture keeps it one pixel wide.
        let third = UniformScale {
            value: 1.0 / 3.0,
            ..UniformScale::default()
        };
        let rgba = paint(3, 3, solid(1, 1, [6, 0, 0, 255]), clip, third);
        assert_eq!(&rgba[16..20], &[6, 0, 0, 255]);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn a_file_quarter_turn_precedes_the_clip() {
        let row = CpuBytes::new(2, 1, vec![1, 0, 0, 255, 2, 0, 0, 255]).unwrap();
        let upright = placed(
            1,
            2,
            row.clone(),
            Clip::default(),
            DisplayQuarter::Clockwise90,
        );
        assert_eq!(upright, vec![1, 0, 0, 255, 2, 0, 0, 255]);

        let half = placed(2, 1, row.clone(), Clip::default(), DisplayQuarter::Half);
        assert_eq!(&half[..8], &[2, 0, 0, 255, 1, 0, 0, 255]);

        let left = placed(
            1,
            2,
            row.clone(),
            Clip::default(),
            DisplayQuarter::Clockwise270,
        );
        assert_eq!(left, vec![2, 0, 0, 255, 1, 0, 0, 255]);

        let flip = Clip {
            flip: Flip {
                horizontal: true,
                ..Flip::default()
            },
            ..Clip::default()
        };
        let after = placed(1, 2, row, flip, DisplayQuarter::Clockwise90);
        assert_eq!(after, vec![1, 0, 0, 255, 2, 0, 0, 255]);
    }

    fn placed(
        width: u32,
        height: u32,
        picture: CpuBytes,
        clip: Clip,
        display: DisplayQuarter,
    ) -> Vec<u8> {
        let transform = transform_of(clip, UniformScale::default(), width, height);
        let Sample::Ready(Picture::Cpu(bytes)) =
            place_with(width, height, picture, transform, display)
        else {
            panic!("ready");
        };
        bytes.into_rgba()
    }

    fn line(property: &str, samples: &[(i64, f64)]) -> CommonKeyframeGroup {
        CommonKeyframeGroup {
            property_type: property.to_owned(),
            keyframe_list: samples
                .iter()
                .map(|(time, value)| CommonKeyframe {
                    time_offset: Micros(*time),
                    values: vec![*value],
                    curve_type: "Line".to_owned(),
                    ..CommonKeyframe::default()
                })
                .collect(),
            ..CommonKeyframeGroup::default()
        }
    }

    fn keyed(start: i64, groups: Vec<CommonKeyframeGroup>) -> Segment {
        let mut segment = Segment::new("segment", "video", TimeRange::new(start, 10_000));
        segment.common_keyframes = groups;
        segment
    }

    #[test]
    fn line_keyframes_move_the_clip_from_the_segment_start() {
        let segment = keyed(1_000, vec![line("KFTypeAlpha", &[(0, 0.0), (1_000, 1.0)])]);
        let (start, _) = segment_clip(&[], &segment, Micros(1_000)).expect("clip");
        assert_eq!(start.alpha, 0.0);
        let (mid, uniform) = segment_clip(&[], &segment, Micros(1_500)).expect("clip");
        assert!((mid.alpha - 0.5).abs() < 1e-9);
        let rgba = paint(1, 1, solid(1, 1, [200, 0, 0, 255]), mid, uniform);
        assert_eq!(rgba, vec![100, 0, 0, 128]);
        let (end, _) = segment_clip(&[], &segment, Micros(3_000)).expect("clip");
        assert_eq!(end.alpha, 1.0);

        let mut curved = line("KFTypeAlpha", &[(0, 0.0), (1_000, 1.0)]);
        curved.keyframe_list[1].curve_type = "CurveIn".to_owned();
        let segment = keyed(0, vec![curved]);
        let (mid, _) = segment_clip(&[], &segment, Micros(500)).expect("clip");
        assert!((mid.alpha - 0.5).abs() < 1e-9);

        let segment = keyed(0, vec![line("alpha", &[(0, 0.25)])]);
        let (held, _) = segment_clip(&[], &segment, Micros(40)).expect("clip");
        assert_eq!(held.alpha, 0.25);

        let later = keyed(
            0,
            vec![
                line("KFTypeAlpha", &[(0, 0.25)]),
                line("KFTypeAlpha", &[(0, 0.75)]),
            ],
        );
        let (clip, _) = segment_clip(&[], &later, Micros(0)).expect("clip");
        assert_eq!(clip.alpha, 0.75);

        let mut blank = line("KFTypeAlpha", &[(0, 0.25)]);
        blank.keyframe_list[0].values.clear();
        let segment = keyed(0, vec![blank]);
        let (clip, _) = segment_clip(&[], &segment, Micros(0)).expect("clip");
        assert_eq!(clip.alpha, 1.0);

        let segment = keyed(0, vec![line("KFTypePositionX", &[(0, 1.0)])]);
        let (moved, uniform) = segment_clip(&[], &segment, Micros(0)).expect("clip");
        let rgba = paint(4, 1, solid(1, 1, [9, 0, 0, 255]), moved, uniform);
        assert_eq!(&rgba[12..16], &[9, 0, 0, 255]);

        let segment = keyed(0, vec![line("KFTypeRotation", &[(0, 90.0)])]);
        let (turned, uniform) = segment_clip(&[], &segment, Micros(0)).expect("clip");
        let picture = CpuBytes::new(3, 1, vec![1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255]).unwrap();
        let rgba = paint(3, 3, picture, turned, uniform);
        assert_eq!([rgba[4], rgba[16], rgba[28]], [3, 2, 1]);
    }

    #[test]
    fn curve_keyframes_follow_the_handles() {
        let mut raised = line("KFTypeAlpha", &[(0, 0.0), (1_000, 1.0)]);
        raised.keyframe_list[0].curve_type = "CurveOut".to_owned();
        raised.keyframe_list[0].right_control.y = 1.0;
        let segment = keyed(0, vec![raised.clone()]);
        let (mid, _) = segment_clip(&[], &segment, Micros(500)).expect("clip");
        assert!((mid.alpha - 0.875).abs() < 1e-12);
        raised.keyframe_list[0].right_control.y = 2.0;
        let segment = keyed(0, vec![raised.clone()]);
        let (mid, _) = segment_clip(&[], &segment, Micros(500)).expect("clip");
        assert!((mid.alpha - 1.25).abs() < 1e-12);
        raised.keyframe_list[0].right_control.y = 1.0;
        let segment = keyed(0, vec![raised.clone()]);
        let (held, _) = segment_clip(&[], &segment, Micros(0)).expect("clip");
        assert_eq!(held.alpha, 0.0);
        let (end, _) = segment_clip(&[], &segment, Micros(5_000)).expect("clip");
        assert_eq!(end.alpha, 1.0);

        raised.keyframe_list[0].curve_type = "FreeCurveInOut".to_owned();
        raised.keyframe_list[0].graph_id = "missing".to_owned();
        let segment = keyed(0, vec![raised]);
        let (mid, _) = segment_clip(&[], &segment, Micros(500)).expect("clip");
        assert!((mid.alpha - 0.875).abs() < 1e-12);

        let mut straight = line("KFTypeAlpha", &[(0, 0.0), (1_000, 1.0)]);
        straight.keyframe_list[0].right_control.y = 1.0;
        straight.keyframe_list[0].curve_type = "Ease".to_owned();
        let segment = keyed(0, vec![straight.clone()]);
        let (mid, _) = segment_clip(&[], &segment, Micros(500)).expect("clip");
        assert!((mid.alpha - 0.5).abs() < 1e-12);
        straight.keyframe_list[0].curve_type = "none".to_owned();
        straight.keyframe_list[0].right_control.y = f64::NAN;
        let segment = keyed(0, vec![straight]);
        let (mid, _) = segment_clip(&[], &segment, Micros(500)).expect("clip");
        assert!((mid.alpha - 0.5).abs() < 1e-12);

        let mut eased = line("KFTypeAlpha", &[(0, 0.0), (1_000, 1.0)]);
        eased.keyframe_list[0].curve_type = "CurveInOut".to_owned();
        eased.keyframe_list[0].right_control.x = 400.0;
        eased.keyframe_list[1].curve_type = "CurveInOut".to_owned();
        eased.keyframe_list[1].left_control.x = -400.0;
        let segment = keyed(0, vec![eased]);
        let (quarter, _) = segment_clip(&[], &segment, Micros(250)).expect("clip");
        assert!((quarter.alpha - 0.13518434228493398).abs() < 1e-12);
        let (three, _) = segment_clip(&[], &segment, Micros(750)).expect("clip");
        assert!((three.alpha - 0.864_815_657_715_066).abs() < 1e-12);

        let mut replaced = line("KFTypeAlpha", &[(0, 0.0), (0, 0.0), (1_000, 1.0)]);
        replaced.keyframe_list[0].curve_type = "CurveOut".to_owned();
        replaced.keyframe_list[0].right_control.y = 1.0;
        let segment = keyed(0, vec![replaced.clone()]);
        let (mid, _) = segment_clip(&[], &segment, Micros(500)).expect("clip");
        assert!((mid.alpha - 0.5).abs() < 1e-12);
        replaced.keyframe_list[1].curve_type = "CurveOut".to_owned();
        replaced.keyframe_list[1].right_control.y = 1.0;
        let segment = keyed(0, vec![replaced]);
        let (mid, _) = segment_clip(&[], &segment, Micros(500)).expect("clip");
        assert!((mid.alpha - 0.875).abs() < 1e-12);

        let mut broken = line("KFTypeAlpha", &[(0, 0.0), (1_000, 1.0)]);
        broken.keyframe_list[0].curve_type = "CurveIn".to_owned();
        broken.keyframe_list[0].right_control.x = f64::INFINITY;
        assert!(segment_clip(&[], &keyed(0, vec![broken]), Micros(500)).is_none());
    }

    #[test]
    fn scale_keyframes_follow_the_uniform_switch() {
        let segment = keyed(
            0,
            vec![
                line("KFTypeScale", &[(0, 2.0)]),
                line("KFTypeScaleX", &[(0, 9.0)]),
            ],
        );
        let (_, uniform) = segment_clip(&[], &segment, Micros(0)).expect("clip");
        assert_eq!(uniform.value, 2.0);
        let transform = transform_of(Clip::default(), uniform, 3, 1);
        assert_eq!(transform.scale_x, 2.0);

        let mut separate = keyed(
            0,
            vec![
                line("KFTypeScale", &[(0, 4.0)]),
                line("KFTypeScaleX", &[(0, 2.0)]),
            ],
        );
        separate.uniform_scale.on = false;
        let (clip, uniform) = segment_clip(&[], &separate, Micros(0)).expect("clip");
        assert_eq!((clip.scale.x, clip.scale.y), (2.0, 4.0));
        assert!(!uniform.on);

        let broken = keyed(0, vec![line("KFTypeAlpha", &[(0, f64::NAN)])]);
        assert!(segment_clip(&[], &broken, Micros(0)).is_none());
        let unused = keyed(0, vec![line("KFTypeScaleY", &[(0, f64::NAN)])]);
        let (_, uniform) = segment_clip(&[], &unused, Micros(0)).expect("clip");
        assert_eq!(uniform.value, 1.0);
        let axes = keyed(
            0,
            vec![
                line("KFTypeScaleX", &[(0, 3.0)]),
                line("KFTypeScaleY", &[(0, 3.0)]),
            ],
        );
        let (_, uniform) = segment_clip(&[], &axes, Micros(0)).expect("clip");
        assert_eq!(uniform.value, 3.0);
        let other = keyed(0, vec![line("KFTypeGlobalAlpha", &[(0, f64::NAN)])]);
        let (clip, _) = segment_clip(&[], &other, Micros(0)).expect("clip");
        assert_eq!(clip.alpha, 1.0);
    }

    #[test]
    fn a_non_finite_clip_is_not_a_transform() {
        let clip = Clip {
            rotation: f64::NAN,
            ..Clip::default()
        };
        assert!(picture_transform(&clip, &UniformScale::default(), 2, 2).is_none());
    }

    fn quad(corners: [f64; 8]) -> Crop {
        Crop {
            upper_left_x: corners[0],
            upper_left_y: corners[1],
            upper_right_x: corners[2],
            upper_right_y: corners[3],
            lower_left_x: corners[4],
            lower_left_y: corners[5],
            lower_right_x: corners[6],
            lower_right_y: corners[7],
        }
    }

    fn full_quad() -> Crop {
        quad([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    }

    fn colored(width: u32, height: u32, pixels: &[[u8; 4]]) -> CpuBytes {
        let mut rgba = Vec::new();
        for pixel in pixels {
            rgba.extend(pixel);
        }
        CpuBytes::new(width, height, rgba).unwrap()
    }

    fn cropped(picture: CpuBytes, crop: Crop, scale: f64) -> Vec<u8> {
        apply_crop(picture, crop, scale).rgba().to_vec()
    }

    #[test]
    fn an_absent_quad_and_a_full_quad_keep_the_picture() {
        let picture = colored(
            2,
            2,
            &[
                [1, 0, 0, 255],
                [2, 0, 0, 255],
                [3, 0, 0, 255],
                [4, 0, 0, 255],
            ],
        );
        let pixels = picture.rgba().to_vec();
        assert_eq!(cropped(picture.clone(), Crop::default(), 1.0), pixels);
        assert_eq!(cropped(picture.clone(), full_quad(), 1.0), pixels);
        assert_eq!(cropped(picture.clone(), full_quad(), 0.0), pixels);
        let mut broken = full_quad();
        broken.upper_left_x = f64::NAN;
        assert_eq!(cropped(picture, broken, 1.0), pixels);
    }

    #[test]
    fn a_half_quad_keeps_that_side_and_a_reversed_edge_swaps_it() {
        let strip = colored(
            4,
            1,
            &[
                [1, 0, 0, 255],
                [2, 0, 0, 255],
                [3, 0, 0, 255],
                [4, 0, 0, 255],
            ],
        );
        let left = quad([0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 1.0]);
        let right = quad([0.5, 0.0, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        assert_eq!(
            cropped(strip.clone(), left, 1.0),
            vec![1, 0, 0, 255, 2, 0, 0, 255]
        );
        assert_eq!(
            cropped(strip.clone(), right, f64::NAN),
            vec![3, 0, 0, 255, 4, 0, 0, 255]
        );
        assert_eq!(cropped(strip.clone(), full_quad(), 2.0), strip.rgba());
        let block = colored(
            4,
            2,
            &[
                [1, 0, 0, 255],
                [2, 0, 0, 255],
                [3, 0, 0, 255],
                [4, 0, 0, 255],
                [5, 0, 0, 255],
                [6, 0, 0, 255],
                [7, 0, 0, 255],
                [8, 0, 0, 255],
            ],
        );
        assert_eq!(
            cropped(block, full_quad(), 2.0),
            vec![6, 0, 0, 255, 7, 0, 0, 255]
        );

        let pair = colored(2, 1, &[[9, 0, 0, 255], [8, 0, 0, 255]]);
        let reversed = quad([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            cropped(pair, reversed, 1.0),
            vec![8, 0, 0, 255, 9, 0, 0, 255]
        );
    }

    #[test]
    fn the_quarter_turn_precedes_the_crop() {
        let picture = colored(
            2,
            2,
            &[
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 0, 255],
            ],
        );
        let transform =
            picture_transform(&Clip::default(), &UniformScale::default(), 1, 2).expect("transform");
        let left = quad([0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 1.0]);
        let Sample::Ready(Picture::Cpu(bytes)) = finish_picture(
            (1, 2),
            picture,
            transform,
            DisplayQuarter::Clockwise90,
            (left, 1.0),
            None,
            &Backdrop::None,
        ) else {
            panic!("ready");
        };
        assert_eq!(bytes.width(), 1);
        assert_eq!(bytes.height(), 2);
        assert_eq!(bytes.rgba(), &[0, 0, 255, 255, 255, 255, 0, 255]);
    }

    #[test]
    fn a_quad_outside_the_picture_is_clear() {
        let picture = colored(2, 1, &[[9, 0, 0, 255], [8, 0, 0, 255]]);
        let outside = quad([1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0, 1.0]);
        assert_eq!(cropped(picture, outside, 1.0), vec![0, 0, 0, 0, 0, 0, 0, 0]);
    }
}
