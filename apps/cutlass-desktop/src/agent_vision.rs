//! Agent-facing frame capture through the editor's real render path.
//!
//! The agent thread owns this service. Its renderer is deliberately private:
//! sharing the preview worker's renderer would contend with scrubbing and mix
//! decoder-cache state between two independently scheduled consumers.

use std::path::{Path, PathBuf};

use cutlass_ai::ImagePart;
use cutlass_models::{MediaId, MediaSource, Project, Rational, RationalTime, TrackKind};
use cutlass_render::Renderer;

use crate::timeline_map::Canvas;

/// Maximum width or height accepted for an agent vision frame.
pub(crate) const MAX_VISION_EDGE: u32 = 768;
pub(crate) const MEDIA_POOL_SHEET_PAGE_SIZE: usize = 12;
pub(crate) const MIN_MEDIA_POOL_SHEET_WIDTH: u32 = 320;
pub(crate) const MAX_MEDIA_POOL_SHEET_WIDTH: u32 = 1024;

const MIN_VISION_EDGE: u32 = 64;
const MAX_LABEL_FILE_NAME_CHARS: usize = 128;
const SHEET_COLUMNS: u32 = 4;
const SHEET_PADDING: u32 = 12;
const SHEET_GAP: u32 = 8;
const SHEET_HEADER_HEIGHT: u32 = 32;
const SHEET_CAPTION_HEIGHT: u32 = 28;
const SHEET_MIN_FRAME_HEIGHT: u32 = 48;
const SHEET_BACKGROUND: [u8; 4] = [18, 20, 25, 255];
const SHEET_TILE_BACKGROUND: [u8; 4] = [29, 32, 39, 255];
const SHEET_FRAME_BACKGROUND: [u8; 4] = [11, 13, 17, 255];
const SHEET_ERROR_BACKGROUND: [u8; 4] = [68, 37, 43, 255];
const SHEET_PRIMARY_TEXT: [u8; 4] = [238, 240, 245, 255];
const SHEET_SECONDARY_TEXT: [u8; 4] = [169, 176, 190, 255];
const SHEET_ERROR_TEXT: [u8; 4] = [255, 178, 186, 255];

#[derive(Debug)]
pub(crate) struct MediaPoolSheet {
    pub image: cutlass_render::RgbaImage,
    pub failures: Vec<MediaPoolSheetFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MediaPoolSheetFailure {
    pub media_id: u64,
    pub detail: String,
}

/// Reused on the agent thread: one private GPU queue + decoder cache, lazily
/// created so non-vision prompts pay no GPU bring-up.
pub(crate) struct AgentVision {
    renderer: Option<Renderer>,
    media_catalog: Vec<MediaCatalogEntry>,
}

impl Default for AgentVision {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentVision {
    pub(crate) fn new() -> Self {
        Self {
            renderer: None,
            media_catalog: Vec::new(),
        }
    }

    /// Render a frame from an isolated snapshot of the current project.
    pub(crate) fn project_frame(
        &mut self,
        project: &Project,
        seconds: f64,
        max_width: u32,
        max_height: u32,
        label_prefix: &str,
    ) -> Result<ImagePart, String> {
        let (max_width, max_height) = vision_dimensions(max_width, max_height)?;
        let rate = project.timeline().frame_rate;
        let duration = project.timeline().duration().value;
        let frame_time = RationalTime::new(seconds_to_tick(seconds, rate, duration)?, rate);

        // The immutable borrow is already a stable snapshot for the complete
        // synchronous render; cloning a large timeline here would only add
        // latency to every frame request.
        self.sync_media_catalog(project)?;
        let image = self
            .renderer()?
            .render_frame_fit(project, frame_time, max_width, max_height)
            .map_err(|error| {
                format!(
                    "could not render project frame at {:.2}s: {error}",
                    frame_time.seconds()
                )
            })?;
        let jpeg = cutlass_render::encode_jpeg(&image)
            .map_err(|error| format!("could not encode project frame as JPEG: {error}"))?;

        Ok(ImagePart::jpeg(
            jpeg,
            format!("{label_prefix} at {:.2}s", frame_time.seconds()),
        ))
    }

    /// Render a source file through a one-clip project, preserving the exact
    /// orientation, color conversion, and compositing behavior of preview.
    pub(crate) fn asset_frame(
        &mut self,
        path: &Path,
        seconds: f64,
        max_width: u32,
        max_height: u32,
    ) -> Result<ImagePart, String> {
        let (max_width, max_height) = vision_dimensions(max_width, max_height)?;
        let (image, frame_time) = self.render_asset_image(path, seconds, max_width, max_height)?;
        let jpeg = cutlass_render::encode_jpeg(&image)
            .map_err(|error| format!("could not encode asset frame as JPEG: {error}"))?;

        Ok(ImagePart::jpeg(jpeg, asset_label(path, frame_time)))
    }

    /// Render one midpoint thumbnail per visual source and compose them into a
    /// single model-friendly image. Per-source failures become placeholder
    /// tiles so one corrupt or offline asset cannot hide the rest of the pool.
    pub(crate) fn media_pool_sheet(
        &mut self,
        sources: &[&MediaSource],
        page: u32,
        total_pages: u32,
        max_width: u32,
    ) -> Result<MediaPoolSheet, String> {
        if sources.len() > MEDIA_POOL_SHEET_PAGE_SIZE {
            return Err(format!(
                "media-pool sheet accepts at most {MEDIA_POOL_SHEET_PAGE_SIZE} visual items"
            ));
        }
        let layout = MediaPoolSheetLayout::new(max_width, sources.len())?;
        let mut previews = Vec::with_capacity(sources.len());
        for source in sources {
            let preview = self
                .render_asset_image(
                    source.path(),
                    sheet_sample_seconds(source),
                    layout.cell_width,
                    layout.frame_height,
                )
                .map(|(image, _)| image);
            previews.push(SheetPreview { source, preview });
        }
        compose_media_pool_sheet(&previews, page, total_pages, layout)
    }

    fn render_asset_image(
        &mut self,
        path: &Path,
        seconds: f64,
        max_width: u32,
        max_height: u32,
    ) -> Result<(cutlass_render::RgbaImage, RationalTime), String> {
        if max_width == 0 || max_height == 0 {
            return Err("asset frame width and height must be greater than zero".into());
        }
        validate_seconds(seconds)?;
        let asset_name = safe_file_name(path);

        let probe = cutlass_decoder::probe(path).map_err(|error| {
            format!(
                "could not inspect asset {asset_name}: {}",
                redact_asset_path(path, &error)
            )
        })?;
        if probe.width == 0 || probe.height == 0 {
            let kind = if probe.has_audio {
                "audio-only"
            } else {
                "nonvisual"
            };
            return Err(format!(
                "asset {asset_name} is {kind}; choose a video or still image"
            ));
        }

        let mut source = if probe.is_image {
            MediaSource::image(path, probe.width, probe.height)
        } else {
            MediaSource::new(
                path,
                probe.width,
                probe.height,
                probe.frame_rate,
                probe.frame_count,
                probe.has_audio,
            )
        };
        // Scratch projects use one stable media-id namespace. This keeps a
        // repeated frame/strip request for the same asset on one warm decoder
        // instead of allocating a fresh cache key for every call.
        source.id = MediaId::from_raw(1);
        let tick = seconds_to_tick(seconds, source.frame_rate, source.duration.value).map_err(
            |error| format!("could not select a frame from asset {asset_name}: {error}"),
        )?;
        let (project, frame_time) = scratch_project(source, tick).map_err(|error| {
            format!("could not prepare asset {asset_name} for rendering: {error}")
        })?;

        // Do not decode directly here: native decoders may return NV12 and
        // encoded orientation metadata. The scratch project guarantees exact
        // parity with the editor's project renderer.
        self.sync_media_catalog(&project)?;
        let image = self
            .renderer()?
            .render_frame_fit(&project, frame_time, max_width, max_height)
            .map_err(|error| {
                format!(
                    "could not render asset {asset_name} at {:.2}s: {}",
                    frame_time.seconds(),
                    redact_asset_path(path, &error)
                )
            })?;
        Ok((image, frame_time))
    }

    fn renderer(&mut self) -> Result<&mut Renderer, String> {
        if self.renderer.is_none() {
            let renderer = Renderer::new_headless()
                .map_err(|error| format!("vision renderer is unavailable: {error}"))?;
            self.renderer = Some(renderer);
        }
        self.renderer
            .as_mut()
            .ok_or_else(|| "vision renderer is unavailable".to_string())
    }

    fn sync_media_catalog(&mut self, project: &Project) -> Result<(), String> {
        let next = media_catalog(project);
        if self.media_catalog != next {
            self.renderer()?.reset_media_sources();
            self.media_catalog = next;
        }
        Ok(())
    }
}

struct SheetPreview<'a> {
    source: &'a MediaSource,
    preview: Result<cutlass_render::RgbaImage, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MediaPoolSheetLayout {
    width: u32,
    height: u32,
    cell_width: u32,
    frame_height: u32,
    row_height: u32,
}

impl MediaPoolSheetLayout {
    fn new(max_width: u32, item_count: usize) -> Result<Self, String> {
        if item_count > MEDIA_POOL_SHEET_PAGE_SIZE {
            return Err(format!(
                "media-pool sheet accepts at most {MEDIA_POOL_SHEET_PAGE_SIZE} visual items"
            ));
        }
        let width = max_width.clamp(MIN_MEDIA_POOL_SHEET_WIDTH, MAX_MEDIA_POOL_SHEET_WIDTH);
        let horizontal_chrome = SHEET_PADDING
            .checked_mul(2)
            .and_then(|padding| padding.checked_add(SHEET_GAP * (SHEET_COLUMNS - 1)))
            .ok_or("media-pool sheet width overflow")?;
        let cell_width = width
            .checked_sub(horizontal_chrome)
            .ok_or("media-pool sheet is too narrow")?
            / SHEET_COLUMNS;
        let frame_height = cell_width
            .checked_mul(9)
            .ok_or("media-pool sheet frame-height overflow")?
            / 16;
        let frame_height = frame_height.max(SHEET_MIN_FRAME_HEIGHT);
        let row_height = frame_height
            .checked_add(SHEET_CAPTION_HEIGHT)
            .ok_or("media-pool sheet row-height overflow")?;
        let row_count = u32::try_from(item_count.max(1))
            .map_err(|_| "media-pool sheet item count overflow")?
            .div_ceil(SHEET_COLUMNS);
        let height = SHEET_HEADER_HEIGHT
            .checked_add(
                row_height
                    .checked_mul(row_count)
                    .ok_or("media-pool sheet height overflow")?,
            )
            .and_then(|height| height.checked_add(SHEET_GAP * row_count.saturating_sub(1)))
            .and_then(|height| height.checked_add(SHEET_PADDING))
            .ok_or("media-pool sheet height overflow")?;

        Ok(Self {
            width,
            height,
            cell_width,
            frame_height,
            row_height,
        })
    }

    fn tile_origin(self, index: usize) -> Result<(i32, i32), String> {
        let index = u32::try_from(index).map_err(|_| "media-pool sheet tile index overflow")?;
        let column = index % SHEET_COLUMNS;
        let row = index / SHEET_COLUMNS;
        let x = SHEET_PADDING
            .checked_add(
                column
                    .checked_mul(self.cell_width + SHEET_GAP)
                    .ok_or("media-pool sheet tile x overflow")?,
            )
            .ok_or("media-pool sheet tile x overflow")?;
        let y = SHEET_HEADER_HEIGHT
            .checked_add(
                row.checked_mul(self.row_height + SHEET_GAP)
                    .ok_or("media-pool sheet tile y overflow")?,
            )
            .ok_or("media-pool sheet tile y overflow")?;
        Ok((
            i32::try_from(x).map_err(|_| "media-pool sheet tile x exceeds i32")?,
            i32::try_from(y).map_err(|_| "media-pool sheet tile y exceeds i32")?,
        ))
    }
}

fn compose_media_pool_sheet(
    previews: &[SheetPreview<'_>],
    page: u32,
    total_pages: u32,
    layout: MediaPoolSheetLayout,
) -> Result<MediaPoolSheet, String> {
    let mut canvas = Canvas::new(layout.width, layout.height, SHEET_BACKGROUND)?;
    canvas.draw_text_clipped(
        &format!("MEDIA POOL  PAGE {page}/{total_pages}"),
        SHEET_PADDING as i32,
        9,
        layout.width.saturating_sub(SHEET_PADDING * 2) as i32,
        9,
        SHEET_PRIMARY_TEXT,
        1,
    );

    if previews.is_empty() {
        canvas.draw_text_clipped(
            "NO VISUAL MEDIA ON THIS PAGE",
            SHEET_PADDING as i32,
            SHEET_HEADER_HEIGHT as i32 + 18,
            layout.width.saturating_sub(SHEET_PADDING * 2) as i32,
            9,
            SHEET_SECONDARY_TEXT,
            1,
        );
        return Ok(MediaPoolSheet {
            image: canvas.into_image(),
            failures: Vec::new(),
        });
    }

    let mut failures = Vec::new();
    for (index, item) in previews.iter().enumerate() {
        let (x, y) = layout.tile_origin(index)?;
        let cell_width = layout.cell_width as i32;
        let frame_height = layout.frame_height as i32;
        canvas.fill_rect(
            x,
            y,
            cell_width,
            layout.row_height as i32,
            SHEET_TILE_BACKGROUND,
        );
        canvas.fill_rect(x, y, cell_width, frame_height, SHEET_FRAME_BACKGROUND);

        let render_error = match &item.preview {
            Ok(image) if canvas.draw_image_centered(image, x, y, cell_width, frame_height) => None,
            Ok(_) => Some("renderer returned a malformed RGBA image".to_string()),
            Err(error) => Some(error.clone()),
        };
        if let Some(detail) = render_error {
            canvas.fill_rect(x, y, cell_width, frame_height, SHEET_ERROR_BACKGROUND);
            canvas.draw_text_clipped(
                "UNAVAILABLE",
                x + 6,
                y + frame_height / 2 - 4,
                cell_width - 12,
                9,
                SHEET_ERROR_TEXT,
                1,
            );
            failures.push(MediaPoolSheetFailure {
                media_id: item.source.id.raw(),
                detail,
            });
        }

        canvas.draw_text_clipped(
            &format!(
                "#{} {}",
                item.source.id.raw(),
                safe_file_name(item.source.path())
            ),
            x + 5,
            y + frame_height + 5,
            cell_width - 10,
            9,
            SHEET_PRIMARY_TEXT,
            1,
        );
        canvas.draw_text_clipped(
            &format!(
                "{}  {}x{}",
                format_sheet_duration(item.source.duration.seconds()),
                item.source.width,
                item.source.height
            ),
            x + 5,
            y + frame_height + 16,
            cell_width - 10,
            9,
            SHEET_SECONDARY_TEXT,
            1,
        );
    }

    Ok(MediaPoolSheet {
        image: canvas.into_image(),
        failures,
    })
}

fn sheet_sample_seconds(source: &MediaSource) -> f64 {
    if source.is_image {
        return 0.0;
    }
    let duration = source.duration.seconds();
    if duration.is_finite() && duration > 0.0 {
        duration / 2.0
    } else {
        0.0
    }
}

fn format_sheet_duration(seconds: f64) -> String {
    if seconds.is_finite() && seconds >= 0.0 {
        format!("{seconds:.1}s")
    } else {
        "unknown duration".to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MediaCatalogEntry {
    id: u64,
    path: PathBuf,
    width: u32,
    height: u32,
    frame_rate: Rational,
    duration_ticks: i64,
    is_image: bool,
}

fn media_catalog(project: &Project) -> Vec<MediaCatalogEntry> {
    let mut catalog: Vec<_> = project
        .media_iter()
        .map(|source| MediaCatalogEntry {
            id: source.id.raw(),
            path: source.path.clone(),
            width: source.width,
            height: source.height,
            frame_rate: source.frame_rate,
            duration_ticks: source.duration.value,
            is_image: source.is_image,
        })
        .collect();
    catalog.sort_unstable_by_key(|source| source.id);
    catalog
}

/// Reject absent dimensions, then bound render/readback work to a small,
/// model-friendly image without allowing callers to request tiny edge cases.
pub(crate) fn vision_dimensions(max_width: u32, max_height: u32) -> Result<(u32, u32), String> {
    if max_width == 0 || max_height == 0 {
        return Err(format!(
            "vision frame width and height must be greater than zero (got {max_width}x{max_height})"
        ));
    }
    Ok((
        max_width.clamp(MIN_VISION_EDGE, MAX_VISION_EDGE),
        max_height.clamp(MIN_VISION_EDGE, MAX_VISION_EDGE),
    ))
}

/// Snap non-negative seconds to the nearest exact frame tick and clamp it to
/// the final available frame. A zero-duration timeline resolves to tick zero,
/// which lets an empty project render its canvas without `duration - 1`
/// underflow; source construction rejects zero-duration visual media.
pub(crate) fn seconds_to_tick(
    seconds: f64,
    rate: Rational,
    duration_ticks: i64,
) -> Result<i64, String> {
    validate_seconds(seconds)?;
    if !rate.is_valid() {
        return Err(format!(
            "frame rate must be positive (got {}/{})",
            rate.num, rate.den
        ));
    }
    if duration_ticks <= 0 {
        return Ok(0);
    }

    let last_tick = duration_ticks.saturating_sub(1);
    let raw_tick = seconds * f64::from(rate.num) / f64::from(rate.den);
    if !raw_tick.is_finite() {
        // A finite but enormous request overflowed only during conversion; it
        // is unambiguously beyond the source and therefore clamps to its tail.
        return Ok(last_tick);
    }
    let snapped = raw_tick.round();
    if snapped <= 0.0 {
        Ok(0)
    } else if snapped >= last_tick as f64 {
        Ok(last_tick)
    } else {
        Ok(snapped as i64)
    }
}

/// Build the proven thumbnail-style scratch project: one main video lane and
/// one clip spanning the complete source. The returned time is defensively
/// clamped even when a caller supplies a raw tick.
pub(crate) fn scratch_project(
    source: MediaSource,
    requested_tick: i64,
) -> Result<(Project, RationalTime), String> {
    if source.width == 0 || source.height == 0 {
        return if source.is_audio_only() {
            Err("source is audio-only and has no visual frame".to_string())
        } else {
            Err(format!(
                "source has no visual frame (reported {}x{})",
                source.width, source.height
            ))
        };
    }
    if !source.frame_rate.is_valid() {
        return Err(format!(
            "source frame rate must be positive (got {}/{})",
            source.frame_rate.num, source.frame_rate.den
        ));
    }
    let duration_ticks = source.duration.value;
    if duration_ticks <= 0 {
        return Err("visual source has no frames to render".to_string());
    }

    let rate = source.frame_rate;
    let source_range = source.full_range();
    let actual_tick = requested_tick.clamp(0, duration_ticks.saturating_sub(1));
    let mut project = Project::new("agent vision asset", rate);
    let media = project.add_media(source);
    let track = project.add_track(TrackKind::Video, "Media");
    project
        .add_clip(track, media, source_range, RationalTime::new(0, rate))
        .map_err(|error| format!("could not build vision scratch clip: {error}"))?;

    Ok((project, RationalTime::new(actual_tick, rate)))
}

/// Provider-facing asset label. Only the final path component is exposed, and
/// controls/oversized synthetic names are contained before entering a prompt.
pub(crate) fn asset_label(path: &Path, frame_time: RationalTime) -> String {
    format!("{} at {:.2}s", safe_file_name(path), frame_time.seconds())
}

fn validate_seconds(seconds: f64) -> Result<(), String> {
    if !seconds.is_finite() {
        return Err("frame time must be a finite number".to_string());
    }
    if seconds < 0.0 {
        return Err(format!("frame time must not be negative (got {seconds}s)"));
    }
    Ok(())
}

pub(crate) fn safe_file_name(path: &Path) -> String {
    let Some(name) = path.file_name() else {
        return "asset".to_string();
    };
    let mut safe = String::with_capacity(name.len().min(MAX_LABEL_FILE_NAME_CHARS));
    for character in name
        .to_string_lossy()
        .chars()
        .take(MAX_LABEL_FILE_NAME_CHARS)
    {
        safe.push(if character.is_control() {
            '\u{fffd}'
        } else {
            character
        });
    }
    if safe.is_empty() {
        "asset".to_string()
    } else {
        safe
    }
}

fn redact_asset_path(path: &Path, error: &dyn std::fmt::Display) -> String {
    let detail = error.to_string();
    let full_path = path.to_string_lossy();
    if full_path.is_empty() {
        detail
    } else {
        detail.replace(full_path.as_ref(), &safe_file_name(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_catalog_is_ordered_and_detects_relinks() {
        let mut project = Project::new("catalog", Rational::FPS_24);
        let second = project.add_media(MediaSource::new(
            "/private/b.mov",
            1920,
            1080,
            Rational::FPS_24,
            48,
            false,
        ));
        let first = project.add_media(MediaSource::image("/private/a.png", 800, 600));

        let before = media_catalog(&project);
        assert!(before.windows(2).all(|pair| pair[0].id < pair[1].id));

        project.media_mut(first).unwrap().path = "/replacement/a.png".into();
        let after = media_catalog(&project);
        assert_ne!(before, after);
        assert!(
            after
                .iter()
                .any(|source| source.id == second.raw() && source.path.ends_with("b.mov"))
        );
    }

    #[test]
    fn provider_errors_redact_full_asset_paths() {
        let path = Path::new("/private/customer-alpha/unreleased/take.mov");
        let detail = format!("decoder could not open {}", path.display());
        let redacted = redact_asset_path(path, &detail);
        assert_eq!(redacted, "decoder could not open take.mov");
        assert!(!redacted.contains("customer-alpha"));
    }

    #[test]
    fn media_pool_sheet_layout_is_bounded_and_pages_twelve_tiles() {
        let narrow = MediaPoolSheetLayout::new(1, MEDIA_POOL_SHEET_PAGE_SIZE).unwrap();
        assert_eq!(narrow.width, MIN_MEDIA_POOL_SHEET_WIDTH);
        assert!(narrow.height <= 768);
        let (last_x, last_y) = narrow.tile_origin(MEDIA_POOL_SHEET_PAGE_SIZE - 1).unwrap();
        assert!(last_x > 0);
        assert!(last_y > SHEET_HEADER_HEIGHT as i32);

        let wide = MediaPoolSheetLayout::new(u32::MAX, 1).unwrap();
        assert_eq!(wide.width, MAX_MEDIA_POOL_SHEET_WIDTH);
        assert!(wide.height < narrow.height);
        assert!(MediaPoolSheetLayout::new(768, MEDIA_POOL_SHEET_PAGE_SIZE + 1).is_err());
    }

    #[test]
    fn media_pool_sheet_composition_keeps_failed_assets_as_placeholders() {
        let mut available = MediaSource::new(
            "/private/available.mov",
            1920,
            1080,
            Rational::FPS_24,
            240,
            false,
        );
        available.id = MediaId::from_raw(7);
        let mut offline = MediaSource::new(
            "/private/offline.mov",
            1920,
            1080,
            Rational::FPS_24,
            240,
            false,
        );
        offline.id = MediaId::from_raw(8);
        let thumbnail = cutlass_render::RgbaImage::new(16, 9, vec![200; 16 * 9 * 4]);
        let previews = [
            SheetPreview {
                source: &available,
                preview: Ok(thumbnail),
            },
            SheetPreview {
                source: &offline,
                preview: Err("offline.mov could not be opened".into()),
            },
        ];
        let layout = MediaPoolSheetLayout::new(768, previews.len()).unwrap();

        let sheet = compose_media_pool_sheet(&previews, 1, 1, layout).unwrap();

        assert!(sheet.image.is_well_formed());
        assert_eq!(sheet.image.width, 768);
        assert_eq!(sheet.image.height, layout.height);
        assert_eq!(
            sheet.failures,
            vec![MediaPoolSheetFailure {
                media_id: 8,
                detail: "offline.mov could not be opened".into(),
            }]
        );
        let (x, y) = layout.tile_origin(1).unwrap();
        assert_eq!(
            sheet.image.pixel(x as u32, y as u32),
            SHEET_ERROR_BACKGROUND
        );
    }

    #[test]
    fn sheet_samples_video_midpoint_and_still_origin() {
        let video = MediaSource::new(
            "/private/video.mov",
            1920,
            1080,
            Rational::FPS_24,
            240,
            false,
        );
        let still = MediaSource::image("/private/still.png", 800, 600);

        assert!((sheet_sample_seconds(&video) - 5.0).abs() < f64::EPSILON);
        assert_eq!(sheet_sample_seconds(&still), 0.0);
    }

    #[test]
    fn opaque_sheet_encodes_as_jpeg_smaller_than_png() {
        let layout = MediaPoolSheetLayout::new(768, 0).unwrap();
        let sheet = compose_media_pool_sheet(&[], 1, 1, layout).unwrap();
        let jpeg = cutlass_render::encode_jpeg(&sheet.image).expect("jpeg");
        let png = cutlass_render::encode_png(&sheet.image).expect("png");
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
        assert!(
            jpeg.len() < png.len(),
            "jpeg {} bytes should beat png {} bytes for an opaque sheet",
            jpeg.len(),
            png.len()
        );
    }
}
