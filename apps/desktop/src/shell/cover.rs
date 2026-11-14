//! The project's cover: the timeline's picture at one time, or a picture
//! from disk cut to the canvas. It is kept as [`COVER_FILE`] in the project
//! folder, drawn when the cover is picked, and the draft says where it came
//! from. Undo and redo draw the file again when they change the cover, so
//! the file always shows the draft's cover.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};

use cutlass_engine::Thumb;
use cutlass_model::{COVER_FILE, Cover, Draft, Micros};
use cutlass_playback::FrameCache;
use cutlass_renderer_software::SoftwareDevice;
use cutlass_session::{Request, Session};

use super::{Shell, ShellError, export_failure, write_replaced};

/// The most rows a cover from disk keeps. A taller picture is shrunk.
const COVER_ROWS: u32 = 1080;

/// JPEG quality out of 100 for a cover from disk, as stills are written.
const COVER_QUALITY: u8 = 92;

/// Small pictures of the whole timeline, drawn on their own thread for the
/// cover dialog's strip.
pub(crate) struct CoverStrip {
    revision: u64,
    cells: usize,
    rows: u32,
    landed: Receiver<(usize, Thumb)>,
    pictures: Vec<Option<Thumb>>,
    /// Bumped each time a picture lands, so the window uploads it.
    version: u64,
}

impl Shell {
    /// The open project's cover, when one was picked.
    #[must_use]
    pub fn cover(&self) -> Option<&Cover> {
        self.draft().and_then(|draft| draft.cover.as_ref())
    }

    /// The cover's picture file, when the project has a cover and the file
    /// is there.
    #[must_use]
    pub fn cover_file(&self) -> Option<PathBuf> {
        self.cover()?;
        let file = self.path.as_ref()?.join(COVER_FILE);
        file.is_file().then_some(file)
    }

    /// Changes each time the cover file is written or removed, so a
    /// picture of it is read again.
    #[must_use]
    pub fn cover_version(&self) -> u64 {
        self.cover_version
    }

    /// Make the timeline's picture at `time` the cover. The picture is the
    /// one export draws, at the canvas size; it is written as a job, and
    /// the cover is set once it lands, as one undo step.
    pub fn set_cover_frame(&mut self, time: Micros) -> Result<(), ShellError> {
        let end = self.cut_duration().0;
        if end <= 0 {
            return Err(ShellError::Unavailable(
                "the timeline has no picture to make a cover of".to_owned(),
            ));
        }
        let time = Micros(time.0.clamp(0, end - 1));
        self.draw_cover(Cover::Frame { time }, true)
    }

    /// Make the picture at `path` the cover, cut to the canvas's shape from
    /// its middle. It is written as a job, and the cover is set once it
    /// lands, as one undo step.
    pub fn set_cover_image(&mut self, path: &Path) -> Result<(), ShellError> {
        let cover = Cover::Image {
            path: path.display().to_string(),
        };
        self.draw_cover(cover, true)
    }

    /// The picture at `path` cut to the canvas as a cover from disk would
    /// be, at most `rows` rows, for the dialog to show before it is picked.
    pub fn cover_image_preview(&self, path: &Path, rows: u32) -> Result<Thumb, ShellError> {
        let (width, height) = self.canvas_shape()?;
        image_cover(&self.readers, path, (width, height), rows)
    }

    /// The picture file at `path` (a project's cover, say) upright and at
    /// most `rows` rows, or `None` when it does not open.
    #[must_use]
    pub fn picture_thumb(&self, path: &Path, rows: u32) -> Option<Thumb> {
        cutlass_engine::frames_of(&self.readers, path, &[Micros(0)], rows)
            .ok()?
            .into_iter()
            .next()
    }

    /// Write the picture `cover` names to the cover file as a job. With
    /// `set`, the cover is stored once the file is written; without, the
    /// draft already names it (after undo or redo).
    fn draw_cover(&mut self, cover: Cover, set: bool) -> Result<(), ShellError> {
        let root = self.path.clone().ok_or_else(|| {
            ShellError::Unavailable("save the project before picking a cover".to_owned())
        })?;
        let destination = root.join(COVER_FILE);
        let draft = self.session_draft()?.clone();
        let canvas = self.canvas_shape()?;
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let fonts = Arc::clone(&self.face);
        let gpu = self.gpu.as_ref().map(|preview| {
            let renderer = preview.renderer();
            (renderer.device().clone(), renderer.queue().clone())
        });
        self.start_job("Cover", move |progress| {
            progress.stage("Drawing the cover");
            let written = match &cover {
                Cover::Frame { time } => {
                    let renderer =
                        gpu.map(|(device, queue)| cutlass_render::Renderer::new(device, queue));
                    cutlass_export::write_still(
                        &cutlass_export::MovieSources {
                            draft: &draft,
                            root: Some(&root),
                            readers: &readers,
                            fonts: &*fonts,
                            renderer: renderer.as_ref(),
                            encoders: &encoders,
                        },
                        *time,
                        cutlass_export::StillFormat::Jpeg,
                        &destination,
                    )
                    .map(|_| ())
                    .map_err(export_failure)
                }
                Cover::Image { path } => image_cover(&readers, Path::new(path), canvas, COVER_ROWS)
                    .and_then(|picture| jpeg(&picture))
                    .and_then(|bytes| write_replaced(&destination, &bytes)),
            };
            Box::new(move |shell: &mut Shell| {
                written?;
                shell.cover_version = shell.cover_version.wrapping_add(1);
                if set {
                    shell.apply(Request::SetCover { cover: Some(cover) })?;
                    shell.set_notice("Cover saved".to_owned());
                }
                Ok(())
            })
        })
    }

    /// After undo or redo: draw the cover file again when the cover the
    /// draft names is not `before`, or remove it when the draft has none.
    pub(super) fn follow_cover(&mut self, before: Option<Cover>) {
        let after = self.cover().cloned();
        if after == before {
            return;
        }
        let result = match after {
            Some(cover) => self.draw_cover(cover, false),
            None => self.remove_cover_file(),
        };
        if let Err(error) = result {
            cutlass_logger::warn!("cover file not redrawn: {error}");
        }
    }

    fn remove_cover_file(&mut self) -> Result<(), ShellError> {
        let Some(root) = self.path.as_ref() else {
            return Ok(());
        };
        match std::fs::remove_file(root.join(COVER_FILE)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(ShellError::Io(error.to_string())),
        }
        self.cover_version = self.cover_version.wrapping_add(1);
        Ok(())
    }

    /// The canvas's width and height.
    fn canvas_shape(&self) -> Result<(u32, u32), ShellError> {
        let draft = self.session_draft()?;
        let width = u32::try_from(draft.canvas_config.width).unwrap_or(0);
        let height = u32::try_from(draft.canvas_config.height).unwrap_or(0);
        if width == 0 || height == 0 {
            return Err(ShellError::Unavailable("the canvas has no size".to_owned()));
        }
        Ok((width, height))
    }

    /// `cells` pictures of the whole timeline, `rows` rows tall, for the
    /// cover dialog's strip: one from the middle of each equal stretch.
    /// They are drawn on their own thread, started on the first ask and
    /// again when the draft changes; a cell not drawn yet is `None`. The
    /// number changes each time a picture lands.
    pub fn cover_strip(&mut self, cells: usize, rows: u32) -> (&[Option<Thumb>], u64) {
        let revision = self.session.as_ref().map_or(0, Session::revision);
        let current = self.cover_strip.as_ref().is_some_and(|strip| {
            strip.revision == revision && strip.cells == cells && strip.rows == rows
        });
        if !current {
            self.cover_strip = self.start_cover_strip(revision, cells, rows);
        }
        let Some(strip) = self.cover_strip.as_mut() else {
            return (&[], 0);
        };
        while let Ok((index, picture)) = strip.landed.try_recv() {
            if let Some(slot) = strip.pictures.get_mut(index) {
                *slot = Some(picture);
                strip.version = strip.version.wrapping_add(1);
            }
        }
        (&strip.pictures, strip.version)
    }

    /// Stop drawing the cover dialog's strip.
    pub fn forget_cover_strip(&mut self) {
        self.cover_strip = None;
    }

    fn start_cover_strip(&self, revision: u64, cells: usize, rows: u32) -> Option<CoverStrip> {
        let draft = self.draft()?.clone();
        let end = self.cut_duration().0;
        if cells == 0 || rows == 0 || end <= 0 {
            return None;
        }
        let times: Vec<Micros> = (0..cells)
            .map(|cell| {
                let middle = (2 * cell as i128 + 1) * i128::from(end) / (2 * cells as i128);
                Micros(i64::try_from(middle).unwrap_or(0).clamp(0, end - 1))
            })
            .collect();
        let readers = self.readers.clone();
        let root = self.path.clone();
        let fonts = Arc::clone(&self.face);
        let (send, landed) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("cover strip".to_owned())
            .spawn(move || {
                draw_strip(
                    draft,
                    &times,
                    rows,
                    &readers,
                    root.as_deref(),
                    &*fonts,
                    &send,
                )
            });
        if let Err(error) = spawned {
            cutlass_logger::warn!("cover strip did not start: {error}");
            return None;
        }
        Some(CoverStrip {
            revision,
            cells,
            rows,
            landed,
            pictures: vec![None; cells],
            version: 0,
        })
    }
}

/// Draw each of `times` and send it shrunk to `rows` rows, until the
/// dialog stops listening.
fn draw_strip(
    draft: Draft,
    times: &[Micros],
    rows: u32,
    readers: &cutlass_eval::ReaderFactory,
    root: Option<&Path>,
    fonts: &dyn cutlass_eval::Fonts,
    send: &mpsc::Sender<(usize, Thumb)>,
) {
    let session = Session::open(draft);
    let mut cache = FrameCache::new();
    cache.set_patient(true);
    let mut device = SoftwareDevice::new();
    for (index, time) in times.iter().enumerate() {
        let picture = match cache.frame(&session, &mut device, *time, readers, root, fonts) {
            Ok(picture) => picture,
            Err(error) => {
                cutlass_logger::warn!("cover strip at {} us: {error}", time.0);
                continue;
            }
        };
        let Some(small) = shrink(picture.width, picture.height, &picture.rgba, rows) else {
            continue;
        };
        if send.send((index, small)).is_err() {
            return;
        }
    }
}

/// The picture at `path`, laid on black, cut to `canvas`'s shape from its
/// middle and at most `rows` rows.
fn image_cover(
    readers: &cutlass_eval::ReaderFactory,
    path: &Path,
    canvas: (u32, u32),
    rows: u32,
) -> Result<Thumb, ShellError> {
    let picture = cutlass_engine::frames_of(readers, path, &[Micros(0)], rows)
        .map_err(|error| ShellError::Unavailable(format!("the picture did not open: {error}")))?
        .into_iter()
        .next()
        .ok_or_else(|| ShellError::Unavailable("the picture did not open".to_owned()))?;
    cut_to(&picture, canvas)
        .ok_or_else(|| ShellError::Unavailable("the picture is empty".to_owned()))
}

/// The middle of `picture` in the shape of `canvas`, opaque. The samples
/// are premultiplied, so on black each keeps its color.
fn cut_to(picture: &Thumb, (canvas_w, canvas_h): (u32, u32)) -> Option<Thumb> {
    let (width, height) = (picture.width, picture.height);
    if width == 0 || height == 0 || canvas_w == 0 || canvas_h == 0 {
        return None;
    }
    // Keep the whole height when the picture is wider than the canvas,
    // else the whole width.
    let (keep_w, keep_h) =
        if u64::from(width) * u64::from(canvas_h) >= u64::from(height) * u64::from(canvas_w) {
            let keep_w = u64::from(height) * u64::from(canvas_w) / u64::from(canvas_h);
            (u32::try_from(keep_w).ok()?.max(1), height)
        } else {
            let keep_h = u64::from(width) * u64::from(canvas_h) / u64::from(canvas_w);
            (width, u32::try_from(keep_h).ok()?.max(1))
        };
    let left = (width - keep_w) / 2;
    let top = (height - keep_h) / 2;
    let stride = width as usize * 4;
    let mut rgba = Vec::with_capacity(keep_w as usize * keep_h as usize * 4);
    for row in top..top + keep_h {
        let start = row as usize * stride + left as usize * 4;
        for pixel in picture
            .rgba
            .get(start..start + keep_w as usize * 4)?
            .as_chunks::<4>()
            .0
        {
            rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
        }
    }
    Some(Thumb {
        width: keep_w,
        height: keep_h,
        rgba,
    })
}

/// `rgba`, `width` by `height`, averaged down to `rows` rows with its
/// shape kept. A picture no taller is copied.
fn shrink(width: u32, height: u32, rgba: &[u8], rows: u32) -> Option<Thumb> {
    if width == 0 || height == 0 || rgba.len() != width as usize * height as usize * 4 {
        return None;
    }
    if height <= rows {
        return Some(Thumb {
            width,
            height,
            rgba: rgba.to_vec(),
        });
    }
    let out_h = rows.max(1);
    let out_w = u32::try_from(
        (u64::from(width) * u64::from(out_h) + u64::from(height) / 2) / u64::from(height),
    )
    .ok()?
    .max(1);
    let mut out = Vec::with_capacity(out_w as usize * out_h as usize * 4);
    for y in 0..out_h {
        let y0 = (u64::from(y) * u64::from(height) / u64::from(out_h)) as usize;
        let y1 = ((u64::from(y + 1) * u64::from(height) / u64::from(out_h)) as usize).max(y0 + 1);
        for x in 0..out_w {
            let x0 = (u64::from(x) * u64::from(width) / u64::from(out_w)) as usize;
            let x1 =
                ((u64::from(x + 1) * u64::from(width) / u64::from(out_w)) as usize).max(x0 + 1);
            let mut sum = [0_u32; 4];
            for row in y0..y1 {
                let line = row * width as usize * 4;
                for column in x0..x1 {
                    let at = line + column * 4;
                    for (channel, total) in sum.iter_mut().enumerate() {
                        *total += u32::from(rgba[at + channel]);
                    }
                }
            }
            let count = ((y1 - y0) * (x1 - x0)) as u32;
            for total in sum {
                out.push(((total + count / 2) / count) as u8);
            }
        }
    }
    Some(Thumb {
        width: out_w,
        height: out_h,
        rgba: out,
    })
}

/// `picture` as a baseline JPEG file.
fn jpeg(picture: &Thumb) -> Result<Vec<u8>, ShellError> {
    let failed = |error: cutlass_decode::CodecError| {
        ShellError::Unavailable(format!("the cover was not written: {error}"))
    };
    let mut planes = cutlass_jpeg::Planes::default();
    planes
        .from_rgb(
            &picture.rgba,
            4,
            picture.width,
            picture.height,
            cutlass_jpeg::Sampling::Yuv420,
        )
        .map_err(failed)?;
    let mut coefficients = cutlass_jpeg::Coefficients::new();
    coefficients
        .transform(&planes.source().map_err(failed)?)
        .map_err(failed)?;
    let mut bytes = Vec::new();
    coefficients
        .write(
            &cutlass_jpeg::WriteSettings {
                quality: COVER_QUALITY,
                restart_interval: 0,
                jfif: true,
            },
            &mut bytes,
        )
        .map_err(failed)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests;
