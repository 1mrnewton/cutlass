//! The picture on a project's card at home: the timeline's first frame as
//! the compositor draws it, kept as [`THUMBNAIL_FILE`] in the project
//! folder. It is apart from the cover the user picks, which only the
//! editor's Cover box and exported files show. A card's picture is drawn
//! again on its own thread whenever the project was saved after it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::SystemTime;

use cutlass_engine::Thumb;
use cutlass_model::{Micros, THUMBNAIL_FILE};

use super::cover::{jpeg, on_black};
use super::{Shell, ShellError, export_failure, write_replaced};

/// The most rows a card's picture keeps: twice the card's height, and some.
const THUMBNAIL_ROWS: u32 = 360;

/// The projects home's pictures, drawn one after another on their own
/// thread.
pub(crate) struct ProjectPictures {
    ask: Sender<PathBuf>,
    landed: Receiver<PathBuf>,
    /// Each project asked for, with its draft's time when it was asked, so
    /// a project is drawn once for each save.
    asked: HashMap<PathBuf, SystemTime>,
    /// How many asked for have not landed yet.
    waiting: usize,
}

impl Shell {
    /// The file of `project`'s card picture, when there is one, and whether
    /// a picture is still being drawn. When the project was saved after its
    /// picture was drawn, or it has none, it is drawn again on its own
    /// thread, and the old one is given until the new one lands.
    pub fn project_thumbnail(&mut self, project: &Path) -> (Option<PathBuf>, bool) {
        let file = project.join(THUMBNAIL_FILE);
        if self.project_pictures.is_none() {
            self.project_pictures = self.start_project_pictures();
        }
        let Some(thumbnails) = self.project_pictures.as_mut() else {
            return (file.is_file().then_some(file), false);
        };
        while thumbnails.landed.try_recv().is_ok() {
            thumbnails.waiting = thumbnails.waiting.saturating_sub(1);
        }
        if let Some(saved) = cutlass_store::saved_at(project) {
            let stale = modified(&file).is_none_or(|drawn| drawn < saved);
            if stale
                && thumbnails.asked.get(project) != Some(&saved)
                && thumbnails.ask.send(project.to_path_buf()).is_ok()
            {
                thumbnails.asked.insert(project.to_path_buf(), saved);
                thumbnails.waiting += 1;
            }
        }
        (file.is_file().then_some(file), thumbnails.waiting > 0)
    }

    fn start_project_pictures(&self) -> Option<ProjectPictures> {
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let fonts = Arc::clone(&self.face);
        let gpu = self.gpu.as_ref().map(|preview| {
            let renderer = preview.renderer();
            (renderer.device().clone(), renderer.queue().clone())
        });
        let (ask, asks) = mpsc::channel::<PathBuf>();
        let (send, landed) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("project thumbnails".to_owned())
            .spawn(move || {
                let renderer =
                    gpu.map(|(device, queue)| cutlass_render::Renderer::new(device, queue));
                while let Ok(project) = asks.recv() {
                    let drawn =
                        draw_thumbnail(&project, &readers, &encoders, &*fonts, renderer.as_ref());
                    if let Err(error) = drawn {
                        cutlass_logger::warn!("project picture for {}: {error}", project.display());
                    }
                    if send.send(project).is_err() {
                        return;
                    }
                }
            });
        if let Err(error) = spawned {
            cutlass_logger::warn!("project pictures did not start: {error}");
            return None;
        }
        Some(ProjectPictures {
            ask,
            landed,
            asked: HashMap::new(),
            waiting: 0,
        })
    }
}

/// Draw the first frame of the project at `project` and write it as its
/// card picture. A timeline with nothing at its start draws black there, as
/// the compositor does; a timeline with nothing on it at all is black too.
fn draw_thumbnail(
    project: &Path,
    readers: &cutlass_eval::ReaderFactory,
    encoders: &cutlass_encode::EncodeFactory,
    fonts: &dyn cutlass_eval::Fonts,
    renderer: Option<&cutlass_render::Renderer>,
) -> Result<(), ShellError> {
    let file = project.join(THUMBNAIL_FILE);
    let draft = cutlass_store::Folder::read(project)
        .map_err(ShellError::Store)?
        .draft;
    if draft.duration.0 <= 0 {
        let picture = black(&draft)
            .ok_or_else(|| ShellError::Unavailable("the canvas has no size".to_owned()))?;
        return write_replaced(&file, &jpeg(&picture)?);
    }
    let (width, height, rgba) = cutlass_export::still_picture(
        &cutlass_export::MovieSources {
            draft: &draft,
            root: Some(project),
            readers,
            fonts,
            renderer,
            encoders,
        },
        Micros(0),
    )
    .map_err(export_failure)?;
    let picture = on_black(width, height, rgba, THUMBNAIL_ROWS)
        .ok_or_else(|| ShellError::Unavailable("the first frame is empty".to_owned()))?;
    write_replaced(&file, &jpeg(&picture)?)
}

/// A black picture in the canvas's shape, at most [`THUMBNAIL_ROWS`] rows.
fn black(draft: &cutlass_model::Draft) -> Option<Thumb> {
    let width = u64::try_from(draft.canvas_config.width).ok()?;
    let height = u64::try_from(draft.canvas_config.height).ok()?;
    if width == 0 || height == 0 {
        return None;
    }
    let rows = height.min(u64::from(THUMBNAIL_ROWS));
    let columns = (width * rows).div_ceil(height).max(1);
    Some(Thumb {
        width: u32::try_from(columns).ok()?,
        height: u32::try_from(rows).ok()?,
        rgba: [0, 0, 0, 255].repeat(usize::try_from(columns * rows).ok()?),
    })
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

#[cfg(test)]
mod tests;
