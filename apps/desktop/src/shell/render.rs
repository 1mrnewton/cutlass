//! Render: draw the selected clips' stretch of the timeline ahead into a
//! movie, so the preview plays that one file there.
//!
//! Preview renders at a small size, Full at the canvas's own. A render
//! stands for the project as it was drawn: an edit leaves it behind, and
//! an undo back to the same project finds it again, as do later runs while
//! the file stays in the cache. The sound still comes from the timeline,
//! and export never reads a render.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cutlass_model::{Micros, TimeRange};
use cutlass_playback::FrameCache;
use cutlass_session::Session;

use super::{ExportSpan, Shell, ShellError, export_failure};

/// One render the preview can play.
pub(crate) struct Rendered {
    key: u64,
    range: TimeRange,
    side: u32,
    pub(crate) session: Session,
}

/// The renders and what playing them needs.
pub(crate) struct Renders {
    folder: PathBuf,
    list: Vec<Rendered>,
    /// Pictures of the renders, apart from the timeline's.
    pub(crate) cache: RefCell<FrameCache>,
    /// The key of the draft last asked about: its id, revision and key.
    memo: RefCell<Option<(String, u64, u64)>>,
}

impl Renders {
    pub(crate) fn new(folder: PathBuf) -> Self {
        Self {
            folder,
            list: Vec::new(),
            cache: RefCell::new(FrameCache::new()),
            memo: RefCell::new(None),
        }
    }

    /// The render of the draft keyed `key` the preview plays at `time`.
    pub(crate) fn at(&self, key: u64, time: Micros) -> Option<&Rendered> {
        self.list
            .iter()
            .filter(|render| render.key == key)
            .filter(|render| {
                let start = render.range.start.0;
                (start..start + render.range.duration.0).contains(&time.0)
            })
            .filter(|render| {
                render
                    .session
                    .draft()
                    .materials
                    .videos
                    .first()
                    .is_some_and(|movie| Path::new(&movie.path).is_file())
            })
            .max_by_key(|render| render.side)
    }

    /// The renders kept for a draft with `key`, read from the folder.
    fn found(&self, key: u64, draft: &cutlass_model::Draft) -> Vec<Rendered> {
        let Ok(entries) = std::fs::read_dir(&self.folder) else {
            return Vec::new();
        };
        entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                let (found, range, side) = cutlass_export::read_render_name(&name)?;
                (found == key).then(|| rendered(draft, &entry.path(), key, range, side))
            })
            .collect()
    }
}

fn rendered(
    draft: &cutlass_model::Draft,
    path: &Path,
    key: u64,
    range: TimeRange,
    side: u32,
) -> Rendered {
    Rendered {
        key,
        session: Session::open(cutlass_export::stand_in(draft, path, &range, key)),
        range,
        side,
    }
}

impl Shell {
    /// The key of the project as the window shows it now.
    fn render_key_now(&self) -> Option<u64> {
        let session = self.session.as_ref()?;
        let draft = session.draft();
        let id = draft.id.to_string();
        let revision = session.revision();
        let mut memo = self.renders.memo.borrow_mut();
        if let Some((seen, at, key)) = memo.as_ref()
            && *seen == id
            && *at == revision
        {
            return Some(*key);
        }
        let key = cutlass_export::render_key(draft);
        *memo = Some((id, revision, key));
        Some(key)
    }

    /// The render the preview plays at `time`, when one stands for the
    /// project as it is and its file is still there. A full render wins.
    pub(crate) fn rendered_at(&self, time: Micros) -> Option<&Rendered> {
        if self.renders.list.is_empty() {
            return None;
        }
        self.renders.at(self.render_key_now()?, time)
    }

    /// The key [`Renders::at`] takes, without asking when there is no render.
    pub(crate) fn render_key_if_any(&self) -> Option<u64> {
        if self.renders.list.is_empty() {
            None
        } else {
            self.render_key_now()
        }
    }

    /// The stretches of the timeline a render stands for now, start and
    /// end, for the ruler.
    #[must_use]
    pub fn rendered_spans(&self) -> Vec<(Micros, Micros)> {
        let Some(key) = self.render_key_now() else {
            return Vec::new();
        };
        self.renders
            .list
            .iter()
            .filter(|render| render.key == key)
            .map(|render| {
                let start = render.range.start;
                (start, Micros(start.0 + render.range.duration.0))
            })
            .collect()
    }

    /// Look in the cache for renders of the project as it opened.
    pub(super) fn find_renders(&mut self) {
        self.renders.list.clear();
        self.renders.cache.borrow_mut().clear();
        let (Some(key), Some(draft)) = (self.render_key_now(), self.draft()) else {
            return;
        };
        let found = self.renders.found(key, draft);
        self.renders.list = found;
    }

    /// Render the selected clips' stretch: at a small size for `full`
    /// false, at the canvas's own for true. It runs as a job; a render
    /// already in the cache is taken at once.
    pub fn render_selected(&mut self, full: bool) -> Result<(), ShellError> {
        let draft = self.session_draft()?.clone();
        let range = self.export_range(&draft, ExportSpan::Selected)?;
        let canvas = (draft.canvas_config.width, draft.canvas_config.height);
        let side = cutlass_export::render_side(canvas.0, canvas.1, full);
        let key = self.render_key_now().ok_or(ShellError::NoProject)?;
        let path = self
            .renders
            .folder
            .join(cutlass_export::render_name(key, &range, side));
        if path.is_file() {
            self.keep_render(rendered(&draft, &path, key, range, side));
            self.set_notice("Rendered".to_owned());
            return Ok(());
        }
        let request = cutlass_export::render_request(&draft, range.clone(), side);
        cutlass_export::validate(&request).map_err(export_failure)?;
        let (width, height) =
            cutlass_export::output_size(canvas.0, canvas.1, side).map_err(export_failure)?;
        let params = cutlass_decode::CodecParams::with_format(
            request.codec,
            width,
            height,
            cutlass_decode::PixelFormat::Rgba8,
        );
        if let Err(error) = self.encoders.open(&params) {
            return Err(ShellError::CodecDeclined(error.to_string()));
        }
        std::fs::create_dir_all(&self.renders.folder)
            .map_err(|error| ShellError::Io(error.to_string()))?;
        let root = self.path.clone();
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let fonts = Arc::clone(&self.face);
        let gpu = self.gpu.as_ref().map(|preview| {
            let renderer = preview.renderer();
            (renderer.device().clone(), renderer.queue().clone())
        });
        self.start_job("Render", move |progress| {
            let renderer = gpu.map(|(device, queue)| cutlass_render::Renderer::new(device, queue));
            progress.stage("Rendering");
            let mut report = |done: u64, total: u64| {
                if total > 0 {
                    progress.fraction(done as f32 / total as f32);
                }
                !progress.cancelled()
            };
            let sources = cutlass_export::MovieSources {
                draft: &draft,
                root: root.as_deref(),
                readers: &readers,
                fonts: &*fonts,
                renderer: renderer.as_ref(),
                encoders: &encoders,
            };
            let written = cutlass_export::write_movie(&sources, &request, &path, &mut report);
            Box::new(move |shell: &mut Shell| {
                written.map_err(export_failure)?;
                shell.keep_render(rendered(&draft, &path, key, range, side));
                shell.set_notice("Rendered".to_owned());
                Ok(())
            })
        })
    }

    /// Keep `render`, in place of one of the same stretch and size.
    fn keep_render(&mut self, render: Rendered) {
        self.renders.list.retain(|kept| {
            !(kept.key == render.key && kept.range == render.range && kept.side == render.side)
        });
        self.renders.list.push(render);
        self.renders.cache.borrow_mut().clear();
    }

    #[cfg(test)]
    pub(crate) fn set_renders_folder(&mut self, folder: PathBuf) {
        self.renders = Renders::new(folder);
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Micros, SegmentId};

    use crate::shell::tests::{Temp, land_job, movie_project, shell};

    #[test]
    fn a_render_plays_in_the_preview_until_an_edit_and_comes_back_on_undo() {
        let temp = Temp::new();
        movie_project(&temp.path);
        let renders = temp.path.join("renders");
        let mut desktop = shell();
        desktop.set_renders_folder(renders.clone());
        desktop.open_folder(&temp.path).unwrap();
        assert!(
            desktop.render_selected(false).is_err(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("two")));
        desktop.render_selected(false).unwrap();
        assert_eq!(
            desktop.job().map(|job| job.label),
            Some("Render".to_owned())
        );
        land_job(&mut desktop);
        assert_eq!(desktop.notice(), "Rendered");
        let files: Vec<_> = std::fs::read_dir(&renders).unwrap().collect();
        assert_eq!(files.len(), 1);
        let spans = desktop.rendered_spans();
        assert_eq!(spans.len(), 1);
        let (start, end) = spans[0];
        let inside = Micros((start.0 + end.0) / 2);
        assert!(desktop.rendered_at(inside).is_some());
        assert!(desktop.rendered_at(Micros(end.0 + 1)).is_none());
        desktop.seek(inside);
        let shown = desktop.preview_bytes().unwrap();
        // Off the render, the same moment drawn live looks the same.
        let path = desktop
            .rendered_at(inside)
            .unwrap()
            .session
            .draft()
            .materials
            .videos[0]
            .path
            .clone();
        let kept = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(desktop.rendered_at(inside).is_none());
        let live = desktop.preview_bytes().unwrap();
        std::fs::write(&path, kept).unwrap();
        assert_eq!((shown.width, shown.height), (live.width, live.height));
        let apart: u64 = shown
            .rgba
            .iter()
            .zip(&live.rgba)
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum();
        assert!(apart / (live.rgba.len() as u64) < 12, "{apart}");
        assert_ne!(shown.rgba, live.rgba, "the render is a coded movie");

        // An edit leaves it behind; undo finds it again.
        desktop.select(Some(SegmentId::new("two")));
        desktop.remove_selected().unwrap();
        assert!(desktop.rendered_spans().is_empty());
        desktop.undo().unwrap();
        assert_eq!(desktop.rendered_spans().len(), 1);

        // A later run finds it in the cache.
        drop(desktop);
        let mut again = shell();
        again.set_renders_folder(renders);
        again.open_folder(&temp.path).unwrap();
        assert_eq!(again.rendered_spans(), spans);
        again.select(Some(SegmentId::new("two")));
        again.render_selected(false).unwrap();
        assert!(again.job().is_none(), "a kept render is taken at once");
    }
}
