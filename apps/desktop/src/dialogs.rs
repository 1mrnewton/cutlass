//! Native file dialogs.
//!
//! The blocking picker runs its own event loop. Opening it from the window
//! event makes the macOS backend try to handle a second event and abort.
//! These dialogs are sheets: the click only starts one, and a later frame
//! reads the path.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Picker {
    OpenProject,
    Import,
    SaveExport,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Finished {
    pub(crate) picker: Picker,
    pub(crate) path: Option<PathBuf>,
}

struct Pending {
    picker: Picker,
    future: Pin<Box<dyn Future<Output = Option<PathBuf>>>>,
}

pub(crate) struct Dialogs {
    pending: Option<Pending>,
}

impl Dialogs {
    pub(crate) fn new() -> Self {
        Self { pending: None }
    }

    pub(crate) fn open_project(&mut self, parent: Option<&eframe::Frame>) {
        let dialog = dialog(parent);
        self.start(Picker::OpenProject, async move {
            dialog
                .pick_folder()
                .await
                .map(|file| file.path().to_path_buf())
        });
    }

    pub(crate) fn import(&mut self, parent: Option<&eframe::Frame>) {
        let dialog = dialog(parent);
        self.start(Picker::Import, async move {
            dialog
                .pick_file()
                .await
                .map(|file| file.path().to_path_buf())
        });
    }

    /// Ask where to write the export, offering `name` with `extension`. A
    /// chosen name without the extension gets it.
    pub(crate) fn save_export(
        &mut self,
        parent: Option<&eframe::Frame>,
        name: &str,
        extension: &str,
    ) {
        let stem = if name.trim().is_empty() {
            "Export"
        } else {
            name.trim()
        };
        let dialog = dialog(parent)
            .set_file_name(format!("{stem}.{extension}"))
            .add_filter(extension.to_uppercase(), &[extension]);
        let extension = extension.to_owned();
        self.start(Picker::SaveExport, async move {
            dialog
                .save_file()
                .await
                .map(|file| with_extension(file.path(), &extension))
        });
    }

    pub(crate) fn poll(&mut self, ctx: &egui::Context) -> Option<Finished> {
        let mut pending = self.pending.take()?;
        let waker = std::task::Waker::noop();
        let mut task = Context::from_waker(waker);
        match pending.future.as_mut().poll(&mut task) {
            Poll::Pending => {
                ctx.request_repaint_after(Duration::from_millis(100));
                self.pending = Some(pending);
                None
            }
            Poll::Ready(path) => Some(Finished {
                picker: pending.picker,
                path,
            }),
        }
    }

    fn start(&mut self, picker: Picker, future: impl Future<Output = Option<PathBuf>> + 'static) {
        if self.pending.is_some() {
            return;
        }
        self.pending = Some(Pending {
            picker,
            future: Box::pin(future),
        });
    }

    #[cfg(test)]
    pub(crate) fn finish_for_test(&mut self, picker: Picker, path: Option<PathBuf>) {
        self.start(picker, std::future::ready(path));
    }
}

/// `path`, ending in `.extension` whatever case it was typed in.
fn with_extension(path: &std::path::Path, extension: &str) -> PathBuf {
    let has = path
        .extension()
        .is_some_and(|typed| typed.eq_ignore_ascii_case(extension));
    if has {
        path.to_path_buf()
    } else {
        let mut named = path.as_os_str().to_owned();
        named.push(".");
        named.push(extension);
        named.into()
    }
}

fn dialog(parent: Option<&eframe::Frame>) -> rfd::AsyncFileDialog {
    match parent {
        Some(parent) => rfd::AsyncFileDialog::new().set_parent(parent),
        None => rfd::AsyncFileDialog::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chosen_export_name_gets_its_extension_once() {
        assert_eq!(
            with_extension(std::path::Path::new("/films/cut"), "mp4"),
            PathBuf::from("/films/cut.mp4")
        );
        assert_eq!(
            with_extension(std::path::Path::new("/films/cut.MP4"), "mp4"),
            PathBuf::from("/films/cut.MP4")
        );
        assert_eq!(
            with_extension(std::path::Path::new("/films/cut.v2"), "mp4"),
            PathBuf::from("/films/cut.v2.mp4")
        );
    }

    #[test]
    fn a_ready_picker_returns_its_path_once() {
        let mut dialogs = Dialogs::new();
        let ctx = egui::Context::default();
        dialogs.finish_for_test(Picker::Import, Some(PathBuf::from("clip.png")));
        dialogs.finish_for_test(Picker::OpenProject, Some(PathBuf::from("other")));
        let finished = dialogs.poll(&ctx).expect("ready");
        assert_eq!(finished.picker, Picker::Import);
        assert_eq!(finished.path, Some(PathBuf::from("clip.png")));
        assert!(dialogs.poll(&ctx).is_none());
    }

    #[test]
    fn a_cancelled_picker_returns_no_path() {
        let mut dialogs = Dialogs::new();
        dialogs.finish_for_test(Picker::Import, None);
        let finished = dialogs.poll(&egui::Context::default()).expect("ready");
        assert_eq!(finished.path, None);
    }
}
