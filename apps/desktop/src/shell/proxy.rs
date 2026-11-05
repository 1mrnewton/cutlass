//! Proxies: smaller copies of 4K and long movies in the Cutlass `Proxies`
//! folder, which the preview plays in place of the source.
//!
//! Only the preview's pictures read a proxy. Probing, thumbnails, sound
//! and export read the source, so an export is drawn from the files the
//! project names. A proxy is named by its source as it is on disk; a
//! changed source is shown from itself until a new proxy is made.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cutlass_engine::ReaderFactory;
use cutlass_model::{Draft, Micros};

use super::{Shell, ShellError, export_failure};

/// The proxies kept for the open project and the readers that play them.
pub(crate) struct Proxies {
    folder: PathBuf,
    /// Each source, by its canonical path, and its proxy.
    made: HashMap<PathBuf, PathBuf>,
    /// The shell's readers opening the proxies, once there is one.
    readers: Option<ReaderFactory>,
}

impl Proxies {
    pub(crate) fn new(folder: PathBuf) -> Self {
        Self {
            folder,
            made: HashMap::new(),
            readers: None,
        }
    }

    /// `readers` opening the proxies, when there are any and `on`.
    pub(crate) fn preview<'a>(&'a self, readers: &'a ReaderFactory, on: bool) -> &'a ReaderFactory {
        match self.readers.as_ref() {
            Some(proxied) if on => proxied,
            _ => readers,
        }
    }
}

/// One movie to draw a proxy of.
struct Wanted {
    source: PathBuf,
    proxy: PathBuf,
    width: u32,
    height: u32,
    duration: Micros,
}

/// The movie files `draft` uses, as paths on disk.
fn movie_files(draft: &Draft, root: Option<&Path>) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = draft
        .materials
        .videos
        .iter()
        .filter(|video| video.kind == "video")
        .filter_map(|video| {
            let stored = Path::new(&video.path);
            if stored.is_absolute() {
                Some(stored.to_path_buf())
            } else {
                cutlass_engine::media_in_project(root?, &video.path)
            }
        })
        .filter_map(|path| path.canonicalize().ok())
        .collect();
    files.sort();
    files.dedup();
    files
}

impl Shell {
    /// The readers the preview's pictures open: through the proxies when
    /// there are any and the setting plays them.
    pub(crate) fn preview_readers(&self) -> &ReaderFactory {
        self.proxies
            .preview(&self.readers, self.settings.playback.proxies)
    }

    /// The sources that have a proxy now.
    #[must_use]
    pub fn proxied(&self) -> Vec<PathBuf> {
        let mut sources: Vec<_> = self.proxies.made.keys().cloned().collect();
        sources.sort();
        sources
    }

    /// Look in the proxies folder for proxies of the project's movies.
    pub(super) fn find_proxies(&mut self) {
        self.proxies.made.clear();
        if let Some(draft) = self.draft() {
            for source in movie_files(draft, self.path.as_deref()) {
                let Some(name) = cutlass_export::proxy_name(&source) else {
                    continue;
                };
                let proxy = self.proxies.folder.join(name);
                if proxy.is_file() {
                    self.proxies.made.insert(source, proxy);
                }
            }
        }
        self.proxies_changed();
    }

    fn proxies_changed(&mut self) {
        self.proxies.readers = (!self.proxies.made.is_empty())
            .then(|| self.readers.with_stand_ins(self.proxies.made.clone()));
        self.preview_cache.borrow_mut().clear();
    }

    /// The project's movies that want a proxy and have none: 4K and long
    /// ones, and the selected clip's movie whatever its size.
    fn wanted_proxies(&self) -> Vec<Wanted> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        let root = self.path.as_deref();
        let chosen: Vec<PathBuf> = self
            .selected_segment()
            .and_then(|segment| {
                let video = draft
                    .materials
                    .videos
                    .iter()
                    .find(|video| video.id == segment.material_id)?;
                let mut single = draft.clone();
                single.materials.videos = vec![video.clone()];
                Some(movie_files(&single, root))
            })
            .unwrap_or_default();
        movie_files(draft, root)
            .into_iter()
            .filter(|source| !self.proxies.made.contains_key(source))
            .filter_map(|source| {
                let info = self.readers.probe(&source).ok()?;
                let (width, height) = (info.width?, info.height?);
                let duration = info.duration?;
                // A turned picture would be turned again over its proxy.
                if info.display != cutlass_container::DisplayQuarter::Zero {
                    return None;
                }
                let wanted = chosen.contains(&source)
                    || cutlass_export::wants_proxy(width, height, duration);
                let name = cutlass_export::proxy_name(&source)?;
                wanted.then(|| Wanted {
                    proxy: self.proxies.folder.join(name),
                    source,
                    width,
                    height,
                    duration,
                })
            })
            .collect()
    }

    /// Draw proxies of the project's 4K and long movies, and of the
    /// selected clip's, as a job. A proxy already in the folder is taken
    /// at once.
    pub fn make_proxies(&mut self) -> Result<(), ShellError> {
        let fps = self.session_draft()?.fps;
        let mut wanted = self.wanted_proxies();
        let before = self.proxies.made.len();
        wanted.retain(|movie| {
            if movie.proxy.is_file() {
                self.proxies
                    .made
                    .insert(movie.source.clone(), movie.proxy.clone());
                false
            } else {
                true
            }
        });
        if self.proxies.made.len() != before {
            self.proxies_changed();
        }
        if wanted.is_empty() {
            self.set_notice("Every movie that needs a proxy has one".to_owned());
            return Ok(());
        }
        std::fs::create_dir_all(&self.proxies.folder)
            .map_err(|error| ShellError::Io(error.to_string()))?;
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let fonts = Arc::clone(&self.face);
        self.start_job("Proxies", move |progress| {
            let count = wanted.len();
            let mut made = Vec::new();
            let mut failed = None;
            for (index, movie) in wanted.into_iter().enumerate() {
                if progress.cancelled() {
                    break;
                }
                progress.stage(&format!("Proxy {} of {count}", index + 1));
                let draft = cutlass_export::proxy_draft(
                    &movie.source,
                    movie.width,
                    movie.height,
                    movie.duration,
                    fps,
                );
                let request = cutlass_export::proxy_request(&draft);
                let mut report = |done: u64, total: u64| {
                    if total > 0 {
                        progress.fraction(done as f32 / total as f32);
                    }
                    !progress.cancelled()
                };
                let sources = cutlass_export::MovieSources {
                    draft: &draft,
                    root: None,
                    readers: &readers,
                    fonts: &*fonts,
                    renderer: None,
                    encoders: &encoders,
                };
                // Written aside first, so a stopped proxy is never found.
                let partial = movie.proxy.with_extension("part");
                let written =
                    cutlass_export::write_movie(&sources, &request, &partial, &mut report)
                        .map_err(export_failure)
                        .and_then(|_| {
                            std::fs::rename(&partial, &movie.proxy)
                                .map_err(|error| ShellError::Io(error.to_string()))
                        });
                match written {
                    Ok(()) => made.push((movie.source, movie.proxy)),
                    Err(error) => {
                        let _ = std::fs::remove_file(&partial);
                        failed = Some(error);
                        break;
                    }
                }
            }
            Box::new(move |shell: &mut Shell| {
                let landed = made.len();
                shell.proxies.made.extend(made);
                shell.proxies_changed();
                if let Some(error) = failed {
                    return Err(error);
                }
                shell.set_notice(match landed {
                    1 => "Made 1 proxy".to_owned(),
                    n => format!("Made {n} proxies"),
                });
                Ok(())
            })
        })
    }

    #[cfg(test)]
    pub(crate) fn set_proxies_folder(&mut self, folder: PathBuf) {
        self.proxies = Proxies::new(folder);
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialVideo, Micros, Segment, SegmentId, TimeRange};
    use cutlass_store::Folder;

    use crate::shell::tests::{Temp, land_job, shell};

    /// A one-second 960 by 480 movie of the engine's pattern picture.
    fn source_movie(path: &std::path::Path) {
        let pattern = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        let mut draft = cutlass_export::proxy_draft(&pattern, 64, 32, Micros(1_000_000), 30.0);
        draft.materials.videos[0].kind = "photo".to_owned();
        let request = cutlass_export::proxy_request(&draft);
        let (decoders, encoders) = crate::register_codecs();
        let readers = crate::readers::readers_with(decoders);
        let fonts = crate::face::FaceTable::default();
        let sources = cutlass_export::MovieSources {
            draft: &draft,
            root: None,
            readers: &readers,
            fonts: &fonts,
            renderer: None,
            encoders: &encoders,
        };
        cutlass_export::write_movie(&sources, &request, path, &mut |_, _| true).unwrap();
    }

    #[test]
    fn a_proxy_plays_in_the_preview_and_never_reaches_export() {
        let temp = Temp::new();
        let movie = temp.path.join("source.mp4");
        source_movie(&movie);
        let project = temp.path.join("project");
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 64;
        draft.canvas_config.height = 32;
        let mut video = MaterialVideo::new("movie");
        video.kind = "video".to_owned();
        video.path = movie.to_string_lossy().into_owned();
        video.width = 960;
        video.height = 480;
        video.duration = Micros(1_000_000);
        draft.materials.videos.push(video);
        let mut clip = Segment::new("clip", "movie", TimeRange::new(0, 1_000_000));
        clip.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(clip);
        draft.duration = Micros(1_000_000);
        Folder::new(draft).save(&project).unwrap();

        let proxies = temp.path.join("proxies");
        let mut desktop = shell();
        desktop.set_proxies_folder(proxies.clone());
        desktop.open_folder(&project).unwrap();
        assert!(desktop.proxied().is_empty());
        let source = movie.canonicalize().unwrap();

        // A small movie gets no proxy unless its clip is chosen.
        desktop.make_proxies().unwrap();
        assert!(desktop.job().is_none());
        assert!(desktop.proxied().is_empty());
        desktop.select(Some(SegmentId::new("clip")));
        desktop.make_proxies().unwrap();
        assert_eq!(
            desktop.job().map(|job| job.label),
            Some("Proxies".to_owned())
        );
        land_job(&mut desktop);
        assert_eq!(desktop.notice(), "Made 1 proxy");
        assert_eq!(desktop.proxied(), vec![source.clone()]);
        let files: Vec<_> = std::fs::read_dir(&proxies)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(files.len(), 1, "{files:?}");
        let proxy = files[0].clone();
        assert_eq!(proxy.extension().unwrap(), "mp4");

        // The preview opens the proxy; export's readers open the source.
        let size = |readers: &cutlass_engine::ReaderFactory| {
            readers.open(&source).unwrap().info().byte_size
        };
        let proxy_size = std::fs::metadata(&proxy).unwrap().len();
        let source_size = std::fs::metadata(&source).unwrap().len();
        assert_ne!(proxy_size, source_size, "the proxy is its own coding");
        assert_eq!(size(desktop.preview_readers()), proxy_size);
        assert_eq!(size(&desktop.readers), source_size);
        desktop.seek(Micros(500_000));
        let shown = desktop.preview_bytes().unwrap();
        assert_eq!((shown.width, shown.height), (64, 32));

        // Off in Settings, the preview reads the source again.
        desktop.update_settings(|settings| settings.playback.proxies = false);
        assert_eq!(size(desktop.preview_readers()), source_size);
        let live = desktop.preview_bytes().unwrap();
        let apart: u64 = shown
            .rgba
            .iter()
            .zip(&live.rgba)
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum();
        assert!(apart / (live.rgba.len() as u64) < 12, "{apart}");
        desktop.update_settings(|settings| settings.playback.proxies = true);

        // A later run finds it; a changed source leaves it behind.
        drop(desktop);
        let mut again = shell();
        again.set_proxies_folder(proxies.clone());
        again.open_folder(&project).unwrap();
        assert_eq!(again.proxied(), vec![source.clone()]);
        again.select(Some(SegmentId::new("clip")));
        again.make_proxies().unwrap();
        assert!(again.job().is_none(), "a kept proxy is taken at once");
        drop(again);
        let bytes = std::fs::read(&source).unwrap();
        std::fs::write(&source, [bytes.as_slice(), b"\0"].concat()).unwrap();
        let mut changed = shell();
        changed.set_proxies_folder(proxies);
        changed.open_folder(&project).unwrap();
        assert!(changed.proxied().is_empty());
    }
}
