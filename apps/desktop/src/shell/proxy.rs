//! Proxies: smaller copies of movies in the Cutlass `Proxies` folder, which
//! the preview plays in place of the source.
//!
//! A project that makes proxies draws one for each 4K or long movie as it
//! comes in, one after another on a thread of their own, so importing,
//! editing and export go on meanwhile. A file chosen by hand in the
//! library has a proxy, or has none, whatever the project makes by itself.
//!
//! Only the preview's pictures read a proxy. Probing, thumbnails, sound
//! and export read the source, so an export is drawn from the files the
//! project names. A proxy is named by its source as it is on disk; a
//! changed source is shown from itself until its new proxy is made.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cutlass_engine::ReaderFactory;
use cutlass_model::Draft;

use super::{Shell, ShellError, export_failure};
use crate::jobs::{Job, Poll, Progress};

/// How a file in the library stands with its proxy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProxyState {
    /// It has no proxy and none is on the way.
    None,
    /// Its proxy waits its turn or is being drawn; how far along, once the
    /// drawing has begun.
    Making(Option<f32>),
    /// The preview plays its proxy.
    Made,
}

/// One movie file of the project.
struct Movie {
    /// The path its material stores, which the project's choices name.
    stored: String,
    /// The file on disk.
    file: PathBuf,
    /// 4K or long: a project that makes proxies makes this one.
    heavy: bool,
    /// Its short side is above a proxy's, so a proxy is smaller than it.
    larger: bool,
}

/// One proxy to draw.
#[derive(Clone)]
struct Wanted {
    source: PathBuf,
    proxy: PathBuf,
}

/// The proxy being drawn.
struct Running {
    source: PathBuf,
    job: Job<Result<(), ShellError>>,
}

/// The proxies the open project plays, and the ones on the way.
pub(crate) struct Proxies {
    folder: PathBuf,
    /// Each source the preview plays a proxy of, by its canonical path.
    made: HashMap<PathBuf, PathBuf>,
    /// The shell's readers opening the proxies, once there is one.
    readers: Option<ReaderFactory>,
    waiting: VecDeque<Wanted>,
    running: Option<Running>,
    /// Stopped proxies whose threads have not ended yet. The next one
    /// waits for them, so two never write the same file.
    stopping: Vec<Job<Result<(), ShellError>>>,
    /// Sources whose proxy could not be drawn, so they are not tried over
    /// and over. Asking for one by hand tries again.
    failed: HashSet<PathBuf>,
    movies: Vec<Movie>,
    /// Each movie material's file.
    by_material: HashMap<String, PathBuf>,
    /// The session revision and the movie materials last looked at.
    seen: Option<(u64, u64)>,
    /// How many times the project's movies were looked through.
    looks: u64,
}

/// Where proxies are kept: the Cutlass `Proxies` folder, and a folder of
/// their own for tests.
pub(crate) fn home_folder() -> PathBuf {
    if cfg!(test) {
        std::env::temp_dir().join(format!("cutlass-test-proxies-{}", std::process::id()))
    } else {
        Shell::home().proxies()
    }
}

impl Drop for Proxies {
    fn drop(&mut self) {
        self.stop();
        for job in &self.stopping {
            job.cancel();
        }
    }
}

impl Proxies {
    pub(crate) fn new(folder: PathBuf) -> Self {
        Self {
            folder,
            made: HashMap::new(),
            readers: None,
            waiting: VecDeque::new(),
            running: None,
            stopping: Vec::new(),
            failed: HashSet::new(),
            movies: Vec::new(),
            by_material: HashMap::new(),
            seen: None,
            looks: 0,
        }
    }

    /// `readers` opening the proxies, when there are any.
    pub(crate) fn preview<'a>(&'a self, readers: &'a ReaderFactory) -> &'a ReaderFactory {
        self.readers.as_ref().unwrap_or(readers)
    }

    /// Stop the proxy being drawn and forget the ones waiting.
    fn stop(&mut self) {
        self.waiting.clear();
        if let Some(running) = self.running.take() {
            running.job.cancel();
            self.stopping.push(running.job);
        }
    }

    fn movie_of(&self, material_id: &str) -> Option<&Movie> {
        let file = self.by_material.get(material_id)?;
        self.movies.iter().find(|movie| &movie.file == file)
    }

    fn state_of(&self, file: &Path) -> ProxyState {
        if self.made.contains_key(file) {
            return ProxyState::Made;
        }
        if let Some(running) = self.running.as_ref()
            && running.source == file
        {
            return ProxyState::Making(running.job.view().fraction);
        }
        if self.waiting.iter().any(|wanted| wanted.source == file) {
            return ProxyState::Making(None);
        }
        ProxyState::None
    }
}

fn side(edge: i32) -> u32 {
    u32::try_from(edge).unwrap_or(0)
}

/// The movie files `draft` uses, each once, and the file of every movie
/// material.
fn movies(draft: &Draft, root: Option<&Path>) -> (Vec<Movie>, HashMap<String, PathBuf>) {
    let mut found: Vec<Movie> = Vec::new();
    let mut by_material = HashMap::new();
    for video in &draft.materials.videos {
        if video.kind != "video" {
            continue;
        }
        let stored = Path::new(&video.path);
        let located = if stored.is_absolute() {
            Some(stored.to_path_buf())
        } else {
            root.and_then(|root| cutlass_engine::media_in_project(root, &video.path))
        };
        let Some(file) = located.and_then(|path| path.canonicalize().ok()) else {
            continue;
        };
        by_material.insert(video.id.as_str().to_owned(), file.clone());
        if found.iter().any(|movie| movie.file == file) {
            continue;
        }
        let (width, height) = (side(video.width), side(video.height));
        found.push(Movie {
            stored: video.path.clone(),
            file,
            heavy: cutlass_export::wants_proxy(width, height, video.duration),
            larger: width.min(height) > cutlass_export::PROXY_SIDE,
        });
    }
    (found, by_material)
}

/// What says which movies `draft` has: a change here is a movie that came
/// or went, or one whose file changed.
fn movies_mark(draft: &Draft) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for video in &draft.materials.videos {
        if video.kind == "video" {
            video.id.as_str().hash(&mut hasher);
            video.path.hash(&mut hasher);
            (video.width, video.height, video.duration.0).hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Draw the proxy of `wanted.source` at the project's `fps`. It is written
/// aside first, so a stopped proxy is never found.
fn draw_proxy(
    wanted: &Wanted,
    fps: f64,
    readers: &ReaderFactory,
    encoders: &cutlass_encode::EncodeFactory,
    fonts: &crate::face::FaceTable,
    progress: &Progress,
) -> Result<(), ShellError> {
    let info = readers
        .probe(&wanted.source)
        .map_err(|error| ShellError::Probe(error.to_string()))?;
    let (Some(width), Some(height), Some(duration)) = (info.width, info.height, info.duration)
    else {
        return Err(ShellError::Unavailable(
            "the movie's size or length could not be read".to_owned(),
        ));
    };
    // A turned picture would be turned again over its proxy.
    if info.display != cutlass_container::DisplayQuarter::Zero {
        return Err(ShellError::Unavailable(
            "a movie stored on its side has no proxy yet".to_owned(),
        ));
    }
    let draft = cutlass_export::proxy_draft(&wanted.source, width, height, duration, fps);
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
        readers,
        fonts,
        renderer: None,
        encoders,
    };
    let partial = wanted.proxy.with_extension("part");
    let written = cutlass_export::write_movie(&sources, &request, &partial, &mut report)
        .map_err(export_failure)
        .and_then(|_| {
            if progress.cancelled() {
                return Err(ShellError::Unavailable("the proxy was stopped".to_owned()));
            }
            std::fs::rename(&partial, &wanted.proxy)
                .map_err(|error| ShellError::Io(error.to_string()))
        });
    if written.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    written
}

impl Shell {
    /// The readers the preview's pictures open: through the proxies when
    /// there are any.
    pub(crate) fn preview_readers(&self) -> &ReaderFactory {
        self.proxies.preview(&self.readers)
    }

    /// The sources the preview plays a proxy of.
    #[must_use]
    pub fn proxied(&self) -> Vec<PathBuf> {
        let mut sources: Vec<_> = self.proxies.made.keys().cloned().collect();
        sources.sort();
        sources
    }

    /// Whether the open project makes proxies of its 4K and long movies
    /// by itself.
    #[must_use]
    pub fn project_proxies(&self) -> bool {
        self.folder
            .as_ref()
            .and_then(|folder| folder.proxies().enabled)
            .unwrap_or(self.settings.playback.proxies)
    }

    /// Have the open project make proxies by itself, or stop. A file
    /// chosen by hand keeps its choice. This is not an undo step.
    pub fn set_project_proxies(&mut self, on: bool) -> Result<(), ShellError> {
        self.choose_proxies(|choices| choices.enabled = Some(on))?;
        self.sync_proxies();
        Ok(())
    }

    /// Change the project's proxy choices. The project then has unsaved
    /// changes.
    fn choose_proxies(
        &mut self,
        change: impl FnOnce(&mut cutlass_store::ProjectProxies),
    ) -> Result<(), ShellError> {
        let folder = self.folder.as_mut().ok_or(ShellError::NoProject)?;
        let before = folder.proxies();
        let mut after = before.clone();
        change(&mut after);
        if after != before {
            folder.set_proxies(&after);
            self.library_edits = self.library_edits.wrapping_add(1);
        }
        Ok(())
    }

    /// Start over for the project that just opened. One that has never
    /// said whether it makes proxies takes the app's setting from here on,
    /// so a later change to the setting is for new projects.
    pub(super) fn open_proxies(&mut self) {
        self.close_proxies();
        let default = self.settings.playback.proxies;
        if let Some(folder) = self.folder.as_mut() {
            let mut choices = folder.proxies();
            if choices.enabled.is_none() {
                choices.enabled = Some(default);
                folder.set_proxies(&choices);
            }
        }
        self.sync_proxies();
    }

    /// Stop making proxies and forget the ones the project played.
    pub(super) fn close_proxies(&mut self) {
        self.proxies.stop();
        self.proxies.failed.clear();
        self.proxies.movies.clear();
        self.proxies.by_material.clear();
        self.proxies.seen = None;
        if !self.proxies.made.is_empty() {
            self.proxies.made.clear();
            self.proxies_changed();
        }
    }

    fn proxies_changed(&mut self) {
        self.proxies.readers = (!self.proxies.made.is_empty())
            .then(|| self.readers.with_stand_ins(self.proxies.made.clone()));
        self.preview_cache.borrow_mut().clear();
    }

    /// Look through the project's movies: play the proxies that are in
    /// the folder, queue the ones that are wanted and missing, and stop
    /// the ones no longer wanted.
    pub(super) fn sync_proxies(&mut self) {
        self.proxies.looks += 1;
        let Some(draft) = self.draft() else {
            return;
        };
        let makes = self.project_proxies();
        let choices = self
            .folder
            .as_ref()
            .map(cutlass_store::Folder::proxies)
            .unwrap_or_default();
        let seen = self
            .session
            .as_ref()
            .map(|session| (session.revision(), movies_mark(draft)));
        let (found, by_material) = movies(draft, self.path.as_deref());
        let mut played = HashMap::new();
        let mut missing = Vec::new();
        for movie in &found {
            if !choices.wants(&movie.stored, makes && movie.heavy) {
                continue;
            }
            let Some(name) = cutlass_export::proxy_name(&movie.file) else {
                continue;
            };
            let proxy = self.proxies.folder.join(name);
            if proxy.is_file() {
                played.insert(movie.file.clone(), proxy);
            } else {
                missing.push(Wanted {
                    source: movie.file.clone(),
                    proxy,
                });
            }
        }
        let proxies = &mut self.proxies;
        proxies.movies = found;
        proxies.by_material = by_material;
        proxies.seen = seen;
        let wanted = |source: &Path| missing.iter().any(|movie| movie.source == source);
        proxies.waiting.retain(|movie| wanted(&movie.source));
        if proxies
            .running
            .as_ref()
            .is_some_and(|running| !wanted(&running.source))
            && let Some(running) = proxies.running.take()
        {
            cutlass_logger::info!("proxy of {} stopped", running.source.display());
            running.job.cancel();
            proxies.stopping.push(running.job);
        }
        for movie in missing {
            let known = proxies.failed.contains(&movie.source)
                || proxies.waiting.iter().any(|w| w.source == movie.source)
                || proxies
                    .running
                    .as_ref()
                    .is_some_and(|running| running.source == movie.source);
            if !known {
                proxies.waiting.push_back(movie);
            }
        }
        if played != proxies.made {
            proxies.made = played;
            self.proxies_changed();
        }
        self.start_next_proxy();
    }

    /// Start the next waiting proxy, once none is being drawn and every
    /// stopped one has ended.
    fn start_next_proxy(&mut self) {
        self.proxies
            .stopping
            .retain(|job| matches!(job.poll(), Poll::Running));
        if self.proxies.running.is_some() || !self.proxies.stopping.is_empty() {
            return;
        }
        let Some(fps) = self.draft().map(|draft| draft.fps) else {
            return;
        };
        let Some(next) = self.proxies.waiting.pop_front() else {
            return;
        };
        let source = next.source.clone();
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let fonts = Arc::clone(&self.face);
        let started = std::fs::create_dir_all(&self.proxies.folder).and_then(|()| {
            Job::start("Proxy", move |progress| {
                progress.stage(&next.source.display().to_string());
                draw_proxy(&next, fps, &readers, &encoders, &fonts, progress)
            })
        });
        match started {
            Ok(job) => self.proxies.running = Some(Running { source, job }),
            Err(error) => {
                cutlass_logger::error!("proxy of {}: {error}", source.display());
                self.proxies.failed.insert(source);
            }
        }
    }

    /// Whether the user asked for the proxy of `source` by hand.
    fn proxy_asked(&self, source: &Path) -> bool {
        let Some(folder) = self.folder.as_ref() else {
            return false;
        };
        self.proxies
            .movies
            .iter()
            .find(|movie| movie.file == source)
            .is_some_and(|movie| folder.proxies().files.get(&movie.stored) == Some(&true))
    }

    /// Keep the proxies going: queue the ones a new movie wants, take the
    /// one that finished, and start the next. True while any is waiting
    /// or being drawn.
    pub fn poll_proxies(&mut self) -> bool {
        // A compound clip's own draft is not the project's list of movies.
        if self.outer.is_empty()
            && let Some(session) = self.session.as_ref()
        {
            let revision = session.revision();
            if self.proxies.seen.map(|(seen, _)| seen) != Some(revision) {
                let mark = movies_mark(session.draft());
                if self.proxies.seen.map(|(_, seen)| seen) == Some(mark) {
                    self.proxies.seen = Some((revision, mark));
                } else {
                    self.sync_proxies();
                }
            }
        }
        let ended = self
            .proxies
            .running
            .as_ref()
            .and_then(|running| match running.job.poll() {
                Poll::Running => None,
                Poll::Finished(outcome) => Some(outcome),
                Poll::Lost => Some(Err(ShellError::Unavailable(
                    "the proxy stopped unexpectedly".to_owned(),
                ))),
            });
        if let Some(outcome) = ended
            && let Some(running) = self.proxies.running.take()
        {
            let elapsed = running.job.elapsed();
            match outcome {
                Ok(()) => {
                    cutlass_logger::info!(
                        "proxy of {} made in {elapsed:.1?}",
                        running.source.display()
                    );
                    self.sync_proxies();
                }
                Err(error) => {
                    cutlass_logger::error!(
                        "proxy of {} failed after {elapsed:.1?}: {error}",
                        running.source.display()
                    );
                    // One made by itself fails quietly; one asked for says why.
                    if self.proxy_asked(&running.source) {
                        self.set_notice(format!("Could not make the proxy: {error}"));
                    }
                    self.proxies.failed.insert(running.source);
                }
            }
        }
        self.start_next_proxy();
        self.proxies.running.is_some()
            || !self.proxies.waiting.is_empty()
            || !self.proxies.stopping.is_empty()
    }

    /// How the library file `material_id` stands with its proxy.
    #[must_use]
    pub fn proxy_state(&self, material_id: &str) -> ProxyState {
        self.proxies
            .by_material
            .get(material_id)
            .map_or(ProxyState::None, |file| self.proxies.state_of(file))
    }

    /// What the library's menu offers for the files `ids`: `Some(true)` to
    /// make a proxy when one of them has none, `Some(false)` to cancel
    /// when every movie among them has one or is getting one, and `None`
    /// when none of them is a movie.
    #[must_use]
    pub fn proxy_offer(&self, ids: &[String]) -> Option<bool> {
        let mut states = ids
            .iter()
            .filter_map(|id| self.proxies.by_material.get(id))
            .map(|file| self.proxies.state_of(file))
            .peekable();
        states.peek()?;
        Some(states.any(|state| state == ProxyState::None))
    }

    /// The paths the project's choices name for the movies among `ids`,
    /// each once, and their files.
    fn chosen_movies(&self, ids: &[String]) -> Vec<(String, PathBuf, bool)> {
        let mut chosen: Vec<(String, PathBuf, bool)> = Vec::new();
        for movie in ids.iter().filter_map(|id| self.proxies.movie_of(id)) {
            if !chosen.iter().any(|(_, file, _)| file == &movie.file) {
                chosen.push((movie.stored.clone(), movie.file.clone(), movie.larger));
            }
        }
        chosen
    }

    /// Make a proxy of each movie among the library files `ids`, whatever
    /// the project makes by itself. One already in the folder plays at
    /// once.
    pub fn make_proxy(&mut self, ids: &[String]) -> Result<(), ShellError> {
        let mut chosen = self.chosen_movies(ids);
        if chosen.is_empty() {
            return Err(ShellError::Unavailable(
                "only a movie can have a proxy".to_owned(),
            ));
        }
        chosen.retain(|(_, _, larger)| *larger);
        if chosen.is_empty() {
            return Err(ShellError::Unavailable(format!(
                "a movie {}p or smaller needs no proxy",
                cutlass_export::PROXY_SIDE
            )));
        }
        for (_, file, _) in &chosen {
            self.proxies.failed.remove(file);
        }
        self.choose_proxies(|choices| {
            for (stored, _, _) in &chosen {
                choices.files.insert(stored.clone(), true);
            }
        })?;
        self.sync_proxies();
        let making = chosen
            .iter()
            .filter(|(_, file, _)| self.proxies.state_of(file) != ProxyState::Made)
            .count();
        self.set_notice(match (making, chosen.len()) {
            (0, 1) => "The proxy is ready".to_owned(),
            (0, _) => "The proxies are ready".to_owned(),
            (1, _) => "Making 1 proxy".to_owned(),
            (n, _) => format!("Making {n} proxies"),
        });
        Ok(())
    }

    /// Have each movie among the library files `ids` play from itself,
    /// whatever the project makes by itself. A proxy being drawn stops; a
    /// finished one stays in the folder, so asking again is at once.
    pub fn cancel_proxy(&mut self, ids: &[String]) -> Result<(), ShellError> {
        let chosen = self.chosen_movies(ids);
        if chosen.is_empty() {
            return Err(ShellError::Unavailable(
                "only a movie can have a proxy".to_owned(),
            ));
        }
        self.choose_proxies(|choices| {
            for (stored, _, _) in &chosen {
                choices.files.insert(stored.clone(), false);
            }
        })?;
        self.sync_proxies();
        self.set_notice(match chosen.len() {
            1 => "Cancelled 1 proxy".to_owned(),
            n => format!("Cancelled {n} proxies"),
        });
        Ok(())
    }

    /// Keep every proxy waiting, as one that is still stopping does, so a
    /// test sees a proxy on its way for as long as it looks.
    #[cfg(test)]
    pub(crate) fn hold_proxies(&mut self) {
        let held = Job::start("Held proxy", |progress| {
            while !progress.cancelled() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Ok(())
        });
        self.proxies.stopping.extend(held);
    }

    #[cfg(test)]
    pub(crate) fn set_proxies_folder(&mut self, folder: PathBuf) {
        self.proxies = Proxies::new(folder);
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange};
    use cutlass_session::{ImportedMedium, Request};
    use cutlass_store::Folder;

    use super::ProxyState;
    use crate::shell::Shell;
    use crate::shell::tests::{Temp, shell};

    /// A fifth of a second of the engine's pattern picture, 64 by 32.
    fn source_movie(path: &Path) {
        let pattern = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        let mut draft = cutlass_export::proxy_draft(&pattern, 64, 32, Micros(200_000), 30.0);
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

    /// A movie material `id` for `file`, said to be `width` by `height`.
    fn movie(id: &str, file: &Path, width: i32, height: i32) -> MaterialVideo {
        let mut video = MaterialVideo::new(id);
        video.kind = "video".to_owned();
        video.path = file.to_string_lossy().into_owned();
        video.width = width;
        video.height = height;
        video.duration = Micros(200_000);
        video
    }

    /// A project in `folder` holding `movies`, the first one on the timeline.
    fn project(folder: &Path, movies: Vec<MaterialVideo>) {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 64;
        draft.canvas_config.height = 32;
        if let Some(first) = movies.first() {
            let mut clip = Segment::new("clip", first.id.as_str(), TimeRange::new(0, 200_000));
            clip.source_range = Some(TimeRange::new(0, 200_000));
            draft.tracks[0].segments.push(clip);
            draft.duration = Micros(200_000);
        }
        draft.materials.videos = movies;
        Folder::new(draft).save(folder).unwrap();
    }

    /// A shell keeping its proxies in `temp`, with the app making proxies
    /// in new projects or not.
    fn desktop(temp: &Temp, makes: bool) -> Shell {
        let mut desktop = shell();
        desktop.set_proxies_folder(temp.path.join("proxies"));
        desktop.update_settings(|settings| settings.playback.proxies = makes);
        desktop
    }

    /// Poll until no proxy is waiting or being drawn.
    fn land_proxies(desktop: &mut Shell) {
        // A debug Rust encoder on a small CI runner is slow.
        let until = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while desktop.poll_proxies() && std::time::Instant::now() < until {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!desktop.poll_proxies(), "the proxies did not finish");
    }

    fn files_in(folder: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(folder).map_or_else(
            |_| Vec::new(),
            |entries| entries.map(|entry| entry.unwrap().path()).collect(),
        )
    }

    fn ids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    #[test]
    fn a_proxy_asked_for_by_hand_plays_in_the_preview_and_never_reaches_export() {
        let temp = Temp::new();
        let file = temp.path.join("source.mp4");
        source_movie(&file);
        let small = temp.path.join("small.mp4");
        std::fs::copy(&file, &small).unwrap();
        let folder = temp.path.join("project");
        project(
            &folder,
            vec![
                movie("movie", &file, 1280, 720),
                movie("small", &small, 640, 360),
            ],
        );
        let proxies = temp.path.join("proxies");
        let mut desktop = desktop(&temp, false);
        desktop.open_folder(&folder).unwrap();
        let source = file.canonicalize().unwrap();

        // The project makes none by itself.
        assert!(!desktop.project_proxies());
        assert!(!desktop.poll_proxies());
        assert!(desktop.proxied().is_empty());
        assert_eq!(desktop.proxy_state("movie"), ProxyState::None);
        assert_eq!(desktop.proxy_offer(&ids(&["movie"])), Some(true));
        assert_eq!(desktop.proxy_offer(&ids(&["nothing"])), None);

        // Only a movie larger than a proxy can have one.
        let refused = desktop.make_proxy(&ids(&["nothing"])).unwrap_err();
        assert_eq!(refused.to_string(), "only a movie can have a proxy");
        let refused = desktop.make_proxy(&ids(&["small"])).unwrap_err();
        assert_eq!(
            refused.to_string(),
            "a movie 480p or smaller needs no proxy"
        );
        assert!(!desktop.unsaved(), "a refusal changes nothing");

        // Asked for, it is drawn beside the work, not as the window's job.
        desktop.make_proxy(&ids(&["movie", "small"])).unwrap();
        assert_eq!(desktop.notice(), "Making 1 proxy");
        assert!(matches!(
            desktop.proxy_state("movie"),
            ProxyState::Making(_)
        ));
        assert_eq!(desktop.proxy_offer(&ids(&["movie"])), Some(false));
        assert!(desktop.job().is_none());
        assert!(desktop.unsaved());
        land_proxies(&mut desktop);
        assert_eq!(desktop.proxy_state("movie"), ProxyState::Made);
        assert_eq!(desktop.proxy_state("small"), ProxyState::None);
        assert_eq!(desktop.proxied(), vec![source.clone()]);
        let files = files_in(&proxies);
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
        desktop.seek(Micros(100_000));
        let shown = desktop.preview_bytes().unwrap();
        assert_eq!((shown.width, shown.height), (64, 32));

        // Cancelled, the preview reads the source again and the file stays.
        desktop.cancel_proxy(&ids(&["movie"])).unwrap();
        assert_eq!(desktop.notice(), "Cancelled 1 proxy");
        assert_eq!(desktop.proxy_state("movie"), ProxyState::None);
        assert_eq!(size(desktop.preview_readers()), source_size);
        let live = desktop.preview_bytes().unwrap();
        let apart: u64 = shown
            .rgba
            .iter()
            .zip(&live.rgba)
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum();
        assert!(apart / (live.rgba.len() as u64) < 12, "{apart}");
        assert!(proxy.is_file());

        // Asked for again, the kept proxy is taken at once.
        desktop.make_proxy(&ids(&["movie"])).unwrap();
        assert_eq!(desktop.notice(), "The proxy is ready");
        assert!(!desktop.poll_proxies());
        assert_eq!(size(desktop.preview_readers()), proxy_size);

        // The choice is the project's: a later run plays it again.
        desktop.save().unwrap();
        drop(desktop);
        let mut again = self::desktop(&temp, false);
        again.open_folder(&folder).unwrap();
        assert_eq!(again.proxied(), vec![source.clone()]);
        assert!(!again.poll_proxies());
        drop(again);

        // A changed source is shown from itself while its new proxy is made.
        let bytes = std::fs::read(&source).unwrap();
        std::fs::write(&source, [bytes.as_slice(), b"\0"].concat()).unwrap();
        let mut changed = self::desktop(&temp, false);
        changed.open_folder(&folder).unwrap();
        assert!(changed.proxied().is_empty());
        assert!(matches!(
            changed.proxy_state("movie"),
            ProxyState::Making(_)
        ));
    }

    #[test]
    fn a_project_that_makes_proxies_makes_one_as_a_heavy_movie_comes_in() {
        let temp = Temp::new();
        let light = temp.path.join("light.mp4");
        source_movie(&light);
        let heavy = temp.path.join("heavy.mp4");
        std::fs::copy(&light, &heavy).unwrap();
        let folder = temp.path.join("project");
        project(&folder, vec![movie("light", &light, 1920, 1080)]);
        let mut desktop = desktop(&temp, true);
        desktop.open_folder(&folder).unwrap();
        assert!(desktop.project_proxies());

        // A movie that is neither 4K nor long gets none.
        assert!(!desktop.poll_proxies());
        assert_eq!(desktop.proxy_state("light"), ProxyState::None);

        // A 4K one is queued the moment it is in the library.
        desktop
            .apply(Request::Import {
                medium: ImportedMedium::Video(Box::new(movie("heavy", &heavy, 3840, 2160))),
            })
            .unwrap();
        assert!(desktop.poll_proxies());
        assert!(matches!(
            desktop.proxy_state("heavy"),
            ProxyState::Making(_)
        ));
        assert!(desktop.job().is_none(), "another import can start");
        land_proxies(&mut desktop);
        assert_eq!(desktop.proxy_state("heavy"), ProxyState::Made);
        assert_eq!(desktop.proxied(), vec![heavy.canonicalize().unwrap()]);
        assert_eq!(desktop.notice(), "", "one made by itself says nothing");

        // Off, the project plays its sources; on again, the kept proxy.
        desktop.set_project_proxies(false).unwrap();
        assert!(desktop.proxied().is_empty());
        assert_eq!(desktop.proxy_offer(&ids(&["heavy"])), Some(true));
        desktop.set_project_proxies(true).unwrap();
        assert_eq!(desktop.proxy_state("heavy"), ProxyState::Made);
        assert!(!desktop.poll_proxies());

        // Cancelled by hand, it stays off though the project makes them.
        desktop.cancel_proxy(&ids(&["heavy"])).unwrap();
        desktop.set_project_proxies(false).unwrap();
        desktop.set_project_proxies(true).unwrap();
        assert_eq!(desktop.proxy_state("heavy"), ProxyState::None);
        assert!(!desktop.poll_proxies());
        desktop.make_proxy(&ids(&["heavy"])).unwrap();
        assert_eq!(desktop.proxy_state("heavy"), ProxyState::Made);

        // The import undone, its proxy is no longer played.
        desktop.undo().unwrap();
        assert!(!desktop.poll_proxies());
        assert!(desktop.proxied().is_empty());
        assert_eq!(desktop.proxy_state("heavy"), ProxyState::None);
    }

    #[test]
    fn stopping_leaves_no_part_made_proxy_and_plays_the_sources() {
        let temp = Temp::new();
        let first = temp.path.join("first.mp4");
        source_movie(&first);
        let second = temp.path.join("second.mp4");
        std::fs::copy(&first, &second).unwrap();
        let folder = temp.path.join("project");
        project(
            &folder,
            vec![
                movie("first", &first, 3840, 2160),
                movie("second", &second, 2160, 3840),
            ],
        );
        let proxies = temp.path.join("proxies");
        let mut desktop = desktop(&temp, true);
        desktop.open_folder(&folder).unwrap();

        // One is drawn at a time; the other waits.
        assert!(desktop.proxies.running.is_some());
        assert_eq!(desktop.proxies.waiting.len(), 1);
        assert_eq!(desktop.proxy_state("second"), ProxyState::Making(None));

        // The waiting one cancelled is gone from the queue.
        desktop.cancel_proxy(&ids(&["second"])).unwrap();
        assert!(desktop.proxies.waiting.is_empty());
        assert_eq!(desktop.proxy_state("second"), ProxyState::None);

        // The project turned off stops the one being drawn.
        desktop.set_project_proxies(false).unwrap();
        assert!(desktop.proxies.running.is_none());
        assert_eq!(desktop.proxy_state("first"), ProxyState::None);
        land_proxies(&mut desktop);
        assert!(desktop.proxied().is_empty());
        let parts: Vec<_> = files_in(&proxies)
            .into_iter()
            .filter(|file| file.extension().is_some_and(|kind| kind == "part"))
            .collect();
        assert!(parts.is_empty(), "{parts:?}");

        // Closing the project stops them too.
        desktop.set_project_proxies(true).unwrap();
        assert!(desktop.proxies.running.is_some() || !desktop.proxies.stopping.is_empty());
        desktop.close_project().unwrap();
        assert!(desktop.proxies.running.is_none());
        assert!(desktop.proxies.waiting.is_empty());
        land_proxies(&mut desktop);
        assert_eq!(desktop.proxy_state("first"), ProxyState::None);
    }

    #[test]
    fn a_movie_that_cannot_be_read_is_tried_once() {
        let temp = Temp::new();
        let broken = temp.path.join("broken.mp4");
        std::fs::write(&broken, b"not a movie").unwrap();
        let folder = temp.path.join("project");
        project(&folder, vec![movie("broken", &broken, 3840, 2160)]);
        let mut desktop = desktop(&temp, true);
        desktop.open_folder(&folder).unwrap();
        assert!(matches!(
            desktop.proxy_state("broken"),
            ProxyState::Making(_)
        ));
        land_proxies(&mut desktop);

        // Made by itself, it fails quietly and is not queued again.
        assert_eq!(desktop.proxy_state("broken"), ProxyState::None);
        assert_eq!(desktop.notice(), "");
        desktop.sync_proxies();
        assert!(!desktop.poll_proxies());
        assert!(files_in(&temp.path.join("proxies")).is_empty());

        // Asked for by hand, it is tried again and says why it failed.
        desktop.make_proxy(&ids(&["broken"])).unwrap();
        land_proxies(&mut desktop);
        assert!(
            desktop.notice().starts_with("Could not make the proxy: "),
            "{}",
            desktop.notice()
        );
        assert_eq!(desktop.proxy_state("broken"), ProxyState::None);
    }

    #[test]
    fn a_project_takes_the_app_setting_the_first_time_it_opens() {
        let temp = Temp::new();
        let folder = temp.path.join("project");
        project(&folder, Vec::new());
        let mut desktop = desktop(&temp, false);
        assert!(!desktop.project_proxies(), "no project follows the app");
        assert_eq!(
            desktop.set_project_proxies(true).unwrap_err().to_string(),
            "no project is open"
        );
        desktop.open_folder(&folder).unwrap();
        assert!(!desktop.project_proxies());
        assert!(!desktop.unsaved(), "opening is not an edit");

        // The app's setting is for new projects from here on.
        desktop.update_settings(|settings| settings.playback.proxies = true);
        assert!(!desktop.project_proxies());
        desktop.save().unwrap();
        drop(desktop);
        let mut again = self::desktop(&temp, true);
        again.open_folder(&folder).unwrap();
        assert!(!again.project_proxies());

        // The project's own switch is kept with it.
        again.set_project_proxies(true).unwrap();
        assert!(again.unsaved());
        again.save().unwrap();
        drop(again);
        let mut last = self::desktop(&temp, false);
        last.open_folder(&folder).unwrap();
        assert!(last.project_proxies());

        let fresh = temp.path.join("fresh");
        project(&fresh, Vec::new());
        let mut new = self::desktop(&temp, true);
        new.open_folder(&fresh).unwrap();
        assert!(new.project_proxies());
    }

    #[test]
    fn polling_looks_through_the_movies_only_when_they_change() {
        let temp = Temp::new();
        let file = temp.path.join("source.mp4");
        std::fs::write(&file, b"not read").unwrap();
        let folder = temp.path.join("project");
        project(&folder, vec![movie("movie", &file, 1280, 720)]);
        let mut desktop = desktop(&temp, true);
        desktop.open_folder(&folder).unwrap();
        let looks = desktop.proxies.looks;
        for _ in 0..200 {
            assert!(!desktop.poll_proxies());
        }
        assert_eq!(desktop.proxies.looks, looks);

        // An edit that leaves the movies alone is not a reason to look.
        desktop
            .apply(Request::SetProjectName {
                name: "Renamed".to_owned(),
            })
            .unwrap();
        desktop.poll_proxies();
        assert_eq!(desktop.proxies.looks, looks);

        // A movie that comes in is.
        desktop
            .apply(Request::Import {
                medium: ImportedMedium::Video(Box::new(movie("other", &file, 1280, 720))),
            })
            .unwrap();
        desktop.poll_proxies();
        desktop.poll_proxies();
        assert_eq!(desktop.proxies.looks, looks + 1);
    }
}
