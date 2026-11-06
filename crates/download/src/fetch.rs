//! The real downloader: pieces of one file fetched over several
//! connections, a piece map beside the partial so a restart picks up where
//! the last run stopped, and the checksum checked before a file counts as
//! finished.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};

use crate::{DownloadError, Downloader, Job, JobId, Priority, Progress, Transport, hash_file};

/// How many times one piece is tried before the transfer fails.
const TRIES: usize = 3;

/// The piece map's first line, so an older or foreign file is not trusted.
const MAP_MAGIC: &str = "cutlass pieces 1";

/// The piece map beside `destination`.
#[must_use]
pub fn piece_map_path(destination: &Path) -> PathBuf {
    let mut name = destination.file_name().unwrap_or_default().to_os_string();
    name.push(".pieces");
    destination.with_file_name(name)
}

/// Which pieces of one file are on disk. It is saved after every piece, so
/// a crash loses at most the pieces in flight.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PieceMap {
    url: String,
    length: u64,
    piece: u64,
    done: Vec<bool>,
}

impl PieceMap {
    fn new(url: &str, length: u64, piece: u64) -> Self {
        let count = length.div_ceil(piece).max(1);
        Self {
            url: url.to_owned(),
            length,
            piece,
            done: vec![false; usize::try_from(count).unwrap_or(usize::MAX)],
        }
    }

    fn span(&self, index: usize) -> (u64, u64) {
        let start = index as u64 * self.piece;
        (start, (start + self.piece).min(self.length))
    }

    fn bytes_done(&self) -> u64 {
        (0..self.done.len())
            .filter(|&index| self.done[index])
            .map(|index| {
                let (start, end) = self.span(index);
                end - start
            })
            .sum()
    }

    fn missing(&self) -> Vec<usize> {
        (0..self.done.len())
            .filter(|&index| !self.done[index])
            .collect()
    }

    fn text(&self) -> String {
        let marks: String = self
            .done
            .iter()
            .map(|&done| if done { '1' } else { '0' })
            .collect();
        format!(
            "{MAP_MAGIC}\n{}\n{}\n{}\n{marks}\n",
            self.url, self.length, self.piece
        )
    }

    fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        if lines.next()? != MAP_MAGIC {
            return None;
        }
        let url = lines.next()?.to_owned();
        let length: u64 = lines.next()?.parse().ok()?;
        let piece: u64 = lines.next()?.parse().ok()?;
        if piece == 0 {
            return None;
        }
        let done: Vec<bool> = lines
            .next()?
            .chars()
            .map(|mark| match mark {
                '1' => Some(true),
                '0' => Some(false),
                _ => None,
            })
            .collect::<Option<_>>()?;
        let map = Self {
            url,
            length,
            piece,
            done,
        };
        (map.done.len() as u64 == length.div_ceil(piece).max(1)).then_some(map)
    }

    /// Written beside the map and renamed over it, so a crash leaves the old
    /// map or the new one, never half of one.
    fn save(&self, path: &Path) -> Result<(), String> {
        let mut scratch = path.as_os_str().to_owned();
        scratch.push(".new");
        let scratch = PathBuf::from(scratch);
        fs::write(&scratch, self.text())
            .and_then(|()| fs::rename(&scratch, path))
            .map_err(|error| format!("{}: {error}", path.display()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum State {
    Running,
    Paused,
    Cancelled,
    Finished,
    Failed(String),
}

/// What the transfer's threads and the caller both see.
struct Shared {
    stop: AtomicBool,
    written: AtomicU64,
    state: Mutex<State>,
}

impl Shared {
    fn state(&self) -> State {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn set(&self, state: State) {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner) = state;
    }

    /// The first failure wins and stops the other connections.
    fn fail(&self, message: String) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if *state == State::Running {
            *state = State::Failed(message);
        }
        self.stop.store(true, Ordering::SeqCst);
    }
}

struct Transfer {
    job: Job,
    shared: Arc<Shared>,
    runner: Option<JoinHandle<()>>,
}

impl Transfer {
    fn halt(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(runner) = self.runner.take() {
            let _ = runner.join();
        }
    }
}

/// Downloads over a [`Transport`]: pieces over several connections, a piece
/// map saved beside the partial, resume after a restart, and the checksum
/// checked before [`Progress::Finished`]. Publishing stays `storage`'s step.
pub struct Fetcher<T: Transport> {
    transport: Arc<T>,
    connections: usize,
    piece: u64,
    next: u64,
    transfers: HashMap<JobId, Transfer>,
}

impl<T: Transport> Fetcher<T> {
    /// Four connections and 1 MiB pieces.
    pub fn new(transport: T) -> Self {
        Self::with(transport, 4, 1 << 20)
    }

    /// `connections` at once for a row on screen (half of them, at least
    /// one, for a background file), `piece` bytes a request.
    pub fn with(transport: T, connections: usize, piece: u64) -> Self {
        Self {
            transport: Arc::new(transport),
            connections: connections.max(1),
            piece: piece.max(1),
            next: 1,
            transfers: HashMap::new(),
        }
    }

    fn run(&self, job: &Job, shared: &Arc<Shared>) -> JoinHandle<()> {
        let transport = Arc::clone(&self.transport);
        let shared = Arc::clone(shared);
        let job = job.clone();
        let connections = match job.priority {
            Priority::Visible => self.connections,
            Priority::Background => self.connections.div_ceil(2),
        };
        let piece = self.piece;
        thread::spawn(move || {
            let outcome = fetch(&*transport, &job, &shared, connections, piece);
            match outcome {
                Ok(true) => shared.set(State::Finished),
                Ok(false) => {}
                Err(message) => shared.fail(message),
            }
        })
    }

    fn transfer(&mut self, id: JobId) -> Result<&mut Transfer, DownloadError> {
        self.transfers
            .get_mut(&id)
            .ok_or(DownloadError::UnknownJob { id })
    }
}

/// One run of a transfer. `Ok(true)` when every byte is in and checked,
/// `Ok(false)` when it was stopped.
fn fetch<T: Transport>(
    transport: &T,
    job: &Job,
    shared: &Shared,
    connections: usize,
    piece: u64,
) -> Result<bool, String> {
    let map_path = piece_map_path(&job.destination);
    let saved = fs::read_to_string(&map_path)
        .ok()
        .and_then(|text| PieceMap::parse(&text))
        .filter(|map| map.url == job.url)
        .filter(|map| fs::metadata(&job.destination).is_ok_and(|meta| meta.len() == map.length));
    let map = match saved {
        Some(map) => map,
        None => {
            let head = transport.head(&job.url)?;
            match head {
                crate::Head {
                    length: Some(length),
                    ranges: true,
                } => {
                    let map = PieceMap::new(&job.url, length, piece);
                    if let Some(parent) = job.destination.parent() {
                        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
                    }
                    let file = File::create(&job.destination)
                        .map_err(|error| format!("{}: {error}", job.destination.display()))?;
                    file.set_len(length).map_err(|error| error.to_string())?;
                    map.save(&map_path)?;
                    map
                }
                _ => return whole(transport, job, shared),
            }
        }
    };
    shared.written.store(map.bytes_done(), Ordering::SeqCst);
    // Taken from the end, so reversed: the file fills from its start.
    let mut missing = map.missing();
    missing.reverse();
    let queue = Mutex::new(missing);
    let map = Mutex::new(map);
    let file = Mutex::new(
        OpenOptions::new()
            .write(true)
            .open(&job.destination)
            .map_err(|error| format!("{}: {error}", job.destination.display()))?,
    );
    thread::scope(|scope| {
        for _ in 0..connections {
            scope.spawn(|| connection(transport, job, shared, &queue, &map, &file, &map_path));
        }
    });
    if shared.stop.load(Ordering::SeqCst) {
        return Ok(false);
    }
    let done = map
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .missing()
        .is_empty();
    if !done {
        return Err(format!("{}: pieces are still missing", job.url));
    }
    drop(file);
    check(job)?;
    let _ = fs::remove_file(&map_path);
    Ok(true)
}

/// One connection: take a missing piece, fetch it, write it, mark it.
fn connection<T: Transport>(
    transport: &T,
    job: &Job,
    shared: &Shared,
    queue: &Mutex<Vec<usize>>,
    map: &Mutex<PieceMap>,
    file: &Mutex<File>,
    map_path: &Path,
) {
    loop {
        if shared.stop.load(Ordering::SeqCst) {
            return;
        }
        let Some(index) = queue.lock().unwrap_or_else(PoisonError::into_inner).pop() else {
            return;
        };
        let (start, end) = map
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .span(index);
        let mut last = String::new();
        let mut fetched = false;
        for _ in 0..TRIES {
            let mut bytes = Vec::with_capacity(usize::try_from(end - start).unwrap_or(0));
            let result = transport.get(&job.url, Some((start, end)), &mut |chunk| {
                bytes.extend_from_slice(chunk);
                !shared.stop.load(Ordering::SeqCst)
            });
            if shared.stop.load(Ordering::SeqCst) {
                return;
            }
            match result {
                Ok(()) if bytes.len() as u64 == end - start => {
                    let written = {
                        let mut file = file.lock().unwrap_or_else(PoisonError::into_inner);
                        file.seek(SeekFrom::Start(start))
                            .and_then(|_| file.write_all(&bytes))
                    };
                    if let Err(error) = written {
                        shared.fail(format!("{}: {error}", job.destination.display()));
                        return;
                    }
                    fetched = true;
                    break;
                }
                Ok(()) => last = format!("{}: a piece came back short", job.url),
                Err(message) => last = message,
            }
        }
        if !fetched {
            shared.fail(last);
            return;
        }
        let mut map = map.lock().unwrap_or_else(PoisonError::into_inner);
        map.done[index] = true;
        if let Err(message) = map.save(map_path) {
            shared.fail(message);
            return;
        }
        shared.written.fetch_add(end - start, Ordering::SeqCst);
    }
}

/// A server with no ranges or no length: one connection, from the start,
/// and no resume.
fn whole<T: Transport>(transport: &T, job: &Job, shared: &Shared) -> Result<bool, String> {
    if let Some(parent) = job.destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut file = File::create(&job.destination)
        .map_err(|error| format!("{}: {error}", job.destination.display()))?;
    shared.written.store(0, Ordering::SeqCst);
    let mut failed = None;
    transport.get(&job.url, None, &mut |chunk| {
        if let Err(error) = file.write_all(chunk) {
            failed = Some(error.to_string());
            return false;
        }
        shared
            .written
            .fetch_add(chunk.len() as u64, Ordering::SeqCst);
        !shared.stop.load(Ordering::SeqCst)
    })?;
    if let Some(message) = failed {
        return Err(message);
    }
    if shared.stop.load(Ordering::SeqCst) {
        return Ok(false);
    }
    drop(file);
    check(job)?;
    Ok(true)
}

/// A file whose checksum does not match is deleted, so the next start
/// fetches it again rather than trusting bad bytes.
fn check(job: &Job) -> Result<(), String> {
    let Some(expected) = job.checksum else {
        return Ok(());
    };
    let (got, _) = hash_file(&job.destination).map_err(|error| error.to_string())?;
    if got == expected {
        return Ok(());
    }
    let _ = fs::remove_file(&job.destination);
    Err(format!(
        "{}: the checksum is {got}, not {expected}; the file was deleted",
        job.url
    ))
}

impl<T: Transport> Downloader for Fetcher<T> {
    fn start(&mut self, job: &Job) -> Result<JobId, DownloadError> {
        let id = JobId(self.next);
        self.next += 1;
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            written: AtomicU64::new(0),
            state: Mutex::new(State::Running),
        });
        let runner = self.run(job, &shared);
        self.transfers.insert(
            id,
            Transfer {
                job: job.clone(),
                shared,
                runner: Some(runner),
            },
        );
        Ok(id)
    }

    fn progress(&mut self, id: JobId) -> Result<Progress, DownloadError> {
        let transfer = self.transfer(id)?;
        let bytes_written = transfer.shared.written.load(Ordering::SeqCst);
        match transfer.shared.state() {
            State::Running => Ok(Progress::Partial { bytes_written }),
            State::Paused => Ok(Progress::Paused { bytes_written }),
            State::Cancelled => Ok(Progress::Cancelled { bytes_written }),
            State::Finished => Ok(Progress::Finished { bytes_written }),
            State::Failed(message) => Err(DownloadError::Failed { message }),
        }
    }

    fn pause(&mut self, id: JobId) -> Result<(), DownloadError> {
        let transfer = self.transfer(id)?;
        transfer.halt();
        if transfer.shared.state() == State::Running {
            transfer.shared.set(State::Paused);
        }
        Ok(())
    }

    fn resume(&mut self, id: JobId) -> Result<(), DownloadError> {
        let transfer = self.transfer(id)?;
        transfer.halt();
        match transfer.shared.state() {
            State::Finished => return Ok(()),
            State::Paused | State::Cancelled | State::Failed(_) | State::Running => {}
        }
        let shared = Arc::clone(&transfer.shared);
        let job = transfer.job.clone();
        shared.stop.store(false, Ordering::SeqCst);
        shared.set(State::Running);
        let runner = self.run(&job, &shared);
        self.transfer(id)?.runner = Some(runner);
        Ok(())
    }

    fn cancel(&mut self, id: JobId) -> Result<(), DownloadError> {
        let transfer = self.transfer(id)?;
        transfer.halt();
        if transfer.shared.state() == State::Running {
            transfer.shared.set(State::Cancelled);
        }
        Ok(())
    }

    fn discard(&mut self, id: JobId) -> Result<(), DownloadError> {
        let mut transfer = self
            .transfers
            .remove(&id)
            .ok_or(DownloadError::UnknownJob { id })?;
        transfer.halt();
        let _ = fs::remove_file(&transfer.job.destination);
        let _ = fs::remove_file(piece_map_path(&transfer.job.destination));
        Ok(())
    }
}

impl<T: Transport> Drop for Fetcher<T> {
    fn drop(&mut self) {
        for transfer in self.transfers.values_mut() {
            transfer.halt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PieceMap;

    #[test]
    fn a_piece_map_reads_back_and_refuses_a_wrong_count() {
        let mut map = PieceMap::new("http://host/file", 2500, 1000);
        assert_eq!(map.done.len(), 3);
        map.done[1] = true;
        assert_eq!(map.span(2), (2000, 2500));
        assert_eq!(map.bytes_done(), 1000);
        assert_eq!(map.missing(), vec![0, 2]);
        assert_eq!(PieceMap::parse(&map.text()), Some(map.clone()));
        let short = map.text().replace("010", "01");
        assert_eq!(PieceMap::parse(&short), None);
        assert_eq!(PieceMap::parse("something else\n"), None);
    }
}
