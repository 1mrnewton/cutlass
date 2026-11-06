//! The real downloader against a small HTTP server on the loopback.

#![cfg(test)]

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use cutlass_download::{
    Checksum, DownloadError, Downloader, Fetcher, Http, Job, Priority, Progress, piece_map_path,
};

/// A server for one file. It counts the bytes it sends, can refuse ranges,
/// and can break any request that starts at or past `break_from`.
struct Server {
    url: String,
    served: Arc<AtomicU64>,
    requests: Arc<AtomicUsize>,
    break_from: Arc<AtomicU64>,
    most_live: Arc<AtomicUsize>,
}

fn serve(body: Vec<u8>, ranges: bool) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/file.bin", listener.local_addr().expect("addr"));
    let body = Arc::new(body);
    let served = Arc::new(AtomicU64::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let break_from = Arc::new(AtomicU64::new(u64::MAX));
    let live = Arc::new(AtomicUsize::new(0));
    let most_live = Arc::new(AtomicUsize::new(0));
    let server = Server {
        url,
        served: Arc::clone(&served),
        requests: Arc::clone(&requests),
        break_from: Arc::clone(&break_from),
        most_live: Arc::clone(&most_live),
    };
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let body = Arc::clone(&body);
            let served = Arc::clone(&served);
            let requests = Arc::clone(&requests);
            let break_from = Arc::clone(&break_from);
            let live = Arc::clone(&live);
            let most_live = Arc::clone(&most_live);
            thread::spawn(move || {
                let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                most_live.fetch_max(now, Ordering::SeqCst);
                let mut reader = BufReader::new(stream.try_clone().expect("clone"));
                let mut stream = stream;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    let mut range = None;
                    loop {
                        let mut header = String::new();
                        if reader.read_line(&mut header).unwrap_or(0) == 0 {
                            break;
                        }
                        let header = header.trim_end();
                        if header.is_empty() {
                            break;
                        }
                        if let Some(value) = header
                            .strip_prefix("Range: bytes=")
                            .or_else(|| header.strip_prefix("range: bytes="))
                        {
                            let (start, end) = value.split_once('-').expect("range");
                            range = Some((
                                start.parse::<u64>().expect("start"),
                                end.parse::<u64>().expect("end") + 1,
                            ));
                        }
                    }
                    requests.fetch_add(1, Ordering::SeqCst);
                    // Give the other connections time to overlap.
                    thread::sleep(Duration::from_millis(5));
                    let total = body.len() as u64;
                    let (status, start, end) = match range {
                        Some((start, end)) if ranges => (206, start, end.min(total)),
                        _ => (200, 0, total),
                    };
                    if start > 0 && start >= break_from.load(Ordering::SeqCst) {
                        break;
                    }
                    let mut head = format!(
                        "HTTP/1.1 {status} {}\r\nContent-Length: {}\r\n",
                        if status == 206 {
                            "Partial Content"
                        } else {
                            "OK"
                        },
                        end - start
                    );
                    if status == 206 {
                        head.push_str(&format!(
                            "Content-Range: bytes {start}-{}/{total}\r\n",
                            end - 1
                        ));
                    }
                    head.push_str("\r\n");
                    let slice = &body[start as usize..end as usize];
                    if stream.write_all(head.as_bytes()).is_err()
                        || stream.write_all(slice).is_err()
                    {
                        break;
                    }
                    served.fetch_add(slice.len() as u64, Ordering::SeqCst);
                }
                live.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    server
}

fn body(length: usize) -> Vec<u8> {
    (0..length).map(|index| (index * 31 % 251) as u8).collect()
}

struct Temp(PathBuf);

impl Temp {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("cutlass-download-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp");
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn wait<D: Downloader>(
    downloader: &mut D,
    id: cutlass_download::JobId,
) -> Result<Progress, DownloadError> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match downloader.progress(id)? {
            Progress::Partial { .. } if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(5));
            }
            other => return Ok(other),
        }
    }
}

fn job(url: &str, destination: &Path, checksum: Option<Checksum>) -> Job {
    Job {
        url: url.to_owned(),
        destination: destination.to_path_buf(),
        checksum,
        priority: Priority::Visible,
    }
}

#[test]
fn a_file_comes_in_pieces_over_several_connections_and_its_checksum_is_checked() {
    let bytes = body(100_000);
    let server = serve(bytes.clone(), true);
    let temp = Temp::new("pieces");
    let destination = temp.0.join("incoming").join("file.bin");
    let mut fetcher = Fetcher::with(Http::new(), 4, 8_000);
    let id = fetcher
        .start(&job(&server.url, &destination, Some(Checksum::of(&bytes))))
        .expect("start");
    assert_eq!(
        wait(&mut fetcher, id).expect("progress"),
        Progress::Finished {
            bytes_written: 100_000
        }
    );
    assert_eq!(std::fs::read(&destination).expect("read"), bytes);
    assert!(!piece_map_path(&destination).exists(), "the map is gone");
    // The one-byte probe and thirteen pieces.
    assert_eq!(server.requests.load(Ordering::SeqCst), 14);
    assert!(
        server.most_live.load(Ordering::SeqCst) > 1,
        "only one connection was used"
    );
}

#[test]
fn a_new_fetcher_takes_up_a_broken_transfer_and_fetches_only_the_missing_pieces() {
    let bytes = body(50_000);
    let server = serve(bytes.clone(), true);
    // Pieces from 30 000 on break, and stay broken through every retry.
    server.break_from.store(30_000, Ordering::SeqCst);
    let temp = Temp::new("resume");
    let destination = temp.0.join("file.bin");
    let checksum = Some(Checksum::of(&bytes));
    {
        // One connection, so the pieces before the break all come in.
        let mut fetcher = Fetcher::with(Http::new(), 1, 10_000);
        let id = fetcher
            .start(&job(&server.url, &destination, checksum))
            .expect("start");
        let error = wait(&mut fetcher, id).expect_err("broken");
        assert!(matches!(error, DownloadError::Failed { .. }), "{error}");
    }
    let map = std::fs::read_to_string(piece_map_path(&destination)).expect("map kept");
    assert!(map.ends_with("11100\n"), "{map}");
    let before = server.served.load(Ordering::SeqCst);
    // As after a restart: a fresh fetcher, the server mended.
    server.break_from.store(u64::MAX, Ordering::SeqCst);
    let mut fetcher = Fetcher::with(Http::new(), 2, 10_000);
    let id = fetcher
        .start(&job(&server.url, &destination, checksum))
        .expect("start");
    assert_eq!(
        wait(&mut fetcher, id).expect("progress"),
        Progress::Finished {
            bytes_written: 50_000
        }
    );
    assert_eq!(std::fs::read(&destination).expect("read"), bytes);
    assert_eq!(
        server.served.load(Ordering::SeqCst) - before,
        20_000,
        "only the two missing pieces came again"
    );
}

#[test]
fn a_wrong_checksum_fails_and_deletes_the_file() {
    let bytes = body(20_000);
    let server = serve(bytes, true);
    let temp = Temp::new("checksum");
    let destination = temp.0.join("file.bin");
    let mut fetcher = Fetcher::with(Http::new(), 3, 4_000);
    let id = fetcher
        .start(&job(
            &server.url,
            &destination,
            Some(Checksum::of(b"something else")),
        ))
        .expect("start");
    let error = wait(&mut fetcher, id).expect_err("mismatch");
    assert!(error.to_string().contains("checksum"), "{error}");
    assert!(!destination.exists());
}

#[test]
fn a_server_without_ranges_sends_the_whole_file_on_one_connection() {
    let bytes = body(30_000);
    let server = serve(bytes.clone(), false);
    let temp = Temp::new("whole");
    let destination = temp.0.join("file.bin");
    let mut fetcher = Fetcher::with(Http::new(), 4, 4_000);
    let id = fetcher
        .start(&job(&server.url, &destination, Some(Checksum::of(&bytes))))
        .expect("start");
    assert_eq!(
        wait(&mut fetcher, id).expect("progress"),
        Progress::Finished {
            bytes_written: 30_000
        }
    );
    assert_eq!(std::fs::read(&destination).expect("read"), bytes);
    assert!(!piece_map_path(&destination).exists());
}

#[test]
fn pause_keeps_the_partial_resume_finishes_and_discard_deletes() {
    let bytes = body(200_000);
    let server = serve(bytes.clone(), true);
    let temp = Temp::new("pause");
    let destination = temp.0.join("file.bin");
    let mut fetcher = Fetcher::with(Http::new(), 1, 2_000);
    let id = fetcher
        .start(&job(&server.url, &destination, Some(Checksum::of(&bytes))))
        .expect("start");
    thread::sleep(Duration::from_millis(30));
    fetcher.pause(id).expect("pause");
    let Progress::Paused { bytes_written } = fetcher.progress(id).expect("progress") else {
        panic!("not paused");
    };
    assert!(bytes_written < 200_000, "{bytes_written}");
    let served = server.served.load(Ordering::SeqCst);
    thread::sleep(Duration::from_millis(30));
    assert_eq!(
        server.served.load(Ordering::SeqCst),
        served,
        "still fetching"
    );
    assert!(destination.exists() && piece_map_path(&destination).exists());
    fetcher.resume(id).expect("resume");
    assert_eq!(
        wait(&mut fetcher, id).expect("progress"),
        Progress::Finished {
            bytes_written: 200_000
        }
    );
    assert_eq!(std::fs::read(&destination).expect("read"), bytes);
    fetcher.discard(id).expect("discard");
    assert!(!destination.exists());
    assert!(matches!(
        fetcher.progress(id),
        Err(DownloadError::UnknownJob { .. })
    ));
}
