//! One real movie, opened with the codecs the desktop registers.
//!
//! The folder is `../local-files` from the checkout. CI does not have it,
//! so the test returns when that folder or a small movie is missing.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, OpenError, ProbeResult};
use cutlass_decode::PixelFormat;
use cutlass_engine::{DecodedFrame, ReaderFactory};
use cutlass_isobmff::IsoBmffFile;
use cutlass_model::{Micros, TimeRange};

#[test]
fn a_real_movie_returns_a_software_picture_and_audio() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return;
    }
    let movies = movies(&root);
    if movies.is_empty() {
        eprintln!("skipped: ../local-files has no small movie");
        return;
    }
    let mut failures = Vec::new();
    for path in &movies {
        match open_one(path) {
            Ok(()) => return,
            Err(error) => failures.push(format!("{}: {error}", path.display())),
        }
    }
    panic!("{}", failures.join("\n"));
}

fn open_one(path: &Path) -> Result<(), String> {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    let (decoders, _) = super::register_codecs();
    *readers.decoders_mut() = decoders;

    let mut media = readers
        .open(path)
        .map_err(|error| format!("open {error}"))?;
    let info = media.info().clone();
    if info.family != ContainerId::Isobmff {
        return Err(format!("family {:?}", info.family));
    }
    let Some(video) = info
        .streams
        .iter()
        .find(|stream| stream.codec == "H264" && stream.width.is_some())
    else {
        return Err("no h264 picture stream".to_owned());
    };
    let width = video.width.unwrap_or(0);
    let height = video.height.unwrap_or(0);
    let held = media
        .decode_at(Micros(0), &TimeRange::new(0, 1_000_000))
        .map_err(|error| format!("decode {error}"))?;
    let DecodedFrame::Coded {
        width: got_width,
        height: got_height,
        pixel,
        bytes,
        ..
    } = held.frame
    else {
        return Err("picture came from the platform decoder".to_owned());
    };
    if got_width != width || got_height != height || pixel != PixelFormat::Yuv420p {
        return Err(format!(
            "picture {got_width}x{got_height} {pixel:?}, stream {width}x{height}"
        ));
    }
    let packed = width
        .saturating_mul(height)
        .saturating_mul(3)
        .saturating_div(2);
    if bytes.len() != usize::try_from(packed).unwrap_or(usize::MAX) {
        return Err(format!(
            "packed length {} for {width}x{height}",
            bytes.len()
        ));
    }
    if let Some(ffmpeg) = ffmpeg() {
        let expected = first_yuv(&ffmpeg, path)?;
        same_picture(&bytes, &expected)?;
    } else {
        eprintln!("skipped: ffmpeg is not installed, so the picture was not compared");
    }

    let Some(audio) = info
        .streams
        .iter()
        .find(|stream| stream.sample_rate.is_some())
    else {
        return Ok(());
    };
    let mut again = readers
        .open(path)
        .map_err(|error| format!("audio open {error}"))?;
    let ready = again
        .pull_audio()
        .map_err(|error| format!("audio {error}"))?
        .ok_or_else(|| "audio stream produced no buffer".to_owned())?;
    if ready.pcm.samples == 0 {
        return Err("audio buffer is empty".to_owned());
    }
    if ready.pcm.rate != audio.sample_rate.unwrap_or(0) {
        return Err(format!(
            "audio rate {} stream {:?}",
            ready.pcm.rate, audio.sample_rate
        ));
    }
    let channels = usize::from(audio.channels.unwrap_or(0));
    if channels > 0 && ready.pcm.layout.len() != channels {
        return Err(format!(
            "audio channels {} stream {channels}",
            ready.pcm.layout.len()
        ));
    }
    Ok(())
}

fn movies(root: &Path) -> Vec<PathBuf> {
    let preferred = root.join("IMG_8146.MP4");
    if preferred.is_file() {
        return vec![preferred];
    }
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !matches!(extension.as_str(), "mp4" | "mov" | "m4v") {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.len() == 0 || meta.len() > 2_000_000 {
            continue;
        }
        found.push((meta.len(), path));
    }
    found.sort_by_key(|(len, _)| *len);
    found.into_iter().map(|(_, path)| path).collect()
}

fn probe_isobmff(header: &[u8]) -> ProbeResult {
    if header.len() < 8 {
        return ProbeResult::NeedBytes {
            have: header.len(),
            need: 8,
        };
    }
    if header.get(4..8) == Some(b"ftyp") {
        ProbeResult::Accept {
            family: ContainerId::Isobmff,
            examined: 8,
        }
    } else {
        ProbeResult::Refuse { examined: 8 }
    }
}

fn open_isobmff(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    Ok(Box::new(IsoBmffFile::open(file)?))
}

fn ffmpeg() -> Option<String> {
    [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    .into_iter()
    .find(|path| Command::new(path).arg("-version").output().is_ok())
    .map(str::to_string)
}

fn first_yuv(ffmpeg: &str, path: &Path) -> Result<Vec<u8>, String> {
    let dir = std::env::temp_dir().join(format!("cutlass-desktop-movie-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let yuv = dir.join("frame.yuv");
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-noautorotate",
            "-i",
        ])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&yuv)
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!(
            "ffmpeg {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let bytes = std::fs::read(&yuv).map_err(|error| error.to_string());
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn same_picture(got: &[u8], expected: &[u8]) -> Result<(), String> {
    if got.len() != expected.len() {
        return Err(format!(
            "picture is {} bytes and ffmpeg is {}",
            got.len(),
            expected.len()
        ));
    }
    if let Some(index) = got
        .iter()
        .zip(expected)
        .position(|(left, right)| left != right)
    {
        return Err(format!(
            "picture differs from ffmpeg at byte {index}: {} vs {}",
            got[index], expected[index]
        ));
    }
    Ok(())
}
