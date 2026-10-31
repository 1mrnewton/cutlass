//! One real movie, opened with the codecs the desktop registers.
//!
//! The folder is `../local-files` from the checkout. CI does not have it,
//! so the test returns when that folder or a small movie is missing.

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_codec::ContainerId;
use cutlass_codec::access::{
    ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat, Transfer, Yuv420p,
};
use cutlass_color::{Input, Matrix, Outcome, Picture as ColorPicture, Request};
use cutlass_engine::DecodedFrame;
use cutlass_eval::{ReaderFactory, Snapshot, frame};
use cutlass_model::{
    Draft, MaterialColorConfig, MaterialVideo, Micros, Segment, TimeRange, TrackAttribute,
};
use cutlass_renderer_software::SoftwareDevice;

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

#[test]
fn a_real_movie_is_the_evaluated_frame() {
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
        match evaluated_frame(path) {
            Ok(()) => return,
            Err(error) => failures.push(format!("{}: {error}", path.display())),
        }
    }
    panic!("{}", failures.join("\n"));
}

fn open_one(path: &Path) -> Result<(), String> {
    let readers = crate::readers::installed_readers();
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
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "cutlass-desktop-movie-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
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

fn evaluated_frame(path: &Path) -> Result<(), String> {
    let readers = crate::readers::installed_readers();
    let mut media = readers
        .open(path)
        .map_err(|error| format!("open {error}"))?;
    let info = media.info().clone();
    let Some(video) = info
        .streams
        .iter()
        .find(|stream| stream.codec == "H264" && stream.width.is_some())
    else {
        return Err("no h264 picture stream".to_owned());
    };
    let held = media
        .decode_at(Micros(0), &TimeRange::new(0, 1_000_000))
        .map_err(|error| format!("decode {error}"))?;
    let DecodedFrame::Coded {
        width,
        height,
        pixel,
        primaries,
        transfer,
        ..
    } = held.frame
    else {
        return Err("picture came from the platform decoder".to_owned());
    };
    if width == 0 || height == 0 || pixel != PixelFormat::Yuv420p {
        return Err(format!("picture {width}x{height} {pixel:?}"));
    }
    if primaries != ColorPrimaries::Bt709 || transfer != Transfer::Bt709 {
        return Err(format!("color {primaries:?} {transfer:?}"));
    }
    if video.width != Some(width) || video.height != Some(height) {
        return Err(format!(
            "stream {:?}x{:?} picture {width}x{height}",
            video.width, video.height
        ));
    }
    let Some(ffmpeg) = ffmpeg() else {
        return Err("ffmpeg is missing".to_owned());
    };
    let yuv = first_yuv(&ffmpeg, path)?;
    let expected = rgba_from_yuv(&yuv, width, height, primaries, transfer)?;
    let draft = movie_draft(path, width, height, true)?;
    same_picture(&render(&readers, &draft, Micros(0))?, &expected)?;

    let mut wide = movie_draft(path, width, height, false)?;
    let canvas_w = width.checked_add(2).ok_or("canvas width")?;
    let canvas_h = height.checked_add(2).ok_or("canvas height")?;
    wide.canvas_config.width = i32::try_from(canvas_w).map_err(|_| "canvas width")?;
    wide.canvas_config.height = i32::try_from(canvas_h).map_err(|_| "canvas height")?;
    centered(
        &render(&readers, &wide, Micros(0))?,
        canvas_w,
        canvas_h,
        &expected,
        width,
        height,
    )?;

    let mut hidden = draft.clone();
    hidden.tracks[0].attribute = TrackAttribute(TrackAttribute::HIDE);
    if render(&readers, &hidden, Micros(0))?
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err("hidden track drew pixels".to_owned());
    }
    if render(&readers, &draft, Micros(2_000_000))?
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err("a time outside the movie drew pixels".to_owned());
    }
    Ok(())
}

fn movie_draft(path: &Path, width: u32, height: u32, under: bool) -> Result<Draft, String> {
    let mut draft = Draft::empty("draft");
    draft.canvas_config.width = i32::try_from(width).map_err(|_| "width")?;
    draft.canvas_config.height = i32::try_from(height).map_err(|_| "height")?;
    let span = TimeRange::new(0, 1_000_000);
    if under {
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "under".into(),
            is_color_clip: true,
            solid_color: "0xFFFF00FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("under", "under", span.clone()));
    }
    let mut video = MaterialVideo::new("movie");
    video.path = path.display().to_string();
    video.width = draft.canvas_config.width;
    video.height = draft.canvas_config.height;
    video.duration = Micros(1_000_000);
    draft.materials.videos.push(video);
    let mut segment = Segment::new("movie", "movie", span.clone());
    segment.source_range = Some(span);
    segment.render_index = 1;
    draft.tracks[0].segments.push(segment);
    Ok(draft)
}

fn render(readers: &ReaderFactory, draft: &Draft, time: Micros) -> Result<Vec<u8>, String> {
    let mut device = SoftwareDevice::new();
    let picture = frame(&Snapshot::new(draft), time, &mut device, readers)
        .map_err(|error| error.to_string())?;
    let width = u32::try_from(draft.canvas_config.width).map_err(|_| "width")?;
    let height = u32::try_from(draft.canvas_config.height).map_err(|_| "height")?;
    if picture.extent.width != width || picture.extent.height != height {
        return Err(format!(
            "frame {}x{} canvas {width}x{height}",
            picture.extent.width, picture.extent.height
        ));
    }
    Ok(picture.rgba)
}

fn centered(
    canvas: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    picture: &[u8],
    pic_w: u32,
    pic_h: u32,
) -> Result<(), String> {
    let x0 = canvas_w.saturating_sub(pic_w) / 2;
    let y0 = canvas_h.saturating_sub(pic_h) / 2;
    if x0 == 0 || y0 == 0 {
        return Err("the picture covers the canvas edge".to_owned());
    }
    let canvas_len = usize::try_from(canvas_w)
        .ok()
        .and_then(|width| width.checked_mul(usize::try_from(canvas_h).ok()?))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("canvas length")?;
    if canvas.len() != canvas_len {
        return Err(format!("canvas is {} bytes", canvas.len()));
    }
    let corner = canvas.get(..4).ok_or("canvas")?;
    if corner != [0, 0, 0, 0] {
        return Err(format!("corner {corner:?}"));
    }
    let canvas_stride = usize::try_from(canvas_w)
        .map_err(|_| "canvas width")?
        .checked_mul(4)
        .ok_or("canvas stride")?;
    let picture_stride = usize::try_from(pic_w)
        .map_err(|_| "picture width")?
        .checked_mul(4)
        .ok_or("picture stride")?;
    for row in 0..usize::try_from(pic_h).map_err(|_| "picture height")? {
        let dest = (usize::try_from(y0).map_err(|_| "origin")? + row)
            .checked_mul(canvas_stride)
            .and_then(|row| row.checked_add(usize::try_from(x0).ok()?.checked_mul(4)?))
            .ok_or("canvas row")?;
        let src = row.checked_mul(picture_stride).ok_or("picture row")?;
        let got = canvas
            .get(dest..dest + picture_stride)
            .ok_or("canvas row")?;
        let expected = picture
            .get(src..src + picture_stride)
            .ok_or("picture row")?;
        if got != expected {
            return Err(format!("centered row {row} differs"));
        }
    }
    Ok(())
}

fn rgba_from_yuv(
    bytes: &[u8],
    width: u32,
    height: u32,
    primaries: ColorPrimaries,
    transfer: Transfer,
) -> Result<Vec<u8>, String> {
    let yuv = split_yuv(width, height, bytes).ok_or("yuv layout")?;
    let matrix =
        Matrix::from_coefficients(MatrixCoefficients::Bt709).map_err(|error| error.to_string())?;
    let picture = ColorPicture::from_yuv420p(
        &yuv,
        ColorRange::Limited,
        matrix,
        primaries,
        cutlass_color::curve_from_codec(transfer),
    );
    match cutlass_color::convert(Input::Cpu(picture), &Request::draft(0)) {
        Ok(Outcome::Working(working)) => Ok(working.rgba),
        Ok(Outcome::Platform { .. }) => Err("color stayed on the platform".to_owned()),
        Err(error) => Err(error.to_string()),
    }
}

fn split_yuv(width: u32, height: u32, bytes: &[u8]) -> Option<Yuv420p> {
    let y_len = usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)?;
    let chroma_width = usize::try_from(width / 2).ok()?;
    let chroma_height = usize::try_from(height / 2).ok()?;
    let chroma_len = chroma_width.checked_mul(chroma_height)?;
    let u_end = y_len.checked_add(chroma_len)?;
    let v_end = u_end.checked_add(chroma_len)?;
    if bytes.len() != v_end {
        return None;
    }
    Yuv420p::try_new(
        width,
        height,
        bytes[..y_len].to_vec(),
        bytes[y_len..u_end].to_vec(),
        bytes[u_end..v_end].to_vec(),
    )
    .ok()
}
