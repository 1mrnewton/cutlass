//! One real movie and one real still, opened with the codecs the desktop registers.
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
use cutlass_eval::{OpenClips, ReaderFactory, Snapshot, frame};
use cutlass_model::{
    Crop, Draft, MaterialCanvas, MaterialColorConfig, MaterialDraft, MaterialVideo, Micros,
    Segment, TimeRange, TrackAttribute,
};
use cutlass_renderer_software::SoftwareDevice;

#[test]
fn a_real_movie_can_be_scanned_for_scene_changes() {
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
    let readers = crate::readers::installed_readers();
    let cuts = cutlass_eval::scene_cuts(&readers, &movies[0], &TimeRange::new(0, 2_000_000))
        .unwrap_or_else(|error| panic!("{error:?}"));
    for cut in cuts {
        assert!(cut.0 > 0 && cut.0 < 2_000_000);
    }
}

#[test]
fn a_real_movie_carries_a_sound_stream() {
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
    let readers = crate::readers::installed_readers();
    let heard = readers
        .carries_sound(&movies[0])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert!(heard);
}

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

#[test]
fn a_real_still_is_the_evaluated_frame() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return;
    }
    let path = root.join("04-motion-graphics.png");
    if !path.is_file() {
        eprintln!("skipped: 04-motion-graphics.png is not in ../local-files");
        return;
    }
    evaluated_still(&path).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn a_nested_still_matches_the_still() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return;
    }
    let path = root.join("04-motion-graphics.png");
    if !path.is_file() {
        eprintln!("skipped: 04-motion-graphics.png is not in ../local-files");
        return;
    }
    nested_still(&path).unwrap_or_else(|error| panic!("{error}"));
}

fn nested_still(path: &Path) -> Result<(), String> {
    let readers = crate::readers::installed_readers();
    let media = readers
        .open(path)
        .map_err(|error| format!("open {error}"))?;
    let Some(stream) = media
        .info()
        .streams
        .iter()
        .find(|stream| stream.codec == "Png" && stream.width.is_some())
    else {
        return Err("no png picture stream".to_owned());
    };
    let width = stream.width.ok_or("width")?;
    let height = stream.height.ok_or("height")?;
    let direct = movie_draft(path, width, height, false)?;
    let mut child = direct.clone();
    child.id = "child".into();
    let mut parent = Draft::empty("parent");
    parent.canvas_config.width = child.canvas_config.width;
    parent.canvas_config.height = child.canvas_config.height;
    let span = TimeRange::new(0, 1_000_000);
    parent.materials.drafts.push(MaterialDraft {
        id: "nest".into(),
        draft: Some(Box::new(child)),
        ..MaterialDraft::default()
    });
    let mut segment = Segment::new("compound", "nest", span.clone());
    segment.source_range = Some(span);
    parent.tracks[0].segments.push(segment);

    let expected = render(&readers, &direct, Micros(0))?;
    let mut device = SoftwareDevice::new();
    let mut clips = OpenClips::new();
    let first = frame(
        &Snapshot::new(&parent),
        Micros(0),
        &mut device,
        &readers,
        &mut clips,
        0,
    )
    .map_err(|error| error.to_string())?;
    let second = frame(
        &Snapshot::new(&parent),
        Micros(0),
        &mut device,
        &readers,
        &mut clips,
        0,
    )
    .map_err(|error| error.to_string())?;
    if first.rgba != second.rgba {
        return Err("the second nested frame changed".to_owned());
    }
    same_picture(&first.rgba, &expected)?;
    if render(&readers, &parent, Micros(2_000_000))?
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err("a time outside the compound drew pixels".to_owned());
    }
    Ok(())
}

fn evaluated_still(path: &Path) -> Result<(), String> {
    let readers = crate::readers::installed_readers();
    let media = readers
        .open(path)
        .map_err(|error| format!("open {error}"))?;
    let Some(stream) = media
        .info()
        .streams
        .iter()
        .find(|stream| stream.codec == "Png" && stream.width.is_some())
    else {
        return Err("no png picture stream".to_owned());
    };
    let width = stream.width.ok_or("width")?;
    let height = stream.height.ok_or("height")?;
    let Some(ffmpeg) = ffmpeg() else {
        return Err("ffmpeg is missing".to_owned());
    };
    let expected = ffmpeg_rgba(&ffmpeg, path)?;
    let draft = movie_draft(path, width, height, true)?;
    same_picture(&render(&readers, &draft, Micros(0))?, &expected)?;

    // Twice the picture and two rows taller: the fit doubles each pixel and
    // leaves one clear row above and below.
    let mut wide = movie_draft(path, width, height, false)?;
    let canvas_w = width.checked_mul(2).ok_or("canvas width")?;
    let canvas_h = height
        .checked_mul(2)
        .and_then(|rows| rows.checked_add(2))
        .ok_or("canvas height")?;
    wide.canvas_config.width = i32::try_from(canvas_w).map_err(|_| "canvas width")?;
    wide.canvas_config.height = i32::try_from(canvas_h).map_err(|_| "canvas height")?;
    doubled(
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
        return Err("a time outside the still drew pixels".to_owned());
    }
    Ok(())
}

fn ffmpeg_rgba(ffmpeg: &str, path: &Path) -> Result<Vec<u8>, String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "cutlass-desktop-still-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let raw = dir.join("frame.rgba");
    let output = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba"])
        .arg(&raw)
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!(
            "ffmpeg rgba failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let bytes = std::fs::read(&raw).map_err(|error| error.to_string());
    let _ = std::fs::remove_dir_all(&dir);
    bytes
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

    // Twice the picture and two rows taller: the fit doubles each pixel and
    // leaves one clear row above and below.
    let mut wide = movie_draft(path, width, height, false)?;
    let canvas_w = width.checked_mul(2).ok_or("canvas width")?;
    let canvas_h = height
        .checked_mul(2)
        .and_then(|rows| rows.checked_add(2))
        .ok_or("canvas height")?;
    wide.canvas_config.width = i32::try_from(canvas_w).map_err(|_| "canvas width")?;
    wide.canvas_config.height = i32::try_from(canvas_h).map_err(|_| "canvas height")?;
    doubled(
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

pub(crate) fn movie_draft(
    path: &Path,
    width: u32,
    height: u32,
    under: bool,
) -> Result<Draft, String> {
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
    let mut clips = OpenClips::new();
    let picture = frame(
        &Snapshot::new(draft),
        time,
        &mut device,
        readers,
        &mut clips,
        0,
    )
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

/// `canvas` holds `picture` scaled up twice, centered between one clear row
/// at the top and one at the bottom.
fn doubled(
    canvas: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    picture: &[u8],
    pic_w: u32,
    pic_h: u32,
) -> Result<(), String> {
    let (canvas_w, canvas_h) = (canvas_w as usize, canvas_h as usize);
    let (pic_w, pic_h) = (pic_w as usize, pic_h as usize);
    if canvas_w != pic_w * 2 || canvas_h != pic_h * 2 + 2 {
        return Err(format!("canvas {canvas_w}x{canvas_h} for {pic_w}x{pic_h}"));
    }
    if canvas.len() != canvas_w * canvas_h * 4 || picture.len() != pic_w * pic_h * 4 {
        return Err(format!("canvas is {} bytes", canvas.len()));
    }
    let stride = canvas_w * 4;
    let last = (canvas_h - 1) * stride;
    if canvas[..stride]
        .iter()
        .chain(&canvas[last..])
        .any(|byte| *byte != 0)
    {
        return Err(format!("corner {:?}", &canvas[..4]));
    }
    for y in 0..pic_h * 2 {
        for x in 0..canvas_w {
            let at = (y + 1) * stride + x * 4;
            let from = ((y / 2) * pic_w + x / 2) * 4;
            if canvas[at..at + 4] != picture[from..from + 4] {
                return Err(format!("doubled pixel {x},{y} differs"));
            }
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

/// One parity case: a name and an edit to the movie draft.
type Edit = fn(&mut Draft);

/// A layer, or a compound's child layer, is a decoded picture the GPU converts.
fn decodes_on_gpu(frame: &cutlass_eval::FrameLayers) -> bool {
    frame.layers.iter().any(|layer| match &layer.source {
        cutlass_eval::GpuSource::Yuv { .. } => true,
        cutlass_eval::GpuSource::Nested(child) => decodes_on_gpu(child),
        cutlass_eval::GpuSource::Rgba(_) => false,
    })
}

#[test]
fn the_gpu_compositor_matches_the_cpu_frame() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    let path = root.join("IMG_8143.MP4");
    if !path.is_file() {
        eprintln!("skipped: ../local-files/IMG_8143.MP4 is missing");
        return;
    }
    let renderer = match cutlass_render::Renderer::headless() {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let fonts = crate::face::FaceTable::load();
    let cases: [(&str, (u32, u32), Edit); 10] = [
        ("same size", (848, 608), |_| {}),
        ("shrunk to fit", (424, 240), |_| {}),
        ("centered", (1000, 700), |_| {}),
        ("moved, turned, flipped, faded", (848, 608), |draft| {
            let clip = &mut draft.tracks[0].segments[1].clip;
            clip.scale.x = 0.6;
            clip.scale.y = 0.45;
            clip.rotation = 30.0;
            clip.transform.x = 0.2;
            clip.transform.y = -0.1;
            clip.flip.horizontal = true;
            clip.alpha = 0.7;
        }),
        ("a quarter turn", (848, 608), |draft| {
            draft.tracks[0].segments[1].clip.rotation = 90.0;
        }),
        ("cropped", (848, 608), |draft| {
            let crop = &mut draft.materials.videos[0].crop;
            crop.upper_left_x = 0.1;
            crop.upper_left_y = 0.2;
            crop.upper_right_x = 0.8;
            crop.upper_right_y = 0.2;
            crop.lower_left_x = 0.1;
            crop.lower_left_y = 0.9;
            crop.lower_right_x = 0.8;
            crop.lower_right_y = 0.9;
        }),
        ("background", (1000, 700), |draft| {
            draft.canvas_config.background = Some("#336699".to_owned());
        }),
        ("blur backdrop", (1000, 700), |draft| {
            let canvas = MaterialCanvas {
                id: "canvas".into(),
                kind: "canvas_blur".to_owned(),
                blur: 0.05,
                ..MaterialCanvas::default()
            };
            draft.tracks[0].segments[1]
                .extra_material_refs
                .push(canvas.id.clone());
            draft.materials.canvases.push(canvas);
            draft.tracks[0].segments[1].clip.scale.x = 0.7;
            draft.tracks[0].segments[1].clip.scale.y = 0.7;
            draft.tracks[0].segments[1].clip.alpha = 0.8;
        }),
        ("nested compound", (848, 608), |draft| {
            // The movie draft becomes a compound's child, moved, turned,
            // faded and cropped on a parent that has a solid under it.
            let mut child = draft.clone();
            child.id = "child".into();
            let mut parent = Draft::empty("parent");
            parent.canvas_config = draft.canvas_config.clone();
            parent.materials.material_colors = draft.materials.material_colors.clone();
            parent.tracks[0]
                .segments
                .push(draft.tracks[0].segments[0].clone());
            let crop = Crop {
                upper_left_x: 0.05,
                upper_left_y: 0.1,
                upper_right_x: 0.9,
                upper_right_y: 0.1,
                lower_left_x: 0.05,
                lower_left_y: 0.95,
                lower_right_x: 0.9,
                lower_right_y: 0.95,
            };
            parent.materials.drafts.push(MaterialDraft {
                id: "nest".into(),
                draft: Some(Box::new(child)),
                crop,
                ..MaterialDraft::default()
            });
            let mut segment = Segment::new("compound", "nest", TimeRange::new(0, 1_000_000));
            segment.source_range = Some(TimeRange::new(0, 1_000_000));
            segment.render_index = 1;
            segment.clip.scale.x = 0.75;
            segment.clip.scale.y = 0.75;
            segment.clip.rotation = 15.0;
            segment.clip.alpha = 0.85;
            parent.tracks[0].segments.push(segment);
            *draft = parent;
        }),
        ("title over video", (848, 608), |draft| {
            let mut text = cutlass_model::MaterialText::new("words");
            text.content = "Smooth playback".to_owned();
            text.font_size = 72.0;
            text.font_path = crate::face::FaceTable::load().name().to_owned();
            text.text_color = "#FFFFFF".to_owned();
            draft.materials.texts.push(text);
            let mut title = Segment::new("title", "words", TimeRange::new(0, 1_000_000));
            title.render_index = 2;
            title.clip.transform.y = -0.5;
            title.clip.rotation = 8.0;
            title.clip.alpha = 0.9;
            draft.tracks[0].segments.push(title);
        }),
    ];
    let readers = crate::readers::installed_readers();
    for (name, (width, height), edit) in cases {
        let mut draft = movie_draft(&path, width, height, true).unwrap();
        edit(&mut draft);
        let time = Micros(500_000);
        let mut device = SoftwareDevice::new();
        let cpu = frame(
            &Snapshot::with_fonts(&draft, &fonts),
            time,
            &mut device,
            &readers,
            &mut OpenClips::new(),
            0,
        )
        .unwrap()
        .rgba;
        let mut clips = OpenClips::new();
        let composed = cutlass_eval::layers_in(
            &Snapshot::with_fonts(&draft, &fonts),
            time,
            &mut device,
            &readers,
            &mut clips,
            0,
            None,
        )
        .unwrap();
        let cutlass_eval::Composed::Layers(layers) = composed else {
            panic!("{name}: composed on the CPU");
        };
        assert!(
            decodes_on_gpu(&layers),
            "{name}: the movie did not reach the GPU"
        );
        if name == "title over video" && !fonts.name().is_empty() {
            assert!(
                layers.layers.iter().any(|layer| matches!(
                    &layer.source,
                    cutlass_eval::GpuSource::Rgba(raster) if raster.width() > 1
                )),
                "the title was not drawn"
            );
        }
        let canvas = renderer.canvas(width, height).unwrap();
        crate::gpu::draw(&renderer, &canvas, &layers).unwrap();
        let gpu = renderer.read(&canvas).unwrap();
        assert_eq!(gpu.len(), cpu.len(), "{name}");
        let pixels = cpu.len() / 4;
        let mut off = 0;
        let mut worst = 0;
        for (a, b) in gpu.chunks(4).zip(cpu.chunks(4)) {
            let diff = a
                .iter()
                .zip(b)
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap_or(0);
            worst = worst.max(diff);
            if diff > 2 {
                off += 1;
            }
        }
        eprintln!("{name}: {off} of {pixels} pixels differ by more than 2, worst {worst}");
        assert!(
            off * 1000 <= pixels,
            "{name}: {off} of {pixels} pixels differ"
        );
    }
}
