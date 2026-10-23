//! A prepared H.264 picture of a source clip.
//!
//! The source plays until the file is valid. The next open, while playback is
//! paused, plays the file. The picture is 8-bit, at most 1920 on the long
//! side, with a keyframe every eight frames. It is not tone-mapped. The sound
//! stays on the source. The file lives in `~/Library/Caches/Cutlass/preview`.
//! A changed source is a new name. There is no size limit.

use std::collections::HashSet;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use cutlass_core::{
    ColorRange, ColorTags, Packet, PixelFormat, Rotation, Timestamp, VideoCodec, VideoFrame,
    VideoStreamConfig,
};
use cutlass_media::{Content, Probe, VideoSession};
use cutlass_mp4::MovieFile;
use cutlass_videotoolbox::{Encoder, EncoderConfig, PixelBuffer, PlaneMut};

/// Longest side of the prepared picture, in pixels.
const LONG_EDGE: u32 = 1920;

/// Frames between forced keyframes, counting the keyframe.
const KEYFRAME_EVERY: u32 = 8;

/// About 5 Mbps at 1080p30, and the cap at 1080p120.
const MIN_BITRATE: f64 = 2_000_000.0;
const MAX_BITRATE: f64 = 20_000_000.0;

/// Start a prepare for `source` unless one is already running or the file is
/// valid. A failure is not tried again this process.
pub(crate) fn ensure(source: &str) {
    if file_path(source).is_none() || ready(source).is_some() {
        return;
    }
    let mut gate = gate();
    if !gate.seen.insert(source.to_owned()) {
        return;
    }
    let Some(sender) = &gate.tx else {
        return;
    };
    let _ = sender.send(source.to_owned());
}

/// The prepared file when its stamp still matches the source.
pub(crate) fn ready(source: &str) -> Option<String> {
    let path = file_path(source)?;
    let meta = std::fs::metadata(path).ok()?;
    let dir = cache_dir()?;
    let key = file_key(path, meta.len(), modified_parts(&meta));
    let mp4 = dir.join(format!("{key}.mp4"));
    let stamp = std::fs::read_to_string(dir.join(format!("{key}.ok"))).ok()?;
    if stamp != stamp_text(meta.len(), modified_parts(&meta)) || !mp4.is_file() {
        return None;
    }
    mp4.to_str().map(str::to_owned)
}

/// Forget a prepared file that could not be opened, and do not write it again
/// this process.
pub(crate) fn discard(source: &str) {
    let Some(path) = file_path(source) else {
        return;
    };
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    if let Some(dir) = cache_dir() {
        let key = file_key(path, meta.len(), modified_parts(&meta));
        let _ = std::fs::remove_file(dir.join(format!("{key}.ok")));
        let _ = std::fs::remove_file(dir.join(format!("{key}.mp4")));
    }
    gate().seen.insert(source.to_owned());
}

struct Gate {
    seen: HashSet<String>,
    tx: Option<std::sync::mpsc::Sender<String>>,
}

fn gate() -> MutexGuard<'static, Gate> {
    static GATE: OnceLock<Mutex<Gate>> = OnceLock::new();
    let mutex = GATE.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("cutlass-preview".to_owned())
            .spawn(move || serve(receiver));
        let tx = match spawned {
            Ok(_) => Some(sender),
            Err(error) => {
                log::warn!("The preview could not prepare pictures: {error}.");
                None
            }
        };
        Mutex::new(Gate {
            seen: HashSet::new(),
            tx,
        })
    });
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn serve(receiver: std::sync::mpsc::Receiver<String>) {
    while let Ok(source) = receiver.recv() {
        if ready(&source).is_some() {
            continue;
        }
        let Some(dir) = cache_dir() else {
            log::warn!("The preview could not prepare {source}: there is no cache folder.");
            continue;
        };
        if let Err(error) = compile_at(&source, dir) {
            log::warn!("The preview could not prepare {source}: {error}.");
        }
    }
}

fn cache_dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let dir = PathBuf::from(home).join("Library/Caches/Cutlass/preview");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    })
    .as_deref()
}

fn compile_at(source: &str, dir: &Path) -> Result<(), String> {
    let path = file_path(source).ok_or("it is not a file")?;
    let meta =
        std::fs::metadata(path).map_err(|error| format!("the file could not be read: {error}"))?;
    let len = meta.len();
    let modified = modified_parts(&meta);
    let key = file_key(path, len, modified);
    let part = dir.join(format!("{key}.part"));
    let mp4 = dir.join(format!("{key}.mp4"));
    log::info!("The preview is preparing {path}.");
    let (bytes, hardware) = encode_file(path)?;
    if let Err(error) = write_bytes(&part, &bytes) {
        let _ = std::fs::remove_file(&part);
        return Err(error);
    }
    if !same_file(path, len, modified) {
        let _ = std::fs::remove_file(&part);
        return Err("the source changed while it was prepared".into());
    }
    std::fs::rename(&part, &mp4).map_err(write_error)?;
    let stamp_part = dir.join(format!("{key}.ok.part"));
    if let Err(error) = write_bytes(&stamp_part, stamp_text(len, modified).as_bytes()) {
        let _ = std::fs::remove_file(&mp4);
        let _ = std::fs::remove_file(&stamp_part);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&stamp_part, dir.join(format!("{key}.ok"))) {
        let _ = std::fs::remove_file(&mp4);
        let _ = std::fs::remove_file(&stamp_part);
        return Err(write_error(error));
    }
    log::info!("The preview will play the prepared picture for {path}.{hardware}");
    Ok(())
}

fn same_file(path: &str, len: u64, modified: Option<(u64, u32)>) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.len() == len && modified_parts(&meta) == modified)
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(write_error)?;
    std::io::Write::write_all(&mut file, bytes).map_err(write_error)?;
    file.sync_all().map_err(write_error)
}

fn write_error(error: impl std::fmt::Display) -> String {
    format!("the preview file could not be written: {error}")
}

fn encode_file(path: &str) -> Result<(Vec<u8>, &'static str), String> {
    let opened = open_source(path)?;
    let mut session = opened.session()?;
    let mut encoder = None;
    let mut pending = None;
    let mut gap = None;
    let mut index = 0;
    let Some(packets) = opened.probe.samples(opened.index) else {
        return Err("the file has no picture".into());
    };
    for packet in packets {
        session.send_packet(packet).map_err(decode_error)?;
        pull(
            &mut session,
            &opened,
            &mut encoder,
            &mut pending,
            &mut gap,
            &mut index,
        )?;
    }
    session.flush().map_err(decode_error)?;
    pull(
        &mut session,
        &opened,
        &mut encoder,
        &mut pending,
        &mut gap,
        &mut index,
    )?;
    let Some(encoder) = encoder else {
        return Err("the picture has no frames".into());
    };
    finish(&opened, encoder, pending, gap, index)
}

fn pull(
    session: &mut VideoSession,
    opened: &Opened,
    encoder: &mut Option<Encoder>,
    pending: &mut Option<Pending>,
    gap: &mut Option<i64>,
    index: &mut u32,
) -> Result<(), String> {
    while let Some(frame) = session.receive_frame().map_err(decode_error)? {
        queue_frame(encoder, opened, pending, gap, index, frame)?;
    }
    Ok(())
}

fn queue_frame(
    encoder: &mut Option<Encoder>,
    opened: &Opened,
    pending: &mut Option<Pending>,
    gap: &mut Option<i64>,
    index: &mut u32,
    frame: VideoFrame<PixelBuffer>,
) -> Result<(), String> {
    let pts = rescale(frame.pts, opened.scale)?;
    if encoder.is_none() {
        let range = color_range(&frame.image, opened.color);
        *encoder = Some(start_encoder(opened, range)?);
    }
    let Some(encoder) = encoder.as_mut() else {
        return Err("the picture could not be encoded".into());
    };
    if let Some(held) = pending.take() {
        if pts <= held.pts {
            return Err("a frame does not move forward".into());
        }
        let duration = pts - held.pts;
        *gap = Some(duration);
        send_held(encoder, opened, held, duration, *index)?;
        *index = index
            .checked_add(1)
            .ok_or("the picture has too many frames")?;
    }
    *pending = Some(Pending {
        image: frame.image,
        pts,
    });
    Ok(())
}

fn finish(
    opened: &Opened,
    mut encoder: Encoder,
    pending: Option<Pending>,
    gap: Option<i64>,
    index: u32,
) -> Result<(Vec<u8>, &'static str), String> {
    if let Some(held) = pending {
        let duration = gap.unwrap_or(1).max(1);
        send_held(&mut encoder, opened, held, duration, index)?;
    }
    encoder.flush().map_err(encode_error)?;
    let mut packets = Vec::new();
    while let Some(packet) = encoder.receive_packet() {
        packets.push(packet.map_err(encode_error)?);
    }
    lay_down(&mut packets, opened.scale)?;
    let bytes = mux(opened, &encoder, packets)?;
    let hardware = match encoder.is_hardware() {
        Ok(true) => " The encoder is hardware.",
        Ok(false) => " The encoder is software.",
        Err(_) => "",
    };
    Ok((bytes, hardware))
}

struct Opened {
    probe: Probe,
    index: usize,
    config: VideoStreamConfig,
    width: u32,
    height: u32,
    scale: NonZeroU32,
    color: Option<ColorTags>,
    rotation: Rotation,
    rate: f64,
}

fn open_source(path: &str) -> Result<Opened, String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("the file could not be read: {error}"))?;
    let probe =
        Probe::parse(&bytes).map_err(|error| format!("the file could not be opened: {error}"))?;
    drop(bytes);
    let index = probe
        .tracks()
        .iter()
        .position(|track| track.enabled && matches!(track.content, Content::Video { .. }))
        .ok_or("the file has no picture")?;
    let (config, color, rotation) = {
        let track = &probe.tracks()[index];
        let Content::Video { config, color, .. } = &track.content else {
            return Err("the file has no picture".into());
        };
        (config.clone(), *color, track.rotation)
    };
    let (width, height) =
        preview_size(config.width, config.height).ok_or("the picture has no size")?;
    let (scale, rate) = {
        let Some(packets) = probe.samples(index) else {
            return Err("the file has no picture".into());
        };
        if packets.is_empty() {
            return Err("the file has no picture".into());
        }
        (packets[0].pts.timescale, frame_rate(packets))
    };
    Ok(Opened {
        probe,
        index,
        config,
        width,
        height,
        scale,
        color,
        rotation,
        rate,
    })
}

impl Opened {
    fn session(&self) -> Result<VideoSession, String> {
        VideoSession::open(
            &self.config,
            self.color,
            None,
            None,
            self.rotation,
            self.probe.samples(self.index).unwrap_or(&[]),
        )
        .map_err(decode_error)
    }
}

fn start_encoder(opened: &Opened, range: ColorRange) -> Result<Encoder, String> {
    Encoder::new(&EncoderConfig {
        codec: VideoCodec::H264,
        width: opened.width,
        height: opened.height,
        pixel_format: PixelFormat::Nv12,
        color_range: range,
        bitrate: Some(bitrate(opened.width, opened.height, opened.rate)),
        keyframe_interval: Some(KEYFRAME_EVERY),
        b_frames: false,
        frame_rate: Some(opened.rate),
        real_time: true,
        require_hardware: false,
    })
    .map_err(encode_error)
}

struct Pending {
    image: PixelBuffer,
    pts: i64,
}

fn send_held(
    encoder: &mut Encoder,
    opened: &Opened,
    held: Pending,
    duration: i64,
    index: u32,
) -> Result<(), String> {
    let mut image = encoder.create_pixel_buffer().map_err(encode_error)?;
    into_nv12(&held.image, &mut image)?;
    encoder
        .send_frame(
            image,
            Timestamp::new(held.pts, opened.scale),
            Some(Timestamp::new(duration, opened.scale)),
            index.is_multiple_of(KEYFRAME_EVERY),
        )
        .map_err(encode_error)
}

fn mux(opened: &Opened, encoder: &Encoder, packets: Vec<Packet>) -> Result<Vec<u8>, String> {
    let record = encoder
        .config_record()
        .ok_or("the encoder gave no config record")?
        .to_vec();
    let config = VideoStreamConfig {
        codec: VideoCodec::H264,
        width: opened.width,
        height: opened.height,
        config_record: record,
    };
    let mut movie = MovieFile::new(opened.scale);
    let written = match opened.color {
        Some(color) => movie.add_video_color(opened.scale, config, color, packets),
        None => movie.add_video(opened.scale, config, packets),
    };
    written.map_err(write_error)?;
    movie.to_bytes().map_err(write_error)
}

fn lay_down(packets: &mut [Packet], scale: NonZeroU32) -> Result<(), String> {
    if packets.is_empty() {
        return Err("the encoder gave no picture".into());
    }
    let mut pts = Vec::with_capacity(packets.len());
    for packet in &*packets {
        pts.push(rescale(packet.pts, scale)?);
    }
    let mut next = 0i64;
    for (index, packet) in packets.iter_mut().enumerate() {
        let duration = match pts.get(index + 1) {
            Some(following) if *following > pts[index] => following - pts[index],
            Some(_) => return Err("a frame does not move forward".into()),
            None => last_duration(packet, scale, index, &pts)?,
        };
        packet.pts = Timestamp::new(pts[index], scale);
        packet.dts = Some(Timestamp::new(next, scale));
        packet.duration = Some(Timestamp::new(duration, scale));
        next = next
            .checked_add(duration)
            .ok_or("the timeline does not fit")?;
    }
    Ok(())
}

fn last_duration(
    packet: &Packet,
    scale: NonZeroU32,
    index: usize,
    pts: &[i64],
) -> Result<i64, String> {
    if let Some(duration) = packet.duration {
        let duration = rescale(duration, scale)?;
        if duration > 0 {
            return Ok(duration);
        }
    }
    let previous = index
        .checked_sub(1)
        .and_then(|index| pts.get(index))
        .copied();
    Ok(previous
        .map(|previous| pts[index] - previous)
        .unwrap_or(1)
        .max(1))
}

fn decode_error(error: impl std::fmt::Display) -> String {
    format!("the picture could not be decoded: {error}")
}

/// `file://` drops that prefix. `content:` is not a file. Same rule as
/// `frames::file_path`.
fn file_path(source: &str) -> Option<&str> {
    if let Some(rest) = source.strip_prefix("file://") {
        let rest = rest.strip_prefix("localhost").unwrap_or(rest);
        return (!rest.is_empty()).then_some(rest);
    }
    if source.starts_with("content:") || source.is_empty() {
        return None;
    }
    Some(source)
}

fn encode_error(error: impl std::fmt::Display) -> String {
    format!("the picture could not be encoded: {error}")
}

fn color_range(frame: &PixelBuffer, tags: Option<ColorTags>) -> ColorRange {
    if let Some(full) = tags.and_then(|tags| tags.full_range) {
        if full {
            return ColorRange::Full;
        }
        return ColorRange::Limited;
    }
    frame
        .format()
        .map(|(_, range)| range)
        .unwrap_or(ColorRange::Limited)
}

fn into_nv12(source: &PixelBuffer, dest: &mut PixelBuffer) -> Result<(), String> {
    let sample = match source.format().map(|(format, _)| format) {
        Some(PixelFormat::Nv12) => Sample::Eight,
        Some(PixelFormat::P010) => Sample::Ten,
        _ => return Err("the frame is not a picture this preview can prepare".into()),
    };
    let copied = source
        .read_planes(|planes| {
            let (Some(luma), Some(chroma)) = (planes.first(), planes.get(1)) else {
                return None;
            };
            Some((copy_plane(luma), copy_plane(chroma)))
        })
        .map_err(|error| format!("the frame could not be read: {error}"))?;
    let Some((luma, chroma)) = copied else {
        return Err("the frame has no planes".into());
    };
    dest.write_planes(|planes| write_scaled(planes, &luma, &chroma, sample))
        .map_err(|error| format!("the frame could not be scaled: {error}"))?
}

fn write_scaled(
    planes: &mut [PlaneMut<'_>],
    luma: &Copied,
    chroma: &Copied,
    sample: Sample,
) -> Result<(), String> {
    if planes.len() < 2 {
        return Err("the encoder buffer has no planes".into());
    }
    let luma_bytes = scaled_bytes(view(luma), sample, 1, plane_shape(&planes[0]));
    let chroma_bytes = scaled_bytes(view(chroma), sample, 2, plane_shape(&planes[1]));
    copy_bytes(planes[0].data, &luma_bytes);
    copy_bytes(planes[1].data, &chroma_bytes);
    Ok(())
}

fn plane_shape(plane: &PlaneMut<'_>) -> (usize, usize, usize, usize) {
    (
        plane.width,
        plane.height,
        plane.bytes_per_row,
        plane.data.len(),
    )
}

fn scaled_bytes(
    src: PlaneView<'_>,
    sample: Sample,
    components: usize,
    shape: (usize, usize, usize, usize),
) -> Vec<u8> {
    let (width, height, stride, len) = shape;
    let mut data = vec![128; len];
    scale_plane(
        src,
        sample,
        components,
        PlaneOut {
            data: &mut data,
            stride,
            width,
            height,
        },
    );
    data
}

fn copy_bytes(dest: &mut [u8], src: &[u8]) {
    let len = dest.len().min(src.len());
    dest[..len].copy_from_slice(&src[..len]);
}

struct Copied {
    data: Vec<u8>,
    stride: usize,
    width: usize,
    height: usize,
}

fn copy_plane(plane: &cutlass_videotoolbox::Plane<'_>) -> Copied {
    Copied {
        data: plane.data.to_vec(),
        stride: plane.bytes_per_row,
        width: plane.width,
        height: plane.height,
    }
}

fn view(plane: &Copied) -> PlaneView<'_> {
    PlaneView {
        data: &plane.data,
        stride: plane.stride,
        width: plane.width,
        height: plane.height,
    }
}

#[derive(Clone, Copy)]
enum Sample {
    Eight,
    Ten,
}

struct PlaneView<'a> {
    data: &'a [u8],
    stride: usize,
    width: usize,
    height: usize,
}

struct PlaneOut<'a> {
    data: &'a mut [u8],
    stride: usize,
    width: usize,
    height: usize,
}

fn scale_plane(src: PlaneView<'_>, sample: Sample, components: usize, dest: PlaneOut<'_>) {
    if src.width == 0 || src.height == 0 || dest.width == 0 || dest.height == 0 || components == 0 {
        return;
    }
    let step = match sample {
        Sample::Eight => 1,
        Sample::Ten => 2,
    };
    for y in 0..dest.height {
        let y0 = y * src.height / dest.height;
        let y1 = ((y + 1) * src.height / dest.height)
            .max(y0 + 1)
            .min(src.height);
        for x in 0..dest.width {
            let x0 = x * src.width / dest.width;
            let x1 = ((x + 1) * src.width / dest.width)
                .max(x0 + 1)
                .min(src.width);
            for component in 0..components {
                let mut sum = 0u64;
                let mut count = 0u64;
                for sy in y0..y1 {
                    let row = sy.saturating_mul(src.stride);
                    for sx in x0..x1 {
                        let at = row
                            .saturating_add(sx.saturating_mul(components.saturating_mul(step)))
                            .saturating_add(component.saturating_mul(step));
                        sum += sample_at(src.data, at, sample);
                        count += 1;
                    }
                }
                let out = y.saturating_mul(dest.stride) + x * components + component;
                if let Some(slot) = dest.data.get_mut(out) {
                    *slot = average(sum, count);
                }
            }
        }
    }
}

fn sample_at(src: &[u8], at: usize, sample: Sample) -> u64 {
    match sample {
        Sample::Eight => u64::from(src.get(at).copied().unwrap_or(0)),
        Sample::Ten => {
            let lo = src.get(at).copied().unwrap_or(0);
            let hi = src.get(at.saturating_add(1)).copied().unwrap_or(0);
            u64::from(u16::from_le_bytes([lo, hi]) >> 8)
        }
    }
}

fn average(sum: u64, count: u64) -> u8 {
    if count == 0 {
        return 0;
    }
    let value = (sum + count / 2) / count;
    u8::try_from(value).unwrap_or(255)
}

fn preview_size(width: u32, height: u32) -> Option<(u32, u32)> {
    if width < 2 || height < 2 {
        return None;
    }
    let (width, height) = if width.max(height) <= LONG_EDGE {
        (width, height)
    } else if width >= height {
        (LONG_EDGE, scale_edge(height, width))
    } else {
        (scale_edge(width, height), LONG_EDGE)
    };
    let width = width & !1;
    let height = height & !1;
    (width >= 2 && height >= 2).then_some((width, height))
}

fn scale_edge(short: u32, long: u32) -> u32 {
    let scaled = (f64::from(short) * f64::from(LONG_EDGE) / f64::from(long)).round();
    if !scaled.is_finite() {
        return 2;
    }
    let scaled = scaled.clamp(2.0, f64::from(LONG_EDGE));
    // The clamp keeps this at most the long edge.
    scaled as u32
}

fn bitrate(width: u32, height: u32, frames_per_second: f64) -> u32 {
    let bits = f64::from(width) * f64::from(height) * frames_per_second * 0.08;
    let bits = bits.clamp(MIN_BITRATE, MAX_BITRATE);
    // The clamp keeps this inside 2 to 20 Mbps.
    bits.round() as u32
}

fn frame_rate(packets: &[Packet]) -> f64 {
    let Some(duration) = packets.iter().find_map(|packet| packet.duration) else {
        return 30.0;
    };
    if duration.value <= 0 {
        return 30.0;
    }
    let rate = f64::from(duration.timescale.get()) / duration.value as f64;
    if rate.is_finite() && (1.0..1000.0).contains(&rate) {
        rate
    } else {
        30.0
    }
}

fn file_key(path: &str, len: u64, modified: Option<(u64, u32)>) -> String {
    let mut state = 0xcbf29ce484222325u64;
    mix(&mut state, path.as_bytes());
    mix(&mut state, &len.to_le_bytes());
    let (secs, nanos) = modified.unwrap_or((0, 0));
    mix(&mut state, &secs.to_le_bytes());
    mix(&mut state, &nanos.to_le_bytes());
    format!("{state:016x}")
}

fn mix(state: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *state ^= u64::from(*byte);
        *state = state.wrapping_mul(0x100_0000_01b3);
    }
}

fn modified_parts(meta: &std::fs::Metadata) -> Option<(u64, u32)> {
    let modified = meta.modified().ok()?;
    let since = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some((since.as_secs(), since.subsec_nanos()))
}

fn stamp_text(len: u64, modified: Option<(u64, u32)>) -> String {
    let (secs, nanos) = modified.unwrap_or((0, 0));
    format!("1\n{len}\n{secs}\n{nanos}\n")
}

fn rescale(time: Timestamp, scale: NonZeroU32) -> Result<i64, String> {
    if time.timescale == scale {
        return Ok(time.value);
    }
    let ticks = i128::from(time.value)
        .checked_mul(i128::from(scale.get()))
        .ok_or("a timestamp does not fit")?;
    let ticks = ticks / i128::from(time.timescale.get());
    i64::try_from(ticks).map_err(|_| "a timestamp does not fit".to_owned())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use cutlass_core::{ColorTags, VideoCodec, VideoStreamConfig};
    use cutlass_mp4::{Kind, Movie, MovieFile};
    use cutlass_videotoolbox::{Encoder, EncoderConfig};

    use super::{
        KEYFRAME_EVERY, PlaneOut, PlaneView, Sample, bitrate, compile_at, file_key, lay_down,
        modified_parts, preview_size, scale_plane,
    };

    #[test]
    fn the_preview_size_fits_the_long_edge() {
        assert_eq!(preview_size(3840, 2160), Some((1920, 1080)));
        assert_eq!(preview_size(2160, 3840), Some((1080, 1920)));
        assert_eq!(preview_size(1280, 720), Some((1280, 720)));
        assert_eq!(preview_size(1919, 1080), Some((1918, 1080)));
        assert_eq!(preview_size(1, 1), None);
    }

    #[test]
    fn the_bitrate_stays_between_two_and_twenty_megabits() {
        assert_eq!(bitrate(1920, 1080, 30.0), 4_976_640);
        assert_eq!(bitrate(1920, 1080, 120.0), 19_906_560);
        assert_eq!(bitrate(1280, 720, 24.0), 2_000_000);
    }

    #[test]
    fn a_changed_file_gets_a_new_name() {
        let same = file_key("/clip.mov", 10, Some((1, 2)));
        assert_eq!(same, file_key("/clip.mov", 10, Some((1, 2))));
        assert_ne!(same, file_key("/clip.mov", 11, Some((1, 2))));
        assert_ne!(same, file_key("/clip.mov", 10, Some((1, 3))));
        assert_ne!(same, file_key("/other.mov", 10, Some((1, 2))));
        assert_eq!(same.len(), 16);
    }

    #[test]
    fn an_area_average_shrinks_a_plane() {
        let src = [10, 20, 30, 40];
        let mut dest = [0];
        scale_plane(
            PlaneView {
                data: &src,
                stride: 2,
                width: 2,
                height: 2,
            },
            Sample::Eight,
            1,
            PlaneOut {
                data: &mut dest,
                stride: 1,
                width: 1,
                height: 1,
            },
        );
        assert_eq!(dest, [25]);
    }

    #[test]
    fn a_ten_bit_plane_keeps_the_top_eight_bits() {
        let src = ten([10, 20, 30, 40]);
        let mut dest = [0];
        scale_plane(
            PlaneView {
                data: &src,
                stride: 4,
                width: 2,
                height: 2,
            },
            Sample::Ten,
            1,
            PlaneOut {
                data: &mut dest,
                stride: 1,
                width: 1,
                height: 1,
            },
        );
        assert_eq!(dest, [25]);
    }

    #[test]
    fn chroma_samples_stay_apart() {
        let src = [8, 240];
        let mut dest = [0, 0];
        scale_plane(
            PlaneView {
                data: &src,
                stride: 2,
                width: 1,
                height: 1,
            },
            Sample::Eight,
            2,
            PlaneOut {
                data: &mut dest,
                stride: 2,
                width: 1,
                height: 1,
            },
        );
        assert_eq!(dest, [8, 240]);
    }

    #[test]
    fn a_clip_prepares_an_h264_picture() {
        let dir = Temp::new();
        let tags = ColorTags {
            primaries: 9,
            transfer: 18,
            matrix: 9,
            full_range: Some(false),
        };
        let source = dir.path.join("source.mp4");
        std::fs::write(&source, gray_movie(tags)).expect("source");
        let source = source.to_str().expect("path");
        compile_at(source, &dir.path).expect("prepare");
        let meta = std::fs::metadata(source).expect("meta");
        let key = file_key(source, meta.len(), modified_parts(&meta));
        let prepared = std::fs::read(dir.path.join(format!("{key}.mp4"))).expect("prepared");
        let movie = Movie::parse(&prepared).expect("parse");
        let track = &movie.tracks()[0];
        assert_eq!(track.color(), Some(tags));
        let Kind::Video(config) = track.kind() else {
            panic!("video");
        };
        assert_eq!(config.codec, VideoCodec::H264);
        assert_eq!((config.width, config.height), (32, 32));
        let samples = track.samples();
        assert_eq!(samples.len(), 9);
        assert!(samples[0].keyframe);
        assert!(samples[8].keyframe);
        assert!(!samples[1].keyframe);
    }

    fn ten(values: [u8; 4]) -> [u8; 8] {
        let mut bytes = [0; 8];
        for (index, value) in values.into_iter().enumerate() {
            let sample = u16::from(value) << 8;
            let pair = sample.to_le_bytes();
            bytes[index * 2] = pair[0];
            bytes[index * 2 + 1] = pair[1];
        }
        bytes
    }

    fn gray_movie(tags: ColorTags) -> Vec<u8> {
        let scale = std::num::NonZeroU32::new(30_000).expect("timescale");
        let mut encoder = Encoder::new(&EncoderConfig {
            keyframe_interval: Some(KEYFRAME_EVERY),
            b_frames: false,
            frame_rate: Some(30.0),
            real_time: true,
            ..EncoderConfig::new(VideoCodec::H264, 32, 32)
        })
        .expect("encoder");
        for index in 0..9u32 {
            let mut image = encoder.create_pixel_buffer().expect("buffer");
            image
                .write_planes(|planes| {
                    for plane in planes {
                        for byte in plane.data.iter_mut() {
                            *byte = 128;
                        }
                    }
                })
                .expect("fill");
            let pts = cutlass_core::Timestamp::new(i64::from(index) * 1000, scale);
            encoder
                .send_frame(
                    image,
                    pts,
                    Some(cutlass_core::Timestamp::new(1000, scale)),
                    index.is_multiple_of(KEYFRAME_EVERY),
                )
                .expect("send");
        }
        encoder.flush().expect("flush");
        let mut packets = Vec::new();
        while let Some(packet) = encoder.receive_packet() {
            packets.push(packet.expect("packet"));
        }
        lay_down(&mut packets, scale).expect("times");
        let config = VideoStreamConfig {
            codec: VideoCodec::H264,
            width: 32,
            height: 32,
            config_record: encoder.config_record().expect("config").to_vec(),
        };
        let mut movie = MovieFile::new(scale);
        movie
            .add_video_color(scale, config, tags, packets)
            .expect("mux");
        movie.to_bytes().expect("bytes")
    }

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir()
                .join(format!("cutlass-preview-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&path).expect("temp");
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
