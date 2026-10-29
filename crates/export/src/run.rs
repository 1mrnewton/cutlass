//! Walk a range, encode each frame, and mux the packets.
//!
//! A frame is included when its start is inside the half-open range. The
//! picture is `eval::frame`, then the export clock. Audio is the same mix
//! playback uses, on the export clock, so a missing clip is an error.

use std::path::{Path, PathBuf};

use cutlass_codec::{
    FrameSize, Packet as FilePacket, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{Mux, MuxStatus, NewStream, TrackDuration, TrackId};
use cutlass_encode::{
    AudioInput, ChannelLayout, CodecParams, CpuFrame, EncodeFactory, Encoder, FrameInput,
    FrameTime, Packet as AccessPacket, Pcm, PixelFormat,
};
use cutlass_eval::{
    AudioError, ClipPcm, ClockKind, Device, EvalError, MixWindow, Snapshot, audio_slice, frame,
    wait_export_frame,
};
use cutlass_model::{Draft, FrameRate, Micros, SegmentId, TimeRange};

use crate::config::{AudioExport, ExportError, ExportRequest, output_size, validate};
use crate::sidecars::{file_name, sidecar_text, write_sidecar};

const MAX_SAMPLES_PER_FRAME: u32 = 480_000;

/// One evaluated picture. `rgba` is the software frame at the canvas size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportedFrame {
    pub time: Micros,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The muxed file and the pictures eval returned.
#[derive(Debug)]
pub struct ExportResult {
    pub frames: Vec<ExportedFrame>,
    pub output_width: u32,
    pub output_height: u32,
    pub finished: cutlass_container::FinishedMux,
    pub sidecar: Option<PathBuf>,
}

/// Borrowed inputs for one export.
pub struct ExportRun<'a, D, M> {
    pub draft: &'a Draft,
    pub device: &'a mut D,
    pub request: &'a ExportRequest,
    pub factory: &'a EncodeFactory,
    pub mux: &'a mut M,
    pub destination: Option<&'a Path>,
}

struct TrackTicks {
    video: TrackId,
    audio: Option<TrackId>,
    video_scale: u32,
    audio_scale: u32,
    video_ticks: i64,
    audio_ticks: i64,
}

/// Frame indexes whose start lies in `range`.
///
/// The start is [`FrameRate::frame_start`]. A frame at the range end is excluded.
pub fn frame_indexes(rate: FrameRate, range: &TimeRange) -> Result<Vec<i64>, ExportError> {
    validate_rate(rate)?;
    if range.duration.0 <= 0 {
        return Err(ExportError::Range);
    }
    let end = range
        .start
        .0
        .checked_add(range.duration.0)
        .ok_or(ExportError::Range)?;
    let Some(first) = first_index(rate, range.start.0, end)? else {
        return Ok(Vec::new());
    };
    let Some(last) = last_index(rate, range.start.0, end)? else {
        return Ok(Vec::new());
    };
    if last < first {
        return Ok(Vec::new());
    }
    let span = u64::try_from(last - first).map_err(|_| ExportError::Limit)?;
    let count = span.checked_add(1).ok_or(ExportError::Limit)?;
    if count > crate::config::MAX_FRAMES {
        return Err(ExportError::Limit);
    }
    let mut indexes = Vec::with_capacity(usize::try_from(count).map_err(|_| ExportError::Limit)?);
    for index in first..=last {
        if rate.contains_frame(range, index) != Ok(true) {
            return Err(ExportError::Range);
        }
        indexes.push(index);
    }
    Ok(indexes)
}

/// Compares software RGBA from playback and export.
///
/// HDR and a buffer that is not tightly packed RGBA are errors. They are not
/// reported as a match.
pub fn same_software_pixels(
    draft: &Draft,
    width: u32,
    height: u32,
    playback: &[u8],
    exported: &[u8],
) -> Result<(), ExportError> {
    if draft_is_hdr(draft) {
        return Err(ExportError::HdrNotThisCase);
    }
    let Some(len) = rgba_len(width, height) else {
        return Err(ExportError::NotSoftwareRgba);
    };
    if playback.len() != len || exported.len() != len {
        return Err(ExportError::NotSoftwareRgba);
    }
    if playback != exported {
        return Err(ExportError::PixelMismatch);
    }
    Ok(())
}

/// Evaluates `draft` over the request range, encodes, and muxes.
///
/// The draft is borrowed and not written. `audio` supplies PCM for each
/// audible segment; mute, volume, and fades are applied by the mix.
pub fn export<D, M, F>(
    mut run: ExportRun<'_, D, M>,
    mut audio: F,
) -> Result<ExportResult, ExportError>
where
    D: Device,
    M: Mux,
    F: FnMut(&SegmentId, u32, u32) -> Result<ClipPcm, ExportError>,
{
    validate(run.request)?;
    if run.mux.family() != run.request.container {
        return Err(ExportError::Container);
    }
    let sidecar_body = prepare_sidecar(run.draft, run.request, run.destination)?;
    let indexes = frame_indexes(run.request.frame_rate, &run.request.range)?;
    let output = output_size(
        run.draft.canvas_config.width,
        run.draft.canvas_config.height,
        run.request.long_side,
    )?;
    let mut video = open_video(run.factory, run.request, output)?;
    let mut audio_encoder = open_audio(run.factory, run.request)?;
    let mut tracks = add_tracks(run.mux, run.request, output)?;
    let frames = match encode_pictures(&mut run, &mut *video, &mut tracks, &indexes, output) {
        Ok(frames) => frames,
        Err(error) => return Err(stop(run.mux, error)),
    };
    if let Err(error) = encode_audio(
        &mut run,
        &mut audio_encoder,
        &mut tracks,
        &indexes,
        &mut audio,
    ) {
        return Err(stop(run.mux, error));
    }
    if let Err(error) = drain_encoders(run.mux, &mut *video, &mut audio_encoder, &mut tracks) {
        return Err(stop(run.mux, error));
    }
    let finished = match finish_file(run.mux, &tracks) {
        Ok(finished) => finished,
        Err(error) => return Err(stop(run.mux, error)),
    };
    let sidecar = write_body(run.destination, run.request, sidecar_body)?;
    Ok(ExportResult {
        frames,
        output_width: output.0,
        output_height: output.1,
        finished,
        sidecar,
    })
}

fn validate_rate(rate: FrameRate) -> Result<(), ExportError> {
    if rate.numerator == 0 || rate.denominator == 0 {
        Err(ExportError::FrameRate)
    } else {
        Ok(())
    }
}

fn first_index(rate: FrameRate, start: i64, end: i64) -> Result<Option<i64>, ExportError> {
    if start >= end || end <= 0 {
        return Ok(None);
    }
    let mut index = if start <= 0 {
        0
    } else {
        rate.frame_index(Micros(start))
            .ok_or(ExportError::FrameRate)?
    };
    for _ in 0..4 {
        let frame_start = rate.frame_start(index).ok_or(ExportError::FrameRate)?;
        if frame_start.0 >= end {
            return Ok(None);
        }
        if frame_start.0 >= start {
            return Ok(Some(index));
        }
        index = index.checked_add(1).ok_or(ExportError::Limit)?;
    }
    Err(ExportError::FrameRate)
}

fn last_index(rate: FrameRate, start: i64, end: i64) -> Result<Option<i64>, ExportError> {
    if end <= 0 {
        return Ok(None);
    }
    let index = rate
        .frame_index(Micros(end - 1))
        .ok_or(ExportError::FrameRate)?;
    let frame_start = rate.frame_start(index).ok_or(ExportError::FrameRate)?;
    if frame_start.0 >= start && frame_start.0 < end {
        Ok(Some(index))
    } else {
        Ok(None)
    }
}

fn draft_is_hdr(draft: &Draft) -> bool {
    if draft.color_space == 67 || draft.color_space == 2084 {
        return true;
    }
    draft.tracks.iter().any(|track| {
        track.segments.iter().any(|segment| {
            segment.hdr_vivid_settings.is_some()
                || segment
                    .hdr_settings
                    .as_ref()
                    .is_some_and(|hdr| hdr.mode != 0)
        })
    })
}

fn prepare_sidecar(
    draft: &Draft,
    request: &ExportRequest,
    destination: Option<&Path>,
) -> Result<Option<String>, ExportError> {
    let Some(sidecar) = &request.sidecar else {
        return Ok(None);
    };
    let body = sidecar_text(draft, &request.range, sidecar.kind)?;
    let _ = file_name(sidecar.kind, &sidecar.name)?;
    match destination {
        Some(dir) if dir.is_dir() => Ok(Some(body)),
        _ => Err(ExportError::Path),
    }
}

fn write_body(
    destination: Option<&Path>,
    request: &ExportRequest,
    body: Option<String>,
) -> Result<Option<PathBuf>, ExportError> {
    let Some(body) = body else {
        return Ok(None);
    };
    let Some(sidecar) = &request.sidecar else {
        return Ok(None);
    };
    let Some(dir) = destination else {
        return Err(ExportError::Path);
    };
    let name = file_name(sidecar.kind, &sidecar.name)?;
    write_sidecar(dir, &name, &body).map(Some)
}

fn open_video(
    factory: &EncodeFactory,
    request: &ExportRequest,
    output: (u32, u32),
) -> Result<Box<dyn Encoder>, ExportError> {
    let mut params =
        CodecParams::with_format(request.codec, output.0, output.1, PixelFormat::Rgba8);
    params.bitrate = request.bitrate;
    factory.open(&params).map_err(ExportError::Encoder)
}

fn open_audio(
    factory: &EncodeFactory,
    request: &ExportRequest,
) -> Result<Option<Box<dyn Encoder>>, ExportError> {
    let Some(audio) = &request.audio else {
        return Ok(None);
    };
    let mut params = CodecParams::audio(audio.codec, audio.sample_rate, layout(audio.channels)?);
    params.bitrate = audio.bitrate;
    factory
        .open(&params)
        .map(Some)
        .map_err(ExportError::Encoder)
}

fn layout(channels: u16) -> Result<ChannelLayout, ExportError> {
    match channels {
        1 => Ok(ChannelLayout::mono()),
        2 => Ok(ChannelLayout::stereo()),
        _ => Err(ExportError::AudioLayout),
    }
}

fn add_tracks<M: Mux>(
    mux: &mut M,
    request: &ExportRequest,
    output: (u32, u32),
) -> Result<TrackTicks, ExportError> {
    let size = FrameSize::try_new(output.0, output.1).map_err(|_| ExportError::Canvas)?;
    let video = mux
        .add_stream(NewStream {
            params: StreamParams::video(StreamCodec::Known(request.codec), size, Vec::new()),
            timescale: Timescale::new(request.frame_rate.numerator),
            language: None,
        })
        .map_err(ExportError::Mux)?;
    let audio = match &request.audio {
        Some(audio) => Some(add_audio(mux, audio)?),
        None => None,
    };
    Ok(TrackTicks {
        video,
        audio,
        video_scale: request.frame_rate.numerator,
        audio_scale: request
            .audio
            .as_ref()
            .map(|audio| audio.sample_rate)
            .unwrap_or(0),
        video_ticks: 0,
        audio_ticks: 0,
    })
}

fn add_audio<M: Mux>(mux: &mut M, audio: &AudioExport) -> Result<TrackId, ExportError> {
    let params = StreamParams::try_audio(
        StreamCodec::Known(audio.codec),
        audio.sample_rate,
        audio.channels,
        Vec::new(),
    )
    .map_err(|_| ExportError::AudioLayout)?;
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(audio.sample_rate),
        language: None,
    })
    .map_err(ExportError::Mux)
}

fn encode_pictures<D, M>(
    run: &mut ExportRun<'_, D, M>,
    video: &mut dyn Encoder,
    tracks: &mut TrackTicks,
    indexes: &[i64],
    output: (u32, u32),
) -> Result<Vec<ExportedFrame>, ExportError>
where
    D: Device,
    M: Mux,
{
    let mut frames = Vec::with_capacity(indexes.len());
    for index in indexes {
        frames.push(encode_picture(run, video, tracks, *index, output)?);
    }
    Ok(frames)
}

fn encode_picture<D, M>(
    run: &mut ExportRun<'_, D, M>,
    video: &mut dyn Encoder,
    tracks: &mut TrackTicks,
    index: i64,
    output: (u32, u32),
) -> Result<ExportedFrame, ExportError>
where
    D: Device,
    M: Mux,
{
    let time = run
        .request
        .frame_rate
        .frame_start(index)
        .ok_or(ExportError::FrameRate)?;
    let ready = Micros(
        time.0
            .checked_add(run.request.late_by)
            .ok_or(ExportError::Range)?,
    );
    let evaluated = evaluated_frame(run.draft, run.device, time, ready)?;
    let bytes = scale_rgba(
        evaluated.width,
        evaluated.height,
        &evaluated.rgba,
        output.0,
        output.1,
    )?;
    let timing = video_time(run.request.frame_rate, index)?;
    let picture = CpuFrame::new(output.0, output.1, PixelFormat::Rgba8, bytes, timing)
        .map_err(ExportError::Encoder)?;
    let packets = video
        .push_frame(&FrameInput::Cpu(picture))
        .map_err(ExportError::Encoder)?;
    write_packets(
        run.mux,
        tracks.video.get(),
        &packets,
        &mut tracks.video_ticks,
    )?;
    Ok(evaluated)
}

fn encode_audio<D, M, F>(
    run: &mut ExportRun<'_, D, M>,
    audio_encoder: &mut Option<Box<dyn Encoder>>,
    tracks: &mut TrackTicks,
    indexes: &[i64],
    audio: &mut F,
) -> Result<(), ExportError>
where
    D: Device,
    M: Mux,
    F: FnMut(&SegmentId, u32, u32) -> Result<ClipPcm, ExportError>,
{
    let Some(encoder) = audio_encoder.as_mut() else {
        return Ok(());
    };
    let Some(settings) = run.request.audio.as_ref() else {
        return Ok(());
    };
    let Some(stream) = tracks.audio else {
        return Ok(());
    };
    for index in indexes {
        let time = run
            .request
            .frame_rate
            .frame_start(*index)
            .ok_or(ExportError::FrameRate)?;
        let samples = mix_window(run.draft, settings, run.request, time, audio)?;
        let packets = encoder
            .push(&AudioInput { pcm: samples })
            .map_err(ExportError::Encoder)?;
        write_packets(run.mux, stream.get(), &packets, &mut tracks.audio_ticks)?;
    }
    Ok(())
}

fn evaluated_frame<D: Device>(
    draft: &Draft,
    device: &mut D,
    time: Micros,
    ready_at: Micros,
) -> Result<ExportedFrame, ExportError> {
    let snapshot = Snapshot::new(draft);
    let picture = frame(&snapshot, time, device).map_err(map_eval)?;
    let picture = wait_export_frame::<D::Error>(time, ready_at, picture).map_err(map_eval)?;
    let width = u32::try_from(draft.canvas_config.width).map_err(|_| ExportError::Canvas)?;
    let height = u32::try_from(draft.canvas_config.height).map_err(|_| ExportError::Canvas)?;
    if picture.extent.width != width || picture.extent.height != height {
        return Err(ExportError::Picture("canvas size".to_owned()));
    }
    Ok(ExportedFrame {
        time,
        width,
        height,
        rgba: picture.rgba,
    })
}

fn map_eval<E: std::fmt::Display>(error: EvalError<E>) -> ExportError {
    match error {
        EvalError::Canvas => ExportError::Canvas,
        EvalError::Wait => ExportError::Wait,
        EvalError::Compose(error) => ExportError::Picture(error.to_string()),
    }
}

fn video_time(rate: FrameRate, index: i64) -> Result<FrameTime, ExportError> {
    let duration = i64::from(rate.denominator);
    let pts = index.checked_mul(duration).ok_or(ExportError::Limit)?;
    FrameTime::new(pts, duration, rate.numerator).map_err(|_| ExportError::FrameRate)
}

fn mix_window<F>(
    draft: &Draft,
    settings: &AudioExport,
    request: &ExportRequest,
    time: Micros,
    audio: &mut F,
) -> Result<Pcm, ExportError>
where
    F: FnMut(&SegmentId, u32, u32) -> Result<ClipPcm, ExportError>,
{
    let next = request
        .frame_rate
        .frame_start(
            request
                .frame_rate
                .frame_index(time)
                .ok_or(ExportError::FrameRate)?
                .checked_add(1)
                .ok_or(ExportError::Limit)?,
        )
        .ok_or(ExportError::FrameRate)?;
    let range_end = request
        .range
        .start
        .0
        .checked_add(request.range.duration.0)
        .ok_or(ExportError::Range)?;
    let end = next.0.min(range_end);
    let frames = samples_between(time.0, end, settings.sample_rate)?;
    if frames > MAX_SAMPLES_PER_FRAME {
        return Err(ExportError::Limit);
    }
    let mut source_error = None;
    let mixed = audio_slice(
        &Snapshot::new(draft),
        time,
        MixWindow {
            rate: settings.sample_rate,
            limiter: settings.limiter,
            frames,
        },
        ClockKind::Export,
        |segment, count, rate| match audio(segment, count, rate) {
            Ok(pcm) => Ok(pcm),
            Err(error) => {
                source_error = Some(error);
                Err(AudioError::Source)
            }
        },
    );
    if let Some(error) = source_error {
        return Err(error);
    }
    let mixed = mixed.map_err(ExportError::Audio)?;
    if mixed.pcm.samples == 0 {
        silence(settings.sample_rate, settings.channels, frames)
    } else {
        Ok(mixed.pcm)
    }
}

fn samples_between(start: i64, end: i64, rate: u32) -> Result<u32, ExportError> {
    if rate == 0 || end < start || start < 0 {
        return Err(ExportError::Range);
    }
    let count = sample_index(end, rate)?
        .checked_sub(sample_index(start, rate)?)
        .ok_or(ExportError::Range)?;
    u32::try_from(count).map_err(|_| ExportError::Limit)
}

fn sample_index(micros: i64, rate: u32) -> Result<i64, ExportError> {
    let product = i128::from(micros)
        .checked_mul(i128::from(rate))
        .ok_or(ExportError::Limit)?;
    i64::try_from(product / 1_000_000).map_err(|_| ExportError::Limit)
}

fn silence(rate: u32, channels: u16, frames: u32) -> Result<Pcm, ExportError> {
    let layout = layout(channels)?;
    let count = usize::try_from(frames).map_err(|_| ExportError::Limit)?;
    let len = count.checked_mul(layout.len()).ok_or(ExportError::Limit)?;
    Pcm::from_f32(rate, layout, vec![0.0; len]).map_err(|_| ExportError::Audio(AudioError::Mix))
}

fn scale_rgba(
    src_w: u32,
    src_h: u32,
    src: &[u8],
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, ExportError> {
    let src_len = rgba_len(src_w, src_h).ok_or(ExportError::Canvas)?;
    if src.len() != src_len {
        return Err(ExportError::Picture("rgba length".to_owned()));
    }
    if src_w == dst_w && src_h == dst_h {
        return Ok(src.to_vec());
    }
    let dst_len = rgba_len(dst_w, dst_h).ok_or(ExportError::Limit)?;
    let mut dst = vec![0_u8; dst_len];
    for y in 0..dst_h {
        let src_y = nearest(y, src_h, dst_h)?;
        for x in 0..dst_w {
            let src_x = nearest(x, src_w, dst_w)?;
            let from = pixel_index(src_x, src_y, src_w)?;
            let to = pixel_index(x, y, dst_w)?;
            copy_pixel(src, from, &mut dst, to)?;
        }
    }
    Ok(dst)
}

fn nearest(pos: u32, src_edge: u32, dst_edge: u32) -> Result<u32, ExportError> {
    if src_edge == 0 || dst_edge == 0 {
        return Err(ExportError::Canvas);
    }
    let value = u64::from(pos) * u64::from(src_edge) / u64::from(dst_edge);
    let value = u32::try_from(value).map_err(|_| ExportError::Limit)?;
    Ok(value.min(src_edge - 1))
}

fn copy_pixel(src: &[u8], from: usize, dst: &mut [u8], to: usize) -> Result<(), ExportError> {
    let src_end = from.checked_add(4).ok_or(ExportError::Limit)?;
    let dst_end = to.checked_add(4).ok_or(ExportError::Limit)?;
    let sample = src
        .get(from..src_end)
        .ok_or_else(|| ExportError::Picture("pixel".to_owned()))?;
    let slot = dst
        .get_mut(to..dst_end)
        .ok_or_else(|| ExportError::Picture("pixel".to_owned()))?;
    slot.copy_from_slice(sample);
    Ok(())
}

fn rgba_len(width: u32, height: u32) -> Option<usize> {
    usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)?
        .checked_mul(4)
}

fn pixel_index(x: u32, y: u32, width: u32) -> Result<usize, ExportError> {
    let x = usize::try_from(x).map_err(|_| ExportError::Limit)?;
    let y = usize::try_from(y).map_err(|_| ExportError::Limit)?;
    let width = usize::try_from(width).map_err(|_| ExportError::Limit)?;
    y.checked_mul(width)
        .and_then(|row| row.checked_add(x))
        .and_then(|pixel| pixel.checked_mul(4))
        .ok_or(ExportError::Limit)
}

fn write_packets<M: Mux>(
    mux: &mut M,
    stream: u32,
    packets: &[AccessPacket],
    ticks: &mut i64,
) -> Result<(), ExportError> {
    for packet in packets {
        let file = file_packet(stream, packet)?;
        let duration = file.timestamp().duration().ok_or(ExportError::Packet)?;
        mux.write_packet(&file).map_err(ExportError::Mux)?;
        *ticks = ticks.checked_add(duration).ok_or(ExportError::Limit)?;
    }
    Ok(())
}

fn file_packet(stream: u32, packet: &AccessPacket) -> Result<FilePacket, ExportError> {
    if !packet.side.is_empty() {
        return Err(ExportError::SideData);
    }
    if packet.duration < 0 {
        return Err(ExportError::Packet);
    }
    let timestamp = Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
        .map_err(|_| ExportError::Packet)?;
    FilePacket::try_new(
        stream,
        packet.data.clone(),
        timestamp,
        Timescale::new(packet.timescale),
        packet.keyframe,
    )
    .map_err(|_| ExportError::Packet)
}

fn drain_encoders<M: Mux>(
    mux: &mut M,
    video: &mut dyn Encoder,
    audio_encoder: &mut Option<Box<dyn Encoder>>,
    tracks: &mut TrackTicks,
) -> Result<(), ExportError> {
    let packets = video.drain().map_err(ExportError::Encoder)?;
    write_packets(mux, tracks.video.get(), &packets, &mut tracks.video_ticks)?;
    if let (Some(encoder), Some(stream)) = (audio_encoder.as_mut(), tracks.audio) {
        let packets = encoder.drain().map_err(ExportError::Encoder)?;
        write_packets(mux, stream.get(), &packets, &mut tracks.audio_ticks)?;
    }
    Ok(())
}

fn finish_file<M: Mux>(
    mux: &mut M,
    tracks: &TrackTicks,
) -> Result<cutlass_container::FinishedMux, ExportError> {
    mux.interleave().map_err(ExportError::Mux)?;
    let mut durations = vec![TrackDuration {
        track: tracks.video,
        timescale: Timescale::new(tracks.video_scale),
        ticks: tracks.video_ticks,
    }];
    if let Some(audio) = tracks.audio {
        durations.push(TrackDuration {
            track: audio,
            timescale: Timescale::new(tracks.audio_scale),
            ticks: tracks.audio_ticks,
        });
    }
    mux.finish(&durations)
        .map_err(|failed| ExportError::Mux(failed.reason().clone()))
}

fn stop<M: Mux>(mux: &mut M, error: ExportError) -> ExportError {
    if mux.status() == MuxStatus::Building {
        mux.cancel();
    }
    error
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use cutlass_codec::access::{CodecError, SideData};
    use cutlass_codec::{CodecId, ContainerId};
    use cutlass_container::{IndexPlace, Limits, Mux, MuxDraft, MuxMode, MuxStatus};
    use cutlass_encode::{
        AudioInput, ChannelLayout, CodecParams, EncodeFactory, Encoder, FrameInput, Packet, Pcm,
        PixelFormat,
    };
    use cutlass_eval::{ClipPcm, Snapshot};
    use cutlass_model::{
        Draft, FrameRate, HdrSettings, MaterialAudioFade, MaterialColorConfig, MaterialId,
        MaterialText, Micros, Segment, TimeRange, Track, TrackAttribute, TrackType, WordColumns,
    };
    use cutlass_playback::present;
    use cutlass_renderer_software::SoftwareDevice;

    use super::{ExportRun, export, frame_indexes, same_software_pixels};
    use crate::config::{AudioExport, ExportError, ExportRequest, SidecarKind, SidecarRequest};

    struct Stamp {
        codec: CodecId,
        width: u32,
        height: u32,
        side: bool,
        cancel: Arc<AtomicBool>,
    }

    impl Encoder for Stamp {
        fn codec(&self) -> CodecId {
            self.codec
        }

        fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
            let Some(samples) = input.pcm.as_f32() else {
                return Err(CodecError::InvalidApi { what: "f32" });
            };
            if samples.is_empty() {
                return Ok(Vec::new());
            }
            let duration = i64::from(input.pcm.samples);
            let data = samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            Ok(vec![Packet::audio(data, 0, duration, input.pcm.rate)])
        }

        fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
            let FrameInput::Cpu(frame) = input else {
                return Err(CodecError::Unsupported {
                    what: "platform picture",
                });
            };
            if frame.width() != self.width || frame.height() != self.height {
                return Err(CodecError::InvalidApi { what: "frame size" });
            }
            let mut packet = Packet::video(
                frame.bytes().to_vec(),
                frame.time().pts(),
                frame.time().duration(),
                frame.time().timescale(),
                true,
            );
            if self.side {
                packet.side.push(SideData::Config(vec![1]));
            }
            Ok(vec![packet])
        }

        fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}

        fn cancel_flag(&self) -> Arc<AtomicBool> {
            Arc::clone(&self.cancel)
        }
    }

    struct CopyAudio {
        pts: i64,
        cancel: Arc<AtomicBool>,
    }

    impl Encoder for CopyAudio {
        fn codec(&self) -> CodecId {
            CodecId::Pcm
        }

        fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
            let Some(samples) = input.pcm.as_f32() else {
                return Err(CodecError::InvalidApi { what: "f32" });
            };
            if samples.is_empty() {
                return Ok(Vec::new());
            }
            let duration = i64::from(input.pcm.samples);
            let data = samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            let packet = Packet::audio(data, self.pts, duration, input.pcm.rate);
            self.pts += duration;
            Ok(vec![packet])
        }

        fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
            Err(CodecError::Unsupported { what: "picture" })
        }

        fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}

        fn cancel_flag(&self) -> Arc<AtomicBool> {
            Arc::clone(&self.cancel)
        }
    }

    fn open_rgba(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        if params.bitrate != 2_000_000 || params.pixel != PixelFormat::Rgba8 {
            return Err(CodecError::Unsupported { what: "picture" });
        }
        Ok(Box::new(Stamp {
            codec: params.codec,
            width: params.width,
            height: params.height,
            side: params.codec == CodecId::Vp9,
            cancel: Arc::new(AtomicBool::new(false)),
        }))
    }

    fn open_pcm(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        if params.codec != CodecId::Pcm || params.bitrate != 128_000 {
            return Err(CodecError::Unsupported { what: "pcm" });
        }
        Ok(Box::new(CopyAudio {
            pts: 0,
            cancel: Arc::new(AtomicBool::new(false)),
        }))
    }

    fn decline(_params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        Err(CodecError::Unsupported { what: "platform" })
    }

    fn factory() -> EncodeFactory {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Gif, open_rgba);
        factory.register(CodecId::H264, open_rgba);
        factory.register(CodecId::Vp9, open_rgba);
        factory.register(CodecId::Pcm, open_pcm);
        factory.register(CodecId::Mjpeg, decline);
        factory
    }

    fn gif_request(range: TimeRange) -> ExportRequest {
        ExportRequest {
            range,
            long_side: 240,
            codec: CodecId::Gif,
            bitrate: 2_000_000,
            frame_rate: FrameRate::FPS_30,
            container: ContainerId::Isobmff,
            audio: None,
            sidecar: None,
            late_by: 0,
        }
    }

    fn color_draft(width: i32, height: i32, span: i64) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.duration = Micros(4_000_000);
        draft.canvas_config.width = width;
        draft.canvas_config.height = height;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "blue".into(),
            is_color_clip: true,
            solid_color: "0xFF0000FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "red".into(),
            is_color_clip: true,
            solid_color: "0xFFFF0000".to_owned(),
            ..MaterialColorConfig::default()
        });
        let mut under = Segment::new("under", "blue", TimeRange::new(0, span));
        under.render_index = 1;
        let mut over = Segment::new("over", "red", TimeRange::new(0, span));
        over.render_index = 2;
        draft.tracks[0].segments.push(over);
        draft.tracks[0].segments.push(under);
        draft
    }

    fn red(width: u32, height: u32) -> Vec<u8> {
        let pixels =
            usize::try_from(width).expect("width") * usize::try_from(height).expect("height");
        [255, 0, 0, 255].repeat(pixels)
    }

    fn clear(width: u32, height: u32) -> Vec<u8> {
        vec![0; usize::try_from(width * height * 4).expect("len")]
    }

    fn samples_of(packet: &cutlass_codec::Packet) -> Vec<f32> {
        packet
            .bytes()
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn frame_starts_inside_the_range_are_included_and_the_end_is_not() {
        let rate = FrameRate::FPS_30;
        assert_eq!(rate.frame_start(1), Some(Micros(33_334)));
        assert_eq!(rate.frame_start(2), Some(Micros(66_667)));
        assert_eq!(
            frame_indexes(rate, &TimeRange::new(0, 1_000_000)).expect("second"),
            (0..30).collect::<Vec<_>>()
        );
        assert_eq!(
            frame_indexes(rate, &TimeRange::new(0, 33_334)).expect("first"),
            vec![0]
        );
        assert_eq!(
            frame_indexes(rate, &TimeRange::new(33_334, 33_333)).expect("second frame"),
            vec![1]
        );
        assert_eq!(
            frame_indexes(rate, &TimeRange::new(1, 1)).expect("empty span"),
            Vec::<i64>::new()
        );
        assert_eq!(
            frame_indexes(rate, &TimeRange::new(0, 0)),
            Err(ExportError::Range)
        );
        assert_eq!(
            frame_indexes(rate, &TimeRange::new(0, i64::MAX)),
            Err(ExportError::Limit)
        );

        let drop = FrameRate::FPS_29_97;
        assert_eq!(drop.frame_start(1), Some(Micros(33_367)));
        assert_eq!(
            frame_indexes(drop, &TimeRange::new(0, 33_367)).expect("one"),
            vec![0]
        );
        assert_eq!(
            frame_indexes(drop, &TimeRange::new(0, 33_368)).expect("two"),
            vec![0, 1]
        );
    }

    #[test]
    fn export_encodes_each_eval_frame_and_muxes_it_in_the_file_timescale() {
        let draft = color_draft(240, 135, 33_334);
        let before = draft.clone();
        let request = gif_request(TimeRange::new(0, 66_667));
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let result = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect("export");
        assert_eq!(draft, before);
        assert_eq!(draft.duration, Micros(4_000_000));
        assert_eq!(result.frames.len(), 2);
        assert_eq!(result.frames[0].rgba, red(240, 135));
        assert_eq!(result.frames[1].rgba, clear(240, 135));
        let packets = result.finished.packets();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].bytes(), red(240, 135));
        assert_eq!(packets[1].bytes(), clear(240, 135));
        assert_eq!(packets[0].timestamp().presentation(), Some(0));
        assert_eq!(packets[1].timestamp().presentation(), Some(1));
        assert_eq!(packets[0].timestamp().duration(), Some(1));
        assert_eq!(packets[0].timescale().ticks_per_second(), 30);
        assert!(packets[0].keyframe());
        assert!(result.finished.interleaved());
        assert_eq!(result.finished.durations()[0].ticks, 2);
        assert_eq!(mux.status(), MuxStatus::Finished);

        let mut playback = SoftwareDevice::new();
        let snapshot = Snapshot::new(&draft);
        let first = present(&snapshot, Micros(0), &mut playback).expect("play");
        let second = present(&snapshot, Micros(33_334), &mut playback).expect("play");
        same_software_pixels(&draft, 240, 135, &first.rgba, &result.frames[0].rgba).expect("lock");
        same_software_pixels(&draft, 240, 135, &second.rgba, &result.frames[1].rgba).expect("lock");
        assert_eq!(first.rgba, red(240, 135));
    }

    #[test]
    fn a_late_picture_is_kept_and_a_declining_encoder_is_named() {
        let draft = color_draft(240, 135, 1_000_000);
        let before = draft.clone();
        let mut request = gif_request(TimeRange::new(0, 33_334));
        request.late_by = 50;
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let result = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect("late");
        assert_eq!(result.finished.packets()[0].bytes(), red(240, 135));
        assert_eq!(draft, before);

        let mut declined = gif_request(TimeRange::new(0, 33_334));
        declined.codec = CodecId::Mjpeg;
        declined.long_side = 480;
        let wide = color_draft(480, 270, 1_000_000);
        let before_wide = wide.clone();
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let error = export(
            ExportRun {
                draft: &wide,
                device: &mut device,
                request: &declined,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect_err("decline");
        assert_eq!(
            error,
            ExportError::Encoder(CodecError::Unsupported {
                what: "encoder rejected the parameters",
            })
        );
        assert_eq!(wide, before_wide);
        assert_eq!(mux.status(), MuxStatus::Building);

        declined.codec = CodecId::Av1;
        let error = export(
            ExportRun {
                draft: &wide,
                device: &mut device,
                request: &declined,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect_err("missing");
        assert_eq!(
            error,
            ExportError::Encoder(CodecError::Unsupported {
                what: "encoder is not registered",
            })
        );
    }

    #[test]
    fn packet_side_data_is_refused_and_the_draft_stays() {
        let draft = color_draft(480, 270, 1_000_000);
        let before = draft.clone();
        let mut request = gif_request(TimeRange::new(0, 33_334));
        request.codec = CodecId::Vp9;
        request.long_side = 480;
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let error = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect_err("side");
        assert_eq!(error, ExportError::SideData);
        assert_eq!(draft, before);
        assert_eq!(mux.status(), MuxStatus::Failed);
    }

    #[test]
    fn hdr_and_yuv_are_not_reported_as_a_software_match() {
        let mut draft = color_draft(2, 2, 1_000);
        let rgba = red(2, 2);
        same_software_pixels(&draft, 2, 2, &rgba, &rgba).expect("sdr");
        draft.tracks[0].segments[0].hdr_settings = Some(HdrSettings {
            mode: 6,
            intensity: 1.0,
            nits: 0.0,
            ..HdrSettings::default()
        });
        assert_eq!(
            same_software_pixels(&draft, 2, 2, &rgba, &rgba),
            Err(ExportError::HdrNotThisCase)
        );
        draft.tracks[0].segments[0].hdr_settings = None;
        draft.color_space = 2084;
        assert_eq!(
            same_software_pixels(&draft, 2, 2, &rgba, &rgba),
            Err(ExportError::HdrNotThisCase)
        );
        let yuv = vec![0_u8; 6];
        assert_eq!(
            same_software_pixels(&color_draft(2, 2, 1_000), 2, 2, &yuv, &yuv),
            Err(ExportError::NotSoftwareRgba)
        );
        let other = clear(2, 2);
        assert_eq!(
            same_software_pixels(&color_draft(2, 2, 1_000), 2, 2, &rgba, &other),
            Err(ExportError::PixelMismatch)
        );
    }

    #[test]
    fn a_hostile_packet_and_the_wrong_container_fail_closed() {
        let draft = color_draft(240, 135, 1_000_000);
        let before = draft.clone();
        let request = gif_request(TimeRange::new(0, 33_334));
        let mut limits = Limits::standard();
        limits.max_packet_bytes = 1;
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            limits,
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let error = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect_err("hostile");
        assert!(matches!(error, ExportError::Mux(_)));
        assert_eq!(mux.status(), MuxStatus::Failed);
        assert_eq!(draft, before);

        let mut mux = MuxDraft::new(
            ContainerId::Matroska,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let error = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect_err("family");
        assert_eq!(error, ExportError::Container);
        assert_eq!(mux.status(), MuxStatus::Building);
        assert_eq!(draft, before);
    }

    fn audio_settings() -> AudioExport {
        AudioExport {
            codec: CodecId::Pcm,
            bitrate: 128_000,
            sample_rate: 4_000,
            channels: 1,
            limiter: 0,
        }
    }

    fn tone(value: f32) -> Pcm {
        Pcm::from_f32(4_000, ChannelLayout::mono(), vec![value; 4]).expect("pcm")
    }

    fn push_sound(draft: &mut Draft, id: &str, volume: f64, attribute: i32, fade: bool) {
        let mut track = Track::main(id);
        track.kind = TrackType::Audio;
        track.attribute = TrackAttribute(attribute);
        let mut segment = Segment::new(id, id, TimeRange::new(0, 1_000_000));
        segment.volume = volume;
        if fade {
            segment.extra_material_refs.push(MaterialId::new("fade"));
            draft.materials.audio_fades.push(MaterialAudioFade {
                id: MaterialId::new("fade"),
                fade_type: 2,
                fade_in_duration: Micros(1_000),
                fade_out_duration: Micros(0),
                ..MaterialAudioFade::default()
            });
        }
        track.segments.push(segment);
        draft.tracks.push(track);
    }

    fn audio_bytes(draft: &Draft, request: &ExportRequest) -> Vec<f32> {
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let result = export(
            ExportRun {
                draft,
                device: &mut device,
                request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |segment, frames, rate| {
                assert_eq!(frames, 4);
                assert_eq!(rate, 4_000);
                let value = if segment.as_str() == "other" {
                    1.0
                } else {
                    0.25
                };
                if segment.as_str() == "loud" {
                    Ok(ClipPcm::Ready(tone(1.0)))
                } else {
                    Ok(ClipPcm::Ready(tone(value)))
                }
            },
        )
        .expect("audio");
        let audio = result
            .finished
            .packets()
            .iter()
            .find(|packet| packet.stream() == 1)
            .expect("audio packet");
        assert_eq!(audio.timescale().ticks_per_second(), 4_000);
        samples_of(audio)
    }

    #[test]
    fn exported_audio_is_the_mix_with_mute_solo_volume_and_fades() {
        let mut request = gif_request(TimeRange::new(0, 1_000));
        request.frame_rate = FrameRate {
            numerator: 1_000,
            denominator: 1,
        };
        request.audio = Some(audio_settings());

        let mut muted = color_draft(240, 135, 1_000_000);
        let before = muted.clone();
        push_sound(&mut muted, "loud", 1.0, TrackAttribute::MUTE, false);
        assert_eq!(audio_bytes(&muted, &request), vec![0.0; 4]);
        assert_eq!(muted.duration, before.duration);

        let mut gained = color_draft(240, 135, 1_000_000);
        push_sound(&mut gained, "loud", 0.5, 0, false);
        assert_eq!(audio_bytes(&gained, &request), vec![0.5; 4]);

        let mut faded = color_draft(240, 135, 1_000_000);
        push_sound(&mut faded, "loud", 1.0, 0, true);
        assert_eq!(audio_bytes(&faded, &request), vec![0.0, 0.25, 0.5, 0.75]);

        let mut solo = color_draft(240, 135, 1_000_000);
        push_sound(&mut solo, "heard", 1.0, TrackAttribute::SOLO, false);
        push_sound(&mut solo, "other", 1.0, TrackAttribute::MUTE_BY_SOLO, false);
        assert_eq!(audio_bytes(&solo, &request), vec![0.25; 4]);

        let mut summed = color_draft(240, 135, 1_000_000);
        push_sound(&mut summed, "left", 1.0, 0, false);
        push_sound(&mut summed, "right", 1.0, 0, false);
        assert_eq!(audio_bytes(&summed, &request), vec![0.5; 4]);

        let mut missing = color_draft(240, 135, 1_000_000);
        push_sound(&mut missing, "loud", 1.0, 0, false);
        let before_missing = missing.clone();
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let error = export(
            ExportRun {
                draft: &missing,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect_err("wait");
        assert_eq!(
            error,
            ExportError::Audio(cutlass_eval::AudioError::NotReady)
        );
        assert_eq!(missing, before_missing);
    }

    #[test]
    fn a_sidecar_is_srt_or_plain_text_and_the_draft_stays() {
        let mut draft = color_draft(240, 135, 1_000_000);
        draft.duration = Micros(9_000);
        let mut text = MaterialText::new("words");
        text.font_name = "StyledFace".to_owned();
        text.words = WordColumns {
            start_time: vec![0],
            end_time: vec![500],
            text: vec!["Hello".to_owned()],
            ..WordColumns::default()
        };
        draft.materials.texts.push(text);
        draft.tracks[0].segments.push(Segment::new(
            "caption",
            "words",
            TimeRange::new(0, 1_000_000),
        ));
        let before = draft.clone();
        let dir = std::env::temp_dir().join(format!("cutlass-export-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let mut request = gif_request(TimeRange::new(0, 33_334));
        request.sidecar = Some(SidecarRequest {
            kind: SidecarKind::Srt,
            name: "captions".to_owned(),
        });
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let result = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: Some(&dir),
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect("sidecar");
        let path = result.sidecar.expect("path");
        let body = std::fs::read_to_string(&path).expect("read");
        assert!(body.contains("Hello"));
        assert!(body.contains("-->"));
        assert!(!body.contains("StyledFace"));
        assert_eq!(draft, before);
        assert_eq!(draft.duration, Micros(9_000));

        request.sidecar.as_mut().expect("sidecar").kind = SidecarKind::Ass;
        let error = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut MuxDraft::new(
                    ContainerId::Isobmff,
                    MuxMode::Progressive,
                    IndexPlace::End,
                    Limits::standard(),
                ),
                destination: Some(&dir),
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect_err("ass");
        assert_eq!(error, ExportError::Ass);
        assert_eq!(draft, before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scaled_output_keeps_the_eval_pixels_and_encodes_the_fitted_size() {
        let draft = color_draft(480, 240, 1_000_000);
        let request = gif_request(TimeRange::new(0, 33_334));
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let result = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect("scale");
        assert_eq!(result.output_width, 240);
        assert_eq!(result.output_height, 120);
        assert_eq!(result.frames[0].rgba, red(480, 240));
        assert_eq!(result.finished.packets()[0].bytes(), red(240, 120));
        let mut playback = SoftwareDevice::new();
        let played = present(&Snapshot::new(&draft), Micros(0), &mut playback).expect("play");
        assert_eq!(played.rgba, result.frames[0].rgba);
    }

    #[test]
    fn an_audio_window_past_the_sample_cap_is_refused() {
        let draft = color_draft(240, 135, 1_000_000);
        let before = draft.clone();
        let mut request = gif_request(TimeRange::new(0, 1_000_000));
        request.frame_rate = FrameRate {
            numerator: 1,
            denominator: 1,
        };
        request.audio = Some(AudioExport {
            codec: CodecId::Pcm,
            bitrate: 128_000,
            sample_rate: u32::MAX,
            channels: 1,
            limiter: 0,
        });
        let mut mux = MuxDraft::new(
            ContainerId::Isobmff,
            MuxMode::Progressive,
            IndexPlace::End,
            Limits::standard(),
        );
        let mut device = SoftwareDevice::new();
        let factory = factory();
        let error = export(
            ExportRun {
                draft: &draft,
                device: &mut device,
                request: &request,
                factory: &factory,
                mux: &mut mux,
                destination: None,
            },
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect_err("limit");
        assert_eq!(error, ExportError::Limit);
        assert_eq!(draft, before);
    }
}
