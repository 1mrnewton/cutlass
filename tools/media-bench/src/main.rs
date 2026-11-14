//! Runs the Rust codecs and containers on one file, end to end.
//!
//! `media-bench decode picture <file>` walks every picture of the first
//! picture stream, in display order. `media-bench decode sound <file>` walks
//! every buffer of the first sound stream. Nothing is converted, resized or
//! kept: the decoder's own samples are counted and dropped, which is the work
//! `ffmpeg -i <file> -f null -` does. `--dump <path>` also writes those
//! samples, for a check against another decoder; a timed run leaves it out.
//! A still is dumped as 16-bit little-endian values, one per sample.
//!
//! `media-bench decode ivf <file>` does the same for a VP8 or VP9 stream in
//! an IVF file, a container the desktop does not open: the frames go
//! straight to the decoder.
//!
//! `media-bench demux <file>` reads every packet of every stream and decodes
//! none. `media-bench encode picture|sound ...` encodes raw 4:2:0 pictures
//! (or raw RGBA ones with `--rgba`, for an encoder that takes those) or raw
//! float sound read from a file and writes the packets' bytes to `--out`; `--decoded <path>` also decodes those packets again with the
//! Rust decoder and writes the pictures, to measure what the encoder kept.
//!
//! Only the Rust codecs are registered, so a platform codec never takes the
//! work. The containers are the ones the desktop opens, and ADTS.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use cutlass_codec::ContainerId;
use cutlass_codec::access::{ChannelLayout, CodecParams, Pcm, PixelFormat, SideData};
use cutlass_container::{Demux, Input, OpenError, ProbeResult};
use cutlass_decode::{CodecId, DecodeFactory, Output};
use cutlass_encode::{AudioInput, CpuFrame, EncodeFactory, FrameInput, FrameTime};
use cutlass_engine::{DecodedFrame, ReaderFactory};

const USAGE: &str = "usage: media-bench decode <picture|sound|ivf> <file> [--dump <path>] [--repeat <n>]
       media-bench demux <file>
       media-bench encode picture <codec> <width> <height> <fps> <bitrate> <raw> [--out <path>] [--decoded <path>] [--rgba]
       media-bench encode sound <codec> <rate> <channels> <bitrate> <raw.f32> [--out <path>]
       media-bench compare <u8|u16|i16|f32> <ours> <theirs> [--channels <n>]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(line) => {
            let _ = writeln!(std::io::stdout(), "{line}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            let _ = writeln!(std::io::stderr(), "media-bench: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<String, String> {
    if let [command, file] = args
        && command == "demux"
    {
        return demux(Path::new(file));
    }
    if let [command, rest @ ..] = args
        && command == "encode"
    {
        return encode(rest);
    }
    let [command, kind, file, rest @ ..] = args else {
        return Err(USAGE.to_owned());
    };
    if command == "compare" {
        let [theirs, rest @ ..] = rest else {
            return Err(USAGE.to_owned());
        };
        let channels = match rest {
            [] => 1,
            [flag, count] if flag == "--channels" => count.parse().map_err(|_| USAGE.to_owned())?,
            _ => return Err(USAGE.to_owned()),
        };
        return compare(kind, Path::new(file), Path::new(theirs), channels);
    }
    if command != "decode" {
        return Err(USAGE.to_owned());
    }
    let mut dump = None;
    let mut repeat = 1_u32;
    let mut rest = rest;
    while let [flag, value, tail @ ..] = rest {
        match flag.as_str() {
            "--dump" => dump = Some(PathBuf::from(value)),
            "--repeat" => repeat = value.parse().map_err(|_| USAGE.to_owned())?,
            _ => return Err(USAGE.to_owned()),
        }
        rest = tail;
    }
    if !rest.is_empty() || repeat == 0 {
        return Err(USAGE.to_owned());
    }
    let mut sink = match dump {
        Some(path) => Some(BufWriter::new(
            File::create(&path).map_err(|error| error.to_string())?,
        )),
        None => None,
    };
    let path = Path::new(file);
    let mut line = String::new();
    // A repeat opens the file again each time, as a player that is handed
    // the same still again does.
    for _ in 0..repeat {
        line = match kind.as_str() {
            "picture" => pictures(path, sink.as_mut())?,
            "sound" => sound(path, sink.as_mut())?,
            "ivf" => ivf(path, sink.as_mut())?,
            _ => return Err(USAGE.to_owned()),
        };
    }
    if let Some(mut sink) = sink {
        sink.flush().map_err(|error| error.to_string())?;
    }
    Ok(line)
}

/// How close two dumps of raw samples are.
///
/// Sound that differs in length is lined up first: the shorter dump is
/// slid over the longer one, up to 8192 sample frames, and the best fit is
/// reported with its lag. `identical` means every byte is the same.
fn compare(kind: &str, ours: &Path, theirs: &Path, channels: usize) -> Result<String, String> {
    let ours = std::fs::read(ours).map_err(|error| error.to_string())?;
    let theirs = std::fs::read(theirs).map_err(|error| error.to_string())?;
    if ours == theirs {
        return Ok(format!("identical bytes {}", ours.len()));
    }
    let (left, right, peak): (Vec<f64>, Vec<f64>, f64) = match kind {
        "u8" => (
            ours.iter().map(|byte| f64::from(*byte)).collect(),
            theirs.iter().map(|byte| f64::from(*byte)).collect(),
            255.0,
        ),
        "u16" => (wide(&ours, false), wide(&theirs, false), 65535.0),
        "i16" => (wide(&ours, true), wide(&theirs, true), 32768.0),
        "f32" => (floats(&ours), floats(&theirs), 1.0),
        _ => return Err(USAGE.to_owned()),
    };
    if left.len() == right.len() {
        return Ok(format!(
            "{} samples {}",
            closeness(&left, &right, peak),
            left.len()
        ));
    }
    if kind == "u8" || kind == "u16" {
        return Err(format!(
            "picture dumps differ in size: {} and {} samples",
            left.len(),
            right.len()
        ));
    }
    let channels = channels.max(1);
    let (long, short) = if left.len() >= right.len() {
        (&left, &right)
    } else {
        (&right, &left)
    };
    let window = short.len().min(48_000 * channels);
    let mut best = (f64::MAX, 0_usize);
    for lag in 0..=8192_usize {
        let at = lag * channels;
        let (Some(a), Some(b)) = (long.get(at..at + window), short.get(..window)) else {
            break;
        };
        let error: f64 = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum();
        if error < best.0 {
            best = (error, lag);
        }
    }
    let at = best.1 * channels;
    let span = short.len().min(long.len() - at);
    let (Some(a), Some(b)) = (long.get(at..at + span), short.get(..span)) else {
        return Err("the dumps do not overlap".to_owned());
    };
    Ok(format!(
        "{} samples {} lag {} frames, lengths {} and {}",
        closeness(a, b, peak),
        span,
        best.1,
        left.len() / channels,
        right.len() / channels
    ))
}

fn wide(bytes: &[u8], signed: bool) -> Vec<f64> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            if signed {
                f64::from(i16::from_le_bytes(*pair))
            } else {
                f64::from(u16::from_le_bytes(*pair))
            }
        })
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|quad| f64::from(f32::from_le_bytes(*quad)))
        .collect()
}

/// Peak signal-to-noise ratio in decibels and the largest single difference.
fn closeness(left: &[f64], right: &[f64], peak: f64) -> String {
    let mut sum = 0.0_f64;
    let mut worst = 0.0_f64;
    for (a, b) in left.iter().zip(right) {
        let diff = (a - b).abs();
        sum += diff * diff;
        worst = worst.max(diff);
    }
    if sum == 0.0 {
        return "equal".to_owned();
    }
    let mean = sum / left.len().max(1) as f64;
    let psnr = 10.0 * (peak * peak / mean).log10();
    format!("psnr {psnr:.2} dB worst {worst}")
}

/// Every picture of the file, counted and dropped.
fn pictures(path: &Path, mut sink: Option<&mut BufWriter<File>>) -> Result<String, String> {
    let mut media = readers().open(path).map_err(|error| format!("{error:?}"))?;
    let mut count = 0_u64;
    let mut bytes = 0_u64;
    let mut shape = String::new();
    while let Some(held) = media.next_picture().map_err(|error| format!("{error:?}"))? {
        let wide: Vec<u8>;
        let data: &[u8] = match &held.frame {
            DecodedFrame::Coded {
                width,
                height,
                pixel,
                bytes,
                ..
            } => {
                if shape.is_empty() {
                    shape = format!("{width}x{height} {pixel:?}");
                }
                bytes.as_slice()
            }
            DecodedFrame::Still(image) => {
                if shape.is_empty() {
                    shape = format!(
                        "{}x{} {:?} {:?}",
                        image.width, image.height, image.color, image.depth
                    );
                }
                // A still keeps one 16-bit value per sample, whatever its
                // depth. Only a dump needs them as bytes.
                wide = if sink.is_some() {
                    image
                        .samples
                        .iter()
                        .flat_map(|sample| sample.to_le_bytes())
                        .collect()
                } else {
                    Vec::new()
                };
                std::hint::black_box(image.samples.as_slice());
                bytes += (image.samples.len() * 2) as u64;
                wide.as_slice()
            }
            DecodedFrame::Platform { .. } | DecodedFrame::Surface { .. } => {
                return Err("a platform decoder took the file".to_owned());
            }
        };
        count += 1;
        if matches!(held.frame, DecodedFrame::Coded { .. }) {
            bytes += data.len() as u64;
        }
        if let Some(sink) = sink.as_deref_mut() {
            sink.write_all(data).map_err(|error| error.to_string())?;
        }
        std::hint::black_box(data);
    }
    if count == 0 {
        return Err("no picture decoded".to_owned());
    }
    Ok(format!("pictures {count} bytes {bytes} {shape}"))
}

/// Every picture of a VP8 or VP9 stream in an IVF file, counted and dropped.
///
/// IVF is a 32-byte header, then a 12-byte header before each frame: the
/// frame's length and its time, little-endian.
fn ivf(path: &Path, mut sink: Option<&mut BufWriter<File>>) -> Result<String, String> {
    let file = std::fs::read(path).map_err(|error| error.to_string())?;
    let header = file.get(..32).ok_or("the IVF header is short")?;
    if &header[..4] != b"DKIF" {
        return Err("not an IVF file".to_owned());
    }
    let width = u32::from(u16::from_le_bytes([header[12], header[13]]));
    let height = u32::from(u16::from_le_bytes([header[14], header[15]]));
    let rate = u32::from_le_bytes([header[16], header[17], header[18], header[19]]).max(1);
    let codec = match &header[8..12] {
        b"VP80" => CodecId::Vp8,
        b"VP90" => CodecId::Vp9,
        _ => return Err("the IVF stream is not VP8 or VP9".to_owned()),
    };
    let params = CodecParams::video(codec, width, height);
    let mut plain = None;
    let mut video = None;
    if codec == CodecId::Vp8 {
        plain = Some(
            rust_decoders()
                .open(&params)
                .map_err(|error| format!("{error:?}"))?,
        );
    } else {
        let mut factory = cutlass_decode::VideoDecodeFactory::new();
        cutlass_vp9::register_decoders(&mut factory);
        video = Some(
            factory
                .open(&params)
                .map_err(|error| format!("{error:?}"))?,
        );
    }
    let mut count = 0_u64;
    let mut bytes = 0_u64;
    let mut emit = |planes: &[&[u8]]| -> Result<(), String> {
        count += 1;
        for plane in planes {
            bytes += plane.len() as u64;
            if let Some(sink) = sink.as_deref_mut() {
                sink.write_all(plane).map_err(|error| error.to_string())?;
            }
            std::hint::black_box(plane);
        }
        Ok(())
    };
    let mut at = 32;
    let mut index = 0_i64;
    let mut pictures = Vec::new();
    while let Some(lead) = file.get(at..at + 12) {
        let length = u32::from_le_bytes([lead[0], lead[1], lead[2], lead[3]]) as usize;
        let data = file
            .get(at + 12..at + 12 + length)
            .ok_or("an IVF frame is cut short")?;
        at += 12 + length;
        let packet = cutlass_decode::Packet {
            stream: 0,
            data: data.to_vec(),
            pts: index,
            duration: 1,
            timescale: rate,
            keyframe: codec == CodecId::Vp8 && data.first().is_some_and(|byte| byte & 1 == 0),
            side: Vec::new(),
        };
        index += 1;
        if let Some(decoder) = plain.as_mut() {
            for data in on_cpu(
                decoder
                    .push(&packet)
                    .map_err(|error| format!("{error:?}"))?,
            )? {
                emit(&[data.as_slice()])?;
            }
        } else if let Some(decoder) = video.as_mut() {
            pictures.extend(
                decoder
                    .push(&packet)
                    .map_err(|error| format!("{error:?}"))?,
            );
            for output in pictures.drain(..) {
                emit(&tight(&output.frame))?;
            }
        }
    }
    if let Some(decoder) = plain.as_mut() {
        for data in on_cpu(decoder.drain().map_err(|error| format!("{error:?}"))?)? {
            emit(&[data.as_slice()])?;
        }
    } else if let Some(decoder) = video.as_mut() {
        for output in decoder.drain().map_err(|error| format!("{error:?}"))? {
            emit(&tight(&output.frame))?;
        }
    }
    if count == 0 {
        return Err("no picture decoded".to_owned());
    }
    Ok(format!(
        "pictures {count} bytes {bytes} {width}x{height} Yuv420p"
    ))
}

/// The bytes of each picture in `outputs`.
fn on_cpu(outputs: Vec<Output>) -> Result<Vec<Vec<u8>>, String> {
    outputs
        .into_iter()
        .filter_map(|output| match output {
            Output::Picture(picture) => Some(picture),
            Output::Audio(_) => None,
        })
        .map(|picture| {
            picture
                .into_cpu()
                .ok_or_else(|| "the picture is not on the CPU".to_owned())
        })
        .collect()
}

/// The three planes of a frame, row by row without any padding.
fn tight(frame: &cutlass_codec::access::VideoFrame) -> Vec<&[u8]> {
    let mut rows = Vec::new();
    let bytes = if matches!(
        frame.format,
        PixelFormat::Yuv420p10
            | PixelFormat::Yuv420p12
            | PixelFormat::Yuv422p10
            | PixelFormat::Yuv422p12
            | PixelFormat::Yuv440p10
            | PixelFormat::Yuv440p12
            | PixelFormat::Yuv444p10
            | PixelFormat::Yuv444p12
            | PixelFormat::Gbrp10
            | PixelFormat::Gbrp12
    ) {
        2
    } else {
        1
    };
    for plane in [&frame.y, &frame.u, &frame.v] {
        let width = plane.width as usize * bytes;
        if plane.stride == width {
            rows.push(plane.samples.as_slice());
            continue;
        }
        for row in plane
            .samples
            .chunks(plane.stride.max(1))
            .take(plane.height as usize)
        {
            rows.push(row.get(..width).unwrap_or(row));
        }
    }
    rows
}

/// Every sound buffer of the file, counted and dropped. A dump is
/// interleaved little-endian samples in the decoder's own format.
fn sound(path: &Path, mut sink: Option<&mut BufWriter<File>>) -> Result<String, String> {
    let mut media = readers().open(path).map_err(|error| format!("{error:?}"))?;
    let mut frames = 0_u64;
    let mut shape = String::new();
    while let Some(ready) = media.pull_audio().map_err(|error| format!("{error:?}"))? {
        let pcm = &ready.pcm;
        if shape.is_empty() {
            shape = format!(
                "{} Hz {} channels {:?}",
                pcm.rate,
                pcm.layout.len(),
                pcm.format
            );
        }
        frames += u64::from(pcm.samples);
        if let Some(sink) = sink.as_deref_mut() {
            if let Some(floats) = pcm.as_f32() {
                for sample in floats {
                    sink.write_all(&sample.to_le_bytes())
                        .map_err(|error| error.to_string())?;
                }
            } else if let Some(ints) = pcm.as_i16() {
                for sample in ints {
                    sink.write_all(&sample.to_le_bytes())
                        .map_err(|error| error.to_string())?;
                }
            }
        }
        std::hint::black_box(pcm);
    }
    if frames == 0 {
        return Err("no sound decoded".to_owned());
    }
    Ok(format!("sound frames {frames} {shape}"))
}

/// Every packet of every stream, read and dropped. Nothing is decoded.
fn demux(path: &Path) -> Result<String, String> {
    type Probe = fn(&[u8]) -> ProbeResult;
    type Open = fn(&Path) -> Result<Box<dyn Demux>, OpenError>;
    let families: &[(Probe, Open)] = &[
        (probe_isobmff, open_isobmff),
        (cutlass_avi::probe, open_avi),
        (cutlass_riff::probe, open_wav),
        (cutlass_aiff::probe, open_aiff),
        (cutlass_caf::probe, open_caf),
        (cutlass_flac_container::probe, open_flac),
        (cutlass_ogg::probe, open_ogg),
        (cutlass_mpegts::probe, open_ts),
        (cutlass_matroska::probe, open_matroska),
        (cutlass_still::probe, open_still),
        (probe_adts, open_adts),
        (probe_mp3, open_mp3),
    ];
    let mut header = vec![0_u8; 65_536];
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut filled = 0;
    while filled < header.len() {
        match file.read(&mut header[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) => return Err(error.to_string()),
        }
    }
    header.truncate(filled);
    drop(file);
    let open = families
        .iter()
        .find(|(probe, _)| matches!(probe(&header), ProbeResult::Accept { .. }))
        .map(|(_, open)| *open)
        .ok_or("no container took the file")?;
    let mut demux = open(path).map_err(|error| error.to_string())?;
    let streams = demux.streams().len();
    let mut packets = 0_u64;
    let mut bytes = 0_u64;
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        packets += 1;
        bytes += packet.bytes().len() as u64;
        std::hint::black_box(packet.bytes());
    }
    Ok(format!("packets {packets} bytes {bytes} streams {streams}"))
}

fn codec_id(name: &str) -> Result<CodecId, String> {
    Ok(match name {
        "h264" => CodecId::H264,
        "av1" => CodecId::Av1,
        "vp8" => CodecId::Vp8,
        "mpeg4" => CodecId::Mpeg4,
        "mpeg2" => CodecId::Mpeg2,
        "mjpeg" => CodecId::Mjpeg,
        "prores" => CodecId::Prores,
        "aac" => CodecId::Aac,
        "opus" => CodecId::Opus,
        "mp3" => CodecId::Mp3,
        "flac" => CodecId::Flac,
        "pcm" => CodecId::Pcm,
        "vorbis" => CodecId::Vorbis,
        "alac" => CodecId::Alac,
        "ac3" => CodecId::Ac3,
        "eac3" => CodecId::Eac3,
        _ => return Err(format!("no encoder named {name}")),
    })
}

fn number<T: std::str::FromStr>(text: &str) -> Result<T, String> {
    text.parse().map_err(|_| USAGE.to_owned())
}

fn encode(args: &[String]) -> Result<String, String> {
    let [kind, codec, a, b, c, d, input, rest @ ..] = args else {
        // Sound has one argument fewer.
        if let [kind, codec, rate, channels, bitrate, input, rest @ ..] = args
            && kind == "sound"
        {
            return encode_sound(
                codec_id(codec)?,
                number(rate)?,
                number(channels)?,
                number(bitrate)?,
                Path::new(input),
                rest,
            );
        }
        return Err(USAGE.to_owned());
    };
    if kind == "sound" {
        let mut tail = vec![d.clone(), input.clone()];
        tail.extend_from_slice(rest);
        let [input, rest @ ..] = tail.as_slice() else {
            return Err(USAGE.to_owned());
        };
        return encode_sound(
            codec_id(codec)?,
            number(a)?,
            number(b)?,
            number(c)?,
            Path::new(input),
            rest,
        );
    }
    if kind != "picture" {
        return Err(USAGE.to_owned());
    }
    encode_pictures(
        codec_id(codec)?,
        number(a)?,
        number(b)?,
        number(c)?,
        number(d)?,
        Path::new(input),
        rest,
    )
}

fn flag<'a>(rest: &'a [String], name: &str) -> Option<&'a str> {
    rest.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
}

/// Raw 4:2:0 or RGBA pictures in, the encoder's packets out.
fn encode_pictures(
    codec: CodecId,
    width: u32,
    height: u32,
    fps: u32,
    bitrate: u32,
    input: &Path,
    rest: &[String],
) -> Result<String, String> {
    let rgba = rest.iter().any(|flag| flag == "--rgba");
    let pixel = if rgba {
        PixelFormat::Rgba8
    } else {
        PixelFormat::Yuv420p
    };
    let mut params = CodecParams::with_format(codec, width, height, pixel);
    params.bitrate = bitrate;
    let mut encoder = rust_encoders()
        .open(&params)
        .map_err(|error| format!("{error:?}"))?;
    let pixels = (width as usize) * (height as usize);
    let frame_len = if rgba { pixels * 4 } else { pixels * 3 / 2 };
    let mut source = File::open(input).map_err(|error| error.to_string())?;
    let mut out = match flag(rest, "--out") {
        Some(path) => Some(BufWriter::new(
            File::create(path).map_err(|error| error.to_string())?,
        )),
        None => None,
    };
    let keep = flag(rest, "--decoded").is_some();
    let mut kept = Vec::new();
    let mut frames = 0_i64;
    let mut count = 0_u64;
    let mut bytes = 0_u64;
    let mut take = |packets: Vec<cutlass_encode::Packet>| -> Result<(), String> {
        for packet in packets {
            count += 1;
            bytes += packet.data.len() as u64;
            if let Some(out) = out.as_mut() {
                out.write_all(&packet.data)
                    .map_err(|error| error.to_string())?;
            }
            if keep {
                kept.push(packet);
            }
        }
        Ok(())
    };
    loop {
        let mut picture = vec![0_u8; frame_len];
        match source.read_exact(&mut picture) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.to_string()),
        }
        let time = FrameTime::new(frames, 1, fps).map_err(|error| format!("{error:?}"))?;
        let frame = CpuFrame::new(width, height, pixel, picture, time)
            .map_err(|error| format!("{error:?}"))?;
        take(
            encoder
                .push_frame(&FrameInput::Cpu(frame))
                .map_err(|error| format!("{error:?}"))?,
        )?;
        frames += 1;
    }
    take(encoder.drain().map_err(|error| format!("{error:?}"))?)?;
    if let Some(mut out) = out {
        out.flush().map_err(|error| error.to_string())?;
    }
    if let Some(path) = flag(rest, "--decoded") {
        decode_again(&params, &kept, Path::new(path))?;
    }
    Ok(format!("pictures {frames} packets {count} bytes {bytes}"))
}

/// The encoder's own packets through the Rust decoder, written as pictures.
fn decode_again(
    params: &CodecParams,
    packets: &[cutlass_encode::Packet],
    path: &Path,
) -> Result<(), String> {
    let mut params = params.clone();
    if let Some(config) = packets
        .iter()
        .flat_map(|packet| &packet.side)
        .find_map(|side| match side {
            SideData::Config(bytes) => Some(bytes.clone()),
            _ => None,
        })
    {
        params.extra = config;
    }
    let mut decoder = rust_decoders()
        .open(&params)
        .map_err(|error| format!("decode again: {error:?}"))?;
    let mut sink = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    let mut write = |outputs: Vec<Output>| -> Result<(), String> {
        for output in outputs {
            if let Output::Picture(picture) = output {
                let bytes = picture.into_cpu().ok_or("the picture is not on the CPU")?;
                sink.write_all(&bytes).map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    };
    for packet in packets {
        write(
            decoder
                .push(packet)
                .map_err(|error| format!("decode again: {error:?}"))?,
        )?;
    }
    write(
        decoder
            .drain()
            .map_err(|error| format!("decode again: {error:?}"))?,
    )?;
    sink.flush().map_err(|error| error.to_string())
}

/// Raw interleaved little-endian float sound in, the encoder's packets out.
fn encode_sound(
    codec: CodecId,
    rate: u32,
    channels: usize,
    bitrate: u32,
    input: &Path,
    rest: &[String],
) -> Result<String, String> {
    let layout = match channels {
        1 => ChannelLayout::mono(),
        2 => ChannelLayout::stereo(),
        _ => return Err("one or two channels".to_owned()),
    };
    let mut params = CodecParams::audio(codec, rate, layout.clone());
    params.bitrate = bitrate;
    let mut encoder = rust_encoders()
        .open(&params)
        .map_err(|error| format!("{error:?}"))?;
    let mut source = File::open(input).map_err(|error| error.to_string())?;
    let mut out = match flag(rest, "--out") {
        Some(path) => Some(BufWriter::new(
            File::create(path).map_err(|error| error.to_string())?,
        )),
        None => None,
    };
    // One second of sound a push, as a mixer hands it over.
    let mut chunk = vec![0_u8; rate as usize * channels * 4];
    let mut frames = 0_u64;
    let mut count = 0_u64;
    let mut bytes = 0_u64;
    let mut take = |packets: Vec<cutlass_encode::Packet>| -> Result<(), String> {
        for packet in packets {
            count += 1;
            bytes += packet.data.len() as u64;
            if let Some(out) = out.as_mut() {
                out.write_all(&packet.data)
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    };
    loop {
        let mut filled = 0;
        while filled < chunk.len() {
            match source.read(&mut chunk[filled..]) {
                Ok(0) => break,
                Ok(read) => filled += read,
                Err(error) => return Err(error.to_string()),
            }
        }
        let whole = filled - filled % (channels * 4);
        if whole == 0 {
            break;
        }
        let samples: Vec<f32> = chunk[..whole]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|quad| f32::from_le_bytes(*quad))
            .collect();
        frames += (samples.len() / channels) as u64;
        let pcm =
            Pcm::from_f32(rate, layout.clone(), samples).map_err(|error| format!("{error:?}"))?;
        take(
            encoder
                .push(&AudioInput { pcm })
                .map_err(|error| format!("{error:?}"))?,
        )?;
        if filled < chunk.len() {
            break;
        }
    }
    take(encoder.drain().map_err(|error| format!("{error:?}"))?)?;
    if let Some(mut out) = out {
        out.flush().map_err(|error| error.to_string())?;
    }
    Ok(format!(
        "sound frames {frames} packets {count} bytes {bytes}"
    ))
}

fn rust_encoders() -> EncodeFactory {
    let mut encoders = EncodeFactory::new();
    cutlass_flac::register_encoders(&mut encoders);
    cutlass_alac::register_encoders(&mut encoders);
    cutlass_mp3::register_encoders(&mut encoders);
    cutlass_ac3::register_encoders(&mut encoders);
    cutlass_opus::register_encoders(&mut encoders);
    cutlass_vorbis::register_encoders(&mut encoders);
    cutlass_mjpeg::register_encoders(&mut encoders);
    cutlass_aac::register_encoders(&mut encoders);
    cutlass_av1::register_encoders(&mut encoders);
    cutlass_pcm::register_encoders(&mut encoders);
    cutlass_h264::register_encoders(&mut encoders);
    cutlass_prores::register_encoders(&mut encoders);
    cutlass_vp8::register_engine_encoders(&mut encoders);
    cutlass_mpeg2::register_encoders(&mut encoders);
    cutlass_mpeg4::register_engine_encoders(&mut encoders);
    encoders
}

/// The containers the desktop opens, decoded with the Rust codecs alone.
fn readers() -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe_isobmff, open_isobmff);
    readers.register_container(cutlass_avi::probe, open_avi);
    readers.register_container(cutlass_riff::probe, open_wav);
    readers.register_container(cutlass_aiff::probe, open_aiff);
    readers.register_container(cutlass_caf::probe, open_caf);
    readers.register_container(cutlass_flac_container::probe, open_flac);
    readers.register_container(cutlass_ogg::probe, open_ogg);
    readers.register_container(cutlass_mpegts::probe, open_ts);
    readers.register_container(cutlass_matroska::probe, open_matroska);
    readers.register_container(cutlass_still::probe, open_still);
    readers.register_container(probe_mp3, open_mp3);
    *readers.decoders_mut() = rust_decoders();
    readers
}

fn rust_decoders() -> DecodeFactory {
    let mut decoders = DecodeFactory::new();
    cutlass_flac::register_decoders(&mut decoders);
    cutlass_alac::register_decoders(&mut decoders);
    cutlass_mp3::register_decoders(&mut decoders);
    cutlass_ac3::register_decoders(&mut decoders);
    cutlass_opus::register_decoders(&mut decoders);
    cutlass_vorbis::register_decoders(&mut decoders);
    cutlass_webp::register_decoders(&mut decoders);
    cutlass_mjpeg::register_decoders(&mut decoders);
    cutlass_png::register_decoders(&mut decoders);
    cutlass_gif::register_decoders(&mut decoders);
    cutlass_jpeg::register_decoders(&mut decoders);
    cutlass_aac::register_decoders(&mut decoders);
    cutlass_av1::register_decoders(&mut decoders);
    cutlass_pcm::register_decoders(&mut decoders);
    cutlass_h264::register_decoders(&mut decoders);
    cutlass_prores::register_decoders(&mut decoders);
    cutlass_vp8::register_engine_decoders(&mut decoders);
    cutlass_hevc::register_decoders(&mut decoders);
    cutlass_mpeg2::register_decoders(&mut decoders);
    cutlass_mpeg4::register_engine_decoders(&mut decoders);
    decoders
}

fn source(error: &std::io::Error) -> OpenError {
    OpenError::Source {
        message: error.to_string(),
    }
}

fn sized(path: &Path) -> Result<(File, u64), OpenError> {
    let file = File::open(path).map_err(|error| source(&error))?;
    let length = file.metadata().map_err(|error| source(&error))?.len();
    Ok((file, length))
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

fn probe_mp3(header: &[u8]) -> ProbeResult {
    let accept = ProbeResult::Accept {
        family: ContainerId::Mp3,
        examined: header.len(),
    };
    match cutlass_mp3_container::probe(header) {
        cutlass_mp3_container::Probe::Matched => accept,
        cutlass_mp3_container::Probe::NeedMore { .. } if header.starts_with(b"ID3") => accept,
        cutlass_mp3_container::Probe::NeedMore { minimum } => ProbeResult::NeedBytes {
            have: header.len(),
            need: usize::try_from(minimum).unwrap_or(usize::MAX),
        },
        cutlass_mp3_container::Probe::Reject => ProbeResult::Refuse {
            examined: header.len(),
        },
    }
}

/// An ADTS sync word: twelve set bits, then a layer of zero.
fn probe_adts(header: &[u8]) -> ProbeResult {
    match header {
        [0xFF, second, ..] if second & 0xF6 == 0xF0 => ProbeResult::Accept {
            family: ContainerId::Adts,
            examined: 2,
        },
        _ => ProbeResult::Refuse { examined: 2 },
    }
}

fn open_adts(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, _) = sized(path)?;
    Ok(Box::new(cutlass_adts::AdtsFile::open(
        std::io::BufReader::new(file),
    )?))
}

fn open_isobmff(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, _) = sized(path)?;
    Ok(Box::new(cutlass_isobmff::IsoBmffFile::open(file)?))
}

fn open_avi(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    Ok(Box::new(cutlass_avi::AviDemux::open(file, length)?))
}

fn open_wav(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    Ok(Box::new(cutlass_riff::WavDemux::open(file, length)?))
}

fn open_aiff(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    Ok(Box::new(cutlass_aiff::AiffDemux::open(file, length)?))
}

fn open_caf(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    Ok(Box::new(cutlass_caf::CafDemux::open(file, length)?))
}

fn open_flac(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    Ok(Box::new(cutlass_flac_container::FlacDemux::open(
        std::io::BufReader::new(file),
        length,
    )?))
}

fn open_ogg(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    Ok(Box::new(cutlass_ogg::OggDemux::open(
        std::io::BufReader::new(file),
        length,
    )?))
}

fn open_ts(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    Ok(Box::new(cutlass_mpegts::TsDemux::open(
        std::io::BufReader::new(file),
        length,
    )?))
}

fn open_matroska(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    // The demuxer reads ahead itself, so the file goes in unbuffered.
    Ok(Box::new(cutlass_matroska::MatroskaDemux::open(
        file, length,
    )?))
}

fn open_still(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, length) = sized(path)?;
    Ok(Box::new(cutlass_still::StillDemux::open(Input::random(
        file,
        length,
        cutlass_still::still_limits(),
    ))?))
}

fn open_mp3(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let (file, _) = sized(path)?;
    let demux =
        cutlass_mp3_container::Mp3Demux::open(std::io::BufReader::new(file)).map_err(|error| {
            OpenError::Source {
                message: error.to_string(),
            }
        })?;
    Ok(Box::new(demux))
}
