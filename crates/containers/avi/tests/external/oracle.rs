//! Independent producers and demuxers: FFmpeg (libavformat) and GStreamer
//! (`avimux`, `avidemux`). Each test skips when its tool is missing.
//!
//! Times are compared in each stream's file timescale: ffprobe's
//! `time_base` is `dwScale/dwRate` reduced, one tick per AVI unit, and this
//! crate's timescale is `dwRate` over the same divisor, so an ffprobe tick
//! is `unit` of ours. Payloads compare by the Adler-32 FFmpeg's `framecrc`
//! prints, and byte for byte against GStreamer's buffers.
//!
//! FFmpeg returns a PCM chunk longer than 1024 blocks as several packets
//! (`avidec.c`, "arbitrary multiplier to avoid tiny packets"); this crate
//! returns one packet per chunk. PCM streams are therefore compared as one
//! byte stream (`-f data`), with every FFmpeg packet's time checked against
//! the byte offset it starts at.

use std::path::Path;

use cutlass_avi::{AviForwardDemux, AviMux, IndexSource};
use cutlass_codec::{CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};

use crate::support::{drain, movie, open_path, output, scratch, stamp, tool_on_path};

/// Adler-32 seeded with zero, as FFmpeg's `framecrc` computes it.
pub fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (0u32, 0u32);
    for chunk in bytes.chunks(5_552) {
        for byte in chunk {
            a += u32::from(*byte);
            b += a;
        }
        a %= 65_521;
        b %= 65_521;
    }
    (b << 16) | a
}

/// One packet as ffprobe and framecrc report it, times in ffprobe ticks.
#[derive(Debug)]
pub struct Probed {
    pub dts: Option<i64>,
    pub duration: Option<i64>,
    pub size: usize,
    pub key: bool,
    pub adler: u32,
}

/// `(num, den)` per stream and every packet per stream, in file order.
pub fn ffprobe(path: &Path) -> (Vec<(i64, i64)>, Vec<Vec<Probed>>) {
    let name = path.display().to_string();
    let streams = output(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "stream=time_base",
            "-of",
            "csv=p=0",
            &name,
        ],
    );
    assert!(
        streams.status.success(),
        "ffprobe: {}",
        String::from_utf8_lossy(&streams.stderr)
    );
    let bases: Vec<(i64, i64)> = String::from_utf8_lossy(&streams.stdout)
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (num, den) = line
                .trim()
                .trim_end_matches(',')
                .split_once('/')
                .expect("time base");
            (num.parse().expect("num"), den.parse().expect("den"))
        })
        .collect();
    let packets = output(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "packet=stream_index,dts,duration,size,flags",
            "-of",
            "csv=p=0",
            &name,
        ],
    );
    assert!(packets.status.success());
    let mut out: Vec<Vec<Probed>> = bases.iter().map(|_| Vec::new()).collect();
    for line in String::from_utf8_lossy(&packets.stdout).lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 {
            continue;
        }
        let stream: usize = fields[0].parse().expect("stream");
        out[stream].push(Probed {
            dts: fields[1].parse().ok(),
            duration: fields[2].parse().ok(),
            size: fields[3].parse().expect("size"),
            key: fields[4].contains('K'),
            adler: 0,
        });
    }
    let crc = output(
        "ffmpeg",
        &[
            "-v", "error", "-i", &name, "-map", "0", "-c", "copy", "-f", "framecrc", "-",
        ],
    );
    assert!(
        crc.status.success(),
        "framecrc: {}",
        String::from_utf8_lossy(&crc.stderr)
    );
    let mut seen = vec![0usize; out.len()];
    for line in String::from_utf8_lossy(&crc.stdout).lines() {
        if line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 6 {
            continue;
        }
        let stream: usize = fields[0].parse().expect("stream");
        let size: usize = fields[4].parse().expect("size");
        let adler = u32::from_str_radix(fields[5].trim_start_matches("0x"), 16).expect("adler");
        let row = &mut out[stream][seen[stream]];
        assert_eq!(
            row.size, size,
            "framecrc and ffprobe disagree on stream {stream}"
        );
        row.adler = adler;
        seen[stream] += 1;
    }
    (bases, out)
}

/// Every packet of `stream` back to back, as `ffmpeg -c copy -f data` writes them.
fn ffmpeg_stream_bytes(path: &Path, stream: usize) -> Vec<u8> {
    let name = path.display().to_string();
    let map = format!("0:{stream}");
    let out = output(
        "ffmpeg",
        &[
            "-v", "error", "-i", &name, "-map", &map, "-c", "copy", "-f", "data", "-",
        ],
    );
    assert!(
        out.status.success(),
        "ffmpeg data: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// A PCM stream: the same bytes, and each FFmpeg packet at the time its
/// first byte has. `block` is the PCM block size, `unit` ffprobe's tick in ours.
fn check_pcm(
    path: &Path,
    stream: usize,
    ours: &[&Packet],
    rows: &[Probed],
    block: usize,
    unit: i64,
) {
    let joined: Vec<u8> = ours
        .iter()
        .flat_map(|packet| packet.bytes().iter().copied())
        .collect();
    assert!(
        joined == ffmpeg_stream_bytes(path, stream),
        "{} stream {stream} bytes",
        path.display()
    );
    let start = ours[0].timestamp().decode().expect("time");
    let mut offset = 0usize;
    for row in rows {
        let blocks = i64::try_from(offset / block).expect("blocks");
        assert_eq!(
            row.dts.map(|dts| dts * unit),
            Some(start + blocks),
            "{} stream {stream}",
            path.display()
        );
        offset += row.size;
    }
    assert_eq!(offset, joined.len());
}

pub fn pcm_block(packet: &Packet, params: &StreamParams) -> Option<usize> {
    if let StreamCodec::Unknown(unknown) = params.codec()
        && matches!(unknown.tag(), b"wav:0006" | b"wav:0007")
    {
        return params.channels().map(usize::from);
    }
    if params.codec() != &StreamCodec::Known(CodecId::Pcm) {
        return None;
    }
    let _ = packet;
    let extra = params.extradata();
    extra
        .get(6..8)
        .map(|bytes| usize::from(u16::from_le_bytes([bytes[0], bytes[1]])))
}

/// Compares this crate's packets with ffprobe's, stream by stream.
pub fn check_against_ffprobe(path: &Path) -> Vec<Packet> {
    let (bases, probed) = ffprobe(path);
    let mut demux = open_path(path);
    let params: Vec<StreamParams> = demux
        .streams()
        .iter()
        .map(|stream| stream.params().clone())
        .collect();
    let scales: Vec<u32> = demux
        .streams()
        .iter()
        .map(|stream| stream.timescale().ticks_per_second())
        .collect();
    let packets = drain(&mut demux);
    assert_eq!(scales.len(), probed.len(), "{}", path.display());
    for (stream, rows) in probed.iter().enumerate() {
        let ours: Vec<&Packet> = packets
            .iter()
            .filter(|packet| packet.stream() == u32::try_from(stream).expect("stream"))
            .collect();
        let (num, den) = bases[stream];
        let scale = i64::from(scales[stream]);
        assert_eq!(
            scale * num % den,
            0,
            "time base {num}/{den} against timescale {scale}"
        );
        let unit = scale * num / den;
        if let Some(block) = ours
            .first()
            .and_then(|packet| pcm_block(packet, &params[stream]))
        {
            check_pcm(path, stream, &ours, rows, block, unit);
            continue;
        }
        assert_eq!(ours.len(), rows.len(), "{} stream {stream}", path.display());
        for (index, (packet, row)) in ours.iter().zip(rows).enumerate() {
            let at = format!("{} stream {stream} packet {index}", path.display());
            assert_eq!(packet.bytes().len(), row.size, "{at}");
            assert_eq!(adler32(packet.bytes()), row.adler, "{at}");
            assert_eq!(packet.keyframe(), row.key, "{at}");
            assert_eq!(
                packet.timestamp().decode(),
                row.dts.map(|dts| dts * unit),
                "{at}"
            );
            let duration = packet.timestamp().duration().expect("duration");
            if params[stream].size().is_some() {
                // A video packet lasts until the next one: through any empty
                // chunks (dropped frames) after it. FFmpeg's demuxer says one
                // unit and its generic layer may guess from the frame rate.
                match rows.get(index + 1).and_then(|next| next.dts.zip(row.dts)) {
                    Some((next, dts)) => assert_eq!(duration, (next - dts) * unit, "{at}"),
                    None => assert!(duration >= unit, "{at}"),
                }
            } else if let Some(span) = row.duration {
                // FFmpeg leaves some first audio packets without a duration.
                assert_eq!(duration, span * unit, "{at}");
            }
        }
    }
    packets
}

fn has_encoder(name: &str) -> bool {
    let out = output("ffmpeg", &["-hide_banner", "-encoders"]);
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .any(|word| word == name)
}

struct Case {
    name: &'static str,
    args: &'static [&'static str],
    encoders: &'static [&'static str],
    codecs: &'static [Option<CodecId>],
}

const SOURCES: [&str; 8] = [
    "-f",
    "lavfi",
    "-i",
    "testsrc=size=160x120:rate=25",
    "-f",
    "lavfi",
    "-i",
    "sine=frequency=440:sample_rate=48000",
];

const CASES: &[Case] = &[
    Case {
        name: "mjpeg-pcm",
        args: &["-c:v", "mjpeg", "-c:a", "pcm_s16le"],
        encoders: &["mjpeg"],
        codecs: &[Some(CodecId::Mjpeg), Some(CodecId::Pcm)],
    },
    Case {
        name: "h264-bframes-aac",
        args: &["-c:v", "libx264", "-bf", "3", "-g", "12", "-c:a", "aac"],
        encoders: &["libx264"],
        codecs: &[Some(CodecId::H264), Some(CodecId::Aac)],
    },
    Case {
        name: "mpeg4-mp3",
        args: &["-c:v", "mpeg4", "-bf", "2", "-c:a", "libmp3lame"],
        encoders: &["libmp3lame"],
        codecs: &[Some(CodecId::Mpeg4), Some(CodecId::Mp3)],
    },
    Case {
        name: "mpeg2-ac3",
        args: &["-c:v", "mpeg2video", "-c:a", "ac3"],
        encoders: &["mpeg2video"],
        codecs: &[Some(CodecId::Mpeg2), Some(CodecId::Ac3)],
    },
    Case {
        name: "vp8-float",
        args: &["-c:v", "libvpx", "-c:a", "pcm_f32le"],
        encoders: &["libvpx"],
        codecs: &[Some(CodecId::Vp8), Some(CodecId::Pcm)],
    },
    Case {
        name: "hevc-s24",
        // FFmpeg 8.1 writes HEVC with a zero FourCC unless told the tag.
        args: &[
            "-c:v",
            "libx265",
            "-tag:v",
            "HEVC",
            "-x265-params",
            "log-level=none",
            "-c:a",
            "pcm_s24le",
        ],
        encoders: &["libx265"],
        codecs: &[Some(CodecId::Hevc), Some(CodecId::Pcm)],
    },
    Case {
        name: "raw-pcm",
        args: &["-c:v", "rawvideo", "-pix_fmt", "bgr24", "-c:a", "pcm_s16le"],
        encoders: &["rawvideo"],
        codecs: &[Some(CodecId::RawVideo), Some(CodecId::Pcm)],
    },
    Case {
        name: "vp9-ac3",
        args: &["-c:v", "libvpx-vp9", "-deadline", "realtime", "-c:a", "ac3"],
        encoders: &["libvpx-vp9", "ac3"],
        codecs: &[Some(CodecId::Vp9), Some(CodecId::Ac3)],
    },
    Case {
        // FFmpeg's framecrc gives a FLAC stream in AVI one packet more
        // than ffprobe does, so the oracle cannot pair them; FLAC is
        // checked through the muxer copy instead.
        name: "av1-pcm",
        args: &["-c:v", "libsvtav1", "-preset", "12", "-c:a", "pcm_s16le"],
        encoders: &["libsvtav1"],
        codecs: &[Some(CodecId::Av1), Some(CodecId::Pcm)],
    },
];

#[test]
fn ffmpeg_made_files_match_ffprobe_packet_for_packet() {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg or ffprobe is missing");
        return;
    }
    let dir = scratch("ffmpeg");
    let mut ran = 0;
    for case in CASES {
        if !case.encoders.iter().all(|name| has_encoder(name)) {
            eprintln!("skipped {}: encoder missing", case.name);
            continue;
        }
        let path = dir.join(format!("{}.avi", case.name));
        let name = path.display().to_string();
        let mut args = vec!["-v", "error", "-y"];
        args.extend(SOURCES);
        args.extend(["-t", "3"]);
        args.extend(case.args);
        args.push(&name);
        let made = output("ffmpeg", &args);
        assert!(
            made.status.success(),
            "{}: {}",
            case.name,
            String::from_utf8_lossy(&made.stderr)
        );
        let demux = open_path(&path);
        assert_eq!(demux.index_source(), IndexSource::Idx1, "{}", case.name);
        for (stream, codec) in demux.streams().iter().zip(case.codecs) {
            match codec {
                Some(id) => assert_eq!(
                    stream.params().codec(),
                    &StreamCodec::Known(*id),
                    "{}",
                    case.name
                ),
                None => assert!(stream.params().codec().is_unknown(), "{}", case.name),
            }
        }
        let packets = check_against_ffprobe(&path);
        assert!(packets.len() > 100, "{}", case.name);
        same_read_forward(&path, &packets);
        ran += 1;
    }
    assert!(ran > 0, "no case ran");
    eprintln!("ffmpeg cases checked: {ran} of {}", CASES.len());
}

/// Reading `path` once, front to back, with no index, gives each stream
/// the packets the indexed reader gave: the same bytes, times, durations
/// and keyframes (the keyframes read from each bitstream here, from
/// `idx1` there).
fn same_read_forward(path: &Path, indexed: &[Packet]) {
    let file = std::fs::File::open(path).expect("open");
    let mut demux = AviForwardDemux::open(std::io::BufReader::new(file)).expect("open forward");
    let read = drain(&mut demux);
    assert_eq!(read.len(), indexed.len(), "{}", path.display());
    for stream in 0..u32::try_from(demux.streams().len()).expect("streams") {
        let left = indexed.iter().filter(|packet| packet.stream() == stream);
        let right = read.iter().filter(|packet| packet.stream() == stream);
        for (index, (left, right)) in left.zip(right).enumerate() {
            assert!(
                left == right,
                "{} stream {stream} packet {index}: {:?} {} {} / {:?} {} {}",
                path.display(),
                left.timestamp(),
                left.bytes().len(),
                left.keyframe(),
                right.timestamp(),
                right.bytes().len(),
                right.keyframe()
            );
        }
    }
}

/// GStreamer's buffers per stream, in order, from `avidemux`.
fn gstreamer_buffers(path: &Path, pads: &[&str]) -> Option<Vec<Vec<Vec<u8>>>> {
    if !tool_on_path("gst-launch-1.0") {
        return None;
    }
    let dir = path.with_extension("gst");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let mut args = vec![
        "-e".to_owned(),
        "alarm 60; exec @ARGV".to_owned(),
        "gst-launch-1.0".to_owned(),
        "-q".to_owned(),
        "filesrc".to_owned(),
        format!("location={}", path.display()),
        "!".to_owned(),
        "avidemux".to_owned(),
        "name=d".to_owned(),
    ];
    for (index, pad) in pads.iter().enumerate() {
        args.extend([
            format!("d.{pad}"),
            "!".to_owned(),
            "queue".to_owned(),
            "!".to_owned(),
            "multifilesink".to_owned(),
            format!("location={}/{index}-%06d", dir.display()),
        ]);
    }
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let run = output("perl", &refs);
    assert!(
        run.status.success(),
        "gst-launch: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .expect("dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .into_string()
                .expect("name")
        })
        .collect();
    names.sort();
    let mut out = vec![Vec::new(); pads.len()];
    for name in names {
        let index: usize = name[..1].parse().expect("index");
        out[index].push(std::fs::read(dir.join(&name)).expect("buffer"));
    }
    Some(out)
}

#[test]
fn a_gstreamer_made_file_matches_ffprobe() {
    if !tool_on_path("gst-launch-1.0") || !tool_on_path("ffprobe") {
        eprintln!("skipped: gst-launch-1.0 or ffprobe is missing");
        return;
    }
    let dir = scratch("gstreamer");
    let path = dir.join("gst-mjpeg-pcm.avi");
    let made = output(
        "gst-launch-1.0",
        &[
            "-q",
            "avimux",
            "name=m",
            "!",
            "filesink",
            &format!("location={}", path.display()),
            "videotestsrc",
            "num-buffers=60",
            "!",
            "video/x-raw,framerate=25/1,width=160,height=120",
            "!",
            "jpegenc",
            "!",
            "m.",
            "audiotestsrc",
            "num-buffers=50",
            "!",
            "audio/x-raw,rate=44100,channels=2,format=S16LE",
            "!",
            "m.",
        ],
    );
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
    let packets = check_against_ffprobe(&path);
    assert_eq!(
        packets.iter().filter(|packet| packet.stream() == 0).count(),
        60
    );
}

fn write_file(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).expect("write");
}

/// Checks a written file against the packets that went in: ffprobe's
/// packets and GStreamer's buffers, per stream.
/// `blocks` is each stream's PCM block size, `None` for other codecs.
/// `exact` asks for every time to come back exactly, as it must when the
/// packets sit on a frame grid.
pub fn check_written(
    path: &Path,
    packets: &[Packet],
    scales: &[u32],
    blocks: &[Option<usize>],
    pads: &[&str],
    exact: bool,
) {
    if tool_on_path("ffprobe") && tool_on_path("ffmpeg") {
        let (bases, probed) = ffprobe(path);
        assert_eq!(probed.len(), scales.len());
        for (stream, rows) in probed.iter().enumerate() {
            let sent: Vec<&Packet> = packets
                .iter()
                .filter(|packet| packet.stream() == u32::try_from(stream).expect("stream"))
                .filter(|packet| !packet.bytes().is_empty())
                .collect();
            let (num, den) = bases[stream];
            if let Some(block) = blocks[stream] {
                let scale = i64::from(scales[stream]);
                check_pcm(path, stream, &sent, rows, block, scale * num / den);
                continue;
            }
            assert_eq!(sent.len(), rows.len(), "stream {stream}");
            for (index, (packet, row)) in sent.iter().zip(rows).enumerate() {
                assert_eq!(
                    packet.bytes().len(),
                    row.size,
                    "stream {stream} packet {index}"
                );
                assert_eq!(
                    adler32(packet.bytes()),
                    row.adler,
                    "stream {stream} packet {index}"
                );
                assert_eq!(packet.keyframe(), row.key, "stream {stream} packet {index}");
                // Seconds: dts * num / den against time / scale. The muxer
                // puts a video frame on the nearest frame of its period,
                // one ffprobe tick, so it may be up to half a tick off; a
                // source on that grid comes back exact.
                let time = i128::from(packet.timestamp().decode().expect("time"));
                let dts = i128::from(row.dts.expect("dts"));
                let scale = i128::from(scales[stream]);
                let error = (dts * i128::from(num) * scale - time * i128::from(den)).abs();
                if exact {
                    assert_eq!(error, 0, "stream {stream} packet {index}");
                } else {
                    assert!(
                        error * 2 <= i128::from(num) * scale,
                        "stream {stream} packet {index}: dts {dts} time {time}"
                    );
                }
            }
        }
    } else {
        eprintln!("skipped ffprobe check: ffmpeg or ffprobe is missing");
    }
    match gstreamer_buffers(path, pads) {
        Some(buffers) => {
            for (stream, got) in buffers.iter().enumerate() {
                if blocks[stream].is_some() {
                    let joined: Vec<u8> = got.concat();
                    let sent: Vec<u8> = packets
                        .iter()
                        .filter(|packet| packet.stream() == u32::try_from(stream).expect("stream"))
                        .flat_map(|packet| packet.bytes().iter().copied())
                        .collect();
                    assert!(joined == sent, "gstreamer stream {stream} bytes");
                    continue;
                }
                let sent: Vec<&[u8]> = packets
                    .iter()
                    .filter(|packet| packet.stream() == u32::try_from(stream).expect("stream"))
                    .filter(|packet| !packet.bytes().is_empty())
                    .map(Packet::bytes)
                    .collect();
                assert_eq!(got.len(), sent.len(), "gstreamer stream {stream}");
                for (index, (left, right)) in got.iter().zip(&sent).enumerate() {
                    assert!(
                        left.as_slice() == *right,
                        "gstreamer stream {stream} buffer {index}"
                    );
                }
            }
        }
        None => eprintln!("skipped GStreamer check: gst-launch-1.0 is missing"),
    }
}

#[test]
fn written_opendml_files_read_back_in_ffprobe_and_gstreamer() {
    let dir = scratch("written");
    for (name, riff) in [("one-riff", 1u64 << 30), ("many-riffs", 96 * 1024)] {
        let written = movie(400, 25, riff);
        if name == "many-riffs" {
            assert!(written.riffs > 4, "{} segments", written.riffs);
        } else {
            assert_eq!(written.riffs, 1);
        }
        let path = dir.join(format!("{name}.avi"));
        write_file(&path, &written.bytes);
        check_written(
            &path,
            &written.packets,
            &[90_000, 48_000],
            &[None, None],
            &["video_0", "audio_0"],
            true,
        );
    }
}

#[test]
fn written_mjpeg_pcm_and_mp3_read_back() {
    let dir = scratch("written-mjpeg");
    let mut mux = AviMux::new(std::io::Cursor::new(Vec::new()));
    let video = NewStream {
        params: StreamParams::video(
            StreamCodec::Known(CodecId::Mjpeg),
            FrameSize::try_new(32, 32).expect("size"),
            Vec::new(),
        ),
        timescale: Timescale::new(30_000),
        language: None,
    };
    mux.add_stream(video).expect("video");
    let pcm = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 44_100, 2, Vec::new())
        .expect("pcm");
    mux.add_stream(NewStream {
        params: pcm,
        timescale: Timescale::new(44_100),
        language: None,
    })
    .expect("pcm");
    let mp3 = StreamParams::try_audio(StreamCodec::Known(CodecId::Mp3), 44_100, 2, Vec::new())
        .expect("mp3");
    mux.add_stream(NewStream {
        params: mp3,
        timescale: Timescale::new(44_100),
        language: None,
    })
    .expect("mp3");
    let mut packets = Vec::new();
    let mut samples = 0i64;
    let mut frames = 0i64;
    for frame in 0..60i64 {
        let time = frame * 1_001;
        let mut jpeg = vec![0xFF, 0xD8];
        jpeg.extend(std::iter::repeat_n(
            u8::try_from(frame).expect("frame"),
            400,
        ));
        jpeg.extend([0xFF, 0xD9]);
        packets.push(stamp(0, jpeg, time, 1_001, 30_000, true));
        while samples * 30_000 <= time * 44_100 {
            packets.push(stamp(1, vec![3; 4 * 1_470], samples, 1_470, 44_100, true));
            samples += 1_470;
        }
        while frames * 1_152 * 30_000 <= time * 44_100 {
            let mut frame_bytes = vec![0xFF, 0xFB, 0x90, 0x64];
            frame_bytes.resize(417, 0x55);
            packets.push(stamp(2, frame_bytes, frames * 1_152, 1_152, 44_100, true));
            frames += 1;
        }
    }
    for packet in &packets {
        mux.write_packet(packet).expect("write");
    }
    mux.finish(&[
        TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(30_000),
            ticks: 60 * 1_001,
        },
        TrackDuration {
            track: TrackId::new(1),
            timescale: Timescale::new(44_100),
            ticks: samples,
        },
        TrackDuration {
            track: TrackId::new(2),
            timescale: Timescale::new(44_100),
            ticks: frames * 1_152,
        },
    ])
    .expect("finish");
    let bytes = mux.into_writer().expect("writer").into_inner();
    let path = dir.join("mjpeg-pcm-mp3.avi");
    write_file(&path, &bytes);
    check_written(
        &path,
        &packets,
        &[30_000, 44_100, 44_100],
        &[None, Some(4), None],
        &["video_0", "audio_0", "audio_1"],
        true,
    );
}

struct Copy {
    name: &'static str,
    /// Inputs after the video and one audio source, and the codec options.
    args: &'static [&'static str],
    encoders: &'static [&'static str],
    codecs: &'static [CodecId],
}

/// FFmpeg makes each file; this crate reads it, `AviMux` writes it again,
/// and ffprobe and GStreamer must read every packet of the copy back.
const COPIES: &[Copy] = &[
    Copy {
        name: "raw-float",
        args: &["-c:v", "rawvideo", "-pix_fmt", "bgr24", "-c:a", "pcm_f32le"],
        encoders: &["rawvideo", "pcm_f32le"],
        codecs: &[CodecId::RawVideo, CodecId::Pcm],
    },
    Copy {
        name: "yuv420-s24",
        args: &[
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s24le",
        ],
        encoders: &["rawvideo", "pcm_s24le"],
        codecs: &[CodecId::RawVideo, CodecId::Pcm],
    },
    Copy {
        name: "vp9-ac3",
        args: &["-c:v", "libvpx-vp9", "-deadline", "realtime", "-c:a", "ac3"],
        encoders: &["libvpx-vp9", "ac3"],
        codecs: &[CodecId::Vp9, CodecId::Ac3],
    },
    Copy {
        name: "av1-eac3",
        args: &["-c:v", "libsvtav1", "-preset", "12", "-c:a", "eac3"],
        encoders: &["libsvtav1", "eac3"],
        codecs: &[CodecId::Av1, CodecId::Eac3],
    },
    Copy {
        name: "mjpeg-flac",
        args: &["-c:v", "mjpeg", "-c:a", "flac"],
        encoders: &["mjpeg", "flac"],
        codecs: &[CodecId::Mjpeg, CodecId::Flac],
    },
];

#[test]
fn ffmpeg_files_copied_through_avimux_read_back() {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg or ffprobe is missing");
        return;
    }
    let dir = scratch("copied");
    let mut ran = 0;
    for copy in COPIES {
        if !copy.encoders.iter().all(|name| has_encoder(name)) {
            eprintln!("skipped {}: encoder missing", copy.name);
            continue;
        }
        let source = dir.join(format!("{}.avi", copy.name));
        let name = source.display().to_string();
        let mut args = vec![
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "2",
            "-ac",
            "2",
        ];
        args.extend_from_slice(copy.args);
        args.push(&name);
        let made = output("ffmpeg", &args);
        assert!(
            made.status.success(),
            "{}: {}",
            copy.name,
            String::from_utf8_lossy(&made.stderr)
        );
        let demux = open_path(&source);
        let codecs: Vec<_> = demux
            .streams()
            .iter()
            .map(|stream| stream.params().codec().clone())
            .collect();
        let wanted: Vec<_> = copy
            .codecs
            .iter()
            .map(|id| StreamCodec::Known(*id))
            .collect();
        assert_eq!(codecs, wanted, "{}", copy.name);
        crate::local::remux(&source, &dir.join(format!("{}-copy.avi", copy.name)));
        eprintln!("  {} copied and read back", copy.name);
        ran += 1;
    }
    assert!(ran > 0, "no case ran");
}

/// MPEG-1 Layer II, A-law and mu-law have no codec id: they keep their
/// wave format tag, and the muxer writes them under it.
#[test]
fn tagged_audio_copies_through_avimux() {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg or ffprobe is missing");
        return;
    }
    let dir = scratch("copied-tags");
    let source = dir.join("tags.avi");
    let name = source.display().to_string();
    let made = output(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "2",
            "-map",
            "0",
            "-map",
            "1",
            "-map",
            "1",
            "-map",
            "1",
            "-c:v",
            "mjpeg",
            "-c:a:0",
            "mp2",
            "-c:a:1",
            "pcm_alaw",
            "-c:a:2",
            "pcm_mulaw",
            "-ar:a:1",
            "8000",
            "-ar:a:2",
            "8000",
            &name,
        ],
    );
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
    let demux = open_path(&source);
    let tags: Vec<_> = demux
        .streams()
        .iter()
        .map(|stream| match stream.params().codec() {
            StreamCodec::Unknown(unknown) => String::from_utf8_lossy(unknown.tag()).into_owned(),
            StreamCodec::Known(id) => format!("{id:?}"),
        })
        .collect();
    assert_eq!(tags, ["Mjpeg", "wav:0050", "wav:0006", "wav:0007"]);
    crate::local::remux(&source, &dir.join("tags-copy.avi"));
}

/// A file over 1 GiB from FFmpeg: two RIFF segments, OpenDML indexes.
/// About 1.6 GB of scratch space and a minute or two.
#[test]
#[ignore = "writes a 1.6 GB file; run with --ignored"]
fn an_ffmpeg_opendml_file_over_one_gib_matches_ffprobe() {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg or ffprobe is missing");
        return;
    }
    let dir = scratch("opendml");
    let path = dir.join("big.avi");
    let name = path.display().to_string();
    let made = output(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=1280x720:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=48000",
            "-t",
            "24",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-c:a",
            "pcm_s16le",
            &name,
        ],
    );
    assert!(made.status.success());
    let demux = open_path(&path);
    assert_eq!(demux.index_source(), IndexSource::OpenDml);
    assert_eq!(demux.riff_count(), 2);
    check_against_ffprobe(&path);
    let _ = std::fs::remove_dir_all(&dir);
}
