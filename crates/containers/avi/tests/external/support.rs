//! Files the tests build, and helpers for tests that shell out to
//! independent tools.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use cutlass_avi::{AviDemux, AviMux, AviMuxOptions};
use cutlass_codec::{CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};

pub fn tool_on_path(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn output(program: &str, args: &[&str]) -> Output {
    Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("{program} failed to start: {err}"))
}

/// A scratch directory under the target directory, emptied first.
pub fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("avi-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

pub fn open_path(path: &Path) -> AviDemux<std::io::BufReader<std::fs::File>> {
    let file = std::fs::File::open(path).expect("file");
    let len = file.metadata().expect("meta").len();
    AviDemux::open(std::io::BufReader::new(file), len)
        .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

pub fn drain<D: Demux>(demux: &mut D) -> Vec<Packet> {
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        out.push(packet);
    }
    out
}

pub fn stamp(
    stream: u32,
    bytes: Vec<u8>,
    time: i64,
    duration: i64,
    scale: u32,
    key: bool,
) -> Packet {
    Packet::try_new(
        stream,
        bytes,
        Timestamp::try_new(Some(time), Some(time), Some(duration)).expect("time"),
        Timescale::new(scale),
        key,
    )
    .expect("packet")
}

/// A movie the muxer writes: H.264 (Annex B, an IDR every `gop` frames)
/// at 25 fps on a 90 kHz clock, and AAC-LC at 48 kHz in 1024-sample frames.
pub struct Movie {
    pub bytes: Vec<u8>,
    pub packets: Vec<Packet>,
    pub riffs: u32,
}

pub fn movie(frames: u32, gop: u32, riff_bytes: u64) -> Movie {
    let options = AviMuxOptions {
        riff_bytes,
        ..AviMuxOptions::default()
    };
    let mut mux = AviMux::with_options(Cursor::new(Vec::new()), options);
    mux.add_stream(NewStream {
        params: StreamParams::video(
            StreamCodec::Known(CodecId::H264),
            FrameSize::try_new(320, 240).expect("size"),
            Vec::new(),
        ),
        timescale: Timescale::new(90_000),
        language: None,
    })
    .expect("video");
    mux.add_stream(NewStream {
        params: StreamParams::try_audio(
            StreamCodec::Known(CodecId::Aac),
            48_000,
            2,
            vec![0x11, 0x90],
        )
        .expect("audio"),
        timescale: Timescale::new(48_000),
        language: None,
    })
    .expect("audio");
    let mut packets = Vec::new();
    let mut audio = 0i64;
    for frame in 0..frames {
        let time = i64::from(frame) * 3_600;
        let key = frame % gop == 0;
        let size = 200 + usize::try_from((frame * 37) % 900).expect("size");
        let mut bytes = vec![u8::try_from(frame % 253).expect("byte"); size];
        bytes[..5].copy_from_slice(&[0, 0, 0, 1, if key { 0x65 } else { 0x41 }]);
        packets.push(stamp(0, bytes, time, 3_600, 90_000, key));
        while audio * 1024 * 90_000 / 48_000 <= time {
            let size = 150 + usize::try_from((audio * 13) % 200).expect("size");
            packets.push(stamp(1, vec![0x21; size], audio * 1024, 1024, 48_000, true));
            audio += 1;
        }
    }
    for packet in &packets {
        mux.write_packet(packet).expect("write");
    }
    mux.finish(&[
        TrackDuration {
            track: TrackId::new(0),
            timescale: Timescale::new(90_000),
            ticks: i64::from(frames) * 3_600,
        },
        TrackDuration {
            track: TrackId::new(1),
            timescale: Timescale::new(48_000),
            ticks: audio * 1024,
        },
    ])
    .expect("finish");
    let riffs = mux.riff_count();
    Movie {
        bytes: mux.into_writer().expect("writer").into_inner(),
        packets,
        riffs,
    }
}
