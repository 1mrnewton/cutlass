//! ffmpeg writes an AIFF, this crate reads the packets, and ffmpeg reads a file this crate wrote.

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_aiff::{AiffDemux, AiffMux};
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn ffmpeg_pcm_matches_this_demux_and_reads_this_mux() {
    if !tool("ffmpeg") || !tool("ffprobe") {
        eprintln!("skipped: ffmpeg and ffprobe are not on PATH");
        return;
    }
    let dir =
        TempDir(std::env::temp_dir().join(format!("cutlass-aiff-oracle-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&dir.0);
    std::fs::create_dir_all(&dir.0).expect("temp");

    round_trip(&dir.0, "s16be", "pcm_s16be", &s16_be(), [1, 1, 0]);
    round_trip(&dir.0, "s16le", "pcm_s16le", &s16_le(), [1, 0, 0]);
    round_trip(&dir.0, "f32be", "pcm_f32be", &f32_be(), [4, 1, 0]);
}

fn round_trip(dir: &Path, raw_format: &str, codec: &str, raw: &[u8], extra: [u8; 3]) {
    let source = dir.join(format!("{raw_format}.pcm"));
    let theirs = dir.join(format!("{raw_format}.aiff"));
    let ours = dir.join(format!("{raw_format}-ours.aiff"));
    let decoded = dir.join(format!("{raw_format}-decoded.pcm"));
    std::fs::write(&source, raw).expect("raw");
    ffmpeg(&[
        "-y",
        "-f",
        raw_format,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-i",
        source.to_str().expect("path"),
        "-c:a",
        codec,
        theirs.to_str().expect("path"),
    ]);
    let packets = read_packets(&theirs);
    let mut got = Vec::<u8>::new();
    for packet in &packets {
        got.extend(packet.bytes());
        assert_eq!(packet.timestamp().duration(), Some(1));
    }
    assert_eq!(got, raw);
    assert_eq!(packets[0].timescale().ticks_per_second(), 16_000);
    let demux = open_file(&theirs);
    assert_eq!(demux.streams()[0].params().extradata(), extra);
    assert_eq!(demux.sample_rate(), 16_000);
    assert_eq!(demux.channels(), 1);
    let probed = ffprobe(&theirs);
    assert!(probed.contains("sample_rate=16000"), "{probed}");
    assert!(probed.contains("channels=1"), "{probed}");
    assert!(probed.contains("pcm"), "{probed}");
    if tool("afinfo") {
        let info = text("afinfo", &[theirs.to_str().expect("path")]);
        assert!(info.contains("16000"), "{info}");
    }

    write_ours(&ours, extra, raw);
    ffmpeg(&[
        "-y",
        "-i",
        ours.to_str().expect("path"),
        "-f",
        raw_format,
        "-c:a",
        codec,
        decoded.to_str().expect("path"),
    ]);
    assert_eq!(std::fs::read(&decoded).expect("decoded"), raw);
}

fn s16_be() -> Vec<u8> {
    vec![0x00, 0x00, 0x7f, 0xff, 0x80, 0x00, 0x00, 0x01]
}

fn s16_le() -> Vec<u8> {
    vec![0x00, 0x00, 0xff, 0x7f, 0x00, 0x80, 0x01, 0x00]
}

fn f32_be() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(1.0f32.to_be_bytes());
    bytes.extend((-0.5f32).to_be_bytes());
    bytes
}

fn write_ours(path: &Path, extra: [u8; 3], raw: &[u8]) {
    let width = match extra[0] {
        1 => 2,
        4 => 4,
        _ => panic!("width"),
    };
    let mut mux = AiffMux::new(std::io::Cursor::new(Vec::new()));
    let params =
        StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 16_000, 1, extra.to_vec())
            .expect("params");
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(16_000),
        language: None,
    })
    .expect("stream");
    let mut index = 0i64;
    for frame in raw.chunks(width) {
        let packet = Packet::try_new(
            0,
            frame.to_vec(),
            Timestamp::try_new(Some(index), Some(index), Some(1)).expect("time"),
            Timescale::new(16_000),
            true,
        )
        .expect("packet");
        mux.write_packet(&packet).expect("write");
        index += 1;
    }
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(16_000),
        ticks: index,
    }])
    .expect("finish");
    std::fs::write(path, mux.into_inner().into_inner()).expect("file");
}

fn read_packets(path: &Path) -> Vec<Packet> {
    let mut demux = open_file(path);
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push(packet);
    }
    packets
}

fn open_file(path: &Path) -> AiffDemux<std::fs::File> {
    let file = std::fs::File::open(path).expect("open");
    let len = file.metadata().expect("meta").len();
    AiffDemux::open(file, len).expect("demux")
}

fn ffprobe(path: &Path) -> String {
    text(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "default=noprint_wrappers=1",
            path.to_str().expect("path"),
        ],
    )
}

fn ffmpeg(args: &[&str]) {
    let output = Command::new("ffmpeg").args(args).output().expect("ffmpeg");
    assert!(
        output.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn text(program: &str, args: &[&str]) -> String {
    let output = Command::new(program).args(args).output().expect(program);
    assert!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn tool(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .is_ok_and(|output| output.status.success())
}
