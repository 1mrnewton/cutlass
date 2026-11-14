//! Constant-bitrate MP3, MPEG-1 Layer II, AC-3 and E-AC-3 stored the way
//! old capture tools stored them: fixed-size chunks that hold several
//! frames and end inside one. FFmpeg makes the elementary stream; the AVI
//! around it is built here, with `dwSampleSize` 1 and a byte clock. Every
//! frame must match what ffprobe returns through FFmpeg's parsers: size,
//! Adler-32 and time.

use std::path::Path;

use crate::oracle::{adler32, ffprobe};
use crate::support::{drain, open_path, output, scratch, tool_on_path};

fn le16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn le32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    le32(&mut out, u32::try_from(body.len()).expect("size"));
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut inner = kind.to_vec();
    inner.extend_from_slice(body);
    chunk(b"LIST", &inner)
}

/// One audio stream of `stream` cut into `chunk_bytes` chunks, indexed by
/// `idx1`, with `tag` and a byte clock of `byte_rate`.
pub fn byte_clock_avi(
    stream: &[u8],
    chunk_bytes: usize,
    tag: u16,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
) -> Vec<u8> {
    let pieces: Vec<&[u8]> = stream.chunks(chunk_bytes).collect();
    let mut avih = Vec::new();
    for value in [
        0,
        byte_rate,
        0,
        0x10,
        u32::try_from(pieces.len()).expect("n"),
        0,
        1,
        0,
        0,
        0,
    ] {
        le32(&mut avih, value);
    }
    for _ in 0..4 {
        le32(&mut avih, 0);
    }
    let mut strh = b"auds".to_vec();
    le32(&mut strh, 0);
    for value in [0, 0, 0] {
        le32(&mut strh, value);
    }
    le32(&mut strh, 1);
    le32(&mut strh, byte_rate);
    le32(&mut strh, 0);
    le32(&mut strh, u32::try_from(stream.len()).expect("len"));
    le32(&mut strh, 0);
    le32(&mut strh, u32::MAX);
    le32(&mut strh, 1);
    for _ in 0..4 {
        le16(&mut strh, 0);
    }
    let mut strf = Vec::new();
    le16(&mut strf, tag);
    le16(&mut strf, channels);
    le32(&mut strf, sample_rate);
    le32(&mut strf, byte_rate);
    le16(&mut strf, 1);
    le16(&mut strf, 0);
    le16(&mut strf, 0);
    let mut strl = chunk(b"strh", &strh);
    strl.extend(chunk(b"strf", &strf));
    let mut hdrl = chunk(b"avih", &avih);
    hdrl.extend(list(b"strl", &strl));
    let mut movi = Vec::new();
    let mut idx1 = Vec::new();
    for piece in &pieces {
        le32(&mut idx1, u32::from_le_bytes(*b"00wb"));
        le32(&mut idx1, 0x10);
        le32(&mut idx1, u32::try_from(movi.len() + 4).expect("offset"));
        le32(&mut idx1, u32::try_from(piece.len()).expect("size"));
        movi.extend(chunk(b"00wb", piece));
    }
    let mut body = b"AVI ".to_vec();
    body.extend(list(b"hdrl", &hdrl));
    body.extend(list(b"movi", &movi));
    body.extend(chunk(b"idx1", &idx1));
    chunk(b"RIFF", &body)
}

struct Case {
    name: &'static str,
    encode: &'static [&'static str],
    format: &'static str,
    tag: u16,
    sample_rate: u32,
    byte_rate: u32,
    chunk: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "mp3-128k",
        encode: &[
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-write_xing",
            "0",
            "-id3v2_version",
            "0",
        ],
        format: "mp3",
        tag: 0x0055,
        sample_rate: 44_100,
        byte_rate: 16_000,
        chunk: 1_000,
    },
    Case {
        name: "mp2-192k",
        encode: &["-c:a", "mp2", "-b:a", "192k"],
        format: "mp2",
        tag: 0x0050,
        sample_rate: 48_000,
        byte_rate: 24_000,
        chunk: 4_608,
    },
    Case {
        name: "ac3-192k",
        encode: &["-c:a", "ac3", "-b:a", "192k"],
        format: "ac3",
        tag: 0x2000,
        sample_rate: 48_000,
        byte_rate: 24_000,
        chunk: 2_000,
    },
];

fn elementary(dir: &Path, case: &Case) -> Option<Vec<u8>> {
    let path = dir.join(format!("{}.{}", case.name, case.format));
    let name = path.display().to_string();
    let rate = case.sample_rate.to_string();
    let mut args = vec![
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=500:duration=4",
        "-ar",
        &rate,
        "-ac",
        "2",
    ];
    args.extend_from_slice(case.encode);
    args.extend(["-f", case.format, &name]);
    let made = output("ffmpeg", &args);
    if !made.status.success() {
        eprintln!(
            "skipped {}: {}",
            case.name,
            String::from_utf8_lossy(&made.stderr).trim()
        );
        return None;
    }
    std::fs::read(&path).ok()
}

#[test]
fn chunks_of_several_frames_match_ffmpeg_parsers() {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped: ffmpeg or ffprobe is missing");
        return;
    }
    let dir = scratch("framed");
    let mut ran = 0;
    for case in CASES {
        let Some(stream) = elementary(&dir, case) else {
            continue;
        };
        let path = dir.join(format!("{}.avi", case.name));
        let file = byte_clock_avi(
            &stream,
            case.chunk,
            case.tag,
            2,
            case.sample_rate,
            case.byte_rate,
        );
        std::fs::write(&path, file).expect("write");
        let (bases, probed) = ffprobe(&path);
        assert_eq!(bases, vec![(1, i64::from(case.byte_rate))], "{}", case.name);
        let mut demux = open_path(&path);
        let packets = drain(&mut demux);
        let theirs = &probed[0];
        assert!(theirs.len() > 20, "{}: {} frames", case.name, theirs.len());
        assert!(
            packets.len() > stream.len() / case.chunk,
            "{}: chunks were not cut",
            case.name
        );
        // FFmpeg's parser drops a partial last frame; this crate returns it.
        let whole = packets.len().min(theirs.len());
        let mut widest = 0;
        assert!(packets.len() - whole <= 1, "{}", case.name);
        for (index, (ours, theirs)) in packets.iter().zip(theirs).take(whole).enumerate() {
            assert_eq!(
                ours.bytes().len(),
                theirs.size,
                "{} frame {index} size",
                case.name
            );
            assert_eq!(
                adler32(ours.bytes()),
                theirs.adler,
                "{} frame {index} bytes",
                case.name
            );
            let dts = theirs.dts.expect("dts");
            let time = ours.timestamp().decode().expect("time");
            widest = widest.max((time - dts).abs());
            // FFmpeg adds each frame's rounded duration to the chunk's time;
            // this crate takes the time of the frame's first byte.
            assert!(
                (time - dts).abs() <= 2,
                "{} frame {index}: {time} against {dts}",
                case.name
            );
        }
        eprintln!(
            "  {}: {} frames from {} chunks, {} more here, times within {widest} ticks",
            case.name,
            whole,
            stream.len().div_ceil(case.chunk),
            packets.len() - whole
        );
        ran += 1;
    }
    assert!(ran > 0, "no case ran");
}
