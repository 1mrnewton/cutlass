//! afconvert and ffmpeg write CAF files, this crate reads them, and both
//! read and decode the files this crate writes. Each part skips, and says
//! so, when its tool is missing (CI runs on Linux without afconvert).

use std::fs::File;
use std::path::Path;

use cutlass_caf::{CafDemux, CafMux};
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};

use crate::support::{
    TempDir, afinfo, ffprobe, open, packets, path, remux, run, source_frames, tool, wave, wave_data,
};

const FRAMES: usize = 66_150;

fn source(dir: &TempDir) -> (std::path::PathBuf, Vec<u8>) {
    let frames = source_frames(FRAMES);
    let file = dir.join("source.wav");
    std::fs::write(&file, wave(&frames, 44_100, 2)).expect("source");
    (file, frames)
}

/// The priming and remainder this crate reads, zero without a table.
fn trim_of(demux: &CafDemux<File>) -> (i64, i64) {
    demux.packet_table().map_or((0, 0), |header| {
        (
            i64::from(header.priming_frames),
            i64::from(header.remainder_frames),
        )
    })
}

/// libavformat times packets from minus the priming and shortens the last
/// one by the remainder; this crate times from zero and reports both as
/// trim. Apart from that the packets must be the same bytes.
fn matches_libavformat(file: &Path) -> bool {
    let Some(probed) = ffprobe(file) else {
        return false;
    };
    let ours = packets(file);
    let (priming, remainder) = trim_of(&open(file));
    let name = file.display();
    assert_eq!(probed.packets.len(), ours.len(), "{name}: packet count");
    for (index, (packet, (pts, duration, size))) in ours.iter().zip(&probed.packets).enumerate() {
        let time = packet.timestamp();
        assert_eq!(
            *pts,
            time.presentation().expect("pts") - priming,
            "{name}: pts {index}"
        );
        assert_eq!(*size, packet.bytes().len(), "{name}: size {index}");
        let mut expect = time.duration().expect("duration");
        if index + 1 == ours.len() {
            expect -= remainder;
        }
        assert_eq!(*duration, expect, "{name}: duration {index}");
    }
    let payload: Vec<u8> = ours
        .iter()
        .flat_map(|packet| packet.bytes().to_vec())
        .collect();
    assert!(payload == probed.payload, "{name}: payload differs");
    true
}

/// afinfo's packet, byte and frame counts against this crate's.
fn matches_afinfo(file: &Path) -> bool {
    let Some(info) = afinfo(file) else {
        return false;
    };
    let demux = open(file);
    let name = file.display();
    assert_eq!(Some(info.packets), demux.packet_count(), "{name}: packets");
    let bytes: usize = packets(file)
        .iter()
        .map(|packet| packet.bytes().len())
        .sum();
    assert_eq!(
        info.bytes,
        u64::try_from(bytes).expect("bytes"),
        "{name}: bytes"
    );
    let total = demux.total_frames().expect("frames");
    match demux.packet_table() {
        Some(header) => {
            assert_eq!(info.priming, u64::from(header.priming_frames), "{name}");
            assert_eq!(info.remainder, u64::from(header.remainder_frames), "{name}");
            assert_eq!(info.valid, header.valid_frames, "{name}");
        }
        None => assert_eq!((info.valid, info.priming, info.remainder), (total, 0, 0)),
    }
    assert!(
        info.valid + info.priming + info.remainder <= total.max(info.valid),
        "{name}"
    );
    true
}

/// Decodes with afconvert to WAVE and returns the samples.
///
/// No `@rate`: with one, afconvert's decode of the same file differs from
/// run to run. And the same file names every time: AudioToolbox's AAC
/// decode depends on the decoding process's memory layout, which the
/// length of its arguments shifts, so two files decoded under names of
/// different lengths can differ in the last bits (README, "AudioToolbox's
/// AAC decode"). With equal arguments the decode is repeatable.
fn af_decode(file: &Path, dir: &TempDir, format: &str) -> Option<Vec<u8>> {
    let input = dir.join("decode-in.caf");
    let out = dir.join("decode-out.wav");
    std::fs::copy(file, &input).expect("copy for decode");
    let result = run(
        "afconvert",
        &["-f", "WAVE", "-d", format, path(&input), path(&out)],
    );
    let _ = std::fs::remove_file(&input);
    result?;
    let decoded = wave_data(&std::fs::read(&out).expect("decoded"));
    let _ = std::fs::remove_file(&out);
    Some(decoded)
}

const AFCONVERT: &[&str] = &[
    "aac", "alac", "opus", "flac", "ulaw", "alaw", "LEI16", "BEI24", "LEF32", "BEF64", "I8",
    "BEI32", "ima4",
];

#[test]
fn afconvert_files_match_libavformat_and_afinfo() {
    if !tool("afconvert") || !tool("afinfo") {
        eprintln!("skipped: afconvert and afinfo are not on PATH (they come with macOS)");
        return;
    }
    let dir = TempDir::new("afconvert");
    let (wav, _) = source(&dir);
    let ffmpeg = tool("ffprobe") && tool("ffmpeg");
    if !ffmpeg {
        eprintln!("ffprobe is not on PATH: comparing with afinfo only");
    }
    for format in AFCONVERT {
        let file = dir.join(&format!("{format}.caf"));
        run(
            "afconvert",
            &["-f", "caff", "-d", format, path(&wav), path(&file)],
        )
        .unwrap_or_else(|| panic!("afconvert {format}"));
        assert!(matches_afinfo(&file), "afinfo read {format}");
        if *format == "ima4" {
            // libavformat 8.1 refuses this file: its pakt says 0 packets
            // and comes before data, and cafdec.c needs the data size then.
            continue;
        }
        if ffmpeg {
            assert!(matches_libavformat(&file), "ffprobe read {format}");
        }
    }
}

const FFMPEG: &[&str] = &[
    "ac3",
    "libmp3lame",
    "alac",
    "flac",
    "pcm_s16be",
    "pcm_s24le",
    "pcm_s32be",
    "pcm_f32le",
    "pcm_f64be",
    "pcm_alaw",
    "pcm_mulaw",
    "pcm_s8",
];

#[test]
fn ffmpeg_files_match_libavformat_and_afinfo() {
    if !tool("ffmpeg") || !tool("ffprobe") {
        eprintln!("skipped: ffmpeg and ffprobe are not on PATH");
        return;
    }
    let dir = TempDir::new("ffmpeg");
    let (wav, _) = source(&dir);
    let afinfo = tool("afinfo");
    for codec in FFMPEG {
        let file = dir.join(&format!("{codec}.caf"));
        if run(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-y",
                "-i",
                path(&wav),
                "-c:a",
                codec,
                path(&file),
            ],
        )
        .is_none()
        {
            eprintln!("skipped {codec}: this ffmpeg cannot write it");
            continue;
        }
        assert!(matches_libavformat(&file), "ffprobe read {codec}");
        if afinfo && !matches_afinfo(&file) {
            eprintln!("afinfo cannot read ffmpeg's {codec} CAF");
        }
    }
}

/// Remuxes each file through this crate; afinfo, ffprobe and afconvert see
/// the same packets and decode the same samples as from the original.
#[test]
fn remuxed_files_read_and_decode_like_the_originals() {
    let apple = tool("afconvert") && tool("afinfo");
    let ffmpeg = tool("ffmpeg") && tool("ffprobe");
    if !apple && !ffmpeg {
        eprintln!("skipped: neither afconvert nor ffmpeg is on PATH");
        return;
    }
    if !apple {
        eprintln!("afconvert is not on PATH: checking remuxes with ffmpeg only");
    }
    if !ffmpeg {
        eprintln!("ffmpeg is not on PATH: checking remuxes with afconvert only");
    }
    let dir = TempDir::new("remux");
    let (wav, frames) = source(&dir);
    let mut originals = Vec::new();
    if apple {
        for format in [
            "aac", "alac", "opus", "flac", "ulaw", "alaw", "LEI16", "BEF64",
        ] {
            let file = dir.join(&format!("af-{format}.caf"));
            run(
                "afconvert",
                &["-f", "caff", "-d", format, path(&wav), path(&file)],
            )
            .expect("afconvert");
            originals.push((
                file,
                format == "alac" || format == "flac" || format == "LEI16",
            ));
        }
    }
    if ffmpeg {
        for codec in ["ac3", "libmp3lame", "alac", "flac", "pcm_s24be"] {
            let file = dir.join(&format!("ff-{codec}.caf"));
            if run(
                "ffmpeg",
                &[
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    path(&wav),
                    "-c:a",
                    codec,
                    path(&file),
                ],
            )
            .is_some()
            {
                originals.push((file, codec == "alac" || codec == "flac"));
            }
        }
    }
    for (original, lossless) in originals {
        let stem = original
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("stem");
        let ours = dir.join(&format!("{stem}-ours.caf"));
        let name = ours.display();
        let flac =
            open(&original).streams()[0].params().codec() == &StreamCodec::Known(CodecId::Flac);
        if flac && open(&original).streams()[0].params().extradata().is_empty() {
            // ffmpeg writes FLAC with no `kuki`; AudioToolbox cannot decode
            // that, so this muxer wants the STREAMINFO and refuses.
            assert!(remux(&original, &ours).is_err(), "{name}");
            continue;
        }
        remux(&original, &ours).unwrap_or_else(|error| panic!("{name}: {error}"));
        let theirs = packets(&original);
        let mine = packets(&ours);
        assert_eq!(theirs.len(), mine.len(), "{name}");
        for (a, b) in theirs.iter().zip(&mine) {
            assert_eq!(a.bytes(), b.bytes(), "{name}");
            assert_eq!(a.timestamp(), b.timestamp(), "{name}");
        }
        assert_eq!(
            open(&original).streams()[0],
            open(&ours).streams()[0],
            "{name}"
        );
        if apple {
            let before = afinfo(&original);
            let after = afinfo(&ours).unwrap_or_else(|| panic!("afinfo cannot read {name}"));
            if let Some(before) = before {
                assert_eq!(before, after, "{name}");
            }
            if let Some(decoded) = af_decode(&original, &dir, "LEF32") {
                let again = af_decode(&ours, &dir, "LEF32")
                    .unwrap_or_else(|| panic!("afconvert cannot decode {name}"));
                assert!(decoded == again, "{name}: afconvert decodes other samples");
            }
            if lossless {
                let exact = af_decode(&ours, &dir, "LEI16").expect("decode");
                assert!(exact == frames, "{name}: not the source samples");
            }
        }
        if ffmpeg {
            let before = ffprobe(&original).expect("ffprobe original");
            let after = ffprobe(&ours).unwrap_or_else(|| panic!("ffprobe cannot read {name}"));
            assert_eq!(before.packets, after.packets, "{name}");
            assert!(before.payload == after.payload, "{name}");
            let decode = |file: &Path| {
                run(
                    "ffmpeg",
                    &["-v", "error", "-i", path(file), "-f", "f32le", "-"],
                )
            };
            let a = decode(&original).expect("ffmpeg decode original");
            let b = decode(&ours).unwrap_or_else(|| panic!("ffmpeg cannot decode {name}"));
            assert!(a == b, "{name}: ffmpeg decodes other samples");
        }
    }
}

fn pcm_stream(extradata: Vec<u8>) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 44_100, 2, extradata)
            .expect("params"),
        timescale: Timescale::new(44_100),
        language: None,
    }
}

fn write_caf(file: &Path, stream: NewStream, units: &[(Vec<u8>, i64)]) {
    let rate = stream.timescale;
    let mut mux = CafMux::new(File::create(file).expect("create"));
    mux.add_stream(stream).expect("add");
    let mut pts = 0;
    for (bytes, frames) in units {
        let time = Timestamp::try_new(Some(pts), Some(pts), Some(*frames)).expect("time");
        let packet = Packet::try_new(0, bytes.clone(), time, rate, true).expect("packet");
        mux.write_packet(&packet).expect("write");
        pts += frames;
    }
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: rate,
        ticks: pts,
    }])
    .expect("finish");
}

/// The source samples in another PCM layout, in runs of 1000 frames.
fn pcm_units(frames: &[u8], convert: impl Fn(i16) -> Vec<u8>) -> Vec<(Vec<u8>, i64)> {
    frames
        .chunks(4000)
        .map(|run| {
            let bytes = run
                .chunks(2)
                .flat_map(|sample| convert(i16::from_le_bytes([sample[0], sample[1]])))
                .collect();
            (bytes, i64::try_from(run.len() / 4).expect("frames"))
        })
        .collect()
}

type Convert = Box<dyn Fn(i16) -> Vec<u8>>;

#[test]
fn pcm_this_crate_writes_decodes_to_the_source() {
    let apple = tool("afconvert") && tool("afinfo");
    let ffmpeg = tool("ffmpeg") && tool("ffprobe");
    if !apple && !ffmpeg {
        eprintln!("skipped: neither afconvert nor ffmpeg is on PATH");
        return;
    }
    let dir = TempDir::new("pcm");
    let frames = source_frames(FRAMES);
    let layouts: [(&str, [u8; 3], Convert); 5] = [
        (
            "s16le",
            [1, 0, 0],
            Box::new(|s: i16| s.to_le_bytes().to_vec()),
        ),
        (
            "s16be",
            [1, 1, 0],
            Box::new(|s: i16| s.to_be_bytes().to_vec()),
        ),
        (
            "s24be",
            [2, 1, 0],
            Box::new(|s: i16| (i32::from(s) << 8).to_be_bytes()[1..].to_vec()),
        ),
        (
            "s32le",
            [3, 0, 0],
            Box::new(|s: i16| (i32::from(s) << 16).to_le_bytes().to_vec()),
        ),
        (
            "f32be",
            [4, 1, 0],
            Box::new(|s: i16| (f32::from(s) / 32_768.0).to_be_bytes().to_vec()),
        ),
    ];
    for (name, extradata, convert) in layouts {
        let file = dir.join(&format!("{name}.caf"));
        write_caf(
            &file,
            pcm_stream(extradata.to_vec()),
            &pcm_units(&frames, convert),
        );
        if apple {
            let info = afinfo(&file).unwrap_or_else(|| panic!("afinfo cannot read {name}"));
            assert_eq!(info.valid, u64::try_from(FRAMES).expect("frames"), "{name}");
            assert_eq!(
                info.packets,
                u64::try_from(FRAMES).expect("frames"),
                "{name}"
            );
            let decoded = af_decode(&file, &dir, "LEI16").expect("afconvert decode");
            assert!(decoded == frames, "{name}: afconvert decodes other samples");
        }
        if ffmpeg {
            assert!(matches_libavformat(&file), "{name}");
            let decoded = run(
                "ffmpeg",
                &["-v", "error", "-i", path(&file), "-f", "s16le", "-"],
            )
            .expect("ffmpeg decode");
            assert!(decoded == frames, "{name}: ffmpeg decodes other samples");
        }
    }
}

/// Splits a raw E-AC-3 stream into sync frames, with the frames of audio
/// in each.
fn eac3_frames(raw: &[u8]) -> Vec<(Vec<u8>, i64)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 6 <= raw.len() {
        assert_eq!(&raw[at..at + 2], [0x0B, 0x77], "sync at {at}");
        let size = ((usize::from(raw[at + 2] & 0x07) << 8) | usize::from(raw[at + 3])) + 1;
        let fscod = raw[at + 4] >> 6;
        let blocks = if fscod == 3 {
            6
        } else {
            [1, 2, 3, 6][usize::from((raw[at + 4] >> 4) & 3)]
        };
        let end = at + size * 2;
        out.push((raw[at..end].to_vec(), blocks * 256));
        at = end;
    }
    out
}

#[test]
fn e_ac_3_this_crate_writes_is_read_by_audiotoolbox() {
    if !tool("ffmpeg") || !tool("afinfo") || !tool("afconvert") {
        eprintln!("skipped: needs ffmpeg to encode and afinfo and afconvert to read");
        return;
    }
    // ffmpeg neither writes nor reads E-AC-3 in CAF (libavformat's caf.c
    // has no `ec-3` tag), so its raw stream is framed here, muxed by this
    // crate, and read back by AudioToolbox.
    let dir = TempDir::new("eac3");
    let (wav, _) = source(&dir);
    let raw = dir.join("source.eac3");
    run(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-y",
            "-i",
            path(&wav),
            "-c:a",
            "eac3",
            path(&raw),
        ],
    )
    .expect("ffmpeg eac3");
    let units = eac3_frames(&std::fs::read(&raw).expect("raw"));
    let file = dir.join("ours.caf");
    let stream = NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(CodecId::Eac3), 44_100, 2, Vec::new())
            .expect("params"),
        timescale: Timescale::new(44_100),
        language: None,
    };
    write_caf(&file, stream, &units);
    let read = packets(&file);
    assert_eq!(read.len(), units.len());
    for (packet, (bytes, _)) in read.iter().zip(&units) {
        assert_eq!(packet.bytes(), &bytes[..]);
    }
    let frames: i64 = units.iter().map(|unit| unit.1).sum();
    let info = afinfo(&file).expect("afinfo reads E-AC-3 in CAF");
    assert_eq!(info.packets, u64::try_from(units.len()).expect("len"));
    assert_eq!(info.valid, u64::try_from(frames).expect("frames"));
    let decoded = af_decode(&file, &dir, "LEF32").expect("afconvert decodes it");
    let decoded = i64::try_from(decoded.len() / 8).expect("frames");
    // AudioToolbox drops its decoder delay (288 frames on this file) when
    // the file names no priming, so the length is close, not equal.
    assert!(
        (frames - decoded).abs() < 1536,
        "{decoded} of {frames} frames"
    );
}
