//! This muxer's output read by ffprobe, ffmpeg and GStreamer.
//!
//! Sources: FFmpeg 8.1 writes MP4 and QuickTime files from its test
//! sources; `cutlass-isobmff` reads them and [`TsMux`] writes them again as
//! TS and as M2TS. Oracles: ffprobe's packets of the source (edit lists
//! ignored, as the remux keeps sample times) against ffprobe's packets of
//! the output, ffmpeg decoding the output with no complaint, GStreamer's
//! `tsdemux` reading it to the end, and a walk over the transport packets
//! for continuity, PCR spacing, table repetition and the PCR lead.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_isobmff::IsoBmffFile;
use cutlass_mpegts::packet::PacketSize;
use cutlass_mpegts::{MuxOptions, TsMux};

use crate::support::{
    ffmpeg, ffmpeg_decode_errors, ffmpeg_present, ffprobe, gstreamer_errors, ours, scratch,
};

const DELAY: i64 = 126_000;

struct Source {
    name: &'static str,
    args: &'static [&'static str],
}

const VIDEO: &str = "testsrc2=size=320x240:rate=25:duration=5";
const AUDIO: &str = "sine=frequency=700:sample_rate=48000:duration=5";

const SOURCES: &[Source] = &[
    Source {
        name: "h264_aac.mp4",
        args: &[
            "-c:v", "libx264", "-preset", "veryfast", "-bf", "2", "-c:a", "aac",
        ],
    },
    Source {
        name: "hevc_opus.mp4",
        args: &[
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=none:bframes=2:keyint=25",
            "-tag:v",
            "hvc1",
            "-c:a",
            "libopus",
        ],
    },
    Source {
        name: "mpeg2_ac3.mov",
        args: &["-c:v", "mpeg2video", "-g", "12", "-bf", "2", "-c:a", "ac3"],
    },
    Source {
        name: "h264_mp3.mp4",
        args: &[
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "libmp3lame",
        ],
    },
    Source {
        name: "h264_eac3.mp4",
        args: &["-c:v", "libx264", "-preset", "veryfast", "-c:a", "eac3"],
    },
    Source {
        name: "mpeg4_aac.mp4",
        args: &["-c:v", "mpeg4", "-bf", "2", "-g", "12", "-c:a", "aac"],
    },
    Source {
        name: "h264_aac_44100.mp4",
        args: &[
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-ar", "44100",
        ],
    },
];

/// Remuxes the MP4 or QuickTime file at `source` into `out`.
pub fn remux(source: &Path, out: &Path, size: PacketSize) -> usize {
    let file = File::open(source).expect("source");
    let mut demux = IsoBmffFile::open(BufReader::new(file)).expect("isobmff");
    let writer = BufWriter::new(File::create(out).expect("create"));
    let mut mux = TsMux::with_options(
        writer,
        MuxOptions {
            packet_size: size,
            ..MuxOptions::default()
        },
    );
    let mut map = Vec::new();
    for stream in demux.streams() {
        let track = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: stream.timescale(),
                language: stream.language().cloned(),
            })
            .expect("a stream the muxer carries");
        map.push((stream.id().get(), track));
    }
    let mut count = 0;
    while let Some(packet) = demux.read_packet().expect("source read") {
        let (_, track) = map
            .iter()
            .find(|(id, _)| *id == packet.stream())
            .expect("track");
        let packet = cutlass_codec::Packet::try_new(
            track.get(),
            packet.bytes().to_vec(),
            packet.timestamp(),
            packet.timescale(),
            packet.keyframe(),
        )
        .expect("packet");
        mux.write_packet(&packet).expect("write");
        count += 1;
    }
    let durations: [TrackDuration; 0] = [];
    mux.finish(&durations).expect("finish");
    count
}

fn to_clock(ticks: i64, (num, den): (i64, i64)) -> i64 {
    let scaled = i128::from(ticks) * 90_000 * i128::from(num);
    let den = i128::from(den);
    i64::try_from((2 * scaled + den).div_euclid(2 * den)).expect("time")
}

/// Packet by packet, the output against the source as ffprobe reads both:
/// time plus the mux delay, keyframes, and audio sizes (AAC gains its ADTS
/// header).
fn compare(source: &Path, out: &Path) -> Vec<String> {
    let before = ffprobe(source, &["-ignore_editlist", "1"]);
    let after = ffprobe(out, &[]);
    let mut problems = Vec::new();
    if before.len() != after.len() {
        problems.push(format!("{} streams, then {}", before.len(), after.len()));
        return problems;
    }
    for (index, (a, b)) in before.iter().zip(&after).enumerate() {
        if a.packets.len() != b.packets.len() {
            problems.push(format!(
                "stream {index} {}: {} packets, then {}",
                a.codec,
                a.packets.len(),
                b.packets.len()
            ));
        }
        for (number, (x, y)) in a.packets.iter().zip(&b.packets).enumerate() {
            let want = x.pts.map(|pts| to_clock(pts, a.time_base) + DELAY);
            let got = y.pts.map(|pts| to_clock(pts, b.time_base));
            let mut fields = Vec::new();
            if let (Some(want), Some(got)) = (want, got)
                && (want - got).abs() > 1
            {
                fields.push("pts");
            }
            if x.key != y.key {
                fields.push("keyframe");
            }
            let size = match a.codec.as_str() {
                "aac" => Some(x.size + 7),
                "opus" | "mp3" | "ac3" | "eac3" | "mp2" => Some(x.size),
                _ => None,
            };
            if size.is_some_and(|size| size != y.size) {
                fields.push("size");
            }
            if !fields.is_empty() {
                problems.push(format!(
                    "stream {index} {} packet {number}: {fields:?} source {x:?} output {y:?}",
                    a.codec
                ));
            }
            if problems.len() > 12 {
                return problems;
            }
        }
    }
    problems
}

/// What a walk over the transport packets found.
#[derive(Debug, Default)]
pub struct Structure {
    pub packets: usize,
    pub continuity_errors: usize,
    pub pcrs: usize,
    pub pcr_backwards: usize,
    /// Longest PCR gap, in 27 MHz ticks.
    pub max_pcr_gap: u64,
    pub pats: usize,
    /// Longest PAT gap by PCR, in 27 MHz ticks.
    pub max_pat_gap: u64,
    /// Least DTS (or PTS) minus the PCR before it, in 27 MHz ticks.
    pub min_lead: i64,
    pub arrival_backwards: usize,
}

pub fn structure(bytes: &[u8], size: PacketSize) -> Structure {
    let (stride, offset) = match size {
        PacketSize::M2ts => (192, 4),
        _ => (188, 0),
    };
    let mut report = Structure {
        min_lead: i64::MAX,
        ..Structure::default()
    };
    let mut continuity: Vec<Option<u8>> = vec![None; 8192];
    let mut last_pcr: Option<u64> = None;
    let mut last_pat: Option<u64> = None;
    let mut last_arrival = None;
    for unit in bytes.chunks_exact(stride) {
        let packet = &unit[offset..];
        assert_eq!(packet[0], 0x47, "sync");
        report.packets += 1;
        if offset == 4 {
            let arrival = u32::from_be_bytes([unit[0], unit[1], unit[2], unit[3]]) & 0x3FFF_FFFF;
            if last_arrival.is_some_and(|last| arrival < last) {
                report.arrival_backwards += 1;
            }
            last_arrival = Some(arrival);
        }
        let pid = usize::from(u16::from_be_bytes([packet[1] & 0x1F, packet[2]]));
        let control = (packet[3] >> 4) & 3;
        let counter = packet[3] & 0x0F;
        let expected = continuity[pid].map(|last| {
            if control & 1 != 0 {
                (last + 1) & 0x0F
            } else {
                last
            }
        });
        if expected.is_some_and(|expected| expected != counter) {
            report.continuity_errors += 1;
        }
        continuity[pid] = Some(counter);
        let mut payload = 4;
        if control & 2 != 0 {
            let length = usize::from(packet[4]);
            payload = 5 + length;
            if length > 0 && packet[5] & 0x10 != 0 {
                let b = &packet[6..12];
                let base = (u64::from(b[0]) << 25)
                    | (u64::from(b[1]) << 17)
                    | (u64::from(b[2]) << 9)
                    | (u64::from(b[3]) << 1)
                    | (u64::from(b[4]) >> 7);
                let pcr = base * 300 + ((u64::from(b[4] & 1) << 8) | u64::from(b[5]));
                if let Some(last) = last_pcr {
                    if pcr < last {
                        report.pcr_backwards += 1;
                    }
                    report.max_pcr_gap = report.max_pcr_gap.max(pcr.saturating_sub(last));
                }
                last_pcr = Some(pcr);
                report.pcrs += 1;
            }
        }
        let start = packet[1] & 0x40 != 0;
        if pid == 0 && start {
            report.pats += 1;
            if let (Some(pcr), Some(last)) = (last_pcr, last_pat) {
                report.max_pat_gap = report.max_pat_gap.max(pcr.saturating_sub(last));
            }
            if last_pcr.is_some() {
                last_pat = last_pcr;
            }
        }
        if control & 1 != 0 && start && pid != 0 && pid != 0x1000 && payload + 19 <= 188 {
            let pes = &packet[payload..];
            if pes[..3] == [0, 0, 1] && pes[7] & 0x80 != 0 {
                let t = if pes[7] & 0x40 != 0 {
                    &pes[14..19]
                } else {
                    &pes[9..14]
                };
                let time = (u64::from((t[0] >> 1) & 7) << 30)
                    | (u64::from(t[1]) << 22)
                    | (u64::from(t[2] >> 1) << 15)
                    | (u64::from(t[3]) << 7)
                    | (u64::from(t[4]) >> 1);
                if let Some(pcr) = last_pcr {
                    let lead = i64::try_from(time * 300).unwrap() - i64::try_from(pcr).unwrap();
                    report.min_lead = report.min_lead.min(lead);
                }
            }
        }
    }
    report
}

/// The structure every output must have: no continuity break, a PCR at
/// least every 100 ms and never backwards, tables at least every 500 ms,
/// every timestamp after the PCR that precedes it, arrival times rising.
pub fn check_structure(report: &Structure) -> Vec<String> {
    let mut problems = Vec::new();
    if report.continuity_errors > 0 {
        problems.push(format!("{} continuity errors", report.continuity_errors));
    }
    if report.pcr_backwards > 0 || report.pcrs < 2 || report.max_pcr_gap > 2_700_000 {
        problems.push(format!("PCR: {report:?}"));
    }
    if report.pats < 2 || report.max_pat_gap > 13_500_000 {
        problems.push(format!("PAT: {report:?}"));
    }
    if report.min_lead < 0 {
        problems.push(format!("a timestamp before its PCR: {report:?}"));
    }
    if report.arrival_backwards > 0 {
        problems.push(format!("arrival times go back: {report:?}"));
    }
    problems
}

fn source(dir: &Path, source: &Source) -> PathBuf {
    let path = dir.join(source.name);
    let mut args = vec!["-f", "lavfi", "-i", VIDEO, "-f", "lavfi", "-i", AUDIO];
    args.extend_from_slice(source.args);
    args.push(path.to_str().unwrap());
    ffmpeg(&args);
    path
}

#[test]
fn remuxed_files_read_back_in_ffprobe_ffmpeg_and_gstreamer() {
    if !ffmpeg_present("remuxed_files_read_back_in_ffprobe_ffmpeg_and_gstreamer") {
        return;
    }
    let dir = scratch("mux");
    let mut failures = Vec::new();
    for entry in SOURCES {
        let input = source(&dir, entry);
        for (size, suffix) in [(PacketSize::Ts, "ts"), (PacketSize::M2ts, "m2ts")] {
            let out = dir.join(format!("{}.{suffix}", entry.name));
            let written = remux(&input, &out, size);
            let mut problems = compare(&input, &out);
            let decode = ffmpeg_decode_errors(&out);
            if !decode.is_empty() {
                problems.push(format!("ffmpeg decode: {decode}"));
            }
            if let Some(gst) = gstreamer_errors(&out)
                && !gst.is_empty()
            {
                problems.push(format!("gstreamer: {gst}"));
            }
            problems.extend(check_structure(&structure(
                &std::fs::read(&out).unwrap(),
                size,
            )));
            // This crate reads its own output: every packet written.
            let (_, read) = ours(&out);
            let total: usize = read.iter().map(|(_, packets)| packets.len()).sum();
            if total != written {
                problems.push(format!("read back {total} of {written} packets"));
            }
            if !problems.is_empty() {
                failures.push(format!(
                    "{} as {suffix}:\n  {}",
                    entry.name,
                    problems.join("\n  ")
                ));
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
