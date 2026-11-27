//! A video track's Colour against libavformat. Files FFmpeg writes with
//! colour tags, VP9, AV1, H.264 and HEVC, open with the primaries,
//! transfer, matrix and range `ffprobe -nofind_stream_info` reads (the
//! demuxer's reading, before any decoder adds its own). A WebM this crate
//! writes with a light level and a mastering display reads in ffprobe with
//! the same values, and FFmpeg's copy of it reads back here the same.
//! Each test says it skipped and passes without `ffmpeg` and `ffprobe`.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::time::Duration;

use cutlass_codec::access::{ColorRange, ContentLight, MasteringDisplay};
use cutlass_codec::{Packet, StreamColor};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_matroska::MatroskaMux;

use crate::support::{ffmpeg, ffmpeg_tools, open, output_timeout, read_all, scratch};

/// Every `key=value` libavformat's demuxer gives for the first video
/// stream, side data included (the first of a repeated key).
fn probed(path: &Path) -> HashMap<String, String> {
    let out = output_timeout(
        "ffprobe",
        &[
            "-v",
            "error",
            "-nofind_stream_info",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "default=nw=1",
            &path.display().to_string(),
        ],
        Duration::from_secs(60),
    )
    .expect("ffprobe");
    assert!(out.status.success(), "ffprobe failed on {}", path.display());
    let mut fields = HashMap::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Some((key, value)) = line.split_once('=') {
            fields
                .entry(key.to_owned())
                .or_insert_with(|| value.to_owned());
        }
    }
    fields
}

/// FFmpeg's names (`libavutil/pixdesc.c`) for the codes the cases use.
fn primaries_code(name: &str) -> Option<u8> {
    match name {
        "unknown" => None,
        "bt709" => Some(1),
        "smpte170m" => Some(6),
        "bt2020" => Some(9),
        "smpte432" => Some(12),
        other => panic!("primaries {other}"),
    }
}

fn transfer_code(name: &str) -> Option<u8> {
    match name {
        "unknown" => None,
        "bt709" => Some(1),
        "smpte170m" => Some(6),
        "iec61966-2-1" => Some(13),
        "smpte2084" => Some(16),
        "arib-std-b67" => Some(18),
        other => panic!("transfer {other}"),
    }
}

fn matrix_code(name: &str) -> Option<u8> {
    match name {
        "unknown" => None,
        "gbr" => Some(0),
        "bt709" => Some(1),
        "smpte170m" => Some(6),
        "bt2020nc" => Some(9),
        other => panic!("matrix {other}"),
    }
}

fn range_of(name: &str) -> ColorRange {
    match name {
        "unknown" => ColorRange::Unspecified,
        "tv" => ColorRange::Limited,
        "pc" => ColorRange::Full,
        other => panic!("range {other}"),
    }
}

/// One FFmpeg file: the encoder's arguments and the tags `setparams` puts
/// on the frames (FFmpeg 9 takes the stream's colour from its frames).
struct Case {
    name: &'static str,
    codec: &'static [&'static str],
    tags: Option<&'static str>,
}

const CASES: &[Case] = &[
    Case {
        name: "vp9_pq.webm",
        codec: &[
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p10le",
            "-b:v",
            "200k",
        ],
        tags: Some("color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv"),
    },
    // The options alone, without tagged frames: FFmpeg 9 writes only the
    // matrix and the range.
    Case {
        name: "vp9_options_only.webm",
        codec: &[
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p10le",
            "-b:v",
            "200k",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
            "-color_range",
            "tv",
        ],
        tags: None,
    },
    Case {
        name: "av1_hlg.mkv",
        codec: &[
            "-c:v",
            "libsvtav1",
            "-pix_fmt",
            "yuv420p10le",
            "-preset",
            "12",
        ],
        tags: Some("color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv"),
    },
    Case {
        name: "h264_601_full.mkv",
        codec: &["-c:v", "libx264", "-pix_fmt", "yuv420p"],
        tags: Some("color_primaries=smpte170m:color_trc=smpte170m:colorspace=smpte170m:range=pc"),
    },
    Case {
        name: "hevc_p3_srgb.mkv",
        codec: &["-c:v", "libx265", "-x265-params", "log-level=error"],
        tags: Some("color_primaries=smpte432:color_trc=iec61966-2-1:colorspace=bt709:range=tv"),
    },
    Case {
        name: "h264_rgb_matrix.mkv",
        codec: &["-c:v", "libx264rgb", "-pix_fmt", "rgb24"],
        tags: Some("color_primaries=bt709:color_trc=bt709:colorspace=gbr:range=pc"),
    },
    Case {
        name: "h264_untagged.mkv",
        codec: &["-c:v", "libx264", "-pix_fmt", "yuv420p"],
        tags: None,
    },
];

#[test]
fn ffmpeg_colour_reads_as_libavformat_reads_it() {
    if !ffmpeg_tools() {
        return;
    }
    let dir = scratch("colour");
    let mut checked = 0;
    for case in CASES {
        let path = dir.join(case.name);
        let path_text = path.display().to_string();
        let mut args = vec!["-f", "lavfi", "-i", "testsrc2=s=160x96:d=0.4"];
        let filter;
        if let Some(tags) = case.tags {
            filter = format!("setparams={tags}");
            args.extend(["-vf", &filter]);
        }
        args.extend_from_slice(case.codec);
        args.push(&path_text);
        if !ffmpeg(&args) {
            eprintln!("skipped {}: ffmpeg could not write it here", case.name);
            continue;
        }
        let theirs = probed(&path);
        let field = |key: &str| theirs.get(key).map_or("unknown", String::as_str);
        let demux = open(&path);
        let ours = demux.streams()[0].params().color();
        let name = case.name;
        assert_eq!(
            ours.primaries,
            primaries_code(field("color_primaries")),
            "{name}"
        );
        assert_eq!(
            ours.transfer,
            transfer_code(field("color_transfer")),
            "{name}"
        );
        assert_eq!(ours.matrix, matrix_code(field("color_space")), "{name}");
        assert_eq!(ours.range, range_of(field("color_range")), "{name}");
        assert_eq!(ours.content_light, None, "{name}");
        assert_eq!(ours.mastering_display, None, "{name}");
        assert!(!theirs.contains_key("side_data_type"), "{name}");
        if case.tags.is_some() {
            assert!(
                ours.primaries.is_some() && ours.transfer.is_some(),
                "{name}"
            );
        }
        checked += 1;
    }
    eprintln!("{checked} of {} FFmpeg files checked", CASES.len());
}

/// An HDR10 stream: BT.2020, PQ, broadcast range, 1000 and 400 cd/m², and
/// a P3 mastering display at D65 from 0.005 to 1000 cd/m².
const HDR10: StreamColor = StreamColor {
    primaries: Some(9),
    transfer: Some(16),
    matrix: Some(9),
    range: ColorRange::Limited,
    content_light: Some(ContentLight {
        max_cll: 1000,
        max_fall: 400,
    }),
    mastering_display: Some(MasteringDisplay {
        primaries: [[44564, 20972], [17367, 45220], [9830, 3932]],
        white_point: [20493, 21561],
        max_luminance: 256_000,
        min_luminance: 82,
    }),
};

/// Copies the one video track of `input` into a WebM at `output`, its
/// stream carrying `color`.
fn write_with_color(input: &Path, output: &Path, color: StreamColor) {
    let mut demux = open(input);
    let stream = demux.streams()[0].clone();
    let packets = read_all(&mut demux);
    let file = std::io::BufWriter::new(File::create(output).expect("create"));
    let mut mux = MatroskaMux::webm(file);
    let track = mux
        .add_stream(NewStream {
            params: stream.params().clone().with_color(color),
            timescale: stream.timescale(),
            language: None,
        })
        .expect("stream");
    let mut end = 0;
    for (index, packet) in packets.iter().enumerate() {
        let pts = packet.timestamp().presentation().expect("pts");
        let next = packets
            .get(index + 1)
            .and_then(|next| next.timestamp().presentation());
        let duration = next.map_or(40, |next| next - pts);
        end = pts + duration;
        let time =
            cutlass_codec::Timestamp::try_new(Some(pts), Some(pts), Some(duration)).expect("time");
        let copy = Packet::try_new(
            track.get(),
            packet.bytes().to_vec(),
            time,
            packet.timescale(),
            packet.keyframe(),
        )
        .expect("packet");
        mux.write_packet(&copy).expect("write");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: stream.timescale(),
        ticks: end,
    }])
    .expect("finish");
}

/// `a/b` as ffprobe prints a rational.
fn fraction(text: &str) -> f64 {
    let (num, den) = text.split_once('/').unwrap_or((text, "1"));
    num.parse::<f64>().expect("numerator") / den.parse::<f64>().expect("denominator")
}

#[test]
fn a_light_level_and_mastering_display_read_in_libavformat_and_back() {
    if !ffmpeg_tools() {
        return;
    }
    let dir = scratch("colour_hdr");
    let source = dir.join("source.webm");
    let source_text = source.display().to_string();
    if !ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=160x96:d=0.4",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "100k",
        &source_text,
    ]) {
        eprintln!("skipped: ffmpeg has no libvpx-vp9");
        return;
    }
    let ours = dir.join("ours.webm");
    write_with_color(&source, &ours, HDR10);

    let theirs = probed(&ours);
    let field = |key: &str| theirs.get(key).map_or("", String::as_str);
    assert_eq!(field("color_primaries"), "bt2020");
    assert_eq!(field("color_transfer"), "smpte2084");
    assert_eq!(field("color_space"), "bt2020nc");
    assert_eq!(field("color_range"), "tv");
    let Some(display) = HDR10.mastering_display else {
        panic!("display");
    };
    let chromaticities = [
        ("red_x", display.primaries[0][0]),
        ("red_y", display.primaries[0][1]),
        ("green_x", display.primaries[1][0]),
        ("green_y", display.primaries[1][1]),
        ("blue_x", display.primaries[2][0]),
        ("blue_y", display.primaries[2][1]),
        ("white_point_x", display.white_point[0]),
        ("white_point_y", display.white_point[1]),
    ];
    for (key, units) in chromaticities {
        let want = f64::from(units) / 65_536.0;
        let got = fraction(field(key));
        assert!((got - want).abs() < 0.5 / 65_536.0, "{key}: {got} {want}");
    }
    let max = fraction(field("max_luminance"));
    let min = fraction(field("min_luminance"));
    assert!((max - 1000.0).abs() < 0.5 / 256.0, "max {max}");
    assert!((min - 82.0 / 16_384.0).abs() < 0.5 / 16_384.0, "min {min}");
    assert_eq!(field("max_content"), "1000");
    assert_eq!(field("max_average"), "400");

    let copy = dir.join("ffmpeg_copy.webm");
    let copy_text = copy.display().to_string();
    let ours_text = ours.display().to_string();
    assert!(ffmpeg(&["-i", &ours_text, "-c", "copy", &copy_text]));
    let demux = open(&copy);
    assert_eq!(demux.streams()[0].params().color(), HDR10);
}
