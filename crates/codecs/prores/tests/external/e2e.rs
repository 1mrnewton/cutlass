//! A real clip through the whole path: ffmpeg writes ProRes, this
//! repository demuxes, decodes, re-encodes and muxes it, and ffprobe,
//! ffmpeg and AVFoundation open the result. Frames, times, the profile and
//! the color tags must come through.

use std::path::Path;
use std::process::Command;

use cutlass_decode::{CodecId, CodecParams, PixelFormat};
use cutlass_prores::{ProresDecoder, ProresEncoder};

use crate::support::{
    Diff, HD, Scratch, Timed, demux_mov, ffmpeg, local, mux_track, reference, run, samples16, tool,
};

fn probe(ffprobe: &Path, movie: &Path) -> Vec<String> {
    let out = Command::new(ffprobe)
        .args([
            "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
            "stream=codec_name,codec_tag_string,width,height,pix_fmt,color_primaries,color_transfer,color_space,r_frame_rate,nb_read_frames",
            "-of", "default=noprint_wrappers=1",
        ])
        .arg(movie)
        .output()
        .expect("ffprobe");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect()
}

fn frame_times(ffprobe: &Path, movie: &Path) -> Vec<String> {
    let out = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts,duration",
            "-of",
            "csv=p=0",
        ])
        .arg(movie)
        .output()
        .expect("ffprobe");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect()
}

struct Trip {
    name: &'static str,
    clip: &'static str,
    filter: &'static str,
    args: &'static [&'static str],
    pix_fmt: &'static str,
    min_psnr: f64,
}

fn round_trip(trip: &Trip) {
    let Some(ffmpeg) = ffmpeg() else { return };
    let Some(ffprobe) = tool("ffprobe") else {
        crate::support::skip("ffprobe");
        return;
    };
    let Some(clip) = local(trip.clip) else { return };
    let scratch = Scratch::new("e2e");
    let original = scratch.path("original.mov");
    let mut args = vec![
        "-v",
        "error",
        "-y",
        "-i",
        clip.to_str().expect("path"),
        "-frames:v",
        "8",
        "-an",
        "-vf",
        trip.filter,
    ];
    args.extend_from_slice(trip.args);
    args.push(original.to_str().expect("path"));
    assert!(run(&ffmpeg, &args), "{}: ffmpeg", trip.name);

    let track = demux_mov(std::fs::read(&original).expect("read"));
    assert_eq!(track.packets.len(), 8, "{}: frames", trip.name);
    let mut params = CodecParams::with_format(
        CodecId::Prores,
        track.width,
        track.height,
        PixelFormat::None,
    );
    params.extra = track.extra.clone();
    let mut decoder = ProresDecoder::open(&params).expect("decoder");
    let mut encoder = ProresEncoder::open(&params).expect("encoder");
    let mut coded = Vec::new();
    let mut tags = None;
    for packet in &track.packets {
        let decoded = decoder.decode(&packet.data).expect("decode");
        let frame = decoded.frame;
        tags = Some((frame.primaries, frame.transfer, frame.matrix));
        coded.push(Timed {
            data: encoder.encode(&frame).expect("encode"),
            pts: packet.pts,
            duration: packet.duration,
        });
    }
    let copy = scratch.path("copy.mov");
    std::fs::write(
        &copy,
        mux_track(
            &coded,
            track.timescale,
            &track.extra,
            track.width,
            track.height,
        ),
    )
    .expect("write");

    let before = probe(&ffprobe, &original);
    let after = probe(&ffprobe, &copy);
    eprintln!(
        "{}\n  ffmpeg's: {before:?}\n  ours:     {after:?}\n  decoded tags: {tags:?}",
        trip.name
    );
    assert_eq!(before, after, "{}: stream description", trip.name);
    assert_eq!(
        frame_times(&ffprobe, &original),
        frame_times(&ffprobe, &copy),
        "{}: packet times",
        trip.name
    );

    let a = samples16(&reference(
        &ffmpeg,
        &original,
        trip.pix_fmt,
        &scratch,
        false,
    ));
    let b = samples16(&reference(&ffmpeg, &copy, trip.pix_fmt, &scratch, false));
    let peak = if trip.pix_fmt.contains("12") {
        4095.0
    } else {
        1023.0
    };
    let diff = Diff::between(&b, &a, peak);
    eprintln!("  re-encode against ffmpeg's stream: {diff}");
    assert!(diff.psnr > trip.min_psnr, "{}: {diff}", trip.name);

    if cfg!(target_vendor = "apple")
        && let Some(avconvert) = tool("avconvert")
    {
        let out = scratch.path("avfoundation.mov");
        let ok = run(
            &avconvert,
            &[
                "--source",
                copy.to_str().expect("path"),
                "--preset",
                "PresetAppleProRes422LPCM",
                "--output",
                out.to_str().expect("path"),
                "--replace",
            ],
        );
        assert!(
            ok && out.is_file(),
            "{}: AVFoundation did not read our movie",
            trip.name
        );
        let reread = probe(&ffprobe, &out);
        eprintln!("  AVFoundation re-wrote: {reread:?}");
        assert!(
            reread.iter().any(|l| l == "nb_read_frames=8"),
            "{}: AVFoundation frames {reread:?}",
            trip.name
        );
    }
}

#[test]
fn a_bt709_hq_movie_comes_back_through_this_crate() {
    round_trip(&Trip {
        name: "1080p HQ BT.709",
        clip: HD,
        filter: "null",
        args: &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ],
        pix_fmt: "yuv422p10le",
        min_psnr: 50.0,
    });
}

#[test]
fn an_hlg_4444_movie_keeps_its_tags() {
    round_trip(&Trip {
        name: "HLG 4444 with alpha",
        clip: "IMG_9197.mov",
        filter: "scale=1280:720,format=yuva444p10le",
        args: &[
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-alpha_bits",
            "16",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "arib-std-b67",
            "-colorspace",
            "bt2020nc",
        ],
        pix_fmt: "yuva444p12le",
        min_psnr: 50.0,
    });
}
