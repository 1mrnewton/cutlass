//! VP9 with alpha against FFmpeg, both ways: this crate's encoder and
//! `cutlass-matroska`'s muxer write a WebM that FFmpeg's `libvpx-vp9`
//! decoder decodes to the pictures and alpha this crate's decoder gives,
//! and a WebM of FFmpeg's `libvpx-vp9` with `yuva420p` decodes here to the
//! YUVA FFmpeg's `libvpx-vp9` decoder gives. Skipped (passing) when
//! `ffmpeg` is not on the `PATH` or has no `libvpx-vp9`.

#![allow(clippy::expect_used, clippy::panic, clippy::print_stderr)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_codec::{
    BlockAddition, CodecId, FrameSize, Packet as FilePacket, StreamCodec, StreamParams, Timescale,
    Timestamp,
};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecParams, Packet, PixelFormat, SideData};
use cutlass_matroska::{MatroskaDemux, MatroskaMux};
use cutlass_vp9::{Vp9Decoder, Vp9Encoder, Vp9Options};

const W: usize = 160;
const H: usize = 96;
const FRAMES: usize = 20;
/// Bytes of a `yuva420p` picture.
const PICTURE: usize = W * H * 2 + 2 * (W / 2) * (H / 2);

fn ffmpeg_with_libvpx() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-h", "encoder=libvpx-vp9"])
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("yuva420p"))
}

fn run(args: &[&str]) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .expect("ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("vp9_alpha");
    std::fs::create_dir_all(&dir).expect("dir");
    dir
}

/// A `yuva420p` source: FFmpeg's test pattern with a moving alpha
/// gradient.
fn source() -> Vec<u8> {
    let size = format!("{W}x{H}");
    let raw = run(&[
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=s={size}:r=30:d=1"),
        "-f",
        "lavfi",
        "-i",
        &format!(
            "nullsrc=s={size}:r=30:d=1,format=gray,\
             geq=lum='128+127*sin((X+6*N)/20)*cos((Y-3*N)/15)'"
        ),
        "-filter_complex",
        "[0][1]alphamerge,format=yuva420p",
        "-frames:v",
        &FRAMES.to_string(),
        "-f",
        "rawvideo",
        "-",
    ]);
    assert_eq!(raw.len(), FRAMES * PICTURE);
    raw
}

/// FFmpeg's `libvpx-vp9` decode of `file` as `yuva420p`.
fn ffmpeg_decode(file: &Path) -> Vec<u8> {
    run(&[
        "-c:v",
        "libvpx-vp9",
        "-i",
        file.to_str().expect("path"),
        "-pix_fmt",
        "yuva420p",
        "-f",
        "rawvideo",
        "-",
    ])
}

/// This crate's decode of the VP9 track of `file`, through
/// `cutlass-matroska`, as `yuva420p`.
fn our_decode(file: &Path, threads: usize) -> Vec<u8> {
    let bytes = std::fs::read(file).expect("read");
    let length = bytes.len() as u64;
    let mut demux = MatroskaDemux::open(std::io::Cursor::new(bytes), length).expect("open");
    assert_eq!(demux.track_entries()[0].alpha_mode, 1);
    let params = CodecParams::with_format(CodecId::Vp9, W as u32, H as u32, PixelFormat::Yuv420p);
    let mut decoder = Vp9Decoder::open_with_threads(&params, threads).expect("decoder");
    let mut out = Vec::new();
    let mut shown = Vec::new();
    while let Some(p) = demux.read_packet().expect("packet") {
        let time = p.timestamp();
        let mut packet = Packet::video(
            p.bytes().to_vec(),
            time.presentation().unwrap_or(0),
            time.duration().unwrap_or(0),
            p.timescale().ticks_per_second(),
            p.keyframe(),
        );
        packet
            .side
            .extend(p.additions().iter().map(|a| SideData::BlockAdditional {
                id: a.id,
                data: a.data.clone(),
            }));
        shown.extend(decoder.push(&packet).expect("decode"));
    }
    shown.extend(decoder.drain().expect("drain"));
    for o in shown {
        let f = &o.frame;
        for plane in [&f.y, &f.u, &f.v] {
            out.extend(plane.packed(1));
        }
        out.extend(f.alpha.as_ref().expect("alpha").plane.packed(1));
    }
    out
}

/// Each plane of each picture of two `yuva420p` files: equal.
fn assert_same(ours: &[u8], theirs: &[u8], what: &str) {
    assert_eq!(ours.len(), theirs.len(), "{what}: length");
    let planes = [
        ("Y", 0, W * H),
        ("U", W * H, W * H + W * H / 4),
        ("V", W * H + W * H / 4, W * H + W * H / 2),
        ("A", W * H + W * H / 2, PICTURE),
    ];
    for (n, (a, b)) in ours
        .as_chunks::<PICTURE>()
        .0
        .iter()
        .zip(theirs.as_chunks::<PICTURE>().0.iter())
        .enumerate()
    {
        for (name, from, to) in planes {
            assert!(a[from..to] == b[from..to], "{what}: picture {n} {name}");
        }
    }
}

#[test]
fn our_webm_with_alpha_decodes_in_ffmpeg_as_it_does_here() {
    if !ffmpeg_with_libvpx() {
        eprintln!("skipped: no ffmpeg with libvpx-vp9");
        return;
    }
    let src = source();
    let opts = Vp9Options {
        qindex: Vp9Encoder::qindex_of_level(30),
        keyint: 8,
        alpha: true,
        threads: 2,
        ..Vp9Options::default()
    };
    let mut encoder = Vp9Encoder::with_options(W as u32, H as u32, opts).expect("encoder");
    let mut coded = Vec::new();
    for (n, picture) in src.as_chunks::<PICTURE>().0.iter().enumerate() {
        let (y, rest) = picture.split_at(W * H);
        let (u, rest) = rest.split_at(W * H / 4);
        let (v, a) = rest.split_at(W * H / 4);
        let planes = [(y, W), (u, W / 2), (v, W / 2), (a, W)];
        coded.extend(
            encoder
                .push_planes_alpha(planes, 1.0 / 30.0, (n as i64, 1, 30))
                .expect("push"),
        );
    }
    coded.extend(encoder.flush_alpha().expect("flush"));
    assert_eq!(coded.len(), FRAMES);

    let mut mux = MatroskaMux::webm(std::io::Cursor::new(Vec::new()));
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::Vp9),
                FrameSize::try_new(W as u32, H as u32).expect("size"),
                Vec::new(),
            ),
            timescale: Timescale::new(30),
            language: None,
        })
        .expect("track");
    for c in &coded {
        let time = Timestamp::try_new(Some(c.clock.0), Some(c.clock.0), Some(1)).expect("time");
        let packet = FilePacket::try_new(
            track.get(),
            c.data.clone(),
            time,
            Timescale::new(30),
            c.keyframe,
        )
        .expect("packet")
        .with_additions(vec![BlockAddition {
            id: 1,
            data: c.alpha.clone(),
        }]);
        mux.write_packet(&packet).expect("write");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(30),
        ticks: FRAMES as i64,
    }])
    .expect("finish");
    let file = dir().join("ours.webm");
    std::fs::write(&file, mux.into_writer().expect("writer").into_inner()).expect("write");

    let theirs = ffmpeg_decode(&file);
    let ours = our_decode(&file, 1);
    assert_same(&ours, &theirs, "ours.webm");
    // The alpha is the source's, coded.
    let alpha_err: f64 = ours
        .as_chunks::<PICTURE>()
        .0
        .iter()
        .zip(src.as_chunks::<PICTURE>().0.iter())
        .flat_map(|(a, b)| a[PICTURE - W * H..].iter().zip(&b[PICTURE - W * H..]))
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / (FRAMES * W * H) as f64;
    let psnr = 10.0 * (255.0 * 255.0 / alpha_err.max(1e-9)).log10();
    assert!(psnr > 35.0, "alpha PSNR {psnr}");
}

#[test]
fn ffmpeg_webm_with_alpha_decodes_here_as_in_ffmpeg() {
    if !ffmpeg_with_libvpx() {
        eprintln!("skipped: no ffmpeg with libvpx-vp9");
        return;
    }
    let src = dir().join("src.yuva");
    std::fs::write(&src, source()).expect("source");
    let file = dir().join("ffmpeg.webm");
    let size = format!("{W}x{H}");
    run(&[
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuva420p",
        "-s",
        &size,
        "-r",
        "30",
        "-i",
        src.to_str().expect("path"),
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-crf",
        "30",
        "-b:v",
        "0",
        "-g",
        "8",
        file.to_str().expect("path"),
    ]);
    let theirs = ffmpeg_decode(&file);
    assert_eq!(theirs.len(), FRAMES * PICTURE);
    for threads in [1, 3] {
        assert_same(&our_decode(&file, threads), &theirs, "ffmpeg.webm");
    }
}
