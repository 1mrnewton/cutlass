//! Playback's and export's VP9 against FFmpeg, through WebM: a WebM of
//! FFmpeg's `libvpx-vp9`, at 8 and at 10 bits, goes through
//! `cutlass-matroska`'s demuxer and the decoder
//! [`cutlass_vp9::register_engine_decoders`] registers and gives the
//! pictures FFmpeg decodes; and a WebM of the encoder
//! [`cutlass_vp9::register_engine_encoders`] registers and
//! `cutlass-matroska`'s muxer decodes in FFmpeg to the pictures it decodes
//! to here. Skipped (passing) when `ffmpeg` is not on the `PATH` or has no
//! `libvpx-vp9`.

#![allow(clippy::expect_used, clippy::panic, clippy::print_stderr)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_codec::{
    CodecId, FrameSize, Packet as FilePacket, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_decode::{
    CodecParams, DecodeFactory, Decoder, Output, Packet, PictureOutput, PictureSamples,
    PixelFormat, SideData,
};
use cutlass_encode::{CpuFrame, EncodeFactory, FrameInput, FrameTime};
use cutlass_matroska::{MatroskaDemux, MatroskaMux};

const W: u32 = 160;
const H: u32 = 96;
const FRAMES: usize = 30;

fn ffmpeg_with_libvpx() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-h", "encoder=libvpx-vp9"])
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("yuv420p10le"))
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
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("vp9_webm_engine");
    std::fs::create_dir_all(&dir).expect("dir");
    dir
}

/// FFmpeg's decode of `file`, as raw `pix_fmt` pictures.
fn ffmpeg_decode(file: &Path, pix_fmt: &str) -> Vec<u8> {
    run(&[
        "-i",
        file.to_str().expect("path"),
        "-pix_fmt",
        pix_fmt,
        "-f",
        "rawvideo",
        "-",
    ])
}

fn decoder() -> Box<dyn Decoder> {
    let mut factory = DecodeFactory::new();
    cutlass_vp9::register_engine_decoders(&mut factory);
    factory
        .open(&CodecParams::video(CodecId::Vp9, W, H))
        .expect("decoder")
}

/// The packets of the one track of the WebM at `file`.
fn demuxed(file: &Path) -> Vec<Packet> {
    let bytes = std::fs::read(file).expect("read");
    let length = bytes.len() as u64;
    let mut demux = MatroskaDemux::open(std::io::Cursor::new(bytes), length).expect("open");
    let mut packets = Vec::new();
    while let Some(p) = demux.read_packet().expect("packet") {
        let time = p.timestamp();
        packets.push(Packet::video(
            p.bytes().to_vec(),
            time.presentation().unwrap_or(0),
            time.duration().unwrap_or(0),
            p.timescale().ticks_per_second(),
            p.keyframe(),
        ));
    }
    packets
}

/// Every picture `decoder` gives for `packets`, drained.
fn decoded(decoder: &mut dyn Decoder, packets: &[Packet]) -> Vec<PictureOutput> {
    let mut outputs = Vec::new();
    for packet in packets {
        outputs.extend(decoder.push(packet).expect("decode"));
    }
    outputs.extend(decoder.drain().expect("drain"));
    outputs
        .into_iter()
        .map(|output| match output {
            Output::Picture(picture) => picture,
            Output::Audio(_) => panic!("audio"),
        })
        .collect()
}

/// The pictures' bytes, one after another, each of `pixel` at the test's
/// size.
fn bytes_of(pictures: &[PictureOutput], pixel: PixelFormat) -> Vec<u8> {
    let mut out = Vec::new();
    for picture in pictures {
        let info = picture.info();
        assert_eq!((info.width, info.height, info.pixel), (W, H, pixel));
        let PictureSamples::Cpu(bytes) = picture.samples() else {
            panic!("platform buffer");
        };
        out.extend_from_slice(bytes);
    }
    out
}

/// Asserts two runs of raw pictures of `picture` bytes each are the same,
/// naming the first picture that is not.
fn assert_same(ours: &[u8], theirs: &[u8], picture: usize, what: &str) {
    assert_eq!(ours.len(), theirs.len(), "{what}: length");
    for (n, (a, b)) in ours.chunks(picture).zip(theirs.chunks(picture)).enumerate() {
        assert!(a == b, "{what}: picture {n}");
    }
}

#[test]
fn a_libvpx_webm_decodes_through_the_demuxer_and_the_adapter_as_in_ffmpeg() {
    if !ffmpeg_with_libvpx() {
        eprintln!("skipped: no ffmpeg with libvpx-vp9");
        return;
    }
    let luma = (W * H) as usize;
    for (pix_fmt, pixel, picture) in [
        ("yuv420p", PixelFormat::Yuv420p, luma * 3 / 2),
        ("yuv420p10le", PixelFormat::Yuv420p10, luma * 3),
        ("yuv444p", PixelFormat::Yuv444p, luma * 3),
        ("yuv422p10le", PixelFormat::Yuv422p10, luma * 4),
    ] {
        let file = dir().join(format!("libvpx-{pix_fmt}.webm"));
        // Two passes with a lookahead: libvpx codes hidden alt-ref frames,
        // which FFmpeg's muxer stores with the frame after them, as
        // superframes.
        let log = dir().join(format!("libvpx-{pix_fmt}"));
        let source = format!("testsrc2=s={W}x{H}:r=30:d=1");
        let encode = [
            "-f",
            "lavfi",
            "-i",
            &source,
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            pix_fmt,
            "-crf",
            "32",
            "-b:v",
            "0",
            "-g",
            "20",
            "-auto-alt-ref",
            "1",
            "-lag-in-frames",
            "16",
            "-passlogfile",
            log.to_str().expect("path"),
        ];
        run(&[&encode[..], &["-pass", "1", "-f", "null", "-"]].concat());
        run(&[&encode[..], &["-pass", "2", file.to_str().expect("path")]].concat());
        let theirs = ffmpeg_decode(&file, pix_fmt);
        assert_eq!(theirs.len(), FRAMES * picture, "{pix_fmt}");

        let packets = demuxed(&file);
        assert_eq!(packets.len(), FRAMES, "{pix_fmt}");
        let mut decoder = decoder();
        let pictures = decoded(decoder.as_mut(), &packets);
        assert_eq!(pictures.len(), FRAMES, "{pix_fmt}");
        assert_same(&bytes_of(&pictures, pixel), &theirs, picture, pix_fmt);
        let times: Vec<i64> = pictures.iter().map(PictureOutput::pts).collect();
        let stamped: Vec<i64> = packets.iter().map(|packet| packet.pts).collect();
        assert_eq!(
            times, stamped,
            "{pix_fmt}: each picture has its packet's time"
        );

        // A seek back to the start: after a reset the same pictures again.
        decoder.reset();
        let again = decoded(decoder.as_mut(), &packets);
        assert_same(&bytes_of(&again, pixel), &theirs, picture, pix_fmt);
    }
}

/// A moving gradient with a block crossing it, as RGBA.
fn rgba(t: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            let block = (x + 150 - 4 * t) % W < 24 && (y + 2 * t) % H < 20;
            out.extend_from_slice(&if block {
                [240, 60, 30, 255]
            } else {
                [(x + 3 * t) as u8, (2 * y + t) as u8, (x + y) as u8, 255]
            });
        }
    }
    out
}

#[test]
fn a_webm_of_the_export_encoder_decodes_in_ffmpeg_as_it_does_here() {
    if !ffmpeg_with_libvpx() {
        eprintln!("skipped: no ffmpeg with libvpx-vp9");
        return;
    }
    let mut encoders = EncodeFactory::new();
    cutlass_vp9::register_engine_encoders(&mut encoders);
    let mut params = CodecParams::with_format(CodecId::Vp9, W, H, PixelFormat::Rgba8);
    params.bitrate = 600_000;
    let mut encoder = encoders.open(&params).expect("encoder");
    let mut coded = Vec::new();
    for t in 0..FRAMES as u32 {
        let time = FrameTime::new(i64::from(t), 1, 30).expect("time");
        let frame = CpuFrame::new(W, H, PixelFormat::Rgba8, rgba(t), time).expect("frame");
        coded.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("push"));
    }
    coded.extend(encoder.drain().expect("drain"));
    assert_eq!(coded.len(), FRAMES);
    assert!(coded[0].keyframe);
    // Export hands the muxer the configuration; WebM stores none for VP9.
    assert!(
        coded
            .iter()
            .flat_map(|packet| &packet.side)
            .all(|side| matches!(side, SideData::Config(_))),
        "only the configuration rides with the packets"
    );

    let mut mux = MatroskaMux::webm(std::io::Cursor::new(Vec::new()));
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::Vp9),
                FrameSize::try_new(W, H).expect("size"),
                Vec::new(),
            ),
            timescale: Timescale::new(30),
            language: None,
        })
        .expect("track");
    for packet in &coded {
        let time = Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(1)).expect("time");
        let written = FilePacket::try_new(
            track.get(),
            packet.data.clone(),
            time,
            Timescale::new(30),
            packet.keyframe,
        )
        .expect("packet");
        mux.write_packet(&written).expect("write");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(30),
        ticks: FRAMES as i64,
    }])
    .expect("finish");
    let file = dir().join("ours.webm");
    std::fs::write(&file, mux.into_writer().expect("writer").into_inner()).expect("write");

    let picture = (W * H * 3 / 2) as usize;
    // FFmpeg's own decoder and libvpx, through FFmpeg, both read it.
    let theirs = ffmpeg_decode(&file, "yuv420p");
    assert_eq!(theirs.len(), FRAMES * picture);
    let libvpx = run(&[
        "-c:v",
        "libvpx-vp9",
        "-i",
        file.to_str().expect("path"),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        "-",
    ]);
    assert_same(&libvpx, &theirs, picture, "libvpx");

    let packets = demuxed(&file);
    assert_eq!(packets.len(), FRAMES);
    let pictures = decoded(decoder().as_mut(), &packets);
    assert_same(
        &bytes_of(&pictures, PixelFormat::Yuv420p),
        &theirs,
        picture,
        "ours.webm",
    );
    // The pictures are the source's, coded: the block is where it was put.
    let last = &theirs[(FRAMES - 1) * picture..];
    let t = FRAMES as u32 - 1;
    let (x, y) = ((4 * t + W - 150 + 12) % W, (H - (2 * t) % H + 10) % H);
    // RGB 240, 60, 30 in BT.709 studio range is Y 98.
    let luma = last[(y * W + x) as usize];
    assert!(luma.abs_diff(98) <= 12, "block luma {luma} at {x},{y}");
}
