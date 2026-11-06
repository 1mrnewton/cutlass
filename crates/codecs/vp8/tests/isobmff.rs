//! Our VP8 packets in an MP4 `vp08` track: muxed with `cutlass-isobmff`,
//! read back, and decoded here and by ffmpeg to the same pictures. The
//! export path (RGBA through the engine factory into `Mp4Mux`) is checked
//! the same way.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This crate is the test harness, so the same exception is named here.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecId, CodecParams, Packet as CodecPacket, SideData};
use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{Plane, VideoFrame};
use cutlass_isobmff::{
    CodecId as BoxCodec, Demux, IsoBmff, IsoBmffMux, MediaKind, MuxOptions, Packet, SliceSource,
    StreamParams, Timescale,
};
use cutlass_vp8::{Vp8Decoder, Vp8Encoder, Vp8Options};

const W: u32 = 96;
const H: u32 = 64;
const FRAMES: u32 = 12;

fn picture(t: u32) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let y: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, r) = ((i % w) as f32, (i / w) as f32);
            (128.0 + 60.0 * ((x + 3.0 * t as f32) * 0.2).sin() + 40.0 * (r * 0.13).cos()) as u8
        })
        .collect();
    let u = vec![110u8; w * h / 4];
    let v: Vec<u8> = (0..w * h / 4)
        .map(|i| (100 + (i + t as usize) % 60) as u8)
        .collect();
    VideoFrame::yuv420p(
        W,
        H,
        Plane::tight(W, H, y).expect("y"),
        Plane::tight(W / 2, H / 2, u).expect("u"),
        Plane::tight(W / 2, H / 2, v).expect("v"),
    )
    .expect("frame")
}

fn decode(packets: &[(Vec<u8>, bool)]) -> Vec<u8> {
    let mut decoder = Vp8Decoder::open(&CodecParams::video(CodecId::Vp8, W, H)).expect("open");
    let mut out = Vec::new();
    for (i, (data, key)) in packets.iter().enumerate() {
        for picture in decoder
            .push(&CodecPacket::video(data.clone(), i as i64, 1, 30, *key))
            .expect("decode")
        {
            out.extend_from_slice(&picture.frame.y.samples);
            out.extend_from_slice(&picture.frame.u.samples);
            out.extend_from_slice(&picture.frame.v.samples);
        }
    }
    out
}

fn ffmpeg_bin() -> Option<PathBuf> {
    [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| {
        Command::new(path)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

#[test]
fn an_mp4_vp08_track_round_trips() {
    let mut encoder = Vp8Encoder::with_options(
        W,
        H,
        Vp8Options {
            keyframe_interval: 5,
            ..Vp8Options::default()
        },
    )
    .expect("encoder");
    let mut packets = Vec::new();
    let mut config = None;
    for t in 0..FRAMES {
        let input = VideoInput {
            frame: picture(t),
            pts: i64::from(t),
            duration: 1,
            timescale: 30,
        };
        for packet in encoder.push(&input).expect("encode") {
            for side in &packet.side {
                if let SideData::Config(bytes) = side {
                    assert!(config.is_none(), "config on one packet only");
                    config = Some(bytes.clone());
                }
            }
            packets.push((packet.data, packet.keyframe));
        }
    }
    let config = config.expect("the first packet carries vpcC");
    assert_eq!(config, encoder.vpcc_record());

    let mut params = StreamParams::video(BoxCodec::Vp8, Timescale(30), W, H);
    params.extradata = config.clone();
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let id = cutlass_isobmff::Mux::add_stream(&mut mux, &params).expect("track");
    for (i, (data, key)) in packets.iter().enumerate() {
        let packet = Packet {
            track_id: id,
            payload: data.clone(),
            dts: i as i64,
            pts: i as i64,
            duration: 1,
            timescale: Timescale(30),
            keyframe: *key,
        };
        cutlass_isobmff::Mux::write_packet(&mut mux, &packet).expect("write");
    }
    let file = mux.finish_into().expect("finish").into_inner();

    // Read back: the same track, record, payloads, and key flags.
    let mut source = SliceSource::new(&file);
    let mut movie = IsoBmff::open(&mut source).expect("open mp4");
    let stream = &movie.streams()[0];
    assert_eq!(stream.params.codec, BoxCodec::Vp8);
    assert_eq!(stream.params.extradata, config);
    assert!(matches!(stream.params.kind, MediaKind::Video(v) if v.width == W && v.height == H));
    let mut read = Vec::new();
    while let Some(packet) = movie.read_packet(&mut source).expect("read") {
        read.push((packet.payload, packet.keyframe));
    }
    assert_eq!(read, packets);
    let ours = decode(&read);
    assert_eq!(ours.len(), (W * H * 3 / 2 * FRAMES) as usize);

    // ffmpeg reads the MP4 and decodes it to the same pictures.
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("ffmpeg missing; the MP4 was checked with our demuxer only");
        return;
    };
    let dir = std::env::temp_dir().join(format!("cutlass-vp8-mp4-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let mp4 = dir.join("ours.mp4");
    std::fs::write(&mp4, &file).expect("mp4");
    for decoder in ["vp8", "libvpx"] {
        let yuv = dir.join(format!("{decoder}.yuv"));
        let status = Command::new(&ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-c:v",
                decoder,
                "-i",
            ])
            .arg(&mp4)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&yuv)
            .output()
            .expect("ffmpeg");
        assert!(
            status.status.success(),
            "{}",
            String::from_utf8_lossy(&status.stderr)
        );
        let theirs = std::fs::read(&yuv).expect("yuv");
        assert!(theirs == ours, "{decoder} decodes the MP4 differently");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// What export does: RGBA frames through the engine's encode factory, the
/// packets and the `vpcC` into `Mp4Mux`. ffmpeg's `vp8` and libvpx decode
/// the file to the pictures our engine decoder gives, and ffprobe reads
/// BT.709 from the `vpcC`.
#[test]
fn the_export_path_writes_an_mp4_other_players_read() {
    use cutlass_codec::{
        CodecId as Id, FrameSize, Packet as MuxPacket, StreamCodec, StreamParams as Params,
        Timescale as Scale, Timestamp,
    };
    use cutlass_container::{Mux as _, NewStream, TrackDuration};
    use cutlass_decode::{DecodeFactory, Output, PictureSamples, PixelFormat};
    use cutlass_encode::{CpuFrame, EncodeFactory, FrameInput, FrameTime};

    let mut encoders = EncodeFactory::new();
    cutlass_vp8::register_engine_encoders(&mut encoders);
    let mut params = CodecParams::with_format(CodecId::Vp8, W, H, PixelFormat::Rgba8);
    params.bitrate = 600_000;
    let mut encoder = encoders.open(&params).expect("encoder");
    let mut packets = Vec::new();
    for t in 0..FRAMES {
        let rgba: Vec<u8> = (0..(W * H) as usize)
            .flat_map(|i| {
                let (x, y) = (i as u32 % W, i as u32 / W);
                [
                    ((x * 5 + t * 7) % 256) as u8,
                    ((y * 3) % 256) as u8,
                    ((x + y + t) % 256) as u8,
                    255,
                ]
            })
            .collect();
        let time = FrameTime::new(i64::from(t), 1, 30).expect("time");
        let frame = CpuFrame::new(W, H, PixelFormat::Rgba8, rgba, time).expect("frame");
        packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("push"));
    }
    packets.extend(encoder.drain().expect("drain"));
    let config = packets[0]
        .side
        .iter()
        .find_map(|side| match side {
            SideData::Config(bytes) => Some(bytes.clone()),
            _ => None,
        })
        .expect("vpcC on the first packet");

    let mut mux = cutlass_isobmff::Mp4Mux::new(Cursor::new(Vec::new()));
    let track = mux
        .add_stream(NewStream {
            params: Params::video(
                StreamCodec::Known(Id::Vp8),
                FrameSize::try_new(W, H).expect("size"),
                Vec::new(),
            ),
            timescale: Scale::new(30),
            language: None,
        })
        .expect("track");
    mux.set_config(track, &config).expect("config");
    for packet in &packets {
        let time = Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
            .expect("time");
        let out = MuxPacket::try_new(
            track.get(),
            packet.data.clone(),
            time,
            Scale::new(30),
            packet.keyframe,
        )
        .expect("packet");
        mux.write_packet(&out).expect("write");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Scale::new(30),
        ticks: i64::from(FRAMES),
    }])
    .expect("finish");
    let file = mux.into_writer().expect("writer").into_inner();

    // Our engine decoder, opened with the file's configuration.
    let mut decoders = DecodeFactory::new();
    cutlass_vp8::register_engine_decoders(&mut decoders);
    let mut read = CodecParams::video(CodecId::Vp8, W, H);
    read.extra = config.clone();
    let mut decoder = decoders.open(&read).expect("decoder");
    let mut ours = Vec::new();
    for packet in &packets {
        for output in cutlass_decode::Decoder::push(decoder.as_mut(), packet).expect("decode") {
            let Output::Picture(picture) = output else {
                panic!("audio");
            };
            let PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("platform");
            };
            ours.extend_from_slice(bytes);
        }
    }
    assert_eq!(ours.len(), (W * H * 3 / 2 * FRAMES) as usize);

    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("ffmpeg missing; the export MP4 was checked with our decoder only");
        return;
    };
    let dir = std::env::temp_dir().join(format!("cutlass-vp8-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let mp4 = dir.join("export.mp4");
    std::fs::write(&mp4, &file).expect("mp4");
    for decoder in ["vp8", "libvpx"] {
        let yuv = dir.join(format!("{decoder}.yuv"));
        let status = Command::new(&ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-c:v",
                decoder,
                "-i",
            ])
            .arg(&mp4)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&yuv)
            .output()
            .expect("ffmpeg");
        assert!(
            status.status.success(),
            "{}",
            String::from_utf8_lossy(&status.stderr)
        );
        let theirs = std::fs::read(&yuv).expect("yuv");
        assert!(
            theirs == ours,
            "{decoder} decodes the export MP4 differently"
        );
    }
    let ffprobe = ffmpeg.with_file_name("ffprobe");
    if let Ok(probe) = Command::new(&ffprobe)
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,color_primaries,color_space,color_transfer",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&mp4)
        .output()
        && probe.status.success()
    {
        let text = String::from_utf8_lossy(&probe.stdout);
        for line in [
            "codec_name=vp8",
            "color_primaries=bt709",
            "color_space=bt709",
            "color_transfer=bt709",
        ] {
            assert!(text.contains(line), "ffprobe: {text}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
