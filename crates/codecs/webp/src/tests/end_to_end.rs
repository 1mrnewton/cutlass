//! End to end through the still container, as import and export use it:
//! a picture from `local-files`, written by `cwebp`, opened by the still
//! demuxer, decoded here, encoded again, muxed by the still muxer, and
//! reopened by `dwebp`, `webpinfo` and ImageIO. Frames of a local video take
//! the same road as an `img2webp` animation, reopened by Pillow's libwebp.
//! Without `local-files` the synthetic picture stands in for the picture,
//! and the animation is skipped.

use std::io::Cursor;
use std::path::Path;

use cutlass_codec::{
    CodecId, FrameSize, Packet as ContainerPacket, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{Demux, Input, Mux, NewStream, TrackDuration, TrackId};
use cutlass_decode::{Decoder, Packet};
use cutlass_encode::{CpuFrame, Encoder, FrameInput, FrameTime, PictureInput, PixelFormat};
use cutlass_still::{StillDemux, StillMux, WEBP_TIMESCALE, still_limits};

use super::animation::check_animations;
use super::lossless_encode::webpinfo_ok;
use super::lossy_encode::psnr;
use super::oracle::{
    cwebp, dwebp_rgba, local_files, require_or_skip, scratch, sips_rgba, tool, video_frame,
};
use super::stills::lossless_sources;
use crate::{WebpDecoder, WebpEncoder, WebpOptions};

fn demux(bytes: Vec<u8>) -> StillDemux<Cursor<Vec<u8>>> {
    let length = bytes.len() as u64;
    StillDemux::open(Input::random(Cursor::new(bytes), length, still_limits())).expect("open")
}

fn packets(demux: &mut impl Demux) -> Vec<ContainerPacket> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push(packet);
    }
    packets
}

/// The container packet as the decoder takes it.
fn access(packet: &ContainerPacket) -> Packet {
    let time = packet.timestamp();
    Packet::video(
        packet.bytes().to_vec(),
        time.presentation_or_decode().unwrap_or(0),
        time.duration().unwrap_or(0),
        packet.timescale().ticks_per_second(),
        true,
    )
}

/// `packets` muxed into one file by the still muxer.
fn mux(packets: &[ContainerPacket], timescale: u32, ticks: i64, looped: bool) -> Vec<u8> {
    let mut mux = StillMux::new();
    mux.add_stream(NewStream {
        params: StreamParams::video(
            StreamCodec::Known(CodecId::Webp),
            FrameSize::try_new(1, 1).expect("size"),
            Vec::new(),
        ),
        timescale: Timescale::new(timescale),
        language: None,
    })
    .expect("stream");
    if looped {
        mux.set_loop_count(0);
    }
    for packet in packets {
        mux.write_packet(packet).expect("write");
    }
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(timescale),
        ticks,
    }])
    .expect("finish");
    mux.bytes().expect("bytes").to_vec()
}

fn encoder(options: WebpOptions) -> WebpEncoder {
    let mut params = cutlass_decode::CodecParams::video(CodecId::Webp, 0, 0);
    params.extra = options.to_extra();
    WebpEncoder::open(&params).expect("encoder")
}

#[test]
fn a_cwebp_picture_goes_through_the_container_and_back_out_to_libwebp_and_imageio() {
    let (Some(cwebp_tool), Some(dwebp)) = (tool("cwebp"), tool("dwebp")) else {
        require_or_skip("cwebp and dwebp");
        return;
    };
    let webpinfo = tool("webpinfo");
    let sips = tool("sips");
    let dir = scratch("end-to-end");
    // The local picture when there is one, else the synthetic one.
    let sources = lossless_sources(&dir);
    let source = sources.last().expect("a source");
    for (label, args, options) in [
        ("lossy", &["-q", "80"][..], WebpOptions::default()),
        ("lossless", &["-lossless"][..], WebpOptions::lossless()),
    ] {
        let first = dir.join(format!("cwebp-{label}.webp"));
        let written = cwebp(&cwebp_tool, source, args, &first);

        // Import: the container hands over the file as one packet.
        let mut opened = demux(written.clone());
        let read = packets(&mut opened);
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].bytes(), written.as_slice());
        let mut decoder =
            WebpDecoder::open(&cutlass_decode::CodecParams::video(CodecId::Webp, 0, 0))
                .expect("decoder");
        let pictures = decoder.decode_picture(&access(&read[0])).expect("decode");
        assert_eq!(pictures.len(), 1);
        let image = pictures[0].image.clone();
        let (_, _, theirs) = dwebp_rgba(&dwebp, &first);
        let channels = image.samples.len() / (image.width * image.height) as usize;
        let ours: Vec<u8> = image
            .samples
            .chunks(channels)
            .flat_map(|p| {
                [
                    p[0] as u8,
                    p[1] as u8,
                    p[2] as u8,
                    p.get(3).map_or(255, |&a| a as u8),
                ]
            })
            .collect();
        assert!(ours == theirs, "{label}: import differs from dwebp");

        // Export: encode the picture again and mux it.
        let (width, height) = (image.width, image.height);
        let coded = encoder(options)
            .push_picture(&PictureInput { image })
            .expect("encode");
        assert_eq!(coded.len(), 1);
        let packet = ContainerPacket::try_new(
            0,
            coded[0].data.clone(),
            Timestamp::missing(),
            Timescale::new(0),
            true,
        )
        .expect("packet");
        let file = mux(&[packet], 0, 0, false);
        assert_eq!(file, coded[0].data, "a still is its one packet");
        let path = dir.join(format!("again-{label}.webp"));
        std::fs::write(&path, &file).expect("write");

        // Reopen with libwebp, webpinfo and ImageIO.
        let (w, h, decoded) = dwebp_rgba(&dwebp, &path);
        assert_eq!((w, h), (width, height));
        let again = super::oracle::decode_still(&file).rgba;
        assert!(again == decoded, "{label}: dwebp and this crate differ");
        if label == "lossless" {
            assert!(decoded == ours, "lossless export changed the picture");
        } else {
            let quality = psnr(&ours, &decoded);
            assert!(quality > 30.0, "{label}: {quality:.2} dB");
        }
        if let Some(webpinfo) = &webpinfo {
            assert!(webpinfo_ok(webpinfo, &path), "{label}: webpinfo");
        }
        if let Some(sips) = &sips {
            let (w, h, imageio) = sips_rgba(sips, &path);
            assert_eq!((w, h), (width, height));
            assert!(psnr(&decoded, &imageio) > 30.0, "{label}: ImageIO");
        }
    }
}

#[test]
fn an_animation_from_a_local_video_goes_through_the_container_and_back_out() {
    let (Some(local), Some(ffmpeg), Some(img2webp)) =
        (local_files(), tool("ffmpeg"), tool("img2webp"))
    else {
        require_or_skip("local-files, ffmpeg and img2webp");
        return;
    };
    let Some(video) = std::fs::read_dir(&local).ok().and_then(|entries| {
        entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "mp4"))
            .min()
    }) else {
        require_or_skip("a video in local-files");
        return;
    };
    let dir = scratch("end-to-end-animation");
    let frames = frames_of(&ffmpeg, &video, &dir, 6);
    if frames.len() < 2 {
        require_or_skip("frames of the local video");
        return;
    }
    let first = dir.join("img2webp.webp");
    let mut command = std::process::Command::new(&img2webp);
    command.args(["-loop", "0", "-lossy", "-q", "70", "-d", "100"]);
    command.args(&frames);
    command.arg("-o").arg(&first);
    assert!(command.status().is_ok_and(|status| status.success()));

    // Import: one packet per frame, each decoded to the whole canvas.
    let mut opened = demux(std::fs::read(&first).expect("read"));
    let read = packets(&mut opened);
    assert_eq!(read.len(), frames.len());
    let mut decoder = WebpDecoder::open(&cutlass_decode::CodecParams::video(CodecId::Webp, 0, 0))
        .expect("decoder");
    let mut canvases = Vec::new();
    for packet in &read {
        let pictures = decoder.decode_picture(&access(packet)).expect("decode");
        assert_eq!(pictures.len(), 1);
        canvases.push(pictures[0].clone());
    }

    // Export: the canvases as frames, timed in milliseconds.
    let mut encoder = encoder(WebpOptions::default());
    let (width, height) = (canvases[0].image.width, canvases[0].image.height);
    let mut end = 0;
    for canvas in &canvases {
        let channels = canvas.image.samples.len() / (width * height) as usize;
        let rgba: Vec<u8> = canvas
            .image
            .samples
            .chunks(channels)
            .flat_map(|p| {
                [
                    p[0] as u8,
                    p[1] as u8,
                    p[2] as u8,
                    p.get(3).map_or(255, |&a| a as u8),
                ]
            })
            .collect();
        let time = FrameTime::new(canvas.pts, canvas.duration, WEBP_TIMESCALE).expect("time");
        end = canvas.pts + canvas.duration;
        let frame = CpuFrame::new(width, height, PixelFormat::Rgba8, rgba, time).expect("frame");
        assert!(
            encoder
                .push_frame(&FrameInput::Cpu(frame))
                .expect("push")
                .is_empty()
        );
    }
    let coded = encoder.drain().expect("drain");
    assert_eq!(coded.len(), canvases.len());
    let muxed: Vec<ContainerPacket> = coded
        .iter()
        .map(|packet| {
            ContainerPacket::try_new(
                0,
                packet.data.clone(),
                Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                    .expect("time"),
                Timescale::new(packet.timescale),
                true,
            )
            .expect("packet")
        })
        .collect();
    let file = mux(&muxed, WEBP_TIMESCALE, end, true);
    let path = dir.join("again.webp");
    std::fs::write(&path, &file).expect("write");

    // Reopen: the container reads every frame and the loop; webpinfo
    // accepts it; libwebp through Pillow composes what this crate does.
    let again = demux(file);
    assert_eq!(again.frame_count(), canvases.len());
    assert_eq!(again.loop_count(), Some(0));
    if let Some(webpinfo) = tool("webpinfo") {
        assert!(webpinfo_ok(&webpinfo, &path));
    }
    check_animations(&dir, &[path]);
}

/// `count` frames of `video`, 160 pixels wide, as PNGs.
fn frames_of(ffmpeg: &Path, video: &Path, dir: &Path, count: usize) -> Vec<std::path::PathBuf> {
    let mut frames = Vec::new();
    for index in 0..count {
        let out = dir.join(format!("frame{index}.png"));
        let filter = format!("select=eq(n\\,{}),scale=160:-2", index * 5);
        if video_frame(ffmpeg, video, &out, &filter) {
            frames.push(out);
        }
    }
    frames
}
