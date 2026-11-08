//! Our MPEG-2 packets in an MP4, as export writes them: RGBA through the
//! engine's encode factory, the packets and the sequence headers into
//! `Mp4Mux`. The file reads back with the same packets and times, our
//! engine decoder plays it in display order, and FFmpeg decodes it to the
//! same pictures within the bound two inverse DCTs allow. Without FFmpeg
//! that last leg says so and skips, or fails when `MPEG2_REQUIRE_ORACLES` is
//! set.

#[cfg(test)]
mod oracles;

#[cfg(test)]
mod isobmff {
    use std::io::Cursor;
    use std::process::Command;

    use cutlass_codec::{
        CodecId as Id, FrameSize, Packet as MuxPacket, StreamCodec, StreamParams, Timescale,
        Timestamp,
    };
    use cutlass_container::{Demux as _, Mux as _, NewStream, TrackDuration};
    use cutlass_decode::{
        CodecId, CodecParams, DecodeFactory, Output, Packet, PictureSamples, PixelFormat, SideData,
    };
    use cutlass_encode::{CpuFrame, EncodeFactory, FrameInput, FrameTime};

    const W: u32 = 176;
    const H: u32 = 144;
    const FRAMES: u32 = 27;
    const RATE: u32 = 25;

    fn rgba(t: u32) -> Vec<u8> {
        (0..W * H)
            .flat_map(|i| {
                let (x, y) = ((i % W) as f32, (i / W) as f32);
                let shift = x - 2.0 * t as f32;
                let wave = 110.0 + 70.0 * (shift * 0.12).sin() * (y * 0.09).cos();
                [
                    wave as u8,
                    (60.0 + y * 0.8) as u8,
                    (200.0 - wave * 0.5) as u8,
                    255,
                ]
            })
            .collect()
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let error: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        if error == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / error).log10()
        }
    }

    #[test]
    fn the_export_path_writes_an_mp4_that_plays_back_and_ffmpeg_reads() {
        let mut encoders = EncodeFactory::new();
        cutlass_mpeg2::register_encoders(&mut encoders);
        let mut params = CodecParams::with_format(CodecId::Mpeg2, W, H, PixelFormat::Rgba8);
        params.bitrate = 600_000;
        let mut encoder = encoders.open(&params).unwrap();
        let mut packets = Vec::new();
        for t in 0..FRAMES {
            let time = FrameTime::new(i64::from(t), 1, RATE).unwrap();
            let frame = CpuFrame::new(W, H, PixelFormat::Rgba8, rgba(t), time).unwrap();
            packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).unwrap());
        }
        packets.extend(encoder.drain().unwrap());
        assert_eq!(packets.len(), FRAMES as usize);
        let config = packets[0]
            .side
            .iter()
            .find_map(|side| match side {
                SideData::Config(bytes) => Some(bytes.clone()),
                _ => None,
            })
            .unwrap();

        let mut mux = cutlass_isobmff::Mp4Mux::new(Cursor::new(Vec::new()));
        let track = mux
            .add_stream(NewStream {
                params: StreamParams::video(
                    StreamCodec::Known(Id::Mpeg2),
                    FrameSize::try_new(W, H).unwrap(),
                    Vec::new(),
                ),
                timescale: Timescale::new(RATE),
                language: None,
            })
            .unwrap();
        mux.set_config(track, &config).unwrap();
        for packet in &packets {
            let time =
                Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                    .unwrap();
            let out = MuxPacket::try_new(
                track.get(),
                packet.data.clone(),
                time,
                Timescale::new(RATE),
                packet.keyframe,
            )
            .unwrap();
            mux.write_packet(&out).unwrap();
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(RATE),
            ticks: i64::from(FRAMES),
        }])
        .unwrap();
        let file = mux.into_writer().unwrap().into_inner();

        // The file gives back the track, its headers, and every packet in
        // decode order with its presentation time.
        let mut movie = cutlass_isobmff::IsoBmffFile::open(Cursor::new(file.clone())).unwrap();
        let stream = movie.streams()[0].clone();
        assert_eq!(stream.params().codec(), &StreamCodec::Known(Id::Mpeg2));
        assert_eq!(stream.params().extradata(), config.as_slice());
        let mut read = Vec::new();
        while let Some(packet) = movie.read_packet().unwrap() {
            let time = packet.timestamp();
            read.push(Packet::video(
                packet.bytes().to_vec(),
                time.presentation_or_decode().unwrap(),
                time.duration().unwrap(),
                RATE,
                packet.keyframe(),
            ));
        }
        assert_eq!(read.len(), packets.len());
        for (back, sent) in read.iter().zip(&packets) {
            assert_eq!(back.data, sent.data);
            assert_eq!(back.pts, sent.pts);
            assert_eq!(back.keyframe, sent.keyframe);
        }

        // Playback opens the decoder with the file's headers and gets every
        // picture once, in display order.
        let mut decoders = DecodeFactory::new();
        cutlass_mpeg2::register_decoders(&mut decoders);
        let mut open = CodecParams::video(CodecId::Mpeg2, W, H);
        open.extra = stream.params().extradata().to_vec();
        let mut decoder = decoders.open(&open).unwrap();
        let mut outputs = Vec::new();
        for packet in &read {
            outputs.extend(cutlass_decode::Decoder::push(decoder.as_mut(), packet).unwrap());
        }
        outputs.extend(cutlass_decode::Decoder::drain(decoder.as_mut()).unwrap());
        let mut ours = Vec::new();
        let mut times = Vec::new();
        for output in outputs {
            let Output::Picture(picture) = output else {
                panic!("sound");
            };
            let PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("platform");
            };
            times.push(picture.pts());
            ours.push(bytes.to_vec());
        }
        assert_eq!(times, (0..i64::from(FRAMES)).collect::<Vec<_>>());

        // Without FFmpeg only our demuxer and decoder checked the MP4.
        let Some(ffmpeg) = crate::oracles::ffmpeg() else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("cutlass-mpeg2-mp4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mp4 = dir.join("export.mp4");
        let yuv = dir.join("ffmpeg.yuv");
        std::fs::write(&mp4, &file).unwrap();
        let run = Command::new(&ffmpeg)
            .args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(["-err_detect", "explode", "-xerror", "-i"])
            .arg(&mp4)
            .args(["-fps_mode", "passthrough", "-f", "rawvideo"])
            .args(["-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let theirs = std::fs::read(&yuv).unwrap();
        let size = (W * H * 3 / 2) as usize;
        assert_eq!(theirs.len(), size * FRAMES as usize);
        for (index, (frame, mine)) in theirs.chunks(size).zip(&ours).enumerate() {
            let quality = psnr(frame, mine);
            assert!(quality >= 50.0, "frame {index}: {quality:.1} dB");
        }

        if let Some(ffprobe) = crate::oracles::ffprobe(&ffmpeg) {
            let probe = Command::new(ffprobe)
                .args(["-v", "error", "-show_entries"])
                .args(["stream=codec_name,profile,width,height,has_b_frames"])
                .args(["-of", "default=noprint_wrappers=1"])
                .arg(&mp4)
                .output()
                .unwrap();
            assert!(
                probe.status.success(),
                "{}",
                String::from_utf8_lossy(&probe.stderr)
            );
            let text = String::from_utf8_lossy(&probe.stdout);
            for line in [
                "codec_name=mpeg2video",
                "profile=Main",
                "width=176",
                "height=144",
                "has_b_frames=1",
            ] {
                assert!(text.contains(line), "ffprobe: {text}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
