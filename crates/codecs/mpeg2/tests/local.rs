//! Real recordings from `../local-files`, both ways through an MP4.
//!
//! FFmpeg, as the test oracle, decodes the opening frames of each clip to
//! 4:2:0. Our encoder codes them into an `Mp4Mux` file that FFmpeg must
//! decode with `-err_detect explode -xerror`, close to the source and to
//! our own decode, at close to the asked rate. FFmpeg's `mpeg2video` codes
//! the same frames into its own MP4 and MOV, which our demuxer must name
//! MPEG-2 and our decoder must play as FFmpeg does.
//!
//! The folder is not in CI, and neither may FFmpeg be; the test then says
//! on stderr what is missing and skips, or fails when
//! `MPEG2_REQUIRE_ORACLES` is set. With the flag, every clip below must be
//! there.

#[cfg(test)]
mod oracles;

#[cfg(test)]
mod local {
    use std::io::Cursor;
    use std::path::Path;
    use std::process::Command;

    use cutlass_codec::{
        CodecId as Id, FrameSize, Packet as MuxPacket, StreamCodec, StreamParams, Timescale,
        Timestamp,
    };
    use cutlass_container::{Demux as _, Mux as _, NewStream, TrackDuration};
    use cutlass_decode::{
        CodecId, CodecParams, DecodeFactory, Output, Packet, PictureSamples, Yuv420p,
    };
    use cutlass_encode::{VideoEncoder, VideoInput};
    use cutlass_mpeg2::Mpeg2Encoder;

    struct Clip {
        name: &'static str,
        /// Scale to fit Main profile, or `None` for the clip's own size.
        scale: Option<(u32, u32)>,
        rate: (u32, u32),
        bitrate: u32,
    }

    const CLIPS: [Clip; 4] = [
        Clip {
            name: "16929689_1920_1080_30fps.mp4",
            scale: None,
            rate: (30_000, 1001),
            bitrate: 12_000_000,
        },
        Clip {
            name: "6905095-hd_1920_1080_25fps.mp4",
            scale: None,
            rate: (25, 1),
            bitrate: 10_000_000,
        },
        Clip {
            name: "IMG_8146.MP4",
            scale: None,
            rate: (30, 1),
            bitrate: 2_000_000,
        },
        Clip {
            name: "7150172-hd_720_1280_50fps.mp4",
            scale: Some((576, 1024)),
            rate: (50, 1),
            bitrate: 6_000_000,
        },
    ];
    const FRAMES: usize = 30;

    fn run(command: &mut Command) {
        let out = command.output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let error = a
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

    /// FFmpeg's strict decode of `file` to 4:2:0, frames in display order.
    fn ffmpeg_decode(ffmpeg: &Path, file: &Path, out: &Path) -> Vec<u8> {
        run(Command::new(ffmpeg)
            .args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(["-err_detect", "explode", "-xerror", "-i"])
            .arg(file)
            .args(["-fps_mode", "passthrough", "-f", "rawvideo"])
            .args(["-pix_fmt", "yuv420p"])
            .arg(out));
        std::fs::read(out).unwrap()
    }

    /// Our engine decoder over every packet of an MP4 or MOV.
    fn our_decode(file: Vec<u8>) -> (StreamParams, Vec<Vec<u8>>) {
        let mut movie = cutlass_isobmff::IsoBmffFile::open(Cursor::new(file)).unwrap();
        let stream = movie.streams()[0].clone();
        let mut decoders = DecodeFactory::new();
        cutlass_mpeg2::register_decoders(&mut decoders);
        let mut params = CodecParams::video(CodecId::Mpeg2, 0, 0);
        params.extra = stream.params().extradata().to_vec();
        let mut decoder = decoders.open(&params).unwrap();
        let mut outputs = Vec::new();
        while let Some(packet) = movie.read_packet().unwrap() {
            let time = packet.timestamp();
            let packet = Packet::video(
                packet.bytes().to_vec(),
                time.presentation_or_decode().unwrap(),
                time.duration().unwrap(),
                packet.timescale().ticks_per_second(),
                packet.keyframe(),
            );
            outputs.extend(cutlass_decode::Decoder::push(decoder.as_mut(), &packet).unwrap());
        }
        outputs.extend(cutlass_decode::Decoder::drain(decoder.as_mut()).unwrap());
        let pictures = outputs
            .into_iter()
            .map(|output| {
                let Output::Picture(picture) = output else {
                    panic!("sound");
                };
                let PictureSamples::Cpu(bytes) = picture.samples() else {
                    panic!("platform");
                };
                bytes.to_vec()
            })
            .collect();
        (stream.params().clone(), pictures)
    }

    /// The share of slices whose header asks for quantiser code 1, the
    /// finest MPEG-2 has.
    fn finest_share(packets: &[Vec<u8>]) -> f64 {
        let (mut finest, mut slices) = (0, 0);
        for data in packets {
            for at in 0..data.len().saturating_sub(4) {
                if data[at..at + 3] == [0, 0, 1] && (1..=0xAF).contains(&data[at + 3]) {
                    slices += 1;
                    if data[at + 4] >> 3 == 1 {
                        finest += 1;
                    }
                }
            }
        }
        f64::from(finest) / f64::from(slices.max(1))
    }

    /// Our encode of `frames` into an MP4, the bytes it coded, and the
    /// share of its slices at the finest quantiser.
    fn our_mp4(clip: &Clip, size: (u32, u32), frames: &[&[u8]]) -> (Vec<u8>, usize, f64) {
        let (width, height) = size;
        let mut params = CodecParams::video(CodecId::Mpeg2, width, height);
        params.bitrate = clip.bitrate;
        let mut encoder = Mpeg2Encoder::open(&params).unwrap();
        let (luma, chroma) = ((width * height) as usize, (width * height / 4) as usize);
        let (scale, duration) = clip.rate;
        let mut packets = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let picture = Yuv420p::try_new(
                width,
                height,
                frame[..luma].to_vec(),
                frame[luma..luma + chroma].to_vec(),
                frame[luma + chroma..].to_vec(),
            )
            .unwrap();
            let input = VideoInput {
                picture,
                pts: index as i64 * i64::from(duration),
                duration: i64::from(duration),
                timescale: scale,
                force_keyframe: false,
            };
            packets.extend(encoder.push(&input).unwrap());
        }
        packets.extend(VideoEncoder::drain(&mut encoder).unwrap());
        let coded = packets.iter().map(|packet| packet.data.len()).sum();
        let data: Vec<Vec<u8>> = packets.iter().map(|packet| packet.data.clone()).collect();
        let finest = finest_share(&data);

        let mut mux = cutlass_isobmff::Mp4Mux::new(Cursor::new(Vec::new()));
        let track = mux
            .add_stream(NewStream {
                params: StreamParams::video(
                    StreamCodec::Known(Id::Mpeg2),
                    FrameSize::try_new(width, height).unwrap(),
                    Vec::new(),
                ),
                timescale: Timescale::new(scale),
                language: None,
            })
            .unwrap();
        mux.set_config(track, encoder.extradata()).unwrap();
        for packet in &packets {
            let time =
                Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
                    .unwrap();
            let out = MuxPacket::try_new(
                track.get(),
                packet.data.clone(),
                time,
                Timescale::new(scale),
                packet.keyframe,
            )
            .unwrap();
            mux.write_packet(&out).unwrap();
        }
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(scale),
            ticks: (frames.len() as i64) * i64::from(duration),
        }])
        .unwrap();
        (mux.into_writer().unwrap().into_inner(), coded, finest)
    }

    fn worst(a: &[u8], b: &[Vec<u8>], frame: usize) -> f64 {
        assert_eq!(a.len(), frame * b.len());
        a.chunks(frame)
            .zip(b)
            .map(|(x, y)| psnr(x, y))
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn local_recordings_round_trip_through_mp4_both_ways() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
        if !root.is_dir() {
            crate::oracles::skip(&root.display().to_string());
            return;
        }
        let Some(ffmpeg) = crate::oracles::ffmpeg() else {
            return;
        };
        let Some(ffprobe) = crate::oracles::ffprobe(&ffmpeg) else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("cutlass-mpeg2-local-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut checked = 0;
        for clip in &CLIPS {
            let source = root.join(clip.name);
            if !source.is_file() {
                crate::oracles::skip(&format!("local-files/{}", clip.name));
                continue;
            }
            let raw = dir.join("source.yuv");
            let mut filter = String::from("format=yuv420p");
            if let Some((w, h)) = clip.scale {
                filter = format!("scale={w}:{h},{filter}");
            }
            run(Command::new(&ffmpeg)
                .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
                .arg(&source)
                .args(["-frames:v", &FRAMES.to_string(), "-vf", &filter])
                .args(["-f", "rawvideo"])
                .arg(&raw));
            let probe = Command::new(&ffprobe)
                .args(["-v", "error", "-select_streams", "v:0"])
                .args(["-show_entries", "stream=width,height", "-of", "csv=p=0"])
                .arg(&source)
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&probe.stdout);
            let mut sides = text.trim().split(',').map(|n| n.parse::<u32>().unwrap());
            let size = clip
                .scale
                .unwrap_or((sides.next().unwrap(), sides.next().unwrap()));
            let frame = (size.0 * size.1 * 3 / 2) as usize;
            let source_bytes = std::fs::read(&raw).unwrap();
            let frames: Vec<&[u8]> = source_bytes.chunks_exact(frame).collect();
            assert_eq!(frames.len(), FRAMES, "{}", clip.name);

            // Ours into an MP4, read by FFmpeg.
            let (mp4, coded, finest) = our_mp4(clip, size, &frames);
            let ours_path = dir.join("ours.mp4");
            std::fs::write(&ours_path, &mp4).unwrap();
            let theirs = ffmpeg_decode(&ffmpeg, &ours_path, &dir.join("ours.yuv"));
            let (params, mine) = our_decode(mp4);
            assert_eq!(params.codec(), &StreamCodec::Known(Id::Mpeg2));
            let agree = worst(&theirs, &mine, frame);
            let owned: Vec<Vec<u8>> = frames.iter().map(|f| f.to_vec()).collect();
            let quality = worst(&theirs, &owned, frame);
            let seconds = FRAMES as f64 * f64::from(clip.rate.1) / f64::from(clip.rate.0);
            let ratio = coded as f64 * 8.0 / seconds / f64::from(clip.bitrate);
            println!(
                "{}: ours {}x{} at {:.2} of the rate, worst frame {quality:.1} dB, \
                 FFmpeg agrees to {agree:.1} dB, {:.0} % of slices at quantiser 1",
                clip.name,
                size.0,
                size.1,
                ratio,
                finest * 100.0
            );
            assert!(agree >= 50.0, "{}: {agree:.1} dB", clip.name);
            assert!(quality >= 30.0, "{}: {quality:.1} dB", clip.name);
            // Thirty frames are one GOP and the next I picture at 50 fps,
            // hence the wide band. Below it only when the picture needs no
            // more: nearly every slice already at the finest quantiser.
            let saturated = ratio < 0.75 && finest >= 0.9;
            assert!(
                (0.75..=1.25).contains(&ratio) || saturated,
                "{}: {ratio:.2}, {:.0} % finest",
                clip.name,
                finest * 100.0
            );

            // FFmpeg's MPEG-2 in its MP4 and its MOV, read by us.
            for extension in ["mp4", "mov"] {
                let file = dir.join(format!("ffmpeg.{extension}"));
                run(Command::new(&ffmpeg)
                    .args(["-y", "-hide_banner", "-loglevel", "error"])
                    .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
                    .args(["-s", &format!("{}x{}", size.0, size.1)])
                    .args(["-r", &format!("{}/{}", clip.rate.0, clip.rate.1), "-i"])
                    .arg(&raw)
                    .args(["-c:v", "mpeg2video", "-g", "12", "-bf", "2"])
                    .args(["-b:v", &clip.bitrate.to_string()])
                    .arg(&file));
                let bytes = std::fs::read(&file).unwrap();
                let theirs = ffmpeg_decode(&ffmpeg, &file, &dir.join("ffmpeg.yuv"));
                let (params, mine) = our_decode(bytes);
                assert_eq!(params.codec(), &StreamCodec::Known(Id::Mpeg2));
                let agree = worst(&theirs, &mine, frame);
                println!(
                    "{}: FFmpeg's {extension} decodes to {agree:.1} dB",
                    clip.name
                );
                assert!(agree >= 50.0, "{} {extension}: {agree:.1} dB", clip.name);
            }
            checked += 1;
        }
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("{checked} of {} recordings checked", CLIPS.len());
    }
}
