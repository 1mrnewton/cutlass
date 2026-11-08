//! The encoder through its public API: what it emits, how a decoder reads
//! it back, its rate, its VBV, and its lifecycle.

#[cfg(test)]
mod encode {
    use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, PixelFormat, Yuv420p};
    use cutlass_encode::{
        CpuFrame, EncodeFactory, FrameInput, FrameTime, VideoEncoder, VideoInput,
    };
    use cutlass_mpeg2::{Mpeg2Decoder, Mpeg2Encoder, register_encoders};

    /// A textured picture moving two samples right and one down per frame,
    /// with fixed noise so the rate, not the quantiser floor, limits it.
    fn picture(width: u32, height: u32, t: u32) -> Yuv420p {
        let (w, h) = (width as usize, height as usize);
        let mut seed = 0x1234_5678u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 28) as f64 - 7.5
        };
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                let (u, v) = (col as f64 - 2.0 * t as f64, row as f64 - t as f64);
                let value = 128.0
                    + 60.0 * (u * 0.15).sin() * (v * 0.11).cos()
                    + 25.0 * ((u * 0.04 + v * 0.07).sin())
                    + noise();
                y[row * w + col] = value.round().clamp(16.0, 235.0) as u8;
            }
        }
        let chroma = |phase: f64| {
            let mut plane = vec![0u8; (w / 2) * (h / 2)];
            for row in 0..h / 2 {
                for col in 0..w / 2 {
                    let u = col as f64 - t as f64;
                    plane[row * (w / 2) + col] =
                        (128.0 + 30.0 * (u * 0.1 + row as f64 * 0.05 + phase).sin()) as u8;
                }
            }
            plane
        };
        Yuv420p::try_new(width, height, y, chroma(0.0), chroma(1.3)).unwrap()
    }

    fn input(width: u32, height: u32, t: u32) -> VideoInput {
        VideoInput {
            picture: picture(width, height, t),
            pts: i64::from(t),
            duration: 1,
            timescale: 25,
            force_keyframe: false,
        }
    }

    fn open(width: u32, height: u32, bitrate: u32, constant: bool) -> Mpeg2Encoder {
        let mut params = CodecParams::video(CodecId::Mpeg2, width, height);
        params.bitrate = bitrate;
        params.constant_rate = constant;
        Mpeg2Encoder::open(&params).unwrap()
    }

    fn encode(encoder: &mut Mpeg2Encoder, width: u32, height: u32, frames: u32) -> Vec<Packet> {
        let mut packets = Vec::new();
        for t in 0..frames {
            packets.extend(encoder.push(&input(width, height, t)).unwrap());
        }
        packets.extend(encoder.drain().unwrap());
        packets
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        }
    }

    /// `(picture_coding_type, temporal_reference)` of a packet's picture.
    fn picture_type(data: &[u8]) -> (u8, u16) {
        let at = data
            .windows(4)
            .position(|w| w == [0, 0, 1, 0])
            .expect("picture start code");
        let header = &data[at + 4..];
        let reference = (u16::from(header[0]) << 2) | u16::from(header[1] >> 6);
        ((header[1] >> 3) & 7, reference)
    }

    #[test]
    fn a_decoder_reads_back_every_picture_in_display_order() {
        let (w, h) = (176, 144);
        let mut encoder = open(w, h, 1_500_000, false);
        let packets = encode(&mut encoder, w, h, 30);
        assert_eq!(packets.len(), 30);
        let mut decoder = Mpeg2Decoder::default();
        let mut frames = Vec::new();
        for packet in &packets {
            frames.extend(decoder.push(packet).unwrap());
        }
        frames.extend(decoder.drain().unwrap());
        assert_eq!(frames.len(), 30);
        for (t, frame) in frames.iter().enumerate() {
            assert_eq!(frame.pts, t as i64, "display order");
            let source = picture(w, h, t as u32);
            assert_eq!((frame.picture.width(), frame.picture.height()), (w, h));
            let y = psnr(source.y(), frame.picture.y());
            let u = psnr(source.u(), frame.picture.u());
            let v = psnr(source.v(), frame.picture.v());
            assert!(
                y > 32.0 && u > 32.0 && v > 32.0,
                "frame {t}: {y:.1} {u:.1} {v:.1}"
            );
        }
    }

    #[test]
    fn packets_leave_in_decode_order_with_display_timestamps() {
        let (w, h) = (64, 48);
        let mut encoder = open(w, h, 0, false);
        let packets = encode(&mut encoder, w, h, 14);
        let order: Vec<(u8, u16, i64, bool)> = packets
            .iter()
            .map(|p| {
                let (kind, reference) = picture_type(&p.data);
                (kind, reference, p.pts, p.keyframe)
            })
            .collect();
        // Twelve pictures per GOP at 25 Hz: I0 P3 B1 B2 P6 B4 B5 P9 B7 B8
        // P11 B10, then I12 and, at drain, P13 as the last anchor.
        let expected = [
            (1, 0, 0, true),
            (2, 3, 3, false),
            (3, 1, 1, false),
            (3, 2, 2, false),
            (2, 6, 6, false),
            (3, 4, 4, false),
            (3, 5, 5, false),
            (2, 9, 9, false),
            (3, 7, 7, false),
            (3, 8, 8, false),
            (2, 11, 11, false),
            (3, 10, 10, false),
            (1, 0, 12, true),
            (2, 1, 13, false),
        ];
        assert_eq!(order, expected);
        // Every I picture starts with the sequence header and a closed GOP.
        for packet in packets.iter().filter(|p| p.keyframe) {
            assert_eq!(&packet.data[..4], &[0, 0, 1, 0xb3]);
            let gop = packet
                .data
                .windows(4)
                .position(|w| w == [0, 0, 1, 0xb8])
                .unwrap();
            assert_eq!(packet.data[gop + 7] & 0x40, 0x40, "closed_gop");
        }
        let last = packets.last().unwrap();
        assert_eq!(&last.data[last.data.len() - 4..], &[0, 0, 1, 0xb7]);
        assert_eq!(
            packets
                .iter()
                .filter(|p| p.data.windows(4).any(|w| w == [0, 0, 1, 0xb7]))
                .count(),
            1
        );
    }

    #[test]
    fn extradata_is_the_sequence_header_and_extension() {
        let mut encoder = open(720, 576, 0, false);
        assert!(encoder.extradata().is_empty(), "no rate before a picture");
        encoder.push(&input(720, 576, 0)).unwrap();
        let extra = encoder.extradata().to_vec();
        assert_eq!(&extra[..4], &[0, 0, 1, 0xb3]);
        // 720 × 576, square samples, 25 Hz.
        assert_eq!(&extra[4..8], &[0x2d, 0x02, 0x40, 0x13]);
        let ext = extra.windows(4).position(|w| w == [0, 0, 1, 0xb5]).unwrap();
        // Sequence extension, Main profile at Main level, progressive 4:2:0.
        assert_eq!(extra[ext + 4] >> 4, 1);
        let profile_level = (extra[ext + 4] << 4) | (extra[ext + 5] >> 4);
        assert_eq!(profile_level, 0x48);
        assert_eq!(extra[ext + 5] & 0x0e, 0x0a);
        assert_eq!(extra.len(), ext + 10);
        let mut decoder = Mpeg2Decoder::open(&{
            let mut params = CodecParams::video(CodecId::Mpeg2, 720, 576);
            params.extra = extra;
            params
        });
        assert!(decoder.as_mut().is_ok());
    }

    /// The `profile_and_level_indication` in an encoder's extradata.
    fn level(width: u32, height: u32, timescale: u32, duration: i64) -> Result<u8, CodecError> {
        let mut encoder = Mpeg2Encoder::open(&CodecParams::video(CodecId::Mpeg2, width, height))?;
        let mut first = input(width, height, 0);
        first.timescale = timescale;
        first.duration = duration;
        encoder.push(&first)?;
        let extra = encoder.extradata();
        let ext = extra.windows(4).position(|w| w == [0, 0, 1, 0xb5]).unwrap();
        Ok((extra[ext + 4] << 4) | (extra[ext + 5] >> 4))
    }

    #[test]
    fn the_level_follows_the_size_and_frame_rate() {
        assert_eq!(level(720, 480, 30_000, 1001), Ok(0x48));
        assert_eq!(level(1280, 720, 50, 1), Ok(0x46));
        // 55.3 million luma samples a second is past High-1440.
        assert_eq!(level(1280, 720, 60, 1), Ok(0x44));
        assert_eq!(level(1920, 1080, 25, 1), Ok(0x44));
        assert_eq!(level(1920, 1080, 30_000, 1001), Ok(0x44));
        // 1080p at 59.94 Hz is twice High level's sample rate: no Main
        // profile level holds it.
        assert!(matches!(
            level(1920, 1080, 60_000, 1001),
            Err(CodecError::Unsupported { .. })
        ));
        // 90 kHz timing at 29.97 still matches Table 6-4.
        assert_eq!(level(640, 360, 90_000, 3003), Ok(0x48));
        assert!(matches!(
            level(640, 360, 15, 1),
            Err(CodecError::Unsupported { .. })
        ));
    }

    #[test]
    fn the_rate_lands_near_the_target() {
        let (w, h) = (352, 288);
        for (target, constant) in [(800_000u32, false), (1_500_000, true)] {
            let mut encoder = open(w, h, target, constant);
            let frames = 48;
            let bytes: usize = encode(&mut encoder, w, h, frames)
                .iter()
                .map(|p| p.data.len())
                .sum();
            let rate = bytes as f64 * 8.0 * 25.0 / f64::from(frames);
            let ratio = rate / f64::from(target);
            assert!(
                (0.9..=1.1).contains(&ratio),
                "{target} {constant}: {rate:.0} ({ratio:.3})"
            );
        }
    }

    /// Fields of a sequence header: `bit_rate` in bit/s, `vbv_buffer_size`
    /// in bits, with their sequence extension bits.
    fn rate_and_buffer(data: &[u8]) -> (f64, f64) {
        let h = &data[4..];
        let rate_low = (u32::from(h[4]) << 10) | (u32::from(h[5]) << 2) | u32::from(h[6] >> 6);
        let vbv_low = (u32::from(h[6] & 0x1f) << 5) | u32::from(h[7] >> 3);
        let ext = data.windows(4).position(|w| w == [0, 0, 1, 0xb5]).unwrap() + 4;
        let e = &data[ext..];
        let rate_high = (u32::from(e[2] & 0x1f) << 7) | u32::from(e[3] >> 1);
        let vbv_high = u32::from(e[4]);
        (
            f64::from((rate_high << 18) | rate_low) * 400.0,
            f64::from((vbv_high << 10) | vbv_low) * 16_384.0,
        )
    }

    /// Replays Annex C from the header fields alone: the buffer fills at
    /// `bit_rate` (capped at full for a variable rate) and each picture is
    /// removed whole one frame period later. Returns the lowest occupancy
    /// seen after a removal.
    fn replay_vbv(packets: &[Packet], frame_rate: f64, constant: bool) -> f64 {
        let (rate, size) = rate_and_buffer(&packets[0].data);
        let mut occupancy = size;
        let mut lowest = f64::MAX;
        for packet in packets {
            occupancy -= packet.data.len() as f64 * 8.0;
            lowest = lowest.min(occupancy);
            occupancy += rate / frame_rate;
            if constant {
                assert!(occupancy <= size + 8.0, "overflow: {occupancy} > {size}");
            }
            occupancy = occupancy.min(size);
        }
        lowest
    }

    #[test]
    fn the_decoder_buffer_never_underflows() {
        let (w, h) = (352, 288);
        for (target, constant) in [(400_000u32, false), (400_000, true), (3_000_000, true)] {
            let mut encoder = open(w, h, target, constant);
            let packets = encode(&mut encoder, w, h, 40);
            let lowest = replay_vbv(&packets, 25.0, constant);
            assert!(lowest >= 0.0, "{target} {constant}: {lowest}");
        }
    }

    #[test]
    fn a_forced_keyframe_closes_the_gop_there() {
        let (w, h) = (64, 48);
        let mut encoder = open(w, h, 0, false);
        let mut packets = Vec::new();
        for t in 0..8 {
            let mut frame = input(w, h, t);
            frame.force_keyframe = t == 5;
            packets.extend(encoder.push(&frame).unwrap());
        }
        packets.extend(encoder.drain().unwrap());
        let order: Vec<(u8, i64)> = packets
            .iter()
            .map(|p| (picture_type(&p.data).0, p.pts))
            .collect();
        // I0 P3 B1 B2, then frame 4 waits; the keyframe at 5 makes it the
        // GOP's last anchor. Drain ends the next GOP with P7 and B6.
        assert_eq!(
            order,
            [
                (1, 0),
                (2, 3),
                (3, 1),
                (3, 2),
                (2, 4),
                (1, 5),
                (2, 7),
                (3, 6)
            ]
        );
        assert!(packets[5].keyframe);
        assert_eq!(packets.iter().filter(|p| p.keyframe).count(), 2);
    }

    #[test]
    fn drain_reset_and_cancel() {
        let (w, h) = (64, 48);
        let mut encoder = open(w, h, 0, false);
        assert!(encoder.drain().unwrap().is_empty(), "nothing pushed");
        for t in 0..4 {
            encoder.push(&input(w, h, t)).unwrap();
        }
        let tail = encoder.drain().unwrap();
        assert!(!tail.is_empty());
        assert!(encoder.drain().unwrap().is_empty(), "second drain");
        // After the sequence ended, the next picture starts a new one.
        let mut next = Vec::new();
        for t in 4..6 {
            next.extend(encoder.push(&input(w, h, t)).unwrap());
        }
        next.extend(encoder.drain().unwrap());
        assert!(next[0].keyframe);
        assert_eq!(&next[0].data[..4], &[0, 0, 1, 0xb3]);

        encoder.push(&input(w, h, 0)).unwrap();
        encoder.push(&input(w, h, 1)).unwrap();
        encoder.reset();
        let after = encode(&mut encoder, w, h, 2);
        assert_eq!(after.len(), 2, "reset dropped the waiting pictures");
        assert!(after[0].keyframe);

        let flag = encoder.cancel_flag();
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            encoder.push(&input(w, h, 0)).unwrap_err(),
            CodecError::Cancelled
        );
        encoder.reset();
        assert!(encoder.push(&input(w, h, 0)).is_ok(), "reset clears cancel");
    }

    #[test]
    fn refusals_leave_the_encoder_as_it_was() {
        let (w, h) = (64, 48);
        let mut encoder = open(w, h, 0, false);
        let mut packets = Vec::new();
        packets.extend(encoder.push(&input(w, h, 0)).unwrap());
        let wrong = VideoInput {
            picture: picture(32, 32, 1),
            ..input(w, h, 1)
        };
        assert!(matches!(
            encoder.push(&wrong),
            Err(CodecError::InvalidApi { .. })
        ));
        let mut bad = input(w, h, 1);
        bad.timescale = 0;
        assert!(encoder.push(&bad).is_err());
        let mut bad = input(w, h, 1);
        bad.duration = -1;
        assert!(encoder.push(&bad).is_err());
        for t in 1..6 {
            packets.extend(encoder.push(&input(w, h, t)).unwrap());
        }
        packets.extend(encoder.drain().unwrap());
        let mut reference = open(w, h, 0, false);
        let clean = encode(&mut reference, w, h, 6);
        assert_eq!(packets, clean);
    }

    #[test]
    fn unsupported_configurations_are_refused() {
        let open = |codec, w, h| Mpeg2Encoder::open(&CodecParams::video(codec, w, h));
        assert!(matches!(
            open(CodecId::H264, 64, 48),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            open(CodecId::Mpeg2, 1922, 1080),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            open(CodecId::Mpeg2, 1920, 1160),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            open(CodecId::Mpeg2, 0, 48),
            Err(CodecError::Unsupported { .. })
        ));
        let mut params = CodecParams::video(CodecId::Mpeg2, 720, 576);
        params.bitrate = 90_000_000;
        let mut encoder = Mpeg2Encoder::open(&params).unwrap();
        assert!(matches!(
            encoder.push(&input(720, 576, 0)),
            Err(CodecError::Unsupported { .. })
        ));
    }

    #[test]
    fn pictures_wait_at_most_three_pushes() {
        let (w, h) = (64, 48);
        let mut encoder = open(w, h, 0, false);
        let mut out = 0usize;
        for t in 0..40u32 {
            out += encoder.push(&input(w, h, t)).unwrap().len();
            let waiting = t as usize + 1 - out;
            assert!(waiting <= 3, "{waiting} pictures waiting after {t}");
        }
    }

    #[test]
    fn padded_sizes_crop_back_on_decode() {
        let (w, h) = (100, 70);
        let mut encoder = open(w, h, 0, false);
        let packets = encode(&mut encoder, w, h, 4);
        let mut decoder = Mpeg2Decoder::default();
        let mut frames = Vec::new();
        for packet in &packets {
            frames.extend(decoder.push(packet).unwrap());
        }
        frames.extend(decoder.drain().unwrap());
        assert_eq!(frames.len(), 4);
        for (t, frame) in frames.iter().enumerate() {
            assert_eq!((frame.picture.width(), frame.picture.height()), (w, h));
            let y = psnr(picture(w, h, t as u32).y(), frame.picture.y());
            assert!(y > 32.0, "frame {t}: {y:.1}");
        }
    }

    #[test]
    fn the_factory_takes_rgba_frames_and_attaches_the_configuration() {
        let mut factory = EncodeFactory::new();
        register_encoders(&mut factory);
        let params = CodecParams::with_format(CodecId::Mpeg2, 64, 48, PixelFormat::Rgba8);
        let mut encoder = factory.open(&params).unwrap();
        let mut packets = Vec::new();
        for t in 0..5i64 {
            let rgba = vec![(t * 20) as u8; 64 * 48 * 4];
            let time = FrameTime::new(t, 1, 25).unwrap();
            let frame = CpuFrame::new(64, 48, PixelFormat::Rgba8, rgba, time).unwrap();
            packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).unwrap());
        }
        packets.extend(encoder.drain().unwrap());
        assert_eq!(packets.len(), 5);
        let configs: Vec<_> = packets
            .iter()
            .flat_map(|p| p.side.iter())
            .filter_map(|side| match side {
                cutlass_decode::SideData::Config(config) => Some(config.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(configs.len(), 1);
        assert_eq!(&configs[0][..4], &[0, 0, 1, 0xb3]);
        assert!(packets[0].side.len() == 1 && packets[0].keyframe);
    }
}
