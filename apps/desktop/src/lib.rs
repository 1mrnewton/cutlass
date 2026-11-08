#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

pub mod assistant;
mod dialogs;
mod face;
mod gpu;
pub mod icons;
pub mod jobs;
pub mod mcp;
mod menu;
pub mod menubar;
mod readers;
pub mod sandbox;

#[cfg(test)]
mod open_movie;
pub mod shell;
pub mod tokens;
pub mod ui;
pub mod view;
pub mod widgets;
pub mod window;

/// Platform openers go first. A codec they decline falls through to the
/// software opener registered after them. The Rust FLAC, ALAC, MP3, Opus
/// and WebP codecs are the exception: they go ahead of the platform.
/// Hardware decode and encode are on.
pub fn register_codecs() -> (cutlass_decode::DecodeFactory, cutlass_encode::EncodeFactory) {
    register_codecs_with(&cutlass_settings::Hardware::default())
}

/// [`register_codecs`] with the user's hardware switches. A switch that is
/// off leaves the platform picture codecs out, so the Rust codec decodes or
/// encodes that id.
pub fn register_codecs_with(
    hardware: &cutlass_settings::Hardware,
) -> (cutlass_decode::DecodeFactory, cutlass_encode::EncodeFactory) {
    let mut decoders = cutlass_decode::DecodeFactory::new();
    let mut encoders = cutlass_encode::EncodeFactory::new();
    // Rust audio codecs go ahead of the platform ones, so a file decodes
    // and encodes the same on every platform. FLAC's encoder hands the muxer
    // the STREAMINFO (sample count and MD5) a FLAC file is finished with;
    // ALAC's hands it the final cookie; MP3's hands it the delay and padding
    // its LAME tag needs; AC-3 and E-AC-3 hand it `dac3` or `dec3`; Opus's
    // hands it the OpusHead, with its pre-skip,
    // and the end trim. The Opus decoder matches the RFC vectors and plays
    // multistream and every API rate. WebP still images follow the same
    // rule, and an animated file's later packets are bare frames only the
    // Rust decoder composes.
    cutlass_flac::register_decoders(&mut decoders);
    cutlass_flac::register_encoders(&mut encoders);
    cutlass_alac::register_decoders(&mut decoders);
    cutlass_alac::register_encoders(&mut encoders);
    cutlass_mp3::register_decoders(&mut decoders);
    cutlass_mp3::register_encoders(&mut encoders);
    cutlass_ac3::register_decoders(&mut decoders);
    cutlass_ac3::register_encoders(&mut encoders);
    cutlass_opus::register_decoders(&mut decoders);
    cutlass_opus::register_encoders(&mut encoders);
    cutlass_webp::register_decoders(&mut decoders);
    cutlass_webp::register_encoders(&mut encoders);
    #[cfg(target_os = "macos")]
    {
        // Pictures stay in their IOSurface; the preview samples them in place.
        cutlass_platform_apple::register_surface_decoders_with(&mut decoders, hardware.decode);
        cutlass_platform_apple::register_encoders_with(&mut encoders, hardware.encode);
    }
    #[cfg(target_os = "windows")]
    {
        cutlass_platform_windows::register_decoders_with(&mut decoders, hardware.decode);
        cutlass_platform_windows::register_encoders_with(&mut encoders, hardware.encode);
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let _ = hardware;
    cutlass_png::register_decoders(&mut decoders);
    cutlass_png::register_encoders(&mut encoders);
    cutlass_gif::register_decoders(&mut decoders);
    cutlass_jpeg::register_decoders(&mut decoders);
    cutlass_jpeg::register_encoders(&mut encoders);
    cutlass_aac::register_decoders(&mut decoders);
    cutlass_aac::register_encoders(&mut encoders);
    cutlass_av1::register_decoders(&mut decoders);
    cutlass_av1::register_encoders(&mut encoders);
    cutlass_pcm::register_decoders(&mut decoders);
    cutlass_pcm::register_encoders(&mut encoders);
    cutlass_h264::register_decoders(&mut decoders);
    cutlass_h264::register_encoders(&mut encoders);
    cutlass_prores::register_decoders(&mut decoders);
    cutlass_prores::register_encoders(&mut encoders);
    cutlass_vp8::register_engine_decoders(&mut decoders);
    cutlass_vp8::register_engine_encoders(&mut encoders);
    // After the platform decoders: the Rust HEVC decoder serves Linux, and
    // the other platforms when hardware decode is off.
    cutlass_hevc::register_decoders(&mut decoders);
    (decoders, encoders)
}

/// The user's settings at `path`, or the defaults.
///
/// A missing file is the defaults. A file that cannot be read is logged,
/// left as it was, and the defaults are used.
#[must_use]
pub fn load_settings(path: &Path) -> cutlass_settings::Settings {
    load_settings_file(path).settings
}

/// As [`load_settings`], with the generation a later save sends. A file
/// that cannot be read gives generation `u64::MAX`, so a save leaves it as
/// it was.
#[must_use]
pub fn load_settings_file(path: &Path) -> cutlass_settings::Loaded {
    match cutlass_settings::load(path) {
        Ok(loaded) => loaded,
        Err(cutlass_settings::SettingsError::Missing { .. }) => cutlass_settings::Loaded {
            generation: 0,
            settings: cutlass_settings::Settings::default(),
        },
        Err(error) => {
            cutlass_logger::warn!("{error}");
            cutlass_settings::Loaded {
                generation: u64::MAX,
                settings: cutlass_settings::Settings::default(),
            }
        }
    }
}

/// Open the process log before the window, record panics in it, and say
/// which build started on which system.
///
/// `None` uses the logs folder of the Cutlass root. A public build writes
/// the file and does not write stderr. `CUTLASS_LOG=debug` keeps DEBUG.
pub fn boot(log_directory: Option<&Path>) -> Result<PathBuf, cutlass_logger::InitError> {
    let opened = match log_directory {
        Some(directory) => cutlass_logger::init(directory),
        None => cutlass_logger::init(&shell::Shell::home().logs()),
    };
    cutlass_logger::install_panic_hook();
    cutlass_logger::info!(
        "Cutlass {} ({} build) on {} {}",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    opened
}

/// Record a failure from process start. The public build keeps it in the file.
pub fn report_error(message: &str) {
    cutlass_logger::error!("{message}");
}

#[cfg(test)]
mod tests {
    use cutlass_decode::{CodecId, CodecParams, ColorPrimaries, PixelFormat};

    fn picture(codec: CodecId, width: u32, height: u32) -> CodecParams {
        let mut params = CodecParams::with_format(codec, width, height, PixelFormat::Rgba8);
        params.primaries = ColorPrimaries::Bt709;
        params
    }

    #[test]
    fn the_desktop_opens_png_after_the_platform_codecs() {
        let (_decoders, encoders) = super::register_codecs();
        encoders
            .open(&picture(CodecId::Png, 0, 0))
            .expect("png encoder");
        #[cfg(target_os = "macos")]
        encoders
            .open(&picture(CodecId::H264, 64, 64))
            .expect("h264 encoder");
    }

    #[test]
    fn the_desktop_codes_webp_in_rust_ahead_of_the_platform() {
        let (decoders, encoders) = super::register_codecs();
        let mut encoder = encoders
            .open(&picture(CodecId::Webp, 0, 0))
            .expect("webp encoder");
        assert_eq!(encoder.codec(), CodecId::Webp);
        let image = cutlass_encode::StillImage::from_samples(
            2,
            2,
            cutlass_encode::ColorType::Rgb,
            cutlass_encode::BitDepth::Eight,
            vec![200; 12],
        );
        let packets = encoder
            .push_picture(&cutlass_encode::PictureInput { image })
            .expect("encode");
        let packet = cutlass_decode::Packet::video(packets[0].data.clone(), 0, 0, 0, true);
        let mut decoder = decoders
            .open(&picture(CodecId::Webp, 2, 2))
            .expect("webp decoder");
        let pictures = decoder.decode_picture(&packet).expect("decode");
        assert_eq!((pictures[0].image.width, pictures[0].image.height), (2, 2));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn hardware_decode_off_leaves_videotoolbox_out() {
        use cutlass_settings::Hardware;

        let hevc = CodecParams::with_format(CodecId::Hevc, 64, 48, PixelFormat::Yuv420p);
        let (on, _) = super::register_codecs_with(&Hardware::default());
        on.open(&hevc).expect("videotoolbox hevc");
        let off = Hardware {
            decode: false,
            ..Hardware::default()
        };
        let (decoders, encoders) = super::register_codecs_with(&off);
        // The HEVC decoder left is the Rust one: it decodes an Annex B frame
        // with no extradata into CPU samples, which VideoToolbox would not.
        let frame = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../crates/codecs/hevc/tests/fixtures/testsrc-64x48.hevc"),
        )
        .expect("fixture");
        let mut rust = decoders.open(&hevc).expect("rust hevc");
        let mut outputs = rust
            .push(&cutlass_decode::Packet::audio(frame, 0, 0, 1))
            .expect("decode");
        outputs.extend(rust.drain().expect("drain"));
        let [cutlass_decode::Output::Picture(decoded)] = outputs.as_slice() else {
            panic!("one picture, got {}", outputs.len());
        };
        assert!(matches!(
            decoded.samples(),
            cutlass_decode::PictureSamples::Cpu(_)
        ));
        let h264 = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        decoders.open(&h264).expect("rust h264");
        encoders
            .open(&picture(CodecId::H264, 64, 64))
            .expect("hardware encode stays on");
    }

    /// A second of stereo tones, MP3 packets from the Rust encoder, and
    /// what `decoders` makes of them.
    fn mp3_round_trip(
        encoders: &cutlass_encode::EncodeFactory,
        decoders: &cutlass_decode::DecodeFactory,
    ) -> (Vec<Vec<u8>>, Vec<f32>) {
        use cutlass_decode::{ChannelLayout, Output, Packet, Pcm};
        use cutlass_encode::AudioInput;

        let rate = 44_100;
        let mut params = CodecParams::audio(CodecId::Mp3, rate, ChannelLayout::stereo());
        params.bitrate = 128_000;
        params.constant_rate = true;
        let mut encoder = encoders.open(&params).expect("mp3 encoder");
        let samples: Vec<f32> = (0..rate as usize)
            .flat_map(|i| {
                let t = i as f32 / rate as f32;
                [
                    0.3 * (t * 440.0 * std::f32::consts::TAU).sin(),
                    0.2 * (t * 660.0 * std::f32::consts::TAU).sin(),
                ]
            })
            .collect();
        let pcm = Pcm::from_f32(rate, ChannelLayout::stereo(), samples).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("encode");
        packets.extend(encoder.drain().expect("drain"));
        let mut decoder = decoders
            .open(&CodecParams::audio(
                CodecId::Mp3,
                rate,
                ChannelLayout::stereo(),
            ))
            .expect("mp3 decoder");
        let mut decoded = Vec::new();
        for packet in &packets {
            let packet = Packet::audio(packet.data.clone(), packet.pts, packet.duration, rate);
            for output in decoder.push(&packet).expect("decode") {
                if let Output::Audio(audio) = output {
                    decoded.extend_from_slice(audio.pcm.as_f32().expect("f32"));
                }
            }
        }
        for output in decoder.drain().expect("drain") {
            if let Output::Audio(audio) = output {
                decoded.extend_from_slice(audio.pcm.as_f32().expect("f32"));
            }
        }
        let bytes = packets.into_iter().map(|packet| packet.data).collect();
        (bytes, decoded)
    }

    #[test]
    fn mp3_is_coded_by_the_rust_codec_ahead_of_the_platform() {
        let (decoders, encoders) = super::register_codecs();
        let mut rust_decoders = cutlass_decode::DecodeFactory::new();
        let mut rust_encoders = cutlass_encode::EncodeFactory::new();
        cutlass_mp3::register_decoders(&mut rust_decoders);
        cutlass_mp3::register_encoders(&mut rust_encoders);
        let app = mp3_round_trip(&encoders, &decoders);
        let rust = mp3_round_trip(&rust_encoders, &rust_decoders);
        assert!(!rust.0.is_empty() && !rust.1.is_empty());
        assert!(app.0 == rust.0, "the app encodes MP3 with the Rust encoder");
        assert!(app.1 == rust.1, "the app decodes MP3 with the Rust decoder");

        // With the platform first, macOS decodes MP3 itself, and differently:
        // the check above would fail if the order were reversed.
        #[cfg(target_os = "macos")]
        {
            let mut reversed = cutlass_decode::DecodeFactory::new();
            cutlass_platform_apple::register_surface_decoders_with(&mut reversed, true);
            cutlass_mp3::register_decoders(&mut reversed);
            let platform = mp3_round_trip(&rust_encoders, &reversed);
            assert!(platform.1 != rust.1, "the platform decoder was not picked");
        }
    }

    #[test]
    fn prores_opens_in_rust_when_the_platform_is_off() {
        use cutlass_settings::Hardware;

        let off = Hardware {
            decode: false,
            encode: false,
            ..Hardware::default()
        };
        let (decoders, encoders) = super::register_codecs_with(&off);
        let mut read = CodecParams::with_format(CodecId::Prores, 64, 48, PixelFormat::Yuv420p);
        read.extra = b"ap4h".to_vec();
        let decoder = decoders.open(&read).expect("rust prores decoder");
        assert_eq!(
            cutlass_decode::Decoder::codec(decoder.as_ref()),
            CodecId::Prores
        );
        let mut write = picture(CodecId::Prores, 64, 48);
        write.extra = b"apch".to_vec();
        let encoder = encoders.open(&write).expect("rust prores encoder");
        assert_eq!(
            cutlass_encode::Encoder::codec(encoder.as_ref()),
            CodecId::Prores
        );
    }

    #[test]
    fn settings_load_from_the_file_or_fall_back_to_defaults() {
        let dir = std::env::temp_dir().join(format!("cutlass-settings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("settings.json");
        assert_eq!(
            super::load_settings(&path),
            cutlass_settings::Settings::default()
        );
        std::fs::write(&path, r#"{"hardware":{"decode":false}}"#).expect("write");
        let loaded = super::load_settings(&path);
        assert!(!loaded.hardware.decode);
        assert!(loaded.hardware.encode);
        std::fs::write(&path, "{").expect("write");
        assert_eq!(
            super::load_settings(&path),
            cutlass_settings::Settings::default()
        );
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    #[test]
    fn the_desktop_opens_jpeg_after_the_platform_codecs() {
        let (decoders, encoders) = super::register_codecs();
        decoders
            .open(&picture(CodecId::Jpeg, 8, 8))
            .expect("jpeg decoder");
        encoders
            .open(&picture(CodecId::Jpeg, 8, 8))
            .expect("jpeg encoder");
    }

    /// The app's WebP decoder is the Rust one, ahead of ImageIO: an
    /// animation's later packets paint onto the canvas the first packet
    /// started, which only `cutlass-webp` keeps.
    #[test]
    fn the_desktop_decodes_webp_in_rust_ahead_of_the_platform() {
        use cutlass_decode::{Decoder, Output, Packet, PictureSamples};

        let file = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../crates/codecs/webp/tests/fixtures/anim-mux.webp"),
        )
        .unwrap();
        // The first frame's packet ends where the second `ANMF` starts.
        let second = file
            .windows(4)
            .enumerate()
            .filter(|(_, word)| *word == b"ANMF")
            .nth(1)
            .unwrap()
            .0;
        let packets = [file[..second].to_vec(), file[second..].to_vec()];
        let samples = |decoder: &mut dyn Decoder| -> Vec<Vec<u8>> {
            let mut out = Vec::new();
            for data in &packets {
                for output in decoder
                    .push(&Packet::audio(data.clone(), 0, 0, 1000))
                    .unwrap()
                {
                    let Output::Picture(picture) = output else {
                        panic!("picture");
                    };
                    let PictureSamples::Cpu(bytes) = picture.samples() else {
                        panic!("the platform decoder answered");
                    };
                    out.push(bytes.to_vec());
                }
            }
            out
        };
        let (decoders, _) = super::register_codecs();
        let mut app = decoders.open(&picture(CodecId::Webp, 40, 30)).unwrap();
        assert_eq!(Decoder::codec(app.as_ref()), CodecId::Webp);
        let mut rust = cutlass_webp::WebpDecoder::open(&picture(CodecId::Webp, 40, 30)).unwrap();
        let from_app = samples(app.as_mut());
        assert_eq!(from_app.len(), 6);
        assert_eq!(from_app, samples(&mut rust));
    }

    #[test]
    fn rust_codecs_open_when_the_platform_has_none() {
        let (decoders, encoders) = super::register_codecs();
        let video = CodecParams::video(CodecId::Av1, 16, 16);
        let decoder = decoders.open(&video).expect("decoder");
        let encoder = encoders.open(&video).expect("encoder");
        assert_eq!(
            cutlass_decode::Decoder::codec(decoder.as_ref()),
            CodecId::Av1
        );
        assert_eq!(
            cutlass_encode::Encoder::codec(encoder.as_ref()),
            CodecId::Av1
        );
    }

    #[test]
    fn pcm_falls_through_when_the_platform_declines_the_layout() {
        use cutlass_decode::{ChannelLayout, Decoder, Output, Packet, SampleFormat};
        use cutlass_encode::{AudioInput, Encoder};

        let (decoders, encoders) = super::register_codecs();
        let mut wide = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        wide.extra = vec![5, 0, 0];
        wide.sample_format = Some(SampleFormat::F32);
        let mut decoder = decoders.open(&wide).expect("pcm decoder");
        let outputs = decoder
            .push(&Packet::audio(0.5f64.to_le_bytes().to_vec(), 0, 1, 48_000))
            .expect("f64 packet");
        let Output::Audio(audio) = &outputs[0] else {
            panic!("audio");
        };
        assert_eq!(Decoder::codec(decoder.as_ref()), CodecId::Pcm);
        assert_eq!(audio.pcm.samples, 1);
        assert_eq!(audio.pcm.as_f32(), Some(&[0.5][..]));

        let mut spec = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        spec.extra = vec![1, 0, 0];
        spec.sample_format = Some(SampleFormat::I16);
        let mut encoder = encoders.open(&spec).expect("pcm encoder");
        assert_eq!(Encoder::codec(encoder.as_ref()), CodecId::Pcm);
        let pcm =
            cutlass_decode::Pcm::from_i16(48_000, ChannelLayout::mono(), vec![7]).expect("pcm");
        let packets = encoder.push(&AudioInput { pcm }).expect("encode");
        assert_eq!(packets.len(), 1);
        assert!(!packets[0].data.is_empty());
    }

    /// Half a second of a stereo chord through `encoder`, in uneven pushes.
    fn opus_packets(encoder: &mut dyn cutlass_encode::Encoder) -> Vec<cutlass_encode::Packet> {
        use cutlass_decode::{ChannelLayout, Pcm};
        let samples: Vec<f32> = (0..2 * 24_000)
            .map(|i| ((i as f32) * 0.013).sin() * 0.4 + ((i as f32) * 0.002).sin() * 0.2)
            .collect();
        let mut packets = Vec::new();
        for chunk in samples.chunks(2 * 1500) {
            let pcm = Pcm::from_f32(48_000, ChannelLayout::stereo(), chunk.to_vec()).expect("pcm");
            let input = cutlass_encode::AudioInput { pcm };
            packets.extend(encoder.push(&input).expect("push"));
        }
        packets.extend(encoder.drain().expect("drain"));
        packets
    }

    /// Every sample `decoder` gives for `packets`.
    fn opus_samples(
        decoder: &mut dyn cutlass_decode::Decoder,
        packets: &[cutlass_encode::Packet],
    ) -> Vec<f32> {
        let mut out = Vec::new();
        for packet in packets {
            for output in decoder.push(packet).expect("push") {
                if let cutlass_decode::Output::Audio(audio) = output {
                    out.extend(audio.pcm.as_f32().expect("f32"));
                }
            }
        }
        out
    }

    #[test]
    fn the_rust_opus_codec_goes_ahead_of_the_platform_one() {
        use cutlass_decode::ChannelLayout;

        let (decoders, encoders) = super::register_codecs();
        let mut spec = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
        spec.bitrate = 96_000;
        let mut app = encoders.open(&spec).expect("app opus encoder");
        let mut rust = cutlass_opus::OpusEncoder::open(&spec).expect("rust opus encoder");
        let packets = opus_packets(&mut rust);
        assert_eq!(opus_packets(app.as_mut()), packets);

        let listen = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
        let mut app = decoders.open(&listen).expect("app opus decoder");
        let mut rust = cutlass_opus::OpusDecoder::open(&listen).expect("rust opus decoder");
        let decoded = opus_samples(&mut rust, &packets);
        assert_eq!(opus_samples(app.as_mut(), &packets), decoded);

        // Platform first, the Apple codecs take Opus and give other bytes
        // and samples: the checks above would fail in that order.
        #[cfg(target_os = "macos")]
        {
            let mut reversed = cutlass_encode::EncodeFactory::new();
            cutlass_platform_apple::register_encoders_with(&mut reversed, true);
            cutlass_opus::register_encoders(&mut reversed);
            let mut platform = reversed.open(&spec).expect("platform opus encoder");
            assert_ne!(opus_packets(platform.as_mut()), packets);

            let mut reversed = cutlass_decode::DecodeFactory::new();
            cutlass_platform_apple::register_surface_decoders_with(&mut reversed, true);
            cutlass_opus::register_decoders(&mut reversed);
            let mut platform = reversed.open(&listen).expect("platform opus decoder");
            assert_ne!(opus_samples(platform.as_mut(), &packets), decoded);
        }
    }

    #[test]
    fn ac3_and_eac3_open_for_import_and_export() {
        use cutlass_decode::ChannelLayout;

        let (decoders, encoders) = super::register_codecs();
        for codec in [CodecId::Ac3, CodecId::Eac3] {
            let spec = CodecParams::audio(codec, 48_000, ChannelLayout::stereo());
            let encoder = encoders.open(&spec).expect("encoder");
            assert_eq!(cutlass_encode::Encoder::codec(encoder.as_ref()), codec);
            let decoder = decoders.open(&spec).expect("decoder");
            assert_eq!(cutlass_decode::Decoder::codec(decoder.as_ref()), codec);
        }
    }

    #[test]
    fn alac_opens_for_import_and_export() {
        use cutlass_decode::{ChannelLayout, SampleFormat};

        let (decoders, encoders) = super::register_codecs();
        let mut spec = CodecParams::audio(CodecId::Alac, 48_000, ChannelLayout::stereo());
        spec.sample_format = Some(SampleFormat::F32);
        let encoder = encoders.open(&spec).expect("alac encoder");
        assert_eq!(
            cutlass_encode::Encoder::codec(encoder.as_ref()),
            CodecId::Alac
        );
        spec.extra = cutlass_alac::Cookie::new(cutlass_alac::AlacConfig::new(48_000, 2, 24, 4096))
            .to_bytes();
        let decoder = decoders.open(&spec).expect("alac decoder");
        assert_eq!(
            cutlass_decode::Decoder::codec(decoder.as_ref()),
            CodecId::Alac
        );
    }

    #[test]
    fn the_rust_alac_encoder_goes_ahead_of_the_platform_one() {
        use cutlass_decode::{ChannelLayout, Pcm, SampleFormat};
        use cutlass_encode::{AudioInput, Encoder};

        let (_, encoders) = super::register_codecs();
        let mut spec = CodecParams::audio(CodecId::Alac, 48_000, ChannelLayout::stereo());
        spec.sample_format = Some(SampleFormat::F32);
        let samples: Vec<f32> = (0..2 * 5000)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let pcm = Pcm::from_f32(48_000, ChannelLayout::stereo(), samples).expect("pcm");
        let input = AudioInput { pcm };
        let encode = |encoder: &mut dyn Encoder| {
            let mut packets = encoder.push(&input).expect("push");
            packets.extend(encoder.drain().expect("drain"));
            packets
        };
        let mut app = encoders.open(&spec).expect("app alac encoder");
        let mut rust = cutlass_alac::AlacEncoder::open(&spec).expect("rust alac encoder");
        assert_eq!(encode(app.as_mut()), encode(&mut rust));
    }

    #[test]
    fn boot_writes_the_log_and_an_error() {
        super::test_log::text();
        super::report_error("window failed");
        let text = super::test_log::text();
        assert!(text.contains("log file"), "{text}");
        assert!(
            text.contains(&format!("Cutlass {}", env!("CARGO_PKG_VERSION"))),
            "{text}"
        );
        assert!(text.contains(std::env::consts::OS), "{text}");
        assert!(text.contains("ERROR"), "{text}");
        assert!(text.contains("window failed"), "{text}");
    }
}

/// One log for the whole test process. The logger is global, so tests that
/// read it share this folder and look for their own words.
#[cfg(test)]
pub(crate) mod test_log {
    use std::path::PathBuf;
    use std::sync::OnceLock;

    /// The live log file, opened through [`super::boot`] on first use.
    pub(crate) fn path() -> &'static PathBuf {
        static PATH: OnceLock<PathBuf> = OnceLock::new();
        PATH.get_or_init(|| {
            let directory =
                std::env::temp_dir().join(format!("cutlass-desktop-log-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&directory);
            super::boot(Some(&directory)).unwrap()
        })
    }

    /// Everything logged so far.
    pub(crate) fn text() -> String {
        std::fs::read_to_string(path()).unwrap_or_default()
    }
}
