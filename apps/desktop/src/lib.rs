#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

mod dialogs;
mod face;
mod gpu;
pub mod icons;
pub mod jobs;
pub mod mcp;
mod menu;
mod readers;

#[cfg(test)]
mod open_movie;
pub mod shell;
pub mod tokens;
pub mod view;
pub mod widgets;
pub mod window;

/// Platform openers go first. A codec they decline falls through to the
/// software opener registered after them. Hardware decode and encode are on.
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
    // and encodes the same on every platform. ALAC is bit-exact either way;
    // the Rust encoder also hands the muxer the final cookie.
    cutlass_alac::register_decoders(&mut decoders);
    cutlass_alac::register_encoders(&mut encoders);
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

/// Open the process log before the window.
///
/// `None` uses the logs folder of the Cutlass root. A public build writes
/// the file and does not write stderr.
pub fn boot(log_directory: Option<&Path>) -> Result<PathBuf, cutlass_logger::InitError> {
    match log_directory {
        Some(directory) => cutlass_logger::init(directory),
        None => cutlass_logger::init(&shell::Shell::home().logs()),
    }
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
        assert!(decoders.open(&hevc).is_err(), "hevc has no rust decoder");
        let h264 = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        decoders.open(&h264).expect("rust h264");
        encoders
            .open(&picture(CodecId::H264, 64, 64))
            .expect("hardware encode stays on");
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
        let path = std::env::temp_dir().join(format!("cutlass-desktop-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        let log = super::boot(Some(&path)).unwrap();
        super::report_error("window failed");
        let text = std::fs::read_to_string(&log).unwrap();
        assert!(text.contains("log file"), "{text}");
        assert!(text.contains("window failed"), "{text}");
        assert!(text.contains("ERROR"), "{text}");
        let _ = std::fs::remove_dir_all(&path);
    }
}
