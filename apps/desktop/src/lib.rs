#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

mod dialogs;
mod face;
pub mod icons;
pub mod probe;
mod readers;

#[cfg(test)]
mod open_movie;
pub mod shell;
pub mod tokens;
pub mod view;
pub mod widgets;
pub mod window;

/// Platform openers go first. A codec they decline falls through to the
/// software opener registered after them.
pub fn register_codecs() -> (cutlass_decode::DecodeFactory, cutlass_encode::EncodeFactory) {
    let mut decoders = cutlass_decode::DecodeFactory::new();
    let mut encoders = cutlass_encode::EncodeFactory::new();
    #[cfg(target_os = "macos")]
    {
        cutlass_platform_apple::register_decoders(&mut decoders);
        cutlass_platform_apple::register_encoders(&mut encoders);
    }
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

/// Open the process log before the window.
///
/// `None` uses the per-user log directory. A public build writes the file
/// and does not write stderr.
pub fn boot(log_directory: Option<&Path>) -> Result<PathBuf, cutlass_logger::InitError> {
    match log_directory {
        Some(directory) => cutlass_logger::init(directory),
        None => cutlass_logger::init_default(),
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
