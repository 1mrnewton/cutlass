#![forbid(unsafe_code)]

use cutlass_decode::DecodeFactory;
use cutlass_encode::EncodeFactory;

/// Platform openers go first. A codec they decline falls through to the
/// software opener registered after them.
fn register_codecs(decode: &mut DecodeFactory, encode: &mut EncodeFactory) {
    #[cfg(target_os = "macos")]
    {
        cutlass_platform_apple::register_decoders(decode);
        cutlass_platform_apple::register_encoders(encode);
    }
    cutlass_png::register_decoders(decode);
    cutlass_png::register_encoders(encode);
    cutlass_aac::register_decoders(decode);
    cutlass_aac::register_encoders(encode);
}

#[allow(clippy::print_stdout)]
fn main() {
    let mut decode = DecodeFactory::new();
    let mut encode = EncodeFactory::new();
    register_codecs(&mut decode, &mut encode);
    println!("cutlass");
}

#[cfg(test)]
mod tests {
    use cutlass_decode::{CodecId, CodecParams, ColorPrimaries, PixelFormat, Transfer};
    use cutlass_encode::EncodeFactory;

    use super::register_codecs;

    fn picture(codec: CodecId, width: u32, height: u32) -> CodecParams {
        CodecParams {
            codec,
            extra: Vec::new(),
            sample_rate: 0,
            layout: None,
            sample_format: None,
            bitrate: 0,
            width,
            height,
            pixel: PixelFormat::Rgba8,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Unspecified,
        }
    }

    #[test]
    fn the_desktop_opens_png_after_the_platform_codecs() {
        let mut decode = cutlass_decode::DecodeFactory::new();
        let mut encode = EncodeFactory::new();
        register_codecs(&mut decode, &mut encode);
        encode
            .open(&picture(CodecId::Png, 0, 0))
            .expect("png encoder");
        #[cfg(target_os = "macos")]
        encode
            .open(&picture(CodecId::H264, 64, 64))
            .expect("h264 encoder");
    }
}
