//! Sets the `engine_native_decode` cfg when the engine should decode video
//! through the platform-native `cutlass_core::VideoDecoder` backend instead of
//! FFmpeg: always on iOS/Android (where FFmpeg's royalty codecs can't ship),
//! and on demand via the `native-decode` feature so the path is testable on a
//! desktop host (e.g. macOS, where the AVFoundation backend is available).

fn main() {
    println!("cargo:rustc-check-cfg=cfg(engine_native_decode)");

    let by_feature = std::env::var_os("CARGO_FEATURE_NATIVE_DECODE").is_some();
    let by_target = matches!(
        std::env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("ios") | Ok("android")
    );
    if by_feature || by_target {
        println!("cargo:rustc-cfg=engine_native_decode");
    }
}
