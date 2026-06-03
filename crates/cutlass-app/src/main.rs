use std::path::Path;

use cutlass_decode::{DecodeOptions, Decoder, HwAccel, ffmpeg_version};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

fn setup_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

fn main() {
    setup_tracing();
    info!(version = ffmpeg_version(), "cutlass-app starting");

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/13232364_3840_2160_24fps.mp4".to_string());

    // Default to hardware decode; override with CUTLASS_HWACCEL (e.g. `none`, `vt`).
    let hw = std::env::var("CUTLASS_HWACCEL")
        .map(|v| cutlass_decode::hw_accel_from_env(&v))
        .unwrap_or(HwAccel::Auto);
    let options = DecodeOptions::default().hw_accel(hw);

    let mut decoder = match Decoder::open_with(Path::new(&path), options) {
        Ok(d) => d,
        Err(e) => {
            warn!(path, error = %e, "failed to open video");
            return;
        }
    };

    let info = decoder.info().clone();
    info!(
        path,
        width = info.width,
        height = info.height,
        ?info.pixel_format,
        hw = info.hw_accel.name(),
        "decoder ready"
    );

    // Smoke check: decode the first frame.
    match decoder.next_frame() {
        Ok(Some(frame)) => info!(
            width = frame.width,
            height = frame.height,
            pts = frame.pts_ticks,
            "decoded first frame"
        ),
        Ok(None) => warn!("stream produced no frames"),
        Err(e) => warn!(error = %e, "decode failed"),
    }
}
