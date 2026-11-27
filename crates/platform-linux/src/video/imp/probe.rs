//! A plain-text account of the hardware video this machine offers, for the
//! `video_probe` example and for bug reports.

use super::{cuda, nvdec, nvenc, va};
use crate::video::backend::{BACKEND_VAR, Backends};

pub fn probe() -> Vec<String> {
    let backends = Backends::from_env();
    let allowed = |on: bool| if on { "allowed" } else { "off" };
    let mut lines = vec![format!(
        "{BACKEND_VAR}: NVIDIA {}, VA-API {}",
        allowed(backends.nvidia),
        allowed(backends.vaapi)
    )];
    match cuda::device() {
        None => lines.push(String::from("NVIDIA: no libcuda.so.1, or no GPU")),
        Some(device) => {
            let version = device.driver_version();
            lines.push(format!(
                "NVIDIA: {} (driver CUDA {}.{})",
                device.name(),
                version / 1000,
                version % 1000 / 10
            ));
            lines.extend(nvdec::report(device));
            lines.extend(nvenc::report());
        }
    }
    lines.extend(va::report());
    lines
}
