//! Video decode: CPU RGBA (`VideoDecoder`) and on macOS zero-copy IOSurface (`VideoDecoderGpu`).
use ffmpeg::decoder::Decoder;
use ffmpeg_next as ffmpeg;

mod cpu;

#[cfg(target_os = "macos")]
pub mod gpu_iosurface;

pub use cpu::{DecodedFrame, VideoDecoder};

#[cfg(target_os = "macos")]
pub use gpu_iosurface::{GpuFrame, VideoDecoderGpu};

#[cfg(target_os = "macos")]
pub(crate) fn attach_videotoolbox(decoder: &mut Decoder) -> bool {
    use ffmpeg::ffi::*;
    use std::ptr;

    unsafe {
        let mut device: *mut AVBufferRef = ptr::null_mut();
        let rc = av_hwdevice_ctx_create(
            &mut device,
            AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            ptr::null(),
            ptr::null_mut(),
            0,
        );
        if rc < 0 {
            if !device.is_null() {
                av_buffer_unref(&mut device);
            }
            return false;
        }

        let avctx = decoder.as_mut_ptr();
        let codec_ref = av_buffer_ref(device);
        if codec_ref.is_null() {
            av_buffer_unref(&mut device);
            return false;
        }
        (*avctx).hw_device_ctx = codec_ref;
        av_buffer_unref(&mut device);
        true
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn attach_videotoolbox(_decoder: &mut Decoder) -> bool {
    false
}
