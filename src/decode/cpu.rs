use ffmpeg::format::{Pixel, context::Input, input};
use ffmpeg::software::scaling::{context::Context as Scaler, flag::Flags};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg_next as ffmpeg;
use std::path::Path;

use super::attach_videotoolbox;

pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub stride: u32,   // bytes_per_row for wgpu
    pub data: Vec<u8>, // RGBA8
}

pub struct VideoDecoder {
    ictx: Input,
    decoder: ffmpeg::decoder::Video,
    scaler: Option<Scaler>,
    /// Scratch buffer when pulling VT frames down to CPU for swscale.
    vt_scratch: VideoFrame,
    hw_decode: bool,
    stream_idx: usize,
    width: u32,
    height: u32,
}

impl VideoDecoder {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ffmpeg::Error> {
        ffmpeg::init()?;
        let ictx = input(&path)?;
        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let stream_idx = stream.index();

        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;

        let mut dec = ctx.decoder();
        let vt_ok = attach_videotoolbox(&mut dec);
        #[cfg(target_os = "macos")]
        if vt_ok {
            log::debug!("CPU path: decoding with VideoToolbox + CPU transfer");
        } else {
            log::debug!("CPU path: software decode");
        }

        let decoder = dec.video()?;
        let (width, height) = (decoder.width(), decoder.height());

        let (hw_decode, scaler) = if vt_ok {
            (true, None)
        } else {
            (
                false,
                Some(Scaler::get(
                    decoder.format(),
                    width,
                    height,
                    Pixel::RGBA,
                    width,
                    height,
                    Flags::BILINEAR,
                )?),
            )
        };

        Ok(Self {
            ictx,
            decoder,
            scaler,
            vt_scratch: VideoFrame::empty(),
            hw_decode,
            stream_idx,
            width,
            height,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn ensure_scaler(
        &mut self,
        src_format: Pixel,
        src_w: u32,
        src_h: u32,
    ) -> Result<(), ffmpeg::Error> {
        let needs_new = match &self.scaler {
            Some(ctx) => {
                let i = ctx.input();
                i.format != src_format || i.width != src_w || i.height != src_h
            }
            None => true,
        };

        if needs_new {
            self.scaler = Some(Scaler::get(
                src_format,
                src_w,
                src_h,
                Pixel::RGBA,
                self.width,
                self.height,
                Flags::BILINEAR,
            )?);
        }
        Ok(())
    }

    fn frame_to_rgba(&mut self, decoded: &VideoFrame) -> Result<DecodedFrame, ffmpeg::Error> {
        let from_hw = self.hw_decode && decoded.format() == Pixel::VIDEOTOOLBOX;

        if from_hw {
            unsafe {
                ffmpeg::ffi::av_frame_unref(self.vt_scratch.as_mut_ptr());
                let ret = ffmpeg::ffi::av_hwframe_transfer_data(
                    self.vt_scratch.as_mut_ptr(),
                    decoded.as_ptr(),
                    0,
                );
                if ret < 0 {
                    return Err(ffmpeg::Error::from(ret));
                }
            }
        }

        let (src_fmt, src_w, src_h) = if from_hw {
            (
                self.vt_scratch.format(),
                self.vt_scratch.width(),
                self.vt_scratch.height(),
            )
        } else {
            (decoded.format(), decoded.width(), decoded.height())
        };

        self.ensure_scaler(src_fmt, src_w, src_h)?;

        let src_ref: &VideoFrame = if from_hw {
            &self.vt_scratch
        } else {
            decoded
        };

        let mut rgba = VideoFrame::empty();
        self.scaler
            .as_mut()
            .expect("scaler set by ensure_scaler")
            .run(src_ref, &mut rgba)?;

        Ok(DecodedFrame {
            width: self.width,
            height: self.height,
            stride: rgba.stride(0) as u32,
            data: rgba.data(0).to_vec(),
        })
    }

    /// pulls the next decoded frame. None = EOF.
    pub fn decode_one(&mut self) -> Result<Option<DecodedFrame>, ffmpeg::Error> {
        let mut decoded = VideoFrame::empty();

        loop {
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                return Ok(Some(self.frame_to_rgba(&decoded)?));
            }

            match self.ictx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() == self.stream_idx {
                        self.decoder.send_packet(&packet)?;
                    }
                }
                None => {
                    self.decoder.send_eof()?;
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        return Ok(Some(self.frame_to_rgba(&decoded)?));
                    }
                    return Ok(None);
                }
            }
        }
    }
}
