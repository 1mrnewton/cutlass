//! A [`VideoEncoder`] behind the [`Encoder`] trait export drives.
//!
//! The Rust video encoders take 4:2:0 pictures and report their decoder
//! configuration once. Export hands every encoder RGBA through
//! [`Encoder::push_frame`] and reads the configuration from the first
//! packet's [`SideData::Config`], as the platform encoders write it. This
//! turns one into the other, so a host with no platform encoder, such as
//! Linux, still exports.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use cutlass_codec::CodecId;
use cutlass_codec::access::{CodecError, FrameError, Packet, PixelFormat, SideData, Yuv420p};

use crate::{AudioInput, CpuFrame, Encoder, FrameInput, VideoEncoder, VideoInput};

pub(crate) struct Planar {
    inner: Box<dyn VideoEncoder>,
    /// The configuration rode on a packet since the open or the last reset.
    configured: bool,
}

impl Planar {
    pub(crate) fn new(inner: Box<dyn VideoEncoder>) -> Self {
        Self {
            inner,
            configured: false,
        }
    }

    /// Put the configuration on the first packet out.
    fn configure(&mut self, mut packets: Vec<Packet>) -> Vec<Packet> {
        if !self.configured
            && let Some(first) = packets.first_mut()
        {
            self.configured = true;
            let extra = self.inner.extradata();
            if !extra.is_empty() {
                first.side.push(SideData::Config(extra.to_vec()));
            }
        }
        packets
    }
}

impl Encoder for Planar {
    fn codec(&self) -> CodecId {
        self.inner.codec()
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "audio input",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frame",
            });
        };
        let picture = yuv420(frame)?;
        let time = frame.time();
        let packets = self.inner.push(&VideoInput {
            picture,
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
            force_keyframe: false,
        })?;
        Ok(self.configure(packets))
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        let packets = self.inner.drain()?;
        Ok(self.configure(packets))
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.configured = false;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.inner.cancel_flag()
    }
}

/// `frame` as BT.709 limited-range 4:2:0, the signal an HD H.264 stream
/// with no colour description is read as. Each chroma sample is the mean
/// of its 2×2 block. Alpha is dropped.
fn yuv420(frame: &CpuFrame) -> Result<Yuv420p, CodecError> {
    let (width, height) = (frame.width(), frame.height());
    match frame.pixel() {
        PixelFormat::Rgba8 => {}
        PixelFormat::Yuv420p => {
            let luma = width as usize * height as usize;
            let bytes = frame.bytes();
            let (y, chroma) = bytes.split_at(luma);
            let (u, v) = chroma.split_at(chroma.len() / 2);
            return Yuv420p::try_new(width, height, y.to_vec(), u.to_vec(), v.to_vec())
                .map_err(frame_error);
        }
        _ => {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
    }
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(CodecError::Unsupported {
            what: "chroma format",
        });
    }
    let (w, h) = (width as usize, height as usize);
    let rgba = frame.bytes();
    let mut y = vec![0_u8; w * h];
    let mut u = vec![0_u8; w * h / 4];
    let mut v = vec![0_u8; w * h / 4];
    for row in (0..h).step_by(2) {
        for col in (0..w).step_by(2) {
            let mut sum = [0_i32; 3];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let at = ((row + dy) * w + col + dx) * 4;
                let rgb = [
                    i32::from(rgba[at]),
                    i32::from(rgba[at + 1]),
                    i32::from(rgba[at + 2]),
                ];
                y[(row + dy) * w + col + dx] = luma(rgb);
                for (total, value) in sum.iter_mut().zip(rgb) {
                    *total += value;
                }
            }
            let mean = sum.map(|total| (total + 2) / 4);
            let (cb, cr) = chroma(mean);
            let at = (row / 2) * (w / 2) + col / 2;
            u[at] = cb;
            v[at] = cr;
        }
    }
    Yuv420p::try_new(width, height, y, u, v).map_err(frame_error)
}

/// BT.709 luma in 16..=235, in 8.8 fixed point.
fn luma([r, g, b]: [i32; 3]) -> u8 {
    clamp(16 + ((47 * r + 157 * g + 16 * b + 128) >> 8))
}

/// BT.709 Cb and Cr in 16..=240, in 8.8 fixed point.
fn chroma([r, g, b]: [i32; 3]) -> (u8, u8) {
    let cb = 128 + ((-26 * r - 87 * g + 113 * b + 128) >> 8);
    let cr = 128 + ((112 * r - 102 * g - 10 * b + 128) >> 8);
    (clamp(cb), clamp(cr))
}

fn clamp(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
}

fn frame_error(error: FrameError) -> CodecError {
    match error {
        FrameError::Limit => CodecError::Limit {
            what: "picture size",
        },
        _ => CodecError::InvalidApi {
            what: "picture size",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{chroma, luma, yuv420};
    use crate::{CpuFrame, FrameTime, PixelFormat};

    fn frame(width: u32, height: u32, rgba: [u8; 4]) -> CpuFrame {
        let bytes = rgba.repeat(width as usize * height as usize);
        let time = FrameTime::new(0, 1, 30).unwrap();
        CpuFrame::new(width, height, PixelFormat::Rgba8, bytes, time).unwrap()
    }

    #[test]
    fn black_white_and_grey_have_no_colour() {
        assert_eq!(luma([0, 0, 0]), 16);
        assert_eq!(luma([255, 255, 255]), 235);
        assert_eq!(chroma([0, 0, 0]), (128, 128));
        assert_eq!(chroma([255, 255, 255]), (128, 128));
        assert_eq!(chroma([128, 128, 128]), (128, 128));
    }

    #[test]
    fn primaries_land_on_the_bt709_values() {
        // Reference limited-range BT.709: red 63/102/240, green 173/42/26,
        // blue 32/240/118. Fixed point is within one step.
        let near = |got: u8, want: u8| got.abs_diff(want) <= 1;
        let red = (luma([255, 0, 0]), chroma([255, 0, 0]));
        assert!(near(red.0, 63) && near(red.1.0, 102) && near(red.1.1, 240));
        let green = (luma([0, 255, 0]), chroma([0, 255, 0]));
        assert!(near(green.0, 173) && near(green.1.0, 42) && near(green.1.1, 26));
        let blue = (luma([0, 0, 255]), chroma([0, 0, 255]));
        assert!(near(blue.0, 32) && near(blue.1.0, 240) && near(blue.1.1, 118));
    }

    #[test]
    fn a_frame_becomes_planes_and_an_odd_edge_is_refused() {
        let planes = yuv420(&frame(4, 2, [255, 255, 255, 0])).unwrap();
        assert_eq!(planes.y(), [235; 8]);
        assert_eq!(planes.u(), [128; 2]);
        assert!(yuv420(&frame(3, 2, [0, 0, 0, 255])).is_err());
    }
}
