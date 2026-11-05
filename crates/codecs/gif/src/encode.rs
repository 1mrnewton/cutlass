//! The GIF encoder: one image per frame, the last held back for the trailer.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, CodecError, CodecId, CodecParams, Encoder, FrameInput, Packet, PixelFormat,
};

use crate::GIF_TICKS_PER_SECOND;
use crate::lzw::image_data;
use crate::quant::quantize;

/// The shortest delay most players honor, in hundredths of a second.
const SHORTEST_DELAY: u16 = 2;

struct Held {
    image: Vec<u8>,
    pts: i64,
    duration: i64,
    timescale: u32,
}

pub struct GifEncoder {
    /// Width and height of the logical screen, set by the first frame.
    screen: Option<(u16, u16)>,
    /// Whether the header has gone out.
    started: bool,
    held: Option<Held>,
    cancel: Arc<AtomicBool>,
}

impl GifEncoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Gif {
            return Err(CodecError::Unsupported { what: "gif" });
        }
        Ok(Self {
            screen: None,
            started: false,
            held: None,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The held frame as a packet, with the header in front when it is the
    /// first and the trailer behind when it is the last.
    fn release(&mut self, last: bool) -> Option<Packet> {
        let held = self.held.take()?;
        let mut data = Vec::with_capacity(held.image.len() + 14);
        if !self.started {
            let (width, height) = self.screen.unwrap_or((1, 1));
            data.extend_from_slice(b"GIF89a");
            data.extend_from_slice(&width.to_le_bytes());
            data.extend_from_slice(&height.to_le_bytes());
            // No global color table; each image carries its own.
            data.extend_from_slice(&[0, 0, 0]);
            self.started = true;
        }
        data.extend_from_slice(&held.image);
        if last {
            data.push(0x3B);
        }
        Some(Packet::video(
            data,
            held.pts,
            held.duration,
            held.timescale,
            true,
        ))
    }
}

/// The delay of a frame in hundredths of a second.
fn delay(duration: i64, timescale: u32) -> u16 {
    if timescale == 0 || duration <= 0 {
        return SHORTEST_DELAY;
    }
    let ticks = duration
        .saturating_mul(GIF_TICKS_PER_SECOND)
        .saturating_add(i64::from(timescale) / 2)
        / i64::from(timescale);
    u16::try_from(ticks).unwrap_or(u16::MAX).max(SHORTEST_DELAY)
}

/// The graphic control extension, the image descriptor with its local color
/// table, and the image data for one frame.
fn image(rgba: &[u8], width: u16, height: u16, delay: u16) -> Vec<u8> {
    let indexed = quantize(rgba);
    let mut out = Vec::with_capacity(rgba.len() / 2 + 800);
    out.extend_from_slice(&[0x21, 0xF9, 4, 0]);
    out.extend_from_slice(&delay.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out.push(0x2C);
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.push(0x80 | (indexed.bits - 1));
    for color in &indexed.palette {
        out.extend_from_slice(color);
    }
    out.extend(image_data(&indexed.indexes, indexed.bits));
    out
}

impl Encoder for GifEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Gif
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "gif encodes pictures",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform picture",
            });
        };
        if frame.pixel() != PixelFormat::Rgba8 {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        let (Ok(width), Ok(height)) = (u16::try_from(frame.width()), u16::try_from(frame.height()))
        else {
            return Err(CodecError::Unsupported { what: "gif size" });
        };
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidApi { what: "gif size" });
        }
        match self.screen {
            None => self.screen = Some((width, height)),
            Some(screen) if screen != (width, height) => {
                return Err(CodecError::InvalidApi {
                    what: "gif frame size changed",
                });
            }
            Some(_) => {}
        }
        let rows = usize::from(width) * 4;
        let bytes = frame.bytes();
        let needed = rows * usize::from(height);
        if bytes.len() < needed {
            return Err(CodecError::InvalidApi {
                what: "short frame",
            });
        }
        let time = frame.time();
        let next = Held {
            image: image(
                &bytes[..needed],
                width,
                height,
                delay(time.duration(), time.timescale()),
            ),
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
        };
        let out = self.release(false).into_iter().collect();
        self.held = Some(next);
        Ok(out)
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(self.release(true).into_iter().collect())
    }

    fn reset(&mut self) {
        self.screen = None;
        self.started = false;
        self.held = None;
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_encode::{CpuFrame, FrameTime};

    use super::*;
    use crate::lzw::tests::expand;

    fn frame(color: [u8; 3], pts: i64) -> FrameInput {
        let mut bytes = Vec::new();
        for y in 0..4u8 {
            for x in 0..6u8 {
                let pixel = if (x + y) % 2 == 0 { color } else { [9, 9, 9] };
                bytes.extend(pixel);
                bytes.push(255);
            }
        }
        let time = FrameTime::new(pts, 1, 10).unwrap();
        FrameInput::Cpu(CpuFrame::new(6, 4, PixelFormat::Rgba8, bytes, time).unwrap())
    }

    /// The pixels and delay of each image in a GIF, read back.
    fn read(file: &[u8]) -> Vec<(u16, Vec<[u8; 3]>)> {
        assert_eq!(&file[..6], b"GIF89a");
        assert_eq!(file.last(), Some(&0x3B));
        let mut at = 13;
        let mut images = Vec::new();
        while file[at] != 0x3B {
            assert_eq!(&file[at..at + 4], &[0x21, 0xF9, 4, 0]);
            let delay = u16::from_le_bytes([file[at + 4], file[at + 5]]);
            at += 8;
            assert_eq!(file[at], 0x2C);
            let packed = file[at + 9];
            at += 10;
            let colors = 1usize << ((packed & 7) + 1);
            let palette: Vec<[u8; 3]> = file[at..at + colors * 3]
                .chunks(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            at += colors * 3;
            let start = at;
            at += 1;
            while file[at] != 0 {
                at += 1 + usize::from(file[at]);
            }
            at += 1;
            let indexes = expand(&file[start..at]);
            images.push((
                delay,
                indexes.iter().map(|i| palette[usize::from(*i)]).collect(),
            ));
        }
        images
    }

    #[test]
    fn frames_join_into_one_gif_with_the_trailer_on_the_last() {
        let mut encoder = GifEncoder::open(&CodecParams::video(CodecId::Gif, 6, 4)).unwrap();
        let mut packets = encoder.push_frame(&frame([200, 10, 10], 0)).unwrap();
        assert!(packets.is_empty(), "the first frame is held back");
        packets.extend(encoder.push_frame(&frame([10, 200, 10], 1)).unwrap());
        packets.extend(encoder.drain().unwrap());
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[1].pts, 1);
        let file: Vec<u8> = packets.iter().flat_map(|p| p.data.clone()).collect();
        let images = read(&file);
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].0, 10);
        assert_eq!(images[0].1[0], [200, 10, 10]);
        assert_eq!(images[0].1[1], [9, 9, 9]);
        assert_eq!(images[1].1[0], [10, 200, 10]);
        assert!(encoder.drain().unwrap().is_empty());
    }

    #[test]
    fn a_short_delay_is_raised_and_a_new_size_refused() {
        assert_eq!(delay(1, 1000), SHORTEST_DELAY);
        assert_eq!(delay(1, 15), 7);
        let mut encoder = GifEncoder::open(&CodecParams::video(CodecId::Gif, 6, 4)).unwrap();
        encoder.push_frame(&frame([1, 2, 3], 0)).unwrap();
        let time = FrameTime::new(1, 1, 10).unwrap();
        let other = CpuFrame::new(2, 2, PixelFormat::Rgba8, vec![0; 16], time).unwrap();
        assert!(encoder.push_frame(&FrameInput::Cpu(other)).is_err());
    }
}
