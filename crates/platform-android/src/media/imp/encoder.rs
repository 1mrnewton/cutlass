//! MediaCodec H.264 and HEVC encode.
//!
//! The platform's pick goes first, which is the phone's hardware encoder,
//! then Android's software encoders by name. Pictures go in through the
//! encoder's input surface as RGBA rows, the way a camera or the screen
//! recorder feeds it; the encoder converts them to its own 4:2:0. A
//! ByteBuffer input was tried first: on the Android emulator's Codec2
//! encoders it encodes whatever the graphics buffer held before, not the
//! bytes written to it, so it could not be checked.
//!
//! A surface buffer's time is the wall clock when it is posted, and the
//! NDK has no call to set it. Rate control is asked for the frame rate of
//! the first picture's duration, and B pictures are asked off, so units
//! leave in input order and each takes the caller's time of the picture at
//! the head of the queue. A keyframe is asked every second, as on Apple and
//! Windows.
//!
//! The encoder turns the RGB into YUV with a matrix of its own choosing.
//! The emulator's software encoders use BT.601 whatever the format's
//! `color-standard` or the surface's dataspace says, and write no matrix in
//! the stream, so a player guesses; on saturated colors that guess can be
//! off by the difference between BT.601 and BT.709.
//!
//! The encoder writes Annex B with the parameter sets in a codec-config
//! buffer. Each unit leaves as 4-byte length-prefixed NAL units without
//! those sets, and the first keyframe carries the `avcC` or `hvcC` as
//! [`SideData::Config`]. After `drain` the input surface has ended, so the
//! next picture builds a new encoder, whose first unit is a keyframe with
//! the record again.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use cutlass_color::nv12::{self, Layout};
use cutlass_decode::nal;
use cutlass_encode::{
    AudioInput, CodecError, CodecId, CodecParams, Encoder, FrameInput, FrameTime, Packet,
    PictureInput, PixelFormat, SideData,
};

use super::ndk::{
    Codec, Dequeued, FLAG_CODEC_CONFIG, FLAG_END_OF_STREAM, FLAG_KEY_FRAME, Format, Window,
};
use crate::media::kind::{Kind, is_software};
use crate::planes::MAX_EDGE;

/// Longest a drain waits for the last unit.
const DRAIN_WAIT: Duration = Duration::from_secs(20);

/// Longest a push waits for the encoder to catch up.
const PUSH_WAIT: Duration = Duration::from_secs(10);

const POLL_US: i64 = 2_000;

/// Pictures posted and not yet out as units. The surface has few buffers,
/// and locking one when none is free would wait on the encoder forever,
/// so a push waits for units first.
const IN_FLIGHT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    Platform,
    Named(&'static str),
}

/// A configured, started encoder and the surface it reads.
struct Running {
    // Field order is drop order: the surface goes before the codec.
    window: Window,
    codec: Codec,
    name: String,
}

pub struct VideoEncoder {
    kind: Kind,
    width: u32,
    height: u32,
    bitrate: u32,
    rate: (u32, u32),
    hardware: bool,
    running: Option<Running>,
    name: String,
    /// A unit has left this encoder since it was built.
    produced: bool,
    attached: bool,
    /// Parameter sets from the codec-config buffer.
    header: Vec<Vec<u8>>,
    /// Times of the pictures posted and not yet out, oldest first.
    waiting: VecDeque<FrameTime>,
    cancel: Arc<AtomicBool>,
}

pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(open_encoder_on(params, true)?))
}

/// The encoder, the platform's pick first when `hardware` is set.
pub fn open_encoder_on(params: &CodecParams, hardware: bool) -> Result<VideoEncoder, CodecError> {
    let kind = Kind::of(params.codec)?;
    if kind.software_encoders().is_empty() {
        return Err(CodecError::Unsupported {
            what: "mediacodec encoder",
        });
    }
    let (width, height) = (params.width, params.height);
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    if width > MAX_EDGE || height > MAX_EDGE || width % 2 != 0 || height % 2 != 0 {
        return Err(CodecError::Unsupported {
            what: "picture size",
        });
    }
    if !matches!(
        params.pixel,
        PixelFormat::None | PixelFormat::Rgba8 | PixelFormat::Yuv420p
    ) {
        return Err(CodecError::Unsupported {
            what: "pixel format",
        });
    }
    let bitrate = target_bitrate(params);
    let rate = (30, 1);
    let running = build(kind, width, height, bitrate, rate, hardware)?;
    Ok(VideoEncoder {
        kind,
        width,
        height,
        bitrate,
        rate,
        hardware,
        name: running.name.clone(),
        running: Some(running),
        produced: false,
        attached: false,
        header: Vec::new(),
        waiting: VecDeque::new(),
        cancel: Arc::new(AtomicBool::new(false)),
    })
}

impl VideoEncoder {
    /// The MediaCodec name encoding, such as `c2.android.avc.encoder`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether a vendor encoder is running rather than Android's software one.
    pub fn hardware(&self) -> bool {
        !is_software(&self.name)
    }
}

impl Encoder for VideoEncoder {
    fn codec(&self) -> CodecId {
        self.kind.codec()
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "audio input",
        })
    }

    fn push_picture(&mut self, _input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frame",
            });
        };
        if frame.width() != self.width || frame.height() != self.height {
            return Err(CodecError::Unsupported { what: "frame" });
        }
        let converted;
        let rgba = match frame.pixel() {
            PixelFormat::Rgba8 => frame.bytes(),
            PixelFormat::Yuv420p => {
                let packed = nv12::from_planar(self.width, self.height, frame.bytes())?;
                converted = nv12::to_rgba(Layout::packed(self.width, self.height), &packed)?;
                &converted
            }
            _ => {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
        };
        let time = frame.time();
        if self.waiting.is_empty() && !self.produced {
            self.start_at(time)?;
        }
        let mut out = Vec::new();
        let start = Instant::now();
        while self.waiting.len() >= IN_FLIGHT {
            if !self.take(POLL_US, &mut out)? && start.elapsed() > PUSH_WAIT {
                return Err(stalled());
            }
        }
        let row = self.width as usize * 4;
        let rows = self.height as usize;
        let running = self.running.as_mut().ok_or_else(refused)?;
        running
            .window
            .post(|buffer, step| {
                for (line, source) in rgba.chunks_exact(row).take(rows).enumerate() {
                    let at = line * step;
                    if let Some(target) = buffer.get_mut(at..at + row) {
                        target.copy_from_slice(source);
                    }
                }
            })
            .map_err(|_| refused())?;
        self.waiting.push_back(time);
        while self.take(0, &mut out)? {}
        Ok(out)
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.waiting.is_empty() && !self.produced {
            return Ok(Vec::new());
        }
        let Some(running) = self.running.as_mut() else {
            return Ok(Vec::new());
        };
        running.codec.end_input().map_err(|_| refused())?;
        let mut out = Vec::new();
        let start = Instant::now();
        loop {
            match self.next_output(POLL_US * 5, &mut out)? {
                Taken::Ended => break,
                Taken::Some => {}
                Taken::Nothing if start.elapsed() > DRAIN_WAIT => return Err(stalled()),
                Taken::Nothing => {}
            }
        }
        self.end();
        Ok(out)
    }

    /// Drop the encoder, so the next picture is a keyframe that carries the
    /// configuration again.
    fn reset(&mut self) {
        self.end();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

enum Taken {
    Nothing,
    Some,
    Ended,
}

impl VideoEncoder {
    /// A fresh encoder at the rate the first picture's duration names, when
    /// the one built does not run at it. Rate control budgets bits per
    /// picture from it.
    fn start_at(&mut self, time: FrameTime) -> Result<(), CodecError> {
        let rate = frame_rate(time);
        if self.running.is_some() && rate == self.rate {
            return Ok(());
        }
        self.running = None;
        let running = build(
            self.kind,
            self.width,
            self.height,
            self.bitrate,
            rate,
            self.hardware,
        )?;
        self.name.clone_from(&running.name);
        self.running = Some(running);
        self.rate = rate;
        Ok(())
    }

    /// The input surface has ended or the stream starts over.
    fn end(&mut self) {
        self.running = None;
        self.waiting.clear();
        self.produced = false;
        self.attached = false;
        self.header.clear();
    }

    fn take(&mut self, timeout_us: i64, out: &mut Vec<Packet>) -> Result<bool, CodecError> {
        Ok(!matches!(
            self.next_output(timeout_us, out)?,
            Taken::Nothing
        ))
    }

    fn next_output(&mut self, timeout_us: i64, out: &mut Vec<Packet>) -> Result<Taken, CodecError> {
        let Some(running) = self.running.as_mut() else {
            return Ok(Taken::Ended);
        };
        match running.codec.dequeue_output(timeout_us) {
            Dequeued::TryAgain => Ok(Taken::Nothing),
            Dequeued::FormatChanged => Ok(Taken::Some),
            Dequeued::Failed(_) => Err(refused()),
            Dequeued::Buffer { index, info } => {
                let bytes = running
                    .codec
                    .with_output(index, info, <[u8]>::to_vec)
                    .ok_or_else(refused);
                running.codec.release_output(index);
                let bytes = bytes?;
                if info.flags & FLAG_CODEC_CONFIG != 0 {
                    self.header = self.sets_in(&bytes);
                } else if !bytes.is_empty()
                    && let Some(packet) = self.packet(&bytes, info.flags & FLAG_KEY_FRAME != 0)?
                {
                    self.produced = true;
                    out.push(packet);
                }
                Ok(if info.flags & FLAG_END_OF_STREAM != 0 {
                    Taken::Ended
                } else {
                    Taken::Some
                })
            }
        }
    }

    fn sets_in(&self, bytes: &[u8]) -> Vec<Vec<u8>> {
        nal::annex_b(bytes)
            .map(|nals| {
                nals.into_iter()
                    .map(|nal| nal.bytes)
                    .filter(|bytes| {
                        self.kind
                            .nal_type(bytes)
                            .is_some_and(|kind| self.kind.is_parameter_set(kind))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// One length-prefixed unit, with the configuration on the first keyframe.
    fn packet(&mut self, bytes: &[u8], flagged_key: bool) -> Result<Option<Packet>, CodecError> {
        let nals = nal::annex_b(bytes)?;
        let mut sets = Vec::new();
        let mut units = Vec::new();
        let mut keyframe = flagged_key;
        for unit in nals {
            match self.kind.nal_type(&unit.bytes) {
                Some(kind) if self.kind.is_parameter_set(kind) => sets.push(unit.bytes),
                Some(kind) if self.kind.is_delimiter(kind) => {}
                kind => {
                    keyframe |= kind.is_some_and(|kind| self.kind.is_keyframe(kind));
                    units.push(unit);
                }
            }
        }
        if units.is_empty() {
            return Ok(None);
        }
        // A unit the encoder made with no picture waiting has no time of
        // its own; it is not a picture of the caller's.
        let Some(time) = self.waiting.pop_front() else {
            return Ok(None);
        };
        let mut packet = Packet {
            stream: 0,
            data: nal::length_prefixed(&units, 4)?,
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
            keyframe,
            side: Vec::new(),
        };
        if keyframe && !self.attached {
            if sets.is_empty() {
                sets.clone_from(&self.header);
            }
            if let Ok(config) = self.kind.write_config(&sets) {
                packet.side.push(SideData::Config(config));
                self.attached = true;
            }
        }
        Ok(Some(packet))
    }
}

/// The first candidate that configures, gives a surface and starts.
fn build(
    kind: Kind,
    width: u32,
    height: u32,
    bitrate: u32,
    rate: (u32, u32),
    hardware: bool,
) -> Result<Running, CodecError> {
    let mut candidates = Vec::new();
    if hardware {
        candidates.push(Candidate::Platform);
    }
    candidates.extend(
        kind.software_encoders()
            .iter()
            .map(|name| Candidate::Named(name)),
    );
    for candidate in candidates {
        let codec = match candidate {
            Candidate::Platform => Codec::by_type(kind.mime(), true),
            Candidate::Named(name) => Codec::by_name(name),
        };
        let Some(mut codec) = codec else {
            continue;
        };
        let Some(format) = Format::new() else {
            continue;
        };
        format.set_str(c"mime", kind.mime());
        format.set_i32(c"width", i32::try_from(width).unwrap_or(0));
        format.set_i32(c"height", i32::try_from(height).unwrap_or(0));
        // `COLOR_FormatSurface`.
        format.set_i32(c"color-format", 0x7F00_0789);
        format.set_i32(c"bitrate", i32::try_from(bitrate).unwrap_or(i32::MAX));
        let fps = (f64::from(rate.0) / f64::from(rate.1.max(1))).round();
        format.set_i32(c"frame-rate", (fps as i32).clamp(1, 240));
        format.set_i32(c"i-frame-interval", 1);
        format.set_i32(c"max-bframes", 0);
        // BT.709, limited range, SDR video: the matrix the RGBA is converted with.
        format.set_i32(c"color-standard", 1);
        format.set_i32(c"color-range", 2);
        format.set_i32(c"color-transfer", 3);
        if codec.configure(&format, true).is_err() {
            continue;
        }
        let Ok(mut window) = codec.input_surface() else {
            continue;
        };
        if window.set_rgba(width, height).is_err() || codec.start().is_err() {
            continue;
        }
        let name = codec.name().unwrap_or_default();
        return Ok(Running {
            window,
            codec,
            name,
        });
    }
    Err(CodecError::Unsupported {
        what: "mediacodec encoder",
    })
}

/// The bitrate asked for, or 8 bits a pixel, as on Apple and Windows.
fn target_bitrate(params: &CodecParams) -> u32 {
    if params.bitrate != 0 {
        return params.bitrate;
    }
    let pixels = u64::from(params.width).saturating_mul(u64::from(params.height));
    u32::try_from(pixels.saturating_mul(8).max(64_000)).unwrap_or(u32::MAX)
}

/// Pictures per second as a reduced fraction, from one picture's duration.
fn frame_rate(time: FrameTime) -> (u32, u32) {
    let numerator = time.timescale();
    let Ok(denominator) = u32::try_from(time.duration()) else {
        return (30, 1);
    };
    if denominator == 0 {
        return (30, 1);
    }
    let divisor = gcd(numerator, denominator);
    (numerator / divisor, denominator / divisor)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn refused() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "mediacodec refused the picture",
    }
}

fn stalled() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "mediacodec stopped taking pictures",
    }
}
