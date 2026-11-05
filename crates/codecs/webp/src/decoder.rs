//! Decoder trait for WebP packets.

use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, Decoder, Output, Packet, StillImage,
    StillOutput,
};

use crate::animation::{Canvas, Decoded};
use crate::limits::{MAX_OUTPUT_PIXELS, picture_ok};
use crate::picture::{self, Picture};
use crate::riff::{self, AnimationHeader, Frame, Layout, Still};

/// Milliseconds: the unit of `ANMF` durations.
const ANIMATION_TIMESCALE: u32 = 1000;

/// The duration a player gives a frame stored with 0, as the `still`
/// container does.
const DEFAULT_FRAME_MS: u32 = 100;

/// Software WebP decoder: lossless, lossy, alpha, and animation.
///
/// One packet is what the `still` container hands over: the file for a
/// still, or one `ANMF` frame of an animation (the first packet also holds
/// `VP8X` and `ANIM`). A still decodes to one picture. An animation frame
/// decodes to the whole canvas after that frame is composed.
pub struct WebpDecoder {
    /// The canvas of the animation being played, from its first packet.
    animation: Option<Animation>,
}

struct Animation {
    canvas: Canvas,
    icc: Vec<u8>,
}

impl WebpDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Webp {
            return Err(CodecError::Unsupported { what: "webp" });
        }
        Ok(Self { animation: None })
    }
}

impl Decoder for WebpDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Webp
    }

    fn push(&mut self, _packet: &Packet) -> Result<Vec<Output>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "webp produces a picture",
        })
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.animation = None;
    }

    fn decode_picture(&mut self, packet: &Packet) -> Result<Vec<StillOutput>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        match riff::parse(&packet.data)? {
            Layout::Still(still) => Ok(vec![StillOutput {
                image: decode_still(&still)?,
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
            }]),
            Layout::Animation { header, frames } => self.compose(packet, header, &frames),
        }
    }
}

impl WebpDecoder {
    /// Compose the frames of one packet. A packet that starts the file
    /// starts a fresh canvas. Nothing changes unless every frame decodes.
    fn compose(
        &mut self,
        packet: &Packet,
        header: Option<AnimationHeader<'_>>,
        frames: &[Frame<'_>],
    ) -> Result<Vec<StillOutput>, CodecError> {
        let starts_file = header.is_some();
        let mut animation = match header {
            Some(header) => Animation {
                canvas: Canvas::new(header.width, header.height)?,
                icc: header.icc.map(<[u8]>::to_vec).unwrap_or_default(),
            },
            None => self.animation.take().ok_or(CodecError::InvalidApi {
                what: "webp frame before the packet that starts the file",
            })?,
        };
        let decoded = decode_frames(&animation.canvas, frames);
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                if !starts_file {
                    self.animation = Some(animation);
                }
                return Err(error);
            }
        };
        let (width, height) = (animation.canvas.width(), animation.canvas.height());
        // One frame keeps the container's clock; a packet holding several
        // frames times them from their stored milliseconds.
        let single = frames.len() == 1;
        let mut clock = if packet.timescale == ANIMATION_TIMESCALE {
            packet.pts
        } else {
            0
        };
        let mut outputs = Vec::with_capacity(frames.len());
        for (frame, decoded) in frames.iter().zip(&decoded) {
            let rgba = animation.canvas.compose(decoded);
            let mut image = StillImage::from_samples(
                width,
                height,
                ColorType::Rgba,
                BitDepth::Eight,
                rgba.into_iter().map(u16::from).collect(),
            );
            image.icc_profile.clone_from(&animation.icc);
            let duration = i64::from(if frame.duration == 0 {
                DEFAULT_FRAME_MS
            } else {
                frame.duration
            });
            outputs.push(if single {
                StillOutput {
                    image,
                    pts: packet.pts,
                    duration: packet.duration,
                    timescale: packet.timescale,
                }
            } else {
                StillOutput {
                    image,
                    pts: clock,
                    duration,
                    timescale: ANIMATION_TIMESCALE,
                }
            });
            clock += duration;
        }
        self.animation = Some(animation);
        Ok(outputs)
    }
}

/// Decode every frame of a packet before composing any.
fn decode_frames(canvas: &Canvas, frames: &[Frame<'_>]) -> Result<Vec<Decoded>, CodecError> {
    let pixels = u64::from(canvas.width()) * u64::from(canvas.height());
    if pixels * frames.len() as u64 > MAX_OUTPUT_PIXELS {
        return Err(CodecError::Limit {
            what: "webp frames in one packet",
        });
    }
    frames.iter().map(|frame| canvas.decode(frame)).collect()
}

fn decode_still(still: &Still<'_>) -> Result<StillImage, CodecError> {
    let decoded = picture::decode(still.bitstream)?;
    if let Some((width, height)) = still.canvas {
        picture_ok(width, height)?;
        if (width, height) != (decoded.width, decoded.height) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "webp canvas does not match the bitstream",
            });
        }
    }
    let mut image = still_image(&decoded);
    if let Some(icc) = still.icc {
        image.icc_profile = icc.to_vec();
    }
    Ok(image)
}

/// RGBA, or RGB when the picture carries no alpha, as 8-bit samples.
pub(crate) fn still_image(picture: &Picture) -> StillImage {
    let (color, samples) = if picture.alpha {
        (
            ColorType::Rgba,
            picture.rgba.iter().map(|&byte| u16::from(byte)).collect(),
        )
    } else {
        (
            ColorType::Rgb,
            picture
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                .map(u16::from)
                .collect(),
        )
    };
    StillImage::from_samples(
        picture.width,
        picture.height,
        color,
        BitDepth::Eight,
        samples,
    )
}
