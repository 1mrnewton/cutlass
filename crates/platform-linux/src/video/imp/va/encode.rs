//! What the VA-API encoders share: the node, profile and entrypoint that
//! encode a codec, its rate control and packed headers, the context and its
//! surfaces, and one picture at a time from upload to coded bytes.
//!
//! Pictures are encoded as FFmpeg's VA-API encoders encode them without B
//! pictures: an IDR every [`rate::KEYFRAME_INTERVAL`] pictures, and P
//! pictures that reference the picture before, one slice each. The driver
//! puts the parameter sets and slice headers this crate writes into the
//! stream when it takes them as packed headers, and writes its own from
//! the parameter buffers otherwise. Each picture is finished before the
//! next goes in, so nothing is held back and a drain has nothing to give.

use std::ffi::c_int;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_color::nv12;
use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, PixelFormat};
use cutlass_encode::{AudioInput, Encoder, FrameInput, PictureInput};

use super::abi::{self, misc_frame_rate, misc_hrd, misc_rate_control, packed_header};
use super::api::{ConfigAttrib, Id};
use super::bits::{Packed, escape};
use super::display::{self, Node};
use super::h264::BACKEND;
use super::header::{Step, rate16};
use super::surface::{Buffer, Config, Context, Picture, Pool, Surface, Upload, failed, refused};
use crate::video::backend::{note_decline, note_open};
use crate::video::imp::blob::{Base, Bits, Blob, Field};
use crate::video::kind::Kind;
use crate::video::limits::video_size;
use crate::video::packets::Packetizer;
use crate::video::rate::{self, Vui};

/// Reconstructed pictures: the one being encoded, its reference, and one
/// spare.
const RECON_SURFACES: usize = 3;

/// The packed headers the encoders write: parameter sets and slice
/// headers, as FFmpeg asks for them.
const WANTED_PACKED: u32 = abi::VA_ENC_PACKED_HEADER_SEQUENCE | abi::VA_ENC_PACKED_HEADER_SLICE;

/// The attributes [`choose`] reads, in the order [`Choice::of`] takes
/// them.
const ATTRIBUTES: [c_int; 9] = [
    abi::VAConfigAttribRTFormat,
    abi::VAConfigAttribRateControl,
    abi::VAConfigAttribEncPackedHeaders,
    abi::VAConfigAttribEncMaxRefFrames,
    abi::VAConfigAttribMaxPictureWidth,
    abi::VAConfigAttribMaxPictureHeight,
    abi::VAConfigAttribPredictionDirection,
    abi::VAConfigAttribEncHEVCFeatures,
    abi::VAConfigAttribEncHEVCBlockSizes,
];

/// What the opener asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    pub codec: CodecId,
    pub width: u32,
    pub height: u32,
    /// The mean bitrate.
    pub bitrate: u32,
    pub constant: bool,
    pub colour: Option<Vui>,
}

impl Settings {
    /// The settings of `params` for `codec`, or a decline so the next
    /// opener is asked.
    pub fn of(params: &CodecParams, codec: CodecId) -> Result<Self, CodecError> {
        if params.codec != codec {
            return Err(CodecError::Unsupported { what: "vaapi" });
        }
        let (width, height) = video_size(params.width, params.height)?;
        if !matches!(
            params.pixel,
            PixelFormat::None | PixelFormat::Rgba8 | PixelFormat::Yuv420p
        ) {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        if super::api::va().is_none() {
            return Err(CodecError::Unsupported { what: "vaapi" });
        }
        Ok(Self {
            codec,
            width,
            height,
            bitrate: rate::target_bitrate(params),
            constant: params.constant_rate,
            colour: Vui::of(params),
        })
    }
}

/// How the bitrate is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RateControl {
    /// `VA_RC_CBR`, `VA_RC_VBR` or `VA_RC_CQP`.
    pub mode: u32,
    /// The mean bitrate.
    pub target: u32,
}

impl RateControl {
    /// CBR for a constant rate and VBR otherwise, each falling back to
    /// the other and then to constant QP; constant QP when the driver
    /// names no modes. `None` when it offers none of the three.
    pub fn choose(offered: u32, constant: bool, target: u32) -> Option<Self> {
        if offered == abi::VA_ATTRIB_NOT_SUPPORTED {
            return Some(Self {
                mode: abi::VA_RC_CQP,
                target,
            });
        }
        let order = if constant {
            [abi::VA_RC_CBR, abi::VA_RC_VBR, abi::VA_RC_CQP]
        } else {
            [abi::VA_RC_VBR, abi::VA_RC_CBR, abi::VA_RC_CQP]
        };
        order
            .into_iter()
            .find(|mode| offered & mode != 0)
            .map(|mode| Self { mode, target })
    }

    /// The rate in the sequence parameters: the mean, none for constant
    /// QP. A driver that reads it after the rate control parameters then
    /// holds the mean.
    pub fn sequence_bits(self) -> u32 {
        if self.mode == abi::VA_RC_CQP {
            0
        } else {
            self.target
        }
    }

    /// The rate the rate control and HRD parameters name, the peak under
    /// VBR, and the percentage of it to aim for.
    fn misc_bits(self) -> (u32, u32) {
        if self.mode == abi::VA_RC_VBR {
            let peak = rate::peak_bitrate(self.target).max(1);
            let share = u64::from(self.target) * 100 / u64::from(peak);
            (peak, u32::try_from(share.clamp(1, 100)).unwrap_or(100))
        } else {
            (self.target, 100)
        }
    }
}

/// A node, profile and entrypoint that encode the codec, and what the
/// driver says about them.
#[derive(Clone, Copy)]
pub struct Choice {
    pub node: &'static Node,
    pub profile: i32,
    pub entrypoint: i32,
    pub rc: RateControl,
    /// Whether the driver names its rate control modes, so the config asks
    /// for one.
    rc_attribute: bool,
    /// The packed headers the driver takes, of [`WANTED_PACKED`].
    packed: u32,
    /// The driver takes no reference pictures, so every picture is an
    /// IDR.
    pub intra_only: bool,
    /// The driver takes no P slices, so P pictures go as B slices with the
    /// same reference in both lists.
    pub gpb: bool,
    /// `VAConfigAttribEncHEVCFeatures`, or `VA_ATTRIB_NOT_SUPPORTED`.
    pub features: u32,
    /// `VAConfigAttribEncHEVCBlockSizes`, or `VA_ATTRIB_NOT_SUPPORTED`.
    pub block_sizes: u32,
}

/// Whether a render node encodes one of `profiles` here.
#[cfg(test)]
pub fn offered(profiles: &[i32]) -> bool {
    super::api::va().is_some()
        && display::nodes().iter().any(|node| {
            profiles.iter().any(|&profile| {
                node.offers(profile, abi::VAEntrypointEncSlice)
                    || node.offers(profile, abi::VAEntrypointEncSliceLP)
            })
        })
}

/// The first node to offer one of `profiles`, in their order, at the full
/// or the low-power entrypoint, with rate control and the picture size.
pub fn choose(settings: &Settings, profiles: &[i32]) -> Result<Choice, CodecError> {
    let mut why = "no render node encodes it";
    for node in display::nodes() {
        for &profile in profiles {
            for entrypoint in [abi::VAEntrypointEncSlice, abi::VAEntrypointEncSliceLP] {
                if !node.offers(profile, entrypoint) {
                    continue;
                }
                match Choice::of(node, profile, entrypoint, settings) {
                    Ok(choice) => return Ok(choice),
                    Err(reason) => why = reason,
                }
            }
        }
    }
    note_decline(BACKEND, settings.codec, true, why);
    Err(CodecError::Unsupported { what: why })
}

impl Choice {
    fn of(
        node: &'static Node,
        profile: i32,
        entrypoint: i32,
        settings: &Settings,
    ) -> Result<Self, &'static str> {
        let values = node.attributes(profile, entrypoint, &ATTRIBUTES);
        let Ok(
            [
                formats,
                modes,
                packed,
                refs,
                max_width,
                max_height,
                direction,
                features,
                blocks,
            ],
        ) = <[u32; 9]>::try_from(values)
        else {
            return Err("VA-API attributes");
        };
        let known = |value: u32| value != abi::VA_ATTRIB_NOT_SUPPORTED;
        if known(formats) && formats & abi::VA_RT_FORMAT_YUV420 == 0 {
            return Err("the GPU does not encode 8-bit 4:2:0");
        }
        if (known(max_width) && settings.width > max_width)
            || (known(max_height) && settings.height > max_height)
        {
            return Err("the picture size is out of the GPU's range");
        }
        let rc = RateControl::choose(modes, settings.constant, settings.bitrate)
            .ok_or("the GPU offers no rate control this encoder uses")?;
        // The low 16 bits are list 0's most, the high list 1's.
        let list0 = if known(refs) { refs & 0xffff } else { 1 };
        Ok(Self {
            node,
            profile,
            entrypoint,
            rc,
            rc_attribute: known(modes),
            packed: if known(packed) {
                packed & WANTED_PACKED
            } else {
                0
            },
            intra_only: list0 == 0,
            gpb: known(direction) && direction & abi::VA_PREDICTION_DIRECTION_BI_NOT_EMPTY != 0,
            features,
            block_sizes: blocks,
        })
    }
}

/// A field of an attribute value such as [`Choice::features`].
pub fn field(value: u32, bits: Bits) -> u32 {
    (value >> bits.shift) & ((1 << bits.width) - 1)
}

/// The surfaces and buffer one picture uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pictures {
    /// Where the reconstructed picture goes.
    pub current: Id,
    /// The picture before, for a P picture.
    pub reference: Option<Id>,
    pub coded: Id,
}

/// One codec's headers and parameter buffers.
pub trait Syntax {
    /// The parameter sets, each a NAL unit before emulation prevention.
    fn sets(&self) -> Vec<Vec<u8>>;

    /// The pictures per second the headers carry, from [`rate16`].
    fn set_rate(&mut self, rate: (u32, u32));

    /// The sequence parameters, sent with every IDR. `period` is the
    /// pictures from one IDR to the next.
    fn sequence(&self, rc: RateControl, period: u32) -> Blob;

    fn picture(&self, step: Step, pictures: Pictures) -> Blob;

    fn slice(&self, step: Step, pictures: Pictures) -> Blob;

    /// The slice NAL unit up to its slice data.
    fn slice_header(&self, step: Step) -> Packed;
}

/// An encode context and the state of the stream it writes. Fields drop
/// in order: the coded buffer and context before the surfaces, and all of
/// them before the config.
pub struct Engine<S> {
    syntax: S,
    coded: Buffer,
    context: Context,
    input: Pool,
    recon: Pool,
    _config: Config,
    upload: Upload,
    /// The reconstructed picture the next P picture references.
    reference: Option<Arc<Surface>>,
    node: &'static Node,
    rc: RateControl,
    packed: u32,
    intra_only: bool,
    /// Pictures per second, from [`rate16`].
    rate: (u32, u32),
    /// The size pictures come in.
    width: u32,
    height: u32,
    /// The next picture is an IDR.
    restart: bool,
    /// Pictures since the last IDR.
    index: u32,
    idr_id: u32,
}

impl<S: Syntax> Engine<S> {
    /// A context for `choice` over surfaces of `size`, for pictures of
    /// `width` by `height`. Every failure declines.
    pub fn open(
        choice: Choice,
        syntax: S,
        size: (u32, u32),
        width: u32,
        height: u32,
    ) -> Result<Self, CodecError> {
        let node = choice.node;
        let mut attributes = vec![ConfigAttrib {
            kind: abi::VAConfigAttribRTFormat,
            value: abi::VA_RT_FORMAT_YUV420,
        }];
        if choice.rc_attribute {
            attributes.push(ConfigAttrib {
                kind: abi::VAConfigAttribRateControl,
                value: choice.rc.mode,
            });
        }
        if choice.packed != 0 {
            attributes.push(ConfigAttrib {
                kind: abi::VAConfigAttribEncPackedHeaders,
                value: choice.packed,
            });
        }
        let config = Config::new(node, choice.profile, choice.entrypoint, &attributes)
            .map_err(|status| refused(node, status, "VA-API could not make an encode config"))?;
        let surfaces = |count| {
            Pool::new(node, size.0, size.1, 8, count)
                .map_err(|status| refused(node, status, "VA-API could not make surfaces"))
        };
        let input = surfaces(1)?;
        let recon = surfaces(RECON_SURFACES)?;
        let context = Context::new(node, &config, size.0, size.1, &recon.ids())
            .map_err(|status| refused(node, status, "VA-API could not make an encode context"))?;
        // Room for a picture past its raw size, as FFmpeg allows.
        let coded_len = size.0 as usize * size.1 as usize * 3 + (1 << 16);
        let coded =
            Buffer::coded(node, context.id, coded_len).map_err(|_| CodecError::Unsupported {
                what: "VA-API could not make a coded buffer",
            })?;
        let mut engine = Self {
            syntax,
            coded,
            context,
            input,
            recon,
            _config: config,
            upload: Upload::new(node),
            reference: None,
            node,
            rc: choice.rc,
            packed: choice.packed,
            intra_only: choice.intra_only,
            rate: rate16(rate::DEFAULT_RATE),
            width,
            height,
            restart: true,
            index: 0,
            idr_id: 0,
        };
        engine.syntax.set_rate(engine.rate);
        Ok(engine)
    }

    pub fn set_rate(&mut self, rate: (u32, u32)) {
        self.rate = rate16(rate);
        self.syntax.set_rate(self.rate);
    }

    /// The next picture starts the stream again with an IDR.
    pub fn restart(&mut self) {
        self.restart = true;
        self.reference = None;
    }

    /// The parameter sets as they go in the stream.
    pub fn sets(&self) -> Vec<Vec<u8>> {
        self.syntax.sets().iter().map(|set| escape(set)).collect()
    }

    /// Encode one picture and return its access unit in Annex B. A
    /// failure starts the stream again at the next picture.
    pub fn encode(&mut self, picture: &Picture<'_>) -> Result<Vec<u8>, CodecError> {
        let idr = self.restart || self.intra_only || self.index >= rate::KEYFRAME_INTERVAL;
        if idr {
            self.index = 0;
            self.reference = None;
        }
        let step = Step {
            idr,
            index: self.index,
            idr_id: self.idr_id,
        };
        match self.encode_step(step, picture) {
            Ok((unit, current)) => {
                self.restart = false;
                self.index += 1;
                if idr {
                    self.idr_id = self.idr_id.wrapping_add(1);
                }
                self.reference = Some(current);
                Ok(unit)
            }
            Err(err) => {
                self.restart();
                Err(err)
            }
        }
    }

    fn encode_step(
        &mut self,
        step: Step,
        picture: &Picture<'_>,
    ) -> Result<(Vec<u8>, Arc<Surface>), CodecError> {
        let busy = CodecError::Malformed {
            at_bit: 0,
            what: "VA-API has no free surface",
        };
        let input = self.input.take().ok_or(busy)?;
        self.upload.write(&input, picture)?;
        let current = self.recon.take().ok_or(busy)?;
        let pictures = Pictures {
            current: current.id,
            reference: self.reference.as_ref().map(|surface| surface.id),
            coded: self.coded.id,
        };
        let (node, context) = (self.node, self.context.id);
        let blob = |kind, blob: &Blob| Buffer::blob(node, context, kind, blob);
        let mut buffers = Vec::new();
        if step.idr {
            let period = if self.intra_only {
                1
            } else {
                rate::KEYFRAME_INTERVAL
            };
            let sequence = self.syntax.sequence(self.rc, period);
            buffers.push(blob(abi::VAEncSequenceParameterBufferType, &sequence)?);
            for misc in self.misc() {
                buffers.push(blob(abi::VAEncMiscParameterBufferType, &misc)?);
            }
            if self.packed & abi::VA_ENC_PACKED_HEADER_SEQUENCE != 0 {
                let sets = Packed::units(&self.syntax.sets());
                buffers.extend(self.packed(abi::VAEncPackedHeaderSequence, &sets)?);
            }
        }
        let params = self.syntax.picture(step, pictures);
        buffers.push(blob(abi::VAEncPictureParameterBufferType, &params)?);
        if self.packed & abi::VA_ENC_PACKED_HEADER_SLICE != 0 {
            let header = self.syntax.slice_header(step);
            buffers.extend(self.packed(abi::VAEncPackedHeaderSlice, &header)?);
        }
        let slice = self.syntax.slice(step, pictures);
        buffers.push(blob(abi::VAEncSliceParameterBufferType, &slice)?);
        self.context.render(input.id, &buffers, &[])?;
        // SAFETY: the surface was made on this display.
        let status = unsafe { (node.va.sync_surface)(node.display, input.id) };
        if status != abi::VA_STATUS_SUCCESS {
            return Err(failed(node, status, "VA-API could not finish an encode"));
        }
        let unit = self.coded.coded_bytes()?;
        if unit.is_empty() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "VA-API wrote no picture",
            });
        }
        Ok((unit, current))
    }

    /// The rate control, HRD and frame rate parameters, each after the
    /// `VAEncMiscParameterBuffer` type.
    fn misc(&self) -> Vec<Blob> {
        let header = abi::misc::SIZE;
        let base = Base(header);
        let make = |kind: i32, size: usize| {
            let mut blob = Blob::new(header + size);
            blob.set(abi::misc::TYPE, kind as u32);
            blob
        };
        let mut out = Vec::new();
        if self.rc.mode != abi::VA_RC_CQP {
            let (bits, percentage) = self.rc.misc_bits();
            let mut control = make(
                abi::VAEncMiscParameterTypeRateControl,
                misc_rate_control::SIZE,
            );
            control.set(misc_rate_control::BITS_PER_SECOND.of(base), bits);
            control.set(misc_rate_control::TARGET_PERCENTAGE.of(base), percentage);
            // A second, as FFmpeg's buffer of one second at that rate gives.
            control.set(misc_rate_control::WINDOW_SIZE.of(base), 1000);
            out.push(control);
            let mut hrd = make(abi::VAEncMiscParameterTypeHRD, misc_hrd::SIZE);
            hrd.set(misc_hrd::BUFFER_SIZE.of(base), bits);
            hrd.set(misc_hrd::INITIAL_BUFFER_FULLNESS.of(base), bits / 4 * 3);
            out.push(hrd);
        }
        let mut frame_rate = make(abi::VAEncMiscParameterTypeFrameRate, misc_frame_rate::SIZE);
        let (numerator, denominator) = self.rate;
        frame_rate.set(
            misc_frame_rate::FRAMERATE.of(base),
            (denominator << 16) | numerator,
        );
        out.push(frame_rate);
        out
    }

    /// A packed header's parameter and data buffers.
    fn packed(&self, kind: i32, header: &Packed) -> Result<[Buffer; 2], CodecError> {
        let (node, context) = (self.node, self.context.id);
        let mut params = Blob::new(packed_header::SIZE);
        params.set(packed_header::TYPE, kind as u32);
        params.set(packed_header::BIT_LENGTH, header.bits);
        params.set(packed_header::HAS_EMULATION_BYTES, 1);
        Ok([
            Buffer::blob(
                node,
                context,
                abi::VAEncPackedHeaderParameterBufferType,
                &params,
            )?,
            Buffer::new(
                node,
                context,
                abi::VAEncPackedHeaderDataBufferType,
                &header.data,
            )?,
        ])
    }
}

/// A VA-API encoder behind [`Encoder`].
pub struct VaEncoder<S> {
    engine: Engine<S>,
    kind: Kind,
    /// A picture went in since the open, a drain or a reset.
    started: bool,
    packets: Packetizer,
    cancel: Arc<AtomicBool>,
}

impl<S: Syntax + Send + 'static> VaEncoder<S> {
    pub fn boxed(engine: Engine<S>, kind: Kind) -> Box<dyn Encoder> {
        let mut packets = Packetizer::new(kind);
        packets.set_header(engine.sets());
        note_open(BACKEND, kind.codec(), true);
        Box::new(Self {
            engine,
            kind,
            started: false,
            packets,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl<S: Syntax + Send> Encoder for VaEncoder<S> {
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
        let (width, height) = (self.engine.width, self.engine.height);
        if frame.width() != width || frame.height() != height {
            return Err(CodecError::Unsupported { what: "frame" });
        }
        let nv12 = match frame.pixel() {
            PixelFormat::Rgba8 => nv12::from_rgba(width, height, frame.bytes())?,
            PixelFormat::Yuv420p => nv12::from_planar(width, height, frame.bytes())?,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
        };
        let time = frame.time();
        if !self.started {
            // The first picture's duration names the rate the headers
            // carry.
            self.engine.set_rate(rate::frame_rate(time));
            self.engine.restart();
            self.packets.set_header(self.engine.sets());
        }
        let unit = self.engine.encode(&Picture {
            nv12: &nv12,
            width,
            height,
        })?;
        self.started = true;
        self.packets.sent(time);
        Ok(self.packets.packet(&unit)?.into_iter().collect())
    }

    /// Every picture was finished as it went in. The next one starts the
    /// stream again with an IDR that carries the configuration.
    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        self.started = false;
        self.packets.reset();
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.started = false;
        self.engine.restart();
        self.packets.reset();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_control_follows_the_offered_modes() {
        let all = abi::VA_RC_CBR | abi::VA_RC_VBR | abi::VA_RC_CQP;
        let mode = |offered, constant| RateControl::choose(offered, constant, 1).map(|rc| rc.mode);
        assert_eq!(mode(all, false), Some(abi::VA_RC_VBR));
        assert_eq!(mode(all, true), Some(abi::VA_RC_CBR));
        assert_eq!(mode(abi::VA_RC_CBR, false), Some(abi::VA_RC_CBR));
        assert_eq!(mode(abi::VA_RC_VBR, true), Some(abi::VA_RC_VBR));
        assert_eq!(mode(abi::VA_RC_CQP, true), Some(abi::VA_RC_CQP));
        assert_eq!(mode(0x100, false), None);
        assert_eq!(
            mode(abi::VA_ATTRIB_NOT_SUPPORTED, false),
            Some(abi::VA_RC_CQP)
        );
    }

    #[test]
    fn variable_rate_aims_at_half_the_peak() {
        let vbr = RateControl {
            mode: abi::VA_RC_VBR,
            target: 8_000_000,
        };
        assert_eq!(vbr.misc_bits(), (16_000_000, 50));
        assert_eq!(vbr.sequence_bits(), 8_000_000);
        let cbr = RateControl {
            mode: abi::VA_RC_CBR,
            ..vbr
        };
        assert_eq!(cbr.misc_bits(), (8_000_000, 100));
        let cqp = RateControl {
            mode: abi::VA_RC_CQP,
            ..vbr
        };
        assert_eq!(cqp.sequence_bits(), 0);
    }

    #[test]
    fn attribute_fields_read_their_bits() {
        let value = (2 << abi::hevc_features::SAO.shift) | (1 << abi::hevc_features::AMP.shift);
        assert_eq!(field(value, abi::hevc_features::SAO), 2);
        assert_eq!(field(value, abi::hevc_features::AMP), 1);
        assert_eq!(field(value, abi::hevc_features::TEMPORAL_MVP), 0);
    }
}
