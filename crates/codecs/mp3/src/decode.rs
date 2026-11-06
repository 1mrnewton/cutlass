//! The Layer III decoder: one frame per push, the bit reservoir, and
//! gapless trim.

use cutlass_decode::{
    AudioOutput, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, Pcm,
    SideData,
};

use crate::bits::BitReader;
use crate::header::{FrameHeader, MAX_FRAME_BYTES, Mode, Version, classify_rate};
use crate::layer3::{self, Parsed, Scalefactors};
use crate::sideinfo::{SHORT, SideInfo, crc16};
use crate::synth::{Synthesis, alias_reduce, hybrid};

/// Samples of delay the Layer III synthesis adds: 528 plus one, the value
/// LAME, FFmpeg and mpg123 all add to the LAME tag's encoder delay.
pub const DECODER_DELAY: u32 = 529;

/// Most reservoir bytes a frame can reach back: 9-bit `main_data_begin`.
const RESERVOIR_BYTES: usize = 511;

/// The stream a decoder has locked onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    version: Version,
    sample_rate: u32,
    channels: usize,
}

/// Gapless trim from the container, in samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Gapless {
    encoder_delay: u32,
    padding: u32,
}

/// Decoded PCM of one frame, before trim.
struct Frame {
    pcm: Vec<f32>,
    samples: usize,
    missing_granules: usize,
    shape: Shape,
}

/// Overlap and filterbank history of one channel.
#[derive(Clone)]
struct Channel {
    overlap: [[f32; 18]; 32],
    synthesis: Synthesis,
}

impl Default for Channel {
    fn default() -> Self {
        Self {
            overlap: [[0.0; 18]; 32],
            synthesis: Synthesis::default(),
        }
    }
}

/// Native Layer III decoder for MPEG-1, MPEG-2 LSF and MPEG-2.5.
///
/// One push is one whole frame, header included, as `cutlass-mp3-container`
/// returns it. Output is interleaved `f32` at full scale 1.0, 1152 samples
/// per channel for MPEG-1 and 576 otherwise.
///
/// The reservoir: after [`Decoder::reset`] or `open`, a frame whose main
/// data starts in bytes this decoder never saw decodes its missing
/// granules as silence and marks their samples in `skip_leading`. Once the
/// decoder knows where the previous main data ended, a `main_data_begin`
/// that reaches past the bytes after it is [`CodecError::Malformed`].
///
/// Gapless: a [`SideData::Gapless`] on a packet sets the encoder delay and
/// padding (the LAME tag fields `cutlass-mp3-container` reports as `Trim`).
/// Output timestamps then count from the first valid sample: `encoder_delay`
/// plus [`DECODER_DELAY`] leading samples are `skip_leading` with a
/// negative `pts`, and the last `padding - 529` samples are held back and
/// returned by `drain` as `skip_trailing`. When the padding is shorter than
/// 529, `drain` releases the missing samples from the filterbank.
pub struct Mp3Decoder {
    open_rate: u32,
    open_channels: usize,
    shape: Option<Shape>,
    reservoir: Vec<u8>,
    exact: bool,
    channels: Box<[Channel; 2]>,
    gapless: Option<Gapless>,
    held: Vec<f32>,
    held_start: i64,
    invalid_until: i64,
    clock: i64,
    timescale: u32,
    drained: bool,
}

impl Mp3Decoder {
    /// Open a decoder.
    ///
    /// `codec` must be [`CodecId::Mp3`]. A non-zero `sample_rate` and a
    /// layout must match the frames. `extra` is ignored: Layer III has no
    /// decoder configuration (a WAV `MPEGLAYER3WAVEFORMAT` blob is not one).
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mp3 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if params.sample_rate != 0 && classify_rate(params.sample_rate).is_none() {
            return Err(CodecError::Unsupported {
                what: "sample rate",
            });
        }
        let open_channels = match params.layout.as_ref().map(ChannelLayout::len) {
            None => 0,
            Some(n @ (1 | 2)) => n,
            Some(_) => return Err(CodecError::Unsupported { what: "channels" }),
        };
        Ok(Self {
            open_rate: params.sample_rate,
            open_channels,
            shape: None,
            reservoir: Vec::with_capacity(RESERVOIR_BYTES + MAX_FRAME_BYTES),
            exact: false,
            channels: Box::default(),
            gapless: None,
            held: Vec::new(),
            held_start: 0,
            invalid_until: 0,
            clock: 0,
            timescale: 0,
            drained: false,
        })
    }

    /// The sample rate and channel count once a frame has been decoded.
    #[must_use]
    pub fn stream(&self) -> Option<(u32, usize)> {
        self.shape.map(|shape| (shape.sample_rate, shape.channels))
    }

    /// Decode one frame. On error the decoder is unchanged.
    fn decode_frame(&mut self, data: &[u8]) -> Result<Frame, CodecError> {
        let Some(head) = data.get(..4) else {
            return Err(CodecError::Truncated {
                at_bit: data.len() as u64 * 8,
                what: "frame header",
            });
        };
        let header = FrameHeader::parse([head[0], head[1], head[2], head[3]])?;
        if header.layer != 3 {
            return Err(CodecError::Unsupported { what: "layer" });
        }
        match header.frame_bytes() {
            Some(size) if data.len() < size => {
                return Err(CodecError::Truncated {
                    at_bit: data.len() as u64 * 8,
                    what: "frame",
                });
            }
            Some(size) if data.len() > size => {
                return Err(CodecError::Malformed {
                    at_bit: size as u64 * 8,
                    what: "frame length",
                });
            }
            _ => {}
        }
        if data.len() > MAX_FRAME_BYTES {
            return Err(CodecError::Limit {
                what: "frame bytes",
            });
        }
        let shape = Shape {
            version: header.version,
            sample_rate: header.sample_rate,
            channels: header.channels(),
        };
        self.check_shape(shape)?;
        let mut pos = 4;
        let mut stored_crc = None;
        if header.protected {
            let Some(bytes) = data.get(4..6) else {
                return Err(CodecError::Truncated {
                    at_bit: data.len() as u64 * 8,
                    what: "crc",
                });
            };
            stored_crc = Some(u16::from_be_bytes([bytes[0], bytes[1]]));
            pos = 6;
        }
        let side_len = header.side_info_bytes();
        let Some(side_bytes) = data.get(pos..pos + side_len) else {
            return Err(CodecError::Truncated {
                at_bit: data.len() as u64 * 8,
                what: "side information",
            });
        };
        if let Some(stored) = stored_crc
            && crc16([head[0], head[1], head[2], head[3]], side_bytes) != stored
        {
            return Err(CodecError::Malformed {
                at_bit: 32,
                what: "crc",
            });
        }
        let mut side = SideInfo::parse(&header, side_bytes)?;
        let payload = &data[pos + side_len..];
        let begin = usize::from(side.main_data_begin);
        let mut main = Vec::with_capacity(begin.max(self.reservoir.len()) + payload.len());
        // Bit offset of the first main-data bit inside `main`. Negative when
        // the frame reaches into bytes this decoder never received.
        let start: i64 = if begin <= self.reservoir.len() {
            main.extend_from_slice(&self.reservoir[self.reservoir.len() - begin..]);
            0
        } else if !self.exact {
            main.extend_from_slice(&self.reservoir);
            -((begin - self.reservoir.len()) as i64 * 8)
        } else {
            return Err(CodecError::Malformed {
                at_bit: (pos as u64) * 8,
                what: "main_data_begin",
            });
        };
        main.extend_from_slice(payload);
        let main_bits = main.len() as i64 * 8;
        let channels = header.channels();
        let granules = header.granules();
        let intensity = header.mode == Mode::JointStereo && header.mode_extension & 1 != 0;
        let mut parsed: Vec<[Parsed; 2]> = Vec::with_capacity(granules);
        let mut missing = 0usize;
        let mut cursor = start;
        for gr in 0..granules {
            let mut pair: [Parsed; 2] = Default::default();
            let present = cursor >= 0;
            if !present {
                missing += 1;
            }
            for ch in 0..channels {
                let mut info = side.granules[gr][ch];
                let length = i64::from(info.part2_3_length);
                let end = cursor + length;
                if end > main_bits {
                    return Err(CodecError::Truncated {
                        at_bit: (pos + side_len) as u64 * 8,
                        what: "main data",
                    });
                }
                if present {
                    let mut reader = BitReader::window(&main, cursor as usize, end as usize);
                    let previous: Scalefactors = if gr == 1 {
                        parsed[0][ch].scalefactors
                    } else {
                        Scalefactors::default()
                    };
                    let scfsi = if gr == 1 && !header.version.lsf() {
                        Some(&side.scfsi[ch])
                    } else {
                        None
                    };
                    let scalefactors = layer3::read_scalefactors(
                        &mut reader,
                        &header,
                        &mut info,
                        scfsi,
                        &previous,
                        intensity && ch == 1,
                    )
                    .map_err(|error| match error {
                        CodecError::Truncated { at_bit, .. } => CodecError::Malformed {
                            at_bit,
                            what: "scalefactors past part2_3_length",
                        },
                        other => other,
                    })?;
                    let slot = &mut pair[ch];
                    slot.nonzero =
                        layer3::read_spectrum(&mut reader, &header, &info, &mut slot.quant)
                            .map_err(|error| match error {
                                CodecError::Truncated { at_bit, .. } => CodecError::Malformed {
                                    at_bit,
                                    what: "big_values past part2_3_length",
                                },
                                other => other,
                            })?;
                    slot.scalefactors = scalefactors;
                }
                pair[ch].info = info;
                side.granules[gr][ch] = info;
                cursor = end;
            }
            parsed.push(pair);
        }

        // Everything after this point is infallible: commit.
        self.shape = Some(shape);
        let new_reservoir_from = if cursor >= 0 {
            self.exact = true;
            (cursor as usize).div_ceil(8).min(main.len())
        } else {
            self.exact = false;
            0
        };
        let tail = &main[new_reservoir_from..];
        let keep = tail.len().min(RESERVOIR_BYTES);
        self.reservoir.clear();
        self.reservoir.extend_from_slice(&tail[tail.len() - keep..]);

        let samples = header.samples();
        let mut pcm = vec![0f32; samples * channels];
        let mut xr = [[0f32; 576]; 2];
        let mut subbands = [[0f32; 32]; 18];
        let mut out = [0f32; 32];
        for (gr, pair) in parsed.iter().enumerate() {
            let present = gr >= missing;
            for ch in 0..channels {
                if present {
                    layer3::requantize(&header, &pair[ch], &mut xr[ch]);
                } else {
                    xr[ch] = [0.0; 576];
                }
            }
            if present && channels == 2 {
                let (left, right) = xr.split_at_mut(1);
                layer3::stereo(&header, &mut left[0], &mut right[0], &pair[1]);
            }
            for ch in 0..channels {
                let info = &pair[ch].info;
                let spectrum = &mut xr[ch];
                let (long_type, long_subbands, boundaries) =
                    if info.window_switching && info.block_type == SHORT {
                        if info.mixed { (0, 2, 1) } else { (0, 0, 0) }
                    } else {
                        let kind = if info.window_switching {
                            info.block_type
                        } else {
                            0
                        };
                        (kind, 32, 31)
                    };
                if present {
                    layer3::reorder(&header, info, spectrum);
                    alias_reduce(spectrum, boundaries);
                }
                let last = spectrum.iter().rposition(|value| *value != 0.0);
                let nonzero_subbands = last.map_or(0, |line| line / 18 + 1);
                let state = &mut self.channels[ch];
                hybrid(
                    spectrum,
                    long_type,
                    long_subbands,
                    nonzero_subbands,
                    &mut state.overlap,
                    &mut subbands,
                );
                for (t, slot) in subbands.iter().enumerate() {
                    state.synthesis.run(slot, &mut out);
                    let base = gr * 576 + t * 32;
                    for (i, sample) in out.iter().enumerate() {
                        pcm[(base + i) * channels + ch] = *sample;
                    }
                }
            }
        }
        Ok(Frame {
            pcm,
            samples,
            missing_granules: missing,
            shape,
        })
    }

    fn check_shape(&self, shape: Shape) -> Result<(), CodecError> {
        if let Some(locked) = self.shape {
            if locked.sample_rate != shape.sample_rate || locked.version != shape.version {
                return Err(CodecError::Malformed {
                    at_bit: 12,
                    what: "sample rate change",
                });
            }
            if locked.channels != shape.channels {
                return Err(CodecError::Malformed {
                    at_bit: 24,
                    what: "channel change",
                });
            }
        }
        if self.open_rate != 0 && self.open_rate != shape.sample_rate {
            return Err(CodecError::Malformed {
                at_bit: 20,
                what: "sample rate change",
            });
        }
        if self.open_channels != 0 && self.open_channels != shape.channels {
            return Err(CodecError::Malformed {
                at_bit: 24,
                what: "channel change",
            });
        }
        Ok(())
    }

    fn layout(channels: usize) -> ChannelLayout {
        if channels == 1 {
            ChannelLayout::mono()
        } else {
            ChannelLayout::stereo()
        }
    }

    /// Ticks of `timescale` for a stream sample index.
    fn ticks(&self, sample: i64, rate: u32) -> i64 {
        if self.timescale == 0 || rate == 0 {
            return 0;
        }
        if self.timescale == rate {
            return sample;
        }
        let scaled = i128::from(sample) * i128::from(self.timescale) / i128::from(rate);
        i64::try_from(scaled).unwrap_or(0)
    }

    fn output(
        &self,
        pcm: Vec<f32>,
        shape: Shape,
        first: i64,
        skip_leading: u32,
        skip_trailing: u32,
    ) -> Result<Output, CodecError> {
        let pcm = Pcm::from_f32(shape.sample_rate, Self::layout(shape.channels), pcm)
            .map_err(|_| CodecError::Limit { what: "pcm" })?;
        Ok(Output::Audio(AudioOutput {
            pcm,
            pts: self.ticks(first, shape.sample_rate),
            timescale: self.timescale,
            skip_leading,
            skip_trailing,
            concealed: false,
        }))
    }

    fn lead(gapless: Gapless) -> i64 {
        i64::from(gapless.encoder_delay) + i64::from(DECODER_DELAY)
    }

    /// Release the filterbank tail past the last frame: one silent granule.
    fn flush_tail(&self, shape: Shape) -> Vec<f32> {
        let channels = shape.channels;
        let mut pcm = vec![0f32; 576 * channels];
        let mut subbands = [[0f32; 32]; 18];
        let mut out = [0f32; 32];
        let silent = [0f32; 576];
        for ch in 0..channels {
            let mut state = self.channels[ch].clone();
            hybrid(&silent, 0, 32, 0, &mut state.overlap, &mut subbands);
            for (t, slot) in subbands.iter().enumerate() {
                state.synthesis.run(slot, &mut out);
                for (i, sample) in out.iter().enumerate() {
                    pcm[(t * 32 + i) * channels + ch] = *sample;
                }
            }
        }
        pcm
    }
}

fn gapless_of(packet: &Packet) -> Option<(Gapless, u32)> {
    packet.side.iter().find_map(|side| match side {
        SideData::Gapless {
            encoder_delay,
            trailing_padding,
            sample_rate,
        } => Some((
            Gapless {
                encoder_delay: *encoder_delay,
                padding: *trailing_padding,
            },
            *sample_rate,
        )),
        _ => None,
    })
}

impl Decoder for Mp3Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Mp3
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let frame = self.decode_frame(&packet.data)?;
        self.drained = false;
        let shape = frame.shape;
        let rate = shape.sample_rate;
        if let Some((gapless, gapless_rate)) = gapless_of(packet)
            && (gapless_rate == rate || gapless_rate == 0)
        {
            self.gapless = Some(gapless);
        }
        self.timescale = packet.timescale;
        let first = if packet.timescale > 0 {
            let scaled = i128::from(packet.pts) * i128::from(rate) / i128::from(packet.timescale);
            i64::try_from(scaled).unwrap_or(self.clock)
        } else {
            self.clock
        };
        let samples = frame.samples as i64;
        self.clock = first.saturating_add(samples);
        let missing = (frame.missing_granules * 576) as i64;
        let Some(gapless) = self.gapless else {
            let output = Output::Audio(AudioOutput {
                pcm: Pcm::from_f32(rate, Self::layout(shape.channels), frame.pcm)
                    .map_err(|_| CodecError::Limit { what: "pcm" })?,
                pts: packet.pts,
                timescale: packet.timescale,
                skip_leading: missing as u32,
                skip_trailing: 0,
                concealed: false,
            });
            return Ok(vec![output]);
        };
        let lead = Self::lead(gapless);
        let channels = shape.channels;
        if self.held.is_empty() {
            self.held_start = first;
        }
        if missing > 0 {
            self.invalid_until = self.invalid_until.max(first + missing);
        }
        let start = self.held_start;
        let mut combined = std::mem::take(&mut self.held);
        combined.extend_from_slice(&frame.pcm);
        let total = combined.len() / channels;
        let hold = gapless.padding.saturating_sub(DECODER_DELAY) as usize;
        let emit = total.saturating_sub(hold);
        self.held = combined.split_off(emit * channels);
        self.held_start = start + emit as i64;
        if emit == 0 {
            return Ok(Vec::new());
        }
        let valid_from = lead.max(self.invalid_until);
        let skip = (valid_from - start).clamp(0, emit as i64) as u32;
        Ok(vec![self.output(combined, shape, start - lead, skip, 0)?])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        self.drained = true;
        let (Some(gapless), Some(shape)) = (self.gapless, self.shape) else {
            return Ok(Vec::new());
        };
        let lead = Self::lead(gapless);
        let channels = shape.channels;
        let start = if self.held.is_empty() {
            self.clock
        } else {
            self.held_start
        };
        let mut pcm = std::mem::take(&mut self.held);
        let held = pcm.len() / channels;
        let wanted_past_end = DECODER_DELAY.saturating_sub(gapless.padding) as usize;
        let trailing = if wanted_past_end > 0 {
            let tail = self.flush_tail(shape);
            pcm.extend_from_slice(&tail);
            576 - wanted_past_end
        } else {
            held
        };
        let total = pcm.len() / channels;
        if total == 0 {
            return Ok(Vec::new());
        }
        let valid_from = lead.max(self.invalid_until);
        let skip = (valid_from - start).clamp(0, total as i64) as u32;
        let trailing = (trailing as u32).min(total as u32 - skip);
        Ok(vec![self.output(
            pcm,
            shape,
            start - lead,
            skip,
            trailing,
        )?])
    }

    fn reset(&mut self) {
        self.reservoir.clear();
        self.exact = false;
        *self.channels = Default::default();
        self.held.clear();
        self.held_start = 0;
        self.invalid_until = 0;
        self.clock = 0;
        self.drained = false;
    }
}
