//! One ALAC frame: syntax elements until `END`.
//!
//! The element syntax is the reference decoder's (`ALACDecoder.cpp`):
//! a 3-bit tag, a 4-bit instance, then for a channel element 12 zero bits,
//! the partial-frame flag, the shifted byte count, the escape flag, the
//! optional 32-bit sample count, and either a compressed body or raw
//! samples. Elements fill channels in frame order. `DSE` and `FIL` carry
//! bytes the decoder skips. `CCE` and `PCE` are not ALAC tools.

use cutlass_decode::CodecError;

use crate::bits::{BitReader, sign_extend};
use crate::config::AlacConfig;
use crate::golomb::{self, AgParams};
use crate::predictor::{MAX_COEFS, integrate_in_place, unpc_block};

pub(crate) const ID_SCE: u32 = 0;
pub(crate) const ID_CPE: u32 = 1;
const ID_CCE: u32 = 2;
pub(crate) const ID_LFE: u32 = 3;
const ID_DSE: u32 = 4;
const ID_PCE: u32 = 5;
const ID_FIL: u32 = 6;
pub(crate) const ID_END: u32 = 7;

/// Scratch buffers kept between frames so a frame does not allocate.
#[derive(Debug, Default)]
pub(crate) struct FrameDecoder {
    residual: [Vec<i32>; 2],
    mixed: [Vec<i32>; 2],
}

#[derive(Clone, Copy)]
struct ChannelHeader {
    mode: u32,
    den_shift: u32,
    pb_factor: u32,
    order: usize,
    coefs: [i16; MAX_COEFS],
}

impl FrameDecoder {
    /// Decode `data` into `out`, interleaved in frame channel order, at the
    /// stream's bit depth. Returns samples per channel. `out` is replaced.
    pub(crate) fn decode(
        &mut self,
        config: &AlacConfig,
        data: &[u8],
        out: &mut Vec<i32>,
    ) -> Result<usize, CodecError> {
        let channels = usize::from(config.channels);
        let mut reader = BitReader::new(data);
        let mut filled = 0usize;
        let mut frames: Option<usize> = None;
        out.clear();
        loop {
            let at = reader.at_bit();
            let tag = reader.read(3, "element tag")?;
            match tag {
                ID_SCE | ID_LFE | ID_CPE => {
                    let width = if tag == ID_CPE { 2 } else { 1 };
                    if filled + width > channels {
                        return Err(CodecError::Malformed {
                            at_bit: at,
                            what: "element past the channel count",
                        });
                    }
                    self.element(config, &mut reader, width, filled, &mut frames, out)?;
                    filled += width;
                }
                ID_DSE => {
                    reader.skip(4, "data element instance")?;
                    let align = reader.read(1, "data element align")? != 0;
                    let mut count = reader.read(8, "data element count")?;
                    if count == 255 {
                        count += reader.read(8, "data element count")?;
                    }
                    if align {
                        reader.align();
                    }
                    reader.skip(count as usize * 8, "data element bytes")?;
                }
                ID_FIL => {
                    let mut count = reader.read(4, "fill count")?;
                    if count == 15 {
                        count += reader.read(8, "fill count")?;
                        count -= 1;
                    }
                    reader.skip(count as usize * 8, "fill bytes")?;
                }
                ID_CCE | ID_PCE => {
                    return Err(CodecError::Unsupported {
                        what: "alac coupling or program element",
                    });
                }
                _ => {
                    // ID_END. Trailing bits after the end are padding.
                    reader.align();
                    break;
                }
            }
        }
        if filled != channels {
            return Err(CodecError::Malformed {
                at_bit: reader.at_bit(),
                what: "frame ended before every channel",
            });
        }
        frames.ok_or(CodecError::Malformed {
            at_bit: reader.at_bit(),
            what: "frame without audio",
        })
    }

    fn element(
        &mut self,
        config: &AlacConfig,
        reader: &mut BitReader<'_>,
        width: usize,
        first: usize,
        frames: &mut Option<usize>,
        out: &mut Vec<i32>,
    ) -> Result<(), CodecError> {
        let channels = usize::from(config.channels);
        let depth = u32::from(config.bit_depth);
        reader.skip(4, "element instance")?;
        let at = reader.at_bit();
        if reader.read(12, "element header")? != 0 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "element header bits",
            });
        }
        let partial = reader.read(1, "partial frame flag")? != 0;
        let at = reader.at_bit();
        let shifted = reader.read(2, "shifted bytes")?;
        if shifted == 3 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "shifted byte count",
            });
        }
        let escape = reader.read(1, "escape flag")? != 0;
        let at = reader.at_bit();
        let count = if partial {
            reader.read(32, "frame sample count")?
        } else {
            config.frame_length
        };
        if count == 0 || count > config.frame_length {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "frame sample count",
            });
        }
        let count = count as usize;
        match *frames {
            None => {
                let total = count.checked_mul(channels).ok_or(CodecError::Limit {
                    what: "frame samples",
                })?;
                out.clear();
                out.resize(total, 0);
                *frames = Some(count);
            }
            Some(known) if known != count => {
                return Err(CodecError::Malformed {
                    at_bit: at,
                    what: "elements disagree on the sample count",
                });
            }
            Some(_) => {}
        }
        if escape {
            if shifted != 0 {
                return Err(CodecError::Malformed {
                    at_bit: at,
                    what: "uncompressed element with shifted bytes",
                });
            }
            let needed = count
                .checked_mul(width)
                .and_then(|samples| samples.checked_mul(depth as usize))
                .ok_or(CodecError::Limit {
                    what: "frame samples",
                })?;
            if needed > reader.remaining() {
                return Err(CodecError::Truncated {
                    at_bit: reader.at_bit(),
                    what: "uncompressed samples",
                });
            }
            for i in 0..count {
                for c in 0..width {
                    out[i * channels + first + c] =
                        reader.read_signed(depth, "uncompressed sample")?;
                }
            }
            return Ok(());
        }
        if (depth == 16 || depth == 20) && shifted != 0 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "shifted bytes below 24-bit",
            });
        }
        let at = reader.at_bit();
        let mix_bits = reader.read(8, "mix bits")?;
        let mix_res = reader.read(8, "mix residue")? as u8 as i8 as i32;
        if width == 2 && mix_res != 0 && mix_bits > 31 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "mix shift",
            });
        }
        let mut headers = [ChannelHeader {
            mode: 0,
            den_shift: 0,
            pb_factor: 0,
            order: 0,
            coefs: [0; MAX_COEFS],
        }; 2];
        for header in headers.iter_mut().take(width) {
            let at = reader.at_bit();
            header.mode = reader.read(4, "prediction mode")?;
            header.den_shift = reader.read(4, "prediction shift")?;
            header.pb_factor = reader.read(3, "golomb factor")?;
            header.order = reader.read(5, "prediction order")? as usize;
            for coef in header.coefs.iter_mut().take(header.order) {
                *coef = reader.read(16, "prediction coefficient")? as u16 as i16;
            }
            if header.den_shift == 0 && (1..31).contains(&header.order) {
                return Err(CodecError::Malformed {
                    at_bit: at,
                    what: "prediction shift of zero",
                });
            }
        }
        let shift = shifted * 8;
        let shift_at = reader.clone();
        if shift != 0 {
            let bits = count
                .checked_mul(width)
                .and_then(|samples| samples.checked_mul(shift as usize))
                .ok_or(CodecError::Limit {
                    what: "frame samples",
                })?;
            reader.skip(bits, "shifted bits")?;
        }
        let chan_bits = depth - shift + (width as u32 - 1);
        if chan_bits > 32 {
            return Err(CodecError::Unsupported {
                what: "33-bit channel pair",
            });
        }
        for (c, header) in headers.iter_mut().enumerate().take(width) {
            let residual = &mut self.residual[c];
            residual.clear();
            residual.resize(count, 0);
            let pb = u32::from(config.pb) * header.pb_factor / 4;
            let params = AgParams::new(u32::from(config.mb), pb, u32::from(config.kb));
            golomb::decompress(params, reader, residual, chan_bits)?;
            if header.mode != 0 {
                integrate_in_place(residual, chan_bits);
            }
            let mixed = &mut self.mixed[c];
            mixed.clear();
            mixed.resize(count, 0);
            unpc_block(
                residual,
                mixed,
                &mut header.coefs,
                header.order,
                chan_bits,
                header.den_shift,
            );
        }
        if width == 2 && mix_res != 0 {
            let [u, v] = &mut self.mixed;
            for (left, right) in u.iter_mut().zip(v.iter_mut()) {
                let diff = *right;
                let l = left
                    .wrapping_add(diff)
                    .wrapping_sub(mix_res.wrapping_mul(diff) >> mix_bits);
                *left = l;
                *right = l.wrapping_sub(diff);
            }
        }
        let mut shifts = shift_at;
        for i in 0..count {
            for c in 0..width {
                let mut value = self.mixed[c][i];
                if shift != 0 {
                    let low = shifts.read(shift, "shifted bits")?;
                    value = ((value as u32) << shift | low) as i32;
                }
                out[i * channels + first + c] = sign_extend(value, depth);
            }
        }
        Ok(())
    }
}
