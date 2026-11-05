//! Frame headers (RFC 6386 Sections 9 and 19) and the state that outlives
//! one frame: probabilities, segmentation, and loop-filter deltas.

use cutlass_decode::CodecError;

use crate::bool_coder::{BoolDecoder, BoolEncoder};
use crate::tables::{
    COEFF_UPDATE_PROBS, DEFAULT_COEFF_PROBS, DEFAULT_MV_PROBS, MV_UPDATE_PROBS, UV_MODE_PROBS,
    Y_MODE_PROBS,
};

/// Coefficient probabilities by block type, band, context, and node.
pub type CoeffProbs = [[[[u8; 11]; 3]; 8]; 4];

/// Largest picture edge a key frame can name (14 bits).
pub const MAX_EDGE: u32 = 16_383;

/// Probabilities that persist between frames (Section 9.11, `refresh_entropy_probs`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probs {
    pub coeff: CoeffProbs,
    pub mv: [[u8; 19]; 2],
    pub y_mode: [u8; 4],
    pub uv_mode: [u8; 3],
}

impl Default for Probs {
    fn default() -> Self {
        Self {
            coeff: DEFAULT_COEFF_PROBS,
            mv: DEFAULT_MV_PROBS,
            y_mode: Y_MODE_PROBS,
            uv_mode: UV_MODE_PROBS,
        }
    }
}

/// Section 9.3. Feature values persist until a key frame or an update.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub update_data: bool,
    pub absolute: bool,
    pub quant: [i8; 4],
    pub filter: [i8; 4],
    pub tree_probs: [u8; 3],
}

/// Section 9.6. Deltas by reference frame (intra, last, golden, altref)
/// and by mode (B_PRED, ZEROMV, other MV modes, SPLITMV).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterDeltas {
    pub enabled: bool,
    pub ref_frame: [i8; 4],
    pub mode: [i8; 4],
}

/// State carried from one frame to the next.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Persist {
    pub probs: Probs,
    pub segmentation: Segmentation,
    pub deltas: FilterDeltas,
}

/// Section 9.6 quantizer indices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantIndices {
    pub y_ac: u8,
    pub y_dc_delta: i8,
    pub y2_dc_delta: i8,
    pub y2_ac_delta: i8,
    pub uv_dc_delta: i8,
    pub uv_ac_delta: i8,
}

/// The first three bytes of every frame, plus the key-frame start code and size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameTag {
    pub key: bool,
    pub version: u8,
    pub show: bool,
    pub first_size: usize,
    pub width: u32,
    pub height: u32,
    pub hscale: u8,
    pub vscale: u8,
}

/// Everything the first partition says before the per-macroblock data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub tag: FrameTag,
    pub color_space: bool,
    pub clamping: bool,
    pub segmentation: Segmentation,
    pub filter_simple: bool,
    pub filter_level: u8,
    pub sharpness: u8,
    pub deltas: FilterDeltas,
    pub partitions: usize,
    pub quant: QuantIndices,
    pub refresh_golden: bool,
    pub refresh_altref: bool,
    pub copy_to_golden: u8,
    pub copy_to_altref: u8,
    pub sign_bias_golden: bool,
    pub sign_bias_altref: bool,
    pub refresh_entropy: bool,
    pub refresh_last: bool,
    /// Probabilities this frame decodes with.
    pub probs: Probs,
    pub skip_enabled: bool,
    pub prob_skip: u8,
    pub prob_intra: u8,
    pub prob_last: u8,
    pub prob_golden: u8,
}

/// A parsed frame: the header, the persistent state after it, the rest of
/// the first partition, and the token partitions.
pub struct Parsed<'a> {
    pub header: FrameHeader,
    pub next: Persist,
    pub modes: BoolDecoder<'a>,
    pub tokens: Vec<&'a [u8]>,
}

fn malformed(at: usize, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}

fn truncated(at: usize, what: &'static str) -> CodecError {
    CodecError::Truncated {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}

/// Reads the uncompressed data chunk (Section 9.1).
pub fn parse_tag(data: &[u8]) -> Result<FrameTag, CodecError> {
    if data.len() < 3 {
        return Err(truncated(data.len(), "vp8 frame tag"));
    }
    let raw = u32::from(data[0]) | (u32::from(data[1]) << 8) | (u32::from(data[2]) << 16);
    let key = raw & 1 == 0;
    let version = ((raw >> 1) & 7) as u8;
    let show = (raw >> 4) & 1 == 1;
    let first_size = (raw >> 5) as usize;
    if version > 3 {
        return Err(CodecError::Unsupported {
            what: "vp8 version above 3",
        });
    }
    let mut tag = FrameTag {
        key,
        version,
        show,
        first_size,
        width: 0,
        height: 0,
        hscale: 0,
        vscale: 0,
    };
    if key {
        if data.len() < 10 {
            return Err(truncated(data.len(), "vp8 key frame header"));
        }
        if data[3..6] != [0x9d, 0x01, 0x2a] {
            return Err(malformed(3, "vp8 start code"));
        }
        let w = u32::from(data[6]) | (u32::from(data[7]) << 8);
        let h = u32::from(data[8]) | (u32::from(data[9]) << 8);
        tag.width = w & 0x3fff;
        tag.hscale = (w >> 14) as u8;
        tag.height = h & 0x3fff;
        tag.vscale = (h >> 14) as u8;
        if tag.width == 0 || tag.height == 0 {
            return Err(malformed(6, "vp8 picture size"));
        }
    }
    Ok(tag)
}

/// Parses a whole frame header. `state` is the state before this frame.
pub fn parse<'a>(data: &'a [u8], state: &Persist) -> Result<Parsed<'a>, CodecError> {
    let tag = parse_tag(data)?;
    let start = if tag.key { 10 } else { 3 };
    let rest = &data[start..];
    if tag.first_size > rest.len() {
        return Err(truncated(data.len(), "vp8 first partition"));
    }
    let (first, after) = rest.split_at(tag.first_size);
    let mut bits = BoolDecoder::new(first);

    let mut next = if tag.key {
        Persist::default()
    } else {
        state.clone()
    };
    let (color_space, clamping) = if tag.key {
        (bits.bit(), bits.bit())
    } else {
        (false, false)
    };

    // Section 9.3.
    let seg = &mut next.segmentation;
    seg.enabled = bits.bit();
    if seg.enabled {
        seg.update_map = bits.bit();
        seg.update_data = bits.bit();
        if seg.update_data {
            seg.absolute = bits.bit();
            for value in &mut seg.quant {
                *value = clamp_i8(bits.maybe_signed(7));
            }
            for value in &mut seg.filter {
                *value = clamp_i8(bits.maybe_signed(6));
            }
        }
        if seg.update_map {
            for prob in &mut seg.tree_probs {
                *prob = if bits.bit() {
                    bits.literal(8) as u8
                } else {
                    255
                };
            }
        }
    } else {
        seg.update_map = false;
        seg.update_data = false;
    }
    let segmentation = *seg;

    // Section 9.6.
    let filter_simple = bits.bit();
    let filter_level = bits.literal(6) as u8;
    let sharpness = bits.literal(3) as u8;
    next.deltas.enabled = bits.bit();
    if next.deltas.enabled && bits.bit() {
        for value in &mut next.deltas.ref_frame {
            if bits.bit() {
                *value = clamp_i8(bits.signed(6));
            }
        }
        for value in &mut next.deltas.mode {
            if bits.bit() {
                *value = clamp_i8(bits.signed(6));
            }
        }
    }
    let deltas = next.deltas;

    // Section 9.5.
    let partitions = 1usize << bits.literal(2);
    let tokens = split_partitions(after, partitions, start + tag.first_size)?;

    // Section 9.6.
    let quant = QuantIndices {
        y_ac: bits.literal(7) as u8,
        y_dc_delta: clamp_i8(bits.maybe_signed(4)),
        y2_dc_delta: clamp_i8(bits.maybe_signed(4)),
        y2_ac_delta: clamp_i8(bits.maybe_signed(4)),
        uv_dc_delta: clamp_i8(bits.maybe_signed(4)),
        uv_ac_delta: clamp_i8(bits.maybe_signed(4)),
    };

    // Sections 9.7 and 9.8.
    let (refresh_golden, refresh_altref, copy_to_golden, copy_to_altref, bias_g, bias_a) =
        if tag.key {
            (true, true, 0, 0, false, false)
        } else {
            let refresh_golden = bits.bit();
            let refresh_altref = bits.bit();
            let copy_to_golden = if refresh_golden {
                0
            } else {
                bits.literal(2) as u8
            };
            let copy_to_altref = if refresh_altref {
                0
            } else {
                bits.literal(2) as u8
            };
            (
                refresh_golden,
                refresh_altref,
                copy_to_golden,
                copy_to_altref,
                bits.bit(),
                bits.bit(),
            )
        };
    let refresh_entropy = bits.bit();
    let refresh_last = if tag.key { true } else { bits.bit() };

    // Section 9.9: the base for this frame. A key frame starts from the
    // defaults, which `next` already holds.
    let base = next.probs.clone();
    let mut probs = base.clone();
    for (i, plane) in probs.coeff.iter_mut().enumerate() {
        for (j, band) in plane.iter_mut().enumerate() {
            for (k, ctx) in band.iter_mut().enumerate() {
                for (l, prob) in ctx.iter_mut().enumerate() {
                    if bits.read(COEFF_UPDATE_PROBS[i][j][k][l]) {
                        *prob = bits.literal(8) as u8;
                    }
                }
            }
        }
    }

    // Section 9.10.
    let skip_enabled = bits.bit();
    let prob_skip = if skip_enabled {
        bits.literal(8) as u8
    } else {
        0
    };
    let (mut prob_intra, mut prob_last, mut prob_golden) = (0, 0, 0);
    if !tag.key {
        prob_intra = bits.literal(8) as u8;
        prob_last = bits.literal(8) as u8;
        prob_golden = bits.literal(8) as u8;
        if bits.bit() {
            for prob in &mut probs.y_mode {
                *prob = bits.literal(8) as u8;
            }
        }
        if bits.bit() {
            for prob in &mut probs.uv_mode {
                *prob = bits.literal(8) as u8;
            }
        }
        // Section 17.2.
        for (component, update) in probs.mv.iter_mut().zip(MV_UPDATE_PROBS.iter()) {
            for (prob, &update_prob) in component.iter_mut().zip(update.iter()) {
                if bits.read(update_prob) {
                    let value = bits.literal(7) as u8;
                    *prob = if value == 0 { 1 } else { value << 1 };
                }
            }
        }
    }

    next.probs = if refresh_entropy { probs.clone() } else { base };

    let header = FrameHeader {
        tag,
        color_space,
        clamping,
        segmentation,
        filter_simple,
        filter_level,
        sharpness,
        deltas,
        partitions,
        quant,
        refresh_golden,
        refresh_altref,
        copy_to_golden,
        copy_to_altref,
        sign_bias_golden: bias_g,
        sign_bias_altref: bias_a,
        refresh_entropy,
        refresh_last,
        probs,
        skip_enabled,
        prob_skip,
        prob_intra,
        prob_last,
        prob_golden,
    };
    Ok(Parsed {
        header,
        next,
        modes: bits,
        tokens,
    })
}

fn clamp_i8(value: i32) -> i8 {
    i8::try_from(value.clamp(-127, 127)).unwrap_or(0)
}

fn split_partitions(data: &[u8], count: usize, offset: usize) -> Result<Vec<&[u8]>, CodecError> {
    let table = 3 * (count - 1);
    if data.len() < table {
        return Err(truncated(offset + data.len(), "vp8 partition sizes"));
    }
    let (sizes, mut rest) = data.split_at(table);
    let mut parts = Vec::with_capacity(count);
    for &[b0, b1, b2] in sizes.as_chunks::<3>().0 {
        let len = usize::from(b0) | (usize::from(b1) << 8) | (usize::from(b2) << 16);
        if len > rest.len() {
            return Err(truncated(offset + data.len(), "vp8 token partition"));
        }
        let (part, tail) = rest.split_at(len);
        parts.push(part);
        rest = tail;
    }
    parts.push(rest);
    Ok(parts)
}

/// What an encoder writes into the first partition before the macroblocks.
pub struct HeaderWriter<'a> {
    pub header: &'a FrameHeader,
    /// The probabilities the decoder holds before this frame.
    pub previous: &'a Probs,
    /// Segmentation the decoder holds before this frame.
    /// Loop-filter deltas the decoder holds before this frame.
    pub previous_deltas: &'a FilterDeltas,
}

impl HeaderWriter<'_> {
    /// Writes the header bits. The caller writes the macroblock modes next.
    pub fn write(&self, bits: &mut BoolEncoder) {
        let h = self.header;
        if h.tag.key {
            bits.bit(h.color_space);
            bits.bit(h.clamping);
        }
        let seg = &h.segmentation;
        bits.bit(seg.enabled);
        if seg.enabled {
            bits.bit(seg.update_map);
            bits.bit(seg.update_data);
            if seg.update_data {
                bits.bit(seg.absolute);
                for &value in &seg.quant {
                    bits.maybe_signed(7, i32::from(value));
                }
                for &value in &seg.filter {
                    bits.maybe_signed(6, i32::from(value));
                }
            }
            if seg.update_map {
                for &prob in &seg.tree_probs {
                    bits.bit(prob != 255);
                    if prob != 255 {
                        bits.literal(8, u32::from(prob));
                    }
                }
            }
        }
        bits.bit(h.filter_simple);
        bits.literal(6, u32::from(h.filter_level));
        bits.literal(3, u32::from(h.sharpness));
        bits.bit(h.deltas.enabled);
        if h.deltas.enabled {
            let old = if h.tag.key {
                FilterDeltas::default()
            } else {
                *self.previous_deltas
            };
            let update = old.ref_frame != h.deltas.ref_frame || old.mode != h.deltas.mode;
            bits.bit(update);
            if update {
                for (&new, &was) in h.deltas.ref_frame.iter().zip(old.ref_frame.iter()) {
                    bits.bit(new != was);
                    if new != was {
                        bits.signed(6, i32::from(new));
                    }
                }
                for (&new, &was) in h.deltas.mode.iter().zip(old.mode.iter()) {
                    bits.bit(new != was);
                    if new != was {
                        bits.signed(6, i32::from(new));
                    }
                }
            }
        }
        bits.literal(2, h.partitions.trailing_zeros());
        let q = &h.quant;
        bits.literal(7, u32::from(q.y_ac));
        for delta in [
            q.y_dc_delta,
            q.y2_dc_delta,
            q.y2_ac_delta,
            q.uv_dc_delta,
            q.uv_ac_delta,
        ] {
            bits.maybe_signed(4, i32::from(delta));
        }
        if !h.tag.key {
            bits.bit(h.refresh_golden);
            bits.bit(h.refresh_altref);
            if !h.refresh_golden {
                bits.literal(2, u32::from(h.copy_to_golden));
            }
            if !h.refresh_altref {
                bits.literal(2, u32::from(h.copy_to_altref));
            }
            bits.bit(h.sign_bias_golden);
            bits.bit(h.sign_bias_altref);
        }
        bits.bit(h.refresh_entropy);
        if !h.tag.key {
            bits.bit(h.refresh_last);
        }
        let base = if h.tag.key {
            Probs::default()
        } else {
            self.previous.clone()
        };
        for (i, plane) in h.probs.coeff.iter().enumerate() {
            for (j, band) in plane.iter().enumerate() {
                for (k, ctx) in band.iter().enumerate() {
                    for (l, &new) in ctx.iter().enumerate() {
                        let changed = new != base.coeff[i][j][k][l];
                        bits.write(COEFF_UPDATE_PROBS[i][j][k][l], changed);
                        if changed {
                            bits.literal(8, u32::from(new));
                        }
                    }
                }
            }
        }
        bits.bit(h.skip_enabled);
        if h.skip_enabled {
            bits.literal(8, u32::from(h.prob_skip));
        }
        if !h.tag.key {
            bits.literal(8, u32::from(h.prob_intra));
            bits.literal(8, u32::from(h.prob_last));
            bits.literal(8, u32::from(h.prob_golden));
            let y_changed = h.probs.y_mode != base.y_mode;
            bits.bit(y_changed);
            if y_changed {
                for &prob in &h.probs.y_mode {
                    bits.literal(8, u32::from(prob));
                }
            }
            let uv_changed = h.probs.uv_mode != base.uv_mode;
            bits.bit(uv_changed);
            if uv_changed {
                for &prob in &h.probs.uv_mode {
                    bits.literal(8, u32::from(prob));
                }
            }
            for (c, component) in h.probs.mv.iter().enumerate() {
                for (n, &new) in component.iter().enumerate() {
                    let changed = new != base.mv[c][n];
                    bits.write(MV_UPDATE_PROBS[c][n], changed);
                    if changed {
                        bits.literal(7, u32::from(new >> 1));
                    }
                }
            }
        }
    }
}

/// Builds the bytes of a frame from its first partition and token partitions.
pub fn assemble(tag: &FrameTag, first: &[u8], tokens: &[Vec<u8>]) -> Result<Vec<u8>, CodecError> {
    if first.len() >= 1 << 19 {
        return Err(CodecError::Limit {
            what: "vp8 first partition above 512 KiB",
        });
    }
    let mut out = Vec::with_capacity(first.len() + tokens.iter().map(Vec::len).sum::<usize>() + 32);
    let raw = u32::from(!tag.key)
        | (u32::from(tag.version) << 1)
        | (u32::from(tag.show) << 4)
        | ((first.len() as u32) << 5);
    out.extend_from_slice(&raw.to_le_bytes()[..3]);
    if tag.key {
        out.extend_from_slice(&[0x9d, 0x01, 0x2a]);
        let w = (tag.width & 0x3fff) | (u32::from(tag.hscale) << 14);
        let h = (tag.height & 0x3fff) | (u32::from(tag.vscale) << 14);
        out.extend_from_slice(&(w as u16).to_le_bytes());
        out.extend_from_slice(&(h as u16).to_le_bytes());
    }
    out.extend_from_slice(first);
    let Some((last, rest)) = tokens.split_last() else {
        return Err(CodecError::InvalidApi {
            what: "vp8 frame needs a token partition",
        });
    };
    for part in rest {
        if part.len() >= 1 << 24 {
            return Err(CodecError::Limit {
                what: "vp8 token partition above 16 MiB",
            });
        }
        out.extend_from_slice(&(part.len() as u32).to_le_bytes()[..3]);
    }
    for part in rest {
        out.extend_from_slice(part);
    }
    out.extend_from_slice(last);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        FilterDeltas, FrameHeader, FrameTag, HeaderWriter, Persist, Probs, QuantIndices,
        Segmentation, assemble, parse, parse_tag,
    };
    use crate::bool_coder::BoolEncoder;
    use cutlass_decode::CodecError;

    fn key_header() -> FrameHeader {
        let mut probs = Probs::default();
        probs.coeff[1][2][0][3] = 17;
        FrameHeader {
            tag: FrameTag {
                key: true,
                version: 0,
                show: true,
                first_size: 0,
                width: 33,
                height: 17,
                hscale: 1,
                vscale: 3,
            },
            color_space: false,
            clamping: false,
            segmentation: Segmentation {
                enabled: true,
                update_map: true,
                update_data: true,
                absolute: false,
                quant: [0, -5, 10, 127],
                filter: [0, 3, -63, 0],
                tree_probs: [128, 255, 3],
            },
            filter_simple: false,
            filter_level: 40,
            sharpness: 3,
            deltas: FilterDeltas {
                enabled: true,
                ref_frame: [2, 0, -2, -2],
                mode: [4, -2, 2, 4],
            },
            partitions: 4,
            quant: QuantIndices {
                y_ac: 60,
                y_dc_delta: -3,
                y2_dc_delta: 0,
                y2_ac_delta: 15,
                uv_dc_delta: -15,
                uv_ac_delta: 0,
            },
            refresh_golden: true,
            refresh_altref: true,
            copy_to_golden: 0,
            copy_to_altref: 0,
            sign_bias_golden: false,
            sign_bias_altref: false,
            refresh_entropy: true,
            refresh_last: true,
            probs,
            skip_enabled: true,
            prob_skip: 200,
            prob_intra: 0,
            prob_last: 0,
            prob_golden: 0,
        }
    }

    fn write(header: &FrameHeader, state: &Persist) -> Vec<u8> {
        let mut bits = BoolEncoder::new();
        HeaderWriter {
            header,
            previous: &state.probs,
            previous_deltas: &state.deltas,
        }
        .write(&mut bits);
        let first = bits.finish();
        let tokens = vec![vec![1, 2, 3]; header.partitions];
        assemble(&header.tag, &first, &tokens).expect("assemble")
    }

    #[test]
    fn a_written_key_header_parses_back() {
        let header = key_header();
        let data = write(&header, &Persist::default());
        let parsed = parse(&data, &Persist::default()).expect("parse");
        let mut want = header.clone();
        want.tag.first_size = parsed.header.tag.first_size;
        assert_eq!(parsed.header, want);
        assert_eq!(parsed.tokens.len(), 4);
        assert!(parsed.tokens.iter().all(|part| *part == [1, 2, 3]));
        assert_eq!(parsed.next.probs, header.probs);
        assert_eq!(parsed.next.segmentation, header.segmentation);
    }

    #[test]
    fn an_inter_header_keeps_unsent_deltas_and_restores_probs() {
        let key = key_header();
        let key_data = write(&key, &Persist::default());
        let state = parse(&key_data, &Persist::default()).expect("key").next;
        let mut inter = key.clone();
        inter.tag.key = false;
        inter.tag.width = 0;
        inter.tag.height = 0;
        inter.tag.hscale = 0;
        inter.tag.vscale = 0;
        inter.color_space = false;
        inter.segmentation.update_data = false;
        inter.segmentation.update_map = false;
        inter.deltas.mode[1] = -9;
        inter.refresh_golden = false;
        inter.refresh_altref = false;
        inter.copy_to_golden = 1;
        inter.copy_to_altref = 2;
        inter.sign_bias_golden = true;
        inter.refresh_entropy = false;
        inter.refresh_last = false;
        inter.probs.mv[0][4] = 254;
        inter.probs.y_mode = [1, 2, 3, 4];
        inter.prob_intra = 9;
        inter.prob_last = 99;
        inter.prob_golden = 199;
        let data = write(&inter, &state);
        let parsed = parse(&data, &state).expect("inter");
        let mut want = inter.clone();
        want.tag.first_size = parsed.header.tag.first_size;
        assert_eq!(parsed.header, want);
        assert_eq!(parsed.next.probs, state.probs, "refresh_entropy 0 restores");
        assert_eq!(parsed.next.deltas.mode, [4, -9, 2, 4]);
    }

    #[test]
    fn short_frames_and_bad_start_codes_fail_closed() {
        assert!(matches!(parse_tag(&[]), Err(CodecError::Truncated { .. })));
        assert!(matches!(
            parse_tag(&[0x10, 0, 0, 0x9d, 0x01, 0x2a, 1, 0]),
            Err(CodecError::Truncated { .. })
        ));
        assert!(matches!(
            parse_tag(&[0x10, 0, 0, 0x9d, 0x01, 0x2b, 1, 0, 1, 0]),
            Err(CodecError::Malformed { .. })
        ));
        assert!(matches!(
            parse_tag(&[0x10, 0, 0, 0x9d, 0x01, 0x2a, 0, 0, 1, 0]),
            Err(CodecError::Malformed { .. })
        ));
        assert!(matches!(
            parse_tag(&[0x18, 0, 0]),
            Err(CodecError::Unsupported { .. })
        ));
        // A first partition longer than the frame.
        assert!(matches!(
            parse(&[0x11, 0x20, 0, 0, 0], &Persist::default()),
            Err(CodecError::Truncated { .. })
        ));
    }

    #[test]
    fn a_partition_table_past_the_end_is_truncated() {
        let header = key_header();
        let mut data = write(&header, &Persist::default());
        let first = parse(&data, &Persist::default())
            .expect("parse")
            .header
            .tag
            .first_size;
        data.truncate(10 + first + 4);
        assert!(matches!(
            parse(&data, &Persist::default()),
            Err(CodecError::Truncated { .. })
        ));
    }
}
