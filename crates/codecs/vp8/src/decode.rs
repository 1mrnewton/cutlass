//! Key frame decode: header, modes, tokens, reconstruction, loop filter.
//!
//! The frame is reconstructed whole before the loop filter runs, so intra
//! prediction reads unfiltered neighbours as section 12 requires. The
//! planes are macroblock-aligned while decoding and cropped to the coded
//! size at the end.

use cutlass_decode::CodecError;

use crate::Vp8Picture;
use crate::bool_decoder::BoolDecoder;
use crate::filter::{Planes, Strength, filter_frame, strength};
use crate::predict::{B_PRED, Edges, implied_subblock_mode, predict_block, predict_subblock};
use crate::tables::{AC_QUANT, COEFF_UPDATE_PROBS, DC_QUANT, DEFAULT_COEFF_PROBS, KF_BMODE_PROBS};
use crate::transform::{idct_add, inverse_wht, wrap16};

/// Section 11.2: B_PRED, then DC, V, H, TM.
const KF_YMODE_TREE: [i8; 8] = [-4, 2, 4, 6, 0, -1, -2, -3];
const KF_YMODE_PROBS: [u8; 4] = [145, 156, 163, 128];
/// Section 11.4.
const UV_MODE_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
const KF_UV_MODE_PROBS: [u8; 3] = [142, 114, 183];
/// Section 11.2, in the subblock mode numbering of `predict`.
const BMODE_TREE: [i8; 18] = [
    0, 2, -1, 4, -2, 6, 8, 12, -3, 10, -5, -6, -4, 14, -7, 16, -8, -9,
];
/// Section 9.3.
const SEGMENT_TREE: [i8; 6] = [2, 4, 0, -1, -2, -3];

/// Section 13.2: tokens 0 to 4 are their value, 5 to 10 the categories,
/// 11 the end of the block.
const DCT_EOB: u8 = 11;
const COEFF_TREE: [i8; 22] = [
    -11, 2, 0, 4, -1, 6, 8, 12, -2, 10, -3, -4, 14, 16, -5, -6, 18, 20, -7, -8, -9, -10,
];
/// The extra-bit probabilities and base value of categories 1 to 6.
const CATEGORIES: [(&[u8], i32); 6] = [
    (&[159], 5),
    (&[165, 145], 7),
    (&[173, 148, 140], 11),
    (&[176, 155, 140, 135], 19),
    (&[180, 157, 141, 134, 130], 35),
    (&[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129], 67),
];
/// Section 13.3.
const BANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];
/// Section 13.3: coded order to raster position.
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Plane types of the token probability table.
const TYPE_Y_AFTER_Y2: usize = 0;
const TYPE_Y2: usize = 1;
const TYPE_UV: usize = 2;
const TYPE_Y: usize = 3;

type CoeffProbs = [[[[u8; 11]; 3]; 8]; 4];

/// Width and height from a key frame's uncompressed header.
pub fn frame_size(data: &[u8]) -> Result<(u32, u32), CodecError> {
    if data.len() < 10 {
        return Err(truncated(0, "vp8 frame header"));
    }
    if data[0] & 1 != 0 {
        return Err(CodecError::Unsupported {
            what: "vp8 inter frame",
        });
    }
    if data[3..6] != [0x9D, 0x01, 0x2A] {
        return Err(malformed(3, "vp8 start code"));
    }
    let width = u32::from(u16::from_le_bytes([data[6], data[7]]) & 0x3FFF);
    let height = u32::from(u16::from_le_bytes([data[8], data[9]]) & 0x3FFF);
    if width == 0 || height == 0 {
        return Err(malformed(6, "vp8 frame size"));
    }
    Ok((width, height))
}

#[derive(Default)]
struct Segmentation {
    enabled: bool,
    update_map: bool,
    absolute: bool,
    quant: [i32; 4],
    filter: [i32; 4],
    probs: [u8; 3],
}

/// Step sizes for one segment, `(dc, ac)` per block type.
#[derive(Clone, Copy, Default)]
struct Steps {
    y: (i32, i32),
    y2: (i32, i32),
    uv: (i32, i32),
}

struct Header {
    segmentation: Segmentation,
    simple_filter: bool,
    filter_level: i32,
    sharpness: u8,
    /// Reference-frame and mode loop filter deltas, when enabled.
    filter_deltas: Option<([i32; 4], [i32; 4])>,
    partitions: usize,
    steps: [Steps; 4],
    probs: Box<CoeffProbs>,
    skip_prob: Option<u8>,
}

fn quant_index(value: i32) -> usize {
    usize::try_from(value.clamp(0, 127)).unwrap_or(0)
}

fn read_header(bd: &mut BoolDecoder<'_>) -> Header {
    // Color space and clamping type: the color space has one defined
    // value, and clamping is always done here.
    let _color_space = bd.read_flag();
    let _clamping = bd.read_flag();
    let mut segmentation = Segmentation {
        probs: [255; 3],
        ..Segmentation::default()
    };
    segmentation.enabled = bd.read_flag();
    if segmentation.enabled {
        segmentation.update_map = bd.read_flag();
        let update_data = bd.read_flag();
        if update_data {
            segmentation.absolute = bd.read_flag();
            for value in &mut segmentation.quant {
                *value = bd.read_optional_signed(7);
            }
            for value in &mut segmentation.filter {
                *value = bd.read_optional_signed(6);
            }
        }
        if segmentation.update_map {
            for prob in &mut segmentation.probs {
                *prob = if bd.read_flag() {
                    u8::try_from(bd.read_literal(8)).unwrap_or(255)
                } else {
                    255
                };
            }
        }
    }
    let simple_filter = bd.read_flag();
    let filter_level = i32::try_from(bd.read_literal(6)).unwrap_or(0);
    let sharpness = u8::try_from(bd.read_literal(3)).unwrap_or(0);
    let mut filter_deltas = None;
    if bd.read_flag() {
        let mut deltas = ([0; 4], [0; 4]);
        if bd.read_flag() {
            for value in deltas.0.iter_mut().chain(deltas.1.iter_mut()) {
                if bd.read_flag() {
                    *value = bd.read_signed(6);
                }
            }
        }
        filter_deltas = Some(deltas);
    }
    let partitions = 1_usize << bd.read_literal(2);
    let base = i32::try_from(bd.read_literal(7)).unwrap_or(0);
    let y_dc = bd.read_optional_signed(4);
    let y2_dc = bd.read_optional_signed(4);
    let y2_ac = bd.read_optional_signed(4);
    let uv_dc = bd.read_optional_signed(4);
    let uv_ac = bd.read_optional_signed(4);
    let mut steps = [Steps::default(); 4];
    for (segment, steps) in steps.iter_mut().enumerate() {
        let q = if !segmentation.enabled {
            base
        } else if segmentation.absolute {
            segmentation.quant[segment]
        } else {
            base + segmentation.quant[segment]
        }
        .clamp(0, 127);
        let dc = |delta: i32| i32::from(DC_QUANT[quant_index(q + delta)]);
        let ac = |delta: i32| i32::from(AC_QUANT[quant_index(q + delta)]);
        *steps = Steps {
            y: (dc(y_dc), ac(0)),
            y2: (dc(y2_dc) * 2, (ac(y2_ac) * 155 / 100).max(8)),
            uv: (dc(uv_dc).min(132), ac(uv_ac)),
        };
    }
    // Key frames: refresh_entropy_probs only matters to later frames.
    let _refresh_entropy = bd.read_flag();
    let mut probs = Box::new(DEFAULT_COEFF_PROBS);
    for (plane, updates) in probs.iter_mut().zip(&COEFF_UPDATE_PROBS) {
        for (band, updates) in plane.iter_mut().zip(updates) {
            for (context, updates) in band.iter_mut().zip(updates) {
                for (prob, &update) in context.iter_mut().zip(updates) {
                    if bd.read_bool(update) {
                        *prob = u8::try_from(bd.read_literal(8)).unwrap_or(128);
                    }
                }
            }
        }
    }
    let skip_prob = bd
        .read_flag()
        .then(|| u8::try_from(bd.read_literal(8)).unwrap_or(0));
    Header {
        segmentation,
        simple_filter,
        filter_level,
        sharpness,
        filter_deltas,
        partitions,
        steps,
        probs,
        skip_prob,
    }
}

/// Split the token partitions that follow the first partition. A size that
/// runs past the frame is a truncation.
fn token_partitions(data: &[u8], count: usize) -> Result<Vec<BoolDecoder<'_>>, CodecError> {
    let sizes_len = 3 * (count - 1);
    if data.len() < sizes_len {
        return Err(truncated(0, "vp8 partition sizes"));
    }
    let (sizes, mut rest) = data.split_at(sizes_len);
    let mut decoders = Vec::with_capacity(count);
    for size in sizes.as_chunks::<3>().0 {
        let size = usize::from(size[0]) | usize::from(size[1]) << 8 | usize::from(size[2]) << 16;
        if size > rest.len() {
            return Err(truncated(0, "vp8 token partition"));
        }
        let (part, tail) = rest.split_at(size);
        decoders.push(BoolDecoder::new(part));
        rest = tail;
    }
    // Both reference decoders refuse an empty last partition: every
    // partition holds at least the end of its bool-coded data.
    if rest.is_empty() {
        return Err(truncated(0, "vp8 last token partition"));
    }
    decoders.push(BoolDecoder::new(rest));
    Ok(decoders)
}

/// The coefficients of one macroblock, already dequantized.
struct Residual {
    y: [[i32; 16]; 16],
    u: [[i32; 16]; 4],
    v: [[i32; 16]; 4],
    /// Any block of the macroblock had a token other than an immediate
    /// end of block.
    any: bool,
}

/// Non-zero flags per macroblock edge: four Y, two U, two V, one Y2.
type Context = [bool; 9];

/// Read one block's tokens into `out`, at raster positions, multiplied by
/// `steps`. Returns whether any token came before the end of block.
fn read_block(
    bd: &mut BoolDecoder<'_>,
    probs: &[[[u8; 11]; 3]; 8],
    first: usize,
    context: usize,
    steps: (i32, i32),
    out: &mut [i32; 16],
) -> bool {
    let mut at = first;
    let mut context = context;
    let mut start = 0;
    while at < 16 {
        let token = bd.read_tree(&COEFF_TREE, &probs[BANDS[at]][context], start);
        if token == DCT_EOB {
            break;
        }
        let magnitude = match token {
            0..=4 => i32::from(token),
            _ => {
                let (extra, base) = CATEGORIES[usize::from(token - 5).min(5)];
                let mut value = 0;
                for &prob in extra {
                    value = (value << 1) | i32::from(bd.read_bool(prob));
                }
                base + value
            }
        };
        let (value, next_context, next_start) = if magnitude == 0 {
            (0, 0, 2)
        } else {
            let signed = if bd.read_flag() {
                -magnitude
            } else {
                magnitude
            };
            (signed, if magnitude == 1 { 1 } else { 2 }, 0)
        };
        let step = if at == 0 { steps.0 } else { steps.1 };
        out[ZIGZAG[at]] = wrap16(value * step);
        context = next_context;
        start = next_start;
        at += 1;
    }
    at > first
}

fn read_residual(
    bd: &mut BoolDecoder<'_>,
    probs: &CoeffProbs,
    steps: Steps,
    has_y2: bool,
    above: &mut Context,
    left: &mut Context,
) -> Residual {
    let mut residual = Residual {
        y: [[0; 16]; 16],
        u: [[0; 16]; 4],
        v: [[0; 16]; 4],
        any: false,
    };
    let (y_type, y_first) = if has_y2 {
        let mut y2 = [0; 16];
        let context = usize::from(above[8]) + usize::from(left[8]);
        let coded = read_block(bd, &probs[TYPE_Y2], 0, context, steps.y2, &mut y2);
        above[8] = coded;
        left[8] = coded;
        residual.any |= coded;
        let dcs = inverse_wht(&y2);
        for (block, dc) in residual.y.iter_mut().zip(dcs) {
            block[0] = dc;
        }
        (TYPE_Y_AFTER_Y2, 1)
    } else {
        (TYPE_Y, 0)
    };
    for (n, block) in residual.y.iter_mut().enumerate() {
        let (row, col) = (n / 4, n % 4);
        let context = usize::from(above[col]) + usize::from(left[row]);
        let coded = read_block(bd, &probs[y_type], y_first, context, steps.y, block);
        above[col] = coded;
        left[row] = coded;
        residual.any |= coded;
    }
    for (plane, offset) in [(&mut residual.u, 4), (&mut residual.v, 6)] {
        for (n, block) in plane.iter_mut().enumerate() {
            let (row, col) = (n / 2, n % 2);
            let context = usize::from(above[offset + col]) + usize::from(left[offset + row]);
            let coded = read_block(bd, &probs[TYPE_UV], 0, context, steps.uv, block);
            above[offset + col] = coded;
            left[offset + row] = coded;
            residual.any |= coded;
        }
    }
    residual
}

/// One plane, macroblock-aligned.
struct Plane {
    data: Vec<u8>,
    stride: usize,
}

impl Plane {
    fn at(&self, x: usize, y: usize) -> u8 {
        self.data.get(y * self.stride + x).copied().unwrap_or(0)
    }

    /// The edges of the `size`-square block at macroblock (`col`, `row`):
    /// the row above, the column to the left, the corner, and the four
    /// pixels above and to the right.
    fn edges(&self, col: usize, row: usize, size: usize, cols: usize) -> BlockEdges {
        let (x, y) = (col * size, row * size);
        let mut edges = BlockEdges {
            above: [127; 16],
            left: [129; 16],
            corner: if row == 0 { 127 } else { 129 },
            above_right: [127; 4],
        };
        if row > 0 {
            for (i, pixel) in edges.above[..size].iter_mut().enumerate() {
                *pixel = self.at(x + i, y - 1);
            }
            if col > 0 {
                edges.corner = self.at(x - 1, y - 1);
            }
            for (i, pixel) in edges.above_right.iter_mut().enumerate() {
                *pixel = if col + 1 < cols {
                    self.at(x + size + i, y - 1)
                } else {
                    self.at(x + size - 1, y - 1)
                };
            }
        }
        if col > 0 {
            for (i, pixel) in edges.left[..size].iter_mut().enumerate() {
                *pixel = self.at(x - 1, y + i);
            }
        }
        edges
    }

    fn write(&mut self, col: usize, row: usize, size: usize, block: &[u8]) {
        let (x, y) = (col * size, row * size);
        for (r, line) in block.chunks_exact(size).enumerate() {
            let at = (y + r) * self.stride + x;
            if let Some(target) = self.data.get_mut(at..at + size) {
                target.copy_from_slice(line);
            }
        }
    }
}

struct BlockEdges {
    above: [u8; 16],
    left: [u8; 16],
    corner: u8,
    above_right: [u8; 4],
}

/// Workspace stride for B_PRED: the corner, sixteen pixels, and four
/// above-right pixels.
const WS: usize = 21;

fn reconstruct_luma(
    plane: &mut Plane,
    (col, row, cols): (usize, usize, usize),
    ymode: u8,
    bmodes: &[u8; 16],
    residual: &Residual,
) {
    let edges = plane.edges(col, row, 16, cols);
    if ymode != B_PRED {
        let mut block = [0_u8; 256];
        predict_block(
            ymode,
            &Edges {
                above: &edges.above,
                left: &edges.left,
                corner: edges.corner,
                has_above: row > 0,
                has_left: col > 0,
            },
            16,
            &mut block,
        );
        for (n, coeffs) in residual.y.iter().enumerate() {
            if coeffs.iter().any(|&c| c != 0) {
                idct_add(coeffs, &mut block, (n / 4) * 64 + (n % 4) * 4, 16);
            }
        }
        plane.write(col, row, 16, &block);
        return;
    }
    let mut ws = [0_u8; WS * 17];
    ws[0] = edges.corner;
    ws[1..17].copy_from_slice(&edges.above);
    ws[17..21].copy_from_slice(&edges.above_right);
    for (r, &left) in edges.left.iter().enumerate() {
        ws[(r + 1) * WS] = left;
    }
    // Subblocks on the right edge below the top row use the same pixels
    // above and to the right as subblock 3 (section 12.3).
    for r in [4, 8, 12] {
        ws[r * WS + 17..r * WS + 21].copy_from_slice(&edges.above_right);
    }
    for (n, &mode) in bmodes.iter().enumerate() {
        let (by, bx) = (n / 4, n % 4);
        let top = 4 * by * WS + 4 * bx;
        let mut above = [0_u8; 9];
        above.copy_from_slice(&ws[top..top + 9]);
        let mut left = [0_u8; 4];
        for (r, pixel) in left.iter_mut().enumerate() {
            *pixel = ws[(4 * by + 1 + r) * WS + 4 * bx];
        }
        let predicted = predict_subblock(mode, &above, &left);
        let origin = (4 * by + 1) * WS + 4 * bx + 1;
        for (r, line) in predicted.iter().enumerate() {
            ws[origin + r * WS..origin + r * WS + 4].copy_from_slice(line);
        }
        let coeffs = &residual.y[n];
        if coeffs.iter().any(|&c| c != 0) {
            idct_add(coeffs, &mut ws, origin, WS);
        }
    }
    let mut block = [0_u8; 256];
    for (r, line) in block.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let at = (r + 1) * WS + 1;
        line.copy_from_slice(&ws[at..at + 16]);
    }
    plane.write(col, row, 16, &block);
}

fn reconstruct_chroma(
    plane: &mut Plane,
    (col, row, cols): (usize, usize, usize),
    mode: u8,
    blocks: &[[i32; 16]; 4],
) {
    let edges = plane.edges(col, row, 8, cols);
    let mut block = [0_u8; 64];
    predict_block(
        mode,
        &Edges {
            above: &edges.above,
            left: &edges.left,
            corner: edges.corner,
            has_above: row > 0,
            has_left: col > 0,
        },
        8,
        &mut block,
    );
    for (n, coeffs) in blocks.iter().enumerate() {
        if coeffs.iter().any(|&c| c != 0) {
            idct_add(coeffs, &mut block, (n / 2) * 32 + (n % 2) * 4, 8);
        }
    }
    plane.write(col, row, 8, &block);
}

fn filter_level(header: &Header, segment: usize, ymode: u8) -> u8 {
    let mut level = header.filter_level;
    let seg = &header.segmentation;
    if seg.enabled {
        level = if seg.absolute {
            seg.filter[segment]
        } else {
            level + seg.filter[segment]
        }
        .clamp(0, 63);
    }
    if let Some((reference, mode)) = header.filter_deltas {
        // Every key frame macroblock is intra; B_PRED has its own delta.
        level += reference[0];
        if ymode == B_PRED {
            level += mode[0];
        }
    }
    u8::try_from(level.clamp(0, 63)).unwrap_or(0)
}

/// Decode one key frame into cropped 4:2:0 planes.
pub fn decode_frame(data: &[u8], max_pixels: usize) -> Result<Vp8Picture, CodecError> {
    let (width, height) = frame_size(data)?;
    let tag = u32::from(data[0]) | u32::from(data[1]) << 8 | u32::from(data[2]) << 16;
    let version = (tag >> 1) & 7;
    let shown = (tag >> 4) & 1 == 1;
    let first_size = usize::try_from(tag >> 5).unwrap_or(usize::MAX);
    if version > 3 {
        return Err(CodecError::Unsupported {
            what: "vp8 version",
        });
    }
    if !shown {
        return Err(CodecError::Unsupported {
            what: "vp8 hidden frame",
        });
    }
    let (w, h) = (
        usize::try_from(width).unwrap_or(usize::MAX),
        usize::try_from(height).unwrap_or(usize::MAX),
    );
    if w.checked_mul(h).is_none_or(|pixels| pixels > max_pixels) {
        return Err(CodecError::Limit {
            what: "vp8 frame size",
        });
    }
    let body = &data[10..];
    if first_size > body.len() {
        return Err(truncated(10, "vp8 first partition"));
    }
    let (first, rest) = body.split_at(first_size);
    let mut bd = BoolDecoder::new(first);
    let header = read_header(&mut bd);
    bd.check("vp8 frame header")?;
    let mut tokens = token_partitions(rest, header.partitions)?;

    let (cols, rows) = (w.div_ceil(16), h.div_ceil(16));
    let mut y = Plane {
        data: vec![0; cols * 16 * rows * 16],
        stride: cols * 16,
    };
    let mut u = Plane {
        data: vec![0; cols * 8 * rows * 8],
        stride: cols * 8,
    };
    let mut v = Plane {
        data: vec![0; cols * 8 * rows * 8],
        stride: cols * 8,
    };
    let mut above_modes = vec![0_u8; cols * 4];
    let mut above_context = vec![[false; 9]; cols];
    let mut strengths = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        let mut left_modes = [0_u8; 4];
        let mut left_context: Context = [false; 9];
        let partition = tokens.len().max(1);
        let Some(token_bd) = tokens.get_mut(row % partition) else {
            return Err(malformed(0, "vp8 partitions"));
        };
        for col in 0..cols {
            let segment = if header.segmentation.update_map {
                usize::from(bd.read_tree(&SEGMENT_TREE, &header.segmentation.probs, 0))
            } else {
                0
            };
            let skip = header.skip_prob.is_some_and(|prob| bd.read_bool(prob));
            let ymode = bd.read_tree(&KF_YMODE_TREE, &KF_YMODE_PROBS, 0);
            let mut bmodes = [implied_subblock_mode(ymode); 16];
            if ymode == B_PRED {
                for n in 0..16 {
                    let (by, bx) = (n / 4, n % 4);
                    let above = if by == 0 {
                        above_modes[col * 4 + bx]
                    } else {
                        bmodes[n - 4]
                    };
                    let left = if bx == 0 {
                        left_modes[by]
                    } else {
                        bmodes[n - 1]
                    };
                    let probs =
                        &KF_BMODE_PROBS[usize::from(above).min(9)][usize::from(left).min(9)];
                    bmodes[n] = bd.read_tree(&BMODE_TREE, probs, 0);
                }
            }
            above_modes[col * 4..col * 4 + 4].copy_from_slice(&bmodes[12..16]);
            left_modes = [bmodes[3], bmodes[7], bmodes[11], bmodes[15]];
            let uvmode = bd.read_tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS, 0);
            let has_y2 = ymode != B_PRED;
            let above_ctx = &mut above_context[col];
            let residual = if skip {
                let keep_y2 = (above_ctx[8], left_context[8]);
                *above_ctx = [false; 9];
                left_context = [false; 9];
                if !has_y2 {
                    above_ctx[8] = keep_y2.0;
                    left_context[8] = keep_y2.1;
                }
                Residual {
                    y: [[0; 16]; 16],
                    u: [[0; 16]; 4],
                    v: [[0; 16]; 4],
                    any: false,
                }
            } else {
                read_residual(
                    token_bd,
                    &header.probs,
                    header.steps[segment.min(3)],
                    has_y2,
                    above_ctx,
                    &mut left_context,
                )
            };
            let place = (col, row, cols);
            reconstruct_luma(&mut y, place, ymode, &bmodes, &residual);
            reconstruct_chroma(&mut u, place, uvmode, &residual.u);
            reconstruct_chroma(&mut v, place, uvmode, &residual.v);
            let level = filter_level(&header, segment.min(3), ymode);
            strengths.push(if header.filter_level == 0 {
                Strength::default()
            } else {
                strength(level, header.sharpness, ymode == B_PRED || residual.any)
            });
        }
        bd.check("vp8 modes")?;
        token_bd.check("vp8 tokens")?;
    }
    filter_frame(
        &mut Planes {
            y: &mut y.data,
            u: &mut u.data,
            v: &mut v.data,
            mb_cols: cols,
            mb_rows: rows,
        },
        &strengths,
        header.simple_filter,
    );
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    Ok(Vp8Picture {
        width,
        height,
        y: crop(&y, w, h),
        u: crop(&u, cw, ch),
        v: crop(&v, cw, ch),
    })
}

fn crop(plane: &Plane, width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height);
    for line in plane.data.chunks_exact(plane.stride).take(height) {
        out.extend_from_slice(&line[..width]);
    }
    out
}

fn truncated(at: usize, what: &'static str) -> CodecError {
    CodecError::Truncated {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}

fn malformed(at: usize, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}
