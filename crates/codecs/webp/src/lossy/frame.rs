//! Decode one VP8 key frame to YUV 4:2:0 planes (RFC 6386 sections 11 to
//! 15). The whole frame is reconstructed first, because intra prediction
//! reads unfiltered pixels, and then loop-filtered in macroblock order.

use cutlass_decode::CodecError;

use super::bool_decoder::BoolDecoder;
use super::dsp::{self, Edge};
use super::header::{self, FilterKind, FilterStrength, FrameHeader, Quant};
use super::predict::{self, B_DC, B_HE, B_TM, B_VE, Edges};
use super::tables::{B_MODE_TREE, BANDS, CAT3, CAT4, CAT5, CAT6, KF_B_MODE_PROBS, ZIGZAG};
use super::yuv::Planes;

/// Token probabilities of one block type: `[band][context][node]`.
type TypeProbs = [[[u8; 11]; 3]; 8];

/// Modes of one macroblock, from the first partition.
#[derive(Clone, Copy)]
struct Modes {
    segment: usize,
    skip: bool,
    is_4x4: bool,
    /// The 16x16 mode in entry 0, or the sixteen sub-block modes.
    luma: [u8; 16],
    chroma: u8,
}

/// Which neighbouring blocks had non-zero coefficients (section 13.3).
#[derive(Clone, Copy, Default)]
struct NonZero {
    y: [bool; 4],
    u: [bool; 2],
    v: [bool; 2],
    y2: bool,
}

fn eof(what: &'static str) -> CodecError {
    CodecError::Truncated { at_bit: 0, what }
}

pub(crate) fn decode(data: &[u8]) -> Result<Planes, CodecError> {
    let FrameHeader {
        width,
        height,
        mut first,
        partitions,
        update_segment_map,
        segment_probs,
        quant,
        filter,
        strengths,
        coeff_probs,
        skip_prob,
    } = header::parse(data)?;
    let width = width as usize;
    let height = height as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let mut planes = Planes {
        width,
        height,
        y: vec![0; y_stride * mb_h * 16],
        y_stride,
        u: vec![0; uv_stride * mb_h * 8],
        v: vec![0; uv_stride * mb_h * 8],
        uv_stride,
    };
    let mut parts: Vec<BoolDecoder<'_>> = partitions.iter().map(|p| BoolDecoder::new(p)).collect();
    let part_mask = parts.len() - 1;
    let mut top_modes = vec![[B_DC; 4]; mb_w];
    let mut top_nz = vec![NonZero::default(); mb_w];
    let mut filters = Vec::with_capacity(if filter == FilterKind::None {
        0
    } else {
        mb_w * mb_h
    });
    let mut row_modes = Vec::with_capacity(mb_w);
    for mb_y in 0..mb_h {
        row_modes.clear();
        let mut left_modes = [B_DC; 4];
        for top in &mut top_modes {
            row_modes.push(read_modes(
                &mut first,
                update_segment_map,
                &segment_probs,
                skip_prob,
                top,
                &mut left_modes,
            ));
        }
        if first.eof() {
            return Err(eof("vp8 first partition"));
        }
        let tokens = &mut parts[mb_y & part_mask];
        let mut left_nz = NonZero::default();
        for (mb_x, modes) in row_modes.iter().enumerate() {
            let mut coeffs = [[0i16; 16]; 24];
            let nz = &mut top_nz[mb_x];
            let has_coeffs = if modes.skip {
                let y2 = (nz.y2, left_nz.y2);
                *nz = NonZero::default();
                left_nz = NonZero::default();
                if modes.is_4x4 {
                    (nz.y2, left_nz.y2) = y2;
                }
                [false; 24]
            } else {
                residuals(
                    tokens,
                    &coeff_probs,
                    &quant[modes.segment],
                    modes.is_4x4,
                    nz,
                    &mut left_nz,
                    &mut coeffs,
                )
            };
            if tokens.eof() {
                return Err(eof("vp8 token partition"));
            }
            reconstruct(&mut planes, mb_x, mb_y, modes, &coeffs, &has_coeffs);
            if filter != FilterKind::None {
                let strength = strengths[modes.segment][usize::from(modes.is_4x4)];
                let inner = modes.is_4x4 || has_coeffs.iter().any(|&nz| nz);
                filters.push((strength, inner));
            }
        }
    }
    if filter != FilterKind::None {
        loop_filter(&mut planes, mb_w, filter, &filters);
    }
    Ok(planes)
}

fn read_modes(
    br: &mut BoolDecoder<'_>,
    update_segment_map: bool,
    segment_probs: &[u8; 3],
    skip_prob: Option<u8>,
    top: &mut [u8; 4],
    left: &mut [u8; 4],
) -> Modes {
    let segment = if update_segment_map {
        if br.bit(segment_probs[0]) {
            2 + usize::from(br.bit(segment_probs[2]))
        } else {
            usize::from(br.bit(segment_probs[1]))
        }
    } else {
        0
    };
    let skip = skip_prob.is_some_and(|prob| br.bit(prob));
    let mut luma = [B_DC; 16];
    // Section 11.2, `kf_ymode_tree`.
    let is_4x4 = !br.bit(145);
    if is_4x4 {
        for row in 0..4 {
            let mut mode = left[row];
            for (col, above) in top.iter_mut().enumerate() {
                let probs = &KF_B_MODE_PROBS[usize::from(*above)][usize::from(mode)];
                mode = br.tree(&B_MODE_TREE, probs) as u8;
                *above = mode;
                luma[row * 4 + col] = mode;
            }
            left[row] = mode;
        }
    } else {
        let mode = if br.bit(156) {
            if br.bit(128) { B_TM } else { B_HE }
        } else if br.bit(163) {
            B_VE
        } else {
            B_DC
        };
        luma[0] = mode;
        // Section 11.3: a 16x16 mode stands for the matching sub-block mode.
        *top = [mode; 4];
        *left = [mode; 4];
    }
    // Section 11.2, `uv_mode_tree`.
    let chroma = if !br.bit(142) {
        B_DC
    } else if !br.bit(114) {
        B_VE
    } else if br.bit(183) {
        B_TM
    } else {
        B_HE
    };
    Modes {
        segment,
        skip,
        is_4x4,
        luma,
        chroma,
    }
}

/// Section 13.2: a token of category 2 or above.
fn large_value(br: &mut BoolDecoder<'_>, p: &[u8; 11]) -> i32 {
    if !br.bit(p[3]) {
        if !br.bit(p[4]) {
            2
        } else {
            3 + i32::from(br.bit(p[5]))
        }
    } else if !br.bit(p[6]) {
        if !br.bit(p[7]) {
            5 + i32::from(br.bit(159))
        } else {
            7 + 2 * i32::from(br.bit(165)) + i32::from(br.bit(145))
        }
    } else {
        let high = usize::from(br.bit(p[8]));
        let low = usize::from(br.bit(p[9 + high]));
        let category = 2 * high + low;
        let extra: &[u8] = match category {
            0 => &CAT3,
            1 => &CAT4,
            2 => &CAT5,
            _ => &CAT6,
        };
        let mut value = 0;
        for &prob in extra {
            value = 2 * value + i32::from(br.bit(prob));
        }
        value + 3 + (8 << category)
    }
}

/// Section 13: the tokens of one block from scan position `first`,
/// dequantized into raster order. Returns one past the last non-zero
/// position, or `first` for an empty block.
fn coefficients(
    br: &mut BoolDecoder<'_>,
    probs: &TypeProbs,
    context: usize,
    dq: [i32; 2],
    first: usize,
    out: &mut [i16; 16],
) -> usize {
    let mut n = first;
    let mut p = &probs[BANDS[n]][context];
    while n < 16 {
        if !br.bit(p[0]) {
            return n;
        }
        while !br.bit(p[1]) {
            n += 1;
            if n == 16 {
                return 16;
            }
            p = &probs[BANDS[n]][0];
        }
        let (value, next_context) = if br.bit(p[2]) {
            (large_value(br, p), 2)
        } else {
            (1, 1)
        };
        let value = if br.bit(128) { -value } else { value };
        // Stored in 16 bits, as the reference decoder does.
        out[ZIGZAG[n]] = (value * dq[usize::from(n > 0)]) as i16;
        n += 1;
        p = &probs[BANDS[n]][next_context];
    }
    16
}

/// The coefficients of one macroblock: sixteen luma blocks, four U, four V.
/// Returns which blocks have any non-zero coefficient.
fn residuals(
    br: &mut BoolDecoder<'_>,
    probs: &[TypeProbs; 4],
    quant: &Quant,
    is_4x4: bool,
    top: &mut NonZero,
    left: &mut NonZero,
    coeffs: &mut [[i16; 16]; 24],
) -> [bool; 24] {
    let mut has = [false; 24];
    let (first, luma_type) = if is_4x4 {
        (0, 3)
    } else {
        let mut y2 = [0i16; 16];
        let context = usize::from(top.y2) + usize::from(left.y2);
        let nz = coefficients(br, &probs[1], context, quant.y2, 0, &mut y2);
        top.y2 = nz > 0;
        left.y2 = nz > 0;
        let dcs = if nz > 1 {
            dsp::inverse_wht(&y2)
        } else {
            [((i32::from(y2[0]) + 3) >> 3) as i16; 16]
        };
        for (block, dc) in coeffs.iter_mut().zip(dcs) {
            block[0] = dc;
        }
        (1, 0)
    };
    for row in 0..4 {
        let mut l = left.y[row];
        for col in 0..4 {
            let index = row * 4 + col;
            let context = usize::from(l) + usize::from(top.y[col]);
            let block = &mut coeffs[index];
            let nz = coefficients(br, &probs[luma_type], context, quant.y1, first, block);
            l = nz > first;
            top.y[col] = l;
            has[index] = nz > 1 || block[0] != 0;
        }
        left.y[row] = l;
    }
    for (plane, offset) in [(0, 16), (1, 20)] {
        let (top_plane, left_plane) = if plane == 0 {
            (&mut top.u, &mut left.u)
        } else {
            (&mut top.v, &mut left.v)
        };
        for (row, left_nz) in left_plane.iter_mut().enumerate() {
            let mut l = *left_nz;
            for (col, top_nz) in top_plane.iter_mut().enumerate() {
                let index = offset + row * 2 + col;
                let context = usize::from(l) + usize::from(*top_nz);
                let block = &mut coeffs[index];
                let nz = coefficients(br, &probs[2], context, quant.uv, 0, block);
                l = nz > 0;
                *top_nz = l;
                has[index] = nz > 1 || block[0] != 0;
            }
            *left_nz = l;
        }
    }
    has
}

/// Pitch of the luma work block: the left column, sixteen pixels, and
/// four pixels above-right.
const LUMA_PITCH: usize = 21;
const CHROMA_PITCH: usize = 9;

/// Fill the edges of a work block from the frame: 127 above the picture,
/// 129 left of it, and the corner from whichever applies.
fn load_edges(
    work: &mut [u8],
    pitch: usize,
    size: usize,
    plane: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
) {
    let x0 = mb_x * size;
    let y0 = mb_y * size;
    if mb_y == 0 {
        work[..pitch].fill(127);
    } else {
        let above = (y0 - 1) * stride;
        work[1..=size].copy_from_slice(&plane[above + x0..above + x0 + size]);
        work[0] = if mb_x == 0 {
            129
        } else {
            plane[above + x0 - 1]
        };
    }
    for row in 0..size {
        work[(row + 1) * pitch] = if mb_x == 0 {
            129
        } else {
            plane[(y0 + row) * stride + x0 - 1]
        };
    }
}

fn store(
    work: &[u8],
    pitch: usize,
    size: usize,
    plane: &mut [u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
) {
    for row in 0..size {
        let from = (row + 1) * pitch + 1;
        let to = (mb_y * size + row) * stride + mb_x * size;
        plane[to..to + size].copy_from_slice(&work[from..from + size]);
    }
}

fn reconstruct(
    planes: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    modes: &Modes,
    coeffs: &[[i16; 16]; 24],
    has: &[bool; 24],
) {
    let mb_w = planes.y_stride / 16;
    let edges = Edges {
        above: mb_y > 0,
        left: mb_x > 0,
    };
    let mut work = [0u8; LUMA_PITCH * 17];
    load_edges(
        &mut work,
        LUMA_PITCH,
        16,
        &planes.y,
        planes.y_stride,
        mb_x,
        mb_y,
    );
    if modes.is_4x4 {
        // Above-right of the macroblock, which sub-blocks on the right
        // column reuse on every row (section 12.3).
        if mb_y > 0 {
            let above = (mb_y * 16 - 1) * planes.y_stride;
            if mb_x + 1 < mb_w {
                let start = above + (mb_x + 1) * 16;
                work[17..21].copy_from_slice(&planes.y[start..start + 4]);
            } else {
                let last = work[16];
                work[17..21].fill(last);
            }
        }
        for row in [4, 8, 12] {
            work.copy_within(17..21, row * LUMA_PITCH + 17);
        }
        for (index, block) in coeffs[..16].iter().enumerate() {
            let at = (1 + (index / 4) * 4) * LUMA_PITCH + 1 + (index % 4) * 4;
            predict::sub_block(&mut work, at, LUMA_PITCH, modes.luma[index]);
            if has[index] {
                dsp::add_idct(block, &mut work, at, LUMA_PITCH);
            }
        }
    } else {
        predict::block(
            &mut work,
            LUMA_PITCH + 1,
            LUMA_PITCH,
            16,
            modes.luma[0],
            edges,
        );
        for (index, block) in coeffs[..16].iter().enumerate() {
            if has[index] {
                let at = (1 + (index / 4) * 4) * LUMA_PITCH + 1 + (index % 4) * 4;
                dsp::add_idct(block, &mut work, at, LUMA_PITCH);
            }
        }
    }
    store(
        &work,
        LUMA_PITCH,
        16,
        &mut planes.y,
        planes.y_stride,
        mb_x,
        mb_y,
    );

    for (plane, offset) in [(&mut planes.u, 16), (&mut planes.v, 20)] {
        let mut work = [0u8; CHROMA_PITCH * 9];
        load_edges(
            &mut work,
            CHROMA_PITCH,
            8,
            plane,
            planes.uv_stride,
            mb_x,
            mb_y,
        );
        predict::block(
            &mut work,
            CHROMA_PITCH + 1,
            CHROMA_PITCH,
            8,
            modes.chroma,
            edges,
        );
        for (index, block) in coeffs[offset..offset + 4].iter().enumerate() {
            if has[offset + index] {
                let at = (1 + (index / 2) * 4) * CHROMA_PITCH + 1 + (index % 2) * 4;
                dsp::add_idct(block, &mut work, at, CHROMA_PITCH);
            }
        }
        store(&work, CHROMA_PITCH, 8, plane, planes.uv_stride, mb_x, mb_y);
    }
}

/// Section 15: filter every macroblock in raster order, left edge, inner
/// vertical edges, top edge, inner horizontal edges.
fn loop_filter(
    planes: &mut Planes,
    mb_w: usize,
    kind: FilterKind,
    filters: &[(FilterStrength, bool)],
) {
    let ys = planes.y_stride;
    let uvs = planes.uv_stride;
    for (index, &(strength, inner)) in filters.iter().enumerate() {
        if strength.limit == 0 {
            continue;
        }
        let (mb_x, mb_y) = (index % mb_w, index / mb_w);
        let y_at = mb_y * 16 * ys + mb_x * 16;
        let uv_at = mb_y * 8 * uvs + mb_x * 8;
        let FilterStrength {
            limit,
            interior,
            hev,
        } = strength;
        // Vertical edges are crossed horizontally; horizontal ones down.
        let across = |at: usize, stride: usize, length: usize| Edge {
            at,
            step: 1,
            pitch: stride,
            length,
        };
        let down = |at: usize, stride: usize, length: usize| Edge {
            at,
            step: stride,
            pitch: 1,
            length,
        };
        if kind == FilterKind::Simple {
            if mb_x > 0 {
                dsp::simple(&mut planes.y, across(y_at, ys, 16), limit + 4);
            }
            if inner {
                for k in [4, 8, 12] {
                    dsp::simple(&mut planes.y, across(y_at + k, ys, 16), limit);
                }
            }
            if mb_y > 0 {
                dsp::simple(&mut planes.y, down(y_at, ys, 16), limit + 4);
            }
            if inner {
                for k in [4, 8, 12] {
                    dsp::simple(&mut planes.y, down(y_at + k * ys, ys, 16), limit);
                }
            }
            continue;
        }
        let edge_limit = limit + 4;
        if mb_x > 0 {
            dsp::normal(
                &mut planes.y,
                across(y_at, ys, 16),
                edge_limit,
                interior,
                hev,
                true,
            );
            dsp::normal(
                &mut planes.u,
                across(uv_at, uvs, 8),
                edge_limit,
                interior,
                hev,
                true,
            );
            dsp::normal(
                &mut planes.v,
                across(uv_at, uvs, 8),
                edge_limit,
                interior,
                hev,
                true,
            );
        }
        if inner {
            for k in [4, 8, 12] {
                dsp::normal(
                    &mut planes.y,
                    across(y_at + k, ys, 16),
                    limit,
                    interior,
                    hev,
                    false,
                );
            }
            dsp::normal(
                &mut planes.u,
                across(uv_at + 4, uvs, 8),
                limit,
                interior,
                hev,
                false,
            );
            dsp::normal(
                &mut planes.v,
                across(uv_at + 4, uvs, 8),
                limit,
                interior,
                hev,
                false,
            );
        }
        if mb_y > 0 {
            dsp::normal(
                &mut planes.y,
                down(y_at, ys, 16),
                edge_limit,
                interior,
                hev,
                true,
            );
            dsp::normal(
                &mut planes.u,
                down(uv_at, uvs, 8),
                edge_limit,
                interior,
                hev,
                true,
            );
            dsp::normal(
                &mut planes.v,
                down(uv_at, uvs, 8),
                edge_limit,
                interior,
                hev,
                true,
            );
        }
        if inner {
            for k in [4, 8, 12] {
                dsp::normal(
                    &mut planes.y,
                    down(y_at + k * ys, ys, 16),
                    limit,
                    interior,
                    hev,
                    false,
                );
            }
            dsp::normal(
                &mut planes.u,
                down(uv_at + 4 * uvs, uvs, 8),
                limit,
                interior,
                hev,
                false,
            );
            dsp::normal(
                &mut planes.v,
                down(uv_at + 4 * uvs, uvs, 8),
                limit,
                interior,
                hev,
                false,
            );
        }
    }
}
