//! Macroblock layer, clauses 7.3.5 and 8. Reconstruction is raster order inside the slice.

use cutlass_decode::CodecError;

use crate::cabac::{self, Cabac};
use crate::cavlc;
use crate::recon;
use crate::set::ZIGZAG4;

use super::{Entropy, Frame, Header, Mb, Mod, Slice, blk_x, blk_y, put_block};

#[derive(Clone, Copy)]
struct Place {
    mx: usize,
    my: usize,
}

#[derive(Clone, Copy)]
struct Cell {
    bx: usize,
    by: usize,
}

struct LumaPaint<'a> {
    i16: bool,
    mode16: u8,
    chroma_mode: u8,
    dc: [i32; 16],
    ac: &'a [[i32; 16]; 16],
    block8: &'a [[i32; 64]; 4],
}

struct ChromaPaint<'a> {
    cb_dc: [i32; 4],
    cr_dc: [i32; 4],
    cb_ac: &'a [[i32; 16]; 4],
    cr_ac: &'a [[i32; 16]; 4],
}

pub(crate) fn reference_lists(
    dpb: &[Frame],
    header: &Header,
    poc: i32,
    max_fn: u32,
) -> (Vec<usize>, Vec<usize>) {
    let short = |frame: &&Frame| frame.referenced && !frame.long_term && !frame.nonexisting;
    let mut shorts: Vec<usize> = dpb
        .iter()
        .enumerate()
        .filter(|(_, frame)| short(frame))
        .map(|(index, _)| index)
        .collect();
    let mut longs: Vec<usize> = dpb
        .iter()
        .enumerate()
        .filter(|(_, frame)| frame.referenced && frame.long_term)
        .map(|(index, _)| index)
        .collect();
    longs.sort_by_key(|&index| dpb[index].long_idx);
    if header.kind == 2 {
        return (Vec::new(), Vec::new());
    }
    if header.kind == 0 {
        shorts.sort_by_key(|&index| {
            std::cmp::Reverse(pic_num(&dpb[index], header.frame_num, max_fn))
        });
        let mut list = shorts;
        list.extend(longs);
        list.truncate(header.num_ref_l0);
        modify(&mut list, dpb, &header.ref_mod_l0, header.frame_num, max_fn);
        return (list, Vec::new());
    }
    let mut past: Vec<usize> = shorts
        .iter()
        .copied()
        .filter(|&index| dpb[index].poc < poc)
        .collect();
    let mut future: Vec<usize> = shorts
        .iter()
        .copied()
        .filter(|&index| dpb[index].poc > poc)
        .collect();
    past.sort_by_key(|&index| std::cmp::Reverse(dpb[index].poc));
    future.sort_by_key(|&index| dpb[index].poc);
    let mut list0 = past.clone();
    list0.extend(future.iter().copied());
    list0.extend(longs.iter().copied());
    let mut list1 = future;
    list1.extend(past);
    list1.extend(longs);
    list0.truncate(header.num_ref_l0);
    list1.truncate(header.num_ref_l1);
    if list0 == list1 && list1.len() > 1 {
        list1.swap(0, 1);
    }
    modify(
        &mut list0,
        dpb,
        &header.ref_mod_l0,
        header.frame_num,
        max_fn,
    );
    modify(
        &mut list1,
        dpb,
        &header.ref_mod_l1,
        header.frame_num,
        max_fn,
    );
    (list0, list1)
}

fn pic_num(frame: &Frame, current: u32, max_fn: u32) -> i32 {
    let mut number = i32::try_from(frame.frame_num).unwrap_or(0);
    if frame.frame_num > current {
        number -= i32::try_from(max_fn).unwrap_or(0);
    }
    number
}

fn modify(list: &mut [usize], dpb: &[Frame], mods: &[Mod], current: u32, max_fn: u32) {
    if mods.is_empty() || list.is_empty() {
        return;
    }
    let mut pred = i32::try_from(current).unwrap_or(0);
    let mut written = 0usize;
    for modification in mods {
        let found = match modification.idc {
            0 => {
                pred -= i32::try_from(modification.arg).unwrap_or(0) + 1;
                find_short(dpb, list, pred, current, max_fn)
            }
            1 => {
                pred += i32::try_from(modification.arg).unwrap_or(0) + 1;
                find_short(dpb, list, pred, current, max_fn)
            }
            2 => dpb.iter().position(|frame| {
                frame.referenced
                    && frame.long_term
                    && frame.long_idx == i32::try_from(modification.arg).unwrap_or(-1)
            }),
            _ => None,
        };
        let Some(picture) = found else {
            continue;
        };
        if written >= list.len() {
            break;
        }
        let mut cursor = picture;
        for slot in list.iter_mut().skip(written) {
            std::mem::swap(slot, &mut cursor);
            if cursor == picture {
                break;
            }
        }
        written += 1;
    }
}

fn find_short(
    dpb: &[Frame],
    list: &[usize],
    number: i32,
    current: u32,
    max_fn: u32,
) -> Option<usize> {
    list.iter()
        .copied()
        .find(|&index| !dpb[index].long_term && pic_num(&dpb[index], current, max_fn) == number)
}

pub(crate) fn decode_mbs(
    pic: &mut Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    slice_id: u16,
    next_mb: &mut usize,
    mut entropy: Entropy<'_, '_>,
) -> Result<(), CodecError> {
    let total = pic.mb_w.saturating_mul(pic.h / 16);
    let mut addr = *next_mb;
    let mut qp = slice.qp;
    while addr < total {
        if slice.kind != 2 {
            match &mut entropy {
                Entropy::Cavlc(bits) => {
                    let run = bits.ue()? as usize;
                    if addr + run > total {
                        return Err(bits.malformed("skip run"));
                    }
                    for _ in 0..run {
                        skip_mb(pic, dpb, slice, slice_id, addr, qp)?;
                        addr += 1;
                    }
                    if addr >= total || (run > 0 && !bits.more_rbsp()) {
                        break;
                    }
                }
                Entropy::Cabac(cab) => {
                    let skipped = cabac_skip(pic, cab, slice.kind, slice_id, addr)?;
                    if skipped {
                        skip_mb(pic, dpb, slice, slice_id, addr, qp)?;
                        let end = cab.terminate()?;
                        addr += 1;
                        if end {
                            break;
                        }
                        continue;
                    }
                }
            }
        }
        qp = one_mb(pic, dpb, slice, slice_id, addr, qp, &mut entropy)?;
        match &mut entropy {
            Entropy::Cavlc(bits) => {
                addr += 1;
                if !bits.more_rbsp() {
                    break;
                }
            }
            Entropy::Cabac(cab) => {
                let end = cab.terminate()?;
                addr += 1;
                if end {
                    break;
                }
            }
        }
    }
    *next_mb = addr;
    Ok(())
}

fn cabac_skip(
    pic: &Frame,
    cab: &mut Cabac<'_>,
    kind: u8,
    slice_id: u16,
    addr: usize,
) -> Result<bool, CodecError> {
    let (mx, my) = xy(pic, addr);
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice_id);
    let above = neighbor(pic, mx as i32, my as i32 - 1, slice_id);
    let mut ctx = 0;
    if left.is_some_and(|mb| !mb.skip) {
        ctx += 1;
    }
    if above.is_some_and(|mb| !mb.skip) {
        ctx += 1;
    }
    if kind == 1 {
        ctx += 13;
    }
    Ok(cab.decision(11 + ctx)? == 1)
}

fn skip_mb(
    pic: &mut Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    slice_id: u16,
    addr: usize,
    qp: i32,
) -> Result<(), CodecError> {
    let (mx, my) = xy(pic, addr);
    let mut mb = Mb::new(slice_id);
    mb.skip = true;
    mb.qp = qp as i8;
    mb.qp_c = recon::chroma_qp(qp, slice.pps.chroma_qp_offset) as i8;
    mb.qp_cr = recon::chroma_qp(qp, slice.pps.second_chroma_qp_offset) as i8;
    mb.off_a = slice.off_a as i8;
    mb.off_b = slice.off_b as i8;
    mb.deblock = slice.deblock;
    if slice.kind == 0 {
        let mv = p_skip_mv(pic, mx, my, slice_id);
        for part in 0..4 {
            mb.ref0[part] = 0;
        }
        for block in 0..16 {
            mb.mv0[block] = mv;
        }
        predict_inter(pic, dpb, slice, mx, my, &mb)?;
    } else {
        direct_16(pic, dpb, slice, mx, my, &mut mb)?;
        predict_inter(pic, dpb, slice, mx, my, &mb)?;
    }
    bind_refs(&mut mb, slice);
    pic.mb[addr] = mb;
    Ok(())
}

fn p_skip_mv(pic: &Frame, mx: usize, my: usize, slice: u16) -> [i16; 2] {
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice);
    let above = neighbor(pic, mx as i32, my as i32 - 1, slice);
    let zero = |mb: &Mb, bx: usize, by: usize| {
        let part = (by / 2) * 2 + (bx / 2);
        mb.ref0[part] == 0 && mb.mv0[by * 4 + bx] == [0, 0]
    };
    if left.is_none_or(|mb| zero(mb, 3, 0)) || above.is_none_or(|mb| zero(mb, 0, 3)) {
        [0, 0]
    } else {
        mvp(
            pic,
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            (16, 16),
            0,
            0,
            &Mb::new(slice),
        )
    }
}

fn one_mb(
    pic: &mut Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    slice_id: u16,
    addr: usize,
    qp: i32,
    entropy: &mut Entropy<'_, '_>,
) -> Result<i32, CodecError> {
    let (mx, my) = xy(pic, addr);
    let mut mb = Mb::new(slice_id);
    mb.off_a = slice.off_a as i8;
    mb.off_b = slice.off_b as i8;
    mb.deblock = slice.deblock;
    let (intra, spec) = mb_type(pic, entropy, slice.kind, mx, my, slice_id)?;
    mb.intra = intra;
    mb.direct16 = !intra && slice.kind == 1 && spec == 0;
    if intra && spec == 25 {
        mb.pcm = true;
        read_pcm(pic, entropy, mx, my, slice)?;
        mb.nnz_y = [16; 16];
        mb.nnz_c = [16; 8];
        mb.qp = 0;
        pic.mb[addr] = mb;
        return Ok(qp);
    }
    let mut qp = qp;
    if intra {
        decode_intra(
            pic,
            entropy,
            slice,
            Place { mx, my },
            spec,
            &mut mb,
            &mut qp,
        )?;
    } else {
        decode_inter(
            pic,
            dpb,
            entropy,
            slice,
            Place { mx, my },
            spec,
            (&mut mb, &mut qp),
        )?;
    }
    mb.qp = qp as i8;
    mb.qp_c = recon::chroma_qp(qp, slice.pps.chroma_qp_offset) as i8;
    mb.qp_cr = recon::chroma_qp(qp, slice.pps.second_chroma_qp_offset) as i8;
    if !mb.intra {
        bind_refs(&mut mb, slice);
    }
    pic.mb[addr] = mb;
    Ok(qp)
}

fn mb_type(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    kind: u8,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<(bool, u32), CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => {
            let value = bits.ue()?;
            if kind == 2 {
                return Ok((true, value));
            }
            let intra_base = if kind == 0 { 5 } else { 23 };
            if value >= intra_base {
                Ok((true, value - intra_base))
            } else {
                Ok((false, value))
            }
        }
        Entropy::Cabac(cab) => cabac_mb_type(pic, cab, kind, mx, my, slice),
    }
}

fn cabac_mb_type(
    pic: &Frame,
    cab: &mut Cabac<'_>,
    kind: u8,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<(bool, u32), CodecError> {
    if kind == 1 {
        let ctx = direct_ctx(pic, mx, my, slice);
        if cab.decision(27 + ctx)? == 0 {
            return Ok((false, 0));
        }
        if cab.decision(27 + 3)? == 0 {
            return Ok((false, 1 + u32::from(cab.decision(27 + 5)?)));
        }
        let mut bits = u32::from(cab.decision(27 + 4)?) << 3;
        bits += u32::from(cab.decision(27 + 5)?) << 2;
        bits += u32::from(cab.decision(27 + 5)?) << 1;
        bits += u32::from(cab.decision(27 + 5)?);
        if bits < 8 {
            return Ok((false, bits + 3));
        }
        if bits == 13 {
            return Ok((true, cabac_intra_type(pic, cab, 32, false, mx, my, slice)?));
        }
        if bits == 14 {
            return Ok((false, 11));
        }
        if bits == 15 {
            return Ok((false, 22));
        }
        bits = (bits << 1) + u32::from(cab.decision(27 + 5)?);
        return Ok((false, bits - 4));
    }
    if kind == 0 {
        if cab.decision(14)? == 0 {
            if cab.decision(15)? == 0 {
                return Ok((false, 3 * u32::from(cab.decision(16)?)));
            }
            return Ok((false, 2 - u32::from(cab.decision(17)?)));
        }
        return Ok((true, cabac_intra_type(pic, cab, 17, false, mx, my, slice)?));
    }
    Ok((true, cabac_intra_type(pic, cab, 3, true, mx, my, slice)?))
}

fn direct_ctx(pic: &Frame, mx: usize, my: usize, slice: u16) -> usize {
    // Clause 9.3.3.1.1.3: a missing neighbor, B_Skip, or B_Direct_16x16 contributes 0.
    let coded = |mb: &Mb| !mb.skip && !mb.direct16;
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice).is_some_and(coded);
    let above = neighbor(pic, mx as i32, my as i32 - 1, slice).is_some_and(coded);
    usize::from(left) + usize::from(above)
}

fn cabac_intra_type(
    pic: &Frame,
    cab: &mut Cabac<'_>,
    base: usize,
    intra_slice: bool,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<u32, CodecError> {
    let intra16 = |mb: &Mb| mb.intra16 || mb.pcm;
    if intra_slice {
        let mut ctx = 0;
        if neighbor(pic, mx as i32 - 1, my as i32, slice).is_some_and(intra16) {
            ctx += 1;
        }
        if neighbor(pic, mx as i32, my as i32 - 1, slice).is_some_and(intra16) {
            ctx += 1;
        }
        if cab.decision(base + ctx)? == 0 {
            return Ok(0);
        }
    } else if cab.decision(base)? == 0 {
        return Ok(0);
    }
    if cab.terminate()? {
        return Ok(25);
    }
    let shift = usize::from(intra_slice);
    // An I slice skips the two prefix contexts before the I_16x16 suffix.
    let origin = base + shift * 2;
    let mut kind = 1u32;
    kind += 12 * u32::from(cab.decision(origin + 1)?);
    if cab.decision(origin + 2)? == 1 {
        kind += 4 + 4 * u32::from(cab.decision(origin + 2 + shift)?);
    }
    kind += 2 * u32::from(cab.decision(origin + 3 + shift)?);
    kind += u32::from(cab.decision(origin + 3 + 2 * shift)?);
    Ok(kind)
}

fn decode_intra(
    pic: &mut Frame,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    spec: u32,
    mb: &mut Mb,
    qp: &mut i32,
) -> Result<(), CodecError> {
    let Place { mx, my } = at;
    if spec > 25 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "intra mb type",
        });
    }
    let i16 = (1..25).contains(&spec);
    mb.intra16 = i16;
    let (mode16, cbp_luma, cbp_chroma) = if i16 {
        let mode = ((spec - 1) % 4) as u8;
        let chroma = (((spec - 1) / 4) % 3) as u8;
        let luma = u8::from(spec >= 13) * 15;
        (mode, luma, chroma)
    } else {
        (0, 0, 0)
    };
    let mut transform8 = false;
    if !i16 && slice.pps.transform_8x8 {
        transform8 = transform8_flag(pic, entropy, mx, my, mb.slice)?;
    }
    mb.transform8 = transform8;
    if !i16 {
        if transform8 {
            read_intra8(pic, entropy, slice, mx, my, mb)?;
        } else {
            read_intra4(pic, entropy, slice, mx, my, mb)?;
        }
    }
    let chroma = read_chroma_mode(pic, entropy, mx, my, mb.slice)?;
    mb.chroma_mode = chroma;
    let (luma_cbp, chroma_cbp) = if i16 {
        (cbp_luma, cbp_chroma)
    } else {
        read_cbp(pic, entropy, mx, my, true, mb.slice)?
    };
    mb.cbp = luma_cbp | (chroma_cbp << 4);
    if luma_cbp > 0 && slice.pps.transform_8x8 && !i16 && !transform8 {
        // Inter-only syntax position. Intra already consumed the flag.
    }
    let mut delta = 0;
    if luma_cbp > 0 || chroma_cbp > 0 || i16 {
        delta = qp_delta(entropy, prev_qpd(pic, mx, my, mb.slice))?;
        *qp = (*qp + delta).rem_euclid(52);
    }
    mb.qpd = delta as i8;
    let mut luma_dc = [0i32; 16];
    let mut ac = [[0i32; 16]; 16];
    let mut block8 = [[0i32; 64]; 4];
    if i16 {
        luma_dc = read_block(
            pic,
            entropy,
            (0, 16, true),
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            mb,
            0,
        )?;
        if luma_cbp != 0 {
            for (i, slot) in ac.iter_mut().enumerate() {
                *slot = read_block(
                    pic,
                    entropy,
                    (1, 15, false),
                    Place { mx, my },
                    Cell {
                        bx: blk_x(i),
                        by: blk_y(i),
                    },
                    mb,
                    0,
                )?;
            }
        }
    } else if transform8 {
        for (i, slot) in block8.iter_mut().enumerate() {
            if luma_cbp & (1 << i) != 0 {
                *slot = read8(pic, entropy, mx, my, i, mb)?;
            }
        }
    } else {
        for (i, slot) in ac.iter_mut().enumerate() {
            let bit = 1 << ((blk_y(i) / 2) * 2 + blk_x(i) / 2);
            if luma_cbp & bit != 0 {
                *slot = read_block(
                    pic,
                    entropy,
                    (2, 16, false),
                    Place { mx, my },
                    Cell {
                        bx: blk_x(i),
                        by: blk_y(i),
                    },
                    mb,
                    0,
                )?;
            }
        }
    }
    let (cb_dc, cr_dc, cb_ac, cr_ac) = read_chroma(pic, entropy, mx, my, chroma_cbp, mb)?;
    recon_intra(
        pic,
        slice,
        Place { mx, my },
        mb,
        LumaPaint {
            i16,
            mode16,
            chroma_mode: chroma,
            dc: luma_dc,
            ac: &ac,
            block8: &block8,
        },
        ChromaPaint {
            cb_dc,
            cr_dc,
            cb_ac: &cb_ac,
            cr_ac: &cr_ac,
        },
        *qp,
    );
    Ok(())
}

fn read_intra4(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
) -> Result<u8, CodecError> {
    for i in 0..16 {
        let bx = blk_x(i);
        let by = blk_y(i);
        let pred = pred_mode(
            mode_at(
                pic,
                mx,
                my,
                bx as i32 - 1,
                by as i32,
                mb,
                slice.pps.constrained_intra,
            ),
            mode_at(
                pic,
                mx,
                my,
                bx as i32,
                by as i32 - 1,
                mb,
                slice.pps.constrained_intra,
            ),
        );
        let mode = read_mode(entropy, pred)?;
        mb.mode4[by * 4 + bx] = mode;
    }
    let _ = slice;
    Ok(0)
}

fn read_intra8(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
) -> Result<u8, CodecError> {
    for i in 0..4 {
        let bx = (i & 1) * 2;
        let by = (i >> 1) * 2;
        let pred = pred_mode(
            mode_at(
                pic,
                mx,
                my,
                bx as i32 - 1,
                by as i32,
                mb,
                slice.pps.constrained_intra,
            ),
            mode_at(
                pic,
                mx,
                my,
                bx as i32,
                by as i32 - 1,
                mb,
                slice.pps.constrained_intra,
            ),
        );
        let mode = read_mode(entropy, pred)?;
        for dy in 0..2 {
            for dx in 0..2 {
                mb.mode4[(by + dy) * 4 + bx + dx] = mode;
            }
        }
    }
    let _ = slice;
    Ok(0)
}

fn mode_at(
    pic: &Frame,
    mx: usize,
    my: usize,
    bx: i32,
    by: i32,
    current: &Mb,
    constrained: bool,
) -> Option<u8> {
    let (mbx, ox) = if bx < 0 {
        (mx as i32 - 1, 3)
    } else if bx > 3 {
        (mx as i32 + 1, 0)
    } else {
        (mx as i32, bx as usize)
    };
    let (mby, oy) = if by < 0 {
        (my as i32 - 1, 3)
    } else if by > 3 {
        (my as i32 + 1, 0)
    } else {
        (my as i32, by as usize)
    };
    if mbx == mx as i32 && mby == my as i32 {
        return Some(current.mode4[oy * 4 + ox]);
    }
    let mb = neighbor(pic, mbx, mby, current.slice)?;
    // Clause 8.3.1.1: a neighbour that is not Intra_4x4 or Intra_8x8 contributes DC.
    // An inter neighbour with constrained intra makes both predictors DC.
    if !mb.intra && constrained {
        return None;
    }
    if mb.intra16 || mb.pcm || !mb.intra {
        return Some(2);
    }
    Some(mb.mode4[oy * 4 + ox])
}

fn pred_mode(left: Option<u8>, above: Option<u8>) -> u8 {
    match (left, above) {
        (Some(left), Some(above)) => left.min(above),
        _ => 2,
    }
}

fn read_mode(entropy: &mut Entropy<'_, '_>, pred: u8) -> Result<u8, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => {
            if bits.bit()? {
                Ok(pred)
            } else {
                let rem = bits.u(3)? as u8;
                Ok(if rem < pred { rem } else { rem + 1 })
            }
        }
        Entropy::Cabac(cab) => {
            if cab.decision(68)? == 1 {
                return Ok(pred);
            }
            let mut rem = 0u8;
            for shift in 0..3 {
                rem |= cab.decision(69)? << shift;
            }
            Ok(if rem < pred { rem } else { rem + 1 })
        }
    }
}

fn read_chroma_mode(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<u8, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => {
            let mode = bits.ue()?;
            if mode > 3 {
                return Err(bits.malformed("chroma mode"));
            }
            Ok(mode as u8)
        }
        Entropy::Cabac(cab) => {
            let mut ctx = 0;
            if neighbor(pic, mx as i32 - 1, my as i32, slice)
                .is_some_and(|mb| mb.intra && !mb.pcm && mb.chroma_mode != 0)
            {
                ctx += 1;
            }
            if neighbor(pic, mx as i32, my as i32 - 1, slice)
                .is_some_and(|mb| mb.intra && !mb.pcm && mb.chroma_mode != 0)
            {
                ctx += 1;
            }
            if cab.decision(64 + ctx)? == 0 {
                return Ok(0);
            }
            if cab.decision(64 + 3)? == 0 {
                return Ok(1);
            }
            if cab.decision(64 + 3)? == 0 {
                Ok(2)
            } else {
                Ok(3)
            }
        }
    }
}

fn read_cbp(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    intra: bool,
    slice: u16,
) -> Result<(u8, u8), CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => {
            let code = bits.ue()? as usize;
            if code >= 48 {
                return Err(bits.malformed("cbp"));
            }
            let value = if intra {
                super::CBP_INTRA[code]
            } else {
                super::CBP_INTER[code]
            };
            Ok((value & 15, value >> 4))
        }
        Entropy::Cabac(cab) => {
            let mut luma = 0u8;
            for bit in 0..4 {
                let ctx = cbp_ctx(pic, mx, my, bit, luma, slice);
                if cab.decision(73 + ctx)? == 1 {
                    luma |= 1 << bit;
                }
            }
            let left = cbp_chroma(pic, mx as i32 - 1, my as i32, slice);
            let above = cbp_chroma(pic, mx as i32, my as i32 - 1, slice);
            let mut ctx = usize::from(left > 0) + 2 * usize::from(above > 0);
            if cab.decision(77 + ctx)? == 0 {
                return Ok((luma, 0));
            }
            ctx = 4 + usize::from(left == 2) + 2 * usize::from(above == 2);
            Ok((luma, 1 + cab.decision(77 + ctx)?))
        }
    }
}

fn cbp_ctx(pic: &Frame, mx: usize, my: usize, bit: usize, current: u8, slice: u16) -> usize {
    let (left_bit, above_bit) = match bit {
        0 => (0x02, 0x04),
        1 => (0x01, 0x08),
        2 => (0x08, 0x01),
        _ => (0x04, 0x02),
    };
    let left_coded = if bit == 1 || bit == 3 {
        current & left_bit != 0
    } else {
        coded_bit(pic, mx as i32 - 1, my as i32, left_bit, slice)
    };
    let above_coded = if bit == 2 || bit == 3 {
        current & above_bit != 0
    } else {
        coded_bit(pic, mx as i32, my as i32 - 1, above_bit, slice)
    };
    usize::from(!left_coded) + 2 * usize::from(!above_coded)
}

fn coded_bit(pic: &Frame, mx: i32, my: i32, bit: u8, slice: u16) -> bool {
    match neighbor(pic, mx, my, slice) {
        Some(mb) => mb.pcm || mb.cbp & bit != 0,
        None => true,
    }
}

fn cbp_chroma(pic: &Frame, mx: i32, my: i32, slice: u16) -> u8 {
    match neighbor(pic, mx, my, slice) {
        Some(mb) if mb.pcm => 2,
        Some(mb) => mb.cbp >> 4,
        None => 0,
    }
}

/// Clause 9.3.3.1.1.5. The first bin uses context 60 or 61. Later bins use 62, then 63.
fn qp_delta(entropy: &mut Entropy<'_, '_>, prev_nonzero: bool) -> Result<i32, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => bits.se(),
        Entropy::Cabac(cab) => {
            let mut ctx = usize::from(prev_nonzero);
            let mut code = 0u32;
            loop {
                if cab.decision(60 + ctx)? == 0 {
                    break;
                }
                code += 1;
                if code > 52 {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "qp delta",
                    });
                }
                ctx = if code == 1 { 2 } else { 3 };
            }
            Ok(if code.is_multiple_of(2) {
                -((code / 2) as i32)
            } else {
                (code / 2) as i32 + 1
            })
        }
    }
}

fn prev_qpd(pic: &Frame, mx: usize, my: usize, slice: u16) -> bool {
    let addr = my * pic.mb_w + mx;
    if addr == 0 {
        return false;
    }
    pic.mb
        .get(addr - 1)
        .is_some_and(|mb| mb.slice == slice && mb.qpd != 0)
}

fn transform8_flag(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<bool, CodecError> {
    let on = |mb: &Mb| mb.transform8;
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice).is_some_and(on);
    let above = neighbor(pic, mx as i32, my as i32 - 1, slice).is_some_and(on);
    flag(entropy, 399 + usize::from(left) + usize::from(above))
}

fn flag(entropy: &mut Entropy<'_, '_>, ctx: usize) -> Result<bool, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => bits.bit(),
        Entropy::Cabac(cab) => Ok(cab.decision(ctx)? == 1),
    }
}

fn read_block(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    ask: (usize, usize, bool),
    at: Place,
    cell: Cell,
    mb: &mut Mb,
    plane: u8,
) -> Result<[i32; 16], CodecError> {
    let (cat, max, dc) = ask;
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let mut out = [0i32; 16];
    let coeffs = match entropy {
        Entropy::Cavlc(bits) => {
            let nc = if cat == 3 {
                -1
            } else {
                nc_of(pic, mx, my, bx, by, mb, plane)
            };
            let coeffs = cavlc::residual(bits, nc, max)?;
            for (index, value) in coeffs.iter().enumerate() {
                out[index] = *value;
            }
            out
        }
        Entropy::Cabac(cab) => {
            // Luma DC (cat 0) has a coded_block_flag. Its neighbours are Intra_16x16 DC blocks.
            let inc = if cat == 0 {
                cbf_dc_luma(pic, mx, my, mb)
            } else if cat == 3 {
                cbf_dc_chroma(pic, mx, my, mb, usize::from(plane.saturating_sub(1)))
            } else {
                cbf_inc(pic, Place { mx, my }, Cell { bx, by }, mb, plane, dc)
            };
            let base = [85, 89, 93, 97, 101][cat.min(4)];
            if cab.decision(base + inc)? == 0 {
                return Ok(out);
            }
            if cat == 0 {
                mb.dc_coded = true;
            } else if cat == 3 {
                let which = usize::from(plane.saturating_sub(1));
                if let Some(flag) = mb.dc_c.get_mut(which) {
                    *flag = true;
                }
            }
            cabac_residual(cab, cat, &mut out[..max])?;
            out
        }
    };
    let count = coeffs.iter().filter(|value| **value != 0).count() as u8;
    if plane == 0 && !dc {
        mb.nnz_y[by * 4 + bx] = count;
    }
    Ok(coeffs)
}

fn read8(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    which: usize,
    mb: &mut Mb,
) -> Result<[i32; 64], CodecError> {
    let mut out = [0i32; 64];
    let bx = (which & 1) * 2;
    let by = (which >> 1) * 2;
    match entropy {
        Entropy::Cavlc(bits) => {
            for i4 in 0..4 {
                let x = bx + (i4 & 1);
                let y = by + (i4 >> 1);
                let nc = nc_of(pic, mx, my, x, y, mb, 0);
                let block = cavlc::residual(bits, nc, 16)?;
                let count = block.iter().filter(|value| **value != 0).count() as u8;
                mb.nnz_y[y * 4 + x] = count;
                for (index, value) in block.iter().enumerate() {
                    out[index * 4 + i4] = *value;
                }
            }
        }
        Entropy::Cabac(cab) => {
            cabac_residual(cab, 5, &mut out)?;
            let count = out.iter().filter(|value| **value != 0).count() as u8;
            for dy in 0..2 {
                for dx in 0..2 {
                    mb.nnz_y[(by + dy) * 4 + bx + dx] = count;
                }
            }
        }
    }
    Ok(out)
}

type ChromaResidual = ([i32; 4], [i32; 4], [[i32; 16]; 4], [[i32; 16]; 4]);

fn read_chroma(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    cbp: u8,
    mb: &mut Mb,
) -> Result<ChromaResidual, CodecError> {
    let mut cb_dc = [0i32; 4];
    let mut cr_dc = [0i32; 4];
    let mut cb_ac = [[0i32; 16]; 4];
    let mut cr_ac = [[0i32; 16]; 4];
    if cbp == 0 {
        return Ok((cb_dc, cr_dc, cb_ac, cr_ac));
    }
    let dc0 = read_block(
        pic,
        entropy,
        (3, 4, true),
        Place { mx, my },
        Cell { bx: 0, by: 0 },
        mb,
        1,
    )?;
    cb_dc.copy_from_slice(&dc0[..4]);
    let dc1 = read_block(
        pic,
        entropy,
        (3, 4, true),
        Place { mx, my },
        Cell { bx: 0, by: 0 },
        mb,
        2,
    )?;
    cr_dc.copy_from_slice(&dc1[..4]);
    if cbp == 2 {
        for (i, slot) in cb_ac.iter_mut().enumerate() {
            *slot = read_block(
                pic,
                entropy,
                (4, 15, false),
                Place { mx, my },
                Cell {
                    bx: i & 1,
                    by: i >> 1,
                },
                mb,
                1,
            )?;
            mb.nnz_c[i] = slot.iter().filter(|value| **value != 0).count() as u8;
        }
        for (i, slot) in cr_ac.iter_mut().enumerate() {
            *slot = read_block(
                pic,
                entropy,
                (4, 15, false),
                Place { mx, my },
                Cell {
                    bx: i & 1,
                    by: i >> 1,
                },
                mb,
                2,
            )?;
            mb.nnz_c[4 + i] = slot.iter().filter(|value| **value != 0).count() as u8;
        }
    }
    Ok((cb_dc, cr_dc, cb_ac, cr_ac))
}

fn nc_of(pic: &Frame, mx: usize, my: usize, bx: usize, by: usize, mb: &Mb, plane: u8) -> i32 {
    let left = nnz_at(pic, mx, my, bx as i32 - 1, by as i32, mb, plane);
    let above = nnz_at(pic, mx, my, bx as i32, by as i32 - 1, mb, plane);
    match (left, above) {
        (Some(a), Some(b)) => (i32::from(a) + i32::from(b) + 1) >> 1,
        (Some(a), None) => i32::from(a),
        (None, Some(b)) => i32::from(b),
        (None, None) => 0,
    }
}

fn nnz_at(
    pic: &Frame,
    mx: usize,
    my: usize,
    bx: i32,
    by: i32,
    current: &Mb,
    plane: u8,
) -> Option<u8> {
    let grid = if plane == 0 { 4 } else { 2 };
    let (mbx, ox) = shift(mx, bx, grid);
    let (mby, oy) = shift(my, by, grid);
    let index = if plane == 0 {
        oy * 4 + ox
    } else {
        usize::from(plane - 1) * 4 + oy * 2 + ox
    };
    if mbx == mx as i32 && mby == my as i32 {
        return Some(if plane == 0 {
            current.nnz_y[index]
        } else {
            current.nnz_c[index]
        });
    }
    let mb = neighbor(pic, mbx, mby, current.slice)?;
    if mb.pcm {
        return Some(16);
    }
    Some(if plane == 0 {
        mb.nnz_y[index]
    } else {
        mb.nnz_c[index]
    })
}

fn shift(mb: usize, block: i32, grid: usize) -> (i32, usize) {
    let last = grid - 1;
    if block < 0 {
        (mb as i32 - 1, last)
    } else if block >= grid as i32 {
        (mb as i32 + 1, 0)
    } else {
        (mb as i32, block as usize)
    }
}

/// Clause 9.3.3.1.1.9 for ctxBlockCat 0. A neighbour contributes its DC flag only when it is Intra_16x16.
fn cbf_dc_luma(pic: &Frame, mx: usize, my: usize, mb: &Mb) -> usize {
    let flag = |dx: i32, dy: i32| -> usize {
        let Some(side) = neighbor(pic, mx as i32 + dx, my as i32 + dy, mb.slice) else {
            return usize::from(mb.intra);
        };
        if side.pcm {
            return 1;
        }
        if !side.intra16 {
            return 0;
        }
        usize::from(side.dc_coded)
    };
    flag(-1, 0) + 2 * flag(0, -1)
}

/// Clause 9.3.3.1.1.9 for ctxBlockCat 3. The neighbour contributes its chroma DC flag.
fn cbf_dc_chroma(pic: &Frame, mx: usize, my: usize, mb: &Mb, which: usize) -> usize {
    let flag = |dx: i32, dy: i32| -> usize {
        let Some(side) = neighbor(pic, mx as i32 + dx, my as i32 + dy, mb.slice) else {
            return usize::from(mb.intra);
        };
        if side.pcm {
            return 1;
        }
        if side.cbp >> 4 == 0 {
            return 0;
        }
        side.dc_c.get(which).copied().map(usize::from).unwrap_or(0)
    };
    flag(-1, 0) + 2 * flag(0, -1)
}

fn cbf_inc(pic: &Frame, at: Place, cell: Cell, mb: &Mb, plane: u8, dc: bool) -> usize {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let left = nnz_at(pic, mx, my, bx as i32 - 1, by as i32, mb, plane)
        .or(dc.then_some(u8::from(mb.intra)));
    let above = nnz_at(pic, mx, my, bx as i32, by as i32 - 1, mb, plane)
        .or(dc.then_some(u8::from(mb.intra)));
    let left = left.unwrap_or(u8::from(mb.intra));
    let above = above.unwrap_or(u8::from(mb.intra));
    usize::from(left > 0) + 2 * usize::from(above > 0)
}

fn cabac_residual(cab: &mut Cabac<'_>, cat: usize, coeffs: &mut [i32]) -> Result<(), CodecError> {
    let (sig_base, last_base, level_base) = match cat {
        0 => (105, 166, 227),
        1 => (120, 181, 237),
        2 => (134, 195, 247),
        3 => (149, 210, 257),
        4 => (152, 213, 266),
        _ => (402, 417, 426),
    };
    let max = coeffs.len();
    let mut index = [0usize; 64];
    let mut count = 0usize;
    let last_coded = max.saturating_sub(1);
    let mut pos = 0;
    while pos < last_coded {
        let sig_ctx = if cat == 5 {
            sig_base + cabac::sig8(pos)
        } else if cat == 3 {
            sig_base + pos.min(2)
        } else {
            sig_base + pos
        };
        if cab.decision(sig_ctx)? == 1 {
            let Some(slot) = index.get_mut(count) else {
                return Err(CodecError::Limit {
                    what: "cabac residual",
                });
            };
            *slot = pos;
            count += 1;
            let last_ctx = if cat == 5 {
                last_base + cabac::last8(pos)
            } else if cat == 3 {
                last_base + pos.min(2)
            } else {
                last_base + pos
            };
            if cab.decision(last_ctx)? == 1 {
                pos = max;
                break;
            }
        }
        pos += 1;
    }
    if pos == last_coded {
        let Some(slot) = index.get_mut(count) else {
            return Err(CodecError::Limit {
                what: "cabac residual",
            });
        };
        *slot = pos;
        count += 1;
    }
    if count == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "cabac residual",
        });
    }
    let mut node = 0usize;
    let level1 = [1usize, 2, 3, 4, 0, 0, 0, 0];
    let gt1 = [5usize, 5, 5, 5, 6, 7, 8, 9];
    let trans1 = [1usize, 2, 3, 3, 4, 5, 6, 7];
    let trans_gt = [4usize, 4, 4, 4, 5, 6, 7, 7];
    for scan in index[..count].iter().rev() {
        let scan = *scan;
        if cab.decision(level_base + level1[node])? == 0 {
            node = trans1[node];
            let sign = cab.bypass()?;
            coeffs[scan] = if sign == 1 { -1 } else { 1 };
        } else {
            let ctx = level_base + gt1[node];
            node = trans_gt[node];
            let mut abs = 2i32;
            while abs < 15 && cab.decision(ctx)? == 1 {
                abs += 1;
            }
            if abs >= 15 {
                let mut zeros = 0i32;
                while cab.bypass()? == 1 && zeros < 20 {
                    zeros += 1;
                }
                let mut mag = 1i32;
                for _ in 0..zeros {
                    mag = (mag << 1) | i32::from(cab.bypass()?);
                }
                abs = mag + 14;
            }
            let sign = cab.bypass()?;
            coeffs[scan] = if sign == 1 { -abs } else { abs };
        }
    }
    Ok(())
}

fn recon_intra(
    pic: &mut Frame,
    slice: &Slice<'_>,
    at: Place,
    mb: &Mb,
    luma: LumaPaint<'_>,
    chroma: ChromaPaint<'_>,
    qp: i32,
) {
    let Place { mx, my } = at;
    let LumaPaint {
        i16,
        mode16,
        chroma_mode,
        dc: luma_dc,
        ac,
        block8,
    } = luma;
    let ChromaPaint {
        cb_dc,
        cr_dc,
        cb_ac,
        cr_ac,
    } = chroma;
    let x0 = mx * 16;
    let y0 = my * 16;
    // Neighbours inside this macroblock are already reconstructed. Publish it
    // before those samples are read back, so availability sees this slice.
    pic.mb[my * pic.mb_w + mx] = mb.clone();
    if i16 {
        let (above, left, above_ok, left_ok) =
            edge16(pic, mx, my, mb.slice, slice.pps.constrained_intra);
        let pred = recon::pred16(mode16, &above, &left, above_ok, left_ok);
        put_block(&mut pic.y, pic.w, x0, y0, &pred, 16);
        let dc = recon::luma_dc(
            &luma_dc_raster(&luma_dc),
            qp,
            i32::from(slice.pps.weight4[0][0]),
        );
        for (i, samples) in ac.iter().enumerate() {
            let bx = blk_x(i);
            let by = blk_y(i);
            let mut coeffs = [0i32; 16];
            if mb.cbp & 15 != 0 {
                coeffs[1..].copy_from_slice(&samples[..15]);
            }
            let mut block = [0i32; 16];
            for n in 0..16 {
                let pos = ZIGZAG4[n];
                let (ry, rx) = (pos / 4, pos % 4);
                let level = if n == 0 { dc[by * 4 + bx] } else { coeffs[n] };
                let weight = i32::from(slice.pps.weight4[0][pos]);
                block[ry * 4 + rx] = if n == 0 {
                    level
                } else {
                    recon::scale4(level, qp, rx, ry, weight)
                };
            }
            if mb.cbp & 15 == 0 {
                block[0] = dc[by * 4 + bx];
            }
            recon::idct4(&mut block);
            recon::add_block(&mut pic.y, pic.w, x0 + bx * 4, y0 + by * 4, &block, 4);
        }
    } else if mb.transform8 {
        for (i, samples) in block8.iter().enumerate() {
            let bx = (i & 1) * 8;
            let by = (i >> 1) * 8;
            let pred = pred8_block(pic, mx, my, bx / 4, by / 4, mb, slice.pps.constrained_intra);
            put_block(&mut pic.y, pic.w, x0 + bx, y0 + by, &pred, 8);
            super::add_scaled(
                &mut pic.y,
                pic.w,
                (x0 + bx, y0 + by),
                samples,
                qp,
                &slice.pps.weight8[0],
                8,
            );
        }
    } else {
        for (i, samples) in ac.iter().enumerate() {
            let bx = blk_x(i);
            let by = blk_y(i);
            let pred = pred4_block(pic, mx, my, bx, by, mb, slice.pps.constrained_intra);
            put_block(&mut pic.y, pic.w, x0 + bx * 4, y0 + by * 4, &pred, 4);
            super::add_scaled(
                &mut pic.y,
                pic.w,
                (x0 + bx * 4, y0 + by * 4),
                samples,
                qp,
                &slice.pps.weight4[0],
                4,
            );
        }
    }
    recon_chroma(
        pic,
        slice,
        Place { mx, my },
        chroma_mode,
        ChromaPaint {
            cb_dc,
            cr_dc,
            cb_ac,
            cr_ac,
        },
        (qp, true, mb.slice),
    );
}

fn luma_dc_raster(scan: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for i in 0..16 {
        out[ZIGZAG4[i]] = scan[i];
    }
    out
}

fn pred4_block(
    pic: &Frame,
    mx: usize,
    my: usize,
    bx: usize,
    by: usize,
    mb: &Mb,
    constrained: bool,
) -> [u8; 16] {
    let ax = mx * 4 + bx;
    let ay = my * 4 + by;
    let mut above = [128u8; 8];
    let mut left = [128u8; 4];
    let here = (mx, my, bx, by);
    let above_ok = row_ok(pic, ax, ay as i32 - 1, mb.slice, constrained, here);
    let left_ok = col_ok(pic, ax as i32 - 1, ay, mb.slice, constrained, here);
    if above_ok {
        for (k, slot) in above[..4].iter_mut().enumerate() {
            *slot = pix(&pic.y, pic.w, pic.h, (ax * 4 + k) as i32, ay as i32 * 4 - 1);
        }
        let right_ok = above_right_ok(pic, mx, my, bx, by, mb.slice, constrained);
        let fill = above[3];
        for (k, slot) in above[4..].iter_mut().enumerate() {
            let x = ax * 4 + k + 4;
            *slot = if right_ok {
                pix(&pic.y, pic.w, pic.h, x as i32, ay as i32 * 4 - 1)
            } else {
                fill
            };
        }
    }
    if left_ok {
        for (k, slot) in left.iter_mut().enumerate() {
            *slot = pix(&pic.y, pic.w, pic.h, ax as i32 * 4 - 1, (ay * 4 + k) as i32);
        }
    }
    let corner = if above_ok && left_ok {
        pix(&pic.y, pic.w, pic.h, ax as i32 * 4 - 1, ay as i32 * 4 - 1)
    } else if above_ok {
        above[0]
    } else if left_ok {
        left[0]
    } else {
        128
    };
    recon::pred4(
        mb.mode4[by * 4 + bx],
        &above,
        &left,
        corner,
        above_ok,
        left_ok,
    )
}

fn pred8_block(
    pic: &Frame,
    mx: usize,
    my: usize,
    bx: usize,
    by: usize,
    mb: &Mb,
    constrained: bool,
) -> [u8; 64] {
    let mode = mb.mode4[by * 4 + bx];
    let mut raw_above = [128u8; 16];
    let mut raw_left = [128u8; 8];
    let x = (mx * 16 + bx * 4) as i32;
    let y = (my * 16 + by * 4) as i32;
    let here = (mx, my, bx, by);
    let above_ok = row_ok(
        pic,
        mx * 4 + bx,
        (my * 4 + by) as i32 - 1,
        mb.slice,
        constrained,
        here,
    );
    let left_ok = col_ok(
        pic,
        (mx * 4 + bx) as i32 - 1,
        my * 4 + by,
        mb.slice,
        constrained,
        here,
    );
    if above_ok {
        for (k, slot) in raw_above.iter_mut().enumerate() {
            *slot = pix(&pic.y, pic.w, pic.h, x + k as i32, y - 1);
        }
    }
    if left_ok {
        for (k, slot) in raw_left.iter_mut().enumerate() {
            *slot = pix(&pic.y, pic.w, pic.h, x - 1, y + k as i32);
        }
    }
    let corner = if above_ok && left_ok {
        pix(&pic.y, pic.w, pic.h, x - 1, y - 1)
    } else {
        128
    };
    let above = filter_row(&raw_above, corner, above_ok, left_ok);
    let left = filter_col(&raw_left, corner, above_ok, left_ok);
    let corner = if above_ok && left_ok {
        avg3(raw_above[0], corner, raw_left[0])
    } else {
        corner
    };
    let mut out = [0u8; 64];
    for row in 0..8 {
        for col in 0..8 {
            out[row * 8 + col] =
                pred8_at(mode, (col, row), &above, &left, corner, (above_ok, left_ok));
        }
    }
    out
}

fn pred8_at(
    mode: u8,
    at: (usize, usize),
    above: &[u8; 16],
    left: &[u8; 8],
    corner: u8,
    present: (bool, bool),
) -> u8 {
    let (x, y) = at;
    let (above_ok, left_ok) = present;
    match mode {
        0 if above_ok => above[x],
        1 if left_ok => left[y],
        3 if above_ok => pred8_down_left(x, y, above),
        4 if above_ok && left_ok => pred8_down_right(x, y, above, left, corner),
        5 if above_ok && left_ok => pred8_vertical_right(x, y, above, left, corner),
        6 if above_ok && left_ok => pred8_horizontal_down(x, y, above, left, corner),
        7 if above_ok => pred8_vertical_left(x, y, above),
        8 if left_ok => pred8_horizontal_up(x, y, left),
        _ => dc8(above, left, above_ok, left_ok),
    }
}

fn pred8_down_left(x: usize, y: usize, above: &[u8; 16]) -> u8 {
    if x == 7 && y == 7 {
        avg3(above[14], above[15], above[15])
    } else {
        avg3(above[x + y], above[x + y + 1], above[x + y + 2])
    }
}

fn pred8_down_right(x: usize, y: usize, above: &[u8; 16], left: &[u8; 8], corner: u8) -> u8 {
    let x = i32::try_from(x).unwrap_or(0);
    let y = i32::try_from(y).unwrap_or(0);
    if x > y {
        avg3(
            above_at(above, corner, x - y - 2),
            above_at(above, corner, x - y - 1),
            above_at(above, corner, x - y),
        )
    } else if x < y {
        avg3(
            left_at(left, corner, y - x - 2),
            left_at(left, corner, y - x - 1),
            left_at(left, corner, y - x),
        )
    } else {
        avg3(above[0], corner, left[0])
    }
}

fn pred8_vertical_right(x: usize, y: usize, above: &[u8; 16], left: &[u8; 8], corner: u8) -> u8 {
    let x = i32::try_from(x).unwrap_or(0);
    let y = i32::try_from(y).unwrap_or(0);
    let z = 2 * x - y;
    let shift = y >> 1;
    if z >= 0 && (z & 1) == 0 {
        let index = x - shift;
        avg2(
            above_at(above, corner, index - 1),
            above_at(above, corner, index),
        )
    } else if z > 0 {
        let index = x - shift;
        avg3(
            above_at(above, corner, index - 2),
            above_at(above, corner, index - 1),
            above_at(above, corner, index),
        )
    } else if z == -1 {
        avg3(left[0], corner, above[0])
    } else {
        let index = y - 2 * x;
        avg3(
            left_at(left, corner, index - 1),
            left_at(left, corner, index - 2),
            left_at(left, corner, index - 3),
        )
    }
}

fn pred8_horizontal_down(x: usize, y: usize, above: &[u8; 16], left: &[u8; 8], corner: u8) -> u8 {
    let x = i32::try_from(x).unwrap_or(0);
    let y = i32::try_from(y).unwrap_or(0);
    let z = 2 * y - x;
    let shift = x >> 1;
    if z >= 0 && (z & 1) == 0 {
        let index = y - shift;
        avg2(
            left_at(left, corner, index - 1),
            left_at(left, corner, index),
        )
    } else if z > 0 {
        let index = y - shift;
        avg3(
            left_at(left, corner, index - 2),
            left_at(left, corner, index - 1),
            left_at(left, corner, index),
        )
    } else if z == -1 {
        avg3(left[0], corner, above[0])
    } else {
        let index = x - 2 * y;
        avg3(
            above_at(above, corner, index - 1),
            above_at(above, corner, index - 2),
            above_at(above, corner, index - 3),
        )
    }
}

fn pred8_vertical_left(x: usize, y: usize, above: &[u8; 16]) -> u8 {
    let base = x + (y >> 1);
    if y.is_multiple_of(2) {
        avg2(above[base], above[base + 1])
    } else {
        avg3(above[base], above[base + 1], above[base + 2])
    }
}

fn pred8_horizontal_up(x: usize, y: usize, left: &[u8; 8]) -> u8 {
    let z = x + 2 * y;
    let base = y + (x >> 1);
    if matches!(z, 0 | 2 | 4 | 6 | 8 | 10 | 12) {
        avg2(left[base], left[base + 1])
    } else if matches!(z, 1 | 3 | 5 | 7 | 9 | 11) {
        avg3(left[base], left[base + 1], left[base + 2])
    } else if z == 13 {
        avg3(left[6], left[7], left[7])
    } else {
        left[7]
    }
}

fn above_at(above: &[u8; 16], corner: u8, index: i32) -> u8 {
    let Ok(index) = usize::try_from(index) else {
        return corner;
    };
    above.get(index).copied().unwrap_or(corner)
}

fn left_at(left: &[u8; 8], corner: u8, index: i32) -> u8 {
    let Ok(index) = usize::try_from(index) else {
        return corner;
    };
    left.get(index).copied().unwrap_or(corner)
}

fn avg2(a: u8, b: u8) -> u8 {
    u8::try_from((i32::from(a) + i32::from(b) + 1) >> 1).unwrap_or(0)
}

fn filter_row(raw: &[u8; 16], corner: u8, above_ok: bool, left_ok: bool) -> [u8; 16] {
    let mut out = *raw;
    if !above_ok {
        return out;
    }
    let left = if left_ok { corner } else { raw[0] };
    out[0] = avg3(left, raw[0], raw[1]);
    for i in 1..15 {
        out[i] = avg3(raw[i - 1], raw[i], raw[i + 1]);
    }
    out[15] = avg3(raw[14], raw[15], raw[15]);
    out
}

fn filter_col(raw: &[u8; 8], corner: u8, above_ok: bool, left_ok: bool) -> [u8; 8] {
    let mut out = *raw;
    if !left_ok {
        return out;
    }
    let up = if above_ok { corner } else { raw[0] };
    out[0] = avg3(up, raw[0], raw[1]);
    for i in 1..7 {
        out[i] = avg3(raw[i - 1], raw[i], raw[i + 1]);
    }
    out[7] = avg3(raw[6], raw[7], raw[7]);
    out
}

fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((i32::from(a) + 2 * i32::from(b) + i32::from(c) + 2) >> 2) as u8
}

fn dc8(above: &[u8], left: &[u8], above_ok: bool, left_ok: bool) -> u8 {
    if above_ok && left_ok {
        let sum = above[..8].iter().map(|v| i32::from(*v)).sum::<i32>()
            + left[..8].iter().map(|v| i32::from(*v)).sum::<i32>();
        ((sum + 8) >> 4) as u8
    } else if above_ok {
        let sum = above[..8].iter().map(|v| i32::from(*v)).sum::<i32>();
        ((sum + 4) >> 3) as u8
    } else if left_ok {
        let sum = left[..8].iter().map(|v| i32::from(*v)).sum::<i32>();
        ((sum + 4) >> 3) as u8
    } else {
        128
    }
}

fn recon_chroma(
    pic: &mut Frame,
    slice: &Slice<'_>,
    at: Place,
    mode: u8,
    paint: ChromaPaint<'_>,
    flags: (i32, bool, u16),
) {
    let Place { mx, my } = at;
    let ChromaPaint {
        cb_dc,
        cr_dc,
        cb_ac,
        cr_ac,
    } = paint;
    let (qp, intra, slice_id) = flags;
    let x0 = mx * 8;
    let y0 = my * 8;
    let qp_cb = recon::chroma_qp(qp, slice.pps.chroma_qp_offset);
    let qp_cr = recon::chroma_qp(qp, slice.pps.second_chroma_qp_offset);
    // Intra chroma uses lists 1 and 2. Inter chroma uses lists 4 and 5.
    let base = if intra { 1 } else { 4 };
    for (cb, dc, ac, qp_c, list) in [
        (true, cb_dc, cb_ac, qp_cb, base),
        (false, cr_dc, cr_ac, qp_cr, base + 1),
    ] {
        let pred = if intra {
            let (above, left, above_ok, left_ok) =
                edge_chroma(pic, mx, my, cb, slice_id, slice.pps.constrained_intra);
            Some(recon::pred_chroma(mode, &above, &left, above_ok, left_ok))
        } else {
            None
        };
        let plane = if cb { &mut pic.u } else { &mut pic.v };
        let stride = pic.w / 2;
        if let Some(pred) = pred {
            put_block(plane, stride, x0, y0, &pred, 8);
        }
        let weight = i32::from(slice.pps.weight4[list][0]);
        let scaled = recon::chroma_dc(dc[0], dc[1], dc[2], dc[3], qp_c, weight);
        for i in 0..4 {
            let bx = i & 1;
            let by = i >> 1;
            let mut coeffs = [0i32; 16];
            coeffs[1..].copy_from_slice(&ac[i][..15]);
            let mut block = [0i32; 16];
            for n in 0..16 {
                let pos = ZIGZAG4[n];
                let (ry, rx) = (pos / 4, pos % 4);
                let level = if n == 0 { scaled[i] } else { coeffs[n] };
                let w = i32::from(slice.pps.weight4[list][pos]);
                block[ry * 4 + rx] = if n == 0 {
                    level
                } else {
                    recon::scale4(level, qp_c, rx, ry, w)
                };
            }
            recon::idct4(&mut block);
            recon::add_block(plane, stride, x0 + bx * 4, y0 + by * 4, &block, 4);
        }
    }
}

#[cfg(test)]
mod oracle {
    use std::process::Command;

    use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder};

    use crate::H264Decoder;

    fn ffmpeg() -> Option<String> {
        [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ]
        .into_iter()
        .find(|path| Command::new(path).arg("-version").output().is_ok())
        .map(str::to_string)
    }

    #[test]
    fn baseline_intra_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-intra");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("intra.h264");
        let yuv = dir.join("intra.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30:duration=1",
                "-frames:v",
                "1",
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=0:bframes=0:weightp=0:8x8dct=0:keyint=1:scenecut=0",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let packet = Packet::video(bytes, 0, 1, 30, true);
        let output = decoder.push(&packet).unwrap();
        let frames = if output.is_empty() {
            decoder.drain().unwrap()
        } else {
            output
        };
        assert_eq!(frames.len(), 1, "expected one picture");
        let picture = &frames[0].picture;
        let mut got = picture.y().to_vec();
        got.extend_from_slice(picture.u());
        got.extend_from_slice(picture.v());
        let mut count = 0usize;
        let mut max = 0u16;
        let mut worst = (0usize, 0u8, 0u8);
        let mut misses = Vec::new();
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            let delta = u16::from(a.abs_diff(*b));
            count += 1;
            if delta >= max {
                max = delta;
                worst = (index, *a, *b);
            }
            if misses.len() < 8 {
                misses.push((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("intra mismatches {count} max {max} worst {worst:?}: {misses:?}");
        }
    }

    #[test]
    fn baseline_ip_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-ip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("ip.h264");
        let yuv = dir.join("ip.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=0:bframes=0:weightp=0:8x8dct=0:keyint=4:scenecut=0:ref=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for unit in access_units(&bytes) {
            let packet = Packet::video(unit, 0, 1, 30, false);
            got.extend(pictures(&decoder.push(&packet).unwrap()));
        }
        got.extend(pictures(&decoder.drain().unwrap()));
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        let mut worst = (0usize, 0u8, 0u8);
        let mut per = [0usize; 8];
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            let delta = a.abs_diff(*b);
            let frame = index / 6144;
            if frame < per.len() {
                per[frame] += 1;
            }
            if delta >= max {
                max = delta;
                worst = (index, *a, *b);
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("ip mismatches {count} max {max} first {first:?} worst {worst:?} per {per:?}");
        }
    }

    #[test]
    fn main_cabac_ipb_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-ipb");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("ipb.h264");
        let yuv = dir.join("ipb.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=1:weightp=0:8x8dct=0:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=spatial",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        let mut per = [0usize; 8];
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            let delta = a.abs_diff(*b);
            let frame = index / 6144;
            if frame < per.len() {
                per[frame] += 1;
            }
            if delta >= max {
                max = delta;
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!(
                "ipb mismatches {count} max {max} first {first:?} per {per:?} got {:?} want {:?}",
                &got[..16],
                &expected[..16]
            );
        }
    }

    #[test]
    fn main_cabac_temporal_direct_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-temporal");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("temporal.h264");
        let yuv = dir.join("temporal.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=1:weightp=0:8x8dct=0:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=temporal",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            let delta = a.abs_diff(*b);
            if delta >= max {
                max = delta;
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("temporal mismatches {count} max {max} first {first:?}");
        }
    }

    #[test]
    fn two_slices_match_ffmpeg_across_the_boundary() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-slices");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("slices.h264");
        let yuv = dir.join("slices.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "3",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=0:slices=2:keyint=30:scenecut=0:weightp=0:8x8dct=0:threads=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            if a.abs_diff(*b) >= max {
                max = a.abs_diff(*b);
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("slice boundary mismatches {count} max {max} first {first:?}");
        }
    }

    #[test]
    fn cropped_picture_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-crop");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("crop.h264");
        let yuv = dir.join("crop.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=70x50:rate=30",
                "-frames:v",
                "1",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=0:keyint=1:scenecut=0:threads=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 70, 50);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        let mut size = (0u32, 0u32);
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => {
                    if let Some(frame) = frames.first() {
                        size = (frame.picture.width(), frame.picture.height());
                    }
                    got.extend(pictures(&frames));
                }
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => {
                if let Some(frame) = frames.first() {
                    size = (frame.picture.width(), frame.picture.height());
                }
                got.extend(pictures(&frames));
            }
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(size, (70, 50), "visible size");
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            if a.abs_diff(*b) >= max {
                max = a.abs_diff(*b);
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("crop mismatches {count} max {max} first {first:?}");
        }
    }

    #[test]
    fn jvt_quant_matrices_match_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-cqm");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("cqm.h264");
        let yuv = dir.join("cqm.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "2",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=0:8x8dct=1:keyint=30:scenecut=0:cqm=jvt:threads=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            if a.abs_diff(*b) >= max {
                max = a.abs_diff(*b);
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("quant matrix mismatches {count} max {max} first {first:?}");
        }
    }

    #[test]
    fn high_cabac_8x8_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-high");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("high.h264");
        let yuv = dir.join("high.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=1:weightp=0:8x8dct=1:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=spatial",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        let mut per = [0usize; 8];
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            let delta = a.abs_diff(*b);
            let frame = index / 6144;
            if frame < per.len() {
                per[frame] += 1;
            }
            if delta >= max {
                max = delta;
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("high mismatches {count} max {max} first {first:?} per {per:?}");
        }
    }

    #[test]
    fn high_implicit_weights_match_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-weightb");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("weightb.h264");
        let yuv = dir.join("weightb.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "6",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=2:weightp=0:weightb=1:8x8dct=1:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=spatial",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a != b {
                count += 1;
                max = max.max(a.abs_diff(*b));
                if first.is_none() {
                    first = Some((index, *a, *b));
                }
            }
        }
        assert_eq!(
            count, 0,
            "weightb mismatches {count} max {max} first {first:?}"
        );
    }

    #[test]
    fn main_explicit_weights_match_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-weightp");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("weightp.h264");
        let yuv = dir.join("weightp.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30,fade=in:0:4",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=0:weightp=1:8x8dct=0:keyint=30:scenecut=0:ref=2:threads=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a != b {
                count += 1;
                max = max.max(a.abs_diff(*b));
                if first.is_none() {
                    first = Some((index, *a, *b));
                }
            }
        }
        assert_eq!(
            count, 0,
            "weightp mismatches {count} max {max} first {first:?}"
        );
    }

    fn pictures(frames: &[cutlass_decode::VideoOutput]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for frame in frames {
            bytes.extend_from_slice(frame.picture.y());
            bytes.extend_from_slice(frame.picture.u());
            bytes.extend_from_slice(frame.picture.v());
        }
        bytes
    }

    fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
        let mut starts = Vec::new();
        let mut i = 0;
        while i + 3 < data.len() {
            if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                let header = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
                starts.push((header, i + 3));
                i += 3;
            } else {
                i += 1;
            }
        }
        let mut units = Vec::new();
        let mut current = Vec::new();
        let mut saw_vcl = false;
        for (n, &(header, body)) in starts.iter().enumerate() {
            let end = starts
                .get(n + 1)
                .map(|(next, _)| *next)
                .unwrap_or(data.len());
            let kind = data[body] & 31;
            let vcl = kind == 1 || kind == 5;
            let starts_picture = vcl && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
            if (kind == 9 || (starts_picture && saw_vcl)) && !current.is_empty() {
                units.push(std::mem::take(&mut current));
                saw_vcl = false;
            }
            current.extend_from_slice(&data[header..end]);
            saw_vcl |= vcl;
        }
        if !current.is_empty() {
            units.push(current);
        }
        units
    }
}

fn edge16(
    pic: &Frame,
    mx: usize,
    my: usize,
    slice: u16,
    constrained: bool,
) -> (Vec<u8>, Vec<u8>, bool, bool) {
    let above_ok = my > 0
        && neighbor(pic, mx as i32, my as i32 - 1, slice)
            .is_some_and(|mb| !constrained || mb.intra);
    let left_ok = mx > 0
        && neighbor(pic, mx as i32 - 1, my as i32, slice)
            .is_some_and(|mb| !constrained || mb.intra);
    let mut above = vec![128u8; 17];
    let mut left = vec![128u8; 17];
    if above_ok {
        for (k, slot) in above[..16].iter_mut().enumerate() {
            *slot = pix(
                &pic.y,
                pic.w,
                pic.h,
                (mx * 16 + k) as i32,
                my as i32 * 16 - 1,
            );
        }
        above[16] = pix(
            &pic.y,
            pic.w,
            pic.h,
            (mx * 16 + 16) as i32,
            my as i32 * 16 - 1,
        );
    }
    if left_ok {
        for (k, slot) in left[..16].iter_mut().enumerate() {
            *slot = pix(
                &pic.y,
                pic.w,
                pic.h,
                mx as i32 * 16 - 1,
                (my * 16 + k) as i32,
            );
        }
        left[16] = pix(
            &pic.y,
            pic.w,
            pic.h,
            mx as i32 * 16 - 1,
            (my * 16 + 16) as i32,
        );
    }
    (above, left, above_ok, left_ok)
}

fn edge_chroma(
    pic: &Frame,
    mx: usize,
    my: usize,
    cb: bool,
    slice_id: u16,
    constrained: bool,
) -> (Vec<u8>, Vec<u8>, bool, bool) {
    let plane = if cb { &pic.u } else { &pic.v };
    let stride = pic.w / 2;
    let height = pic.h / 2;
    let above_ok = my > 0
        && neighbor(pic, mx as i32, my as i32 - 1, slice_id)
            .is_some_and(|mb| !constrained || mb.intra);
    let left_ok = mx > 0
        && neighbor(pic, mx as i32 - 1, my as i32, slice_id)
            .is_some_and(|mb| !constrained || mb.intra);
    let mut above = vec![128u8; 9];
    let mut left = vec![128u8; 9];
    if above_ok {
        for (k, slot) in above[..8].iter_mut().enumerate() {
            *slot = pix(
                plane,
                stride,
                height,
                (mx * 8 + k) as i32,
                my as i32 * 8 - 1,
            );
        }
        above[8] = pix(
            plane,
            stride,
            height,
            (mx * 8 + 8) as i32,
            my as i32 * 8 - 1,
        );
    }
    if left_ok {
        for (k, slot) in left[..8].iter_mut().enumerate() {
            *slot = pix(
                plane,
                stride,
                height,
                mx as i32 * 8 - 1,
                (my * 8 + k) as i32,
            );
        }
        left[8] = pix(
            plane,
            stride,
            height,
            mx as i32 * 8 - 1,
            (my * 8 + 8) as i32,
        );
    }
    (above, left, above_ok, left_ok)
}

fn row_ok(
    pic: &Frame,
    ax: usize,
    ay: i32,
    slice: u16,
    constrained: bool,
    here: (usize, usize, usize, usize),
) -> bool {
    if ay < 0 {
        return false;
    }
    let (cur_mx, cur_my, cur_bx, cur_by) = here;
    let mbx = (ax / 4) as i32;
    let mby = ay / 4;
    if mbx == cur_mx as i32 && mby == cur_my as i32 {
        return scan_of(ax % 4, ay as usize % 4) < scan_of(cur_bx, cur_by);
    }
    let Some(mb) = neighbor(pic, mbx, mby, slice) else {
        return false;
    };
    !constrained || mb.intra
}

fn col_ok(
    pic: &Frame,
    ax: i32,
    ay: usize,
    slice: u16,
    constrained: bool,
    here: (usize, usize, usize, usize),
) -> bool {
    if ax < 0 {
        return false;
    }
    let (cur_mx, cur_my, cur_bx, cur_by) = here;
    let mbx = ax / 4;
    let mby = (ay / 4) as i32;
    if mbx == cur_mx as i32 && mby == cur_my as i32 {
        return scan_of(ax as usize % 4, ay % 4) < scan_of(cur_bx, cur_by);
    }
    let Some(mb) = neighbor(pic, mbx, mby, slice) else {
        return false;
    };
    !constrained || mb.intra
}

fn above_right_ok(
    pic: &Frame,
    mx: usize,
    my: usize,
    bx: usize,
    by: usize,
    slice: u16,
    constrained: bool,
) -> bool {
    if by == 0 {
        let mbx = mx + usize::from(bx == 3);
        return neighbor(pic, mbx as i32, my as i32 - 1, slice)
            .is_some_and(|mb| !constrained || mb.intra);
    }
    if bx == 3 {
        return false;
    }
    let target = scan_of(bx + 1, by - 1);
    let current = scan_of(bx, by);
    target < current
}

fn scan_of(bx: usize, by: usize) -> usize {
    match (bx, by) {
        (0, 0) => 0,
        (1, 0) => 1,
        (0, 1) => 2,
        (1, 1) => 3,
        (2, 0) => 4,
        (3, 0) => 5,
        (2, 1) => 6,
        (3, 1) => 7,
        (0, 2) => 8,
        (1, 2) => 9,
        (0, 3) => 10,
        (1, 3) => 11,
        (2, 2) => 12,
        (3, 2) => 13,
        (2, 3) => 14,
        _ => 15,
    }
}

fn pix(plane: &[u8], stride: usize, height: usize, x: i32, y: i32) -> u8 {
    if stride == 0 || height == 0 {
        return 128;
    }
    let x = x.clamp(0, stride as i32 - 1) as usize;
    let y = y.clamp(0, height as i32 - 1) as usize;
    plane.get(y * stride + x).copied().unwrap_or(128)
}

fn decode_inter(
    pic: &mut Frame,
    dpb: &[Frame],
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    spec: u32,
    work: (&mut Mb, &mut i32),
) -> Result<(), CodecError> {
    let Place { mx, my } = at;
    let (mb, qp) = work;
    let sub = if slice.kind == 1 {
        spec == 22
    } else {
        spec == 3 || spec == 4
    };
    if slice.kind == 1 && spec == 0 {
        direct_16(pic, dpb, slice, mx, my, mb)?;
    } else if sub {
        read_sub(pic, dpb, entropy, slice, Place { mx, my }, spec, mb)?;
    } else {
        read_parts(pic, entropy, slice, mx, my, spec, mb)?;
    }
    let (luma_cbp, chroma_cbp) = read_cbp(pic, entropy, mx, my, false, mb.slice)?;
    mb.cbp = luma_cbp | (chroma_cbp << 4);
    let direct16 = slice.kind == 1 && spec == 0;
    let allow8 = if direct16 {
        slice.sps.direct_8x8_inference
    } else {
        mb.partition8
    };
    if luma_cbp > 0 && slice.pps.transform_8x8 && allow8 {
        mb.transform8 = transform8_flag(pic, entropy, mx, my, mb.slice)?;
    }
    let mut delta = 0;
    if luma_cbp > 0 || chroma_cbp > 0 {
        delta = qp_delta(entropy, prev_qpd(pic, mx, my, mb.slice))?;
        *qp = (*qp + delta).rem_euclid(52);
    }
    mb.qpd = delta as i8;
    predict_inter(pic, dpb, slice, mx, my, mb)?;
    if mb.transform8 {
        for i in 0..4 {
            if luma_cbp & (1 << i) != 0 {
                let block = read8(pic, entropy, mx, my, i, mb)?;
                let bx = (i & 1) * 8;
                let by = (i >> 1) * 8;
                super::add_scaled(
                    &mut pic.y,
                    pic.w,
                    (mx * 16 + bx, my * 16 + by),
                    &block,
                    *qp,
                    &slice.pps.weight8[1],
                    8,
                );
            }
        }
    } else {
        for i in 0..16 {
            let bit = 1 << ((blk_y(i) / 2) * 2 + blk_x(i) / 2);
            if luma_cbp & bit != 0 {
                let block = read_block(
                    pic,
                    entropy,
                    (2, 16, false),
                    Place { mx, my },
                    Cell {
                        bx: blk_x(i),
                        by: blk_y(i),
                    },
                    mb,
                    0,
                )?;
                super::add_scaled(
                    &mut pic.y,
                    pic.w,
                    (mx * 16 + blk_x(i) * 4, my * 16 + blk_y(i) * 4),
                    &block,
                    *qp,
                    &slice.pps.weight4[3],
                    4,
                );
            }
        }
    }
    let (cb_dc, cr_dc, cb_ac, cr_ac) = read_chroma(pic, entropy, mx, my, chroma_cbp, mb)?;
    recon_chroma(
        pic,
        slice,
        Place { mx, my },
        0,
        ChromaPaint {
            cb_dc,
            cr_dc,
            cb_ac: &cb_ac,
            cr_ac: &cr_ac,
        },
        (*qp, false, mb.slice),
    );
    Ok(())
}

fn read_parts(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    spec: u32,
    mb: &mut Mb,
) -> Result<(), CodecError> {
    let parts = partitions(slice.kind, spec);
    // Clause 7.3.5.1: every list 0 value, then every list 1 value.
    for part in &parts {
        if part.l0 && slice.num_ref_l0 > 1 {
            let r = read_te(
                entropy,
                slice.num_ref_l0,
                ref_ctx(pic, mx, my, part.x / 4, part.y / 4, 0, mb),
            )?;
            fill_ref(&mut mb.ref0, part, r as i8);
        } else if part.l0 {
            fill_ref(&mut mb.ref0, part, 0);
        }
    }
    for part in &parts {
        if part.l1 && slice.num_ref_l1 > 1 {
            let r = read_te(
                entropy,
                slice.num_ref_l1,
                ref_ctx(pic, mx, my, part.x / 4, part.y / 4, 1, mb),
            )?;
            fill_ref(&mut mb.ref1, part, r as i8);
        } else if part.l1 {
            fill_ref(&mut mb.ref1, part, 0);
        }
    }
    for part in &parts {
        if part.l0 {
            read_mvd_part(pic, entropy, slice, Place { mx, my }, part, 0, mb)?;
        }
    }
    for part in &parts {
        if part.l1 {
            read_mvd_part(pic, entropy, slice, Place { mx, my }, part, 1, mb)?;
        }
    }
    Ok(())
}

struct Part {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    l0: bool,
    l1: bool,
}

fn partitions(kind: u8, spec: u32) -> Vec<Part> {
    if kind == 0 {
        return match spec {
            0 => vec![part(0, 0, 16, 16, true, false)],
            1 => vec![
                part(0, 0, 16, 8, true, false),
                part(0, 8, 16, 8, true, false),
            ],
            2 => vec![
                part(0, 0, 8, 16, true, false),
                part(8, 0, 8, 16, true, false),
            ],
            _ => Vec::new(),
        };
    }
    let (a, b, wide) = match spec {
        1 => (1, 0, true),
        2 => (0, 1, true),
        3 => (1, 1, true),
        4 => (1, 1, true),
        5 => (1, 1, false),
        6 => (0, 1, true),
        7 => (0, 1, false),
        8 => (1, 0, true),
        9 => (1, 0, false),
        10 => (0, 1, true),
        11 => (0, 1, false),
        12 => (1, 1, true),
        13 => (1, 1, false),
        14 => (0, 1, true),
        15 => (0, 1, false),
        16 => (1, 1, true),
        17 => (1, 1, false),
        18 => (1, 0, true),
        19 => (1, 0, false),
        20 => (1, 1, true),
        21 => (1, 1, false),
        _ => return vec![part(0, 0, 16, 16, true, false)],
    };
    let _ = (a, b);
    let lists = b_lists(spec);
    if wide && matches!(spec, 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20) {
        vec![
            part(0, 0, 16, 8, lists.0.0, lists.0.1),
            part(0, 8, 16, 8, lists.1.0, lists.1.1),
        ]
    } else if matches!(spec, 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21) {
        vec![
            part(0, 0, 8, 16, lists.0.0, lists.0.1),
            part(8, 0, 8, 16, lists.1.0, lists.1.1),
        ]
    } else {
        vec![part(0, 0, 16, 16, lists.0.0, lists.0.1)]
    }
}

fn b_lists(spec: u32) -> ((bool, bool), (bool, bool)) {
    match spec {
        1 => ((true, false), (true, false)),
        2 => ((false, true), (false, true)),
        3 => ((true, true), (true, true)),
        4 | 5 => ((true, false), (true, false)),
        6 | 7 => ((false, true), (false, true)),
        8 | 9 => ((true, false), (false, true)),
        10 | 11 => ((false, true), (true, false)),
        12 | 13 => ((true, false), (true, true)),
        14 | 15 => ((false, true), (true, true)),
        16 | 17 => ((true, true), (true, false)),
        18 | 19 => ((true, true), (false, true)),
        _ => ((true, true), (true, true)),
    }
}

fn part(x: usize, y: usize, w: usize, h: usize, l0: bool, l1: bool) -> Part {
    Part { x, y, w, h, l0, l1 }
}

fn fill_ref(refs: &mut [i8; 4], part: &Part, value: i8) {
    for y in (part.y / 8)..((part.y + part.h) / 8) {
        for x in (part.x / 8)..((part.x + part.w) / 8) {
            refs[y * 2 + x] = value;
        }
    }
}

fn read_te(entropy: &mut Entropy<'_, '_>, range: usize, ctx: usize) -> Result<u32, CodecError> {
    if range <= 1 {
        return Ok(0);
    }
    match entropy {
        Entropy::Cavlc(bits) => {
            if range == 2 {
                Ok(1 - u32::from(bits.bit()?))
            } else {
                bits.ue()
            }
        }
        Entropy::Cabac(cab) => {
            let mut value = 0u32;
            let mut context = ctx;
            while cab.decision(54 + context)? == 1 {
                value += 1;
                context = (context >> 2) + 4;
                if value >= range as u32 {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "ref idx",
                    });
                }
            }
            Ok(value)
        }
    }
}

fn ref_ctx(pic: &Frame, mx: usize, my: usize, bx: usize, by: usize, list: usize, mb: &Mb) -> usize {
    let left = ref_at(pic, mx, my, bx as i32 - 1, by as i32, list, mb);
    let above = ref_at(pic, mx, my, bx as i32, by as i32 - 1, list, mb);
    usize::from(left > 0) + 2 * usize::from(above > 0)
}

fn ref_at(pic: &Frame, mx: usize, my: usize, bx: i32, by: i32, list: usize, current: &Mb) -> i8 {
    let (mbx, ox) = if bx < 0 {
        (mx as i32 - 1, 1)
    } else {
        (mx as i32, (bx as usize) / 2)
    };
    let (mby, oy) = if by < 0 {
        (my as i32 - 1, 1)
    } else {
        (my as i32, (by as usize) / 2)
    };
    let index = oy * 2 + ox.min(1);
    if mbx == mx as i32 && mby == my as i32 {
        return if list == 0 {
            current.ref0[index]
        } else {
            current.ref1[index]
        };
    }
    neighbor(pic, mbx, mby, current.slice)
        .map(|mb| {
            if list == 0 {
                mb.ref0[index]
            } else {
                mb.ref1[index]
            }
        })
        .unwrap_or(-1)
}

fn read_mvd_part(
    pic: &Frame,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    part: &Part,
    list: usize,
    mb: &mut Mb,
) -> Result<(), CodecError> {
    let Place { mx, my } = at;
    let x = part.x / 4;
    let y = part.y / 4;
    let pred = mvp(
        pic,
        Place { mx, my },
        Cell { bx: x, by: y },
        (part.w, part.h),
        list,
        ref_of(mb, x, y, list),
        mb,
    );
    let dx = read_mvd(
        entropy,
        mvd_sum(pic, Place { mx, my }, Cell { bx: x, by: y }, (list, 0), mb),
        40,
    )?;
    let dy = read_mvd(
        entropy,
        mvd_sum(pic, Place { mx, my }, Cell { bx: x, by: y }, (list, 1), mb),
        47,
    )?;
    let mv = [
        super::clip_ref(i32::from(pred[0]) + dx),
        super::clip_ref(i32::from(pred[1]) + dy),
    ];
    let mvd = [super::clip_ref(dx), super::clip_ref(dy)];
    let bw = part.w / 4;
    let bh = part.h / 4;
    for by in 0..bh {
        for bx in 0..bw {
            let index = (y + by) * 4 + x + bx;
            if list == 0 {
                mb.mv0[index] = mv;
                mb.mvd0[index] = mvd;
            } else {
                mb.mv1[index] = mv;
                mb.mvd1[index] = mvd;
            }
            mb.ready |= 1 << index;
        }
    }
    let _ = slice;
    Ok(())
}

fn ref_of(mb: &Mb, bx: usize, by: usize, list: usize) -> i8 {
    let index = (by / 2) * 2 + (bx / 2);
    if list == 0 {
        mb.ref0[index]
    } else {
        mb.ref1[index]
    }
}

fn read_mvd(entropy: &mut Entropy<'_, '_>, sum: i32, base: usize) -> Result<i32, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => bits.se(),
        Entropy::Cabac(cab) => {
            let ctx = usize::from(sum > 2) + usize::from(sum > 32);
            if cab.decision(base + ctx)? == 0 {
                return Ok(0);
            }
            let mut value = 1i32;
            let mut context = base + 3;
            while value < 9 && cab.decision(context)? == 1 {
                if value < 4 {
                    context += 1;
                }
                value += 1;
            }
            if value >= 9 {
                let mut k = 3i32;
                while cab.bypass()? == 1 {
                    value += 1 << k;
                    k += 1;
                    if k > 16 {
                        return Err(CodecError::Malformed {
                            at_bit: 0,
                            what: "mvd",
                        });
                    }
                }
                for bit in (0..k).rev() {
                    value += i32::from(cab.bypass()?) << bit;
                }
            }
            let sign = cab.bypass()?;
            Ok(if sign == 1 { -value } else { value })
        }
    }
}

fn mvd_sum(pic: &Frame, at: Place, cell: Cell, which: (usize, usize), mb: &Mb) -> i32 {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let (list, comp) = which;
    let left = mvd_at(
        pic,
        Place { mx, my },
        (bx as i32 - 1, by as i32),
        (list, comp),
        mb,
    );
    let above = mvd_at(
        pic,
        Place { mx, my },
        (bx as i32, by as i32 - 1),
        (list, comp),
        mb,
    );
    left.abs() + above.abs()
}

fn mvd_at(pic: &Frame, at: Place, cell: (i32, i32), which: (usize, usize), current: &Mb) -> i32 {
    let Place { mx, my } = at;
    let (bx, by) = cell;
    let (list, comp) = which;
    let (mbx, ox) = shift(mx, bx, 4);
    let (mby, oy) = shift(my, by, 4);
    let index = oy * 4 + ox;
    if mbx == mx as i32 && mby == my as i32 {
        let mv = if list == 0 {
            current.mvd0[index]
        } else {
            current.mvd1[index]
        };
        return i32::from(mv[comp]);
    }
    neighbor(pic, mbx, mby, current.slice)
        .map(|mb| {
            i32::from(if list == 0 {
                mb.mvd0[index][comp]
            } else {
                mb.mvd1[index][comp]
            })
        })
        .unwrap_or(0)
}

#[derive(Clone, Copy)]
struct Motion {
    available: bool,
    mv: [i16; 2],
    reference: i8,
}

fn mvp(
    pic: &Frame,
    at: Place,
    cell: Cell,
    size: (usize, usize),
    list: usize,
    reference: i8,
    current: &Mb,
) -> [i16; 2] {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let (w, h) = size;
    let a = part_motion(pic, mx, my, bx as i32 - 1, by as i32, list, current);
    let mut b = part_motion(pic, mx, my, bx as i32, by as i32 - 1, list, current);
    let above_right = part_motion(
        pic,
        mx,
        my,
        bx as i32 + (w / 4) as i32,
        by as i32 - 1,
        list,
        current,
    );
    let mut c = if above_right.available {
        above_right
    } else {
        part_motion(pic, mx, my, bx as i32 - 1, by as i32 - 1, list, current)
    };
    if w == 16 && h == 8 && by == 0 && b.reference == reference {
        return b.mv;
    }
    if w == 16 && h == 8 && by == 2 && a.reference == reference {
        return a.mv;
    }
    if w == 8 && h == 16 && bx == 0 && a.reference == reference {
        return a.mv;
    }
    if w == 8 && h == 16 && bx == 2 && c.reference == reference {
        return c.mv;
    }
    if !b.available && !c.available && a.available {
        b = a;
        c = a;
    }
    let same = [a, b, c]
        .into_iter()
        .filter(|n| n.reference == reference)
        .collect::<Vec<_>>();
    if same.len() == 1 {
        return same[0].mv;
    }
    [
        median(a.mv[0], b.mv[0], c.mv[0]),
        median(a.mv[1], b.mv[1], c.mv[1]),
    ]
}

fn median(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.max(c)).min(b.max(c))
}

fn part_motion(
    pic: &Frame,
    mx: usize,
    my: usize,
    bx: i32,
    by: i32,
    list: usize,
    current: &Mb,
) -> Motion {
    let (mbx, ox) = shift(mx, bx, 4);
    let (mby, oy) = shift(my, by, 4);
    let index = oy * 4 + ox;
    if mbx == mx as i32 && mby == my as i32 {
        if current.ready & (1 << index) == 0 {
            return Motion {
                available: false,
                mv: [0, 0],
                reference: -1,
            };
        }
        return motion_of(current, index, list);
    }
    let Some(mb) = neighbor(pic, mbx, mby, current.slice) else {
        return Motion {
            available: false,
            mv: [0, 0],
            reference: -1,
        };
    };
    motion_of(mb, index, list)
}

fn motion_of(mb: &Mb, index: usize, list: usize) -> Motion {
    if mb.intra {
        return Motion {
            available: true,
            mv: [0, 0],
            reference: -1,
        };
    }
    let part = (index / 4 / 2) * 2 + ((index % 4) / 2);
    let reference = if list == 0 {
        mb.ref0[part]
    } else {
        mb.ref1[part]
    };
    if reference < 0 {
        return Motion {
            available: true,
            mv: [0, 0],
            reference: -1,
        };
    }
    let mv = if list == 0 {
        mb.mv0[index]
    } else {
        mb.mv1[index]
    };
    Motion {
        available: true,
        mv,
        reference,
    }
}

fn read_sub(
    pic: &Frame,
    dpb: &[Frame],
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    spec: u32,
    mb: &mut Mb,
) -> Result<(), CodecError> {
    let Place { mx, my } = at;
    let mut kinds = [0u32; 4];
    for slot in &mut kinds {
        *slot = match entropy {
            Entropy::Cavlc(bits) => bits.ue()?,
            Entropy::Cabac(cab) => {
                if slice.kind == 0 {
                    if cab.decision(21)? == 1 {
                        0
                    } else if cab.decision(22)? == 0 {
                        1
                    } else if cab.decision(23)? == 1 {
                        2
                    } else {
                        3
                    }
                } else if cab.decision(36)? == 0 {
                    0
                } else if cab.decision(37)? == 0 {
                    1 + u32::from(cab.decision(39)?)
                } else {
                    let mut kind = 3u32;
                    if cab.decision(38)? == 1 {
                        if cab.decision(39)? == 1 {
                            *slot = 11 + u32::from(cab.decision(39)?);
                            continue;
                        }
                        kind += 4;
                    }
                    kind += 2 * u32::from(cab.decision(39)?);
                    kind += u32::from(cab.decision(39)?);
                    kind
                }
            }
        };
    }
    for kind in kinds {
        if kind == 0 && slice.kind == 1 {
            if !slice.sps.direct_8x8_inference {
                mb.partition8 = false;
            }
            continue;
        }
        let (_, _, w, h) = sub_shape(slice.kind, kind);
        if w < 8 || h < 8 {
            mb.partition8 = false;
        }
    }
    let ref0_forced = spec == 4;
    let direct = |kind: u32| kind == 0 && slice.kind == 1;
    // Clause 7.3.5.2: list 0 references, list 1 references, list 0 deltas, list 1 deltas.
    for (i, kind) in kinds.iter().copied().enumerate() {
        if direct(kind) {
            continue;
        }
        let (l0, _) = sub_lists(slice.kind, kind);
        if !l0 {
            continue;
        }
        let x = (i & 1) * 2;
        let y = (i >> 1) * 2;
        let r = if ref0_forced || slice.num_ref_l0 <= 1 {
            0
        } else {
            read_te(entropy, slice.num_ref_l0, ref_ctx(pic, mx, my, x, y, 0, mb))? as i8
        };
        mb.ref0[(y / 2) * 2 + (x / 2)] = r;
    }
    for (i, kind) in kinds.iter().copied().enumerate() {
        if direct(kind) {
            continue;
        }
        let (_, l1) = sub_lists(slice.kind, kind);
        if !l1 {
            continue;
        }
        let x = (i & 1) * 2;
        let y = (i >> 1) * 2;
        let r = if slice.num_ref_l1 <= 1 {
            0
        } else {
            read_te(entropy, slice.num_ref_l1, ref_ctx(pic, mx, my, x, y, 1, mb))? as i8
        };
        mb.ref1[(y / 2) * 2 + (x / 2)] = r;
    }
    for (i, kind) in kinds.iter().copied().enumerate() {
        if direct(kind) {
            let x0 = (i & 1) * 8;
            let y0 = (i >> 1) * 8;
            direct_8(pic, dpb, slice, Place { mx, my }, (x0, y0), mb);
        }
    }
    for list in 0..2 {
        for (i, kind) in kinds.iter().copied().enumerate() {
            if direct(kind) {
                continue;
            }
            let (l0, l1, w, h) = sub_shape(slice.kind, kind);
            if (list == 0 && !l0) || (list == 1 && !l1) {
                continue;
            }
            let x0 = (i & 1) * 8;
            let y0 = (i >> 1) * 8;
            for (px, py) in sub_origins(x0, y0, w, h) {
                let part = part(px, py, w, h, list == 0, list == 1);
                read_mvd_part(pic, entropy, slice, Place { mx, my }, &part, list, mb)?;
            }
        }
    }
    Ok(())
}

fn sub_origins(x0: usize, y0: usize, w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut origins = Vec::new();
    let mut y = y0;
    while y < y0 + 8 {
        let mut x = x0;
        while x < x0 + 8 {
            origins.push((x, y));
            x += w;
        }
        y += h;
    }
    origins
}

fn sub_lists(kind: u8, spec: u32) -> (bool, bool) {
    if kind == 0 {
        return (true, false);
    }
    match spec {
        0 | 1 | 4 | 5 | 10 => (true, false),
        2 | 6 | 7 | 11 => (false, true),
        _ => (true, true),
    }
}

fn sub_shape(kind: u8, spec: u32) -> (bool, bool, usize, usize) {
    let (l0, l1) = sub_lists(kind, spec);
    let (w, h) = if kind == 0 {
        match spec {
            1 => (8, 4),
            2 => (4, 8),
            3 => (4, 4),
            _ => (8, 8),
        }
    } else {
        match spec {
            4 | 6 | 8 => (8, 4),
            5 | 7 | 9 => (4, 8),
            10..=12 => (4, 4),
            _ => (8, 8),
        }
    };
    (l0, l1, w, h)
}

fn direct_8(
    pic: &Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    at: Place,
    origin: (usize, usize),
    mb: &mut Mb,
) {
    let Place { mx, my } = at;
    let (x, y) = origin;
    let mut blocks = [(0usize, 0usize); 4];
    let mut n = 0;
    for by in 0..2 {
        for bx in 0..2 {
            blocks[n] = (x / 4 + bx, y / 4 + by);
            n += 1;
        }
    }
    apply_direct(pic, dpb, slice, mx, my, mb, &blocks);
}

fn direct_16(
    pic: &Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
) -> Result<(), CodecError> {
    let mut blocks = [(0usize, 0usize); 16];
    for (index, slot) in blocks.iter_mut().enumerate() {
        *slot = (index % 4, index / 4);
    }
    apply_direct(pic, dpb, slice, mx, my, mb, &blocks);
    Ok(())
}

/// Clause 8.4.1.2.3. Scale the co-located motion vector by the picture-order distance.
fn temporal_direct(
    pic: &Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
    blocks: &[(usize, usize)],
) {
    let addr = my * pic.mb_w + mx;
    let col = slice
        .ref1
        .first()
        .and_then(|&index| dpb.get(index))
        .and_then(|frame| frame.mb.get(addr).map(|col| (frame.poc, col)));
    for &(bx, by) in blocks {
        let index = by * 4 + bx;
        let part = (by / 2) * 2 + (bx / 2);
        let (ref0, mv0, mv1) = temporal_block(slice, col, bx, by);
        mb.mv0[index] = mv0;
        mb.mv1[index] = mv1;
        mb.ref0[part] = ref0;
        mb.ref1[part] = 0;
        mb.ready |= 1 << index;
    }
}

fn temporal_block(
    slice: &Slice<'_>,
    col: Option<(i32, &Mb)>,
    bx: usize,
    by: usize,
) -> (i8, [i16; 2], [i16; 2]) {
    let Some((col_poc, col)) = col else {
        return (0, [0, 0], [0, 0]);
    };
    if col.intra || col.pcm {
        return (0, [0, 0], [0, 0]);
    }
    let part = (by / 2) * 2 + (bx / 2);
    let index = if slice.sps.direct_8x8_inference {
        (by & !1) * 4 + (bx & !1)
    } else {
        by * 4 + bx
    };
    let (mv_col, ref_poc) = if col.ref0[part] >= 0 {
        (col.mv0[index], col.ref_poc0[part])
    } else if col.ref1[part] >= 0 {
        (col.mv1[index], col.ref_poc1[part])
    } else {
        return (0, [0, 0], [0, 0]);
    };
    let ref0 = slice
        .poc0
        .iter()
        .position(|&poc| poc == ref_poc)
        .and_then(|index| i8::try_from(index).ok())
        .unwrap_or(0);
    let Some(&pic0) = slice.poc0.get(usize::from(u8::try_from(ref0).unwrap_or(0))) else {
        return (0, [0, 0], [0, 0]);
    };
    let long = slice
        .long0
        .get(usize::from(u8::try_from(ref0).unwrap_or(0)))
        .copied()
        .unwrap_or(false);
    let td = (col_poc - pic0).clamp(-128, 127);
    if long || td == 0 {
        return (ref0, mv_col, [0, 0]);
    }
    let tb = (slice.poc - pic0).clamp(-128, 127);
    let tx = (16_384 + (td.abs() / 2)) / td;
    let factor = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
    let mv0 = [scale_mv(factor, mv_col[0]), scale_mv(factor, mv_col[1])];
    let mv1 = [
        clip_mv(i32::from(mv0[0]) - i32::from(mv_col[0])),
        clip_mv(i32::from(mv0[1]) - i32::from(mv_col[1])),
    ];
    (ref0, mv0, mv1)
}

fn scale_mv(factor: i32, component: i16) -> i16 {
    clip_mv((factor * i32::from(component) + 128) >> 8)
}

fn clip_mv(value: i32) -> i16 {
    i16::try_from(value.clamp(-32_768, 32_767)).unwrap_or(0)
}

/// Clause 8.4.1.2.2. One reference pair for the macroblock; each 4×4 may be forced to a zero vector.
fn apply_direct(
    pic: &Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
    blocks: &[(usize, usize)],
) {
    if !slice.spatial_direct {
        temporal_direct(pic, dpb, slice, mx, my, mb, blocks);
        return;
    }
    let (ref0, ref1, zero) = direct_refs(pic, mx, my, mb);
    let pred0 = if zero || ref0 < 0 {
        [0, 0]
    } else {
        mvp(
            pic,
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            (16, 16),
            0,
            ref0,
            mb,
        )
    };
    let pred1 = if zero || ref1 < 0 {
        [0, 0]
    } else {
        mvp(
            pic,
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            (16, 16),
            1,
            ref1,
            mb,
        )
    };
    let addr = my * pic.mb_w + mx;
    for &(bx, by) in blocks {
        let index = by * 4 + bx;
        let col = if slice.sps.direct_8x8_inference {
            (by & !1) * 4 + (bx & !1)
        } else {
            index
        };
        let col_zero = col_zero(dpb, slice, addr, col);
        mb.mv0[index] = if zero || ref0 < 0 || (ref0 == 0 && col_zero) {
            [0, 0]
        } else {
            pred0
        };
        mb.mv1[index] = if zero || ref1 < 0 || (ref1 == 0 && col_zero) {
            [0, 0]
        } else {
            pred1
        };
        let part = (by / 2) * 2 + bx / 2;
        mb.ref0[part] = if zero { 0 } else { ref0 };
        mb.ref1[part] = if zero { 0 } else { ref1 };
        mb.ready |= 1 << index;
    }
}

fn direct_refs(pic: &Frame, mx: usize, my: usize, mb: &Mb) -> (i8, i8, bool) {
    let reference = |list: usize| {
        let a = part_motion(pic, mx, my, -1, 0, list, mb);
        let b = part_motion(pic, mx, my, 0, -1, list, mb);
        let right = part_motion(pic, mx, my, 4, -1, list, mb);
        let c = if right.available {
            right
        } else {
            part_motion(pic, mx, my, -1, -1, list, mb)
        };
        min_positive(a.reference, min_positive(b.reference, c.reference))
    };
    let mut ref0 = reference(0);
    let mut ref1 = reference(1);
    if ref0 < 0 && ref1 < 0 {
        ref0 = 0;
        ref1 = 0;
        return (ref0, ref1, true);
    }
    (ref0, ref1, false)
}

fn min_positive(left: i8, right: i8) -> i8 {
    if left >= 0 && right >= 0 {
        left.min(right)
    } else {
        left.max(right)
    }
}

fn col_zero(dpb: &[Frame], slice: &Slice<'_>, addr: usize, index: usize) -> bool {
    let Some(&frame_index) = slice.ref1.first() else {
        return false;
    };
    let Some(frame) = dpb.get(frame_index) else {
        return false;
    };
    if frame.long_term {
        return false;
    }
    let Some(col) = frame.mb.get(addr) else {
        return false;
    };
    if col.intra || col.pcm {
        return false;
    }
    let part = (index / 4 / 2) * 2 + ((index % 4) / 2);
    let (mv, reference) = if col.ref0[part] >= 0 {
        (col.mv0[index], col.ref0[part])
    } else if col.ref1[part] >= 0 {
        (col.mv1[index], col.ref1[part])
    } else {
        return false;
    };
    reference == 0 && (-1..=1).contains(&mv[0]) && (-1..=1).contains(&mv[1])
}

fn predict_inter(
    pic: &mut Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &Mb,
) -> Result<(), CodecError> {
    let mut done = [false; 16];
    for by in 0..4 {
        for bx in 0..4 {
            let index = by * 4 + bx;
            if done[index] {
                continue;
            }
            let r0 = ref_of(mb, bx, by, 0);
            let r1 = ref_of(mb, bx, by, 1);
            let mv0 = mb.mv0[index];
            let mv1 = mb.mv1[index];
            let mut bw = 1usize;
            while bx + bw < 4 && same_motion(mb, bx + bw, by, r0, r1, mv0, mv1) {
                bw += 1;
            }
            let mut bh = 1usize;
            while by + bh < 4 {
                let row = (0..bw).all(|dx| {
                    !done[(by + bh) * 4 + bx + dx]
                        && same_motion(mb, bx + dx, by + bh, r0, r1, mv0, mv1)
                });
                if !row {
                    break;
                }
                bh += 1;
            }
            for dy in 0..bh {
                for dx in 0..bw {
                    done[(by + dy) * 4 + bx + dx] = true;
                }
            }
            mc_block(
                pic,
                dpb,
                slice,
                Block {
                    x: mx * 16 + bx * 4,
                    y: my * 16 + by * 4,
                    w: bw * 4,
                    h: bh * 4,
                },
                McMotion { r0, r1, mv0, mv1 },
            )?;
        }
    }
    Ok(())
}

fn same_motion(
    mb: &Mb,
    bx: usize,
    by: usize,
    r0: i8,
    r1: i8,
    mv0: [i16; 2],
    mv1: [i16; 2],
) -> bool {
    let index = by * 4 + bx;
    ref_of(mb, bx, by, 0) == r0
        && ref_of(mb, bx, by, 1) == r1
        && mb.mv0[index] == mv0
        && mb.mv1[index] == mv1
}

struct Block {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

struct McMotion {
    r0: i8,
    r1: i8,
    mv0: [i16; 2],
    mv1: [i16; 2],
}

fn mc_block(
    pic: &mut Frame,
    dpb: &[Frame],
    slice: &Slice<'_>,
    block: Block,
    motion: McMotion,
) -> Result<(), CodecError> {
    let Block { x, y, w, h } = block;
    let McMotion { r0, r1, mv0, mv1 } = motion;
    let mut luma0 = [0u8; 256];
    let mut luma1 = [0u8; 256];
    if r0 >= 0 {
        let frame = reference_frame(dpb, slice, 0, r0)?;
        predict_luma(&mut luma0, frame, x, y, w, h, mv0);
    }
    if r1 >= 0 {
        let frame = reference_frame(dpb, slice, 1, r1)?;
        predict_luma(&mut luma1, frame, x, y, w, h, mv1);
    }
    let cw = w / 2;
    let ch = h / 2;
    let mut u0 = [0u8; 64];
    let mut v0 = [0u8; 64];
    let mut u1 = [0u8; 64];
    let mut v1 = [0u8; 64];
    if r0 >= 0 && cw > 0 && ch > 0 {
        let frame = reference_frame(dpb, slice, 0, r0)?;
        predict_chroma((&mut u0, &mut v0), frame, (x, y), (cw, ch), mv0);
    }
    if r1 >= 0 && cw > 0 && ch > 0 {
        let frame = reference_frame(dpb, slice, 1, r1)?;
        predict_chroma((&mut u1, &mut v1), frame, (x, y), (cw, ch), mv1);
    }
    store_block(
        &mut pic.y,
        pic.w,
        (x, y),
        (w, h),
        (&luma0, &luma1),
        mix(slice, r0, r1, -1),
        0,
    );
    if cw > 0 && ch > 0 {
        let stride = pic.w / 2;
        let origin = (x / 2, y / 2);
        let size = (cw, ch);
        store_block(
            &mut pic.u,
            stride,
            origin,
            size,
            (&u0, &u1),
            mix(slice, r0, r1, 0),
            128,
        );
        store_block(
            &mut pic.v,
            stride,
            origin,
            size,
            (&v0, &v1),
            mix(slice, r0, r1, 1),
            128,
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Mix {
    Fill,
    Copy(usize),
    Uni {
        weight: i32,
        offset: i32,
        denom: i32,
        from: usize,
    },
    Avg,
    Weighted(recon::BiWeight),
}

fn mix(slice: &Slice<'_>, r0: i8, r1: i8, plane: i8) -> Mix {
    if r0 >= 0 && r1 >= 0 {
        if slice.weighted {
            Mix::Weighted(explicit_bi(slice, r0, r1, plane))
        } else if slice.pps.weighted_bipred == 2 && slice.kind == 1 {
            let (w0, w1) = implicit_weights(slice, r0, r1);
            Mix::Weighted(recon::BiWeight {
                w0,
                w1,
                o0: 0,
                o1: 0,
                denom: 5,
            })
        } else {
            Mix::Avg
        }
    } else if r0 >= 0 {
        uni_mix(slice, 0, r0, plane)
    } else if r1 >= 0 {
        uni_mix(slice, 1, r1, plane)
    } else {
        Mix::Fill
    }
}

fn explicit_bi(slice: &Slice<'_>, r0: i8, r1: i8, plane: i8) -> recon::BiWeight {
    let left = usize::try_from(r0).unwrap_or(0);
    let right = usize::try_from(r1).unwrap_or(0);
    if plane < 0 {
        recon::BiWeight {
            w0: slice.yw[0][left],
            w1: slice.yw[1][right],
            o0: slice.yo[0][left],
            o1: slice.yo[1][right],
            denom: slice.luma_denom,
        }
    } else {
        let plane = usize::try_from(plane).unwrap_or(0);
        recon::BiWeight {
            w0: slice.cw[0][left][plane],
            w1: slice.cw[1][right][plane],
            o0: slice.co[0][left][plane],
            o1: slice.co[1][right][plane],
            denom: slice.chroma_denom,
        }
    }
}

fn uni_mix(slice: &Slice<'_>, list: usize, reference: i8, plane: i8) -> Mix {
    if !slice.weighted {
        return Mix::Copy(list);
    }
    let index = usize::try_from(reference).unwrap_or(0);
    if plane < 0 {
        Mix::Uni {
            weight: slice.yw[list][index],
            offset: slice.yo[list][index],
            denom: slice.luma_denom,
            from: list,
        }
    } else {
        let plane = usize::try_from(plane).unwrap_or(0);
        Mix::Uni {
            weight: slice.cw[list][index][plane],
            offset: slice.co[list][index][plane],
            denom: slice.chroma_denom,
            from: list,
        }
    }
}

fn store_block(
    dst: &mut [u8],
    stride: usize,
    origin: (usize, usize),
    size: (usize, usize),
    pred: (&[u8], &[u8]),
    mix: Mix,
    fill: u8,
) {
    let (x, y) = origin;
    let (w, h) = size;
    let (left, right) = pred;
    match mix {
        Mix::Fill => {
            for row in 0..h {
                let at = (y + row) * stride + x;
                dst[at..at + w].fill(fill);
            }
        }
        Mix::Copy(which) => {
            let src = if which == 0 { left } else { right };
            for row in 0..h {
                let at = (y + row) * stride + x;
                dst[at..at + w].copy_from_slice(&src[row * w..row * w + w]);
            }
        }
        Mix::Uni {
            weight,
            offset,
            denom,
            from,
        } => {
            let src = if from == 0 { left } else { right };
            for row in 0..h {
                let at = (y + row) * stride + x;
                for col in 0..w {
                    dst[at + col] = recon::weight_uni(src[row * w + col], weight, offset, denom);
                }
            }
        }
        Mix::Avg => {
            for row in 0..h {
                let at = (y + row) * stride + x;
                let a = &left[row * w..row * w + w];
                let b = &right[row * w..row * w + w];
                for col in 0..w {
                    dst[at + col] = recon::avg(a[col], b[col]);
                }
            }
        }
        Mix::Weighted(weight)
            if weight.w0 == 32
                && weight.w1 == 32
                && weight.o0 == 0
                && weight.o1 == 0
                && weight.denom == 5 =>
        {
            for row in 0..h {
                let at = (y + row) * stride + x;
                let a = &left[row * w..row * w + w];
                let b = &right[row * w..row * w + w];
                for col in 0..w {
                    dst[at + col] = recon::avg(a[col], b[col]);
                }
            }
        }
        Mix::Weighted(weight) => {
            for row in 0..h {
                let at = (y + row) * stride + x;
                recon::weight_row(
                    &mut dst[at..at + w],
                    &left[row * w..row * w + w],
                    &right[row * w..row * w + w],
                    weight,
                );
            }
        }
    }
}

fn reference_frame<'a>(
    dpb: &'a [Frame],
    slice: &Slice<'_>,
    list: usize,
    reference: i8,
) -> Result<&'a Frame, CodecError> {
    let refs = if list == 0 { &slice.ref0 } else { &slice.ref1 };
    let index = usize::try_from(reference).map_err(|_| CodecError::Malformed {
        at_bit: 0,
        what: "ref idx",
    })?;
    let Some(&frame_index) = refs.get(index) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "ref idx",
        });
    };
    let Some(frame) = dpb.get(frame_index) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "dpb",
        });
    };
    if frame.nonexisting {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "non-existing reference",
        });
    }
    Ok(frame)
}

fn predict_luma(
    dst: &mut [u8],
    frame: &Frame,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv: [i16; 2],
) {
    let mvx = i32::from(mv[0]);
    let mvy = i32::from(mv[1]);
    recon::put_luma_block(
        dst,
        w,
        &frame.y,
        frame.w,
        (
            i32::try_from(frame.w).unwrap_or(0),
            i32::try_from(frame.h).unwrap_or(0),
        ),
        recon::BlockMc {
            x: i32::try_from(x).unwrap_or(0) + (mvx >> 2),
            y: i32::try_from(y).unwrap_or(0) + (mvy >> 2),
            w,
            h,
            x_frac: mvx & 3,
            y_frac: mvy & 3,
        },
    );
}

fn predict_chroma(
    planes: (&mut [u8], &mut [u8]),
    frame: &Frame,
    at: (usize, usize),
    size: (usize, usize),
    mv: [i16; 2],
) {
    let (u, v) = planes;
    let (x, y) = at;
    let (w, h) = size;
    let mvx = i32::from(mv[0]);
    let mvy = i32::from(mv[1]);
    let block = recon::BlockMc {
        x: i32::try_from(x / 2).unwrap_or(0) + (mvx >> 3),
        y: i32::try_from(y / 2).unwrap_or(0) + (mvy >> 3),
        w,
        h,
        x_frac: mvx & 7,
        y_frac: mvy & 7,
    };
    let pic = (
        i32::try_from(frame.w / 2).unwrap_or(0),
        i32::try_from(frame.h / 2).unwrap_or(0),
    );
    recon::put_chroma_block(u, w, &frame.u, frame.w / 2, pic, block);
    recon::put_chroma_block(v, w, &frame.v, frame.w / 2, pic, block);
}

/// Clause 8.4.3 implicit bipred weights from the picture-order distance.
fn implicit_weights(slice: &Slice<'_>, r0: i8, r1: i8) -> (i32, i32) {
    let Some(poc0) = listed(&slice.poc0, r0) else {
        return (32, 32);
    };
    let Some(poc1) = listed(&slice.poc1, r1) else {
        return (32, 32);
    };
    let long = listed(&slice.long0, r0).unwrap_or(true) || listed(&slice.long1, r1).unwrap_or(true);
    let td = (poc1 - poc0).clamp(-128, 127);
    if td == 0 || long {
        return (32, 32);
    }
    let tb = (slice.poc - poc0).clamp(-128, 127);
    let tx = (16_384 + (td / 2).abs()) / td;
    let dist = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
    let w1 = dist >> 2;
    if !(-64..=128).contains(&w1) {
        return (32, 32);
    }
    (64 - w1, w1)
}

fn listed<T: Copy>(values: &[T], index: i8) -> Option<T> {
    let index = usize::try_from(index).ok()?;
    values.get(index).copied()
}

fn read_pcm(
    pic: &mut Frame,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    slice: &Slice<'_>,
) -> Result<(), CodecError> {
    let x0 = mx * 16;
    let y0 = my * 16;
    match entropy {
        Entropy::Cavlc(bits) => {
            while bits.pos() % 8 != 0 {
                if bits.bit()? {
                    return Err(bits.malformed("pcm align"));
                }
            }
            for row in 0..16 {
                for col in 0..16 {
                    pic.y[(y0 + row) * pic.w + x0 + col] = bits.u(8)? as u8;
                }
            }
            for plane in [&mut pic.u, &mut pic.v] {
                let stride = pic.w / 2;
                for row in 0..8 {
                    for col in 0..8 {
                        plane[(y0 / 2 + row) * stride + x0 / 2 + col] = bits.u(8)? as u8;
                    }
                }
            }
        }
        Entropy::Cabac(cab) => {
            let bytes = cab.tail();
            if bytes.len() < 384 {
                return Err(CodecError::Truncated {
                    at_bit: 0,
                    what: "pcm",
                });
            }
            for row in 0..16 {
                for col in 0..16 {
                    pic.y[(y0 + row) * pic.w + x0 + col] = bytes[row * 16 + col];
                }
            }
            for row in 0..8 {
                for col in 0..8 {
                    let stride = pic.w / 2;
                    pic.u[(y0 / 2 + row) * stride + x0 / 2 + col] = bytes[256 + row * 8 + col];
                    pic.v[(y0 / 2 + row) * stride + x0 / 2 + col] = bytes[320 + row * 8 + col];
                }
            }
            let rest = &bytes[384..];
            **cab = Cabac::open(rest, slice.qp, if slice.kind == 2 { 0 } else { 1 })?;
        }
    }
    Ok(())
}

fn neighbor(pic: &Frame, mx: i32, my: i32, slice: u16) -> Option<&Mb> {
    if mx < 0 || my < 0 || mx as usize >= pic.mb_w || my as usize >= pic.h / 16 {
        return None;
    }
    let mb = pic.mb.get(my as usize * pic.mb_w + mx as usize)?;
    if mb.slice != slice {
        return None;
    }
    Some(mb)
}

fn xy(pic: &Frame, addr: usize) -> (usize, usize) {
    (addr % pic.mb_w, addr / pic.mb_w)
}

pub(crate) fn deblock(pic: &mut Frame) {
    let mb_w = pic.mb_w;
    let mb_h = pic.h / 16;
    for my in 0..mb_h {
        for mx in 0..mb_w {
            let index = my * mb_w + mx;
            if pic
                .mb
                .get(index)
                .is_some_and(|mb| mb.deblock == 1 || mb.pcm)
            {
                continue;
            }
            let transform8 = pic.mb.get(index).is_some_and(|mb| mb.transform8);
            let off = offsets(pic, index);
            for edge in 0..4 {
                if edge == 0 && (mx == 0 || !filter_boundary(pic, mx, my, true)) {
                    continue;
                }
                let x = mx * 16 + edge * 4;
                let bs = edge_strength(pic, mx, my, edge, true);
                // An 8×8 transform has no luma edge at 4 or 12.
                if x >= 4 && !(transform8 && (edge == 1 || edge == 3)) {
                    let qpav = edge_qp(pic, mx, my, true, true, edge == 0);
                    filter_groups(
                        &mut pic.y,
                        pic.w,
                        (x, my * 16),
                        (true, false, 4),
                        bs,
                        qpav,
                        off,
                    );
                }
                if edge % 2 == 0 {
                    let cx = mx * 8 + edge * 2;
                    if cx >= 2 {
                        let qpc = edge_qp(pic, mx, my, false, true, edge == 0);
                        filter_groups(
                            &mut pic.u,
                            pic.w / 2,
                            (cx, my * 8),
                            (true, true, 2),
                            bs,
                            qpc,
                            off,
                        );
                        filter_groups(
                            &mut pic.v,
                            pic.w / 2,
                            (cx, my * 8),
                            (true, true, 2),
                            bs,
                            qpc,
                            off,
                        );
                    }
                }
            }
            for edge in 0..4 {
                if edge == 0 && (my == 0 || !filter_boundary(pic, mx, my, false)) {
                    continue;
                }
                let y = my * 16 + edge * 4;
                let bs = edge_strength(pic, mx, my, edge, false);
                if y >= 4 && !(transform8 && (edge == 1 || edge == 3)) {
                    let qpav = edge_qp(pic, mx, my, true, false, edge == 0);
                    filter_groups(
                        &mut pic.y,
                        pic.w,
                        (mx * 16, y),
                        (false, false, 4),
                        bs,
                        qpav,
                        off,
                    );
                }
                if edge % 2 == 0 {
                    let cy = my * 8 + edge * 2;
                    if cy >= 2 {
                        let qpc = edge_qp(pic, mx, my, false, false, edge == 0);
                        filter_groups(
                            &mut pic.u,
                            pic.w / 2,
                            (mx * 8, cy),
                            (false, true, 2),
                            bs,
                            qpc,
                            off,
                        );
                        filter_groups(
                            &mut pic.v,
                            pic.w / 2,
                            (mx * 8, cy),
                            (false, true, 2),
                            bs,
                            qpc,
                            off,
                        );
                    }
                }
            }
        }
    }
}

fn edge_strength(pic: &Frame, mx: usize, my: usize, edge: usize, vertical: bool) -> [u8; 4] {
    let Some(current) = pic.mb.get(my * pic.mb_w + mx) else {
        return [0; 4];
    };
    let neighbor = if edge == 0 {
        let (nx, ny) = if vertical {
            (mx as i32 - 1, my as i32)
        } else {
            (mx as i32, my as i32 - 1)
        };
        coded_mb(pic, nx, ny)
    } else {
        None
    };
    // Intra covers the whole macroblock, so every sample of the edge agrees.
    if edge == 0 {
        if current.intra || neighbor.is_some_and(|mb| mb.intra) {
            return [4; 4];
        }
    } else if current.intra {
        return [3; 4];
    }
    let mut out = [0u8; 4];
    for (line, sample) in out.iter_mut().enumerate() {
        *sample = inter_strength(current, neighbor, edge, line, vertical);
    }
    out
}

fn inter_strength(q: &Mb, neighbor: Option<&Mb>, edge: usize, line: usize, vertical: bool) -> u8 {
    let (qx, qy) = if vertical { (edge, line) } else { (line, edge) };
    let q_nnz = q.nnz_y[qy * 4 + qx];
    let q_pred = preds_at(q, qx, qy);
    let (p_nnz, p_pred) = if edge == 0 {
        let (ox, oy) = if vertical { (3, line) } else { (line, 3) };
        match neighbor {
            Some(p) => {
                let nnz = if p.pcm { 16 } else { p.nnz_y[oy * 4 + ox] };
                (nnz, preds_at(p, ox, oy))
            }
            None => (0, [(-1, [0, 0]), (-1, [0, 0])]),
        }
    } else {
        let (ox, oy) = if vertical {
            (edge - 1, line)
        } else {
            (line, edge - 1)
        };
        (q.nnz_y[oy * 4 + ox], preds_at(q, ox, oy))
    };
    if p_nnz > 0 || q_nnz > 0 {
        return 2;
    }
    motion_strength(p_pred, q_pred)
}

fn filter_groups(
    plane: &mut [u8],
    stride: usize,
    at: (usize, usize),
    kind: (bool, bool, usize),
    bs: [u8; 4],
    qpav: i32,
    off: (i32, i32),
) {
    let (x, y) = at;
    let (vertical, chroma, step) = kind;
    let (offset_a, offset_b) = off;
    if bs[0] != 0 && bs[1] == bs[0] && bs[2] == bs[0] && bs[3] == bs[0] {
        recon::filter_edge(
            plane,
            stride,
            x,
            y,
            recon::Edge {
                vertical,
                bs: bs[0],
                qpav,
                offset_a,
                offset_b,
                chroma,
                len: step * 4,
            },
        );
        return;
    }
    for (group, group_bs) in bs.into_iter().enumerate() {
        if group_bs == 0 {
            continue;
        }
        let shift = group * step;
        let (gx, gy) = if vertical {
            (x, y + shift)
        } else {
            (x + shift, y)
        };
        recon::filter_edge(
            plane,
            stride,
            gx,
            gy,
            recon::Edge {
                vertical,
                bs: group_bs,
                qpav,
                offset_a,
                offset_b,
                chroma,
                len: step,
            },
        );
    }
}

fn offsets(pic: &Frame, index: usize) -> (i32, i32) {
    pic.mb
        .get(index)
        .map(|mb| (i32::from(mb.off_a), i32::from(mb.off_b)))
        .unwrap_or((0, 0))
}

fn edge_qp(pic: &Frame, mx: usize, my: usize, luma: bool, vertical: bool, boundary: bool) -> i32 {
    let index = my * pic.mb_w + mx;
    let current = pic
        .mb
        .get(index)
        .map(|mb| {
            if luma {
                i32::from(mb.qp)
            } else {
                i32::from(mb.qp_c)
            }
        })
        .unwrap_or(26);
    if !boundary {
        return current;
    }
    let (nx, ny) = if vertical {
        (mx as i32 - 1, my as i32)
    } else {
        (mx as i32, my as i32 - 1)
    };
    let Some(other) = coded_mb(pic, nx, ny) else {
        return current;
    };
    let side = if luma {
        i32::from(other.qp)
    } else {
        i32::from(other.qp_c)
    };
    (current + side + 1) >> 1
}

/// Clause 8.7.2.1. Compares the pictures actually referenced, on either list.
fn motion_strength(p: [(i8, [i16; 2]); 2], q: [(i8, [i16; 2]); 2]) -> u8 {
    let p_used = u8::from(p[0].0 >= 0) + u8::from(p[1].0 >= 0);
    let q_used = u8::from(q[0].0 >= 0) + u8::from(q[1].0 >= 0);
    if p_used != q_used {
        return 1;
    }
    if p_used == 0 {
        return 0;
    }
    if p_used == 1 {
        let p = if p[0].0 >= 0 { p[0] } else { p[1] };
        let q = if q[0].0 >= 0 { q[0] } else { q[1] };
        return u8::from(p.0 != q.0 || mv_apart(p.1, q.1));
    }
    if !same_pictures(p[0].0, p[1].0, q[0].0, q[1].0) {
        return 1;
    }
    if p[0].0 != p[1].0 {
        let q_mv = |slot: i8| {
            if q[0].0 == slot { q[0].1 } else { q[1].1 }
        };
        return u8::from(mv_apart(p[0].1, q_mv(p[0].0)) || mv_apart(p[1].1, q_mv(p[1].0)));
    }
    let same_list = mv_apart(p[0].1, q[0].1) || mv_apart(p[1].1, q[1].1);
    let crossed = mv_apart(p[0].1, q[1].1) || mv_apart(p[1].1, q[0].1);
    u8::from(same_list && crossed)
}

fn same_pictures(a: i8, b: i8, c: i8, d: i8) -> bool {
    (a == c && b == d) || (a == d && b == c)
}

fn mv_apart(a: [i16; 2], b: [i16; 2]) -> bool {
    (a[0] - b[0]).abs() >= 4 || (a[1] - b[1]).abs() >= 4
}

fn preds_at(mb: &Mb, bx: usize, by: usize) -> [(i8, [i16; 2]); 2] {
    let index = by * 4 + bx;
    let part = (by / 2) * 2 + (bx / 2);
    [
        (mb.slot0[part], mb.mv0[index]),
        (mb.slot1[part], mb.mv1[index]),
    ]
}

fn bind_refs(mb: &mut Mb, slice: &Slice<'_>) {
    for i in 0..4 {
        mb.slot0[i] = ref_slot(&slice.ref0, mb.ref0[i]);
        mb.slot1[i] = ref_slot(&slice.ref1, mb.ref1[i]);
        mb.ref_poc0[i] = listed_poc(&slice.poc0, mb.ref0[i]);
        mb.ref_poc1[i] = listed_poc(&slice.poc1, mb.ref1[i]);
    }
}

fn listed_poc(pocs: &[i32], index: i8) -> i32 {
    let Ok(index) = usize::try_from(index) else {
        return 0;
    };
    pocs.get(index).copied().unwrap_or(0)
}

fn ref_slot(list: &[usize], index: i8) -> i8 {
    let Ok(index) = usize::try_from(index) else {
        return -1;
    };
    let Some(slot) = list.get(index) else {
        return -1;
    };
    i8::try_from(*slot).unwrap_or(-1)
}

/// A macroblock edge is filtered across slices unless this slice asked not to.
fn filter_boundary(pic: &Frame, mx: usize, my: usize, vertical: bool) -> bool {
    let Some(mb) = coded_mb(pic, mx as i32, my as i32) else {
        return false;
    };
    if mb.deblock != 2 {
        return true;
    }
    let (nx, ny) = if vertical {
        (mx as i32 - 1, my as i32)
    } else {
        (mx as i32, my as i32 - 1)
    };
    coded_mb(pic, nx, ny).is_some_and(|other| other.slice == mb.slice)
}

fn coded_mb(pic: &Frame, mx: i32, my: i32) -> Option<&Mb> {
    if mx < 0 || my < 0 || mx as usize >= pic.mb_w || my as usize >= pic.h / 16 {
        return None;
    }
    let mb = pic.mb.get(my as usize * pic.mb_w + mx as usize)?;
    if mb.slice == u16::MAX { None } else { Some(mb) }
}
