//! Marker walk and scan decode.
//!
//! Baseline and extended sequential frames inverse-transform each block as it
//! is read, so the sample planes are the only large buffer. A progressive
//! frame keeps coefficients until the last scan, and those coefficients have
//! their own 256 MiB cap. `docs/codecs.md` is the picture that comes out.

use cutlass_core::Still;

use crate::bits::{Bits, extend};
use crate::color::ycbcr;
use crate::error::Error;
use crate::huff::Huff;
use crate::idct::{dequant, idct, level};

/// Each edge, and each decoded buffer, stops here. `docs/codecs.md`.
const MAX_EDGE: u32 = 32_768;
const MAX_BYTES: usize = 256 * 1024 * 1024;

/// Decodes one JPEG still.
///
/// The bytes are the file. Baseline, extended sequential, and progressive
/// frames are read at 8 bits. A 12-bit frame, a lossless frame, and arithmetic
/// coding are refused. EXIF is not applied.
pub fn decode(bytes: &[u8]) -> Result<Still, Error> {
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return Err(Error::Signature);
    }
    let mut pos = 2;
    let mut tables = Tables::default();
    let mut frame: Option<Frame> = None;
    let mut scans = 0u32;
    loop {
        if pos >= bytes.len() {
            return Err(truncated(pos, 2));
        }
        let (code, next) = marker_at(bytes, pos)?;
        pos = next;
        match code {
            0xD8 => return Err(Error::Inconsistent("a second start marker")),
            0xD9 => {
                if pos != bytes.len() {
                    return Err(Error::Inconsistent("a byte follows the end marker"));
                }
                break;
            }
            0xDA => {
                let frame = frame
                    .as_mut()
                    .ok_or(Error::Inconsistent("a scan comes before the frame"))?;
                if !frame.progressive && scans > 0 {
                    return Err(Error::Inconsistent("a sequential frame has a second scan"));
                }
                let (scan, entropy_at) = read_sos(bytes, pos, frame)?;
                let end = entropy_end(bytes, entropy_at)?;
                decode_scan(frame, &tables, &scan, &bytes[entropy_at..end], entropy_at)?;
                scans = scans.checked_add(1).ok_or(Error::TooLarge)?;
                pos = end;
            }
            0xDB => pos = read_dqt(bytes, pos, &mut tables)?,
            0xC4 => pos = read_dht(bytes, pos, &mut tables)?,
            0xDD => pos = read_dri(bytes, pos, &mut tables)?,
            0xC0..=0xC2 => {
                if frame.is_some() {
                    return Err(Error::Inconsistent("a second frame"));
                }
                let progressive = code == 0xC2;
                let (parsed, next) = read_sof(bytes, pos, progressive)?;
                pos = next;
                frame = Some(parsed);
            }
            0xFE => pos = skip_segment(bytes, pos)?,
            code if (0xE0..=0xEF).contains(&code) => {
                pos = read_app(bytes, pos, code, &mut tables)?;
            }
            0xC3 | 0xC7 | 0xCB | 0xCF => {
                return Err(Error::Unsupported("a lossless frame is not supported"));
            }
            0xC5 | 0xC6 => {
                return Err(Error::Unsupported("a differential frame is not supported"));
            }
            0xC9 | 0xCA | 0xCC | 0xCD | 0xCE => {
                return Err(Error::Unsupported("arithmetic coding is not supported"));
            }
            0xC8 => {
                return Err(Error::Unsupported(
                    "a JPEG extension frame is not supported",
                ));
            }
            0xDC => return Err(Error::Unsupported("the height is defined later")),
            0xD0..=0xD7 => {
                return Err(Error::Inconsistent("a restart marker is outside a scan"));
            }
            _ => return Err(Error::Unsupported("a marker is not supported")),
        }
    }
    let mut frame = frame.ok_or(Error::Inconsistent("the file has no frame"))?;
    if scans == 0 {
        return Err(Error::Inconsistent("the file has no scan"));
    }
    if frame.progressive {
        transform(&mut frame, &tables)?;
    }
    paint(&frame, tables.adobe)
}

#[derive(Default)]
struct Tables {
    quant: [Option<[u16; 64]>; 4],
    dc: [Option<Huff>; 4],
    ac: [Option<Huff>; 4],
    restart: u16,
    adobe: Option<u8>,
}

impl Tables {
    fn quant(&self, id: u8) -> Result<&[u16; 64], Error> {
        self.slot(self.quant.get(usize::from(id)), "a quant table is missing")
    }

    fn dc(&self, id: u8) -> Result<&Huff, Error> {
        self.slot(
            self.dc.get(usize::from(id)),
            "a DC Huffman table is missing",
        )
    }

    fn ac(&self, id: u8) -> Result<&Huff, Error> {
        self.slot(
            self.ac.get(usize::from(id)),
            "an AC Huffman table is missing",
        )
    }

    fn slot<'a, T>(
        &'a self,
        slot: Option<&'a Option<T>>,
        message: &'static str,
    ) -> Result<&'a T, Error> {
        slot.ok_or(Error::Inconsistent(message))?
            .as_ref()
            .ok_or(Error::Inconsistent(message))
    }
}

struct Frame {
    progressive: bool,
    width: u32,
    height: u32,
    h_max: u8,
    v_max: u8,
    mcus_x: usize,
    mcus_y: usize,
    comps: Vec<Comp>,
}

struct Comp {
    id: u8,
    h: u8,
    v: u8,
    tq: u8,
    stride: usize,
    blocks_x: usize,
    blocks_y: usize,
    samples: Vec<u8>,
    coefs: Vec<i32>,
    dc_shift: Option<u8>,
    ac_shift: [Option<u8>; 64],
}

struct Spec {
    id: u8,
    h: u8,
    v: u8,
    tq: u8,
}

struct Scan {
    comps: Vec<ScanComp>,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
}

#[derive(Clone, Copy)]
struct ScanComp {
    index: usize,
    dc: u8,
    ac: u8,
}

#[derive(Clone, Copy)]
struct Band {
    ss: u8,
    se: u8,
    al: u8,
}

struct Cursor<'a> {
    bits: Bits<'a>,
    preds: [i32; 4],
    eobrun: u32,
    index: u32,
    rst: u8,
    interval: u16,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8], base: usize, interval: u16) -> Self {
        Self {
            bits: Bits::new(data, base),
            preds: [0; 4],
            eobrun: 0,
            index: 0,
            rst: 0,
            interval,
        }
    }

    /// T.81 B.2.1 and F.2.1. Restart interval counts data units, then the
    /// predictors and the end-of-band run go back to zero.
    fn restart_if_due(&mut self) -> Result<(), Error> {
        if self.interval != 0
            && self.index > 0
            && self.index.is_multiple_of(u32::from(self.interval))
        {
            self.bits.discard();
            let code = self.bits.marker()?;
            if code != 0xD0 + self.rst {
                return Err(Error::Inconsistent("a restart marker is out of order"));
            }
            self.rst = (self.rst + 1) % 8;
            self.preds = [0; 4];
            self.eobrun = 0;
        }
        self.index = self.index.checked_add(1).ok_or(Error::TooLarge)?;
        Ok(())
    }
}

fn read_sof(bytes: &[u8], pos: usize, progressive: bool) -> Result<(Frame, usize), Error> {
    let (payload, end) = segment(bytes, pos)?;
    if payload.len() < 6 {
        return Err(Error::Inconsistent("a frame segment is short"));
    }
    let precision = payload[0];
    if precision != 8 {
        return Err(Error::Unsupported("the sample precision is not 8 bits"));
    }
    let height = u32::from(u16::from_be_bytes([payload[1], payload[2]]));
    let width = u32::from(u16::from_be_bytes([payload[3], payload[4]]));
    let count = payload[5];
    if height == 0 {
        return Err(Error::Unsupported("the height is defined later"));
    }
    if width == 0 {
        return Err(Error::Inconsistent("the width is zero"));
    }
    if count != 1 && count != 3 {
        return Err(Error::Unsupported("a frame component count is not 1 or 3"));
    }
    let need = 6 + usize::from(count) * 3;
    if payload.len() != need {
        return Err(Error::Inconsistent("a frame segment has the wrong length"));
    }
    if width > MAX_EDGE || height > MAX_EDGE {
        return Err(Error::TooLarge);
    }
    let _rgba = rgba_len(width, height)?;

    let mut specs = Vec::with_capacity(usize::from(count));
    let mut h_max = 0u8;
    let mut v_max = 0u8;
    let mut sum = 0u32;
    for index in 0..usize::from(count) {
        let at = 6 + index * 3;
        let id = payload[at];
        let hv = payload[at + 1];
        let h = hv >> 4;
        let v = hv & 0x0F;
        let tq = payload[at + 2];
        if h == 0 || v == 0 {
            return Err(Error::Inconsistent("a sampling factor is zero"));
        }
        if h > 4 || v > 4 {
            return Err(Error::Inconsistent("a sampling factor is past 4"));
        }
        if tq > 3 {
            return Err(Error::Inconsistent("a quant table id is past 3"));
        }
        if specs.iter().any(|spec: &Spec| spec.id == id) {
            return Err(Error::Inconsistent("a component id is repeated"));
        }
        sum = sum.saturating_add(u32::from(h) * u32::from(v));
        if sum > 10 {
            return Err(Error::Inconsistent(
                "the sampling factors are past the baseline limit",
            ));
        }
        h_max = h_max.max(h);
        v_max = v_max.max(v);
        specs.push(Spec { id, h, v, tq });
    }

    let mcu_w = 8 * u32::from(h_max);
    let mcu_h = 8 * u32::from(v_max);
    let mcus_x = as_usize(width.div_ceil(mcu_w))?;
    let mcus_y = as_usize(height.div_ceil(mcu_h))?;
    let mut plane_bytes = 0usize;
    let mut coef_bytes = 0usize;
    let mut layouts = Vec::with_capacity(specs.len());
    for spec in &specs {
        let blocks_x = mcus_x
            .checked_mul(usize::from(spec.h))
            .ok_or(Error::TooLarge)?;
        let blocks_y = mcus_y
            .checked_mul(usize::from(spec.v))
            .ok_or(Error::TooLarge)?;
        let stride = blocks_x.checked_mul(8).ok_or(Error::TooLarge)?;
        let rows = blocks_y.checked_mul(8).ok_or(Error::TooLarge)?;
        let plane = stride.checked_mul(rows).ok_or(Error::TooLarge)?;
        plane_bytes = plane_bytes.checked_add(plane).ok_or(Error::TooLarge)?;
        if plane_bytes > MAX_BYTES {
            return Err(Error::TooLarge);
        }
        let coefs = if progressive {
            let count = blocks_x
                .checked_mul(blocks_y)
                .and_then(|blocks| blocks.checked_mul(64))
                .ok_or(Error::TooLarge)?;
            let bytes = count.checked_mul(4).ok_or(Error::TooLarge)?;
            coef_bytes = coef_bytes.checked_add(bytes).ok_or(Error::TooLarge)?;
            if coef_bytes > MAX_BYTES {
                return Err(Error::TooLarge);
            }
            count
        } else {
            0
        };
        layouts.push((blocks_x, blocks_y, stride, plane, coefs));
    }

    let mut comps = Vec::with_capacity(specs.len());
    for (spec, (blocks_x, blocks_y, stride, plane, coefs)) in specs.into_iter().zip(layouts) {
        comps.push(Comp {
            id: spec.id,
            h: spec.h,
            v: spec.v,
            tq: spec.tq,
            stride,
            blocks_x,
            blocks_y,
            samples: vec![0; plane],
            coefs: vec![0; coefs],
            dc_shift: None,
            ac_shift: [None; 64],
        });
    }
    Ok((
        Frame {
            progressive,
            width,
            height,
            h_max,
            v_max,
            mcus_x,
            mcus_y,
            comps,
        },
        end,
    ))
}

fn read_sos(bytes: &[u8], pos: usize, frame: &Frame) -> Result<(Scan, usize), Error> {
    let (payload, end) = segment(bytes, pos)?;
    if payload.is_empty() {
        return Err(Error::Inconsistent("a scan segment is empty"));
    }
    let count = payload[0];
    if count == 0 || count > 4 {
        return Err(Error::Inconsistent(
            "a scan component count is not 1 through 4",
        ));
    }
    let need = 1 + usize::from(count) * 2 + 3;
    if payload.len() != need {
        return Err(Error::Inconsistent("a scan segment has the wrong length"));
    }
    let mut comps = Vec::with_capacity(usize::from(count));
    for index in 0..usize::from(count) {
        let id = payload[1 + index * 2];
        let selector = payload[2 + index * 2];
        let dc = selector >> 4;
        let ac = selector & 0x0F;
        if dc > 3 || ac > 3 {
            return Err(Error::Inconsistent("a Huffman table id is past 3"));
        }
        let Some(component) = frame.comps.iter().position(|comp| comp.id == id) else {
            return Err(Error::Inconsistent("a scan names an unknown component"));
        };
        if comps.iter().any(|scan: &ScanComp| scan.index == component) {
            return Err(Error::Inconsistent("a scan repeats a component"));
        }
        comps.push(ScanComp {
            index: component,
            dc,
            ac,
        });
    }
    let at = 1 + usize::from(count) * 2;
    let ss = payload[at];
    let se = payload[at + 1];
    let successive = payload[at + 2];
    Ok((
        Scan {
            comps,
            ss,
            se,
            ah: successive >> 4,
            al: successive & 0x0F,
        },
        end,
    ))
}

fn read_dqt(bytes: &[u8], pos: usize, tables: &mut Tables) -> Result<usize, Error> {
    let (payload, end) = segment(bytes, pos)?;
    if payload.is_empty() {
        return Err(Error::Inconsistent("a quant segment is empty"));
    }
    let mut index = 0;
    while index < payload.len() {
        let packed = payload[index];
        index += 1;
        let precision = packed >> 4;
        let id = packed & 0x0F;
        if id > 3 {
            return Err(Error::Inconsistent("a quant table id is past 3"));
        }
        if precision != 0 {
            return Err(Error::Unsupported("a quant table is not 8 bits"));
        }
        let Some(bytes) = payload.get(index..index + 64) else {
            return Err(Error::Inconsistent("a quant table is short"));
        };
        let mut table = [0u16; 64];
        for (slot, byte) in table.iter_mut().zip(bytes) {
            *slot = u16::from(*byte);
        }
        index += 64;
        tables.quant[usize::from(id)] = Some(table);
    }
    Ok(end)
}

fn read_dht(bytes: &[u8], pos: usize, tables: &mut Tables) -> Result<usize, Error> {
    let (payload, end) = segment(bytes, pos)?;
    if payload.is_empty() {
        return Err(Error::Inconsistent("a Huffman segment is empty"));
    }
    let mut index = 0;
    while index < payload.len() {
        let packed = payload[index];
        index += 1;
        let class = packed >> 4;
        let id = packed & 0x0F;
        if class > 1 || id > 3 {
            return Err(Error::Inconsistent("a Huffman table id is past 3"));
        }
        let Some(counts_bytes) = payload.get(index..index + 16) else {
            return Err(Error::Inconsistent("a Huffman table is short"));
        };
        let mut counts = [0u8; 16];
        counts.copy_from_slice(counts_bytes);
        index += 16;
        let total: usize = counts.iter().copied().map(usize::from).sum();
        if total > 256 {
            return Err(Error::Inconsistent("a Huffman table has too many symbols"));
        }
        let Some(symbols) = payload.get(index..index + total) else {
            return Err(Error::Inconsistent(
                "a Huffman table length does not match its counts",
            ));
        };
        index += total;
        let table = Huff::build(&counts, symbols)?;
        if class == 0 {
            tables.dc[usize::from(id)] = Some(table);
        } else {
            tables.ac[usize::from(id)] = Some(table);
        }
    }
    Ok(end)
}

fn read_dri(bytes: &[u8], pos: usize, tables: &mut Tables) -> Result<usize, Error> {
    let (payload, end) = segment(bytes, pos)?;
    if payload.len() != 2 {
        return Err(Error::Inconsistent(
            "a restart segment has the wrong length",
        ));
    }
    tables.restart = u16::from_be_bytes([payload[0], payload[1]]);
    Ok(end)
}

fn read_app(bytes: &[u8], pos: usize, code: u8, tables: &mut Tables) -> Result<usize, Error> {
    let (payload, end) = segment(bytes, pos)?;
    // T.81 B.2.4. Adobe APP14 carries a color transform in its 13th byte.
    if code == 0xEE && payload.len() >= 13 && payload.starts_with(b"Adobe\0") {
        tables.adobe = Some(payload[12]);
    }
    Ok(end)
}

fn skip_segment(bytes: &[u8], pos: usize) -> Result<usize, Error> {
    Ok(segment(bytes, pos)?.1)
}

fn decode_scan(
    frame: &mut Frame,
    tables: &Tables,
    scan: &Scan,
    entropy: &[u8],
    base: usize,
) -> Result<(), Error> {
    if frame.progressive {
        if scan.ss > 63 || scan.se > 63 || scan.ss > scan.se {
            return Err(Error::Inconsistent(
                "a spectral selection is past the block",
            ));
        }
        if scan.ah > 13 || scan.al > 13 {
            return Err(Error::Inconsistent("a successive approximation is past 13"));
        }
        if scan.ah != 0 {
            let step = scan
                .al
                .checked_add(1)
                .ok_or(Error::Inconsistent("a refinement does not step one bit"))?;
            if scan.ah != step {
                return Err(Error::Inconsistent("a refinement does not step one bit"));
            }
        }
        if scan.ss == 0 && scan.se != 0 {
            return Err(Error::Inconsistent("a progressive scan mixes DC and AC"));
        }
        if scan.se > 0 && scan.comps.len() != 1 {
            return Err(Error::Inconsistent(
                "an AC scan has more than one component",
            ));
        }
    } else {
        if scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0 {
            return Err(Error::Inconsistent(
                "a sequential scan is not the whole block",
            ));
        }
        if scan.comps.len() != frame.comps.len() {
            return Err(Error::Inconsistent("a sequential scan skips a component"));
        }
    }

    let mut cur = Cursor::new(entropy, base, tables.restart);
    if !frame.progressive {
        sequential(frame, scan, &mut cur, tables)?;
    } else if scan.se == 0 && scan.ah == 0 {
        dc_first(frame, scan, &mut cur, tables)?;
    } else if scan.se == 0 {
        dc_refine(frame, scan, &mut cur, tables)?;
    } else if scan.ah == 0 {
        ac_first(frame, scan, &mut cur, tables)?;
    } else {
        ac_refine(frame, scan, &mut cur, tables)?;
    }
    Ok(())
}

fn sequential(
    frame: &mut Frame,
    scan: &Scan,
    cur: &mut Cursor<'_>,
    tables: &Tables,
) -> Result<(), Error> {
    for sc in &scan.comps {
        let tq = frame.comps[sc.index].tq;
        let _ = tables.quant(tq)?;
        let _ = tables.dc(sc.dc)?;
        let _ = tables.ac(sc.ac)?;
    }
    each_data_unit(frame, scan, cur, |frame, cur, sc, bx, by| {
        let comp = frame
            .comps
            .get_mut(sc.index)
            .ok_or(Error::Inconsistent("a scan names an unknown component"))?;
        sequential_block(comp, cur, sc, tables, (bx, by))
    })
}

fn sequential_block(
    comp: &mut Comp,
    cur: &mut Cursor<'_>,
    sc: ScanComp,
    tables: &Tables,
    at: (usize, usize),
) -> Result<(), Error> {
    let quant = *tables.quant(comp.tq)?;
    let dc = tables.dc(sc.dc)?;
    let pred = *cur
        .preds
        .get(sc.index)
        .ok_or(Error::Inconsistent("a component index is past the frame"))?;
    let next = read_dc(&mut cur.bits, dc, pred)?;
    *cur.preds
        .get_mut(sc.index)
        .ok_or(Error::Inconsistent("a component index is past the frame"))? = next;
    let mut zz = [0i32; 64];
    zz[0] = next;
    let ac = tables.ac(sc.ac)?;
    read_ac(&mut cur.bits, ac, &mut zz)?;
    let natural = dequant(&zz, &quant)?;
    let spatial = idct(&natural)?;
    write_block(comp, at, &spatial)
}

fn dc_first(
    frame: &mut Frame,
    scan: &Scan,
    cur: &mut Cursor<'_>,
    tables: &Tables,
) -> Result<(), Error> {
    for sc in &scan.comps {
        check_dc(&frame.comps[sc.index], scan.ah)?;
        let _ = tables.dc(sc.dc)?;
    }
    let al = scan.al;
    each_data_unit(frame, scan, cur, |frame, cur, sc, bx, by| {
        let comp = frame
            .comps
            .get_mut(sc.index)
            .ok_or(Error::Inconsistent("a scan names an unknown component"))?;
        let dc = tables.dc(sc.dc)?;
        let pred = *cur
            .preds
            .get(sc.index)
            .ok_or(Error::Inconsistent("a component index is past the frame"))?;
        let next = read_dc(&mut cur.bits, dc, pred)?;
        *cur.preds
            .get_mut(sc.index)
            .ok_or(Error::Inconsistent("a component index is past the frame"))? = next;
        let shifted = next
            .checked_shl(u32::from(al))
            .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
        *coef_mut(comp, (bx, by), 0)? = shifted;
        Ok(())
    })?;
    for sc in &scan.comps {
        frame.comps[sc.index].dc_shift = Some(al);
    }
    Ok(())
}

fn dc_refine(
    frame: &mut Frame,
    scan: &Scan,
    cur: &mut Cursor<'_>,
    _tables: &Tables,
) -> Result<(), Error> {
    for sc in &scan.comps {
        check_dc(&frame.comps[sc.index], scan.ah)?;
    }
    let al = scan.al;
    each_data_unit(frame, scan, cur, |frame, cur, sc, bx, by| {
        let comp = frame
            .comps
            .get_mut(sc.index)
            .ok_or(Error::Inconsistent("a scan names an unknown component"))?;
        let bit = cur.bits.bit()?;
        if bit == 1 {
            add_magnitude(coef_mut(comp, (bx, by), 0)?, al)?;
        }
        Ok(())
    })?;
    for sc in &scan.comps {
        frame.comps[sc.index].dc_shift = Some(al);
    }
    Ok(())
}

fn ac_first(
    frame: &mut Frame,
    scan: &Scan,
    cur: &mut Cursor<'_>,
    tables: &Tables,
) -> Result<(), Error> {
    let sc = scan.comps[0];
    check_ac(&frame.comps[sc.index], scan)?;
    let ac = tables.ac(sc.ac)?;
    let band = Band {
        ss: scan.ss,
        se: scan.se,
        al: scan.al,
    };
    each_data_unit(frame, scan, cur, |frame, cur, sc, bx, by| {
        let comp = frame
            .comps
            .get_mut(sc.index)
            .ok_or(Error::Inconsistent("a scan names an unknown component"))?;
        ac_first_block(comp, &mut cur.bits, ac, band, (bx, by), &mut cur.eobrun)
    })?;
    if cur.eobrun != 0 {
        return Err(Error::Inconsistent(
            "a progressive end-of-band run passes the last block",
        ));
    }
    remember_ac(&mut frame.comps[sc.index], band);
    Ok(())
}

fn ac_refine(
    frame: &mut Frame,
    scan: &Scan,
    cur: &mut Cursor<'_>,
    tables: &Tables,
) -> Result<(), Error> {
    let sc = scan.comps[0];
    check_ac(&frame.comps[sc.index], scan)?;
    let ac = tables.ac(sc.ac)?;
    let band = Band {
        ss: scan.ss,
        se: scan.se,
        al: scan.al,
    };
    each_data_unit(frame, scan, cur, |frame, cur, sc, bx, by| {
        let comp = frame
            .comps
            .get_mut(sc.index)
            .ok_or(Error::Inconsistent("a scan names an unknown component"))?;
        ac_refine_block(comp, &mut cur.bits, ac, band, (bx, by), &mut cur.eobrun)
    })?;
    if cur.eobrun != 0 {
        return Err(Error::Inconsistent(
            "a progressive end-of-band run passes the last block",
        ));
    }
    remember_ac(&mut frame.comps[sc.index], band);
    Ok(())
}

fn each_data_unit(
    frame: &mut Frame,
    scan: &Scan,
    cur: &mut Cursor<'_>,
    mut block: impl FnMut(&mut Frame, &mut Cursor<'_>, ScanComp, usize, usize) -> Result<(), Error>,
) -> Result<(), Error> {
    if scan.comps.is_empty() {
        return Err(Error::Inconsistent("a scan has no components"));
    }
    // One component is its own block raster. More than one is an interleaved MCU.
    if scan.comps.len() == 1 {
        let sc = scan.comps[0];
        let comp = frame
            .comps
            .get(sc.index)
            .ok_or(Error::Inconsistent("a scan names an unknown component"))?;
        let blocks_x = comp.blocks_x;
        let blocks_y = comp.blocks_y;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                cur.restart_if_due()?;
                block(frame, cur, sc, bx, by)?;
            }
        }
        return Ok(());
    }
    let mcus_x = frame.mcus_x;
    let mcus_y = frame.mcus_y;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            cur.restart_if_due()?;
            for slot in 0..scan.comps.len() {
                let sc = scan.comps[slot];
                let comp = frame
                    .comps
                    .get(sc.index)
                    .ok_or(Error::Inconsistent("a scan names an unknown component"))?;
                let h = usize::from(comp.h);
                let v = usize::from(comp.v);
                for vy in 0..v {
                    for hx in 0..h {
                        block(frame, cur, sc, mx * h + hx, my * v + vy)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn read_dc(bits: &mut Bits<'_>, dc: &Huff, pred: i32) -> Result<i32, Error> {
    let category = dc.decode(bits)?;
    if category > 11 {
        return Err(Error::Inconsistent("a DC category is past 11"));
    }
    let extra = if category == 0 {
        0
    } else {
        bits.bits(category)?
    };
    let diff = extend(extra, category)?;
    pred.checked_add(diff)
        .ok_or(Error::Inconsistent("a DC coefficient does not fit"))
}

fn read_ac(bits: &mut Bits<'_>, ac: &Huff, zz: &mut [i32; 64]) -> Result<(), Error> {
    let mut k = 1usize;
    while k < 64 {
        let symbol = ac.decode(bits)?;
        let run = usize::from(symbol >> 4);
        let size = symbol & 0x0F;
        if size == 0 {
            if run == 15 {
                k = k
                    .checked_add(16)
                    .ok_or(Error::Inconsistent("a zero run passes the end of a block"))?;
                if k > 64 {
                    return Err(Error::Inconsistent("a zero run passes the end of a block"));
                }
                continue;
            }
            if run == 0 {
                break;
            }
            return Err(Error::Inconsistent("an AC symbol is not a sequential code"));
        }
        if size > 10 {
            return Err(Error::Inconsistent("an AC category is past 10"));
        }
        k = k.checked_add(run).ok_or(Error::Inconsistent(
            "a coefficient is past the end of a block",
        ))?;
        if k >= 64 {
            return Err(Error::Inconsistent(
                "a coefficient is past the end of a block",
            ));
        }
        zz[k] = extend(bits.bits(size)?, size)?;
        k += 1;
    }
    Ok(())
}

fn ac_first_block(
    comp: &mut Comp,
    bits: &mut Bits<'_>,
    ac: &Huff,
    band: Band,
    at: (usize, usize),
    eobrun: &mut u32,
) -> Result<(), Error> {
    // T.81 G.1.2.2. An end-of-band run covers this block and the ones after it.
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = usize::from(band.ss);
    let se = usize::from(band.se);
    while k <= se {
        let symbol = ac.decode(bits)?;
        let run = symbol >> 4;
        let size = symbol & 0x0F;
        if size == 0 {
            if run == 15 {
                k = k
                    .checked_add(16)
                    .ok_or(Error::Inconsistent("a zero run passes the end of a block"))?;
                if k > se + 1 {
                    return Err(Error::Inconsistent("a zero run passes the end of a block"));
                }
                continue;
            }
            *eobrun = eob_run(bits, run)?;
            break;
        }
        if size > 10 {
            return Err(Error::Inconsistent("an AC category is past 10"));
        }
        k = k.checked_add(usize::from(run)).ok_or(Error::Inconsistent(
            "a coefficient is past the end of a block",
        ))?;
        if k > se {
            return Err(Error::Inconsistent(
                "a coefficient is past the end of a block",
            ));
        }
        let magnitude = extend(bits.bits(size)?, size)?;
        let shifted = magnitude
            .checked_shl(u32::from(band.al))
            .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
        *coef_mut(comp, at, k)? = shifted;
        k += 1;
    }
    if *eobrun > 0 {
        *eobrun -= 1;
    }
    Ok(())
}

fn ac_refine_block(
    comp: &mut Comp,
    bits: &mut Bits<'_>,
    ac: &Huff,
    band: Band,
    at: (usize, usize),
    eobrun: &mut u32,
) -> Result<(), Error> {
    // T.81 G.1.2.3. The sign bit follows the Huffman symbol. The run then
    // refines nonzero coefficients and counts only zeros.
    let mut k = usize::from(band.ss);
    let se = usize::from(band.se);
    if *eobrun == 0 {
        while k <= se {
            let symbol = ac.decode(bits)?;
            let run = symbol >> 4;
            let size = symbol & 0x0F;
            if size == 0 {
                if run == 15 {
                    k = skip_zeros(comp, bits, band.al, at, k, se, 16)?;
                    continue;
                }
                *eobrun = eob_run(bits, run)?;
                break;
            }
            if size != 1 {
                return Err(Error::Inconsistent(
                    "a refined coefficient is wider than one bit",
                ));
            }
            let sign = bits.bit()?;
            let magnitude = 1i32
                .checked_shl(u32::from(band.al))
                .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
            let value = if sign == 1 { magnitude } else { -magnitude };
            k = skip_zeros(comp, bits, band.al, at, k, se, run)?;
            if k > se {
                return Err(Error::Inconsistent(
                    "a coefficient is past the end of a block",
                ));
            }
            let slot = coef_mut(comp, at, k)?;
            if *slot != 0 {
                return Err(Error::Inconsistent("a new coefficient lands on a nonzero"));
            }
            *slot = value;
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= se {
            if coef_get(comp, at, k)? != 0 {
                refine_magnitude(bits, coef_mut(comp, at, k)?, band.al)?;
            }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}

fn skip_zeros(
    comp: &mut Comp,
    bits: &mut Bits<'_>,
    al: u8,
    at: (usize, usize),
    mut k: usize,
    se: usize,
    mut left: u8,
) -> Result<usize, Error> {
    while left > 0 {
        if k > se {
            return Err(Error::Inconsistent("a zero run passes the end of a block"));
        }
        if coef_get(comp, at, k)? != 0 {
            refine_magnitude(bits, coef_mut(comp, at, k)?, al)?;
        } else {
            left -= 1;
        }
        k += 1;
    }
    Ok(k)
}

fn eob_run(bits: &mut Bits<'_>, run: u8) -> Result<u32, Error> {
    if run > 14 {
        return Err(Error::Inconsistent("an end-of-band run is not a code"));
    }
    let extra = if run == 0 {
        0
    } else {
        u32::from(bits.bits(run)?)
    };
    let base = 1u32
        .checked_shl(u32::from(run))
        .ok_or(Error::Inconsistent("an end-of-band run does not fit"))?;
    base.checked_add(extra)
        .ok_or(Error::Inconsistent("an end-of-band run does not fit"))
}

fn refine_magnitude(bits: &mut Bits<'_>, coef: &mut i32, al: u8) -> Result<(), Error> {
    if bits.bit()? == 0 {
        return Ok(());
    }
    add_magnitude(coef, al)
}

fn add_magnitude(coef: &mut i32, al: u8) -> Result<(), Error> {
    let bit_value = coef_bit(al)?;
    if *coef > 0 {
        *coef = coef
            .checked_add(bit_value)
            .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
    } else if *coef < 0 {
        *coef = coef
            .checked_sub(bit_value)
            .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
    } else {
        return Err(Error::Inconsistent(
            "a refinement bit changes a zero coefficient",
        ));
    }
    Ok(())
}

fn coef_bit(al: u8) -> Result<i32, Error> {
    1i32.checked_shl(u32::from(al))
        .ok_or(Error::Inconsistent("a coefficient does not fit"))
}

fn check_dc(comp: &Comp, ah: u8) -> Result<(), Error> {
    match (ah, comp.dc_shift) {
        (0, None) => Ok(()),
        (0, Some(_)) => Err(Error::Inconsistent("a DC band is sent twice")),
        (_, None) => Err(Error::Inconsistent(
            "a refinement comes before the first DC scan",
        )),
        (ah, Some(prev)) if prev == ah => Ok(()),
        _ => Err(Error::Inconsistent("a refinement skips a bit")),
    }
}

fn check_ac(comp: &Comp, scan: &Scan) -> Result<(), Error> {
    for k in usize::from(scan.ss)..=usize::from(scan.se) {
        match (scan.ah, comp.ac_shift[k]) {
            (0, None) => {}
            (0, Some(_)) => {
                return Err(Error::Inconsistent("an AC band is sent twice"));
            }
            (_, None) => {
                return Err(Error::Inconsistent(
                    "a refinement comes before the coefficient",
                ));
            }
            (ah, Some(prev)) if prev == ah => {}
            _ => return Err(Error::Inconsistent("a refinement skips a bit")),
        }
    }
    Ok(())
}

fn remember_ac(comp: &mut Comp, band: Band) {
    for k in usize::from(band.ss)..=usize::from(band.se) {
        comp.ac_shift[k] = Some(band.al);
    }
}

fn coef_get(comp: &Comp, at: (usize, usize), k: usize) -> Result<i32, Error> {
    let index = coef_index(comp, at, k)?;
    comp.coefs
        .get(index)
        .copied()
        .ok_or(Error::Inconsistent("a block falls outside its plane"))
}

fn coef_mut(comp: &mut Comp, at: (usize, usize), k: usize) -> Result<&mut i32, Error> {
    let index = coef_index(comp, at, k)?;
    comp.coefs
        .get_mut(index)
        .ok_or(Error::Inconsistent("a block falls outside its plane"))
}

fn coef_index(comp: &Comp, at: (usize, usize), k: usize) -> Result<usize, Error> {
    let (bx, by) = at;
    if bx >= comp.blocks_x || by >= comp.blocks_y || k >= 64 {
        return Err(Error::Inconsistent("a block falls outside its plane"));
    }
    by.checked_mul(comp.blocks_x)
        .and_then(|block| block.checked_add(bx))
        .and_then(|block| block.checked_mul(64))
        .and_then(|block| block.checked_add(k))
        .ok_or(Error::TooLarge)
}

fn write_block(comp: &mut Comp, at: (usize, usize), spatial: &[i32; 64]) -> Result<(), Error> {
    let (bx, by) = at;
    for y in 0..8 {
        for x in 0..8 {
            let row = by * 8 + y;
            let col = bx * 8 + x;
            let index = row
                .checked_mul(comp.stride)
                .and_then(|value| value.checked_add(col))
                .ok_or(Error::TooLarge)?;
            let slot = comp
                .samples
                .get_mut(index)
                .ok_or(Error::Inconsistent("a block falls outside its plane"))?;
            *slot = level(spatial[y * 8 + x]);
        }
    }
    Ok(())
}

fn transform(frame: &mut Frame, tables: &Tables) -> Result<(), Error> {
    for index in 0..frame.comps.len() {
        let tq = frame.comps[index].tq;
        let quant = *tables.quant(tq)?;
        let blocks_x = frame.comps[index].blocks_x;
        let blocks_y = frame.comps[index].blocks_y;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut zz = [0i32; 64];
                for (k, coef) in zz.iter_mut().enumerate() {
                    *coef = coef_get(&frame.comps[index], (bx, by), k)?;
                }
                let natural = dequant(&zz, &quant)?;
                let spatial = idct(&natural)?;
                write_block(&mut frame.comps[index], (bx, by), &spatial)?;
            }
        }
    }
    Ok(())
}

fn paint(frame: &Frame, adobe: Option<u8>) -> Result<Still, Error> {
    let rgb = match adobe {
        Some(0) => {
            if frame.comps.len() != 3 {
                return Err(Error::Inconsistent("an RGB marker needs three components"));
            }
            true
        }
        Some(1) | None => false,
        Some(2) => return Err(Error::Unsupported("YCCK is not supported")),
        Some(_) => return Err(Error::Unsupported("an Adobe transform is not supported")),
    };
    if frame.comps.len() != 1 && frame.comps.len() != 3 {
        return Err(Error::Unsupported("a frame component count is not 1 or 3"));
    }
    let mut pixels = vec![0u8; rgba_len(frame.width, frame.height)?];
    let width = as_usize(frame.width)?;
    for y in 0..frame.height {
        for x in 0..frame.width {
            let first = sample(frame, 0, x, y)?;
            let (red, green, blue) = if frame.comps.len() == 1 {
                (first, first, first)
            } else if rgb {
                (first, sample(frame, 1, x, y)?, sample(frame, 2, x, y)?)
            } else {
                let converted = ycbcr(first, sample(frame, 1, x, y)?, sample(frame, 2, x, y)?);
                (converted[0], converted[1], converted[2])
            };
            let x = as_usize(x)?;
            let y = as_usize(y)?;
            let at = y
                .checked_mul(width)
                .and_then(|row| row.checked_add(x))
                .and_then(|pixel| pixel.checked_mul(4))
                .ok_or(Error::TooLarge)?;
            let pixel = pixels
                .get_mut(at..at + 4)
                .ok_or(Error::Inconsistent("a pixel falls outside the picture"))?;
            pixel.copy_from_slice(&[red, green, blue, 255]);
        }
    }
    Ok(Still::new(frame.width, frame.height, pixels).expect("the RGBA buffer matches the frame"))
}

fn sample(frame: &Frame, index: usize, x: u32, y: u32) -> Result<u8, Error> {
    let comp = frame
        .comps
        .get(index)
        .ok_or(Error::Inconsistent("a component is missing"))?;
    if frame.h_max == 0 || frame.v_max == 0 {
        return Err(Error::Inconsistent("a sampling factor is zero"));
    }
    // Integer repeat: chroma sample (x * H / Hmax, y * V / Vmax).
    let cx = u32::from(comp.h).checked_mul(x).ok_or(Error::TooLarge)? / u32::from(frame.h_max);
    let cy = u32::from(comp.v).checked_mul(y).ok_or(Error::TooLarge)? / u32::from(frame.v_max);
    let cx = as_usize(cx)?;
    let cy = as_usize(cy)?;
    let at = cy
        .checked_mul(comp.stride)
        .and_then(|row| row.checked_add(cx))
        .ok_or(Error::TooLarge)?;
    comp.samples
        .get(at)
        .copied()
        .ok_or(Error::Inconsistent("a sample falls outside its plane"))
}

fn rgba_len(width: u32, height: u32) -> Result<usize, Error> {
    let bytes = as_usize(width)?
        .checked_mul(as_usize(height)?)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Error::TooLarge)?;
    if bytes > MAX_BYTES {
        Err(Error::TooLarge)
    } else {
        Ok(bytes)
    }
}

fn as_usize(value: u32) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::TooLarge)
}

fn marker_at(bytes: &[u8], mut pos: usize) -> Result<(u8, usize), Error> {
    if pos >= bytes.len() {
        return Err(truncated(pos, 2));
    }
    if bytes[pos] != 0xFF {
        return Err(Error::Inconsistent("a marker is missing"));
    }
    pos += 1;
    while pos < bytes.len() && bytes[pos] == 0xFF {
        pos += 1;
    }
    if pos >= bytes.len() {
        return Err(truncated(pos, 1));
    }
    let code = bytes[pos];
    Ok((code, pos + 1))
}

fn segment(bytes: &[u8], pos: usize) -> Result<(&[u8], usize), Error> {
    let len = usize::from(read_u16(bytes, pos)?);
    if len < 2 {
        return Err(Error::Inconsistent(
            "a segment length is shorter than itself",
        ));
    }
    let end = pos.checked_add(len).ok_or(Error::TooLarge)?;
    if end > bytes.len() {
        return Err(truncated(pos, len));
    }
    Ok((&bytes[pos + 2..end], end))
}

fn read_u16(bytes: &[u8], pos: usize) -> Result<u16, Error> {
    let end = pos.checked_add(2).ok_or(Error::TooLarge)?;
    if end > bytes.len() {
        return Err(truncated(pos, 2));
    }
    Ok(u16::from_be_bytes([bytes[pos], bytes[pos + 1]]))
}

fn entropy_end(bytes: &[u8], mut index: usize) -> Result<usize, Error> {
    while index < bytes.len() {
        if bytes[index] != 0xFF {
            index += 1;
            continue;
        }
        let mut next = index + 1;
        while next < bytes.len() && bytes[next] == 0xFF {
            next += 1;
        }
        if next >= bytes.len() {
            return Err(truncated(index, 2));
        }
        let code = bytes[next];
        if code == 0x00 || (0xD0..=0xD7).contains(&code) {
            index = next + 1;
            continue;
        }
        return Ok(index);
    }
    Err(Error::Inconsistent("the file has no end"))
}

fn truncated(at: usize, need: usize) -> Error {
    Error::Truncated {
        at: u64::try_from(at).unwrap_or(u64::MAX),
        need: u64::try_from(need).unwrap_or(u64::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::decode;
    use crate::error::Error;
    use crate::idct::{dequant, idct, level};
    use cutlass_core::Still;

    #[test]
    fn a_zero_block_is_twelve_zero_bits() {
        let bytes = entropy(&[vec![(0, zz_dc(0))]], 1, 0);
        assert_eq!(bytes, [0x00, 0x0F]);
    }

    #[test]
    fn a_zero_block_is_mid_grey() {
        let still = open(&grey(8, 8, &[zz_dc(0)], 0));
        assert_grey_blocks(&still, 1, &[zz_dc(0)]);
    }

    #[test]
    fn a_dc_of_sixteen_is_two_above_grey() {
        let still = open(&grey(8, 8, &[zz_dc(16)], 0));
        assert_grey_blocks(&still, 1, &[zz_dc(16)]);
        assert_eq!(pixel_at(&still, 0, 0), [130, 130, 130, 255]);
    }

    #[test]
    fn a_negative_dc_drops_below_grey() {
        let still = open(&grey(8, 8, &[zz_dc(-16)], 0));
        assert_grey_blocks(&still, 1, &[zz_dc(-16)]);
        assert_eq!(pixel_at(&still, 0, 0)[0], 126);
    }

    #[test]
    fn the_dc_predictor_carries_into_the_next_block() {
        let blocks = [zz_dc(16), zz_dc(32)];
        let still = open(&grey(16, 8, &blocks, 0));
        assert_grey_blocks(&still, 2, &blocks);
        assert_eq!(pixel_at(&still, 0, 0)[0], 130);
        assert_eq!(pixel_at(&still, 8, 0)[0], 132);
    }

    #[test]
    fn a_restart_resets_the_dc_predictor() {
        let blocks = [zz_dc(16), zz_dc(16)];
        let still = open(&grey(16, 8, &blocks, 1));
        assert_grey_blocks(&still, 2, &blocks);
        assert_eq!(pixel_at(&still, 0, 0)[0], 130);
        assert_eq!(pixel_at(&still, 8, 0)[0], 130);
    }

    #[test]
    fn a_one_pixel_image_crops_the_block() {
        let still = open(&grey(1, 1, &[zz_dc(16)], 0));
        assert_eq!(still.width(), 1);
        assert_eq!(still.height(), 1);
        assert_eq!(still.pixels(), [130, 130, 130, 255]);
    }

    #[test]
    fn a_stuffed_ff_stays_a_coefficient() {
        let still = open(&grey(8, 8, &[zz_dc(2047)], 0));
        assert_color(&still, [255, 255, 255, 255]);
    }

    #[test]
    fn a_horizontal_ac_matches_the_fixed_idct() {
        let mut zz = [0i32; 64];
        zz[1] = 80;
        let still = open(&grey(8, 8, &[zz], 0));
        assert_grey_blocks(&still, 1, &[zz]);
        assert_eq!(pixel_at(&still, 0, 0), [142, 142, 142, 255]);
    }

    #[test]
    fn a_zero_run_reaches_a_later_coefficient() {
        let mut zz = [0i32; 64];
        zz[17] = 80;
        let still = open(&grey(8, 8, &[zz], 0));
        assert_grey_blocks(&still, 1, &[zz]);
    }

    #[test]
    fn subsampled_chroma_repeats_across_the_luma_block() {
        let y = zz_dc(0);
        let cb = zz_dc(16);
        let cr = zz_dc(0);
        let mcu = vec![(0, y), (0, y), (0, y), (0, y), (1, cb), (2, cr)];
        let bytes = encode(&Parts {
            width: 16,
            height: 16,
            comps: vec![(1, 2, 2), (2, 1, 1), (3, 1, 1)],
            scans: vec![scan_entropy(&[1, 2, 3], &entropy(&[mcu], 3, 0))],
            ..Parts::grey(16, 16)
        });
        let still = open(&bytes);
        assert_color(&still, [128, 127, 132, 255]);
    }

    #[test]
    fn horizontal_subsampling_keeps_neutral_chroma() {
        let zero = zz_dc(0);
        let mcu = vec![(0, zero), (0, zero), (1, zero), (2, zero)];
        let bytes = encode(&Parts {
            width: 16,
            height: 8,
            comps: vec![(1, 2, 1), (2, 1, 1), (3, 1, 1)],
            scans: vec![scan_entropy(&[1, 2, 3], &entropy(&[mcu], 3, 0))],
            ..Parts::grey(16, 8)
        });
        assert_color(&open(&bytes), [128, 128, 128, 255]);
    }

    #[test]
    fn an_app_segment_and_a_comment_do_not_change_pixels() {
        let plain = grey(8, 8, &[zz_dc(0)], 0);
        let mut parts = Parts::grey(8, 8);
        parts.apps.push((0xE0, b"JFIF\0".to_vec()));
        parts.apps.push((0xE1, b"Exif\0\0junk".to_vec()));
        parts.apps.push((0xFE, b"no turn".to_vec()));
        parts.scans = vec![scan_entropy(&[1], &entropy(&[vec![(0, zz_dc(0))]], 1, 0))];
        assert_eq!(open(&plain).pixels(), open(&encode(&parts)).pixels());
    }

    #[test]
    fn adobe_rgb_is_copied_and_ycbcr_is_converted() {
        let blocks = [zz_dc(-944), zz_dc(-864), zz_dc(-784)];
        let mcus = vec![
            blocks
                .iter()
                .enumerate()
                .map(|(index, zz)| (index, *zz))
                .collect(),
        ];
        let entropy = entropy(&mcus, 3, 0);
        let comps = vec![(1, 1, 1), (2, 1, 1), (3, 1, 1)];
        let base = Parts {
            width: 8,
            height: 8,
            comps: comps.clone(),
            scans: vec![scan_entropy(&[1, 2, 3], &entropy)],
            ..Parts::grey(8, 8)
        };
        let converted = open(&encode(&base));
        assert_color(&converted, [0, 117, 0, 255]);
        let rgb = Parts {
            adobe: Some(0),
            ..base
        };
        assert_color(&open(&encode(&rgb)), [10, 20, 30, 255]);
    }

    #[test]
    fn extended_sequential_matches_baseline() {
        let mut parts = Parts::grey(8, 8);
        parts.sof = 0xC1;
        parts.scans = vec![scan_entropy(&[1], &entropy(&[vec![(0, zz_dc(0))]], 1, 0))];
        assert_color(&open(&encode(&parts)), [128, 128, 128, 255]);
    }

    #[test]
    fn progressive_dc_shifts_before_the_level() {
        let mut parts = Parts::grey(8, 8);
        parts.sof = 0xC2;
        parts.scans = vec![ScanBytes {
            ids: vec![1],
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
            entropy: dc_entropy(&[8]),
        }];
        let still = open(&encode(&parts));
        assert_eq!(pixel_at(&still, 0, 0)[0], 130);
    }

    #[test]
    fn progressive_dc_refinement_adds_the_low_bit() {
        let mut first = Parts::grey(8, 8);
        first.sof = 0xC2;
        // Unshifted 1, point transform 4, stores 16. The next bit adds 8.
        first.scans = vec![ScanBytes {
            ids: vec![1],
            ss: 0,
            se: 0,
            ah: 0,
            al: 4,
            entropy: dc_entropy(&[1]),
        }];
        assert_eq!(
            pixel_at(&open(&encode(&first)), 0, 0)[0],
            block_samples(&zz_dc(16))[0]
        );
        first.scans.push(ScanBytes {
            ids: vec![1],
            ss: 0,
            se: 0,
            ah: 4,
            al: 3,
            entropy: raw_bits(&[1]),
        });
        let refined = block_samples(&zz_dc(24))[0];
        assert_ne!(refined, block_samples(&zz_dc(16))[0]);
        assert_eq!(pixel_at(&open(&encode(&first)), 0, 0)[0], refined);
    }

    #[test]
    fn a_progressive_ac_scan_matches_the_sequential_block() {
        let mut zz = [0i32; 64];
        zz[1] = 80;
        let sequential = open(&grey(8, 8, &[zz], 0));
        let mut parts = Parts::grey(8, 8);
        parts.sof = 0xC2;
        parts.scans = vec![
            ScanBytes {
                ids: vec![1],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
                entropy: dc_entropy(&[0]),
            },
            ScanBytes {
                ids: vec![1],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
                entropy: ac_entropy(&zz),
            },
        ];
        assert_eq!(open(&encode(&parts)).pixels(), sequential.pixels());
    }

    #[test]
    fn progressive_ac_refinement_adds_sixteen() {
        let mut high = [0i32; 64];
        high[1] = 2;
        let mut kept = [0i32; 64];
        kept[1] = 64;
        let mut bumped = [0i32; 64];
        bumped[1] = 80;
        let mut parts = Parts::grey(8, 8);
        parts.sof = 0xC2;
        parts.scans = vec![ScanBytes {
            ids: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 5,
            entropy: ac_entropy(&high),
        }];
        let bit0 = parts.clone();
        let mut stay = bit0.clone();
        stay.scans.push(ScanBytes {
            ids: vec![1],
            ss: 1,
            se: 63,
            ah: 5,
            al: 4,
            entropy: refine_eob_bit(0),
        });
        let mut add = parts;
        add.scans.push(ScanBytes {
            ids: vec![1],
            ss: 1,
            se: 63,
            ah: 5,
            al: 4,
            entropy: refine_eob_bit(1),
        });
        let stayed = open(&encode(&stay));
        let added = open(&encode(&add));
        assert_grey_blocks(&stayed, 1, &[kept]);
        assert_grey_blocks(&added, 1, &[bumped]);
        assert_ne!(stayed.pixels(), added.pixels());
    }

    #[test]
    fn a_new_refined_coefficient_keeps_the_sign_with_the_symbol() {
        let mut high = [0i32; 64];
        high[1] = 2;
        let mut expected = [0i32; 64];
        expected[1] = 64;
        expected[3] = 16;
        let mut parts = Parts::grey(8, 8);
        parts.sof = 0xC2;
        parts.scans = vec![
            ScanBytes {
                ids: vec![1],
                ss: 1,
                se: 63,
                ah: 0,
                al: 5,
                entropy: ac_entropy(&high),
            },
            ScanBytes {
                ids: vec![1],
                ss: 1,
                se: 63,
                ah: 5,
                al: 4,
                entropy: refine_new_after_one(1, 0),
            },
        ];
        assert_grey_blocks(&open(&encode(&parts)), 1, &[expected]);
    }

    #[test]
    fn an_end_of_band_run_covers_the_next_block() {
        let mut first = [0i32; 64];
        first[1] = 80;
        let second = [0i32; 64];
        let mut parts = Parts::grey(16, 8);
        parts.sof = 0xC2;
        parts.scans = vec![ScanBytes {
            ids: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
            entropy: ac_value_then_eobrun(1, 80, 1, 0),
        }];
        assert_grey_blocks(&open(&encode(&parts)), 2, &[first, second]);
    }

    #[test]
    fn interleaved_progressive_dc_is_mid_grey() {
        let bytes = encode(&Parts {
            sof: 0xC2,
            width: 8,
            height: 8,
            comps: vec![(1, 1, 1), (2, 1, 1), (3, 1, 1)],
            scans: vec![ScanBytes {
                ids: vec![1, 2, 3],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
                entropy: dc_entropy(&[0, 0, 0]),
            }],
            ..Parts::grey(8, 8)
        });
        assert_color(&open(&bytes), [128, 128, 128, 255]);
    }

    #[test]
    fn ycck_is_refused() {
        let mut parts = Parts::grey(8, 8);
        parts.adobe = Some(2);
        parts.scans = vec![scan_entropy(&[1], &entropy(&[vec![(0, zz_dc(0))]], 1, 0))];
        assert_eq!(
            decode(&encode(&parts)),
            Err(Error::Unsupported("YCCK is not supported"))
        );
    }

    #[test]
    fn the_file_must_start_with_soi() {
        assert_eq!(decode(&[]), Err(Error::Signature));
        assert_eq!(decode(&[0xFF, 0xD9]), Err(Error::Signature));
    }

    #[test]
    fn a_short_file_ends_before_the_frame() {
        assert!(matches!(
            decode(&[0xFF, 0xD8]),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn a_frame_without_eoi_has_no_end() {
        let mut bytes = grey(8, 8, &[zz_dc(0)], 0);
        bytes.pop();
        bytes.pop();
        assert_eq!(
            decode(&bytes),
            Err(Error::Inconsistent("the file has no end"))
        );
    }

    #[test]
    fn a_byte_after_eoi_is_an_error() {
        let mut bytes = grey(8, 8, &[zz_dc(0)], 0);
        bytes.push(0);
        assert_eq!(
            decode(&bytes),
            Err(Error::Inconsistent("a byte follows the end marker"))
        );
    }

    #[test]
    fn soi_and_eoi_alone_have_no_frame() {
        assert_eq!(
            decode(&[0xFF, 0xD8, 0xFF, 0xD9]),
            Err(Error::Inconsistent("the file has no frame"))
        );
    }

    #[test]
    fn twelve_bit_precision_is_refused() {
        let mut parts = Parts::grey(8, 8);
        parts.precision = 12;
        assert_eq!(
            decode(&encode(&parts)),
            Err(Error::Unsupported("the sample precision is not 8 bits"))
        );
    }

    #[test]
    fn four_components_are_refused() {
        let parts = Parts {
            comps: vec![(1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1)],
            ..Parts::grey(8, 8)
        };
        assert_eq!(
            decode(&encode(&parts)),
            Err(Error::Unsupported("a frame component count is not 1 or 3"))
        );
    }

    #[test]
    fn arithmetic_and_lossless_frames_are_refused() {
        assert_eq!(
            decode(&[0xFF, 0xD8, 0xFF, 0xC9]),
            Err(Error::Unsupported("arithmetic coding is not supported"))
        );
        assert_eq!(
            decode(&[0xFF, 0xD8, 0xFF, 0xC3]),
            Err(Error::Unsupported("a lossless frame is not supported"))
        );
    }

    #[test]
    fn a_zero_edge_is_refused() {
        let mut width = Parts::grey(0, 8);
        width.scans.clear();
        assert_eq!(
            decode(&encode(&width)),
            Err(Error::Inconsistent("the width is zero"))
        );
        let mut height = Parts::grey(8, 0);
        height.scans.clear();
        assert_eq!(
            decode(&encode(&height)),
            Err(Error::Unsupported("the height is defined later"))
        );
    }

    #[test]
    fn a_zero_sampling_factor_is_refused() {
        let parts = Parts {
            comps: vec![(1, 0, 1)],
            ..Parts::grey(8, 8)
        };
        assert_eq!(
            decode(&encode(&parts)),
            Err(Error::Inconsistent("a sampling factor is zero"))
        );
    }

    #[test]
    fn sampling_past_the_baseline_limit_is_refused() {
        let parts = Parts {
            comps: vec![(1, 2, 2), (2, 2, 2), (3, 2, 2)],
            ..Parts::grey(8, 8)
        };
        assert_eq!(
            decode(&encode(&parts)),
            Err(Error::Inconsistent(
                "the sampling factors are past the baseline limit"
            ))
        );
    }

    #[test]
    fn an_edge_past_the_limit_allocates_nothing() {
        let mut parts = Parts::grey(40_000, 8);
        parts.scans.clear();
        assert_eq!(decode(&encode(&parts)), Err(Error::TooLarge));
    }

    #[test]
    fn an_rgba_buffer_past_the_limit_allocates_nothing() {
        let mut parts = Parts::grey(8192, 8193);
        parts.scans.clear();
        assert_eq!(decode(&encode(&parts)), Err(Error::TooLarge));
    }

    #[test]
    fn progressive_coefficients_past_the_limit_allocate_nothing() {
        let mut parts = Parts::grey(8192, 8192);
        parts.sof = 0xC2;
        parts.comps = vec![(1, 1, 1), (2, 1, 1), (3, 1, 1)];
        parts.scans.clear();
        assert_eq!(decode(&encode(&parts)), Err(Error::TooLarge));
    }

    #[test]
    fn a_marker_inside_a_coefficient_is_an_error() {
        let mut parts = Parts::grey(8, 8);
        parts.scans = vec![scan_entropy(&[1], &[0x00, 0xFF, 0xD0])];
        assert_eq!(
            decode(&encode(&parts)),
            Err(Error::Inconsistent("a marker sits inside a coefficient"))
        );
    }

    #[test]
    fn a_missing_restart_is_an_error() {
        let mut parts = Parts::grey(16, 8);
        parts.restart = 1;
        parts.scans = vec![scan_entropy(
            &[1],
            &entropy(&[vec![(0, zz_dc(16))], vec![(0, zz_dc(16))]], 1, 0),
        )];
        assert_eq!(
            decode(&encode(&parts)),
            Err(Error::Inconsistent("a restart marker is missing"))
        );
    }

    #[test]
    fn a_restart_out_of_order_is_an_error() {
        let mut bytes = grey(16, 8, &[zz_dc(16), zz_dc(16)], 1);
        let found = bytes.windows(2).position(|pair| pair == [0xFF, 0xD0]);
        let at = found.expect("the test file has RST0");
        bytes[at + 1] = 0xD1;
        assert_eq!(
            decode(&bytes),
            Err(Error::Inconsistent("a restart marker is out of order"))
        );
    }

    #[test]
    fn a_second_sequential_scan_is_an_error() {
        let mut parts = Parts::grey(8, 8);
        let entropy = entropy(&[vec![(0, zz_dc(0))]], 1, 0);
        parts.scans = vec![scan_entropy(&[1], &entropy), scan_entropy(&[1], &entropy)];
        assert_eq!(
            decode(&encode(&parts)),
            Err(Error::Inconsistent("a sequential frame has a second scan"))
        );
    }

    fn open(bytes: &[u8]) -> Still {
        decode(bytes).expect("the test file decodes")
    }

    fn assert_color(still: &Still, expected: [u8; 4]) {
        assert!(!still.pixels().is_empty());
        for pixel in still.pixels().chunks(4) {
            assert_eq!(pixel, expected);
        }
    }

    fn assert_grey_blocks(still: &Still, across: u32, blocks: &[[i32; 64]]) {
        let decoded: Vec<[u8; 64]> = blocks.iter().map(block_samples).collect();
        for y in 0..still.height() {
            for x in 0..still.width() {
                let index = usize::try_from((y / 8) * across + x / 8).expect("block index");
                let sx = usize::try_from(x % 8).expect("x");
                let sy = usize::try_from(y % 8).expect("y");
                let value = decoded[index][sy * 8 + sx];
                assert_eq!(
                    pixel_at(still, x, y),
                    [value, value, value, 255],
                    "pixel {x},{y}"
                );
            }
        }
    }

    fn block_samples(zz: &[i32; 64]) -> [u8; 64] {
        let quant = [1u16; 64];
        let natural = dequant(zz, &quant).expect("dequant");
        let spatial = idct(&natural).expect("idct");
        let mut out = [0u8; 64];
        for (slot, value) in out.iter_mut().zip(spatial) {
            *slot = level(value);
        }
        out
    }

    fn pixel_at(still: &Still, x: u32, y: u32) -> [u8; 4] {
        let width = usize::try_from(still.width()).expect("width");
        let x = usize::try_from(x).expect("x");
        let y = usize::try_from(y).expect("y");
        let at = (y * width + x) * 4;
        let bytes = still.pixels();
        [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]
    }

    fn grey(width: u16, height: u16, blocks: &[[i32; 64]], restart: u16) -> Vec<u8> {
        let mcus = blocks
            .iter()
            .map(|block| vec![(0usize, *block)])
            .collect::<Vec<_>>();
        let mut parts = Parts::grey(width, height);
        parts.restart = restart;
        parts.scans = vec![scan_entropy(&[1], &entropy(&mcus, 1, restart))];
        encode(&parts)
    }

    fn zz_dc(dc: i32) -> [i32; 64] {
        let mut zz = [0; 64];
        zz[0] = dc;
        zz
    }

    #[derive(Clone)]
    struct Parts {
        sof: u8,
        precision: u8,
        width: u16,
        height: u16,
        comps: Vec<(u8, u8, u8)>,
        restart: u16,
        adobe: Option<u8>,
        apps: Vec<(u8, Vec<u8>)>,
        scans: Vec<ScanBytes>,
    }

    impl Parts {
        fn grey(width: u16, height: u16) -> Self {
            Self {
                sof: 0xC0,
                precision: 8,
                width,
                height,
                comps: vec![(1, 1, 1)],
                restart: 0,
                adobe: None,
                apps: Vec::new(),
                scans: Vec::new(),
            }
        }
    }

    #[derive(Clone)]
    struct ScanBytes {
        ids: Vec<u8>,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        entropy: Vec<u8>,
    }

    fn scan_entropy(ids: &[u8], entropy: &[u8]) -> ScanBytes {
        ScanBytes {
            ids: ids.to_vec(),
            ss: 0,
            se: 63,
            ah: 0,
            al: 0,
            entropy: entropy.to_vec(),
        }
    }

    fn encode(parts: &Parts) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8];
        for (marker, payload) in &parts.apps {
            // Comments use marker 0xFE. APP markers are 0xE0 through 0xEF.
            segment(&mut out, *marker, payload);
        }
        if let Some(transform) = parts.adobe {
            segment(&mut out, 0xEE, &adobe(transform));
        }
        segment(&mut out, 0xDB, &dqt_ones());
        segment(&mut out, 0xC4, &dht_dc());
        segment(&mut out, 0xC4, &dht_ac());
        if parts.restart != 0 {
            segment(&mut out, 0xDD, &parts.restart.to_be_bytes());
        }
        segment(&mut out, parts.sof, &sof(parts));
        for scan in &parts.scans {
            segment(&mut out, 0xDA, &sos(scan));
            out.extend_from_slice(&scan.entropy);
        }
        out.extend_from_slice(&[0xFF, 0xD9]);
        out
    }

    fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        out.push(0xFF);
        out.push(marker);
        let len = u16::try_from(payload.len() + 2).expect("a test segment fits");
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(payload);
    }

    fn sof(parts: &Parts) -> Vec<u8> {
        let mut payload = vec![parts.precision];
        payload.extend_from_slice(&parts.height.to_be_bytes());
        payload.extend_from_slice(&parts.width.to_be_bytes());
        payload.push(u8::try_from(parts.comps.len()).expect("component count fits"));
        for (id, h, v) in &parts.comps {
            payload.push(*id);
            payload.push((h << 4) | v);
            payload.push(0);
        }
        payload
    }

    fn sos(scan: &ScanBytes) -> Vec<u8> {
        let mut payload = vec![u8::try_from(scan.ids.len()).expect("scan count fits")];
        for id in &scan.ids {
            payload.push(*id);
            payload.push(0);
        }
        payload.push(scan.ss);
        payload.push(scan.se);
        payload.push((scan.ah << 4) | scan.al);
        payload
    }

    fn dqt_ones() -> Vec<u8> {
        let mut payload = vec![0x00];
        payload.extend(std::iter::repeat_n(1u8, 64));
        payload
    }

    fn dht_dc() -> Vec<u8> {
        let mut payload = vec![0x00];
        let mut counts = [0u8; 16];
        counts[3] = 12;
        payload.extend(counts);
        payload.extend(0..12);
        payload
    }

    fn dht_ac() -> Vec<u8> {
        let mut payload = vec![0x10];
        let mut counts = [0u8; 16];
        counts[7] = 251;
        payload.extend(counts);
        payload.extend(0..=250);
        payload
    }

    fn adobe(transform: u8) -> Vec<u8> {
        let mut payload = b"Adobe\0".to_vec();
        payload.extend_from_slice(&100u16.to_be_bytes());
        payload.extend_from_slice(&[0, 0, 0, 0]);
        payload.push(transform);
        payload
    }

    fn entropy(mcus: &[Vec<(usize, [i32; 64])>], comps: usize, restart: u16) -> Vec<u8> {
        let mut writer = Writer::default();
        let mut pred = vec![0i32; comps];
        for (index, mcu) in mcus.iter().enumerate() {
            if restart != 0 && index > 0 && index.is_multiple_of(usize::from(restart)) {
                let which = index / usize::from(restart) - 1;
                writer.restart(u8::try_from(which % 8).expect("a restart index fits"));
                pred.fill(0);
            }
            for (comp, zz) in mcu {
                write_block(&mut writer, &mut pred[*comp], zz);
            }
        }
        writer.finish();
        writer.bytes
    }

    fn dc_entropy(values: &[i32]) -> Vec<u8> {
        let mut writer = Writer::default();
        let mut pred = 0i32;
        for value in values {
            let diff = value.checked_sub(pred).expect("a test DC difference fits");
            pred = *value;
            write_dc(&mut writer, diff);
        }
        writer.finish();
        writer.bytes
    }

    fn ac_entropy(zz: &[i32; 64]) -> Vec<u8> {
        let mut writer = Writer::default();
        write_ac_band(&mut writer, zz, 1, 63);
        writer.finish();
        writer.bytes
    }

    fn ac_value_then_eobrun(k: usize, value: i32, run: u8, extra: u32) -> Vec<u8> {
        let mut writer = Writer::default();
        let mut pos = 1usize;
        let mut zeros = 0u8;
        while pos < k {
            zeros += 1;
            if zeros == 16 {
                writer.write_bits(0xF0, 8);
                zeros = 0;
            }
            pos += 1;
        }
        let cat = category(value);
        writer.write_bits((u32::from(zeros) << 4) | u32::from(cat), 8);
        if cat > 0 {
            writer.write_bits(mag_bits(value, cat), cat);
        }
        writer.write_bits(u32::from(run) << 4, 8);
        if run > 0 {
            writer.write_bits(extra, run);
        }
        writer.finish();
        writer.bytes
    }

    fn raw_bits(bits: &[u8]) -> Vec<u8> {
        let mut writer = Writer::default();
        for bit in bits {
            writer.write_bits(u32::from(*bit), 1);
        }
        writer.finish();
        writer.bytes
    }

    fn refine_eob_bit(bit: u8) -> Vec<u8> {
        let mut writer = Writer::default();
        writer.write_bits(0, 8);
        writer.write_bits(u32::from(bit), 1);
        writer.finish();
        writer.bytes
    }

    fn refine_new_after_one(sign: u8, refine: u8) -> Vec<u8> {
        let mut writer = Writer::default();
        writer.write_bits(0x11, 8);
        writer.write_bits(u32::from(sign), 1);
        writer.write_bits(u32::from(refine), 1);
        writer.write_bits(0, 8);
        writer.finish();
        writer.bytes
    }

    fn write_block(writer: &mut Writer, pred: &mut i32, zz: &[i32; 64]) {
        let dc = zz[0];
        let diff = dc.checked_sub(*pred).expect("a test DC difference fits");
        *pred = dc;
        write_dc(writer, diff);
        write_ac_band(writer, zz, 1, 63);
    }

    fn write_dc(writer: &mut Writer, diff: i32) {
        let cat = category(diff);
        writer.write_bits(u32::from(cat), 4);
        if cat > 0 {
            writer.write_bits(mag_bits(diff, cat), cat);
        }
    }

    fn write_ac_band(writer: &mut Writer, zz: &[i32; 64], ss: usize, se: usize) {
        let mut k = ss;
        let mut run = 0u8;
        while k <= se {
            if zz[k] == 0 {
                run += 1;
                k += 1;
                continue;
            }
            while run >= 16 {
                writer.write_bits(0xF0, 8);
                run -= 16;
            }
            let cat = category(zz[k]);
            writer.write_bits((u32::from(run) << 4) | u32::from(cat), 8);
            if cat > 0 {
                writer.write_bits(mag_bits(zz[k], cat), cat);
            }
            run = 0;
            k += 1;
        }
        if run > 0 {
            writer.write_bits(0, 8);
        }
    }

    fn category(value: i32) -> u8 {
        let magnitude = value.unsigned_abs();
        if magnitude == 0 {
            0
        } else {
            u8::try_from(32 - magnitude.leading_zeros()).expect("a test category fits in a byte")
        }
    }

    fn mag_bits(value: i32, cat: u8) -> u32 {
        if value < 0 {
            let span = (1i32 << cat) - 1;
            u32::try_from(value + span).expect("negative magnitude bits are non-negative")
        } else {
            u32::try_from(value).expect("a positive magnitude fits")
        }
    }

    #[derive(Default)]
    struct Writer {
        bytes: Vec<u8>,
        acc: u32,
        n: u8,
    }

    impl Writer {
        fn write_bits(&mut self, value: u32, count: u8) {
            assert!(count <= 32, "a test code is at most 32 bits");
            for bit in (0..count).rev() {
                let one = (value >> u32::from(bit)) & 1;
                self.acc = (self.acc << 1) | one;
                self.n += 1;
                if self.n == 8 {
                    self.emit(u8::try_from(self.acc).expect("a byte fits"));
                    self.acc = 0;
                    self.n = 0;
                }
            }
        }

        fn emit(&mut self, byte: u8) {
            self.bytes.push(byte);
            if byte == 0xFF {
                self.bytes.push(0x00);
            }
        }

        fn finish(&mut self) {
            if self.n == 0 {
                return;
            }
            let pad = 8 - self.n;
            self.acc = (self.acc << pad) | ((1u32 << pad) - 1);
            self.emit(u8::try_from(self.acc).expect("a byte fits"));
            self.n = 0;
            self.acc = 0;
        }

        fn restart(&mut self, index: u8) {
            self.finish();
            self.bytes.push(0xFF);
            self.bytes.push(0xD0 + index);
        }
    }
}
