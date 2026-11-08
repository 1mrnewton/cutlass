//! Baseline sequential encoder core.
//!
//! [`Coefficients::transform`] runs the forward DCT once. The same
//! coefficients can then be counted at several qualities and written at one.
//! Output is baseline SOF0 with the Annex K tables scaled the way the
//! Independent JPEG Group scales them, and the Annex K.3 Huffman tables.

use cutlass_decode::CodecError;

use crate::color::rgb_to_ycbcr;
use crate::dct::fdct_block;
use crate::huffman::{HuffEncode, encode_table};
use crate::limits::dimensions_ok;
use crate::tables::{
    AC_CHROMA_COUNTS, AC_CHROMA_SYMBOLS, AC_LUMA_COUNTS, AC_LUMA_SYMBOLS, CHROMINANCE_QUANT,
    DC_CHROMA_COUNTS, DC_CHROMA_SYMBOLS, DC_LUMA_COUNTS, DC_LUMA_SYMBOLS, LUMINANCE_QUANT, ZIGZAG,
    quality_scale, scale_quant,
};

/// Component layout written by the encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sampling {
    Gray,
    Yuv444,
    Yuv422,
    Yuv420,
}

impl Sampling {
    /// Luma sampling factors; chroma is always 1×1.
    fn luma_factors(self) -> (u8, u8) {
        match self {
            Self::Gray | Self::Yuv444 => (1, 1),
            Self::Yuv422 => (2, 1),
            Self::Yuv420 => (2, 2),
        }
    }

    fn components(self) -> usize {
        if self == Self::Gray { 1 } else { 3 }
    }

    /// Chroma plane size for a picture `width` × `height`.
    #[must_use]
    pub fn chroma_size(self, width: u32, height: u32) -> (u32, u32) {
        let (h, v) = self.luma_factors();
        (width.div_ceil(u32::from(h)), height.div_ceil(u32::from(v)))
    }
}

/// One source plane: rows `stride` bytes apart.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRef<'a> {
    pub samples: &'a [u8],
    pub stride: usize,
}

/// Source planes at the sizes of their sampling. Grey reads only `y`.
#[derive(Clone, Copy, Debug)]
pub struct Source<'a> {
    pub width: u32,
    pub height: u32,
    pub sampling: Sampling,
    pub y: PlaneRef<'a>,
    pub cb: PlaneRef<'a>,
    pub cr: PlaneRef<'a>,
}

/// Owned JFIF YCbCr planes, tightly packed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Planes {
    pub width: u32,
    pub height: u32,
    pub sampling: Option<Sampling>,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

impl Planes {
    /// Convert interleaved RGB or RGBA (`channels` 3 or 4, alpha ignored)
    /// to JFIF YCbCr and subsample chroma by averaging, as libjpeg does.
    /// Grey takes the luma only.
    pub fn from_rgb(
        &mut self,
        pixels: &[u8],
        channels: usize,
        width: u32,
        height: u32,
        sampling: Sampling,
    ) -> Result<(), CodecError> {
        if !(3..=4).contains(&channels) {
            return Err(invalid());
        }
        if !dimensions_ok(width, height) {
            return Err(CodecError::Limit { what: "jpeg size" });
        }
        let w = usize::try_from(width).map_err(|_| invalid())?;
        let h = usize::try_from(height).map_err(|_| invalid())?;
        if pixels.len() != w * h * channels {
            return Err(invalid());
        }
        let pixels_len = w * h;
        resize(&mut self.y, pixels_len)?;
        let color = sampling != Sampling::Gray;
        if color {
            // Full-size chroma first, then subsample in place of the copy.
            resize(&mut self.cb, pixels_len)?;
            resize(&mut self.cr, pixels_len)?;
        }
        for (index, pixel) in pixels.chunks_exact(channels).enumerate() {
            let (y, cb, cr) = rgb_to_ycbcr(
                i32::from(pixel[0]),
                i32::from(pixel[1]),
                i32::from(pixel[2]),
            );
            self.y[index] = y;
            if color {
                self.cb[index] = cb;
                self.cr[index] = cr;
            }
        }
        if color {
            self.cb = subsample(&self.cb, w, h, sampling)?;
            self.cr = subsample(&self.cr, w, h, sampling)?;
        } else {
            self.cb.clear();
            self.cr.clear();
        }
        self.width = width;
        self.height = height;
        self.sampling = Some(sampling);
        Ok(())
    }

    /// Borrow these planes as encoder input.
    pub fn source(&self) -> Result<Source<'_>, CodecError> {
        let sampling = self.sampling.ok_or_else(invalid)?;
        let (cw, _) = sampling.chroma_size(self.width, self.height);
        Ok(Source {
            width: self.width,
            height: self.height,
            sampling,
            y: PlaneRef {
                samples: &self.y,
                stride: usize::try_from(self.width).map_err(|_| invalid())?,
            },
            cb: PlaneRef {
                samples: &self.cb,
                stride: usize::try_from(cw).map_err(|_| invalid())?,
            },
            cr: PlaneRef {
                samples: &self.cr,
                stride: usize::try_from(cw).map_err(|_| invalid())?,
            },
        })
    }
}

fn invalid() -> CodecError {
    CodecError::InvalidApi {
        what: "jpeg picture",
    }
}

fn resize<T: Clone + Default>(values: &mut Vec<T>, len: usize) -> Result<(), CodecError> {
    values.clear();
    values.try_reserve(len).map_err(|_| CodecError::Limit {
        what: "jpeg picture",
    })?;
    values.resize(len, T::default());
    Ok(())
}

/// Average 2×1 or 2×2 neighbours with libjpeg's alternating rounding bias.
/// The right and bottom edges repeat the last sample.
fn subsample(
    full: &[u8],
    width: usize,
    height: usize,
    sampling: Sampling,
) -> Result<Vec<u8>, CodecError> {
    let (h, v) = sampling.luma_factors();
    let (cw, ch) = (
        width.div_ceil(usize::from(h)),
        height.div_ceil(usize::from(v)),
    );
    let mut out = Vec::new();
    resize(&mut out, cw * ch)?;
    let at = |x: usize, y: usize| u16::from(full[y.min(height - 1) * width + x.min(width - 1)]);
    for cy in 0..ch {
        for cx in 0..cw {
            let value = match (h, v) {
                (2, 2) => {
                    let bias = if cx % 2 == 0 { 1 } else { 2 };
                    let (x, y) = (cx * 2, cy * 2);
                    (at(x, y) + at(x + 1, y) + at(x, y + 1) + at(x + 1, y + 1) + bias) >> 2
                }
                (2, 1) => {
                    let bias = u16::from(cx % 2 == 1);
                    (at(cx * 2, cy) + at(cx * 2 + 1, cy) + bias) >> 1
                }
                _ => at(cx, cy),
            };
            out[cy * cw + cx] = u8::try_from(value).unwrap_or(u8::MAX);
        }
    }
    Ok(out)
}

struct CoefComp {
    h: u8,
    v: u8,
    /// Blocks per row and column, whole MCUs.
    blocks_w: usize,
    blocks_h: usize,
    /// Block by block, zigzag order, scaled by 8.
    data: Vec<i16>,
}

/// Forward-transformed picture, before quantization.
#[derive(Default)]
pub struct Coefficients {
    width: u32,
    height: u32,
    sampling: Option<Sampling>,
    comps: Vec<CoefComp>,
}

/// What [`Coefficients::write`] puts in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteSettings {
    /// 1..=100, the Independent JPEG Group scale.
    pub quality: u8,
    /// MCUs between restart markers. Zero writes none.
    pub restart_interval: u16,
    /// Write an APP0 JFIF segment.
    pub jfif: bool,
}

struct Coder {
    /// Natural order, as DQT stores them after the zigzag.
    quant: [[u16; 64]; 2],
    /// Zigzag order.
    quant_zz: [[u16; 64]; 2],
    /// Reciprocals of `quant_zz * 8`, see [`divide`].
    recip: [[u64; 64]; 2],
    dc: [HuffEncode; 2],
    ac: [HuffEncode; 2],
}

impl Coder {
    fn new(quality: u8) -> Result<Self, CodecError> {
        if !(1..=100).contains(&quality) {
            return Err(CodecError::InvalidApi {
                what: "jpeg quality",
            });
        }
        let scale = quality_scale(quality);
        let quant = [
            scale_quant(&LUMINANCE_QUANT, scale),
            scale_quant(&CHROMINANCE_QUANT, scale),
        ];
        let quant_zz = quant.map(|table| ZIGZAG.map(|natural| table[natural]));
        let recip =
            quant_zz.map(|table| table.map(|step| (1u64 << 32).div_ceil(u64::from(step) << 3)));
        Ok(Self {
            quant,
            quant_zz,
            recip,
            dc: [
                encode_table(&DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS)?,
                encode_table(&DC_CHROMA_COUNTS, &DC_CHROMA_SYMBOLS)?,
            ],
            ac: [
                encode_table(&AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS)?,
                encode_table(&AC_CHROMA_COUNTS, &AC_CHROMA_SYMBOLS)?,
            ],
        })
    }
}

/// `round(|value| / (step * 8))` with libjpeg's rounding, by reciprocal.
/// Exact while `numerator * divisor < 2^32`, which holds for 8-bit input:
/// the numerator is under 2^15 and the divisor at most 2040.
#[inline]
pub(crate) fn divide(value: i32, step: u16, recip: u64) -> i32 {
    let magnitude = u64::from(value.unsigned_abs() + (u32::from(step) << 2));
    let quotient = i32::try_from((magnitude * recip) >> 32).unwrap_or(i32::MAX);
    if value < 0 { -quotient } else { quotient }
}

impl Coefficients {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Transform `source`, reusing this value's buffers.
    pub fn transform(&mut self, source: &Source<'_>) -> Result<(), CodecError> {
        if !dimensions_ok(source.width, source.height) {
            return Err(CodecError::Limit { what: "jpeg size" });
        }
        let sampling = source.sampling;
        let (max_h, max_v) = sampling.luma_factors();
        let mcu_cols =
            usize::try_from(source.width.div_ceil(u32::from(max_h) * 8)).map_err(|_| invalid())?;
        let mcu_rows =
            usize::try_from(source.height.div_ceil(u32::from(max_v) * 8)).map_err(|_| invalid())?;
        let (cw, ch) = sampling.chroma_size(source.width, source.height);
        let planes = [
            (source.y, source.width, source.height, max_h, max_v),
            (source.cb, cw, ch, 1, 1),
            (source.cr, cw, ch, 1, 1),
        ];
        self.comps.truncate(sampling.components());
        for (index, (plane, width, height, h, v)) in
            planes.into_iter().take(sampling.components()).enumerate()
        {
            let width = usize::try_from(width).map_err(|_| invalid())?;
            let height = usize::try_from(height).map_err(|_| invalid())?;
            if plane.stride < width || plane.samples.len() < (height - 1) * plane.stride + width {
                return Err(invalid());
            }
            let blocks_w = mcu_cols * usize::from(h);
            let blocks_h = mcu_rows * usize::from(v);
            if self.comps.len() <= index {
                self.comps.push(CoefComp {
                    h,
                    v,
                    blocks_w: 0,
                    blocks_h: 0,
                    data: Vec::new(),
                });
            }
            let comp = &mut self.comps[index];
            comp.h = h;
            comp.v = v;
            comp.blocks_w = blocks_w;
            comp.blocks_h = blocks_h;
            resize(&mut comp.data, blocks_w * blocks_h * 64)?;
            let mut block = [0i32; 64];
            for by in 0..blocks_h {
                for bx in 0..blocks_w {
                    for row in 0..8 {
                        let y = (by * 8 + row).min(height - 1);
                        let line = &plane.samples[y * plane.stride..y * plane.stride + width];
                        for col in 0..8 {
                            let x = (bx * 8 + col).min(width - 1);
                            block[row * 8 + col] = i32::from(line[x]) - 128;
                        }
                    }
                    fdct_block(&mut block);
                    let start = (by * blocks_w + bx) * 64;
                    for (slot, natural) in comp.data[start..start + 64].iter_mut().zip(ZIGZAG) {
                        *slot = i16::try_from(block[natural]).unwrap_or(0);
                    }
                }
            }
        }
        self.width = source.width;
        self.height = source.height;
        self.sampling = Some(sampling);
        Ok(())
    }

    /// Entropy-coded bits of the scan at `quality`, without stuffing,
    /// restart markers, or headers.
    pub fn scan_bits(&self, quality: u8) -> Result<u64, CodecError> {
        let coder = Coder::new(quality)?;
        let mut sink = Counter { bits: 0 };
        self.scan(&coder, 0, &mut sink)?;
        Ok(sink.bits)
    }

    /// Bytes of everything but the scan: markers, tables, and headers.
    pub fn header_len(&self, settings: &WriteSettings) -> Result<usize, CodecError> {
        let mut out = Vec::new();
        let coder = Coder::new(settings.quality)?;
        self.headers(&coder, settings, &mut out)?;
        Ok(out.len() + 2)
    }

    /// Append one complete JPEG image to `out`.
    pub fn write(&self, settings: &WriteSettings, out: &mut Vec<u8>) -> Result<(), CodecError> {
        let coder = Coder::new(settings.quality)?;
        self.headers(&coder, settings, out)?;
        let mut sink = Writer {
            out,
            acc: 0,
            nbits: 0,
        };
        self.scan(&coder, settings.restart_interval, &mut sink)?;
        sink.align();
        out.extend_from_slice(&[0xff, 0xd9]);
        Ok(())
    }

    fn headers(
        &self,
        coder: &Coder,
        settings: &WriteSettings,
        out: &mut Vec<u8>,
    ) -> Result<(), CodecError> {
        let sampling = self.sampling.ok_or_else(invalid)?;
        let color = sampling != Sampling::Gray;
        out.extend_from_slice(&[0xff, 0xd8]);
        if settings.jfif {
            segment(
                out,
                0xe0,
                &[b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
            )?;
        }
        let mut dqt = Vec::with_capacity(130);
        for (id, table) in coder
            .quant
            .iter()
            .enumerate()
            .take(if color { 2 } else { 1 })
        {
            dqt.push(u8::try_from(id).unwrap_or(0));
            dqt.extend(
                ZIGZAG
                    .iter()
                    .map(|natural| u8::try_from(table[*natural]).unwrap_or(u8::MAX)),
            );
        }
        segment(out, 0xdb, &dqt)?;
        let height = u16::try_from(self.height).map_err(|_| invalid())?;
        let width = u16::try_from(self.width).map_err(|_| invalid())?;
        let mut sof = vec![8];
        sof.extend(height.to_be_bytes());
        sof.extend(width.to_be_bytes());
        sof.push(u8::try_from(self.comps.len()).unwrap_or(1));
        for (index, comp) in self.comps.iter().enumerate() {
            let id = u8::try_from(index + 1).unwrap_or(1);
            sof.extend([id, (comp.h << 4) | comp.v, u8::from(index > 0)]);
        }
        segment(out, 0xc0, &sof)?;
        let mut dht = Vec::new();
        for class_id in 0..if color { 2 } else { 1 } {
            for (class, table) in [(0u8, &coder.dc[class_id]), (1u8, &coder.ac[class_id])] {
                dht.push((class << 4) | u8::try_from(class_id).unwrap_or(0));
                dht.extend_from_slice(&table.counts);
                dht.extend_from_slice(&table.symbols);
            }
        }
        segment(out, 0xc4, &dht)?;
        if settings.restart_interval > 0 {
            segment(out, 0xdd, &settings.restart_interval.to_be_bytes())?;
        }
        let mut sos = vec![u8::try_from(self.comps.len()).unwrap_or(1)];
        for index in 0..self.comps.len() {
            let id = u8::try_from(index + 1).unwrap_or(1);
            sos.extend([id, if index > 0 { 0x11 } else { 0x00 }]);
        }
        sos.extend([0, 63, 0]);
        segment(out, 0xda, &sos)
    }

    fn scan(&self, coder: &Coder, restart: u16, sink: &mut impl Sink) -> Result<(), CodecError> {
        let sampling = self.sampling.ok_or_else(invalid)?;
        let (max_h, max_v) = sampling.luma_factors();
        let mcu_cols =
            usize::try_from(self.width.div_ceil(u32::from(max_h) * 8)).map_err(|_| invalid())?;
        let mcu_rows =
            usize::try_from(self.height.div_ceil(u32::from(max_v) * 8)).map_err(|_| invalid())?;
        let mut pred = [0i32; 3];
        let mut left = restart;
        let mut next_rst = 0u8;
        let mut zig = [0i32; 64];
        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                if restart > 0 {
                    if left == 0 {
                        sink.restart(next_rst);
                        next_rst = (next_rst + 1) & 7;
                        pred = [0; 3];
                        left = restart;
                    }
                    left -= 1;
                }
                for (index, comp) in self.comps.iter().enumerate() {
                    let table = usize::from(index > 0);
                    let (h, v) = (usize::from(comp.h), usize::from(comp.v));
                    for local_y in 0..v {
                        for local_x in 0..h {
                            let bx = mcu_x * h + local_x;
                            let by = mcu_y * v + local_y;
                            let start = (by * comp.blocks_w + bx) * 64;
                            let Some(block) = comp.data.get(start..start + 64) else {
                                return Err(invalid());
                            };
                            let steps = &coder.quant_zz[table];
                            let recips = &coder.recip[table];
                            for k in 0..64 {
                                zig[k] = divide(i32::from(block[k]), steps[k], recips[k]);
                            }
                            encode_block(
                                sink,
                                &zig,
                                &mut pred[index],
                                &coder.dc[table],
                                &coder.ac[table],
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn segment(out: &mut Vec<u8>, marker: u8, body: &[u8]) -> Result<(), CodecError> {
    let len = u16::try_from(body.len() + 2).map_err(|_| invalid())?;
    out.extend_from_slice(&[0xff, marker]);
    out.extend(len.to_be_bytes());
    out.extend_from_slice(body);
    Ok(())
}

trait Sink {
    fn put(&mut self, bits: u32, len: u32);
    fn restart(&mut self, index: u8);
}

struct Counter {
    bits: u64,
}

impl Sink for Counter {
    #[inline]
    fn put(&mut self, _bits: u32, len: u32) {
        self.bits += u64::from(len);
    }

    fn restart(&mut self, _index: u8) {
        self.bits = self.bits.next_multiple_of(8) + 16;
    }
}

struct Writer<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl Writer<'_> {
    /// Pad the last byte with ones.
    fn align(&mut self) {
        let pad = (8 - self.nbits % 8) % 8;
        if pad > 0 {
            self.put((1 << pad) - 1, pad);
        }
        while self.nbits >= 8 {
            self.emit_byte();
        }
    }

    #[inline]
    fn emit_byte(&mut self) {
        self.nbits -= 8;
        let byte = u8::try_from((self.acc >> self.nbits) & 0xff).unwrap_or(0);
        self.out.push(byte);
        if byte == 0xff {
            self.out.push(0);
        }
    }
}

impl Sink for Writer<'_> {
    #[inline]
    fn put(&mut self, bits: u32, len: u32) {
        self.acc = (self.acc << len) | u64::from(bits);
        self.nbits += len;
        if self.nbits >= 32 {
            let word = u32::try_from((self.acc >> (self.nbits - 32)) & 0xffff_ffff).unwrap_or(0);
            if !word.to_be_bytes().contains(&0xff) {
                self.out.extend_from_slice(&word.to_be_bytes());
                self.nbits -= 32;
            } else {
                while self.nbits >= 8 {
                    self.emit_byte();
                }
            }
            self.acc &= (1u64 << self.nbits) - 1;
        }
    }

    fn restart(&mut self, index: u8) {
        self.align();
        self.out.extend_from_slice(&[0xff, 0xd0 + index]);
    }
}

#[inline]
fn category(value: i32) -> u32 {
    32 - value.unsigned_abs().leading_zeros()
}

#[inline]
fn magnitude_bits(value: i32, size: u32) -> u32 {
    if value < 0 {
        (value - 1).cast_unsigned() & ((1 << size) - 1)
    } else {
        value.unsigned_abs()
    }
}

fn encode_block(
    sink: &mut impl Sink,
    zig: &[i32; 64],
    pred: &mut i32,
    dc: &HuffEncode,
    ac: &HuffEncode,
) -> Result<(), CodecError> {
    let diff = zig[0] - *pred;
    *pred = zig[0];
    let size = category(diff);
    if size > 11 {
        return Err(CodecError::Limit {
            what: "jpeg coefficient",
        });
    }
    let symbol = usize::try_from(size).unwrap_or(0);
    put_code(sink, dc, symbol, magnitude_bits(diff, size), size)?;
    let mut run = 0usize;
    for value in &zig[1..] {
        if *value == 0 {
            run += 1;
            continue;
        }
        while run >= 16 {
            put_code(sink, ac, 0xf0, 0, 0)?;
            run -= 16;
        }
        let size = category(*value);
        if size > 10 {
            return Err(CodecError::Limit {
                what: "jpeg coefficient",
            });
        }
        let symbol = (run << 4) | usize::try_from(size).unwrap_or(0);
        put_code(sink, ac, symbol, magnitude_bits(*value, size), size)?;
        run = 0;
    }
    if run > 0 {
        put_code(sink, ac, 0x00, 0, 0)?;
    }
    Ok(())
}

#[inline]
fn put_code(
    sink: &mut impl Sink,
    table: &HuffEncode,
    symbol: usize,
    extra: u32,
    size: u32,
) -> Result<(), CodecError> {
    let len = u32::from(table.len[symbol]);
    if len == 0 {
        return Err(CodecError::Limit {
            what: "jpeg coefficient",
        });
    }
    sink.put((u32::from(table.code[symbol]) << size) | extra, len + size);
    Ok(())
}
