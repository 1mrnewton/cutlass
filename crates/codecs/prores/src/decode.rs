//! Frame decoding: slices in parallel, each into its own band of rows.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{CodecError, PixelFormat, Plane, VideoFrame};
use cutlass_pool::Pool;

use crate::bits::BitReader;
use crate::color;
use crate::dct::idct;
use crate::header::{
    AlphaBits, Chroma, FRAME_PREFIX, FrameHeader, Interlace, Picture, SliceRef, parse_frame_header,
    parse_picture,
};
use crate::tables::{
    FIRST_DC_CODEBOOK, FIRST_DC_SYMBOL, FIRST_LEVEL, FIRST_RUN, INTERLACED_SCAN, PROGRESSIVE_SCAN,
    dc_codebook, level_codebook, qscale, run_codebook,
};

/// Most blocks one component of a slice holds: 8 macroblocks of 4.
const MAX_BLOCKS: usize = 32;
/// Largest dequantized coefficient kept, in eighths: |F| up to 4096.
const MAX_EIGHTHS: i64 = 32_768;
/// Largest coefficient magnitude kept before dequantization.
const MAX_LEVEL: u32 = 1 << 20;

/// A decoded frame and the header it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    pub header: FrameHeader,
    pub frame: VideoFrame,
}

/// Decodes one frame to `depth`-bit samples (10 or 12), slices spread over `pool`.
pub fn decode_frame(
    data: &[u8],
    depth: u8,
    pool: &Pool,
    cancel: &AtomicBool,
) -> Result<DecodedFrame, CodecError> {
    if depth != 10 && depth != 12 {
        return Err(CodecError::InvalidApi {
            what: "output bit depth",
        });
    }
    let header = parse_frame_header(data)?;
    let frame = &data[..header.frame_size];
    let mut at = FRAME_PREFIX + header.header_size;
    let mut pictures = Vec::with_capacity(2);
    for index in 0..header.pictures() {
        let (picture, end) = parse_picture(frame, &header, index, at)?;
        pictures.push(picture);
        at = end;
    }
    let width = header.width as usize;
    let height = header.height as usize;
    let chroma_width = match header.chroma {
        Chroma::Yuv422 => width.div_ceil(2),
        Chroma::Yuv444 => width,
    };
    let band_rows = if header.interlace == Interlace::Progressive {
        16
    } else {
        32
    };
    let alpha_bytes = if header.alpha == AlphaBits::Sixteen {
        2
    } else {
        1
    };
    let mut y = vec![0u8; width * 2 * height];
    let mut u = vec![0u8; chroma_width * 2 * height];
    let mut v = vec![0u8; chroma_width * 2 * height];
    let mut a = (header.alpha != AlphaBits::None).then(|| vec![0u8; width * alpha_bytes * height]);
    let band_count = height.div_ceil(band_rows);
    let mut alpha_chunks: Vec<Option<&mut [u8]>> = match a.as_mut() {
        Some(buf) => buf
            .chunks_mut(width * alpha_bytes * band_rows)
            .map(Some)
            .collect(),
        None => (0..band_count).map(|_| None).collect(),
    };
    let bands: Vec<Mutex<Band<'_>>> = y
        .chunks_mut(width * 2 * band_rows)
        .zip(u.chunks_mut(chroma_width * 2 * band_rows))
        .zip(v.chunks_mut(chroma_width * 2 * band_rows))
        .zip(alpha_chunks.drain(..))
        .map(|(((y, u), v), a)| {
            Mutex::new(Band {
                rows: y.len() / (width * 2),
                y,
                u,
                v,
                a,
            })
        })
        .collect();
    let ctx = Context {
        frame,
        header: &header,
        depth,
        width,
        chroma_width,
        alpha_bytes,
        scan: if header.interlace == Interlace::Progressive {
            &PROGRESSIVE_SCAN
        } else {
            &INTERLACED_SCAN
        },
    };
    let results = pool.slices(bands.len(), |index| {
        if cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let Some(cell) = bands.get(index) else {
            return Err(CodecError::InvalidApi { what: "band index" });
        };
        let mut band = cell
            .lock()
            .map_err(|_| CodecError::InvalidApi { what: "band lock" })?;
        let mut scratch = Scratch::new();
        for picture in &pictures {
            let row = index as u32;
            if row >= picture.height_in_mb {
                continue;
            }
            let start = index * picture.slices_per_row;
            let Some(slices) = picture.slices.get(start..start + picture.slices_per_row) else {
                return Err(CodecError::InvalidApi { what: "slice row" });
            };
            for slice in slices {
                ctx.slice(slice, picture, row, &mut band, &mut scratch)?;
            }
        }
        Ok(())
    });
    drop(bands);
    for result in results {
        result?;
    }
    let format = match (header.chroma, depth) {
        (Chroma::Yuv422, 10) => PixelFormat::Yuv422p10,
        (Chroma::Yuv422, _) => PixelFormat::Yuv422p12,
        (Chroma::Yuv444, 10) => PixelFormat::Yuv444p10,
        (Chroma::Yuv444, _) => PixelFormat::Yuv444p12,
    };
    let mut out = VideoFrame::planar(
        format,
        header.width,
        header.height,
        Plane::bytes(header.width, header.height, 2, y)?,
        Plane::bytes(chroma_width as u32, header.height, 2, u)?,
        Plane::bytes(chroma_width as u32, header.height, 2, v)?,
    )?;
    out.range = cutlass_decode::ColorRange::Limited;
    out.primaries = color::primaries(header.primaries);
    out.transfer = color::transfer(header.transfer);
    out.matrix = color::matrix(header.matrix);
    if let Some(alpha) = a {
        out = out.with_alpha(
            header.alpha.bits(),
            Plane::bytes(header.width, header.height, alpha_bytes, alpha)?,
        )?;
    }
    Ok(DecodedFrame { header, frame: out })
}

/// The rows of every plane one job writes.
struct Band<'a> {
    rows: usize,
    y: &'a mut [u8],
    u: &'a mut [u8],
    v: &'a mut [u8],
    a: Option<&'a mut [u8]>,
}

struct Context<'a> {
    frame: &'a [u8],
    header: &'a FrameHeader,
    depth: u8,
    width: usize,
    chroma_width: usize,
    alpha_bytes: usize,
    scan: &'a [u8; 64],
}

/// Per-job working memory.
struct Scratch {
    coefs: Box<[[i32; 64]; MAX_BLOCKS]>,
    alpha: Vec<u16>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            coefs: Box::new([[0; 64]; MAX_BLOCKS]),
            alpha: Vec::new(),
        }
    }
}

/// Where a block's samples land: rows of one plane inside a band.
struct Target<'a> {
    buf: &'a mut [u8],
    /// Bytes from one frame row to the next.
    stride: usize,
    /// Samples in a frame row; columns past it are cropped.
    width: usize,
    /// Frame rows in the band; rows past it are cropped.
    rows: usize,
    /// Band row of picture row 0 of this macroblock row, and the step between picture rows.
    first: usize,
    step: usize,
}

fn be16(data: &[u8], at: usize) -> usize {
    match data.get(at..at + 2) {
        Some(bytes) => usize::from(u16::from_be_bytes([bytes[0], bytes[1]])),
        None => 0,
    }
}

impl Context<'_> {
    fn slice(
        &self,
        slice: &SliceRef,
        picture: &Picture,
        row: u32,
        band: &mut Band<'_>,
        scratch: &mut Scratch,
    ) -> Result<(), CodecError> {
        let base = slice.offset as u64 * 8;
        let malformed = |what| CodecError::Malformed { at_bit: base, what };
        let data = self
            .frame
            .get(slice.offset..slice.offset + slice.size)
            .ok_or(malformed("slice"))?;
        let header_size = usize::from(data[0] >> 3);
        let alpha = self.header.alpha != AlphaBits::None;
        let fixed = if alpha { 8 } else { 6 };
        if header_size < fixed || header_size > data.len() {
            return Err(malformed("slice header size"));
        }
        let index = data[1];
        if index == 0 || index > 224 {
            return Err(malformed("quantization index"));
        }
        let qscale = qscale(index) as i32;
        let y_size = be16(data, 2);
        let cb_size = be16(data, 4);
        let body = data.len() - header_size;
        let cr_size = if alpha {
            be16(data, 6)
        } else {
            body.checked_sub(y_size + cb_size)
                .ok_or(malformed("slice component sizes"))?
        };
        if y_size + cb_size + cr_size > body {
            return Err(malformed("slice component sizes"));
        }
        let y_at = header_size;
        let cb_at = y_at + y_size;
        let cr_at = cb_at + cb_size;
        let a_at = cr_at + cr_size;
        let mbs = slice.mbs as usize;
        let first = picture.first_row as usize;
        let step = picture.row_step as usize;
        let rows = band.rows;
        let scan = self.scan;

        let luma_blocks = 4 * mbs;
        let mut weights = [0i32; 64];
        for (w, &m) in weights.iter_mut().zip(&self.header.luma_matrix) {
            *w = i32::from(m) * qscale;
        }
        let ac = component(
            &data[y_at..cb_at],
            base + y_at as u64 * 8,
            luma_blocks,
            scan,
            &mut scratch.coefs,
        )?;
        let mut target = Target {
            buf: band.y,
            stride: self.width * 2,
            width: self.width,
            rows,
            first,
            step,
        };
        let x0 = slice.mb_x as usize * 16;
        for block in 0..luma_blocks {
            let mb = block / 4;
            let sub = block % 4;
            let x = x0 + mb * 16 + (sub & 1) * 8;
            let y = (sub >> 1) * 8;
            self.put(
                &scratch.coefs[block],
                ac & (1 << block) != 0,
                &weights,
                &mut target,
                x,
                y,
            );
        }

        for (w, &m) in weights.iter_mut().zip(&self.header.chroma_matrix) {
            *w = i32::from(m) * qscale;
        }
        let (chroma_blocks, chroma_x0) = match self.header.chroma {
            Chroma::Yuv422 => (2 * mbs, slice.mb_x as usize * 8),
            Chroma::Yuv444 => (4 * mbs, slice.mb_x as usize * 16),
        };
        for (plane, at, end) in [(0, cb_at, cr_at), (1, cr_at, a_at)] {
            let ac = component(
                &data[at..end],
                base + at as u64 * 8,
                chroma_blocks,
                scan,
                &mut scratch.coefs,
            )?;
            let buf: &mut [u8] = if plane == 0 { band.u } else { band.v };
            let mut target = Target {
                buf,
                stride: self.chroma_width * 2,
                width: self.chroma_width,
                rows,
                first,
                step,
            };
            for block in 0..chroma_blocks {
                let (x, y) = match self.header.chroma {
                    Chroma::Yuv422 => (chroma_x0 + (block / 2) * 8, (block % 2) * 8),
                    Chroma::Yuv444 => {
                        let sub = block % 4;
                        (chroma_x0 + (block / 4) * 16 + (sub >> 1) * 8, (sub & 1) * 8)
                    }
                };
                self.put(
                    &scratch.coefs[block],
                    ac & (1 << block) != 0,
                    &weights,
                    &mut target,
                    x,
                    y,
                );
            }
        }

        if let Some(out) = band.a.as_deref_mut() {
            let visible = (picture.height - row * 16).min(16) as usize;
            let columns = 16 * mbs;
            let count = columns * visible;
            let alpha_data = &data[a_at..];
            if alpha_data.is_empty() {
                return Err(CodecError::Malformed {
                    at_bit: base + a_at as u64 * 8,
                    what: "empty alpha data",
                });
            }
            let sixteen = self.header.alpha == AlphaBits::Sixteen;
            unpack_alpha(
                alpha_data,
                base + a_at as u64 * 8,
                sixteen,
                count,
                &mut scratch.alpha,
            )?;
            let x0 = slice.mb_x as usize * 16;
            let stride = self.width * self.alpha_bytes;
            for r in 0..visible {
                let band_row = first + step * r;
                if band_row >= rows {
                    break;
                }
                let line = &scratch.alpha[r * columns..(r + 1) * columns];
                let keep = columns.min(self.width.saturating_sub(x0));
                let start = band_row * stride + x0 * self.alpha_bytes;
                if sixteen {
                    let dest = &mut out[start..start + keep * 2];
                    for (pair, &value) in dest.as_chunks_mut::<2>().0.iter_mut().zip(line) {
                        *pair = value.to_le_bytes();
                    }
                } else {
                    let dest = &mut out[start..start + keep];
                    for (byte, &value) in dest.iter_mut().zip(line) {
                        *byte = value as u8;
                    }
                }
            }
        }
        Ok(())
    }

    /// Dequantizes, transforms, and stores one block.
    fn put(
        &self,
        coefs: &[i32; 64],
        has_ac: bool,
        weights: &[i32; 64],
        target: &mut Target<'_>,
        x: usize,
        y: usize,
    ) {
        // Eighths of F become output-scale samples: 2^(b - 9) / 8.
        let scale = f32::from(1u16 << (self.depth - 9)) / 8.0;
        let bias = f32::from(1u16 << (self.depth - 1)) + 0.5;
        let low = 4i32;
        let high = (1i32 << self.depth) - 5;
        let mut block = [0f32; 64];
        if has_ac {
            for k in 0..64 {
                block[k] = dequant(coefs[k], weights[k]) * scale;
            }
            idct(&mut block);
        } else {
            block = [dequant(coefs[0], weights[0]) * scale / 8.0; 64];
        }
        for yy in 0..8 {
            let band_row = target.first + target.step * (y + yy);
            if band_row >= target.rows {
                break;
            }
            if x >= target.width {
                break;
            }
            let keep = (target.width - x).min(8);
            let start = band_row * target.stride + x * 2;
            let Some(dest) = target.buf.get_mut(start..start + keep * 2) else {
                break;
            };
            for (pair, &value) in dest
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(&block[yy * 8..yy * 8 + 8])
            {
                let sample = ((value + bias).floor() as i32).clamp(low, high) as u16;
                *pair = sample.to_le_bytes();
            }
        }
    }
}

#[inline]
fn dequant(coef: i32, weight: i32) -> f32 {
    (i64::from(coef) * i64::from(weight)).clamp(-MAX_EIGHTHS, MAX_EIGHTHS) as f32
}

/// Decodes one component's DC and AC coefficients into `coefs[..blocks]`,
/// raster order. Returns a mask of the blocks that have an AC coefficient.
fn component(
    data: &[u8],
    base: u64,
    blocks: usize,
    scan: &[u8; 64],
    coefs: &mut [[i32; 64]; MAX_BLOCKS],
) -> Result<u32, CodecError> {
    let mut bits = BitReader::new(data, base);
    for block in coefs.iter_mut().take(blocks) {
        *block = [0; 64];
    }
    // DC: the first coefficient, then signed differences.
    let code = bits.code(FIRST_DC_CODEBOOK)?;
    let mut dc = to_signed(code);
    coefs[0][0] = dc.clamp(-(MAX_LEVEL as i32), MAX_LEVEL as i32);
    let mut previous = FIRST_DC_SYMBOL;
    let mut negative = false;
    for block in coefs.iter_mut().take(blocks).skip(1) {
        let code = bits.code(dc_codebook(previous))?;
        if code == 0 {
            negative = false;
        } else if code & 1 == 1 {
            negative = !negative;
        }
        let magnitude = code.div_ceil(2).min(MAX_LEVEL) as i32;
        dc = dc.saturating_add(if negative { -magnitude } else { magnitude });
        block[0] = dc.clamp(-(MAX_LEVEL as i32), MAX_LEVEL as i32);
        previous = code;
    }
    // AC: runs and levels interleaved across the blocks.
    let log2 = blocks.trailing_zeros();
    let mask = blocks - 1;
    let limit = 64usize << log2;
    let mut pos = mask;
    let mut run = FIRST_RUN;
    let mut level = FIRST_LEVEL;
    let mut has_ac = 0u32;
    while !bits.at_end_of_data() {
        run = bits.code(run_codebook(run))?;
        pos += run as usize + 1;
        if pos >= limit {
            return Err(bits.malformed("coefficient run past the block"));
        }
        level = bits.code(level_codebook(level))?.saturating_add(1);
        let sign = bits.read(1);
        let block = pos & mask;
        let index = usize::from(scan[pos >> log2]);
        let magnitude = level.min(MAX_LEVEL) as i32;
        coefs[block][index] = if sign == 1 { -magnitude } else { magnitude };
        has_ac |= 1 << block;
    }
    if bits.overrun() {
        return Err(bits.malformed("component data overrun"));
    }
    Ok(has_ac)
}

#[inline]
fn to_signed(code: u32) -> i32 {
    let half = (code >> 1) as i32;
    if code & 1 == 1 { -half - 1 } else { half }
}

/// Decodes `count` raster alpha values into `out`.
fn unpack_alpha(
    data: &[u8],
    base: u64,
    sixteen: bool,
    count: usize,
    out: &mut Vec<u16>,
) -> Result<(), CodecError> {
    out.clear();
    out.resize(count, 0);
    let mut bits = BitReader::new(data, base);
    let (full, short) = if sixteen { (16, 7) } else { (8, 4) };
    let mask: i32 = (1 << full) - 1;
    let mut alpha: i32 = mask;
    let mut index = 0;
    'outer: while index < count {
        loop {
            let difference = if bits.read(1) == 1 {
                bits.read(full) as i32
            } else {
                let code = bits.read(short) as i32;
                let magnitude = (code + 2) >> 1;
                if code & 1 == 1 { -magnitude } else { magnitude }
            };
            alpha = (alpha + difference) & mask;
            out[index] = alpha as u16;
            index += 1;
            if index >= count {
                break 'outer;
            }
            if bits.read(1) == 0 {
                break;
            }
        }
        let mut run = bits.read(4) as usize;
        if run == 0 {
            run = bits.read(11) as usize;
        }
        let run = run.min(count - index);
        out[index..index + run].fill(alpha as u16);
        index += run;
    }
    if bits.overrun() {
        return Err(bits.malformed("alpha data overrun"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{to_signed, unpack_alpha};
    use crate::bits::BitWriter;

    #[test]
    fn signed_symbols_alternate() {
        let values: Vec<i32> = (0..6).map(to_signed).collect();
        assert_eq!(values, vec![0, -1, 1, -2, 2, -3]);
    }

    #[test]
    fn alpha_runs_repeat_the_last_value() {
        // 8-bit: start at 255. "1" + 0x00 gives 255; then "0" run of 3 more;
        // then "0" + short code 3 (|d| = 2, negative) gives 253, end.
        let mut w = BitWriter::default();
        w.put(1, 1);
        w.put(8, 0);
        w.put(1, 0);
        w.put(4, 3);
        w.put(1, 0);
        w.put(4, 0b0011);
        let bytes = w.finish();
        let mut out = Vec::new();
        unpack_alpha(&bytes, 0, false, 5, &mut out).expect("alpha");
        assert_eq!(out, vec![255, 255, 255, 255, 253]);
    }

    #[test]
    fn a_long_run_is_clamped_to_the_slice() {
        let mut w = BitWriter::default();
        w.put(1, 1);
        w.put(8, 7);
        w.put(1, 0);
        w.put(4, 0);
        w.put(11, 2000);
        let bytes = w.finish();
        let mut out = Vec::new();
        unpack_alpha(&bytes, 0, false, 10, &mut out).expect("alpha");
        assert_eq!(out, vec![6; 10]);
    }
}
