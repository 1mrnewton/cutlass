//! Frame decoding: slices in parallel, each into its own band of rows.
//!
//! A job decodes one macroblock row of every picture into its own working
//! rows, a few tens of kilobytes that stay in cache, and then writes its
//! band of the output from them: the wide samples, or the 8-bit 4:2:0 the
//! app shows. Nothing is left for one thread to do after the jobs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use cutlass_decode::{CodecError, PixelFormat, Plane, VideoFrame};
use cutlass_pool::Pool;

use crate::bits::{BitReader, codeword};
use crate::color;
use crate::convert::{Narrow, chroma_row_422, chroma_row_444, luma_row};
use crate::dct::{basis, idct_columns, transposed};
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
/// Working sets and returned planes kept for the next frame.
const MAX_KEPT: usize = 32;

/// A decoded frame and the header it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    pub header: FrameHeader,
    pub frame: VideoFrame,
}

/// A frame as tight 8-bit 4:2:0: Y, then U, then V.
pub(crate) struct Picture420 {
    pub header: FrameHeader,
    /// The even size 4:2:0 needs; an odd edge grows by one repeated sample.
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

/// Memory a decoder keeps between frames: each job's working rows, and
/// planes the caller handed back.
#[derive(Default)]
pub(crate) struct Reuse {
    scratch: Mutex<Vec<Scratch>>,
    spare: Mutex<Vec<Vec<u8>>>,
}

impl Reuse {
    fn scratch(&self) -> Scratch {
        lock(&self.scratch).pop().unwrap_or_default()
    }

    fn keep_scratch(&self, scratch: Scratch) {
        let mut kept = lock(&self.scratch);
        if kept.len() < MAX_KEPT {
            kept.push(scratch);
        }
    }

    /// A buffer of `len` bytes. One handed back earlier is not cleared:
    /// a frame that decodes writes every sample of every plane.
    fn plane(&self, len: usize) -> Vec<u8> {
        let mut spare = lock(&self.spare);
        match spare.iter().position(|buffer| buffer.len() == len) {
            Some(at) => spare.swap_remove(at),
            None => {
                drop(spare);
                vec![0u8; len]
            }
        }
    }

    /// Takes a buffer back for a later frame of the same size.
    pub(crate) fn keep_plane(&self, buffer: Vec<u8>) {
        let mut spare = lock(&self.spare);
        if spare.len() >= MAX_KEPT {
            spare.remove(0);
        }
        spare.push(buffer);
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Decodes one frame to `depth`-bit samples (10 or 12), slices spread over `pool`.
pub fn decode_frame(
    data: &[u8],
    depth: u8,
    pool: &Pool,
    cancel: &AtomicBool,
) -> Result<DecodedFrame, CodecError> {
    decode_wide(data, depth, pool, cancel, &Reuse::default())
}

/// The headers of one frame and where its rows go.
struct Parsed {
    header: FrameHeader,
    pictures: Vec<Picture>,
    frame_size: usize,
    width: usize,
    height: usize,
    chroma_width: usize,
    /// Frame rows one job writes: a macroblock row of every picture.
    band_rows: usize,
    alpha_bytes: usize,
}

impl Parsed {
    fn new(data: &[u8], depth: u8) -> Result<Self, CodecError> {
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
        Ok(Self {
            frame_size: header.frame_size,
            width,
            height: header.height as usize,
            chroma_width: match header.chroma {
                Chroma::Yuv422 => width.div_ceil(2),
                Chroma::Yuv444 => width,
            },
            band_rows: if header.interlace == Interlace::Progressive {
                16
            } else {
                32
            },
            alpha_bytes: if header.alpha == AlphaBits::Sixteen {
                2
            } else {
                1
            },
            header,
            pictures,
        })
    }

    fn bands(&self) -> usize {
        self.height.div_ceil(self.band_rows)
    }

    fn context<'a>(&'a self, data: &'a [u8], depth: u8) -> Context<'a> {
        let scan = if self.header.interlace == Interlace::Progressive {
            &PROGRESSIVE_SCAN
        } else {
            &INTERLACED_SCAN
        };
        let padded = self.header.width_in_mb() as usize * 16;
        Context {
            frame: &data[..self.frame_size],
            header: &self.header,
            pictures: &self.pictures,
            depth,
            width: self.width,
            height: self.height,
            chroma_width: self.chroma_width,
            band_rows: self.band_rows,
            alpha_bytes: self.alpha_bytes,
            stride: padded,
            chroma_stride: match self.header.chroma {
                Chroma::Yuv422 => padded / 2,
                Chroma::Yuv444 => padded,
            },
            scan: scan.map(transposed),
            basis: basis(),
        }
    }
}

/// Decodes one frame to wide planes, with alpha when the frame has it.
pub(crate) fn decode_wide(
    data: &[u8],
    depth: u8,
    pool: &Pool,
    cancel: &AtomicBool,
    reuse: &Reuse,
) -> Result<DecodedFrame, CodecError> {
    let parsed = Parsed::new(data, depth)?;
    let (width, height) = (parsed.width, parsed.height);
    let (chroma_width, band_rows) = (parsed.chroma_width, parsed.band_rows);
    let alpha_bytes = parsed.alpha_bytes;
    let mut y = reuse.plane(width * 2 * height);
    let mut u = reuse.plane(chroma_width * 2 * height);
    let mut v = reuse.plane(chroma_width * 2 * height);
    let mut a =
        (parsed.header.alpha != AlphaBits::None).then(|| reuse.plane(width * alpha_bytes * height));
    let mut alpha_chunks: Vec<Option<&mut [u8]>> = match a.as_mut() {
        Some(buf) => buf
            .chunks_mut(width * alpha_bytes * band_rows)
            .map(Some)
            .collect(),
        None => (0..parsed.bands()).map(|_| None).collect(),
    };
    let bands: Vec<Mutex<Out<'_>>> = y
        .chunks_mut(width * 2 * band_rows)
        .zip(u.chunks_mut(chroma_width * 2 * band_rows))
        .zip(v.chunks_mut(chroma_width * 2 * band_rows))
        .zip(alpha_chunks.drain(..))
        .map(|(((y, u), v), a)| Mutex::new(Out::Wide { y, u, v, a }))
        .collect();
    let ctx = parsed.context(data, depth);
    let outcome = run_bands(&ctx, &bands, pool, cancel, reuse);
    drop(bands);
    if let Err(error) = outcome {
        for buffer in [Some(y), Some(u), Some(v), a].into_iter().flatten() {
            reuse.keep_plane(buffer);
        }
        return Err(error);
    }
    let header = parsed.header;
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

/// Decodes one frame straight to 8-bit 4:2:0. Alpha is checked and dropped.
pub(crate) fn decode_yuv420(
    data: &[u8],
    depth: u8,
    pool: &Pool,
    cancel: &AtomicBool,
    reuse: &Reuse,
) -> Result<Picture420, CodecError> {
    let parsed = Parsed::new(data, depth)?;
    let out_w = parsed.width.div_ceil(2) * 2;
    let out_h = parsed.height.div_ceil(2) * 2;
    let band_rows = parsed.band_rows;
    let mut bytes = vec![0u8; out_w * out_h * 3 / 2];
    let (y, chroma) = bytes.split_at_mut(out_w * out_h);
    let (u, v) = chroma.split_at_mut(out_w / 2 * (out_h / 2));
    let chroma_band = out_w / 2 * (band_rows / 2);
    let bands: Vec<Mutex<Out<'_>>> = y
        .chunks_mut(out_w * band_rows)
        .zip(u.chunks_mut(chroma_band))
        .zip(v.chunks_mut(chroma_band))
        .map(|((y, u), v)| Mutex::new(Out::Narrow { y, u, v }))
        .collect();
    let ctx = parsed.context(data, depth);
    let outcome = run_bands(&ctx, &bands, pool, cancel, reuse);
    drop(bands);
    outcome?;
    Ok(Picture420 {
        header: parsed.header,
        width: out_w as u32,
        height: out_h as u32,
        bytes,
    })
}

/// Runs every band's job on `pool`. The first band that failed, in picture
/// order, gives the error.
fn run_bands(
    ctx: &Context<'_>,
    bands: &[Mutex<Out<'_>>],
    pool: &Pool,
    cancel: &AtomicBool,
    reuse: &Reuse,
) -> Result<(), CodecError> {
    let results = pool.slices(bands.len(), |index| {
        if cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let Some(cell) = bands.get(index) else {
            return Err(CodecError::InvalidApi { what: "band index" });
        };
        let mut out = cell
            .lock()
            .map_err(|_| CodecError::InvalidApi { what: "band lock" })?;
        ctx.job(index, &mut out, reuse)
    });
    results.into_iter().collect()
}

/// The rows of the output one job writes.
enum Out<'a> {
    /// 16-bit little-endian planes, and alpha at its coded width.
    Wide {
        y: &'a mut [u8],
        u: &'a mut [u8],
        v: &'a mut [u8],
        a: Option<&'a mut [u8]>,
    },
    /// 8-bit 4:2:0 at the even output size.
    Narrow {
        y: &'a mut [u8],
        u: &'a mut [u8],
        v: &'a mut [u8],
    },
}

struct Context<'a> {
    frame: &'a [u8],
    header: &'a FrameHeader,
    pictures: &'a [Picture],
    depth: u8,
    width: usize,
    height: usize,
    chroma_width: usize,
    band_rows: usize,
    alpha_bytes: usize,
    /// Samples in a working row: the width in whole macroblocks.
    stride: usize,
    chroma_stride: usize,
    /// The scan, giving each coefficient's place in the transform's layout.
    scan: [u8; 64],
    basis: &'static [[f32; 8]; 8],
}

/// Per-job working memory.
struct Scratch {
    coefs: Box<[[f32; 64]; MAX_BLOCKS]>,
    alpha: Vec<u16>,
    /// The band's samples, every macroblock whole: `band_rows` rows of
    /// `stride` (`chroma_stride`) samples.
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            coefs: Box::new([[0.0; 64]; MAX_BLOCKS]),
            alpha: Vec::new(),
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
        }
    }
}

/// How samples leave the transform: offset, floor, clamp to the legal range.
#[derive(Clone, Copy)]
struct Store {
    bias: f32,
    low: f32,
    high: f32,
}

/// Where a block lands in a working plane: column `x`, from row `row`
/// every `step` rows.
#[derive(Clone, Copy)]
struct Place {
    stride: usize,
    x: usize,
    row: usize,
    step: usize,
}

fn be16(data: &[u8], at: usize) -> usize {
    match data.get(at..at + 2) {
        Some(bytes) => usize::from(u16::from_be_bytes([bytes[0], bytes[1]])),
        None => 0,
    }
}

impl Context<'_> {
    /// One band, compiled for the widest vectors the CPU has.
    fn job(&self, index: usize, out: &mut Out<'_>, reuse: &Reuse) -> Result<(), CodecError> {
        let mut scratch = reuse.scratch();
        let result = cutlass_simd::wide::wide(
            #[inline(always)]
            || self.band(index, out, &mut scratch),
        );
        reuse.keep_scratch(scratch);
        result
    }

    #[inline(always)]
    fn band(
        &self,
        index: usize,
        out: &mut Out<'_>,
        scratch: &mut Scratch,
    ) -> Result<(), CodecError> {
        scratch.y.resize(self.stride * self.band_rows, 0);
        scratch.u.resize(self.chroma_stride * self.band_rows, 0);
        scratch.v.resize(self.chroma_stride * self.band_rows, 0);
        let top = index * self.band_rows;
        // Frame rows in this band.
        let rows = self.height.saturating_sub(top).min(self.band_rows);
        let row = index as u32;
        for picture in self.pictures {
            if row >= picture.height_in_mb {
                continue;
            }
            let start = index * picture.slices_per_row;
            let Some(slices) = picture.slices.get(start..start + picture.slices_per_row) else {
                return Err(CodecError::InvalidApi { what: "slice row" });
            };
            for slice in slices {
                let alpha = match out {
                    Out::Wide { a, .. } => a.as_deref_mut(),
                    Out::Narrow { .. } => None,
                };
                self.slice(slice, picture, row, rows, scratch, alpha)?;
            }
        }
        match out {
            Out::Wide { y, u, v, .. } => {
                copy_rows(&scratch.y, self.stride, self.width, y);
                copy_rows(&scratch.u, self.chroma_stride, self.chroma_width, u);
                copy_rows(&scratch.v, self.chroma_stride, self.chroma_width, v);
            }
            Out::Narrow { y, u, v } => self.narrow(rows, scratch, y, u, v),
        }
        Ok(())
    }

    /// Writes the band as 8-bit 4:2:0. Rows and columns past the picture
    /// repeat its last.
    #[inline(always)]
    fn narrow(&self, rows: usize, scratch: &Scratch, y: &mut [u8], u: &mut [u8], v: &mut [u8]) {
        if rows == 0 {
            return;
        }
        let narrow = Narrow::new(self.depth);
        let out_w = self.width.div_ceil(2) * 2;
        let last = rows - 1;
        for (r, dst) in y.chunks_exact_mut(out_w).enumerate() {
            let at = r.min(last) * self.stride;
            if let Some(src) = scratch.y.get(at..at + self.width) {
                luma_row(src, dst, narrow);
            }
        }
        for (plane, out) in [(&scratch.u, u), (&scratch.v, v)] {
            for (r, dst) in out.chunks_exact_mut(out_w / 2).enumerate() {
                let a = (2 * r).min(last) * self.chroma_stride;
                let b = (2 * r + 1).min(last) * self.chroma_stride;
                let (Some(a), Some(b)) = (
                    plane.get(a..a + self.chroma_width),
                    plane.get(b..b + self.chroma_width),
                ) else {
                    continue;
                };
                match self.header.chroma {
                    Chroma::Yuv422 => chroma_row_422(a, b, dst, narrow),
                    Chroma::Yuv444 => chroma_row_444(a, b, dst, narrow),
                }
            }
        }
    }

    #[inline(always)]
    fn slice(
        &self,
        slice: &SliceRef,
        picture: &Picture,
        row: u32,
        rows: usize,
        scratch: &mut Scratch,
        alpha_out: Option<&mut [u8]>,
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
        // Eighths of F become output-scale samples: 2^(b - 9) / 8.
        let scale = f32::from(1u16 << (self.depth - 9)) / 8.0;
        let store = Store {
            bias: f32::from(1u16 << (self.depth - 1)) + 0.5,
            low: 4.0,
            high: ((1i32 << self.depth) - 5) as f32,
        };

        let luma_blocks = 4 * mbs;
        let weights = weights_of(&self.header.luma_matrix, qscale);
        let ac = component(
            &data[y_at..cb_at],
            base + y_at as u64 * 8,
            luma_blocks,
            &self.scan,
            &weights,
            scale,
            &mut scratch.coefs,
        )?;
        let x0 = slice.mb_x as usize * 16;
        for block in 0..luma_blocks {
            let mb = block / 4;
            let sub = block % 4;
            let x = x0 + mb * 16 + (sub & 1) * 8;
            let y = (sub >> 1) * 8;
            let at = Place {
                stride: self.stride,
                x,
                row: first + step * y,
                step,
            };
            self.put(
                &scratch.coefs[block],
                ac & (1 << block) != 0,
                store,
                &mut scratch.y,
                at,
            );
        }

        let weights = weights_of(&self.header.chroma_matrix, qscale);
        let (chroma_blocks, chroma_x0) = match self.header.chroma {
            Chroma::Yuv422 => (2 * mbs, slice.mb_x as usize * 8),
            Chroma::Yuv444 => (4 * mbs, slice.mb_x as usize * 16),
        };
        for (plane, at, end) in [(0, cb_at, cr_at), (1, cr_at, a_at)] {
            let ac = component(
                &data[at..end],
                base + at as u64 * 8,
                chroma_blocks,
                &self.scan,
                &weights,
                scale,
                &mut scratch.coefs,
            )?;
            let buf = if plane == 0 {
                &mut scratch.u
            } else {
                &mut scratch.v
            };
            for block in 0..chroma_blocks {
                let (x, y) = match self.header.chroma {
                    Chroma::Yuv422 => (chroma_x0 + (block / 2) * 8, (block % 2) * 8),
                    Chroma::Yuv444 => {
                        let sub = block % 4;
                        (chroma_x0 + (block / 4) * 16 + (sub >> 1) * 8, (sub & 1) * 8)
                    }
                };
                let at = Place {
                    stride: self.chroma_stride,
                    x,
                    row: first + step * y,
                    step,
                };
                self.put(
                    &scratch.coefs[block],
                    ac & (1 << block) != 0,
                    store,
                    buf,
                    at,
                );
            }
        }

        if alpha {
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
            let Some(out) = alpha_out else {
                return Ok(());
            };
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

    /// Transforms one block and stores it at `at`.
    #[inline(always)]
    fn put(&self, coefs: &[f32; 64], has_ac: bool, store: Store, plane: &mut [u16], at: Place) {
        if !has_ac {
            // A flat block: one sample, repeated.
            let flat = store.samples(&[coefs[0] / 8.0; 8]);
            for yy in 0..8 {
                let start = (at.row + at.step * yy) * at.stride + at.x;
                if let Some(dest) = plane.get_mut(start..start + 8) {
                    dest.copy_from_slice(&flat);
                }
            }
            return;
        }
        let block = idct_columns(coefs, self.basis);
        for (yy, values) in block.iter().enumerate() {
            let start = (at.row + at.step * yy) * at.stride + at.x;
            if let Some(dest) = plane.get_mut(start..start + 8) {
                dest.copy_from_slice(&store.samples(values));
            }
        }
    }
}

/// Two to the 23rd: from here up a single-precision value is a whole number.
const WHOLE: f32 = 8_388_608.0;

impl Store {
    /// Eight transform outputs as samples: `floor(value + bias)`, clamped.
    ///
    /// The clamp comes first, so every value is positive and small.
    /// Adding [`WHOLE`] then rounds it to the nearest whole number, which
    /// sits in the low bits of the sum, and one comes off where that
    /// rounded up. This is the floor without a float-to-integer cast per
    /// sample, which the compiler will not do eight at a time.
    #[inline(always)]
    fn samples(self, values: &[f32; 8]) -> [u16; 8] {
        let mut out = [0u16; 8];
        for (sample, &value) in out.iter_mut().zip(values) {
            let biased = value + self.bias;
            let raised = if biased > self.low { biased } else { self.low };
            let kept = if raised < self.high {
                raised
            } else {
                self.high
            };
            let lifted = kept + WHOLE;
            let nearest = lifted - WHOLE;
            let whole = lifted.to_bits() & 0x007f_ffff;
            *sample = (whole - u32::from(nearest > kept)) as u16;
        }
        out
    }
}

/// Copies `width` samples of each working row into tight little-endian rows.
#[inline(always)]
fn copy_rows(samples: &[u16], stride: usize, width: usize, out: &mut [u8]) {
    if stride == 0 || width == 0 {
        return;
    }
    for (row, dest) in samples
        .chunks_exact(stride)
        .zip(out.chunks_exact_mut(width * 2))
    {
        for (pair, &value) in dest.as_chunks_mut::<2>().0.iter_mut().zip(row) {
            *pair = value.to_le_bytes();
        }
    }
}

/// Each coefficient's weight times the slice's scale, in the transform's layout.
#[inline(always)]
fn weights_of(matrix: &[u8; 64], qscale: i32) -> [i32; 64] {
    let mut out = [0i32; 64];
    for (raster, &weight) in matrix.iter().enumerate() {
        out[usize::from(transposed(raster as u8))] = i32::from(weight) * qscale;
    }
    out
}

/// A dequantized coefficient at output scale.
#[inline(always)]
fn dequant(coef: i32, weight: i32, scale: f32) -> f32 {
    (i64::from(coef) * i64::from(weight)).clamp(-MAX_EIGHTHS, MAX_EIGHTHS) as f32 * scale
}

/// Decodes one component's DC and AC coefficients into `coefs[..blocks]`,
/// dequantized, in the transform's layout. Returns a mask of the blocks
/// that have an AC coefficient.
#[inline(always)]
fn component(
    data: &[u8],
    base: u64,
    blocks: usize,
    scan: &[u8; 64],
    weights: &[i32; 64],
    scale: f32,
    coefs: &mut [[f32; 64]; MAX_BLOCKS],
) -> Result<u32, CodecError> {
    let mut bits = BitReader::new(data, base);
    for block in coefs.iter_mut().take(blocks) {
        *block = [0.0; 64];
    }
    // DC: the first coefficient, then signed differences.
    let code = bits.code(FIRST_DC_CODEBOOK)?;
    let mut dc = to_signed(code);
    let limit = MAX_LEVEL as i32;
    coefs[0][0] = dequant(dc.clamp(-limit, limit), weights[0], scale);
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
        block[0] = dequant(dc.clamp(-limit, limit), weights[0], scale);
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
    loop {
        // One refill covers the run and, unless the run was long, the level
        // and its sign.
        bits.refill();
        if bits.ended() {
            break;
        }
        let (value, len) = codeword(bits.top32(), run_codebook(run));
        if len > 31 {
            return Err(bits.malformed("codeword"));
        }
        bits.skip(len);
        run = value;
        pos += run as usize + 1;
        if pos >= limit {
            return Err(bits.malformed("coefficient run past the block"));
        }
        bits.ensure32();
        let buf = bits.top32();
        let (value, len) = codeword(buf, level_codebook(level));
        if len > 31 {
            return Err(bits.malformed("codeword"));
        }
        level = value.saturating_add(1);
        // The sign is the bit after the codeword, inside the same 32.
        let sign = (buf >> (31 - len)) & 1;
        bits.skip(len + 1);
        let block = pos & mask;
        let index = usize::from(scan[pos >> log2]);
        let magnitude = level.min(MAX_LEVEL) as i32;
        let signed = if sign == 1 { -magnitude } else { magnitude };
        coefs[block][index] = dequant(signed, weights[index], scale);
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

/// The alpha unpacker, for the encoder's tests.
#[cfg(test)]
pub(crate) fn unpack_alpha_for_tests(data: &[u8], sixteen: bool, count: usize, out: &mut Vec<u16>) {
    if let Err(err) = unpack_alpha(data, 0, sixteen, count, out) {
        panic!("alpha: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::{Store, to_signed, unpack_alpha};
    use crate::bits::BitWriter;

    #[test]
    fn stored_samples_are_the_clamped_floor() {
        for depth in [10u8, 12] {
            let store = Store {
                bias: f32::from(1u16 << (depth - 1)) + 0.5,
                low: 4.0,
                high: ((1i32 << depth) - 5) as f32,
            };
            let want = |value: f32| {
                ((value + store.bias).floor() as i32).clamp(4, (1 << depth) - 5) as u16
            };
            let mut values = vec![
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -0.0,
                1e30,
                -1e30,
            ];
            // Every whole and half step across the range, and its neighbours.
            for k in -5000i32..5000 {
                for part in [-0.5f32, -0.25, 0.0, 0.25, 0.5] {
                    let value = k as f32 + part;
                    values.push(value);
                    for off in 1..3 {
                        values.push(f32::from_bits(value.to_bits().wrapping_add(off)));
                        values.push(f32::from_bits(value.to_bits().wrapping_sub(off)));
                    }
                }
            }
            let mut state = 9u32;
            for _ in 0..200_000 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                values.push((state as f32 / 4_294_967_296.0 - 0.5) * 12_000.0);
            }
            for group in values.chunks(8) {
                let mut eight = [0f32; 8];
                eight[..group.len()].copy_from_slice(group);
                let got = store.samples(&eight);
                for (value, sample) in eight.iter().zip(got) {
                    assert_eq!(sample, want(*value), "{value} at {depth} bits");
                }
            }
        }
    }

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
