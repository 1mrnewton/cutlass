//! A whole frame: slices in parallel, one quantizer for the frame tuned to
//! the profile's bit budget, then the headers.
//!
//! A pass takes each slice once: its samples are read from the caller's
//! picture, transformed, and then quantized and written at each
//! quantization index the pass asks for, while the coefficients are still
//! in the worker's cache. The first pass asks for the last frame's index
//! and the one finer, which is all a frame needs when its index has not
//! moved. A frame whose index moves costs one more pass for each index the
//! search asks about.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use cutlass_decode::CodecError;
use cutlass_pool::Pool;

use super::rate::{matrices, target_frame_bits};
use super::slice::{Work, steps};
use super::source::Picture;
use crate::dct::basis;
use crate::header::{Chroma, FRAME_HEADER_FIXED, ICPF, slice_widths};
use crate::profile::Profile;

/// Every slice is 8 macroblocks wide where the row allows.
const LOG2_SLICE: u32 = 3;
/// Where the search starts without a previous frame: HQ's usual index.
const FIRST_GUESS: u8 = 4;
/// A slice's size is a 16-bit field.
const MAX_SLICE: usize = 65_535;
/// Working sets kept between frames: one for each worker of the largest pool.
const MAX_KEPT: usize = 32;
/// Row buffers kept: the rows of the tallest frame, twice over.
const MAX_KEPT_ROWS: usize = 2048;
/// Our encoder's four-character identifier.
pub(crate) const ENCODER_ID: [u8; 4] = *b"ctls";

/// Color code points written into the frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColorCodes {
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
}

/// How one frame is to be coded.
#[derive(Clone, Copy)]
pub(crate) struct Settings {
    pub profile: Profile,
    pub color: ColorCodes,
    pub budget_scale: f64,
    /// The quantization index the last frame settled on.
    pub hint: Option<u8>,
}

/// How a frame came out: its bytes and the quantizer the rate control chose.
pub(crate) struct Encoded {
    pub bytes: Vec<u8>,
    /// Quantization index most slices used.
    pub index: u8,
}

/// Memory an encoder keeps between frames: each job's working set, and
/// the buffers the rows are coded into.
#[derive(Default)]
pub(crate) struct Kept {
    work: Mutex<Vec<Work>>,
    rows: Mutex<Vec<Vec<u8>>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// One macroblock row, coded at each index of a pass.
struct RowOut {
    bytes: Vec<u8>,
    /// Where each slice lies in `bytes`: slice by slice, one start and
    /// length for each index of the pass.
    ranges: Vec<(u32, u32)>,
    /// Bytes of alpha in each slice.
    alpha: Vec<u32>,
}

/// Every slice of the frame coded at a few quantization indices.
struct Pass {
    indices: Vec<u8>,
    rows: Vec<RowOut>,
}

struct Frame<'a> {
    picture: &'a Picture<'a>,
    widths: Vec<u32>,
    rows: usize,
    luma_matrix: &'static [u8; 64],
    chroma_matrix: &'static [u8; 64],
    pool: &'a Pool,
    kept: &'a Kept,
}

impl Frame<'_> {
    fn slices(&self) -> usize {
        self.rows * self.widths.len()
    }

    /// Codes row `row` at every index in `indices`: each slice is read and
    /// transformed once, then quantized and written once per index.
    #[inline(always)]
    fn row(&self, row: usize, indices: &[u8], steps: &[([f32; 64], [f32; 64])]) -> RowOut {
        let mut work = lock(&self.kept.work).pop().unwrap_or_default();
        let mut bytes = lock(&self.kept.rows).pop().unwrap_or_default();
        bytes.clear();
        let mut ranges = Vec::with_capacity(self.widths.len() * indices.len());
        let mut alpha = Vec::with_capacity(self.widths.len());
        let basis = basis();
        let mut mb_x = 0usize;
        for &mbs in &self.widths {
            work.prepare(self.picture, basis, row, mb_x, mbs as usize);
            alpha.push(work.alpha_len() as u32);
            for (&index, (luma, chroma)) in indices.iter().zip(steps) {
                let start = bytes.len();
                let len = work.write(index, luma, chroma, &mut bytes);
                ranges.push((start as u32, len as u32));
            }
            mb_x += mbs as usize;
        }
        let mut kept = lock(&self.kept.work);
        if kept.len() < MAX_KEPT {
            kept.push(work);
        }
        RowOut {
            bytes,
            ranges,
            alpha,
        }
    }

    /// One pass over the frame on the pool.
    fn pass(&self, indices: &[u8]) -> Pass {
        let steps: Vec<([f32; 64], [f32; 64])> = indices
            .iter()
            .map(|&index| {
                (
                    steps(self.luma_matrix, index),
                    steps(self.chroma_matrix, index),
                )
            })
            .collect();
        let rows = self.pool.slices(self.rows, |row| {
            cutlass_simd::wide::wide(
                #[inline(always)]
                || self.row(row, indices, &steps),
            )
        });
        Pass {
            indices: indices.to_vec(),
            rows,
        }
    }

    /// Hands a pass's row buffers back for the next one.
    fn release(&self, pass: Pass) {
        let mut kept = lock(&self.kept.rows);
        for row in pass.rows {
            if kept.len() < MAX_KEPT_ROWS {
                kept.push(row.bytes);
            }
        }
    }

    /// One slice coded alone at `index`, for a slice that came out over
    /// the size limit.
    fn recode(&self, slice: usize, index: u8) -> Vec<u8> {
        let per_row = self.widths.len().max(1);
        let (row, column) = (slice / per_row, slice % per_row);
        let mb_x: u32 = self.widths.iter().take(column).sum();
        let mbs = self.widths.get(column).copied().unwrap_or(1);
        let mut work = lock(&self.kept.work).pop().unwrap_or_default();
        let mut out = Vec::new();
        cutlass_simd::wide::wide(
            #[inline(always)]
            || {
                work.prepare(self.picture, basis(), row, mb_x as usize, mbs as usize);
                let luma = steps(self.luma_matrix, index);
                let chroma = steps(self.chroma_matrix, index);
                work.write(index, &luma, &chroma, &mut out);
            },
        );
        lock(&self.kept.work).push(work);
        out
    }
}

/// What the passes so far have coded.
#[derive(Default)]
struct Coded {
    passes: Vec<Pass>,
    /// For each index coded: its pass, its place in that pass, and the
    /// frame's bytes there without alpha.
    at: BTreeMap<u8, (usize, usize, u64)>,
}

impl Coded {
    fn add(&mut self, pass: Pass) {
        let number = self.passes.len();
        let per = pass.indices.len();
        for (place, &index) in pass.indices.iter().enumerate() {
            let mut total = 0u64;
            for row in &pass.rows {
                for (slice, &alpha) in row.alpha.iter().enumerate() {
                    let len = row
                        .ranges
                        .get(slice * per + place)
                        .map_or(0, |range| range.1);
                    total += u64::from(len.saturating_sub(alpha));
                }
            }
            self.at.entry(index).or_insert((number, place, total));
        }
        self.passes.push(pass);
    }

    /// The frame's bytes at `index` without alpha, when a pass has coded it.
    fn total(&self, index: u8) -> Option<u64> {
        self.at.get(&index).map(|entry| entry.2)
    }

    /// The bytes of every slice at `index`, in picture order.
    fn slices(&self, index: u8) -> Vec<&[u8]> {
        let Some(&(number, place, _)) = self.at.get(&index) else {
            return Vec::new();
        };
        let Some(pass) = self.passes.get(number) else {
            return Vec::new();
        };
        let per = pass.indices.len();
        let mut out = Vec::new();
        for row in &pass.rows {
            for slice in 0..row.alpha.len() {
                let (start, len) = row
                    .ranges
                    .get(slice * per + place)
                    .copied()
                    .unwrap_or((0, 0));
                let (start, len) = (start as usize, len as usize);
                out.push(row.bytes.get(start..start + len).unwrap_or_default());
            }
        }
        out
    }
}

pub(crate) fn encode_frame(
    picture: &Picture<'_>,
    settings: Settings,
    pool: &Pool,
    cancel: &AtomicBool,
    kept: &Kept,
) -> Result<Encoded, CodecError> {
    let profile = settings.profile;
    let width_in_mb = picture.width.div_ceil(16);
    let (luma_matrix, chroma_matrix) = matrices(profile);
    let frame = Frame {
        picture,
        widths: slice_widths(width_in_mb, LOG2_SLICE),
        rows: picture.height.div_ceil(16) as usize,
        luma_matrix,
        chroma_matrix,
        pool,
        kept,
    };
    let mut coded = Coded::default();
    let outcome = search(&frame, settings, cancel, &mut coded);
    for pass in coded.passes {
        frame.release(pass);
    }
    outcome
}

/// Finds the frame's quantization index and puts the frame together.
fn search(
    frame: &Frame<'_>,
    settings: Settings,
    cancel: &AtomicBool,
    coded: &mut Coded,
) -> Result<Encoded, CodecError> {
    let picture = frame.picture;
    let cancelled = || cancel.load(Ordering::Relaxed);
    if cancelled() {
        return Err(CodecError::Cancelled);
    }
    let header_bytes = 8 + frame_header_size() + 8 + 2 * frame.slices();
    let target = (target_frame_bits(settings.profile, picture.width, picture.height) as f64
        * settings.budget_scale) as u64;
    let budget = target.saturating_sub(header_bytes as u64 * 8) / 8;

    // The first pass codes the frame at the last frame's index and at the
    // one finer. When the index has not moved, the search below finds
    // every answer it asks for already there.
    let start = settings.hint.unwrap_or(FIRST_GUESS).clamp(1, 224);
    let first = [start, start - 1];
    coded.add(frame.pass(if start > 1 { &first } else { &first[..1] }));
    // Bytes of every slice, without alpha, at one quantization index. An
    // index no pass has coded costs a pass, and `also` is coded with it.
    let total_at = |index: u8, also: Option<u8>, coded: &mut Coded| -> u64 {
        if coded.total(index).is_none() {
            match also.filter(|&extra| extra != index && coded.total(extra).is_none()) {
                Some(extra) => coded.add(frame.pass(&[index, extra])),
                None => coded.add(frame.pass(&[index])),
            }
        }
        coded.total(index).unwrap_or(u64::MAX)
    };

    // The finest index whose frame fits the budget, searched outward from
    // the last frame's index: galloping, then bisecting the bracket. Video
    // changes slowly, so most frames settle on the first pass.
    let mut fits = |index: u8, also: Option<u8>| total_at(index, also, coded) <= budget;
    let (mut bad, mut good) = if fits(start, None) {
        let mut good = start;
        let mut step = 1u8;
        loop {
            if good == 1 {
                break (0, 1);
            }
            if cancelled() {
                return Err(CodecError::Cancelled);
            }
            let next = good.saturating_sub(step).max(1);
            // If `next` does not fit, the bisection asks about the index
            // above it, so that one is coded in the same pass.
            if fits(next, Some(next + 1)) {
                good = next;
                step = step.saturating_mul(2);
            } else {
                break (next, good);
            }
        }
    } else {
        let mut bad = start;
        let mut step = 1u8;
        loop {
            if bad == 224 {
                break (223, 224);
            }
            if cancelled() {
                return Err(CodecError::Cancelled);
            }
            let next = bad.saturating_add(step).min(224);
            if next == 224 || fits(next, None) {
                break (bad, next);
            }
            bad = next;
            step = step.saturating_mul(2);
        }
    };
    while good - bad > 1 {
        if cancelled() {
            return Err(CodecError::Cancelled);
        }
        let mid = bad + (good - bad) / 2;
        if fits(mid, None) {
            good = mid;
        } else {
            bad = mid;
        }
    }
    let chosen = good;
    let finer = chosen - 1;
    let total = total_at(chosen, (finer > 0).then_some(finer), coded);
    if finer > 0 {
        total_at(finer, None, coded);
    }
    if cancelled() {
        return Err(CodecError::Cancelled);
    }
    let at_chosen = coded.slices(chosen);
    let mut picked: Vec<(&[u8], u8)> = at_chosen.iter().map(|&bytes| (bytes, chosen)).collect();
    // Spend what is left: move slices one index finer, spread over the picture.
    if finer > 0 {
        let at_finer = coded.slices(finer);
        let mut left = budget.saturating_sub(total);
        let mut order: Vec<usize> = (0..picked.len()).collect();
        order.sort_by(|&a, &b| {
            let fa = (a as f64 * 0.618_033_988_75).fract();
            let fb = (b as f64 * 0.618_033_988_75).fract();
            fa.total_cmp(&fb)
        });
        for slice in order {
            let (Some(&fine), Some(entry)) = (at_finer.get(slice), picked.get_mut(slice)) else {
                continue;
            };
            // Both hold the same alpha, so it cancels.
            let extra = (fine.len() as u64).saturating_sub(entry.0.len() as u64);
            if extra <= left {
                left -= extra;
                *entry = (fine, finer);
            }
        }
    }

    // A slice past the 16-bit size field is coded again, coarser.
    let mut recoded: Vec<(usize, Vec<u8>)> = Vec::new();
    for (slice, &(bytes, index)) in picked.iter().enumerate() {
        if bytes.len() <= MAX_SLICE {
            continue;
        }
        let mut index = index;
        let mut again = Vec::new();
        while index < 224 {
            index += 1;
            again = frame.recode(slice, index);
            if again.len() <= MAX_SLICE {
                break;
            }
        }
        if again.is_empty() || again.len() > MAX_SLICE {
            return Err(CodecError::Limit { what: "slice size" });
        }
        recoded.push((slice, again));
    }
    let mut out_slices: Vec<&[u8]> = picked.iter().map(|&(bytes, _)| bytes).collect();
    for (slice, bytes) in &recoded {
        if let Some(entry) = out_slices.get_mut(*slice) {
            *entry = bytes;
        }
    }
    let bytes = assemble(picture, settings.profile, settings.color, &out_slices)?;
    Ok(Encoded {
        bytes,
        index: chosen,
    })
}

fn frame_header_size() -> usize {
    FRAME_HEADER_FIXED + 128
}

fn assemble(
    picture: &Picture<'_>,
    profile: Profile,
    color: ColorCodes,
    slices: &[&[u8]],
) -> Result<Vec<u8>, CodecError> {
    let (luma_matrix, chroma_matrix) = matrices(profile);
    let header_size = frame_header_size();
    let table = 2 * slices.len();
    let data: usize = slices.iter().map(|slice| slice.len()).sum();
    let picture_size = 8 + table + data;
    let frame_size = 8 + header_size + picture_size;
    let frame_size_u32 =
        u32::try_from(frame_size).map_err(|_| CodecError::Limit { what: "frame size" })?;
    let mut out = Vec::with_capacity(frame_size);
    out.extend_from_slice(&frame_size_u32.to_be_bytes());
    out.extend_from_slice(&ICPF);
    // Frame header.
    out.extend_from_slice(&(header_size as u16).to_be_bytes());
    out.push(0);
    out.push(u8::from(
        picture.chroma == Chroma::Yuv444 || picture.alpha.is_some(),
    ));
    out.extend_from_slice(&ENCODER_ID);
    out.extend_from_slice(&(picture.width as u16).to_be_bytes());
    out.extend_from_slice(&(picture.height as u16).to_be_bytes());
    let chroma_format: u8 = match picture.chroma {
        Chroma::Yuv422 => 2,
        Chroma::Yuv444 => 3,
    };
    out.push(chroma_format << 6);
    out.push(0);
    out.push(color.primaries);
    out.push(color.transfer);
    out.push(color.matrix);
    out.push(match picture.alpha {
        None => 0,
        Some(true) => 2,
        Some(false) => 1,
    });
    out.push(0);
    out.push(3);
    out.extend_from_slice(luma_matrix);
    out.extend_from_slice(chroma_matrix);
    // Picture header.
    out.push(8 << 3);
    out.extend_from_slice(&(picture_size as u32).to_be_bytes());
    out.extend_from_slice(&(slices.len().min(usize::from(u16::MAX)) as u16).to_be_bytes());
    out.push((LOG2_SLICE as u8) << 4);
    for slice in slices {
        out.extend_from_slice(&(slice.len() as u16).to_be_bytes());
    }
    for slice in slices {
        out.extend_from_slice(slice);
    }
    Ok(out)
}
