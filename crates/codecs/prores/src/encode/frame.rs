//! A whole frame: slices in parallel, one quantizer for the frame tuned to
//! the profile's bit budget, then the headers.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::CodecError;
use cutlass_pool::Pool;

use super::rate::{matrices, target_frame_bits};
use super::slice::{Planes, Prepared, Work, slice_bytes, steps, write_slice};
use crate::header::{Chroma, FRAME_HEADER_FIXED, ICPF, slice_widths};
use crate::profile::Profile;

/// Every slice is 8 macroblocks wide where the row allows.
const LOG2_SLICE: u32 = 3;
/// A slice's size is a 16-bit field.
const MAX_SLICE: u32 = 65_535;
/// Our encoder's four-character identifier.
pub(crate) const ENCODER_ID: [u8; 4] = *b"ctls";

/// Color code points written into the frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColorCodes {
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
}

/// How a frame came out: its bytes and the quantizer the rate control chose.
pub(crate) struct Encoded {
    pub bytes: Vec<u8>,
    /// Quantization index most slices used.
    pub index: u8,
}

struct Geometry {
    rows: usize,
    widths: Vec<u32>,
}

impl Geometry {
    fn slices(&self) -> usize {
        self.rows * self.widths.len()
    }
}

pub(crate) fn encode_frame(
    planes: &Planes,
    profile: Profile,
    color: ColorCodes,
    budget_scale: f64,
    pool: &Pool,
    cancel: &AtomicBool,
) -> Result<Encoded, CodecError> {
    let width_in_mb = planes.width.div_ceil(16);
    let geometry = Geometry {
        rows: planes.height.div_ceil(16) as usize,
        widths: slice_widths(width_in_mb, LOG2_SLICE),
    };
    let per_row = geometry.widths.len();
    let cancelled = || cancel.load(Ordering::Relaxed);

    let prepared: Vec<Vec<Prepared>> = pool.slices(geometry.rows, |row| {
        let mut mb_x = 0usize;
        let mut out = Vec::with_capacity(per_row);
        for &mbs in &geometry.widths {
            out.push(planes.prepare(row, mb_x, mbs as usize));
            mb_x += mbs as usize;
        }
        out
    });
    if cancelled() {
        return Err(CodecError::Cancelled);
    }

    let (luma_matrix, chroma_matrix) = matrices(profile);
    let alpha = planes.alpha.is_some();
    let slice_header: u32 = if alpha { 8 } else { 6 };
    let header_bytes = 8 + frame_header_size() + 8 + 2 * geometry.slices();
    let target =
        (target_frame_bits(profile, planes.width, planes.height) as f64 * budget_scale) as u64;
    let budget = target.saturating_sub(header_bytes as u64 * 8) / 8;

    // Bytes of every slice, without alpha, at one quantization index.
    let probe = |index: u8| -> Vec<u32> {
        let luma = steps(luma_matrix, index);
        let chroma = steps(chroma_matrix, index);
        pool.slices(geometry.rows, |row| {
            let mut work = Work::default();
            prepared[row]
                .iter()
                .map(|slice| slice_bytes(slice, &luma, &chroma, slice_header, &mut work))
                .collect::<Vec<u32>>()
        })
        .into_iter()
        .flatten()
        .collect()
    };
    let mut probes: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
    let total_at = |index: u8, probes: &mut BTreeMap<u8, Vec<u32>>| -> u64 {
        let sizes = probes.entry(index).or_insert_with(|| probe(index));
        sizes.iter().map(|&s| u64::from(s)).sum()
    };

    // The finest index whose frame fits the budget.
    let chosen = if total_at(1, &mut probes) <= budget {
        1
    } else if total_at(224, &mut probes) > budget {
        224
    } else {
        let (mut low, mut high) = (1u8, 224u8);
        while high - low > 1 {
            if cancelled() {
                return Err(CodecError::Cancelled);
            }
            let mid = low + (high - low) / 2;
            if total_at(mid, &mut probes) <= budget {
                high = mid;
            } else {
                low = mid;
            }
        }
        high
    };
    let mut indices = vec![chosen; geometry.slices()];
    let mut sizes = probes.get(&chosen).cloned().unwrap_or_default();
    // Spend what is left: move slices one index finer, spread over the picture.
    if chosen > 1 {
        let finer = chosen - 1;
        let total: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
        let mut left = budget.saturating_sub(total);
        let finer_sizes = probes.entry(finer).or_insert_with(|| probe(finer)).clone();
        let count = sizes.len();
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| {
            let fa = (a as f64 * 0.618_033_988_75).fract();
            let fb = (b as f64 * 0.618_033_988_75).fract();
            fa.total_cmp(&fb)
        });
        for slice in order {
            let extra = u64::from(finer_sizes[slice].saturating_sub(sizes[slice]));
            if extra <= left {
                left -= extra;
                indices[slice] = finer;
                sizes[slice] = finer_sizes[slice];
            }
        }
    }
    if cancelled() {
        return Err(CodecError::Cancelled);
    }

    // Write, coarsening any slice past the 16-bit size field.
    let slices: Vec<Vec<Vec<u8>>> = pool.slices(geometry.rows, |row| {
        let mut work = Work::default();
        prepared[row]
            .iter()
            .enumerate()
            .map(|(column, slice)| {
                let mut index = indices[row * per_row + column];
                loop {
                    let luma = steps(luma_matrix, index);
                    let chroma = steps(chroma_matrix, index);
                    let bytes = write_slice(slice, index, &luma, &chroma, &mut work);
                    if bytes.len() as u32 <= MAX_SLICE || index == 224 {
                        return bytes;
                    }
                    index += 1;
                }
            })
            .collect()
    });
    let mut out_slices = Vec::with_capacity(geometry.slices());
    for row in slices {
        for slice in row {
            if slice.len() as u32 > MAX_SLICE {
                return Err(CodecError::Limit { what: "slice size" });
            }
            out_slices.push(slice);
        }
    }

    let bytes = assemble(planes, profile, color, &out_slices)?;
    Ok(Encoded {
        bytes,
        index: chosen,
    })
}

fn frame_header_size() -> usize {
    FRAME_HEADER_FIXED + 128
}

fn assemble(
    planes: &Planes,
    profile: Profile,
    color: ColorCodes,
    slices: &[Vec<u8>],
) -> Result<Vec<u8>, CodecError> {
    let (luma_matrix, chroma_matrix) = matrices(profile);
    let header_size = frame_header_size();
    let table = 2 * slices.len();
    let data: usize = slices.iter().map(Vec::len).sum();
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
        planes.chroma == Chroma::Yuv444 || planes.alpha.is_some(),
    ));
    out.extend_from_slice(&ENCODER_ID);
    out.extend_from_slice(&(planes.width as u16).to_be_bytes());
    out.extend_from_slice(&(planes.height as u16).to_be_bytes());
    let chroma_format: u8 = match planes.chroma {
        Chroma::Yuv422 => 2,
        Chroma::Yuv444 => 3,
    };
    out.push(chroma_format << 6);
    out.push(0);
    out.push(color.primaries);
    out.push(color.transfer);
    out.push(color.matrix);
    out.push(match &planes.alpha {
        None => 0,
        Some(alpha) if alpha.sixteen => 2,
        Some(_) => 1,
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
