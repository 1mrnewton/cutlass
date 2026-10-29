//! Film grain synthesis from the output process of the AV1 specification.
//!
//! The noise is added to the picture that is shown. Reference frames keep the
//! picture from before this step.

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::header::{INTER_FRAME, Sequence};

#[path = "grain_table.rs"]
mod grain_table;

const GRAIN_WIDTH: usize = 82;
const GRAIN_HEIGHT: usize = 73;
const CHROMA_WIDTH: usize = 44;
const CHROMA_HEIGHT: usize = 38;

/// Film grain parameters for one shown picture.
#[derive(Clone, Copy, Debug)]
pub struct Grain {
    pub apply: bool,
    pub seed: u16,
    num_y: u8,
    y_value: [u8; 14],
    y_scaling: [u8; 14],
    chroma_from_luma: bool,
    num_cb: u8,
    num_cr: u8,
    cb_value: [u8; 10],
    cb_scaling: [u8; 10],
    cr_value: [u8; 10],
    cr_scaling: [u8; 10],
    scaling_minus_8: u8,
    ar_lag: u8,
    ar_y: [i16; 24],
    ar_cb: [i16; 25],
    ar_cr: [i16; 25],
    ar_shift_minus_6: u8,
    grain_scale_shift: u8,
    cb_mult: i32,
    cb_luma_mult: i32,
    cb_offset: i32,
    cr_mult: i32,
    cr_luma_mult: i32,
    cr_offset: i32,
    overlap: bool,
    clip_restricted: bool,
}

impl Grain {
    pub const fn off() -> Self {
        Self {
            apply: false,
            seed: 0,
            num_y: 0,
            y_value: [0; 14],
            y_scaling: [0; 14],
            chroma_from_luma: false,
            num_cb: 0,
            num_cr: 0,
            cb_value: [0; 10],
            cb_scaling: [0; 10],
            cr_value: [0; 10],
            cr_scaling: [0; 10],
            scaling_minus_8: 0,
            ar_lag: 0,
            ar_y: [0; 24],
            ar_cb: [0; 25],
            ar_cr: [0; 25],
            ar_shift_minus_6: 0,
            grain_scale_shift: 0,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            overlap: false,
            clip_restricted: false,
        }
    }
}

pub fn read(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    frame_type: u8,
    show_frame: bool,
    showable: bool,
) -> Result<Grain, CodecError> {
    if !sequence.film_grain || (!show_frame && !showable) {
        return Ok(Grain::off());
    }
    if !bits.read_bool()? {
        return Ok(Grain::off());
    }
    let seed = bits.read(16)? as u16;
    let update = if frame_type == INTER_FRAME {
        bits.read_bool()?
    } else {
        true
    };
    if !update {
        let _ = bits.read(3)?;
        return Err(CodecError::Unsupported {
            what: "film grain reference",
        });
    }
    let mut grain = Grain::off();
    grain.apply = true;
    grain.seed = seed;
    grain.num_y = bits.read(4)? as u8;
    if grain.num_y > 14 {
        return Err(malformed(bits));
    }
    for index in 0..usize::from(grain.num_y) {
        grain.y_value[index] = bits.read(8)? as u8;
        grain.y_scaling[index] = bits.read(8)? as u8;
    }
    grain.chroma_from_luma = bits.read_bool()?;
    let skip_chroma = grain.chroma_from_luma || grain.num_y == 0;
    if !skip_chroma {
        grain.num_cb = bits.read(4)? as u8;
        if grain.num_cb > 10 {
            return Err(malformed(bits));
        }
        for index in 0..usize::from(grain.num_cb) {
            grain.cb_value[index] = bits.read(8)? as u8;
            grain.cb_scaling[index] = bits.read(8)? as u8;
        }
        grain.num_cr = bits.read(4)? as u8;
        if grain.num_cr > 10 {
            return Err(malformed(bits));
        }
        for index in 0..usize::from(grain.num_cr) {
            grain.cr_value[index] = bits.read(8)? as u8;
            grain.cr_scaling[index] = bits.read(8)? as u8;
        }
    }
    grain.scaling_minus_8 = bits.read(2)? as u8;
    grain.ar_lag = bits.read(2)? as u8;
    let lag = usize::from(grain.ar_lag);
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma = if grain.num_y > 0 {
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if grain.num_y > 0 {
        for index in 0..num_pos_luma {
            grain.ar_y[index] = bits.read(8)? as i16 - 128;
        }
    }
    if grain.chroma_from_luma || grain.num_cb > 0 {
        for index in 0..num_pos_chroma {
            grain.ar_cb[index] = bits.read(8)? as i16 - 128;
        }
    }
    if grain.chroma_from_luma || grain.num_cr > 0 {
        for index in 0..num_pos_chroma {
            grain.ar_cr[index] = bits.read(8)? as i16 - 128;
        }
    }
    grain.ar_shift_minus_6 = bits.read(2)? as u8;
    grain.grain_scale_shift = bits.read(2)? as u8;
    if grain.num_cb > 0 {
        grain.cb_mult = bits.read(8)? as i32;
        grain.cb_luma_mult = bits.read(8)? as i32;
        grain.cb_offset = bits.read(9)? as i32;
    }
    if grain.num_cr > 0 {
        grain.cr_mult = bits.read(8)? as i32;
        grain.cr_luma_mult = bits.read(8)? as i32;
        grain.cr_offset = bits.read(9)? as i32;
    }
    grain.overlap = bits.read_bool()?;
    grain.clip_restricted = bits.read_bool()?;
    Ok(grain)
}

pub fn overlay(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    grain: &Grain,
) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if width == 0 || height == 0 || y.len() < width * height {
        return None;
    }
    let chroma_w = width.div_ceil(2);
    let chroma_h = height.div_ceil(2);
    if u.len() < chroma_w * chroma_h || v.len() < chroma_w * chroma_h {
        return None;
    }
    let mut out_y = y[..width * height].to_vec();
    let mut out_u = u[..chroma_w * chroma_h].to_vec();
    let mut out_v = v[..chroma_w * chroma_h].to_vec();
    let luma = generate_luma(grain);
    let (cb, cr) = generate_chroma(grain, &luma);
    let scaling = scaling_luts(grain);
    add_noise(Noise {
        y: &mut out_y,
        u: &mut out_u,
        v: &mut out_v,
        width,
        height,
        grain,
        luma: &luma,
        cb: &cb,
        cr: &cr,
        scaling: &scaling,
    });
    Some((out_y, out_u, out_v))
}

struct Noise<'a> {
    y: &'a mut [u8],
    u: &'a mut [u8],
    v: &'a mut [u8],
    width: usize,
    height: usize,
    grain: &'a Grain,
    luma: &'a [i32],
    cb: &'a [i32],
    cr: &'a [i32],
    scaling: &'a [[i32; 256]; 3],
}

fn add_noise(noise: Noise<'_>) {
    let Noise {
        y,
        u,
        v,
        width,
        height,
        grain,
        luma,
        cb,
        cr,
        scaling,
    } = noise;
    let stripes = stripe_count(height);
    let luma_stride = width + 64;
    let chroma_w = width.div_ceil(2);
    let chroma_stride = chroma_w + 64;
    let mut luma_stripes = vec![0i32; stripes * 34 * luma_stride];
    let mut cb_stripes = vec![0i32; stripes * 17 * chroma_stride];
    let mut cr_stripes = vec![0i32; stripes * 17 * chroma_stride];
    let mut luma_num = 0usize;
    let mut row = 0usize;
    while row < height.div_ceil(2) {
        let mut register = grain.seed ^ (((luma_num * 37 + 178) & 255) as u16) << 8;
        register ^= ((luma_num * 173 + 105) & 255) as u16;
        let mut col = 0usize;
        while col < width.div_ceil(2) {
            let rand = random(&mut register, 8);
            let offset_x = (rand >> 4) as usize;
            let offset_y = (rand & 15) as usize;
            copy_block(
                &mut luma_stripes,
                Block {
                    stripe: luma_num,
                    stride: luma_stride,
                    rows: 34,
                    origin: col * 2,
                    offset_x: 9 + offset_x * 2,
                    offset_y: 9 + offset_y * 2,
                    grain_w: GRAIN_WIDTH,
                    overlap: grain.overlap && col > 0,
                    subsampled: false,
                },
                luma,
            );
            copy_block(
                &mut cb_stripes,
                Block {
                    stripe: luma_num,
                    stride: chroma_stride,
                    rows: 17,
                    origin: col,
                    offset_x: 6 + offset_x,
                    offset_y: 6 + offset_y,
                    grain_w: CHROMA_WIDTH,
                    overlap: grain.overlap && col > 0,
                    subsampled: true,
                },
                cb,
            );
            copy_block(
                &mut cr_stripes,
                Block {
                    stripe: luma_num,
                    stride: chroma_stride,
                    rows: 17,
                    origin: col,
                    offset_x: 6 + offset_x,
                    offset_y: 6 + offset_y,
                    grain_w: CHROMA_WIDTH,
                    overlap: grain.overlap && col > 0,
                    subsampled: true,
                },
                cr,
            );
            col += 16;
        }
        luma_num += 1;
        row += 16;
    }
    let (min_value, max_luma, max_chroma) = if grain.clip_restricted {
        (16, 235, 240)
    } else {
        (0, 255, 255)
    };
    let shift = i32::from(grain.scaling_minus_8) + 8;
    let chroma_h = height.div_ceil(2);
    for sample_y in 0..chroma_h {
        let luma_y = sample_y * 2;
        for sample_x in 0..chroma_w {
            let luma_x = sample_x * 2;
            let next_x = (luma_x + 1).min(width - 1);
            let average = round2(
                i32::from(y[luma_y * width + luma_x]) + i32::from(y[luma_y * width + next_x]),
                1,
            );
            if grain.num_cb > 0 || grain.chroma_from_luma {
                blend_chroma(
                    u,
                    Chroma {
                        x: sample_x,
                        y: sample_y,
                        stride: chroma_w,
                        noise: sample_noise(
                            &cb_stripes,
                            sample_y,
                            sample_x,
                            chroma_stride,
                            17,
                            grain.overlap,
                            true,
                        ),
                        scale: scaling[1][merged_chroma(
                            grain,
                            average,
                            u[sample_y * chroma_w + sample_x],
                            true,
                        ) as usize],
                        shift,
                        min_value,
                        max_value: max_chroma,
                    },
                );
            }
            if grain.num_cr > 0 || grain.chroma_from_luma {
                blend_chroma(
                    v,
                    Chroma {
                        x: sample_x,
                        y: sample_y,
                        stride: chroma_w,
                        noise: sample_noise(
                            &cr_stripes,
                            sample_y,
                            sample_x,
                            chroma_stride,
                            17,
                            grain.overlap,
                            true,
                        ),
                        scale: scaling[2][merged_chroma(
                            grain,
                            average,
                            v[sample_y * chroma_w + sample_x],
                            false,
                        ) as usize],
                        shift,
                        min_value,
                        max_value: max_chroma,
                    },
                );
            }
        }
    }
    for sample_y in 0..height {
        for sample_x in 0..width {
            if grain.num_y == 0 {
                continue;
            }
            let orig = i32::from(y[sample_y * width + sample_x]);
            let sample = sample_noise(
                &luma_stripes,
                sample_y,
                sample_x,
                luma_stride,
                34,
                grain.overlap,
                false,
            );
            let scaled = round2(scaling[0][orig as usize] * sample, shift);
            y[sample_y * width + sample_x] = clip3(min_value, max_luma, orig + scaled) as u8;
        }
    }
}

struct Chroma {
    x: usize,
    y: usize,
    stride: usize,
    noise: i32,
    scale: i32,
    shift: i32,
    min_value: i32,
    max_value: i32,
}

fn blend_chroma(plane: &mut [u8], chroma: Chroma) {
    let slot = chroma.y * chroma.stride + chroma.x;
    let orig = i32::from(plane[slot]);
    let scaled = round2(chroma.scale * chroma.noise, chroma.shift);
    plane[slot] = clip3(chroma.min_value, chroma.max_value, orig + scaled) as u8;
}

fn merged_chroma(grain: &Grain, average: i32, orig: u8, cb: bool) -> i32 {
    if grain.chroma_from_luma {
        return average.clamp(0, 255);
    }
    let (mult, luma_mult, offset) = if cb {
        (grain.cb_mult, grain.cb_luma_mult, grain.cb_offset)
    } else {
        (grain.cr_mult, grain.cr_luma_mult, grain.cr_offset)
    };
    let combined = average * (luma_mult - 128) + i32::from(orig) * (mult - 128);
    clip3(0, 255, (combined >> 6) + (offset - 256))
}

fn sample_noise(
    stripes: &[i32],
    y: usize,
    x: usize,
    stride: usize,
    rows: usize,
    overlap: bool,
    sub_y: bool,
) -> i32 {
    let shift = if sub_y { 4 } else { 5 };
    let luma_num = y >> shift;
    let i = y - (luma_num << shift);
    let mut g = stripes[stripe_at(luma_num, i, x, stride, rows)];
    let overlap_row = luma_num > 0 && overlap && if sub_y { i < 1 } else { i < 2 };
    if overlap_row {
        let old_row = if sub_y { i + 16 } else { i + 32 };
        let old = stripes[stripe_at(luma_num - 1, old_row, x, stride, rows)];
        let (old_w, new_w) = if sub_y {
            (23, 22)
        } else if i == 0 {
            (27, 17)
        } else {
            (17, 27)
        };
        g = clip3(-128, 127, round2(old * old_w + g * new_w, 5));
    }
    g
}

fn stripe_at(stripe: usize, row: usize, x: usize, stride: usize, rows: usize) -> usize {
    (stripe * rows + row) * stride + x
}

fn stripe_count(height: usize) -> usize {
    height.div_ceil(2).div_ceil(16).max(1)
}

struct Block {
    stripe: usize,
    stride: usize,
    rows: usize,
    origin: usize,
    offset_x: usize,
    offset_y: usize,
    grain_w: usize,
    overlap: bool,
    subsampled: bool,
}

fn copy_block(dest: &mut [i32], block: Block, grain: &[i32]) {
    let grain_h = if block.subsampled {
        CHROMA_HEIGHT
    } else {
        GRAIN_HEIGHT
    };
    for row in 0..block.rows {
        for col in 0..block.rows {
            let gy = block.offset_y + row;
            let gx = block.offset_x + col;
            let mut g = if gy < grain_h && gx < block.grain_w {
                grain[gy * block.grain_w + gx]
            } else {
                0
            };
            let at = stripe_at(
                block.stripe,
                row,
                block.origin + col,
                block.stride,
                block.rows,
            );
            if block.overlap && !block.subsampled && col < 2 {
                let old = dest[at];
                let (old_w, new_w) = if col == 0 { (27, 17) } else { (17, 27) };
                g = clip3(-128, 127, round2(old * old_w + g * new_w, 5));
            } else if block.overlap && block.subsampled && col == 0 {
                let old = dest[at];
                g = clip3(-128, 127, round2(old * 23 + g * 22, 5));
            }
            if at < dest.len() {
                dest[at] = g;
            }
        }
    }
}

fn generate_luma(grain: &Grain) -> Vec<i32> {
    let mut plane = vec![0i32; GRAIN_HEIGHT * GRAIN_WIDTH];
    let shift = 4 + i32::from(grain.grain_scale_shift);
    let mut register = grain.seed;
    for slot in &mut plane {
        let g = if grain.num_y > 0 {
            i32::from(grain_table::GAUSSIAN[random(&mut register, 11) as usize])
        } else {
            0
        };
        *slot = round2(g, shift);
    }
    autoregressive(
        &mut plane,
        GRAIN_WIDTH,
        GRAIN_HEIGHT,
        grain.ar_lag,
        &grain.ar_y,
        grain.ar_shift_minus_6,
    );
    plane
}

fn generate_chroma(grain: &Grain, luma: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut cb = vec![0i32; CHROMA_HEIGHT * CHROMA_WIDTH];
    let mut cr = vec![0i32; CHROMA_HEIGHT * CHROMA_WIDTH];
    let shift = 4 + i32::from(grain.grain_scale_shift);
    let mut register = grain.seed ^ 0xb524;
    for slot in &mut cb {
        let g = if grain.num_cb > 0 || grain.chroma_from_luma {
            i32::from(grain_table::GAUSSIAN[random(&mut register, 11) as usize])
        } else {
            0
        };
        *slot = round2(g, shift);
    }
    register = grain.seed ^ 0x49d8;
    for slot in &mut cr {
        let g = if grain.num_cr > 0 || grain.chroma_from_luma {
            i32::from(grain_table::GAUSSIAN[random(&mut register, 11) as usize])
        } else {
            0
        };
        *slot = round2(g, shift);
    }
    chroma_autoregressive(&mut cb, &mut cr, grain, luma);
    (cb, cr)
}

fn autoregressive(
    plane: &mut [i32],
    width: usize,
    height: usize,
    lag: u8,
    coeffs: &[i16],
    shift_minus_6: u8,
) {
    let shift = i32::from(shift_minus_6) + 6;
    let lag = usize::from(lag);
    if lag == 0 {
        return;
    }
    for y in 3..height {
        for x in 3..width - 3 {
            let mut sum = 0i32;
            let mut pos = 0usize;
            for delta_row in -(lag as i32)..=0 {
                for delta_col in -(lag as i32)..=lag as i32 {
                    if delta_row == 0 && delta_col == 0 {
                        break;
                    }
                    let sample = plane
                        [(y as i32 + delta_row) as usize * width + (x as i32 + delta_col) as usize];
                    sum += sample * i32::from(coeffs[pos]);
                    pos += 1;
                }
            }
            let value = plane[y * width + x] + round2(sum, shift);
            plane[y * width + x] = clip3(-128, 127, value);
        }
    }
}

fn chroma_autoregressive(cb: &mut [i32], cr: &mut [i32], grain: &Grain, luma: &[i32]) {
    let shift = i32::from(grain.ar_shift_minus_6) + 6;
    let lag = usize::from(grain.ar_lag);
    for y in 3..CHROMA_HEIGHT {
        for x in 3..CHROMA_WIDTH - 3 {
            let mut s0 = 0i32;
            let mut s1 = 0i32;
            let mut pos = 0usize;
            for delta_row in -(lag as i32)..=0 {
                for delta_col in -(lag as i32)..=lag as i32 {
                    let c0 = i32::from(grain.ar_cb[pos]);
                    let c1 = i32::from(grain.ar_cr[pos]);
                    if delta_row == 0 && delta_col == 0 {
                        if grain.num_y > 0 {
                            let luma_x = ((x - 3) << 1) + 3;
                            let luma_y = ((y - 3) << 1) + 3;
                            let mut mixed = 0i32;
                            for row in 0..2 {
                                for col in 0..2 {
                                    mixed += luma[(luma_y + row) * GRAIN_WIDTH + luma_x + col];
                                }
                            }
                            let mixed = round2(mixed, 2);
                            s0 += mixed * c0;
                            s1 += mixed * c1;
                        }
                        break;
                    }
                    let cb_sample = cb[(y as i32 + delta_row) as usize * CHROMA_WIDTH
                        + (x as i32 + delta_col) as usize];
                    let cr_sample = cr[(y as i32 + delta_row) as usize * CHROMA_WIDTH
                        + (x as i32 + delta_col) as usize];
                    s0 += cb_sample * c0;
                    s1 += cr_sample * c1;
                    pos += 1;
                }
            }
            cb[y * CHROMA_WIDTH + x] =
                clip3(-128, 127, cb[y * CHROMA_WIDTH + x] + round2(s0, shift));
            cr[y * CHROMA_WIDTH + x] =
                clip3(-128, 127, cr[y * CHROMA_WIDTH + x] + round2(s1, shift));
        }
    }
}

fn scaling_luts(grain: &Grain) -> [[i32; 256]; 3] {
    let mut luts = [[0i32; 256]; 3];
    fill_lut(&mut luts[0], grain.num_y, &grain.y_value, &grain.y_scaling);
    if grain.chroma_from_luma {
        luts[1] = luts[0];
        luts[2] = luts[0];
    } else {
        fill_lut(
            &mut luts[1],
            grain.num_cb,
            &grain.cb_value,
            &grain.cb_scaling,
        );
        fill_lut(
            &mut luts[2],
            grain.num_cr,
            &grain.cr_value,
            &grain.cr_scaling,
        );
    }
    luts
}

fn fill_lut(lut: &mut [i32; 256], count: u8, values: &[u8], scaling: &[u8]) {
    let count = usize::from(count);
    if count == 0 {
        return;
    }
    let first = usize::from(values[0]);
    for slot in &mut lut[..first] {
        *slot = i32::from(scaling[0]);
    }
    for index in 0..count - 1 {
        let delta_y = i32::from(scaling[index + 1]) - i32::from(scaling[index]);
        let delta_x = i32::from(values[index + 1]) - i32::from(values[index]);
        if delta_x <= 0 {
            continue;
        }
        let delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
        for step in 0..delta_x {
            let value = i32::from(scaling[index]) + ((step * delta + 32768) >> 16);
            let at = usize::from(values[index]) + step as usize;
            if at < lut.len() {
                lut[at] = value;
            }
        }
    }
    let last = usize::from(values[count - 1]);
    for slot in &mut lut[last..] {
        *slot = i32::from(scaling[count - 1]);
    }
}

fn random(register: &mut u16, bits: u32) -> u32 {
    let r = u32::from(*register);
    let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    let r = (r >> 1) | (bit << 15);
    *register = r as u16;
    (r >> (16 - bits)) & ((1 << bits) - 1)
}

fn round2(value: i32, bits: i32) -> i32 {
    if bits <= 0 {
        value
    } else {
        (value + (1 << (bits - 1))) >> bits
    }
}

fn clip3(min: i32, max: i32, value: i32) -> i32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn malformed(bits: &BitReader<'_>) -> CodecError {
    CodecError::Malformed {
        at_bit: bits.position() as u64,
        what: "film grain",
    }
}

#[cfg(test)]
mod tests {
    use super::grain_table::GAUSSIAN;

    #[test]
    fn the_gaussian_table_starts_where_the_specification_does() {
        assert_eq!(GAUSSIAN.len(), 2048);
        assert_eq!(GAUSSIAN[0], 56);
        assert_eq!(GAUSSIAN[1], 568);
        assert_eq!(GAUSSIAN[2], -180);
    }
}
