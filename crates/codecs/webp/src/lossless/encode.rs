//! The VP8L encoder (RFC 9649): transforms chosen per image, then the
//! entropy-coded image.
//!
//! An image with at most 256 colors is coded as palette indices, packed
//! several to a byte when there are at most 16. Any other image is coded
//! with subtract-green, a predictor chosen per tile, and from effort 3 a
//! cross-color transform chosen per tile. From effort 7 the encoder codes
//! several candidates and keeps the smallest.

use super::bits::BitWriter;
use super::decode::{SIGNATURE, subsample};
use super::entropy::{Effort, write_image};
use super::transform::{color_delta, predict};

/// Settings of one lossless encode.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    /// 0 (fastest) to 9 (smallest), as `cwebp -z`.
    pub effort: u8,
    /// Keep the RGB of fully transparent pixels. Otherwise they become
    /// black, which codes smaller and looks the same.
    pub exact: bool,
}

/// Per-channel difference modulo 256, the inverse of [`add_pixels`].
#[inline]
fn sub_pixels(a: u32, b: u32) -> u32 {
    let alpha_green = ((a | 0x00ff_00ff).wrapping_sub(b & 0xff00_ff00)) & 0xff00_ff00;
    let red_blue = ((a | 0xff00_ff00).wrapping_sub(b & 0x00ff_00ff)) & 0x00ff_00ff;
    alpha_green | red_blue
}

fn effort(level: u8) -> Effort {
    let chain = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024][usize::from(level.min(9))];
    Effort {
        chain,
        window: match level {
            0..=2 => 1 << 16,
            3..=5 => 1 << 18,
            _ => 1 << 20,
        },
        lazy: level >= 5,
        cache_search: level >= 4,
        group_bits: match level {
            0..=2 => None,
            3 | 4 => Some(6),
            5 | 6 => Some(5),
            _ => Some(4),
        },
        optimal: match level {
            0..=2 => 0,
            3..=8 => 1,
            _ => 2,
        },
    }
}

/// A complete `VP8L` bitstream, header included.
pub(crate) fn encode(argb: &[u32], width: u32, height: u32, config: Config) -> Vec<u8> {
    let alpha = argb.iter().any(|&pixel| pixel >> 24 != 0xff);
    let mut writer = BitWriter::with_capacity(argb.len());
    writer.write(u32::from(SIGNATURE), 8);
    writer.write(width - 1, 14);
    writer.write(height - 1, 14);
    writer.write(u32::from(alpha), 1);
    writer.write(0, 3);
    let stream = encode_stream(argb, width, height, config);
    let mut bytes = writer.finish();
    // The header is exactly five bytes, so the stream starts on a byte.
    bytes.extend_from_slice(&stream);
    bytes
}

/// A headerless image stream, as an `ALPH` chunk carries one.
pub(crate) fn encode_stream(argb: &[u32], width: u32, height: u32, config: Config) -> Vec<u8> {
    let mut pixels = argb.to_vec();
    if !config.exact {
        for pixel in &mut pixels {
            if *pixel >> 24 == 0 {
                *pixel = 0;
            }
        }
    }
    let width = width as usize;
    let height = height as usize;
    let palette = palette(&pixels);
    // Below effort 7 only one of palette and spatial is coded: the one a
    // quick entropy estimate prefers.
    let try_spatial = match &palette {
        None => true,
        Some(_) if config.effort >= 7 => true,
        Some(palette) => !palette_wins(&pixels, width, palette),
    };
    let palette = palette.filter(|_| config.effort >= 7 || !try_spatial);
    let mut candidates: Vec<Plan> = Vec::new();
    if let Some(palette) = &palette {
        candidates.push(Plan::Palette {
            palette: palette.clone(),
            predictor: false,
        });
        if config.effort >= 8 && palette.len() > 16 {
            candidates.push(Plan::Palette {
                palette: palette.clone(),
                predictor: true,
            });
        }
    }
    if try_spatial {
        let bits = predictor_bits(config.effort, width, height);
        candidates.push(Plan::Spatial { bits });
        if config.effort >= 8 {
            candidates.push(Plan::Spatial {
                bits: bits.saturating_sub(1).max(2),
            });
        }
        if config.effort >= 9 {
            candidates.push(Plan::Spatial { bits: bits + 1 });
        }
    }
    let mut best: Option<Vec<u8>> = None;
    for plan in candidates {
        let bytes = run(&plan, &pixels, width, height, config.effort);
        if best.as_ref().is_none_or(|best| bytes.len() < best.len()) {
            best = Some(bytes);
        }
    }
    best.unwrap_or_default()
}

enum Plan {
    Palette { palette: Vec<u32>, predictor: bool },
    Spatial { bits: u32 },
}

fn predictor_bits(effort: u8, width: usize, height: usize) -> u32 {
    let base: u32 = match effort {
        0 => 6,
        1 | 2 => 5,
        3..=6 => 4,
        _ => 3,
    };
    // Keep the mode image small for large pictures.
    let mut bits = base;
    while bits < 6 && (width * height) >> (2 * bits) > 1 << 16 {
        bits += 1;
    }
    bits
}

fn run(plan: &Plan, pixels: &[u32], width: usize, height: usize, level: u8) -> Vec<u8> {
    let mut writer = BitWriter::with_capacity(pixels.len());
    let effort = effort(level);
    match plan {
        Plan::Palette { palette, predictor } => {
            let (indices, coded_width) = index_image(pixels, width, palette);
            writer.write(1, 1);
            writer.write(3, 2);
            writer.write(palette.len() as u32 - 1, 8);
            let deltas: Vec<u32> = palette
                .iter()
                .enumerate()
                .map(|(i, &color)| {
                    if i == 0 {
                        color
                    } else {
                        sub_pixels(color, palette[i - 1])
                    }
                })
                .collect();
            write_image(
                &mut writer,
                &deltas,
                deltas.len(),
                1,
                &Effort::SUB_IMAGE,
                false,
            );
            let mut image = indices;
            if *predictor {
                let bits = predictor_bits(level, coded_width, height);
                image = apply_predictor(&mut writer, &image, coded_width, height, bits);
            }
            writer.write(0, 1);
            write_image(&mut writer, &image, coded_width, height, &effort, true);
        }
        Plan::Spatial { bits } => {
            let mut image: Vec<u32> = pixels.iter().map(|&p| subtract_green(p)).collect();
            writer.write(1, 1);
            writer.write(2, 2);
            image = apply_predictor(&mut writer, &image, width, height, *bits);
            if level >= 3 {
                image = apply_cross_color(&mut writer, &image, width, height, *bits + 1);
            }
            writer.write(0, 1);
            write_image(&mut writer, &image, width, height, &effort, true);
        }
    }
    writer.finish()
}

fn subtract_green(pixel: u32) -> u32 {
    let green = (pixel >> 8) & 0xff;
    let red_blue = (pixel & 0x00ff_00ff) | 0xff00_ff00;
    let red_blue = red_blue.wrapping_sub((green << 16) | green) & 0x00ff_00ff;
    (pixel & 0xff00_ff00) | red_blue
}

/// The distinct colors, sorted, when there are at most 256.
fn palette(pixels: &[u32]) -> Option<Vec<u32>> {
    let mut colors: Vec<u32> = Vec::with_capacity(257);
    let mut last = None;
    for &pixel in pixels {
        if last == Some(pixel) {
            continue;
        }
        last = Some(pixel);
        if let Err(at) = colors.binary_search(&pixel) {
            if colors.len() == 256 {
                return None;
            }
            colors.insert(at, pixel);
        }
    }
    Some(colors)
}

/// Whether palette indices look cheaper than spatial residuals: the
/// entropy of the indices against that of each channel's difference from
/// the left neighbour after subtract-green. Small palettes always win,
/// since several indices share a byte.
fn palette_wins(pixels: &[u32], width: usize, palette: &[u32]) -> bool {
    if palette.len() <= 16 {
        return true;
    }
    let mut indices = [0u32; 256];
    let mut channels = [[0u32; 256]; 4];
    let mut index_of = std::collections::HashMap::with_capacity(palette.len());
    for (index, &color) in palette.iter().enumerate() {
        index_of.insert(color, index);
    }
    for (at, &pixel) in pixels.iter().enumerate() {
        indices[index_of.get(&pixel).copied().unwrap_or(0)] += 1;
        let neighbour = if at % width > 0 {
            pixels[at - 1]
        } else if at >= width {
            pixels[at - width]
        } else {
            0xff00_0000
        };
        let residual = sub_pixels(subtract_green(pixel), subtract_green(neighbour));
        for (channel, counts) in channels.iter_mut().enumerate() {
            counts[((residual >> (8 * channel)) & 0xff) as usize] += 1;
        }
    }
    let spatial: f64 = channels.iter().map(entropy).sum();
    entropy(&indices) <= spatial
}

/// Palette indices, packed `8 >> bits` bits each into green.
fn index_image(pixels: &[u32], width: usize, palette: &[u32]) -> (Vec<u32>, usize) {
    let bits: u32 = match palette.len() {
        0..=2 => 3,
        3..=4 => 2,
        5..=16 => 1,
        _ => 0,
    };
    let coded_width = subsample(width as u32, bits) as usize;
    let per_pixel = 8 >> bits;
    let per_byte = 1usize << bits;
    let height = pixels.len() / width;
    let mut out = vec![0xff00_0000u32; coded_width * height];
    for (row, line) in pixels.chunks_exact(width).enumerate() {
        for (x, pixel) in line.iter().enumerate() {
            let index = palette.binary_search(pixel).unwrap_or(0) as u32;
            let shift = (x % per_byte) as u32 * per_pixel;
            out[row * coded_width + x / per_byte] |= index << (8 + shift);
        }
    }
    (out, coded_width)
}

/// Shannon bits of a 256-bin histogram.
fn entropy(histogram: &[u32; 256]) -> f64 {
    let mut total = 0u32;
    let mut sum = 0.0;
    for &count in histogram {
        if count > 0 {
            total += count;
            sum -= f64::from(count) * f64::from(count).log2();
        }
    }
    if total == 0 {
        return 0.0;
    }
    sum + f64::from(total) * f64::from(total).log2()
}

/// `x log2 x`.
#[inline]
fn x_log_x(x: u32) -> f64 {
    if x == 0 {
        0.0
    } else {
        let x = f64::from(x);
        x * x.log2()
    }
}

/// How much the Shannon bits of `previous` grow when `added` joins it,
/// looking only at the bins `added` touches.
fn added_entropy(previous: &[u32; 256], added: &[u32; 256]) -> f64 {
    let mut before = 0u32;
    let mut after = 0u32;
    let mut bins = 0.0;
    for (&p, &a) in previous.iter().zip(added) {
        before += p;
        if a > 0 {
            after += a;
            bins += x_log_x(p + a) - x_log_x(p);
        }
    }
    let total_after = before + after;
    (x_log_x(total_after) - x_log_x(before)) - bins
}

/// Choose a predictor per tile, write the transform, and return the
/// residuals.
fn apply_predictor(
    writer: &mut BitWriter,
    image: &[u32],
    width: usize,
    height: usize,
    bits: u32,
) -> Vec<u32> {
    let tiles_wide = subsample(width as u32, bits) as usize;
    let tiles_high = subsample(height as u32, bits) as usize;
    let tile = 1usize << bits;
    let mut modes = vec![0xff00_0000u32 | (11 << 8); tiles_wide * tiles_high];
    // Residuals so far, per channel, to favour modes that agree with them.
    let mut accumulated = [[0u32; 256]; 4];
    let neighbours = |x: usize, y: usize| {
        let at = y * width + x;
        let up = at - width;
        // The top-right of the last column is the first pixel of this row.
        let top_right = image[up + 1];
        (image[at - 1], image[up], image[up - 1], top_right)
    };
    for ty in 0..tiles_high {
        for tx in 0..tiles_wide {
            let x0 = tx * tile;
            let y0 = ty * tile;
            let x1 = (x0 + tile).min(width);
            let y1 = (y0 + tile).min(height);
            let mut best = (f64::INFINITY, 11u32, [[0u32; 256]; 4]);
            for mode in 0..14u32 {
                let mut histogram = [[0u32; 256]; 4];
                for y in y0.max(1)..y1 {
                    for x in x0.max(1)..x1 {
                        let (left, top, top_left, top_right) = neighbours(x, y);
                        let residual = sub_pixels(
                            image[y * width + x],
                            predict(mode, left, top, top_left, top_right),
                        );
                        for (channel, counts) in histogram.iter_mut().enumerate() {
                            counts[((residual >> (8 * channel)) & 0xff) as usize] += 1;
                        }
                    }
                }
                let mut cost = 0.0;
                for (counts, previous) in histogram.iter().zip(&accumulated) {
                    cost += added_entropy(previous, counts);
                }
                if cost < best.0 {
                    best = (cost, mode, histogram);
                }
            }
            modes[ty * tiles_wide + tx] = 0xff00_0000 | (best.1 << 8);
            for (counts, previous) in best.2.iter().zip(accumulated.iter_mut()) {
                for (a, b) in previous.iter_mut().zip(counts) {
                    *a += b;
                }
            }
        }
    }
    writer.write(1, 1);
    writer.write(0, 2);
    writer.write(bits - 2, 3);
    write_image(
        writer,
        &modes,
        tiles_wide,
        tiles_high,
        &Effort::SUB_IMAGE,
        false,
    );
    let mut residuals = vec![0u32; image.len()];
    for y in 0..height {
        for x in 0..width {
            let at = y * width + x;
            let guess = if y == 0 {
                if x == 0 { 0xff00_0000 } else { image[at - 1] }
            } else if x == 0 {
                image[at - width]
            } else {
                let mode = (modes[(y >> bits) * tiles_wide + (x >> bits)] >> 8) & 0xf;
                let (left, top, top_left, top_right) = neighbours(x, y);
                predict(mode, left, top, top_left, top_right)
            };
            residuals[at] = sub_pixels(image[at], guess);
        }
    }
    residuals
}

/// Red after removing what green predicts.
#[inline]
fn red_residual(pixel: u32, green_to_red: u8) -> u8 {
    let green = ((pixel >> 8) & 0xff) as u8;
    let red = ((pixel >> 16) & 0xff) as i32;
    ((red - color_delta(green_to_red, green)) & 0xff) as u8
}

/// Blue after removing what green and red predict.
#[inline]
fn blue_residual(pixel: u32, green_to_blue: u8, red_to_blue: u8) -> u8 {
    let green = ((pixel >> 8) & 0xff) as u8;
    let red = ((pixel >> 16) & 0xff) as u8;
    let blue = (pixel & 0xff) as i32;
    ((blue - color_delta(green_to_blue, green) - color_delta(red_to_blue, red)) & 0xff) as u8
}

/// The multiplier in `-128..128` (as a byte) that `cost` likes best,
/// coarse to fine.
fn search(mut cost: impl FnMut(u8) -> f64) -> u8 {
    let mut best = (cost(0), 0i32);
    for value in (-128..128).step_by(16) {
        let c = cost(value as i8 as u8);
        if c < best.0 {
            best = (c, value);
        }
    }
    let mut step = 8;
    while step > 0 {
        let center = best.1;
        for value in [center - step, center + step] {
            if (-128..128).contains(&value) {
                let c = cost(value as i8 as u8);
                if c < best.0 {
                    best = (c, value);
                }
            }
        }
        step /= 2;
    }
    best.1 as i8 as u8
}

/// Choose cross-color multipliers per tile, write the transform, and
/// return the transformed image.
fn apply_cross_color(
    writer: &mut BitWriter,
    image: &[u32],
    width: usize,
    height: usize,
    bits: u32,
) -> Vec<u32> {
    let bits = bits.min(9);
    let tiles_wide = subsample(width as u32, bits) as usize;
    let tiles_high = subsample(height as u32, bits) as usize;
    let tile = 1usize << bits;
    let mut multipliers = vec![0xff00_0000u32; tiles_wide * tiles_high];
    let mut out = image.to_vec();
    for ty in 0..tiles_high {
        for tx in 0..tiles_wide {
            let x0 = tx * tile;
            let y0 = ty * tile;
            let x1 = (x0 + tile).min(width);
            let y1 = (y0 + tile).min(height);
            let tile_pixels: Vec<u32> = (y0..y1)
                .flat_map(|y| image[y * width + x0..y * width + x1].iter().copied())
                .collect();
            let histogram_cost = |values: &mut dyn Iterator<Item = u8>| {
                let mut counts = [0u32; 256];
                for value in values {
                    counts[usize::from(value)] += 1;
                }
                entropy(&counts)
            };
            let green_to_red =
                search(|m| histogram_cost(&mut tile_pixels.iter().map(|&p| red_residual(p, m))));
            let green_to_blue = search(|m| {
                histogram_cost(&mut tile_pixels.iter().map(|&p| blue_residual(p, m, 0)))
            });
            let red_to_blue = search(|m| {
                histogram_cost(
                    &mut tile_pixels
                        .iter()
                        .map(|&p| blue_residual(p, green_to_blue, m)),
                )
            });
            multipliers[ty * tiles_wide + tx] = 0xff00_0000
                | (u32::from(red_to_blue) << 16)
                | (u32::from(green_to_blue) << 8)
                | u32::from(green_to_red);
            for y in y0..y1 {
                for x in x0..x1 {
                    let pixel = image[y * width + x];
                    let red = red_residual(pixel, green_to_red);
                    let blue = blue_residual(pixel, green_to_blue, red_to_blue);
                    out[y * width + x] =
                        (pixel & 0xff00_ff00) | (u32::from(red) << 16) | u32::from(blue);
                }
            }
        }
    }
    writer.write(1, 1);
    writer.write(1, 2);
    writer.write(bits - 2, 3);
    write_image(
        writer,
        &multipliers,
        tiles_wide,
        tiles_high,
        &Effort::SUB_IMAGE,
        false,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::{Config, encode, sub_pixels};
    use crate::lossless::decode::{add_pixels, decode};

    #[test]
    fn sub_pixels_inverts_add_pixels() {
        for (a, b) in [
            (0x0102_0304u32, 0xff80_7f01u32),
            (0, 0xffff_ffff),
            (0x8000_0001, 0x7fff_ffff),
        ] {
            assert_eq!(add_pixels(sub_pixels(a, b), b), a);
        }
    }

    fn round_trip(argb: &[u32], width: u32, height: u32) {
        for effort in 0..=9 {
            let config = Config {
                effort,
                exact: true,
            };
            let bytes = encode(argb, width, height, config);
            let (header, decoded) = decode(&bytes).expect("decode");
            assert_eq!((header.width, header.height), (width, height));
            assert!(decoded == argb, "effort {effort}");
        }
    }

    #[test]
    fn every_effort_round_trips() {
        let (width, height) = (37u32, 23u32);
        let mut seed = 7u32;
        let noisy: Vec<u32> = (0..width * height)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let x = i % width;
                let y = i / width;
                (((x * 7) & 0xff) << 16)
                    | (((y * 11) & 0xff) << 8)
                    | ((seed >> 24) & 0x0f)
                    | 0xff00_0000
            })
            .collect();
        round_trip(&noisy, width, height);
        // Gradients make the top-right modes win, including at the last
        // column, whose top-right is the first pixel of its own row.
        let rgba = crate::tests::oracle_synthetic(97, 61);
        let gradient: Vec<u32> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| u32::from_be_bytes([p[3], p[0], p[1], p[2]]))
            .collect();
        round_trip(&gradient, 97, 61);
        let few: Vec<u32> = (0..width * height)
            .map(|i| [0xff00_0000, 0x80ff_0000, 0x0000_00ff][(i % 7 % 3) as usize])
            .collect();
        round_trip(&few, width, height);
        round_trip(&[0x1234_5678], 1, 1);
        let ramp: Vec<u32> = (0..300u32)
            .map(|i| 0xff00_0000 | (i * 0x0001_0203))
            .collect();
        round_trip(&ramp, 300, 1);
        round_trip(&ramp, 1, 300);
    }
}
