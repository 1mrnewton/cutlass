//! A palette of at most 256 colors for one frame, by median cut over the
//! frame's colors at five bits a channel.

use std::collections::HashMap;

/// The palette and each pixel's index into it.
pub(crate) struct Indexed {
    /// RGB triples. The length is a power of two from 2 to 256.
    pub palette: Vec<[u8; 3]>,
    /// log2 of the palette length.
    pub bits: u8,
    pub indexes: Vec<u8>,
}

#[derive(Clone)]
struct Bucket {
    /// Five-bit color key and how many pixels have it.
    colors: Vec<(u16, u32)>,
}

fn key(rgb: [u8; 3]) -> u16 {
    (u16::from(rgb[0] >> 3) << 10) | (u16::from(rgb[1] >> 3) << 5) | u16::from(rgb[2] >> 3)
}

fn channel(key: u16, which: usize) -> u8 {
    ((key >> (10 - 5 * which)) & 31) as u8
}

impl Bucket {
    fn count(&self) -> u64 {
        self.colors.iter().map(|(_, count)| u64::from(*count)).sum()
    }

    /// The channel with the widest spread, and that spread.
    fn widest(&self) -> (usize, u8) {
        (0..3)
            .map(|which| {
                let (low, high) = self
                    .colors
                    .iter()
                    .fold((31u8, 0u8), |(low, high), (key, _)| {
                        let value = channel(*key, which);
                        (low.min(value), high.max(value))
                    });
                (which, high.saturating_sub(low))
            })
            .max_by_key(|(_, spread)| *spread)
            .unwrap_or((0, 0))
    }

    fn split(mut self) -> (Bucket, Bucket) {
        let (which, _) = self.widest();
        self.colors.sort_by_key(|(key, _)| channel(*key, which));
        let half = self.count() / 2;
        let mut seen = 0u64;
        let mut cut = 1;
        for (at, (_, count)) in self.colors.iter().enumerate() {
            seen += u64::from(*count);
            if seen >= half {
                cut = (at + 1).clamp(1, self.colors.len() - 1);
                break;
            }
        }
        let upper = self.colors.split_off(cut);
        (self, Bucket { colors: upper })
    }

    fn mean(&self, pixels: &HashMap<u16, [u64; 4]>) -> [u8; 3] {
        let mut sum = [0u64; 4];
        for (key, _) in &self.colors {
            if let Some(total) = pixels.get(key) {
                for channel in 0..4 {
                    sum[channel] += total[channel];
                }
            }
        }
        let count = sum[3].max(1);
        [
            (sum[0] / count) as u8,
            (sum[1] / count) as u8,
            (sum[2] / count) as u8,
        ]
    }
}

/// Index `rgba` (tightly packed, alpha ignored) into a palette of at most
/// 256 colors.
pub(crate) fn quantize(rgba: &[u8]) -> Indexed {
    // Per five-bit key: summed full-precision RGB and the pixel count.
    let mut pixels: HashMap<u16, [u64; 4]> = HashMap::new();
    for pixel in rgba.as_chunks::<4>().0 {
        let entry = pixels
            .entry(key([pixel[0], pixel[1], pixel[2]]))
            .or_insert([0; 4]);
        entry[0] += u64::from(pixel[0]);
        entry[1] += u64::from(pixel[1]);
        entry[2] += u64::from(pixel[2]);
        entry[3] += 1;
    }
    let mut buckets = vec![Bucket {
        colors: pixels
            .iter()
            .map(|(key, total)| (*key, u32::try_from(total[3]).unwrap_or(u32::MAX)))
            .collect(),
    }];
    while buckets.len() < 256 {
        let Some((at, _)) = buckets
            .iter()
            .enumerate()
            .filter(|(_, bucket)| bucket.colors.len() > 1)
            .max_by_key(|(_, bucket)| (bucket.widest().1, bucket.count()))
        else {
            break;
        };
        let (low, high) = buckets.swap_remove(at).split();
        buckets.push(low);
        buckets.push(high);
    }
    let mut palette: Vec<[u8; 3]> = buckets.iter().map(|bucket| bucket.mean(&pixels)).collect();
    let mut lookup: HashMap<u16, u8> = HashMap::new();
    for (index, bucket) in buckets.iter().enumerate() {
        for (key, _) in &bucket.colors {
            lookup.insert(*key, u8::try_from(index).unwrap_or(255));
        }
    }
    let mut bits = 1u8;
    while (1usize << bits) < palette.len() {
        bits += 1;
    }
    palette.resize(1 << bits, [0, 0, 0]);
    let indexes = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            lookup
                .get(&key([pixel[0], pixel[1], pixel[2]]))
                .copied()
                .unwrap_or(0)
        })
        .collect();
    Indexed {
        palette,
        bits,
        indexes,
    }
}

#[cfg(test)]
mod tests {
    use super::quantize;

    #[test]
    fn few_colors_are_kept_exactly() {
        let mut rgba = Vec::new();
        for color in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]] {
            rgba.extend(color);
            rgba.push(255);
        }
        let indexed = quantize(&rgba);
        assert_eq!(indexed.palette.len(), 4);
        for (pixel, index) in rgba.as_chunks::<4>().0.iter().zip(&indexed.indexes) {
            assert_eq!(
                indexed.palette[usize::from(*index)],
                [pixel[0], pixel[1], pixel[2]]
            );
        }
    }

    #[test]
    fn a_gradient_lands_within_a_small_error() {
        let mut rgba = Vec::new();
        for x in 0..256u32 {
            for y in 0..64u32 {
                rgba.extend([x as u8, (y * 4) as u8, ((x + y) / 2) as u8, 255]);
            }
        }
        let indexed = quantize(&rgba);
        assert!(indexed.palette.len() <= 256);
        let worst = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .zip(&indexed.indexes)
            .map(|(pixel, index)| {
                let color = indexed.palette[usize::from(*index)];
                (0..3)
                    .map(|c| i32::from(pixel[c]).abs_diff(i32::from(color[c])))
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert!(worst <= 40, "worst channel error {worst}");
    }
}
