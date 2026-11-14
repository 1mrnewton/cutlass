//! A TIFF picture from a color strike, decoded to premultiplied RGBA.
//!
//! Read: the baseline kinds a font tool writes. One picture in strips,
//! eight bits a sample, in gray of either sense, a palette, or RGB, with or
//! without one sample of alpha, stored plain, as PackBits or as LZW, with or
//! without each sample stored as its step from the one before.
//!
//! Refused as unsupported: tiles, a plane for each sample, other depths,
//! other packings (Deflate, JPEG and the fax codes), a picture stored on
//! its side, and colors other than those above.

use cutlass_font::FontError;

use crate::TextError;

/// Most fields read from the picture's directory.
const MAX_FIELDS: usize = 512;

/// The codes an LZW table holds at most: twelve bits of them.
const LZW_CODES: usize = 4_096;
const LZW_CLEAR: usize = 256;
const LZW_END: usize = 257;

const TRUNCATED: TextError = TextError::Font(FontError::Truncated);
const BAD: TextError = TextError::Font(FontError::BadTable);
const UNSUPPORTED: TextError = TextError::Font(FontError::Unsupported);

/// Decode `picture`, whose directory declares `width` by `height`, to
/// premultiplied RGBA with row 0 at the top. The caller has checked the
/// size against what the atlas holds.
pub(super) fn decode(picture: &[u8], width: u32, height: u32) -> Result<Vec<u8>, TextError> {
    let file = File::open(picture)?;
    let fields = file.fields()?;
    // The reader of the font and this one read the same directory.
    if fields.width != width || fields.height != height {
        return Err(BAD);
    }
    let layout = fields.layout()?;
    let samples = file.samples(&fields, &layout)?;
    layout.color(&fields, &samples)
}

/// The picture's bytes and the order its numbers are stored in.
struct File<'a> {
    bytes: &'a [u8],
    big: bool,
}

/// What the directory says of the picture, with the defaults the format
/// gives a field that is absent.
struct Fields {
    width: u32,
    height: u32,
    bits: Vec<u32>,
    packing: u32,
    sense: Option<u32>,
    strip_starts: Vec<u32>,
    strip_lengths: Vec<u32>,
    samples: u32,
    rows_per_strip: u32,
    planes: u32,
    predictor: u32,
    palette: Vec<u32>,
    extra: Vec<u32>,
    turned: u32,
    tiled: bool,
}

/// How a pixel's samples make a color.
enum Color {
    /// `true` when white is zero.
    Gray(bool),
    Palette,
    Rgb,
}

/// What the sample after the color ones is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Alpha {
    None,
    /// The colors are already multiplied by it.
    Multiplied,
    Straight,
}

struct Layout {
    color: Color,
    alpha: Alpha,
    /// Samples stored for one pixel.
    stored: usize,
}

impl<'a> File<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self, TextError> {
        let big = match bytes.get(..4) {
            Some(b"MM\0*") => true,
            Some(b"II*\0") => false,
            Some(_) => return Err(BAD),
            None => return Err(TRUNCATED),
        };
        Ok(Self { bytes, big })
    }

    fn short(&self, at: usize) -> Result<u32, TextError> {
        let end = at.checked_add(2).ok_or(BAD)?;
        let pair: [u8; 2] = self
            .bytes
            .get(at..end)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(TRUNCATED)?;
        Ok(u32::from(if self.big {
            u16::from_be_bytes(pair)
        } else {
            u16::from_le_bytes(pair)
        }))
    }

    fn long(&self, at: usize) -> Result<u32, TextError> {
        let end = at.checked_add(4).ok_or(BAD)?;
        let four: [u8; 4] = self
            .bytes
            .get(at..end)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(TRUNCATED)?;
        Ok(if self.big {
            u32::from_be_bytes(four)
        } else {
            u32::from_le_bytes(four)
        })
    }

    /// The numbers of the field at `at`: bytes, 16-bit or 32-bit numbers,
    /// at most `most` of them. They sit in the field when four bytes hold
    /// them and where the field points when not.
    fn numbers(&self, at: usize, most: usize) -> Result<Vec<u32>, TextError> {
        let kind = self.short(at + 2)?;
        let count = usize::try_from(self.long(at + 4)?).map_err(|_| BAD)?;
        if count > most {
            return Err(TextError::Limit);
        }
        let size = match kind {
            1 => 1,
            3 => 2,
            4 => 4,
            _ => return Err(BAD),
        };
        let start = if count * size <= 4 {
            at + 8
        } else {
            usize::try_from(self.long(at + 8)?).map_err(|_| BAD)?
        };
        (0..count)
            .map(|index| {
                let at = start.checked_add(index * size).ok_or(BAD)?;
                match size {
                    1 => self
                        .bytes
                        .get(at)
                        .map(|byte| u32::from(*byte))
                        .ok_or(TRUNCATED),
                    2 => self.short(at),
                    _ => self.long(at),
                }
            })
            .collect()
    }

    /// One number, the first when a field holds several.
    fn number(&self, at: usize) -> Result<u32, TextError> {
        self.numbers(at, 4)?.first().copied().ok_or(BAD)
    }

    fn fields(&self) -> Result<Fields, TextError> {
        let directory = usize::try_from(self.long(4)?).map_err(|_| BAD)?;
        let count = usize::try_from(self.short(directory)?).map_err(|_| BAD)?;
        let mut fields = Fields {
            width: 0,
            height: 0,
            bits: vec![1],
            packing: 1,
            sense: None,
            strip_starts: Vec::new(),
            strip_lengths: Vec::new(),
            samples: 1,
            rows_per_strip: u32::MAX,
            planes: 1,
            predictor: 1,
            palette: Vec::new(),
            extra: Vec::new(),
            turned: 1,
            tiled: false,
        };
        for index in 0..count.min(MAX_FIELDS) {
            let at = directory.checked_add(2 + index * 12).ok_or(BAD)?;
            // A strip for every row is the most a picture has.
            let rows = usize::try_from(fields.height.clamp(1, 1 << 16)).map_err(|_| BAD)?;
            match self.short(at)? {
                256 => fields.width = self.number(at)?,
                257 => fields.height = self.number(at)?,
                258 => fields.bits = self.numbers(at, 8)?,
                259 => fields.packing = self.number(at)?,
                262 => fields.sense = Some(self.number(at)?),
                273 => fields.strip_starts = self.numbers(at, rows)?,
                274 => fields.turned = self.number(at)?,
                277 => fields.samples = self.number(at)?,
                278 => fields.rows_per_strip = self.number(at)?,
                279 => fields.strip_lengths = self.numbers(at, rows)?,
                284 => fields.planes = self.number(at)?,
                317 => fields.predictor = self.number(at)?,
                320 => fields.palette = self.numbers(at, 3 * 256)?,
                322..=325 => fields.tiled = true,
                338 => fields.extra = self.numbers(at, 8)?,
                _ => {}
            }
        }
        if fields.width == 0 || fields.height == 0 {
            return Err(BAD);
        }
        Ok(fields)
    }

    /// Every sample of the picture, row after row, unpacked from its
    /// strips and with the steps between samples added back up.
    fn samples(&self, fields: &Fields, layout: &Layout) -> Result<Vec<u8>, TextError> {
        let width = usize::try_from(fields.width).map_err(|_| TextError::Limit)?;
        let height = usize::try_from(fields.height).map_err(|_| TextError::Limit)?;
        let row = width.checked_mul(layout.stored).ok_or(TextError::Limit)?;
        let all = row.checked_mul(height).ok_or(TextError::Limit)?;
        let rows_per_strip = usize::try_from(fields.rows_per_strip)
            .unwrap_or(usize::MAX)
            .min(height);
        if rows_per_strip == 0 {
            return Err(BAD);
        }
        let mut out = Vec::with_capacity(all);
        let strips = fields.strip_starts.iter().zip(&fields.strip_lengths);
        for (start, length) in strips {
            if out.len() >= all {
                break;
            }
            let need = (rows_per_strip * row).min(all - out.len());
            let start = usize::try_from(*start).map_err(|_| BAD)?;
            let end = usize::try_from(*length)
                .ok()
                .and_then(|length| start.checked_add(length))
                .ok_or(BAD)?;
            let strip = self.bytes.get(start..end).ok_or(TRUNCATED)?;
            match fields.packing {
                1 => out.extend_from_slice(strip.get(..need).ok_or(TRUNCATED)?),
                5 => unpack_lzw(strip, need, &mut out)?,
                32_773 => unpack_bits(strip, need, &mut out)?,
                _ => return Err(UNSUPPORTED),
            }
        }
        if out.len() != all {
            return Err(TRUNCATED);
        }
        match fields.predictor {
            1 => {}
            2 => {
                for line in out.chunks_exact_mut(row) {
                    for at in layout.stored..line.len() {
                        line[at] = line[at].wrapping_add(line[at - layout.stored]);
                    }
                }
            }
            _ => return Err(UNSUPPORTED),
        }
        Ok(out)
    }
}

impl Fields {
    fn layout(&self) -> Result<Layout, TextError> {
        if self.tiled || self.planes != 1 || self.turned != 1 {
            return Err(UNSUPPORTED);
        }
        if self.bits.iter().any(|bits| *bits != 8) {
            return Err(UNSUPPORTED);
        }
        let (color, colors) = match self.sense.ok_or(BAD)? {
            0 => (Color::Gray(true), 1),
            1 => (Color::Gray(false), 1),
            2 => (Color::Rgb, 3),
            3 => (Color::Palette, 1),
            _ => return Err(UNSUPPORTED),
        };
        let stored = usize::try_from(self.samples).map_err(|_| BAD)?;
        // One more sample than the color needs is alpha when the picture
        // says so, and is passed over when it does not.
        let alpha = match (stored.checked_sub(colors), self.extra.first()) {
            (Some(0), _) => Alpha::None,
            (Some(1), Some(1)) => Alpha::Multiplied,
            (Some(1), Some(2)) => Alpha::Straight,
            (Some(1), _) => Alpha::None,
            _ => return Err(UNSUPPORTED),
        };
        if matches!(color, Color::Palette) && self.palette.len() != 3 * 256 {
            return Err(BAD);
        }
        Ok(Layout {
            color,
            alpha,
            stored,
        })
    }
}

impl Layout {
    /// The picture's samples as premultiplied RGBA.
    fn color(&self, fields: &Fields, samples: &[u8]) -> Result<Vec<u8>, TextError> {
        let mut out = Vec::with_capacity(samples.len() / self.stored * 4);
        // A palette's colors are 16 bits each, every red, then every
        // green, then every blue.
        let entry = |index: u8, part: usize| {
            let value = fields
                .palette
                .get(part * 256 + usize::from(index))
                .copied()
                .unwrap_or(0);
            u8::try_from(value >> 8).unwrap_or(u8::MAX)
        };
        for pixel in samples.chunks_exact(self.stored) {
            let (rgb, used) = match self.color {
                Color::Gray(white_is_zero) => {
                    let gray = if white_is_zero {
                        u8::MAX - pixel[0]
                    } else {
                        pixel[0]
                    };
                    ([gray; 3], 1)
                }
                Color::Palette => (
                    [entry(pixel[0], 0), entry(pixel[0], 1), entry(pixel[0], 2)],
                    1,
                ),
                Color::Rgb => ([pixel[0], pixel[1], pixel[2]], 3),
            };
            let alpha = pixel.get(used).copied().unwrap_or(u8::MAX);
            match self.alpha {
                Alpha::None => out.extend([rgb[0], rgb[1], rgb[2], u8::MAX]),
                // A color past its alpha is not a color: it is held to it.
                Alpha::Multiplied => out.extend([
                    rgb[0].min(alpha),
                    rgb[1].min(alpha),
                    rgb[2].min(alpha),
                    alpha,
                ]),
                Alpha::Straight => super::push(&mut out, [rgb[0], rgb[1], rgb[2], alpha]),
            }
        }
        Ok(out)
    }
}

/// Unpack `need` bytes of PackBits from `strip` onto `out`: a count, then
/// that many bytes as they are or one byte that many times.
fn unpack_bits(strip: &[u8], need: usize, out: &mut Vec<u8>) -> Result<(), TextError> {
    let end = out.len() + need;
    let mut at = 0;
    while out.len() < end {
        let count = i8::from_be_bytes([*strip.get(at).ok_or(TRUNCATED)?]);
        at += 1;
        let run = usize::from(count.unsigned_abs()) + 1;
        if count >= 0 {
            out.extend_from_slice(strip.get(at..at + run).ok_or(TRUNCATED)?);
            at += run;
        } else if count != i8::MIN {
            let byte = *strip.get(at).ok_or(TRUNCATED)?;
            out.extend(std::iter::repeat_n(byte, run));
            at += 1;
        }
    }
    out.truncate(end);
    Ok(())
}

/// Unpack `need` bytes of LZW from `strip` onto `out`, as TIFF writes it:
/// codes high bit first, from nine bits to twelve, each a byte, a string
/// seen before, the order to start the table again, or the end. A code
/// grows by a bit one string sooner than in a GIF.
fn unpack_lzw(strip: &[u8], need: usize, out: &mut Vec<u8>) -> Result<(), TextError> {
    let end = out.len() + need;
    // Each string is another string and one more byte.
    let mut before = [0_u16; LZW_CODES];
    let mut last = [0_u8; LZW_CODES];
    let mut first = [0_u8; LZW_CODES];
    let mut length = [0_u16; LZW_CODES];
    for byte in 0..=u8::MAX {
        let code = usize::from(byte);
        last[code] = byte;
        first[code] = byte;
        length[code] = 1;
    }
    let mut next = LZW_END + 1;
    let mut width = 9_u32;
    let mut previous: Option<usize> = None;
    let (mut bits, mut held, mut at) = (0_u32, 0_u32, 0_usize);
    while out.len() < end {
        while held < width {
            bits = (bits << 8) | u32::from(*strip.get(at).ok_or(TRUNCATED)?);
            held += 8;
            at += 1;
        }
        held -= width;
        let code = usize::try_from((bits >> held) & ((1 << width) - 1)).map_err(|_| BAD)?;
        if code == LZW_END {
            break;
        }
        if code == LZW_CLEAR {
            next = LZW_END + 1;
            width = 9;
            previous = None;
            continue;
        }
        // A code the table does not hold yet is the string before it and
        // that string's first byte again.
        let known = code < LZW_CLEAR || (code > LZW_END && code < next);
        let (string, extra) = match (known, previous) {
            (true, _) => (code, None),
            (false, Some(previous)) if code == next => (previous, Some(first[previous])),
            _ => return Err(BAD),
        };
        let start = out.len();
        let size = usize::from(length[string]);
        out.resize(start + size, 0);
        let mut link = string;
        for slot in out[start..].iter_mut().rev() {
            *slot = last[link];
            link = usize::from(before[link]);
        }
        out.extend(extra);
        if let Some(previous) = previous
            && next < LZW_CODES
        {
            before[next] = u16::try_from(previous).map_err(|_| BAD)?;
            last[next] = first[string];
            first[next] = first[previous];
            length[next] = length[previous].saturating_add(1);
            next += 1;
        }
        if next + 1 >= (1 << width) && width < 12 {
            width += 1;
        }
        previous = Some(code);
    }
    if out.len() < end {
        return Err(TRUNCATED);
    }
    out.truncate(end);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: u32 = 48;
    const HEIGHT: u32 = 32;

    /// The picture every fixture was written from, as straight RGBA: steps
    /// of color on the left half, with alpha that runs out lower down, and
    /// on the right half numbers from a rule that never settles.
    fn pattern() -> Vec<[u8; 4]> {
        let mut state = 7_u32;
        let mut noise = || {
            state = (state.wrapping_mul(1_103_515_245).wrapping_add(12_345)) & 0x7FFF_FFFF;
            u8::try_from((state >> 16) & 0xFF).unwrap_or(0)
        };
        let byte = |value: u32| u8::try_from(value % 256).unwrap_or(0);
        let mut out = Vec::new();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                if x < 24 {
                    let alpha = if y < 16 { 255 } else { byte(x * 11) };
                    out.push([byte(x * 10), byte(y * 8), byte((x + y) * 5), alpha]);
                } else {
                    let (red, green, blue, alpha) = (noise(), noise(), noise(), noise());
                    out.push([red, green, blue, [0, 64, 128, 255][usize::from(alpha & 3)]]);
                }
            }
        }
        out
    }

    fn premultiplied(pixels: impl Iterator<Item = [u8; 4]>) -> Vec<u8> {
        let mut out = Vec::new();
        for pixel in pixels {
            super::super::push(&mut out, pixel);
        }
        out
    }

    fn opaque(pixel: [u8; 4]) -> [u8; 4] {
        [pixel[0], pixel[1], pixel[2], 255]
    }

    #[test]
    fn each_packing_of_a_color_picture_decodes_to_the_pixels_it_was_written_from() {
        let wanted = premultiplied(pattern().into_iter().map(opaque));
        // PackBits and LZW, each as FFmpeg writes it, low byte first.
        for fixture in [
            &include_bytes!("../../../fixtures/tiff/rgb_packbits.tiff")[..],
            &include_bytes!("../../../fixtures/tiff/rgb_lzw.tiff")[..],
        ] {
            assert_eq!(decode(fixture, WIDTH, HEIGHT).expect("picture"), wanted);
        }
    }

    #[test]
    fn alpha_is_read_straight_or_already_multiplied() {
        let wanted = premultiplied(pattern().into_iter());
        let straight = include_bytes!("../../../fixtures/tiff/rgba_lzw.tiff");
        assert_eq!(decode(straight, WIDTH, HEIGHT).expect("picture"), wanted);
        // The Mac writes its numbers high byte first and stores each
        // sample as its step from the one before.
        let mac = include_bytes!("../../../fixtures/tiff/mac_rgba_lzw.tiff");
        assert_eq!(&mac[..4], b"MM\0*");
        assert_eq!(decode(mac, WIDTH, HEIGHT).expect("picture"), wanted);

        // The same samples called multiplied are kept as they are, held
        // to their alpha. The field is the one of kind 3 numbered 338.
        let mut multiplied = straight.to_vec();
        let file = File::open(straight).expect("file");
        let directory = usize::try_from(file.long(4).expect("directory")).expect("fits");
        let fields = usize::try_from(file.short(directory).expect("count")).expect("fits");
        let extra = (0..fields)
            .map(|index| directory + 2 + index * 12)
            .find(|at| file.short(*at) == Ok(338))
            .expect("extra samples");
        assert_eq!(multiplied[extra + 8], 2);
        multiplied[extra + 8] = 1;
        let held: Vec<u8> = pattern()
            .into_iter()
            .flat_map(|[red, green, blue, alpha]| {
                [red.min(alpha), green.min(alpha), blue.min(alpha), alpha]
            })
            .collect();
        assert_eq!(decode(&multiplied, WIDTH, HEIGHT).expect("picture"), held);
        // Called nothing at all, the sample is passed over.
        multiplied[extra + 8] = 0;
        assert_eq!(
            decode(&multiplied, WIDTH, HEIGHT).expect("picture"),
            premultiplied(pattern().into_iter().map(opaque))
        );
    }

    #[test]
    fn gray_and_a_palette_become_color() {
        let gray = |[red, _, _, _]: [u8; 4]| [red, red, red, 255];
        let packed = include_bytes!("../../../fixtures/tiff/gray_packbits.tiff");
        assert_eq!(
            decode(packed, WIDTH, HEIGHT).expect("gray"),
            premultiplied(pattern().into_iter().map(gray))
        );
        let with_alpha = include_bytes!("../../../fixtures/tiff/gray_alpha_plain.tiff");
        assert_eq!(
            decode(with_alpha, WIDTH, HEIGHT).expect("gray and alpha"),
            premultiplied(
                pattern()
                    .into_iter()
                    .map(|[red, _, _, alpha]| [red, red, red, alpha])
            )
        );
        // White as zero turns every gray over.
        let mut turned = packed.to_vec();
        let file = File::open(packed).expect("file");
        let directory = usize::try_from(file.long(4).expect("directory")).expect("fits");
        let fields = usize::try_from(file.short(directory).expect("count")).expect("fits");
        let sense = (0..fields)
            .map(|index| directory + 2 + index * 12)
            .find(|at| file.short(*at) == Ok(262))
            .expect("sense");
        turned[sense + 8] = 0;
        assert_eq!(
            decode(&turned, WIDTH, HEIGHT).expect("turned"),
            premultiplied(pattern().into_iter().map(|[red, _, _, _]| [
                255 - red,
                255 - red,
                255 - red,
                255
            ]))
        );
        // A palette, as FFmpeg reads the same file back.
        let palette = include_bytes!("../../../fixtures/tiff/palette_lzw.tiff");
        let read_back = include_bytes!("../../../fixtures/tiff/palette_lzw.rgb");
        let wanted: Vec<u8> = read_back
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|[red, green, blue]| [*red, *green, *blue, 255])
            .collect();
        assert_eq!(decode(palette, WIDTH, HEIGHT).expect("palette"), wanted);
    }

    #[test]
    fn a_picture_this_does_not_read_is_refused_by_name() {
        let picture = include_bytes!("../../../fixtures/tiff/rgb_lzw.tiff");
        let mac = include_bytes!("../../../fixtures/tiff/mac_rgba_lzw.tiff");
        // The picture with the one number of field `tag` made `value`.
        let with = |picture: &[u8], tag: u32, value: u8| {
            let file = File::open(picture).expect("file");
            let directory = usize::try_from(file.long(4).expect("directory")).expect("fits");
            let fields = usize::try_from(file.short(directory).expect("count")).expect("fits");
            let field = (0..fields)
                .map(|index| directory + 2 + index * 12)
                .find(|at| file.short(*at) == Ok(tag))
                .expect("field");
            let mut changed = picture.to_vec();
            changed[field + 8 + usize::from(file.big)] = value;
            decode(&changed, WIDTH, HEIGHT)
        };
        // Deflate, JPEG, a fax code; CMYK.
        for (tag, value) in [(259, 8), (259, 7), (259, 3), (262, 5)] {
            assert!(
                matches!(with(picture, tag, value), Err(UNSUPPORTED)),
                "{tag} as {value}"
            );
        }
        // On its side; a plane for each sample; steps in floating point.
        for (tag, value) in [(274, 6), (284, 2), (317, 3)] {
            assert!(
                matches!(with(mac, tag, value), Err(UNSUPPORTED)),
                "{tag} as {value}"
            );
        }
        // The size the font was told is the size the picture must say.
        assert!(matches!(decode(picture, WIDTH, HEIGHT + 1), Err(BAD)));
        assert!(matches!(decode(b"%PDF-1.4", 2, 2), Err(BAD)));
        assert!(matches!(decode(b"II", 2, 2), Err(TRUNCATED)));
    }

    #[test]
    fn a_torn_picture_is_an_error_and_never_a_panic_or_a_wait() {
        let started = std::time::Instant::now();
        for fixture in [
            &include_bytes!("../../../fixtures/tiff/rgb_packbits.tiff")[..],
            &include_bytes!("../../../fixtures/tiff/rgba_lzw.tiff")[..],
            &include_bytes!("../../../fixtures/tiff/mac_rgba_lzw.tiff")[..],
            &include_bytes!("../../../fixtures/tiff/palette_lzw.tiff")[..],
        ] {
            // Cut anywhere.
            for end in (0..fixture.len()).step_by(7) {
                let _ = decode(&fixture[..end], WIDTH, HEIGHT);
            }
            // Bytes turned anywhere, by a rule that repeats.
            let mut state = 99_u32;
            for _ in 0..600 {
                let mut torn = fixture.to_vec();
                for _ in 0..3 {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let at = usize::try_from(state >> 8).unwrap_or(0) % torn.len();
                    torn[at] ^= u8::try_from(state & 0xFF).unwrap_or(1) | 1;
                }
                if let Ok(pixels) = decode(&torn, WIDTH, HEIGHT) {
                    assert_eq!(pixels.len(), 48 * 32 * 4);
                }
            }
        }
        // PackBits that only ever says "nothing", and LZW that only ever
        // starts its table again, run out of bytes and stop.
        let mut out = Vec::new();
        assert!(unpack_bits(&[0x80; 100_000], 1_000, &mut out).is_err());
        // 256 in nine bits, over and over.
        let clears: Vec<u8> = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00]
            .iter()
            .copied()
            .cycle()
            .take(90_000)
            .collect();
        assert!(unpack_lzw(&clears, 1_000, &mut out).is_err());
        assert!(out.is_empty());
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }
}
