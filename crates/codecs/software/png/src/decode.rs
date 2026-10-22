//! Chunk walk and zlib inflate. Painting lives in `expand`.

use cutlass_core::Still;

use crate::crc;
use crate::error::{ChunkKind, Error};
use crate::expand;
use crate::image::{self, ColorKind, Info, Key};

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

/// Decodes one PNG still.
///
/// The bytes are the file. A missing chunk, a bad checksum, or a picture past
/// the limit is an error. Gamma and an EXIF orientation are not applied.
pub fn decode(bytes: &[u8]) -> Result<Still, Error> {
    let parsed = read_chunks(bytes)?;
    let rgba = image::rgba_len(parsed.info.width, parsed.info.height)?;
    let expected = image::inflated_len(&parsed.info)?;
    let mut raw = vec![0u8; expected];
    let slices = parsed.idat.iter().map(|(start, end)| &bytes[*start..*end]);
    let written =
        miniz_oxide::inflate::decompress_slice_iter_to_slice(&mut raw, slices, true, false)
            .map_err(|_| Error::Inflate)?;
    if written != expected {
        return Err(Error::Inflate);
    }
    let mut pixels = vec![0u8; rgba];
    expand::paint(
        &mut raw,
        &parsed.info,
        &parsed.palette,
        &parsed.key,
        &mut pixels,
    )?;
    Still::new(parsed.info.width, parsed.info.height, pixels)
        .map_err(|_| Error::Inconsistent("the picture buffer does not match its size"))
}

struct Parsed {
    info: Info,
    palette: Vec<[u8; 3]>,
    key: Key,
    idat: Vec<(usize, usize)>,
}

struct Building {
    info: Option<Info>,
    palette: Vec<[u8; 3]>,
    saw_palette: bool,
    key: Key,
    saw_key: bool,
    idat: Vec<(usize, usize)>,
    idat_closed: bool,
    ended: bool,
}

fn read_chunks(bytes: &[u8]) -> Result<Parsed, Error> {
    if bytes.len() < SIGNATURE.len() || bytes[..SIGNATURE.len()] != SIGNATURE {
        return Err(Error::Signature);
    }
    let mut file = Building {
        info: None,
        palette: Vec::new(),
        saw_palette: false,
        key: Key::None,
        saw_key: false,
        idat: Vec::new(),
        idat_closed: false,
        ended: false,
    };
    let mut at = SIGNATURE.len();
    while at < bytes.len() {
        if file.ended {
            return Err(Error::Inconsistent("a chunk follows IEND"));
        }
        if bytes.len().saturating_sub(at) < 8 {
            return Err(truncated(at, 8));
        }
        let length = u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        let kind = [bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]];
        let data_at = at + 8;
        let data_len = usize::try_from(length).map_err(|_| Error::TooLarge)?;
        let Some(data_end) = data_at.checked_add(data_len) else {
            return Err(Error::TooLarge);
        };
        let Some(crc_end) = data_end.checked_add(4) else {
            return Err(Error::TooLarge);
        };
        if crc_end > bytes.len() {
            let need = 8usize
                .checked_add(data_len)
                .and_then(|count| count.checked_add(4))
                .ok_or(Error::TooLarge)?;
            return Err(truncated(at, need));
        }
        let data = &bytes[data_at..data_end];
        let stored = u32::from_be_bytes([
            bytes[data_end],
            bytes[data_end + 1],
            bytes[data_end + 2],
            bytes[data_end + 3],
        ]);
        if stored != crc::of(&kind, data) {
            return Err(Error::Checksum {
                name: ChunkKind(kind),
            });
        }
        accept(&mut file, kind, data, (data_at, data_end))?;
        at = crc_end;
    }
    if !file.ended {
        return Err(Error::Inconsistent("the file has no IEND"));
    }
    let Some(info) = file.info else {
        return Err(Error::Inconsistent("the file has no IHDR"));
    };
    if file.idat.is_empty() {
        return Err(Error::Inconsistent("the file has no image data"));
    }
    if info.color == ColorKind::Indexed && !file.saw_palette {
        return Err(Error::Inconsistent("an indexed image has no palette"));
    }
    Ok(Parsed {
        info,
        palette: file.palette,
        key: file.key,
        idat: file.idat,
    })
}

fn accept(
    file: &mut Building,
    kind: [u8; 4],
    data: &[u8],
    range: (usize, usize),
) -> Result<(), Error> {
    if file.info.is_none() && &kind != b"IHDR" {
        return Err(Error::Inconsistent("IHDR is not first"));
    }
    match &kind {
        b"IHDR" => {
            if file.info.is_some() {
                return Err(Error::Inconsistent("the file has two IHDR chunks"));
            }
            file.info = Some(parse_ihdr(data)?);
        }
        b"PLTE" => {
            if file.saw_palette {
                return Err(Error::Inconsistent("the file has two palettes"));
            }
            if !file.idat.is_empty() {
                return Err(Error::Inconsistent("the palette follows the image data"));
            }
            let info = file.info.ok_or(Error::Inconsistent("IHDR is not first"))?;
            file.palette = parse_palette(data, info.color, info.bit_depth)?;
            file.saw_palette = true;
        }
        b"tRNS" => {
            if file.saw_key {
                return Err(Error::Inconsistent("the file has two transparency keys"));
            }
            if !file.idat.is_empty() {
                return Err(Error::Inconsistent(
                    "the transparency key follows the image data",
                ));
            }
            let info = file.info.ok_or(Error::Inconsistent("IHDR is not first"))?;
            file.key = parse_key(data, info.color, file.palette.len())?;
            file.saw_key = true;
        }
        b"IDAT" => {
            if file.idat_closed {
                return Err(Error::Inconsistent("IDAT chunks are not consecutive"));
            }
            file.idat.push(range);
        }
        b"IEND" => {
            if !data.is_empty() {
                return Err(Error::Inconsistent("IEND is not empty"));
            }
            file.ended = true;
        }
        _ => {
            if is_critical(kind) {
                return Err(Error::Critical {
                    name: ChunkKind(kind),
                });
            }
            if !file.idat.is_empty() {
                file.idat_closed = true;
            }
        }
    }
    if &kind != b"IDAT" && !file.idat.is_empty() {
        file.idat_closed = true;
    }
    Ok(())
}

fn parse_ihdr(data: &[u8]) -> Result<Info, Error> {
    if data.len() != 13 {
        return Err(Error::Inconsistent("IHDR is not 13 bytes"));
    }
    let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    let bit_depth = data[8];
    let color = color_kind(data[9])?;
    if data[10] != 0 {
        return Err(Error::Unsupported("the compression method is not 0"));
    }
    if data[11] != 0 {
        return Err(Error::Unsupported("the filter method is not 0"));
    }
    let interlace = match data[12] {
        0 => false,
        1 => true,
        _ => {
            return Err(Error::Unsupported("the interlace method is not 0 or 1"));
        }
    };
    if !depth_allowed(color, bit_depth) {
        return Err(Error::Unsupported(
            "the bit depth is not allowed for the color type",
        ));
    }
    let info = Info {
        width,
        height,
        bit_depth,
        color,
        interlace,
    };
    image::limits(&info)?;
    Ok(info)
}

fn color_kind(byte: u8) -> Result<ColorKind, Error> {
    match byte {
        0 => Ok(ColorKind::Grey),
        2 => Ok(ColorKind::Truecolor),
        3 => Ok(ColorKind::Indexed),
        4 => Ok(ColorKind::GreyAlpha),
        6 => Ok(ColorKind::TruecolorAlpha),
        _ => Err(Error::Unsupported("the color type is not a still")),
    }
}

fn depth_allowed(color: ColorKind, depth: u8) -> bool {
    match color {
        ColorKind::Grey => matches!(depth, 1 | 2 | 4 | 8 | 16),
        ColorKind::Truecolor | ColorKind::GreyAlpha | ColorKind::TruecolorAlpha => {
            matches!(depth, 8 | 16)
        }
        ColorKind::Indexed => matches!(depth, 1 | 2 | 4 | 8),
    }
}

fn parse_palette(data: &[u8], color: ColorKind, bit_depth: u8) -> Result<Vec<[u8; 3]>, Error> {
    match color {
        ColorKind::Grey | ColorKind::GreyAlpha => {
            return Err(Error::Inconsistent("a grey image has a palette"));
        }
        ColorKind::Truecolor | ColorKind::TruecolorAlpha | ColorKind::Indexed => {}
    }
    if data.is_empty() || !data.len().is_multiple_of(3) || data.len() > 256 * 3 {
        return Err(Error::Inconsistent(
            "the palette length is not a set of colors",
        ));
    }
    let count = data.len() / 3;
    if color == ColorKind::Indexed {
        if bit_depth > 8 {
            return Err(Error::Unsupported(
                "the bit depth is not allowed for the color type",
            ));
        }
        let max = 1usize << usize::from(bit_depth);
        if count > max {
            return Err(Error::Inconsistent(
                "the palette is wider than the bit depth",
            ));
        }
    }
    let mut palette = Vec::with_capacity(count);
    if color == ColorKind::Indexed {
        for entry in data.chunks_exact(3) {
            palette.push([entry[0], entry[1], entry[2]]);
        }
    }
    Ok(palette)
}

fn parse_key(data: &[u8], color: ColorKind, palette_len: usize) -> Result<Key, Error> {
    match color {
        ColorKind::Grey => {
            let Some(bytes) = data.get(..2) else {
                return Err(Error::Inconsistent(
                    "the grey transparency key is not two bytes",
                ));
            };
            if data.len() != 2 {
                return Err(Error::Inconsistent(
                    "the grey transparency key is not two bytes",
                ));
            }
            Ok(Key::Grey(u16::from_be_bytes([bytes[0], bytes[1]])))
        }
        ColorKind::Truecolor => {
            if data.len() != 6 {
                return Err(Error::Inconsistent(
                    "the color transparency key is not six bytes",
                ));
            }
            Ok(Key::Rgb([
                u16::from_be_bytes([data[0], data[1]]),
                u16::from_be_bytes([data[2], data[3]]),
                u16::from_be_bytes([data[4], data[5]]),
            ]))
        }
        ColorKind::Indexed => {
            if palette_len == 0 {
                return Err(Error::Inconsistent(
                    "the transparency key is ahead of the palette",
                ));
            }
            if data.len() > palette_len {
                return Err(Error::Inconsistent(
                    "the transparency key is longer than the palette",
                ));
            }
            Ok(Key::Index(data.to_vec()))
        }
        ColorKind::GreyAlpha | ColorKind::TruecolorAlpha => {
            Err(Error::Inconsistent("an alpha image has a transparency key"))
        }
    }
}

fn is_critical(kind: [u8; 4]) -> bool {
    kind[0] & 0x20 == 0
}

fn truncated(at: usize, need: usize) -> Error {
    Error::Truncated {
        at: u64::try_from(at).expect("a byte offset fits in u64"),
        need: u64::try_from(need).expect("a byte count fits in u64"),
    }
}

#[cfg(test)]
mod tests {
    use super::{SIGNATURE, decode};
    use crate::crc;
    use crate::error::{ChunkKind, Error};

    #[test]
    fn a_red_pixel_is_opaque() {
        let file = png(1, 1, 8, 2, 0, &[], &[0, 255, 0, 0]);
        let still = decode(&file).expect("red");
        assert_eq!(still.width(), 1);
        assert_eq!(still.pixels(), &[255, 0, 0, 255]);
    }

    #[test]
    fn an_alpha_pixel_keeps_its_coverage() {
        let file = png(1, 1, 8, 6, 0, &[], &[0, 10, 20, 30, 40]);
        let still = decode(&file).expect("rgba");
        assert_eq!(still.pixels(), &[10, 20, 30, 40]);
    }

    #[test]
    fn grey_bytes_are_copied_into_rgb() {
        let file = png(2, 1, 8, 0, 0, &[], &[0, 1, 2]);
        let still = decode(&file).expect("grey");
        assert_eq!(still.pixels(), &[1, 1, 1, 255, 2, 2, 2, 255]);
    }

    #[test]
    fn grey_alpha_keeps_both_samples() {
        let file = png(1, 1, 8, 4, 0, &[], &[0, 10, 20]);
        let still = decode(&file).expect("grey alpha");
        assert_eq!(still.pixels(), &[10, 10, 10, 20]);
    }

    #[test]
    fn a_one_bit_row_repeats_its_high_bit() {
        let file = png(8, 1, 1, 0, 0, &[], &[0, 0b1011_0000]);
        let still = decode(&file).expect("one bit");
        let mut pixels = Vec::new();
        for sample in [255, 0, 255, 255, 0, 0, 0, 0] {
            pixels.extend([sample, sample, sample, 255]);
        }
        assert_eq!(still.pixels(), pixels.as_slice());
    }

    #[test]
    fn a_two_bit_sample_repeats_into_the_byte() {
        let file = png(1, 1, 2, 0, 0, &[], &[0, 0x40]);
        let still = decode(&file).expect("two bit");
        assert_eq!(still.pixels(), &[85, 85, 85, 255]);
    }

    #[test]
    fn a_palette_index_uses_the_entry() {
        let file = png(
            1,
            1,
            8,
            3,
            0,
            &[(b"PLTE", &[255, 0, 0]), (b"tRNS", &[128])],
            &[0, 0],
        );
        let still = decode(&file).expect("indexed");
        assert_eq!(still.pixels(), &[255, 0, 0, 128]);
    }

    #[test]
    fn a_four_bit_index_is_packed_from_the_high_bits() {
        let palette = [0, 0, 0, 0, 255, 0, 0, 0, 255];
        let file = png(
            2,
            1,
            4,
            3,
            0,
            &[(b"PLTE", &palette), (b"tRNS", &[255, 128])],
            &[0, 0x12],
        );
        let still = decode(&file).expect("packed index");
        assert_eq!(still.pixels(), &[0, 255, 0, 128, 0, 0, 255, 255]);
    }

    #[test]
    fn an_index_past_the_palette_is_rejected() {
        let file = png(1, 1, 8, 3, 0, &[(b"PLTE", &[255, 0, 0])], &[0, 1]);
        assert_eq!(
            decode(&file),
            Err(Error::Inconsistent("a pixel is outside the palette"))
        );
    }

    #[test]
    fn a_grey_key_is_transparent() {
        let file = png(2, 1, 8, 0, 0, &[(b"tRNS", &[0x00, 0x05])], &[0, 5, 6]);
        let still = decode(&file).expect("grey key");
        assert_eq!(still.pixels(), &[5, 5, 5, 0, 6, 6, 6, 255]);
    }

    #[test]
    fn a_key_masks_bits_above_the_sample() {
        let file = png(1, 1, 8, 0, 0, &[(b"tRNS", &[0x01, 0x05])], &[0, 5]);
        let still = decode(&file).expect("masked key");
        assert_eq!(still.pixels(), &[5, 5, 5, 0]);
    }

    #[test]
    fn a_color_key_is_transparent() {
        let key = [0x00, 0x01, 0x00, 0x02, 0x00, 0x03];
        let file = png(1, 1, 8, 2, 0, &[(b"tRNS", &key)], &[0, 1, 2, 3]);
        let still = decode(&file).expect("color key");
        assert_eq!(still.pixels(), &[1, 2, 3, 0]);
    }

    #[test]
    fn a_sixteen_bit_sample_keeps_its_top_byte() {
        let file = png(1, 1, 16, 0, 0, &[], &[0, 0x12, 0xAB]);
        let still = decode(&file).expect("sixteen bit");
        assert_eq!(still.pixels(), &[0x12, 0x12, 0x12, 255]);
    }

    #[test]
    fn a_sixteen_bit_key_uses_the_whole_sample() {
        let file = png(
            1,
            1,
            16,
            0,
            0,
            &[(b"tRNS", &[0x12, 0xFF])],
            &[0, 0x12, 0x00],
        );
        let still = decode(&file).expect("sixteen bit key");
        assert_eq!(still.pixels(), &[0x12, 0x12, 0x12, 255]);
    }

    #[test]
    fn sub_reconstructs_the_byte_to_the_left() {
        let file = png(2, 1, 8, 2, 0, &[], &[1, 1, 2, 3, 3, 3, 3]);
        let still = decode(&file).expect("sub");
        assert_eq!(still.pixels(), &[1, 2, 3, 255, 4, 5, 6, 255]);
    }

    #[test]
    fn up_reconstructs_the_byte_above() {
        let file = png(1, 2, 8, 2, 0, &[], &[0, 10, 20, 30, 2, 1, 2, 3]);
        let still = decode(&file).expect("up");
        assert_eq!(still.pixels(), &[10, 20, 30, 255, 11, 22, 33, 255]);
    }

    #[test]
    fn average_uses_the_byte_to_the_left_and_above() {
        let file = png(2, 2, 8, 0, 0, &[], &[0, 10, 20, 3, 7, 24]);
        let still = decode(&file).expect("average");
        assert_eq!(
            still.pixels(),
            &[
                10, 10, 10, 255, 20, 20, 20, 255, 12, 12, 12, 255, 40, 40, 40, 255,
            ]
        );
    }

    #[test]
    fn paeth_uses_the_neighbor_bytes() {
        let file = png(2, 2, 8, 0, 0, &[], &[0, 10, 20, 4, 0, 10]);
        let still = decode(&file).expect("paeth");
        assert_eq!(
            still.pixels(),
            &[
                10, 10, 10, 255, 20, 20, 20, 255, 10, 10, 10, 255, 30, 30, 30, 255,
            ]
        );
    }

    #[test]
    fn adam7_places_the_spec_passes() {
        // Pass 1 is (0, 0), pass 6 is (1, 0), pass 7 is (0, 1) then (1, 1).
        let file = png(2, 2, 8, 0, 1, &[], &[0, 1, 0, 2, 0, 3, 4]);
        let still = decode(&file).expect("adam7");
        assert_eq!(
            still.pixels(),
            &[1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255,]
        );
    }

    #[test]
    fn a_filter_applies_inside_one_pass() {
        // The last pass is Sub, so its second byte is reconstructed from its first.
        let file = png(2, 2, 8, 0, 1, &[], &[0, 1, 0, 2, 1, 3, 1]);
        let still = decode(&file).expect("pass filter");
        assert_eq!(
            still.pixels(),
            &[1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255,]
        );
    }

    #[test]
    fn an_interlaced_picture_matches_the_plain_one() {
        let pixels: Vec<u8> = (0..64).collect();
        let plain = png(8, 8, 8, 0, 0, &[], &grey_rows(8, &pixels));
        let laced = png(8, 8, 8, 0, 1, &[], &adam7_rows(8, 8, &pixels));
        let plain = decode(&plain).expect("plain");
        let laced = decode(&laced).expect("laced");
        assert_eq!(laced.pixels(), plain.pixels());
    }

    #[test]
    fn two_idat_chunks_are_one_stream() {
        let scanlines = [0u8, 9, 8, 7];
        let zlib = store(&scanlines);
        let mid = zlib.len() / 2;
        let mut file = Vec::from(SIGNATURE);
        file.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 2, 0)));
        file.extend(chunk(b"IDAT", &zlib[..mid]));
        file.extend(chunk(b"IDAT", &zlib[mid..]));
        file.extend(chunk(b"IEND", &[]));
        let still = decode(&file).expect("split idat");
        assert_eq!(still.pixels(), &[9, 8, 7, 255]);
    }

    #[test]
    fn a_gamma_chunk_does_not_change_the_samples() {
        let file = png(1, 1, 8, 0, 0, &[(b"gAMA", &[0, 0, 0xb1, 0x8f])], &[0, 128]);
        let still = decode(&file).expect("gamma");
        assert_eq!(still.pixels(), &[128, 128, 128, 255]);
    }

    #[test]
    fn an_exif_chunk_does_not_turn_the_picture() {
        let file = png(1, 1, 8, 2, 0, &[(b"eXIf", &[0, 1, 2, 3])], &[0, 4, 5, 6]);
        let still = decode(&file).expect("exif");
        assert_eq!(still.pixels(), &[4, 5, 6, 255]);
    }

    #[test]
    fn a_bad_signature_is_rejected() {
        assert_eq!(decode(b"not a png"), Err(Error::Signature));
    }

    #[test]
    fn a_bad_checksum_is_rejected() {
        let mut file = png(1, 1, 8, 0, 0, &[], &[0, 1]);
        file[28] ^= 1;
        assert_eq!(
            decode(&file),
            Err(Error::Checksum {
                name: ChunkKind(*b"IHDR"),
            })
        );
    }

    #[test]
    fn a_zero_edge_is_rejected() {
        let file = header_only(0, 1, 8, 0, 0);
        assert_eq!(
            decode(&file),
            Err(Error::Inconsistent("width or height is zero"))
        );
    }

    #[test]
    fn an_edge_past_the_limit_is_rejected() {
        let file = header_only(32_769, 1, 8, 0, 0);
        assert_eq!(decode(&file), Err(Error::TooLarge));
    }

    #[test]
    fn a_picture_past_the_byte_limit_is_rejected() {
        let file = header_only(8_192, 8_193, 8, 0, 0);
        assert_eq!(decode(&file), Err(Error::TooLarge));
    }

    #[test]
    fn a_short_file_is_rejected() {
        let mut file = Vec::from(SIGNATURE);
        file.extend([0, 0, 0, 1]);
        assert_eq!(decode(&file), Err(Error::Truncated { at: 8, need: 8 }));
    }

    #[test]
    fn a_chunk_past_the_end_is_rejected() {
        let mut file = Vec::from(SIGNATURE);
        file.extend(100u32.to_be_bytes());
        file.extend(b"IHDR");
        assert_eq!(decode(&file), Err(Error::Truncated { at: 8, need: 112 }));
    }

    #[test]
    fn an_unknown_critical_chunk_is_rejected() {
        let file = png(1, 1, 8, 0, 0, &[(b"NOPE", &[1])], &[0, 1]);
        assert_eq!(
            decode(&file),
            Err(Error::Critical {
                name: ChunkKind(*b"NOPE"),
            })
        );
    }

    #[test]
    fn a_text_chunk_between_idat_chunks_is_rejected() {
        let zlib = store(&[0, 1]);
        let mid = zlib.len() / 2;
        let mut file = Vec::from(SIGNATURE);
        file.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
        file.extend(chunk(b"IDAT", &zlib[..mid]));
        file.extend(chunk(b"tEXt", b"Comment\0hi"));
        file.extend(chunk(b"IDAT", &zlib[mid..]));
        file.extend(chunk(b"IEND", &[]));
        assert_eq!(
            decode(&file),
            Err(Error::Inconsistent("IDAT chunks are not consecutive"))
        );
    }

    #[test]
    fn an_indexed_image_needs_a_palette() {
        let file = png(1, 1, 8, 3, 0, &[], &[0, 0]);
        assert_eq!(
            decode(&file),
            Err(Error::Inconsistent("an indexed image has no palette"))
        );
    }

    #[test]
    fn a_grey_palette_is_rejected() {
        let file = png(1, 1, 8, 0, 0, &[(b"PLTE", &[0, 0, 0])], &[0, 1]);
        assert_eq!(
            decode(&file),
            Err(Error::Inconsistent("a grey image has a palette"))
        );
    }

    #[test]
    fn an_alpha_image_rejects_a_key() {
        let file = png(1, 1, 8, 6, 0, &[(b"tRNS", &[0])], &[0, 1, 2, 3, 4]);
        assert_eq!(
            decode(&file),
            Err(Error::Inconsistent("an alpha image has a transparency key"))
        );
    }

    #[test]
    fn a_filter_type_past_four_is_rejected() {
        let file = png(1, 1, 8, 2, 0, &[], &[5, 1, 2, 3]);
        assert_eq!(
            decode(&file),
            Err(Error::Unsupported("the filter type is not 0 through 4"))
        );
    }

    #[test]
    fn a_short_zlib_stream_is_rejected() {
        let file = png_idat(1, 1, 8, 0, 0, &store(&[0]));
        assert_eq!(decode(&file), Err(Error::Inflate));
    }

    #[test]
    fn a_bad_zlib_stream_is_rejected() {
        let file = png_idat(1, 1, 8, 0, 0, &[0, 1, 2, 3]);
        assert_eq!(decode(&file), Err(Error::Inflate));
    }

    #[test]
    fn bytes_after_iend_are_rejected() {
        let mut file = png(1, 1, 8, 0, 0, &[], &[0, 1]);
        file.push(0);
        assert_eq!(
            decode(&file),
            Err(Error::Inconsistent("a chunk follows IEND"))
        );
    }

    #[test]
    fn a_bit_depth_the_color_type_forbids_is_rejected() {
        let file = header_only(1, 1, 1, 2, 0);
        assert_eq!(
            decode(&file),
            Err(Error::Unsupported(
                "the bit depth is not allowed for the color type"
            ))
        );
    }

    #[test]
    fn the_file_needs_an_iend() {
        let file = header_only(1, 1, 8, 0, 0);
        assert_eq!(
            decode(&file),
            Err(Error::Inconsistent("the file has no IEND"))
        );
    }

    #[test]
    fn iend_is_empty() {
        let mut file = Vec::from(SIGNATURE);
        file.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
        file.extend(chunk(b"IDAT", &store(&[0, 1])));
        file.extend(chunk(b"IEND", &[0]));
        assert_eq!(decode(&file), Err(Error::Inconsistent("IEND is not empty")));
    }

    fn png(
        width: u32,
        height: u32,
        depth: u8,
        color: u8,
        interlace: u8,
        extras: &[(&[u8; 4], &[u8])],
        scanlines: &[u8],
    ) -> Vec<u8> {
        png_idat_with(
            width,
            height,
            depth,
            color,
            interlace,
            extras,
            &store(scanlines),
        )
    }

    fn png_idat(
        width: u32,
        height: u32,
        depth: u8,
        color: u8,
        interlace: u8,
        idat: &[u8],
    ) -> Vec<u8> {
        png_idat_with(width, height, depth, color, interlace, &[], idat)
    }

    fn png_idat_with(
        width: u32,
        height: u32,
        depth: u8,
        color: u8,
        interlace: u8,
        extras: &[(&[u8; 4], &[u8])],
        idat: &[u8],
    ) -> Vec<u8> {
        let mut file = Vec::from(SIGNATURE);
        file.extend(chunk(
            b"IHDR",
            &ihdr(width, height, depth, color, interlace),
        ));
        for (kind, data) in extras {
            file.extend(chunk(kind, data));
        }
        file.extend(chunk(b"IDAT", idat));
        file.extend(chunk(b"IEND", &[]));
        file
    }

    fn header_only(width: u32, height: u32, depth: u8, color: u8, interlace: u8) -> Vec<u8> {
        let mut file = Vec::from(SIGNATURE);
        file.extend(chunk(
            b"IHDR",
            &ihdr(width, height, depth, color, interlace),
        ));
        file
    }

    fn ihdr(width: u32, height: u32, depth: u8, color: u8, interlace: u8) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(width.to_be_bytes());
        data.extend(height.to_be_bytes());
        data.extend([depth, color, 0, 0, interlace]);
        data
    }

    fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let len = u32::try_from(data.len()).expect("a test chunk fits in u32");
        let mut out = Vec::new();
        out.extend(len.to_be_bytes());
        out.extend(kind);
        out.extend(data);
        out.extend(crc::of(kind, data).to_be_bytes());
        out
    }

    fn store(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        let mut rest = data;
        if rest.is_empty() {
            out.push(1);
            out.extend(0u16.to_le_bytes());
            out.extend(0xffffu16.to_le_bytes());
        }
        while !rest.is_empty() {
            let take = rest.len().min(65_535);
            let last = take == rest.len();
            out.push(if last { 1 } else { 0 });
            let len = u16::try_from(take).expect("a stored block fits in u16");
            out.extend(len.to_le_bytes());
            out.extend((!len).to_le_bytes());
            out.extend(&rest[..take]);
            rest = &rest[take..];
        }
        out.extend(adler32(data).to_be_bytes());
        out
    }

    fn adler32(data: &[u8]) -> u32 {
        let mut a = 1u32;
        let mut b = 0u32;
        for &byte in data {
            a = (a + u32::from(byte)) % 65_521;
            b = (b + a) % 65_521;
        }
        (b << 16) | a
    }

    fn grey_rows(width: usize, pixels: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in pixels.chunks(width) {
            out.push(0);
            out.extend(row);
        }
        out
    }

    fn adam7_rows(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        const PASSES: [(u32, u32, u32, u32); 7] = [
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        ];
        let mut out = Vec::new();
        for (x0, y0, x_step, y_step) in PASSES {
            if x0 >= width || y0 >= height {
                continue;
            }
            let cols = (width - x0 - 1) / x_step + 1;
            let rows = (height - y0 - 1) / y_step + 1;
            for row in 0..rows {
                out.push(0);
                for col in 0..cols {
                    let x = x0 + col * x_step;
                    let y = y0 + row * y_step;
                    let index = usize::try_from(y * width + x).expect("a test pixel fits");
                    out.push(pixels[index]);
                }
            }
        }
        out
    }
}
