//! RIFF container. A still is one `VP8L` image. An animation keeps the first frame.

use cutlass_core::Still;

use crate::error::Error;
use crate::lossless::{Picture, decode_vp8l, into_rgba, pixel_count};

/// Decodes a WebP file into one straight RGBA8 still.
///
/// `docs/codecs.md` lists what is read. Lossy VP8 is refused.
pub fn decode(bytes: &[u8]) -> Result<Still, Error> {
    let picture = container(bytes)?;
    let rgba = into_rgba(&picture.pixels);
    Ok(Still::new(picture.width, picture.height, rgba)
        .expect("the decoded buffer matches the still size"))
}

fn container(bytes: &[u8]) -> Result<Picture, Error> {
    let payload = riff_payload(bytes)?;
    let mut input = Input::new(payload);
    if input.remaining() == 0 {
        return Err(Error::Inconsistent("the file has no image"));
    }
    let (id, data) = input.chunk()?;
    if id == *b"VP8L" {
        let image = decode_vp8l(data)?;
        if input.remaining() != 0 {
            return Err(Error::Inconsistent("a chunk follows a simple image"));
        }
        return Ok(image);
    }
    if id == *b"VP8 " {
        return Err(Error::Unsupported("lossy WebP is not supported"));
    }
    if id != *b"VP8X" {
        return Err(Error::Inconsistent("a chunk is out of order"));
    }
    let ext = extended(data)?;
    assemble(&mut input, ext)
}

struct Extended {
    animation: bool,
    width: u32,
    height: u32,
}

/// WebP container, extended header. Reserved bits are ignored.
fn extended(data: &[u8]) -> Result<Extended, Error> {
    if data.len() != 10 {
        return Err(Error::Inconsistent("an extended header is the wrong size"));
    }
    // Bit 1 of the first byte is the animation flag. WebP container, VP8X.
    let animation = (data[0] & 0x02) != 0;
    let width = u24(&data[4..7]).checked_add(1).ok_or(Error::TooLarge)?;
    let height = u24(&data[7..10]).checked_add(1).ok_or(Error::TooLarge)?;
    pixel_count(width, height)?;
    Ok(Extended {
        animation,
        width,
        height,
    })
}

fn assemble(input: &mut Input<'_>, ext: Extended) -> Result<Picture, Error> {
    let mut background = 0u32;
    let mut saw_anim = false;
    let mut saw_alpha = false;
    let mut picture: Option<Picture> = None;
    while input.remaining() > 0 {
        let (id, data) = input.chunk()?;
        if id == *b"VP8X" {
            return Err(Error::Inconsistent("a second extended header"));
        } else if id == *b"ICCP" || id == *b"EXIF" || id == *b"XMP " {
            // Profiles and metadata are not applied. The picture is not turned.
        } else if id == *b"ALPH" {
            if picture.is_some() || saw_alpha {
                return Err(Error::Inconsistent("an alpha chunk is out of order"));
            }
            saw_alpha = true;
        } else if id == *b"ANIM" {
            if !ext.animation {
                continue;
            }
            if saw_anim {
                return Err(Error::Inconsistent("a second animation header"));
            }
            background = anim_background(data)?;
            saw_anim = true;
        } else if id == *b"ANMF" {
            if !ext.animation {
                return Err(Error::Inconsistent("an animation frame is out of order"));
            }
            if picture.is_some() {
                continue;
            }
            if !saw_anim {
                return Err(Error::Inconsistent("an animation frame is out of order"));
            }
            picture = Some(draw_frame(&ext, background, data, saw_alpha)?);
        } else if id == *b"VP8 " {
            return Err(Error::Unsupported("lossy WebP is not supported"));
        } else if id == *b"VP8L" {
            if ext.animation {
                return Err(Error::Inconsistent("a chunk is out of order"));
            }
            if picture.is_some() {
                return Err(Error::Inconsistent("a second image"));
            }
            if saw_alpha {
                return Err(Error::Inconsistent("a lossless frame has an alpha chunk"));
            }
            let image = decode_vp8l(data)?;
            if image.width != ext.width || image.height != ext.height {
                return Err(Error::Inconsistent("the canvas does not match the image"));
            }
            picture = Some(image);
        } else if picture.is_none() {
            return Err(Error::Inconsistent("a chunk is out of order"));
        }
    }
    if ext.animation {
        if !saw_anim {
            return Err(Error::Inconsistent("an animation has no header"));
        }
        picture.ok_or(Error::Inconsistent("an animation has no frame"))
    } else {
        picture.ok_or(Error::Inconsistent("the file has no image"))
    }
}

/// Background is stored blue, green, red, alpha.
fn anim_background(data: &[u8]) -> Result<u32, Error> {
    if data.len() != 6 {
        return Err(Error::Inconsistent("an animation header is the wrong size"));
    }
    Ok((u32::from(data[3]) << 24)
        | (u32::from(data[2]) << 16)
        | (u32::from(data[1]) << 8)
        | u32::from(data[0]))
}

fn draw_frame(
    ext: &Extended,
    background: u32,
    data: &[u8],
    saw_alpha: bool,
) -> Result<Picture, Error> {
    if data.len() < 16 {
        return Err(Error::Truncated {
            at: u64::try_from(data.len()).unwrap_or(u64::MAX),
            need: 16,
        });
    }
    // Origins are stored in units of two pixels. WebP container, ANMF.
    let x = u24(&data[0..3]).checked_mul(2).ok_or(Error::TooLarge)?;
    let y = u24(&data[3..6]).checked_mul(2).ok_or(Error::TooLarge)?;
    let fw = u24(&data[6..9]).checked_add(1).ok_or(Error::TooLarge)?;
    let fh = u24(&data[9..12]).checked_add(1).ok_or(Error::TooLarge)?;
    let overwrite = (data[15] & 0x02) != 0;
    if x.checked_add(fw).ok_or(Error::TooLarge)? > ext.width
        || y.checked_add(fh).ok_or(Error::TooLarge)? > ext.height
    {
        return Err(Error::Inconsistent("a frame does not fit on the canvas"));
    }
    let frame = frame_image(&data[16..], fw, fh, saw_alpha)?;
    blit(ext, background, x, y, &frame, overwrite)
}

fn frame_image(data: &[u8], fw: u32, fh: u32, mut saw_alpha: bool) -> Result<Picture, Error> {
    let mut input = Input::new(data);
    let mut frame = None;
    while input.remaining() > 0 {
        let (id, chunk) = input.chunk()?;
        if id == *b"ALPH" {
            if saw_alpha || frame.is_some() {
                return Err(Error::Inconsistent("an alpha chunk is out of order"));
            }
            saw_alpha = true;
        } else if id == *b"VP8 " {
            return Err(Error::Unsupported("lossy WebP is not supported"));
        } else if id == *b"VP8L" {
            if frame.is_some() {
                return Err(Error::Inconsistent("a second image"));
            }
            if saw_alpha {
                return Err(Error::Inconsistent("a lossless frame has an alpha chunk"));
            }
            let image = decode_vp8l(chunk)?;
            if image.width != fw || image.height != fh {
                return Err(Error::Inconsistent("a frame does not match its image"));
            }
            frame = Some(image);
        } else if frame.is_none() {
            return Err(Error::Inconsistent("a chunk is out of order"));
        }
    }
    frame.ok_or(Error::Inconsistent("an animation frame has no image"))
}

fn blit(
    ext: &Extended,
    background: u32,
    x: u32,
    y: u32,
    frame: &Picture,
    overwrite: bool,
) -> Result<Picture, Error> {
    let count = pixel_count(ext.width, ext.height)?;
    let mut canvas = vec![background; count];
    let canvas_w = usize::try_from(ext.width).map_err(|_| Error::TooLarge)?;
    let x0 = usize::try_from(x).map_err(|_| Error::TooLarge)?;
    let y0 = usize::try_from(y).map_err(|_| Error::TooLarge)?;
    let fw = usize::try_from(frame.width).map_err(|_| Error::TooLarge)?;
    let fh = usize::try_from(frame.height).map_err(|_| Error::TooLarge)?;
    for row in 0..fh {
        for col in 0..fw {
            let src = frame.pixels[row * fw + col];
            let at = (y0 + row) * canvas_w + (x0 + col);
            canvas[at] = if overwrite {
                src
            } else {
                // WebP container, alpha blending, on the stored samples.
                blend(canvas[at], src)?
            };
        }
    }
    Ok(Picture {
        width: ext.width,
        height: ext.height,
        pixels: canvas,
    })
}

/// Truncating integer form of the container blend. Channels are not linearized.
fn blend(dst: u32, src: u32) -> Result<u32, Error> {
    let sa = (src >> 24) & 0xff;
    let da = (dst >> 24) & 0xff;
    let inv = 255 - sa;
    let out_a = sa + (da * inv) / 255;
    if out_a == 0 {
        return Ok(0);
    }
    let mut out = out_a << 24;
    for shift in [16u32, 8, 0] {
        let s = (src >> shift) & 0xff;
        let d = (dst >> shift) & 0xff;
        let mixed = (s * sa + (d * da * inv) / 255) / out_a;
        let channel = u8::try_from(mixed)
            .map_err(|_| Error::Inconsistent("a blended channel does not fit"))?;
        out |= u32::from(channel) << shift;
    }
    Ok(out)
}

fn u24(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

struct Input<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Input<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.at
    }

    fn chunk(&mut self) -> Result<([u8; 4], &'a [u8]), Error> {
        let id = self.fourcc()?;
        let size = self.u32()?;
        let len = usize::try_from(size).map_err(|_| Error::TooLarge)?;
        let data = self.take(len)?;
        if !size.is_multiple_of(2) {
            let pad = self.take(1)?;
            if pad[0] != 0 {
                return Err(Error::Inconsistent("a padding byte is not zero"));
            }
        }
        Ok((id, data))
    }

    fn fourcc(&mut self) -> Result<[u8; 4], Error> {
        let bytes = self.take(4)?;
        Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn u32(&mut self) -> Result<u32, Error> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let Some(end) = self.at.checked_add(n) else {
            return Err(Error::TooLarge);
        };
        if end > self.data.len() {
            return Err(Error::Truncated {
                at: u64::try_from(self.at).unwrap_or(u64::MAX),
                need: u64::try_from(n).unwrap_or(u64::MAX),
            });
        }
        let slice = &self.data[self.at..end];
        self.at = end;
        Ok(slice)
    }
}

/// Chunks that follow the `WEBP` form type.
fn riff_payload(bytes: &[u8]) -> Result<&[u8], Error> {
    if bytes.len() < 4 || &bytes[0..4] != b"RIFF" {
        return Err(Error::Signature);
    }
    if bytes.len() < 8 {
        return Err(Error::Truncated { at: 4, need: 4 });
    }
    let size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if size < 4 {
        return Err(Error::Truncated { at: 8, need: 4 });
    }
    let size_us = usize::try_from(size).map_err(|_| Error::TooLarge)?;
    let end = 8usize.checked_add(size_us).ok_or(Error::TooLarge)?;
    if bytes.len() < end {
        return Err(Error::Truncated {
            at: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            need: u64::try_from(end - bytes.len()).unwrap_or(u64::MAX),
        });
    }
    if bytes.len() > end {
        return Err(Error::Inconsistent("a byte follows the file"));
    }
    if &bytes[8..12] != b"WEBP" {
        return Err(Error::Signature);
    }
    Ok(&bytes[12..])
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_n;

    use cutlass_core::Still;

    use super::{Error, decode};
    use crate::bits::Writer;
    use crate::lossless::cache_index;

    /// Opaque black. The bytes are the bit-order lock for a simple leaf.
    const BLACK: &[u8] = &[0x2f, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88, 0xfe, 0x07];
    /// Green 1. The bytes lock an 8-bit symbol, low bit first.
    const GREEN: &[u8] = &[0x2f, 0x00, 0x00, 0x00, 0x00, 0x68, 0x40, 0x44, 0xff, 0x03];

    fn px(a: u8, r: u8, g: u8, b: u8) -> u32 {
        (u32::from(a) << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
    }

    fn refused(bytes: &[u8]) -> Error {
        decode(bytes).expect_err("the file is refused")
    }

    fn write_header(w: &mut Writer, width: u32, height: u32, alpha_used: bool) {
        w.bits(0x2f, 8);
        w.bits(width - 1, 14);
        w.bits(height - 1, 14);
        w.bit(u32::from(alpha_used));
        w.bits(0, 3);
    }

    fn simple_leaf(w: &mut Writer, symbol: u32) {
        w.bit(1);
        w.bit(0);
        if symbol <= 1 {
            w.bit(0);
            w.bit(symbol);
        } else {
            w.bit(1);
            w.bits(symbol, 8);
        }
    }

    fn simple_two(w: &mut Writer, first: u32, second: u32) {
        w.bit(1);
        w.bit(1);
        if first <= 1 {
            w.bit(0);
            w.bit(first);
        } else {
            w.bit(1);
            w.bits(first, 8);
        }
        w.bits(second, 8);
    }

    fn channel_values(pixels: &[u32], shift: u32) -> Vec<u8> {
        let mut vals = Vec::new();
        for pixel in pixels {
            let value = u8::try_from((pixel >> shift) & 0xff).expect("a channel fits");
            if !vals.contains(&value) {
                vals.push(value);
            }
        }
        vals.sort_unstable();
        vals
    }

    fn write_channel(w: &mut Writer, vals: &[u8]) {
        match vals {
            [one] => simple_leaf(w, u32::from(*one)),
            [lo, hi] => simple_two(w, u32::from(*lo), u32::from(*hi)),
            _ => panic!("a test channel has more than two symbols"),
        }
    }

    fn write_value(w: &mut Writer, vals: &[u8], value: u8) {
        if vals.len() < 2 {
            return;
        }
        w.bit(u32::from(value != vals[0]));
    }

    /// Cache bit, optional meta bit, then one literal per channel. At most two symbols.
    fn write_literals(w: &mut Writer, pixels: &[u32], meta_bit: bool) {
        w.bit(0);
        if meta_bit {
            w.bit(0);
        }
        let channels: Vec<Vec<u8>> = [8u32, 16, 0, 24]
            .into_iter()
            .map(|shift| channel_values(pixels, shift))
            .collect();
        for vals in &channels {
            write_channel(w, vals);
        }
        simple_leaf(w, 0);
        for pixel in pixels {
            for (shift, vals) in [8u32, 16, 0, 24].into_iter().zip(&channels) {
                let value = u8::try_from((pixel >> shift) & 0xff).expect("a channel fits");
                write_value(w, vals, value);
            }
        }
    }

    fn write_predictor(w: &mut Writer, size_bits: u32, modes: &[u32]) {
        w.bit(1);
        w.bits(0, 2);
        w.bits(size_bits - 2, 3);
        write_literals(w, modes, false);
    }

    fn write_color(w: &mut Writer, size_bits: u32, elems: &[u32]) {
        w.bit(1);
        w.bits(1, 2);
        w.bits(size_bits - 2, 3);
        write_literals(w, elems, false);
    }

    fn write_green(w: &mut Writer) {
        w.bit(1);
        w.bits(2, 2);
    }

    fn write_index(w: &mut Writer, table_size: u32, deltas: &[u32]) {
        w.bit(1);
        w.bits(3, 2);
        w.bits(table_size - 1, 8);
        write_literals(w, deltas, false);
    }

    /// Lengths of 0, 1, and 2. The code-length tree is symbols 0, 1, and 2.
    fn write_normal(w: &mut Writer, lengths: &[u8]) {
        w.bit(0);
        w.bits(5, 4);
        for len in [0u32, 0, 1, 2, 2, 0, 0, 0, 0] {
            w.bits(len, 3);
        }
        let n = u32::try_from(lengths.len()).expect("an alphabet fits");
        w.bit(1);
        let (k, width) = if n <= 5 {
            (0, 2)
        } else if n <= 17 {
            (1, 4)
        } else if n <= 65 {
            (2, 6)
        } else if n <= 257 {
            (3, 8)
        } else {
            (4, 10)
        };
        w.bits(k, 3);
        w.bits(n - 2, width);
        for &len in lengths {
            match len {
                0 => w.bit(0),
                1 => {
                    w.bit(1);
                    w.bit(0);
                }
                2 => {
                    w.bit(1);
                    w.bit(1);
                }
                _ => panic!("a test length is 0, 1, or 2"),
            }
        }
    }

    fn assign(lengths: &[u8]) -> Vec<Option<(u32, u8)>> {
        let mut count = [0u32; 16];
        let mut nonzero = 0usize;
        let mut only = 0usize;
        for (index, &len) in lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            count[usize::from(len)] += 1;
            nonzero += 1;
            only = index;
        }
        let mut out = vec![None; lengths.len()];
        if nonzero == 1 && lengths[only] == 1 {
            out[only] = Some((0, 0));
            return out;
        }
        let mut code = 0u32;
        let mut next = [0u32; 16];
        for len in 1..16 {
            code = (code + count[len - 1]) << 1;
            next[len] = code;
        }
        let mut cursor = next;
        for (index, &len) in lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            out[index] = Some((cursor[usize::from(len)], len));
            cursor[usize::from(len)] += 1;
        }
        out
    }

    fn write_symbol(w: &mut Writer, lengths: &[u8], symbol: usize) {
        let (code, n) = assign(lengths)[symbol].expect("the test symbol is in the tree");
        for shift in (0..n).rev() {
            w.bit((code >> shift) & 1);
        }
    }

    fn riff(chunks: &[([u8; 4], &[u8])]) -> Vec<u8> {
        let mut body = Vec::from(&b"WEBP"[..]);
        for (id, data) in chunks {
            body.extend_from_slice(id);
            let len = u32::try_from(data.len()).expect("a test chunk fits");
            body.extend_from_slice(&len.to_le_bytes());
            body.extend_from_slice(data);
            if !data.len().is_multiple_of(2) {
                body.push(0);
            }
        }
        let mut out = Vec::from(&b"RIFF"[..]);
        let size = u32::try_from(body.len()).expect("a test file fits");
        out.extend_from_slice(&size.to_le_bytes());
        out.extend(body);
        out
    }

    fn vp8l_file(payload: &[u8]) -> Vec<u8> {
        riff(&[(*b"VP8L", payload)])
    }

    fn push_u24(out: &mut Vec<u8>, value: u32) {
        let bytes = value.to_le_bytes();
        out.extend_from_slice(&bytes[..3]);
    }

    fn vp8x(flags: u8, width: u32, height: u32) -> Vec<u8> {
        let mut out = vec![flags, 0, 0, 0];
        push_u24(&mut out, width - 1);
        push_u24(&mut out, height - 1);
        out
    }

    fn anim(b: u8, g: u8, r: u8, a: u8) -> Vec<u8> {
        vec![b, g, r, a, 0, 0]
    }

    fn anmf(
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        overwrite: bool,
        chunks: &[([u8; 4], &[u8])],
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        push_u24(&mut payload, x / 2);
        push_u24(&mut payload, y / 2);
        push_u24(&mut payload, w - 1);
        push_u24(&mut payload, h - 1);
        push_u24(&mut payload, 0);
        payload.push(if overwrite { 0x02 } else { 0 });
        for (id, data) in chunks {
            payload.extend_from_slice(id);
            let len = u32::try_from(data.len()).expect("a test chunk fits");
            payload.extend_from_slice(&len.to_le_bytes());
            payload.extend_from_slice(data);
            if !data.len().is_multiple_of(2) {
                payload.push(0);
            }
        }
        payload
    }

    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
        let pixel = px(rgba[3], rgba[0], rgba[1], rgba[2]);
        let n = usize::try_from(width).expect("width") * usize::try_from(height).expect("height");
        let mut writer = Writer::new();
        write_header(&mut writer, width, height, false);
        writer.bit(0);
        write_literals(&mut writer, &vec![pixel; n], true);
        writer.finish()
    }

    fn header_only(width: u32, height: u32) -> Vec<u8> {
        let mut writer = Writer::new();
        write_header(&mut writer, width, height, false);
        vp8l_file(&writer.finish())
    }

    #[test]
    fn a_hand_built_black_pixel_is_opaque_black() {
        let still = decode(&vp8l_file(BLACK)).expect("black");
        assert_eq!(still.width(), 1);
        assert_eq!(still.height(), 1);
        assert_eq!(still.pixels(), &[0, 0, 0, 255]);
    }

    #[test]
    fn the_writer_matches_the_hand_built_black_pixel() {
        assert_eq!(solid(1, 1, [0, 0, 0, 255]), BLACK);
    }

    #[test]
    fn a_hand_built_green_symbol_is_stored_low_bit_first() {
        let still = decode(&vp8l_file(GREEN)).expect("green");
        assert_eq!(still.pixels(), &[0, 1, 0, 255]);
    }

    #[test]
    fn the_writer_encodes_a_one_bit_green_symbol() {
        // Symbol 1 may be one bit or eight. The hand-built stream uses eight.
        let still = decode(&vp8l_file(&solid(1, 1, [0, 1, 0, 255]))).expect("green");
        assert_eq!(still.pixels(), &[0, 1, 0, 255]);
    }

    #[test]
    fn an_alpha_hint_does_not_change_the_sample() {
        let mut writer = Writer::new();
        write_header(&mut writer, 1, 1, false);
        writer.bit(0);
        write_literals(&mut writer, &[px(0, 1, 2, 3)], true);
        let still = decode(&vp8l_file(&writer.finish())).expect("alpha");
        assert_eq!(still.pixels(), &[1, 2, 3, 0]);
    }

    #[test]
    fn subtract_green_adds_green_back() {
        let mut writer = Writer::new();
        write_header(&mut writer, 1, 1, false);
        write_green(&mut writer);
        writer.bit(0);
        write_literals(&mut writer, &[px(255, 1, 2, 3)], true);
        let still = decode(&vp8l_file(&writer.finish())).expect("green");
        assert_eq!(still.pixels(), &[3, 2, 5, 255]);
    }

    #[test]
    fn a_color_transform_uses_the_new_red_for_blue() {
        let mut writer = Writer::new();
        write_header(&mut writer, 1, 1, false);
        write_color(&mut writer, 2, &[px(255, 32, 0, 32)]);
        writer.bit(0);
        write_literals(&mut writer, &[px(255, 0, 40, 0)], true);
        let still = decode(&vp8l_file(&writer.finish())).expect("color");
        assert_eq!(still.pixels(), &[40, 40, 40, 255]);
    }

    #[test]
    fn a_negative_color_delta_wraps() {
        let mut writer = Writer::new();
        write_header(&mut writer, 1, 1, false);
        write_color(&mut writer, 2, &[px(255, 0, 0, 224)]);
        writer.bit(0);
        write_literals(&mut writer, &[px(255, 0, 40, 0)], true);
        let still = decode(&vp8l_file(&writer.finish())).expect("color");
        assert_eq!(still.pixels(), &[216, 40, 0, 255]);
    }

    fn predictor_still(mode: u8) -> Still {
        let residuals = [
            px(0, 0, 0, 0),
            px(0, 0, 10, 0),
            px(0, 0, 0, 0),
            px(0, 20, 0, 0),
            px(0, 0, 0, 0),
            px(0, 0, 0, 0),
        ];
        let mut writer = Writer::new();
        write_header(&mut writer, 3, 2, false);
        write_predictor(&mut writer, 2, &[px(0, 0, mode, 0)]);
        writer.bit(0);
        write_literals(&mut writer, &residuals, true);
        decode(&vp8l_file(&writer.finish())).expect("predictor")
    }

    #[test]
    fn predictor_modes_reconstruct_the_fixture() {
        let borders = [0, 0, 0, 255, 0, 10, 0, 255, 0, 10, 0, 255, 20, 0, 0, 255];
        // (mode, pixel (1, 1), pixel (2, 1)). The rightmost top-right is the row start.
        let cases = [
            (0, [0, 0, 0, 255], [0, 0, 0, 255]),
            (1, [20, 0, 0, 255], [20, 0, 0, 255]),
            (2, [0, 10, 0, 255], [0, 10, 0, 255]),
            (3, [0, 10, 0, 255], [20, 0, 0, 255]),
            (4, [0, 0, 0, 255], [0, 10, 0, 255]),
            (5, [5, 7, 0, 255], [6, 6, 0, 255]),
            (6, [10, 0, 0, 255], [5, 5, 0, 255]),
            (7, [10, 5, 0, 255], [5, 7, 0, 255]),
            (8, [0, 5, 0, 255], [0, 10, 0, 255]),
            (9, [0, 10, 0, 255], [10, 5, 0, 255]),
            (10, [5, 5, 0, 255], [6, 6, 0, 255]),
            (11, [20, 0, 0, 255], [20, 0, 0, 255]),
            (12, [20, 10, 0, 255], [20, 10, 0, 255]),
            (13, [15, 7, 0, 255], [10, 7, 0, 255]),
        ];
        for (mode, mid, right) in cases {
            let still = predictor_still(mode);
            assert_eq!(&still.pixels()[..16], &borders, "mode {mode} border");
            assert_eq!(&still.pixels()[16..20], &mid, "mode {mode}");
            assert_eq!(&still.pixels()[20..], &right, "mode {mode} right");
        }
    }

    #[test]
    fn a_border_pixel_ignores_a_bad_predictor() {
        let still = predictor_still_size(1, 1, 14);
        assert_eq!(still.pixels(), &[0, 0, 0, 255]);
    }

    fn predictor_still_size(width: u32, height: u32, mode: u8) -> Still {
        let n = usize::try_from(width).expect("width") * usize::try_from(height).expect("height");
        let mut writer = Writer::new();
        write_header(&mut writer, width, height, false);
        write_predictor(&mut writer, 2, &[px(0, 0, mode, 0)]);
        writer.bit(0);
        write_literals(&mut writer, &vec![0; n], true);
        decode(&vp8l_file(&writer.finish())).expect("border")
    }

    #[test]
    fn an_interior_predictor_past_thirteen_is_refused() {
        assert!(matches!(
            refused(&{
                let residuals = [0, px(0, 0, 10, 0), 0, px(0, 20, 0, 0), 0, 0];
                let mut writer = Writer::new();
                write_header(&mut writer, 3, 2, false);
                write_predictor(&mut writer, 2, &[px(0, 0, 14, 0)]);
                writer.bit(0);
                write_literals(&mut writer, &residuals, true);
                vp8l_file(&writer.finish())
            }),
            Error::Inconsistent("a predictor is not one of 14")
        ));
    }

    #[test]
    fn a_repeated_transform_is_refused() {
        let mut writer = Writer::new();
        write_header(&mut writer, 1, 1, false);
        write_green(&mut writer);
        write_green(&mut writer);
        assert!(matches!(
            refused(&vp8l_file(&writer.finish())),
            Error::Inconsistent("a transform is repeated")
        ));
    }

    #[test]
    fn a_bundled_palette_unpacks_low_bits_first() {
        let mut writer = Writer::new();
        write_header(&mut writer, 2, 1, false);
        write_index(&mut writer, 2, &[px(255, 1, 2, 3), px(0, 3, 3, 3)]);
        writer.bit(0);
        write_literals(&mut writer, &[px(255, 0, 2, 0)], true);
        let still = decode(&vp8l_file(&writer.finish())).expect("index");
        assert_eq!(still.pixels(), &[1, 2, 3, 255, 4, 5, 6, 255]);
    }

    #[test]
    fn an_index_past_the_palette_is_transparent_black() {
        let mut writer = Writer::new();
        write_header(&mut writer, 1, 1, true);
        write_index(&mut writer, 1, &[px(255, 9, 9, 9)]);
        writer.bit(0);
        write_literals(&mut writer, &[px(0, 0, 1, 0)], true);
        let still = decode(&vp8l_file(&writer.finish())).expect("index");
        assert_eq!(still.pixels(), &[0, 0, 0, 0]);
    }

    #[test]
    fn a_predictor_after_the_palette_in_the_file_runs_first() {
        let mut deltas = vec![px(255, 10, 20, 30)];
        deltas.extend(repeat_n(0, 16));
        let mut writer = Writer::new();
        write_header(&mut writer, 1, 1, true);
        write_predictor(&mut writer, 2, &[0]);
        write_index(&mut writer, 17, &deltas);
        writer.bit(0);
        write_literals(&mut writer, &[0], true);
        let still = decode(&vp8l_file(&writer.finish())).expect("order");
        assert_eq!(still.pixels(), &[10, 20, 30, 254]);
    }

    #[test]
    fn a_predictor_on_the_packed_pixel_runs_before_unpacking() {
        let mut writer = Writer::new();
        write_header(&mut writer, 8, 1, false);
        write_index(&mut writer, 2, &[px(255, 1, 2, 3), px(0, 3, 3, 3)]);
        write_predictor(&mut writer, 2, &[0]);
        writer.bit(0);
        write_literals(&mut writer, &[px(0, 0, 0xaa, 0)], true);
        let still = decode(&vp8l_file(&writer.finish())).expect("order");
        let mut expected = Vec::new();
        for index in 0..8 {
            if index % 2 == 0 {
                expected.extend([1, 2, 3, 255]);
            } else {
                expected.extend([4, 5, 6, 255]);
            }
        }
        assert_eq!(still.pixels(), expected.as_slice());
    }

    #[test]
    fn a_backward_reference_overlaps_itself() {
        let mut lengths = vec![0u8; 258];
        lengths[5] = 1;
        lengths[257] = 1;
        let mut writer = Writer::new();
        write_header(&mut writer, 3, 1, false);
        writer.bit(0);
        writer.bit(0);
        writer.bit(0);
        write_normal(&mut writer, &lengths);
        simple_leaf(&mut writer, 0);
        simple_leaf(&mut writer, 0);
        simple_leaf(&mut writer, 255);
        simple_leaf(&mut writer, 1);
        write_symbol(&mut writer, &lengths, 5);
        write_symbol(&mut writer, &lengths, 257);
        let still = decode(&vp8l_file(&writer.finish())).expect("copy");
        assert_eq!(still.pixels(), &[0, 5, 0, 255, 0, 5, 0, 255, 0, 5, 0, 255]);
    }

    #[test]
    fn distance_code_five_is_two_rows_up() {
        let mut lengths = vec![0u8; 257];
        lengths[1] = 1;
        lengths[9] = 2;
        lengths[256] = 2;
        let mut writer = Writer::new();
        write_header(&mut writer, 2, 3, false);
        writer.bit(0);
        writer.bit(0);
        writer.bit(0);
        write_normal(&mut writer, &lengths);
        simple_leaf(&mut writer, 0);
        simple_leaf(&mut writer, 0);
        simple_leaf(&mut writer, 255);
        simple_leaf(&mut writer, 4);
        for symbol in [1usize, 1, 9, 9] {
            write_symbol(&mut writer, &lengths, symbol);
        }
        write_symbol(&mut writer, &lengths, 256);
        writer.bit(0);
        write_symbol(&mut writer, &lengths, 1);
        let still = decode(&vp8l_file(&writer.finish())).expect("distance");
        assert_eq!(
            still.pixels(),
            &[
                0, 1, 0, 255, 0, 1, 0, 255, 0, 9, 0, 255, 0, 9, 0, 255, 0, 1, 0, 255, 0, 1, 0, 255,
            ]
        );
    }

    #[test]
    fn a_color_cache_hit_reuses_the_hashed_color() {
        assert_eq!(cache_index(0xff00_ff00, 1), 1);
        let mut lengths = vec![0u8; 282];
        lengths[255] = 1;
        lengths[281] = 1;
        let mut writer = Writer::new();
        write_header(&mut writer, 2, 1, false);
        writer.bit(0);
        writer.bit(1);
        writer.bits(1, 4);
        writer.bit(0);
        write_normal(&mut writer, &lengths);
        simple_leaf(&mut writer, 0);
        simple_leaf(&mut writer, 0);
        simple_leaf(&mut writer, 255);
        simple_leaf(&mut writer, 0);
        write_symbol(&mut writer, &lengths, 255);
        write_symbol(&mut writer, &lengths, 281);
        let still = decode(&vp8l_file(&writer.finish())).expect("cache");
        assert_eq!(still.pixels(), &[0, 255, 0, 255, 0, 255, 0, 255]);
    }

    #[test]
    fn a_color_cache_of_zero_bits_is_refused() {
        let mut writer = Writer::new();
        write_header(&mut writer, 1, 1, false);
        writer.bit(0);
        writer.bit(1);
        writer.bits(0, 4);
        assert!(matches!(
            refused(&vp8l_file(&writer.finish())),
            Error::Inconsistent("a color cache size is not supported")
        ));
    }

    #[test]
    fn an_entropy_image_selects_the_prefix_group() {
        let mut writer = Writer::new();
        write_header(&mut writer, 8, 1, false);
        writer.bit(0);
        writer.bit(0);
        writer.bit(1);
        writer.bits(0, 3);
        write_literals(&mut writer, &[px(0, 0, 0, 0), px(0, 1, 0, 0)], false);
        for group in 0..257 {
            let green = if group == 0 {
                10
            } else if group == 256 {
                20
            } else {
                0
            };
            let alpha = if group == 0 || group == 256 { 255 } else { 0 };
            simple_leaf(&mut writer, green);
            simple_leaf(&mut writer, 0);
            simple_leaf(&mut writer, 0);
            simple_leaf(&mut writer, alpha);
            simple_leaf(&mut writer, 0);
        }
        let still = decode(&vp8l_file(&writer.finish())).expect("entropy");
        let mut expected = Vec::new();
        for _ in 0..4 {
            expected.extend([0, 10, 0, 255]);
        }
        for _ in 0..4 {
            expected.extend([0, 20, 0, 255]);
        }
        assert_eq!(still.pixels(), expected.as_slice());
    }

    #[test]
    fn a_picture_just_over_the_buffer_limit_is_too_large() {
        assert_eq!(refused(&header_only(8192, 8193)), Error::TooLarge);
        assert_eq!(refused(&header_only(16384, 4097)), Error::TooLarge);
    }

    #[test]
    fn a_picture_at_the_buffer_limit_is_not_too_large() {
        assert!(matches!(
            refused(&header_only(8192, 8192)),
            Error::Truncated { .. }
        ));
    }

    #[test]
    fn a_nonzero_version_is_refused() {
        assert!(matches!(
            refused(&vp8l_file(&[0x2f, 0x00, 0x00, 0x00, 0x20])),
            Error::Inconsistent("the version is not 0")
        ));
    }

    #[test]
    fn a_bad_lossless_signature_is_refused() {
        assert!(matches!(
            refused(&vp8l_file(&[0x2e, 0x00, 0x00, 0x00, 0x00])),
            Error::Inconsistent("the lossless signature is not 0x2f")
        ));
    }

    #[test]
    fn a_file_that_is_not_riff_is_refused() {
        assert_eq!(refused(&[]), Error::Signature);
        assert_eq!(refused(b"nope"), Error::Signature);
    }

    #[test]
    fn a_short_riff_header_is_truncated() {
        assert!(matches!(refused(b"RIFF"), Error::Truncated { .. }));
    }

    #[test]
    fn a_riff_form_that_is_not_webp_is_refused() {
        let mut file = Vec::from(&b"RIFF"[..]);
        file.extend(4u32.to_le_bytes());
        file.extend(b"XXXX");
        assert_eq!(refused(&file), Error::Signature);
    }

    #[test]
    fn a_webp_with_no_image_is_refused() {
        let mut file = Vec::from(&b"RIFF"[..]);
        file.extend(4u32.to_le_bytes());
        file.extend(b"WEBP");
        assert!(matches!(
            refused(&file),
            Error::Inconsistent("the file has no image")
        ));
    }

    #[test]
    fn a_lossy_frame_is_refused() {
        assert!(matches!(
            refused(&riff(&[(*b"VP8 ", &[0, 0, 0, 0])])),
            Error::Unsupported("lossy WebP is not supported")
        ));
    }

    #[test]
    fn profiles_and_metadata_do_not_change_the_pixels() {
        let payload = solid(1, 1, [0, 0, 0, 255]);
        let header = vp8x(0x28, 1, 1);
        let icc = b"icc";
        let exif = b"exif-not-a-rotation";
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"ICCP", icc),
            (*b"VP8L", &payload),
            (*b"EXIF", exif),
        ]);
        let still = decode(&file).expect("extended");
        assert_eq!(still.pixels(), &[0, 0, 0, 255]);
    }

    #[test]
    fn a_byte_after_the_riff_payload_is_refused() {
        let mut file = vp8l_file(BLACK);
        file.push(0);
        assert!(matches!(
            refused(&file),
            Error::Inconsistent("a byte follows the file")
        ));
    }

    #[test]
    fn a_nonzero_padding_byte_is_refused() {
        let mut file = vp8l_file(BLACK);
        *file.last_mut().expect("the file has a pad byte") = 1;
        assert!(matches!(
            refused(&file),
            Error::Inconsistent("a padding byte is not zero")
        ));
    }

    #[test]
    fn an_unknown_chunk_before_the_image_is_refused() {
        let payload = solid(1, 1, [0, 0, 0, 255]);
        let header = vp8x(0, 1, 1);
        let unknown = b"no";
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"XYZZ", unknown),
            (*b"VP8L", &payload),
        ]);
        assert!(matches!(
            refused(&file),
            Error::Inconsistent("a chunk is out of order")
        ));
    }

    #[test]
    fn an_unknown_chunk_after_the_image_is_skipped() {
        let payload = solid(1, 1, [0, 0, 0, 255]);
        let header = vp8x(0, 1, 1);
        let unknown = b"no";
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"VP8L", &payload),
            (*b"XYZZ", unknown),
        ]);
        let still = decode(&file).expect("unknown");
        assert_eq!(still.pixels(), &[0, 0, 0, 255]);
    }

    #[test]
    fn a_canvas_that_does_not_match_the_image_is_refused() {
        let payload = solid(1, 1, [0, 0, 0, 255]);
        let header = vp8x(0, 2, 1);
        let file = riff(&[(*b"VP8X", &header), (*b"VP8L", &payload)]);
        assert!(matches!(
            refused(&file),
            Error::Inconsistent("the canvas does not match the image")
        ));
    }

    #[test]
    fn a_lossless_frame_with_an_alpha_chunk_is_refused() {
        let payload = solid(1, 1, [0, 0, 0, 255]);
        let header = vp8x(0x10, 1, 1);
        let alpha = [0u8; 4];
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"ALPH", &alpha),
            (*b"VP8L", &payload),
        ]);
        assert!(matches!(
            refused(&file),
            Error::Inconsistent("a lossless frame has an alpha chunk")
        ));
    }

    #[test]
    fn an_animation_keeps_the_background_and_the_first_frame() {
        let frame = solid(1, 1, [0, 0, 0, 255]);
        let header = vp8x(0x02, 2, 1);
        let background = anim(0, 0, 255, 255);
        let first = anmf(0, 0, 1, 1, true, &[(*b"VP8L", &frame)]);
        let bad = [0u8];
        let second = anmf(0, 0, 1, 1, true, &[(*b"VP8L", &bad)]);
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"ANIM", &background),
            (*b"ANMF", &first),
            (*b"ANMF", &second),
        ]);
        let still = decode(&file).expect("animation");
        assert_eq!(still.width(), 2);
        assert_eq!(still.pixels(), &[0, 0, 0, 255, 255, 0, 0, 255]);
    }

    #[test]
    fn a_frame_origin_is_twice_the_stored_value() {
        let frame = solid(1, 1, [255, 0, 0, 255]);
        let header = vp8x(0x02, 4, 1);
        let background = anim(255, 0, 0, 255);
        let placed = anmf(2, 0, 1, 1, true, &[(*b"VP8L", &frame)]);
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"ANIM", &background),
            (*b"ANMF", &placed),
        ]);
        let still = decode(&file).expect("origin");
        assert_eq!(
            still.pixels(),
            &[
                0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 255
            ]
        );
    }

    #[test]
    fn blending_truncates_in_the_stored_space() {
        let frame = solid(1, 1, [200, 0, 0, 100]);
        let header = vp8x(0x02, 1, 1);
        let background = anim(0, 0, 50, 200);
        let placed = anmf(0, 0, 1, 1, false, &[(*b"VP8L", &frame)]);
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"ANIM", &background),
            (*b"ANMF", &placed),
        ]);
        let still = decode(&file).expect("blend");
        assert_eq!(still.pixels(), &[118, 0, 0, 221]);
    }

    #[test]
    fn an_overwrite_keeps_the_frame_alpha() {
        let frame = solid(1, 1, [200, 0, 0, 100]);
        let header = vp8x(0x02, 1, 1);
        let background = anim(0, 0, 50, 200);
        let placed = anmf(0, 0, 1, 1, true, &[(*b"VP8L", &frame)]);
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"ANIM", &background),
            (*b"ANMF", &placed),
        ]);
        let still = decode(&file).expect("overwrite");
        assert_eq!(still.pixels(), &[200, 0, 0, 100]);
    }

    #[test]
    fn a_lossy_animation_frame_is_refused() {
        let header = vp8x(0x02, 1, 1);
        let background = anim(0, 0, 0, 255);
        let bits = [0u8; 4];
        let frame = anmf(0, 0, 1, 1, true, &[(*b"VP8 ", &bits)]);
        let file = riff(&[
            (*b"VP8X", &header),
            (*b"ANIM", &background),
            (*b"ANMF", &frame),
        ]);
        assert!(matches!(
            refused(&file),
            Error::Unsupported("lossy WebP is not supported")
        ));
    }
}
