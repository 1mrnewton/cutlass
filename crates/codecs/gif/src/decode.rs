//! GIF pictures from the still container's packets.
//!
//! The first packet carries the header, the logical screen and the global
//! color table; every packet carries graphic-control extensions and images.
//! The decoder keeps the screen between packets and paints each image onto
//! it, so every output is the whole screen as RGBA with the transparent
//! index left clear. The disposal of the previous image (keep, clear its
//! rectangle, or put back what was under it) runs before the next is drawn.
//! Packets must arrive in order from the first; a seek starts again at the
//! first packet, which is the only keyframe.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, Transfer,
};

/// The largest screen this decoder paints, in pixels.
const MAX_PIXELS: usize = 16_384 * 16_384 / 4;

/// LZW codes are at most twelve bits wide.
const MAX_CODES: usize = 4096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Rect {
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}

/// What a graphic-control extension says about the next image.
#[derive(Clone, Copy, Debug, Default)]
struct Control {
    disposal: u8,
    transparent: Option<u8>,
}

struct Screen {
    width: usize,
    height: usize,
    global: Vec<[u8; 3]>,
    canvas: Vec<u8>,
    /// The image drawn last, its disposal, and for disposal 3 what was
    /// under it.
    last: Option<(Rect, u8, Option<Vec<u8>>)>,
}

/// A GIF decoder that keeps the screen from packet to packet.
#[derive(Default)]
pub struct GifDecoder {
    screen: Option<Screen>,
    control: Control,
}

impl GifDecoder {
    /// Open a decoder. The screen size comes from the first packet.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Gif {
            return Err(CodecError::InvalidApi { what: "gif params" });
        }
        Ok(Self::default())
    }

    fn picture(&self, screen: &Screen, packet: &Packet) -> Result<Output, CodecError> {
        let _ = self;
        let info = PictureInfo {
            width: u32::try_from(screen.width).map_err(|_| limit("gif width"))?,
            height: u32::try_from(screen.height).map_err(|_| limit("gif height"))?,
            pixel: PixelFormat::Rgba8,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Srgb,
        };
        Ok(Output::Picture(PictureOutput::cpu(
            info,
            screen.canvas.clone(),
            packet.pts,
            packet.timescale,
        )?))
    }
}

impl Decoder for GifDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Gif
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let data = &packet.data;
        let mut at = 0;
        if data.starts_with(b"GIF8") {
            let (screen, next) = header(data)?;
            self.screen = Some(screen);
            self.control = Control::default();
            at = next;
        }
        let mut screen = self
            .screen
            .take()
            .ok_or(CodecError::InvalidApi { what: "gif header" })?;
        let mut outputs = Vec::new();
        let result = (|| {
            while at < data.len() {
                match data[at] {
                    0x21 => at = self.extension(data, at)?,
                    0x2C => {
                        at = self.image(&mut screen, data, at)?;
                        outputs.push(self.picture(&screen, packet)?);
                    }
                    0x3B => break,
                    // Some writers pad after the last block.
                    0x00 => at += 1,
                    _ => return Err(malformed(at, "gif block")),
                }
            }
            Ok(())
        })();
        self.screen = Some(screen);
        result?;
        Ok(outputs)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.screen = None;
        self.control = Control::default();
    }
}

impl GifDecoder {
    fn extension(&mut self, data: &[u8], at: usize) -> Result<usize, CodecError> {
        let label = *data
            .get(at + 1)
            .ok_or_else(|| truncated(at, "gif extension"))?;
        let mut index = at + 2;
        if label == 0xF9 {
            let size = usize::from(*data.get(index).ok_or_else(|| truncated(index, "gce"))?);
            if size >= 4 && index + 1 + size <= data.len() {
                let packed = data[index + 1];
                self.control = Control {
                    disposal: (packed >> 2) & 0x07,
                    transparent: (packed & 0x01 != 0).then_some(data[index + 4]),
                };
            }
        }
        // Skip the sub-blocks through the terminator.
        loop {
            let size = usize::from(
                *data
                    .get(index)
                    .ok_or_else(|| truncated(index, "gif sub-block"))?,
            );
            index += 1;
            if size == 0 {
                return Ok(index);
            }
            index += size;
            if index > data.len() {
                return Err(truncated(index, "gif sub-block"));
            }
        }
    }

    fn image(&mut self, screen: &mut Screen, data: &[u8], at: usize) -> Result<usize, CodecError> {
        if at + 10 > data.len() {
            return Err(truncated(at, "gif image descriptor"));
        }
        let rect = Rect {
            left: usize::from(u16::from_le_bytes([data[at + 1], data[at + 2]])),
            top: usize::from(u16::from_le_bytes([data[at + 3], data[at + 4]])),
            width: usize::from(u16::from_le_bytes([data[at + 5], data[at + 6]])),
            height: usize::from(u16::from_le_bytes([data[at + 7], data[at + 8]])),
        };
        let packed = data[at + 9];
        let mut index = at + 10;
        let local = if packed & 0x80 != 0 {
            let count = 2_usize << (packed & 0x07);
            let table = data
                .get(index..index + count * 3)
                .ok_or_else(|| truncated(index, "gif local table"))?;
            index += count * 3;
            Some(palette(table))
        } else {
            None
        };
        let interlaced = packed & 0x40 != 0;
        let min_size = *data
            .get(index)
            .ok_or_else(|| truncated(index, "gif code size"))?;
        index += 1;
        let mut packed_codes = Vec::new();
        loop {
            let size = usize::from(
                *data
                    .get(index)
                    .ok_or_else(|| truncated(index, "gif image data"))?,
            );
            index += 1;
            if size == 0 {
                break;
            }
            let block = data
                .get(index..index + size)
                .ok_or_else(|| truncated(index, "gif image data"))?;
            packed_codes.extend_from_slice(block);
            index += size;
        }
        let pixels = rect
            .width
            .checked_mul(rect.height)
            .filter(|pixels| *pixels <= MAX_PIXELS)
            .ok_or_else(|| limit("gif image size"))?;
        let indexes = expand(&packed_codes, min_size, pixels)?;

        dispose(screen);
        let control = std::mem::take(&mut self.control);
        let saved = (control.disposal == 3).then(|| screen.canvas.clone());
        let colors = local.as_deref().unwrap_or(&screen.global);
        for (n, &color) in indexes.iter().enumerate() {
            if Some(color) == control.transparent {
                continue;
            }
            let Some(rgb) = colors.get(usize::from(color)) else {
                continue;
            };
            let column = n % rect.width;
            let row = row_of(n / rect.width, rect.height, interlaced);
            let (x, y) = (rect.left + column, rect.top + row);
            if x >= screen.width || y >= screen.height {
                continue;
            }
            let at = (y * screen.width + x) * 4;
            screen.canvas[at..at + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
        screen.last = Some((rect, control.disposal, saved));
        Ok(index)
    }
}

fn header(data: &[u8]) -> Result<(Screen, usize), CodecError> {
    if data.len() < 13 || !(data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a")) {
        return Err(malformed(0, "gif signature"));
    }
    let width = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let height = usize::from(u16::from_le_bytes([data[8], data[9]]));
    let pixels = width
        .checked_mul(height)
        .filter(|pixels| *pixels > 0 && *pixels <= MAX_PIXELS)
        .ok_or_else(|| limit("gif screen"))?;
    let packed = data[10];
    let mut at = 13;
    let global = if packed & 0x80 != 0 {
        let count = 2_usize << (packed & 0x07);
        let table = data
            .get(at..at + count * 3)
            .ok_or_else(|| truncated(at, "gif global table"))?;
        at += count * 3;
        palette(table)
    } else {
        Vec::new()
    };
    Ok((
        Screen {
            width,
            height,
            global,
            canvas: vec![0; pixels * 4],
            last: None,
        },
        at,
    ))
}

fn palette(table: &[u8]) -> Vec<[u8; 3]> {
    table.as_chunks::<3>().0.to_vec()
}

/// Undo the last image as its disposal asks before the next is drawn.
fn dispose(screen: &mut Screen) {
    let Some((rect, disposal, saved)) = screen.last.take() else {
        return;
    };
    match disposal {
        2 => {
            for y in rect.top..(rect.top + rect.height).min(screen.height) {
                for x in rect.left..(rect.left + rect.width).min(screen.width) {
                    let at = (y * screen.width + x) * 4;
                    screen.canvas[at..at + 4].fill(0);
                }
            }
        }
        3 => {
            if let Some(saved) = saved {
                screen.canvas = saved;
            }
        }
        _ => {}
    }
}

/// The screen row of the `pass_row`-th stored row: interlaced images store
/// every eighth row from 0, every eighth from 4, every fourth from 2, then
/// every second from 1.
fn row_of(stored: usize, height: usize, interlaced: bool) -> usize {
    if !interlaced {
        return stored;
    }
    let mut stored = stored;
    for (start, step) in [(0, 8), (4, 8), (2, 4), (1, 2)] {
        let rows = height.saturating_sub(start).div_ceil(step);
        if stored < rows {
            return start + stored * step;
        }
        stored -= rows;
    }
    height
}

/// Decode LZW codes, least significant bit first, into at most `wanted`
/// color indexes. A short stream leaves the rest of the image index 0's
/// place untouched by returning fewer indexes.
fn expand(packed: &[u8], min_size: u8, wanted: usize) -> Result<Vec<u8>, CodecError> {
    if !(1..=11).contains(&min_size) {
        return Err(malformed(0, "gif code size"));
    }
    let clear = 1_usize << min_size;
    let end = clear + 1;
    let mut prefix = vec![0_u16; MAX_CODES];
    let mut suffix = vec![0_u8; MAX_CODES];
    let mut first = vec![0_u8; MAX_CODES];
    for code in 0..clear {
        suffix[code] = u8::try_from(code).unwrap_or(u8::MAX);
        first[code] = suffix[code];
    }
    let mut next = end + 1;
    let mut width = u32::from(min_size) + 1;
    let mut previous: Option<usize> = None;
    let mut out = Vec::with_capacity(wanted);
    let mut stack = Vec::with_capacity(MAX_CODES);
    let (mut held, mut count) = (0_u32, 0_u32);
    let mut bytes = packed.iter();
    while out.len() < wanted {
        while count < width {
            let Some(&byte) = bytes.next() else {
                return Ok(out);
            };
            held |= u32::from(byte) << count;
            count += 8;
        }
        let code = (held & ((1 << width) - 1)) as usize;
        held >>= width;
        count -= width;
        if code == clear {
            next = end + 1;
            width = u32::from(min_size) + 1;
            previous = None;
            continue;
        }
        if code == end {
            break;
        }
        let Some(last) = previous else {
            if code >= clear {
                return Err(malformed(0, "gif first code"));
            }
            out.push(suffix[code]);
            previous = Some(code);
            continue;
        };
        let head = if code < next {
            first[code]
        } else if code == next {
            first[last]
        } else {
            return Err(malformed(0, "gif code"));
        };
        if next < MAX_CODES {
            prefix[next] = u16::try_from(last).unwrap_or(0);
            suffix[next] = head;
            first[next] = first[last];
            next += 1;
            if next == 1 << width && width < 12 {
                width += 1;
            }
        }
        // Walk the code back to its first index, then write it forwards.
        let mut walk = code;
        stack.clear();
        while walk > end {
            stack.push(suffix[walk]);
            walk = usize::from(prefix[walk]);
        }
        stack.push(suffix[walk]);
        out.extend(stack.iter().rev().take(wanted - out.len()));
        previous = Some(code);
    }
    Ok(out)
}

fn truncated(at: usize, what: &'static str) -> CodecError {
    CodecError::Truncated {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}

fn malformed(at: usize, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}

fn limit(what: &'static str) -> CodecError {
    CodecError::Limit { what }
}

#[cfg(test)]
mod tests {
    use cutlass_decode::PictureSamples;
    use cutlass_encode::{CpuFrame, Encoder, FrameInput, FrameTime};

    use super::*;
    use crate::GifEncoder;

    fn params(width: u32, height: u32) -> CodecParams {
        CodecParams::with_format(CodecId::Gif, width, height, PixelFormat::Rgba8)
    }

    fn rgba(picture: &Output) -> Vec<u8> {
        let Output::Picture(picture) = picture else {
            panic!("picture");
        };
        match picture.samples() {
            PictureSamples::Cpu(bytes) => bytes.to_vec(),
            _ => panic!("cpu"),
        }
    }

    fn packet(data: Vec<u8>, pts: i64) -> Packet {
        Packet {
            stream: 0,
            data,
            pts,
            duration: 10,
            timescale: 100,
            keyframe: pts == 0,
            side: Vec::new(),
        }
    }

    #[test]
    fn the_encoders_frames_come_back_in_order() {
        // Few enough colors that the palette keeps each one exactly.
        let red = [200_u8, 10, 10, 255].repeat(6 * 4);
        let mut blue = [10_u8, 10, 200, 255].repeat(6 * 4);
        blue[0..4].copy_from_slice(&[10, 200, 10, 255]);
        let mut encoder = GifEncoder::open(&CodecParams::video(CodecId::Gif, 6, 4)).unwrap();
        let mut packets = Vec::new();
        for (n, frame) in [&red, &blue].into_iter().enumerate() {
            let time = FrameTime::new(i64::try_from(n).unwrap(), 1, 10).unwrap();
            let input = CpuFrame::new(6, 4, PixelFormat::Rgba8, frame.clone(), time).unwrap();
            packets.extend(encoder.push_frame(&FrameInput::Cpu(input)).unwrap());
        }
        packets.extend(encoder.drain().unwrap());
        let packets: Vec<Vec<u8>> = packets.into_iter().map(|packet| packet.data).collect();
        assert_eq!(packets.len(), 2);
        let mut decoder = GifDecoder::open(&params(6, 4)).unwrap();
        let first = decoder.push(&packet(packets[0].clone(), 0)).unwrap();
        let second = decoder.push(&packet(packets[1].clone(), 10)).unwrap();
        assert_eq!(rgba(&first[0]), red);
        assert_eq!(rgba(&second[0]), blue);
        // A later packet with no header first is refused after a reset.
        decoder.reset();
        assert!(decoder.push(&packet(packets[1].clone(), 10)).is_err());
    }

    /// A hand-written GIF: a 4x2 screen, a two-color global table, and
    /// frames written with the crate's LZW so each case is exact.
    type Drawn = (Rect, u8, Option<u8>, Vec<u8>, bool);

    fn file(frames: &[Drawn]) -> Vec<u8> {
        let mut out = b"GIF89a".to_vec();
        out.extend_from_slice(&4_u16.to_le_bytes());
        out.extend_from_slice(&2_u16.to_le_bytes());
        out.extend_from_slice(&[0x80, 0, 0]);
        out.extend_from_slice(&[255, 0, 0, 0, 0, 255]);
        for (rect, disposal, transparent, indexes, interlaced) in frames {
            let flag = u8::from(transparent.is_some());
            out.extend_from_slice(&[0x21, 0xF9, 4, (disposal << 2) | flag, 10, 0]);
            out.push(transparent.unwrap_or(0));
            out.push(0);
            out.push(0x2C);
            for value in [rect.left, rect.top, rect.width, rect.height] {
                out.extend_from_slice(&u16::try_from(value).unwrap().to_le_bytes());
            }
            out.push(if *interlaced { 0x40 } else { 0 });
            out.extend_from_slice(&crate::lzw::image_data(indexes, 1));
        }
        out.push(0x3B);
        out
    }

    const RED: [u8; 4] = [255, 0, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const CLEAR: [u8; 4] = [0, 0, 0, 0];

    fn pixels(output: &Output) -> Vec<[u8; 4]> {
        rgba(output).as_chunks::<4>().0.to_vec()
    }

    #[test]
    fn transparency_and_each_disposal_paint_the_screen() {
        let whole = Rect {
            left: 0,
            top: 0,
            width: 4,
            height: 2,
        };
        let corner = Rect {
            left: 1,
            top: 0,
            width: 2,
            height: 1,
        };
        let data = file(&[
            // Red everywhere but one transparent pixel; kept.
            (whole, 1, Some(1), vec![0, 0, 0, 0, 0, 0, 0, 1], false),
            // Blue in the corner, put back afterwards.
            (corner, 3, None, vec![1, 1], false),
            // Blue in the corner again, cleared afterwards.
            (corner, 2, None, vec![1, 1], false),
            // A transparent pixel over the cleared corner stays clear.
            (corner, 1, Some(0), vec![0, 1], false),
        ]);
        let mut decoder = GifDecoder::open(&params(4, 2)).unwrap();
        let frames = decoder.push(&packet(data, 0)).unwrap();
        assert_eq!(frames.len(), 4);
        assert_eq!(
            pixels(&frames[0]),
            [RED, RED, RED, RED, RED, RED, RED, CLEAR]
        );
        assert_eq!(
            pixels(&frames[1]),
            [RED, BLUE, BLUE, RED, RED, RED, RED, CLEAR]
        );
        // The red under the corner came back before the third was drawn.
        assert_eq!(
            pixels(&frames[2]),
            [RED, BLUE, BLUE, RED, RED, RED, RED, CLEAR]
        );
        assert_eq!(
            pixels(&frames[3]),
            [RED, CLEAR, BLUE, RED, RED, RED, RED, CLEAR]
        );
    }

    #[test]
    fn interlaced_rows_land_in_their_places() {
        assert_eq!(
            (0..8).map(|row| row_of(row, 8, true)).collect::<Vec<_>>(),
            [0, 4, 2, 6, 1, 3, 5, 7]
        );
        let tall = Rect {
            left: 0,
            top: 0,
            width: 4,
            height: 2,
        };
        // Stored rows: blue first (row 0), red second (row 1).
        let data = file(&[(tall, 0, None, vec![1, 1, 1, 1, 0, 0, 0, 0], true)]);
        let frames = GifDecoder::default().push(&packet(data, 0)).unwrap();
        assert_eq!(
            pixels(&frames[0]),
            [BLUE, BLUE, BLUE, BLUE, RED, RED, RED, RED]
        );
    }

    #[test]
    fn hostile_and_short_files_are_errors_not_panics() {
        let mut decoder = GifDecoder::default();
        assert!(decoder.push(&packet(b"GIF89a".to_vec(), 0)).is_err());
        let mut huge = b"GIF89a".to_vec();
        huge.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0]);
        assert!(matches!(
            decoder.push(&packet(huge, 0)),
            Err(CodecError::Limit { .. })
        ));
        let whole = Rect {
            left: 0,
            top: 0,
            width: 4,
            height: 2,
        };
        let data = file(&[(whole, 0, None, vec![0; 8], false)]);
        for cut in 0..data.len() {
            let _ = GifDecoder::default().push(&packet(data[..cut].to_vec(), 0));
        }
        let mut flipped = data.clone();
        for at in 13..flipped.len() {
            flipped[at] ^= 0x5A;
            let _ = GifDecoder::default().push(&packet(flipped.clone(), 0));
            flipped[at] ^= 0x5A;
        }
        assert!(matches!(
            expand(&[0xFF; 4], 0, 8),
            Err(CodecError::Malformed { .. })
        ));
    }
}
