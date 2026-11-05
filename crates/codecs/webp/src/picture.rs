//! RIFF chunks and the one picture they describe.
//!
//! A picture is `VP8 ` (lossy, optionally after an `ALPH` chunk) or `VP8L`
//! (lossless, alpha inside). Other chunks between them are skipped, as the
//! container specification asks of unknown chunks.

use cutlass_decode::CodecError;

use crate::alpha::decode_alpha;
use crate::lossless;
use crate::yuv::to_rgba;

/// One chunk: its four-character code and payload, padding excluded.
pub(crate) struct Chunk<'a> {
    pub fourcc: [u8; 4],
    pub payload: &'a [u8],
    /// Where the payload starts in the slice that was walked.
    pub at: usize,
}

/// Walk the chunks of `data` in order. A chunk that claims more bytes than
/// remain is truncated; a missing pad byte after the last chunk is not.
pub(crate) fn chunks(data: &[u8]) -> Result<Vec<Chunk<'_>>, CodecError> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < data.len() {
        let Some(header) = data.get(at..at + 8) else {
            return Err(truncated(at, "riff chunk header"));
        };
        let fourcc = [header[0], header[1], header[2], header[3]];
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let size = usize::try_from(size).unwrap_or(usize::MAX);
        let start = at + 8;
        let Some(payload) = start.checked_add(size).and_then(|end| data.get(start..end)) else {
            return Err(truncated(start, "riff chunk"));
        };
        out.push(Chunk {
            fourcc,
            payload,
            at: start,
        });
        at = start + size + (size & 1);
    }
    Ok(out)
}

pub(crate) fn u24(bytes: &[u8], at: usize) -> u32 {
    u32::from(bytes[at]) | u32::from(bytes[at + 1]) << 8 | u32::from(bytes[at + 2]) << 16
}

/// A decoded picture, straight RGBA.
pub(crate) struct Rgba {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
    /// The bitstream says it uses alpha: an `ALPH` chunk, or the VP8L flag.
    pub has_alpha: bool,
}

/// The size of the picture among `chunks`, without decoding it.
pub(crate) fn picture_size(chunks: &[Chunk<'_>]) -> Result<(u32, u32), CodecError> {
    for chunk in chunks {
        match &chunk.fourcc {
            b"VP8 " => return cutlass_vp8::frame_size(chunk.payload),
            b"VP8L" => return lossless::size(chunk.payload),
            _ => {}
        }
    }
    Err(malformed(0, "webp picture missing"))
}

/// Decode the picture among `chunks`.
pub(crate) fn decode_picture(chunks: &[Chunk<'_>], max_pixels: usize) -> Result<Rgba, CodecError> {
    let mut alpha_chunk = None;
    for chunk in chunks {
        match &chunk.fourcc {
            b"ALPH" => {
                if alpha_chunk.is_none() {
                    alpha_chunk = Some(chunk.payload);
                }
            }
            b"VP8 " => {
                let frame = cutlass_vp8::decode_frame(chunk.payload, max_pixels)?;
                let width = usize::try_from(frame.width).unwrap_or(0);
                let height = usize::try_from(frame.height).unwrap_or(0);
                let alpha = alpha_chunk
                    .map(|data| decode_alpha(data, width, height, max_pixels))
                    .transpose()?;
                return Ok(Rgba {
                    width,
                    height,
                    pixels: to_rgba(&frame, alpha.as_deref()),
                    has_alpha: alpha.is_some(),
                });
            }
            b"VP8L" => {
                let image = lossless::decode(chunk.payload, max_pixels)?;
                let mut pixels = Vec::with_capacity(image.pixels.len() * 4);
                for argb in image.pixels {
                    let [a, r, g, b] = argb.to_be_bytes();
                    pixels.extend_from_slice(&[r, g, b, a]);
                }
                return Ok(Rgba {
                    width: image.width,
                    height: image.height,
                    pixels,
                    has_alpha: image.has_alpha,
                });
            }
            _ => {}
        }
    }
    Err(malformed(0, "webp picture missing"))
}

pub(crate) fn truncated(at: usize, what: &'static str) -> CodecError {
    CodecError::Truncated {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}

pub(crate) fn malformed(at: usize, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_walk_with_padding_and_refuse_overruns() {
        let mut data = b"ABCD".to_vec();
        data.extend_from_slice(&3_u32.to_le_bytes());
        data.extend_from_slice(&[1, 2, 3, 0]);
        data.extend_from_slice(b"EFGH");
        data.extend_from_slice(&1_u32.to_le_bytes());
        data.push(9);
        let walked = chunks(&data).unwrap();
        assert_eq!(walked.len(), 2);
        assert_eq!(walked[0].payload, [1, 2, 3]);
        assert_eq!(&walked[1].fourcc, b"EFGH");
        assert_eq!(walked[1].payload, [9]);
        let mut long = data.clone();
        long[4] = 200;
        assert!(chunks(&long).is_err());
        assert!(chunks(&data[..6]).is_err());
        let mut huge = data;
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(chunks(&huge).is_err());
    }
}
