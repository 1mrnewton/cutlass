//! The chunks inside one WebP packet (RFC 9649 section 2).
//!
//! The `still` container identifies the file, splits an animation into one
//! packet per `ANMF` chunk, and stamps the frame times. The packet it hands
//! over is file bytes: the whole file for a still, and for an animation the
//! file start through the first `ANMF` chunk, then each later `ANMF` chunk.
//! This module walks the chunks of such a packet to find the bitstreams and
//! the frame fields the pixels depend on. It does not decide packet
//! boundaries or timing. It also writes those chunks for the encoder.

use cutlass_decode::CodecError;

use crate::limits::{MAX_CHUNKS_PER_PACKET, MAX_FRAMES_PER_PACKET, MAX_ICC_BYTES};
use crate::lossless::decode::header;

/// The coded picture of a still or of one frame.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Bitstream<'a> {
    /// A `VP8 ` key frame and, for an extended file, its `ALPH` payload.
    Lossy {
        vp8: &'a [u8],
        alpha: Option<&'a [u8]>,
    },
    /// A `VP8L` bitstream, header included.
    Lossless(&'a [u8]),
}

/// One still picture.
pub(crate) struct Still<'a> {
    pub bitstream: Bitstream<'a>,
    /// `VP8X` canvas size, which must match the bitstream.
    pub canvas: Option<(u32, u32)>,
    pub icc: Option<&'a [u8]>,
}

struct Chunk<'a> {
    fourcc: [u8; 4],
    data: &'a [u8],
    /// Byte offset of the chunk header in the walked buffer.
    at: usize,
}

fn bit(at: usize) -> u64 {
    (at as u64).saturating_mul(8)
}

/// The chunks of `data[start..end]`, in order. A chunk that runs past `end`
/// is truncated.
fn walk(data: &[u8], start: usize, end: usize) -> Result<Vec<Chunk<'_>>, CodecError> {
    let mut chunks = Vec::new();
    let mut at = start;
    while at < end {
        let (chunk, next) = chunk_at(data, at, end)?;
        chunks.push(chunk);
        if chunks.len() > MAX_CHUNKS_PER_PACKET {
            return Err(CodecError::Limit {
                what: "webp chunks in one packet",
            });
        }
        at = next;
    }
    Ok(chunks)
}

/// The chunk at `at`, which must end by `end`, and where the next starts.
fn chunk_at(data: &[u8], at: usize, end: usize) -> Result<(Chunk<'_>, usize), CodecError> {
    if end - at < 8 {
        return Err(CodecError::Truncated {
            at_bit: bit(at),
            what: "webp chunk header",
        });
    }
    let fourcc = [data[at], data[at + 1], data[at + 2], data[at + 3]];
    let size = u32::from_le_bytes([data[at + 4], data[at + 5], data[at + 6], data[at + 7]]);
    let size = usize::try_from(size).map_err(|_| CodecError::Limit {
        what: "webp chunk size",
    })?;
    let body = at + 8;
    if size > end - body {
        return Err(CodecError::Truncated {
            at_bit: bit(at),
            what: "webp chunk",
        });
    }
    let padded = body + size + (size & 1);
    if padded > end {
        return Err(CodecError::Truncated {
            at_bit: bit(at),
            what: "webp chunk padding",
        });
    }
    let chunk = Chunk {
        fourcc,
        data: &data[body..body + size],
        at,
    };
    Ok((chunk, padded))
}

fn u24(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

/// `VP8X` flag of an animation.
const VP8X_ANIMATION: u8 = 0x02;

/// What one packet holds.
pub(crate) enum Layout<'a> {
    Still(Still<'a>),
    /// Frames of an animation. The header comes with the packet that starts
    /// the file; later packets carry `ANMF` chunks only.
    Animation {
        header: Option<AnimationHeader<'a>>,
        frames: Vec<Frame<'a>>,
    },
}

/// The canvas of an animation, from `VP8X`, and its profile.
pub(crate) struct AnimationHeader<'a> {
    pub width: u32,
    pub height: u32,
    pub icc: Option<&'a [u8]>,
}

/// One `ANMF` chunk (RFC 9649 section 2.7.1.1).
pub(crate) struct Frame<'a> {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Milliseconds as stored.
    pub duration: u32,
    pub dispose_background: bool,
    /// The blending bit is clear: alpha-blend over the canvas.
    pub blend: bool,
    /// An `ALPH` chunk, or a `VP8L` header that sets its alpha hint.
    pub has_alpha: bool,
    pub bitstream: Bitstream<'a>,
}

/// Read the chunks of one packet.
pub(crate) fn parse(data: &[u8]) -> Result<Layout<'_>, CodecError> {
    if !data.starts_with(b"RIFF") {
        return later_frames(data);
    }
    if data.len() < 12 {
        return Err(CodecError::Truncated {
            at_bit: bit(data.len()),
            what: "webp riff header",
        });
    }
    if &data[8..12] != b"WEBP" {
        return Err(CodecError::Malformed {
            at_bit: 64,
            what: "webp riff header",
        });
    }
    let declared = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let declared_end = declared.saturating_add(8);
    // The first packet of a split animation declares the whole file.
    let short = declared_end > data.len();
    let chunks = walk(data, 12, declared_end.min(data.len()))?;
    let Some(first) = chunks.first() else {
        return Err(CodecError::Truncated {
            at_bit: 96,
            what: "webp first chunk",
        });
    };
    let truncated = || CodecError::Truncated {
        at_bit: bit(data.len()),
        what: "webp riff form",
    };
    match &first.fourcc {
        b"VP8L" | b"VP8 " if short => Err(truncated()),
        b"VP8L" => Ok(Layout::Still(Still {
            bitstream: Bitstream::Lossless(first.data),
            canvas: None,
            icc: None,
        })),
        b"VP8 " => Ok(Layout::Still(Still {
            bitstream: Bitstream::Lossy {
                vp8: first.data,
                alpha: None,
            },
            canvas: None,
            icc: None,
        })),
        b"VP8X" => {
            let layout = extended(&chunks)?;
            if short && matches!(layout, Layout::Still(_)) {
                return Err(truncated());
            }
            Ok(layout)
        }
        _ => Err(CodecError::Malformed {
            at_bit: bit(first.at),
            what: "webp first chunk",
        }),
    }
}

/// A packet after the first of an animation: `ANMF` chunks, perhaps after
/// unknown chunks, and for the last packet whatever follows in the file.
fn later_frames(data: &[u8]) -> Result<Layout<'_>, CodecError> {
    let mut frames = Vec::new();
    let mut at = 0;
    let mut count = 0;
    while at < data.len() {
        let (chunk, next) = match chunk_at(data, at, data.len()) {
            Ok(found) => found,
            // Bytes past the RIFF form end the last packet; anything else
            // that is not a whole chunk is refused.
            Err(_) if !frames.is_empty() => break,
            Err(error) => return Err(error),
        };
        count += 1;
        if count > MAX_CHUNKS_PER_PACKET {
            return Err(CodecError::Limit {
                what: "webp chunks in one packet",
            });
        }
        if &chunk.fourcc == b"ANMF" {
            push_frame(&mut frames, &chunk)?;
        }
        at = next;
    }
    if frames.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "webp packet has no frame",
        });
    }
    Ok(Layout::Animation {
        header: None,
        frames,
    })
}

fn extended<'a>(chunks: &[Chunk<'a>]) -> Result<Layout<'a>, CodecError> {
    let vp8x = &chunks[0];
    if vp8x.data.len() < 10 {
        return Err(CodecError::Truncated {
            at_bit: bit(vp8x.at),
            what: "webp vp8x",
        });
    }
    let width = u24(&vp8x.data[4..7]) + 1;
    let height = u24(&vp8x.data[7..10]) + 1;
    let icc = chunks
        .iter()
        .find(|chunk| &chunk.fourcc == b"ICCP")
        .map(|chunk| chunk.data);
    if let Some(profile) = icc
        && profile.len() > MAX_ICC_BYTES
    {
        return Err(CodecError::Limit {
            what: "webp icc profile",
        });
    }
    if vp8x.data[0] & VP8X_ANIMATION != 0 {
        return animation(chunks, width, height, icc);
    }
    let Some(picture) = chunks
        .iter()
        .find(|chunk| &chunk.fourcc == b"VP8 " || &chunk.fourcc == b"VP8L")
    else {
        return Err(CodecError::Malformed {
            at_bit: bit(vp8x.at),
            what: "webp still has no bitstream",
        });
    };
    Ok(Layout::Still(Still {
        bitstream: bitstream(picture, chunks),
        canvas: Some((width, height)),
        icc,
    }))
}

/// The picture chunk as a bitstream. `ALPH` belongs to a lossy picture
/// only; `VP8L` carries its own alpha.
fn bitstream<'a>(picture: &Chunk<'a>, siblings: &[Chunk<'a>]) -> Bitstream<'a> {
    if &picture.fourcc == b"VP8L" {
        Bitstream::Lossless(picture.data)
    } else {
        Bitstream::Lossy {
            vp8: picture.data,
            alpha: siblings
                .iter()
                .find(|chunk| &chunk.fourcc == b"ALPH")
                .map(|chunk| chunk.data),
        }
    }
}

fn animation<'a>(
    chunks: &[Chunk<'a>],
    width: u32,
    height: u32,
    icc: Option<&'a [u8]>,
) -> Result<Layout<'a>, CodecError> {
    let mut frames = Vec::new();
    let mut saw_anim = false;
    for chunk in chunks {
        match &chunk.fourcc {
            b"ANIM" if chunk.data.len() < 6 => {
                return Err(CodecError::Truncated {
                    at_bit: bit(chunk.at),
                    what: "webp anim chunk",
                });
            }
            b"ANIM" => saw_anim = true,
            b"ANMF" if !saw_anim => {
                return Err(CodecError::Malformed {
                    at_bit: bit(chunk.at),
                    what: "webp anmf before anim",
                });
            }
            b"ANMF" => push_frame(&mut frames, chunk)?,
            _ => {}
        }
    }
    if frames.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "webp animation has no frame",
        });
    }
    Ok(Layout::Animation {
        header: Some(AnimationHeader { width, height, icc }),
        frames,
    })
}

fn push_frame<'a>(frames: &mut Vec<Frame<'a>>, chunk: &Chunk<'a>) -> Result<(), CodecError> {
    if frames.len() >= MAX_FRAMES_PER_PACKET {
        return Err(CodecError::Limit {
            what: "webp frames in one packet",
        });
    }
    frames.push(frame(chunk)?);
    Ok(())
}

fn frame<'a>(chunk: &Chunk<'a>) -> Result<Frame<'a>, CodecError> {
    let data = chunk.data;
    if data.len() < 16 {
        return Err(CodecError::Truncated {
            at_bit: bit(chunk.at),
            what: "webp anmf header",
        });
    }
    let inner = walk(data, 16, data.len())?;
    let Some(picture) = inner
        .iter()
        .find(|inner| &inner.fourcc == b"VP8 " || &inner.fourcc == b"VP8L")
    else {
        return Err(CodecError::Malformed {
            at_bit: bit(chunk.at),
            what: "webp frame has no bitstream",
        });
    };
    let bitstream = bitstream(picture, &inner);
    let has_alpha = match bitstream {
        Bitstream::Lossy { alpha, .. } => alpha.is_some(),
        Bitstream::Lossless(data) => header(data).is_ok_and(|header| header.alpha),
    };
    let flags = data[15];
    Ok(Frame {
        x: u24(&data[0..3]) * 2,
        y: u24(&data[3..6]) * 2,
        width: u24(&data[6..9]) + 1,
        height: u24(&data[9..12]) + 1,
        duration: u24(&data[12..15]),
        dispose_background: flags & 0x01 != 0,
        blend: flags & 0x02 == 0,
        has_alpha,
        bitstream,
    })
}

/// Append one chunk, padded to an even length.
pub(crate) fn put_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

/// A whole file of `chunks`.
pub(crate) fn file(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut body = Vec::new();
    for (fourcc, payload) in chunks {
        put_chunk(&mut body, fourcc, payload);
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32 + 4).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(&body);
    out
}

/// The chunks of a whole file, for tests that rebuild one.
#[cfg(test)]
pub(crate) fn chunks(data: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    walk(data, 12, data.len())
        .unwrap_or_default()
        .into_iter()
        .map(|chunk| (chunk.fourcc, chunk.data.to_vec()))
        .collect()
}

/// The ten-byte `VP8X` payload.
pub(crate) fn vp8x(flags: u8, width: u32, height: u32) -> [u8; 10] {
    let w = (width - 1).to_le_bytes();
    let h = (height - 1).to_le_bytes();
    [flags, 0, 0, 0, w[0], w[1], w[2], h[0], h[1], h[2]]
}

/// `VP8X` flag bits.
pub(crate) const FLAG_ICC: u8 = 0x20;
pub(crate) const FLAG_ALPHA: u8 = 0x10;
pub(crate) const FLAG_ANIMATION: u8 = 0x02;

/// A still file. It is the simple form, one bitstream chunk, unless an
/// ICC profile or a lossy alpha plane needs the extended form.
pub(crate) fn write_still(
    bitstream: Bitstream<'_>,
    width: u32,
    height: u32,
    alpha: bool,
    icc: Option<&[u8]>,
) -> Vec<u8> {
    let lossy_alpha = matches!(bitstream, Bitstream::Lossy { alpha: Some(_), .. });
    let mut chunks: Vec<(&[u8; 4], &[u8])> = Vec::with_capacity(4);
    let header;
    if icc.is_some() || lossy_alpha {
        let mut flags = 0;
        if icc.is_some() {
            flags |= FLAG_ICC;
        }
        if alpha {
            flags |= FLAG_ALPHA;
        }
        header = vp8x(flags, width, height);
        chunks.push((b"VP8X", &header));
    }
    if let Some(icc) = icc {
        chunks.push((b"ICCP", icc));
    }
    push_bitstream(&mut chunks, bitstream);
    file(&chunks)
}

/// The `ALPH` and `VP8 `, or the `VP8L`, chunk of a bitstream.
pub(crate) fn push_bitstream<'a>(
    chunks: &mut Vec<(&'static [u8; 4], &'a [u8])>,
    bitstream: Bitstream<'a>,
) {
    match bitstream {
        Bitstream::Lossy { vp8, alpha } => {
            if let Some(alpha) = alpha {
                chunks.push((b"ALPH", alpha));
            }
            chunks.push((b"VP8 ", vp8));
        }
        Bitstream::Lossless(vp8l) => chunks.push((b"VP8L", vp8l)),
    }
}

#[cfg(test)]
mod tests {
    use super::{Bitstream, Layout, Still, file, parse, put_chunk, vp8x};

    fn still_of(data: &[u8]) -> Still<'_> {
        match parse(data).unwrap() {
            Layout::Still(still) => still,
            Layout::Animation { .. } => panic!("animation"),
        }
    }

    #[test]
    fn a_simple_lossless_file_is_one_bitstream() {
        let data = file(&[(b"VP8L", &[0x2f, 1, 2, 3, 4])]);
        let still = still_of(&data);
        assert!(matches!(still.bitstream, Bitstream::Lossless(bytes) if bytes.len() == 5));
        assert!(still.canvas.is_none());
    }

    #[test]
    fn an_extended_still_carries_its_canvas_and_icc() {
        let data = file(&[
            (b"VP8X", &vp8x(0x20, 3, 5)),
            (b"ICCP", b"icc"),
            (b"VP8L", &[0x2f, 1, 2, 3, 4]),
        ]);
        let still = still_of(&data);
        assert_eq!(still.canvas, Some((3, 5)));
        assert_eq!(still.icc, Some(&b"icc"[..]));
    }

    #[test]
    fn alph_belongs_to_a_lossy_picture_only() {
        let lossy = file(&[
            (b"VP8X", &vp8x(0x10, 3, 5)),
            (b"ALPH", &[0, 1, 2]),
            (b"VP8 ", &[9; 11]),
        ]);
        let still = still_of(&lossy);
        let Bitstream::Lossy { vp8, alpha } = still.bitstream else {
            panic!("lossy");
        };
        assert_eq!(vp8.len(), 11);
        assert_eq!(alpha, Some(&[0u8, 1, 2][..]));
        let lossless = file(&[
            (b"VP8X", &vp8x(0x10, 3, 5)),
            (b"ALPH", &[0, 1, 2]),
            (b"VP8L", &[0x2f, 1, 2, 3, 4]),
        ]);
        assert!(matches!(
            still_of(&lossless).bitstream,
            Bitstream::Lossless(_)
        ));
    }

    #[test]
    fn a_chunk_past_the_form_is_truncated() {
        let mut data = file(&[(b"VP8L", &[0x2f, 1, 2, 3, 4, 5])]);
        data.truncate(data.len() - 2);
        assert!(parse(&data).is_err());
        let mut lying = file(&[(b"VP8L", &[0x2f, 1, 2, 3, 4, 5])]);
        lying[16] = 0xff;
        assert!(parse(&lying).is_err());
    }

    #[test]
    fn frames_read_offsets_flags_and_bitstreams() {
        let mut anmf = vec![1, 0, 0, 2, 0, 0, 9, 0, 0, 4, 0, 0, 100, 0, 0, 0b11];
        put_chunk(&mut anmf, b"VP8L", &[0x2f, 0, 0, 0, 0x10]);
        let data = file(&[
            (b"VP8X", &vp8x(0x02, 40, 40)),
            (b"ANIM", &[1, 2, 3, 4, 5, 0]),
            (b"ANMF", &anmf),
        ]);
        let Layout::Animation { header, frames } = parse(&data).unwrap() else {
            panic!("animation");
        };
        let header = header.unwrap();
        assert_eq!((header.width, header.height), (40, 40));
        let frame = &frames[0];
        assert_eq!((frame.x, frame.y, frame.width, frame.height), (2, 4, 10, 5));
        assert_eq!(frame.duration, 100);
        assert!(frame.dispose_background);
        assert!(!frame.blend);
        assert!(frame.has_alpha);
        // A later packet is the ANMF chunk, with whatever trails the file.
        let mut later = Vec::new();
        put_chunk(&mut later, b"ANMF", &anmf);
        put_chunk(&mut later, b"EXIF", b"exif");
        later.extend_from_slice(b"junk");
        let Layout::Animation { header, frames } = parse(&later).unwrap() else {
            panic!("animation");
        };
        assert!(header.is_none());
        assert_eq!(frames.len(), 1);
        // The first packet of a split file declares more than it holds.
        let mut first = data.clone();
        first[4..8].copy_from_slice(&(data.len() as u32 + 100).to_le_bytes());
        assert!(matches!(parse(&first).unwrap(), Layout::Animation { .. }));
    }

    #[test]
    fn an_animation_without_anim_is_malformed() {
        let mut anmf = vec![0; 16];
        put_chunk(&mut anmf, b"VP8L", &[0x2f, 0, 0, 0, 0]);
        let data = file(&[(b"VP8X", &vp8x(0x02, 4, 4)), (b"ANMF", &anmf)]);
        assert!(parse(&data).is_err());
        assert!(parse(b"junk").is_err());
    }
}
