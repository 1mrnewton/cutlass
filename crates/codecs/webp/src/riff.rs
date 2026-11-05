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

use crate::limits::{MAX_CHUNKS_PER_PACKET, MAX_ICC_BYTES};

/// The coded picture of a still.
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
        chunks.push(Chunk {
            fourcc,
            data: &data[body..body + size],
            at,
        });
        if chunks.len() > MAX_CHUNKS_PER_PACKET {
            return Err(CodecError::Limit {
                what: "webp chunks in one packet",
            });
        }
        let padded = body + size + (size & 1);
        if padded > end {
            return Err(CodecError::Truncated {
                at_bit: bit(at),
                what: "webp chunk padding",
            });
        }
        at = padded;
    }
    Ok(chunks)
}

fn u24(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

/// Read the chunks of one packet.
pub(crate) fn parse(data: &[u8]) -> Result<Still<'_>, CodecError> {
    if data.len() < 12 {
        return Err(CodecError::Truncated {
            at_bit: bit(data.len()),
            what: "webp riff header",
        });
    }
    if &data[..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "webp riff header",
        });
    }
    let declared = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let end = declared.saturating_add(8);
    if end > data.len() {
        return Err(CodecError::Truncated {
            at_bit: bit(data.len()),
            what: "webp riff form",
        });
    }
    let chunks = walk(data, 12, end)?;
    let Some(first) = chunks.first() else {
        return Err(CodecError::Truncated {
            at_bit: 96,
            what: "webp first chunk",
        });
    };
    match &first.fourcc {
        b"VP8L" => Ok(Still {
            bitstream: Bitstream::Lossless(first.data),
            canvas: None,
            icc: None,
        }),
        b"VP8 " => Ok(Still {
            bitstream: Bitstream::Lossy {
                vp8: first.data,
                alpha: None,
            },
            canvas: None,
            icc: None,
        }),
        b"VP8X" => extended(&chunks),
        _ => Err(CodecError::Malformed {
            at_bit: bit(first.at),
            what: "webp first chunk",
        }),
    }
}

fn extended<'a>(chunks: &[Chunk<'a>]) -> Result<Still<'a>, CodecError> {
    let vp8x = &chunks[0];
    if vp8x.data.len() < 10 {
        return Err(CodecError::Truncated {
            at_bit: bit(vp8x.at),
            what: "webp vp8x",
        });
    }
    if vp8x.data[0] & 0x02 != 0 {
        return Err(CodecError::Unsupported {
            what: "webp animation",
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
    let Some(picture) = chunks
        .iter()
        .find(|chunk| &chunk.fourcc == b"VP8 " || &chunk.fourcc == b"VP8L")
    else {
        return Err(CodecError::Malformed {
            at_bit: bit(vp8x.at),
            what: "webp still has no bitstream",
        });
    };
    // `ALPH` belongs to a lossy picture only; `VP8L` carries its own.
    let bitstream = if &picture.fourcc == b"VP8L" {
        Bitstream::Lossless(picture.data)
    } else {
        Bitstream::Lossy {
            vp8: picture.data,
            alpha: chunks
                .iter()
                .find(|chunk| &chunk.fourcc == b"ALPH")
                .map(|chunk| chunk.data),
        }
    };
    Ok(Still {
        bitstream,
        canvas: Some((width, height)),
        icc,
    })
}

#[cfg(test)]
mod tests {
    use super::{Bitstream, parse};

    fn put_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
        out.extend_from_slice(fourcc);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
    }

    fn file(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
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

    #[test]
    fn a_simple_lossless_file_is_one_bitstream() {
        let data = file(&[(b"VP8L", &[0x2f, 1, 2, 3, 4])]);
        let still = parse(&data).unwrap();
        assert!(matches!(still.bitstream, Bitstream::Lossless(bytes) if bytes.len() == 5));
        assert!(still.canvas.is_none());
    }

    #[test]
    fn an_extended_still_carries_its_canvas_and_icc() {
        let data = file(&[
            (b"VP8X", &[0x20, 0, 0, 0, 2, 0, 0, 4, 0, 0]),
            (b"ICCP", b"icc"),
            (b"VP8L", &[0x2f, 1, 2, 3, 4]),
        ]);
        let still = parse(&data).unwrap();
        assert_eq!(still.canvas, Some((3, 5)));
        assert_eq!(still.icc, Some(&b"icc"[..]));
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
}
