//! Malformed cookies and frames: each is refused with an error, none
//! panics or loops, and the decoder decodes the next good frame as before.

use cutlass_decode::CodecError;

use super::{encode_all, synth};
use crate::bits::{BitSink, BitWriter};
use crate::frame::{ID_CPE, ID_END, ID_SCE};
use crate::{AlacConfig, AlacDecoder, Cookie, EncodeMode, EncoderConfig, default_layout};

const ID_CCE: u32 = 2;
const ID_DSE: u32 = 4;
const ID_PCE: u32 = 5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn cookie(channels: u8, depth: u8) -> Vec<u8> {
    Cookie::new(AlacConfig::new(44_100, channels, depth, 4096)).to_bytes()
}

/// An element header: tag, instance, twelve unused bits, the partial flag
/// with its count, the shifted byte count, and the escape flag.
fn header(w: &mut BitWriter, tag: u32, count: Option<u32>, shift: u32, escape: bool) {
    w.put(tag, 3);
    w.put(0, 4);
    w.put(0, 12);
    w.put(u32::from(count.is_some()), 1);
    w.put(shift, 2);
    w.put(u32::from(escape), 1);
    if let Some(count) = count {
        w.put(count, 32);
    }
}

/// A compressed channel's parameters: mode, shift, adaptation, order, and
/// zero coefficients.
fn channel(w: &mut BitWriter, den_shift: u32, order: u32) {
    w.put(den_shift, 8);
    w.put((4 << 5) | order, 8);
    for _ in 0..order {
        w.put(0, 16);
    }
}

fn finish(mut w: BitWriter) -> Vec<u8> {
    w.put(ID_END, 3);
    w.align();
    w.into_bytes()
}

/// A mono 16-bit escape frame of `count` samples.
fn escape_mono(count: u32) -> Vec<u8> {
    let mut w = BitWriter::default();
    header(&mut w, ID_SCE, Some(count), 0, true);
    for i in 0..count {
        w.put(i & 0xffff, 16);
    }
    finish(w)
}

fn refused(decoder: &mut AlacDecoder, frame: &[u8]) -> CodecError {
    match decoder.decode_frame(frame) {
        Ok(decoded) => panic!("accepted a hostile frame: {} samples", decoded.frames),
        Err(error) => error,
    }
}

#[test]
fn hand_built_frames_decode_and_their_faults_are_named() {
    let mono = cookie(1, 16);
    let mut decoder = AlacDecoder::new(&mono).expect("decoder");
    let good = decoder.decode_frame(&escape_mono(5)).expect("good");
    assert_eq!(good.samples, [0, 1, 2, 3, 4]);

    // Unused header bits set.
    let mut w = BitWriter::default();
    w.put(ID_SCE, 3);
    w.put(0, 4);
    w.put(1, 12);
    assert!(matches!(
        refused(&mut decoder, &finish(w)),
        CodecError::Malformed { .. }
    ));

    // Three shifted bytes.
    let mut w = BitWriter::default();
    header(&mut w, ID_SCE, None, 3, false);
    assert!(matches!(
        refused(&mut decoder, &finish(w)),
        CodecError::Malformed { .. }
    ));

    // A partial count of zero, and one past the frame length.
    for count in [0, 4097] {
        let mut w = BitWriter::default();
        header(&mut w, ID_SCE, Some(count), 0, true);
        assert!(decoder.decode_frame(&finish(w)).is_err(), "count {count}");
    }

    // Shifted bytes in a 16-bit stream.
    let mut w = BitWriter::default();
    header(&mut w, ID_SCE, Some(4), 1, false);
    assert!(matches!(
        refused(&mut decoder, &finish(w)),
        CodecError::Malformed { .. }
    ));

    // A prediction shift of zero with a real predictor.
    let mut w = BitWriter::default();
    header(&mut w, ID_SCE, Some(4), 0, false);
    w.put(0, 8);
    w.put(0, 8);
    channel(&mut w, 0, 8);
    assert!(matches!(
        refused(&mut decoder, &finish(w)),
        CodecError::Malformed { .. }
    ));

    // A pair in a mono stream, and the coupling and program elements.
    let mut w = BitWriter::default();
    header(&mut w, ID_CPE, Some(4), 0, true);
    assert!(matches!(
        refused(&mut decoder, &finish(w)),
        CodecError::Malformed { .. }
    ));
    for tag in [ID_CCE, ID_PCE] {
        let mut w = BitWriter::default();
        w.put(tag, 3);
        assert!(matches!(
            refused(&mut decoder, &finish(w)),
            CodecError::Unsupported { .. }
        ));
    }

    // Nothing but the end, and an end before the second channel.
    assert!(decoder.decode_frame(&finish(BitWriter::default())).is_err());
    let mut stereo = AlacDecoder::new(&cookie(2, 16)).expect("stereo");
    assert!(stereo.decode_frame(&escape_mono(3)).is_err());

    // A data element that runs past the packet.
    let mut w = BitWriter::default();
    w.put(ID_DSE, 3);
    w.put(0, 4);
    w.put(0, 1);
    w.put(200, 8);
    assert!(matches!(
        refused(&mut decoder, &finish(w)),
        CodecError::Truncated { .. }
    ));

    // A data element and a fill element before the audio are skipped.
    let mut w = BitWriter::default();
    w.put(ID_DSE, 3);
    w.put(0, 4);
    w.put(1, 1);
    w.put(2, 8);
    w.align();
    w.put(0xabcd, 16);
    w.put(6, 3);
    w.put(1, 4);
    w.put(0xee, 8);
    header(&mut w, ID_SCE, Some(2), 0, true);
    w.put(7, 16);
    w.put(0xffff, 16);
    let skipped = decoder.decode_frame(&finish(w)).expect("skipped");
    assert_eq!(skipped.samples, [7, -1]);

    // Two elements that disagree on the sample count.
    let mut three = AlacDecoder::new(&cookie(3, 16)).expect("three");
    let mut w = BitWriter::default();
    header(&mut w, ID_SCE, Some(2), 0, true);
    w.put(1, 16);
    w.put(2, 16);
    header(&mut w, ID_CPE, Some(3), 0, true);
    for v in 0..6 {
        w.put(v, 16);
    }
    assert!(matches!(
        refused(&mut three, &finish(w)),
        CodecError::Malformed { .. }
    ));

    // An escape frame cut short.
    let whole = escape_mono(100);
    assert!(matches!(
        refused(&mut decoder, &whole[..50]),
        CodecError::Truncated { .. }
    ));

    // An oversized packet, and the decoder is still as it was.
    assert!(matches!(
        refused(&mut decoder, &vec![0; crate::MAX_PACKET_BYTES + 1]),
        CodecError::Limit { .. }
    ));
    assert_eq!(decoder.decode_frame(&escape_mono(5)).expect("again"), good);
}

/// Real frames at every depth and count, then every truncation, every
/// single-bit flip in the first bytes, and random bytes after a real
/// header. Errors are fine; panics and wrong lengths are not.
#[test]
fn damaged_real_frames_never_panic() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for (channels, depth, mode) in [
        (1u8, 16u8, EncodeMode::Default),
        (2, 16, EncodeMode::Default),
        (2, 24, EncodeMode::Fast),
        (2, 32, EncodeMode::Default),
        (6, 20, EncodeMode::Default),
        (8, 24, EncodeMode::Default),
    ] {
        let samples = synth(usize::from(channels), 3000, u32::from(depth), 99);
        let config = EncoderConfig {
            frame_length: 1024,
            mode,
            ..EncoderConfig::new(44_100, channels, depth)
        };
        let (cookie, packets) = encode_all(config, &default_layout(channels), &samples);
        let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
        let limit = usize::from(channels) * 1024;
        let check = |decoder: &mut AlacDecoder, data: &[u8]| {
            if let Ok(frame) = decoder.decode_frame(data) {
                assert!(frame.samples.len() <= limit);
                assert_eq!(frame.samples.len(), frame.frames * usize::from(channels));
            }
        };
        for packet in &packets {
            let data = &packet.data;
            let expected = decoder.decode_frame(data).expect("real frame");
            for cut in 0..data.len() {
                check(&mut decoder, &data[..cut]);
            }
            for bit in 0..(data.len() * 8).min(400) {
                let mut flipped = data.clone();
                flipped[bit / 8] ^= 0x80 >> (bit % 8);
                check(&mut decoder, &flipped);
            }
            for _ in 0..200 {
                let mut noisy = data.clone();
                let from = rng.below(noisy.len());
                for byte in &mut noisy[from..] {
                    *byte = rng.next() as u8;
                }
                check(&mut decoder, &noisy);
            }
            assert_eq!(decoder.decode_frame(data).expect("again"), expected);
        }
    }
}

#[test]
fn random_bytes_never_panic() {
    let mut rng = Rng(0xfeed_5eed_0bad_cafe);
    for (channels, depth) in [(1u8, 16u8), (2, 16), (2, 24), (5, 32), (8, 20)] {
        let mut decoder = AlacDecoder::new(&cookie(channels, depth)).expect("decoder");
        for _ in 0..20_000 {
            let len = 1 + rng.below(300);
            let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let _ = decoder.decode_frame(&data);
        }
    }
}

#[test]
fn malformed_cookies_are_refused() {
    let good = cookie(6, 24);
    assert!(Cookie::parse(&good).is_ok());
    for cut in 0..good.len() {
        // The layout atom is optional, so only a cut config must fail.
        let result = Cookie::parse(&good[..cut]);
        if cut < crate::CONFIG_BYTES {
            assert!(result.is_err(), "cut at {cut}");
        }
    }
    let field = |at: usize, value: u8| {
        let mut bytes = good.clone();
        bytes[at] = value;
        Cookie::parse(&bytes)
    };
    // Version, frame length, depth, kb, channel count.
    assert!(matches!(field(4, 1), Err(CodecError::Unsupported { .. })));
    let mut zero_length = good.clone();
    zero_length[..4].fill(0);
    zero_length[4] = 1;
    assert!(Cookie::parse(&zero_length).is_err());
    assert!(matches!(field(5, 18), Err(CodecError::Unsupported { .. })));
    assert!(field(8, 32).is_err());
    assert!(field(9, 0).is_err());
    assert!(matches!(field(9, 9), Err(CodecError::Unsupported { .. })));
    // A layout for another channel count.
    let mut tag = good.clone();
    tag[crate::CONFIG_BYTES + 15] = 5;
    assert!(matches!(
        Cookie::parse(&tag),
        Err(CodecError::Malformed { .. })
    ));
    let mut long = good.clone();
    long[..4].copy_from_slice(&(crate::MAX_FRAME_LENGTH + 1).to_be_bytes());
    assert!(matches!(
        Cookie::parse(&long),
        Err(CodecError::Limit { .. })
    ));

    let mut rng = Rng(0x0dd5_eed5);
    for _ in 0..50_000 {
        let len = rng.below(80);
        let mut bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        if len > 12 && rng.below(2) == 0 {
            bytes[4..8].copy_from_slice(b"alac");
        }
        if let Ok(cookie) = Cookie::parse(&bytes) {
            let _ = AlacDecoder::new(&cookie.to_bytes());
        }
    }
}
