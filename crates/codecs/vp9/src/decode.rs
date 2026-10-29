//! Stateful profile 0 decoder.
//!
//! A failed frame leaves references and probabilities as they were. Reference
//! slots update only after reconstruction succeeds. `show_existing_frame`
//! displays a stored picture and does not refresh those slots.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, Packet, Plane, VideoDecoder, VideoFrame, VideoOutput,
};
use cutlass_encode::PixelFormat;

use crate::geom::{self, SWITCHABLE, TX_MODE_SELECT};
use crate::header::{Header, Parsed, Persist, parse};
use crate::prob::{self, Counts, FrameContext};
use crate::recon::{Motion, Picture, RefSlots, Reference, reconstruct};

#[derive(Clone)]
pub struct Decoder {
    persist: Persist,
    refs: [Option<Arc<Reference>>; 8],
    prev: Option<Arc<Motion>>,
    last_seg: Vec<u8>,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            persist: Persist::new(),
            refs: [const { None }; 8],
            prev: None,
            last_seg: Vec::new(),
        }
    }

    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<Picture>, CodecError> {
        if packet.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let parsed = parse(&self.persist, packet)?;
        if parsed.header.show_existing {
            let picture = self.shown(&parsed.header)?;
            self.persist.last_width = parsed.header.width;
            self.persist.last_height = parsed.header.height;
            return Ok(Some(picture));
        }
        let prev = self.prev.clone();
        let last_seg = seg_source(&self.last_seg, &parsed.header);
        let built = {
            let slots = self.slots();
            reconstruct(&parsed, &slots, prev.as_deref(), &last_seg)?
        };
        self.commit(
            &parsed,
            &built.picture,
            &built.counts,
            built.motion,
            built.segmap,
        );
        if parsed.header.show_frame {
            Ok(Some(built.picture))
        } else {
            Ok(None)
        }
    }

    pub fn reset(&mut self) {
        self.persist = Persist::new();
        self.refs = [const { None }; 8];
        self.prev = None;
        self.last_seg.clear();
    }

    fn slots(&self) -> RefSlots<'_> {
        RefSlots {
            frames: std::array::from_fn(|index| self.refs[index].as_deref()),
        }
    }

    fn shown(&self, header: &Header) -> Result<Picture, CodecError> {
        let slot = usize::from(header.frame_to_show);
        let reference =
            self.refs
                .get(slot)
                .and_then(|slot| slot.as_ref())
                .ok_or(CodecError::Malformed {
                    at_bit: 0,
                    what: "shown reference",
                })?;
        Ok(Picture {
            y: reference.y.samples.clone(),
            u: reference.u.samples.clone(),
            v: reference.v.samples.clone(),
            width: reference.width,
            height: reference.height,
        })
    }

    fn commit(
        &mut self,
        parsed: &Parsed<'_>,
        picture: &Picture,
        counts: &Counts,
        motion: Motion,
        segmap: Vec<u8>,
    ) {
        let header = &parsed.header;
        let after_key = self.persist.key_frame;
        let fc = adapted(parsed, counts, after_key);
        self.persist.width = header.width;
        self.persist.height = header.height;
        self.persist.last_width = header.width;
        self.persist.last_height = header.height;
        self.persist.need_resync = header.need_resync;
        self.persist.key_frame = header.key_frame;
        self.persist.intra_only = header.intra_only;
        self.persist.last_show_frame = header.show_frame;
        self.persist.sign_bias = header.sign_bias;
        self.persist.seg = header.seg.clone();
        self.persist.ref_deltas = header.ref_deltas;
        self.persist.mode_deltas = header.mode_deltas;
        self.persist.contexts = parsed.contexts.clone();
        if header.refresh_context {
            let index = usize::from(header.context_idx);
            self.persist.contexts[index] = fc;
        }
        self.prev = Some(Arc::new(motion));
        self.store_refs(header, picture);
        self.store_seg(header, segmap);
    }

    fn store_seg(&mut self, header: &Header, segmap: Vec<u8>) {
        let cells = geom::mi_count(header.width).saturating_mul(geom::mi_count(header.height));
        if header.seg.enabled {
            self.last_seg = segmap;
        } else if header.is_intra() || header.error_resilient {
            self.last_seg = vec![0u8; cells];
        }
    }

    fn store_refs(&mut self, header: &Header, picture: &Picture) {
        if header.refresh_flags == 0 {
            return;
        }
        let Ok(reference) = Reference::from_picture(picture) else {
            return;
        };
        let shared = Arc::new(reference);
        for slot in 0..8 {
            let bit = 1u8 << u32::try_from(slot).unwrap_or(0);
            if header.refresh_flags & bit == 0 {
                continue;
            }
            self.refs[slot] = Some(Arc::clone(&shared));
            self.persist.ref_ok[slot] = true;
            self.persist.ref_w[slot] = header.width;
            self.persist.ref_h[slot] = header.height;
        }
    }
}

fn seg_source(last: &[u8], header: &Header) -> Vec<u8> {
    let cells = geom::mi_count(header.width).saturating_mul(geom::mi_count(header.height));
    if header.is_intra() || header.error_resilient || last.len() != cells {
        vec![0u8; cells]
    } else {
        last.to_vec()
    }
}

fn adapted(parsed: &Parsed<'_>, counts: &Counts, after_key: bool) -> FrameContext {
    let header = &parsed.header;
    let mut fc = parsed.fc.clone();
    if header.error_resilient || header.frame_parallel {
        return fc;
    }
    let pre = &parsed.contexts[usize::from(header.context_idx)];
    let key = header.is_intra();
    for tx in 0..4 {
        prob::adapt_coef(&mut fc, pre, counts, tx, key, after_key);
    }
    if !key {
        prob::adapt_modes(
            &mut fc,
            pre,
            counts,
            header.tx_mode == TX_MODE_SELECT,
            header.interp == SWITCHABLE,
        );
        prob::adapt_mv(&mut fc, pre, counts, header.allow_hp);
    }
    fc
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Profile-0 decoder on the video codec contract.
///
/// `push` consumes one access unit. A hidden frame returns no picture.
/// `drain` is empty. `reset` drops reference pictures. A failed `push`
/// leaves those pictures as they were.
pub struct Vp9Decoder {
    inner: Decoder,
    cancel: Arc<AtomicBool>,
}

impl Vp9Decoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp9 {
            return Err(CodecError::Unsupported {
                what: "vp9 decoder",
            });
        }
        if params.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "vp9 pixel format",
            });
        }
        check_open_size(params.width, params.height)?;
        Ok(Self {
            inner: Decoder::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl VideoDecoder for Vp9Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let parts = split_superframe(&packet.data)?;
        let mut checkpoint = (parts.len() > 1).then(|| self.inner.clone());
        let mut outputs = Vec::new();
        for part in parts {
            match self.inner.decode(part) {
                Ok(Some(picture)) => outputs.push(VideoOutput {
                    frame: picture_frame(&picture)?,
                    pts: packet.pts,
                    duration: packet.duration,
                    timescale: packet.timescale,
                }),
                Ok(None) => {}
                Err(err) => {
                    if let Some(saved) = checkpoint.take() {
                        self.inner = saved;
                    }
                    return Err(err);
                }
            }
        }
        Ok(outputs)
    }

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

fn picture_frame(picture: &Picture) -> Result<VideoFrame, CodecError> {
    let chroma_w = picture.width.div_ceil(2);
    let chroma_h = picture.height.div_ceil(2);
    VideoFrame::yuv420p(
        picture.width,
        picture.height,
        Plane::tight(picture.width, picture.height, picture.y.clone())?,
        Plane::tight(chroma_w, chroma_h, picture.u.clone())?,
        Plane::tight(chroma_w, chroma_h, picture.v.clone())?,
    )
}

fn split_superframe(data: &[u8]) -> Result<Vec<&[u8]>, CodecError> {
    let Some(marker) = data.last().copied() else {
        return Ok(vec![data]);
    };
    if marker & 0xe0 != 0xc0 {
        return Ok(vec![data]);
    }
    let frames = usize::from(marker & 0x7) + 1;
    let mag = usize::from((marker >> 3) & 0x3) + 1;
    let index = 2 + mag * frames;
    if data.len() < index || data[data.len() - index] != marker {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "superframe",
        });
    }
    let base = data.len() - index + 1;
    let payload = data.len() - index;
    let mut offset = 0usize;
    let mut parts = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut size = 0usize;
        for byte in 0..mag {
            let shift = u32::try_from(8 * byte).unwrap_or(0);
            size |= usize::from(data[base + frame * mag + byte]) << shift;
        }
        let end = offset
            .checked_add(size)
            .ok_or(CodecError::Limit { what: "superframe" })?;
        if end > payload {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "superframe",
            });
        }
        parts.push(&data[offset..end]);
        offset = end;
    }
    if offset != payload {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "superframe",
        });
    }
    Ok(parts)
}

fn check_open_size(width: u32, height: u32) -> Result<(), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    if width > geom::MAX_EDGE || height > geom::MAX_EDGE {
        return Err(CodecError::Limit {
            what: "vp9 frame edge",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Decoder;
    use cutlass_encode::VideoEncoder;

    #[test]
    fn normal_inter_matches_ffvp9() {
        let frames = [
            include_bytes!("../fixtures/vp9_normal32_0.bin").as_slice(),
            include_bytes!("../fixtures/vp9_normal32_1.bin").as_slice(),
            include_bytes!("../fixtures/vp9_normal32_2.bin").as_slice(),
        ];
        let yuv = include_bytes!("../fixtures/vp9_normal32.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let picture = decoder
                .decode(frame)
                .unwrap_or_else(|err| panic!("frame {index} {err}"))
                .unwrap_or_else(|| panic!("frame {index} hidden"));
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got.len(), yuv.len());
        let mismatch = got.iter().zip(yuv.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "first mismatch {mismatch:?}");
    }

    #[test]
    fn altref_inter_matches_ffvp9() {
        let frames = [
            include_bytes!("../fixtures/vp9_altref64_0.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_1.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_2.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_3.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_4.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_5.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_6.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_7.bin").as_slice(),
        ];
        let yuv = include_bytes!("../fixtures/vp9_altref64.yuv");
        decode_all(&frames, yuv);
    }

    fn decode_all(frames: &[&[u8]], yuv: &[u8]) {
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let picture = decoder
                .decode(frame)
                .unwrap_or_else(|err| panic!("frame {index} {err}"))
                .unwrap_or_else(|| panic!("frame {index} hidden"));
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got.len(), yuv.len());
        let mismatch = got.iter().zip(yuv.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "first mismatch {mismatch:?}");
    }

    #[test]
    fn error_resilient_inter_matches_ffvp9() {
        let frames = [
            include_bytes!("../fixtures/vp9_inter32_0.bin").as_slice(),
            include_bytes!("../fixtures/vp9_inter32_1.bin").as_slice(),
        ];
        let yuv = include_bytes!("../fixtures/vp9_inter32.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for frame in frames {
            let picture = decoder.decode(frame).expect("frame").expect("shown");
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got.len(), yuv.len());
        let mismatch = got.iter().zip(yuv.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "first mismatch {mismatch:?}");
    }

    #[test]
    fn failed_inter_does_not_drop_references() {
        let key = include_bytes!("../fixtures/vp9_inter32_0.bin");
        let mut decoder = Decoder::new();
        let shown = decoder.decode(key).expect("key").expect("shown key");
        let broken = [0x87, 0x00];
        let err = decoder.decode(&broken);
        assert!(err.is_err());
        let inter = include_bytes!("../fixtures/vp9_inter32_1.bin");
        let picture = decoder
            .decode(inter)
            .expect("refs survived")
            .expect("shown");
        let mut got = shown.y.clone();
        got.extend_from_slice(&shown.u);
        got.extend_from_slice(&shown.v);
        got.extend_from_slice(&picture.y);
        got.extend_from_slice(&picture.u);
        got.extend_from_slice(&picture.v);
        let yuv = include_bytes!("../fixtures/vp9_inter32.yuv");
        assert_eq!(got.as_slice(), yuv.as_slice());
    }

    #[test]
    fn decoder_fuzz_does_not_panic() {
        let secs = std::env::var("VP9_FUZZ_SECS")
            .ok()
            .and_then(|text| text.parse::<u64>().ok())
            .unwrap_or(2);
        let seed = std::env::var("VP9_FUZZ_SEED")
            .ok()
            .and_then(|text| text.parse::<u64>().ok())
            .unwrap_or(0x5650_3900_d4af);
        let corpus = fuzz_corpus();
        let mut rng = FuzzRng { state: seed };
        let started = std::time::Instant::now();
        let mut inputs = 0u64;
        let mut decoded = 0u64;
        let limit = std::time::Duration::from_secs(secs);
        while started.elapsed() < limit {
            let bytes = mutate(&mut rng, &corpus);
            let mut decoder = Decoder::new();
            if let Ok(Some(picture)) = decoder.decode(&bytes) {
                assert!(sane(&picture), "seed {seed} input {inputs}");
                decoded += 1;
            }
            if inputs.is_multiple_of(32) {
                let mut decoder = Decoder::new();
                let key = &corpus[0];
                decoder.decode(key).expect("corpus key").expect("shown");
                let _ = decoder.decode(&bytes);
                let again = decoder.decode(key).expect("key after noise");
                assert!(again.is_some(), "seed {seed} dropped the key");
            }
            inputs += 1;
        }
        eprintln!(
            "vp9 fuzz seed {seed:#x} seconds {secs} inputs {inputs} pictures {decoded} findings 0"
        );
    }

    fn fuzz_corpus() -> Vec<Vec<u8>> {
        let mut encoder = crate::encode::Vp9Encoder::with_quantizer(32, 16, 32).expect("encoder");
        let mut frames = Vec::new();
        for index in 0..4 {
            let frame = fuzz_picture(index);
            let packets = encoder.push(&frame).expect("corpus");
            frames.push(packets[0].data.clone());
        }
        frames.push(vec![0xa0]);
        frames.push(vec![0x00]);
        frames.push(vec![0xc0]);
        frames
    }

    fn fuzz_picture(index: u32) -> cutlass_encode::VideoInput {
        let width = 32u32;
        let height = 16u32;
        let mut y = Vec::new();
        let mut u = Vec::new();
        let mut v = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let value = row
                    .wrapping_mul(3)
                    .wrapping_add(col)
                    .wrapping_add(index * 5);
                y.push(u8::try_from(value % 220).unwrap_or(0));
            }
        }
        for row in 0..height / 2 {
            for col in 0..width / 2 {
                u.push(u8::try_from((row + col + index) % 80 + 40).unwrap_or(40));
                v.push(u8::try_from((col * 2 + index) % 70 + 60).unwrap_or(60));
            }
        }
        let frame = cutlass_encode::VideoFrame::yuv420p(
            width,
            height,
            cutlass_encode::Plane::tight(width, height, y).expect("y"),
            cutlass_encode::Plane::tight(width / 2, height / 2, u).expect("u"),
            cutlass_encode::Plane::tight(width / 2, height / 2, v).expect("v"),
        )
        .expect("frame");
        cutlass_encode::VideoInput {
            frame,
            pts: i64::from(index),
            duration: 1,
            timescale: 30,
        }
    }

    fn mutate(rng: &mut FuzzRng, corpus: &[Vec<u8>]) -> Vec<u8> {
        let pick = rng.below(corpus.len() + 1);
        if pick == corpus.len() {
            let len = rng.below(2048);
            return (0..len).map(|_| rng.byte()).collect();
        }
        let mut bytes = corpus[pick].clone();
        if bytes.is_empty() {
            return bytes;
        }
        match rng.below(4) {
            0 => {
                let at = rng.below(bytes.len());
                bytes[at] ^= rng.byte() | 1;
            }
            1 => {
                let at = rng.below(bytes.len());
                bytes.truncate(at);
            }
            2 => {
                let at = rng.below(bytes.len() + 1);
                bytes.insert(at, rng.byte());
            }
            _ => {
                let extra: Vec<u8> = (0..rng.below(16)).map(|_| rng.byte()).collect();
                bytes.extend(extra);
            }
        }
        bytes
    }

    fn sane(picture: &crate::recon::Picture) -> bool {
        let width = usize::try_from(picture.width).unwrap_or(usize::MAX);
        let height = usize::try_from(picture.height).unwrap_or(usize::MAX);
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        picture.width > 0
            && picture.height > 0
            && picture.width <= crate::geom::MAX_EDGE
            && picture.height <= crate::geom::MAX_EDGE
            && picture.y.len() == width.saturating_mul(height)
            && picture.u.len() == chroma_w.saturating_mul(chroma_h)
            && picture.v.len() == picture.u.len()
    }

    struct FuzzRng {
        state: u64,
    }

    impl FuzzRng {
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.state
        }

        fn below(&mut self, n: usize) -> usize {
            if n == 0 {
                return 0;
            }
            usize::try_from(self.next() % u64::try_from(n).unwrap_or(1)).unwrap_or(0)
        }

        fn byte(&mut self) -> u8 {
            u8::try_from(self.next() & 0xff).unwrap_or(0)
        }
    }

    #[test]
    fn scaled_references_match_libvpx() {
        let frames: [&[u8]; 3] = [
            include_bytes!("../fixtures/vp9_scale_0.bin"),
            include_bytes!("../fixtures/vp9_scale_1.bin"),
            include_bytes!("../fixtures/vp9_scale_2.bin"),
        ];
        let yuv = include_bytes!("../fixtures/vp9_scale.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        let mut sizes = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let picture = decoder
                .decode(frame)
                .unwrap_or_else(|err| panic!("frame {index}: {err}"))
                .expect("shown");
            sizes.push((picture.width, picture.height));
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(sizes, [(64, 64), (32, 32), (52, 52)]);
        assert_eq!(got, yuv);
    }

    #[test]
    fn segmentation_matches_libvpx() {
        let frames: [&[u8]; 8] = [
            include_bytes!("../fixtures/vp9_aq_0.bin"),
            include_bytes!("../fixtures/vp9_aq_1.bin"),
            include_bytes!("../fixtures/vp9_aq_2.bin"),
            include_bytes!("../fixtures/vp9_aq_3.bin"),
            include_bytes!("../fixtures/vp9_aq_4.bin"),
            include_bytes!("../fixtures/vp9_aq_5.bin"),
            include_bytes!("../fixtures/vp9_aq_6.bin"),
            include_bytes!("../fixtures/vp9_aq_7.bin"),
        ];
        let yuv = include_bytes!("../fixtures/vp9_aq.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for frame in frames {
            let picture = decoder.decode(frame).expect("frame").expect("shown");
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got.len(), yuv.len());
        let mismatch = got.iter().zip(yuv.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "first mismatch {mismatch:?}");
    }

    #[test]
    fn reset_drops_references() {
        let key = include_bytes!("../fixtures/vp9_inter32_0.bin");
        let inter = include_bytes!("../fixtures/vp9_inter32_1.bin");
        let mut decoder = Decoder::new();
        decoder.decode(key).expect("key");
        decoder.reset();
        let err = decoder.decode(inter);
        assert!(err.is_err());
    }
}
