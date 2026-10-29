//! Stateful profile 0 decoder.
//!
//! A failed frame leaves references and probabilities as they were. Reference
//! slots update only after reconstruction succeeds. `show_existing_frame`
//! displays a stored picture and does not refresh those slots.

use std::sync::Arc;

use cutlass_decode::CodecError;

use crate::header::{Header, Parsed, Persist, parse};
use crate::recon::{Picture, RefSlots, Reference, reconstruct};

pub struct Decoder {
    persist: Persist,
    refs: [Option<Arc<Reference>>; 8],
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            persist: Persist::new(),
            refs: [const { None }; 8],
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
        let picture = {
            let slots = self.slots();
            reconstruct(&parsed, &slots)?
        };
        self.commit(&parsed, &picture);
        if parsed.header.show_frame {
            Ok(Some(picture))
        } else {
            Ok(None)
        }
    }

    pub fn reset(&mut self) {
        self.persist = Persist::new();
        self.refs = [const { None }; 8];
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

    fn commit(&mut self, parsed: &Parsed<'_>, picture: &Picture) {
        let header = &parsed.header;
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
            self.persist.contexts[index] = parsed.fc.clone();
        }
        self.store_refs(header, picture);
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

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Decoder;

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
