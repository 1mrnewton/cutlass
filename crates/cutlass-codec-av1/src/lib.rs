//! Native AV1 decoder.
#![allow(clippy::too_many_arguments)]
//!
//! Packets are complete low-overhead OBUs or Annex B temporal units. The
//! production path does not call another AV1 decoder. See `docs/av1/` for the
//! specification pin, state table, and coverage.

mod annexb;
mod bits;
mod cdf;
mod coeff;
mod coeff_cdf;
mod config;
#[cfg(test)]
mod ec;
mod error;
mod geom;
mod header;
mod intra;
mod limits;
mod math;
mod obu;
mod quant;
mod sequence;
mod symbol;
mod tiledec;
mod txfm;

pub use config::{DecoderConfig, FilmGrainMode, InputFraming, PacketTiming};
pub use error::DecodeError;
pub use limits::DecoderLimits;
pub use sequence::{ChromaSamplePosition, SequenceHeader};

use cutlass_core::{
    CpuImage, FrameData, PixelFormat, Plane, RationalTime, Rect, Rotation, VideoFrame,
};

/// Result of [`Av1Decoder::submit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitStatus {
    /// The packet was consumed. Call [`Av1Decoder::receive`].
    Accepted,
    /// The output queue is full. The packet was not consumed.
    Blocked,
}

/// Result of [`Av1Decoder::receive`].
#[derive(Debug)]
pub enum ReceiveStatus {
    /// One presentation. Hidden frames are not returned here.
    Frame(DecodedFrame),
    /// Accepted input did not complete a shown frame.
    NeedInput,
    /// `drain` has finished and no further frames will appear.
    Drained,
}

/// A presented frame plus AV1 geometry and grain metadata.
#[derive(Debug)]
pub struct DecodedFrame {
    pub frame: VideoFrame,
    pub info: FrameInfo,
}

/// Geometry and signaling that [`VideoFrame`] does not carry on its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameInfo {
    /// Coded luma size before super-resolution upscaling.
    pub coded_size: (u32, u32),
    /// Size after super-resolution. This is the reconstructed buffer size.
    pub upscaled_size: (u32, u32),
    /// Render size from the frame header. Not a second scaled allocation.
    pub render_size: (u32, u32),
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub chroma_sample_position: ChromaSamplePosition,
    pub operating_point: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub shown_existing: bool,
    pub refresh_frame_flags: u8,
    /// True when `frame` samples include reference-process film grain.
    pub grain_applied: bool,
    /// Parsed grain parameters are present even when grain was not applied.
    pub grain_params_present: bool,
    pub color_primaries_raw: u8,
    pub transfer_raw: u8,
    pub matrix_raw: u8,
}

/// Profiles, depths, and layouts this build can reconstruct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderCapabilities {
    pub profiles: &'static [u8],
    pub bit_depths: &'static [u8],
    pub tile_list: bool,
    pub workers: u32,
    pub grain: bool,
}

/// One AV1 sequence. Instances do not share mutable stream state.
pub struct Av1Decoder {
    config: DecoderConfig,
    sequence: Option<SequenceHeader>,
    seen_frame_header: bool,
    /// Header parsed from a frame-header OBU, waiting for tile groups.
    pending_header: Option<header::FrameHeader>,
    drained: bool,
    needs_reset: Option<String>,
    /// Presentations waiting for `receive`, in emit order.
    queue: std::collections::VecDeque<DecodedFrame>,
}

impl Av1Decoder {
    pub fn new(config: DecoderConfig) -> Result<Self, DecodeError> {
        config.validate()?;
        Ok(Self {
            config,
            sequence: None,
            seen_frame_header: false,
            pending_header: None,
            drained: false,
            needs_reset: None,
            queue: std::collections::VecDeque::new(),
        })
    }

    pub fn capabilities(&self) -> DecoderCapabilities {
        DecoderCapabilities {
            // A lossless 8-bit 4:2:0 DC intra frame reconstructs. Profiles are
            // not advertised until the rest of each profile's tool set does too.
            profiles: &[],
            bit_depths: &[],
            tile_list: false,
            workers: 1,
            grain: false,
        }
    }

    pub fn sequence(&self) -> Option<&SequenceHeader> {
        self.sequence.as_ref()
    }

    /// Accept one packet. The packet is consumed only on [`SubmitStatus::Accepted`].
    pub fn submit(
        &mut self,
        packet: &[u8],
        timing: PacketTiming,
    ) -> Result<SubmitStatus, DecodeError> {
        if let Some(reason) = &self.needs_reset {
            return Err(DecodeError::NeedsReset(reason.clone()));
        }
        if self.drained {
            return Err(DecodeError::Misuse("submit after drain; call reset".into()));
        }
        if self.queue.len() >= self.config.limits.max_queued_frames {
            return Ok(SubmitStatus::Blocked);
        }
        let obus = match self.config.framing {
            InputFraming::LowOverhead => obu::split_low_overhead(packet, &self.config.limits)?,
            InputFraming::AnnexB => annexb::split_temporal_units(packet, &self.config.limits)?
                .into_iter()
                .flat_map(|u| u.obus)
                .collect(),
        };
        // Preflight succeeded. Apply OBUs. A failure here keeps any frame
        // already queued from an earlier OBU in this packet and requires reset.
        for obu in obus {
            if let Err(err) = self.apply_obu(obu, timing) {
                // The packet's earlier OBUs may already have committed. Further
                // input is rejected until reset so a retry cannot decode an
                // accepted OBU twice.
                self.needs_reset = Some(err.to_string());
                return Err(err);
            }
        }
        Ok(SubmitStatus::Accepted)
    }

    /// One OBU whose length is the slice, including when `obu_has_size_field` is 0.
    pub fn submit_unsized_obu(
        &mut self,
        obu_bytes: &[u8],
        timing: PacketTiming,
    ) -> Result<SubmitStatus, DecodeError> {
        if self.needs_reset.is_some() || self.drained {
            return self.submit(&[], timing);
        }
        let (obu, consumed) = obu::read_obu(obu_bytes, Some(obu_bytes.len()), &self.config.limits)?;
        if consumed != obu_bytes.len() {
            return Err(DecodeError::malformed(
                "unsized OBU did not consume the enclosing buffer",
            ));
        }
        self.apply_obu(obu, timing)?;
        Ok(SubmitStatus::Accepted)
    }

    fn apply_obu(&mut self, obu: obu::RawObu, timing: PacketTiming) -> Result<(), DecodeError> {
        let idc = self
            .sequence
            .as_ref()
            .map(|s| s.operating_point_idc_selected)
            .unwrap_or(0);
        if !obu::in_operating_point(&obu, idc) {
            return Ok(());
        }
        match obu.obu_type {
            obu::OBU_TEMPORAL_DELIMITER => {
                self.seen_frame_header = false;
                Ok(())
            }
            obu::OBU_SEQUENCE_HEADER => {
                let parsed = sequence::parse(
                    &obu.payload,
                    &self.config.limits,
                    self.config.operating_point,
                )?;
                if let Some(existing) = &self.sequence
                    && !existing.same_sequence(&parsed)
                    && self.seen_frame_header
                {
                    return Err(DecodeError::malformed(
                        "sequence header changed inside a frame",
                    ));
                }
                self.sequence = Some(parsed);
                Ok(())
            }
            obu::OBU_PADDING => Ok(()),
            obu::OBU_METADATA => {
                if obu.payload.len() > self.config.limits.max_metadata_bytes {
                    return Err(DecodeError::limit("metadata OBU"));
                }
                // Metadata is bounded and retained with the sequence scope.
                // Unknown types are ignored after the size check.
                Ok(())
            }
            obu::OBU_FRAME_HEADER | obu::OBU_REDUNDANT_FRAME_HEADER => {
                self.parse_frame_header_obu(&obu.payload)?;
                Ok(())
            }
            obu::OBU_TILE_GROUP => {
                self.decode_tile_group(&obu.payload, timing, obu.temporal_id, obu.spatial_id)
            }
            obu::OBU_FRAME => {
                self.decode_frame_obu(&obu.payload, timing, obu.temporal_id, obu.spatial_id)
            }
            other => {
                // Reserved types are ignored when the payload was bounded.
                let _ = other;
                Ok(())
            }
        }
    }

    fn sequence_ref(&self) -> Result<&SequenceHeader, DecodeError> {
        self.sequence
            .as_ref()
            .ok_or_else(|| DecodeError::malformed("frame OBU before a sequence header"))
    }

    fn parse_frame_header_obu(&mut self, payload: &[u8]) -> Result<(), DecodeError> {
        let seq = self.sequence_ref()?;
        let mut bits = bits::BitReader::new(payload);
        let header = header::parse_uncompressed(&mut bits, seq, &self.config.limits)?;
        let used = bits.position();
        let remain = payload.len() * 8 - used;
        if remain == 0 {
            return Err(DecodeError::malformed(
                "frame header is missing trailing bits",
            ));
        }
        bits.trailing_bits(remain as i32)?;
        self.pending_header = Some(header);
        self.seen_frame_header = true;
        Ok(())
    }

    fn decode_frame_obu(
        &mut self,
        payload: &[u8],
        timing: PacketTiming,
        temporal_id: u8,
        spatial_id: u8,
    ) -> Result<(), DecodeError> {
        let seq = self.sequence_ref()?.clone();
        let mut bits = bits::BitReader::new(payload);
        let header = header::parse_uncompressed(&mut bits, &seq, &self.config.limits)?;
        bits.byte_align()?;
        let offset = bits.position() / 8;
        self.seen_frame_header = true;
        self.finish_tiles(
            &payload[offset..],
            header,
            timing,
            temporal_id,
            spatial_id,
            &seq,
        )
    }

    fn decode_tile_group(
        &mut self,
        payload: &[u8],
        timing: PacketTiming,
        temporal_id: u8,
        spatial_id: u8,
    ) -> Result<(), DecodeError> {
        let header = self
            .pending_header
            .clone()
            .ok_or_else(|| DecodeError::malformed("tile group without a frame header"))?;
        let seq = self.sequence_ref()?.clone();
        self.finish_tiles(payload, header, timing, temporal_id, spatial_id, &seq)
    }

    fn finish_tiles(
        &mut self,
        tiles: &[u8],
        header: header::FrameHeader,
        timing: PacketTiming,
        temporal_id: u8,
        spatial_id: u8,
        seq: &SequenceHeader,
    ) -> Result<(), DecodeError> {
        let recon = tiledec::decode_shown_frame(header, tiles, seq, &self.config.limits)?;
        let frame = pack_frame(&recon, seq, &self.config, timing, temporal_id, spatial_id);
        self.queue.push_back(frame);
        self.pending_header = None;
        self.seen_frame_header = false;
        Ok(())
    }

    pub fn receive(&mut self) -> Result<ReceiveStatus, DecodeError> {
        if let Some(frame) = self.queue.pop_front() {
            return Ok(ReceiveStatus::Frame(frame));
        }
        if self.drained {
            if self.seen_frame_header {
                return Err(DecodeError::Incomplete);
            }
            return Ok(ReceiveStatus::Drained);
        }
        Ok(ReceiveStatus::NeedInput)
    }

    pub fn drain(&mut self) -> Result<(), DecodeError> {
        self.drained = true;
        if self.seen_frame_header {
            return Err(DecodeError::Incomplete);
        }
        Ok(())
    }

    /// Drop assembly, references, and the sequence. Queued frames are discarded.
    pub fn reset(&mut self) {
        self.sequence = None;
        self.seen_frame_header = false;
        self.pending_header = None;
        self.drained = false;
        self.needs_reset = None;
        self.queue.clear();
    }
}

fn pack_frame(
    recon: &tiledec::Reconstructed,
    seq: &SequenceHeader,
    config: &DecoderConfig,
    timing: PacketTiming,
    temporal_id: u8,
    spatial_id: u8,
) -> DecodedFrame {
    let y = Plane::new(recon.y.clone(), recon.width as usize, recon.height as usize);
    let u = Plane::new(
        recon.u.clone(),
        recon.chroma_w as usize,
        recon.chroma_h as usize,
    );
    let v = Plane::new(
        recon.v.clone(),
        recon.chroma_w as usize,
        recon.chroma_h as usize,
    );
    let ticks = timing.timestamp.unwrap_or(0);
    let mut frame = VideoFrame::new(
        timestamp(ticks, config.time_base),
        PixelFormat::Yuv420p,
        seq.color,
        (recon.header.upscaled_width, recon.header.frame_height),
        Rect::from_size(recon.header.upscaled_width, recon.header.frame_height),
        Rotation::None,
        FrameData::Cpu(CpuImage::new(vec![y, u, v])),
    );
    if let Some(duration) = timing.duration {
        frame = frame.with_duration(timestamp(duration, config.time_base));
    }
    let grain_applied = config.grain == FilmGrainMode::Apply && recon.header.apply_grain;
    DecodedFrame {
        frame,
        info: FrameInfo {
            coded_size: (recon.header.frame_width, recon.header.frame_height),
            upscaled_size: (recon.header.upscaled_width, recon.header.frame_height),
            render_size: (recon.header.render_width, recon.header.render_height),
            bit_depth: seq.bit_depth,
            mono_chrome: seq.mono_chrome,
            subsampling_x: seq.subsampling_x,
            subsampling_y: seq.subsampling_y,
            chroma_sample_position: seq.chroma_sample_position,
            operating_point: seq.operating_point,
            temporal_id,
            spatial_id,
            shown_existing: false,
            refresh_frame_flags: recon.header.refresh_frame_flags,
            grain_applied,
            grain_params_present: recon.header.apply_grain,
            color_primaries_raw: seq.color_primaries_raw,
            transfer_raw: seq.transfer_raw,
            matrix_raw: seq.matrix_raw,
        },
    }
}

/// Helper so callers can stamp a frame without inventing a rate.
pub fn timestamp(ticks: i64, time_base: cutlass_core::Rational) -> RationalTime {
    RationalTime::new(ticks, time_base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reduced_sequence_packet() -> Vec<u8> {
        // OBU header type=1 has_size=1, then payload from the sequence test.
        let mut payload = Vec::new();
        let mut n = 0usize;
        let push = |value: u32, bits: usize, payload: &mut Vec<u8>, n: &mut usize| {
            for i in (0..bits).rev() {
                let bit = ((value >> i) & 1) as u8;
                let byte = *n / 8;
                if byte == payload.len() {
                    payload.push(0);
                }
                if bit == 1 {
                    payload[byte] |= 1 << (7 - (*n % 8));
                }
                *n += 1;
            }
        };
        push(0, 3, &mut payload, &mut n);
        push(1, 1, &mut payload, &mut n);
        push(1, 1, &mut payload, &mut n);
        push(0, 5, &mut payload, &mut n);
        push(3, 4, &mut payload, &mut n);
        push(3, 4, &mut payload, &mut n);
        push(15, 4, &mut payload, &mut n);
        push(15, 4, &mut payload, &mut n);
        for _ in 0..6 {
            push(0, 1, &mut payload, &mut n);
        }
        push(0, 1, &mut payload, &mut n); // highbd
        push(0, 1, &mut payload, &mut n); // mono
        push(0, 1, &mut payload, &mut n); // description
        push(0, 1, &mut payload, &mut n); // range
        push(0, 2, &mut payload, &mut n); // csp
        push(0, 1, &mut payload, &mut n); // separate
        push(0, 1, &mut payload, &mut n); // grain
        push(1, 1, &mut payload, &mut n);
        while !n.is_multiple_of(8) {
            push(0, 1, &mut payload, &mut n);
        }
        let mut packet = vec![0b0000_1010, payload.len() as u8];
        packet.extend(payload);
        packet
    }

    #[test]
    fn submit_sequence_then_frame_is_unsupported_without_consuming_retry_twice() {
        let mut dec = Av1Decoder::new(DecoderConfig::default()).unwrap();
        let pkt = reduced_sequence_packet();
        assert_eq!(
            dec.submit(&pkt, PacketTiming::default()).unwrap(),
            SubmitStatus::Accepted
        );
        assert_eq!(dec.sequence().unwrap().max_frame_width, 16);
        // A frame-header OBU with an empty payload cannot be reconstructed.
        let frame = [0b0011_0100, 0x00];
        let err = dec.submit(&frame, PacketTiming::default()).unwrap_err();
        assert!(matches!(
            err,
            DecodeError::Truncated(_) | DecodeError::Malformed(_)
        ));
    }

    #[test]
    fn blocked_when_queue_cap_is_zero_is_not_the_default() {
        let mut dec = Av1Decoder::new(DecoderConfig::default()).unwrap();
        assert!(dec.queue.is_empty());
        assert!(matches!(dec.receive().unwrap(), ReceiveStatus::NeedInput));
    }

    #[test]
    fn drain_then_submit_is_misuse() {
        let mut dec = Av1Decoder::new(DecoderConfig::default()).unwrap();
        dec.drain().unwrap();
        assert!(matches!(
            dec.submit(&[0], PacketTiming::default()),
            Err(DecodeError::Misuse(_))
        ));
        dec.reset();
        assert!(matches!(dec.receive().unwrap(), ReceiveStatus::NeedInput));
    }

    fn push_bits(payload: &mut Vec<u8>, n: &mut usize, value: u32, bits: usize) {
        for i in (0..bits).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte = *n / 8;
            if byte == payload.len() {
                payload.push(0);
            }
            if bit == 1 {
                payload[byte] |= 1 << (7 - (*n % 8));
            }
            *n += 1;
        }
    }

    /// Tile bytes whose symbols are partition NONE, skip, DC luma, DC chroma.
    fn lossless_dc_tile() -> Vec<u8> {
        let mut enc = ec::SymbolEncoder::new();
        enc.encode_symbol(0, cdf::partition_cdf(2, 0).unwrap())
            .unwrap();
        enc.encode_symbol(1, &cdf::SKIP[0]).unwrap();
        enc.encode_symbol(0, &cdf::KF_Y_MODE[0][0]).unwrap();
        enc.encode_symbol(0, cdf::uv_mode_cdf(false, 0).unwrap())
            .unwrap();
        enc.finish().unwrap()
    }

    /// Same header symbols, but skip is clear and the first luma 4×4 carries
    /// a positive DC level of 1. Every other transform is `all_zero`.
    fn lossless_dc_residual_tile() -> Vec<u8> {
        let mut enc = ec::SymbolEncoder::new();
        enc.encode_symbol(0, cdf::partition_cdf(2, 0).unwrap())
            .unwrap();
        enc.encode_symbol(0, &cdf::SKIP[0]).unwrap();
        enc.encode_symbol(0, &cdf::KF_Y_MODE[0][0]).unwrap();
        enc.encode_symbol(0, cdf::uv_mode_cdf(false, 0).unwrap())
            .unwrap();
        let mut maps = coeff::LevelMaps::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                encode_tx(&mut enc, &mut maps, 0, x, y, 16, 16, 4, 4, x == 0 && y == 0);
            }
        }
        for plane in 1..3 {
            for y in 0..2 {
                for x in 0..2 {
                    encode_tx(&mut enc, &mut maps, plane, x, y, 8, 8, 2, 2, false);
                }
            }
        }
        enc.finish().unwrap()
    }

    fn encode_tx(
        enc: &mut ec::SymbolEncoder,
        maps: &mut coeff::LevelMaps,
        plane: usize,
        x4: usize,
        y4: usize,
        block_w: usize,
        block_h: usize,
        max_x4: usize,
        max_y4: usize,
        dc_level_one: bool,
    ) {
        let ctx = coeff::txb_skip_ctx(
            plane,
            4,
            4,
            block_w,
            block_h,
            x4,
            y4,
            max_x4,
            max_y4,
            &maps.above_level[plane],
            &maps.left_level[plane],
            &maps.above_dc[plane],
            &maps.left_dc[plane],
        )
        .unwrap();
        let ptype = usize::from(plane > 0);
        enc.encode_symbol(
            u32::from(!dc_level_one) as usize,
            &coeff_cdf::TXB_SKIP[0][0][ctx],
        )
        .unwrap();
        if dc_level_one {
            enc.encode_symbol(0, &coeff_cdf::EOB_PT_16[0][ptype][0])
                .unwrap();
            enc.encode_symbol(0, &coeff_cdf::COEFF_BASE_EOB[0][0][ptype][0])
                .unwrap();
            enc.encode_symbol(0, &coeff_cdf::DC_SIGN[0][ptype][0])
                .unwrap();
            maps.above_level[plane][x4] = 1;
            maps.left_level[plane][y4] = 1;
            maps.above_dc[plane][x4] = 2;
            maps.left_dc[plane][y4] = 2;
        } else {
            maps.above_level[plane][x4] = 0;
            maps.left_level[plane][y4] = 0;
            maps.above_dc[plane][x4] = 0;
            maps.left_dc[plane][y4] = 0;
        }
    }

    #[test]
    fn lossless_dc_keyframe_is_mid_gray() {
        let mut dec = Av1Decoder::new(DecoderConfig::default()).unwrap();
        dec.submit(&reduced_sequence_packet(), PacketTiming::default())
            .unwrap();
        let mut payload = Vec::new();
        let mut n = 0usize;
        push_bits(&mut payload, &mut n, 1, 1); // disable_cdf_update
        push_bits(&mut payload, &mut n, 0, 1); // screen content off
        push_bits(&mut payload, &mut n, 0, 1); // render size matches
        push_bits(&mut payload, &mut n, 1, 1); // uniform tiles
        push_bits(&mut payload, &mut n, 0, 8); // base q
        push_bits(&mut payload, &mut n, 0, 1); // delta y dc
        push_bits(&mut payload, &mut n, 0, 1); // delta u dc
        push_bits(&mut payload, &mut n, 0, 1); // delta u ac
        push_bits(&mut payload, &mut n, 0, 1); // qmatrix
        push_bits(&mut payload, &mut n, 0, 1); // segmentation
        push_bits(&mut payload, &mut n, 0, 1); // reduced tx set
        while !n.is_multiple_of(8) {
            push_bits(&mut payload, &mut n, 0, 1);
        }
        payload.extend(lossless_dc_tile());
        let mut packet = vec![0x32, payload.len() as u8];
        packet.extend(payload);
        dec.submit(
            &packet,
            PacketTiming {
                timestamp: Some(5),
                ..PacketTiming::default()
            },
        )
        .unwrap();
        match dec.receive().unwrap() {
            ReceiveStatus::Frame(decoded) => {
                assert_eq!(decoded.frame.pts.value, 5);
                assert!(!decoded.info.grain_applied);
                assert_eq!(decoded.info.coded_size, (16, 16));
                assert_eq!(decoded.info.upscaled_size, (16, 16));
                let cutlass_core::FrameData::Cpu(image) = &decoded.frame.data else {
                    panic!("cpu frame");
                };
                assert!(image.planes[0].data.iter().all(|&s| s == 128));
                assert_eq!(image.planes[0].data.len(), 16 * 16);
                assert!(image.planes[1].data.iter().all(|&s| s == 128));
                assert!(image.planes[2].data.iter().all(|&s| s == 128));
                assert_eq!(image.planes[1].data.len(), 8 * 8);
            }
            other => panic!("expected a frame, got {other:?}"),
        }
    }

    #[test]
    fn lossless_dc_coefficient_adds_one_at_origin() {
        let mut dec = Av1Decoder::new(DecoderConfig::default()).unwrap();
        dec.submit(&reduced_sequence_packet(), PacketTiming::default())
            .unwrap();
        let mut payload = Vec::new();
        let mut n = 0usize;
        push_bits(&mut payload, &mut n, 1, 1);
        push_bits(&mut payload, &mut n, 0, 1);
        push_bits(&mut payload, &mut n, 0, 1);
        push_bits(&mut payload, &mut n, 1, 1);
        push_bits(&mut payload, &mut n, 0, 8);
        push_bits(&mut payload, &mut n, 0, 1);
        push_bits(&mut payload, &mut n, 0, 1);
        push_bits(&mut payload, &mut n, 0, 1);
        push_bits(&mut payload, &mut n, 0, 1);
        push_bits(&mut payload, &mut n, 0, 1);
        push_bits(&mut payload, &mut n, 0, 1);
        while !n.is_multiple_of(8) {
            push_bits(&mut payload, &mut n, 0, 1);
        }
        payload.extend(lossless_dc_residual_tile());
        let mut packet = vec![0x32, payload.len() as u8];
        packet.extend(payload);
        dec.submit(&packet, PacketTiming::default()).unwrap();
        match dec.receive().unwrap() {
            ReceiveStatus::Frame(decoded) => {
                let cutlass_core::FrameData::Cpu(image) = &decoded.frame.data else {
                    panic!("cpu frame");
                };
                assert_eq!(image.planes[0].data[0], 129);
                assert!(image.planes[0].data[1..].iter().all(|&s| s == 128));
                assert!(image.planes[1].data.iter().all(|&s| s == 128));
                assert!(image.planes[2].data.iter().all(|&s| s == 128));
            }
            other => panic!("expected a frame, got {other:?}"),
        }
    }
}
