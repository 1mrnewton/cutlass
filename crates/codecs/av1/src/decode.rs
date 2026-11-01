//! Public AV1 decoder. One access unit is one temporal unit.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, Decoder, Output, Packet, PictureOutput, PixelFormat,
};

use crate::yuv::{Plane, ShownTags, Yuv420, gray_output, gray_wide, picture_output, wide_output};

use crate::header::{
    FrameInfo, Inherited, SavedLoopFilter, Segmentation, Sequence, TilePart, annex_b_obus,
    header_is_addressable, obu_payload, parse_with_motion, sequence_obu, tile_list, tile_span,
    unit_obus,
};
use crate::inter::{RefView, SavedMotion};
use crate::tile::{Cdfs, DecodedPicture, Reconstructed, decode_tile_parts, decode_with_refs};

use std::sync::Arc;

struct WidePlanes {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
    width: usize,
    height: usize,
}

struct Reference {
    picture: Yuv420,
    /// Native samples when the sequence is 10-bit. Eight-bit slots leave this empty.
    wide: Option<Arc<WidePlanes>>,
    render_width: u32,
    render_height: u32,
    cdfs: Arc<Cdfs>,
    gm: [[i32; 6]; 8],
    field: Arc<SavedMotion>,
    lf: SavedLoopFilter,
    segmentation: Segmentation,
    segments: Arc<Vec<u8>>,
    grain: crate::grain::Grain,
    mi_cols: u32,
    mi_rows: u32,
}

struct Held {
    picture: Yuv420,
    wide: Option<Arc<WidePlanes>>,
    cdfs: Arc<Cdfs>,
    motion: Arc<SavedMotion>,
    segmentation: Segmentation,
    segments: Arc<Vec<u8>>,
    mi_cols: u32,
    mi_rows: u32,
}

/// A frame header waiting for its tile groups. Nothing is stored until the groups cover every tile.
struct PendingTiles {
    sequence: Sequence,
    frame: FrameInfo,
    next: usize,
    count: usize,
    parts: Vec<TilePart>,
}

struct KeptSlot {
    slot: usize,
    reference: Option<Reference>,
    hint: u32,
}

struct FrameUndo {
    stored: Vec<KeptSlot>,
    stale: [Option<Reference>; 8],
}

/// Rust AV1 decoder. Reference slots stay inside this type.
pub struct Av1Decoder {
    sequence: Option<Sequence>,
    refs: [Option<Reference>; 8],
    order_hints: [u32; 8],
    /// `choose_operating_point`. Point 0 is the first point in the sequence.
    operating_point: u8,
}

impl Av1Decoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Av1 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if params.pixel != PixelFormat::None
            && params.pixel != PixelFormat::Yuv420p
            && params.pixel != PixelFormat::Yuv420p10
            && params.pixel != PixelFormat::Yuv420p12
            && params.pixel != PixelFormat::Yuv422p
            && params.pixel != PixelFormat::Yuv422p10
            && params.pixel != PixelFormat::Yuv422p12
            && params.pixel != PixelFormat::Yuv444p
            && params.pixel != PixelFormat::Yuv444p10
            && params.pixel != PixelFormat::Yuv444p12
            && params.pixel != PixelFormat::Gray8
            && params.pixel != PixelFormat::Gray10
        {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        if !crate::bits::signed_and_little_endian_hold()
            || !crate::symbol::literal_holds()
            || !crate::tables::default_tables_match_their_shapes()
            || !crate::dims::rectangular_transforms_are_numbered()
        {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "default tables",
            });
        }
        Ok(Self {
            sequence: None,
            refs: std::array::from_fn(|_| None),
            order_hints: [0; 8],
            operating_point: 0,
        })
    }

    /// Select the operating point from section 6.4.1. The default is point 0.
    ///
    /// An index past the sequence's last point is refused once that sequence is known.
    pub fn select_operating_point(&mut self, index: u8) -> Result<(), CodecError> {
        if index > 31 {
            return Err(CodecError::Unsupported {
                what: "operating point",
            });
        }
        if let Some(sequence) = &self.sequence
            && usize::from(index) > usize::from(sequence.operating_points)
        {
            return Err(CodecError::Unsupported {
                what: "operating point",
            });
        }
        self.operating_point = index;
        Ok(())
    }
}

impl Decoder for Av1Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Av1
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let sequence_before = self.sequence.clone();
        let mut undo = Vec::new();
        match self.decode_unit(packet, &mut undo) {
            Ok(shown) => Ok(shown),
            Err(error) => {
                self.undo_frames(undo);
                self.sequence = sequence_before;
                Err(error)
            }
        }
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.refs = std::array::from_fn(|_| None);
        self.order_hints = [0; 8];
    }
}

impl Av1Decoder {
    /// Every frame OBU in the temporal unit, hidden frames first.
    fn decode_unit(
        &mut self,
        packet: &Packet,
        undo: &mut Vec<FrameUndo>,
    ) -> Result<Vec<Output>, CodecError> {
        let annex = annex_b_obus(&packet.data)?;
        let low;
        let obus = match &annex {
            Some(obus) => obus.as_slice(),
            None => {
                low = unit_obus(&packet.data)?;
                low.as_slice()
            }
        };
        let mut prefix = Vec::new();
        let mut shown = Vec::new();
        let mut saw_frame = false;
        let mut dropped = false;
        let mut pending: Option<PendingTiles> = None;
        for obu in obus {
            if obu.kind != 1 && obu.kind != 2 && !self.keeps_layer(obu.bytes)? {
                dropped = true;
                continue;
            }
            match obu.kind {
                // Padding, metadata, and a reserved type carry no frame.
                // A reserved payload is not inspected.
                0 | 5 | 9..=14 | 15 => {}
                2 => {
                    if pending.is_some() {
                        return Err(unfinished_tiles());
                    }
                }
                1 => {
                    if pending.is_some() {
                        return Err(unfinished_tiles());
                    }
                    let sequence = sequence_obu(obu.bytes)?;
                    if usize::from(self.operating_point) > usize::from(sequence.operating_points) {
                        return Err(CodecError::Unsupported {
                            what: "operating point",
                        });
                    }
                    self.sequence = Some(sequence);
                    prefix.extend_from_slice(obu.bytes);
                }
                6 => {
                    if pending.is_some() {
                        return Err(unfinished_tiles());
                    }
                    saw_frame = true;
                    let mut bytes = std::mem::take(&mut prefix);
                    bytes.extend_from_slice(obu.bytes);
                    record(&mut shown, undo, self.decode_frame(&bytes, packet)?);
                }
                3 => {
                    if pending.is_some() {
                        return Err(unfinished_tiles());
                    }
                    saw_frame = true;
                    let mut bytes = std::mem::take(&mut prefix);
                    bytes.extend_from_slice(obu.bytes);
                    let (sequence, frame) = self.read_header(&bytes, None)?;
                    if frame.show_existing {
                        record(
                            &mut shown,
                            undo,
                            self.commit_frame(packet, sequence, frame, None)?,
                        );
                    } else {
                        let count =
                            (frame.tile_cols as usize).saturating_mul(frame.tile_rows as usize);
                        if count == 0 {
                            return Err(CodecError::Malformed {
                                at_bit: 0,
                                what: "tile grid",
                            });
                        }
                        pending = Some(PendingTiles {
                            sequence,
                            frame,
                            next: 0,
                            count,
                            parts: Vec::new(),
                        });
                    }
                }
                7 => {
                    let Some(open) = pending.as_ref() else {
                        return Err(CodecError::Malformed {
                            at_bit: 0,
                            what: "redundant frame header",
                        });
                    };
                    // A redundant header repeats the open frame. It adds no tiles.
                    self.read_header(obu.bytes, Some(&open.sequence))?;
                }
                4 => {
                    let payload = obu_payload(obu.bytes)?;
                    let done = {
                        let Some(open) = pending.as_mut() else {
                            return Err(CodecError::Unsupported {
                                what: "tile group outside a frame",
                            });
                        };
                        let span = tile_span(&open.frame, payload)?;
                        let covered = span.end.saturating_sub(span.start).saturating_add(1);
                        if span.start != open.next
                            || span.end >= open.count
                            || span.parts.len() != covered
                        {
                            return Err(unfinished_tiles());
                        }
                        open.next = span.end + 1;
                        open.parts.extend(span.parts);
                        open.next == open.count
                    };
                    if done {
                        let Some(open) = pending.take() else {
                            return Err(unfinished_tiles());
                        };
                        record(
                            &mut shown,
                            undo,
                            self.commit_frame(
                                packet,
                                open.sequence,
                                open.frame,
                                Some(&open.parts),
                            )?,
                        );
                    }
                }
                8 => {
                    let Some(open) = pending.take() else {
                        return Err(tile_list_error());
                    };
                    if open.next != 0 {
                        return Err(tile_list_error());
                    }
                    let payload = obu_payload(obu.bytes)?;
                    record(
                        &mut shown,
                        undo,
                        self.show_tile_list(packet, open.sequence, open.frame, payload)?,
                    );
                }
                _ => {
                    return Err(CodecError::Unsupported {
                        what: "reserved obu",
                    });
                }
            }
        }
        if pending.is_some() {
            return Err(unfinished_tiles());
        }
        if !saw_frame {
            if dropped {
                return Ok(shown);
            }
            return Err(CodecError::InvalidApi {
                what: "missing frame",
            });
        }
        Ok(shown)
    }

    /// Section 5.3.1. A sequence header and a temporal delimiter stay.
    /// Any other OBU is dropped when its layer is outside the selected point.
    fn keeps_layer(&self, obu: &[u8]) -> Result<bool, CodecError> {
        let Some(sequence) = &self.sequence else {
            return Ok(true);
        };
        let idc = sequence
            .operating_idc
            .get(usize::from(self.operating_point))
            .copied()
            .ok_or(CodecError::Unsupported {
                what: "operating point",
            })?;
        if idc == 0 {
            return Ok(true);
        }
        let (extension, temporal, spatial) = Self::obu_layer(obu)?;
        if !extension {
            return Ok(true);
        }
        let in_temporal = (idc >> u16::from(temporal)) & 1 == 1;
        let in_spatial = (idc >> (u16::from(spatial) + 8)) & 1 == 1;
        Ok(in_temporal && in_spatial)
    }

    /// `temporal_id` and `spatial_id` from an OBU extension header.
    fn obu_layer(bytes: &[u8]) -> Result<(bool, u8, u8), CodecError> {
        let header = bytes.first().copied().ok_or(CodecError::Truncated {
            at_bit: 0,
            what: "obu",
        })?;
        if header & 0x04 == 0 {
            return Ok((false, 0, 0));
        }
        let extension = bytes.get(1).copied().ok_or(CodecError::Truncated {
            at_bit: 8,
            what: "obu",
        })?;
        Ok((true, extension >> 5, (extension >> 3) & 0x3))
    }

    /// Decode one frame OBU. A hidden frame updates reference slots and shows nothing.
    fn decode_frame(
        &mut self,
        bytes: &[u8],
        packet: &Packet,
    ) -> Result<(Option<PictureOutput>, FrameUndo), CodecError> {
        let (sequence, frame) = self.read_header(bytes, None)?;
        self.commit_frame(packet, sequence, frame, None)
    }

    fn read_header(
        &self,
        bytes: &[u8],
        stored: Option<&Sequence>,
    ) -> Result<(Sequence, FrameInfo), CodecError> {
        let saved_motion =
            std::array::from_fn(|slot| self.refs[slot].as_ref().map(|reference| reference.gm));
        let ref_sizes = std::array::from_fn(|slot| {
            self.refs[slot].as_ref().map(|reference| {
                let (width, height) = reference.coded_size();
                (
                    width,
                    height,
                    reference.render_width,
                    reference.render_height,
                )
            })
        });
        let inherited = std::array::from_fn(|slot| {
            self.refs[slot]
                .as_ref()
                .map(|reference| Inherited {
                    lf: reference.lf,
                    seg: reference.segmentation,
                })
                .unwrap_or_else(Inherited::defaults)
        });
        let (sequence, mut frame) = parse_with_motion(
            bytes,
            stored.or(self.sequence.as_ref()),
            &saved_motion,
            &self.order_hints,
            &ref_sizes,
            &inherited,
        )?;
        self.attach_prev_segments(&mut frame);
        self.resolve_grain(&mut frame)?;
        header_is_addressable(&sequence, &frame)?;
        Ok((sequence, frame))
    }

    fn commit_frame(
        &mut self,
        packet: &Packet,
        sequence: Sequence,
        mut frame: FrameInfo,
        parts: Option<&[TilePart]>,
    ) -> Result<(Option<PictureOutput>, FrameUndo), CodecError> {
        let stale = self.take_stale_refs(&frame);
        let held = match self.picture_before_grain(&sequence, &frame, parts) {
            Ok(decoded) => decoded,
            Err(error) => {
                self.restore_refs(stale);
                return Err(error);
            }
        };
        if let Some(reference) = self.refs[usize::from(frame.frame_to_show)].as_ref()
            && frame.show_existing
        {
            frame.render_width = reference.render_width;
            frame.render_height = reference.render_height;
        }
        let stored = self.store(&frame, &held);
        let tags = shown_tags(&sequence, &frame, self.operating_point);
        let monochrome = sequence.monochrome;
        self.sequence = Some(sequence);
        if !frame.show_frame {
            return Ok((None, FrameUndo { stored, stale }));
        }
        let picture = match publish_held(
            &held,
            &frame,
            monochrome,
            &tags,
            packet.pts,
            packet.timescale,
        ) {
            Ok(picture) => picture,
            Err(error) => {
                self.undo_store(stored);
                self.restore_refs(stale);
                return Err(error);
            }
        };
        Ok((Some(picture), FrameUndo { stored, stale }))
    }

    fn decode_picture(
        &self,
        sequence: &Sequence,
        frame: &FrameInfo,
        parts: Option<&[TilePart]>,
    ) -> Result<Reconstructed, CodecError> {
        let carried = self.carried_cdf(frame)?;
        let views = self.ref_views();
        match parts {
            Some(parts) => {
                decode_tile_parts(sequence, frame, &views, carried.as_deref(), parts, true)
            }
            None => decode_with_refs(sequence, frame, &views, carried.as_deref()),
        }
    }

    /// `load_cdfs(ref_frame_idx[primary_ref_frame])` when the frame names a primary reference.
    /// An intra frame initializes its CDFs instead.
    fn carried_cdf(&self, frame: &FrameInfo) -> Result<Option<Arc<Cdfs>>, CodecError> {
        if frame.intra || frame.primary_ref >= 7 {
            return Ok(None);
        }
        let index = usize::from(frame.primary_ref);
        let slot = usize::from(frame.ref_frames[index]);
        let Some(reference) = &self.refs[slot] else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        };
        Ok(Some(Arc::clone(&reference.cdfs)))
    }

    fn ref_views(&self) -> [Option<RefView<'_>>; 8] {
        std::array::from_fn(|slot| {
            self.refs[slot].as_ref().map(|reference| RefView {
                y: reference.picture.y.samples(),
                u: reference.picture.u.samples(),
                v: reference.picture.v.samples(),
                width: reference.picture.y.width() as usize,
                height: reference.picture.y.height() as usize,
                frame_type: reference.field.frame_type,
                mi_cols: reference.field.mi_cols,
                mi_rows: reference.field.mi_rows,
                saved_hints: reference.field.saved_hints,
                mf_refs: &reference.field.refs,
                mf_mvs: &reference.field.mvs,
            })
        })
    }

    /// `load_grain_params`. Every parameter comes from the named slot except `grain_seed`.
    fn resolve_grain(&self, frame: &mut FrameInfo) -> Result<(), CodecError> {
        let Some(slot) = frame.grain.copy_from else {
            return Ok(());
        };
        let slot = usize::from(slot);
        let named = u8::try_from(slot).unwrap_or(255);
        if slot >= 8 || !frame.ref_frames.contains(&named) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "film grain",
            });
        }
        let Some(reference) = self.refs[slot].as_ref() else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        };
        let seed = frame.grain.seed;
        frame.grain = reference.grain;
        frame.grain.seed = seed;
        frame.grain.copy_from = None;
        Ok(())
    }

    fn attach_prev_segments(&self, frame: &mut FrameInfo) {
        if !frame.segmentation.enabled || frame.primary_ref >= 7 {
            return;
        }
        let Some(slot) = frame
            .ref_frames
            .get(usize::from(frame.primary_ref))
            .copied()
        else {
            return;
        };
        let Some(reference) = self.refs[usize::from(slot)].as_ref() else {
            return;
        };
        if reference.mi_cols == frame.mi_cols && reference.mi_rows == frame.mi_rows {
            frame.prev_segment_ids = (*reference.segments).clone();
        }
    }

    fn store(&mut self, frame: &FrameInfo, held: &Held) -> Vec<KeptSlot> {
        let mut kept = Vec::new();
        for slot in 0..8 {
            if frame.refresh_flags & (1 << slot) != 0 {
                kept.push(KeptSlot {
                    slot,
                    reference: self.refs[slot].take(),
                    hint: self.order_hints[slot],
                });
                self.refs[slot] = Some(Reference {
                    picture: held.picture.clone(),
                    wide: held.wide.clone(),
                    render_width: frame.render_width,
                    render_height: frame.render_height,
                    cdfs: Arc::clone(&held.cdfs),
                    gm: frame.gm_params,
                    field: Arc::clone(&held.motion),
                    lf: SavedLoopFilter {
                        ref_deltas: frame.lf_ref_deltas,
                        mode_deltas: frame.lf_mode_deltas,
                    },
                    segmentation: held.segmentation,
                    segments: Arc::clone(&held.segments),
                    grain: frame.grain,
                    mi_cols: held.mi_cols,
                    mi_rows: held.mi_rows,
                });
                self.order_hints[slot] = frame.order_hint;
            }
        }
        kept
    }

    fn undo_store(&mut self, kept: Vec<KeptSlot>) {
        for slot in kept.into_iter().rev() {
            self.order_hints[slot.slot] = slot.hint;
            self.refs[slot.slot] = slot.reference;
        }
    }

    fn undo_frames(&mut self, frames: Vec<FrameUndo>) {
        for frame in frames.into_iter().rev() {
            self.undo_store(frame.stored);
            self.restore_refs(frame.stale);
        }
    }

    fn picture_before_grain(
        &self,
        sequence: &Sequence,
        frame: &FrameInfo,
        parts: Option<&[TilePart]>,
    ) -> Result<Held, CodecError> {
        if sequence.bit_depth != 8 {
            if sequence.bit_depth != 10 && sequence.bit_depth != 12 {
                return Err(CodecError::Unsupported { what: "bit depth" });
            }
            return self.hold_wide(sequence, frame, parts);
        }
        if frame.show_existing {
            let Some(stored) = self.refs[usize::from(frame.frame_to_show)].as_ref() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "missing reference",
                });
            };
            return Ok(Held {
                picture: stored.picture.clone(),
                wide: None,
                cdfs: Arc::clone(&stored.cdfs),
                motion: Arc::clone(&stored.field),
                segmentation: stored.segmentation,
                segments: Arc::clone(&stored.segments),
                mi_cols: stored.mi_cols,
                mi_rows: stored.mi_rows,
            });
        }
        let reconstructed = self.decode_picture(sequence, frame, parts)?;
        Ok(Held {
            picture: owned(
                reconstructed.picture,
                sequence.subsampling_x,
                sequence.subsampling_y,
            )?,
            wide: None,
            cdfs: Arc::new(reconstructed.cdfs),
            motion: Arc::new(reconstructed.motion),
            segmentation: frame.segmentation,
            segments: Arc::new(reconstructed.segments),
            mi_cols: frame.mi_cols,
            mi_rows: frame.mi_rows,
        })
    }

    /// Profile 0 `high_bitdepth`. Samples stay in `u16`. A monochrome sequence
    /// publishes the luma plane. Film grain is added only to the shown picture.
    fn hold_wide(
        &self,
        sequence: &Sequence,
        frame: &FrameInfo,
        parts: Option<&[TilePart]>,
    ) -> Result<Held, CodecError> {
        if frame.show_existing {
            let Some(stored) = self.refs[usize::from(frame.frame_to_show)].as_ref() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "missing reference",
                });
            };
            let Some(wide) = stored.wide.clone() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "missing reference",
                });
            };
            return Ok(Held {
                picture: stored.picture.clone(),
                wide: Some(wide),
                cdfs: Arc::clone(&stored.cdfs),
                motion: Arc::clone(&stored.field),
                segmentation: stored.segmentation,
                segments: Arc::clone(&stored.segments),
                mi_cols: stored.mi_cols,
                mi_rows: stored.mi_rows,
            });
        }
        let carried = self.carried_cdf(frame)?;
        let views = self.wide_views();
        let reconstructed = match parts {
            Some(parts) => {
                decode_tile_parts(sequence, frame, &views, carried.as_deref(), parts, true)
            }
            None => {
                let generated = crate::header::tile_parts(frame)?;
                decode_tile_parts(
                    sequence,
                    frame,
                    &views,
                    carried.as_deref(),
                    &generated,
                    true,
                )
            }
        }?;
        let picture = reconstructed.picture;
        Ok(Held {
            picture: placeholder_picture()?,
            wide: Some(Arc::new(WidePlanes {
                width: picture.width,
                height: picture.height,
                y: picture.y,
                u: picture.u,
                v: picture.v,
            })),
            cdfs: Arc::new(reconstructed.cdfs),
            motion: Arc::new(reconstructed.motion),
            segmentation: frame.segmentation,
            segments: Arc::new(reconstructed.segments),
            mi_cols: frame.mi_cols,
            mi_rows: frame.mi_rows,
        })
    }

    fn wide_views(&self) -> [Option<RefView<'_, u16>>; 8] {
        std::array::from_fn(|slot| {
            let reference = self.refs[slot].as_ref()?;
            let wide = reference.wide.as_ref()?;
            Some(RefView {
                y: wide.y.as_slice(),
                u: wide.u.as_slice(),
                v: wide.v.as_slice(),
                width: wide.width,
                height: wide.height,
                frame_type: reference.field.frame_type,
                mi_cols: reference.field.mi_cols,
                mi_rows: reference.field.mi_rows,
                saved_hints: reference.field.saved_hints,
                mf_refs: &reference.field.refs,
                mf_mvs: &reference.field.mvs,
            })
        })
    }

    fn take_stale_refs(&mut self, frame: &FrameInfo) -> [Option<Reference>; 8] {
        let mut saved = std::array::from_fn(|_| None);
        let Some(hints) = frame.signaled_order_hints else {
            return saved;
        };
        for slot in 0..8 {
            if hints[slot] != self.order_hints[slot] {
                saved[slot] = self.refs[slot].take();
            }
        }
        saved
    }

    fn restore_refs(&mut self, saved: [Option<Reference>; 8]) {
        for (slot, reference) in saved.into_iter().enumerate() {
            if let Some(reference) = reference {
                self.refs[slot] = Some(reference);
            }
        }
    }

    /// Section 7.3. Each listed tile is decoded from one stored reference and copied
    /// into the output mosaic. The list does not refresh reference slots, and it does
    /// not run the loop filter, CDEF, super resolution, or restoration.
    ///
    /// Anchor frames are the eight reference slots. The specification allows 128
    /// external anchors; this decoder does not keep that extra buffer.
    fn show_tile_list(
        &mut self,
        packet: &Packet,
        sequence: Sequence,
        frame: FrameInfo,
        payload: &[u8],
    ) -> Result<(Option<PictureOutput>, FrameUndo), CodecError> {
        if frame.show_existing || frame.intra || frame.upscaled_width != frame.width {
            return Err(tile_list_error());
        }
        if !sequence.subsampling_x || !sequence.subsampling_y {
            return Err(CodecError::Unsupported { what: "chroma" });
        }
        let list = tile_list(payload)?;
        let (tile_w, tile_h) = uniform_tile_size(&frame)?;
        let out_w = list
            .tiles_wide
            .checked_mul(tile_w)
            .ok_or(CodecError::Limit { what: "frame size" })?;
        let out_h = list
            .tiles_high
            .checked_mul(tile_h)
            .ok_or(CodecError::Limit { what: "frame size" })?;
        if out_w == 0 || out_h == 0 || out_w > 4096 || out_h > 4096 {
            return Err(CodecError::Limit { what: "frame size" });
        }
        let chroma_w = out_w / 2;
        let chroma_h = out_h / 2;
        let y_len = out_w
            .checked_mul(out_h)
            .ok_or(CodecError::Limit { what: "frame size" })?;
        let c_len = chroma_w
            .checked_mul(chroma_h)
            .ok_or(CodecError::Limit { what: "frame size" })?;
        let mut y = vec![0u8; y_len];
        let mut u = vec![128u8; c_len];
        let mut v = vec![128u8; c_len];
        let carried = self.carried_cdf(&frame)?;
        for (index, entry) in list.entries.into_iter().enumerate() {
            if usize::from(entry.anchor) >= self.refs.len()
                || entry.row >= frame.tile_rows as usize
                || entry.col >= frame.tile_cols as usize
            {
                return Err(tile_list_error());
            }
            let Some(&row_start) = frame.mi_row_starts.get(entry.row) else {
                return Err(tile_list_error());
            };
            let Some(&row_end) = frame.mi_row_starts.get(entry.row + 1) else {
                return Err(tile_list_error());
            };
            let Some(&col_start) = frame.mi_col_starts.get(entry.col) else {
                return Err(tile_list_error());
            };
            let Some(&col_end) = frame.mi_col_starts.get(entry.col + 1) else {
                return Err(tile_list_error());
            };
            let views = self.views_for_anchor(&frame, usize::from(entry.anchor))?;
            let part = TilePart {
                row: row_start as usize,
                row_end: row_end as usize,
                col: col_start as usize,
                col_end: col_end as usize,
                data: entry.data,
            };
            let picture = decode_tile_parts(
                &sequence,
                &frame,
                &views,
                carried.as_deref(),
                &[part],
                false,
            )?
            .picture;
            let dx = (index % list.tiles_wide).saturating_mul(tile_w);
            let dy = (index / list.tiles_wide).saturating_mul(tile_h);
            let sx = col_start as usize * 4;
            let sy = row_start as usize * 4;
            copy_rect(PlaneCopy {
                dst: &mut y,
                dst_stride: out_w,
                src: &picture.y,
                src_stride: picture.width,
                dx,
                dy,
                sx,
                sy,
                width: tile_w,
                height: tile_h,
            })?;
            let src_c = picture.width.div_ceil(2);
            copy_rect(PlaneCopy {
                dst: &mut u,
                dst_stride: chroma_w,
                src: &picture.u,
                src_stride: src_c,
                dx: dx / 2,
                dy: dy / 2,
                sx: sx / 2,
                sy: sy / 2,
                width: tile_w / 2,
                height: tile_h / 2,
            })?;
            copy_rect(PlaneCopy {
                dst: &mut v,
                dst_stride: chroma_w,
                src: &picture.v,
                src_stride: src_c,
                dx: dx / 2,
                dy: dy / 2,
                sx: sx / 2,
                sy: sy / 2,
                width: tile_w / 2,
                height: tile_h / 2,
            })?;
        }
        let picture = owned(
            DecodedPicture {
                y,
                u,
                v,
                width: out_w,
                height: out_h,
            },
            true,
            true,
        )?;
        let tags = shown_tags(&sequence, &frame, self.operating_point);
        let shown = if frame.show_frame {
            Some(publish(
                &picture,
                sequence.monochrome,
                &tags,
                packet.pts,
                packet.timescale,
            )?)
        } else {
            None
        };
        self.sequence = Some(sequence);
        Ok((
            shown,
            FrameUndo {
                stored: Vec::new(),
                stale: std::array::from_fn(|_| None),
            },
        ))
    }

    /// The camera-tile process replaces `FrameStore[ref_frame_idx[0]]` with the anchor.
    fn views_for_anchor(
        &self,
        frame: &FrameInfo,
        anchor: usize,
    ) -> Result<[Option<RefView<'_>>; 8], CodecError> {
        let Some(slot) = frame.ref_frames.first().copied() else {
            return Err(tile_list_error());
        };
        let slot = usize::from(slot);
        if slot >= 8 || anchor >= 8 {
            return Err(tile_list_error());
        }
        if self.refs[anchor].is_none() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        }
        Ok(std::array::from_fn(|index| {
            let source = if index == slot { anchor } else { index };
            self.refs[source].as_ref().map(|reference| RefView {
                y: reference.picture.y.samples(),
                u: reference.picture.u.samples(),
                v: reference.picture.v.samples(),
                width: reference.picture.y.width() as usize,
                height: reference.picture.y.height() as usize,
                frame_type: reference.field.frame_type,
                mi_cols: reference.field.mi_cols,
                mi_rows: reference.field.mi_rows,
                saved_hints: reference.field.saved_hints,
                mf_refs: &reference.field.refs,
                mf_mvs: &reference.field.mvs,
            })
        }))
    }
}

fn unfinished_tiles() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "tile group",
    }
}

fn tile_list_error() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "tile list",
    }
}

/// The camera-tile output uses one tile width and one tile height.
fn uniform_tile_size(frame: &FrameInfo) -> Result<(usize, usize), CodecError> {
    let cols = frame.mi_col_starts.len().saturating_sub(1);
    let rows = frame.mi_row_starts.len().saturating_sub(1);
    if cols == 0
        || rows == 0
        || cols != frame.tile_cols as usize
        || rows != frame.tile_rows as usize
    {
        return Err(tile_list_error());
    }
    let width = frame.mi_col_starts[1].saturating_sub(frame.mi_col_starts[0]);
    let height = frame.mi_row_starts[1].saturating_sub(frame.mi_row_starts[0]);
    if width == 0 || height == 0 {
        return Err(tile_list_error());
    }
    for col in 1..cols {
        let start = frame.mi_col_starts[col];
        let end = frame.mi_col_starts[col + 1];
        if end.saturating_sub(start) != width {
            return Err(tile_list_error());
        }
    }
    for row in 1..rows {
        let start = frame.mi_row_starts[row];
        let end = frame.mi_row_starts[row + 1];
        if end.saturating_sub(start) != height {
            return Err(tile_list_error());
        }
    }
    let px_w = usize::try_from(width).unwrap_or(0).saturating_mul(4);
    let px_h = usize::try_from(height).unwrap_or(0).saturating_mul(4);
    if px_w == 0 || px_h == 0 || !px_w.is_multiple_of(2) || !px_h.is_multiple_of(2) {
        return Err(tile_list_error());
    }
    Ok((px_w, px_h))
}

struct PlaneCopy<'a> {
    dst: &'a mut [u8],
    dst_stride: usize,
    src: &'a [u8],
    src_stride: usize,
    dx: usize,
    dy: usize,
    sx: usize,
    sy: usize,
    width: usize,
    height: usize,
}

fn copy_rect(plane: PlaneCopy<'_>) -> Result<(), CodecError> {
    if plane.width == 0 || plane.height == 0 || plane.dst_stride == 0 || plane.src_stride == 0 {
        return Err(tile_list_error());
    }
    for row in 0..plane.height {
        let Some(from) = (plane.sy + row)
            .checked_mul(plane.src_stride)
            .and_then(|off| off.checked_add(plane.sx))
        else {
            return Err(tile_list_error());
        };
        let Some(to) = (plane.dy + row)
            .checked_mul(plane.dst_stride)
            .and_then(|off| off.checked_add(plane.dx))
        else {
            return Err(tile_list_error());
        };
        let Some(src_row) = plane.src.get(from..from + plane.width) else {
            return Err(tile_list_error());
        };
        let Some(dst_row) = plane.dst.get_mut(to..to + plane.width) else {
            return Err(tile_list_error());
        };
        dst_row.copy_from_slice(src_row);
    }
    Ok(())
}

fn record(
    shown: &mut Vec<Output>,
    undo: &mut Vec<FrameUndo>,
    decoded: (Option<PictureOutput>, FrameUndo),
) {
    let (picture, frame_undo) = decoded;
    undo.push(frame_undo);
    if let Some(picture) = picture {
        shown.push(Output::Picture(picture));
    }
}

impl Reference {
    fn coded_size(&self) -> (u32, u32) {
        if let Some(wide) = &self.wide {
            (
                u32::try_from(wide.width).unwrap_or(0),
                u32::try_from(wide.height).unwrap_or(0),
            )
        } else {
            (self.picture.y.width(), self.picture.y.height())
        }
    }
}

fn placeholder_picture() -> Result<Yuv420, CodecError> {
    Yuv420::new(
        Plane::new(2, 2, vec![0; 4])?,
        Plane::new(1, 1, vec![128])?,
        Plane::new(1, 1, vec![128])?,
    )
}

fn shown_tags(sequence: &Sequence, frame: &FrameInfo, operating_point: u8) -> ShownTags {
    let (
        num_units_in_display_tick,
        time_scale,
        equal_picture_interval,
        num_ticks_per_picture_minus_1,
    ) = match sequence.timing {
        Some(timing) => (
            timing.num_units_in_display_tick,
            timing.time_scale,
            timing.equal_picture_interval,
            timing.num_ticks_per_picture_minus_1,
        ),
        None => (0, 0, false, None),
    };
    let num_units_in_decoding_tick = sequence
        .decoder_model
        .map(|model| model.num_units_in_decoding_tick)
        .unwrap_or(0);
    let seq_level_idx = sequence.seq_level_idx[usize::from(operating_point)];
    ShownTags {
        bit_depth: sequence.bit_depth,
        render_width: frame.render_width,
        render_height: frame.render_height,
        primaries: sequence.color_primaries,
        transfer: sequence.transfer_characteristics,
        matrix: sequence.matrix_coefficients,
        chroma_position: sequence.chroma_sample_position,
        num_units_in_display_tick,
        time_scale,
        equal_picture_interval,
        num_ticks_per_picture_minus_1,
        num_units_in_decoding_tick,
        seq_level_idx,
    }
}

fn publish_held(
    held: &Held,
    frame: &FrameInfo,
    monochrome: bool,
    tags: &ShownTags,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    let Some(wide) = &held.wide else {
        let picture = shown(&held.picture, frame)?;
        return publish(&picture, monochrome, tags, pts, timescale);
    };
    let width = u32::try_from(wide.width).map_err(|_| CodecError::Limit { what: "frame size" })?;
    let height =
        u32::try_from(wide.height).map_err(|_| CodecError::Limit { what: "frame size" })?;
    if monochrome {
        if tags.bit_depth != 10 {
            return Err(CodecError::Unsupported { what: "bit depth" });
        }
        if !frame.grain.apply {
            return gray_wide(&wide.y, width, height, tags, pts, timescale);
        }
        let Some((y, _, _)) = crate::grain::overlay(
            &wide.y,
            &wide.u,
            &wide.v,
            wide.width,
            wide.height,
            &frame.grain,
            10,
        ) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "film grain",
            });
        };
        return gray_wide(&y, width, height, tags, pts, timescale);
    }
    if frame.grain.apply && tags.bit_depth != 10 {
        return Err(CodecError::Unsupported { what: "film grain" });
    }
    if !frame.grain.apply {
        return wide_output(
            &wide.y,
            &wide.u,
            &wide.v,
            width,
            height,
            tags,
            (pts, timescale),
        );
    }
    let chroma_w = if frame.grain.sub_x {
        wide.width.div_ceil(2)
    } else {
        wide.width
    };
    let chroma_h = if frame.grain.sub_y {
        wide.height.div_ceil(2)
    } else {
        wide.height
    };
    if wide.u.len() != chroma_w.saturating_mul(chroma_h) {
        return Err(CodecError::Unsupported { what: "film grain" });
    }
    let Some((y, u, v)) = crate::grain::overlay(
        &wide.y,
        &wide.u,
        &wide.v,
        wide.width,
        wide.height,
        &frame.grain,
        10,
    ) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "film grain",
        });
    };
    wide_output(&y, &u, &v, width, height, tags, (pts, timescale))
}

fn publish(
    picture: &Yuv420,
    monochrome: bool,
    tags: &ShownTags,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    if monochrome {
        gray_output(picture, tags, pts, timescale)
    } else {
        picture_output(picture, tags, pts, timescale)
    }
}

fn shown(picture: &Yuv420, frame: &FrameInfo) -> Result<Yuv420, CodecError> {
    if !frame.grain.apply {
        return Ok(picture.clone());
    }
    let width = picture.y.width() as usize;
    let height = picture.y.height() as usize;
    let chroma_w = if frame.grain.sub_x {
        picture.y.width().div_ceil(2)
    } else {
        picture.y.width()
    };
    let chroma_h = if frame.grain.sub_y {
        picture.y.height().div_ceil(2)
    } else {
        picture.y.height()
    };
    if picture.u.width() != chroma_w || picture.u.height() != chroma_h {
        return Err(CodecError::Unsupported { what: "film grain" });
    }
    let Some((y, u, v)) = crate::grain::overlay(
        picture.y.samples(),
        picture.u.samples(),
        picture.v.samples(),
        width,
        height,
        &frame.grain,
        8,
    ) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "film grain",
        });
    };
    owned(
        DecodedPicture {
            y,
            u,
            v,
            width,
            height,
        },
        frame.grain.sub_x,
        frame.grain.sub_y,
    )
}

fn owned(picture: DecodedPicture, sub_x: bool, sub_y: bool) -> Result<Yuv420, CodecError> {
    let width =
        u32::try_from(picture.width).map_err(|_| CodecError::Limit { what: "frame size" })?;
    let height =
        u32::try_from(picture.height).map_err(|_| CodecError::Limit { what: "frame size" })?;
    let chroma_w = if sub_x { width.div_ceil(2) } else { width };
    let chroma_h = if sub_y { height.div_ceil(2) } else { height };
    Yuv420::subsampled(
        Plane::new(width, height, picture.y)?,
        Plane::new(chroma_w, chroma_h, picture.u)?,
        Plane::new(chroma_w, chroma_h, picture.v)?,
        sub_x,
        sub_y,
    )
}

#[cfg(test)]
mod tests {
    use super::Av1Decoder;
    use crate::encode::Av1Encoder;
    use crate::register_decoders;
    use cutlass_decode::{CodecId, CodecParams, DecodeFactory, Decoder};

    use crate::yuv::{AsView, Plane, VideoInput, Yuv420};

    fn solid(width: u32, height: u32, sample: u8) -> Yuv420 {
        let y = vec![sample; (width * height) as usize];
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let chroma = vec![128u8; (cw * ch) as usize];
        Yuv420::new(
            Plane::new(width, height, y).expect("y"),
            Plane::new(cw, ch, chroma.clone()).expect("u"),
            Plane::new(cw, ch, chroma).expect("v"),
        )
        .expect("yuv")
    }

    fn input(picture: Yuv420, pts: i64) -> VideoInput {
        VideoInput {
            picture,
            pts,
            duration: 1,
            timescale: 30,
        }
    }

    #[test]
    fn the_factory_opens_the_rust_decoder() {
        let mut factory = DecodeFactory::new();
        register_decoders(&mut factory);
        let params = CodecParams::video(CodecId::Av1, 16, 16);
        let decoder = factory.open(&params).expect("open");
        assert_eq!(decoder.codec(), CodecId::Av1);
    }

    #[test]
    fn an_empty_packet_is_refused() {
        let params = CodecParams::video(CodecId::Av1, 16, 16);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packet = cutlass_decode::Packet {
            stream: 0,
            data: Vec::new(),
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: true,
            side: Vec::new(),
        };
        assert!(matches!(
            decoder.push(&packet),
            Err(cutlass_decode::CodecError::InvalidApi { .. })
        ));
    }

    #[test]
    fn a_truncated_packet_leaves_no_picture() {
        let params = CodecParams::video(CodecId::Av1, 16, 16);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packet = cutlass_decode::Packet {
            stream: 0,
            data: vec![0x12, 0x00],
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: true,
            side: Vec::new(),
        };
        assert!(decoder.push(&packet).is_err());
        assert!(decoder.drain().expect("drain").is_empty());
        assert!(decoder.drain().expect("drain again").is_empty());
    }

    #[test]
    fn a_repeated_picture_shows_the_stored_reference() {
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_quality(40);
        let picture = solid(32, 32, 40);
        let first = encoder
            .push_picture(&input(picture.clone(), 0))
            .expect("first");
        let second = encoder.push_picture(&input(picture, 1)).expect("second");
        assert!(second[0].data.len() < first[0].data.len());
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let a = decoder.push(&first[0]).expect("decode first");
        let b = decoder.push(&second[0]).expect("decode second");
        assert_eq!(
            a[0].as_video().expect("video").picture.y.samples(),
            b[0].as_video().expect("video").picture.y.samples()
        );
    }

    #[test]
    fn a_bad_packet_keeps_the_reference() {
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        let picture = solid(32, 32, 90);
        let first = encoder
            .push_picture(&input(picture.clone(), 0))
            .expect("first");
        let second = encoder.push_picture(&input(picture, 1)).expect("second");
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let shown = decoder.push(&first[0]).expect("decode");
        let mut broken = second[0].clone();
        broken.data = vec![0xff, 0x00, 0x01];
        assert!(decoder.push(&broken).is_err());
        let again = decoder.push(&second[0]).expect("reference still there");
        assert_eq!(
            shown[0].as_video().expect("video").picture.y.samples(),
            again[0].as_video().expect("video").picture.y.samples()
        );
    }

    #[test]
    fn reset_drops_the_reference() {
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        let picture = solid(32, 32, 70);
        let first = encoder
            .push_picture(&input(picture.clone(), 0))
            .expect("first");
        let second = encoder.push_picture(&input(picture, 1)).expect("second");
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        decoder.push(&first[0]).expect("decode");
        decoder.reset();
        assert!(decoder.push(&second[0]).is_err());
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn hostile_bytes_leave_the_reference() {
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        let picture = solid(32, 32, 50);
        let first = encoder
            .push_picture(&input(picture.clone(), 0))
            .expect("first");
        let repeat = encoder.push_picture(&input(picture, 1)).expect("repeat");
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let shown = decoder.push(&first[0]).expect("decode");
        let mut state = 0x1234_5678u32;
        for _ in 0..200 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = (state % 80) as usize;
            if len == 0 {
                continue;
            }
            let mut data = vec![0u8; len];
            for byte in &mut data {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 16) as u8;
            }
            let packet = cutlass_decode::Packet {
                stream: 0,
                data,
                pts: 0,
                duration: 1,
                timescale: 30,
                keyframe: false,
                side: Vec::new(),
            };
            // A buffer with no size field may be a legal show-existing frame.
            // Either way the stored reference has to survive.
            let _ = decoder.push(&packet);
        }
        let again = decoder.push(&repeat[0]).expect("reference still there");
        assert_eq!(
            shown[0].as_video().expect("video").picture.y.samples(),
            again[0].as_video().expect("video").picture.y.samples()
        );
    }

    fn frame_digest(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    fn leb128(mut value: usize) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            byte |= 0x80;
            out.push(byte);
        }
    }

    fn leb_read(bytes: &[u8]) -> (usize, usize) {
        let mut value = 0usize;
        for (index, byte) in bytes.iter().take(8).enumerate() {
            value |= usize::from(byte & 0x7f) << (index * 7);
            if byte & 0x80 == 0 {
                return (value, index + 1);
            }
        }
        panic!("leb128");
    }

    fn frame_unit(obus: &[&[u8]]) -> Vec<u8> {
        let mut unit = Vec::new();
        for obu in obus {
            unit.extend(leb128(obu.len()));
            unit.extend_from_slice(obu);
        }
        let mut out = Vec::new();
        out.extend(leb128(unit.len()));
        out.extend(unit);
        out
    }

    fn annex_b_temporal(units: &[Vec<u8>]) -> Vec<u8> {
        let mut body = Vec::new();
        for unit in units {
            body.extend_from_slice(unit);
        }
        let mut out = Vec::new();
        out.extend(leb128(body.len()));
        out.extend(body);
        out
    }

    fn annex_b_packet(obus: &[crate::header::UnitObu<'_>]) -> Vec<u8> {
        let bytes: Vec<&[u8]> = obus.iter().map(|obu| obu.bytes).collect();
        annex_b_temporal(&[frame_unit(&bytes)])
    }

    fn push_error(decoder: &mut Av1Decoder, data: &[u8], pts: i64) -> cutlass_decode::CodecError {
        decoder
            .push(&cutlass_decode::Packet {
                stream: 0,
                data: data.to_vec(),
                pts,
                duration: 1,
                timescale: 30,
                keyframe: pts == 0,
                side: Vec::new(),
            })
            .expect_err("packet")
    }

    fn push_frame(decoder: &mut Av1Decoder, data: &[u8], pts: i64) -> Vec<u8> {
        let shown = decoder
            .push(&cutlass_decode::Packet {
                stream: 0,
                data: data.to_vec(),
                pts,
                duration: 1,
                timescale: 30,
                keyframe: pts == 0,
                side: Vec::new(),
            })
            .unwrap_or_else(|error| panic!("frame {pts}: {error:?}"));
        let picture = &shown[0].as_video().expect("video").picture;
        let mut packed = picture.y.samples().to_vec();
        packed.extend_from_slice(picture.u.samples());
        packed.extend_from_slice(picture.v.samples());
        packed
    }

    #[test]
    fn chroma_restoration_stripes_match_dav1d() {
        // aomenc, 256×128 4:2:0. CDEF is on, and every plane selects self-guided
        // restoration with 256-sample units. A chroma stripe is 32 samples
        // tall. The digest is `ffmpeg -c:v libdav1d -threads 1`.
        let frames = ivf_payloads(include_bytes!("../fixtures/restore-420.ivf"));
        assert_eq!(frames.len(), 1);
        let (sequence, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert!(sequence.subsampling_x && sequence.subsampling_y);
        assert!(sequence.enable_cdef);
        assert_eq!(frame.width, 256);
        assert_eq!(frame.height, 128);
        assert_eq!(frame.restoration.plane, [crate::header::RESTORE_SGRPROJ; 3]);
        assert_eq!(frame.restoration.unit, [256; 3]);
        let params = CodecParams::video(CodecId::Av1, 256, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, frames[0], 0);
        assert_eq!(packed.len(), 256 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), 0x0102_7216_ee37_1c58);
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn restoration_on_444_and_422_matches_dav1d() {
        // aomenc --enable-restoration=1 --enable-cdef=0, one frame, cq 40.
        // Profile 1 is 512×128 4:4:4. Chroma is full size, so each chroma plane
        // has two 256-wide restoration units. Profile 2 is 256×512 4:2:2.
        // Chroma keeps the full height, so each chroma plane has two 256-tall
        // units, and the unit row follows the luma sample eight rows above
        // the block. Both frames select self-guided restoration on every plane.
        // Digests are `ffmpeg -c:v libdav1d -threads 1`.
        let cases = [
            (
                include_bytes!("../fixtures/restore-444.ivf").as_slice(),
                1u8,
                false,
                false,
                512u32,
                128u32,
                cutlass_decode::PixelFormat::Yuv444p,
                512 * 128 * 3,
                0x390e_7514_ad62_fb6d,
            ),
            (
                include_bytes!("../fixtures/restore-422.ivf").as_slice(),
                2,
                true,
                false,
                256,
                512,
                cutlass_decode::PixelFormat::Yuv422p,
                256 * 512 + 128 * 512 * 2,
                0xd3c2_410f_120e_df46,
            ),
        ];
        for (file, profile, sub_x, sub_y, width, height, pixel, length, digest) in cases {
            let frames = ivf_payloads(file);
            assert_eq!(frames.len(), 1, "profile {profile}");
            let (sequence, key) = crate::header::parse_unit(frames[0], None).expect("sequence");
            assert_eq!(sequence.profile, profile);
            assert_eq!(sequence.subsampling_x, sub_x);
            assert_eq!(sequence.subsampling_y, sub_y);
            assert_eq!(key.width, width);
            assert_eq!(key.height, height);
            assert_eq!(
                key.restoration.plane,
                [crate::header::RESTORE_SGRPROJ; 3],
                "profile {profile}"
            );
            assert_eq!(key.restoration.unit, [256; 3], "profile {profile}");
            let params = CodecParams::video(CodecId::Av1, width, height);
            let mut decoder = Av1Decoder::open(&params).expect("open");
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: frames[0].to_vec(),
                    pts: 0,
                    duration: 1,
                    timescale: 30,
                    keyframe: true,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("profile {profile}: {error:?}"));
            assert_eq!(shown.len(), 1, "profile {profile}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, pixel, "profile {profile}");
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), length, "profile {profile}");
            assert_eq!(frame_digest(bytes), digest, "profile {profile}");
            assert!(decoder.drain().expect("drain").is_empty());
        }
    }

    #[test]
    fn cdef_index_covers_a_block_wider_than_64() {
        // aomenc --sb-size=128 --min-partition-size=128 --max-partition-size=128.
        // The 128×128 block reads one CDEF index and that index covers all four
        // 64×64 units. The digest is `ffmpeg -c:v libdav1d -threads 1`.
        let frames = ivf_payloads(include_bytes!("../fixtures/cdef-wide.ivf"));
        assert_eq!(frames.len(), 1);
        let (sequence, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert!(sequence.use_128);
        assert!(sequence.enable_cdef);
        assert_eq!(frame.width, 128);
        assert_eq!(frame.height, 128);
        assert_ne!(frame.cdef.y_pri[0], 0);
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, frames[0], 0);
        assert_eq!(packed.len(), 128 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), 0xc184_d76c_fa98_bc01);
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn restoration_units_keep_the_updated_cdf() {
        // 512×128 noise. Luma is self-guided with 128×128 units, four of them.
        // CDF update is on, so each unit after the first reads the adapted
        // probability. Chroma selects no restoration. The digest is
        // `ffmpeg -c:v libdav1d -threads 1`.
        let frames = ivf_payloads(include_bytes!("../fixtures/restore-cdf.ivf"));
        assert_eq!(frames.len(), 1);
        let (_, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert!(!frame.disable_cdf_update);
        assert_eq!(frame.restoration.plane[0], crate::header::RESTORE_SGRPROJ);
        assert_eq!(frame.restoration.plane[1], crate::header::RESTORE_NONE);
        assert_eq!(frame.restoration.plane[2], crate::header::RESTORE_NONE);
        assert_eq!(frame.restoration.unit[0], 128);
        let params = CodecParams::video(CodecId::Av1, 512, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, frames[0], 0);
        assert_eq!(packed.len(), 512 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), 0x9517_c294_09ce_d34f);
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn single_reference_frames_match_dav1d() {
        // aomenc 64x64, tools that this decoder still rejects turned off.
        // Digests are dav1d 1.5.3 output of those three frames.
        let frames: [&[u8]; 3] = [
            &[
                0x12, 0x00, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xf0, 0x10, 0x60, 0x02,
                0x32, 0x60, 0x19, 0x41, 0x41, 0x20, 0x00, 0x00, 0x04, 0x00, 0xdd, 0x2b, 0xdc, 0x5e,
                0xf3, 0x06, 0x3b, 0x9e, 0x5e, 0x5e, 0xf3, 0xf7, 0x13, 0xef, 0xce, 0xc6, 0x63, 0x08,
                0x57, 0xed, 0xcf, 0x8e, 0x26, 0xaa, 0xb5, 0xf7, 0xa3, 0x89, 0x0b, 0x71, 0xf4, 0xab,
                0x0d, 0xb5, 0xf4, 0xb8, 0x42, 0xbf, 0xf6, 0xa2, 0x71, 0xa9, 0x69, 0x33, 0x51, 0x89,
                0x6e, 0x63, 0x48, 0x19, 0x2e, 0xce, 0x25, 0x01, 0x0c, 0x4c, 0x18, 0x45, 0xc0, 0x16,
                0x90, 0x2a, 0x4f, 0xcf, 0x88, 0x3f, 0x49, 0x90, 0x9b, 0x87, 0xbf, 0xa3, 0xf9, 0xbf,
                0xf1, 0xf2, 0x52, 0x7c, 0x3d, 0x8b, 0x97, 0x81, 0x36, 0xf1, 0xa5, 0x8d, 0xed, 0x18,
            ],
            &[
                0x12, 0x00, 0x32, 0x52, 0x3c, 0xa0, 0xa4, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x0a, 0x88, 0x00, 0x00, 0x20,
                0x00, 0xdd, 0x6d, 0xf7, 0x97, 0xfc, 0x56, 0xc9, 0xce, 0x0d, 0x35, 0x39, 0xea, 0x88,
                0xd7, 0x1f, 0x2a, 0x39, 0x32, 0x7d, 0x8d, 0x7d, 0x92, 0x8e, 0x8e, 0x54, 0x9c, 0x75,
                0x85, 0x5b, 0xf7, 0xe9, 0x4c, 0x9a, 0x01, 0x72, 0x2b, 0x02, 0x82, 0x8a, 0xa5, 0xe1,
                0x86, 0x99, 0xaf, 0x2e, 0x7d, 0x8f, 0x8d, 0x64, 0x6a, 0x74, 0xe7, 0xf2, 0x9d, 0x00,
                0xfb, 0xd4,
            ],
            &[
                0x12, 0x00, 0x32, 0x57, 0x3c, 0xa0, 0xa8, 0x08, 0x00, 0x01, 0x00, 0x00, 0x80, 0x00,
                0x48, 0x00, 0x00, 0x00, 0x10, 0x00, 0x08, 0x00, 0x05, 0x0a, 0x80, 0x00, 0x00, 0x20,
                0x00, 0xdd, 0x6d, 0xf7, 0x96, 0xe2, 0xd5, 0xb0, 0xde, 0xe8, 0xb6, 0xed, 0x1c, 0xa2,
                0x64, 0xcc, 0x1b, 0xcd, 0xa7, 0x14, 0x84, 0x74, 0x10, 0xa7, 0x55, 0xaf, 0xd6, 0x1d,
                0xc3, 0x49, 0x08, 0x63, 0xfc, 0x50, 0xc7, 0x56, 0x91, 0xe5, 0xbf, 0xd4, 0x0e, 0x26,
                0x0d, 0x9c, 0xbe, 0x7b, 0x04, 0xeb, 0xa6, 0x86, 0x96, 0x5a, 0x37, 0xa7, 0x0d, 0xa8,
                0x5d, 0xdb, 0xf5, 0x2f, 0x9b, 0xc6, 0x40,
            ],
        ];
        let expected = [
            0x95c8_589e_9dee_2325,
            0x241e_7719_9e22_2325,
            0xb8bc_e8ee_6376_2325,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
            if index == 1 {
                assert_eq!(packed[0], 40);
                assert_eq!(packed[24], 180);
                assert_eq!(packed[56], 40);
            }
        }
    }

    #[test]
    fn a_primary_reference_cdf_matches_dav1d() {
        // aomenc 64×64. The keyframe adapts its CDFs. Each inter frame loads the
        // primary reference and keeps adapting. Digests are
        // `ffmpeg -c:v libdav1d -threads 1` of those three frames.
        let frames: [&[u8]; 3] = [
            &[
                0x12, 0x00, 0x0a, 0x09, 0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x88, 0x30, 0x01, 0x32,
                0x14, 0x10, 0x48, 0x00, 0x00, 0x01, 0x00, 0xb4, 0x5a, 0xcc, 0xa6, 0x0b, 0xe2, 0xb6,
                0x29, 0x06, 0x56, 0x79, 0xa1, 0xd0, 0xe0,
            ],
            &[
                0x12, 0x00, 0x32, 0x2c, 0x30, 0xc0, 0x40, 0x00, 0x00, 0x41, 0x51, 0x00, 0x41, 0x00,
                0x00, 0x40, 0x00, 0xd1, 0x1f, 0xd1, 0x7c, 0xfb, 0xef, 0xec, 0x11, 0xcb, 0x95, 0x22,
                0xc9, 0x7d, 0x61, 0x78, 0xc2, 0x2f, 0x74, 0x36, 0x15, 0x06, 0x7f, 0xd0, 0xdc, 0x8d,
                0xb2, 0xb5, 0x4f, 0x96, 0xd5, 0xf6,
            ],
            &[
                0x12, 0x00, 0x32, 0x14, 0x30, 0x60, 0x80, 0x02, 0x00, 0x41, 0x50, 0x00, 0x41, 0x00,
                0x00, 0x40, 0x00, 0xcd, 0xb9, 0x61, 0x4d, 0x7b, 0xb7, 0xa0,
            ],
        ];
        let expected = [
            0x95c8_589e_9dee_2325,
            0xa64d_3fcf_0838_ab25,
            0x1377_4ca4_510c_0b25,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            if index > 0 {
                assert!(!frame.error_resilient, "frame {index}");
                assert!(frame.primary_ref < 7, "frame {index}");
                assert!(!frame.disable_cdf_update, "frame {index}");
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn an_inter_frame_selects_its_transform_size() {
        // aomenc 64×64. Both frames select the transform size. The inter frame
        // is not error resilient and splits one inter transform. Digests are
        // `ffmpeg -c:v libdav1d -threads 1` of those two frames.
        let frames: [&[u8]; 2] = [
            &[
                0x12, 0x00, 0x0a, 0x09, 0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x88, 0xb0, 0x01, 0x32,
                0x3e, 0x10, 0x40, 0xc0, 0x00, 0x01, 0x40, 0xce, 0xfc, 0x66, 0x8e, 0x0f, 0x23, 0x32,
                0x2f, 0x8e, 0xdc, 0x0f, 0x18, 0x3c, 0x6b, 0xfc, 0xcb, 0x29, 0x71, 0x05, 0x28, 0xf5,
                0x67, 0x21, 0x81, 0x5f, 0xe4, 0xac, 0x02, 0xd0, 0x3c, 0x01, 0x07, 0xe7, 0xf1, 0x21,
                0xdc, 0x34, 0x07, 0x2d, 0xeb, 0x4e, 0x0e, 0xdf, 0xe7, 0xa1, 0x36, 0x48, 0xee, 0xd0,
                0x46, 0xc7, 0x83, 0xcb, 0x04, 0x26, 0xec,
            ],
            &[
                0x12, 0x00, 0x32, 0xb2, 0x01, 0x30, 0xc0, 0x40, 0x00, 0x00, 0x41, 0x0e, 0x00, 0x00,
                0x05, 0x00, 0x00, 0xb6, 0x98, 0x70, 0xeb, 0x68, 0xe6, 0x88, 0x1a, 0x6a, 0x0b, 0x3e,
                0x47, 0xbb, 0x0e, 0xf0, 0x67, 0xd1, 0x28, 0xfe, 0xc5, 0x0c, 0xdc, 0x53, 0xdd, 0xbc,
                0x6a, 0xaf, 0xbd, 0xf1, 0x0c, 0x0f, 0x0e, 0x9d, 0x2c, 0xb5, 0x08, 0x99, 0x22, 0x4c,
                0xb8, 0x0d, 0x76, 0x50, 0x3d, 0xda, 0x37, 0x08, 0xe2, 0xa9, 0x3c, 0xc3, 0x52, 0xe2,
                0x2a, 0x38, 0xe5, 0x6b, 0xf8, 0x89, 0x35, 0x75, 0xb0, 0x14, 0x8d, 0xf2, 0xa6, 0xec,
                0xc5, 0xbe, 0xbe, 0xe8, 0xe0, 0x2f, 0x24, 0xee, 0xeb, 0xb0, 0xe0, 0x60, 0x78, 0x48,
                0xad, 0x94, 0x56, 0x6a, 0xe5, 0xa5, 0x00, 0x20, 0x37, 0x70, 0x3a, 0x33, 0x52, 0x63,
                0xde, 0x3f, 0x35, 0x8b, 0x67, 0xd6, 0x46, 0xc3, 0x84, 0x4b, 0x89, 0x1e, 0x69, 0x0e,
                0x59, 0x5e, 0x14, 0xb3, 0xa4, 0x20, 0x3a, 0x86, 0x03, 0xa0, 0xbd, 0x7a, 0x0d, 0xc0,
                0x7f, 0x0d, 0x89, 0xac, 0xf9, 0xcf, 0xe4, 0x79, 0xce, 0xe2, 0x64, 0x0b, 0x85, 0x2f,
                0x95, 0x33, 0x56, 0xc8, 0xb9, 0x82, 0x05, 0x58, 0xdf, 0xc9, 0x8e, 0xe6, 0xf1, 0xc0,
                0x01, 0x5e, 0x02, 0x7c, 0xa7, 0x44, 0x2a, 0xf3, 0x3c, 0xe4, 0x57, 0xa1, 0x67, 0xc5,
                0x80,
            ],
        ];
        let expected = [0x216f_5038_1f9a_4325, 0x7515_42f8_12fe_45cb];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(frame.tx_select, "frame {index}");
            if index == 1 {
                assert!(!frame.error_resilient);
                assert!(frame.primary_ref < 7);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn superblock_128_matches_dav1d() {
        // aomenc --sb-size=128. Digests are dav1d 1.5.3 with in-loop filters.
        expect_superblock_128(
            &[include_bytes!("../fixtures/sb128_flat_0.obu")],
            &[0xf206_7a5e_2040_2325],
        );
        expect_superblock_128(
            &[
                include_bytes!("../fixtures/sb128_0.obu"),
                include_bytes!("../fixtures/sb128_1.obu"),
            ],
            &[0x4d41_30d9_78c5_115f, 0xcdad_99ad_6d88_94d9],
        );
        expect_superblock_128(
            &[
                include_bytes!("../fixtures/sb128_cdef_0.obu"),
                include_bytes!("../fixtures/sb128_cdef_1.obu"),
            ],
            &[0x1fef_bc5c_9457_b890, 0xfb94_9310_8fde_56d6],
        );
        expect_superblock_128(
            &[
                include_bytes!("../fixtures/sb128_delta_0.obu"),
                include_bytes!("../fixtures/sb128_delta_1.obu"),
            ],
            &[0x6aa5_395a_d395_4f83, 0xdd97_7f4a_be1b_1ff2],
        );
    }

    fn expect_superblock_128(frames: &[&[u8]], expected: &[u64]) {
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("parse");
        assert!(sequence.use_128);
        let params = CodecParams::video(CodecId::Av1, 256, 256);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 256 * 256 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn palette_screen_matches_dav1d() {
        // aomenc --tune-content=screen, from a scaled screen recording.
        // The keyframe uses palette colors. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 4] = [
            include_bytes!("../fixtures/sc_0.obu"),
            include_bytes!("../fixtures/sc_1.obu"),
            include_bytes!("../fixtures/sc_2.obu"),
            include_bytes!("../fixtures/sc_3.obu"),
        ];
        let expected = [
            0x7ea0_91e8_4fa7_1841,
            0xe747_2a4c_3197_44ed,
            0x9228_c426_43d5_ed6e,
            0x8903_d6d5_9e45_fe3e,
        ];
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 128 * 128 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn intrabc_and_integer_motion_match_dav1d() {
        // aomenc --tune-content=screen on a 256×256 graphic, then the same
        // picture again. The keyframe copies a block inside the frame. The
        // inter frame forces integer motion vectors. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 2] = [
            include_bytes!("../fixtures/ibc_0.obu"),
            include_bytes!("../fixtures/ibc_1.obu"),
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            if index == 0 {
                assert!(frame.allow_intrabc, "frame {index}");
            } else {
                assert!(frame.force_integer_mv, "frame {index}");
            }
            sequence = Some(seq);
        }
        let expected = [0x8734_a364_5973_d327, 0x8734_a364_5973_d327];
        let params = CodecParams::video(CodecId::Av1, 256, 256);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 256 * 256 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn lossless_walsh_hadamard_matches_dav1d() {
        // aomenc --lossless=1, from a scaled photographic frame. base_q_idx is 0,
        // so every block is a 4×4 Walsh-Hadamard transform and the frame header
        // omits the loop-filter levels and the CDEF strengths. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 4] = [
            include_bytes!("../fixtures/ll_0.obu"),
            include_bytes!("../fixtures/ll_1.obu"),
            include_bytes!("../fixtures/ll_2.obu"),
            include_bytes!("../fixtures/ll_3.obu"),
        ];
        let expected = [
            0xf5c7_fad9_a756_c601,
            0xb764_46d4_5d8d_96ea,
            0x863c_b381_802e_9309,
            0xe4a4_2ebe_186b_d4e6,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(!frame.show_existing, "frame {index}");
            assert!(frame.lossless, "frame {index}");
            assert_eq!(frame.base_q, 0, "frame {index}");
            assert!(!frame.tx_select, "frame {index}");
            assert_eq!(frame.lf_level, [0, 0, 0, 0], "frame {index}");
            assert_eq!(frame.cdef.bits, 0, "frame {index}");
            assert!(!frame.restoration.uses, "frame {index}");
            if index == 0 {
                assert_eq!(frame.frame_type, crate::header::KEY_FRAME);
            } else {
                assert_eq!(frame.frame_type, crate::header::INTER_FRAME);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 128 * 128 * 3 / 2, "frame {index}");
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn cdef_keyframe_matches_dav1d() {
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0xa0, 0x08, 0x32, 0x5d, 0x98, 0x00,
            0x00, 0x00, 0x50, 0x00, 0x00, 0xdd, 0x2c, 0x71, 0xeb, 0x75, 0x3d, 0xd3, 0x1d, 0xee,
            0xd1, 0x89, 0x93, 0x0d, 0x1e, 0x84, 0x56, 0x77, 0x27, 0xd2, 0xf1, 0x87, 0x90, 0x4f,
            0xde, 0xfe, 0xb2, 0xea, 0x6e, 0x8c, 0xea, 0xa3, 0xf4, 0x66, 0x6b, 0xa3, 0x49, 0x81,
            0x4e, 0xbe, 0x91, 0xb9, 0x53, 0x73, 0x6f, 0xb9, 0x26, 0x91, 0x44, 0xfb, 0x52, 0x34,
            0x48, 0x27, 0xcb, 0x34, 0x3c, 0x69, 0xc6, 0xdc, 0x65, 0x96, 0xd4, 0xa9, 0x60, 0x18,
            0xe4, 0x65, 0x95, 0xb9, 0x6f, 0x3c, 0x0c, 0xf2, 0xdb, 0x9c, 0xc7, 0x37, 0x87, 0xea,
            0x0d, 0x76, 0x1a, 0x98, 0x9a, 0x22, 0x5f,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(frame_digest(&packed), 0x09af_16ec_4c1d_d325);
    }

    #[test]
    fn cdef_across_a_stripe_matches_dav1d() {
        // 64×128, taller than one 64-sample stripe. Strength crosses the boundary.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0xbf, 0xfe, 0x50, 0x04, 0x32, 0xe5, 0x01, 0x92,
            0x80, 0x00, 0x00, 0x24, 0x00, 0x00, 0xdd, 0x2c, 0x22, 0x57, 0x63, 0x90, 0xeb, 0xc5,
            0x4d, 0xa9, 0x51, 0x17, 0x2b, 0xee, 0xf8, 0xad, 0x93, 0xf2, 0x66, 0x45, 0xc4, 0x24,
            0x50, 0x47, 0xe1, 0x63, 0x49, 0xba, 0x0b, 0x04, 0x7b, 0x97, 0x6a, 0x65, 0xf5, 0x8f,
            0x9a, 0x06, 0x6d, 0xb5, 0x71, 0xb7, 0x80, 0xe6, 0x38, 0x97, 0x51, 0x56, 0xf2, 0x06,
            0xeb, 0xf2, 0x71, 0x65, 0x37, 0x72, 0x59, 0x98, 0x95, 0x2b, 0x7c, 0x57, 0x4f, 0x59,
            0xc9, 0x62, 0x30, 0x2f, 0x35, 0x9a, 0x17, 0x17, 0xad, 0x23, 0xf5, 0x64, 0xa4, 0x31,
            0x15, 0x1f, 0xe8, 0x9b, 0xd8, 0x56, 0x16, 0xcd, 0xb6, 0xae, 0x72, 0x23, 0xd9, 0xc5,
            0x1f, 0xb7, 0x43, 0x8c, 0xcd, 0xda, 0x99, 0x7e, 0xa8, 0x4b, 0x36, 0x02, 0x10, 0xf0,
            0x6b, 0x91, 0xc9, 0x92, 0x57, 0xaf, 0xd8, 0x34, 0xc1, 0x1a, 0x27, 0x70, 0xae, 0xb5,
            0xfe, 0x11, 0xed, 0xd1, 0xdf, 0xbf, 0x40, 0xcd, 0xa3, 0x54, 0x1e, 0xea, 0x65, 0x97,
            0x77, 0x78, 0x56, 0x3e, 0xa9, 0xf4, 0x9f, 0x93, 0x40, 0xaf, 0xe3, 0x52, 0x8f, 0xc1,
            0x48, 0x10, 0xc2, 0xd9, 0xb6, 0xd5, 0xce, 0x46, 0x29, 0x1c, 0xcc, 0x55, 0x89, 0x0b,
            0x04, 0x7b, 0x97, 0x6a, 0x66, 0x04, 0x18, 0xff, 0x9a, 0xc9, 0x59, 0xc9, 0x17, 0x3d,
            0xdd, 0xe9, 0xd7, 0x89, 0x19, 0x70, 0xd6, 0xc1, 0xf5, 0xfa, 0xb4, 0x8f, 0x58, 0x28,
            0xe9, 0x0a, 0x69, 0x02, 0x3e, 0x08, 0xf5, 0xf9, 0x33, 0x81, 0x31, 0x66, 0x61, 0xf3,
            0x08, 0xde, 0xa8, 0x49, 0xba, 0x89, 0x09, 0xa5, 0x0e, 0x92, 0x2d, 0xc8, 0xc5, 0x23,
            0x99, 0x8a, 0xb1, 0x10,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), 0xa1d3_4899_7983_e0e5);
    }

    #[test]
    fn superres_ramp_matches_dav1d() {
        // 64×64 shown, coded 43 wide. The horizontal ramp forces every phase.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0xc0, 0x08, 0x32, 0x60, 0xad, 0x50,
            0x00, 0x00, 0x04, 0x00, 0xdd, 0x2c, 0x6c, 0xb6, 0xfe, 0x1e, 0x1f, 0xeb, 0x1b, 0xdf,
            0x39, 0x81, 0x7a, 0xea, 0xe6, 0x3f, 0xf8, 0xc2, 0xaa, 0x35, 0xf5, 0x77, 0x02, 0x56,
            0x6a, 0x8a, 0x5a, 0xe4, 0xd7, 0xae, 0xbd, 0x50, 0x44, 0x15, 0x13, 0x54, 0x24, 0xde,
            0xb9, 0xdf, 0xbc, 0xa4, 0xf7, 0x09, 0x28, 0x6e, 0x81, 0xad, 0x21, 0x8e, 0xf4, 0xad,
            0xf0, 0x47, 0xb6, 0x65, 0x9a, 0x17, 0x17, 0xad, 0x21, 0x8e, 0xf4, 0xad, 0xf0, 0x47,
            0xb6, 0x73, 0xde, 0x64, 0xe5, 0xc5, 0xde, 0xc4, 0x47, 0xaf, 0xca, 0xe7, 0xb2, 0xd8,
            0xd6, 0x17, 0xcf, 0xae, 0xc0, 0xef, 0x26, 0x81, 0x66, 0x2a,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 2);
        assert_eq!(packed[63], 188);
        assert_eq!(frame_digest(&packed), 0x336f_e213_3c36_3325);
    }

    #[test]
    fn restoration_left_off_matches_dav1d() {
        // Restoration is enabled in the sequence and every plane selects none.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0x80, 0x08, 0x32, 0x59, 0x95, 0x00,
            0x00, 0x00, 0x40, 0xdd, 0x2b, 0xfa, 0x7e, 0xc8, 0x8b, 0x61, 0xfb, 0x0c, 0xeb, 0xc2,
            0x6c, 0x8b, 0x0d, 0x04, 0x5c, 0x1d, 0x90, 0x28, 0xbb, 0x24, 0x05, 0xcc, 0x0f, 0x7d,
            0x45, 0xfa, 0xac, 0xac, 0x84, 0x6d, 0x2c, 0x8b, 0x0c, 0x94, 0xe3, 0x14, 0xc9, 0x69,
            0xcb, 0xaa, 0x78, 0x8c, 0xe8, 0xe1, 0xb3, 0x5c, 0x40, 0x51, 0x76, 0x48, 0x0b, 0x98,
            0x1f, 0x0f, 0x5b, 0x78, 0x8c, 0xe6, 0x3a, 0xee, 0xe9, 0x39, 0x45, 0x93, 0xa9, 0xe2,
            0x33, 0x98, 0xeb, 0xbb, 0xa4, 0xe5, 0x14, 0x50, 0x83, 0x21, 0xee, 0x8f, 0xa2, 0xa5,
            0x04, 0x34, 0x80,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert!(packed[..64 * 64].iter().all(|sample| *sample == 40));
        assert_eq!(frame_digest(&packed), 0x2067_6a1f_17e4_0325);
    }

    #[test]
    fn self_guided_restoration_matches_dav1d() {
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0xb0, 0x08, 0x32, 0xb1, 0x07, 0x97,
            0x80, 0x00, 0x74, 0x00, 0x04, 0x9e, 0x01, 0x87, 0x00, 0xc9, 0xb9, 0xad, 0x13, 0xc8,
            0x1b, 0x9c, 0x13, 0x56, 0x42, 0x1a, 0x48, 0x42, 0x21, 0x18, 0x28, 0xbd, 0x33, 0x0f,
            0xae, 0x15, 0x33, 0x3c, 0xa6, 0x07, 0x37, 0xfe, 0xeb, 0xac, 0x12, 0xa9, 0x10, 0x8d,
            0x94, 0x1d, 0x12, 0xe8, 0xb7, 0x5c, 0x33, 0x74, 0xe2, 0x6b, 0xfb, 0xd5, 0xa7, 0x7f,
            0x32, 0x7a, 0xaa, 0xc7, 0xd6, 0x9e, 0x4f, 0xe5, 0xd6, 0x24, 0x4a, 0x31, 0x5f, 0xf1,
            0x49, 0xd6, 0xfb, 0xef, 0x03, 0x9e, 0xf8, 0x19, 0x45, 0x8f, 0x7f, 0xa0, 0x1a, 0x58,
            0x9e, 0xfb, 0xdf, 0x89, 0x6b, 0x12, 0x07, 0x67, 0x3b, 0x26, 0x66, 0x2a, 0xdd, 0x98,
            0x8a, 0xc9, 0x2a, 0xf2, 0x8f, 0xb8, 0xcc, 0x99, 0x05, 0x20, 0xc3, 0x3b, 0x81, 0x52,
            0xed, 0x37, 0x4d, 0x1b, 0xe1, 0xee, 0x2d, 0xf7, 0x3d, 0x9b, 0x2d, 0xca, 0xda, 0x13,
            0xa6, 0x28, 0x44, 0xde, 0x25, 0x7f, 0x9a, 0x9f, 0xa0, 0xe1, 0x9f, 0x4f, 0x70, 0x57,
            0xfd, 0x4d, 0x42, 0x9d, 0x5a, 0xf2, 0xc5, 0xc6, 0x83, 0x4c, 0x8c, 0xe6, 0xa8, 0xc6,
            0x04, 0x34, 0x75, 0xf7, 0x89, 0xf9, 0x87, 0x05, 0x9d, 0x03, 0xb4, 0x42, 0x65, 0xc4,
            0x1b, 0x37, 0x02, 0x84, 0x36, 0x57, 0xa1, 0x4c, 0x16, 0x4a, 0xe2, 0xa1, 0x93, 0x02,
            0xb9, 0x9e, 0xf1, 0x27, 0xf6, 0xf6, 0x07, 0xef, 0x0e, 0xfd, 0x5f, 0x8d, 0xde, 0x71,
            0x41, 0xe7, 0x57, 0x9d, 0x12, 0x8e, 0xa5, 0xb1, 0x7f, 0x6f, 0x06, 0x46, 0x3c, 0xa5,
            0x49, 0xca, 0x89, 0x0b, 0x04, 0x03, 0xa6, 0xfc, 0x4f, 0xd8, 0x5e, 0x53, 0xe4, 0xe2,
            0x81, 0xbb, 0x7a, 0x55, 0x01, 0x38, 0xe3, 0x2b, 0x6d, 0x3b, 0x7a, 0xa4, 0x42, 0x12,
            0x83, 0x43, 0xe8, 0x97, 0x1f, 0x39, 0xb9, 0x02, 0xba, 0xd8, 0x1e, 0x8a, 0xab, 0x6c,
            0x9a, 0xe2, 0x1a, 0xcf, 0x77, 0xc9, 0x8e, 0xeb, 0xcc, 0x40, 0x50, 0xa6, 0x1e, 0x0e,
            0x51, 0xcc, 0x27, 0x59, 0xf3, 0x81, 0x68, 0xdd, 0x55, 0xe3, 0xf1, 0x19, 0x40, 0x67,
            0x28, 0xb4, 0x01, 0x26, 0x79, 0x23, 0x9b, 0xa0, 0x55, 0xba, 0x87, 0x80, 0x81, 0xf8,
            0xcc, 0x68, 0x9e, 0x61, 0x75, 0x6b, 0x07, 0xbe, 0xcc, 0xb0, 0x76, 0xa4, 0x68, 0x81,
            0xf2, 0x28, 0xfc, 0x6d, 0x92, 0x77, 0xfe, 0x50, 0x6b, 0xf1, 0x4b, 0x0a, 0x58, 0x84,
            0x8c, 0xb3, 0xdd, 0xef, 0xca, 0x45, 0x9f, 0xfc, 0x70, 0xcb, 0x30, 0x87, 0x77, 0x1b,
            0x97, 0x17, 0xe7, 0x59, 0x88, 0x40, 0x12, 0xf1, 0x36, 0xb4, 0x0f, 0xf0, 0xcc, 0x56,
            0x80, 0xed, 0xc8, 0xa4, 0x2b, 0x16, 0xb4, 0xb3, 0x6e, 0x2b, 0x1a, 0x25, 0x64, 0x97,
            0x80, 0x77, 0xee, 0x02, 0x2e, 0xa6, 0x4b, 0x05, 0x14, 0x68, 0x40, 0xa4, 0x21, 0xfb,
            0x9c, 0x43, 0x48, 0x2d, 0xe1, 0x0b, 0x49, 0x6f, 0xfa, 0x17, 0x6e, 0x46, 0x49, 0x66,
            0xec, 0x1a, 0x02, 0x82, 0xe3, 0x22, 0xe8, 0x18, 0x63, 0x89, 0xe8, 0x25, 0xff, 0xdd,
            0xd5, 0x81, 0xf6, 0xbe, 0x35, 0x66, 0xf7, 0x46, 0x00, 0xaf, 0xa9, 0xbc, 0x91, 0x4e,
            0xb7, 0x00, 0xd5, 0xe4, 0x0b, 0x74, 0x4f, 0xc4, 0x50, 0x16, 0x13, 0xa7, 0x34, 0x69,
            0x62, 0x39, 0xec, 0xc3, 0x90, 0x1e, 0x93, 0xe1, 0x34, 0x6e, 0xbd, 0x72, 0xa8, 0x83,
            0xda, 0x8e, 0xd0, 0xab, 0x5d, 0xde, 0xf5, 0xb4, 0x81, 0x33, 0x72, 0x44, 0xc8, 0xa8,
            0xdd, 0x04, 0x18, 0x0c, 0x4a, 0xf6, 0xb1, 0xb4, 0x29, 0x56, 0xc5, 0xfc, 0xe6, 0x46,
            0xb5, 0x51, 0x7d, 0x34, 0xda, 0x16, 0x6d, 0x6a, 0xfa, 0x05, 0x29, 0xd3, 0x8b, 0x12,
            0xea, 0x02, 0x73, 0x9f, 0xa1, 0x01, 0xdb, 0xfc, 0xc6, 0x06, 0xcf, 0x7b, 0xf3, 0x28,
            0x0d, 0x88, 0x7e, 0xf0, 0x07, 0x56, 0xde, 0x3a, 0x15, 0xa9, 0x92, 0xf8, 0x8a, 0x0d,
            0x08, 0x7b, 0x08, 0x90, 0xcf, 0x35, 0xfe, 0x4a, 0x23, 0x80, 0xee, 0xf5, 0x2f, 0xc9,
            0x9e, 0x67, 0xf2, 0xec, 0xf2, 0x5d, 0xf4, 0xa3, 0x72, 0x55, 0xaa, 0xb7, 0xf1, 0xf6,
            0x04, 0x34, 0x3d, 0x58, 0xdb, 0x8c, 0x03, 0x34, 0x10, 0x7d, 0x5a, 0x58, 0xff, 0xde,
            0x32, 0xcd, 0x3b, 0x5f, 0x23, 0x3d, 0x27, 0xbb, 0x28, 0xf2, 0x14, 0xe9, 0xfb, 0x46,
            0x90, 0xa4, 0x1b, 0x66, 0x0c, 0xc8, 0x28, 0xa0, 0x12, 0xbb, 0x31, 0xda, 0xb4, 0x12,
            0x97, 0xa5, 0x6b, 0x8f, 0xd3, 0x32, 0x33, 0x87, 0x96, 0x8b, 0xba, 0x26, 0x68, 0xc7,
            0xb6, 0xfb, 0xf2, 0x1d, 0x92, 0xa8, 0x3f, 0x90, 0xcf, 0xc0, 0x7b, 0x7c, 0xea, 0x3d,
            0x76, 0xa2, 0x5c, 0xf9, 0x3c, 0xd8, 0xdb, 0x07, 0xc0, 0x51, 0xa2, 0x6a, 0xd3, 0x4e,
            0x44, 0x82, 0x22, 0x8a, 0x17, 0x79, 0xb5, 0x81, 0x64, 0x1b, 0x0d, 0x78, 0x2b, 0x2a,
            0x5b, 0x38, 0x2b, 0xc1, 0x53, 0x42, 0x54, 0xdb, 0x32, 0x99, 0x53, 0x0c, 0x69, 0xd3,
            0x63, 0xbf, 0x02, 0xda, 0x73, 0xe2, 0x00, 0xec, 0x83, 0x58, 0x22, 0x7b, 0x06, 0x24,
            0xe7, 0x0d, 0x34, 0xec, 0x98, 0xfb, 0xf9, 0x91, 0xb5, 0xb6, 0xc3, 0x45, 0xbd, 0x30,
            0x5d, 0x93, 0x97, 0x22, 0x25, 0x75, 0xaf, 0x90, 0x24, 0x2a, 0x8a, 0x3d, 0xe5, 0x1f,
            0x8b, 0x91, 0xfa, 0x6d, 0xf3, 0x84, 0xac, 0x48, 0xc2, 0xda, 0xa2, 0xc8, 0x53, 0x65,
            0x84, 0x1d, 0xe8, 0x77, 0x50, 0x37, 0x1b, 0x3f, 0xff, 0xb7, 0x86, 0x46, 0xc9, 0x9f,
            0xf9, 0x5d, 0x85, 0xe8, 0xd6, 0x08, 0xb3, 0x5f, 0xf3, 0xad, 0x25, 0x46, 0x94, 0xde,
            0xe0, 0x49, 0x38, 0x84, 0x63, 0xc3, 0x1d, 0xc0, 0x03, 0x80, 0x52, 0x2b, 0x04, 0xa8,
            0x53, 0x55, 0x5d, 0xca, 0x34, 0xeb, 0x21, 0xc9, 0xe2, 0xe3, 0x4c, 0xec, 0x64, 0x15,
            0x38, 0x77, 0x62, 0x09, 0x0e, 0xea, 0x83, 0x10, 0xf7, 0x3b, 0xa6, 0x79, 0xe4, 0x54,
            0x0e, 0xee, 0xf8, 0x5f, 0x3d, 0xc7, 0xd0, 0xb8, 0x4b, 0xdb, 0x0b, 0xdd, 0xfb, 0x0f,
            0xbb, 0x40, 0x50, 0x86, 0xa5, 0xdc, 0xe3, 0x0b, 0x2e, 0x63, 0x6e, 0xa3, 0x7e, 0x7b,
            0xd4, 0x4b, 0x03, 0xe1, 0x4f, 0x00, 0x82, 0x40, 0xa6, 0xff, 0xf7, 0x62, 0x07, 0x3d,
            0xf0, 0x4c, 0x18, 0x46, 0x0d, 0xae, 0xac, 0xe7, 0xba, 0xfe, 0x9f, 0xf3, 0x3e, 0x0f,
            0x78, 0x55, 0x47, 0x4b, 0xf2, 0xd0, 0x16, 0x70, 0x74, 0x91, 0xf1, 0x26, 0x75, 0xd3,
            0x3a, 0x44, 0x18, 0x0a, 0x82, 0x73, 0xda, 0x21, 0xff, 0xda, 0x09, 0x8c, 0x46, 0x4d,
            0x98, 0xa5, 0xbc, 0x43, 0x29, 0x07, 0x55, 0x7f, 0xe3, 0x39, 0x16, 0x8c, 0xdc, 0x60,
            0x04, 0x7d, 0xc6, 0x64, 0xc8, 0x28, 0xd1, 0xc9, 0x58, 0x05, 0x1f, 0xe0, 0x13, 0xa0,
            0xfc, 0xb5, 0x74, 0xdd, 0x8e, 0x26, 0x89, 0x63, 0x17, 0x82, 0x5d, 0xf4, 0x90, 0xe5,
            0x0c, 0x49, 0x3f, 0x77, 0x9a, 0xa4, 0xcd, 0x21, 0xf4, 0xf8, 0xec, 0x34, 0x31, 0x9a,
            0x7b, 0xd9, 0xcd, 0xca, 0xe9, 0x89, 0x26, 0x46, 0x31, 0xb7, 0x58, 0xef, 0xa0, 0x2c,
            0xc7, 0xfb, 0x59, 0x2e, 0x40, 0x40,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 6);
        assert_eq!(packed[32], 20);
        assert_eq!(frame_digest(&packed), 0x1cc8_07a0_f463_5138);
    }

    #[test]
    fn wiener_restoration_matches_dav1d() {
        // 64×64 ramp. Luma uses the Wiener filter. Chroma selects none.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0x90, 0x08, 0x32, 0x6f, 0x97, 0x00,
            0x00, 0x00, 0x50, 0x70, 0xf9, 0x3f, 0xd7, 0xf9, 0x35, 0x67, 0x1f, 0x2b, 0x13, 0x59,
            0xa1, 0xb2, 0x71, 0xe2, 0x26, 0x01, 0x57, 0xdc, 0x58, 0xb9, 0xf6, 0x60, 0xe1, 0x5d,
            0x7d, 0xa5, 0xf5, 0xf7, 0x61, 0x2d, 0xe6, 0x2e, 0x21, 0x2c, 0xea, 0x71, 0xfe, 0x6f,
            0x04, 0x1b, 0x7e, 0x22, 0xc3, 0x00, 0x69, 0x2d, 0x2b, 0x9d, 0x65, 0xd4, 0x36, 0xc3,
            0x7a, 0x77, 0x33, 0x8a, 0x23, 0xd8, 0x47, 0xa8, 0x2e, 0xa4, 0x33, 0x8a, 0x20, 0x01,
            0x19, 0x2d, 0x5d, 0xa9, 0xdb, 0x42, 0xc2, 0x16, 0x9c, 0xf6, 0x2e, 0x9a, 0xa6, 0x2e,
            0xf6, 0x42, 0x8b, 0x42, 0x00, 0x0f, 0x33, 0x72, 0x1a, 0x00, 0x5f, 0x3e, 0x00, 0x69,
            0xf0, 0x8d, 0x8c, 0x9a, 0xda, 0x8d, 0xcf, 0x08, 0xf5, 0x05, 0xd4,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 2);
        assert_eq!(packed[63], 187);
        assert_eq!(frame_digest(&packed), 0x68b2_c8a0_4688_cf25);
    }

    #[test]
    fn switchable_restoration_matches_dav1d() {
        // 512×64. The luma plane lets each restoration unit choose its filter.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x21, 0x7f, 0xff, 0x92, 0x01, 0x32, 0x90, 0x3a, 0x93,
            0x80, 0x00, 0x00, 0x24, 0x10, 0xae, 0x84, 0x2f, 0xae, 0x7a, 0x47, 0x16, 0xb1, 0x31,
            0xae, 0x82, 0x12, 0x26, 0xe5, 0x97, 0x4b, 0x6a, 0xdb, 0x05, 0x87, 0xf5, 0xd7, 0x8a,
            0xe9, 0x05, 0x11, 0xde, 0x25, 0x45, 0x81, 0xaa, 0x33, 0xd3, 0x58, 0x1b, 0xe8, 0xc4,
            0xc8, 0x6d, 0x94, 0xbb, 0x53, 0x93, 0x27, 0x22, 0xd0, 0xed, 0x69, 0xc3, 0xd3, 0xca,
            0xda, 0x04, 0xf3, 0x57, 0x61, 0x3f, 0x15, 0xf5, 0xb1, 0x6d, 0xc4, 0xc1, 0xda, 0xf2,
            0xda, 0xdd, 0xdd, 0x16, 0x92, 0x66, 0x0c, 0x63, 0x06, 0x93, 0x08, 0x97, 0xe9, 0x5d,
            0x00, 0x31, 0x93, 0x61, 0xdc, 0xf8, 0x4d, 0xf8, 0x06, 0x2d, 0x08, 0x00, 0x3d, 0xa8,
            0x8d, 0xbf, 0x79, 0xed, 0x32, 0xd7, 0x7f, 0x38, 0xb0, 0x34, 0x7b, 0x79, 0x49, 0x90,
            0x23, 0xdf, 0xce, 0x2c, 0x0d, 0x1e, 0xde, 0x55, 0x0f, 0x28, 0x89, 0xfe, 0xed, 0xe2,
            0x8d, 0x4b, 0x35, 0x45, 0xa3, 0x83, 0x46, 0x55, 0x07, 0x0a, 0xd1, 0x86, 0xc8, 0x68,
            0x33, 0x04, 0x74, 0x99, 0xa6, 0x7c, 0x60, 0xe2, 0x36, 0x5e, 0x2d, 0x06, 0x1e, 0xef,
            0x5a, 0x48, 0x84, 0x66, 0x07, 0x32, 0x4b, 0x1f, 0xa0, 0x21, 0x4f, 0xff, 0x12, 0xef,
            0x69, 0xa7, 0x44, 0x3a, 0x9d, 0x5e, 0x2a, 0xc8, 0x1a, 0x20, 0xe4, 0x22, 0xbc, 0x22,
            0x90, 0x50, 0xd4, 0x9a, 0xfc, 0xae, 0xf5, 0xa4, 0x88, 0x46, 0x2e, 0xcf, 0xc0, 0x27,
            0xae, 0x78, 0xe6, 0xca, 0x21, 0x87, 0xc0, 0xd1, 0xec, 0x7d, 0x0c, 0x71, 0xf4, 0xdb,
            0x3c, 0x27, 0x70, 0x72, 0xb1, 0xc6, 0x83, 0x2f, 0xba, 0xd9, 0xe0, 0x9e, 0x87, 0x55,
            0x77, 0xc6, 0xef, 0x2c, 0xb1, 0x28, 0xab, 0xa3, 0x1a, 0x4d, 0x1f, 0xe5, 0xa1, 0x0d,
            0x43, 0x72, 0x2d, 0x3e, 0xe3, 0xcd, 0x6d, 0xb6, 0xd9, 0xf6, 0x0b, 0xc0, 0xfa, 0x31,
            0x08, 0xb0, 0x67, 0xa2, 0xe3, 0x58, 0xba, 0xfb, 0xc0, 0x11, 0x39, 0x9e, 0x31, 0x35,
            0xa3, 0x98, 0x1c, 0x1c, 0xce, 0xee, 0xb6, 0x66, 0xae, 0x86, 0x4a, 0xaf, 0xac, 0xc0,
            0xbb, 0x36, 0x1a, 0x57, 0x59, 0xf2, 0x63, 0x4a, 0xeb, 0x3a, 0xe7, 0x51, 0x48, 0x2e,
            0x13, 0x2b, 0x53, 0x86, 0x52, 0x49, 0x85, 0xc2, 0xb6, 0x3f, 0x28, 0x5c, 0xac, 0x2c,
            0x14, 0x19, 0x77, 0xc6, 0xef, 0x2c, 0x8c, 0x18, 0xa1, 0x61, 0x60, 0x03, 0xa8, 0xf6,
            0xf2, 0x92, 0xf5, 0x26, 0xbf, 0x2b, 0xbd, 0x69, 0x22, 0x11, 0x8a, 0xd6, 0x13, 0xd6,
            0x5e, 0x5f, 0x76, 0x29, 0x8b, 0x1d, 0xb1, 0x48, 0x5f, 0xfe, 0xa7, 0x5f, 0x4d, 0xe1,
            0xfb, 0x8e, 0xae, 0x63, 0x34, 0xc6, 0x63, 0xce, 0x6f, 0x92, 0xcf, 0x64, 0x16, 0xad,
            0x84, 0x3a, 0xd8, 0x9e, 0x88, 0xdc, 0x0a, 0xee, 0xef, 0x65, 0x18, 0x56, 0xfe, 0x48,
            0x32, 0x10, 0xde, 0xe8, 0x60, 0x84, 0x19, 0x57, 0x9a, 0x00, 0xb8, 0x70, 0xe8, 0x58,
            0x35, 0xd3, 0x1f, 0x60, 0x4a, 0xc5, 0x21, 0xb6, 0xb8, 0xdd, 0xe5, 0x91, 0x83, 0x14,
            0x2c, 0x2c, 0x00, 0x75, 0x1e, 0xde, 0x52, 0x5e, 0xa4, 0xd7, 0xe5, 0x77, 0xad, 0x24,
            0x42, 0x31, 0x5a, 0xc2, 0x7a, 0xcb, 0x68, 0xeb, 0xde, 0x51, 0x0c, 0x3e, 0x06, 0x8f,
            0x63, 0xe8, 0x63, 0x8f, 0xa6, 0xd9, 0xce, 0x17, 0x09, 0x95, 0x97, 0xb0, 0xbb, 0x3e,
            0x88, 0xd2, 0x0f, 0xf4, 0xe8, 0x80, 0xf0, 0xde, 0x77, 0x24, 0x5e, 0xb3, 0x14, 0xdd,
            0x96, 0x80, 0x3d, 0x86, 0xf0, 0xa5, 0x42, 0x3d, 0x4b, 0xdd, 0x3b, 0x8b, 0x6d, 0x6c,
            0x47, 0xcd, 0x2d, 0x2b, 0x05, 0x54, 0x6b, 0x89, 0x1e, 0xcb, 0xf6, 0x2c, 0x1b, 0x48,
            0xfd, 0xd3, 0x94, 0x94, 0x4b, 0xac, 0x75, 0xc6, 0x8c, 0xf7, 0xf6, 0x21, 0x0a, 0x0b,
            0x03, 0xa9, 0xae, 0x23, 0x91, 0x59, 0x9e, 0x64, 0xf6, 0x2d, 0x05, 0x26, 0xd3, 0xff,
            0x87, 0x7f, 0x8e, 0xca, 0x7e, 0xfd, 0xfe, 0x4a, 0x6b, 0xcf, 0xbe, 0xdc, 0xfe, 0xd0,
            0xfb, 0x2d, 0xdd, 0xc2, 0x41, 0xf4, 0xd4, 0x67, 0x48, 0x3b, 0x43, 0x88, 0x8a, 0xd8,
            0x76, 0x4b, 0x71, 0xf0, 0xfa, 0x0b, 0x8d, 0x65, 0x58, 0x47, 0x71, 0x1b, 0xbf, 0x49,
            0x10, 0xa6, 0x3e, 0x98, 0x68, 0xf6, 0x8c, 0x9f, 0x2e, 0xe9, 0x67, 0x17, 0x09, 0xc0,
            0x5b, 0x17, 0xdc, 0x9b, 0x14, 0x10, 0xb3, 0x31, 0x53, 0x78, 0x95, 0x22, 0xa8, 0xc4,
            0xf7, 0x78, 0x2e, 0xa6, 0x23, 0x41, 0x63, 0x07, 0xfe, 0x02, 0x3d, 0xe7, 0xbf, 0x24,
            0xb5, 0xf4, 0x9c, 0xbe, 0x85, 0x62, 0xc1, 0xa4, 0x7c, 0x68, 0x03, 0x7e, 0x36, 0x49,
            0x51, 0xf5, 0xe1, 0xbb, 0x32, 0xa5, 0x25, 0xb8, 0xd0, 0xcf, 0x68, 0x35, 0x95, 0x70,
            0x76, 0xa6, 0x2e, 0x16, 0xe3, 0xa4, 0xc5, 0xc3, 0x97, 0xfd, 0x25, 0xbe, 0x3f, 0x09,
            0xd1, 0xca, 0xf1, 0xee, 0x96, 0xee, 0x49, 0xb2, 0xfe, 0xe9, 0x6c, 0xb2, 0x7b, 0x42,
            0x96, 0x66, 0x39, 0x81, 0xe7, 0x05, 0x48, 0x36, 0xd2, 0xb6, 0x0f, 0xc0, 0x86, 0x05,
            0xc0, 0x3d, 0xb3, 0xa5, 0xc0, 0x42, 0xdd, 0xd6, 0xc8, 0x05, 0xd6, 0xc7, 0x22, 0x41,
            0x95, 0x5e, 0xe9, 0x32, 0x27, 0x4b, 0xfd, 0x0b, 0xcc, 0x03, 0xc5, 0xa9, 0x2b, 0x01,
            0x42, 0xae, 0x79, 0x24, 0x98, 0xd5, 0x28, 0x42, 0x82, 0x94, 0x40, 0xad, 0x75, 0xcc,
            0x13, 0x34, 0x8d, 0x7e, 0x5c, 0xa0, 0x31, 0x3b, 0x77, 0xc0, 0x62, 0x16, 0xdf, 0x2c,
            0x90, 0x39, 0x11, 0x49, 0x61, 0x40, 0xb6, 0xa2, 0x2d, 0x53, 0x63, 0x30, 0xb9, 0x8c,
            0x50, 0xb5, 0x3d, 0x42, 0x5c, 0x8d, 0x51, 0x56, 0xc1, 0xf8, 0xa4, 0x62, 0xc2, 0xc6,
            0x08, 0x95, 0xea, 0xd6, 0x77, 0x0a, 0x94, 0x85, 0x4e, 0xc7, 0xe1, 0x03, 0x04, 0xfc,
            0x32, 0x1d, 0xa8, 0x17, 0xbe, 0x12, 0x90, 0x4a, 0xe8, 0x5e, 0xd1, 0xa6, 0x70, 0x24,
            0x23, 0xfc, 0xec, 0xbc, 0xee, 0xb1, 0xa0, 0x91, 0x0f, 0xc2, 0x2a, 0xce, 0x93, 0x6c,
            0x58, 0x54, 0xcd, 0x95, 0xdc, 0x66, 0x5e, 0x02, 0x0c, 0xb6, 0x80, 0x50, 0x82, 0xe1,
            0x97, 0x06, 0xd2, 0xea, 0xf2, 0x77, 0x28, 0x43, 0x82, 0xf1, 0xad, 0x81, 0x72, 0xb9,
            0x48, 0xfa, 0x18, 0x08, 0x87, 0xd7, 0x92, 0x0b, 0xd9, 0x9d, 0x1d, 0x66, 0x1b, 0xd0,
            0x5b, 0x03, 0x30, 0xc1, 0xbb, 0x06, 0xa1, 0xe1, 0x44, 0x1a, 0xda, 0xa8, 0x3a, 0x01,
            0x4e, 0xd7, 0x63, 0x02, 0x3e, 0x17, 0x38, 0x05, 0x21, 0xcd, 0x17, 0xbc, 0xb9, 0x60,
            0xe1, 0xfa, 0x1d, 0x9d, 0x98, 0xe7, 0xa3, 0xab, 0x58, 0xa1, 0xd7, 0xdc, 0xfe, 0x96,
            0xd8, 0x8a, 0xe5, 0xa5, 0x9a, 0x3f, 0xf4, 0x78, 0xa0, 0x1f, 0x7f, 0x78, 0x0d, 0xc6,
            0xad, 0x96, 0xe6, 0xf0, 0xc9, 0xba, 0x57, 0x48, 0x11, 0x1e, 0x60, 0xdf, 0x75, 0x22,
            0x2e, 0xf8, 0x38, 0x46, 0x26, 0x1e, 0xc8, 0x87, 0x73, 0x0f, 0x6e, 0x9f, 0x6d, 0x61,
            0x4b, 0xf5, 0x0a, 0x0e, 0x9d, 0xdb, 0x3c, 0x25, 0x62, 0xac, 0xf2, 0xa5, 0xd8, 0xea,
            0xaa, 0xd7, 0x04, 0x26, 0x8a, 0xc8, 0x6b, 0x49, 0x3d, 0x4f, 0x5e, 0x90, 0x52, 0xc0,
            0xf9, 0xf3, 0xbb, 0xc4, 0x6c, 0x34, 0xe6, 0xd5, 0x68, 0x5c, 0x72, 0x8f, 0xde, 0x3a,
            0xca, 0xb7, 0x21, 0x81, 0xd4, 0xe8, 0xae, 0x7b, 0x90, 0x68, 0x45, 0xe8, 0xde, 0x77,
            0xc8, 0x6a, 0x50, 0x5d, 0x57, 0x91, 0x00, 0x5e, 0x5e, 0xe1, 0x53, 0xc1, 0x54, 0x5e,
            0xa7, 0x66, 0x81, 0x71, 0xdb, 0x7a, 0x82, 0x09, 0x32, 0x34, 0x38, 0x2e, 0x21, 0x67,
            0x88, 0xc3, 0x04, 0x8d, 0xb6, 0x1f, 0x7b, 0x09, 0x26, 0x97, 0x83, 0xc5, 0xbf, 0xba,
            0x26, 0xdf, 0x81, 0xe8, 0x2b, 0x19, 0xc2, 0x26, 0xec, 0xd5, 0xa9, 0xb4, 0xfa, 0x3b,
            0x21, 0x7a, 0xb9, 0x56, 0x5a, 0x12, 0xcf, 0xea, 0x10, 0xe6, 0x0b, 0x7f, 0xf6, 0x8e,
            0x42, 0x87, 0xcc, 0xdf, 0x40, 0xd8, 0xa1, 0x0a, 0x7a, 0xfa, 0x54, 0xd8, 0x85, 0xb9,
            0x65, 0x03, 0x3b, 0x4a, 0x27, 0xf9, 0x66, 0x48, 0x44, 0x05, 0x23, 0x98, 0x50, 0x1f,
            0x2a, 0x2c, 0x92, 0x2c, 0x31, 0x73, 0x3f, 0xe5, 0xd6, 0xe2, 0xa0, 0x55, 0x25, 0xcc,
            0x46, 0x19, 0xd3, 0x8f, 0x22, 0x1f, 0x83, 0x9e, 0x18, 0x5b, 0x11, 0xbb, 0x54, 0x56,
            0xd5, 0xfb, 0x31, 0xab, 0x6c, 0x6d, 0xa2, 0x77, 0x5d, 0x23, 0xa2, 0x04, 0x39, 0xa3,
            0x54, 0x21, 0x3e, 0xc3, 0xe7, 0x71, 0xbd, 0x5f, 0x1c, 0x32, 0xea, 0xf0, 0x1d, 0x4b,
            0xb8, 0xc2, 0xb4, 0xf3, 0xee, 0x1c, 0xbd, 0x51, 0x06, 0x5f, 0x90, 0xd1, 0x7e, 0xf5,
            0x76, 0x8e, 0x4e, 0x0b, 0x56, 0x02, 0x1f, 0x88, 0x6b, 0x33, 0x6f, 0xfd, 0x67, 0x46,
            0xa4, 0x09, 0x7d, 0x88, 0xa2, 0x0c, 0xca, 0x7a, 0x3d, 0xed, 0xbc, 0xcd, 0x56, 0xe5,
            0x7c, 0xc2, 0x3f, 0x9e, 0x94, 0xef, 0x97, 0x45, 0x76, 0x4d, 0x35, 0x49, 0x1c, 0xf5,
            0x73, 0xb6, 0x33, 0x78, 0x35, 0x88, 0xa9, 0xea, 0xc2, 0x1f, 0xf1, 0x20, 0x7d, 0x4f,
            0x34, 0xd6, 0xd9, 0x0a, 0x97, 0xee, 0x27, 0x31, 0xa5, 0x68, 0x0d, 0x60, 0x98, 0x33,
            0x74, 0xbd, 0x9d, 0xfa, 0x7f, 0x8b, 0x71, 0xf7, 0xd9, 0x12, 0x0f, 0xa2, 0xea, 0xb4,
            0x7f, 0xd9, 0x2d, 0x50, 0xda, 0x81, 0xf2, 0xac, 0xf8, 0x94, 0x2b, 0x06, 0x53, 0xe3,
            0x6c, 0xf3, 0xdd, 0xdc, 0x41, 0x5e, 0x38, 0xf8, 0x2b, 0xea, 0xb4, 0xc1, 0x01, 0x13,
            0x3e, 0xe4, 0xa8, 0xa6, 0x76, 0x03, 0x9b, 0xe3, 0x5e, 0x79, 0x9b, 0xb3, 0x6b, 0xf9,
            0x08, 0x12, 0x3b, 0x52, 0x69, 0x23, 0x95, 0x4d, 0xcf, 0x3c, 0x38, 0xa7, 0x6c, 0xed,
            0x1d, 0xab, 0xe2, 0x3e, 0xd0, 0xb8, 0xc9, 0xaa, 0xfe, 0x94, 0x2b, 0x10, 0x50, 0xcf,
            0xfc, 0x63, 0x07, 0x06, 0x44, 0xb4, 0xc1, 0x85, 0xa1, 0x55, 0xc3, 0x1d, 0xc7, 0x3e,
            0x80, 0xb4, 0xb8, 0xef, 0x53, 0x18, 0x59, 0x05, 0xb8, 0x42, 0x2b, 0x38, 0xd2, 0x05,
            0x71, 0x36, 0xf6, 0xa9, 0xf5, 0x31, 0xd9, 0x37, 0x41, 0x16, 0x9d, 0x89, 0x35, 0x4e,
            0xf9, 0x02, 0xe3, 0x5e, 0x07, 0x7d, 0x8c, 0x62, 0x64, 0xb2, 0x4d, 0x1b, 0xde, 0xab,
            0xf8, 0x5a, 0xac, 0x2c, 0x6f, 0x6a, 0xb4, 0xc9, 0xd0, 0x61, 0x22, 0x97, 0x6c, 0xc9,
            0xc5, 0x1f, 0x65, 0xe8, 0x86, 0x19, 0x66, 0x53, 0x05, 0x4f, 0x31, 0xf5, 0x29, 0x81,
            0xda, 0x96, 0xc3, 0x35, 0x55, 0x0e, 0x35, 0xb8, 0x18, 0x44, 0xb3, 0x2e, 0x9c, 0x38,
            0xdd, 0x38, 0x35, 0x5f, 0x0c, 0xe4, 0x09, 0xc2, 0x6e, 0x2e, 0xd0, 0x92, 0x44, 0xed,
            0xf9, 0x20, 0x74, 0x0c, 0x0d, 0xf7, 0xf8, 0x42, 0x30, 0x37, 0xa2, 0x5d, 0x0f, 0xcd,
            0x67, 0x74, 0x18, 0x1b, 0xdd, 0x3d, 0x0a, 0x9f, 0x1c, 0xeb, 0x3d, 0x04, 0x0a, 0x02,
            0x06, 0x2f, 0xbf, 0xbf, 0xd5, 0x55, 0x60, 0x71, 0x69, 0x04, 0x66, 0xf1, 0x3b, 0x9b,
            0xa5, 0x4e, 0xa0, 0x87, 0xa7, 0x37, 0x66, 0x73, 0x3b, 0x10, 0xe7, 0x10, 0x17, 0xf7,
            0x0a, 0x45, 0x34, 0x9b, 0xb7, 0x43, 0x52, 0xd1, 0x6f, 0xa8, 0x5e, 0xbc, 0x2d, 0xb9,
            0xc8, 0x13, 0xa8, 0xec, 0x9e, 0xc0, 0x44, 0x36, 0x08, 0x73, 0x18, 0xd0, 0x6f, 0x19,
            0x7a, 0x33, 0x76, 0x10, 0x4b, 0xbd, 0xa1, 0x78, 0x58, 0x1b, 0x27, 0x5b, 0x67, 0xf4,
            0x99, 0x1c, 0x23, 0x32, 0xc1, 0x93, 0x9e, 0xa5, 0x34, 0x57, 0xad, 0xdf, 0x74, 0x56,
            0x64, 0xa5, 0x5b, 0xf7, 0x1f, 0x8d, 0xa8, 0x19, 0x42, 0x03, 0x3f, 0x37, 0x2f, 0x7e,
            0x17, 0xe6, 0xe0, 0xd1, 0x69, 0x08, 0x71, 0x92, 0x1a, 0x63, 0x10, 0x0f, 0x14, 0x11,
            0xe8, 0x36, 0x9e, 0x20, 0x35, 0xb6, 0xaf, 0x69, 0xd0, 0xe9, 0x86, 0x32, 0x2e, 0x10,
            0x35, 0x53, 0x7f, 0xba, 0xd2, 0xf7, 0x40, 0x20, 0xfc, 0x23, 0xc0, 0xd5, 0x32, 0x8c,
            0x44, 0x73, 0xb4, 0x4e, 0xf9, 0x4f, 0x4d, 0xea, 0xe0, 0x95, 0xf5, 0xe1, 0x7e, 0xb7,
            0x63, 0x7f, 0xa1, 0x96, 0xae, 0xb2, 0x43, 0x23, 0x3a, 0xe5, 0x06, 0x5e, 0x10, 0x01,
            0x91, 0xe7, 0x3b, 0x72, 0xd8, 0xfd, 0x32, 0xa2, 0x9e, 0xcd, 0x69, 0x59, 0x91, 0xd6,
            0x57, 0xb1, 0xa1, 0x89, 0x95, 0x04, 0xcc, 0xc9, 0x43, 0xd7, 0xef, 0xa1, 0xb3, 0xea,
            0x88, 0x07, 0xbb, 0xa0, 0x2d, 0x65, 0x74, 0xf2, 0x31, 0xa2, 0x8b, 0xda, 0x8c, 0xb7,
            0x45, 0xf7, 0x14, 0xfb, 0x04, 0x9a, 0x94, 0xac, 0x17, 0x8c, 0x4f, 0x79, 0x55, 0x32,
            0x41, 0x8e, 0xbd, 0xbe, 0x44, 0x2c, 0xcf, 0x8f, 0xd6, 0x60, 0x8b, 0x9a, 0x5e, 0x78,
            0x7d, 0xce, 0x80, 0x5b, 0x7e, 0x32, 0xe0, 0xbf, 0x41, 0xc4, 0x75, 0xf3, 0x34, 0x15,
            0x01, 0xc0, 0x4d, 0x4d, 0x0b, 0x1d, 0x5c, 0xaa, 0xe9, 0x0e, 0xd2, 0x83, 0x3d, 0x6c,
            0xd5, 0x92, 0xcf, 0xe5, 0x20, 0x45, 0x25, 0x1c, 0x3e, 0x39, 0xd5, 0x57, 0x5f, 0xc6,
            0xc5, 0xb0, 0xa7, 0x6d, 0x2d, 0x47, 0x8a, 0x3c, 0x79, 0x1d, 0xe7, 0xcb, 0x5c, 0xd3,
            0x40, 0xe6, 0x23, 0xe5, 0x84, 0xff, 0x29, 0x16, 0x58, 0x4e, 0x33, 0xc7, 0xe0, 0xab,
            0x56, 0xcf, 0x2c, 0x81, 0xd6, 0xc8, 0xa4, 0x44, 0x5c, 0x07, 0x0a, 0x84, 0x03, 0x1b,
            0x03, 0x1e, 0x5a, 0x2e, 0xf8, 0x31, 0x37, 0x04, 0xc7, 0x81, 0x8e, 0xf1, 0x34, 0x8c,
            0xcc, 0xe2, 0x64, 0x84, 0x7c, 0xdd, 0x49, 0x06, 0xdf, 0xd5, 0xf4, 0x23, 0x2f, 0x54,
            0x69, 0x9a, 0x5e, 0x81, 0x1c, 0xe9, 0x37, 0x87, 0x5c, 0xff, 0x84, 0x23, 0xd8, 0x7b,
            0x54, 0x5c, 0x63, 0x06, 0xe0, 0x0c, 0xa7, 0xb6, 0x0e, 0x19, 0xe9, 0xa5, 0xa0, 0xb0,
            0x09, 0xd1, 0x35, 0x45, 0x38, 0xdf, 0x85, 0xbd, 0x42, 0x23, 0xc2, 0xfe, 0x4b, 0x09,
            0xfc, 0x26, 0x11, 0xad, 0x60, 0x22, 0x23, 0xca, 0x7b, 0x63, 0xe8, 0xd7, 0x95, 0x71,
            0x38, 0x40, 0xd8, 0xb2, 0xcc, 0x8f, 0xfd, 0xea, 0xee, 0x38, 0xcc, 0xbc, 0x04, 0x19,
            0x4d, 0x82, 0x1a, 0xb1, 0xd8, 0x76, 0xac, 0xf1, 0x8c, 0x12, 0x02, 0x71, 0xbd, 0x0c,
            0x57, 0xa7, 0xb1, 0x31, 0xfe, 0x17, 0x9c, 0xf4, 0xe4, 0x00, 0xb3, 0x57, 0x26, 0x01,
            0xf9, 0x15, 0x41, 0x94, 0x49, 0xca, 0x32, 0x58, 0xcd, 0x0c, 0x3b, 0x31, 0x9c, 0x1b,
            0x4a, 0x34, 0xa7, 0x0f, 0xb9, 0x62, 0xf7, 0x7d, 0xf1, 0xc8, 0xc1, 0x6a, 0xa2, 0x6a,
            0x6b, 0x7a, 0x7a, 0x93, 0x3f, 0x76, 0x4d, 0xcc, 0x47, 0xbf, 0x73, 0xe3, 0xb8, 0xd0,
            0x00, 0xef, 0x07, 0xaf, 0x49, 0xae, 0x15, 0xfc, 0x88, 0xe3, 0x56, 0xa6, 0x37, 0xcc,
            0xbe, 0x23, 0x72, 0xa6, 0x0f, 0x3a, 0xfc, 0x3e, 0xf6, 0x5a, 0xc9, 0x4c, 0xcb, 0x74,
            0x3d, 0x9c, 0xe9, 0x96, 0xc7, 0x7c, 0x15, 0x7d, 0x3f, 0xf9, 0x84, 0xdf, 0x81, 0x40,
            0x49, 0x37, 0xca, 0x3c, 0xb4, 0x46, 0x58, 0xaa, 0x22, 0xe0, 0x7f, 0xfa, 0x4b, 0x46,
            0x9a, 0x67, 0xe2, 0x08, 0xad, 0x63, 0x99, 0x42, 0x28, 0xc2, 0x52, 0x0a, 0x87, 0xab,
            0xcf, 0xf7, 0xc6, 0xea, 0x97, 0x4d, 0x70, 0x3c, 0xf1, 0xa6, 0x14, 0x18, 0xde, 0x6c,
            0xc8, 0x0d, 0x6f, 0xdc, 0xe5, 0xa4, 0x7e, 0x81, 0x99, 0xd0, 0xe8, 0xc7, 0x88, 0xa9,
            0x7e, 0xe6, 0xc4, 0x04, 0x75, 0x02, 0x8e, 0x6d, 0xcb, 0x64, 0xc4, 0x80, 0xe1, 0x7d,
            0x0f, 0x5e, 0xbd, 0xc8, 0x6a, 0x5a, 0xc3, 0x5d, 0x30, 0x01, 0x81, 0x8d, 0x5f, 0xc1,
            0x7c, 0x8d, 0xfe, 0xd6, 0xfc, 0xd5, 0xb7, 0x36, 0x0c, 0x12, 0x01, 0xc2, 0x54, 0x42,
            0x8a, 0x4f, 0xab, 0xf7, 0xfa, 0xe1, 0xc4, 0xc6, 0x09, 0x96, 0xd6, 0x71, 0xd8, 0xd1,
            0x6a, 0xf7, 0x61, 0x6b, 0x96, 0x71, 0x39, 0xf3, 0x82, 0x1e, 0x86, 0x62, 0xe0, 0x1f,
            0x22, 0x0f, 0x1c, 0x81, 0x5e, 0x2c, 0x02, 0x2a, 0xf7, 0x5a, 0xb6, 0x09, 0x10, 0x02,
            0xc1, 0x11, 0x9d, 0x3f, 0xe3, 0xb9, 0xdf, 0xb1, 0x4e, 0x56, 0x8b, 0x15, 0x57, 0x09,
            0xd0, 0x42, 0xca, 0xd0, 0x34, 0xae, 0x8e, 0x56, 0x6c, 0x1c, 0x1b, 0x03, 0xa2, 0xc1,
            0xca, 0x80, 0x16, 0x77, 0xf0, 0x65, 0x66, 0xe4, 0xc1, 0x86, 0xc8, 0x43, 0xfd, 0xf4,
            0x13, 0x14, 0xbc, 0x23, 0x31, 0x76, 0x1a, 0x0f, 0x49, 0xd5, 0x1e, 0x15, 0xa4, 0x81,
            0x0d, 0x91, 0xdc, 0xb5, 0x45, 0xa6, 0x33, 0x18, 0xf6, 0xde, 0x33, 0x4c, 0x8a, 0xdb,
            0x39, 0x69, 0x69, 0xbf, 0x2a, 0x88, 0xfe, 0x03, 0x24, 0x0b, 0x43, 0x83, 0x11, 0xc1,
            0xde, 0x54, 0xe7, 0x96, 0xf7, 0x72, 0xb4, 0xc7, 0x0c, 0x68, 0x08, 0xbd, 0x44, 0xca,
            0x82, 0x2a, 0x8a, 0xf1, 0xa4, 0xd0, 0x37, 0x0c, 0xdd, 0xd2, 0xbf, 0xee, 0x45, 0x0f,
            0xf5, 0x95, 0x2a, 0x34, 0xbf, 0x08, 0xaa, 0x2e, 0x76, 0x10, 0xeb, 0x4b, 0x4e, 0xe6,
            0x21, 0x11, 0x3b, 0xff, 0xc8, 0xf3, 0xe1, 0xd5, 0x37, 0x38, 0xc4, 0x39, 0x68, 0x4b,
            0xa9, 0x5d, 0x2d, 0xd0, 0x5c, 0xe3, 0x8e, 0x04, 0x8e, 0x05, 0xee, 0xbe, 0x0f, 0xef,
            0x3d, 0x1d, 0x20, 0xbb, 0x82, 0xf9, 0xf5, 0xdb, 0x56, 0x7b, 0xf5, 0x36, 0x72, 0x68,
            0xc7, 0x4e, 0x77, 0x2c, 0xb7, 0x15, 0x30, 0xea, 0x0f, 0xb9, 0x54, 0x44, 0x51, 0x22,
            0x5f, 0x3e, 0xca, 0x05, 0x65, 0x3f, 0x86, 0xaa, 0x93, 0x11, 0x48, 0x98, 0x5f, 0x48,
            0x17, 0x45, 0x61, 0x3d, 0x87, 0xd0, 0x34, 0x6a, 0xec, 0xb5, 0x8c, 0x3a, 0x0c, 0x17,
            0x71, 0x25, 0x59, 0x98, 0xf3, 0xcf, 0xc5, 0xa2, 0x61, 0xf0, 0x3c, 0xbc, 0xe4, 0xad,
            0x09, 0xa7, 0x7f, 0x16, 0x7a, 0xe6, 0xb0, 0xa4, 0x3a, 0xb3, 0xd7, 0x1c, 0x48, 0xa4,
            0xbf, 0xd4, 0x3a, 0xcf, 0xdd, 0x31, 0x45, 0x3c, 0x76, 0xbb, 0x41, 0x46, 0x9f, 0x6e,
            0x13, 0x2d, 0x1e, 0x67, 0x1b, 0x07, 0xeb, 0x50, 0xa4, 0xdb, 0xc3, 0x92, 0x0a, 0xef,
            0xf7, 0x9f, 0x2a, 0x3c, 0xf2, 0x1d, 0xba, 0x74, 0x1b, 0xc2, 0xc9, 0x3b, 0x26, 0x08,
            0x18, 0x7b, 0xbb, 0x85, 0x94, 0x28, 0x18, 0x70, 0xdf, 0x81, 0x37, 0x78, 0xca, 0x93,
            0x4e, 0xd1, 0xfb, 0xac, 0xf7, 0xf7, 0xa5, 0x41, 0xe3, 0xe3, 0x08, 0x84, 0xf1, 0xbb,
            0xc7, 0xfa, 0xe2, 0x77, 0x35, 0x79, 0x21, 0x50, 0x17, 0x11, 0xef, 0x39, 0x7f, 0x21,
            0x9d, 0x2d, 0x10, 0x98, 0x4c, 0xe5, 0xe2, 0x9d, 0x1a, 0xe9, 0x7c, 0x3a, 0x35, 0xf0,
            0x13, 0xde, 0x64, 0xc6, 0x83, 0xc7, 0x13, 0x91, 0x0a, 0xc2, 0x82, 0x57, 0x7e, 0x62,
            0x89, 0x04, 0xf8, 0x5f, 0x26, 0xd0, 0x1c, 0xbb, 0xdd, 0xde, 0x5a, 0x99, 0x2b, 0xef,
            0xcb, 0xe8, 0xe8, 0x78, 0x1f, 0x93, 0x59, 0xb5, 0xb4, 0xb4, 0x85, 0x10, 0xd5, 0x3b,
            0xd5, 0xf3, 0xbf, 0x4a, 0xb1, 0xa5, 0x32, 0x16, 0x76, 0xeb, 0x96, 0x38, 0xdb, 0x1c,
            0x6c, 0x10, 0xec, 0x45, 0x37, 0xd3, 0xbf, 0xa1, 0xc1, 0x63, 0xd1, 0xba, 0x8b, 0xfd,
            0x5a, 0xae, 0xb9, 0x1d, 0x0e, 0x8a, 0x2d, 0x02, 0x29, 0x4c, 0x0f, 0x9d, 0x9e, 0xd2,
            0xd8, 0xb3, 0xf6, 0xe7, 0xda, 0xfb, 0x4e, 0x09, 0xf7, 0xaa, 0x95, 0x57, 0x3f, 0xd1,
            0x44, 0x58, 0x96, 0x35, 0x82, 0x49, 0xa1, 0x99, 0x26, 0x5b, 0xc6, 0x67, 0x6e, 0x41,
            0x44, 0x63, 0xf3, 0x93, 0x66, 0xd4, 0x5b, 0xea, 0xc8, 0x94, 0x64, 0x28, 0x01, 0x83,
            0x26, 0xfc, 0xba, 0x8c, 0x20, 0xde, 0xed, 0xe7, 0xf1, 0x0f, 0x56, 0x46, 0x3e, 0xc5,
            0x6e, 0xc2, 0x04, 0x0d, 0xa1, 0x1f, 0x37, 0xde, 0x29, 0x81, 0xdf, 0x74, 0xae, 0xc8,
            0x93, 0x88, 0xcd, 0xf2, 0x53, 0xc3, 0x8e, 0x89, 0xb9, 0x7a, 0x98, 0x04, 0xfe, 0xde,
            0x8c, 0xa0, 0x89, 0x8d, 0x3d, 0x91, 0x40, 0x54, 0x06, 0x93, 0x44, 0x46, 0xd1, 0x37,
            0xec, 0x50, 0x85, 0x62, 0xce, 0x27, 0x74, 0xdd, 0xd0, 0x1d, 0x71, 0x56, 0xf6, 0x46,
            0x29, 0x33, 0xf2, 0x94, 0xe3, 0x3e, 0x6e, 0x4b, 0xe3, 0xe4, 0xa0, 0xef, 0x3d, 0x40,
            0x26, 0x4b, 0x8b, 0x9c, 0xb1, 0xb2, 0x2c, 0x15, 0x85, 0xa9, 0x99, 0x01, 0xe6, 0xd2,
            0xd1, 0xc1, 0x90, 0xeb, 0xfa, 0x94, 0x16, 0x57, 0x2f, 0x20, 0x07, 0xcf, 0x36, 0x0b,
            0x73, 0xe9, 0xe3, 0xa9, 0xf8, 0x93, 0x29, 0xaf, 0x16, 0x04, 0x0e, 0x40, 0xed, 0xfc,
            0xf9, 0x87, 0x49, 0x4c, 0x4d, 0xf4, 0xdc, 0xb8, 0xb4, 0xab, 0x71, 0x29, 0xdf, 0xbe,
            0x86, 0x2f, 0x43, 0xa2, 0x30, 0x36, 0x20, 0x2b, 0xa7, 0x1a, 0xcb, 0xf3, 0x19, 0x55,
            0xef, 0x54, 0x7c, 0xb7, 0xfa, 0x06, 0xc8, 0x04, 0xf7, 0xe9, 0xce, 0xf5, 0x0a, 0x11,
            0x62, 0xfd, 0x56, 0xc0, 0x72, 0x4f, 0x3c, 0x3b, 0x82, 0xa9, 0x4b, 0x8f, 0xbc, 0x1a,
            0x8e, 0x5d, 0x2f, 0xc6, 0x98, 0x46, 0x0a, 0xc1, 0xae, 0x4f, 0x0c, 0x1d, 0x2a, 0x28,
            0x45, 0x52, 0x44, 0xe5, 0x92, 0x88, 0x4f, 0x48, 0x15, 0x58, 0xdd, 0x89, 0xbb, 0x47,
            0x40, 0x2d, 0x7a, 0x06, 0xee, 0x7e, 0x1d, 0x02, 0x4c, 0xd8, 0x30, 0x04, 0x40, 0x54,
            0x81, 0x75, 0x2d, 0x34, 0xf1, 0x00, 0x4c, 0x2b, 0x56, 0x01, 0xdd, 0xcd, 0x31, 0x97,
            0xdf, 0x1a, 0xb9, 0x3d, 0xa9, 0x1c, 0xfa, 0x40, 0x27, 0xef, 0x8e, 0x20, 0x0d, 0x11,
            0x83, 0xdd, 0xf5, 0x3c, 0x30, 0xdd, 0x6e, 0x8c, 0x4d, 0x48, 0xdd, 0x89, 0x96, 0xc3,
            0x96, 0x0c, 0x02, 0x6d, 0x58, 0x14, 0x9d, 0x37, 0x14, 0x71, 0x8f, 0x53, 0xcb, 0xcc,
            0x23, 0xec, 0xb0, 0x0a, 0xff, 0x00, 0xe2, 0xe7, 0xbb, 0x4e, 0xff, 0x4b, 0xd4, 0xc5,
            0x8f, 0x61, 0x24, 0x81, 0x43, 0x1f, 0x22, 0x3c, 0x1e, 0x95, 0x1f, 0x7e, 0x8c, 0x11,
            0xa4, 0x51, 0xbb, 0x77, 0x8a, 0xd2, 0x2e, 0x0a, 0x74, 0xef, 0x95, 0xe9, 0x29, 0x37,
            0xdb, 0xe6, 0xda, 0xcb, 0x54, 0x9b, 0xbb, 0xef, 0xca, 0x85, 0xde, 0x53, 0x9a, 0xea,
            0x32, 0x69, 0x48, 0x96, 0x80, 0x91, 0xb1, 0xca, 0xf6, 0x1f, 0xed, 0x07, 0x0b, 0x8c,
            0x83, 0x5c, 0xe9, 0xe2, 0x53, 0xf3, 0x01, 0x95, 0x83, 0x99, 0x92, 0x57, 0xf5, 0x3f,
            0x03, 0xb5, 0x95, 0x06, 0x4f, 0x90, 0x4b, 0x67, 0x66, 0x96, 0xdc, 0xf7, 0xb0, 0x58,
            0x2d, 0x10, 0x42, 0x88, 0x58, 0x68, 0x43, 0x02, 0x31, 0x7a, 0x9d, 0x50, 0xe0, 0x6d,
            0xc8, 0x84, 0xab, 0xa3, 0xc1, 0xbd, 0x15, 0x3e, 0x5b, 0xd6, 0xb5, 0x72, 0xcf, 0x68,
            0xef, 0xc5, 0x2b, 0x2b, 0x25, 0x37, 0x45, 0x62, 0xfd, 0x2f, 0xd9, 0x31, 0x6f, 0x7f,
            0xcd, 0x34, 0x91, 0x77, 0x84, 0xc7, 0x13, 0x95, 0x52, 0x87, 0x6e, 0xd2, 0x35, 0x98,
            0x38, 0xe1, 0x9b, 0x09, 0x60, 0x57, 0xd6, 0xa8, 0xf2, 0x5e, 0x9e, 0xf7, 0xbc, 0xba,
            0xed, 0xb2, 0xef, 0x78, 0x55, 0x31, 0xc9, 0xdc, 0x0f, 0xb5, 0x68, 0x05, 0x97, 0x3f,
            0x73, 0xa6, 0x31, 0xea, 0xd2, 0x8f, 0xed, 0xe6, 0xeb, 0x70, 0x9f, 0x59, 0xb7, 0x38,
            0x07, 0xa8, 0x2d, 0xc6, 0x5e, 0xe3, 0x86, 0xcd, 0x2a, 0x87, 0x8f, 0x3b, 0x8c, 0x18,
            0xcd, 0x89, 0x31, 0x7c, 0xb3, 0xbd, 0xee, 0x56, 0x03, 0xb7, 0x9a, 0x57, 0x5b, 0x35,
            0xdd, 0x5e, 0x80, 0xab, 0x99, 0x1f, 0xba, 0xf2, 0x36, 0x38, 0x42, 0x5e, 0x2b, 0x9c,
            0xc3, 0x3c, 0x7b, 0x3b, 0x9b, 0xc4, 0xd6, 0xba, 0xd8, 0x49, 0x84, 0xb1, 0x04, 0xf1,
            0x16, 0xb9, 0x57, 0xa2, 0x6d, 0xb9, 0x23, 0xab, 0xb0, 0xe0, 0x22, 0x8d, 0xf7, 0x5e,
            0xb4, 0x7c, 0x21, 0x86, 0xe6, 0xcc, 0xdc, 0xdb, 0x7d, 0x64, 0x05, 0x33, 0x35, 0xcf,
            0xa6, 0xbe, 0xed, 0x9a, 0xd1, 0xa5, 0x68, 0xaa, 0xe2, 0x95, 0x72, 0xcc, 0xb4, 0xfa,
            0x24, 0x6a, 0x25, 0x73, 0xff, 0xa5, 0x1a, 0x2e, 0x4a, 0x29, 0xb2, 0x59, 0xfa, 0x29,
            0x48, 0xd0, 0xb3, 0xbf, 0xe2, 0xd6, 0x26, 0x1d, 0xe2, 0xe5, 0xff, 0x5e, 0xed, 0x8a,
            0xf7, 0xe2, 0xdd, 0x78, 0x16, 0x96, 0x78, 0x5d, 0xdb, 0x62, 0xf2, 0xfb, 0x0d, 0xdc,
            0xea, 0x0b, 0x5f, 0x4c, 0x4a, 0x31, 0xf7, 0xdc, 0x61, 0xc3, 0x54, 0xe8, 0xb5, 0x53,
            0xa7, 0x02, 0x5c, 0x97, 0x0f, 0xf9, 0x2d, 0xb2, 0x18, 0x97, 0x6e, 0xe4, 0xde, 0xbf,
            0x61, 0x55, 0xdc, 0xb3, 0x61, 0x42, 0xeb, 0x4d, 0x68, 0xe4, 0xdf, 0x0e, 0x57, 0xca,
            0x1d, 0xe0, 0x1b, 0x01, 0x0f, 0x4e, 0xaf, 0x1d, 0x72, 0xb5, 0x56, 0x56, 0xa1, 0x1d,
            0x47, 0x85, 0x11, 0x00, 0x7d, 0x52, 0x8f, 0x9d, 0x48, 0x81, 0xa1, 0x74, 0x20, 0xd4,
            0x40, 0xfe, 0x55, 0x01, 0x11, 0x51, 0x1e, 0x48, 0xf5, 0x49, 0x37, 0x45, 0x3c, 0x11,
            0xee, 0xe9, 0x77, 0x41, 0x68, 0xc2, 0x73, 0x1d, 0x69, 0xb9, 0xd9, 0xe5, 0xf9, 0xc7,
            0xa8, 0x6d, 0x30, 0xbb, 0x95, 0x8d, 0x49, 0x98, 0x9d, 0xb3, 0x6e, 0x86, 0x4b, 0x76,
            0xd8, 0xf6, 0xc4, 0x0a, 0x3f, 0x7a, 0x40, 0xc0, 0x2c, 0x40, 0x96, 0x8a, 0x96, 0x4c,
            0x75, 0xad, 0x40, 0xe9, 0x6d, 0xcb, 0x69, 0x20, 0xc4, 0x19, 0xbe, 0x55, 0x40, 0xb8,
            0x4f, 0xc8, 0x83, 0xc2, 0xcc, 0x1d, 0xac, 0x25, 0xd5, 0x17, 0xa5, 0xa9, 0x04, 0x62,
            0xc7, 0xb0, 0x99, 0x0b, 0xbd, 0x86, 0xf6, 0xb8, 0x9e, 0x7b, 0x4a, 0x0a, 0xe7, 0x57,
            0x78, 0x88, 0x08, 0x86, 0x72, 0x42, 0xad, 0x18, 0x66, 0x7a, 0xed, 0x0a, 0xb6, 0x05,
            0xa5, 0x5c, 0xf4, 0x77, 0x29, 0x47, 0x7b, 0x29, 0x0c, 0x91, 0x23, 0x44, 0x28, 0xd2,
            0x7d, 0x97, 0x93, 0xdf, 0x32, 0xfd, 0xed, 0x71, 0x3b, 0x0b, 0xbe, 0x45, 0xaa, 0xc2,
            0x67, 0x7b, 0x08, 0x6f, 0x25, 0x98, 0x96, 0xca, 0xa6, 0x03, 0xea, 0x6d, 0x39, 0x6b,
            0x95, 0xd4, 0x46, 0xda, 0x5e, 0x58, 0x42, 0xe1, 0x0c, 0x0d, 0xfd, 0x15, 0x66, 0x46,
            0xa5, 0x48, 0x85, 0xe6, 0x19, 0xd2, 0x11, 0x22, 0x90, 0x1f, 0xd0, 0x03, 0x27, 0xf0,
            0xe1, 0x50, 0x60, 0x9c, 0x1c, 0x85, 0xa5, 0xe4, 0x86, 0x14, 0x61, 0xab, 0xa6, 0x58,
            0xe0, 0xb0, 0x96, 0xb4, 0xda, 0x4b, 0x6e, 0xc7, 0x84, 0x3c, 0xa4, 0x0c, 0xfa, 0x8b,
            0x3d, 0x8c, 0x8e, 0x2b, 0xac, 0xf2, 0x11, 0x93, 0x22, 0xc9, 0xe3, 0x3c, 0xd1, 0xff,
            0x2f, 0x42, 0xba, 0x83, 0x32, 0xc8, 0xce, 0xa8, 0xa2, 0x9a, 0x19, 0x08, 0x01, 0xbd,
            0xce, 0xfb, 0x53, 0xd1, 0x07, 0x67, 0x3a, 0x1c, 0xff, 0x67, 0x64, 0x4a, 0x01, 0xa8,
            0x3c, 0xf9, 0x97, 0xdf, 0xa2, 0xf1, 0xe8, 0xb7, 0xd0, 0xea, 0x0b, 0x3d, 0x8b, 0x87,
            0x2c, 0xe4, 0x1c, 0xaf, 0x9d, 0xeb, 0x78, 0xa5, 0xf9, 0x0c, 0xa7, 0xec, 0x49, 0x4c,
            0x22, 0xfd, 0x5b, 0x62, 0x07, 0x08, 0x5f, 0x76, 0x7a, 0xae, 0xac, 0x8d, 0x36, 0xb8,
            0x0d, 0x80, 0x8d, 0x1e, 0x2a, 0x2f, 0x6d, 0x90, 0x53, 0xb3, 0x25, 0xb1, 0x0b, 0x9c,
            0x68, 0xbc, 0xad, 0xf1, 0xe9, 0x28, 0x95, 0xbb, 0x09, 0xdb, 0x68, 0x3c, 0xba, 0x2e,
            0x4c, 0x27, 0x30, 0xc3, 0x54, 0xea, 0x5d, 0xc5, 0xb2, 0x83, 0x53, 0xe1, 0x94, 0xf6,
            0xab, 0xbd, 0x3d, 0xd0, 0x1c, 0x31, 0x8e, 0x66, 0x40, 0xad, 0x56, 0x10, 0x5b, 0x61,
            0xc5, 0xe5, 0xaf, 0xbe, 0x27, 0x4f, 0xdf, 0xd4, 0x99, 0xe6, 0xbf, 0x4c, 0x4f, 0xd1,
            0x1e, 0x37, 0xb8, 0xa6, 0xea, 0x0a, 0x82, 0x8b, 0x4d, 0xb9, 0x34, 0x50, 0x39, 0x27,
            0xcb, 0xc4, 0xa6, 0xb6, 0x10, 0xe8, 0x0c, 0xb6, 0xe0, 0x4c, 0xde, 0xa7, 0xfe, 0x66,
            0xab, 0x60, 0x64, 0x20, 0x07, 0x02, 0xee, 0x04, 0x87, 0xc2, 0xd5, 0xc2, 0x99, 0xdb,
            0x5c, 0x5f, 0x91, 0x26, 0x97, 0x95, 0x1e, 0x79, 0x0f, 0x25, 0x93, 0x03, 0x65, 0xa8,
            0x6c, 0xa7, 0x72, 0x26, 0x33, 0x7b, 0xb8, 0x82, 0xff, 0xf1, 0xda, 0x66, 0xd6, 0xac,
            0x35, 0x8c, 0x6a, 0x74, 0x42, 0x8d, 0x39, 0x9d, 0x68, 0x8b, 0x81, 0xef, 0x5e, 0x5c,
            0x88, 0x00, 0xf1, 0x16, 0x46, 0x8b, 0x80, 0xad, 0xa5, 0x5a, 0x53, 0x5a, 0x54, 0x59,
            0xd3, 0x7e, 0x1e, 0xfb, 0x72, 0xdc, 0xb7, 0xfb, 0x35, 0xb2, 0x9d, 0x92, 0xce, 0xaa,
            0x10, 0x34, 0xc6, 0x2e, 0x78, 0x84, 0x4d, 0xe6, 0x16, 0xea, 0xef, 0x0b, 0x5c, 0xa6,
            0x8e, 0xba, 0xa1, 0x90, 0x16, 0x53, 0x26, 0x19, 0x68, 0xe4, 0xdf, 0x0e, 0x5a, 0x95,
            0xd0, 0x2d, 0xae, 0xc5, 0xdb, 0x96, 0x9a, 0x9e, 0x79, 0xa0, 0xa0, 0xbf, 0xc6, 0xc8,
            0x9a, 0xfa, 0x2a, 0x5a, 0x90, 0x4d, 0x2a, 0x5f, 0x05, 0x5a, 0x5c, 0x47, 0xdb, 0x5f,
            0x71, 0x36, 0x14, 0xc7, 0x49, 0xe2, 0x3b, 0x1c, 0xd9, 0x24, 0x28, 0x24, 0x5d, 0x24,
            0x69, 0x0f, 0xe1, 0xc4, 0xf7, 0xb0, 0x42, 0x6a, 0x6b, 0xf2, 0x39, 0x46, 0x5f, 0xf1,
            0x28, 0x3b, 0x9b, 0x2c, 0x97, 0x04, 0x5f, 0x72, 0x82, 0xce, 0x31, 0x3b, 0x5d, 0xc6,
            0xbb, 0xe6, 0x45, 0xec, 0x24, 0xd4, 0x7b, 0xdd, 0x26, 0x57, 0x4a, 0xce, 0x12, 0x8e,
            0x9d, 0x6d, 0x80, 0xc1, 0x3b, 0xbc, 0x8f, 0xf0, 0xfb, 0x25, 0xc5, 0x2f, 0xc8, 0x64,
            0xb9, 0x3a, 0xbd, 0x10, 0x73, 0xa0, 0x07, 0x65, 0x53, 0x8a, 0x62, 0x60, 0x93, 0xa8,
            0x6c, 0xe8, 0x69, 0x54, 0xb5, 0x01, 0x8b, 0x9a, 0xd0, 0xee, 0xdf, 0x58, 0xd2, 0x08,
            0xc6, 0x00, 0x50, 0x29, 0xb0, 0xb8, 0x81, 0xac, 0xab, 0xbf, 0x15, 0x76, 0x95, 0x15,
            0x12, 0xc1, 0xeb, 0x27, 0xd2, 0xc0, 0xed, 0x09, 0x58, 0xcc, 0xc3, 0x53, 0xfe, 0x53,
            0x27, 0x8f, 0x1c, 0x64, 0x84, 0xb8, 0x77, 0x54, 0x7a, 0x84, 0x73, 0x3f, 0x21, 0x97,
            0xab, 0xfc, 0xbe, 0xd1, 0xae, 0x22, 0xb5, 0xec, 0x58, 0x0d, 0x72, 0x5d, 0x8b, 0x3a,
            0x34, 0xa7, 0xdd, 0x7f, 0x01, 0x11, 0x2c, 0x25, 0x44, 0xc1, 0xa7, 0xc0, 0xbc, 0xbd,
            0xf6, 0xdf, 0xec, 0xff, 0x24, 0xa8, 0xe5, 0x56, 0x05, 0x78, 0x35, 0xb6, 0x72, 0x29,
            0xfa, 0xf4, 0xd8, 0x07, 0xa1, 0x64, 0x61, 0x91, 0xf0, 0x91, 0xd0, 0x61, 0xc7, 0x96,
            0xb0, 0x8f, 0x9a, 0x1b, 0xf1, 0xf2, 0xd0, 0x95, 0x38, 0x32, 0x66, 0x7d, 0x78, 0x04,
            0xcb, 0xcd, 0xa2, 0xd3, 0x67, 0x24, 0x37, 0xe0, 0x5c, 0x5a, 0xbc, 0xd7, 0xaf, 0x5e,
            0x9b, 0x35, 0x47, 0x5c, 0x84, 0x42, 0x23, 0xf1, 0x82, 0x4f, 0x0d, 0x97, 0x9e, 0xaa,
            0xa7, 0x2a, 0xeb, 0x9b, 0xac, 0x6e, 0x1c, 0xc3, 0x2d, 0x5c, 0xd0, 0x14, 0xfa, 0xcc,
            0xf4, 0xaa, 0x4e, 0x39, 0x55, 0x69, 0x65, 0x73, 0x7d, 0x4b, 0x3d, 0x84, 0xea, 0x0d,
            0xc0, 0xce, 0xfe, 0x2f, 0x39, 0xb5, 0x09, 0x4b, 0x69, 0x85, 0xf9, 0xc2, 0xc2, 0x9b,
            0x91, 0xa7, 0xc3, 0x76, 0x82, 0x2e, 0x34, 0x8f, 0x79, 0x44, 0xee, 0x3d, 0x26, 0x13,
            0x70, 0x05, 0x47, 0x88, 0x6e, 0xc5, 0x81, 0xcb, 0x33, 0x06, 0x19, 0xfd, 0xdd, 0xc3,
            0x38, 0x47, 0x78, 0x04, 0xdf, 0x92, 0xa2, 0xe6, 0xea, 0xbf, 0x39, 0xd4, 0x66, 0xb6,
            0x93, 0xd7, 0x4c, 0x55, 0xe9, 0x65, 0x64, 0xe3, 0x36, 0x6c, 0xcc, 0x78, 0x18, 0xe3,
            0x99, 0x7c, 0x80, 0x94, 0x7a, 0x8a, 0x3d, 0x85, 0x5c, 0x09, 0x22, 0x7c, 0x71, 0xa4,
            0x35, 0x6b, 0xfa, 0x02, 0x39, 0x1a, 0xaf, 0xcb, 0x5a, 0x2c, 0xae, 0xb9, 0x5d, 0x43,
            0x40, 0xee, 0x99, 0xc8, 0xbf, 0x62, 0x1f, 0x0c, 0x33, 0xf5, 0x86, 0xc3, 0x71, 0x5b,
            0x73, 0xb1, 0x23, 0xb4, 0x62, 0x42, 0xd6, 0x07, 0x43, 0x82, 0xb9, 0x39, 0xc4, 0x2c,
            0x6f, 0x73, 0x68, 0xd6, 0x6b, 0xe0, 0x42, 0x2f, 0x3a, 0x71, 0x91, 0x97, 0xd6, 0x0e,
            0x78, 0x97, 0x0d, 0x0b, 0x4d, 0x5d, 0xa7, 0x8f, 0xf4, 0x04, 0x8a, 0x1f, 0x98, 0xba,
            0x63, 0x96, 0xdb, 0x46, 0x4e, 0x9f, 0x32, 0x80, 0x1a, 0x38, 0x15, 0x48, 0x41, 0x4e,
            0x37, 0xde, 0xf4, 0x74, 0x08, 0x83, 0x70, 0x60, 0xe2, 0xd4, 0xe1, 0x33, 0x77, 0x65,
            0x6a, 0x2b, 0x6e, 0x14, 0x11, 0x7b, 0xfb, 0xa4, 0x8e, 0x97, 0x6c, 0x25, 0xeb, 0xc3,
            0xf7, 0x67, 0x21, 0x94, 0xef, 0xe1, 0x1d, 0xec, 0x8f, 0x86, 0x9e, 0x3f, 0x39, 0xb7,
            0xb2, 0x1d, 0xcf, 0xd6, 0xce, 0x90, 0x30, 0xd9, 0x84, 0xea, 0x44, 0x05, 0xe9, 0x05,
            0xca, 0x2f, 0xc7, 0xff, 0xb0, 0x00, 0x52, 0xbf, 0x64, 0x21, 0x58, 0xd4, 0x75, 0x7f,
            0xb4, 0x56, 0x4b, 0x53, 0x16, 0x34, 0xf6, 0x06, 0x85, 0x78, 0x93, 0x2c, 0x53, 0x64,
            0x2d, 0xeb, 0xbc, 0x11, 0xdd, 0x46, 0x54, 0xde, 0x3a, 0x2f, 0x79, 0x17, 0x66, 0x21,
            0x0c, 0xb5, 0x37, 0xbe, 0xac, 0xdd, 0x15, 0x67, 0xcd, 0x40, 0x93, 0xf3, 0x34, 0x93,
            0x5f, 0x8d, 0xe6, 0xac, 0x87, 0xe0, 0x53, 0xb0, 0x69, 0xa5, 0xd0, 0xc4, 0xfd, 0x21,
            0x7f, 0x3e, 0x0b, 0xeb, 0x1e, 0x80, 0x87, 0x8f, 0x9e, 0x3d, 0x73, 0x13, 0xfc, 0x3b,
            0xc2, 0x5f, 0xdc, 0xe3, 0x6e, 0x28, 0x8a, 0x10, 0x07, 0x21, 0x6e, 0xc0, 0x46, 0xc1,
            0x4a, 0x29, 0xa4, 0x78, 0x40, 0xbf, 0xc2, 0x85, 0x5c, 0xad, 0xba, 0x07, 0xec, 0x6a,
            0xd6, 0x84, 0x5a, 0xa8, 0xf8, 0x10, 0xc0, 0x6c, 0xd4, 0x26, 0x69, 0x14, 0x3c, 0x01,
            0x99, 0xf9, 0xee, 0x07, 0xc0, 0x97, 0x57, 0x4a, 0xda, 0x1d, 0xc5, 0x64, 0x40, 0xf8,
            0x05, 0x37, 0x7a, 0xc5, 0xad, 0x4d, 0xe6, 0x1b, 0x4b, 0x72, 0x44, 0x73, 0x61, 0xc9,
            0x91, 0x85, 0x53, 0xa2, 0x73, 0xeb, 0x41, 0x71, 0x42, 0xd8, 0x74, 0x09, 0x03, 0x46,
            0x70, 0x8d, 0xe8, 0x24, 0x41, 0x1b, 0x99, 0xb6, 0xb3, 0xcf, 0x71, 0xe9, 0x5d, 0x59,
            0xe2, 0x4f, 0xc3, 0xbb, 0x86, 0x45, 0xad, 0x62, 0x49, 0xbe, 0xed, 0xd2, 0x57, 0x15,
            0xba, 0xe5, 0xdf, 0xf2, 0xcf, 0x08, 0xf3, 0xfc, 0x08, 0x9a, 0x7b, 0xb6, 0x8f, 0x96,
            0x83, 0x36, 0xda, 0x94, 0xff, 0xc5, 0x3d, 0xd6, 0x27, 0xd0, 0xe7, 0xe6, 0xf1, 0xf9,
            0x5a, 0x6a, 0xce, 0x00, 0x77, 0x15, 0x09, 0x35, 0x66, 0xdf, 0xe0, 0xcd, 0x50, 0x3e,
            0xce, 0x22, 0x4d, 0x26, 0xaa, 0x01, 0x30, 0xc1, 0x23, 0x6d, 0x87, 0xe2, 0xee, 0x83,
            0xdc, 0x1c, 0xc3, 0x04, 0x8d, 0xb6, 0x21, 0x04, 0xfd, 0xb2, 0x1f, 0x3c, 0xb1, 0xb2,
            0x2c, 0x15, 0x85, 0x9a, 0xe4, 0xdf, 0x18, 0x47, 0x98, 0xb3, 0x50, 0x29, 0xa9, 0xf3,
            0x07, 0x5a, 0xbe, 0x33, 0xde, 0x72, 0x80, 0xb5, 0xc1, 0x6a, 0xbf, 0x89, 0xa6, 0x51,
            0x73, 0x83, 0x65, 0x0c, 0x81, 0x23, 0xd9, 0x8b, 0xe9, 0x11, 0xee, 0x0e, 0x7b, 0x21,
            0x05, 0xae, 0x91, 0x22, 0xee, 0xe2, 0x69, 0xf4, 0x48, 0xd4, 0x4a, 0xe7, 0xff, 0x4a,
            0xc7, 0xab, 0xfe, 0x3b, 0xf4, 0xfe, 0xc5, 0x97, 0x0a, 0xec, 0x74, 0xe7, 0x84, 0xb0,
            0xd5, 0x29, 0x3e, 0x79, 0x60, 0x02, 0x34, 0xd6, 0xfe, 0xca, 0xb6, 0x18, 0x81, 0x25,
            0x33, 0x8a, 0xfc, 0x85, 0x12, 0x2c, 0x9b, 0x80, 0x70, 0x7d, 0xfd, 0xe9, 0x4b, 0xb7,
            0x58, 0xa5, 0xbb, 0xec, 0x3c, 0x7f, 0x7f, 0xf1, 0x7e, 0xa3, 0xd1, 0xb9, 0xe3, 0xad,
            0x38, 0xd8, 0x1a, 0x54, 0x07, 0x48, 0x73, 0xc3, 0xfd, 0xc2, 0x91, 0x4d, 0x26, 0xe0,
            0x58, 0xda, 0x9c, 0xbe, 0xec, 0xf7, 0x45, 0x6f, 0xbf, 0x0e, 0x6b, 0x98, 0x7e, 0xcd,
            0x7d, 0x71, 0x5c, 0x11, 0x9f, 0x80, 0x12, 0xe8, 0x3c, 0x8f, 0x15, 0xa3, 0x03, 0x67,
            0x87, 0xca, 0xcb, 0xea, 0x5b, 0x33, 0xf7, 0x51, 0xd2, 0x2e, 0xee, 0x76, 0x5c, 0x38,
            0x26, 0xc4, 0x20, 0x4f, 0xbf, 0x66, 0x04, 0xce, 0xee, 0xeb, 0x37, 0x7a, 0xa2, 0x39,
            0x8b, 0x83, 0x22, 0xd5, 0xdb, 0x49, 0x16, 0x47, 0xdf, 0xdf, 0xfe, 0xc2, 0xba, 0xcd,
            0x6e, 0x3d, 0x05, 0x28, 0x8e, 0xf8, 0x34, 0x2c, 0x9a, 0x35, 0x7b, 0xd6, 0x4e, 0x54,
            0x97, 0xc8, 0xa8, 0x76, 0x6c, 0x72, 0x38, 0x74, 0x0f, 0x48, 0x23, 0xbc, 0xab, 0x42,
            0x4c, 0xe9, 0xb6, 0x1b, 0xeb, 0x5e, 0xae, 0xa3, 0x2c, 0x7c, 0x8d, 0x3e, 0x9f, 0x5e,
            0x91, 0x06, 0x2a, 0x04, 0x6f, 0x83, 0xad, 0x39, 0x88, 0x8e, 0x4d, 0x05, 0xc0, 0xb5,
            0x65, 0x88, 0x59, 0x13, 0x38, 0xcc, 0x91, 0xbb, 0xb0, 0xc1, 0x2c, 0xf3, 0x97, 0x4a,
            0x2f, 0xc7, 0x7d, 0xa2, 0xe0, 0xfc, 0x06, 0x16, 0xd1, 0xee, 0x45, 0x7a, 0x2f, 0xd1,
            0x5c, 0xd2, 0x14, 0xcd, 0x0f, 0xf7, 0x36, 0xe5, 0x99, 0x63, 0xfd, 0x8a, 0x6c, 0x5f,
            0x94, 0x2f, 0x58, 0x69, 0x78, 0x7f, 0x99, 0x68, 0x8d, 0x54, 0x67, 0xd0, 0xd2, 0xc6,
            0x47, 0x38, 0x30, 0x4e, 0xde, 0xf3, 0xe7, 0xc8, 0xe5, 0xeb, 0x0c, 0xf9, 0x4f, 0x42,
            0x7d, 0xf2, 0xab, 0x10, 0x1c, 0x21, 0x1b, 0x5a, 0x02, 0x70, 0xd9, 0x6d, 0x21, 0x91,
            0xb5, 0x11, 0xaa, 0x7d, 0x60, 0xc9, 0x7c, 0x3d, 0xa5, 0x4b, 0x60, 0xb6, 0x41, 0xa9,
            0x40, 0x3e, 0xa7, 0x8b, 0x26, 0xb0, 0x4f, 0xca, 0x3c, 0xfb, 0x21, 0xba, 0x39, 0xfe,
            0xbb, 0x61, 0x24, 0xca, 0xa2, 0x58, 0x7f, 0xdc, 0x24, 0xc6, 0x59, 0xb3, 0xad, 0x2e,
            0x0e, 0x41, 0x97, 0xa1, 0x9a, 0x6b, 0x9d, 0xd4, 0xaa, 0x69, 0x39, 0x33, 0x70, 0x78,
            0xa2, 0x90, 0x81, 0x25, 0x56, 0xb3, 0x23, 0x1e, 0x01, 0xc5, 0x0a, 0xd4, 0xc9, 0x0c,
            0xfd, 0xca, 0x2a, 0xa6, 0x49, 0x33, 0x6c, 0xbd, 0x51, 0x27, 0xa2, 0x89, 0x6c, 0x41,
            0xd8, 0xe3, 0xfd, 0xf9, 0xd0, 0xd5, 0x48, 0x17, 0x37, 0x29, 0xb9, 0x28, 0x5b, 0x78,
            0x48, 0x56, 0x3c, 0x19, 0xb3, 0xb9, 0xb7, 0x27, 0x7f, 0x22, 0x12, 0xd7, 0xbd, 0x20,
            0x4d, 0x5b, 0xc5, 0xb8, 0x94, 0x85, 0xe9, 0x17, 0xef, 0x47, 0x4d, 0xb3, 0xb9, 0xe6,
            0x87, 0xc1, 0x62, 0x34, 0x0c, 0x45, 0xb5, 0x85, 0xea, 0x33, 0x84, 0x43, 0x2b, 0xe0,
            0xa5, 0x34, 0xe0, 0x03, 0x2e, 0x92, 0xb3, 0xf4, 0xba, 0xad, 0xcd, 0xa8, 0x47, 0x4d,
            0x6a, 0x81, 0x70, 0x58, 0x54, 0xd3, 0x8e, 0x41, 0x7d, 0xcc, 0x63, 0xd7, 0xf7, 0x54,
            0x01, 0x5f, 0x8d, 0xe6, 0x84, 0xce, 0x4e, 0x1b, 0x17, 0x3f, 0x3b, 0x45, 0x36, 0x69,
            0x77, 0xa2, 0xea, 0xc6, 0x7e, 0x8f, 0x36, 0x33, 0x1c, 0xf0, 0xb0, 0x6e, 0x4d, 0x2b,
            0xbc, 0xac, 0x32, 0xde, 0xe5, 0xd8, 0x76, 0x29, 0x39, 0xf4, 0x4b, 0x8e, 0xba, 0x22,
            0xb2, 0x70, 0xda, 0xe1, 0xc6, 0x87, 0x92, 0x84, 0x9a, 0x01, 0xad, 0xee, 0xe8, 0xba,
            0x22, 0x74, 0x54, 0x33, 0x59, 0x0c, 0x35, 0x8e, 0x39, 0xbe, 0xa4, 0xaf, 0xd5, 0x80,
            0xd0, 0xec, 0xe6, 0x85, 0xe5, 0x7f, 0x00, 0x24, 0x5d, 0xd5, 0x19, 0xd7, 0x46, 0xe1,
            0x33, 0x43, 0x30, 0x09, 0xf7, 0xec, 0x36, 0xc9, 0x48, 0x73, 0xe2, 0xee, 0x94, 0xbc,
            0xb6, 0x35, 0x23, 0x39, 0x2b, 0x07, 0x44, 0x75, 0xd9, 0xa3, 0xe0, 0x23, 0x2d, 0xb6,
            0x12, 0xed, 0x47, 0x34, 0x1b, 0x5c, 0x07, 0xfa, 0x74, 0xe3, 0x4d, 0x09, 0xe3, 0x1f,
            0x4c, 0x99, 0xbe, 0x3e, 0xd8, 0x53, 0xaa, 0x75, 0xd6, 0x6d, 0xeb, 0x6d, 0x0b, 0x94,
            0x26, 0xc9, 0x7b, 0x35, 0xea, 0x7a, 0xde, 0x50, 0x10, 0xc2, 0x31, 0x9f, 0x83, 0x60,
            0xcc, 0xde, 0x42, 0x00, 0x66, 0x47, 0xf5, 0xbc, 0x39, 0x57, 0xb9, 0x56, 0x10, 0x86,
            0x18, 0x0f, 0x89, 0xe3, 0x22, 0xbc, 0x40, 0x3e, 0x67, 0xb4, 0x82, 0x36, 0xa5, 0x33,
            0x80, 0xeb, 0xd5, 0xe7, 0xf5, 0x49, 0x04, 0x36, 0xdf, 0xb4, 0x54, 0xc9, 0x73, 0x29,
            0x6e, 0x95, 0x41, 0x5f, 0x1c, 0x93, 0xdc, 0x8c, 0xb1, 0xb8, 0x0b, 0xfb, 0x60, 0xb3,
            0x43, 0x3f, 0x18, 0xbe, 0x12, 0x35, 0x4c, 0x2b, 0x9d, 0x41, 0xfb, 0x62, 0xab, 0x89,
            0xc4, 0xe4, 0x4b, 0xf3, 0x01, 0xba, 0xc8, 0x3d, 0xe6, 0x60, 0xa5, 0xf9, 0xbb, 0x03,
            0x87, 0x27, 0xce, 0x46, 0x07, 0x62, 0x7d, 0x35, 0x31, 0x76, 0x91, 0x79, 0x62, 0x87,
            0xa5, 0xb6, 0xce, 0x0a, 0xbd, 0xc8, 0x54, 0xea, 0x1a, 0x34, 0x7c, 0xf0, 0xd0, 0x0a,
            0x4e, 0x7f, 0x3a, 0x5e, 0x17, 0xcc, 0x95, 0x9c, 0x17, 0x79, 0x63, 0x78, 0xd0, 0x20,
            0xcf, 0xbe, 0xc7, 0x0d, 0x60, 0xf9, 0x06, 0x8f, 0xae, 0x3e, 0xc0, 0xcd, 0x5e, 0x3c,
            0x6e, 0x45, 0x0f, 0xd4, 0x0f, 0x75, 0x51, 0xc8, 0xc3, 0x9a, 0x37, 0xbe, 0xd3, 0xcf,
            0x44, 0x97, 0xc6, 0x5a, 0x26, 0x9b, 0xa6, 0x2b, 0x40, 0x56, 0xdd, 0xc8, 0xbc, 0xc6,
            0x21, 0xa6, 0x2e, 0xfe, 0xb8, 0xc3, 0x35, 0x72, 0xac, 0xb1, 0x63, 0xfa, 0xd6, 0xa6,
            0x5b, 0xe3, 0xcb, 0xf8, 0xb2, 0x7f, 0x1f, 0x5d, 0x4a, 0x58, 0x64, 0x40, 0x8c, 0x73,
            0x4b, 0xc0, 0xa7, 0x65, 0x24, 0xde, 0x5a, 0x85, 0xdf, 0xf9, 0xca, 0x69, 0xa9, 0xf5,
            0x90, 0x0b, 0xad, 0x8e, 0x44, 0x83, 0x2a, 0xbe, 0xb2, 0xb8, 0x00, 0x20, 0xfe, 0x9d,
            0x8a, 0xe5, 0x1f, 0x0b, 0x67, 0x1b, 0x66, 0x54, 0xb3, 0xa3, 0xac, 0x74, 0x91, 0xdc,
            0xf4, 0x2e, 0xe1, 0x1c, 0xdb, 0x82, 0x47, 0x04, 0x8f, 0x6f, 0x2e, 0x86, 0xa1, 0x27,
            0xed, 0x3d, 0xf2, 0x82, 0x0b, 0x2b, 0x42, 0x35, 0xc9, 0xbf, 0x64, 0x05, 0x83, 0x4b,
            0x49, 0x57, 0xcf, 0xec, 0x2e, 0x8c, 0x9d, 0xb8, 0x5b, 0x14, 0x33, 0x1a, 0x1c, 0xb8,
            0xcd, 0x6e, 0xd0, 0xe0, 0xfc, 0xb0, 0x35, 0x88, 0x7f, 0x00, 0x67, 0x9c, 0x78, 0x40,
            0x74, 0xfc, 0xf3, 0x20, 0x10, 0x15, 0xc6, 0x04, 0xba, 0x94, 0x15, 0xd5, 0x28, 0x37,
            0xf2, 0xd1, 0x22, 0x39, 0xb3, 0x2e, 0x71, 0x1d, 0x2e, 0x80, 0x3d, 0xf6, 0x2d, 0x4a,
            0xdb, 0xd1, 0x5a, 0x99, 0xf0, 0x77, 0x71, 0x1b, 0x8a, 0xf4, 0x54, 0xe0, 0xc0, 0x45,
            0x6c, 0xbc, 0x29, 0x9d, 0xe8, 0x0e, 0x94, 0xeb, 0x74, 0xdf, 0x7f, 0xbd, 0x2f, 0x44,
            0x8e, 0xa9, 0xea, 0x72, 0xd6, 0x9d, 0x9e, 0x4b, 0x9b, 0x0d, 0x51, 0xa2, 0xa6, 0x4b,
            0x3b, 0x82, 0xb5, 0xa4, 0x76, 0x5f, 0xe3, 0x43, 0x44, 0x64, 0x88, 0x20, 0x88, 0x96,
            0x12, 0xa2, 0x61, 0x70, 0xa1, 0x50, 0x51, 0xde, 0x1b, 0x6c, 0x54, 0x4e, 0x79, 0xe6,
            0x23, 0xec, 0x55, 0xee, 0x6a, 0x0c, 0xf2, 0x48, 0x7a, 0x4c, 0x3d, 0xc3, 0x47, 0x2a,
            0x37, 0x9e, 0x07, 0xcd, 0x1c, 0xc3, 0x87, 0x80, 0x98, 0x2a, 0x83, 0x22, 0xd8, 0x34,
            0x24, 0x77, 0xab, 0x34, 0x74, 0x6c, 0xb5, 0x9a, 0x55, 0x9b, 0xb8, 0x28, 0x38, 0x30,
            0xc6, 0xcc, 0x04, 0x77, 0x67, 0x05, 0x1e, 0xcf, 0x69, 0x22, 0xd0, 0x38, 0x08, 0x53,
            0x02, 0x67, 0xa3, 0x79, 0xac, 0xf6, 0x76, 0x04, 0xa8, 0x6d, 0x85, 0xbc, 0x5d, 0x9f,
            0x4e, 0x91, 0x82, 0x68, 0x9c, 0xc3, 0xa5, 0x65, 0x78, 0x43, 0xe0, 0xfe, 0xe2, 0x1e,
            0xa3, 0xe6, 0xde, 0xb9, 0x52, 0x24, 0x06, 0xb0, 0x31, 0xea, 0xfe, 0x38, 0xe9, 0x4f,
            0x9b, 0x8c, 0x7a, 0x40, 0xaa, 0xc6, 0xee, 0xc6, 0x57, 0x3d, 0x39, 0xfc, 0xd3, 0xef,
            0xee, 0x6e, 0x40, 0xb9, 0xb9, 0x4d, 0xca, 0xd4, 0x40, 0xcf, 0x0a, 0xae, 0x11, 0x74,
            0xa9, 0xe7, 0x48, 0x39, 0x47, 0x0c, 0x71, 0x2d, 0x6e, 0x2a, 0x04, 0x8f, 0xb6, 0xf4,
            0xac, 0xb0, 0x0a, 0xd9, 0xc4, 0x36, 0x7d, 0x49, 0x2f, 0x23, 0x50, 0x93, 0x9e, 0xef,
            0x05, 0x07, 0x5a, 0xd1, 0xdc, 0xe7, 0x1c, 0xed, 0xa4, 0x92, 0x5a, 0xc3, 0xe2, 0xa7,
            0xed, 0x17, 0x2b, 0x90, 0x6a, 0x0c, 0x4c, 0x27, 0x41, 0x64, 0x28, 0x28, 0xb4, 0x07,
            0xd9, 0x3d, 0xe6, 0xf0, 0x45, 0xe7, 0x19, 0x14, 0xb4, 0xcc, 0x1c, 0xf7, 0x24, 0x8f,
            0x8d, 0xf5, 0xe0, 0xa2, 0x11, 0xe1, 0xf4, 0xc9, 0xef, 0x5c, 0x61, 0x85, 0xf8, 0x2c,
            0x42, 0x39, 0xbf, 0x24, 0x7d, 0xcd, 0x4b, 0x36, 0x53, 0x15, 0x30, 0x73, 0x76, 0xd8,
            0xdf, 0x94, 0x25, 0x53, 0x65, 0x01, 0x4c, 0x23, 0x1e, 0x97, 0x23, 0xf4, 0x0f, 0xfd,
            0x87, 0x4a, 0x94, 0xc4, 0xb5, 0xea, 0x29, 0xd5, 0xc5, 0x58, 0x7c, 0xd7, 0x39, 0x17,
            0xac, 0xac, 0x94, 0xda, 0x2c, 0xaa, 0xda, 0x9f, 0xfa, 0x56, 0xec, 0xe5, 0xb8, 0x82,
            0x5d, 0x5d, 0x67, 0xea, 0x5a, 0x66, 0xa7, 0x10, 0xef, 0x03, 0x2f, 0xe0, 0x4f, 0x14,
            0x42, 0x7a, 0x40, 0xb2, 0xcd, 0x26, 0x80, 0x51, 0x85, 0xa7, 0x45, 0xfd, 0x18, 0x48,
            0xf2, 0x9c, 0x3e, 0xab, 0x9f, 0xef, 0x9b, 0x16, 0xf0, 0x2e, 0xeb, 0x68, 0x8e, 0xf5,
            0xc8, 0xf6, 0xbe, 0x2d, 0x86, 0xf1, 0x48, 0xde, 0x8f, 0xc0, 0x8a, 0x4d, 0xc6, 0x26,
            0x61, 0xb2, 0x1c, 0xda, 0xca, 0xdb, 0x78, 0x08, 0xbe, 0xe9, 0xd2, 0x76, 0xac, 0xb6,
            0xfe, 0x5d, 0x90, 0xb0, 0x64, 0x5f, 0xac, 0x2f, 0x2f, 0xf8, 0x19, 0xc8, 0xb6, 0xa7,
            0xec, 0xd8, 0x2e, 0x67, 0x6d, 0x01, 0xd1, 0x28, 0xc0, 0x86, 0xc1, 0x36, 0x65, 0xe8,
            0xff, 0xfe, 0xad, 0x37, 0xbf, 0x2e, 0x2a, 0xe7, 0x9a, 0xba, 0xe2, 0xa5, 0xa0, 0xca,
            0xe9, 0x9f, 0x3b, 0x98, 0x49, 0x6a, 0x31, 0x61, 0xfb, 0x0f, 0x8b, 0x3e, 0x3c, 0x4f,
            0xf1, 0xb4, 0x1f, 0x59, 0xdb, 0x83, 0x54, 0x35, 0x70, 0x51, 0x8b, 0x00, 0x57, 0x75,
            0x6f, 0xbc, 0x03, 0xe6, 0x64, 0x91, 0x18, 0x1e, 0x59, 0xdd, 0x93, 0x44, 0x64, 0xd7,
            0x52, 0xe0, 0xad, 0xd4, 0xbd, 0x74, 0x45, 0xd9, 0x03, 0x6a, 0x9e, 0x3d, 0xf9, 0x1c,
            0x48, 0x2b, 0x01, 0xf0, 0x5a, 0x9f, 0xf2, 0xc6, 0x3e, 0x02, 0x08, 0xc1, 0x6c, 0xb0,
            0x9d, 0xce, 0x9a, 0x24, 0x8b, 0x00, 0x60, 0xc7, 0xa5, 0x60, 0x86, 0xb5, 0x92, 0xdb,
            0xec, 0xa5, 0xa8, 0xdc, 0x55, 0xf7, 0x2a, 0xb7, 0xa7, 0x5a, 0x56, 0xb0, 0x02, 0x55,
            0xeb, 0x3b, 0x3c, 0xa6, 0xc3, 0x21, 0xfe, 0xc6, 0x87, 0x3d, 0x62, 0xf9, 0x9f, 0x4a,
            0xe5, 0x56, 0x73, 0xb8, 0xfc, 0x3a, 0xd8, 0x35, 0x75, 0x16, 0x4a, 0x25, 0x2e, 0xc2,
            0x2b, 0xfb, 0x84, 0x6d, 0x48, 0x64, 0x64, 0x6a, 0xe1, 0x20, 0xa9, 0xe3, 0x56, 0xef,
            0xde, 0xa7, 0xc1, 0x10, 0x01, 0x42, 0x68, 0xb9, 0x7f, 0xdc, 0x38, 0x2f, 0xb1, 0x0b,
            0x18, 0x85, 0x77, 0x0a, 0xd4, 0x5e, 0x8c, 0x73, 0x73, 0xc4, 0x08, 0x70, 0x5c, 0x21,
            0xf5, 0xba, 0xec, 0x00, 0x68, 0x59, 0x7d, 0x2b, 0x31, 0x83, 0x22, 0x5a, 0xf5, 0x0c,
            0x82, 0x2f, 0x43, 0x0a, 0xc7, 0x64, 0xd6, 0xff, 0x95, 0x2e, 0x64, 0x87, 0xf2, 0x18,
            0x71, 0x32, 0xef, 0x9e, 0xed, 0x0c, 0xc3, 0xbd, 0x5a, 0x34, 0x07, 0xe6, 0xa9, 0xdf,
            0x2d, 0x22, 0x03, 0x59, 0x5c, 0xfe, 0xcd, 0xa7, 0x54, 0x71, 0xd2, 0xd2, 0x58, 0xad,
            0xb7, 0x78, 0x83, 0x0b, 0x6e, 0x15, 0x34, 0xe3, 0x8e, 0x55, 0x5c, 0x18, 0xaf, 0xdd,
            0x50, 0x06, 0x3d, 0x0e, 0x4a, 0xc2, 0x3d, 0x20, 0xd7, 0x04, 0xd6, 0xa6, 0xcb, 0xc3,
            0xd9, 0xf7, 0x4a, 0x78, 0xcf, 0xd6, 0xb9, 0x07, 0xf0, 0x36, 0x88, 0x3c, 0x5e, 0x1e,
            0x93, 0x72, 0xee, 0xd6, 0x0a, 0x46, 0x7d, 0x1e, 0xde, 0xde, 0xa0, 0x75, 0xb4, 0xfb,
            0xab, 0xda, 0x60, 0xe7, 0x36, 0xfe, 0xd3, 0x90, 0x26, 0xee, 0x8b, 0xc5, 0xe2, 0xde,
            0x2a, 0x4a, 0x9c, 0x86, 0x7f, 0x83, 0x1c, 0xa4, 0x85, 0x41, 0x6c, 0xab, 0x29, 0x21,
            0xa3, 0xf2, 0xa6, 0x95, 0x3e, 0x5a, 0x9d, 0x74, 0xd5, 0x2a, 0xe1, 0xa6, 0x3c, 0x69,
            0x88, 0xd0, 0x58, 0xc1, 0xfe, 0xac, 0xf6, 0x56, 0x7d, 0xa6, 0xa1, 0x73, 0x0b, 0xbc,
            0x80, 0xf3, 0x4c, 0xa1, 0x91, 0xcc, 0xdf, 0x27, 0x91, 0xbb, 0x7a, 0x3f, 0xb4, 0x2a,
            0xf1, 0xa6, 0xec, 0xc5, 0xb1, 0x51, 0x7c, 0x8d, 0xd3, 0x5a, 0xe5, 0x95, 0xc4, 0x68,
            0xa3, 0xb3, 0xe0, 0xaf, 0x69, 0xfd, 0xee, 0x96, 0x48, 0x95, 0x74, 0x78, 0x37, 0xa2,
            0xa7, 0xcb, 0x79, 0xe4, 0x94, 0x26, 0x41, 0x84, 0x1b, 0xfa, 0x87, 0x57, 0x77, 0x1d,
            0xd6, 0x0b, 0x8d, 0x60, 0xa0, 0x94, 0x0b, 0x33, 0x0c, 0x45, 0x8c, 0x7c, 0xc5, 0xeb,
            0x6f, 0x5b, 0xf1, 0x3a, 0x2a, 0x22, 0xb2, 0xbf, 0x63, 0xf0, 0x9e, 0x29, 0xd7, 0xbd,
            0x18, 0x1e, 0x09, 0x2c, 0x92, 0x84, 0x61, 0x79, 0xfa, 0xf3, 0x38, 0xca, 0x04, 0xae,
            0x9c, 0xa2, 0xfd, 0x4e, 0x14, 0xf5, 0xa1, 0x40, 0x50, 0x12, 0xad, 0xc7, 0x81, 0x8a,
            0x27, 0xa5, 0xe9, 0x24, 0x48, 0xa2, 0xce, 0x45, 0x28, 0x3e, 0xed, 0x9c, 0xc1, 0x29,
            0x93, 0xff, 0xd9, 0x06, 0x90, 0x37, 0x02, 0x4f, 0x1b, 0x9b, 0xf7, 0xad, 0x7d, 0xfc,
            0x48, 0xb4, 0x0c, 0x53, 0x02, 0xb7, 0x59, 0xfe, 0x02, 0x0f, 0x49, 0xc8, 0x55, 0x4d,
            0x22, 0xe1, 0xdb, 0xab, 0xb3, 0x26, 0x54, 0xbb, 0xd5, 0xcd, 0x55, 0xc6, 0x6c, 0x76,
            0x7c, 0x64, 0xc0, 0xc9, 0x87, 0x67, 0xef, 0xf2, 0x8e, 0x46, 0x2f, 0x48, 0xed, 0xf0,
            0x10, 0xee, 0x7f, 0x92, 0x0c, 0xd9, 0x0e, 0x9f, 0xf3, 0x58, 0x22, 0x87, 0x2c, 0x76,
            0x4b, 0xaa, 0xca, 0x6a, 0xb6, 0xb0, 0xad, 0xcf, 0xd9, 0x7e, 0x65, 0x25, 0xe3, 0xb6,
            0xea, 0x42, 0x0b, 0xec, 0xde, 0x7b, 0x3a, 0xaa, 0x75, 0xe2, 0x97, 0xf9, 0x98, 0x79,
            0x43, 0x0a, 0xfc, 0x0e, 0x69, 0xdd, 0x44, 0x8e, 0x9c, 0xd4, 0xe6, 0xca, 0x1e, 0xbf,
            0x2c, 0x18, 0x14, 0xd6, 0xac, 0x43, 0x35, 0x88, 0xe7, 0xeb, 0x5f, 0xdb, 0xdb, 0x80,
            0xbf, 0x42, 0xb8, 0xac, 0x0c, 0xb8, 0x39, 0xd3, 0xbf, 0xf6, 0xe6, 0x67, 0xb1, 0xfc,
            0xe0, 0xaf, 0xba, 0x45, 0xb2, 0xbb, 0x8b, 0x79, 0xba, 0xdb, 0x6e, 0x7d, 0x09, 0xec,
            0xf6, 0xed, 0x4f, 0x06, 0x33, 0x06, 0x9a, 0xd7, 0x81, 0xed, 0xec, 0x33, 0x7f, 0x9a,
            0x65, 0x52, 0x98, 0xcb, 0x5e, 0x2f, 0x6b, 0xe0, 0x02, 0x4c, 0x44, 0x90, 0x92, 0x96,
            0x4e, 0x21, 0x6b, 0x11, 0x00, 0x8e, 0x06, 0xe3, 0xb6, 0x37, 0x35, 0x22, 0x8d, 0x7a,
            0xb2, 0x4e, 0x55, 0xd5, 0x50, 0xe9, 0x31, 0xd8, 0x4b, 0xfd, 0xde, 0x7e, 0x40, 0x9f,
            0xb1, 0x2e, 0x48, 0xfd, 0x03, 0xf6, 0x35, 0x6b, 0xfd, 0xb0, 0xb5, 0xa4, 0xe4, 0x75,
            0x6c, 0xae, 0xd2, 0xec, 0x77, 0x13, 0x39, 0xf5, 0x96, 0xd8, 0x24, 0x06, 0xde, 0x79,
            0xe0, 0x60, 0x40, 0x81, 0x3b, 0x05, 0x2d, 0x5a, 0x6b, 0x90, 0x3f, 0x7e, 0x66, 0x9f,
            0xe3, 0x3f, 0x86, 0x1b, 0xb0, 0xd8, 0xc9, 0xa5, 0x27, 0x62, 0x40, 0xe6, 0xac, 0xf8,
            0xb4, 0x0a, 0xcc, 0x08, 0x0c, 0x05, 0x1f, 0xe3, 0x89, 0x91, 0x9e, 0x71, 0xcb, 0x6d,
            0x81, 0x3b, 0xbc, 0xd4, 0x55, 0xe6, 0x69, 0x5b, 0xc6, 0xb4, 0x8c, 0x6d, 0xc3, 0xf3,
            0xa3, 0xbd, 0xe2, 0xc4, 0x56, 0x3d, 0x36, 0x39, 0xa4, 0x6d, 0x0d, 0x58, 0x02, 0x90,
            0xe9, 0x13, 0x04, 0x4a, 0x62, 0xa8, 0x53, 0x89, 0xae, 0x3c, 0xd3, 0x2e, 0xc9, 0xfa,
            0x48, 0x6a, 0xdf, 0x08, 0x82, 0xfd, 0x79, 0xb1, 0x53, 0x6f, 0x4c, 0x68, 0x48, 0xea,
            0x30, 0x41, 0x95, 0x22, 0x93, 0x17, 0x32, 0x72, 0xfd, 0x7a, 0x61, 0x4f, 0x22, 0x13,
            0xfd, 0xbc, 0x89, 0xaf, 0xca, 0xfc, 0x5d, 0xfa, 0x17, 0xdd, 0x8a, 0xe5, 0xe8, 0x35,
            0xc5, 0x11, 0xfe, 0xe9, 0x10, 0xa5, 0x0b, 0xf1, 0x90, 0x97, 0x8b, 0x07, 0xc8, 0x90,
            0x2a, 0x27, 0xec, 0xce, 0x2f, 0x23, 0x84, 0x4c, 0x5d, 0x82, 0x83, 0xa7, 0x3b, 0x87,
            0xfd, 0x5d, 0x91, 0x01, 0x2e, 0x34, 0xcd, 0xac, 0x86, 0x03, 0x9f, 0x6f, 0x3e, 0x9b,
            0xc4, 0x94, 0xfe, 0x0a, 0xe3, 0xfc, 0x46, 0x7a, 0xbe, 0xea, 0xb5, 0xda, 0x6c, 0xe9,
            0x1a, 0x29, 0x0e, 0x70, 0x61, 0x64, 0xc7, 0xf4, 0xae, 0xc0, 0x63, 0xb8, 0xa8, 0x49,
            0xab, 0x36, 0xfa, 0xfc, 0xe3, 0xcf, 0x5e, 0x2d, 0x55, 0x1a, 0x10, 0xda, 0x11, 0xbf,
            0xa7, 0x8f, 0x88, 0xf0, 0xb4, 0x38, 0x31, 0x1c, 0x0e, 0xc0, 0x3c, 0x39, 0x7b, 0x43,
            0x92, 0xd5, 0x25, 0x40, 0x98, 0xd4, 0x46, 0x8e, 0xae, 0xb6, 0x69, 0x71, 0xb4, 0xb5,
            0xf0, 0x9e, 0x77, 0x62, 0x39, 0x26, 0xd8, 0xfc, 0x6e, 0xcc, 0xbb, 0x5f, 0x09, 0x39,
            0x13, 0x86, 0xb9, 0xdf, 0x86, 0xc9, 0xee, 0xc1, 0xe0, 0xb5, 0x58, 0x2d, 0x10, 0x0f,
            0xfc, 0x8d, 0x34, 0x6e, 0x2b, 0x1a, 0x1e, 0x7d, 0x9f, 0x40, 0x39, 0x5c, 0xc0, 0x39,
            0xc7, 0x00, 0xfa, 0x71, 0x8b, 0x79, 0x0f, 0xee, 0xfe, 0xd6, 0x20, 0xa6, 0xdc, 0x63,
            0xc8, 0xea, 0x8e, 0x45, 0xbe, 0xcd, 0x7e, 0x34, 0xcd, 0x03, 0xf6, 0x35, 0x6b, 0x42,
            0x2d, 0x17, 0x9c, 0x85, 0x9a, 0x38, 0xa4, 0x93, 0x36, 0x96, 0x36, 0x17, 0xb8, 0x97,
            0xb0, 0xdd, 0x63, 0xfc, 0xba, 0x4b, 0x92, 0xd6, 0xaa, 0xa4, 0xfe, 0x65, 0xcc, 0xdc,
            0x18, 0x61, 0xa6, 0xdc, 0x1e, 0x30, 0x30, 0x3c, 0xd7, 0x7d, 0xd4, 0x2a, 0x61, 0x7a,
            0x6a, 0x92, 0x71, 0x51, 0xb4, 0x80, 0xc9, 0x39, 0x33, 0xf1, 0x29, 0x34, 0x5f, 0x29,
            0xc6, 0x5c, 0xac, 0x18, 0x32, 0x59, 0x45, 0x35, 0x40, 0x3c, 0x75, 0x49, 0x78, 0xa5,
            0x11, 0x22, 0x64, 0x33, 0x12, 0xb5, 0xd3, 0x5f, 0xb6, 0x03, 0xdc, 0xe7, 0x83, 0x74,
            0xf6, 0x3c, 0x8d, 0xa8, 0x93, 0x97, 0x84, 0xfa, 0x67, 0x42, 0x92, 0x27, 0xca, 0xfd,
            0x63, 0x2a, 0x19, 0x47, 0x6d, 0xd9, 0x5d, 0x72, 0xfe, 0xb3, 0x80, 0xfc, 0x39, 0xa2,
            0xc3, 0xf9, 0x50, 0xbe, 0x6d, 0x54, 0x91, 0x25, 0xe6, 0xfb, 0x92, 0x88, 0x3f, 0xc8,
            0x8d, 0x18, 0x40, 0x26, 0xf0, 0xb3, 0x9f, 0x38, 0x22, 0x52, 0x07, 0xd5, 0xa7, 0x97,
            0xd5, 0xd1, 0xf7, 0xe4, 0x3a, 0x98, 0xd3, 0xa3, 0x0e, 0xee, 0x3d, 0xf8, 0xb5, 0x96,
            0xf6, 0x1d, 0x36, 0x98, 0x0e, 0x7b, 0x91, 0xdb, 0x09, 0x3c, 0x27, 0xb8, 0xba, 0x1b,
            0x38, 0x23, 0xf9, 0x00, 0x02, 0x12, 0x7c, 0x27, 0xf9, 0x51, 0xe7, 0x90, 0xe7, 0x40,
            0x08, 0xa1, 0x2c, 0xdc, 0x13, 0xe1, 0x9e, 0xd0, 0xa6, 0x05, 0x80, 0x9b, 0x3b, 0x83,
            0x39, 0xe2, 0xcb, 0xb3, 0xe0, 0x47, 0x76, 0x1e, 0xab, 0xea, 0x2a, 0xfd, 0x41, 0x9a,
            0xaa, 0xd4, 0xd5, 0xb4, 0xd6, 0x46, 0xe2, 0x8b, 0x74, 0x89, 0x80, 0xfb, 0x2e, 0xdf,
            0x94, 0xd0, 0x06, 0xc3, 0x21, 0x0c, 0x56, 0x17, 0xf6, 0x83, 0xa8, 0xde, 0xa5, 0xc9,
            0x60, 0x0a, 0xf6, 0xa3, 0xf6, 0x31, 0x08, 0x5b, 0x85, 0x77, 0x31, 0x65, 0xed, 0x3b,
            0x3c, 0x08, 0x89, 0x29, 0x58, 0x4c, 0xf0, 0x91, 0x54, 0x8f, 0x3f, 0x10, 0x2f, 0x58,
            0x41, 0xfe, 0xdb, 0xa4, 0x0c, 0x7c, 0x30, 0x4b, 0x97, 0xdb, 0x4c, 0x84, 0x45, 0xf1,
            0x8f, 0xa0, 0x46, 0xbe, 0xc4, 0x79, 0x91, 0xf0, 0x08, 0x75, 0x6c, 0xb3, 0x84, 0x72,
            0x5d, 0x0c, 0xd1, 0x46, 0x41, 0x41, 0x58, 0x3a, 0x18, 0xea, 0xa5, 0xa6, 0xe6, 0xa6,
            0xfe, 0xdb, 0x94, 0xfc, 0xcf, 0xaf, 0x2f, 0x7c, 0x4b, 0x36, 0x6c, 0x51, 0x25, 0x0b,
            0x5c, 0xb6, 0x79, 0x25, 0x89, 0x64, 0x1f, 0xe1, 0x0c, 0x0d, 0xfe, 0xe9, 0x32, 0x78,
            0xbc, 0x3d, 0x26, 0xf6, 0x60, 0xca, 0xce, 0x3a, 0x32, 0x93, 0x08, 0x3f, 0x53, 0x8a,
            0x60, 0x8c, 0x73, 0x32, 0x11, 0x1a, 0xda, 0xda, 0x9f, 0x03, 0x4a, 0x32, 0xf7, 0x0d,
            0x36, 0x85, 0x64, 0x87, 0x20, 0x34, 0x7d, 0x37, 0x87, 0xa3, 0x94, 0x7f, 0x29, 0x74,
            0x5c, 0xf4, 0x8f, 0xa5, 0x5e, 0xde, 0x02, 0x1b, 0xcb, 0x80, 0x9c, 0xe1, 0x29, 0x81,
            0xa9, 0xc6, 0xe1, 0x16, 0xcd, 0xfd, 0xc6, 0x42, 0x8d, 0x44, 0xe4, 0x44, 0x76, 0xf1,
            0xc7, 0xdd, 0xfe, 0x93, 0x4d, 0x71, 0x55, 0x13, 0xa6, 0x1f, 0xba, 0xff, 0x3a, 0x28,
            0x7d, 0x12, 0xa1, 0xb0, 0xba, 0x40, 0x28, 0xd8, 0x11, 0x28, 0x7d, 0x55, 0x75, 0x19,
            0x1b, 0xcd, 0x26, 0x22, 0x0d, 0x8d, 0x34, 0x6a, 0xca, 0x89, 0x60, 0xc3, 0x84, 0x64,
            0xbb, 0xa8, 0x71, 0x1b, 0x60,
        ];
        let params = CodecParams::video(CodecId::Av1, 512, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 512 * 64 * 3 / 2);
        assert_eq!(packed[0], 129);
        assert_eq!(packed[8 * 512 + 60], 221);
        assert_eq!(frame_digest(&packed), 0x00a4_13e5_8cce_fe0e);
    }

    #[test]
    fn film_grain_matches_dav1d() {
        // 64×64 flat keyframe. Luma grain comes from a table; chroma grain is off.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0x80, 0x18, 0x32, 0x4b, 0x92, 0x80,
            0x00, 0x00, 0x54, 0xc0, 0xe4, 0x80, 0x0a, 0x3f, 0xca, 0x00, 0x00, 0x00, 0xdd, 0x34,
            0x64, 0x6d, 0x44, 0x2f, 0x18, 0xf2, 0xb3, 0x7b, 0x13, 0x59, 0xe2, 0xbc, 0x54, 0x04,
            0x9c, 0xdb, 0xf4, 0x91, 0x0b, 0xc6, 0x3c, 0xab, 0x93, 0x2a, 0xc2, 0x46, 0xd3, 0xb3,
            0xf3, 0xc5, 0x02, 0x68, 0x9b, 0x05, 0x50, 0xb9, 0x4f, 0x0c, 0x69, 0xad, 0xcd, 0xd7,
            0x9a, 0xd5, 0xba, 0x22, 0x90, 0xc9, 0x69, 0xcf, 0x6f, 0xd2, 0x5a, 0x52, 0xc9, 0x89,
            0x9f, 0xdb, 0xa9,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 125);
        assert_eq!(frame_digest(&packed), 0x0378_f39d_a086_7aec);
    }

    #[test]
    fn film_grain_with_chroma_and_overlap_matches_dav1d() {
        // Same flat keyframe. Luma and chroma grain use lag-1 coefficients and overlap.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0x80, 0x18, 0x32, 0x63, 0x92, 0x80,
            0x00, 0x00, 0x54, 0x01, 0x8c, 0x80, 0x05, 0x3f, 0xcf, 0x03, 0x00, 0x3c, 0x30, 0x03,
            0x2a, 0x19, 0xdc, 0x2d, 0xc8, 0x05, 0xf0, 0x19, 0xdc, 0x01, 0xfa, 0x0f, 0xe6, 0x22,
            0x00, 0x28, 0x18, 0x0c, 0x85, 0x0a, 0x06, 0xe4, 0xdd, 0x34, 0x64, 0x6d, 0x44, 0x2f,
            0x18, 0xf2, 0xb3, 0x7b, 0x13, 0x59, 0xe2, 0xbc, 0x54, 0x04, 0x9c, 0xdb, 0xf4, 0x91,
            0x0b, 0xc6, 0x3c, 0xab, 0x93, 0x2a, 0xc2, 0x46, 0xd3, 0xb3, 0xf3, 0xc5, 0x02, 0x68,
            0x9b, 0x05, 0x50, 0xb9, 0x4f, 0x0c, 0x69, 0xad, 0xcd, 0xd7, 0x9a, 0xd5, 0xba, 0x22,
            0x90, 0xc9, 0x69, 0xcf, 0x6f, 0xd2, 0x5a, 0x52, 0xc9, 0x89, 0x9f, 0xdb, 0xa9,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 118);
        assert_eq!(packed[64 * 64], 121);
        assert_eq!(frame_digest(&packed), 0xd63e_96e3_d39a_e49d);
    }

    #[test]
    fn film_grain_stays_off_the_reference() {
        // The second frame predicts the first. Grain is on the shown pictures
        // and not on the samples the second frame predicts from.
        let frames: [&[u8]; 2] = [
            &[
                0x12, 0x00, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xf0, 0x11, 0x60, 0x06,
                0x32, 0x66, 0x18, 0x04, 0x01, 0x28, 0x00, 0x00, 0x05, 0x40, 0x18, 0xc8, 0x00, 0x53,
                0xfc, 0xf0, 0x30, 0x03, 0xc3, 0x00, 0x32, 0xa1, 0x9d, 0xc2, 0xdc, 0x80, 0x5f, 0x01,
                0x9d, 0xc0, 0x1f, 0xa0, 0xfe, 0x62, 0x20, 0x02, 0x81, 0x80, 0xc8, 0x50, 0xa0, 0x6e,
                0x40, 0xdd, 0x34, 0x64, 0x6d, 0x44, 0x2f, 0x18, 0xf2, 0xb3, 0x7b, 0x13, 0x59, 0xe2,
                0xbc, 0x54, 0x04, 0x9c, 0xdb, 0xf4, 0x91, 0x0b, 0xc6, 0x3c, 0xab, 0x93, 0x2a, 0xc2,
                0x46, 0xd3, 0xb3, 0xf3, 0xc5, 0x02, 0x68, 0x9b, 0x05, 0x50, 0xb9, 0x4f, 0x0c, 0x69,
                0xad, 0xcd, 0xd7, 0x9a, 0xd5, 0xba, 0x22, 0x90, 0xc9, 0x69, 0xcf, 0x6f, 0xd2, 0x5a,
                0x52, 0xc9, 0x89, 0x9f, 0xdb, 0xa9,
            ],
            &[
                0x12, 0x00, 0x32, 0x55, 0x3c, 0x02, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x09, 0x40, 0x00, 0x00, 0x20,
                0x02, 0x1b, 0x31, 0x20, 0x01, 0x4f, 0xf3, 0xc0, 0xc0, 0x0f, 0x0c, 0x00, 0xca, 0x86,
                0x77, 0x0b, 0x72, 0x01, 0x7c, 0x06, 0x77, 0x00, 0x7e, 0x83, 0xf9, 0x88, 0x80, 0x0a,
                0x06, 0x03, 0x21, 0x42, 0x81, 0xb9, 0x00, 0xdd, 0x6d, 0xf7, 0x98, 0x26, 0x99, 0x45,
                0xc8, 0xfc, 0xd1, 0x87, 0x26, 0xe0, 0x44, 0xa9, 0x42, 0xe4, 0x7d, 0x49, 0xf7, 0x0d,
                0x11, 0xd4, 0x22, 0x38, 0x58,
            ],
        ];
        let digests = [0xd63e_96e3_d39a_e49d, 0x9fee_c5fb_8546_9aaf];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(frame_digest(&packed), digests[index]);
        }
    }
    #[test]
    fn film_grain_scaled_from_luma_matches_dav1d() {
        // Chroma grain is scaled from the luma grain, with lag-2 coefficients.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0x80, 0x18, 0x32, 0x6e, 0x92, 0x80,
            0x00, 0x00, 0x54, 0x00, 0x1c, 0x50, 0x0c, 0xa4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x18, 0x90, 0xdd, 0x34, 0x64, 0x6d, 0x44, 0x2f, 0x18, 0xf2, 0xb3,
            0x7b, 0x13, 0x59, 0xe2, 0xbc, 0x54, 0x04, 0x9c, 0xdb, 0xf4, 0x91, 0x0b, 0xc6, 0x3c,
            0xab, 0x93, 0x2a, 0xc2, 0x46, 0xd3, 0xb3, 0xf3, 0xc5, 0x02, 0x68, 0x9b, 0x05, 0x50,
            0xb9, 0x4f, 0x0c, 0x69, 0xad, 0xcd, 0xd7, 0x9a, 0xd5, 0xba, 0x22, 0x90, 0xc9, 0x69,
            0xcf, 0x6f, 0xd2, 0x5a, 0x52, 0xc9, 0x89, 0x9f, 0xdb, 0xa9,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed[0], 153);
        assert_eq!(packed[64 * 64], 153);
        assert_eq!(frame_digest(&packed), 0xae59_b5fc_69f0_dd0e);
    }

    #[test]
    fn film_grain_copied_from_a_reference_matches_dav1d() {
        // Three 64×64 frames. The inter frames name reference slot 0 for every
        // grain parameter and carry their own seed. The shown pictures match
        // dav1d. A show-existing frame shows the stored picture, which is the
        // reconstruction from before grain.
        let frames: [&[u8]; 3] = [
            include_bytes!("../fixtures/fgref_0.obu"),
            include_bytes!("../fixtures/fgref_1.obu"),
            include_bytes!("../fixtures/fgref_2.obu"),
        ];
        let expected = [
            0xe40f_dc7b_d200_6040,
            0x2ef9_787f_853b_6707,
            0xfedc_eacb_858d_ccf5,
        ];
        let (sequence, key) = crate::header::parse_unit(frames[0], None).expect("key");
        assert!(key.grain.apply);
        assert!(key.grain.copy_from.is_none());
        assert!(!sequence.frame_id_present);
        for frame in &frames[1..] {
            let (_sequence, info) =
                crate::header::parse_unit(frame, Some(&sequence)).expect("inter");
            assert_eq!(info.grain.copy_from, Some(0));
            assert!(info.ref_frames.contains(&0));
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, frames[0], 0);
        assert_eq!(frame_digest(&packed), expected[0]);
        decoder.reset();
        let missing = push_error(&mut decoder, frames[1], 1);
        assert!(matches!(
            missing,
            cutlass_decode::CodecError::Malformed {
                what: "missing reference",
                ..
            }
        ));
        let packed = push_frame(&mut decoder, frames[0], 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(frame_digest(&packed), expected[0]);
        // Frame-header OBU. show_existing_frame, slot 0, then the trailing bit.
        let again = push_frame(&mut decoder, &[0x12, 0x00, 0x1a, 0x01, 0x88], 1);
        assert_eq!(frame_digest(&again), 0x09b9_a1c6_19d6_bbb2);
        assert_ne!(frame_digest(&again), expected[0]);
        for (index, payload) in frames[1..].iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, (index + 2) as i64);
            assert_eq!(
                frame_digest(&packed),
                expected[index + 1],
                "frame {}",
                index + 1
            );
        }
    }

    #[test]
    fn order_hint_frames_match_dav1d() {
        // Error-resilient single-reference frames. Order hint is on, and each
        // frame repeats the eight reference hints before its own reference list.
        let frames: [&[u8]; 3] = [
            &[
                0x12, 0x00, 0x0a, 0x0b, 0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xf0, 0x11, 0x9e, 0x00,
                0x10, 0x32, 0x62, 0x18, 0x08, 0x08, 0x02, 0xde, 0x00, 0x00, 0x0a, 0xdd, 0x2b, 0xae,
                0xc2, 0x82, 0x32, 0x75, 0x92, 0x7a, 0x7e, 0xcd, 0x7a, 0xc6, 0xd6, 0x22, 0x37, 0xd0,
                0x6c, 0x45, 0xb7, 0xe1, 0x94, 0xab, 0x0b, 0x3e, 0xf0, 0x04, 0x4e, 0x9f, 0xff, 0x05,
                0x3c, 0x6c, 0xa7, 0x0b, 0x52, 0x71, 0x30, 0x78, 0x2b, 0x48, 0x7c, 0xc1, 0x98, 0x2b,
                0x49, 0x10, 0x8c, 0x56, 0xb0, 0x9e, 0xb2, 0xe2, 0xee, 0xe8, 0xca, 0x81, 0x54, 0xf8,
                0xf7, 0x94, 0x43, 0x6d, 0x87, 0x89, 0x0b, 0xf6, 0x1f, 0x03, 0x47, 0xb1, 0xf4, 0xe4,
                0x1a, 0x1d, 0xde, 0x27, 0xfd, 0xc1, 0xca, 0xc7, 0x1a, 0x0c, 0xbe, 0xeb, 0x67, 0x82,
                0x7a, 0x1d, 0x50,
            ],
            &[
                0x12, 0x00, 0x32, 0xab, 0x01, 0x3c, 0x04, 0x08, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x0b, 0xb0, 0x00, 0x00, 0x20, 0x00, 0xdd, 0x39, 0x8e, 0xaa,
                0xcb, 0x7f, 0x5c, 0x59, 0x4f, 0x8e, 0x1a, 0x4f, 0x5f, 0xf8, 0xeb, 0x51, 0x89, 0xe5,
                0x62, 0xef, 0xfc, 0x53, 0xbe, 0x40, 0xff, 0x14, 0xaa, 0xa7, 0x31, 0x14, 0xb4, 0x4a,
                0x2e, 0x12, 0xab, 0x24, 0x97, 0xf3, 0x10, 0x3f, 0x56, 0x91, 0x42, 0x2f, 0x96, 0x93,
                0xd7, 0xfe, 0x3a, 0xd4, 0x62, 0x79, 0x56, 0x92, 0x29, 0x16, 0x10, 0xd1, 0xa8, 0x77,
                0x51, 0xdc, 0xb8, 0xc6, 0xc6, 0x56, 0xda, 0xc4, 0x3e, 0x39, 0x76, 0xe4, 0x4b, 0xcf,
                0x81, 0xfe, 0x5c, 0x87, 0x88, 0x43, 0x77, 0x0a, 0x53, 0xb2, 0x9a, 0x72, 0x96, 0xbd,
                0xae, 0x95, 0x2a, 0x9e, 0xe6, 0x56, 0x23, 0x12, 0x8e, 0xd6, 0x75, 0x1c, 0x1d, 0x61,
                0x3c, 0xab, 0x5d, 0x37, 0x27, 0x95, 0x52, 0x2a, 0x8a, 0x0a, 0x29, 0x88, 0x07, 0x75,
                0x1c, 0x1d, 0x61, 0x3c, 0xab, 0x49, 0x14, 0x8b, 0x08, 0x68, 0xd4, 0x3b, 0xa8, 0xee,
                0x5c, 0x63, 0x63, 0x2b, 0x6d, 0x62, 0x1f, 0x1a,
            ],
            &[
                0x12, 0x00, 0x32, 0xa6, 0x01, 0x3c, 0x04, 0x0c, 0x08, 0x10, 0x00, 0x10, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x80, 0x00, 0x40, 0x00, 0x20, 0x00, 0x10,
                0x00, 0x08, 0x00, 0x05, 0x0b, 0xc0, 0x00, 0x00, 0x20, 0x00, 0xdd, 0x39, 0x8e, 0x70,
                0x12, 0x2a, 0x62, 0x04, 0xf2, 0xb3, 0x5c, 0xcb, 0x5d, 0xcc, 0xef, 0xbd, 0x88, 0x74,
                0x55, 0xb2, 0x22, 0x7d, 0xb1, 0x02, 0x6d, 0x43, 0x1b, 0x65, 0x7a, 0x89, 0x93, 0xbd,
                0x16, 0x3d, 0xf3, 0x83, 0x9d, 0x44, 0xdc, 0x28, 0x97, 0xa8, 0xb3, 0x5c, 0xcb, 0x5d,
                0xcc, 0xef, 0xbd, 0x88, 0x71, 0x21, 0xd4, 0xea, 0xa8, 0xab, 0x74, 0x99, 0xed, 0x20,
                0x67, 0x10, 0x10, 0xfd, 0x8f, 0xa8, 0x66, 0xff, 0x07, 0x36, 0xfa, 0x47, 0x9a, 0xa0,
                0x2b, 0x25, 0xd4, 0x63, 0x62, 0xf5, 0x0c, 0xd6, 0x74, 0xf9, 0xc6, 0x0e, 0x41, 0x19,
                0x14, 0x77, 0x02, 0xf9, 0xb4, 0x19, 0xcb, 0x08, 0xa7, 0xf5, 0x82, 0x8a, 0x66, 0xde,
                0x54, 0x62, 0x56, 0xfc, 0x13, 0xf0, 0x34, 0x40, 0x53, 0x94, 0x08, 0x96, 0x02, 0x68,
                0x32, 0x92, 0xf2, 0x79, 0x31, 0x01, 0x0f, 0xd8, 0xfc, 0x7b, 0xd8, 0x9b, 0x53, 0xf6,
                0x3e, 0xa1, 0x9b,
            ],
        ];
        let digests = [
            0xb015_7009_2ef5_c325,
            0xb152_7972_70b9_2b25,
            0x23d6_1a82_815a_5225,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(frame_digest(&packed), digests[index]);
        }
        let mut stale = frames[1].to_vec();
        stale[9] ^= 1;
        let mut decoder = Av1Decoder::open(&params).expect("open");
        push_frame(&mut decoder, frames[0], 0);
        let error = decoder
            .push(&cutlass_decode::Packet {
                stream: 0,
                data: stale,
                pts: 1,
                duration: 1,
                timescale: 30,
                keyframe: false,
                side: Vec::new(),
            })
            .expect_err("stale order hint");
        assert!(matches!(
            error,
            cutlass_decode::CodecError::Malformed {
                what: "missing reference",
                ..
            }
        ));
        let packed = push_frame(&mut decoder, frames[1], 1);
        assert_eq!(frame_digest(&packed), digests[1]);
    }

    #[test]
    fn tile_groups_match_dav1d() {
        // 128×128, four tiles. The frame header is its own OBU and each tile is
        // its own group. The second frame is inter and has no sequence header.
        let frames: [&[u8]; 2] = [
            include_bytes!("../fixtures/tg_0.obu"),
            include_bytes!("../fixtures/tg_1.obu"),
        ];
        let expected = [0xb1a2_c0fe_a403_1499, 0xc653_6a11_a972_4b1c];
        let first = crate::header::unit_obus(frames[0]).expect("obus");
        let second = crate::header::unit_obus(frames[1]).expect("obus");
        assert_eq!(
            first.iter().map(|obu| obu.kind).collect::<Vec<_>>(),
            vec![2, 1, 3, 4, 4, 4, 4]
        );
        assert_eq!(
            second.iter().map(|obu| obu.kind).collect::<Vec<_>>(),
            vec![2, 3, 4, 4, 4, 4]
        );
        let mut header = Vec::new();
        for obu in first.iter().filter(|obu| obu.kind == 1 || obu.kind == 3) {
            header.extend_from_slice(obu.bytes);
        }
        let (_sequence, frame) = crate::header::parse_unit(&header, None).expect("header");
        assert_eq!(frame.tile_cols, 2);
        assert_eq!(frame.tile_rows, 2);
        assert!(frame.tile.is_empty());

        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let mut groups = Vec::new();
        for obu in first.iter().filter(|obu| obu.kind == 4) {
            groups.extend_from_slice(obu.bytes);
        }
        let missing = push_error(&mut decoder, &groups, 0);
        assert!(matches!(
            missing,
            cutlass_decode::CodecError::Unsupported {
                what: "tile group outside a frame",
            }
        ));
        let mut short = Vec::new();
        for obu in first.iter().take(first.len() - 1) {
            short.extend_from_slice(obu.bytes);
        }
        let short = push_error(&mut decoder, &short, 0);
        assert!(matches!(
            short,
            cutlass_decode::CodecError::Malformed {
                what: "tile group",
                ..
            }
        ));
        let mut gap = Vec::new();
        let mut skipped = false;
        for obu in &first {
            if obu.kind == 4 && !skipped {
                skipped = true;
                continue;
            }
            gap.extend_from_slice(obu.bytes);
        }
        let gap = push_error(&mut decoder, &gap, 0);
        assert!(matches!(
            gap,
            cutlass_decode::CodecError::Malformed {
                what: "tile group",
                ..
            }
        ));

        let packed = push_frame(&mut decoder, frames[0], 0);
        assert_eq!(packed.len(), 128 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), expected[0]);
        let again = push_error(&mut decoder, &groups, 1);
        assert!(matches!(
            again,
            cutlass_decode::CodecError::Unsupported {
                what: "tile group outside a frame",
            }
        ));
        let packed = push_frame(&mut decoder, frames[1], 1);
        assert_eq!(packed.len(), 128 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), expected[1]);
    }

    #[test]
    fn annex_b_lengths_match_dav1d() {
        // The same 128×128 frames, each wrapped as one Annex B temporal unit.
        // Lengths sit in front of the temporal unit, the frame unit, and every OBU.
        let frames: [&[u8]; 2] = [
            include_bytes!("../fixtures/tg_0.obu"),
            include_bytes!("../fixtures/tg_1.obu"),
        ];
        let expected = [0xb1a2_c0fe_a403_1499, 0xc653_6a11_a972_4b1c];
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let first = crate::header::unit_obus(frames[0]).expect("obus");
        let wrapped = annex_b_packet(&first);
        let packed = push_frame(&mut decoder, &wrapped, 0);
        assert_eq!(packed.len(), 128 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), expected[0]);

        let (size, used) = leb_read(&wrapped);
        assert_eq!(size, wrapped.len() - used);
        let mut inconsistent = wrapped.clone();
        let (_, frame_at) = leb_read(&inconsistent[used..]);
        let (_, obu_at) = leb_read(&inconsistent[used + frame_at..]);
        let size_at = used + frame_at + obu_at + 1;
        assert_eq!(inconsistent[size_at - 1], 0x12);
        assert_eq!(inconsistent[size_at], 0x00);
        inconsistent[size_at] = 0x01;
        let error = push_error(&mut decoder, &inconsistent, 1);
        assert!(matches!(
            error,
            cutlass_decode::CodecError::Malformed {
                what: "obu length",
                ..
            }
        ));
        let mut body = wrapped[used..].to_vec();
        body.pop();
        let mut short = leb128(body.len());
        short.extend_from_slice(&body);
        let error = push_error(&mut decoder, &short, 1);
        assert!(matches!(
            error,
            cutlass_decode::CodecError::Truncated {
                what: "annex b",
                ..
            }
        ));

        let second = crate::header::unit_obus(frames[1]).expect("obus");
        let packed = push_frame(&mut decoder, &annex_b_packet(&second), 1);
        assert_eq!(packed.len(), 128 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), expected[1]);

        // One temporal unit, two frame units. The second unit has no delimiter.
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let first_bytes: Vec<&[u8]> = first.iter().map(|obu| obu.bytes).collect();
        let second_bytes: Vec<&[u8]> = second.iter().skip(1).map(|obu| obu.bytes).collect();
        assert_eq!((second[0].bytes[0] >> 3) & 0xf, 2);
        let both = annex_b_temporal(&[frame_unit(&first_bytes), frame_unit(&second_bytes)]);
        let shown = decoder
            .push(&cutlass_decode::Packet {
                stream: 0,
                data: both,
                pts: 0,
                duration: 1,
                timescale: 30,
                keyframe: true,
                side: Vec::new(),
            })
            .expect("two frame units");
        assert_eq!(shown.len(), 2);
        for (index, output) in shown.iter().enumerate() {
            let picture = &output.as_video().expect("video").picture;
            let mut packed = picture.y.samples().to_vec();
            packed.extend_from_slice(picture.u.samples());
            packed.extend_from_slice(picture.v.samples());
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn large_scale_tile_list_matches_dav1d() {
        // 128×128 keyframe and one inter frame. CDF update, CDEF, restoration,
        // the loop filter, and order hint are off, and the grid is four 64×64
        // tiles. dav1d decodes those frame OBUs. The tile-list packet below is
        // the inter frame's header plus those same tile bytes. dav1d does not
        // decode a tile-list OBU; the mosaic is compared to dav1d's inter frame.
        let frames: [&[u8]; 2] = [
            include_bytes!("../fixtures/lst_0.obu"),
            include_bytes!("../fixtures/lst_1.obu"),
        ];
        let expected = [0x6103_2e26_fc9f_2ab5, 0x909f_ce3d_18f4_c300];
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let key = push_frame(&mut decoder, frames[0], 0);
        assert_eq!(key.len(), 128 * 128 * 3 / 2);
        assert_eq!(frame_digest(&key), expected[0]);
        let inter = push_frame(&mut decoder, frames[1], 1);
        assert_eq!(inter.len(), key.len());
        assert_eq!(frame_digest(&inter), expected[1]);

        let key_obus = crate::header::unit_obus(frames[0]).expect("key obus");
        let inter_obus = crate::header::unit_obus(frames[1]).expect("inter obus");
        let sequence = key_obus
            .iter()
            .find(|obu| obu.kind == 1)
            .expect("sequence")
            .bytes;
        let frame_obu = inter_obus
            .iter()
            .find(|obu| obu.kind == 6)
            .expect("frame")
            .bytes;
        let mut parsed = sequence.to_vec();
        parsed.extend_from_slice(frame_obu);
        let (_sequence, info) = crate::header::parse_unit(&parsed, None).expect("inter header");
        assert!(!info.intra);
        assert!(!info.show_existing);
        assert!(info.disable_cdf_update);
        assert_eq!(info.lf_level, [0, 0, 0, 0]);
        assert_eq!((info.tile_cols, info.tile_rows), (2, 2));
        assert_eq!(info.upscaled_width, info.width);
        assert!(!info.grain.apply);
        assert_eq!(info.ref_frames[0], 0);
        let payload = crate::header::obu_payload(frame_obu).expect("frame payload");
        assert!(payload.ends_with(&info.tile));
        let raw_header = &payload[..payload.len() - info.tile.len()];
        let mut headers = Vec::new();
        for pad in 0..8 {
            let candidate = header_with_trailing(raw_header, pad);
            let mut probe = sequence.to_vec();
            probe.extend(obu_with_size(3, &candidate));
            let Ok((_sequence, parsed)) = crate::header::parse_unit(&probe, None) else {
                continue;
            };
            if parsed.frame_type == info.frame_type
                && parsed.width == info.width
                && parsed.height == info.height
                && parsed.base_q == info.base_q
                && parsed.disable_cdf_update == info.disable_cdf_update
                && parsed.ref_frames == info.ref_frames
                && parsed.lf_level == info.lf_level
                && parsed.mi_col_starts == info.mi_col_starts
                && parsed.mi_row_starts == info.mi_row_starts
                && parsed.refresh_flags == info.refresh_flags
                && parsed.primary_ref == info.primary_ref
                && parsed.reference_select == info.reference_select
            {
                headers.push(candidate);
            }
        }
        assert_eq!(headers.len(), 1, "frame header trailing bit");
        let header = &headers[0];
        let parts = crate::header::tile_parts(&info).expect("tiles");
        assert_eq!(parts.len(), 4);
        let mut listed = Vec::new();
        for part in &parts {
            let row = info
                .mi_row_starts
                .iter()
                .position(|start| *start as usize == part.row)
                .expect("tile row") as u8;
            let col = info
                .mi_col_starts
                .iter()
                .position(|start| *start as usize == part.col)
                .expect("tile col") as u8;
            listed.push((0u8, row, col, part.data.as_slice()));
        }
        assert_eq!(
            listed
                .iter()
                .map(|(_, row, col, _)| (*row, *col))
                .collect::<Vec<_>>(),
            vec![(0, 0), (0, 1), (1, 0), (1, 1)]
        );

        let mut mosaic = Av1Decoder::open(&params).expect("open");
        push_frame(&mut mosaic, frames[0], 0);
        let listed_packet = list_packet(sequence, header, &listed, 2, 2);
        let picture = push_frame(&mut mosaic, &listed_packet, 1);
        assert_eq!(frame_digest(&picture), expected[1]);
        let again = push_frame(&mut mosaic, frames[1], 2);
        assert_eq!(frame_digest(&again), expected[1]);

        let swapped = vec![listed[1], listed[0], listed[3], listed[2]];
        let swapped = list_packet(sequence, header, &swapped, 2, 2);
        let mut reorder = Av1Decoder::open(&params).expect("open");
        push_frame(&mut reorder, frames[0], 0);
        let picture = push_frame(&mut reorder, &swapped, 1);
        assert_eq!(frame_digest(&picture), frame_digest(&swap_columns(&inter)));

        let one = list_packet(sequence, header, &listed[3..], 1, 1);
        let mut single = Av1Decoder::open(&params).expect("open");
        push_frame(&mut single, frames[0], 0);
        let picture = push_frame(&mut single, &one, 1);
        assert_eq!(picture.len(), 64 * 64 * 3 / 2);
        assert_eq!(
            frame_digest(&picture),
            frame_digest(&crop_tile(&inter, 1, 1))
        );

        let corner = list_packet(sequence, header, &listed[..1], 2, 2);
        let mut partial = Av1Decoder::open(&params).expect("open");
        push_frame(&mut partial, frames[0], 0);
        let picture = push_frame(&mut partial, &corner, 1);
        assert_eq!(picture[64], 0);
        assert_eq!(picture[128 * 128 + 32], 128);
        assert_eq!(
            frame_digest(&crop_tile(&picture, 0, 0)),
            frame_digest(&crop_tile(&inter, 0, 0))
        );

        let mut fresh = Av1Decoder::open(&params).expect("open");
        let missing = push_error(&mut fresh, &listed_packet, 0);
        assert!(matches!(
            missing,
            cutlass_decode::CodecError::Malformed {
                what: "missing reference",
                ..
            }
        ));
        let listed_obus = crate::header::unit_obus(&listed_packet).expect("list obus");
        let list_obu = listed_obus
            .iter()
            .find(|obu| obu.kind == 8)
            .expect("tile list obu");
        let orphan = push_error(&mut fresh, list_obu.bytes, 0);
        assert!(matches!(
            orphan,
            cutlass_decode::CodecError::Malformed {
                what: "tile list",
                ..
            }
        ));
        push_frame(&mut fresh, frames[0], 0);
        let mut anchored = listed.clone();
        anchored[0].0 = 8;
        let high = list_packet(sequence, header, &anchored, 2, 2);
        let high = push_error(&mut fresh, &high, 1);
        assert!(matches!(
            high,
            cutlass_decode::CodecError::Malformed {
                what: "tile list",
                ..
            }
        ));
        let mut overrun = listed_packet.clone();
        let list_at = overrun
            .windows(list_obu.bytes.len())
            .position(|window| window == list_obu.bytes)
            .expect("list obu");
        let list_payload = crate::header::obu_payload(list_obu.bytes).expect("list payload");
        let size_at = list_at + (list_obu.bytes.len() - list_payload.len()) + 7;
        overrun[size_at] = 0xff;
        overrun[size_at + 1] = 0xff;
        let overrun = push_error(&mut fresh, &overrun, 1);
        assert!(matches!(
            overrun,
            cutlass_decode::CodecError::Truncated {
                what: "tile list",
                ..
            }
        ));
        let inter = push_frame(&mut fresh, frames[1], 2);
        assert_eq!(frame_digest(&inter), expected[1]);
    }

    fn header_with_trailing(header: &[u8], pad: usize) -> Vec<u8> {
        if pad == 0 {
            let mut out = header.to_vec();
            out.push(0x80);
            return out;
        }
        let mut out = header.to_vec();
        let last = out.last_mut().expect("header byte");
        let keep = !((1u8 << pad) - 1);
        let trail = 1u8 << (pad - 1);
        *last = (*last & keep) | trail;
        out
    }

    fn obu_with_size(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![(kind << 3) | 0x02];
        out.extend(leb128(payload.len()));
        out.extend_from_slice(payload);
        out
    }

    fn list_packet(
        sequence: &[u8],
        header: &[u8],
        entries: &[(u8, u8, u8, &[u8])],
        wide: u8,
        high: u8,
    ) -> Vec<u8> {
        let mut payload = vec![wide - 1, high - 1];
        let count = u16::try_from(entries.len()).expect("count") - 1;
        payload.extend(count.to_be_bytes());
        for (anchor, row, col, data) in entries {
            payload.push(*anchor);
            payload.push(*row);
            payload.push(*col);
            let size = u16::try_from(data.len()).expect("tile size") - 1;
            payload.extend(size.to_be_bytes());
            payload.extend_from_slice(data);
        }
        let mut packet = vec![0x12, 0x00];
        packet.extend_from_slice(sequence);
        packet.extend(obu_with_size(3, header));
        packet.extend(obu_with_size(8, &payload));
        packet
    }

    fn crop_tile(packed: &[u8], col: usize, row: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 * 64 * 3 / 2);
        for line in 0..64 {
            let start = (row * 64 + line) * 128 + col * 64;
            out.extend_from_slice(&packed[start..start + 64]);
        }
        for plane in 0..2 {
            let chroma = &packed[128 * 128 + plane * 64 * 64..];
            for line in 0..32 {
                let start = (row * 32 + line) * 64 + col * 32;
                out.extend_from_slice(&chroma[start..start + 32]);
            }
        }
        out
    }

    fn swap_columns(packed: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; packed.len()];
        let order = [(1usize, 0usize), (0, 0), (1, 1), (0, 1)];
        for (index, (sx, sy)) in order.iter().enumerate() {
            let dx = index % 2;
            let dy = index / 2;
            for line in 0..64 {
                let from = (sy * 64 + line) * 128 + sx * 64;
                let to = (dy * 64 + line) * 128 + dx * 64;
                out[to..to + 64].copy_from_slice(&packed[from..from + 64]);
            }
        }
        for plane in 0..2 {
            let base = 128 * 128 + plane * 64 * 64;
            for (index, (sx, sy)) in order.iter().enumerate() {
                let dx = index % 2;
                let dy = index / 2;
                for line in 0..32 {
                    let from = base + (sy * 32 + line) * 64 + sx * 32;
                    let to = base + (dy * 32 + line) * 64 + dx * 32;
                    out[to..to + 32].copy_from_slice(&packed[from..from + 32]);
                }
            }
        }
        out
    }

    #[test]
    fn nonuniform_tiles_match_dav1d() {
        // 320×192 keyframes. Superblocks are 64 pixels, so the grid is five by
        // three. The signaled widths are 2, 2, 1 and the heights are 2, 1.
        // A 4-row chroma block just inside the shorter row must not read the
        // tile above: that neighbor is two luma rows back, outside this tile.
        let frames: [&[u8]; 2] = [
            include_bytes!("../fixtures/nu_0.obu"),
            include_bytes!("../fixtures/nu_1.obu"),
        ];
        let expected = [0xf2df_1c82_2f33_5300, 0x6878_5ee4_efd0_358f];
        let (_sequence, frame) = crate::header::parse_unit(frames[0], None).expect("header");
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 192);
        assert_eq!(frame.tile_cols, 3);
        assert_eq!(frame.tile_rows, 2);
        assert_eq!(frame.mi_col_starts, vec![0, 32, 64, 80]);
        assert_eq!(frame.mi_row_starts, vec![0, 32, 48]);
        let params = CodecParams::video(CodecId::Av1, 320, 192);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 320 * 192 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn four_uniform_tiles_match_dav1d() {
        // 128×128 keyframe split into a 2×2 uniform grid. Each tile has its own
        // entropy stream, and prediction does not cross a tile edge.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x19, 0xbf, 0xff, 0x20, 0x02, 0x32, 0x89, 0x0a, 0x9e,
            0x54, 0x00, 0x00, 0x01, 0x40, 0x00, 0x38, 0x01, 0xdd, 0x2b, 0xc0, 0x46, 0xe9, 0xd9,
            0xf4, 0xa0, 0xa0, 0xa8, 0x22, 0xec, 0x0f, 0xd7, 0x71, 0x12, 0x36, 0x96, 0xc1, 0xde,
            0xcc, 0xe2, 0x13, 0x83, 0x53, 0x38, 0x00, 0x65, 0x83, 0xec, 0x02, 0x4a, 0x5e, 0xbb,
            0x2c, 0xdc, 0x39, 0xae, 0xc0, 0x33, 0x27, 0x5b, 0x82, 0xbe, 0x6f, 0x76, 0x7a, 0x84,
            0x9d, 0x15, 0xe9, 0x30, 0x76, 0x93, 0x29, 0xe0, 0x58, 0x68, 0xab, 0xe2, 0x18, 0x56,
            0xde, 0x46, 0x6f, 0xe7, 0xbb, 0x95, 0x8b, 0x66, 0x9a, 0x0a, 0x83, 0xdb, 0xf8, 0xe4,
            0x25, 0x66, 0xd7, 0x14, 0x27, 0xa3, 0x6b, 0xfa, 0x5b, 0xc7, 0x2e, 0x52, 0xcd, 0xb1,
            0x2e, 0x30, 0xa2, 0x21, 0x9a, 0xb3, 0x09, 0x57, 0x96, 0x18, 0x68, 0xda, 0x44, 0x09,
            0x88, 0x98, 0xfc, 0x07, 0x5d, 0xe9, 0x75, 0xd0, 0x60, 0xb2, 0xd0, 0x95, 0x9b, 0xc3,
            0x92, 0xd7, 0xaa, 0xe0, 0x16, 0xd1, 0xf0, 0x82, 0xda, 0x2a, 0xa9, 0x0b, 0x60, 0x18,
            0xc9, 0x65, 0x0f, 0xc9, 0x6d, 0xf9, 0x33, 0x24, 0x75, 0x39, 0x18, 0xb6, 0xcc, 0xa3,
            0x85, 0x37, 0x0e, 0x8d, 0x49, 0x01, 0xee, 0x01, 0xcb, 0x5f, 0x80, 0x19, 0x03, 0x53,
            0x6c, 0x9c, 0xb7, 0x97, 0xd3, 0x43, 0x6e, 0xfb, 0x8f, 0xc6, 0x71, 0x7f, 0x5a, 0x21,
            0x04, 0x65, 0x79, 0xd4, 0x9a, 0xe8, 0x60, 0x05, 0x4c, 0x80, 0xeb, 0x6c, 0xb3, 0x47,
            0x15, 0x16, 0x50, 0xeb, 0x1c, 0xc4, 0x18, 0x85, 0xd3, 0x68, 0xef, 0x05, 0xf2, 0xa5,
            0x71, 0x6f, 0x4c, 0xbe, 0x2d, 0x63, 0x8c, 0xfd, 0x52, 0xc5, 0x7c, 0x2f, 0x0e, 0x56,
            0x74, 0x56, 0x69, 0x26, 0x8c, 0x4c, 0xbb, 0x79, 0x07, 0x36, 0xfd, 0x20, 0x9d, 0xd5,
            0xfa, 0x43, 0x9f, 0x80, 0xb7, 0x0c, 0xf6, 0xb2, 0x6b, 0x8f, 0x53, 0x1c, 0xf8, 0xbf,
            0x46, 0x91, 0x12, 0x72, 0xa8, 0x62, 0x5a, 0xde, 0xe0, 0x07, 0x14, 0x0e, 0x7d, 0x6b,
            0xc6, 0x7c, 0x1d, 0x76, 0xe3, 0xed, 0x09, 0xb4, 0x66, 0x30, 0x5b, 0x55, 0x4f, 0x95,
            0x7e, 0x66, 0xc3, 0xa4, 0x57, 0xb6, 0x30, 0x8a, 0x56, 0xd6, 0xa2, 0x23, 0x83, 0x82,
            0x25, 0xbe, 0xf8, 0x75, 0x0f, 0x6f, 0xe1, 0xa9, 0x72, 0x47, 0x6e, 0x79, 0x81, 0x87,
            0x77, 0xcc, 0xee, 0xb5, 0xc9, 0x78, 0x1f, 0xdc, 0x8b, 0x78, 0x9c, 0x72, 0xc0, 0x07,
            0x01, 0xdd, 0x5b, 0x13, 0x9a, 0x08, 0xce, 0x17, 0x6a, 0x46, 0x36, 0x29, 0xd4, 0x54,
            0x23, 0xab, 0x7e, 0x06, 0xdf, 0x71, 0x98, 0x7f, 0xa0, 0x0f, 0x0c, 0x92, 0x1f, 0xae,
            0xa8, 0x99, 0xe8, 0xb2, 0x2c, 0x29, 0x00, 0x7f, 0xae, 0xdf, 0xf9, 0x6f, 0xdb, 0x6d,
            0x6a, 0x22, 0x38, 0x38, 0x22, 0x5b, 0xef, 0x87, 0x50, 0xf6, 0xfe, 0x1a, 0x7b, 0xbd,
            0xf3, 0x5c, 0x03, 0x30, 0x90, 0x46, 0xc2, 0x4b, 0x11, 0xa8, 0x19, 0x24, 0xb3, 0x1a,
            0x4d, 0xc4, 0xb5, 0x90, 0x76, 0x45, 0x94, 0x42, 0x71, 0xf7, 0x11, 0xbe, 0x1e, 0x11,
            0x26, 0x7e, 0x8f, 0xd3, 0xc8, 0x09, 0x7d, 0x09, 0x28, 0x7f, 0x69, 0x2e, 0x18, 0xc7,
            0x87, 0xc0, 0xff, 0xf1, 0xc8, 0x18, 0x06, 0x6f, 0x17, 0xeb, 0xc8, 0xe3, 0x4f, 0x5b,
            0x1a, 0xb0, 0x6c, 0xfc, 0x0f, 0xb6, 0xd1, 0x34, 0x2c, 0x83, 0xc4, 0xb0, 0xfc, 0x97,
            0x8f, 0x37, 0x69, 0x9a, 0x61, 0xda, 0x90, 0x55, 0xfc, 0x3c, 0x7c, 0x0d, 0x2f, 0x63,
            0xb2, 0x48, 0xdf, 0xfd, 0x42, 0x58, 0x8c, 0x21, 0x98, 0xb2, 0xa1, 0xda, 0xf4, 0x91,
            0x63, 0xde, 0x3e, 0x67, 0x33, 0x73, 0x5a, 0x3f, 0x44, 0xed, 0xbf, 0x26, 0x2e, 0x63,
            0x1e, 0xbd, 0x57, 0x28, 0xaf, 0xb6, 0xca, 0x7b, 0x5c, 0xfd, 0x55, 0x5c, 0x06, 0xac,
            0x5b, 0x23, 0xaa, 0xbb, 0x44, 0x62, 0xba, 0x22, 0xf8, 0xad, 0xb6, 0xb8, 0x8c, 0x9e,
            0xfb, 0x73, 0x57, 0x9b, 0x3e, 0x38, 0xb0, 0x90, 0x22, 0x41, 0x7f, 0x19, 0xb6, 0x2e,
            0x53, 0xdb, 0x88, 0xdc, 0xb1, 0x51, 0xa6, 0x8e, 0xfe, 0xd4, 0xff, 0x42, 0x58, 0xd8,
            0x04, 0x58, 0x80, 0xc9, 0xd5, 0xac, 0x00, 0xd4, 0xb6, 0x4f, 0xfb, 0xea, 0x08, 0x83,
            0x55, 0x25, 0x7d, 0x3a, 0x89, 0x65, 0x98, 0xa2, 0x81, 0x8d, 0x13, 0x73, 0xe9, 0x50,
            0xe1, 0x59, 0x13, 0x0f, 0x7a, 0xe5, 0x8d, 0xad, 0x6f, 0x2a, 0xe8, 0x51, 0xbc, 0x81,
            0x01, 0xdd, 0x2b, 0xc0, 0x46, 0xd3, 0xb6, 0x8c, 0x24, 0x8e, 0x61, 0x56, 0xae, 0x8e,
            0xa3, 0xd7, 0xe2, 0x49, 0x6a, 0x2d, 0x64, 0x65, 0xb9, 0xf3, 0x18, 0xbc, 0xda, 0xdc,
            0x54, 0x23, 0xe2, 0xf2, 0xd6, 0xcf, 0x0d, 0x41, 0x61, 0xfa, 0xf1, 0xe3, 0x02, 0xb1,
            0x76, 0xbe, 0x58, 0x01, 0x7d, 0x32, 0x25, 0xc2, 0x2a, 0x32, 0x75, 0xdc, 0x86, 0xf7,
            0xa2, 0x56, 0x53, 0x7d, 0xc3, 0x2d, 0x61, 0x8f, 0xf5, 0xd2, 0x52, 0x22, 0xe6, 0xfc,
            0x3e, 0xe1, 0x13, 0x16, 0xba, 0xfb, 0x77, 0xc9, 0x49, 0x8c, 0xea, 0x06, 0x16, 0x88,
            0x98, 0x4b, 0xf6, 0xf8, 0x44, 0x9a, 0x0a, 0x74, 0x2b, 0x95, 0x95, 0xac, 0xe3, 0xe0,
            0x14, 0x1b, 0xac, 0x48, 0xf1, 0x4a, 0xd7, 0x70, 0x0f, 0xe8, 0xfb, 0x41, 0xfc, 0x05,
            0xb8, 0x2f, 0xe0, 0x9d, 0x37, 0x73, 0xff, 0x42, 0xdb, 0xab, 0x11, 0xc0, 0x38, 0x19,
            0x5e, 0x77, 0x9e, 0xec, 0xed, 0xce, 0xeb, 0x03, 0xd7, 0x14, 0xb2, 0x5f, 0x2c, 0xc2,
            0x3e, 0x35, 0x74, 0x3f, 0x01, 0xb9, 0xb6, 0x52, 0x5a, 0x9b, 0x42, 0xe0, 0x5b, 0x37,
            0x4b, 0x0b, 0x8a, 0x4d, 0x0d, 0x43, 0x39, 0x9f, 0x31, 0x89, 0xb9, 0x76, 0xcd, 0xc8,
            0x74, 0xfa, 0xd7, 0x2e, 0x66, 0x68, 0x22, 0xaf, 0x40, 0x0b, 0x1b, 0xc2, 0x7b, 0xa1,
            0x26, 0xa3, 0x98, 0xdd, 0xde, 0xeb, 0xf8, 0xa2, 0xfb, 0x5f, 0x8e, 0x48, 0xd0, 0xe2,
            0xa0, 0x0f, 0x61, 0x48, 0xd0, 0xa3, 0xce, 0x17, 0x29, 0xd3, 0xd9, 0xa5, 0xad, 0xfb,
            0xe8, 0x5a, 0x37, 0x5e, 0x33, 0x55, 0xb6, 0xed, 0x7b, 0x01, 0x00, 0x28, 0xd8, 0x99,
            0xc3, 0x1e, 0x65, 0xa0, 0x77, 0x04, 0xda, 0x53, 0xff, 0xdc, 0x03, 0x96, 0xc5, 0x7a,
            0xb3, 0xda, 0xe7, 0xee, 0x78, 0xcc, 0x41, 0x86, 0x41, 0x1a, 0x36, 0x51, 0xbe, 0xda,
            0x08, 0xde, 0x4e, 0x3b, 0xc0, 0x65, 0xd3, 0x34, 0xcc, 0xcc, 0x1d, 0xb3, 0x59, 0x29,
            0x16, 0x1a, 0xa8, 0x4b, 0x7d, 0xfa, 0xe6, 0xe5, 0xa1, 0xf7, 0x8d, 0xd9, 0xec, 0xf1,
            0x76, 0xf3, 0x10, 0x68, 0xb2, 0x83, 0xe8, 0x22, 0xe5, 0x98, 0xc4, 0x88, 0x0b, 0x1d,
            0x78, 0x53, 0x00, 0x98, 0x55, 0xa0, 0x46, 0xdb, 0xb1, 0x4c, 0x12, 0x55, 0x63, 0x8d,
            0x0e, 0x19, 0xbb, 0x23, 0x47, 0x38, 0xa9, 0x30, 0x59, 0x2f, 0x47, 0xdf, 0x87, 0x15,
            0xfe, 0x8e, 0xfe, 0xa7, 0x6a, 0x0e, 0x61, 0xa5, 0xd3, 0xf5, 0xd2, 0x09, 0x53, 0xe7,
            0x4e, 0xb6, 0x6f, 0xdb, 0x71, 0x4c, 0xfd, 0xef, 0xef, 0x80, 0x7b, 0x85, 0xa3, 0x86,
            0x19, 0x98, 0x9b, 0x17, 0x10, 0x3f, 0xa2, 0xd8, 0x86, 0x4d, 0x05, 0xd7, 0x91, 0xc6,
            0x9e, 0x9a, 0x05, 0xde, 0xcc, 0xae, 0x1b, 0xaa, 0x00, 0xbb, 0x24, 0xce, 0xc4, 0x77,
            0xaa, 0xab, 0x80, 0xce, 0x19, 0x4e, 0x47, 0x5b, 0xa0, 0xdd, 0x2b, 0xc0, 0x46, 0xe9,
            0xd9, 0xf4, 0xa0, 0xa0, 0xa8, 0x22, 0xec, 0x0f, 0xd7, 0x71, 0x12, 0x36, 0x96, 0xc1,
            0xde, 0xcc, 0xe2, 0x13, 0x83, 0x53, 0x38, 0x00, 0x65, 0x83, 0xec, 0x02, 0x4a, 0x5e,
            0xbb, 0x2c, 0xdc, 0x39, 0xae, 0xc0, 0x33, 0x27, 0x5b, 0x82, 0xbe, 0x6f, 0x76, 0x7a,
            0x84, 0x9d, 0x15, 0xe9, 0x30, 0x76, 0x93, 0x29, 0xe0, 0x58, 0x68, 0xab, 0xe2, 0x18,
            0x56, 0xde, 0x46, 0x6f, 0xe7, 0xbb, 0x95, 0x8b, 0x66, 0x9a, 0x0a, 0x83, 0xdb, 0xf8,
            0xe4, 0x25, 0x66, 0xd7, 0x14, 0x27, 0xa3, 0x6b, 0xfa, 0x5b, 0xc7, 0x2e, 0x52, 0xcd,
            0xb1, 0x2e, 0x30, 0xa2, 0x21, 0x9a, 0xb3, 0x09, 0x57, 0x96, 0x18, 0x68, 0xda, 0x44,
            0x09, 0x88, 0x98, 0xfc, 0x07, 0x5d, 0xe9, 0x75, 0xd0, 0x60, 0xb2, 0xd0, 0x95, 0x9b,
            0xc3, 0x92, 0xd7, 0xaa, 0xe0, 0x16, 0xd1, 0xf0, 0x82, 0xda, 0x2a, 0xa9, 0x0b, 0x60,
            0x18, 0xc9, 0x65, 0x0f, 0xc9, 0x6d, 0xf9, 0x33, 0x24, 0x75, 0x39, 0x18, 0xb6, 0xcc,
            0xa3, 0x85, 0x37, 0x0e, 0x8d, 0x49, 0x01, 0xee, 0x01, 0xcb, 0x5f, 0x80, 0x19, 0x03,
            0x53, 0x6c, 0x9c, 0xb7, 0x97, 0xd3, 0x43, 0x6e, 0xfb, 0x8f, 0xc6, 0x71, 0x7f, 0x5a,
            0x21, 0x04, 0x65, 0x79, 0xd4, 0x9a, 0xe8, 0x60, 0x05, 0x4c, 0x80, 0xeb, 0x6c, 0xb3,
            0x47, 0x15, 0x16, 0x50, 0xeb, 0x1c, 0xc4, 0x18, 0x85, 0xd3, 0x68, 0xef, 0x05, 0xf2,
            0xa5, 0x71, 0x6f, 0x4c, 0xbe, 0x2d, 0x63, 0x8c, 0xfd, 0x52, 0xc5, 0x7c, 0x2f, 0x0e,
            0x56, 0x74, 0x56, 0x69, 0x26, 0x8c, 0x4c, 0xbb, 0x79, 0x07, 0x36, 0xfd, 0x20, 0x9d,
            0xd5, 0xfa, 0x43, 0x9f, 0x80, 0xb7, 0x0c, 0xf6, 0xb2, 0x6b, 0x8f, 0x53, 0x1c, 0xf8,
            0xbf, 0x46, 0x91, 0x12, 0x72, 0xa8, 0x62, 0x5a, 0xde, 0xe0, 0x07, 0x14, 0x0e, 0x7d,
            0x6b, 0xc6, 0x7c, 0x1d, 0x76, 0xe3, 0xed, 0x09, 0xb4, 0x66, 0x30, 0x5b, 0x55, 0x4f,
            0x95, 0x7e, 0x66, 0xc3, 0xa4, 0x57, 0xb6, 0x30, 0x8a, 0x56, 0xd6, 0xa2, 0x23, 0x83,
            0x82, 0x25, 0xbe, 0xf8, 0x75, 0x0f, 0x6f, 0xe1, 0xa9, 0x72, 0x47, 0x6e, 0x79, 0x81,
            0x87, 0x77, 0xcc, 0xee, 0xb5, 0xc9, 0x78, 0x1f, 0xdc, 0x8b, 0x78, 0x9c, 0x72, 0xc0,
        ];
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 128 * 128 * 3 / 2);
        assert_eq!(packed[0], 43);
        assert_eq!(frame_digest(&packed), 0xf38e_b8da_938e_0dea);
    }
    #[test]
    fn chroma_above_right_matches_dav1d() {
        // 64×64 keyframe whose chroma is not flat. Above-right availability is
        // in the plane's own 4×4 grid, so a diagonal chroma mode can see the
        // block that finished on the row above.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0x80, 0x08, 0x32, 0xb9, 0x04, 0x96,
            0x00, 0x00, 0x00, 0x50, 0xdd, 0x2b, 0xee, 0xa5, 0xa6, 0x0a, 0x19, 0xba, 0x0f, 0x19,
            0xb8, 0x1f, 0x85, 0xaf, 0x18, 0xe5, 0x85, 0x52, 0xc6, 0x97, 0x67, 0xc7, 0x8e, 0xff,
            0xda, 0xf3, 0x97, 0xc0, 0x4f, 0xb7, 0x40, 0xbd, 0xad, 0x6a, 0x79, 0xc7, 0xe5, 0x75,
            0xfc, 0x00, 0xc9, 0x16, 0x32, 0x94, 0x23, 0x42, 0x02, 0xa9, 0x74, 0xd1, 0xe9, 0x41,
            0x2b, 0xe5, 0x93, 0xc0, 0x10, 0xbd, 0x23, 0xb0, 0x3d, 0x9e, 0x6f, 0x7a, 0x24, 0xb9,
            0xa0, 0x13, 0xa6, 0xf7, 0xe4, 0x7a, 0x4a, 0xcd, 0x2a, 0x6f, 0xd2, 0x0d, 0x24, 0x68,
            0xf7, 0x84, 0xfa, 0x8b, 0xb0, 0x3e, 0x6d, 0xaf, 0x87, 0xb3, 0xd7, 0x30, 0xc9, 0x54,
            0x61, 0x33, 0xf9, 0xdb, 0x26, 0xad, 0xde, 0xd3, 0x9d, 0xc2, 0x98, 0x39, 0x4c, 0x21,
            0x15, 0x97, 0xe7, 0xdb, 0x8f, 0x93, 0x09, 0x4c, 0x7b, 0xee, 0x9c, 0xe8, 0x8d, 0x6d,
            0x34, 0xec, 0x6f, 0x2a, 0x41, 0xe1, 0xdf, 0x11, 0xaf, 0x43, 0xba, 0x5c, 0x4b, 0x5b,
            0x1a, 0xea, 0x1a, 0xa1, 0x83, 0x86, 0xb7, 0xca, 0xbe, 0x8a, 0x63, 0x3e, 0x3a, 0x4d,
            0xc1, 0x09, 0x25, 0x92, 0x58, 0xbb, 0x1e, 0xae, 0x26, 0x8d, 0xbd, 0x48, 0x46, 0x3d,
            0x19, 0x8e, 0xd4, 0xc0, 0x80, 0x9c, 0xce, 0x4b, 0x0d, 0xac, 0x8f, 0x5c, 0xa7, 0x56,
            0xb5, 0xb0, 0xe6, 0x87, 0x0d, 0x96, 0xd4, 0xd2, 0xae, 0xd0, 0x56, 0x9f, 0x64, 0x86,
            0x52, 0x06, 0xe6, 0x8b, 0x2c, 0x58, 0xb2, 0xc2, 0x74, 0xd1, 0x0f, 0x07, 0xa3, 0x76,
            0xba, 0x5e, 0x4a, 0x4d, 0xdb, 0x41, 0x6b, 0xca, 0x7f, 0x6c, 0xc2, 0xf5, 0xad, 0x18,
            0x1c, 0x5e, 0xde, 0x08, 0x8d, 0x7a, 0x3b, 0x06, 0xb5, 0xfa, 0xb8, 0x10, 0x4c, 0x48,
            0x2e, 0xe2, 0x8c, 0x68, 0xa7, 0x26, 0xfd, 0xdd, 0xd3, 0x9a, 0xf3, 0x34, 0x37, 0xa5,
            0xe5, 0x8e, 0xcd, 0x8b, 0x54, 0x65, 0x9b, 0x6f, 0x46, 0x24, 0x02, 0x4f, 0x06, 0xa5,
            0x8d, 0x4f, 0xb7, 0x06, 0xae, 0x88, 0xa6, 0xc1, 0x82, 0x3f, 0x84, 0xa8, 0xab, 0x09,
            0x5a, 0x82, 0xf7, 0x0a, 0x9b, 0x83, 0x51, 0x23, 0x84, 0x2b, 0xd4, 0xbc, 0xaa, 0x93,
            0xcb, 0x4d, 0x3e, 0xd6, 0xdf, 0x3d, 0xb5, 0x93, 0x6e, 0x0b, 0x61, 0xd0, 0xb6, 0x04,
            0x6d, 0xf5, 0xc0, 0x94, 0x48, 0xee, 0x86, 0x4b, 0x1d, 0xb5, 0xef, 0x0b, 0x29, 0x0a,
            0x3d, 0x01, 0x3e, 0xe8, 0x0f, 0x0d, 0xe6, 0x73, 0x35, 0x3d, 0x0c, 0xa6, 0x6f, 0x6d,
            0xa9, 0xb4, 0x7f, 0x95, 0x83, 0x3a, 0x13, 0x84, 0x84, 0xd1, 0x71, 0x71, 0xf5, 0xe5,
            0x01, 0x7f, 0x9f, 0xfe, 0x7e, 0x57, 0x9b, 0x1f, 0xa9, 0x9b, 0x48, 0x08, 0xbb, 0x88,
            0x01, 0x9a, 0xf4, 0x4c, 0x98, 0x35, 0x8e, 0x24, 0x11, 0x5c, 0x42, 0x04, 0x3b, 0x61,
            0x23, 0x74, 0x6f, 0xa2, 0xc5, 0x50, 0x75, 0x60, 0xe1, 0xc8, 0xf0, 0x4a, 0x3f, 0x8d,
            0xc0, 0x40, 0x5e, 0xf2, 0x40, 0xb1, 0x14, 0x4c, 0x85, 0x7f, 0x0a, 0xbb, 0xaf, 0xb4,
            0x4e, 0x4e, 0x11, 0xca, 0x38, 0x1d, 0x7f, 0xc6, 0x6f, 0x96, 0x08, 0xfe, 0x88, 0xfa,
            0xe3, 0x1b, 0x8c, 0x1a, 0x55, 0x3a, 0x7b, 0xe2, 0x69, 0x30, 0xd3, 0xa7, 0xfe, 0x7d,
            0xaf, 0x5b, 0x17, 0x2d, 0xd0, 0x5c, 0xcd, 0xa0, 0x1a, 0x10, 0x3c, 0x47, 0xbd, 0xfb,
            0x70, 0x3e, 0x6a, 0x6b, 0xcf, 0x92, 0xaa, 0xf5, 0x37, 0x09, 0x87, 0x44, 0xbb, 0xe8,
            0xcb, 0xda, 0x4f, 0x88, 0x01, 0x6d, 0x9b, 0x77, 0xa1, 0x6a, 0x5b, 0x61, 0xda, 0x49,
            0x19, 0x62, 0x6f, 0xb2, 0x2a, 0xe9, 0x70, 0xbc, 0x53, 0x22, 0xda, 0x46, 0x87, 0xb1,
            0xc4, 0x2f, 0x2b, 0x92, 0x9b, 0x25, 0xc5, 0xd5, 0xc6, 0x7d, 0x3d, 0x49, 0xc8, 0x8b,
            0x25, 0xc5, 0xe2, 0x72, 0xcb, 0xc8, 0x89, 0xa7, 0xc2, 0xb4, 0xc1, 0x1e, 0x3a, 0x73,
            0x77, 0x39, 0x6d, 0xde, 0x7d, 0x9d, 0x79, 0x71, 0x36, 0x9b, 0x92, 0x8d, 0x58, 0xee,
            0xc0, 0x39, 0xc9, 0x6d, 0xf8, 0x1b, 0x7b, 0xa7, 0xe7, 0x0c, 0xe8, 0xc0, 0x64, 0xc5,
            0x14, 0x4c, 0x18, 0x2f, 0x2e, 0x0e, 0x52, 0x0f, 0xab, 0x49, 0x8a, 0x0b, 0xf9, 0xc1,
            0xea, 0xd3, 0x5b, 0x9a, 0x31, 0x7b, 0xf7, 0xf4,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 21);
        assert_eq!(packed[64 * 64 + 4 * 32], 85);
        assert_eq!(frame_digest(&packed), 0x746c_bddf_90ff_e41d);
    }
    #[test]
    fn quantizer_matrix_matches_dav1d() {
        // 64×64 keyframe. Level 3 of the specification quantizer matrix scales
        // each coefficient before the inverse transform. Identity transforms
        // skip the matrix. Chroma is not flat, so the chroma plane of the
        // matrix is on the path.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0x80, 0x08, 0x32, 0xfa, 0x03, 0x96,
            0x01, 0x33, 0x00, 0x00, 0x50, 0xdd, 0x2b, 0xc0, 0xc5, 0xb6, 0x74, 0xca, 0xd8, 0xdd,
            0x1e, 0x69, 0x07, 0xce, 0x33, 0xfd, 0x25, 0xd4, 0x22, 0x57, 0x7a, 0xef, 0x53, 0xc2,
            0x1f, 0x35, 0xfc, 0x46, 0x74, 0xa4, 0xc2, 0x56, 0xc9, 0x9c, 0xa4, 0x25, 0x09, 0xd8,
            0xbc, 0x51, 0xbb, 0x40, 0xa8, 0x55, 0x90, 0xc8, 0x4e, 0xd4, 0x16, 0x3c, 0x6b, 0x73,
            0xa2, 0xcb, 0x66, 0x9d, 0x2e, 0x72, 0x44, 0xb0, 0x31, 0x1c, 0x07, 0xb5, 0x9e, 0x26,
            0xfa, 0x75, 0x8c, 0xe5, 0x96, 0xb4, 0xd8, 0x23, 0x1b, 0x55, 0x44, 0x76, 0x46, 0xf1,
            0x03, 0x15, 0x98, 0xb0, 0xfd, 0x41, 0x23, 0x02, 0xc3, 0x9b, 0x4e, 0xf0, 0x5f, 0x06,
            0x87, 0xc2, 0xf4, 0xb2, 0xd8, 0x10, 0x11, 0xf1, 0x9e, 0xc9, 0x07, 0x0f, 0xa1, 0xb0,
            0x47, 0x20, 0x92, 0xf2, 0xe2, 0xc9, 0xa9, 0x09, 0x6f, 0x43, 0x20, 0x7b, 0xf7, 0xe7,
            0x77, 0x90, 0x0e, 0x39, 0xd9, 0xf9, 0x25, 0xc2, 0x1c, 0x91, 0x02, 0xc2, 0xc5, 0x75,
            0x9a, 0xec, 0x95, 0xec, 0x54, 0xb5, 0x1c, 0x4c, 0x60, 0xd8, 0xa3, 0xd6, 0x18, 0xd5,
            0x6a, 0x28, 0x1c, 0xf4, 0x8c, 0x8b, 0x94, 0x18, 0x35, 0x81, 0xc4, 0xe7, 0x3c, 0x4f,
            0xc0, 0xad, 0x6c, 0x1b, 0xcd, 0x1a, 0x77, 0x5e, 0x2d, 0xf1, 0x91, 0x8a, 0xf1, 0xcc,
            0x18, 0xb9, 0x9f, 0x59, 0xcb, 0x31, 0x6e, 0x35, 0xc9, 0x1a, 0x0b, 0x69, 0xe7, 0xc2,
            0x07, 0xbc, 0x26, 0xd0, 0x03, 0x8d, 0xdf, 0xad, 0x69, 0x56, 0xd9, 0x90, 0xb7, 0x02,
            0x1a, 0x0b, 0xf7, 0x51, 0x44, 0xb3, 0x6c, 0x16, 0x0d, 0x6e, 0x4f, 0x61, 0xff, 0x31,
            0x8b, 0xae, 0xc5, 0x9e, 0xea, 0xb2, 0xc7, 0x48, 0x3f, 0xdb, 0x54, 0x30, 0x59, 0x0b,
            0x55, 0x00, 0x05, 0xdd, 0x00, 0x03, 0x90, 0x50, 0x24, 0x3b, 0xeb, 0x78, 0xea, 0x6b,
            0x12, 0x9a, 0x45, 0xbf, 0x41, 0x7f, 0xf4, 0xff, 0x82, 0x7c, 0x7c, 0xce, 0x36, 0x0f,
            0x3a, 0x27, 0x43, 0x32, 0x18, 0x71, 0x16, 0x7a, 0x32, 0xd4, 0xec, 0x65, 0xfc, 0xf2,
            0x77, 0x3f, 0xe9, 0x04, 0x4b, 0x1b, 0xd7, 0xca, 0xb9, 0xee, 0xb4, 0x63, 0x73, 0x41,
            0xc7, 0xba, 0x56, 0x26, 0x59, 0x39, 0xeb, 0xd1, 0xcc, 0x22, 0x7f, 0x93, 0x93, 0x80,
            0xb2, 0x01, 0xb7, 0x7f, 0x68, 0x65, 0xf9, 0xdb, 0x22, 0x56, 0x0e, 0xd5, 0x90, 0x93,
            0x84, 0xc7, 0xad, 0x34, 0x61, 0x36, 0xc4, 0x92, 0xba, 0x37, 0xbd, 0x6f, 0x4c, 0x01,
            0xe6, 0x67, 0xbf, 0x0a, 0x30, 0x7d, 0xb0, 0xf1, 0x5a, 0xa3, 0xf8, 0xa3, 0xba, 0xca,
            0xf9, 0xac, 0x09, 0xd5, 0xd4, 0x66, 0xbe, 0xfe, 0x2b, 0xc6, 0x6c, 0x2b, 0x4b, 0x6a,
            0x1c, 0x17, 0xfe, 0xca, 0x88, 0x81, 0x31, 0xbb, 0xf5, 0xac, 0x2f, 0xb8, 0x7c, 0x00,
            0xa6, 0x7d, 0x95, 0xd6, 0x5a, 0x21, 0xea, 0x39, 0x3c, 0x71, 0xa2, 0x5f, 0x42, 0xa6,
            0x2a, 0xff, 0x66, 0x72, 0x99, 0xfb, 0x1f, 0xa3, 0xe4, 0xf9, 0x86, 0x2d, 0x8c, 0x2f,
            0x0c, 0x6c, 0xb9, 0x1b, 0xcc, 0x73, 0xa7, 0x38, 0x2b, 0x92, 0xb8, 0x24, 0x4a, 0x69,
            0xdb, 0xb3, 0x6d, 0xbb, 0xa4, 0xde, 0xfd, 0x04, 0x12, 0x9b, 0x44, 0x8c, 0xc3, 0xfe,
            0x78, 0x95, 0xbb, 0xbc, 0xf5, 0xd2, 0x3f, 0xde, 0x58, 0x93, 0x86, 0xdd, 0xa9, 0xac,
            0x4a, 0x69, 0x16, 0xfd, 0x05, 0xff, 0xd3, 0xfe, 0x09, 0xee, 0x9a, 0x4c, 0x9e, 0x41,
            0xae, 0xf6, 0xb9, 0xdd, 0x85, 0xc7, 0x51, 0xfa, 0x13, 0x7e, 0x68, 0xff, 0x4b, 0x00,
            0x3c, 0xa5, 0x21, 0x8b, 0x2a, 0xb1, 0xc1, 0xee, 0x31, 0x02, 0xff, 0x15, 0x72, 0x48,
            0xaf, 0x7e, 0x87, 0xea, 0x0b, 0xcb, 0xdc, 0xaf, 0x29, 0x36, 0x8a, 0x5c, 0xbb, 0xb2,
            0xe8,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 23);
        assert_eq!(packed[64 * 64], 82);
        assert_eq!(packed[64 * 64 + 4 * 32 + 11], 95);
        assert_eq!(frame_digest(&packed), 0xdecb_74cf_64fd_bdda);
    }
    #[test]
    fn superblock_quantizer_delta_matches_dav1d() {
        // 128×128 keyframe. The frame header names a quantizer delta, and the
        // first coded block of a superblock may move the running index.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x19, 0xbf, 0xff, 0x00, 0x02, 0x32, 0xe2, 0x09, 0x92,
            0x80, 0x18, 0x00, 0x02, 0x80, 0x97, 0xc8, 0x64, 0x18, 0xb4, 0x40, 0xa8, 0xea, 0x64,
            0x29, 0xce, 0x3a, 0x97, 0x36, 0xba, 0x7f, 0xf9, 0xfc, 0xa5, 0xea, 0x67, 0x26, 0xab,
            0x77, 0x99, 0x2f, 0xfc, 0x70, 0x97, 0xef, 0x69, 0xcc, 0x60, 0x80, 0x53, 0x7f, 0x5e,
            0x6d, 0x6a, 0x62, 0xc2, 0x53, 0x18, 0x63, 0x2e, 0x61, 0xc3, 0x28, 0x4e, 0x4e, 0x8e,
            0x49, 0xf2, 0xee, 0x29, 0x58, 0xe4, 0x89, 0xd1, 0x0e, 0x84, 0x5b, 0xc6, 0xce, 0xec,
            0xf9, 0xf7, 0xe8, 0xce, 0x68, 0x10, 0x5c, 0x39, 0x08, 0x9e, 0x58, 0xbc, 0x69, 0x34,
            0x38, 0xf6, 0xc4, 0xab, 0x99, 0x82, 0x9e, 0x9b, 0x41, 0xba, 0x6d, 0x87, 0xcd, 0x82,
            0xbc, 0x88, 0x8f, 0xc7, 0x29, 0xee, 0xdb, 0xc8, 0x12, 0xf9, 0xdd, 0x62, 0x37, 0xf6,
            0x4b, 0x3b, 0xa5, 0xbc, 0x35, 0x93, 0x21, 0xe8, 0x87, 0xaf, 0xf6, 0x44, 0xe1, 0xaa,
            0x98, 0x45, 0xa0, 0x94, 0xa6, 0xec, 0xb2, 0xc3, 0xe2, 0x5b, 0x2b, 0x4e, 0x73, 0x9c,
            0x6c, 0x21, 0x3b, 0x15, 0x87, 0x7d, 0x0c, 0xe3, 0x6d, 0x98, 0x97, 0x08, 0x10, 0xe4,
            0xe2, 0xf1, 0x30, 0xe8, 0x3e, 0x3a, 0xbb, 0x2c, 0x12, 0xe0, 0x3b, 0x6f, 0x9b, 0xd3,
            0x13, 0xce, 0xa8, 0x40, 0x56, 0xa2, 0x58, 0x56, 0x91, 0x5f, 0x9e, 0x22, 0xe2, 0x51,
            0x65, 0xd8, 0xa6, 0x57, 0xe7, 0x55, 0xaa, 0xf7, 0x6f, 0xd2, 0xac, 0x7e, 0xbe, 0xe5,
            0xdc, 0xa4, 0x02, 0xfd, 0x1f, 0xef, 0x01, 0xbc, 0x9a, 0xdf, 0x87, 0x21, 0xec, 0x83,
            0x2d, 0x00, 0x8a, 0xf3, 0x86, 0x5b, 0x8e, 0xd5, 0x2c, 0xdf, 0xce, 0x60, 0xb2, 0x63,
            0xe4, 0x57, 0xf7, 0xf6, 0xb9, 0x40, 0xc0, 0x03, 0xb8, 0x0f, 0xc7, 0xd9, 0x6e, 0x3b,
            0x65, 0x30, 0x1f, 0x1d, 0xb4, 0xb9, 0x92, 0xaa, 0xee, 0x7d, 0x47, 0x18, 0x59, 0x36,
            0xc6, 0xa3, 0xa3, 0xd6, 0xcc, 0x64, 0x1a, 0x1e, 0xb2, 0x63, 0x07, 0x52, 0x41, 0x44,
            0x58, 0xd4, 0x0e, 0x13, 0x14, 0x3c, 0xee, 0x86, 0xc2, 0x3a, 0xcd, 0x4f, 0x62, 0xdc,
            0x12, 0x2e, 0xb6, 0x9f, 0x7d, 0x37, 0x27, 0x8c, 0x9c, 0x11, 0x51, 0xe9, 0xf0, 0x8e,
            0x6a, 0x9b, 0xe4, 0xba, 0x08, 0xea, 0xa6, 0x9d, 0x81, 0x50, 0x38, 0x5c, 0x9e, 0x1f,
            0x1d, 0x6a, 0x01, 0xe1, 0x18, 0x0a, 0x92, 0x6b, 0x49, 0xb6, 0xc5, 0x0f, 0xa1, 0x9d,
            0xb0, 0xf4, 0x03, 0x58, 0x2d, 0x43, 0x5b, 0xdc, 0xac, 0x63, 0x6f, 0x05, 0x51, 0xf6,
            0xdf, 0x7e, 0x8b, 0xd4, 0x00, 0x14, 0x94, 0x9c, 0x93, 0x89, 0xb6, 0xc8, 0x86, 0x5f,
            0xa1, 0x03, 0xe4, 0x8b, 0x0b, 0x97, 0xb1, 0x69, 0x23, 0x51, 0x09, 0x97, 0x0e, 0x9d,
            0xa9, 0xbf, 0xa4, 0x6f, 0x68, 0xec, 0x18, 0x7c, 0xdc, 0x5a, 0xe3, 0x34, 0x16, 0x85,
            0x33, 0x25, 0x8b, 0x47, 0x7e, 0x54, 0x74, 0x8c, 0x77, 0xe5, 0x2d, 0x59, 0xbf, 0x26,
            0x5c, 0xb9, 0x46, 0x4b, 0xb1, 0x6c, 0x27, 0x38, 0x0e, 0xd1, 0x30, 0x49, 0xb4, 0xda,
            0x33, 0xa5, 0x6d, 0xdb, 0xc5, 0xaf, 0x46, 0x1f, 0xa0, 0x2b, 0xea, 0x94, 0x84, 0x61,
            0x7e, 0x72, 0x56, 0x63, 0xa6, 0x56, 0x64, 0x8d, 0xcd, 0xff, 0xfc, 0xfe, 0x52, 0xf5,
            0x33, 0xca, 0x5b, 0x7d, 0x76, 0xa2, 0x37, 0x3a, 0x70, 0x25, 0xe9, 0xf3, 0x40, 0x75,
            0x81, 0x28, 0x52, 0x0e, 0x3a, 0xaa, 0xbc, 0xb2, 0xea, 0xe5, 0x1a, 0x7d, 0xa4, 0x20,
            0xf7, 0x45, 0x50, 0x24, 0xbb, 0x1e, 0x34, 0xae, 0x5f, 0x43, 0xa7, 0x47, 0xf2, 0x66,
            0x85, 0x52, 0xf0, 0x3b, 0x92, 0x44, 0xa3, 0x33, 0x3c, 0x1f, 0xd9, 0xf5, 0x9e, 0x67,
            0x4c, 0xf5, 0x76, 0x08, 0x81, 0x1c, 0x31, 0x61, 0x82, 0xd9, 0xb3, 0x71, 0x27, 0x22,
            0xa9, 0x9b, 0x0b, 0x28, 0x75, 0x10, 0xb4, 0x58, 0x30, 0x3f, 0x69, 0x1d, 0xab, 0x54,
            0xd9, 0xda, 0x98, 0x23, 0x1f, 0x74, 0x2b, 0xea, 0x7c, 0x8f, 0x2b, 0x42, 0x0f, 0xa9,
            0x19, 0x08, 0x7d, 0x72, 0x3b, 0x12, 0x8d, 0x3b, 0xcc, 0x6f, 0xa2, 0x18, 0xc5, 0xfb,
            0x52, 0xb6, 0x00, 0x68, 0x82, 0x49, 0xfe, 0x74, 0x22, 0xae, 0x8a, 0x09, 0x66, 0x12,
            0x18, 0x27, 0x4e, 0xd7, 0xdc, 0xdd, 0x1f, 0x54, 0x32, 0xe3, 0x47, 0x29, 0x80, 0x91,
            0x30, 0x19, 0xc6, 0xeb, 0xed, 0xc5, 0x89, 0xd6, 0xb8, 0x5b, 0xb6, 0xcb, 0x70, 0x17,
            0xe3, 0x64, 0xac, 0x09, 0x35, 0xaf, 0x99, 0x1f, 0xcc, 0x1d, 0xa8, 0x77, 0x7b, 0x11,
            0x82, 0x25, 0x8f, 0xcc, 0x02, 0x73, 0x06, 0xdc, 0x64, 0x63, 0xce, 0x94, 0xa4, 0x62,
            0x4f, 0xdc, 0xe9, 0x52, 0x8a, 0x69, 0x8c, 0xd3, 0x6e, 0x4e, 0xfc, 0x7d, 0x75, 0xe1,
            0x4d, 0x13, 0xfa, 0xec, 0x05, 0x8a, 0x72, 0x06, 0xf6, 0xac, 0x0b, 0x90, 0x2e, 0x80,
            0xf0, 0x23, 0x9d, 0xb9, 0x73, 0x3e, 0x8b, 0xec, 0xd3, 0x68, 0x2f, 0x08, 0xe4, 0x8c,
            0x13, 0x97, 0x4a, 0x09, 0x3c, 0xc2, 0xf5, 0x2b, 0x55, 0x6f, 0x60, 0xf1, 0x8c, 0x8b,
            0xfb, 0xfa, 0x25, 0x1c, 0xa3, 0x66, 0xb7, 0xa0, 0x45, 0xdc, 0x30, 0x78, 0xfa, 0xf3,
            0x6a, 0x17, 0x6d, 0xa9, 0xcd, 0xc0, 0xe7, 0x3b, 0x52, 0x68, 0x92, 0xb0, 0xb2, 0x52,
            0x85, 0x83, 0xac, 0xde, 0xf4, 0x70, 0x59, 0x4c, 0x6e, 0x04, 0xab, 0x3d, 0x16, 0xab,
            0x7c, 0x32, 0x11, 0x2d, 0x1b, 0x8f, 0x18, 0x08, 0xe1, 0x20, 0x80, 0x0c, 0xd6, 0x09,
            0xdd, 0xf4, 0x89, 0x08, 0xc2, 0x84, 0x2d, 0xda, 0xc1, 0xb0, 0x50, 0x1b, 0xeb, 0x66,
            0x3b, 0xe7, 0x71, 0x62, 0x7c, 0xe9, 0x42, 0xde, 0xd2, 0x02, 0x27, 0x98, 0xe4, 0x7a,
            0x32, 0x0b, 0xab, 0x3e, 0xeb, 0xda, 0x50, 0x55, 0x2a, 0x3a, 0x85, 0x76, 0x3e, 0x87,
            0x99, 0xb5, 0xda, 0x01, 0x1e, 0xba, 0x74, 0xe0, 0xd0, 0x27, 0x62, 0x1a, 0x1f, 0xae,
            0xbc, 0x76, 0xf0, 0xc9, 0xd1, 0xcc, 0x6a, 0xe7, 0x59, 0x0e, 0xb6, 0xf6, 0x48, 0x37,
            0x1e, 0x6d, 0x00, 0xcf, 0xe8, 0xa8, 0xcf, 0xd5, 0x5f, 0x16, 0x4c, 0xba, 0x1d, 0xe5,
            0x9d, 0x75, 0x1b, 0xde, 0x2c, 0x2a, 0x4d, 0x10, 0xff, 0x77, 0x1b, 0x4b, 0x5a, 0x02,
            0x83, 0xce, 0x35, 0xa9, 0x83, 0xee, 0xfa, 0x74, 0xbf, 0x8e, 0x1f, 0x31, 0xe7, 0x4f,
            0x6f, 0xc9, 0xa0, 0x52, 0x65, 0x61, 0x2e, 0xd1, 0x39, 0xfe, 0x23, 0x2b, 0x25, 0xe2,
            0xba, 0xf1, 0x73, 0xff, 0x67, 0xea, 0x0f, 0x21, 0x05, 0x16, 0xda, 0x23, 0x27, 0x6b,
            0x9f, 0x01, 0x2e, 0x3f, 0x5a, 0xaa, 0x06, 0x46, 0xa4, 0xd5, 0x80, 0x5c, 0xbe, 0x61,
            0xfd, 0x3a, 0xd4, 0x73, 0x83, 0xa7, 0x91, 0x3e, 0xb8, 0x2e, 0x01, 0xd9, 0x70, 0x48,
            0x10, 0xfe, 0x98, 0x79, 0x39, 0x85, 0x11, 0x83, 0x7a, 0x45, 0x40, 0x49, 0xe8, 0x4b,
            0xcf, 0x97, 0x50, 0x3c, 0x7f, 0x31, 0xa1, 0x2c, 0xf2, 0x4d, 0x37, 0x6c, 0x88, 0xe3,
            0xe6, 0x20, 0x15, 0x77, 0x23, 0xd6, 0x4a, 0xe4, 0xe8, 0xe4, 0x9f, 0x2e, 0xe2, 0x95,
            0x8e, 0x48, 0x6d, 0x35, 0x16, 0xdf, 0x0a, 0x0b, 0xce, 0x90, 0xe5, 0xcc, 0x9f, 0x2f,
            0xb5, 0x48, 0x15, 0x1c, 0xe1, 0x54, 0xa2, 0x87, 0xc4, 0xaa, 0x6e, 0x1c, 0x12, 0xf1,
            0xd5, 0x4a, 0xf6, 0x1a, 0xed, 0x5a, 0xca, 0xbb, 0x07, 0xa2, 0x1b, 0x3c, 0x14, 0xc9,
            0xf1, 0x4b, 0x7c, 0x30, 0x9e, 0x56, 0x33, 0x79, 0xcd, 0x31, 0xb4, 0x46, 0x1e, 0xda,
            0xc6, 0xca, 0xa0, 0xba, 0xb1, 0xa0, 0x32, 0xc9, 0x09, 0xf0, 0xc0, 0x2e, 0x7e, 0x93,
            0x50, 0x75, 0x38, 0x3a, 0x34, 0x83, 0x08, 0xae, 0x07, 0x95, 0xbf, 0x04, 0xec, 0x27,
            0x9d, 0x57, 0xd3, 0x50, 0x26, 0x12, 0x92, 0x5e, 0xf3, 0xf9, 0xd7, 0xe1, 0x34, 0xb0,
            0x9f, 0x7a, 0x1b, 0xca, 0xb8, 0xc4, 0xf4, 0xd5, 0x9d, 0x9c, 0x64, 0x13, 0xa1, 0x6c,
            0x04, 0x53, 0xa8, 0xb1, 0x8c, 0xc6, 0x2e, 0x1e, 0x1d, 0xd1, 0xc2, 0x9e, 0xeb, 0x4c,
            0x34, 0x5b, 0x2b, 0x6f, 0x29, 0xb1, 0x99, 0x89, 0xec, 0x58, 0xa6, 0x27, 0x7e, 0x4f,
            0xbb, 0x4e, 0x4f, 0x40, 0x4d, 0xe0, 0x0d, 0xd0, 0x23, 0xbf, 0xb4, 0x13, 0x6d, 0x76,
            0x53, 0xda, 0x3a, 0xe6, 0xd3, 0xa8, 0x13, 0xad, 0x4f, 0xad, 0x79, 0xbf, 0xff, 0xf5,
            0xb2, 0x58, 0x67, 0xed, 0xbb, 0x35, 0xc8, 0xc7, 0x1c, 0xc0, 0x4a, 0xc0, 0xad, 0x68,
            0x76, 0xa1, 0x37, 0xbc, 0x48, 0x7f, 0xec, 0x69, 0xd4, 0x2c, 0x4a, 0x2c, 0xb7, 0xe0,
            0x60, 0x98, 0x5a, 0x9e, 0x13, 0x73, 0x47, 0xef, 0x3c, 0x57, 0x4b, 0x28, 0xf4, 0xb8,
            0xfb, 0x74, 0xe4, 0x8c, 0x1f, 0x65, 0x58, 0x76, 0x46, 0xea, 0x9d, 0x8c, 0xe2, 0x74,
            0x0e, 0x53, 0x6d, 0xcb, 0xce, 0x30, 0x7f, 0xe9, 0x8e, 0x70, 0x6e, 0xda, 0xb0, 0xc8,
            0x3c, 0x70, 0xee, 0xc5, 0x71, 0xd8, 0xa6, 0x73, 0xbf, 0xa9, 0xe4, 0x06, 0x52, 0xdc,
            0x72, 0xc4, 0x93, 0xe5, 0xdc, 0x48, 0xd6, 0x3b, 0xbf, 0x37, 0x90, 0x5e, 0xa0, 0x03,
            0x65, 0xf8, 0xb8, 0x32, 0xf0, 0xd2, 0xcc, 0xd5, 0x8f, 0xd0, 0x84, 0xa9, 0x19, 0xcf,
            0x1c, 0x6f, 0xf7, 0x41, 0x4e, 0xd1, 0xc1, 0xbf, 0x66, 0x83, 0xff, 0xf0, 0xa0, 0x41,
            0xc5, 0x95, 0xd5, 0xcd, 0xbf, 0xa1, 0xef, 0xa6, 0xa4, 0x60, 0x98, 0xca, 0xc4, 0x6f,
            0x6b, 0xbd, 0x20,
        ];
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 128 * 128 * 3 / 2);
        assert_eq!(packed[0], 240);
        assert_eq!(packed[64], 205);
        assert_eq!(frame_digest(&packed), 0x9c5f_415c_3c06_47ed);
    }
    #[test]
    fn a_filter_intra_mode_picture_matches_dav1d() {
        // aomenc, 64×64 vertical gradient, max partition 16, other intra modes
        // off. Twenty-five blocks select filter intra. CDEF, restoration, and
        // super resolution are off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p`.
        let file = include_bytes!("../fixtures/filter-intra.ivf");
        let frames = ivf_payloads(file);
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert!(sequence.enable_filter_intra);
        let expected = [0x6ab9_1134_1ed1_2065, 0x9dcc_ede2_e005_dee5];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn filter_intra_matches_dav1d() {
        // 64×64 keyframe. Two luma blocks use the recursive intra filter
        // (horizontal and Paeth). The shown picture matches dav1d.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfd, 0x80, 0x08, 0x32, 0xec, 0x03, 0x95,
            0x00, 0x00, 0x00, 0x50, 0xd8, 0x01, 0x50, 0x77, 0x67, 0x34, 0x6f, 0xff, 0xed, 0x02,
            0x5b, 0xe6, 0x54, 0x14, 0xfd, 0xed, 0x0f, 0x7f, 0x33, 0xdd, 0xc4, 0xe1, 0xfc, 0x9f,
            0x8f, 0x28, 0x52, 0x1f, 0x17, 0x99, 0xd0, 0x50, 0xfc, 0x7d, 0x6a, 0x02, 0x9a, 0x9e,
            0xe5, 0x22, 0x2e, 0x6f, 0xc3, 0xca, 0x78, 0xd2, 0x42, 0xc8, 0x5e, 0xc3, 0xfa, 0xaa,
            0xc4, 0x87, 0xa7, 0x74, 0x75, 0x71, 0xa4, 0x3d, 0x55, 0x1c, 0x09, 0x15, 0xb7, 0x2b,
            0x3b, 0x3c, 0x25, 0x50, 0x0e, 0x51, 0x2d, 0x39, 0x51, 0x02, 0x45, 0x62, 0x31, 0x0b,
            0x79, 0xb3, 0x92, 0x52, 0x05, 0x2c, 0x2c, 0x00, 0x60, 0xa0, 0x79, 0x5d, 0x3a, 0x5a,
            0xc5, 0x2f, 0xd9, 0xd3, 0x89, 0xc8, 0x00, 0xdd, 0x3e, 0xb4, 0xb2, 0x79, 0x6f, 0x01,
            0x90, 0x13, 0xcc, 0x2e, 0x9e, 0xad, 0x20, 0xe9, 0xd9, 0x95, 0x40, 0xdc, 0xe3, 0x7f,
            0xf5, 0x92, 0xc3, 0x4b, 0x39, 0x30, 0x57, 0x62, 0x2a, 0xf7, 0x68, 0xb5, 0xdc, 0x43,
            0x0c, 0x09, 0x47, 0xfc, 0x5d, 0xf1, 0x27, 0x1a, 0xe7, 0xaf, 0x9a, 0xb3, 0x1b, 0x06,
            0x35, 0x29, 0x84, 0xf8, 0x56, 0xfe, 0x86, 0xe0, 0xd5, 0x06, 0x15, 0x22, 0x72, 0x83,
            0x04, 0xbb, 0x62, 0x13, 0x21, 0x4b, 0x9a, 0x5b, 0x51, 0x30, 0xac, 0x65, 0xfe, 0x76,
            0x18, 0x18, 0x36, 0x78, 0xe2, 0x39, 0xcc, 0x9d, 0xd8, 0x32, 0xb0, 0xc1, 0x3c, 0xdf,
            0x14, 0xa8, 0xfe, 0xaf, 0x56, 0x2a, 0xd0, 0xb6, 0x91, 0x12, 0xb0, 0xe2, 0xc9, 0x56,
            0x09, 0xa6, 0x1a, 0x0c, 0x77, 0xa2, 0x6d, 0x8a, 0xaa, 0x42, 0x94, 0x3c, 0x1e, 0x6f,
            0xaf, 0xc6, 0x94, 0x24, 0xf3, 0x55, 0x57, 0x12, 0x0c, 0xf1, 0xd8, 0xc1, 0x29, 0x42,
            0x16, 0x17, 0x63, 0xc0, 0xd4, 0x14, 0xf4, 0xdf, 0xa2, 0x7f, 0x9e, 0x50, 0x92, 0x04,
            0x0c, 0xe1, 0x80, 0x25, 0x52, 0x81, 0x02, 0xe9, 0xa5, 0xa9, 0x6a, 0xae, 0x52, 0x35,
            0x20, 0xce, 0x5a, 0xcf, 0xff, 0xdf, 0xb0, 0x54, 0xe8, 0x38, 0x39, 0x30, 0x3e, 0xd5,
            0xc1, 0xa3, 0x79, 0xd8, 0x80, 0x4a, 0xbf, 0xba, 0xff, 0x48, 0x91, 0x70, 0x39, 0xb8,
            0xdc, 0xf7, 0xcc, 0xbb, 0x2c, 0x1a, 0x92, 0xce, 0xed, 0x1f, 0xde, 0x19, 0x64, 0x30,
            0x2e, 0x47, 0x7b, 0x7f, 0x3c, 0x75, 0x8d, 0xc8, 0xd7, 0x0c, 0x7e, 0x1f, 0xb2, 0xb7,
            0x3e, 0xe7, 0x60, 0xd7, 0x29, 0x72, 0x6e, 0xb1, 0x37, 0x5a, 0x7e, 0xa9, 0x30, 0x65,
            0xba, 0xf5, 0x0f, 0x26, 0xbd, 0x49, 0xde, 0x43, 0xb5, 0x61, 0x98, 0x42, 0xb5, 0x7d,
            0x04, 0xc6, 0x9c, 0xd2, 0x41, 0xed, 0x2c, 0xde, 0x76, 0xbf, 0xde, 0xd0, 0xfe, 0x33,
            0x53, 0x19, 0xb9, 0x9d, 0x81, 0x78, 0xd6, 0x29, 0x1c, 0xa3, 0x9e, 0x2d, 0xb8, 0xda,
            0xa6, 0xd9, 0xe9, 0x4a, 0xdb, 0x29, 0x54, 0x38, 0xab, 0x08, 0x09, 0xb3, 0x30, 0xc1,
            0x60, 0x08, 0xa6, 0xed, 0x34, 0xd4, 0x5f, 0xdb, 0xcb, 0x35, 0x58, 0x78, 0x91, 0x4e,
            0xc1, 0x8f, 0x54, 0xa6, 0xad, 0x23, 0xa8, 0x80, 0x5f, 0x4d, 0x84, 0xd9, 0x53, 0xef,
            0xff, 0x81, 0xd1, 0x08, 0x40, 0x9e, 0x88, 0x1f, 0x9f, 0x62, 0xd3, 0xa8, 0x27, 0xcd,
            0x00, 0x11, 0xb3, 0x40, 0x4c, 0x42, 0x18, 0x5d, 0x9e, 0x7e, 0x74, 0xe2, 0xb9, 0x85,
            0xd4, 0xf1, 0xe2, 0x20, 0x04, 0xa4, 0xee, 0x95, 0xff, 0x80, 0xb1, 0x64, 0xf6, 0x0e,
            0x9a, 0xad, 0xf4, 0x3e, 0xc8, 0x52, 0x60, 0x67, 0x05, 0xeb, 0x02, 0xb5, 0x03, 0x23,
            0x56, 0x53, 0x46, 0xc7, 0xc6, 0x09, 0x47, 0x95, 0x2c, 0xba, 0x20, 0xc9, 0xf5, 0x37,
            0x50,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 42);
        assert_eq!(frame_digest(&packed), 0x7006_b339_44a6_241d);
    }

    #[test]
    fn directional_transforms_match_dav1d() {
        // 64×64 keyframe. Transform size is selected and the extended intra
        // set is available. The shown picture matches dav1d.
        let payload = [
            0x12, 0x00, 0x0a, 0x06, 0x18, 0x15, 0x7f, 0xfc, 0x80, 0x08, 0x32, 0x8f, 0x05, 0x9a,
            0x00, 0x12, 0x14, 0xd5, 0xc5, 0x00, 0xdd, 0x31, 0x63, 0xb1, 0x94, 0x5f, 0x5c, 0xe1,
            0xcd, 0xb4, 0xec, 0x7a, 0xbf, 0x0a, 0xed, 0x08, 0x41, 0xc7, 0x45, 0xed, 0x15, 0xf2,
            0x83, 0x81, 0x8f, 0x80, 0x11, 0xbb, 0x13, 0xcb, 0xd0, 0xed, 0xd5, 0x95, 0x35, 0x46,
            0x48, 0x1e, 0xe2, 0xab, 0xc0, 0x72, 0xac, 0x13, 0x04, 0x09, 0x1e, 0xf3, 0xc4, 0x66,
            0x71, 0x2e, 0x4a, 0xc9, 0x2f, 0x47, 0x5f, 0xc8, 0x99, 0x6d, 0x55, 0x08, 0xd5, 0xac,
            0xbb, 0xeb, 0xe1, 0x6f, 0xb8, 0x6d, 0xd3, 0xe0, 0xe3, 0xc1, 0xc8, 0x05, 0xd7, 0x53,
            0x54, 0x1e, 0xa6, 0x8a, 0xc2, 0x07, 0x8f, 0x05, 0x54, 0xd8, 0x82, 0xb3, 0x77, 0xc5,
            0x98, 0xbc, 0xc0, 0xad, 0x0b, 0x90, 0x01, 0xae, 0x74, 0x7e, 0x8b, 0xf9, 0xfc, 0xd8,
            0x1a, 0xf1, 0xbf, 0xfb, 0xdf, 0xbc, 0xf0, 0x8b, 0x1d, 0xff, 0x23, 0xdd, 0x62, 0x42,
            0x13, 0x66, 0x71, 0x40, 0xd5, 0x63, 0x29, 0x07, 0x31, 0xf1, 0x7a, 0x97, 0x9c, 0x65,
            0x96, 0xc5, 0xa6, 0xa8, 0x5d, 0x29, 0x60, 0x27, 0xad, 0x4f, 0x78, 0x39, 0x6a, 0xa5,
            0x3c, 0xca, 0x56, 0x92, 0x19, 0xb1, 0x04, 0x6c, 0x69, 0xe9, 0x2c, 0x65, 0xc8, 0xbe,
            0xa3, 0xf8, 0xe9, 0xe9, 0x35, 0xec, 0xe2, 0xbf, 0x77, 0x7f, 0xd1, 0x7e, 0x39, 0xc9,
            0x67, 0x03, 0x2f, 0x18, 0x79, 0x7b, 0x28, 0xf3, 0xdf, 0x2c, 0xa4, 0xf7, 0x5a, 0xa1,
            0x04, 0x95, 0x4c, 0x85, 0x0d, 0x8e, 0xfb, 0x92, 0x19, 0xd0, 0x43, 0x6a, 0x4e, 0x78,
            0x1e, 0x44, 0x1b, 0x6a, 0xe8, 0xf0, 0x50, 0x1b, 0xa8, 0x91, 0x84, 0x3b, 0x25, 0xcc,
            0xe2, 0x37, 0x84, 0xf7, 0x45, 0x71, 0x11, 0xe7, 0xd6, 0xde, 0xfc, 0xae, 0x6b, 0x56,
            0x4e, 0xdf, 0x01, 0xd0, 0x5a, 0x70, 0xdf, 0x53, 0x6b, 0xd7, 0xf8, 0xb1, 0xdd, 0xc6,
            0x24, 0xf1, 0xf0, 0x97, 0x73, 0xa8, 0x3b, 0x30, 0x1b, 0x71, 0xd6, 0x34, 0x7c, 0x9d,
            0xf1, 0xb4, 0x73, 0x24, 0xca, 0x93, 0xbb, 0x23, 0x6f, 0xd4, 0xa9, 0xee, 0x1f, 0x8d,
            0xf0, 0x88, 0xf6, 0xa1, 0xd5, 0xfb, 0x6b, 0x79, 0xd4, 0xcd, 0xa5, 0xd6, 0xfd, 0xd9,
            0x33, 0xce, 0x55, 0x83, 0xe5, 0xf1, 0x05, 0x34, 0x8e, 0x0f, 0x53, 0x9c, 0x01, 0x0d,
            0x20, 0xd7, 0xc6, 0x2a, 0xd2, 0x2d, 0x1a, 0x48, 0x15, 0x9f, 0x3d, 0x1c, 0xae, 0x07,
            0x97, 0x3c, 0xd1, 0xe3, 0xde, 0xda, 0xdd, 0xb1, 0xf6, 0xad, 0xd1, 0x5e, 0x74, 0x0f,
            0xc6, 0xa8, 0x93, 0x50, 0x4c, 0x70, 0x7b, 0xcf, 0x8e, 0xc2, 0xc1, 0x7c, 0x1b, 0x82,
            0xa0, 0x73, 0xb3, 0x15, 0x24, 0x60, 0x98, 0xb0, 0x6e, 0x97, 0x57, 0x3a, 0x48, 0xf2,
            0x45, 0x53, 0x43, 0xcf, 0xac, 0xcc, 0xa9, 0xd2, 0xb9, 0x3d, 0x8a, 0x3a, 0xca, 0x8f,
            0x02, 0x0e, 0x5b, 0x45, 0xf4, 0xd0, 0xe5, 0x68, 0x9a, 0x57, 0xee, 0x22, 0xbf, 0x9e,
            0xb4, 0x84, 0x34, 0x5c, 0x56, 0x92, 0xcc, 0xc3, 0x2f, 0xe1, 0xb0, 0x0c, 0xd5, 0x6f,
            0x0d, 0x8c, 0x56, 0x27, 0x0f, 0xfd, 0xc4, 0x93, 0xff, 0xcc, 0x32, 0xd7, 0x33, 0x6c,
            0x91, 0x69, 0x84, 0x89, 0xa9, 0x3a, 0x78, 0x68, 0x6c, 0xf1, 0x87, 0x16, 0x59, 0x94,
            0x78, 0x2d, 0x47, 0x5c, 0xe5, 0xc7, 0x2c, 0x48, 0xd3, 0x94, 0xbb, 0x7c, 0x3b, 0xc6,
            0x83, 0x00, 0x76, 0x17, 0x8c, 0xc7, 0xf5, 0x7b, 0xb5, 0xa4, 0xb4, 0x95, 0xb0, 0xae,
            0x9c, 0x0a, 0x08, 0x99, 0xa9, 0x68, 0xff, 0x31, 0x6b, 0xa7, 0x82, 0xd5, 0xae, 0xc3,
            0x82, 0x65, 0x2e, 0x39, 0xe9, 0x48, 0x9f, 0x7b, 0x4c, 0x0e, 0x39, 0x2f, 0x25, 0x1e,
            0x34, 0x51, 0xe3, 0x4a, 0x4c, 0xd9, 0xaf, 0x22, 0x25, 0x69, 0x12, 0xb0, 0xc8, 0x2c,
            0xb4, 0xab, 0xd2, 0xb3, 0x1d, 0x7d, 0x42, 0x65, 0x35, 0x98, 0xa5, 0x89, 0xf2, 0x29,
            0xe3, 0x1b, 0x9f, 0xcf, 0x24, 0x90, 0x92, 0xa3, 0x1f, 0x98, 0xb0, 0x93, 0x7e, 0x74,
            0xa6, 0xab, 0xbe, 0x34, 0x8a, 0x6b, 0x6b, 0x35, 0xd0, 0xd4, 0x5f, 0xc9, 0x89, 0xcc,
            0xbf, 0x9f, 0xe0, 0x58, 0x1c, 0xc8, 0x39, 0x98, 0xbf, 0xcf, 0xc4, 0xff, 0xaf, 0xde,
            0x6f, 0xce, 0x71, 0xed, 0x34, 0x5e, 0xac, 0xf0, 0x4f, 0x0b, 0xb1, 0x08, 0x7c, 0xae,
            0x0c, 0x83, 0x2b, 0x87, 0xc4, 0x5f, 0x92, 0x11, 0x12, 0x32, 0x69, 0x3c, 0x98, 0xcb,
            0x85, 0xe1, 0xe0, 0x29, 0xea, 0x0c, 0xc4, 0x9f, 0xd4, 0xc6, 0x59, 0x42, 0x27, 0x49,
            0x00, 0x0d, 0x55, 0xc3, 0xc7, 0x38, 0x8b, 0x57, 0x08, 0xbc, 0xb2, 0xc0, 0xed, 0xe2,
            0x13, 0x71, 0x50, 0x56, 0xcc, 0xa4, 0xf4, 0x98, 0xeb, 0x25, 0x5b, 0x77, 0x22, 0xd1,
            0xaa, 0xa6, 0x76, 0xe4, 0x68, 0xd3, 0xeb, 0x8b, 0xed, 0xc0, 0x8c, 0xc9, 0x9c, 0x26,
            0xe2, 0xff, 0x05, 0x9c, 0xf4, 0x67, 0x4d, 0x00, 0xdc, 0xc0,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &payload, 0);
        assert_eq!(packed.len(), 64 * 64 * 3 / 2);
        assert_eq!(packed[0], 51);
        assert_eq!(frame_digest(&packed), 0x5e0f_45d9_39c1_a370);
    }

    #[test]
    fn rotzoom_global_motion_matches_dav1d() {
        // aomenc 128×128. The inter frame uses a rotation-zoom model, high
        // precision motion vectors, and switchable filters in both directions.
        // Some blocks use a different horizontal filter from the vertical one.
        // Digests are `ffmpeg -c:v libdav1d -threads 1` of those three frames.
        let frames: [&[u8]; 3] = [
            include_bytes!("../fixtures/rotzoom_0.obu"),
            include_bytes!("../fixtures/rotzoom_1.obu"),
            include_bytes!("../fixtures/rotzoom_2.obu"),
        ];
        let expected = [
            0x453f_7daf_99a0_ebd2,
            0xd04f_763d_2dee_d1f2,
            0xb89c_65a9_84a2_b011,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(seq.enable_dual_filter);
            if index == 1 {
                assert!(frame.allow_high_precision_mv);
                assert!(frame.filter_switchable);
                assert_eq!(frame.gm_type[1], 2);
                assert_eq!(
                    frame.gm_params[1],
                    [613_376, 46_080, 66_148, 1_216, -1_216, 66_148]
                );
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 128 * 128 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn a_short_reference_picture_matches_dav1d() {
        // aomenc 64×64, six frames, constant quality 10. CDEF, restoration, and
        // super resolution are off. The last frame's header is rewritten so it
        // signals only LAST and GOLDEN. The other five names are filled from
        // the order hints stored in the eight slots. Blocks in that frame
        // predict from LAST2, which is slot 2. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p`.
        let file = include_bytes!("../fixtures/short-refs.ivf");
        let frames = ivf_payloads(file);
        assert_eq!(frames.len(), 6);
        let mut sequence = None;
        let mut hints = [0u32; 8];
        let mut sizes = [None; 8];
        let mut last = None;
        for payload in &frames {
            let (seq, frame) = crate::header::parse_with_motion(
                payload,
                sequence.as_ref(),
                &[None; 8],
                &hints,
                &sizes,
                &[crate::header::Inherited::defaults(); 8],
            )
            .expect("parse");
            let size = Some((
                frame.upscaled_width,
                frame.height,
                frame.render_width,
                frame.render_height,
            ));
            for slot in 0..8 {
                if frame.refresh_flags & (1u8 << slot) != 0 {
                    hints[slot] = frame.order_hint;
                    sizes[slot] = size;
                }
            }
            last = Some(frame);
            sequence = Some(seq);
        }
        let frame = last.expect("frame");
        assert!(!frame.intra);
        assert_eq!(frame.order_hint, 5);
        assert_eq!(frame.ref_frames, [3, 2, 1, 4, 7, 6, 5]);
        let expected = [
            (64 * 64 * 3 / 2, 0xe485_041a_9516_0325),
            (64 * 64 * 3 / 2, 0x99f9_f1e8_464b_d462),
            (64 * 64 * 3 / 2, 0x63e8_8322_e74e_6325),
            (64 * 64 * 3 / 2, 0x256a_1da7_2db1_44af),
            (64 * 64 * 3 / 2, 0xd421_caac_351f_1c9c),
            (64 * 64 * 3 / 2, 0x63e8_8322_e74e_6325),
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), expected[index].0, "frame {index}");
            assert_eq!(frame_digest(&packed), expected[index].1, "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn ten_bit_restoration_on_444_and_422_matches_dav1d() {
        // aomenc --enable-restoration=1 --bit-depth=10, one frame, cq 40.
        // Profile 1 is 512×128 4:4:4. Luma is self-guided. Each chroma plane
        // is switchable, in 64-sample units, and the frame updates those CDFs.
        // Profile 2 is 256×512 4:2:2. Luma restoration is off. Each chroma
        // plane is self-guided, in 256-sample units, so the full-height chroma
        // has two rows. Digests are `ffmpeg -c:v libdav1d -threads 1`.
        let cases = [
            (
                include_bytes!("../fixtures/restore-444-10.ivf").as_slice(),
                1u8,
                false,
                false,
                512u32,
                128u32,
                [
                    crate::header::RESTORE_SGRPROJ,
                    crate::header::RESTORE_SWITCHABLE,
                    crate::header::RESTORE_SWITCHABLE,
                ],
                [64u16, 64, 64],
                cutlass_decode::PixelFormat::Yuv444p10,
                512 * 128 * 6,
                0x362f_dff0_9fd5_194d,
            ),
            (
                include_bytes!("../fixtures/restore-422-10.ivf").as_slice(),
                2,
                true,
                false,
                256,
                512,
                [
                    crate::header::RESTORE_NONE,
                    crate::header::RESTORE_SGRPROJ,
                    crate::header::RESTORE_SGRPROJ,
                ],
                [256, 256, 256],
                cutlass_decode::PixelFormat::Yuv422p10,
                256 * 512 * 4,
                0xf32e_8649_9a71_9522,
            ),
        ];
        for (file, profile, sub_x, sub_y, width, height, planes, units, pixel, length, digest) in
            cases
        {
            let frames = ivf_payloads(file);
            assert_eq!(frames.len(), 1, "profile {profile}");
            let (sequence, key) = crate::header::parse_unit(frames[0], None).expect("sequence");
            assert_eq!(sequence.profile, profile);
            assert_eq!(sequence.bit_depth, 10);
            assert_eq!(sequence.subsampling_x, sub_x);
            assert_eq!(sequence.subsampling_y, sub_y);
            assert!(!key.disable_cdf_update, "profile {profile}");
            assert_eq!(key.width, width);
            assert_eq!(key.height, height);
            assert_eq!(key.restoration.plane, planes, "profile {profile}");
            assert_eq!(key.restoration.unit, units, "profile {profile}");
            let params = CodecParams::video(CodecId::Av1, width, height);
            let mut decoder = Av1Decoder::open(&params).expect("open");
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: frames[0].to_vec(),
                    pts: 0,
                    duration: 1,
                    timescale: 30,
                    keyframe: true,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("profile {profile}: {error:?}"));
            assert_eq!(shown.len(), 1, "profile {profile}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, pixel, "profile {profile}");
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), length, "profile {profile}");
            assert_eq!(frame_digest(bytes), digest, "profile {profile}");
            assert!(decoder.drain().expect("drain").is_empty());
        }
    }

    #[test]
    fn ten_bit_film_grain_on_444_and_422_matches_dav1d() {
        // aomenc --film-grain-test=1 --bit-depth=10, 64×64, two frames. CDEF
        // and restoration are off. One stream is profile 1 4:4:4 and one is
        // profile 2 4:2:2. Samples stay in the low 10 bits. Digests are
        // `ffmpeg -c:v libdav1d -threads 1`.
        let cases = [
            (
                include_bytes!("../fixtures/grain-444-10.ivf").as_slice(),
                1u8,
                false,
                false,
                cutlass_decode::PixelFormat::Yuv444p10,
                64 * 64 * 6,
                [0x97e3_411c_470e_6d86, 0xb0de_0a05_8938_67be],
            ),
            (
                include_bytes!("../fixtures/grain-422-10.ivf").as_slice(),
                2,
                true,
                false,
                cutlass_decode::PixelFormat::Yuv422p10,
                64 * 64 * 4,
                [0x8104_4273_3a51_f53a, 0x00ea_ecc5_a61d_0d78],
            ),
        ];
        for (file, profile, sub_x, sub_y, pixel, length, expected) in cases {
            let frames = ivf_payloads(file);
            assert_eq!(frames.len(), 2, "profile {profile}");
            let (sequence, key) = crate::header::parse_unit(frames[0], None).expect("sequence");
            assert_eq!(sequence.profile, profile);
            assert_eq!(sequence.bit_depth, 10);
            assert_eq!(sequence.subsampling_x, sub_x);
            assert_eq!(sequence.subsampling_y, sub_y);
            assert!(key.grain.apply);
            assert_eq!(key.grain.sub_x, sub_x);
            assert_eq!(key.grain.sub_y, sub_y);
            let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
            assert!(!inter.intra);
            assert!(inter.grain.apply);
            let params = CodecParams::video(CodecId::Av1, 64, 64);
            let mut decoder = Av1Decoder::open(&params).expect("open");
            for (index, payload) in frames.iter().enumerate() {
                let shown = decoder
                    .push(&cutlass_decode::Packet {
                        stream: 0,
                        data: payload.to_vec(),
                        pts: index as i64,
                        duration: 1,
                        timescale: 30,
                        keyframe: index == 0,
                        side: Vec::new(),
                    })
                    .unwrap_or_else(|error| panic!("profile {profile} frame {index}: {error:?}"));
                assert_eq!(shown.len(), 1, "profile {profile} frame {index}");
                let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                    panic!("picture");
                };
                let info = picture.info();
                assert_eq!(info.pixel, pixel, "profile {profile} frame {index}");
                let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                    panic!("cpu");
                };
                assert_eq!(bytes.len(), length, "profile {profile} frame {index}");
                assert_eq!(
                    frame_digest(bytes),
                    expected[index],
                    "profile {profile} frame {index}"
                );
            }
            assert!(decoder.drain().expect("drain").is_empty());
        }
    }

    #[test]
    fn film_grain_on_444_and_422_matches_dav1d() {
        // aomenc --film-grain-test=1, 64×64, two frames. CDEF and restoration
        // are off. One stream is profile 1 4:4:4 and one is profile 2 4:2:2.
        // Chroma grain uses the full template when that axis is not subsampled.
        // Digests are `ffmpeg -c:v libdav1d -threads 1`.
        let cases = [
            (
                include_bytes!("../fixtures/grain-444.ivf").as_slice(),
                1u8,
                false,
                false,
                cutlass_decode::PixelFormat::Yuv444p,
                64 * 64 * 3,
                [0xc834_7b7a_6690_9501, 0x098c_8ff0_695d_0b65],
            ),
            (
                include_bytes!("../fixtures/grain-422.ivf").as_slice(),
                2,
                true,
                false,
                cutlass_decode::PixelFormat::Yuv422p,
                64 * 64 + 32 * 64 * 2,
                [0xed8a_94d1_ac1f_64af, 0x32a1_2ff4_b5fe_4ad3],
            ),
        ];
        for (file, profile, sub_x, sub_y, pixel, length, expected) in cases {
            let frames = ivf_payloads(file);
            assert_eq!(frames.len(), 2);
            let (sequence, key) = crate::header::parse_unit(frames[0], None).expect("sequence");
            assert_eq!(sequence.profile, profile);
            assert_eq!(sequence.subsampling_x, sub_x);
            assert_eq!(sequence.subsampling_y, sub_y);
            assert!(key.grain.apply);
            assert_eq!(key.grain.sub_x, sub_x);
            assert_eq!(key.grain.sub_y, sub_y);
            let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
            assert!(!inter.intra);
            assert!(inter.grain.apply);
            let params = CodecParams::video(CodecId::Av1, 64, 64);
            let mut decoder = Av1Decoder::open(&params).expect("open");
            for (index, payload) in frames.iter().enumerate() {
                let shown = decoder
                    .push(&cutlass_decode::Packet {
                        stream: 0,
                        data: payload.to_vec(),
                        pts: index as i64,
                        duration: 1,
                        timescale: 30,
                        keyframe: index == 0,
                        side: Vec::new(),
                    })
                    .unwrap_or_else(|error| panic!("profile {profile} frame {index}: {error:?}"));
                assert_eq!(shown.len(), 1, "profile {profile} frame {index}");
                let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                    panic!("picture");
                };
                let info = picture.info();
                assert_eq!(info.pixel, pixel, "profile {profile} frame {index}");
                let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                    panic!("cpu");
                };
                assert_eq!(bytes.len(), length, "profile {profile} frame {index}");
                assert_eq!(
                    frame_digest(bytes),
                    expected[index],
                    "profile {profile} frame {index}"
                );
            }
            assert!(decoder.drain().expect("drain").is_empty());
        }
    }

    #[test]
    fn a_lossless_segment_picture_matches_dav1d() {
        // aomenc --lossless=1, 64×64, two frames. CDEF, restoration, and super
        // resolution are off. The inter frame header is rewritten so base_q_idx
        // is 8, segmentation is on, the map is not updated, and segment 0's
        // quantizer delta is -8. The other segments stay at 8, so the frame is
        // not coded lossless. Every block keeps segment 0 from the key and is a
        // 4×4 Walsh-Hadamard transform. The tile bytes are the original lossless
        // inter frame. Digests are `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p`.
        let file = include_bytes!("../fixtures/lossless-segment.ivf");
        let frames = ivf_payloads(file);
        assert_eq!(frames.len(), 2);
        let mut sequence = None;
        let mut headers = Vec::new();
        for payload in &frames {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            headers.push(frame);
            sequence = Some(seq);
        }
        assert!(headers[0].lossless);
        assert_eq!(headers[0].base_q, 0);
        assert!(!headers[0].segmentation.enabled);
        assert_eq!(headers[0].width, 64);
        assert_eq!(headers[0].height, 64);
        assert!(!headers[1].lossless);
        assert_eq!(headers[1].base_q, 8);
        assert!(headers[1].segmentation.enabled);
        assert!(!headers[1].segmentation.update_map);
        assert!(headers[1].segmentation.active(0, crate::header::SEG_ALT_Q));
        assert_eq!(
            i32::from(headers[1].base_q)
                + headers[1].segmentation.value(0, crate::header::SEG_ALT_Q),
            0
        );
        assert!(!headers[1].segmentation.active(1, crate::header::SEG_ALT_Q));
        let expected = [
            (64 * 64 * 3 / 2, 0xd4e1_7006_4dfa_ae30),
            (64 * 64 * 3 / 2, 0xc473_19cf_f33e_95bf),
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), expected[index].0, "frame {index}");
            assert_eq!(frame_digest(&packed), expected[index].1, "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_scaled_reference_global_motion_picture_matches_dav1d() {
        // aomenc, error resilient, 64×64 superblocks. The key is 64×64 and the
        // inter frames are 128×128, so the reference is scaled. The first inter
        // frame's global model is rotation-zoom. Those blocks sample the
        // projected vector through the scaled motion filter. Local warp,
        // restoration, CDEF, and super resolution are off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p`.
        let file = include_bytes!("../fixtures/scaled-warp.ivf");
        let frames = ivf_payloads(file);
        assert_eq!(frames.len(), 3);
        let mut sequence = None;
        let mut sizes = Vec::new();
        for payload in &frames {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            sizes.push((frame.width, frame.height, frame.gm_type));
            sequence = Some(seq);
        }
        assert_eq!(sizes[0].0, 64);
        assert_eq!(sizes[0].1, 64);
        assert_eq!(sizes[1].0, 128);
        assert_eq!(sizes[1].1, 128);
        assert_eq!(sizes[1].2[1], 2);
        let expected = [
            (64 * 64 * 3 / 2, 0x8419_0e04_1574_0ec3),
            (128 * 128 * 3 / 2, 0xf311_8759_99f9_2b97),
            (128 * 128 * 3 / 2, 0xd8ec_38de_6d66_7c35),
        ];
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), expected[index].0, "frame {index}");
            assert_eq!(frame_digest(&packed), expected[index].1, "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn an_interintra_wedge_picture_matches_dav1d() {
        // aomenc 64×64. Six blocks on the inter frame blend one reference with
        // intra through a luma wedge. The wedge sign is 0. Restoration, CDEF,
        // and super resolution are off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p`.
        let file = include_bytes!("../fixtures/interintra-wedge.ivf");
        let frames = ivf_payloads(file);
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert!(sequence.enable_interintra);
        let expected = [0x79f7_4b34_bf00_4135, 0xf0cd_13c5_915c_11ca];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn smooth_interintra_matches_dav1d() {
        // aomenc 64×64. The inter frame blends one 16×16 block with DC intra.
        // The wedge symbol is 0. Digests are `ffmpeg -c:v libdav1d -threads 1`.
        let frames: [&[u8]; 2] = [
            include_bytes!("../fixtures/interintra_0.obu"),
            include_bytes!("../fixtures/interintra_1.obu"),
        ];
        let expected = [0xe08b_dc0c_5a92_ee12, 0xf8fe_56b7_21e1_855d];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(seq.enable_interintra);
            if index == 1 {
                assert!(frame.allow_high_precision_mv);
                assert!(!frame.reference_select);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn average_compound_matches_dav1d() {
        // aomenc 64×64. Later frames set reference_select and average two
        // references. Masked compound and distance weights are off, so the
        // predictors stay at the compound filter precision and the average
        // rounds the sum. Digests are `ffmpeg -c:v libdav1d -threads 1`.
        let frames: [&[u8]; 5] = [
            include_bytes!("../fixtures/compound_0.obu"),
            include_bytes!("../fixtures/compound_1.obu"),
            include_bytes!("../fixtures/compound_2.obu"),
            include_bytes!("../fixtures/compound_3.obu"),
            include_bytes!("../fixtures/compound_4.obu"),
        ];
        let expected = [
            0xfe5f_9b89_ae87_0b0f,
            0x2872_93cd_3351_db0a,
            0x7e37_e459_41b5_7851,
            0x5bfb_a649_6f7b_8281,
            0xd519_f6d7_94f4_2cdf,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(!seq.enable_order_hint);
            assert!(!seq.enable_masked);
            assert!(!seq.enable_jnt);
            if index >= 2 {
                assert!(frame.reference_select);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn distance_compound_matches_dav1d() {
        // aomenc 64×64 with order hint. Later frames set reference_select and
        // may weight the two predictors by order-hint distance. Masked
        // compound is off. The frame does not signal skip mode. Digests are
        // `ffmpeg -c:v libdav1d -threads 1`.
        let frames: [&[u8]; 5] = [
            include_bytes!("../fixtures/distance_0.obu"),
            include_bytes!("../fixtures/distance_1.obu"),
            include_bytes!("../fixtures/distance_2.obu"),
            include_bytes!("../fixtures/distance_3.obu"),
            include_bytes!("../fixtures/distance_4.obu"),
        ];
        let expected = [
            0xfe5f_9b89_ae87_0b0f,
            0x2872_93cd_3351_db0a,
            0x6ee4_5945_eef4_d3ef,
            0xf558_739c_be0b_97d8,
            0xd8f8_2c4e_99cf_d80b,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(seq.enable_order_hint);
            assert!(seq.enable_jnt);
            assert!(!seq.enable_masked);
            if index >= 2 {
                assert!(frame.reference_select);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn masked_compound_matches_dav1d() {
        // aomenc 64×64 with masked compound and order hint off, so skip mode
        // is not signaled. Later frames set reference_select. A compound block
        // blends with a wedge or a difference mask. Digests are
        // `ffmpeg -c:v libdav1d -threads 1`.
        let frames: [&[u8]; 5] = [
            include_bytes!("../fixtures/masked_0.obu"),
            include_bytes!("../fixtures/masked_1.obu"),
            include_bytes!("../fixtures/masked_2.obu"),
            include_bytes!("../fixtures/masked_3.obu"),
            include_bytes!("../fixtures/masked_4.obu"),
        ];
        let expected = [
            0xfe5f_9b89_ae87_0b0f,
            0x2872_93cd_3351_db0a,
            0xba4a_27af_1e96_b104,
            0xbf61_2dcf_b440_08ab,
            0x197e_b9c0_0513_3b51,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(seq.enable_masked);
            assert!(!seq.enable_order_hint);
            if index >= 2 {
                assert!(frame.reference_select);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn difference_compound_matches_dav1d() {
        // Same source as the wedge stream, with wedge compound turned off.
        // A compound block that asks for a mask uses the difference weight.
        let frames: [&[u8]; 5] = [
            include_bytes!("../fixtures/diffwtd_0.obu"),
            include_bytes!("../fixtures/diffwtd_1.obu"),
            include_bytes!("../fixtures/diffwtd_2.obu"),
            include_bytes!("../fixtures/diffwtd_3.obu"),
            include_bytes!("../fixtures/diffwtd_4.obu"),
        ];
        let expected = [
            0xfe5f_9b89_ae87_0b0f,
            0x2872_93cd_3351_db0a,
            0xd4f2_2750_7b89_37d7,
            0xd0a6_f70e_63ff_6f91,
            0xb942_e0b9_20d6_ba82,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(seq.enable_masked);
            assert!(!seq.enable_order_hint);
            if index >= 2 {
                assert!(frame.reference_select);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn skip_mode_matches_dav1d() {
        // aomenc 64×64 luma ramp with a short golden-frame pyramid. Some
        // temporal units hold a hidden reference before the shown frame.
        // A skip-mode block predicts the nearest compound pair and writes
        // no residual. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 16] = [
            include_bytes!("../fixtures/skip_0.obu"),
            include_bytes!("../fixtures/skip_1.obu"),
            include_bytes!("../fixtures/skip_2.obu"),
            include_bytes!("../fixtures/skip_3.obu"),
            include_bytes!("../fixtures/skip_4.obu"),
            include_bytes!("../fixtures/skip_5.obu"),
            include_bytes!("../fixtures/skip_6.obu"),
            include_bytes!("../fixtures/skip_7.obu"),
            include_bytes!("../fixtures/skip_8.obu"),
            include_bytes!("../fixtures/skip_9.obu"),
            include_bytes!("../fixtures/skip_10.obu"),
            include_bytes!("../fixtures/skip_11.obu"),
            include_bytes!("../fixtures/skip_12.obu"),
            include_bytes!("../fixtures/skip_13.obu"),
            include_bytes!("../fixtures/skip_14.obu"),
            include_bytes!("../fixtures/skip_15.obu"),
        ];
        let expected = [
            0x7c7f_f1cc_caf8_0325,
            0x7d47_ecca_8864_0325,
            0x8ddf_9224_2738_0325,
            0x2067_6a1f_17e4_0325,
            0x8945_a93f_11e8_0325,
            0x1bcd_813a_0294_0325,
            0xfa5b_03a5_62e8_0325,
            0xb570_a2bd_73b4_0325,
            0x7067_64bc_2198_0325,
            0xb6e1_bb9f_8ba4_0325,
            0x11b7_8b2f_4238_0325,
            0xf7fe_688f_8e04_0325,
            0x692a_5500_50e8_0325,
            0xa7f3_2795_e674_0325,
            0xc6cf_71aa_fe48_0325,
            0xbc4c_a205_d3b4_0325,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn local_warp_matches_dav1d() {
        // aomenc 64×64 rotating ramp. Local warp is on, overlapped motion and
        // global motion are off. A block that selects local warp predicts
        // through the neighbor fit. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 8] = [
            include_bytes!("../fixtures/warp_0.obu"),
            include_bytes!("../fixtures/warp_1.obu"),
            include_bytes!("../fixtures/warp_2.obu"),
            include_bytes!("../fixtures/warp_3.obu"),
            include_bytes!("../fixtures/warp_4.obu"),
            include_bytes!("../fixtures/warp_5.obu"),
            include_bytes!("../fixtures/warp_6.obu"),
            include_bytes!("../fixtures/warp_7.obu"),
        ];
        let expected = [
            0xe7c3_cd2d_808b_176a,
            0xc8a7_97da_0d6c_3034,
            0xdf9e_2073_3cbd_7890,
            0xf27b_d8a3_2653_7ead,
            0x99d0_9e09_77f8_133c,
            0xb181_1245_9630_1235,
            0x7e61_2c3b_fe43_2419,
            0xfc90_db15_e81e_065e,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            if index == 0 {
                assert!(seq.enable_warped);
            }
            if index > 0 {
                assert!(frame.allow_warped);
                assert!(frame.motion_mode_switchable);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn overlapped_motion_matches_dav1d() {
        // Same rotating ramp, with overlapped motion on and local warp off.
        // A block blends the predictors of the neighboring motion vectors.
        // Digests are dav1d 1.5.3.
        let frames: [&[u8]; 8] = [
            include_bytes!("../fixtures/obmc_0.obu"),
            include_bytes!("../fixtures/obmc_1.obu"),
            include_bytes!("../fixtures/obmc_2.obu"),
            include_bytes!("../fixtures/obmc_3.obu"),
            include_bytes!("../fixtures/obmc_4.obu"),
            include_bytes!("../fixtures/obmc_5.obu"),
            include_bytes!("../fixtures/obmc_6.obu"),
            include_bytes!("../fixtures/obmc_7.obu"),
        ];
        let expected = [
            0xe7c3_cd2d_808b_176a,
            0xcca6_568f_5899_0985,
            0x245f_f039_e396_5c23,
            0xdda6_2c5f_deb6_0bd0,
            0xf8f5_8219_6e8f_1906,
            0x1541_be60_1543_2419,
            0x954e_f037_d8c8_5719,
            0xd6de_133c_3aa1_697f,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            if index == 0 {
                assert!(!seq.enable_warped);
            }
            if index > 0 {
                assert!(!frame.allow_warped);
                assert!(frame.motion_mode_switchable);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn reference_motion_matches_dav1d() {
        // aomenc 64×64 translating ramp. Order hint and reference motion
        // vectors are on. A block's candidate stack includes the field
        // projected from an earlier frame. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 8] = [
            include_bytes!("../fixtures/refmv_0.obu"),
            include_bytes!("../fixtures/refmv_1.obu"),
            include_bytes!("../fixtures/refmv_2.obu"),
            include_bytes!("../fixtures/refmv_3.obu"),
            include_bytes!("../fixtures/refmv_4.obu"),
            include_bytes!("../fixtures/refmv_5.obu"),
            include_bytes!("../fixtures/refmv_6.obu"),
            include_bytes!("../fixtures/refmv_7.obu"),
        ];
        let expected = [
            0xcfef_e949_6c51_7954,
            0x9fdf_f457_5e9a_9914,
            0x6b24_2ef4_2f4e_c08a,
            0x50b8_1bd4_5032_17d8,
            0xd2b7_389b_37ea_96d2,
            0xf9e7_ceb6_a08d_c10b,
            0x0554_021b_7090_f238,
            0x145b_519c_22cd_e7ac,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            if index == 0 {
                assert!(seq.enable_ref_mvs);
                assert!(seq.enable_order_hint);
            }
            if index > 0 {
                assert!(frame.use_ref_frame_mvs);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn inter_superres_matches_dav1d() {
        // aomenc 64×64 shown, coded narrower. The inter frames sample the
        // upscaled reference through the scaled motion filter. Digests are
        // dav1d 1.5.3.
        let frames: [&[u8]; 4] = [
            include_bytes!("../fixtures/super_0.obu"),
            include_bytes!("../fixtures/super_1.obu"),
            include_bytes!("../fixtures/super_2.obu"),
            include_bytes!("../fixtures/super_3.obu"),
        ];
        let expected = [
            0x2eba_7fac_4977_792a,
            0xd1d6_7332_739c_ae46,
            0x1ac4_0e41_5928_5a78,
            0xdc78_a476_9c66_1bc2,
        ];
        let mut sequence = None;
        for payload in frames {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(seq.enable_superres);
            assert!(frame.upscaled_width > frame.width);
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn frame_size_from_a_reference_matches_dav1d() {
        // The sequence is 64×64. Every frame is shown at 32×32, so the size
        // is not the sequence size. A later frame copies that size from a
        // reference instead of coding it again. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 4] = [
            include_bytes!("../fixtures/size_0.obu"),
            include_bytes!("../fixtures/size_1.obu"),
            include_bytes!("../fixtures/size_2.obu"),
            include_bytes!("../fixtures/size_3.obu"),
        ];
        let expected = [
            0x66fd_0ea1_018d_ba92,
            0x49ba_3638_8cd8_c716,
            0xaf94_1afd_3c5d_08f4,
            0xe8bb_81a2_b771_96f8,
        ];
        // A later frame copies its size from a reference. A parse that has
        // not stored that reference cannot read the size bits, because they
        // are not there.
        let mut sequence = None;
        let mut copied = false;
        for (index, payload) in frames.iter().enumerate() {
            match crate::header::parse_unit(payload, sequence.as_ref()) {
                Ok((seq, frame)) => {
                    assert_eq!(frame.width, 32, "frame {index}");
                    assert_eq!(frame.height, 32, "frame {index}");
                    sequence = Some(seq);
                }
                Err(cutlass_decode::CodecError::Malformed {
                    what: "missing reference",
                    ..
                }) => {
                    assert!(index > 0);
                    copied = true;
                }
                Err(err) => panic!("frame {index}: {err}"),
            }
        }
        assert!(copied);
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 32 * 32 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn scaled_reference_matches_dav1d() {
        // The key is 32×32. The next frame is 64×64 and samples that smaller
        // reference. Digests are `ffmpeg -c:v libdav1d -threads 1 -noautoscale`.
        let frames: [&[u8]; 4] = [
            include_bytes!("../fixtures/scale_0.obu"),
            include_bytes!("../fixtures/scale_1.obu"),
            include_bytes!("../fixtures/scale_2.obu"),
            include_bytes!("../fixtures/scale_3.obu"),
        ];
        let expected = [
            0x66fd_0ea1_018d_ba92,
            0xfe7a_b0ac_d0a7_15fd,
            0xfb97_91fc_340c_7254,
            0x0571_778f_0ca8_1061,
        ];
        let lengths = [
            32 * 32 * 3 / 2,
            64 * 64 * 3 / 2,
            64 * 64 * 3 / 2,
            64 * 64 * 3 / 2,
        ];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), lengths[index], "frame {index}");
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn loop_filter_delta_update_matches_dav1d() {
        // The keyframe replaces the intra filter delta. Later frames keep that
        // value when they do not write a new one. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 4] = [
            include_bytes!("../fixtures/lfdelta_0.obu"),
            include_bytes!("../fixtures/lfdelta_1.obu"),
            include_bytes!("../fixtures/lfdelta_2.obu"),
            include_bytes!("../fixtures/lfdelta_3.obu"),
        ];
        let expected = [
            0x00ec_51b0_a70a_febb,
            0x3e9b_4d41_a410_05e5,
            0x9ec7_fa19_6d23_3a89,
            0x9674_3f13_bccc_56b5,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            if index == 0 {
                assert!(frame.lf_delta_enabled);
                assert_eq!(frame.lf_ref_deltas[0], -16);
            } else {
                assert_ne!(frame.primary_ref, 7, "frame {index}");
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 64 * 64 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn superblock_loop_filter_delta_matches_dav1d() {
        // aomenc keyframes with a quantizer delta and one loop-filter delta
        // per superblock. The luma filter levels are non-zero, so that delta
        // changes samples. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 2] = [
            include_bytes!("../fixtures/lfblock_0.obu"),
            include_bytes!("../fixtures/lfblock_1.obu"),
        ];
        let expected = [0xf892_381c_0e3d_8319, 0xf892_381c_0e3d_8319];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(frame.delta_q_present, "frame {index}");
            assert!(frame.delta_lf_present, "frame {index}");
            assert!(!frame.delta_lf_multi, "frame {index}");
            assert_ne!(frame.lf_level[0], 0, "frame {index}");
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 128 * 128 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn segmentation_quantizer_matches_dav1d() {
        // aomenc variance adaptive quantization. The keyframe assigns a
        // quantizer delta to each segment. A later frame keeps the previous
        // map and does not rewrite the feature values. Digests are dav1d 1.5.3.
        let frames: [&[u8]; 4] = [
            include_bytes!("../fixtures/seg_0.obu"),
            include_bytes!("../fixtures/seg_1.obu"),
            include_bytes!("../fixtures/seg_2.obu"),
            include_bytes!("../fixtures/seg_3.obu"),
        ];
        let expected = [
            0xdcbd_5a56_a7f2_81f8,
            0x6e04_7338_a20a_5d6f,
            0x6e04_7338_a20a_5d6f,
            0x6e04_7338_a20a_5d6f,
        ];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) =
                crate::header::parse_unit(payload, sequence.as_ref()).expect("parse");
            assert!(frame.segmentation.enabled, "frame {index}");
            if index == 0 {
                assert!(frame.segmentation.update_data);
                assert!(frame.segmentation.update_map);
                assert!(!frame.segmentation.preskip);
                assert_eq!(frame.segmentation.data[0][0], -50);
                assert_eq!(frame.segmentation.data[7][0], 25);
            }
            if index == 3 {
                assert!(!frame.segmentation.update_map);
                assert!(!frame.segmentation.update_data);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 128 * 128 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn switch_frame_matches_dav1d() {
        // SVT-AV1 low delay, with a switch frame at index 3. That frame is
        // predicted as an inter frame, is error resilient, and refreshes every
        // reference slot. The frames after it use those slots. Digests are
        // dav1d 1.5.3 with in-loop filters.
        let frames: [&[u8]; 6] = [
            include_bytes!("../fixtures/sw_0.obu"),
            include_bytes!("../fixtures/sw_1.obu"),
            include_bytes!("../fixtures/sw_2.obu"),
            include_bytes!("../fixtures/sw_3.obu"),
            include_bytes!("../fixtures/sw_4.obu"),
            include_bytes!("../fixtures/sw_5.obu"),
        ];
        let expected = [
            0x832d_5f10_e25b_b379,
            0xe711_b7e7_2938_348c,
            0x148f_178d_d2f4_1d6c,
            0x82c3_51a8_c80f_9fbe,
            0x4814_1f75_474a_da47,
            0x7c8f_46f6_4e71_07c6,
        ];
        let sizes = [Some((128, 128, 128, 128)); 8];
        let saved = [None; 8];
        let hints = [0; 8];
        let inherited = [crate::header::Inherited::defaults(); 8];
        let mut sequence = None;
        for (index, payload) in frames.iter().enumerate() {
            let (seq, frame) = crate::header::parse_with_motion(
                payload,
                sequence.as_ref(),
                &saved,
                &hints,
                &sizes,
                &inherited,
            )
            .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            if index == 3 {
                assert_eq!(frame.frame_type, crate::header::SWITCH_FRAME);
                assert!(!frame.intra);
                assert!(frame.error_resilient);
                assert_eq!(frame.refresh_flags, 0xff);
                assert!(frame.primary_ref >= 7);
            }
            sequence = Some(seq);
        }
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let packed = push_frame(&mut decoder, payload, index as i64);
            assert_eq!(packed.len(), 128 * 128 * 3 / 2);
            assert_eq!(frame_digest(&packed), expected[index], "frame {index}");
        }
    }

    #[test]
    fn a_reserved_obu_is_skipped() {
        // The picture is the flat 128 superblock keyframe. A reserved OBU
        // sits between the sequence and the frame. Digests are dav1d 1.5.3.
        let frame = include_bytes!("../fixtures/sb128_flat_0.obu");
        let expected = 0xf206_7a5e_2040_2325;
        let params = CodecParams::video(CodecId::Av1, 256, 256);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let plain = push_frame(&mut decoder, frame, 0);
        assert_eq!(frame_digest(&plain), expected);

        let mut with_reserved = Vec::new();
        with_reserved.extend_from_slice(&frame[..11]);
        with_reserved.extend_from_slice(&reserved_obu(9, &[0xff, 0x01]));
        with_reserved.extend_from_slice(&frame[11..]);
        with_reserved.splice(0..0, reserved_obu(0, &[0x80]));
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &with_reserved, 0);
        assert_eq!(frame_digest(&packed), expected);

        let mut overrun = reserved_obu(14, &[0x80]);
        overrun[1] = 50;
        overrun.extend_from_slice(frame);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        assert!(
            decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: overrun,
                    pts: 0,
                    duration: 1,
                    timescale: 30,
                    keyframe: true,
                    side: Vec::new(),
                })
                .is_err()
        );
    }

    #[test]
    fn an_obu_without_a_size_uses_the_packet_end() {
        // The last OBU of the flat 128 superblock keyframe drops its size
        // field. The packet length is the size. A missing trailing byte fails.
        // The digest is dav1d 1.5.3.
        let frame = include_bytes!("../fixtures/sb128_flat_0.obu");
        let expected = 0xf206_7a5e_2040_2325;
        let open = without_last_size(frame);
        let params = CodecParams::video(CodecId::Av1, 256, 256);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, &open, 0);
        assert_eq!(packed.len(), 256 * 256 * 3 / 2);
        assert_eq!(frame_digest(&packed), expected);

        let mut short = open;
        short.pop();
        let mut decoder = Av1Decoder::open(&params).expect("open");
        assert!(
            decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: short,
                    pts: 0,
                    duration: 1,
                    timescale: 30,
                    keyframe: true,
                    side: Vec::new(),
                })
                .is_err()
        );
    }

    #[test]
    fn a_monochrome_picture_matches_dav1d() {
        // aomenc --monochrome, 128×128, three frames. Filter intra and an
        // interintra wedge are off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt gray`.
        let file = include_bytes!("../fixtures/mono.ivf");
        let frames = ivf_payloads(file);
        let expected = [
            0x1df5_4802_6a88_f184,
            0xe533_b05b_9f50_a964,
            0x4d6f_93f9_0895_ca96,
        ];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert!(sequence.monochrome);
        assert_eq!(sequence.bit_depth, 8);
        let gray =
            CodecParams::with_format(CodecId::Av1, 128, 128, cutlass_decode::PixelFormat::Gray8);
        assert!(Av1Decoder::open(&gray).is_ok());
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Gray8);
            assert_eq!((info.width, info.height), (128, 128));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 128 * 128, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_ten_bit_monochrome_picture_matches_dav1d() {
        // aomenc --monochrome --bit-depth=10, 128×128, one keyframe and one
        // inter frame. Filter intra, an interintra wedge, restoration, and
        // super resolution are off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt gray10le`.
        let file = include_bytes!("../fixtures/mono10.ivf");
        let frames = ivf_payloads(file);
        let expected = [0x86ae_d48a_6970_1217, 0x15dc_062f_948b_bdbe];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert!(sequence.monochrome);
        assert_eq!(sequence.bit_depth, 10);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let gray =
            CodecParams::with_format(CodecId::Av1, 128, 128, cutlass_decode::PixelFormat::Gray10);
        assert!(Av1Decoder::open(&gray).is_ok());
        let params = CodecParams::video(CodecId::Av1, 128, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Gray10);
            assert_eq!((info.width, info.height), (128, 128));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 128 * 128 * 2, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn an_operating_point_keeps_its_temporal_layer() {
        // Two operating points. Point 0 is temporal layer 0. Point 1 also
        // includes temporal layer 1, a second copy of the keyframe. Digests are
        // `ffmpeg -c:v libdav1d -threads 1` and the same command with `-oppoint 1`.
        let file = include_bytes!("../fixtures/operating-point.ivf");
        let frames = ivf_payloads(file);
        assert_eq!(frames.len(), 3);
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.operating_points, 1);
        assert_eq!(sequence.operating_idc[0], 0x101);
        assert_eq!(sequence.operating_idc[1], 0x103);
        let base = [0xf021_b9cb_b654_3518, 0x21e9_8a09_7459_161c];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("point 0 frame {index}: {error:?}"));
            if index == 2 {
                assert!(shown.is_empty(), "temporal layer");
                continue;
            }
            assert_eq!(shown.len(), 1, "point 0 frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(frame_digest(bytes), base[index], "point 0 frame {index}");
        }
        let mut layered = Av1Decoder::open(&params).expect("open");
        layered.select_operating_point(1).expect("point 1");
        let layered_digest = [base[0], base[1], base[0]];
        for (index, payload) in frames.iter().enumerate() {
            let shown = layered
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index != 1,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("point 1 frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "point 1 frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(
                frame_digest(bytes),
                layered_digest[index],
                "point 1 frame {index}"
            );
        }
        let mut refused = Av1Decoder::open(&params).expect("open");
        refused.select_operating_point(2).expect("stored");
        let error = refused
            .push(&cutlass_decode::Packet {
                stream: 0,
                data: frames[0].to_vec(),
                pts: 0,
                duration: 1,
                timescale: 30,
                keyframe: true,
                side: Vec::new(),
            })
            .expect_err("past the last point");
        assert_eq!(
            error,
            cutlass_decode::CodecError::Unsupported {
                what: "operating point",
            }
        );
    }

    #[test]
    fn colour_timing_and_render_size_stay_on_the_picture() {
        // aomenc 64×64, one keyframe and one inter frame, with BT.709 colour,
        // colocated chroma, and a constant 30 Hz timing_info. The keyframe's
        // display size is 80×40. The sample picture stays 64×64. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p`.
        let file = include_bytes!("../fixtures/color.ivf");
        let frames = ivf_payloads(file);
        let (sequence, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.color_primaries, 1);
        assert_eq!(sequence.transfer_characteristics, 1);
        assert_eq!(sequence.matrix_coefficients, 1);
        assert_eq!(sequence.chroma_sample_position, 2);
        let timing = sequence.timing.expect("timing");
        assert_eq!(timing.num_units_in_display_tick, 1);
        assert_eq!(timing.time_scale, 30);
        assert!(timing.equal_picture_interval);
        assert_eq!(timing.num_ticks_per_picture_minus_1, Some(0));
        assert_eq!((frame.render_width, frame.render_height), (80, 40));
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert_eq!((inter.render_width, inter.render_height), (64, 64));
        let expected = [0x1b33_c4d2_4867_47f8, 0x7963_d39f_992f_c21f];
        let renders = [(80, 40), (64, 64)];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv420p);
            assert_eq!((info.width, info.height), (64, 64));
            assert_eq!((info.render_width, info.render_height), renders[index]);
            assert_eq!(info.primaries, cutlass_decode::ColorPrimaries::Bt709);
            assert_eq!(info.transfer, cutlass_decode::Transfer::Bt709);
            assert_eq!(info.matrix, cutlass_decode::ColorMatrix::Bt709);
            assert_eq!(
                info.chroma_position,
                cutlass_decode::ChromaSamplePosition::Colocated
            );
            assert_eq!(
                (info.cicp_primaries, info.cicp_transfer, info.cicp_matrix),
                (1, 1, 1)
            );
            assert_eq!(info.num_units_in_display_tick, 1);
            assert_eq!(info.time_scale, 30);
            assert!(info.equal_picture_interval);
            assert_eq!(info.num_ticks_per_picture_minus_1, Some(0));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3 / 2, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn decoder_model_info_is_parsed_and_the_picture_matches_dav1d() {
        // aomenc --timing-info=model, 64×64, one keyframe and one inter frame.
        // Restoration, super resolution, filter intra, and an interintra wedge
        // are off. The model is parsed and not scheduled. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p`.
        let file = include_bytes!("../fixtures/model.ivf");
        let frames = ivf_payloads(file);
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        let model = sequence.decoder_model.expect("decoder model");
        assert_eq!(model.buffer_delay_length_minus_1, 15);
        assert_eq!(model.num_units_in_decoding_tick, 1);
        assert_eq!(model.buffer_removal_time_length_minus_1, 9);
        assert_eq!(model.frame_presentation_time_length_minus_1, 9);
        assert!(model.present[0]);
        assert!(!model.present[1]);
        let timing = sequence.timing.expect("timing");
        assert_eq!(timing.num_units_in_display_tick, 1);
        assert_eq!(timing.time_scale, 30);
        assert!(!timing.equal_picture_interval);
        assert_eq!(timing.num_ticks_per_picture_minus_1, None);
        let expected = [0x1b33_c4d2_4867_47f8, 0x7963_d39f_992f_c21f];
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
            assert_eq!(picture.info().num_units_in_decoding_tick, 1);
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_level_three_picture_wider_than_4096_matches_dav1d() {
        // aomenc --target-seq-level-idx=4, 4352×64, one keyframe and one inter
        // frame. Level 3.0 allows this edge. Restoration, super resolution,
        // filter intra, and an interintra wedge are off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p`.
        let file = include_bytes!("../fixtures/level.ivf");
        let frames = ivf_payloads(file);
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.seq_level_idx[0], 4);
        let expected = [0x419d_ca99_d7ff_bfd7, 0x47af_41d4_871f_de86];
        let params = CodecParams::video(CodecId::Av1, 4352, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!((info.width, info.height), (4352, 64));
            assert_eq!(info.seq_level_idx, 4);
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 4352 * 64 * 3 / 2);
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_picture_past_every_defined_level_is_refused() {
        // 16448×64 is wider than level 6.3. dav1d decodes it. The check runs
        // before the frame buffer is allocated.
        let file = include_bytes!("../fixtures/wide.ivf");
        let frames = ivf_payloads(file);
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let error = decoder
            .push(&cutlass_decode::Packet {
                stream: 0,
                data: frames[0].to_vec(),
                pts: 0,
                duration: 1,
                timescale: 30,
                keyframe: true,
                side: Vec::new(),
            })
            .expect_err("past every defined level");
        assert!(
            matches!(
                error,
                cutlass_decode::CodecError::Limit { what: "frame size" }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn a_twelve_bit_profile_two_444_picture_matches_dav1d() {
        // aomenc --i444 --bit-depth=12, 64×64, one keyframe and one inter frame.
        // The encoder selects profile 2. Restoration, super resolution, filter
        // intra, and an interintra wedge are off. Film grain is off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv444p12le`.
        let file = include_bytes!("../fixtures/yuv444-12.ivf");
        let frames = ivf_payloads(file);
        let expected = [0x9ba7_f873_40f0_da67, 0x1c37_dcc4_bbdb_ffdc];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.profile, 2);
        assert_eq!(sequence.bit_depth, 12);
        assert!(!sequence.subsampling_x);
        assert!(!sequence.subsampling_y);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let wide =
            CodecParams::with_format(CodecId::Av1, 64, 64, cutlass_decode::PixelFormat::Yuv444p12);
        assert!(Av1Decoder::open(&wide).is_ok());
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv444p12);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 6, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_twelve_bit_profile_two_422_picture_matches_dav1d() {
        // aomenc --i422 --bit-depth=12, 64×64, one keyframe and one inter frame.
        // The encoder selects profile 2. Restoration, super resolution, filter
        // intra, and an interintra wedge are off. Film grain is off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv422p12le`.
        let file = include_bytes!("../fixtures/yuv422-12.ivf");
        let frames = ivf_payloads(file);
        let expected = [0x9cc4_cc09_c07e_dab1, 0xa7dc_8376_997d_f350];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.profile, 2);
        assert_eq!(sequence.bit_depth, 12);
        assert!(sequence.subsampling_x);
        assert!(!sequence.subsampling_y);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let wide =
            CodecParams::with_format(CodecId::Av1, 64, 64, cutlass_decode::PixelFormat::Yuv422p12);
        assert!(Av1Decoder::open(&wide).is_ok());
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv422p12);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 4, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn wiener_units_keep_the_updated_cdf() {
        // aomenc --sharpness=5 --cq-level=32, 512×128. Every plane is Wiener
        // with 128-sample units. Luma has four units and the frame updates
        // that CDF. Chroma is 4:2:0, so each chroma plane has two units and a
        // stripe is 32 samples. The digest is `ffmpeg -c:v libdav1d -threads 1`.
        let frames = ivf_payloads(include_bytes!("../fixtures/restore-wiener.ivf"));
        assert_eq!(frames.len(), 1);
        let (sequence, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert!(sequence.subsampling_x && sequence.subsampling_y);
        assert!(sequence.enable_cdef);
        assert!(!frame.disable_cdf_update);
        assert_eq!(frame.width, 512);
        assert_eq!(frame.height, 128);
        assert_eq!(frame.restoration.plane, [crate::header::RESTORE_WIENER; 3]);
        assert_eq!(frame.restoration.unit, [128; 3]);
        let params = CodecParams::video(CodecId::Av1, 512, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packed = push_frame(&mut decoder, frames[0], 0);
        assert_eq!(packed.len(), 512 * 128 * 3 / 2);
        assert_eq!(frame_digest(&packed), 0xdf63_3cd9_dfae_14d3);
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn twelve_bit_restoration_matches_dav1d() {
        // aomenc --bit-depth=12, 256×128 4:2:0. Luma is self-guided. Chroma
        // restoration is off. CDEF is off. Samples stay in the low 12 bits.
        // The digest is `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p12le`.
        let frames = ivf_payloads(include_bytes!("../fixtures/restore-12.ivf"));
        assert_eq!(frames.len(), 1);
        let (sequence, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.profile, 2);
        assert_eq!(sequence.bit_depth, 12);
        assert!(sequence.subsampling_x && sequence.subsampling_y);
        assert_eq!(frame.width, 256);
        assert_eq!(frame.height, 128);
        assert_eq!(
            frame.restoration.plane,
            [
                crate::header::RESTORE_SGRPROJ,
                crate::header::RESTORE_NONE,
                crate::header::RESTORE_NONE,
            ]
        );
        assert_eq!(frame.restoration.unit, [256, 256, 256]);
        let params = CodecParams::video(CodecId::Av1, 256, 128);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let shown = decoder
            .push(&cutlass_decode::Packet {
                stream: 0,
                data: frames[0].to_vec(),
                pts: 0,
                duration: 1,
                timescale: 30,
                keyframe: true,
                side: Vec::new(),
            })
            .expect("frame");
        assert_eq!(shown.len(), 1);
        let cutlass_decode::Output::Picture(picture) = &shown[0] else {
            panic!("picture");
        };
        let info = picture.info();
        assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv420p12);
        let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
            panic!("cpu");
        };
        assert_eq!(bytes.len(), 256 * 128 * 3);
        assert_eq!(frame_digest(bytes), 0x4969_32d8_3fe2_20c9);
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_twelve_bit_profile_two_picture_matches_dav1d() {
        // aomenc --profile=2 --i420 --bit-depth=12, 64×64, one keyframe and one
        // inter frame. Restoration, super resolution, filter intra, and an
        // interintra wedge are off. Film grain is off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p12le`.
        let file = include_bytes!("../fixtures/yuv420-12.ivf");
        let frames = ivf_payloads(file);
        let expected = [0xa23a_10c6_2e39_0ed4, 0xfc00_e897_7397_b338];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.profile, 2);
        assert_eq!(sequence.bit_depth, 12);
        assert!(sequence.subsampling_x);
        assert!(sequence.subsampling_y);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let wide =
            CodecParams::with_format(CodecId::Av1, 64, 64, cutlass_decode::PixelFormat::Yuv420p12);
        assert!(Av1Decoder::open(&wide).is_ok());
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv420p12);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_ten_bit_profile_two_picture_matches_dav1d() {
        // aomenc --profile=2 --i422 --bit-depth=10, 64×64, one keyframe and one
        // inter frame. Restoration, super resolution, filter intra, and an
        // interintra wedge are off. Film grain is off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv422p10le`.
        let file = include_bytes!("../fixtures/yuv422-10.ivf");
        let frames = ivf_payloads(file);
        let expected = [0x001b_2070_d9b7_c298, 0x8d13_4376_1d14_82e1];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.profile, 2);
        assert_eq!(sequence.bit_depth, 10);
        assert!(sequence.subsampling_x);
        assert!(!sequence.subsampling_y);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let wide =
            CodecParams::with_format(CodecId::Av1, 64, 64, cutlass_decode::PixelFormat::Yuv422p10);
        assert!(Av1Decoder::open(&wide).is_ok());
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv422p10);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 4, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_profile_two_picture_matches_dav1d() {
        // aomenc --profile=2 --i422, 64×64, one keyframe and one inter frame.
        // Restoration, super resolution, filter intra, and an interintra wedge
        // are off. Film grain is off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv422p`.
        let file = include_bytes!("../fixtures/yuv422.ivf");
        let frames = ivf_payloads(file);
        let expected = [0xce61_9143_3646_31bb, 0x5e03_15f4_b833_cd90];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.profile, 2);
        assert_eq!(sequence.bit_depth, 8);
        assert!(sequence.subsampling_x);
        assert!(!sequence.subsampling_y);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let wide =
            CodecParams::with_format(CodecId::Av1, 64, 64, cutlass_decode::PixelFormat::Yuv422p);
        assert!(Av1Decoder::open(&wide).is_ok());
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv422p);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 2, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_profile_one_picture_matches_dav1d() {
        // aomenc --profile=1 --i444, 64×64, one keyframe and one inter frame.
        // Restoration, super resolution, filter intra, and an interintra wedge
        // are off. Film grain is off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv444p`.
        let file = include_bytes!("../fixtures/yuv444.ivf");
        let frames = ivf_payloads(file);
        let expected = [0x8205_072f_5468_587b, 0xf644_9906_e40d_6902];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.profile, 1);
        assert_eq!(sequence.bit_depth, 8);
        assert!(!sequence.subsampling_x);
        assert!(!sequence.subsampling_y);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let wide =
            CodecParams::with_format(CodecId::Av1, 64, 64, cutlass_decode::PixelFormat::Yuv444p);
        assert!(Av1Decoder::open(&wide).is_ok());
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv444p);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_ten_bit_profile_one_picture_matches_dav1d() {
        // aomenc --profile=1 --i444 --bit-depth=10, 64×64, one keyframe and one
        // inter frame. Restoration, super resolution, filter intra, and an
        // interintra wedge are off. Film grain is off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv444p10le`.
        let file = include_bytes!("../fixtures/yuv444-10.ivf");
        let frames = ivf_payloads(file);
        let expected = [0x581e_f035_c698_6f47, 0xdff1_b576_fb7f_aa2c];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.profile, 1);
        assert_eq!(sequence.bit_depth, 10);
        assert!(!sequence.subsampling_x);
        assert!(!sequence.subsampling_y);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let wide =
            CodecParams::with_format(CodecId::Av1, 64, 64, cutlass_decode::PixelFormat::Yuv444p10);
        assert!(Av1Decoder::open(&wide).is_ok());
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv444p10);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 6, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_ten_bit_picture_matches_dav1d() {
        // aomenc profile 0, 10-bit, 64×64, one keyframe and one inter frame.
        // Restoration, super resolution, film grain, filter intra, and an
        // interintra wedge are off. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p10le`.
        let file = include_bytes!("../fixtures/ten.ivf");
        let frames = ivf_payloads(file);
        let expected = [0x20d1_bb0c_7997_8616, 0xb811_99a4_24e0_cf92];
        let (sequence, _) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.bit_depth, 10);
        assert!(!sequence.monochrome);
        let wide =
            CodecParams::with_format(CodecId::Av1, 64, 64, cutlass_decode::PixelFormat::Yuv420p10);
        assert!(Av1Decoder::open(&wide).is_ok());
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            assert!(shown[0].as_video().is_none(), "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv420p10);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_ten_bit_restoration_picture_matches_dav1d() {
        // aomenc profile 0, 10-bit, 64×64 keyframes. One luma plane uses the
        // Wiener filter and one uses self-guided restoration. Chroma selects
        // none. CDEF, the loop filter, and super resolution are off. Digests
        // are `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p10le`.
        let cases = [
            (
                include_bytes!("../fixtures/ten-wiener.ivf").as_slice(),
                1u8,
                0x940b_250a_982f_0c81u64,
            ),
            (
                include_bytes!("../fixtures/ten-sgr.ivf").as_slice(),
                2u8,
                0xd51c_3ef3_fbd2_f5c1u64,
            ),
        ];
        for (file, luma, expected) in cases {
            let frames = ivf_payloads(file);
            let (sequence, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
            assert_eq!(sequence.bit_depth, 10);
            assert!(sequence.enable_restoration);
            assert_eq!(frame.restoration.plane[0], luma);
            assert!(frame.restoration.uses);
            let params = CodecParams::video(CodecId::Av1, 64, 64);
            let mut decoder = Av1Decoder::open(&params).expect("open");
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: frames[0].to_vec(),
                    pts: 0,
                    duration: 1,
                    timescale: 30,
                    keyframe: true,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("luma {luma}: {error:?}"));
            assert_eq!(shown.len(), 1);
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv420p10);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3);
            assert_eq!(frame_digest(bytes), expected, "luma {luma}");
        }
    }

    #[test]
    fn a_ten_bit_grain_picture_matches_dav1d() {
        // aomenc profile 0, 10-bit, 64×64, one keyframe and one inter frame.
        // Film grain test vector 1 is on. Restoration and super resolution are
        // off. The shown pictures include the grain. Digests are
        // `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p10le`.
        let file = include_bytes!("../fixtures/ten-grain.ivf");
        let frames = ivf_payloads(file);
        let expected = [0xf5d3_1373_87c4_71e3, 0x5686_4b37_12f3_6d66];
        let (sequence, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.bit_depth, 10);
        assert!(frame.grain.apply);
        let (_, inter) = crate::header::parse_unit(frames[1], Some(&sequence)).expect("inter");
        assert!(!inter.intra);
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv420p10);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn a_ten_bit_superres_picture_matches_dav1d() {
        // aomenc profile 0, 10-bit, shown 64×64, coded narrower. The inter
        // frame samples the upscaled reference. Restoration and film grain
        // are off. Digests are `ffmpeg -c:v libdav1d -threads 1 -pix_fmt yuv420p10le`.
        let file = include_bytes!("../fixtures/ten-sr.ivf");
        let frames = ivf_payloads(file);
        let expected = [0x5b6c_62a7_42d3_8e10, 0x4eb2_aa07_fece_99cb];
        let (sequence, frame) = crate::header::parse_unit(frames[0], None).expect("sequence");
        assert_eq!(sequence.bit_depth, 10);
        assert!(sequence.enable_superres);
        assert!(frame.upscaled_width > frame.width);
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        for (index, payload) in frames.iter().enumerate() {
            let shown = decoder
                .push(&cutlass_decode::Packet {
                    stream: 0,
                    data: payload.to_vec(),
                    pts: index as i64,
                    duration: 1,
                    timescale: 30,
                    keyframe: index == 0,
                    side: Vec::new(),
                })
                .unwrap_or_else(|error| panic!("frame {index}: {error:?}"));
            assert_eq!(shown.len(), 1, "frame {index}");
            let cutlass_decode::Output::Picture(picture) = &shown[0] else {
                panic!("picture");
            };
            let info = picture.info();
            assert_eq!(info.pixel, cutlass_decode::PixelFormat::Yuv420p10);
            assert_eq!((info.width, info.height), (64, 64));
            let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            assert_eq!(bytes.len(), 64 * 64 * 3, "frame {index}");
            assert_eq!(frame_digest(bytes), expected[index], "frame {index}");
        }
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn every_shown_frame_of_a_real_file_matches_dav1d() {
        // The file stays in ../local-files. CI skips when that folder is
        // absent. A present file must match every shown picture from
        // `ffmpeg -c:v libdav1d -threads 1`, including frames after the key.
        let Some(input) = real_av1_file() else {
            return;
        };
        if !ffmpeg_installed() {
            return;
        }
        let ivf_path =
            std::env::temp_dir().join(format!("cutlass-av1-real-{}.ivf", std::process::id()));
        let extracted = std::process::Command::new("ffmpeg")
            .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
            .arg(&input)
            .args(["-map", "0:v:0", "-c:v", "copy", "-f", "ivf"])
            .arg(&ivf_path)
            .output()
            .expect("ffmpeg");
        assert!(
            extracted.status.success(),
            "{}",
            String::from_utf8_lossy(&extracted.stderr)
        );
        let _remove_ivf = TempFile(ivf_path.clone());
        let file = std::fs::read(&ivf_path).expect("ivf");
        let frames = ivf_payloads(&file);

        let mut child = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-c:v",
                "libdav1d",
                "-threads",
                "1",
                "-i",
            ])
            .arg(&ivf_path)
            .args([
                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1",
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("libdav1d");
        let mut stdout = child.stdout.take().expect("stdout");
        let mut stderr = child.stderr.take().expect("stderr");
        let mut logged = Some(std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = std::io::Read::read_to_end(&mut stderr, &mut bytes);
            bytes
        }));

        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let mut oracle = Vec::new();
        let mut shown = 0usize;
        for (index, payload) in frames.iter().enumerate() {
            let packet = cutlass_decode::Packet {
                stream: 0,
                data: payload.to_vec(),
                pts: i64::try_from(index).unwrap_or(i64::MAX),
                duration: 1,
                timescale: 30,
                keyframe: index == 0,
                side: Vec::new(),
            };
            let pictures = decoder
                .push(&packet)
                .unwrap_or_else(|error| panic!("packet {index} failed: {error:?}"));
            for picture in &pictures {
                let cutlass_decode::Output::Picture(picture) = picture else {
                    panic!("shown {shown} is not a picture");
                };
                let cutlass_decode::PictureSamples::Cpu(bytes) = picture.samples() else {
                    panic!("shown {shown} is not cpu samples");
                };
                if oracle.len() != bytes.len() {
                    oracle.resize(bytes.len(), 0);
                }
                if let Err(error) =
                    std::io::Read::read_exact(&mut stdout, &mut oracle[..bytes.len()])
                {
                    panic!(
                        "shown {shown} from packet {index}: {error}: {}",
                        oracle_note(&mut logged)
                    );
                }
                if let Some(at) = bytes
                    .iter()
                    .zip(oracle.iter())
                    .position(|(left, right)| left != right)
                {
                    panic!(
                        "shown {shown} from packet {index} differs at byte {at}: {:02x} vs {:02x} ({}x{})",
                        bytes[at],
                        oracle[at],
                        picture.info().width,
                        picture.info().height
                    );
                }
                shown += 1;
            }
        }
        let drained = decoder.drain().expect("drain");
        assert!(drained.is_empty(), "drain returned {}", drained.len());
        let mut extra = [0u8; 1];
        let trailing = std::io::Read::read(&mut stdout, &mut extra).expect("trailing");
        assert_eq!(trailing, 0, "dav1d returned another frame");
        let status = child.wait().expect("wait");
        assert!(status.success(), "{}", oracle_note(&mut logged));
        assert!(shown > 1, "only the first shown frame was compared");
    }

    struct TempFile(std::path::PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn oracle_note(logged: &mut Option<std::thread::JoinHandle<Vec<u8>>>) -> String {
        let Some(handle) = logged.take() else {
            return String::new();
        };
        let bytes = handle.join().unwrap_or_default();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn real_av1_file() -> Option<std::path::PathBuf> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
        if !root.is_dir() {
            return None;
        }
        let mut found = Vec::new();
        let entries = std::fs::read_dir(root).ok()?;
        for entry in entries {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let name = name.to_ascii_lowercase();
            let video = name.ends_with(".mp4") || name.ends_with(".webm") || name.ends_with(".mov");
            if video && name.contains("av1") {
                found.push(path);
            }
        }
        found.sort();
        found.into_iter().next()
    }

    fn ffmpeg_installed() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    fn ivf_payloads(file: &[u8]) -> Vec<&[u8]> {
        assert!(file.len() >= 32 && &file[..4] == b"DKIF", "ivf header");
        let header = usize::from(u16::from_le_bytes([file[6], file[7]]));
        assert!(header >= 32 && file.len() >= header, "ivf header length");
        let mut offset = header;
        let mut frames = Vec::new();
        while offset + 12 <= file.len() {
            let size = usize::try_from(u32::from_le_bytes([
                file[offset],
                file[offset + 1],
                file[offset + 2],
                file[offset + 3],
            ]))
            .expect("frame size");
            offset += 12;
            assert!(offset + size <= file.len(), "ivf frame ends past the file");
            frames.push(&file[offset..offset + size]);
            offset += size;
        }
        assert_eq!(offset, file.len(), "bytes after the last ivf frame");
        frames
    }

    fn without_last_size(packet: &[u8]) -> Vec<u8> {
        let mut index = 0;
        let mut last = 0;
        while index < packet.len() {
            last = index;
            let header = packet[index];
            assert_eq!(header & 0x04, 0, "extension");
            assert_ne!(header & 0x02, 0, "size");
            index += 1;
            let size = usize::from(packet[index]);
            assert!(packet[index] < 128, "leb128");
            index += 1 + size;
        }
        assert_eq!(index, packet.len());
        let mut out = packet[..last].to_vec();
        out.push(packet[last] & !0x02);
        out.extend_from_slice(&packet[last + 2..]);
        out
    }

    fn reserved_obu(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(2 + payload.len());
        bytes.push((kind << 3) | 0x02);
        bytes.push(u8::try_from(payload.len()).expect("size"));
        bytes.extend_from_slice(payload);
        bytes
    }
}
