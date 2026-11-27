//! The decoding process up to reconstruction, shared by the software
//! decoder and the stateless front end: the NAL units of an access unit,
//! parameter sets, slice segment headers, picture order counts (clause
//! 8.3.1), reference picture sets and marking (clause 8.3.2), pictures
//! generated for missing references (clause 8.3.3), reference picture lists
//! (clause 8.3.4), and the decoded picture buffer with its output order
//! (clause C.5.2).
//!
//! A [`Backend`] reconstructs the pictures: the software decoder, or an
//! accelerator behind [`crate::stateless`]. The buffer holds what the
//! backend gives back for each picture.

use std::sync::Arc;

use cutlass_decode::{CodecError, Packet};

use crate::nal::{self, Nal};
use crate::ps::{self, Pps, Sps};
use crate::slice::{self, Parsed, SliceHeader, SliceType};

const MAX_SPS: usize = 16;
const MAX_PPS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Marking {
    Unused,
    Short,
    Long,
}

/// The list of the current picture's reference picture set (clause 8.3.2)
/// a reference picture is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpsSet {
    /// `RefPicSetStCurrBefore`: short-term, before the current picture in
    /// output order, used by it.
    StCurrBefore,
    /// `RefPicSetStCurrAfter`: short-term, after the current picture, used
    /// by it.
    StCurrAfter,
    /// `RefPicSetLtCurr`: long-term, used by the current picture.
    LtCurr,
    /// `RefPicSetStFoll` or `RefPicSetLtFoll`: kept for later pictures, not
    /// used by the current one.
    Foll,
}

/// What an output picture carries besides its samples.
#[derive(Clone, Debug)]
pub(crate) struct Meta {
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub keyframe: bool,
    pub sps: Arc<Sps>,
}

/// A picture chosen for output, in output order.
pub(crate) type Bumped<P> = (P, Arc<Meta>);

#[derive(Clone)]
struct DpbPic<P> {
    pic: P,
    poc: i32,
    marking: Marking,
    /// Waiting for output.
    output: bool,
    /// Identity of the picture in the buffer.
    uid: u64,
    /// `None` for a picture generated for a missing reference.
    meta: Option<Arc<Meta>>,
}

/// A picture of the buffer marked as used for reference once the current
/// picture's reference picture set is applied.
#[derive(Clone)]
pub(crate) struct Ref<P> {
    pub pic: P,
    pub poc: i32,
    pub long_term: bool,
    /// Identity of the picture in the buffer.
    pub uid: u64,
    pub set: RpsSet,
}

/// What a backend learns at the first slice segment of a picture.
pub(crate) struct Start<'a, P> {
    pub sps: &'a Arc<Sps>,
    pub pps: &'a Arc<Pps>,
    pub nal: &'a Nal,
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// The picture is output (`PicOutputFlag`).
    pub output: bool,
    /// A later picture may reference this one: it is not a sub-layer
    /// non-reference picture of the highest sub-layer.
    pub reference: bool,
    /// Every picture of the buffer marked as used for reference.
    pub refs: &'a [Ref<P>],
}

/// One slice segment of the picture a backend builds.
pub(crate) struct SliceIn {
    pub nal: Arc<Nal>,
    /// The header, a dependent slice segment's with the values it inherits.
    pub header: SliceHeader,
    /// Byte offset of the slice data in the RBSP after the NAL unit header.
    pub data_offset: usize,
    /// Bits of the `st_ref_pic_set()` in the header of the independent
    /// slice segment.
    pub st_rps_bits: u32,
    /// Where each substream after the first starts, in RBSP bytes from
    /// `data_offset`.
    pub entries: Vec<usize>,
    /// `SliceAddrRs`: the first CTB of the independent slice segment.
    pub slice_addr: u32,
    /// `RefPicList0` and `RefPicList1`, as indices into the picture's
    /// references.
    pub ref_lists: [Vec<usize>; 2],
}

/// What reconstructs the pictures the front end reads.
pub(crate) trait Backend {
    /// A decoded picture, as the buffer holds it.
    type Pic: Clone;
    /// A picture whose slice segments are being read.
    type Building;

    /// The pictures from here on use `sps`, a different SPS from the
    /// last picture's (or the first).
    fn sequence(&mut self, sps: &Arc<Sps>) -> Result<(), CodecError>;
    /// A picture standing in for a missing reference of order count `poc`.
    fn missing(&mut self, sps: &Arc<Sps>, poc: i32) -> Result<Self::Pic, CodecError>;
    /// A picture starts.
    fn start(&mut self, start: Start<'_, Self::Pic>) -> Result<Self::Building, CodecError>;
    /// One more slice segment of the picture.
    fn slice(&mut self, building: &mut Self::Building, slice: SliceIn) -> Result<(), CodecError>;
    /// Every slice segment is read: decode the picture, or start its
    /// decode, and return what the buffer holds. `data` is the packet the
    /// NAL units' spans index.
    fn finish(&mut self, building: Self::Building, data: &[u8]) -> Result<Self::Pic, CodecError>;
}

/// The picture whose slice segments are being read.
struct Current<B> {
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    poc: i32,
    output: bool,
    meta: Arc<Meta>,
    uid: u64,
    /// The last independent slice segment header, for dependent ones, and
    /// the bits of its short-term set.
    independent: Option<(SliceHeader, u32)>,
    slice_addr: u32,
    /// `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter` and
    /// `RefPicSetLtCurr`, as indices into the references handed to the
    /// backend.
    curr: [Vec<usize>; 3],
    building: B,
}

/// Everything the headers of a push may change. A failed push leaves the
/// last good copy.
#[derive(Clone)]
pub(crate) struct Front<P> {
    sps: Vec<Option<Arc<Sps>>>,
    pps: Vec<Option<Arc<Pps>>>,
    dpb: Vec<DpbPic<P>>,
    /// `PicOrderCntVal` of the previous `TemporalId` 0 picture.
    prev_tid0_poc: i32,
    /// An IRAP picture has started a coded video sequence.
    started: bool,
    /// `NoRaslOutputFlag` of the last IRAP picture.
    no_rasl_output: bool,
    /// An end of sequence came before the next picture.
    after_eos: bool,
    next_uid: u64,
    /// The SPS of the last picture started.
    active: Option<Arc<Sps>>,
}

impl<P: Clone> Front<P> {
    pub(crate) fn new() -> Self {
        Self {
            sps: vec![None; MAX_SPS],
            pps: vec![None; MAX_PPS],
            dpb: Vec::new(),
            prev_tid0_poc: 0,
            started: false,
            no_rasl_output: true,
            after_eos: false,
            next_uid: 1,
            active: None,
        }
    }

    fn uid(&mut self) -> u64 {
        let uid = self.next_uid;
        self.next_uid += 1;
        uid
    }

    fn store_param(&mut self, nal: &Nal) -> Result<(), CodecError> {
        match nal.kind {
            nal::SPS => {
                let sps = ps::parse_sps(&nal.rbsp)?;
                let id = usize::from(sps.sps_seq_parameter_set_id);
                self.sps[id] = Some(Arc::new(sps));
            }
            nal::PPS => {
                let pps = ps::parse_pps(&nal.rbsp)?;
                let id = usize::from(pps.pps_pic_parameter_set_id);
                self.pps[id] = Some(Arc::new(pps));
            }
            _ => {}
        }
        Ok(())
    }

    fn sets(&self, pps_id: u8) -> Option<(Arc<Sps>, Arc<Pps>)> {
        let pps = self.pps.get(usize::from(pps_id))?.clone()?;
        let sps = self
            .sps
            .get(usize::from(pps.pps_seq_parameter_set_id))?
            .clone()?;
        Some((sps, pps))
    }

    /// Store the parameter sets of the extradata: an `hvcC` record, whose
    /// NAL unit length size replaces `length_size`, or Annex B.
    pub(crate) fn apply_extra(
        &mut self,
        extra: &[u8],
        length_size: &mut u8,
    ) -> Result<(), CodecError> {
        if extra.is_empty() {
            return Ok(());
        }
        let nals = if nal::is_hvcc(extra) {
            let (size, nals) = nal::hvcc(extra)?;
            *length_size = size;
            nals
        } else {
            nal::access_unit(extra, 0)?
        };
        for nal in &nals {
            self.store_param(nal)?;
        }
        Ok(())
    }

    /// The bumping process: output the picture with the smallest order
    /// count while too many wait (clause C.5.2.4, as the reference decoder
    /// counts them).
    fn bump(&mut self, max_output: usize, max_dpb: usize, out: &mut Vec<Bumped<P>>) {
        loop {
            let waiting = self.dpb.iter().filter(|p| p.output).count();
            if waiting == 0 || (waiting <= max_output && self.dpb.len() <= max_dpb) {
                break;
            }
            let Some(index) = self
                .dpb
                .iter()
                .enumerate()
                .filter(|(_, p)| p.output)
                .min_by_key(|(_, p)| p.poc)
                .map(|(i, _)| i)
            else {
                break;
            };
            self.dpb[index].output = false;
            if let Some(meta) = &self.dpb[index].meta {
                out.push((self.dpb[index].pic.clone(), meta.clone()));
            }
            self.dpb
                .retain(|p| p.output || p.marking != Marking::Unused);
        }
    }

    /// Output every picture waiting, in output order.
    pub(crate) fn flush(&mut self, out: &mut Vec<Bumped<P>>) {
        self.bump(0, usize::MAX, out);
        self.dpb
            .retain(|p| p.output || p.marking != Marking::Unused);
    }

    /// Read one access unit: every picture in it goes to `backend`, and
    /// the pictures chosen for output to `out`.
    pub(crate) fn access_unit<B: Backend<Pic = P>>(
        &mut self,
        backend: &mut B,
        packet: &Packet,
        length_size: u8,
        out: &mut Vec<Bumped<P>>,
    ) -> Result<(), CodecError> {
        let data = &packet.data;
        let nals = nal::access_unit(data, length_size)?;
        let mut current: Option<Current<B::Building>> = None;
        let mut skipping = false;
        for nal in nals {
            if nal.layer != 0 {
                continue;
            }
            match nal.kind {
                nal::SPS | nal::PPS => self.store_param(&nal)?,
                nal::EOS | nal::EOB => {
                    if let Some(done) = current.take() {
                        self.finish(backend, done, data, out)?;
                    }
                    self.after_eos = true;
                }
                _ if nal.is_slice() => {
                    let first = nal.rbsp.first().is_some_and(|b| b & 0x80 != 0);
                    if first {
                        if let Some(done) = current.take() {
                            self.finish(backend, done, data, out)?;
                        }
                        let parsed = self.parse(&nal, None)?;
                        current = self.start_picture(backend, &nal, &parsed, packet, out)?;
                        skipping = current.is_none();
                        if let Some(picture) = current.as_mut() {
                            add_slice(backend, picture, Arc::new(nal), parsed)?;
                        }
                    } else if !skipping && let Some(picture) = current.as_mut() {
                        let prev = picture.independent.as_ref().map(|(h, _)| h);
                        let parsed = self.parse(&nal, prev)?;
                        add_slice(backend, picture, Arc::new(nal), parsed)?;
                    }
                }
                _ => {}
            }
        }
        if let Some(done) = current.take() {
            self.finish(backend, done, data, out)?;
        }
        Ok(())
    }

    fn parse(&self, nal: &Nal, prev: Option<&SliceHeader>) -> Result<Parsed, CodecError> {
        slice::parse_header(nal, &|id| self.sets(id), prev)
    }

    /// Start a picture at its first slice segment: order count, reference
    /// picture set, missing references, and the output of prior pictures.
    /// `None` skips the picture.
    fn start_picture<B: Backend<Pic = P>>(
        &mut self,
        backend: &mut B,
        nal: &Nal,
        parsed: &Parsed,
        packet: &Packet,
        out: &mut Vec<Bumped<P>>,
    ) -> Result<Option<Current<B::Building>>, CodecError> {
        let header = &parsed.header;
        let (sps, pps) = (&parsed.sps, &parsed.pps);
        let irap = nal.is_irap();
        if irap {
            self.no_rasl_output = nal.is_idr() || nal.is_bla() || !self.started || self.after_eos;
            self.started = true;
        } else if !self.started {
            return Ok(None);
        }
        if nal.is_rasl() && self.no_rasl_output {
            return Ok(None);
        }
        let new_sequence = irap && self.no_rasl_output;
        let flushed_by_eos = self.after_eos;
        self.after_eos = false;

        // Picture order count (clause 8.3.1).
        let max_lsb = 1i32 << sps.log2_max_poc_lsb();
        let lsb = header.slice_pic_order_cnt_lsb as i32;
        let msb = if new_sequence {
            0
        } else {
            let prev_lsb = self.prev_tid0_poc & (max_lsb - 1);
            let prev_msb = self.prev_tid0_poc - prev_lsb;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        let poc = msb + lsb;
        if nal.temporal_id == 0 && !nal.is_rasl() && !nal.is_radl() && !nal.is_sub_layer_non_ref() {
            self.prev_tid0_poc = poc;
        }

        // Output of prior pictures at a new sequence (clause C.5.2.2).
        if new_sequence {
            let discard =
                header.no_output_of_prior_pics_flag || (nal.kind == nal::CRA && flushed_by_eos);
            if discard {
                self.dpb.clear();
            } else {
                for p in &mut self.dpb {
                    p.marking = Marking::Unused;
                }
                self.flush(out);
            }
            self.dpb.clear();
        }

        if !self
            .active
            .as_ref()
            .is_some_and(|a| Arc::ptr_eq(a, sps) || **a == **sps)
        {
            backend.sequence(sps)?;
            self.active = Some(sps.clone());
        }

        // Reference picture set (clause 8.3.2).
        let mut keep = vec![Marking::Unused; self.dpb.len()];
        let mut lt_curr = Vec::new();
        for lt in &header.long_term {
            let (value, exact) = lt.target(poc, header.slice_pic_order_cnt_lsb, max_lsb);
            let found = self.dpb.iter().position(|p| {
                p.marking != Marking::Unused
                    && if exact {
                        p.poc == value
                    } else {
                        p.poc & (max_lsb - 1) == value
                    }
            });
            if let Some(i) = found {
                keep[i] = Marking::Long;
            }
            if lt.used_by_curr_pic_lt_flag {
                lt_curr.push((value, found));
            }
        }
        let mut st_lists = [Vec::new(), Vec::new()];
        let rps = &header.st_ref_pic_set;
        let deltas = [
            (&rps.delta_poc_s0, &rps.used_by_curr_pic_s0, 0usize),
            (&rps.delta_poc_s1, &rps.used_by_curr_pic_s1, 1usize),
        ];
        for (values, used, list) in deltas {
            for (&delta, &used) in values.iter().zip(used.iter()) {
                let target = poc + delta;
                let found = self.dpb.iter().enumerate().position(|(i, p)| {
                    p.marking == Marking::Short && keep[i] != Marking::Long && p.poc == target
                });
                if let Some(i) = found {
                    keep[i] = Marking::Short;
                }
                if used {
                    st_lists[list].push((target, found));
                }
            }
        }
        for (p, mark) in self.dpb.iter_mut().zip(&keep) {
            p.marking = *mark;
        }
        // Generate what is missing (clause 8.3.3), never output.
        let [before, after] = st_lists;
        let before = self.resolve(backend, sps, before, false)?;
        let after = self.resolve(backend, sps, after, false)?;
        let lt_curr = self.resolve(backend, sps, lt_curr, true)?;
        self.dpb
            .retain(|p| p.output || p.marking != Marking::Unused);

        // Every picture marked as used for reference, and the lists of the
        // current picture's set as indices into them. Each picture the sets
        // name is among them: it is marked, so it stayed.
        let set_of = |uid: u64| {
            if before.contains(&uid) {
                RpsSet::StCurrBefore
            } else if after.contains(&uid) {
                RpsSet::StCurrAfter
            } else if lt_curr.contains(&uid) {
                RpsSet::LtCurr
            } else {
                RpsSet::Foll
            }
        };
        let refs: Vec<Ref<P>> = self
            .dpb
            .iter()
            .filter(|p| p.marking != Marking::Unused)
            .map(|p| Ref {
                pic: p.pic.clone(),
                poc: p.poc,
                long_term: p.marking == Marking::Long,
                uid: p.uid,
                set: set_of(p.uid),
            })
            .collect();
        let curr = [before, after, lt_curr].map(|uids| {
            uids.iter()
                .filter_map(|&uid| refs.iter().position(|r| r.uid == uid))
                .collect::<Vec<usize>>()
        });

        let output = header.pic_output_flag && !(nal.is_rasl() && self.no_rasl_output);
        let reference =
            !(nal.is_sub_layer_non_ref() && nal.temporal_id >= sps.sps_max_sub_layers_minus1);
        let building = backend.start(Start {
            sps,
            pps,
            nal,
            poc,
            output,
            reference,
            refs: &refs,
        })?;
        let uid = self.uid();
        Ok(Some(Current {
            sps: sps.clone(),
            pps: pps.clone(),
            poc,
            output,
            meta: Arc::new(Meta {
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
                keyframe: irap,
                sps: sps.clone(),
            }),
            uid,
            independent: None,
            slice_addr: 0,
            curr,
            building,
        }))
    }

    /// The buffer pictures of the order counts in `wanted`, each found in
    /// the buffer at its index or else generated and added: their
    /// identities.
    fn resolve<B: Backend<Pic = P>>(
        &mut self,
        backend: &mut B,
        sps: &Arc<Sps>,
        wanted: Vec<(i32, Option<usize>)>,
        long: bool,
    ) -> Result<Vec<u64>, CodecError> {
        let mut uids = Vec::with_capacity(wanted.len());
        for (value, found) in wanted {
            let uid = match found.and_then(|i| self.dpb.get(i)) {
                Some(p) => p.uid,
                None => {
                    let pic = backend.missing(sps, value)?;
                    let uid = self.uid();
                    self.dpb.push(DpbPic {
                        pic,
                        poc: value,
                        marking: if long { Marking::Long } else { Marking::Short },
                        output: false,
                        uid,
                        meta: None,
                    });
                    uid
                }
            };
            uids.push(uid);
        }
        Ok(uids)
    }

    /// The picture's slice segments are all read: the backend decodes it
    /// and it enters the picture buffer (clause C.5.2.3).
    fn finish<B: Backend<Pic = P>>(
        &mut self,
        backend: &mut B,
        current: Current<B::Building>,
        data: &[u8],
        out: &mut Vec<Bumped<P>>,
    ) -> Result<(), CodecError> {
        let Current {
            sps,
            poc,
            output,
            meta,
            uid,
            building,
            ..
        } = current;
        let pic = backend.finish(building, data)?;
        self.dpb.push(DpbPic {
            pic,
            poc,
            marking: Marking::Short,
            output,
            uid,
            meta: Some(meta),
        });
        self.bump(sps.max_num_reorder(), sps.max_dec_pic_buffering(), out);
        Ok(())
    }
}

/// Record one slice segment of the current picture.
fn add_slice<B: Backend>(
    backend: &mut B,
    picture: &mut Current<B::Building>,
    nal: Arc<Nal>,
    parsed: Parsed,
) -> Result<(), CodecError> {
    let Parsed {
        header,
        data_offset,
        st_rps_bits,
        ..
    } = parsed;
    let st_rps_bits = if !header.dependent_slice_segment_flag {
        picture.slice_addr = header.slice_segment_address;
        picture.independent = Some((header.clone(), st_rps_bits));
        st_rps_bits
    } else if let Some((_, bits)) = &picture.independent {
        *bits
    } else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "dependent slice without a slice",
        });
    };
    if picture.pps.pps_pic_parameter_set_id != header.slice_pic_parameter_set_id {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pps changed inside a picture",
        });
    }
    let ref_lists = ref_lists(&picture.curr, &header)?;
    // Substream starts, from coded offsets to RBSP offsets in the data.
    let Some(data) = nal.rbsp.get(data_offset..) else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "slice data",
        });
    };
    let mut entries = Vec::with_capacity(header.entry_point_offset_minus1.len());
    let mut raw = nal.raw_of(data_offset);
    for &minus1 in &header.entry_point_offset_minus1 {
        raw = raw.saturating_add(minus1.saturating_add(1) as usize);
        let at = nal.rbsp_of(raw).saturating_sub(data_offset);
        if at > data.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "entry point",
            });
        }
        entries.push(at);
    }
    let sps = &picture.sps;
    if header.slice_segment_address >= sps.ctb_width() * sps.ctb_height() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice runs past the picture",
        });
    }
    backend.slice(
        &mut picture.building,
        SliceIn {
            nal,
            header,
            data_offset,
            st_rps_bits,
            entries,
            slice_addr: picture.slice_addr,
            ref_lists,
        },
    )
}

/// Reference picture lists of one slice (clause 8.3.4): `RefPicList0` and
/// `RefPicList1` as indices, from the indices of `RefPicSetStCurrBefore`,
/// `RefPicSetStCurrAfter` and `RefPicSetLtCurr` in `curr`.
fn ref_lists(curr: &[Vec<usize>; 3], header: &SliceHeader) -> Result<[Vec<usize>; 2], CodecError> {
    if header.slice_type == SliceType::I {
        return Ok([Vec::new(), Vec::new()]);
    }
    let [before, after, lt_curr] = curr;
    let total = before.len() + after.len() + lt_curr.len();
    if total == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter slice without references",
        });
    }
    let mut lists = [Vec::new(), Vec::new()];
    let count = if header.slice_type == SliceType::B {
        2
    } else {
        1
    };
    let num_ref_idx = header.num_ref_idx_active();
    for (list, out) in lists.iter_mut().enumerate().take(count) {
        let wanted = num_ref_idx[list];
        let size = wanted.max(total);
        let order: [&Vec<usize>; 3] = if list == 0 {
            [before, after, lt_curr]
        } else {
            [after, before, lt_curr]
        };
        let mut temp = Vec::with_capacity(size);
        while temp.len() < size {
            for part in order {
                for &r in part {
                    if temp.len() < size {
                        temp.push(r);
                    }
                }
            }
        }
        for i in 0..wanted {
            let index = match header.list_entry(list) {
                Some(entries) => *entries.get(i).ok_or(CodecError::Malformed {
                    at_bit: 0,
                    what: "list_entry",
                })? as usize,
                None => i,
            };
            let Some(&r) = temp.get(index) else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "list_entry",
                });
            };
            out.push(r);
        }
    }
    Ok(lists)
}
