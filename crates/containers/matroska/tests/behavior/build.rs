//! Hand-built Matroska files for the behavior tests.
//!
//! Each helper returns the bytes of one element, header included, so a test
//! can lay out any structure: known or unknown sizes, Cues before or after
//! the Clusters, laced and grouped blocks, and broken sizes.

use cutlass_matroska::ebml::{
    UNKNOWN_SIZE, put_bytes, put_float, put_id, put_master, put_size_width, put_uint,
};
use cutlass_matroska::ids;

pub fn element(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    put_master(&mut out, id, body);
    out
}

pub fn unknown(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    put_id(&mut out, id);
    out.extend_from_slice(&UNKNOWN_SIZE);
    out.extend_from_slice(body);
    out
}

pub fn uint(id: u32, value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    put_uint(&mut out, id, value);
    out
}

pub fn bytes(id: u32, value: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    put_bytes(&mut out, id, value);
    out
}

pub fn ebml_header(doc_type: &str) -> Vec<u8> {
    let mut body = Vec::new();
    put_uint(&mut body, ids::EBML_VERSION, 1);
    put_uint(&mut body, ids::EBML_READ_VERSION, 1);
    put_uint(&mut body, ids::EBML_MAX_ID_LENGTH, 4);
    put_uint(&mut body, ids::EBML_MAX_SIZE_LENGTH, 8);
    put_bytes(&mut body, ids::DOC_TYPE, doc_type.as_bytes());
    put_uint(&mut body, ids::DOC_TYPE_VERSION, 4);
    put_uint(&mut body, ids::DOC_TYPE_READ_VERSION, 2);
    element(ids::EBML, &body)
}

pub fn info(scale: u64, duration: Option<f64>) -> Vec<u8> {
    let mut body = Vec::new();
    put_uint(&mut body, ids::TIMESTAMP_SCALE, scale);
    if let Some(duration) = duration {
        put_float(&mut body, ids::DURATION, duration);
    }
    put_bytes(&mut body, ids::MUXING_APP, b"test");
    element(ids::INFO, &body)
}

#[derive(Clone)]
pub struct Track {
    pub number: u64,
    pub kind: u64,
    pub codec: &'static str,
    pub private: Vec<u8>,
    pub default_duration: Option<u64>,
    /// Raw children appended to the TrackEntry.
    pub extra: Vec<u8>,
}

impl Track {
    pub fn video(number: u64, codec: &'static str) -> Self {
        let mut extra = Vec::new();
        let mut video = Vec::new();
        put_uint(&mut video, ids::PIXEL_WIDTH, 64);
        put_uint(&mut video, ids::PIXEL_HEIGHT, 48);
        put_master(&mut extra, ids::VIDEO, &video);
        Self {
            number,
            kind: 1,
            codec,
            private: Vec::new(),
            default_duration: None,
            extra,
        }
    }

    pub fn audio(number: u64, codec: &'static str, rate: f64, channels: u64) -> Self {
        let mut extra = Vec::new();
        let mut audio = Vec::new();
        put_float(&mut audio, ids::SAMPLING_FREQUENCY, rate);
        put_uint(&mut audio, ids::CHANNELS, channels);
        put_master(&mut extra, ids::AUDIO, &audio);
        Self {
            number,
            kind: 2,
            codec,
            private: Vec::new(),
            default_duration: None,
            extra,
        }
    }

    pub fn with_private(mut self, private: &[u8]) -> Self {
        self.private = private.to_vec();
        self
    }

    pub fn with_default_duration(mut self, ns: u64) -> Self {
        self.default_duration = Some(ns);
        self
    }

    pub fn with_extra(mut self, extra: &[u8]) -> Self {
        self.extra.extend_from_slice(extra);
        self
    }

    pub fn entry(&self) -> Vec<u8> {
        let mut body = Vec::new();
        put_uint(&mut body, ids::TRACK_NUMBER, self.number);
        put_uint(&mut body, ids::TRACK_UID, self.number + 1000);
        put_uint(&mut body, ids::TRACK_TYPE, self.kind);
        put_bytes(&mut body, ids::CODEC_ID, self.codec.as_bytes());
        if !self.private.is_empty() {
            put_bytes(&mut body, ids::CODEC_PRIVATE, &self.private);
        }
        if let Some(ns) = self.default_duration {
            put_uint(&mut body, ids::DEFAULT_DURATION, ns);
        }
        body.extend_from_slice(&self.extra);
        element(ids::TRACK_ENTRY, &body)
    }
}

pub fn tracks(list: &[Track]) -> Vec<u8> {
    let body: Vec<u8> = list.iter().flat_map(Track::entry).collect();
    element(ids::TRACKS, &body)
}

/// The block header: track (one octet), relative time, flags.
pub fn block_head(track: u64, relative: i16, flags: u8) -> Vec<u8> {
    let mut out = Vec::new();
    put_size_width(&mut out, track, 1);
    out.extend_from_slice(&relative.to_be_bytes());
    out.push(flags);
    out
}

pub fn simple(track: u64, relative: i16, key: bool, payload: &[u8]) -> Vec<u8> {
    let mut body = block_head(track, relative, if key { 0x80 } else { 0 });
    body.extend_from_slice(payload);
    element(ids::SIMPLE_BLOCK, &body)
}

/// A SimpleBlock with Xiph lacing of `frames`.
pub fn xiph(track: u64, relative: i16, frames: &[&[u8]]) -> Vec<u8> {
    let mut body = block_head(track, relative, 0x80 | 0x02);
    body.push(u8::try_from(frames.len() - 1).expect("count"));
    let sizes: Vec<usize> = frames[..frames.len() - 1].iter().map(|f| f.len()).collect();
    body.extend(cutlass_matroska::block::xiph_sizes(&sizes));
    for frame in frames {
        body.extend_from_slice(frame);
    }
    element(ids::SIMPLE_BLOCK, &body)
}

/// A SimpleBlock with fixed-size lacing.
pub fn fixed(track: u64, relative: i16, frames: &[&[u8]]) -> Vec<u8> {
    let mut body = block_head(track, relative, 0x80 | 0x04);
    body.push(u8::try_from(frames.len() - 1).expect("count"));
    for frame in frames {
        body.extend_from_slice(frame);
    }
    element(ids::SIMPLE_BLOCK, &body)
}

/// A SimpleBlock with EBML lacing (two-octet size codes).
pub fn ebml_laced(track: u64, relative: i16, frames: &[&[u8]]) -> Vec<u8> {
    let mut body = block_head(track, relative, 0x80 | 0x06);
    body.push(u8::try_from(frames.len() - 1).expect("count"));
    let sizes: Vec<i64> = frames
        .iter()
        .map(|f| i64::try_from(f.len()).expect("len"))
        .collect();
    put_size_width(&mut body, u64::try_from(sizes[0]).expect("size"), 2);
    for pair in sizes[..sizes.len() - 1].windows(2) {
        put_size_width(
            &mut body,
            u64::try_from(pair[1] - pair[0] + 8191).expect("delta"),
            2,
        );
    }
    for frame in frames {
        body.extend_from_slice(frame);
    }
    element(ids::SIMPLE_BLOCK, &body)
}

/// A BlockGroup: a Block, an optional BlockDuration, and a ReferenceBlock
/// when `reference` is set.
pub fn group(
    track: u64,
    relative: i16,
    payload: &[u8],
    duration: Option<u64>,
    reference: Option<i64>,
) -> Vec<u8> {
    let mut block = block_head(track, relative, 0);
    block.extend_from_slice(payload);
    let mut body = element(ids::BLOCK, &block);
    if let Some(duration) = duration {
        body.extend(uint(ids::BLOCK_DURATION, duration));
    }
    if let Some(reference) = reference {
        body.extend(bytes(ids::REFERENCE_BLOCK, &reference.to_be_bytes()[7..]));
    }
    element(ids::BLOCK_GROUP, &body)
}

pub fn cluster(timestamp: u64, blocks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = uint(ids::TIMESTAMP, timestamp);
    for block in blocks {
        body.extend_from_slice(block);
    }
    element(ids::CLUSTER, &body)
}

pub fn unknown_cluster(timestamp: u64, blocks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = uint(ids::TIMESTAMP, timestamp);
    for block in blocks {
        body.extend_from_slice(block);
    }
    unknown(ids::CLUSTER, &body)
}

/// Cues from (time, track, cluster Segment Position).
pub fn cues(points: &[(u64, u64, u64)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (time, track, position) in points {
        let mut positions = uint(ids::CUE_TRACK, *track);
        positions.extend(uint(ids::CUE_CLUSTER_POSITION, *position));
        let mut point = uint(ids::CUE_TIME, *time);
        point.extend(element(ids::CUE_TRACK_POSITIONS, &positions));
        body.extend(element(ids::CUE_POINT, &point));
    }
    element(ids::CUES, &body)
}

pub fn seek_head(entries: &[(u32, u64)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (id, position) in entries {
        let mut seek = bytes(
            ids::SEEK_ID,
            &id.to_be_bytes()[4 - cutlass_matroska::ebml::id_width(*id)..],
        );
        seek.extend(uint(ids::SEEK_POSITION, *position));
        body.extend(element(ids::SEEK, &seek));
    }
    element(ids::SEEK_HEAD, &body)
}

/// A whole file: EBML header, then a Segment of `children`.
pub fn file(doc_type: &str, children: &[Vec<u8>]) -> Vec<u8> {
    let mut out = ebml_header(doc_type);
    let body: Vec<u8> = children.concat();
    out.extend(element(ids::SEGMENT, &body));
    out
}

/// A whole file with an unknown-size Segment, as a live recorder writes.
pub fn live_file(doc_type: &str, children: &[Vec<u8>]) -> Vec<u8> {
    let mut out = ebml_header(doc_type);
    out.extend(unknown(ids::SEGMENT, &children.concat()));
    out
}

/// Builds a Segment of `children` where the SeekHead (first) points at the
/// elements it names, wherever they ended up.
pub fn with_seek_head(wanted: &[u32], children: &[Vec<u8>]) -> Vec<u8> {
    let mut positions: Vec<(u32, u64)> = wanted.iter().map(|id| (*id, 0)).collect();
    loop {
        let head = seek_head(&positions);
        let mut at = head.len() as u64;
        let mut next = positions.clone();
        for child in children {
            let id = cutlass_matroska::ebml::read_id(child).expect("id").0;
            for entry in &mut next {
                if entry.0 == id {
                    entry.1 = at;
                }
            }
            at += child.len() as u64;
        }
        if next == positions {
            let mut all = vec![head];
            all.extend_from_slice(children);
            return file("matroska", &all);
        }
        positions = next;
    }
}
