//! Demux and mux traits.
//!
//! A container owns framing: which bytes are one packet, which bytes are
//! extradata, which codec id that stream has, and the timestamps in the
//! file timescale. It does not decode pictures or samples, and it does not
//! convert those timestamps to microseconds. Format parsers live under
//! `crates/containers`. `engine` and `export` call these traits.
//!
//! # Scope
//!
//! This crate is the shared contract. It does not open MP4, MP3, or any
//! other file format. [`ListDemux`] and [`MuxDraft`] hold packets the
//! caller already split so the contract can run without a family crate.
//! `engine`'s reader and `export`'s writer are still empty; those paths
//! are not exercised here.
//!
//! Timestamps are ticks of a [`cutlass_codec::Timescale`]. An edit list,
//! encoder delay, trailing padding, and gaps are exposed on the stream and
//! are not applied to packet timestamps. One read returns one access unit,
//! in decode order. An unknown codec is [`cutlass_codec::StreamCodec::Unknown`]
//! on that stream. A failed mux is a [`FailedMux`], which has no durations.
//!
//! # Limits
//!
//! [`Limits::standard`] caps one header, one access unit, the track count,
//! one extradata blob, one edit list, and one index. It does not cap the
//! length of a file. The input is not buffered whole.
//!
//! # Capability matrix
//!
//! | Capability | Demux | Mux | Status |
//! | --- | --- | --- | --- |
//! | Probe window | Accept, refuse, or need more bytes | Outside this crate | Verified on the trait |
//! | Owned input | Random or forward, consumed bytes, hostile size refused before allocation | Writes stay in the draft until finish | Verified |
//! | Streams | Track, codec, extradata, timescale, language, frame size or rate and channels | Add stream, reject unknown codec | Verified |
//! | Unknown codec | Typed [`cutlass_codec::StreamCodec::Unknown`], stream kept | [`MuxError::UnknownCodec`], not complete | Verified |
//! | Packets | One access unit, decode order, file timescale | Payload bytes unchanged | Verified |
//! | Edit, trim, gaps | Exposed, not applied to timestamps | Not a document clock | Verified |
//! | Seek | Keyframe landing, preroll, index or scan, stale packets dropped | Outside this crate | Verified |
//! | Open failure | No streams published | — | Verified |
//! | Read failure | No packet from that call, streams remain | — | Verified |
//! | Finish | — | Durations only on [`FinishedMux`] | Verified |
//! | Fragmented output | Outside this crate | Mode is distinct from progressive | Verified as a flag |
//! | Real file formats | Outside this crate | Outside this crate | Not run |
//!
//! A family crate fills the format rows. Fuzzing a byte parser is not run
//! here because this crate does not parse one.

#![forbid(unsafe_code)]

mod demux;
mod error;
mod input;
mod limits;
mod list;
mod mux;
mod probe;
mod seek;
mod stream;
mod time;

pub use demux::Demux;
pub use error::{MuxError, OpenError, ReadError, SeekError, SourceError};
pub use input::{Access, Input};
pub use limits::Limits;
pub use list::{ListDemux, ListRequest, OpenFailure, PacketPlan};
pub use mux::{
    FailedMux, FinishedMux, IndexPlace, Mux, MuxDraft, MuxMode, MuxStatus, NewStream, TrackDuration,
};
pub use probe::{HeaderVerdict, Probe, ProbeResult, decide_probe};
pub use seek::{
    IndexEntry, Landing, ScanEntry, SeekLanding, SeekTo, ensure_index_sorted, reserve_index,
    seek_by_scan, seek_with_index, suppress_stale,
};
pub use stream::{Edit, EditError, Gap, GapError, Language, LanguageError, Stream, TrackId, Trim};
pub use time::{CompareError, compare_ticks};

#[cfg(test)]
mod contract;
