//! AVI and OpenDML AVI 2.0 framing.
//!
//! The crate finds streams, codec ids, extradata, chunk tables, and
//! timestamps in each stream's own `dwRate / dwScale` clock. It does not
//! decode payloads.
//!
//! Normative layout: Microsoft "AVI RIFF File Reference" (`AVIMAINHEADER`,
//! `AVISTREAMHEADER`, `AVIOLDINDEX`, `AVISUPERINDEX`, `AVISTDINDEX`) and
//! the OpenDML AVI File Format Extensions 1.02. Where those leave a choice
//! open, this crate reads files the way FFmpeg's demuxer does, so times
//! agree with `ffprobe`; each such choice is documented where it is made.

#![forbid(unsafe_code)]

mod bytes;
mod demux;
mod format;
mod header;
mod index;
mod layout;
mod limits;
mod mux;
mod probe;
mod sniff;
mod source;
mod track;

pub use demux::AviDemux;
pub use format::{StreamKind, VideoFormat};
pub use header::{
    AVIF_HASINDEX, AVIF_ISINTERLEAVED, AVIF_MUSTUSEINDEX, AVIF_TRUSTCKTYPE, MainHeader,
    StreamHeader,
};
pub use index::{Fallback, IndexSource};
pub use limits::{
    MAX_CHUNKS, MAX_DRIFT_SECONDS, MAX_RESYNC, MAX_RIFFS, MAX_SKIPPED_FRAMES, MAX_STREAMS,
    MAX_SUPER_ENTRIES, PROBE_BYTES, SNIFF_BYTES,
};
pub use mux::{AviMux, AviMuxOptions};
pub use probe::{AviFamily, probe};

#[cfg(test)]
mod behavior;
