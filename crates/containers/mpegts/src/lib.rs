//! MPEG transport stream demux and mux (ITU-T H.222.0 | ISO/IEC 13818-1).
//!
//! The support record, limits, and measurements are in `README.md` next to
//! this crate.

#![forbid(unsafe_code)]

pub mod crc;
mod demux;
pub mod es;
mod limits;
mod mux;
pub mod packet;
pub mod pes;
pub mod pipeline;
mod probe;
pub mod psi;
pub mod section;
mod source;
pub mod time;

pub use demux::{Program, SeekCost, TIMESCALE, TsDemux};
pub use limits::{
    MAX_BUFFERED_BYTES, MAX_DESCRIPTORS, MAX_HELD_PACKETS, MAX_OPEN_SCAN_BYTES, MAX_PARAMS_BYTES,
    MAX_PARAMS_PES, MAX_PROBE_WINDOW_BYTES, MAX_PROGRAMS, MAX_RANDOM_ACCESS_SCAN_BYTES,
    MAX_RESYNC_BYTES, MAX_SECTION_BYTES, MAX_SEEK_PROBES, MAX_SEEK_WIDENINGS, MAX_TAIL_SCAN_BYTES,
    PROBE_BYTES, SYNC_BYTE, SYNC_RUN, TS_PACKET_BYTES,
};
pub use mux::{MuxOptions, TsMux};
pub use pipeline::Counters;
pub use probe::{TsFamily, probe};
