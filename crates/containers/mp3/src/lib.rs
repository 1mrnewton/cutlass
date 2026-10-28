//! MPEG audio Layer III framing.
//!
//! The file is a sequence of Layer III frames, optionally wrapped by an
//! ID3v2 tag at the start and an ID3v1 or APEv2 tag at the end. A Xing,
//! Info, or VBRI header in the first frame carries duration and a seek
//! index. This crate does not decode samples. Frame bytes, including the
//! four-byte header, are the packet a Layer III decoder accepts.
//!
//! Normative layout: ISO/IEC 11172-3:1993 and ISO/IEC 13818-3:1998.
//! MPEG-2.5 sample rates are the Fraunhofer extension. The Xing/Info tag
//! and the LAME delay fields follow the tag LAME writes; VBRI follows the
//! Fraunhofer header at byte 36 of the first frame. ID3v2 sizes are the
//! synchsafe integers in the id3.org 2.4.0 header.

#![forbid(unsafe_code)]

mod demux;
mod header;
mod id3;
mod limits;
mod mux;
mod probe;
mod tag;

pub use cutlass_container::{ContainerError, Demux, Mux, Probe, SeekLanding, Stream, Trim};
pub use demux::{IndexKind, Mp3Demux};
pub use limits::{MAX_FRAME_BYTES, MAX_ID3_BYTES, MAX_RESYNC_BYTES, SEEK_WINDOW_BYTES};
pub use mux::Mp3Mux;
pub use probe::probe;

#[cfg(test)]
mod property;
