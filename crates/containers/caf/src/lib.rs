//! Core Audio Format framing.
//!
//! The crate walks the chunks of a `.caf` file, finds the codec, its
//! extradata and the packet boundaries, and returns packets timed in frames
//! at the file's sample rate. It does not decode the sound. It also writes
//! packets back into a CAF file.
//!
//! Normative layout: Apple's *Core Audio Format Specification* and
//! `CAFFile.h` (AudioToolbox). Every multi-byte field is big-endian. The
//! crate's README holds the support matrix and the readiness report.

#![forbid(unsafe_code)]

mod bytes;
mod chan;
mod codec;
mod cookie;
mod demux;
mod desc;
mod error;
mod features;
mod info;
mod limits;
mod mux;
mod pakt;
mod probe;
mod varint;
mod walk;

pub use chan::{
    ChannelDescription, ChannelLayout, TAG_MONO, TAG_STEREO, TAG_USE_BITMAP, TAG_USE_DESCRIPTIONS,
};
pub use demux::CafDemux;
pub use desc::{Description, FLAG_FLOAT, FLAG_LITTLE_ENDIAN};
pub use features::{FEATURES, Feature};
pub use info::InfoEntry;
pub use limits::{
    CHECKPOINT_EVERY, MAX_BITS_PER_CHANNEL, MAX_CHAN_BYTES, MAX_CHANNELS, MAX_CHUNKS,
    MAX_FRAMES_PER_PACKET, MAX_INFO_BYTES, MAX_VARINT_BYTES, PCM_PACKET_BYTES, PROBE_BYTES,
};
pub use mux::CafMux;
pub use pakt::PacketTableHeader;
pub use probe::{CafFamily, probe};
pub use walk::Chunk;

#[cfg(test)]
mod tests;
