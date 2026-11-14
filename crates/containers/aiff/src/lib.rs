//! AIFF and AIFF-C framing.
//!
//! The crate finds `COMM`, `SSND`, the codec id, extradata, and timestamps
//! in the file timescale. The timescale is the sample rate. It does not
//! decode samples.
//!
//! Normative layout: Audio Interchange File Format 1.3 (Apple, 1989) and
//! AIFF-C (Apple, 1991). A declared size that runs past its parent is an
//! error. One linear packet is one sample frame. A packetized compression
//! block stays compressed, and its duration is the sample frames that block
//! represents. `QDM2`, `QDMC`, `Qclp`, `G722`, `G726`, `G728`, and any other
//! unnamed fourcc are refused.
//!
//! PCM extradata is the three-byte layout the PCM decoder reads: coding,
//! endian, then interleaved. Coding byte 6 is signed 8-bit. Float and
//! little-endian integer are written as AIFF-C.
//! Big-endian integer is written as AIFF.

#![forbid(unsafe_code)]

mod demux;
mod format;
mod limits;
mod mux;
mod probe;
mod rate;
mod walk;

pub use demux::AiffDemux;
pub use mux::AiffMux;
pub use probe::{AiffFamily, probe};

#[cfg(test)]
mod behavior;

/// One row of the support matrix. A `false` flag is unsupported in this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Feature {
    pub name: &'static str,
    pub demux: bool,
    pub mux: bool,
}

/// What this build opens and writes. Unsupported rows stay in the list.
pub const FEATURES: &[Feature] = &[
    Feature {
        name: "aiff",
        demux: true,
        mux: true,
    },
    Feature {
        name: "aifc",
        demux: true,
        mux: true,
    },
    Feature {
        name: "pcm-integer-be",
        demux: true,
        mux: true,
    },
    Feature {
        name: "pcm-integer-le",
        demux: true,
        mux: true,
    },
    Feature {
        name: "pcm-float",
        demux: true,
        mux: true,
    },
    Feature {
        name: "pcm-8",
        demux: true,
        mux: true,
    },
    Feature {
        name: "alaw-ulaw",
        demux: true,
        mux: false,
    },
    Feature {
        name: "ima4",
        demux: true,
        mux: false,
    },
    Feature {
        name: "mace",
        demux: true,
        mux: false,
    },
    Feature {
        name: "gsm",
        demux: true,
        mux: false,
    },
    Feature {
        name: "other-compression",
        demux: false,
        mux: false,
    },
    Feature {
        name: "ssnd-offset",
        demux: true,
        mux: false,
    },
    Feature {
        name: "unknown-chunk",
        demux: true,
        mux: false,
    },
];
