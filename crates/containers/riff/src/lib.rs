//! WAV and RF64 framing.
//!
//! The crate finds chunks, the codec id, extradata, and timestamps in the
//! file timescale. The timescale is the sample rate. It does not decode
//! samples, and it does not turn a compressed payload into PCM.
//!
//! Normative layout: Multimedia Programming Interface and Data Specifications
//! 1.0 (IBM/Microsoft, 1991), Microsoft WAVEFORMATEX and WAVEFORMATEXTENSIBLE,
//! and EBU Tech 3306 for RF64. A declared size that runs past its parent is
//! an error. FFmpeg, GStreamer, and libsndfile sometimes clip that size;
//! this crate follows the declared bound.

#![forbid(unsafe_code)]

mod bytes;
mod demux;
mod format;
mod limits;
mod mpeg;
mod mux;
mod probe;
mod walk;

pub use demux::WavDemux;
pub use format::WavFormat;
pub use limits::{
    CHANNEL_MASK_OFFSET, EXTRA_HEADER, MAX_CHUNKS, MAX_DEPTH, MAX_DS64_ROWS, MAX_KEPT_CHUNK,
    MAX_MPEG_RESYNC, PROBE_BYTES,
};
pub use mux::WavMux;
pub use probe::{WavFamily, probe};
pub use walk::{Brand, DroppedChunk, KeptChunk};

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
        name: "riff",
        demux: true,
        mux: true,
    },
    Feature {
        name: "rf64",
        demux: true,
        mux: true,
    },
    Feature {
        name: "pcm-integer",
        demux: true,
        mux: true,
    },
    Feature {
        name: "pcm-float",
        demux: true,
        mux: true,
    },
    Feature {
        name: "waveformatextensible",
        demux: true,
        mux: true,
    },
    Feature {
        name: "alaw-mulaw",
        demux: true,
        mux: false,
    },
    Feature {
        name: "mpeg-in-wav",
        demux: true,
        mux: true,
    },
    Feature {
        name: "adpcm-blocks",
        demux: true,
        mux: true,
    },
    Feature {
        name: "unknown-codec",
        demux: true,
        mux: false,
    },
    Feature {
        name: "unknown-chunk-kept",
        demux: true,
        mux: true,
    },
    Feature {
        name: "unknown-chunk-dropped",
        demux: true,
        mux: false,
    },
    Feature {
        name: "fact",
        demux: true,
        mux: true,
    },
    Feature {
        name: "ds64",
        demux: true,
        mux: true,
    },
    Feature {
        name: "odd-pad",
        demux: true,
        mux: true,
    },
    Feature {
        name: "rifx",
        demux: false,
        mux: false,
    },
    Feature {
        name: "bw64",
        demux: false,
        mux: false,
    },
    Feature {
        name: "aiff",
        demux: false,
        mux: false,
    },
    Feature {
        name: "wavl",
        demux: false,
        mux: false,
    },
    Feature {
        name: "smv",
        demux: false,
        mux: false,
    },
    Feature {
        name: "language",
        demux: false,
        mux: false,
    },
    Feature {
        name: "sample-decode",
        demux: false,
        mux: false,
    },
];

#[cfg(test)]
mod tests {
    use super::{FEATURES, Feature};

    fn feature(name: &str) -> Feature {
        *FEATURES
            .iter()
            .find(|feature| feature.name == name)
            .expect(name)
    }

    #[test]
    fn the_support_matrix_names_implemented_and_unsupported_rows() {
        assert!(feature("riff").demux && feature("riff").mux);
        assert!(feature("rf64").demux && feature("pcm-float").demux);
        assert!(feature("waveformatextensible").mux);
        assert!(feature("unknown-codec").demux);
        assert!(!feature("unknown-codec").mux);
        assert!(feature("alaw-mulaw").demux);
        assert!(!feature("alaw-mulaw").mux);
        assert!(feature("unknown-chunk-kept").mux);
        assert!(!feature("unknown-chunk-dropped").mux);
        for name in [
            "rifx",
            "bw64",
            "aiff",
            "wavl",
            "smv",
            "language",
            "sample-decode",
        ] {
            let row = feature(name);
            assert!(!row.demux && !row.mux, "{name}");
        }
    }
}
