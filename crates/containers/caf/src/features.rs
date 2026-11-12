//! The support matrix.

/// One row of the support matrix. A `false` flag is unsupported in this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Feature {
    pub name: &'static str,
    pub demux: bool,
    pub mux: bool,
}

/// What this build opens and writes. Unsupported rows stay in the list; the
/// README says what each row means and how it was checked.
pub const FEATURES: &[Feature] = &[
    Feature {
        name: "caf-version-1",
        demux: true,
        mux: true,
    },
    Feature {
        name: "lpcm-integer",
        demux: true,
        mux: true,
    },
    Feature {
        name: "lpcm-float",
        demux: true,
        mux: true,
    },
    Feature {
        name: "alaw-ulaw",
        demux: true,
        mux: true,
    },
    Feature {
        name: "aac",
        demux: true,
        mux: true,
    },
    Feature {
        name: "alac",
        demux: true,
        mux: true,
    },
    Feature {
        name: "flac",
        demux: true,
        mux: true,
    },
    Feature {
        name: "opus",
        demux: true,
        mux: true,
    },
    Feature {
        name: "mp3",
        demux: true,
        mux: true,
    },
    Feature {
        name: "ac-3",
        demux: true,
        mux: true,
    },
    Feature {
        name: "e-ac-3",
        demux: true,
        mux: true,
    },
    Feature {
        name: "other-format-framed",
        demux: true,
        mux: false,
    },
    Feature {
        name: "packet-table",
        demux: true,
        mux: true,
    },
    Feature {
        name: "priming-remainder",
        demux: true,
        mux: true,
    },
    Feature {
        name: "data-size-unknown",
        demux: true,
        mux: true,
    },
    Feature {
        name: "chan",
        demux: true,
        mux: true,
    },
    Feature {
        name: "info",
        demux: true,
        mux: true,
    },
    Feature {
        name: "other-chunks",
        demux: true,
        mux: false,
    },
    Feature {
        name: "seek",
        demux: true,
        mux: false,
    },
    Feature {
        name: "caf-version-other",
        demux: false,
        mux: false,
    },
];

#[cfg(test)]
mod tests {
    use super::FEATURES;

    #[test]
    fn each_row_is_named_once() {
        let mut names = Vec::new();
        for feature in FEATURES {
            assert!(!names.contains(&feature.name), "{}", feature.name);
            names.push(feature.name);
        }
        let row = |name: &str| {
            FEATURES
                .iter()
                .find(|feature| feature.name == name)
                .expect(name)
        };
        for name in [
            "lpcm-integer",
            "aac",
            "alac",
            "flac",
            "opus",
            "packet-table",
        ] {
            assert!(row(name).demux && row(name).mux, "{name}");
        }
        assert!(row("other-format-framed").demux && !row("other-format-framed").mux);
        assert!(!row("caf-version-other").demux);
    }
}
