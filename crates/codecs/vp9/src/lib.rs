//! VP9 profile 0 decoder and encoder.
//!
//! The support record is `STATUS.md` next to this crate. Profiles 1, 2, and 3
//! are rejected. Playback and a WebM container are outside this crate.

#![forbid(unsafe_code)]

mod bits;
mod coef;
mod decode;
mod geom;
mod header;
mod intra;
mod lpf;
mod prob;
mod recon;
mod tables;
mod txfm;

pub use decode::Decoder;

#[cfg(test)]
mod tests {
    use super::tables::COEF_PROBS;

    #[test]
    fn default_dc_coef_prob_matches_libvpx() {
        assert_eq!(COEF_PROBS[0][0][0][0][0], [195, 29, 183]);
    }
}
