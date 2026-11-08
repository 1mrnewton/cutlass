//! The Vorbis encoder: generated books, a masking floor, and residue
//! quantization. Packet timing and the [`cutlass_encode::Encoder`] wrapper
//! live in [`encode`].

mod books;
mod encode;
mod floorfit;
mod profile;
mod psy;
mod quant;
mod resenc;

pub use encode::{Options, VorbisEncoder};
