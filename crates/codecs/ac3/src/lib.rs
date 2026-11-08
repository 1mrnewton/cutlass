//! AC-3 and E-AC-3 (ATSC A/52:2018, including Annex E).
//!
//! The support record is `STATUS.md` next to this crate.

#![forbid(unsafe_code)]

mod aht;
mod bitalloc;
mod bits;
mod crc;
mod exponents;
mod header;
mod mdct;
mod spx;
mod tables;
