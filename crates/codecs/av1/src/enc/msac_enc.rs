//! Placeholder until the symbol encoder lands.
#![allow(dead_code)]
#[derive(Default)]
pub struct MsacEnc;
impl MsacEnc {
    pub fn new(_allow_update: bool) -> Self {
        Self
    }
    pub fn bool_adapt(&mut self, _cdf: &mut [u16; 2], _bit: bool) {}
    pub fn bool_prob(&mut self, _f: u32, _bit: bool) {}
    pub fn symbol<const N: usize>(&mut self, _cdf: &mut [u16; N], _s: usize) {}
    pub fn literal(&mut self, _n: u32, _v: u32) {}
    pub fn golomb(&mut self, _v: u32) {}
    pub fn ns(&mut self, _n: u32, _v: u32) {}
    pub fn finish(self) -> Vec<u8> {
        Vec::new()
    }
}
