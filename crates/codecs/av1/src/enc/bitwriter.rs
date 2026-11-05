//! Placeholder until the bit writer lands.
#![allow(dead_code)]
#[derive(Default)]
pub struct BitWriter;
impl BitWriter {
    pub fn new() -> Self {
        Self
    }
    pub fn f(&mut self, _n: u32, _v: u32) {}
    pub fn byte_alignment(&mut self) {}
    pub fn bytes(&self) -> Vec<u8> {
        Vec::new()
    }
}
pub fn write_obu(_out: &mut Vec<u8>, _t: u8, _tid: u8, _sid: u8, _ext: bool, _p: &[u8]) {}
