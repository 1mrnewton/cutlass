//! Placeholder until the forward transform lands.
#![allow(dead_code)]
pub fn fwd_txfm(_src: &[i32], _ss: usize, coefs: &mut [i32], _tx: usize, _t: usize, _bd: u32) {
    for c in coefs.iter_mut() {
        *c = 0;
    }
}
pub fn fwd_wht4x4(_src: &[i32], _ss: usize, coefs: &mut [i32]) {
    for c in coefs.iter_mut() {
        *c = 0;
    }
}
