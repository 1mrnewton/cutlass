//! The similarity the camera path is built from.

use super::*;

const MIDDLE: (f64, f64) = (120.0, 90.0);

#[test]
fn a_similarity_composes_and_inverts() {
    let a = Sim::about(MIDDLE, 3.0, -2.0, 0.1, 1.05);
    let b = Sim::about(MIDDLE, -1.0, 4.0, -0.05, 0.98);
    let p = (17.0, 140.0);
    let both = a.then(&b).apply(p);
    let step = b.apply(a.apply(p));
    assert!((both.0 - step.0).abs() < 1e-9 && (both.1 - step.1).abs() < 1e-9);
    let back = a.inverse().unwrap().apply(a.apply(p));
    assert!((back.0 - p.0).abs() < 1e-9 && (back.1 - p.1).abs() < 1e-9);
    let flat = Sim {
        a: 0.0,
        b: 0.0,
        tx: 1.0,
        ty: 1.0,
    };
    assert!(flat.inverse().is_none());
    assert_eq!(Sim::IDENTITY.apply(p), p);
}
