use super::{Edge, Thresh, Wd, filter, filter_scalar};

#[test]
fn matches_the_scalar_twin() {
    let mut seed = 5u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        seed >> 8
    };
    let stride = 48;
    let mut ran = 0;
    for case in 0..6000 {
        // Smooth pictures with steps, so that every filter path runs.
        let base = (rnd() % 200) as i32;
        let step = (rnd() % 40) as i32 - 20;
        let noise = rnd() % 4;
        let buf: Vec<u8> = (0..stride * 40)
            .map(|i| {
                let (x, y) = ((i % stride) as i32, (i / stride) as i32);
                let s = if x >= 24 || y >= 20 { step } else { 0 };
                (base + s + (rnd() % (noise + 1)) as i32).clamp(0, 255) as u8
            })
            .collect();
        let vertical = case % 2 == 0;
        let pick = |r: u32| [None, Some(Wd::W4), Some(Wd::W8), Some(Wd::W16)][(r % 4) as usize];
        let wd = [pick(rnd()), pick(rnd())];
        let t = |r: u32| {
            let level = (r % 63 + 1) as u8;
            Thresh {
                blimit: 2 * (level + 2) + 4,
                limit: (level >> 1).max(1),
                hev: level >> 4,
            }
        };
        let thr = [t(rnd()), t(rnd())];
        let at = if vertical {
            4 * stride + 24
        } else {
            20 * stride + 8
        };
        let mut want = buf.clone();
        filter_scalar(Edge {
            buf: &mut want,
            at,
            stride,
            vertical,
            wd,
            thr,
        });
        let mut got = buf.clone();
        if filter(Edge {
            buf: &mut got,
            at,
            stride,
            vertical,
            wd,
            thr,
        }) {
            ran += 1;
            assert_eq!(got, want, "case {case} vertical {vertical} {wd:?} {thr:?}");
        }
    }
    if super::super::avx2() {
        assert!(ran > 4000, "{ran}");
    }
}

#[test]
fn two_edges_at_once_match_two_scalar_ones() {
    use super::{Edge32, filter32};
    let mut seed = 11u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        seed >> 8
    };
    let stride = 64;
    let mut ran = 0;
    for case in 0..6000 {
        let base = (rnd() % 200) as i32;
        let step = (rnd() % 40) as i32 - 20;
        let noise = rnd() % 4;
        let buf: Vec<u8> = (0..stride * 40)
            .map(|i| {
                let (x, y) = ((i % stride) as i32, (i / stride) as i32);
                let s = if y >= 20 { step } else { 0 } + if x >= 40 { step / 2 } else { 0 };
                (base + s + (rnd() % (noise + 1)) as i32).clamp(0, 255) as u8
            })
            .collect();
        let pick = |r: u32| [None, Some(Wd::W4), Some(Wd::W8), Some(Wd::W16)][(r % 4) as usize];
        let wd = [pick(rnd()), pick(rnd()), pick(rnd()), pick(rnd())];
        let t = |r: u32| {
            let level = (r % 63 + 1) as u8;
            Thresh {
                blimit: 2 * (level + 2) + 4,
                limit: (level >> 1).max(1),
                hev: level >> 4,
            }
        };
        let thr = [t(rnd()), t(rnd()), t(rnd()), t(rnd())];
        let at = 20 * stride + 8;
        let mut want = buf.clone();
        for h in 0..2 {
            filter_scalar(Edge {
                buf: &mut want,
                at: at + 16 * h,
                stride,
                vertical: false,
                wd: [wd[2 * h], wd[2 * h + 1]],
                thr: [thr[2 * h], thr[2 * h + 1]],
            });
        }
        let mut got = buf.clone();
        if filter32(Edge32 {
            buf: &mut got,
            at,
            stride,
            wd,
            thr,
        }) {
            ran += 1;
            assert_eq!(got, want, "case {case} {wd:?} {thr:?}");
        }
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        assert!(ran > 5000);
    }
}
