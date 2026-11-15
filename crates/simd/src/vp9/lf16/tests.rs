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
    for case in 0..12000 {
        let bd = if case % 4 < 2 { 10 } else { 12 };
        let max = (1i32 << bd) - 1;
        let sh = bd - 8;
        // Smooth pictures with steps, so that every filter path runs, at
        // the bottom and the top of the range too.
        let base = match case % 7 {
            0 => 0,
            1 => max - (rnd() % 8) as i32,
            _ => (rnd() % (max as u32 + 1)) as i32,
        };
        let step = ((rnd() % 40) as i32 - 20) << sh;
        let noise = (rnd() % 4) << (rnd() % (sh + 1));
        let buf: Vec<u16> = (0..stride * 40)
            .map(|i| {
                let (x, y) = ((i % stride) as i32, (i / stride) as i32);
                let s = if x >= 24 || y >= 20 { step } else { 0 };
                if case % 11 == 3 {
                    // Alternating extremes.
                    [0, max][(x + y) as usize % 2] as u16
                } else {
                    (base + s + (rnd() % (noise + 1)) as i32).clamp(0, max) as u16
                }
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
        let thr = if case % 13 == 0 {
            [Thresh {
                blimit: 255,
                limit: 63,
                hev: 15,
            }; 2]
        } else {
            [t(rnd()), t(rnd())]
        };
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
            bd,
        });
        let mut got = buf.clone();
        if filter(Edge {
            buf: &mut got,
            at,
            stride,
            vertical,
            wd,
            thr,
            bd,
        }) {
            ran += 1;
            assert_eq!(
                got, want,
                "case {case} bd {bd} vertical {vertical} {wd:?} {thr:?}"
            );
        }
    }
    if super::super::avx2() {
        assert!(ran > 8000, "{ran}");
    }
}
