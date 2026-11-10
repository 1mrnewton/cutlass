use std::sync::Arc;

use super::super::frame::Picture;
use super::{Mirror, PicturePixel, band_of, band_overlap};

/// A 4:2:0 picture whose samples tell their plane and row.
fn picture() -> Picture<u8> {
    let mut pic = Picture::<u8>::new(64, 256, 3, true, true, 8);
    for (p, pl) in pic.planes.iter_mut().enumerate() {
        let stride = pl.stride;
        for (i, v) in pl.data.iter_mut().enumerate() {
            *v = (p * 100 + i / stride) as u8;
        }
    }
    pic
}

/// The bands of luma rows 0..64, 64..128 and 128..192, each with the
/// overlap of 64x64 superblocks above its rows.
fn mirror(pic: &Picture<u8>) -> Mirror<u8> {
    let mut m = Mirror::new(u8::view_rows);
    for k in 0..3 {
        let b = band_of(
            pic,
            64 * k,
            64 * (k + 1),
            band_overlap(false),
            Default::default(),
        );
        m.bands.push(Arc::new(b));
    }
    m
}

#[test]
fn a_band_holds_the_rows_above_the_ones_it_adds() {
    let pic = picture();
    let b = band_of(&pic, 128, 192, band_overlap(false), Default::default());
    assert_eq!((b.y0, b.y1), (128, 192), "the rows it adds");
    // 80 luma rows above, 40 chroma rows.
    assert_eq!(b.row0, [48, 24, 24]);
    assert_eq!(b.nrows, [144, 72, 72]);
    // The first band has nothing above it.
    let first = band_of(&pic, 0, 64, band_overlap(false), Default::default());
    assert_eq!((first.row0, first.nrows), ([0; 3], [64, 32, 32]));
    // 128x128 superblocks take blocks twice as high.
    assert!(band_overlap(true) >= 128 + 7);
    assert!(band_overlap(false) >= 64 + 7);
}

#[test]
fn the_rows_of_a_block_across_two_bands_lie_in_one() {
    let pic = picture();
    let m = mirror(&pic);
    // A 64-row block and the 7 rows its filters read, from row 60: over
    // the edge at 64 and the one at 128.
    for (p, y0, y1) in [
        (0, 60, 131),
        (0, 0, 64),
        (0, 121, 192),
        (1, 30, 69),
        (2, 60, 96),
    ] {
        let (rows, stride) = m.rows(p, y0, y1).expect("one band holds them");
        assert_eq!(stride, pic.planes[p].w);
        for y in y0..y1 {
            let want = (p * 100 + y) as u8;
            assert_eq!(rows[(y - y0) * stride], want, "plane {p} row {y}");
            assert_eq!(rows[(y - y0) * stride + stride - 1], want);
        }
    }
}

#[test]
fn rows_not_published_or_out_of_reach_are_not_found() {
    let pic = picture();
    let m = mirror(&pic);
    assert!(m.rows(0, 180, 193).is_none(), "row 192 is not published");
    assert!(m.rows(0, 300, 310).is_none());
    // More rows than a band overlaps: no band holds them all.
    assert!(m.rows(0, 20, 192).is_none());
    assert!(m.rows(0, 70, 70).is_none(), "no rows asked for");
    assert!(Mirror::<u8>::new(u8::view_rows).rows(0, 0, 8).is_none());
}
