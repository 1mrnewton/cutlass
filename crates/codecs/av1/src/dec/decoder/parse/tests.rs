use std::collections::BTreeMap;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, OnceLock};

use super::super::super::frame::{MfBand, RefData};
use super::super::super::mvpred::MfEntry;
use super::super::super::tile::ReconLog;
use super::super::super::types::Mv;
use super::super::{MfClose, MfShared, Rect, RowMsg};
use super::{RowEnd, Sink};

/// A log told apart by the number of its coefficients.
fn log(tag: usize) -> ReconLog {
    ReconLog {
        coefs: vec![0; tag],
        ..ReconLog::default()
    }
}

/// The end of luma 4x4 rows `r0..r1` of a tile, with `refs` as the saved
/// reference of every 8x8 position of its `w8` columns.
fn end(r0: usize, r1: usize, w8: usize, refs: i8) -> RowEnd {
    RowEnd {
        msg: RowMsg {
            r0,
            r1,
            ..RowMsg::default()
        },
        motion: vec![MfBand {
            refs: vec![refs; 8 * w8],
            mvs: vec![Mv::ZERO; 8 * w8],
        }],
        seg: Rect::default(),
    }
}

fn sink<'a>(
    tx: &'a std::sync::mpsc::Sender<(Option<RowMsg>, ReconLog)>,
    cols: usize,
    data: Option<&'a RefData>,
) -> Sink<'a> {
    Sink {
        tx,
        cols,
        next: 0,
        held: BTreeMap::new(),
        data,
        seg: Vec::new(),
        seg_stride: 1,
    }
}

/// What was sent: each log's tag, and its message as `(r0, row_end)`.
fn sent(rx: &Receiver<(Option<RowMsg>, ReconLog)>) -> Vec<(usize, Option<(usize, bool)>)> {
    rx.try_iter()
        .map(|(m, l)| (l.coefs.len(), m.map(|m| (m.r0, m.row_end))))
        .collect()
}

#[test]
fn a_row_goes_out_once_every_column_has_ended_it() {
    let (tx, rx) = channel();
    let mut s = sink(&tx, 2, None);
    // The second column ends the row first.
    s.push(1, 0, &mut vec![log(3), log(4)], Some(end(0, 16, 2, 1)));
    assert!(
        sent(&rx).is_empty(),
        "the first column has not ended the row"
    );
    s.push(0, 0, &mut vec![log(1)], None);
    assert!(sent(&rx).is_empty());
    s.push(0, 0, &mut vec![log(2)], Some(end(0, 16, 2, 1)));
    // Column by column, each column's message with its last log, and the
    // row ending with the last column.
    assert_eq!(
        sent(&rx),
        [
            (1, None),
            (2, Some((0, false))),
            (3, None),
            (4, Some((0, true)))
        ]
    );
    assert_eq!(s.next, 16);
}

#[test]
fn rows_go_out_in_order_whatever_order_they_end_in() {
    let (tx, rx) = channel();
    let mut s = sink(&tx, 1, None);
    s.push(0, 32, &mut vec![log(3)], Some(end(32, 48, 2, 1)));
    s.push(0, 16, &mut vec![log(2)], Some(end(16, 32, 2, 1)));
    assert!(sent(&rx).is_empty(), "the first row is not there");
    s.push(0, 0, &mut vec![log(1)], Some(end(0, 16, 2, 1)));
    assert_eq!(
        sent(&rx),
        [
            (1, Some((0, true))),
            (2, Some((16, true))),
            (3, Some((32, true)))
        ]
    );
    assert!(s.held.is_empty());
}

#[test]
fn a_lone_column_sends_the_logs_of_the_next_row_at_once() {
    let (tx, rx) = channel();
    let mut s = sink(&tx, 1, None);
    s.push(0, 0, &mut vec![log(1)], None);
    assert_eq!(sent(&rx), [(1, None)], "nothing to wait for");
    // A later row waits for the one before it.
    s.push(0, 16, &mut vec![log(3)], None);
    assert!(sent(&rx).is_empty());
    s.push(0, 0, &mut vec![log(2)], Some(end(0, 16, 2, 1)));
    assert_eq!(sent(&rx), [(2, Some((0, true)))]);
    s.push(0, 16, &mut vec![log(4)], Some(end(16, 32, 2, 1)));
    assert_eq!(sent(&rx), [(3, None), (4, Some((16, true)))]);
}

#[test]
fn a_whole_row_saves_its_motion_column_beside_column() {
    // 8 columns and 16 rows of 4x4 units: one band of 8 rows of 4.
    let rd = RefData::new(8, 16, 1, [0; 8]);
    let (tx, _rx) = channel();
    let mut s = sink(&tx, 2, Some(&rd));
    s.push(0, 0, &mut vec![log(1)], Some(end(0, 16, 1, 5)));
    assert!(!rd.has_band(0), "half a row is no band");
    s.push(1, 0, &mut vec![log(2)], Some(end(0, 16, 3, 7)));
    let band = rd.band(0).expect("band");
    assert_eq!(band.refs.len(), 8 * 4);
    for row in band.refs.chunks(4) {
        assert_eq!(row, [5, 7, 7, 7]);
    }
    assert_eq!(band.mvs.len(), 8 * 4);
}

#[test]
fn a_rect_copies_its_area_out_and_back() {
    let src: Vec<u8> = (0..20).collect();
    let mut r = Rect::default();
    r.take(&src, 5, 1..3, 2..4);
    assert_eq!(r.data, [7, 8, 12, 13]);
    let mut dst = vec![0u8; 20];
    r.put(&mut dst, 5);
    let want: Vec<u8> = (0..20)
        .map(|i| if [7, 8, 12, 13].contains(&i) { i } else { 0 })
        .collect();
    assert_eq!(dst, want);
    // An area past the array is clipped to it.
    r.take(&src, 5, 3..usize::MAX, 4..9);
    assert_eq!((r.rows.clone(), r.cols.clone()), (3..4, 4..5));
    assert_eq!(r.data, [19]);
    // An empty area writes nothing, and a short target is left alone.
    r.take(&src, 5, 0..0, 0..0);
    r.put(&mut dst, 5);
    assert_eq!(dst, want);
    r.take(&src, 5, 3..4, 0..5);
    r.put(&mut dst[..17], 5);
    assert_eq!(dst, want);
}

#[test]
fn a_motion_field_that_ends_early_leaves_its_bands_unset() {
    // 20 rows of 3: bands of 8, 8 and 4 rows.
    let field = Arc::new(MfShared {
        bands: (0..3).map(|_| OnceLock::new()).collect(),
        w8: 3,
        h8: 20,
    });
    let moved = MfEntry {
        mv: Mv::new(4, 4),
        offset: 2,
    };
    let _ = field.bands[0].set(vec![moved; 24]);
    drop(MfClose(field.clone()));
    assert_eq!(
        field.bands[0].get(),
        Some(&vec![moved; 24]),
        "a set band stays"
    );
    assert_eq!(field.bands[1].get(), Some(&vec![MfEntry::default(); 24]));
    assert_eq!(field.bands[2].get(), Some(&vec![MfEntry::default(); 12]));
}

#[test]
fn rows_copied_out_of_a_picture_land_in_the_same_rows_of_another() {
    use super::super::super::frame::Picture;
    use super::super::{copy_rows_in, copy_rows_out};
    // 4:2:0, 64x48: chroma rows are half the luma rows.
    let mut src = Picture::<u8>::new(64, 48, 3, true, true, 8);
    for (p, pl) in src.planes.iter_mut().enumerate() {
        let stride = pl.stride;
        for (i, v) in pl.data.iter_mut().enumerate() {
            *v = (1 + p + i / stride) as u8;
        }
    }
    let mut dst = Picture::<u8>::new(64, 48, 3, true, true, 8);
    // Luma 4x4 rows 4..8: luma rows 16..32, chroma rows 8..16.
    let bufs = copy_rows_out(&src, 4, 8, false, Default::default());
    assert_eq!(bufs[0].len(), 16 * src.planes[0].stride);
    assert_eq!(bufs[1].len(), 8 * src.planes[1].stride);
    copy_rows_in(&mut dst, 4, &bufs);
    for (p, (d, s)) in dst.planes.iter().zip(&src.planes).enumerate() {
        let (y0, y1) = if p == 0 { (16, 32) } else { (8, 16) };
        for (y, (dr, sr)) in d
            .data
            .chunks(d.stride)
            .zip(s.data.chunks(s.stride))
            .enumerate()
        {
            if (y0..y1).contains(&y) {
                assert_eq!(dr, sr, "plane {p} row {y}");
            } else {
                assert!(dr.iter().all(|&v| v == 0), "plane {p} row {y} is untouched");
            }
        }
    }
    // The last rows run to the end of the planes, past the frame.
    let bufs = copy_rows_out(&src, 8, 12, true, bufs);
    let height = src.planes[0].data.len() / src.planes[0].stride;
    assert!(height > 48);
    assert_eq!(bufs[0].len(), (height - 32) * src.planes[0].stride);
    copy_rows_in(&mut dst, 8, &bufs);
    assert_eq!(
        dst.planes[0].data[32 * dst.planes[0].stride..],
        src.planes[0].data[32 * src.planes[0].stride..]
    );
    // Rows past the picture copy nothing, and write nothing.
    let none = copy_rows_out(&src, 4000, 4004, false, bufs);
    assert!(none.iter().all(Vec::is_empty));
}
