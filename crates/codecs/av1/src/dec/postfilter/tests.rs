use super::super::frame::FrameState;
use super::super::obu::{FrameHeader, SequenceHeader};
use super::super::types::{
    BLOCK_8X8, BLOCK_16X16, BLOCK_64X64, GLOBALMV, INTRA_FRAME, MiFields, MiInfo, NEWMV,
};
use super::{FilterInfo, LF_BLOCK_EDGES, LF_SKIP, LfInfo, LfLevels};

/// A 64x48 frame with a level for each loop filter index.
fn header() -> FrameHeader {
    let mut hdr = FrameHeader {
        mi_cols: 16,
        mi_rows: 12,
        frame_width: 64,
        frame_height: 48,
        ..FrameHeader::default()
    };
    hdr.loop_filter.level = [20, 24, 12, 8];
    hdr
}

#[test]
fn a_block_gets_the_level_of_each_loop_filter_index() {
    let hdr = header();
    let lv = LfLevels::new(&hdr);
    let i = lv.info(&hdr, 1, 0, NEWMV, false, BLOCK_16X16, [0; 4]);
    assert_eq!(i.lvl, [20, 24, 12, 8]);
    assert_eq!((i.flags, i.bsize), (0, BLOCK_16X16));
}

#[test]
fn the_block_delta_moves_the_levels() {
    let mut hdr = header();
    let lv = LfLevels::new(&hdr);
    // One delta for every index.
    let i = lv.info(&hdr, 1, 0, NEWMV, false, BLOCK_8X8, [3, -9, 1, 1]);
    assert_eq!(i.lvl, [23, 27, 15, 11]);
    // One delta for each index, and a level stays in 0..=63.
    hdr.delta_lf_multi = true;
    let lv = LfLevels::new(&hdr);
    let i = lv.info(&hdr, 1, 0, NEWMV, false, BLOCK_8X8, [3, -40, 60, 0]);
    assert_eq!(i.lvl, [23, 0, 63, 8]);
}

#[test]
fn the_reference_and_mode_deltas_move_the_levels() {
    let mut hdr = header();
    hdr.loop_filter.delta_enabled = true;
    hdr.loop_filter.ref_deltas = [1, 0, 0, 0, -1, 0, -1, -1];
    hdr.loop_filter.mode_deltas = [0, 2];
    let lv = LfLevels::new(&hdr);
    // An intra block takes the intra delta alone.
    assert_eq!(
        lv.info(&hdr, INTRA_FRAME, 0, 0, false, BLOCK_8X8, [0; 4])
            .lvl[0],
        21
    );
    // A NEWMV block on reference 4 takes its delta and the mode delta.
    assert_eq!(
        lv.info(&hdr, 4, 0, NEWMV, false, BLOCK_8X8, [0; 4]).lvl[0],
        21
    );
    // A GLOBALMV block takes the first mode delta.
    assert_eq!(
        lv.info(&hdr, 4, 0, GLOBALMV, false, BLOCK_8X8, [0; 4]).lvl[0],
        19
    );
    // With a block delta the same sums are worked out, not looked up.
    assert_eq!(
        lv.info(&hdr, 4, 0, NEWMV, false, BLOCK_8X8, [2, 0, 0, 0])
            .lvl[0],
        23
    );
}

#[test]
fn a_skipped_block_says_so_and_an_inter_one_keeps_only_its_own_edges() {
    let hdr = header();
    let lv = LfLevels::new(&hdr);
    let flags = |rf: i8, skip: bool| lv.info(&hdr, rf, 0, NEWMV, skip, BLOCK_8X8, [0; 4]).flags;
    assert_eq!(flags(1, false), 0);
    assert_eq!(flags(INTRA_FRAME, true), LF_SKIP);
    assert_eq!(flags(2, true), LF_SKIP | LF_BLOCK_EDGES);
}

#[test]
fn a_stored_block_covers_its_units_inside_the_frame() {
    let hdr = header();
    let fs = FrameState::new(&SequenceHeader::default(), &hdr);
    let mut info = FilterInfo::default();
    info.size_for(&fs);
    let block = LfInfo {
        lvl: [9, 9, 9, 9],
        flags: LF_SKIP,
        bsize: BLOCK_64X64,
        pad: [0; 2],
    };
    // A 64x64 block at row 8: 4 of its 16 rows are in the 12-row frame.
    info.store((8, 0), (fs.mi_rows, fs.mi_cols), block);
    let st = info.stride;
    for r in 0..fs.mi_rows + 4 {
        for c in 0..fs.mi_cols {
            let inside = (8..12).contains(&r);
            assert_eq!(
                info.units[r * st + c] == block.bytes(),
                inside,
                "row {r} column {c}"
            );
        }
    }
    assert!(info.skip(8, 15) && !info.skip(7, 15));
    // A block past the frame stores nothing.
    info.store((40, 40), (fs.mi_rows, fs.mi_cols), block);
}

#[test]
fn rows_filled_from_the_mode_info_match_blocks_stored_one_by_one() {
    let hdr = FrameHeader {
        delta_lf_present: true,
        ..header()
    };
    let mut fs = FrameState::new(&SequenceHeader::default(), &hdr);
    fs.delta_lf.resize(fs.mi.len(), [0; 4]);
    let lv = LfLevels::new(&hdr);
    // 16x16 blocks, then 8x8 blocks in the last 4 rows: each with its
    // own reference, segment and delta.
    let mut want = FilterInfo::default();
    want.size_for(&fs);
    let st = fs.mi_stride;
    let mut n = 0u8;
    let mut put = |fs: &mut FrameState, want: &mut FilterInfo, r: usize, c: usize, bsize: u8| {
        n += 1;
        let m: MiInfo = MiFields {
            ref_frame: [(n % 5) as i8, -1],
            bsize,
            y_mode: GLOBALMV - 2 + n % 4,
            skip: n.is_multiple_of(3),
            segment_id: n % 8,
            ..MiFields::default()
        }
        .into();
        let delta_lf = [(n % 7) as i8 - 3, 0, 0, 0];
        let size = if bsize == BLOCK_16X16 { 4 } else { 2 };
        for y in 0..size {
            fs.mi[(r + y) * st + c..][..size].fill(m);
            fs.delta_lf[(r + y) * st + c..][..size].fill(delta_lf);
        }
        let of = lv.info(
            &hdr,
            m.ref_frame[0],
            m.segment_id(),
            m.y_mode,
            m.skip(),
            bsize,
            delta_lf,
        );
        want.store((r, c), (12, 16), of);
    };
    for r in (0..8).step_by(4) {
        for c in (0..16).step_by(4) {
            put(&mut fs, &mut want, r, c, BLOCK_16X16);
        }
    }
    for r in (8..12).step_by(2) {
        for c in (0..16).step_by(2) {
            put(&mut fs, &mut want, r, c, BLOCK_8X8);
        }
    }
    let mut got = FilterInfo::default();
    got.size_for(&fs);
    got.fill_rows(&hdr, &lv, &fs, 0..usize::MAX);
    for r in 0..12 {
        assert_eq!(
            got.units[r * st..r * st + 16],
            want.units[r * st..r * st + 16],
            "row {r}"
        );
    }
    // Only the rows asked for are filled.
    let mut part = FilterInfo::default();
    part.size_for(&fs);
    part.fill_rows(&hdr, &lv, &fs, 4..8);
    assert_eq!(
        part.units[4 * st..4 * st + 16],
        want.units[4 * st..4 * st + 16]
    );
    assert!(part.units[..4 * st].iter().all(|u| *u == [0; 8]));
    assert!(part.units[8 * st..].iter().all(|u| *u == [0; 8]));
}
