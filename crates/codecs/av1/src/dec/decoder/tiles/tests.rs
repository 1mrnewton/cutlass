use super::super::{AnyPicture, Av1Core, DecodedFrame};
use super::{TileArea, bands};
use cutlass_decode::CodecError;

/// The frame payloads of a fixture IVF file.
fn ivf(name: &str) -> Vec<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    let data = std::fs::read(path).expect("fixture");
    let mut at = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut out = Vec::new();
    while at + 12 <= data.len() {
        let size = u32::from_le_bytes(data[at..at + 4].try_into().expect("size")) as usize;
        at += 12;
        out.push(data[at..at + size].to_vec());
        at += size;
    }
    out
}

/// The visible samples of each shown frame.
fn samples(frames: &[(u64, DecodedFrame)]) -> Vec<Vec<u16>> {
    frames
        .iter()
        .map(|(_, f)| {
            let mut out = Vec::new();
            match &*f.picture {
                AnyPicture::U8(p) => {
                    for pl in &p.planes {
                        for y in 0..pl.h {
                            out.extend(
                                pl.data[y * pl.stride..][..pl.w]
                                    .iter()
                                    .map(|&v| u16::from(v)),
                            );
                        }
                    }
                }
                AnyPicture::U16(p) => {
                    for pl in &p.planes {
                        for y in 0..pl.h {
                            out.extend_from_slice(&pl.data[y * pl.stride..][..pl.w]);
                        }
                    }
                }
            }
            out
        })
        .collect()
}

fn decode(core: &mut Av1Core, packets: &[Vec<u8>]) -> Result<Vec<(u64, DecodedFrame)>, CodecError> {
    let mut out = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        out.extend(core.decode_delayed(p, i as u64)?);
    }
    out.extend(core.drain()?);
    Ok(out)
}

fn core(threads: usize) -> Av1Core {
    let mut core = Av1Core::new();
    core.threads = threads;
    core
}

/// A decoder that takes one frame at a time, so the tiles of a frame
/// decode into shells.
fn one_frame_core(threads: usize) -> Av1Core {
    let mut core = core(threads);
    core.frames_at_once = false;
    core
}

#[test]
fn helpers_decode_tiles_and_give_their_shells_back() {
    // 4352x64 in two tile columns: a key frame, then an inter frame.
    let packets = ivf("level.ivf");
    let one = samples(&decode(&mut core(1), &packets).expect("one thread"));
    assert_eq!(one.len(), 2);
    let mut four = one_frame_core(4);
    let got = samples(&decode(&mut four, &packets).expect("four threads"));
    assert_eq!(got, one);
    let sp = super::super::lock(&four.px8.spares);
    assert_eq!(sp.tile_shells_out, 0, "every shell lent comes back");
    // Two tiles: one helper beside the frame's thread.
    assert_eq!(sp.tile_shells.len(), 1);
}

#[test]
fn a_frame_decodes_its_tiles_alone_when_every_shell_is_lent() {
    let packets = ivf("level.ivf");
    let one = samples(&decode(&mut core(1), &packets).expect("one thread"));
    let mut four = one_frame_core(4);
    super::super::lock(&four.px8.spares).tile_shells_out = 3;
    let got = samples(&decode(&mut four, &packets).expect("four threads"));
    assert_eq!(got, one);
    let sp = super::super::lock(&four.px8.spares);
    assert!(sp.tile_shells.is_empty(), "no shell past the budget");
    assert_eq!(sp.tile_shells_out, 3);
}

#[test]
fn many_tiles_on_few_threads_match_one_thread() {
    // 8x4 tiles (128x128 superblocks) over 3 threads: each thread decodes
    // several tiles, its shell keeping the tiles before.
    let packets = ivf("sb128-tiles.ivf");
    let one = samples(&decode(&mut core(1), &packets).expect("one thread"));
    for threads in [2, 3, 5] {
        let got = samples(&decode(&mut one_frame_core(threads), &packets).expect("threads"));
        assert_eq!(got, one, "{threads} threads");
    }
}

#[test]
fn a_frame_taken_alone_shows_what_frames_at_once_show() {
    for name in ["level.ivf", "sb128-tiles.ivf"] {
        let packets = ivf(name);
        let at_once = samples(&decode(&mut core(4), &packets).expect("at once"));
        let alone = samples(&decode(&mut one_frame_core(4), &packets).expect("alone"));
        assert_eq!(alone, at_once, "{name}");
    }
}

/// The packets of the fixture files `name_0.obu`, `name_1.obu`, and so on.
fn obus(name: &str, count: usize) -> Vec<Vec<u8>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    (0..count)
        .map(|i| std::fs::read(dir.join(format!("{name}_{i}.obu"))).expect("fixture"))
        .collect()
}

#[test]
fn a_block_that_stops_for_reference_rows_is_predicted_again_once_they_are_final() {
    // A block stops at the first rows of a reference still being decoded
    // that are not published, and is taken up again: ordinary and warped
    // blocks alike. With the wait ahead of each chunk of blocks, the
    // samples are the same.
    let streams = [
        ivf("sb128-tiles.ivf"),
        ivf("level.ivf"),
        obus("warp", 8),
        obus("rotzoom", 3),
    ];
    let mut waits = 0;
    // Whether a reference is still being decoded when a frame reads it is
    // a matter of timing: some round has one that is.
    for _ in 0..50 {
        for packets in &streams {
            let one = samples(&decode(&mut core(1), packets).expect("one thread"));
            let mut four = core(4);
            let got = samples(&decode(&mut four, packets).expect("four threads"));
            assert_eq!(got, one);
            waits += super::super::lock(&four.px8.spares).row_waits;
            let mut ahead = core(4);
            super::super::lock(&ahead.px8.spares).rows_ahead = true;
            let got = samples(&decode(&mut ahead, packets).expect("waits ahead"));
            assert_eq!(got, one);
        }
        if waits > 0 {
            break;
        }
    }
    assert!(waits > 0, "no block waited for rows");
}

#[test]
fn a_reset_keeps_the_choice_of_one_frame_at_a_time() {
    let mut core = one_frame_core(3);
    core.reset();
    assert!(!core.frames_at_once);
    assert!(
        Av1Core::new().frames_at_once,
        "frames decode at once by default"
    );
}

#[test]
fn a_bad_tile_fails_the_frame_as_it_does_on_one_thread() {
    let mut packets = ivf("level.ivf");
    // Cut the inter frame inside its second tile.
    let n = packets[1].len();
    packets[1].truncate(n - n / 4);
    let one = decode(&mut core(1), &packets).map(|f| f.len());
    assert!(one.is_err(), "the cut frame fails");
    for mut four in [core(4), one_frame_core(4)] {
        let four = decode(&mut four, &packets).map(|f| f.len());
        assert_eq!(format!("{four:?}"), format!("{one:?}"));
    }
}

#[test]
fn a_tile_area_runs_to_the_array_ends_on_the_frame_edges() {
    let a = TileArea {
        rows: 16..32,
        cols: 32..48,
        last_row: false,
        last_col: false,
    };
    assert_eq!(a.span((0, 0), false, 100, 50), (16..32, 32..48));
    assert_eq!(a.span((1, 1), false, 100, 50), (8..16, 16..24));
    assert_eq!(a.span((1, 1), true, 400, 200), (32..64, 64..96));
    let edge = TileArea {
        last_row: true,
        last_col: true,
        ..a
    };
    assert_eq!(edge.span((0, 0), false, 100, 50), (16..50, 32..100));
}

#[test]
fn bands_cut_an_array_into_its_rows() {
    let mut a: Vec<u32> = (0..12).collect();
    let b = bands(&mut a, 3, &[0, 1, 3, 4]);
    let rows: Vec<_> = b.iter().map(|b| b.rows.clone()).collect();
    assert_eq!(rows, [0..1, 1..3, 3..4]);
    let data: Vec<&[u32]> = b.iter().map(|b| &*b.data).collect();
    assert_eq!(data, [&[0, 1, 2][..], &[3, 4, 5, 6, 7, 8], &[9, 10, 11]]);
}

#[test]
fn a_band_copies_only_its_own_rows() {
    let src: Vec<u8> = (0..16).collect();
    let mut dst = vec![0u8; 16];
    let mut b = bands(&mut dst, 4, &[0, 2, 4]);
    b[1].copy(&src, &(1..4), &(1..3));
    drop(b);
    assert_eq!(
        dst,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 0, 0, 13, 14, 0],
        "rows 2 and 3 of band 2..4, columns 1..3"
    );
}
