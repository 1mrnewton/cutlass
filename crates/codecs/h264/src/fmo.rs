//! Slice groups, clause 8.2.2.
//!
//! One group is raster order and stores nothing. Several groups build
//! `MbToSliceGroupMap` at the start of the slice, then each macroblock's
//! successor is the next address in that same group.

use cutlass_decode::CodecError;

use crate::set::SliceGroups;

#[derive(Clone, Copy)]
pub(crate) struct Grid {
    pub mb_w: usize,
    pub frame_mb_h: usize,
    pub frame_only: bool,
    pub mbaff: bool,
    pub field: bool,
}

pub(crate) fn map_units(mb_w: usize, frame_mb_h: usize, frame_only: bool) -> u32 {
    let map_h = if frame_only {
        frame_mb_h
    } else {
        frame_mb_h / 2
    };
    (mb_w as u32).saturating_mul(map_h as u32)
}

pub(crate) fn ceil_log2(value: u32) -> u32 {
    if value <= 1 {
        0
    } else {
        u32::BITS - (value - 1).leading_zeros()
    }
}

pub(crate) fn cycle_bits(map_units: u32, rate: u32) -> Result<u32, CodecError> {
    if rate == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice group change",
        });
    }
    Ok(ceil_log2(map_units / rate + 1))
}

/// Successor of each macroblock address. Empty when the picture has one group.
pub(crate) fn links(
    grid: Grid,
    groups: &SliceGroups,
    cycle: u32,
) -> Result<Vec<usize>, CodecError> {
    if groups.minus1 == 0 {
        return Ok(Vec::new());
    }
    Ok(successor(&mb_map(grid, groups, cycle)?))
}

fn successor(map: &[u8]) -> Vec<usize> {
    let end = map.len();
    let mut next = vec![end; end];
    let mut last = [end; 8];
    for (i, &group) in map.iter().enumerate().rev() {
        let group = usize::from(group);
        if let Some(tail) = last.get_mut(group) {
            next[i] = *tail;
            *tail = i;
        }
    }
    next
}

fn mb_map(grid: Grid, groups: &SliceGroups, cycle: u32) -> Result<Vec<u8>, CodecError> {
    let map_h = if grid.frame_only {
        grid.frame_mb_h
    } else {
        grid.frame_mb_h / 2
    };
    if grid.mb_w == 0 || map_h == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice group map",
        });
    }
    let units = grid.mb_w * map_h;
    let pic_mbs = if grid.field {
        units
    } else {
        grid.mb_w * grid.frame_mb_h
    };
    let mut map = vec![0u8; units];
    match groups.map_type {
        0 => interleaved(&mut map, &groups.run, groups.minus1)?,
        1 => dispersed(&mut map, grid.mb_w, groups.minus1),
        2 => foreground(
            &mut map,
            grid.mb_w,
            groups.minus1,
            &groups.top_left,
            &groups.bottom_right,
        )?,
        3..=5 => {
            if groups.minus1 != 1 {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "slice group map",
                });
            }
            let grown = group0(cycle, groups.rate, units);
            match groups.map_type {
                3 => box_out(&mut map, grid.mb_w, map_h, groups.direction, grown)?,
                4 => raster(&mut map, groups.direction, grown),
                _ => wipe(&mut map, grid.mb_w, map_h, groups.direction, grown),
            }
        }
        6 => {
            if groups.explicit.len() != units {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "slice group id",
                });
            }
            map.copy_from_slice(&groups.explicit);
        }
        _ => {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "slice group map",
            });
        }
    }
    to_macroblocks(map, grid, pic_mbs)
}

fn group0(cycle: u32, rate: u32, units: usize) -> usize {
    u64::from(cycle)
        .saturating_mul(u64::from(rate))
        .min(units as u64) as usize
}

fn interleaved(map: &mut [u8], runs: &[u32], minus1: u8) -> Result<(), CodecError> {
    let groups = usize::from(minus1);
    if runs.len() != groups + 1 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice group map",
        });
    }
    let size = map.len();
    let mut i = 0usize;
    let mut guard = 0usize;
    while i < size {
        guard += 1;
        if guard > size {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "slice group map",
            });
        }
        let mut group = 0usize;
        while group <= groups && i < size {
            let run = runs[group] as usize;
            let mut j = 0usize;
            while j <= run && i + j < size {
                map[i + j] = group as u8;
                j += 1;
            }
            i = i.saturating_add(run).saturating_add(1);
            group += 1;
        }
    }
    Ok(())
}

fn dispersed(map: &mut [u8], width: usize, minus1: u8) {
    let count = usize::from(minus1) + 1;
    for (i, cell) in map.iter_mut().enumerate() {
        let x = i % width;
        let y = i / width;
        *cell = ((x + (y * count) / 2) % count) as u8;
    }
}

fn foreground(
    map: &mut [u8],
    width: usize,
    minus1: u8,
    top_left: &[u32],
    bottom_right: &[u32],
) -> Result<(), CodecError> {
    map.fill(minus1);
    if top_left.len() != usize::from(minus1) || bottom_right.len() != top_left.len() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice group map",
        });
    }
    for group in (0..usize::from(minus1)).rev() {
        let top = top_left[group] as usize;
        let bottom = bottom_right[group] as usize;
        if top >= map.len() || bottom >= map.len() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "slice group map",
            });
        }
        let x0 = top % width;
        let y0 = top / width;
        let x1 = bottom % width;
        let y1 = bottom / width;
        if x1 < x0 || y1 < y0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "slice group map",
            });
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                map[y * width + x] = group as u8;
            }
        }
    }
    Ok(())
}

fn box_out(
    map: &mut [u8],
    width: usize,
    height: usize,
    direction: bool,
    count: usize,
) -> Result<(), CodecError> {
    map.fill(1);
    let width = width as i32;
    let height = height as i32;
    let flag = i32::from(direction);
    let mut x = (width - flag) / 2;
    let mut y = (height - flag) / 2;
    let mut left = x;
    let mut top = y;
    let mut right = x;
    let mut bottom = y;
    let mut x_dir = flag - 1;
    let mut y_dir = flag;
    let mut filled = 0usize;
    let mut steps = 0usize;
    while filled < count {
        steps += 1;
        if steps > map.len() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "slice group map",
            });
        }
        if x < 0 || y < 0 || x >= width || y >= height {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "slice group map",
            });
        }
        let index = (y as usize) * (width as usize) + (x as usize);
        let vacant = map[index] == 1;
        if vacant {
            map[index] = 0;
        }
        if x_dir == -1 && x == left {
            left = (left - 1).max(0);
            x = left;
            x_dir = 0;
            y_dir = 2 * flag - 1;
        } else if x_dir == 1 && x == right {
            right = (right + 1).min(width - 1);
            x = right;
            x_dir = 0;
            y_dir = 1 - 2 * flag;
        } else if y_dir == -1 && y == top {
            top = (top - 1).max(0);
            y = top;
            x_dir = 1 - 2 * flag;
            y_dir = 0;
        } else if y_dir == 1 && y == bottom {
            bottom = (bottom + 1).min(height - 1);
            y = bottom;
            x_dir = 2 * flag - 1;
            y_dir = 0;
        } else {
            x += x_dir;
            y += y_dir;
        }
        filled += usize::from(vacant);
    }
    Ok(())
}

fn raster(map: &mut [u8], direction: bool, grown: usize) {
    let upper = if direction { map.len() - grown } else { grown };
    let flag = u8::from(direction);
    for (i, cell) in map.iter_mut().enumerate() {
        *cell = if i < upper { flag } else { 1 - flag };
    }
}

fn wipe(map: &mut [u8], width: usize, height: usize, direction: bool, grown: usize) {
    let flag = u8::from(direction);
    let mut k = 0usize;
    for x in 0..width {
        for y in 0..height {
            map[y * width + x] = if k < grown { flag } else { 1 - flag };
            k += 1;
        }
    }
}

fn to_macroblocks(units: Vec<u8>, grid: Grid, pic_mbs: usize) -> Result<Vec<u8>, CodecError> {
    if grid.frame_only || grid.field {
        if units.len() != pic_mbs {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "slice group map",
            });
        }
        return Ok(units);
    }
    if pic_mbs != units.len().saturating_mul(2) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice group map",
        });
    }
    let mut out = vec![0u8; pic_mbs];
    if grid.mbaff {
        for (i, cell) in out.iter_mut().enumerate() {
            *cell = units[i / 2];
        }
    } else {
        for (i, cell) in out.iter_mut().enumerate() {
            let unit = (i / (2 * grid.mb_w)) * grid.mb_w + (i % grid.mb_w);
            *cell = units[unit];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::process::{Command, Stdio};

    use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder};

    use super::{Grid, ceil_log2, cycle_bits, mb_map};
    use crate::H264Decoder;
    use crate::nal::annex_b_nal;
    use crate::set::SliceGroups;

    const FRAME: Grid = Grid {
        mb_w: 2,
        frame_mb_h: 2,
        frame_only: true,
        mbaff: false,
        field: false,
    };

    fn groups(map_type: u8) -> SliceGroups {
        let mut groups = SliceGroups::none();
        groups.minus1 = 1;
        groups.map_type = map_type;
        groups.rate = 1;
        groups
    }

    #[test]
    fn map_formulas_follow_the_recommendation() {
        let mut runs = groups(0);
        runs.run = vec![0, 0];
        assert_eq!(mb_map(FRAME, &runs, 0).unwrap(), vec![0, 1, 0, 1]);
        runs.run = vec![10, 0];
        assert_eq!(mb_map(FRAME, &runs, 0).unwrap(), vec![0, 0, 0, 0]);

        assert_eq!(mb_map(FRAME, &groups(1), 0).unwrap(), vec![0, 1, 1, 0]);

        let mut rect = groups(2);
        rect.top_left = vec![3];
        rect.bottom_right = vec![3];
        assert_eq!(mb_map(FRAME, &rect, 0).unwrap(), vec![1, 1, 1, 0]);
        rect.minus1 = 2;
        rect.top_left = vec![0, 0];
        rect.bottom_right = vec![0, 3];
        assert_eq!(mb_map(FRAME, &rect, 0).unwrap(), vec![0, 1, 1, 1]);

        let mut grow = groups(3);
        assert_eq!(mb_map(FRAME, &grow, 1).unwrap(), vec![1, 1, 1, 0]);
        grow.direction = true;
        assert_eq!(mb_map(FRAME, &grow, 1).unwrap(), vec![0, 1, 1, 1]);
        grow.direction = false;
        assert_eq!(mb_map(FRAME, &grow, 4).unwrap(), vec![0, 0, 0, 0]);

        grow.map_type = 4;
        assert_eq!(mb_map(FRAME, &grow, 2).unwrap(), vec![0, 0, 1, 1]);
        grow.direction = true;
        assert_eq!(mb_map(FRAME, &grow, 2).unwrap(), vec![1, 1, 0, 0]);

        grow.map_type = 5;
        grow.direction = false;
        assert_eq!(mb_map(FRAME, &grow, 2).unwrap(), vec![0, 1, 0, 1]);
        grow.direction = true;
        assert_eq!(mb_map(FRAME, &grow, 2).unwrap(), vec![1, 0, 1, 0]);

        let mut explicit = groups(6);
        explicit.explicit = vec![0, 1, 1, 0];
        assert_eq!(mb_map(FRAME, &explicit, 0).unwrap(), vec![0, 1, 1, 0]);

        let pair = Grid {
            frame_only: false,
            ..FRAME
        };
        explicit.explicit = vec![0, 1];
        assert_eq!(mb_map(pair, &explicit, 0).unwrap(), vec![0, 1, 0, 1]);
        let adaptive = Grid {
            mbaff: true,
            ..pair
        };
        assert_eq!(mb_map(adaptive, &explicit, 0).unwrap(), vec![0, 0, 1, 1]);
        let field = Grid {
            field: true,
            mbaff: false,
            ..pair
        };
        assert_eq!(mb_map(field, &explicit, 0).unwrap(), vec![0, 1]);

        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(cycle_bits(4, 1).unwrap(), 3);
        assert_eq!(cycle_bits(2, 1).unwrap(), 2);
        assert!(
            super::links(FRAME, &SliceGroups::none(), 0)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn interleaved_runs_visit_every_other_macroblock() {
        assert_picture(
            Kind::Progressive,
            &MapSpec::Runs([0, 0]),
            &[slice(0, 40, 2, None), slice(1, 200, 2, None)],
            64,
            16,
            &[(0, 40), (1, 200), (2, 40), (3, 200)],
        );
    }

    #[test]
    fn dispersed_groups_place_the_checkerboard() {
        assert_picture(
            Kind::Progressive,
            &MapSpec::Dispersed,
            &[slice(0, 40, 2, None), slice(1, 200, 2, None)],
            32,
            32,
            &[(0, 40), (1, 200), (2, 200), (3, 40)],
        );
    }

    #[test]
    fn foreground_rectangle_starts_at_the_last_macroblock() {
        assert_picture(
            Kind::Progressive,
            &MapSpec::Rect(3, 3),
            &[slice(3, 40, 1, None), slice(0, 200, 3, None)],
            32,
            32,
            &[(0, 200), (1, 200), (2, 200), (3, 40)],
        );
    }

    #[test]
    fn box_out_grows_from_the_centre() {
        let cycle = Some((1, 3));
        assert_picture(
            Kind::Progressive,
            &MapSpec::Grow {
                kind: 3,
                direction: false,
                rate_minus1: 0,
            },
            &[slice(3, 40, 1, cycle), slice(0, 200, 3, cycle)],
            32,
            32,
            &[(0, 200), (1, 200), (2, 200), (3, 40)],
        );
    }

    #[test]
    fn reverse_raster_grows_from_the_end() {
        let cycle = Some((2, 3));
        assert_picture(
            Kind::Progressive,
            &MapSpec::Grow {
                kind: 4,
                direction: true,
                rate_minus1: 0,
            },
            &[slice(2, 40, 2, cycle), slice(0, 200, 2, cycle)],
            32,
            32,
            &[(0, 200), (1, 200), (2, 40), (3, 40)],
        );
    }

    #[test]
    fn wipe_grows_down_the_columns() {
        let cycle = Some((2, 3));
        assert_picture(
            Kind::Progressive,
            &MapSpec::Grow {
                kind: 5,
                direction: false,
                rate_minus1: 0,
            },
            &[slice(0, 40, 2, cycle), slice(1, 200, 2, cycle)],
            32,
            32,
            &[(0, 40), (1, 200), (2, 40), (3, 200)],
        );
    }

    #[test]
    fn explicit_ids_place_the_macroblocks() {
        assert_picture(
            Kind::Progressive,
            &MapSpec::Explicit(&[0, 1, 1, 0]),
            &[slice(0, 40, 2, None), slice(1, 200, 2, None)],
            32,
            32,
            &[(0, 40), (1, 200), (2, 200), (3, 40)],
        );
    }

    #[test]
    fn an_interlaced_frame_repeats_each_map_unit_on_both_rows() {
        assert_picture(
            Kind::Interlaced,
            &MapSpec::Explicit(&[0, 1]),
            &[slice(0, 40, 2, None), slice(1, 200, 2, None)],
            32,
            32,
            &[(0, 40), (1, 200), (2, 40), (3, 200)],
        );
    }

    #[test]
    fn an_adaptive_pair_shares_one_map_unit() {
        assert_picture(
            Kind::Adaptive,
            &MapSpec::Explicit(&[0, 1]),
            &[slice(0, 40, 2, None), slice(1, 200, 2, None)],
            32,
            32,
            &[(0, 40), (1, 200), (2, 40), (3, 200)],
        );
    }

    #[test]
    fn a_repeated_macroblock_is_rejected_and_the_next_picture_decodes() {
        let bad = access(
            Kind::Progressive,
            &MapSpec::Runs([0, 0]),
            &[slice(0, 40, 1, None), slice(0, 200, 1, None)],
            32,
            16,
        );
        rejects(&bad, "first macroblock");
    }

    #[test]
    fn a_slice_group_map_past_six_is_rejected() {
        let mut body = Writer::new();
        body.ue(0);
        body.ue(0);
        body.bit(false);
        body.bit(false);
        body.ue(1);
        body.ue(7);
        rejects(&parameter_set(&body.bytes), "slice group map");
    }

    #[test]
    fn more_than_eight_slice_groups_are_rejected() {
        let mut body = Writer::new();
        body.ue(0);
        body.ue(0);
        body.bit(false);
        body.bit(false);
        body.ue(8);
        rejects(&parameter_set(&body.bytes), "slice groups");
    }

    #[test]
    fn an_explicit_id_past_the_group_count_is_rejected() {
        let mut body = Writer::new();
        body.ue(0);
        body.ue(0);
        body.bit(false);
        body.bit(false);
        body.ue(2);
        body.ue(6);
        body.ue(0);
        body.bits(3, 2);
        rejects(&parameter_set(&body.bytes), "slice group id");
    }

    fn rejects(bad: &[u8], what: &str) {
        let mut params = CodecParams::video(CodecId::H264, 32, 16);
        params.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&params).unwrap();
        let err = decoder
            .push(&Packet::video(bad.to_vec(), 0, 1, 30, false))
            .unwrap_err();
        assert!(
            matches!(err, CodecError::Malformed { what: got, .. } if got == what),
            "{err:?}"
        );
        let annex = access(
            Kind::Progressive,
            &MapSpec::Runs([0, 0]),
            &[slice(0, 40, 1, None), slice(1, 200, 1, None)],
            32,
            16,
        );
        let mut pictures = decoder.push(&Packet::video(annex, 1, 1, 30, true)).unwrap();
        pictures.extend(decoder.drain().unwrap());
        assert_eq!(pictures.len(), 1);
        assert_eq!(
            pictures[0].picture.y(),
            blocks(32, 16, &[(0, 40), (1, 200)]).as_slice()
        );
    }

    fn parameter_set(pps: &[u8]) -> Vec<u8> {
        let mut out = annex_b_nal(3, 7, &sps(Kind::Progressive, 16, 16));
        out.extend(annex_b_nal(3, 8, pps));
        out
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Progressive,
        Interlaced,
        Adaptive,
    }

    enum MapSpec {
        Runs([u32; 2]),
        Dispersed,
        Rect(u32, u32),
        Grow {
            kind: u32,
            direction: bool,
            rate_minus1: u32,
        },
        Explicit(&'static [u8]),
    }

    struct SliceBits {
        first_mb: u32,
        luma: u8,
        mbs: usize,
        cycle: Option<(u32, u32)>,
    }

    fn slice(first_mb: u32, luma: u8, mbs: usize, cycle: Option<(u32, u32)>) -> SliceBits {
        SliceBits {
            first_mb,
            luma,
            mbs,
            cycle,
        }
    }

    fn assert_picture(
        kind: Kind,
        map: &MapSpec,
        slices: &[SliceBits],
        width: usize,
        height: usize,
        samples: &[(usize, u8)],
    ) {
        let annex = access(kind, map, slices, width, height);
        let (y, u, v) = decode(&annex, width as u32, height as u32);
        assert_eq!(y, blocks(width, height, samples));
        let chroma = vec![128u8; width * height / 4];
        assert_eq!(u, chroma);
        assert_eq!(v, chroma);
        if let Some(ffmpeg) = ffmpeg_bin()
            && let Ok(raw) = ffmpeg_raw(&ffmpeg, &annex, width, height)
        {
            let y_len = width * height;
            let c_len = y_len / 4;
            assert_eq!(&raw[..y_len], y.as_slice(), "ffmpeg luma");
            assert_eq!(&raw[y_len..y_len + c_len], u.as_slice(), "ffmpeg cb");
            assert_eq!(&raw[y_len + c_len..], v.as_slice(), "ffmpeg cr");
        }
    }

    fn access(
        kind: Kind,
        map: &MapSpec,
        slices: &[SliceBits],
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        let mut out = annex_b_nal(3, 7, &sps(kind, width, height));
        out.extend(annex_b_nal(3, 8, &pps(map)));
        for spec in slices {
            out.extend(annex_b_nal(3, 5, &slice_rbsp(kind, spec)));
        }
        out
    }

    fn sps(kind: Kind, width: usize, height: usize) -> Vec<u8> {
        let mb_w = width / 16;
        let frame_h = height / 16;
        let map_h = if kind == Kind::Progressive {
            frame_h
        } else {
            frame_h / 2
        };
        let mut w = Writer::new();
        w.bits(66, 8);
        w.bits(0, 8);
        w.bits(30, 8);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(1);
        w.bit(false);
        w.ue((mb_w - 1) as u32);
        w.ue((map_h - 1) as u32);
        w.bit(kind == Kind::Progressive);
        if kind != Kind::Progressive {
            w.bit(kind == Kind::Adaptive);
        }
        w.bit(true);
        w.bit(false);
        w.bit(false);
        w.trailing();
        w.bytes
    }

    fn pps(map: &MapSpec) -> Vec<u8> {
        let mut w = Writer::new();
        w.ue(0);
        w.ue(0);
        w.bit(false);
        w.bit(false);
        w.ue(1);
        match map {
            MapSpec::Runs(runs) => {
                w.ue(0);
                for run in runs {
                    w.ue(*run);
                }
            }
            MapSpec::Dispersed => w.ue(1),
            MapSpec::Rect(top, bottom) => {
                w.ue(2);
                w.ue(*top);
                w.ue(*bottom);
            }
            MapSpec::Grow {
                kind,
                direction,
                rate_minus1,
            } => {
                w.ue(*kind);
                w.bit(*direction);
                w.ue(*rate_minus1);
            }
            MapSpec::Explicit(ids) => {
                w.ue(6);
                w.ue((ids.len() - 1) as u32);
                for id in *ids {
                    w.bit(*id == 1);
                }
            }
        }
        w.ue(0);
        w.ue(0);
        w.bit(false);
        w.bits(0, 2);
        w.se(0);
        w.se(0);
        w.se(0);
        w.bit(true);
        w.bit(false);
        w.bit(false);
        w.trailing();
        w.bytes
    }

    fn slice_rbsp(kind: Kind, spec: &SliceBits) -> Vec<u8> {
        let mut w = Writer::new();
        w.ue(spec.first_mb);
        w.ue(2);
        w.ue(0);
        w.bits(0, 4);
        if kind != Kind::Progressive {
            w.bit(false);
        }
        w.ue(0);
        w.bits(0, 4);
        w.bit(false);
        w.bit(false);
        w.se(0);
        w.ue(1);
        if let Some((cycle, width)) = spec.cycle {
            w.bits(cycle, width);
        }
        for index in 0..spec.mbs {
            if kind == Kind::Adaptive && index.is_multiple_of(2) {
                w.bit(false);
            }
            w.ue(25);
            w.align();
            w.bytes.extend(std::iter::repeat_n(spec.luma, 256));
            w.bytes.extend(std::iter::repeat_n(128, 128));
            w.bit = w.bytes.len() * 8;
        }
        w.trailing();
        w.bytes
    }

    fn decode(annex: &[u8], width: u32, height: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut params = CodecParams::video(CodecId::H264, width, height);
        params.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut pictures = decoder
            .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
            .unwrap();
        pictures.extend(decoder.drain().unwrap());
        assert_eq!(pictures.len(), 1);
        let picture = &pictures[0];
        (
            picture.picture.y().to_vec(),
            picture.picture.u().to_vec(),
            picture.picture.v().to_vec(),
        )
    }

    fn blocks(width: usize, height: usize, mbs: &[(usize, u8)]) -> Vec<u8> {
        let mb_w = width / 16;
        let mut plane = vec![0u8; width * height];
        for &(addr, sample) in mbs {
            let mx = addr % mb_w;
            let my = addr / mb_w;
            for row in 0..16 {
                for col in 0..16 {
                    plane[(my * 16 + row) * width + mx * 16 + col] = sample;
                }
            }
        }
        plane
    }

    fn ffmpeg_bin() -> Option<std::path::PathBuf> {
        for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] {
            let path = std::path::PathBuf::from(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
        None
    }

    fn ffmpeg_raw(
        ffmpeg: &std::path::Path,
        annex: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, String> {
        let mut child = Command::new(ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "none",
                "-threads",
                "1",
                "-f",
                "h264",
                "-i",
                "pipe:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| err.to_string())?;
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(annex)
            .map_err(|err| err.to_string())?;
        drop(child.stdin.take());
        let output = child.wait_with_output().map_err(|err| err.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        let frame = width * height * 3 / 2;
        if output.stdout.len() < frame {
            return Err(format!("short decode {}", output.stdout.len()));
        }
        Ok(output.stdout)
    }

    struct Writer {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl Writer {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit: 0,
            }
        }

        fn bit(&mut self, value: bool) {
            let index = self.bit / 8;
            if index == self.bytes.len() {
                self.bytes.push(0);
            }
            if value {
                self.bytes[index] |= 1 << (7 - (self.bit % 8));
            }
            self.bit += 1;
        }

        fn bits(&mut self, value: u32, n: u32) {
            for shift in (0..n).rev() {
                self.bit(((value >> shift) & 1) == 1);
            }
        }

        fn ue(&mut self, value: u32) {
            let code = value + 1;
            let zeros = 31 - code.leading_zeros();
            for _ in 0..zeros {
                self.bit(false);
            }
            self.bits(code, zeros + 1);
        }

        fn se(&mut self, value: i32) {
            let code = if value > 0 {
                value.unsigned_abs() * 2 - 1
            } else {
                value.unsigned_abs() * 2
            };
            self.ue(code);
        }

        fn align(&mut self) {
            while !self.bit.is_multiple_of(8) {
                self.bit(false);
            }
        }

        fn trailing(&mut self) {
            self.bit(true);
            while !self.bit.is_multiple_of(8) {
                self.bit(false);
            }
        }
    }
}
