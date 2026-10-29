//! Intra prediction for 8-bit 4:2:0.
//!
//! Edge samples follow libvpx: unavailable above is 127, unavailable left is
//! 129, and samples past the crop repeat the last sample inside the crop.

#![allow(dead_code)]

pub struct Edge<'a> {
    pub samples: &'a [u8],
    pub stride: usize,
    pub crop_w: usize,
    pub crop_h: usize,
    pub x: usize,
    pub y: usize,
    pub have_left: bool,
    pub have_above: bool,
    pub have_right: bool,
}

#[derive(Clone, Copy)]
pub struct Site {
    pub tx: u8,
    pub stride: usize,
    pub crop_w: usize,
    pub crop_h: usize,
    pub x: usize,
    pub y: usize,
    pub have_left: bool,
    pub have_above: bool,
    pub have_right: bool,
}

pub fn predict(mode: u8, samples: &mut [u8], site: Site) {
    let Site {
        tx,
        stride,
        crop_w,
        crop_h,
        x,
        y,
        have_left,
        have_above,
        have_right,
    } = site;
    let size = 4usize << tx;
    let mut above = vec![127u8; size * 2 + 1];
    let mut left = vec![129u8; size];
    let (corner, used_left, used_above) = {
        let edge = Edge {
            samples,
            stride,
            crop_w,
            crop_h,
            x,
            y,
            have_left,
            have_above,
            have_right,
        };
        fill_edges(&edge, size, &mut above, &mut left);
        (above[0], edge.have_left, edge.have_above)
    };
    let above_row = &above[1..];
    let mut pred = vec![0u8; size * size];
    match mode {
        1 => vertical(&mut pred, size, above_row),
        2 => horizontal(&mut pred, size, &left),
        3 => d45(&mut pred, size, above_row),
        4 => d135(&mut pred, size, above_row, &left, corner),
        5 => d117(&mut pred, size, above_row, &left, corner),
        6 => d153(&mut pred, size, above_row, &left, corner),
        7 => d207(&mut pred, size, &left),
        8 => d63(&mut pred, size, above_row),
        9 => tm(&mut pred, size, above_row, &left, corner),
        _ => dc(&mut pred, size, above_row, &left, used_left, used_above),
    }
    for row in 0..size {
        let src = &pred[row * size..row * size + size];
        let at = (y + row) * stride + x;
        if let Some(slot) = samples.get_mut(at..at + size) {
            slot.copy_from_slice(src);
        }
    }
}

fn fill_edges(edge: &Edge<'_>, size: usize, above: &mut [u8], left: &mut [u8]) {
    if edge.have_left {
        for (i, slot) in left.iter_mut().enumerate().take(size) {
            *slot = sample(edge, edge.x as isize - 1, edge.y as isize + i as isize);
        }
    }
    let above_len = if edge.have_right && size == 4 {
        size * 2
    } else {
        size
    };
    if edge.have_above {
        for i in 0..above_len {
            above[i + 1] = sample(edge, edge.x as isize + i as isize, edge.y as isize - 1);
        }
        for i in above_len..size * 2 {
            above[i + 1] = above[above_len];
        }
        above[0] = if edge.have_left {
            sample(edge, edge.x as isize - 1, edge.y as isize - 1)
        } else {
            129
        };
    } else {
        for slot in &mut above[1..] {
            *slot = 127;
        }
        above[0] = 127;
    }
}

fn sample(edge: &Edge<'_>, x: isize, y: isize) -> u8 {
    let max_x = edge.crop_w.saturating_sub(1) as isize;
    let max_y = edge.crop_h.saturating_sub(1) as isize;
    let x = usize::try_from(x.clamp(0, max_x)).unwrap_or(0);
    let y = usize::try_from(y.clamp(0, max_y)).unwrap_or(0);
    edge.samples
        .get(y * edge.stride + x)
        .copied()
        .unwrap_or(128)
}

fn vertical(pred: &mut [u8], size: usize, above: &[u8]) {
    for row in 0..size {
        pred[row * size..row * size + size].copy_from_slice(&above[..size]);
    }
}

fn horizontal(pred: &mut [u8], size: usize, left: &[u8]) {
    for row in 0..size {
        for col in 0..size {
            pred[row * size + col] = left[row];
        }
    }
}

fn dc(pred: &mut [u8], size: usize, above: &[u8], left: &[u8], have_left: bool, have_above: bool) {
    let log2 = size.trailing_zeros();
    let avg = if have_left && have_above {
        let sum = side_sum(left, size) + side_sum(above, size);
        (sum + size) >> (log2 + 1)
    } else if have_left {
        let sum = side_sum(left, size);
        (sum + (1 << (log2 - 1))) >> log2
    } else if have_above {
        let sum = side_sum(above, size);
        (sum + (1 << (log2 - 1))) >> log2
    } else {
        128
    };
    let avg = u8::try_from(avg).unwrap_or(128);
    pred.fill(avg);
}

fn side_sum(samples: &[u8], size: usize) -> usize {
    samples
        .iter()
        .take(size)
        .map(|sample| usize::from(*sample))
        .sum()
}

fn d45(pred: &mut [u8], size: usize, above: &[u8]) {
    for row in 0..size {
        for col in 0..size {
            pred[row * size + col] = if row + col + 2 < size * 2 {
                round2(above[row + col], above[row + col + 1], above[row + col + 2])
            } else {
                above[size * 2 - 1]
            };
        }
    }
}

fn d63(pred: &mut [u8], size: usize, above: &[u8]) {
    for row in 0..size {
        for col in 0..size {
            let base = row / 2 + col;
            pred[row * size + col] = if row & 1 == 1 {
                round2(above[base], above[base + 1], above[base + 2])
            } else {
                round1(above[base], above[base + 1])
            };
        }
    }
}

fn d207(pred: &mut [u8], size: usize, left: &[u8]) {
    for col in 0..size {
        pred[(size - 1) * size + col] = left[size - 1];
    }
    for row in 0..size - 1 {
        pred[row * size] = round1(left[row], left[row + 1]);
    }
    for row in 0..size - 2 {
        pred[row * size + 1] = round2(left[row], left[row + 1], left[row + 2]);
    }
    pred[(size - 2) * size + 1] = round2(left[size - 2], left[size - 1], left[size - 1]);
    let mut row = size - 2;
    loop {
        for col in 2..size {
            pred[row * size + col] = pred[(row + 1) * size + col - 2];
        }
        if row == 0 {
            break;
        }
        row -= 1;
    }
}

fn d117(pred: &mut [u8], size: usize, above: &[u8], left: &[u8], corner: u8) {
    for col in 0..size {
        let previous = if col == 0 { corner } else { above[col - 1] };
        pred[col] = round1(previous, above[col]);
    }
    pred[size] = round2(left[0], corner, above[0]);
    for col in 1..size {
        let back = if col == 1 { corner } else { above[col - 2] };
        pred[size + col] = round2(back, above[col - 1], above[col]);
    }
    pred[2 * size] = round2(corner, left[0], left[1]);
    for row in 3..size {
        pred[row * size] = round2(left[row - 3], left[row - 2], left[row - 1]);
    }
    for row in 2..size {
        for col in 1..size {
            pred[row * size + col] = pred[(row - 2) * size + col - 1];
        }
    }
}

fn d135(pred: &mut [u8], size: usize, above: &[u8], left: &[u8], corner: u8) {
    pred[0] = round2(left[0], corner, above[0]);
    for col in 1..size {
        let back = if col == 1 { corner } else { above[col - 2] };
        pred[col] = round2(back, above[col - 1], above[col]);
    }
    pred[size] = round2(corner, left[0], left[1]);
    for row in 2..size {
        pred[row * size] = round2(left[row - 2], left[row - 1], left[row]);
    }
    for row in 1..size {
        for col in 1..size {
            pred[row * size + col] = pred[(row - 1) * size + col - 1];
        }
    }
}

fn d153(pred: &mut [u8], size: usize, above: &[u8], left: &[u8], corner: u8) {
    pred[0] = round1(left[0], corner);
    for row in 1..size {
        pred[row * size] = round1(left[row - 1], left[row]);
    }
    pred[1] = round2(left[0], corner, above[0]);
    pred[size + 1] = round2(corner, left[0], left[1]);
    for row in 2..size {
        pred[row * size + 1] = round2(left[row - 2], left[row - 1], left[row]);
    }
    for col in 2..size {
        let a = if col == 2 { corner } else { above[col - 3] };
        let b = if col == 2 { above[0] } else { above[col - 2] };
        let c = above[col - 1];
        pred[col] = round2(a, b, c);
    }
    for row in 1..size {
        for col in 2..size {
            pred[row * size + col] = pred[(row - 1) * size + col - 2];
        }
    }
}

fn tm(pred: &mut [u8], size: usize, above: &[u8], left: &[u8], corner: u8) {
    for row in 0..size {
        for col in 0..size {
            let value = i32::from(above[col]) + i32::from(left[row]) - i32::from(corner);
            pred[row * size + col] = clip(value);
        }
    }
}

fn round1(a: u8, b: u8) -> u8 {
    u8::try_from((usize::from(a) + usize::from(b) + 1) >> 1).unwrap_or(0)
}

fn round2(a: u8, b: u8, c: u8) -> u8 {
    let sum = usize::from(a) + usize::from(b) * 2 + usize::from(c) + 2;
    u8::try_from(sum >> 2).unwrap_or(0)
}

fn clip(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
}
