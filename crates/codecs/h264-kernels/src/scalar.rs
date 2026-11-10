//! The plain version of every kernel. It is what a CPU without a vector
//! version runs, and what each vector version is tested against.

use crate::deblock::{Lines, Rows};

/// See [`crate::deblock::luma_normal_rows`].
pub fn luma_normal_rows(rows: &mut Rows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    for column in 0..16 {
        let tc0 = i32::from(tc0[column / 4]);
        if tc0 < 0 {
            continue;
        }
        let [_, p2, p1, p0, q0, q1, q2, _] = line(rows, column);
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let ap = (p2 - p0).abs();
        let aq = (q2 - q0).abs();
        let tc = tc0 + i32::from(ap < beta) + i32::from(aq < beta);
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        rows[3][column] = clip(p0 + delta);
        rows[4][column] = clip(q0 - delta);
        if ap < beta {
            let d = ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
            rows[2][column] = clip(p1 + d);
        }
        if aq < beta {
            let d = ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
            rows[5][column] = clip(q1 + d);
        }
    }
}

/// See [`crate::deblock::luma_strong_rows`].
pub fn luma_strong_rows(rows: &mut Rows, alpha: i32, beta: i32) {
    for column in 0..16 {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = line(rows, column);
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let small = (p0 - q0).abs() < ((alpha >> 2) + 2);
        if small && (p2 - p0).abs() < beta {
            rows[3][column] = clip((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            rows[2][column] = clip((p2 + p1 + p0 + q0 + 2) >> 2);
            rows[1][column] = clip((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
        } else {
            rows[3][column] = clip((2 * p1 + p0 + q1 + 2) >> 2);
        }
        if small && (q2 - q0).abs() < beta {
            rows[4][column] = clip((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            rows[5][column] = clip((p0 + q0 + q1 + q2 + 2) >> 2);
            rows[6][column] = clip((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
        } else {
            rows[4][column] = clip((2 * q1 + q0 + p1 + 2) >> 2);
        }
    }
}

/// See [`crate::deblock::luma_normal_lines`].
pub fn luma_normal_lines(lines: &mut Lines, alpha: i32, beta: i32, tc0: [i8; 4]) {
    let mut rows = turned(lines);
    luma_normal_rows(&mut rows, alpha, beta, tc0);
    turn_back(&rows, lines);
}

/// See [`crate::deblock::luma_strong_lines`].
pub fn luma_strong_lines(lines: &mut Lines, alpha: i32, beta: i32) {
    let mut rows = turned(lines);
    luma_strong_rows(&mut rows, alpha, beta);
    turn_back(&rows, lines);
}

fn line(rows: &Rows, column: usize) -> [i32; 8] {
    let mut line = [0i32; 8];
    for (sample, row) in line.iter_mut().zip(rows) {
        *sample = i32::from(row[column]);
    }
    line
}

fn clip(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn turned(lines: &Lines) -> Rows {
    let mut rows = [[0u8; 16]; 8];
    for (index, line) in lines.iter().enumerate() {
        for (row, &sample) in rows.iter_mut().zip(line) {
            row[index] = sample;
        }
    }
    rows
}

fn turn_back(rows: &Rows, lines: &mut Lines) {
    for (index, line) in lines.iter_mut().enumerate() {
        for (sample, row) in line.iter_mut().zip(rows) {
            *sample = row[index];
        }
    }
}
