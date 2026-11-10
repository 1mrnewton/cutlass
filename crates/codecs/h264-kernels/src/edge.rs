//! Deblocking one whole edge in a plane, written once and built twice:
//! over the NEON filters and over the plain ones. The samples around the
//! edge are read from the plane, filtered, and written back in one
//! function, so the filters inline into it.
//!
//! Everything is safe indexing into the caller's plane. An edge whose
//! samples do not all lie in it is left alone and reported.

/// One edge in a plane.
#[derive(Clone, Copy)]
pub struct Edge {
    /// The first q0 sample: the edge's first sample on its right or lower
    /// side.
    pub at: usize,
    /// Samples between one picture row of the edge and the next.
    pub down: usize,
    /// The edge runs down the picture; otherwise along a row.
    pub vertical: bool,
    pub alpha: i32,
    pub beta: i32,
    /// The clipping value of each quarter of the edge for a strength below
    /// 4, negative for a quarter that is not filtered. `None` is strength 4.
    pub tc0: Option<[i8; 4]>,
}

/// How one edge of a macroblock is filtered: its thresholds, and `tc0` as
/// in [`Edge`].
#[derive(Clone, Copy, Debug)]
pub struct Filter {
    pub alpha: i32,
    pub beta: i32,
    pub tc0: Option<[i8; 4]>,
}

/// The edges of one macroblock in one plane, filtered in the order the
/// standard gives: those that run down the picture, left to right, then
/// those that run along it, top to bottom. `None` is an edge left alone.
///
/// Edge `n` is `4 * n` samples into a 16×16 luma macroblock and `2 * n`
/// into an 8×8 chroma one.
#[derive(Clone, Copy, Debug)]
pub struct MbEdges {
    /// The macroblock's first sample.
    pub at: usize,
    /// Samples between one picture row and the next.
    pub down: usize,
    pub vertical: [Option<Filter>; 4],
    pub horizontal: [Option<Filter>; 4],
}

impl MbEdges {
    /// No edge is filtered.
    pub fn is_empty(&self) -> bool {
        self.vertical
            .iter()
            .chain(&self.horizontal)
            .all(Option::is_none)
    }

    /// Whether every sample these edges touch is in a plane of `len`
    /// samples, for a macroblock `size` samples wide and high there: 16 in
    /// luma, 8 in 4:2:0 chroma.
    pub fn fits(&self, len: usize, size: usize) -> bool {
        mb_fits(len, self, size, size / 4)
    }
}

/// Whether every sample the edges of a macroblock touch is in a plane of
/// `len` samples. `size` is the macroblock's width and height there, and
/// `reach` how many samples before an edge its filter reads.
pub(crate) fn mb_fits(len: usize, edges: &MbEdges, size: usize, reach: usize) -> bool {
    let MbEdges {
        at,
        down,
        vertical,
        horizontal,
    } = *edges;
    if down < size + reach {
        return false;
    }
    let Some(end) = (size - 1)
        .checked_mul(down)
        .and_then(|rows| rows.checked_add(size))
        .and_then(|span| span.checked_add(at))
    else {
        return false;
    };
    if end > len {
        return false;
    }
    if vertical[0].is_some() && at < reach {
        return false;
    }
    !(horizontal[0].is_some() && at < reach * down)
}

/// Calls `filter` with each edge of `edges` that is filtered, in order.
/// `step` is the samples between one edge and the next.
pub(crate) fn each_edge(edges: &MbEdges, step: usize, mut filter: impl FnMut(Edge)) {
    for (index, entry) in edges.vertical.iter().enumerate() {
        if let Some(entry) = entry {
            filter(Edge {
                at: edges.at + step * index,
                down: edges.down,
                vertical: true,
                alpha: entry.alpha,
                beta: entry.beta,
                tc0: entry.tc0,
            });
        }
    }
    for (index, entry) in edges.horizontal.iter().enumerate() {
        if let Some(entry) = entry {
            filter(Edge {
                at: edges.at + step * index * edges.down,
                down: edges.down,
                vertical: false,
                alpha: entry.alpha,
                beta: entry.beta,
                tc0: entry.tc0,
            });
        }
    }
}

/// The edge functions over whichever filters are in scope where this is
/// used: `luma_normal_rows`, `luma_strong_rows`, `luma_normal_lines`,
/// `luma_strong_lines`, `chroma_normal`, `chroma_strong`, and
/// `chroma_lines`, which reads eight lines of four samples as four rows of
/// eight. The visibility and the attribute, if any, go on each.
macro_rules! edge_kernels {
    ($visibility:vis, $(#[$feature:meta])*) => {
        /// A 16-sample luma edge. `false` means its samples are not all in
        /// the plane and nothing was done.
        $(#[$feature])*
        $visibility fn luma_edge(plane: &mut [u8], edge: $crate::edge::Edge) -> bool {
            let $crate::edge::Edge {
                at,
                down,
                vertical,
                alpha,
                beta,
                tc0,
            } = edge;
            if vertical {
                // Sixteen lines across the edge, eight samples each.
                let Some(first) = at.checked_sub(4) else {
                    return false;
                };
                if first + 15 * down + 8 > plane.len() {
                    return false;
                }
                let mut lines = [[0u8; 8]; 16];
                for (index, line) in lines.iter_mut().enumerate() {
                    let start = first + index * down;
                    line.copy_from_slice(&plane[start..start + 8]);
                }
                match tc0 {
                    Some(tc0) => luma_normal_lines(&mut lines, alpha, beta, tc0),
                    None => luma_strong_lines(&mut lines, alpha, beta),
                }
                for (index, line) in lines.iter().enumerate() {
                    let start = first + index * down;
                    plane[start..start + 8].copy_from_slice(line);
                }
                return true;
            }
            // Eight rows around the edge, sixteen samples each.
            let Some(first) = at.checked_sub(4 * down) else {
                return false;
            };
            if first + 7 * down + 16 > plane.len() {
                return false;
            }
            let mut rows = [[0u8; 16]; 8];
            for (index, row) in rows.iter_mut().enumerate() {
                let start = first + index * down;
                row.copy_from_slice(&plane[start..start + 16]);
            }
            match tc0 {
                Some(tc0) => luma_normal_rows(&mut rows, alpha, beta, tc0),
                None => luma_strong_rows(&mut rows, alpha, beta),
            }
            // The filter never changes p3 or q3.
            for (index, row) in rows.iter().enumerate().skip(1).take(6) {
                let start = first + index * down;
                plane[start..start + 16].copy_from_slice(row);
            }
            true
        }

        /// An 8-sample chroma edge. `tc0` is per two samples. `false` as
        /// for `luma_edge`.
        $(#[$feature])*
        $visibility fn chroma_edge(plane: &mut [u8], edge: $crate::edge::Edge) -> bool {
            let $crate::edge::Edge {
                at,
                down,
                vertical,
                alpha,
                beta,
                tc0,
            } = edge;
            if vertical {
                // Eight lines across the edge, each p1, p0, q0, q1.
                let Some(first) = at.checked_sub(2) else {
                    return false;
                };
                if first + 7 * down + 4 > plane.len() {
                    return false;
                }
                let mut lines = [[0u8; 4]; 8];
                for (index, line) in lines.iter_mut().enumerate() {
                    let start = first + index * down;
                    line.copy_from_slice(&plane[start..start + 4]);
                }
                let mut rows = chroma_lines(&lines);
                match tc0 {
                    Some(tc0) => chroma_normal(&mut rows, alpha, beta, tc0),
                    None => chroma_strong(&mut rows, alpha, beta),
                }
                // A chroma filter changes only p0 and q0.
                for (index, (&p0, &q0)) in rows[1].iter().zip(&rows[2]).enumerate() {
                    let start = first + index * down;
                    plane[start + 1..start + 3].copy_from_slice(&[p0, q0]);
                }
                return true;
            }
            // Four rows around the edge, eight samples each.
            let Some(first) = at.checked_sub(2 * down) else {
                return false;
            };
            if first + 3 * down + 8 > plane.len() {
                return false;
            }
            let mut rows = [[0u8; 8]; 4];
            for (index, row) in rows.iter_mut().enumerate() {
                let start = first + index * down;
                row.copy_from_slice(&plane[start..start + 8]);
            }
            match tc0 {
                Some(tc0) => chroma_normal(&mut rows, alpha, beta, tc0),
                None => chroma_strong(&mut rows, alpha, beta),
            }
            for (index, row) in rows.iter().enumerate().skip(1).take(2) {
                let start = first + index * down;
                plane[start..start + 8].copy_from_slice(row);
            }
            true
        }
    };
}
pub(crate) use edge_kernels;
