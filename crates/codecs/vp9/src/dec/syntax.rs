//! What the parser leaves for reconstruction: per tile column, every
//! block's modes and vectors, the end of block of each of its transform
//! blocks, and their nonzero coefficients, in decoding order.

use super::frame::Mv;

/// One block.
#[derive(Clone, Copy, Debug)]
pub struct BlockRec {
    /// Position in 8×8 units.
    pub row: u16,
    pub col: u16,
    pub bsize: u8,
    /// Whether the block has transform blocks (it did not code skip).
    pub coded: bool,
    pub tx: u8,
    pub uv_tx: u8,
    pub mode: u8,
    pub bmode: [u8; 4],
    pub uv: u8,
    pub interp: u8,
    pub refs: [u8; 2],
    pub bmv: [[Mv; 2]; 4],
}

/// One transform block inside the frame, of a coded block.
#[derive(Clone, Copy, Debug, Default)]
pub struct TxRec {
    pub eob: u16,
    /// How many coefficients follow in `pos` and `val`.
    pub n: u16,
    /// Rows and columns that can hold a nonzero coefficient.
    pub nzr: u8,
    pub nzc: u8,
}

/// Where each superblock row's syntax ends.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowEnd {
    pub blocks: usize,
    pub txs: usize,
    pub coefs: usize,
}

/// The syntax of one tile column, over every tile row.
#[derive(Default)]
pub struct TileSyntax {
    pub col0: usize,
    pub col1: usize,
    pub blocks: Vec<BlockRec>,
    pub txs: Vec<TxRec>,
    /// Coefficient positions, transposed (`column * n + row`).
    pub pos: Vec<u16>,
    pub val: Vec<i32>,
    pub rows: Vec<RowEnd>,
}

impl TileSyntax {
    /// Empties the syntax for columns `col0..col1` of a frame of
    /// `sb_rows` superblock rows, keeping its buffers.
    pub fn reset(&mut self, col0: usize, col1: usize, sb_rows: usize) {
        self.col0 = col0;
        self.col1 = col1;
        self.blocks.clear();
        self.txs.clear();
        self.pos.clear();
        self.val.clear();
        self.rows.clear();
        self.rows.resize(sb_rows, RowEnd::default());
    }

    /// Marks the end of superblock row `sb_row`.
    pub fn end_row(&mut self, sb_row: usize) {
        if let Some(r) = self.rows.get_mut(sb_row) {
            *r = RowEnd {
                blocks: self.blocks.len(),
                txs: self.txs.len(),
                coefs: self.pos.len(),
            };
        }
    }

    /// Where superblock row `sb_row` starts.
    pub fn row_start(&self, sb_row: usize) -> RowEnd {
        if sb_row == 0 {
            RowEnd::default()
        } else {
            self.rows.get(sb_row - 1).copied().unwrap_or_default()
        }
    }
}
