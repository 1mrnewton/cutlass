//! Transport packets from the input, with the packet grid found and found
//! again after corruption.
//!
//! The input is read in large chunks into one buffer and each packet is
//! handed out in place, so no transport packet is copied. When a packet
//! does not start with the sync byte where the grid says, the bytes are
//! searched for the next run of [`SYNC_RUN`] sync bytes at any of the three
//! strides, and the skipped bytes are counted.
//!
//! While a file is opened every byte read is kept, so the first packets can
//! be read again without a second pass over the input: a forward input
//! cannot seek, and a random one saves the bytes.

use std::io::Read;

use cutlass_container::{Input, SourceError};

use crate::limits::{MAX_RESYNC_BYTES, SYNC_BYTE, SYNC_RUN, TS_PACKET_BYTES};
use crate::packet::{PacketSize, find_grid};

/// Bytes asked of the input at a time.
const CHUNK: usize = 192 * 1024;

/// The read position a failed seek puts back.
pub struct Saved {
    file: u64,
    rest: Vec<u8>,
    size: PacketSize,
    synced: bool,
    eof: bool,
}

pub struct Source<R> {
    input: Input<R>,
    buf: Vec<u8>,
    /// Start of the next packet in `buf`, prefix included.
    at: usize,
    /// File offset of `buf[0]`.
    base: u64,
    size: PacketSize,
    synced: bool,
    eof: bool,
    /// Keep every byte read, for [`Source::rewind`].
    keep: bool,
    pub resyncs: u64,
    pub skipped: u64,
    /// Bytes at the end of the input too short for a packet.
    pub tail: u64,
}

/// What [`Source::next`] found.
pub enum Next {
    /// A packet: its file offset, and where its 188 bytes start in the
    /// buffer.
    Packet(u64, usize),
    End,
}

impl<R: Read> Source<R> {
    pub const fn new(input: Input<R>) -> Self {
        Self {
            input,
            buf: Vec::new(),
            at: 0,
            base: 0,
            size: PacketSize::Ts,
            synced: false,
            eof: false,
            keep: false,
            resyncs: 0,
            skipped: 0,
            tail: 0,
        }
    }

    pub const fn size(&self) -> PacketSize {
        self.size
    }

    /// File offset of the next packet.
    pub const fn position(&self) -> u64 {
        self.base + self.at as u64
    }

    pub fn consumed(&self) -> u64 {
        self.input.consumed()
    }

    /// Keeps every byte from here on, until [`Source::rewind`].
    pub fn keep_from_here(&mut self) {
        self.compact();
        self.keep = true;
    }

    /// Stops keeping bytes read.
    pub const fn release(&mut self) {
        self.keep = false;
    }

    /// Reads again from `position`, which must be inside the bytes kept
    /// since [`Source::keep_from_here`].
    pub fn rewind(&mut self, position: u64) -> bool {
        let Some(at) = position.checked_sub(self.base) else {
            return false;
        };
        let Ok(at) = usize::try_from(at) else {
            return false;
        };
        if at > self.buf.len() {
            return false;
        }
        self.at = at;
        true
    }

    /// The bytes of the packet at `index`, from [`Next::Packet`].
    pub fn packet(&self, index: usize) -> &[u8; TS_PACKET_BYTES] {
        // `next` checked that the whole packet is buffered.
        self.buf[index..index + TS_PACKET_BYTES]
            .try_into()
            .unwrap_or(&[0; TS_PACKET_BYTES])
    }

    fn compact(&mut self) {
        if self.keep || self.at == 0 {
            return;
        }
        self.buf.drain(..self.at);
        self.base += self.at as u64;
        self.at = 0;
    }

    /// Makes at least `want` bytes available from `at`, unless the input
    /// ends first.
    fn fill(&mut self, want: usize) -> Result<(), SourceError> {
        while self.buf.len() - self.at < want && !self.eof {
            if self.at > CHUNK {
                self.compact();
            }
            let len = self.buf.len();
            self.buf.resize(len + CHUNK, 0);
            let read = self.input.read(&mut self.buf[len..]);
            let got = *read.as_ref().unwrap_or(&0);
            self.buf.truncate(len + got);
            read?;
            if got == 0 {
                self.eof = true;
            }
        }
        Ok(())
    }

    /// Finds the packet grid from `at`, skipping what is not on it.
    fn sync(&mut self) -> Result<bool, SourceError> {
        // A full run at the largest stride.
        let need = SYNC_RUN * PacketSize::Fec.stride();
        loop {
            self.fill(need + PacketSize::Fec.stride())?;
            let window = &self.buf[self.at..];
            if window.len() < TS_PACKET_BYTES {
                self.skipped += window.len() as u64;
                self.at = self.buf.len();
                return Ok(false);
            }
            // Away from the end of the input, only starts with room for a
            // whole run are tried, so a short run is never taken there.
            let search = if self.eof {
                window.len()
            } else {
                window.len() - need
            }
            .min(MAX_RESYNC_BYTES);
            match find_grid(window, search, SYNC_RUN) {
                Some(grid) if grid.sync_at >= grid.size.sync_offset() => {
                    let start = grid.sync_at - grid.size.sync_offset();
                    self.skipped += start as u64;
                    self.at += start;
                    self.size = grid.size;
                    self.synced = true;
                    return Ok(true);
                }
                Some(grid) => {
                    // The prefix of the first packet is not buffered.
                    self.skipped += grid.sync_at as u64 + 1;
                    self.at += grid.sync_at + 1;
                }
                None => {
                    self.skipped += search as u64;
                    self.at += search;
                }
            }
        }
    }

    /// The next packet on the grid.
    pub fn next(&mut self) -> Result<Next, SourceError> {
        loop {
            if !self.synced && !self.sync()? {
                return Ok(Next::End);
            }
            let stride = self.size.stride();
            self.fill(stride)?;
            let available = self.buf.len() - self.at;
            let offset = self.size.sync_offset();
            if available < offset + TS_PACKET_BYTES {
                self.tail += available as u64;
                self.at = self.buf.len();
                return Ok(Next::End);
            }
            if self.buf[self.at + offset] == SYNC_BYTE {
                let position = self.position();
                let index = self.at + offset;
                self.at += stride.min(available);
                return Ok(Next::Packet(position, index));
            }
            self.resyncs += 1;
            self.synced = false;
            self.at += 1;
            self.skipped += 1;
        }
    }

    /// The read position, to put back after a failed seek.
    pub fn save(&self) -> Saved {
        Saved {
            file: self.base + self.buf.len() as u64,
            rest: self.buf[self.at..].to_vec(),
            size: self.size,
            synced: self.synced,
            eof: self.eof,
        }
    }
}

impl<R: Read + std::io::Seek> Source<R> {
    /// Reads from `position` on, finding the grid there.
    pub fn seek(&mut self, position: u64) -> Result<(), SourceError> {
        self.input.seek(position)?;
        self.buf.clear();
        self.at = 0;
        self.base = position;
        self.synced = false;
        self.eof = false;
        Ok(())
    }

    /// Reads from `position` on, keeping the packet size: `position` is a
    /// packet start.
    pub fn seek_packet(&mut self, position: u64) -> Result<(), SourceError> {
        self.seek(position)?;
        self.synced = true;
        Ok(())
    }

    pub fn restore(&mut self, saved: Saved) -> Result<(), SourceError> {
        self.input.seek(saved.file)?;
        self.base = saved.file - saved.rest.len() as u64;
        self.buf = saved.rest;
        self.at = 0;
        self.size = saved.size;
        self.synced = saved.synced;
        self.eof = saved.eof;
        Ok(())
    }
}
