use super::Error;
use super::{Packet, StreamParams};

/// Writes packets into a container.
///
/// `finish` writes the index and the duration. A writer that is dropped before
/// `finish` does not look like a complete file.
pub trait Mux {
    fn add_stream(&mut self, params: &StreamParams) -> Result<u32, Error>;
    fn write_packet(&mut self, packet: &Packet) -> Result<(), Error>;
    fn finish(&mut self) -> Result<(), Error>;
}
