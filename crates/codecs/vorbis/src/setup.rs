//! The setup header (Vorbis I section 4.2.4): codebooks, time placeholders,
//! floors, residues, mappings, and modes.

#[cfg(test)]
use crate::bits::BitWriter;
use crate::bits::{BitReader, ilog};
#[cfg(test)]
use crate::codebook::BookSpec;
use crate::codebook::{Budget, Codebook};
#[cfg(test)]
use crate::header::setup_writer;
use crate::header::{malformed, setup_reader};
use cutlass_decode::CodecError;

/// Floor 1 points, the spec's limit.
pub(crate) const MAX_FLOOR1_VALUES: usize = 65;

/// A decoded setup header, ready for audio packets.
#[derive(Clone, Debug)]
pub(crate) struct Setup {
    pub books: Vec<Codebook>,
    pub floors: Vec<Floor>,
    pub residues: Vec<Residue>,
    pub mappings: Vec<Mapping>,
    pub modes: Vec<Mode>,
}

/// A setup header as written: book descriptions instead of decode tables.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SetupSpec {
    pub books: Vec<BookSpec>,
    pub floors: Vec<Floor>,
    pub residues: Vec<Residue>,
    pub mappings: Vec<Mapping>,
    pub modes: Vec<Mode>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Floor {
    Zero(Floor0),
    One(Floor1),
}

/// Floor 0 (section 6): an LSP curve on a Bark-warped scale.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Floor0 {
    pub order: usize,
    pub rate: u32,
    pub bark_map_size: u32,
    pub amplitude_bits: u32,
    pub amplitude_offset: u32,
    pub books: Vec<u8>,
    /// The Bark map for the short and the long block, `n / 2` entries
    /// each. Built from the blocksizes, not written.
    pub maps: [Vec<i32>; 2],
}

/// Floor 1 (section 7): a piecewise-linear curve in dB.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Floor1 {
    pub partition_classes: Vec<u8>,
    pub classes: Vec<Floor1Class>,
    /// 1 to 4: the dB step of one Y unit is `multiplier` table steps.
    pub multiplier: u8,
    pub range_bits: u8,
    /// X of every point in stream order; 0 and `2^range_bits` first.
    pub xs: Vec<u16>,
    /// Point indices by increasing X.
    pub sorted: Vec<usize>,
    /// For point `i >= 2`, the earlier points just below and above it.
    pub low: Vec<usize>,
    pub high: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Floor1Class {
    pub dimensions: u8,
    pub subclass_bits: u8,
    pub masterbook: u8,
    /// One per subclass; `None` codes Y as 0.
    pub subclass_books: Vec<Option<u8>>,
}

/// A residue (section 8).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Residue {
    pub kind: u8,
    pub begin: u32,
    pub end: u32,
    pub partition_size: u32,
    pub classifications: u8,
    pub classbook: u8,
    /// Per classification, the book of each of the eight passes.
    pub books: Vec<[Option<u8>; 8]>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Mapping {
    /// Magnitude and angle channel of each coupling step.
    pub coupling: Vec<(u8, u8)>,
    /// Submap of each channel.
    pub mux: Vec<u8>,
    /// Floor and residue of each submap.
    pub submaps: Vec<(u8, u8)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mode {
    /// Long block.
    pub long: bool,
    pub mapping: u8,
}

impl Setup {
    /// Parses a setup header packet.
    ///
    /// # Errors
    ///
    /// `Malformed` or `Truncated` as section 4.2.4 says the stream is
    /// undecodable; `Limit` when the books exceed this crate's bounds.
    pub(crate) fn parse(
        packet: &[u8],
        channels: usize,
        blocksizes: [usize; 2],
    ) -> Result<Self, CodecError> {
        let mut reader = setup_reader(packet)?;
        let reader = &mut reader;
        let mut budget = Budget::default();
        let book_count = reader.take(8, "setup codebooks")? as usize + 1;
        let mut books = Vec::with_capacity(book_count);
        for _ in 0..book_count {
            books.push(Codebook::parse(reader, &mut budget)?);
        }
        let floors_etc = SetupTail::parse(reader, channels, blocksizes, book_count, &|book| {
            books
                .get(usize::from(book))
                .is_some_and(|b| b.values.is_some())
        })?;
        Ok(Self {
            books,
            floors: floors_etc.floors,
            residues: floors_etc.residues,
            mappings: floors_etc.mappings,
            modes: floors_etc.modes,
        })
    }
}

#[cfg(test)]
impl SetupSpec {
    /// Parses a setup header into its written form.
    #[cfg(test)]
    pub(crate) fn parse(
        packet: &[u8],
        channels: usize,
        blocksizes: [usize; 2],
    ) -> Result<Self, CodecError> {
        let mut reader = setup_reader(packet)?;
        let reader = &mut reader;
        let mut budget = Budget::default();
        let book_count = reader.take(8, "setup codebooks")? as usize + 1;
        let mut books = Vec::with_capacity(book_count);
        for _ in 0..book_count {
            books.push(BookSpec::parse(reader, &mut budget)?);
        }
        let tail = SetupTail::parse(reader, channels, blocksizes, book_count, &|book| {
            books
                .get(usize::from(book))
                .is_some_and(|b| b.lookup.is_some())
        })?;
        Ok(Self {
            books,
            floors: tail.floors,
            residues: tail.residues,
            mappings: tail.mappings,
            modes: tail.modes,
        })
    }

    /// The setup header packet.
    pub(crate) fn write(&self, channels: usize) -> Vec<u8> {
        let mut writer = setup_writer();
        let w = &mut writer;
        w.write(self.books.len() as u32 - 1, 8);
        for book in &self.books {
            book.write(w);
        }
        // One time-domain placeholder, type 0.
        w.write(0, 6);
        w.write(0, 16);
        w.write(self.floors.len() as u32 - 1, 6);
        for floor in &self.floors {
            match floor {
                Floor::Zero(floor) => {
                    w.write(0, 16);
                    w.write(floor.order as u32, 8);
                    w.write(floor.rate, 16);
                    w.write(floor.bark_map_size, 16);
                    w.write(floor.amplitude_bits, 6);
                    w.write(floor.amplitude_offset, 8);
                    w.write(floor.books.len() as u32 - 1, 4);
                    for &book in &floor.books {
                        w.write(u32::from(book), 8);
                    }
                }
                Floor::One(floor) => {
                    w.write(1, 16);
                    write_floor1(w, floor);
                }
            }
        }
        w.write(self.residues.len() as u32 - 1, 6);
        for residue in &self.residues {
            write_residue(w, residue);
        }
        w.write(self.mappings.len() as u32 - 1, 6);
        for mapping in &self.mappings {
            write_mapping(w, mapping, channels);
        }
        w.write(self.modes.len() as u32 - 1, 6);
        for mode in &self.modes {
            w.write_bit(mode.long);
            w.write(0, 16);
            w.write(0, 16);
            w.write(u32::from(mode.mapping), 8);
        }
        w.write_bit(true);
        writer.finish()
    }
}

struct SetupTail {
    floors: Vec<Floor>,
    residues: Vec<Residue>,
    mappings: Vec<Mapping>,
    modes: Vec<Mode>,
}

impl SetupTail {
    fn parse(
        reader: &mut BitReader<'_>,
        channels: usize,
        blocksizes: [usize; 2],
        book_count: usize,
        has_values: &dyn Fn(u8) -> bool,
    ) -> Result<Self, CodecError> {
        let times = reader.take(6, "setup time count")? + 1;
        for _ in 0..times {
            let at = reader.position();
            if reader.take(16, "setup time")? != 0 {
                return Err(malformed(at, "time domain type"));
            }
        }
        let floor_count = reader.take(6, "setup floor count")? as usize + 1;
        let mut floors = Vec::with_capacity(floor_count);
        for _ in 0..floor_count {
            let at = reader.position();
            match reader.take(16, "floor type")? {
                0 => {
                    let order = reader.take(8, "floor 0")? as usize;
                    let rate = reader.take(16, "floor 0")?;
                    let bark_map_size = reader.take(16, "floor 0")?;
                    let amplitude_bits = reader.take(6, "floor 0")?;
                    let amplitude_offset = reader.take(8, "floor 0")?;
                    let count = reader.take(4, "floor 0")? as usize + 1;
                    let mut books = Vec::with_capacity(count);
                    for _ in 0..count {
                        books.push(read_book(reader, book_count, "floor 0 book")?);
                    }
                    if order == 0 || rate == 0 || bark_map_size == 0 {
                        return Err(malformed(at, "floor 0 configuration"));
                    }
                    let mut floor = Floor0 {
                        order,
                        rate,
                        bark_map_size,
                        amplitude_bits,
                        amplitude_offset,
                        books,
                        maps: [Vec::new(), Vec::new()],
                    };
                    floor.maps = [
                        floor.bark_map(blocksizes[0] / 2),
                        floor.bark_map(blocksizes[1] / 2),
                    ];
                    floors.push(Floor::Zero(floor));
                }
                1 => floors.push(Floor::One(parse_floor1(reader, book_count)?)),
                _ => return Err(malformed(at, "floor type")),
            }
        }
        let residue_count = reader.take(6, "setup residue count")? as usize + 1;
        let mut residues = Vec::with_capacity(residue_count);
        for _ in 0..residue_count {
            residues.push(parse_residue(reader, book_count, has_values)?);
        }
        let mapping_count = reader.take(6, "setup mapping count")? as usize + 1;
        let mut mappings = Vec::with_capacity(mapping_count);
        for _ in 0..mapping_count {
            mappings.push(parse_mapping(reader, channels, floor_count, residue_count)?);
        }
        let mode_count = reader.take(6, "setup mode count")? as usize + 1;
        let mut modes = Vec::with_capacity(mode_count);
        for _ in 0..mode_count {
            let at = reader.position();
            let long = reader.take_bit("mode")?;
            let window = reader.take(16, "mode")?;
            let transform = reader.take(16, "mode")?;
            let mapping = reader.take(8, "mode")? as usize;
            if window != 0 || transform != 0 || mapping >= mapping_count {
                return Err(malformed(at, "mode configuration"));
            }
            modes.push(Mode {
                long,
                mapping: mapping as u8,
            });
        }
        let at = reader.position();
        if !reader.take_bit("setup framing bit")? {
            return Err(malformed(at, "setup framing bit"));
        }
        Ok(Self {
            floors,
            residues,
            mappings,
            modes,
        })
    }
}

/// A codebook number, checked against the books the setup declared.
fn read_book(
    reader: &mut BitReader<'_>,
    book_count: usize,
    what: &'static str,
) -> Result<u8, CodecError> {
    let at = reader.position();
    let book = reader.take(8, what)? as usize;
    if book >= book_count {
        return Err(malformed(at, what));
    }
    Ok(book as u8)
}

fn parse_floor1(reader: &mut BitReader<'_>, book_count: usize) -> Result<Floor1, CodecError> {
    let at = reader.position();
    let partitions = reader.take(5, "floor 1")? as usize;
    let mut partition_classes = Vec::with_capacity(partitions);
    for _ in 0..partitions {
        partition_classes.push(reader.take(4, "floor 1")? as u8);
    }
    let class_count = partition_classes
        .iter()
        .map(|&c| usize::from(c) + 1)
        .max()
        .unwrap_or(0);
    let mut classes = Vec::with_capacity(class_count);
    for _ in 0..class_count {
        let dimensions = reader.take(3, "floor 1 class")? as u8 + 1;
        let subclass_bits = reader.take(2, "floor 1 class")? as u8;
        let masterbook = if subclass_bits > 0 {
            read_book(reader, book_count, "floor 1 masterbook")?
        } else {
            0
        };
        let mut subclass_books = Vec::with_capacity(1 << subclass_bits);
        for _ in 0..(1 << subclass_bits) {
            let book_at = reader.position();
            let value = reader.take(8, "floor 1 subclass book")? as usize;
            subclass_books.push(match value {
                0 => None,
                value if value - 1 < book_count => Some((value - 1) as u8),
                _ => return Err(malformed(book_at, "floor 1 subclass book")),
            });
        }
        classes.push(Floor1Class {
            dimensions,
            subclass_bits,
            masterbook,
            subclass_books,
        });
    }
    let multiplier = reader.take(2, "floor 1")? as u8 + 1;
    let range_bits = reader.take(4, "floor 1")? as u8;
    let mut xs: Vec<u16> = vec![0, 1 << range_bits];
    for &class in &partition_classes {
        for _ in 0..classes[usize::from(class)].dimensions {
            if xs.len() >= MAX_FLOOR1_VALUES {
                return Err(malformed(reader.position(), "floor 1 over 65 points"));
            }
            xs.push(reader.take(u32::from(range_bits), "floor 1 X")? as u16);
        }
    }
    Floor1::new(partition_classes, classes, multiplier, range_bits, xs)
        .ok_or(malformed(at, "floor 1 X values repeat"))
}

impl Floor1 {
    /// A floor 1 with its point order and neighbors; `None` when two
    /// points share an X.
    pub(crate) fn new(
        partition_classes: Vec<u8>,
        classes: Vec<Floor1Class>,
        multiplier: u8,
        range_bits: u8,
        xs: Vec<u16>,
    ) -> Option<Self> {
        let mut sorted: Vec<usize> = (0..xs.len()).collect();
        sorted.sort_by_key(|&i| xs[i]);
        if sorted.windows(2).any(|pair| xs[pair[0]] == xs[pair[1]]) {
            return None;
        }
        let mut low = vec![0; xs.len()];
        let mut high = vec![1; xs.len()];
        for i in 2..xs.len() {
            let x = xs[i];
            let mut best_low: Option<usize> = None;
            let mut best_high: Option<usize> = None;
            for (n, &other) in xs.iter().enumerate().take(i) {
                if other < x && best_low.is_none_or(|b| other > xs[b]) {
                    best_low = Some(n);
                }
                if other > x && best_high.is_none_or(|b| other < xs[b]) {
                    best_high = Some(n);
                }
            }
            low[i] = best_low.unwrap_or(0);
            high[i] = best_high.unwrap_or(1);
        }
        Some(Self {
            partition_classes,
            classes,
            multiplier,
            range_bits,
            xs,
            sorted,
            low,
            high,
        })
    }
}

#[cfg(test)]
fn write_floor1(w: &mut BitWriter, floor: &Floor1) {
    w.write(floor.partition_classes.len() as u32, 5);
    for &class in &floor.partition_classes {
        w.write(u32::from(class), 4);
    }
    for class in &floor.classes {
        w.write(u32::from(class.dimensions) - 1, 3);
        w.write(u32::from(class.subclass_bits), 2);
        if class.subclass_bits > 0 {
            w.write(u32::from(class.masterbook), 8);
        }
        for book in &class.subclass_books {
            w.write(book.map_or(0, |b| u32::from(b) + 1), 8);
        }
    }
    w.write(u32::from(floor.multiplier) - 1, 2);
    w.write(u32::from(floor.range_bits), 4);
    for &x in &floor.xs[2..] {
        w.write(u32::from(x), u32::from(floor.range_bits));
    }
}

fn parse_residue(
    reader: &mut BitReader<'_>,
    book_count: usize,
    has_values: &dyn Fn(u8) -> bool,
) -> Result<Residue, CodecError> {
    let at = reader.position();
    let kind = reader.take(16, "residue type")?;
    if kind > 2 {
        return Err(malformed(at, "residue type"));
    }
    let begin = reader.take(24, "residue")?;
    let end = reader.take(24, "residue")?;
    let partition_size = reader.take(24, "residue")? + 1;
    let classifications = reader.take(6, "residue")? as u8 + 1;
    let classbook = read_book(reader, book_count, "residue classbook")?;
    let mut cascades = Vec::with_capacity(usize::from(classifications));
    for _ in 0..classifications {
        let low = reader.take(3, "residue cascade")?;
        let high = if reader.take_bit("residue cascade")? {
            reader.take(5, "residue cascade")?
        } else {
            0
        };
        cascades.push(high * 8 + low);
    }
    let mut books = Vec::with_capacity(cascades.len());
    for cascade in cascades {
        let mut passes = [None; 8];
        for (pass, slot) in passes.iter_mut().enumerate() {
            if cascade & (1 << pass) != 0 {
                let book_at = reader.position();
                let number = read_book(reader, book_count, "residue book")?;
                if !has_values(number) {
                    return Err(malformed(book_at, "residue book without values"));
                }
                *slot = Some(number);
            }
        }
        books.push(passes);
    }
    Ok(Residue {
        kind: kind as u8,
        begin,
        end,
        partition_size,
        classifications,
        classbook,
        books,
    })
}

#[cfg(test)]
fn write_residue(w: &mut BitWriter, residue: &Residue) {
    w.write(u32::from(residue.kind), 16);
    w.write(residue.begin, 24);
    w.write(residue.end, 24);
    w.write(residue.partition_size - 1, 24);
    w.write(u32::from(residue.classifications) - 1, 6);
    w.write(u32::from(residue.classbook), 8);
    for passes in &residue.books {
        let cascade = passes
            .iter()
            .enumerate()
            .filter(|(_, book)| book.is_some())
            .fold(0_u32, |bits, (pass, _)| bits | (1 << pass));
        w.write(cascade & 7, 3);
        w.write_bit(cascade > 7);
        if cascade > 7 {
            w.write(cascade >> 3, 5);
        }
    }
    for passes in &residue.books {
        for book in passes.iter().flatten() {
            w.write(u32::from(*book), 8);
        }
    }
}

fn parse_mapping(
    reader: &mut BitReader<'_>,
    channels: usize,
    floor_count: usize,
    residue_count: usize,
) -> Result<Mapping, CodecError> {
    let at = reader.position();
    if reader.take(16, "mapping type")? != 0 {
        return Err(malformed(at, "mapping type"));
    }
    let submap_count = if reader.take_bit("mapping")? {
        reader.take(4, "mapping")? as usize + 1
    } else {
        1
    };
    let mut coupling = Vec::new();
    if reader.take_bit("mapping")? {
        let steps = reader.take(8, "mapping coupling")? as usize + 1;
        let width = ilog(channels as u32 - 1);
        for _ in 0..steps {
            let step_at = reader.position();
            let magnitude = reader.take(width, "mapping coupling")? as usize;
            let angle = reader.take(width, "mapping coupling")? as usize;
            if magnitude == angle || magnitude >= channels || angle >= channels {
                return Err(malformed(step_at, "coupling channels"));
            }
            coupling.push((magnitude as u8, angle as u8));
        }
    }
    let reserved_at = reader.position();
    if reader.take(2, "mapping reserved")? != 0 {
        return Err(malformed(reserved_at, "mapping reserved bits"));
    }
    let mut mux = vec![0_u8; channels];
    if submap_count > 1 {
        for slot in &mut mux {
            let mux_at = reader.position();
            let value = reader.take(4, "mapping mux")? as usize;
            if value >= submap_count {
                return Err(malformed(mux_at, "mapping mux"));
            }
            *slot = value as u8;
        }
    }
    let mut submaps = Vec::with_capacity(submap_count);
    for _ in 0..submap_count {
        reader.take(8, "mapping submap")?;
        let floor_at = reader.position();
        let floor = reader.take(8, "mapping submap")? as usize;
        let residue = reader.take(8, "mapping submap")? as usize;
        if floor >= floor_count || residue >= residue_count {
            return Err(malformed(floor_at, "mapping submap"));
        }
        submaps.push((floor as u8, residue as u8));
    }
    Ok(Mapping {
        coupling,
        mux,
        submaps,
    })
}

#[cfg(test)]
fn write_mapping(w: &mut BitWriter, mapping: &Mapping, channels: usize) {
    w.write(0, 16);
    w.write_bit(mapping.submaps.len() > 1);
    if mapping.submaps.len() > 1 {
        w.write(mapping.submaps.len() as u32 - 1, 4);
    }
    w.write_bit(!mapping.coupling.is_empty());
    if !mapping.coupling.is_empty() {
        w.write(mapping.coupling.len() as u32 - 1, 8);
        let width = ilog(channels as u32 - 1);
        for &(magnitude, angle) in &mapping.coupling {
            w.write(u32::from(magnitude), width);
            w.write(u32::from(angle), width);
        }
    }
    w.write(0, 2);
    if mapping.submaps.len() > 1 {
        for &submap in &mapping.mux {
            w.write(u32::from(submap), 4);
        }
    }
    for &(floor, residue) in &mapping.submaps {
        w.write(0, 8);
        w.write(u32::from(floor), 8);
        w.write(u32::from(residue), 8);
    }
}

impl Floor0 {
    /// The Bark map of section 6.2.3 for a half block of `n` samples, with
    /// a trailing `-1` sentinel. Arithmetic as the reference: the
    /// frequency in `f32`, the Bark function in `f64`, the scale in `f32`.
    pub(crate) fn bark_map(&self, n: usize) -> Vec<i32> {
        let bark = |f: f32| -> f64 {
            let f = f64::from(f);
            13.1 * (0.00074 * f).atan() + 2.24 * (1.85e-8 * f * f).atan() + 1e-4 * f
        };
        let size = self.bark_map_size as i32;
        let scale = (f64::from(self.bark_map_size) / bark(self.rate as f32 / 2.0)) as f32;
        let mut map = Vec::with_capacity(n + 1);
        for i in 0..n {
            let frequency = (self.rate as f32 / 2.0) / n as f32 * i as f32;
            let value = (bark(frequency) * f64::from(scale)).floor() as i32;
            map.push(value.min(size - 1));
        }
        map.push(-1);
        map
    }
}

#[cfg(test)]
mod tests {
    use super::{Floor, Floor0, Floor1, Floor1Class, Mapping, Mode, Residue, Setup, SetupSpec};
    use crate::codebook::{BookSpec, Lookup, float32_pack};

    pub(crate) fn scalar_book(entries: usize) -> BookSpec {
        let bits = entries.trailing_zeros() as u8;
        BookSpec {
            dimensions: 1,
            lengths: vec![bits.max(1); entries],
            lookup: None,
        }
    }

    pub(crate) fn vq_book() -> BookSpec {
        BookSpec {
            dimensions: 2,
            lengths: vec![4; 16],
            lookup: Some(Lookup {
                kind: 1,
                minimum: float32_pack(-2.0),
                delta: float32_pack(1.0),
                value_bits: 2,
                sequence: false,
                multiplicands: vec![0, 1, 2, 3],
            }),
        }
    }

    fn spec() -> SetupSpec {
        let floor1 = Floor1::new(
            vec![0, 1],
            vec![
                Floor1Class {
                    dimensions: 2,
                    subclass_bits: 0,
                    masterbook: 0,
                    subclass_books: vec![Some(0)],
                },
                Floor1Class {
                    dimensions: 3,
                    subclass_bits: 1,
                    masterbook: 2,
                    subclass_books: vec![None, Some(0)],
                },
            ],
            2,
            7,
            vec![0, 128, 64, 32, 96, 16, 48],
        )
        .unwrap();
        SetupSpec {
            books: vec![scalar_book(128), vq_book(), scalar_book(2)],
            floors: vec![
                Floor::One(floor1),
                Floor::Zero(Floor0 {
                    order: 16,
                    rate: 44_100,
                    bark_map_size: 256,
                    amplitude_bits: 6,
                    amplitude_offset: 100,
                    books: vec![1],
                    maps: [Vec::new(), Vec::new()],
                }),
            ],
            residues: vec![Residue {
                kind: 2,
                begin: 0,
                end: 256,
                partition_size: 16,
                classifications: 2,
                classbook: 2,
                books: vec![
                    [None; 8],
                    [Some(1), None, None, Some(1), None, None, None, None],
                ],
            }],
            mappings: vec![Mapping {
                coupling: vec![(0, 1)],
                mux: vec![0, 0],
                submaps: vec![(0, 0)],
            }],
            modes: vec![
                Mode {
                    long: false,
                    mapping: 0,
                },
                Mode {
                    long: true,
                    mapping: 0,
                },
            ],
        }
    }

    #[test]
    fn a_setup_round_trips_and_builds() {
        let spec = spec();
        let packet = spec.write(2);
        let mut back = SetupSpec::parse(&packet, 2, [256, 2048]).unwrap();
        // The Bark maps are derived, not written.
        if let Floor::Zero(floor) = &mut back.floors[1] {
            assert_eq!(floor.maps[0].len(), 129);
            assert_eq!(floor.maps[1].len(), 1025);
            assert_eq!(floor.maps[0][0], 0);
            assert_eq!(floor.maps[1][1024], -1);
            assert!(floor.maps[1][..1024].windows(2).all(|p| p[0] <= p[1]));
            assert!(floor.maps[1][1023] <= 255);
            floor.maps = [Vec::new(), Vec::new()];
        }
        assert_eq!(back, spec);
        let setup = Setup::parse(&packet, 2, [256, 2048]).unwrap();
        assert_eq!(setup.books.len(), 3);
        assert_eq!(setup.modes.len(), 2);
    }

    #[test]
    fn floor1_neighbors_are_the_closest_earlier_points() {
        let Floor::One(floor) = &spec().floors[0] else {
            unreachable!()
        };
        // X: 0 128 64 32 96 16 48.
        assert_eq!(floor.sorted, [0, 5, 3, 6, 2, 4, 1]);
        assert_eq!((floor.low[2], floor.high[2]), (0, 1));
        assert_eq!((floor.low[3], floor.high[3]), (0, 2));
        assert_eq!((floor.low[4], floor.high[4]), (2, 1));
        assert_eq!((floor.low[5], floor.high[5]), (0, 3));
        assert_eq!((floor.low[6], floor.high[6]), (3, 2));
        assert!(Floor1::new(vec![], vec![], 1, 4, vec![0, 16, 16]).is_none());
    }

    #[test]
    fn broken_setups_are_refused() {
        let good = spec().write(2);
        // Every truncation fails cleanly.
        for len in 0..good.len() - 1 {
            assert!(
                Setup::parse(&good[..len], 2, [256, 2048]).is_err(),
                "length {len}"
            );
        }
        // Coupling a channel with itself.
        let mut same = spec();
        same.mappings[0].coupling = vec![(1, 1)];
        assert!(Setup::parse(&same.write(2), 2, [256, 2048]).is_err());
        // A residue book with no values.
        let mut scalar = spec();
        scalar.residues[0].books[1][0] = Some(0);
        assert!(Setup::parse(&scalar.write(2), 2, [256, 2048]).is_err());
        // A mode naming a missing mapping.
        let mut mode = spec();
        mode.modes[1].mapping = 3;
        assert!(Setup::parse(&mode.write(2), 2, [256, 2048]).is_err());
        // A floor book past the last one.
        let mut book = spec();
        if let Floor::Zero(floor) = &mut book.floors[1] {
            floor.books = vec![9];
        }
        assert!(Setup::parse(&book.write(2), 2, [256, 2048]).is_err());
        // A missing framing bit: the last bit written is the highest set
        // bit of the last byte.
        let mut unframed = good.clone();
        let last = unframed.len() - 1;
        let top = 7 - unframed[last].leading_zeros();
        unframed[last] &= !(1 << top);
        assert!(Setup::parse(&unframed, 2, [256, 2048]).is_err());
        // One channel cannot couple: both channel numbers read as zero.
        assert!(Setup::parse(&good, 1, [256, 2048]).is_err());
    }
}
