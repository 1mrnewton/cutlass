//! What a variable font varies along: its axes and named instances
//! (`fvar`), how a setting becomes a place along an axis (`avar`), and the
//! store of deltas the other tables share.
//!
//! A place along an axis is a number from -1 through 0, the face as it is
//! drawn with no setting, to 1, kept as the 2.14 fixed-point value the
//! tables themselves use: 16,384 is 1. Every delta is weighted by how near
//! the place is to the peak of the region the delta was drawn for.

use crate::layout::Tag;
use crate::{FontError, bytes_at, u16_at, u32_at};

/// `value` to the nearest whole number, a half going up: the rounding
/// HarfBuzz uses wherever a varied value becomes whole, kept so that a
/// sum that lands on a half comes out the same here.
pub(crate) fn nearest(value: f32) -> f32 {
    (value + 0.5).floor()
}

/// The most axes a face may vary along here. A face of the widest design
/// space in use has under twenty.
pub(crate) const MAX_AXES: usize = 64;

/// The most axes whose places the layout tables carry with them. A face
/// with more is laid out with the values of its default instance.
pub(crate) const MAX_LAYOUT_AXES: usize = 32;

/// The most named instances listed.
const MAX_INSTANCES: usize = 1_024;

/// One axis a face varies along, in the units a user sets it in.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    /// What varies: `wght` for weight, `wdth` for width, `slnt` for slant,
    /// `ital` for italic, `opsz` for optical size, or a tag of the face's
    /// own.
    pub tag: Tag,
    pub min: f32,
    pub default: f32,
    pub max: f32,
    /// The face asks that the axis not be shown to a user.
    pub hidden: bool,
    /// What the face calls the axis, when it has a readable name.
    pub name: Option<String>,
}

/// One instance a face names: a setting of every axis, such as Bold.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    /// The style's name, such as "Bold Condensed".
    pub name: Option<String>,
    /// The name of the instance as a font of its own.
    pub postscript_name: Option<String>,
    /// A value for each axis, in the order of the face's axes.
    pub coordinates: Vec<f32>,
}

/// One axis as `fvar` stores it.
#[derive(Clone, Copy, Debug)]
struct AxisRecord {
    tag: Tag,
    min: f32,
    default: f32,
    max: f32,
    hidden: bool,
    name_id: u16,
}

/// The `fvar` table of a face.
#[derive(Clone, Debug)]
pub(crate) struct Fvar<'a> {
    axes: Vec<AxisRecord>,
    /// The instance records, each `instance_size` bytes.
    instances: &'a [u8],
    instance_size: usize,
}

fn fixed(data: &[u8], at: usize) -> Result<f32, FontError> {
    let bytes = bytes_at(data, at, 4)?;
    let raw = i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    #[allow(clippy::cast_precision_loss)]
    Ok(raw as f32 / 65_536.0)
}

impl<'a> Fvar<'a> {
    /// Read the table. One with no axes is no table.
    pub(crate) fn read(data: &'a [u8]) -> Result<Option<Self>, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::Unsupported);
        }
        let axes_at = usize::from(u16_at(data, 4)?);
        let axis_count = usize::from(u16_at(data, 8)?);
        let axis_size = usize::from(u16_at(data, 10)?);
        let instance_count = usize::from(u16_at(data, 12)?);
        let instance_size = usize::from(u16_at(data, 14)?);
        if axis_count == 0 {
            return Ok(None);
        }
        if axis_count > MAX_AXES || axis_size < 20 {
            return Err(FontError::BadTable);
        }
        let mut axes = Vec::with_capacity(axis_count);
        for index in 0..axis_count {
            let record = bytes_at(data, axes_at + index * axis_size, 20)?;
            let (min, default, max) = (fixed(record, 4)?, fixed(record, 8)?, fixed(record, 12)?);
            if !(min <= default && default <= max) {
                return Err(FontError::BadTable);
            }
            axes.push(AxisRecord {
                tag: [record[0], record[1], record[2], record[3]],
                min,
                default,
                max,
                hidden: u16_at(record, 16)? & 1 != 0,
                name_id: u16_at(record, 18)?,
            });
        }
        // An instance is two name ids, a value for each axis, and
        // sometimes a third name id. Records of another size are not
        // instances this reader lists.
        let plain = 4 + axis_count * 4;
        let instances = if instance_size == plain || instance_size == plain + 2 {
            let count = instance_count.min(MAX_INSTANCES);
            bytes_at(
                data,
                axes_at + axis_count * axis_size,
                count * instance_size,
            )?
        } else {
            &[]
        };
        Ok(Some(Self {
            axes,
            instances,
            instance_size,
        }))
    }

    pub(crate) fn axis_count(&self) -> usize {
        self.axes.len()
    }

    /// The axes, each named from the face's `name` table.
    pub(crate) fn axes(&self, names: Option<&[u8]>) -> Vec<Axis> {
        self.axes
            .iter()
            .map(|axis| Axis {
                tag: axis.tag,
                min: axis.min,
                default: axis.default,
                max: axis.max,
                hidden: axis.hidden,
                name: names.and_then(|names| crate::faces::name(names, axis.name_id)),
            })
            .collect()
    }

    /// Each instance as the table stores it: the id of its name, the id
    /// of its name as a font of its own when it has one, and a value for
    /// each axis.
    pub(crate) fn records(&self) -> impl Iterator<Item = (u16, Option<u16>, Vec<f32>)> + '_ {
        let size = self.instance_size.max(1);
        let axes = self.axes.len();
        self.instances.chunks_exact(size).filter_map(move |record| {
            let name = u16_at(record, 0).ok()?;
            let coordinates: Vec<f32> = (0..axes)
                .filter_map(|axis| fixed(record, 4 + axis * 4).ok())
                .collect();
            if coordinates.len() != axes {
                return None;
            }
            // 0xFFFF, and for this name 0, is no name.
            let postscript = (size == 6 + axes * 4)
                .then(|| u16_at(record, 4 + axes * 4).ok())
                .flatten()
                .filter(|id| *id != 0 && *id != 0xFFFF);
            Some((name, postscript, coordinates))
        })
    }

    pub(crate) fn instances(&self, names: Option<&[u8]>) -> Vec<Instance> {
        let named = |id: u16| {
            names
                .filter(|_| id != 0xFFFF)
                .and_then(|names| crate::faces::name(names, id))
        };
        self.records()
            .map(|(name, postscript, coordinates)| Instance {
                name: named(name),
                postscript_name: postscript.and_then(named),
                coordinates,
            })
            .collect()
    }

    /// The place along each axis of `settings`: an axis and a value in
    /// the axis's own units. A value past an axis's ends is its end, an
    /// axis not named is at its default, an axis the face lacks is passed
    /// over, and of two settings of one axis the later counts. `avar`,
    /// when the face has it, bends each axis after. The places are worked
    /// out in 16.16 fixed point and only then cut to 2.14, as HarfBuzz
    /// works them out, so each comes to the same number.
    pub(crate) fn normalize(
        &self,
        settings: &[(Tag, f32)],
        avar: Option<&[u8]>,
    ) -> Result<Vec<i16>, FontError> {
        let mut fine = vec![0_i32; self.axes.len()];
        for (tag, value) in settings {
            for (axis, place) in self.axes.iter().zip(&mut fine) {
                if axis.tag == *tag {
                    *place = axis.place(*value);
                }
            }
        }
        if let Some(avar) = avar {
            bend(avar, &mut fine)?;
        }
        Ok(fine.into_iter().map(cut).collect())
    }

    /// The value each axis has under `settings`, in its own units: the
    /// last setting that names it, kept within its ends, and its default
    /// when none does or the value is no number.
    fn values(&self, settings: &[(Tag, f32)]) -> Vec<f32> {
        self.axes
            .iter()
            .map(|axis| {
                settings
                    .iter()
                    .rev()
                    .find(|(tag, _)| *tag == axis.tag)
                    .map(|(_, value)| *value)
                    .filter(|value| value.is_finite())
                    .map_or(axis.default, |value| value.clamp(axis.min, axis.max))
            })
            .collect()
    }
}

/// 1 as a 16.16 fixed-point number.
const FINE_ONE: i32 = 65_536;

impl AxisRecord {
    /// Where `value` is along the axis, in 16.16 fixed point.
    fn place(&self, value: f32) -> i32 {
        if !value.is_finite() {
            return 0;
        }
        let value = value.clamp(self.min, self.max);
        let along = if value < self.default {
            (value - self.default) / (self.default - self.min)
        } else if value > self.default {
            (value - self.default) / (self.max - self.default)
        } else {
            0.0
        };
        #[allow(clippy::cast_possible_truncation)]
        let place = nearest(along * 65_536.0).clamp(-65_536.0, 65_536.0) as i32;
        place
    }
}

/// Bend each of `places`, in 16.16 fixed point, by its axis's segment map
/// in `avar`. A table of the second version then moves each axis by
/// where all of them are: it has a store of deltas like the other
/// tables', a row for each axis, asked where the axes are after their
/// own bending.
fn bend(avar: &[u8], places: &mut [i32]) -> Result<(), FontError> {
    let version = u16_at(avar, 0)?;
    if !(1..=2).contains(&version) {
        return Ok(());
    }
    let axes = usize::from(u16_at(avar, 6)?);
    if axes > MAX_AXES {
        return Err(FontError::BadTable);
    }
    let mut at = 8;
    for axis in 0..axes {
        let count = usize::from(u16_at(avar, at)?);
        let pairs = bytes_at(avar, at + 2, count * 4)?;
        if let Some(place) = places.get_mut(axis) {
            *place = bent(pairs, *place);
        }
        at += 2 + count * 4;
    }
    if version < 2 {
        return Ok(());
    }
    // After the maps: where the map from an axis to its row is, and
    // where the store is. No map gives each axis the row of its number.
    let offset = |at: usize| {
        usize::try_from(u32_at(avar, at)?)
            .map_err(|_| FontError::BadTable)
            .map(|offset| (offset != 0).then_some(offset))
    };
    let Some(store) = offset(at + 4)? else {
        return Ok(());
    };
    let store = Store::read(avar.get(store..).ok_or(FontError::Truncated)?)?;
    let map = match offset(at)? {
        Some(map) => Some(IndexMap::read(
            avar.get(map..).ok_or(FontError::Truncated)?,
        )?),
        None => None,
    };
    // The store is asked in the 2.14 the other tables use; what it gives
    // back is 2.14 as well.
    let coarse: Vec<i16> = places.iter().map(|place| cut(*place)).collect();
    for (axis, place) in places.iter_mut().enumerate() {
        let (outer, inner) = map.as_ref().map_or((0, axis), |map| map.get(axis));
        let delta = store.delta_at(outer, inner, &coarse)?;
        #[allow(clippy::cast_possible_truncation)]
        let moved = nearest(delta * 4.0).clamp(-262_144.0, 262_144.0) as i32;
        *place = (*place + moved).clamp(-FINE_ONE, FINE_ONE);
    }
    Ok(())
}

/// A place in 16.16 fixed point as the 2.14 the tables use.
fn cut(place: i32) -> i16 {
    i16::try_from((place.clamp(-FINE_ONE, FINE_ONE) + 2) >> 2).unwrap_or_default()
}

/// `place` through a map of pairs, each a place and what it becomes, in
/// rising order: straight lines between the pairs. The pairs are 2.14
/// numbers and `place` 16.16. The cases for pairs that repeat a place
/// are the ones HarfBuzz settled on to match what Core Text does with
/// the system's own faces.
fn bent(pairs: &[u8], place: i32) -> i32 {
    let from = |index: usize| {
        let at = index * 4;
        i32::from(i16::from_be_bytes([pairs[at], pairs[at + 1]])) * 4
    };
    let to = |index: usize| {
        let at = index * 4 + 2;
        i32::from(i16::from_be_bytes([pairs[at], pairs[at + 1]])) * 4
    };
    let count = pairs.len() / 4;
    if count == 0 {
        return place;
    }
    if count == 1 {
        return place - from(0) + to(0);
    }
    // A map may say twice where the ends go.
    let (mut start, mut end) = (0, count);
    if from(start) == -FINE_ONE && to(start) == -FINE_ONE && from(start + 1) == -FINE_ONE {
        start += 1;
    }
    if from(end - 1) == FINE_ONE && to(end - 1) == FINE_ONE && from(end - 2) == FINE_ONE {
        end -= 1;
    }
    if let Some(first) = (start..end).find(|index| from(*index) == place) {
        let mut last = first;
        while last + 1 < end && from(last + 1) == place {
            last += 1;
        }
        // One pair for the place is the pair; of three, the middle one;
        // otherwise the one that maps nearer to 0.
        if first + 2 == last {
            return to(first + 1);
        }
        let takes_first =
            first == last || place > 0 || (place == 0 && to(first).abs() < to(last).abs());
        return if takes_first { to(first) } else { to(last) };
    }
    let Some(after) = (start..end).find(|index| place < from(*index)) else {
        return place - from(end - 1) + to(end - 1);
    };
    if after == 0 {
        return place - from(0) + to(0);
    }
    let before = after - 1;
    #[allow(clippy::cast_precision_loss)]
    let (span, rise, along) = (
        (from(after) - from(before)) as f32,
        (to(after) - to(before)) as f32,
        (place - from(before)) as f32,
    );
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let mapped = nearest(to(before) as f32 + rise * along / span) as i32;
    mapped
}

/// The places along a face's axes, carried by value inside the layout
/// tables: at most [`MAX_LAYOUT_AXES`] of them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Places {
    count: u8,
    values: [i16; MAX_LAYOUT_AXES],
}

impl Places {
    /// `places` to carry; `None` when every one is 0, where nothing
    /// varies, or there are more than can be carried.
    pub(crate) fn of(places: &[i16]) -> Option<Self> {
        if places.len() > MAX_LAYOUT_AXES || places.iter().all(|place| *place == 0) {
            return None;
        }
        let mut values = [0; MAX_LAYOUT_AXES];
        values[..places.len()].copy_from_slice(places);
        Some(Self {
            count: u8::try_from(places.len()).ok()?,
            values,
        })
    }

    pub(crate) fn get(&self) -> &[i16] {
        &self.values[..usize::from(self.count)]
    }
}

/// How much a delta drawn for the region from `start` through `peak` to
/// `end` counts at `place`: 1 at the peak, 0 at and past the ends.
fn weight(start: i16, peak: i16, end: i16, place: i16) -> f32 {
    // An axis whose peak is 0 does not weigh on the delta at all.
    if peak == 0 || place == peak {
        return 1.0;
    }
    if place == 0 {
        return 0.0;
    }
    // Nor does a region out of order, or one that crosses 0 with its
    // peak elsewhere.
    if start > peak || peak > end || (start < 0 && end > 0) {
        return 1.0;
    }
    if place <= start || end <= place {
        return 0.0;
    }
    let (start, peak, end, place) = (
        f32::from(start),
        f32::from(peak),
        f32::from(end),
        f32::from(place),
    );
    if place < peak {
        (place - start) / (peak - start)
    } else {
        (end - place) / (end - peak)
    }
}

/// An item variation store: rows of deltas, each row a number for each of
/// the regions its group names.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Store<'a> {
    data: &'a [u8],
    /// The region list: for each region, a start, a peak and an end for
    /// each axis.
    regions: &'a [u8],
    axes: usize,
    groups: usize,
}

impl<'a> Store<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::Unsupported);
        }
        let regions_at = usize::try_from(u32_at(data, 2)?).map_err(|_| FontError::BadTable)?;
        let groups = usize::from(u16_at(data, 6)?);
        let _offsets = bytes_at(data, 8, groups * 4)?;
        let list = data.get(regions_at..).ok_or(FontError::Truncated)?;
        let axes = usize::from(u16_at(list, 0)?);
        let count = usize::from(u16_at(list, 2)?);
        if axes > MAX_AXES {
            return Err(FontError::BadTable);
        }
        Ok(Self {
            data,
            regions: bytes_at(list, 4, count * axes * 6)?,
            axes,
            groups,
        })
    }

    fn region_count(&self) -> usize {
        if self.axes == 0 {
            0
        } else {
            self.regions.len() / (self.axes * 6)
        }
    }

    /// How much region `region` counts at `places`. A region the store
    /// lacks counts nothing.
    fn region_weight(&self, region: usize, places: &[i16]) -> f32 {
        let size = self.axes * 6;
        let Some(record) = region
            .checked_mul(size)
            .and_then(|at| self.regions.get(at..at + size))
        else {
            return 0.0;
        };
        let mut total = 1.0;
        for (axis, triple) in record.as_chunks::<6>().0.iter().enumerate() {
            let read = |at: usize| i16::from_be_bytes([triple[at], triple[at + 1]]);
            let place = places.get(axis).copied().unwrap_or(0);
            let one = weight(read(0), read(2), read(4), place);
            if one == 0.0 {
                return 0.0;
            }
            total *= one;
        }
        total
    }

    /// How much every region counts at `places`, to reuse for many rows.
    pub(crate) fn weights(&self, places: &[i16]) -> Vec<f32> {
        (0..self.region_count())
            .map(|region| self.region_weight(region, places))
            .collect()
    }

    /// Row `inner` of group `outer`, with each of its numbers weighted by
    /// `weigh`, which is given a region's index. A row the store lacks is
    /// no change.
    fn row(
        &self,
        outer: usize,
        inner: usize,
        weigh: impl Fn(usize) -> f32,
    ) -> Result<f32, FontError> {
        if outer >= self.groups {
            return Ok(0.0);
        }
        let at =
            usize::try_from(u32_at(self.data, 8 + outer * 4)?).map_err(|_| FontError::BadTable)?;
        let group = self.data.get(at..).ok_or(FontError::Truncated)?;
        let rows = usize::from(u16_at(group, 0)?);
        let words = u16_at(group, 2)?;
        let columns = usize::from(u16_at(group, 4)?);
        if inner >= rows {
            return Ok(0.0);
        }
        // The first numbers of a row are words and the rest bytes, or,
        // in a group of long numbers, 32 bits and then words.
        let long = words & 0x8000 != 0;
        let wide = usize::from(words & 0x7FFF);
        if wide > columns {
            return Err(FontError::BadTable);
        }
        let (big, small) = if long { (4, 2) } else { (2, 1) };
        let row_size = wide * big + (columns - wide) * small;
        let regions = bytes_at(group, 6, columns * 2)?;
        let row = bytes_at(group, 6 + columns * 2 + inner * row_size, row_size)?;
        let mut total = 0.0;
        let mut at = 0;
        for (column, region) in regions.as_chunks::<2>().0.iter().enumerate() {
            let size = if column < wide { big } else { small };
            let bytes = &row[at..at + size];
            at += size;
            let delta = match size {
                1 => i32::from(i8::from_be_bytes([bytes[0]])),
                2 => i32::from(i16::from_be_bytes([bytes[0], bytes[1]])),
                _ => i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            };
            if delta == 0 {
                continue;
            }
            let weight = weigh(usize::from(u16::from_be_bytes(*region)));
            if weight != 0.0 {
                #[allow(clippy::cast_precision_loss)]
                let delta = delta as f32;
                total += delta * weight;
            }
        }
        Ok(total)
    }

    /// The change row `inner` of group `outer` makes, with `weights` from
    /// [`Self::weights`].
    pub(crate) fn delta(
        &self,
        outer: usize,
        inner: usize,
        weights: &[f32],
    ) -> Result<f32, FontError> {
        self.row(outer, inner, |region| {
            weights.get(region).copied().unwrap_or(0.0)
        })
    }

    /// The change row `inner` of group `outer` makes at `places`, for a
    /// caller that asks for few rows.
    pub(crate) fn delta_at(
        &self,
        outer: usize,
        inner: usize,
        places: &[i16],
    ) -> Result<f32, FontError> {
        self.row(outer, inner, |region| self.region_weight(region, places))
    }

    /// How much each region group `outer` has numbers for counts at
    /// `places`, in the order the group lists them: what a CFF2 glyph's
    /// blends are weighted by. A group of more than `most` regions is
    /// refused, and a group the store lacks has none.
    pub(crate) fn group_weights(
        &self,
        outer: usize,
        places: &[i16],
        most: usize,
    ) -> Result<Vec<f32>, FontError> {
        if outer >= self.groups {
            return Ok(Vec::new());
        }
        let at =
            usize::try_from(u32_at(self.data, 8 + outer * 4)?).map_err(|_| FontError::BadTable)?;
        let group = self.data.get(at..).ok_or(FontError::Truncated)?;
        let columns = usize::from(u16_at(group, 4)?);
        if columns > most {
            return Err(FontError::BadTable);
        }
        Ok(bytes_at(group, 6, columns * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|region| self.region_weight(usize::from(u16::from_be_bytes(*region)), places))
            .collect())
    }
}

/// A map from a glyph, or another index, to a row of a [`Store`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexMap<'a> {
    entries: &'a [u8],
    entry_size: usize,
    inner_bits: u32,
}

impl<'a> IndexMap<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        let format = *data.first().ok_or(FontError::Truncated)?;
        let entry_format = *data.get(1).ok_or(FontError::Truncated)?;
        let (count, at) = match format {
            0 => (usize::from(u16_at(data, 2)?), 4),
            1 => (
                usize::try_from(u32_at(data, 2)?).map_err(|_| FontError::BadTable)?,
                6,
            ),
            _ => return Err(FontError::Unsupported),
        };
        let entry_size = usize::from((entry_format >> 4) & 3) + 1;
        let bytes = count.checked_mul(entry_size).ok_or(FontError::BadTable)?;
        Ok(Self {
            entries: bytes_at(data, at, bytes)?,
            entry_size,
            inner_bits: u32::from(entry_format & 0x0F) + 1,
        })
    }

    /// The group and row for `index`. An index past the end of the map
    /// has the last entry's, and an empty map maps an index to itself.
    pub(crate) fn get(&self, index: usize) -> (usize, usize) {
        let count = self.entries.len() / self.entry_size;
        if count == 0 {
            return (index >> 16, index & 0xFFFF);
        }
        let at = index.min(count - 1) * self.entry_size;
        let mut entry = 0_u32;
        for byte in &self.entries[at..at + self.entry_size] {
            entry = entry << 8 | u32::from(*byte);
        }
        let outer = entry >> self.inner_bits;
        let inner = entry & ((1_u32 << self.inner_bits) - 1);
        (outer as usize, inner as usize)
    }
}

/// `HVAR`: how each glyph's advance changes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hvar<'a> {
    store: Store<'a>,
    advances: Option<IndexMap<'a>>,
}

impl<'a> Hvar<'a> {
    pub(crate) fn read(data: &'a [u8]) -> Result<Self, FontError> {
        if u16_at(data, 0)? != 1 {
            return Err(FontError::Unsupported);
        }
        let offset = |at: usize| {
            usize::try_from(u32_at(data, at)?)
                .map_err(|_| FontError::BadTable)
                .map(|offset| (offset != 0).then_some(offset))
        };
        let store = offset(4)?.ok_or(FontError::BadTable)?;
        let store = Store::read(data.get(store..).ok_or(FontError::Truncated)?)?;
        let advances = match offset(8)? {
            Some(at) => Some(IndexMap::read(data.get(at..).ok_or(FontError::Truncated)?)?),
            None => None,
        };
        Ok(Self { store, advances })
    }

    pub(crate) fn weights(&self, places: &[i16]) -> Vec<f32> {
        self.store.weights(places)
    }

    /// How much wider `glyph` is, with `weights` from [`Self::weights`].
    pub(crate) fn advance_delta(&self, glyph: u16, weights: &[f32]) -> Result<f32, FontError> {
        let (outer, inner) = match &self.advances {
            Some(map) => map.get(usize::from(glyph)),
            None => (0, usize::from(glyph)),
        };
        self.store.delta(outer, inner, weights)
    }
}

/// The deltas of a layout table's values: the store `GDEF` holds and the
/// places they are asked at.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Deltas<'a> {
    store: Store<'a>,
    places: Places,
}

impl<'a> Deltas<'a> {
    pub(crate) fn new(store: Store<'a>, places: &[i16]) -> Option<Self> {
        Some(Self {
            store,
            places: Places::of(places)?,
        })
    }

    /// What the table at `table`, when it is one that names a row of the
    /// store and not a list of pixel sizes, adds to a value: rounded, as
    /// the values it adds to are whole.
    pub(crate) fn of(&self, table: &[u8]) -> Result<i32, FontError> {
        if u16_at(table, 4)? != 0x8000 {
            return Ok(0);
        }
        let (outer, inner) = (u16_at(table, 0)?, u16_at(table, 2)?);
        let delta =
            self.store
                .delta_at(usize::from(outer), usize::from(inner), self.places.get())?;
        #[allow(clippy::cast_possible_truncation)]
        let whole = nearest(delta).clamp(-65_536.0, 65_536.0) as i32;
        Ok(whole)
    }
}

/// `value` moved by `delta`, kept inside what 16 bits hold.
pub(crate) fn moved(value: i16, delta: i32) -> i16 {
    i16::try_from((i32::from(value) + delta).clamp(i32::from(i16::MIN), i32::from(i16::MAX)))
        .unwrap_or(value)
}

/// What a face varies along, the tables that say what changes, and where
/// along its axes the face is set.
#[derive(Clone, Debug)]
pub(crate) struct Variation<'a> {
    pub(crate) fvar: Fvar<'a>,
    names: Option<&'a [u8]>,
    avar: Option<&'a [u8]>,
    pub(crate) gvar: Option<crate::gvar::Gvar<'a>>,
    pub(crate) hvar: Option<Hvar<'a>>,
    /// The store of deltas the layout tables' values name rows of.
    pub(crate) layout: Option<Store<'a>>,
    /// The place along each axis: every one 0 as the face is read.
    pub(crate) places: Vec<i16>,
    /// The value each axis is set to, in the axis's own units; none
    /// while every axis is at its default.
    values: Vec<f32>,
    /// Whether any place is not 0.
    pub(crate) set: bool,
    /// How much each shared peak of `gvar`, and each region of `HVAR`,
    /// counts at `places`.
    pub(crate) gvar_weights: Vec<f32>,
    pub(crate) hvar_weights: Vec<f32>,
    /// The fingerprint of the face at its default.
    plain: u64,
}

impl<'a> Variation<'a> {
    /// What the face of `directory` varies along. A face with no `fvar`
    /// table, or one that does not read, does not vary; a table of moves
    /// that does not read moves nothing.
    pub(crate) fn read(directory: &crate::Directory<'a>, fingerprint: u64) -> Option<Self> {
        let fvar = Fvar::read(directory.table(b"fvar").ok()?).ok()??;
        let axes = fvar.axis_count();
        Some(Self {
            names: directory.table(b"name").ok(),
            avar: directory.table(b"avar").ok(),
            gvar: directory
                .table(b"gvar")
                .ok()
                .and_then(|table| crate::gvar::Gvar::read(table, axes).ok()),
            hvar: directory
                .table(b"HVAR")
                .ok()
                .and_then(|table| Hvar::read(table).ok()),
            layout: directory.table(b"GDEF").ok().and_then(layout_store),
            places: vec![0; axes],
            values: Vec::new(),
            set: false,
            gvar_weights: Vec::new(),
            hvar_weights: Vec::new(),
            plain: fingerprint,
            fvar,
        })
    }

    pub(crate) fn axes(&self) -> Vec<Axis> {
        self.fvar.axes(self.names)
    }

    pub(crate) fn instances(&self) -> Vec<Instance> {
        self.fvar.instances(self.names)
    }

    /// Set the face to `settings`, and give its fingerprint there.
    pub(crate) fn set(&mut self, settings: &[(Tag, f32)]) -> Result<u64, FontError> {
        let places = self.fvar.normalize(settings, self.avar)?;
        self.values = self.fvar.values(settings);
        self.set = places.iter().any(|place| *place != 0);
        self.gvar_weights = match (&self.gvar, self.set) {
            (Some(gvar), true) => gvar.weights(&places),
            _ => Vec::new(),
        };
        self.hvar_weights = match (&self.hvar, self.set) {
            (Some(hvar), true) => hvar.weights(&places),
            _ => Vec::new(),
        };
        self.places = places;
        Ok(self.fingerprint())
    }

    /// Each axis's tag and the value it is set to, in the axis's own
    /// units and the face's order.
    pub(crate) fn settings(&self) -> Vec<(Tag, f32)> {
        let defaults;
        let values = if self.values.is_empty() {
            defaults = self.fvar.values(&[]);
            &defaults
        } else {
            &self.values
        };
        self.fvar
            .axes
            .iter()
            .zip(values)
            .map(|(axis, value)| (axis.tag, *value))
            .collect()
    }

    /// The deltas of the layout tables' values where the face is set;
    /// none at its default.
    pub(crate) fn layout_deltas(&self) -> Option<Deltas<'a>> {
        Deltas::new(self.layout?, &self.places)
    }

    /// A number for the face where it is set: the face's own at its
    /// default, and another for every other setting.
    fn fingerprint(&self) -> u64 {
        if !self.set {
            return self.plain;
        }
        // FNV-1a, 64 bits, over the face's own and each place.
        let mut hash = 0xCBF2_9CE4_8422_2325_u64;
        let mut mix = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
            }
        };
        mix(&self.plain.to_be_bytes());
        for place in &self.places {
            mix(&place.to_be_bytes());
        }
        hash
    }
}

/// The store of deltas in a `GDEF` table, from version 1.3 on.
fn layout_store(gdef: &[u8]) -> Option<Store<'_>> {
    if u16_at(gdef, 0).ok()? != 1 || u16_at(gdef, 2).ok()? < 3 {
        return None;
    }
    let at = usize::try_from(u32_at(gdef, 14).ok()?).ok()?;
    if at == 0 {
        return None;
    }
    Store::read(gdef.get(at..)?).ok()
}
