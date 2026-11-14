//! Small fonts built in code, for tests.
//!
//! A table is a list of 16-bit values and of child tables; each child is
//! written after the values and its place stored where it was listed, the
//! way every OpenType layout table points at its parts.

/// One 16-bit slot of a table.
pub(crate) enum Part {
    /// A value.
    V(u16),
    /// A child table, stored as its offset from this table's start.
    Off(Vec<u8>),
    /// The offset of a table that is not there.
    Null,
}

pub(crate) use Part::{Null, Off, V};

/// The table `parts` describe.
pub(crate) fn table(parts: Vec<Part>) -> Vec<u8> {
    let mut at = parts.len() * 2;
    let mut head = Vec::new();
    let mut tail = Vec::new();
    for part in parts {
        match part {
            V(value) => head.extend(value.to_be_bytes()),
            Null => head.extend([0, 0]),
            Off(child) => {
                head.extend(u16::try_from(at).expect("offset").to_be_bytes());
                at += child.len();
                tail.extend(child);
            }
        }
    }
    head.extend(tail);
    head
}

/// `values`, big endian.
pub(crate) fn be(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

fn count(length: usize) -> u16 {
    u16::try_from(length).expect("count")
}

/// A signed value as the 16 bits a table stores.
pub(crate) fn signed(value: i16) -> u16 {
    u16::from_be_bytes(value.to_be_bytes())
}

/// A tag as two values.
fn tag(tag: &[u8; 4]) -> [Part; 2] {
    [
        V(u16::from_be_bytes([tag[0], tag[1]])),
        V(u16::from_be_bytes([tag[2], tag[3]])),
    ]
}

/// A coverage table of `glyphs`, which are in order.
pub(crate) fn coverage(glyphs: &[u16]) -> Vec<u8> {
    let mut values = vec![1, count(glyphs.len())];
    values.extend(glyphs);
    be(&values)
}

/// A coverage table of ranges: each a first glyph and a last.
pub(crate) fn coverage_ranges(ranges: &[(u16, u16)]) -> Vec<u8> {
    let mut values = vec![2, count(ranges.len())];
    let mut index = 0;
    for (first, last) in ranges {
        values.extend([*first, *last, index]);
        index += last - first + 1;
    }
    be(&values)
}

/// A class table of ranges: each a first glyph, a last, and their class.
pub(crate) fn classes(ranges: &[(u16, u16, u16)]) -> Vec<u8> {
    let mut values = vec![2, count(ranges.len())];
    for (first, last, class) in ranges {
        values.extend([*first, *last, *class]);
    }
    be(&values)
}

/// One lookup: its type, its flag, and its subtables.
pub(crate) struct LookupSpec {
    pub(crate) kind: u16,
    pub(crate) flag: u16,
    pub(crate) mark_set: Option<u16>,
    pub(crate) subtables: Vec<Vec<u8>>,
}

impl LookupSpec {
    pub(crate) fn new(kind: u16, subtables: Vec<Vec<u8>>) -> Self {
        Self {
            kind,
            flag: 0,
            mark_set: None,
            subtables,
        }
    }

    pub(crate) fn flag(mut self, flag: u16) -> Self {
        self.flag = flag;
        self
    }
}

/// A `GSUB` or `GPOS` table. Each script has one default language that
/// turns on the features listed for it, by index.
pub(crate) fn layout_table(
    scripts: &[(&[u8; 4], &[u16])],
    features: &[(&[u8; 4], &[u16])],
    lookups: Vec<LookupSpec>,
) -> Vec<u8> {
    let mut script_parts = vec![V(count(scripts.len()))];
    for (name, on) in scripts {
        let mut system = vec![0, 0xFFFF, count(on.len())];
        system.extend(*on);
        script_parts.extend(tag(name));
        script_parts.push(Off(table(vec![Off(be(&system)), V(0)])));
    }
    let mut feature_parts = vec![V(count(features.len()))];
    for (name, runs) in features {
        let mut feature = vec![0, count(runs.len())];
        feature.extend(*runs);
        feature_parts.extend(tag(name));
        feature_parts.push(Off(be(&feature)));
    }
    let mut lookup_parts = vec![V(count(lookups.len()))];
    for lookup in lookups {
        let flag = lookup.flag | if lookup.mark_set.is_some() { 0x10 } else { 0 };
        let mut parts = vec![V(lookup.kind), V(flag), V(count(lookup.subtables.len()))];
        parts.extend(lookup.subtables.into_iter().map(Off));
        parts.extend(lookup.mark_set.map(V));
        lookup_parts.push(Off(table(parts)));
    }
    table(vec![
        V(1),
        V(0),
        Off(table(script_parts)),
        Off(table(feature_parts)),
        Off(table(lookup_parts)),
    ])
}

/// A `GDEF` table of glyph classes and mark attachment classes, each as
/// ranges, and mark sets, each a list of glyphs in order.
pub(crate) fn gdef(
    glyph_classes: &[(u16, u16, u16)],
    mark_classes: &[(u16, u16, u16)],
    mark_sets: &[&[u16]],
) -> Vec<u8> {
    let class_table = |ranges: &[(u16, u16, u16)]| {
        if ranges.is_empty() {
            Null
        } else {
            Off(classes(ranges))
        }
    };
    // The mark sets are found by 32-bit offsets, so they are written by hand.
    let mut sets = be(&[1, count(mark_sets.len())]);
    let mut at = 4 + mark_sets.len() * 4;
    let mut tail = Vec::new();
    for set in mark_sets {
        sets.extend(u32::try_from(at).expect("offset").to_be_bytes());
        let coverage = coverage(set);
        at += coverage.len();
        tail.extend(coverage);
    }
    sets.extend(tail);
    table(vec![
        V(1),
        V(2),
        class_table(glyph_classes),
        Null,
        Null,
        class_table(mark_classes),
        if mark_sets.is_empty() {
            Null
        } else {
            Off(sets)
        },
    ])
}

/// A single substitution listing a substitute for each covered glyph.
pub(crate) fn single(pairs: &[(u16, u16)]) -> Vec<u8> {
    let covered: Vec<u16> = pairs.iter().map(|pair| pair.0).collect();
    let mut parts = vec![V(2), Off(coverage(&covered)), V(count(pairs.len()))];
    parts.extend(pairs.iter().map(|pair| V(pair.1)));
    table(parts)
}

/// A multiple substitution, or with `kind` 3 the alternates of each glyph.
pub(crate) fn sequences(lists: &[(u16, &[u16])]) -> Vec<u8> {
    let covered: Vec<u16> = lists.iter().map(|list| list.0).collect();
    let mut parts = vec![V(1), Off(coverage(&covered)), V(count(lists.len()))];
    for (_, glyphs) in lists {
        let mut values = vec![count(glyphs.len())];
        values.extend(*glyphs);
        parts.push(Off(be(&values)));
    }
    table(parts)
}

/// The glyphs after a ligature's first, and the glyph they make with it.
pub(crate) type Made<'a> = (&'a [u16], u16);

/// Ligatures: for each first glyph, the ligatures it starts, each the
/// glyphs after the first and the glyph they make.
pub(crate) fn ligatures(sets: &[(u16, &[Made<'_>])]) -> Vec<u8> {
    let covered: Vec<u16> = sets.iter().map(|set| set.0).collect();
    let mut parts = vec![V(1), Off(coverage(&covered)), V(count(sets.len()))];
    for (_, made) in sets {
        let mut set = vec![V(count(made.len()))];
        for (rest, glyph) in *made {
            let mut values = vec![*glyph, count(rest.len() + 1)];
            values.extend(*rest);
            set.push(Off(be(&values)));
        }
        parts.push(Off(table(set)));
    }
    table(parts)
}

/// A chained context by coverage: the glyphs before, nearest first, the
/// input, the glyphs after, and the lookups to run at places in the input.
pub(crate) fn chain(
    before: &[&[u16]],
    input: &[&[u16]],
    after: &[&[u16]],
    records: &[(u16, u16)],
) -> Vec<u8> {
    let mut parts = vec![V(3)];
    for group in [before, input, after] {
        parts.push(V(count(group.len())));
        parts.extend(group.iter().map(|glyphs| Off(coverage(glyphs))));
    }
    parts.push(V(count(records.len())));
    for (place, lookup) in records {
        parts.extend([V(*place), V(*lookup)]);
    }
    table(parts)
}

/// An anchor at a point.
pub(crate) fn anchor(x: i16, y: i16) -> Vec<u8> {
    be(&[1, signed(x), signed(y)])
}

/// Kerning by pairs: for each first glyph, its second glyphs in order and
/// how much the first's advance changes.
pub(crate) fn pairs(sets: &[(u16, &[(u16, i16)])]) -> Vec<u8> {
    let covered: Vec<u16> = sets.iter().map(|set| set.0).collect();
    let mut parts = vec![
        V(1),
        Off(coverage(&covered)),
        V(4),
        V(0),
        V(count(sets.len())),
    ];
    for (_, seconds) in sets {
        let mut values = vec![count(seconds.len())];
        for (second, advance) in *seconds {
            values.extend([*second, signed(*advance)]);
        }
        parts.push(Off(be(&values)));
    }
    table(parts)
}

/// Marks set on bases, or on other marks: each mark with its class and
/// anchor, and each base with an anchor for each class.
pub(crate) fn mark_attach(
    marks: &[(u16, u16, (i16, i16))],
    bases: &[(u16, &[(i16, i16)])],
) -> Vec<u8> {
    let class_count = bases.first().map_or(0, |base| base.1.len());
    let mut mark_array = vec![V(count(marks.len()))];
    for (_, class, at) in marks {
        mark_array.extend([V(*class), Off(anchor(at.0, at.1))]);
    }
    let mut base_array = vec![V(count(bases.len()))];
    for (_, anchors) in bases {
        base_array.extend(anchors.iter().map(|at| Off(anchor(at.0, at.1))));
    }
    let mark_glyphs: Vec<u16> = marks.iter().map(|mark| mark.0).collect();
    let base_glyphs: Vec<u16> = bases.iter().map(|base| base.0).collect();
    table(vec![
        V(1),
        Off(coverage(&mark_glyphs)),
        Off(coverage(&base_glyphs)),
        V(count(class_count)),
        Off(table(mark_array)),
        Off(table(base_array)),
    ])
}

/// One glyph of a built font: its advance and, when it draws, the corners
/// of its box.
#[derive(Clone, Copy)]
pub(crate) struct Glyph {
    pub(crate) advance: u16,
    pub(crate) outline: Option<[i16; 4]>,
}

/// A box glyph `advance` wide.
pub(crate) fn boxed(advance: u16) -> Glyph {
    let right = i16::try_from(advance).expect("advance") - 50;
    Glyph {
        advance,
        outline: Some([50, 0, right, 700]),
    }
}

/// A character, the variation selector after it, and the glyph the two
/// draw: `None` when they draw as the character alone does.
pub(crate) type Varied = (char, char, Option<u16>);

/// A `cmap` table of `map`, in character order, and of `varied`. One group
/// for each character: format 12 holds every plane. The pairs with a
/// selector, when there are any, go in a format 14 subtable before it.
pub(crate) fn character_map(map: &[(char, u16)], varied: &[Varied]) -> Vec<u8> {
    let mut groups = be(&[12, 0]);
    let count = u32::try_from(map.len()).expect("groups");
    groups.extend((16 + count * 12).to_be_bytes());
    groups.extend(0_u32.to_be_bytes());
    groups.extend(count.to_be_bytes());
    for (character, glyph) in map {
        groups.extend(u32::from(*character).to_be_bytes());
        groups.extend(u32::from(*character).to_be_bytes());
        groups.extend(u32::from(*glyph).to_be_bytes());
    }
    if varied.is_empty() {
        let mut cmap = be(&[0, 1, 3, 10, 0, 12]);
        cmap.extend(groups);
        return cmap;
    }
    let three = |value: u32| value.to_be_bytes()[1..].to_vec();
    let mut selectors: Vec<char> = varied.iter().map(|(_, selector, _)| *selector).collect();
    selectors.sort_unstable();
    selectors.dedup();
    let mut records = Vec::new();
    let mut lists = Vec::new();
    let lists_at = 10 + selectors.len() * 11;
    for selector in &selectors {
        let mut of: Vec<&Varied> = varied
            .iter()
            .filter(|(_, with, _)| with == selector)
            .collect();
        of.sort_by_key(|(character, _, _)| *character);
        let plain: Vec<u32> = of
            .iter()
            .filter(|(_, _, glyph)| glyph.is_none())
            .map(|(character, _, _)| u32::from(*character))
            .collect();
        let own: Vec<(u32, u16)> = of
            .iter()
            .filter_map(|(character, _, glyph)| Some((u32::from(*character), (*glyph)?)))
            .collect();
        records.extend(three(u32::from(*selector)));
        let mut place = |empty: bool, list: Vec<u8>| {
            let at = if empty { 0 } else { lists_at + lists.len() };
            records.extend(u32::try_from(at).expect("offset").to_be_bytes());
            if !empty {
                lists.extend(list);
            }
        };
        let mut ranges = u32::try_from(plain.len())
            .expect("ranges")
            .to_be_bytes()
            .to_vec();
        for character in &plain {
            ranges.extend(three(*character));
            ranges.push(0);
        }
        place(plain.is_empty(), ranges);
        let mut mappings = u32::try_from(own.len())
            .expect("mappings")
            .to_be_bytes()
            .to_vec();
        for (character, glyph) in &own {
            mappings.extend(three(*character));
            mappings.extend(glyph.to_be_bytes());
        }
        place(own.is_empty(), mappings);
    }
    let mut variants = be(&[14]);
    variants.extend(
        u32::try_from(lists_at + lists.len())
            .expect("length")
            .to_be_bytes(),
    );
    variants.extend(
        u32::try_from(selectors.len())
            .expect("selectors")
            .to_be_bytes(),
    );
    variants.extend(records);
    variants.extend(lists);
    // Two records: the selectors under platform 0, encoding 5, and then
    // the characters.
    let mut cmap = be(&[0, 2, 0, 5]);
    cmap.extend(20_u32.to_be_bytes());
    cmap.extend(be(&[3, 10]));
    cmap.extend(
        u32::try_from(20 + variants.len())
            .expect("offset")
            .to_be_bytes(),
    );
    cmap.extend(variants);
    cmap.extend(groups);
    cmap
}

/// A TrueType font, 1000 units to the em, of `glyphs` (the first is the
/// missing glyph), a character map in character order, and extra tables.
pub(crate) fn font(
    glyphs: &[Glyph],
    map: &[(char, u16)],
    extra: &[(&[u8; 4], Vec<u8>)],
) -> Vec<u8> {
    let total = count(glyphs.len());
    let mut head = vec![0_u8; 54];
    head[..4].copy_from_slice(&[0, 1, 0, 0]);
    head[12..16].copy_from_slice(&0x5F0F_3CF5_u32.to_be_bytes());
    head[18..20].copy_from_slice(&1_000_u16.to_be_bytes());
    head[50..52].copy_from_slice(&1_u16.to_be_bytes());
    let mut hhea = vec![0_u8; 36];
    hhea[..4].copy_from_slice(&[0, 1, 0, 0]);
    hhea[4..6].copy_from_slice(&800_u16.to_be_bytes());
    hhea[6..8].copy_from_slice(&signed(-200).to_be_bytes());
    hhea[34..36].copy_from_slice(&total.to_be_bytes());
    let mut maxp = vec![0_u8; 32];
    maxp[..4].copy_from_slice(&[0, 1, 0, 0]);
    maxp[4..6].copy_from_slice(&total.to_be_bytes());
    let mut hmtx = Vec::new();
    let mut glyf = Vec::new();
    let mut loca = Vec::new();
    for glyph in glyphs {
        hmtx.extend(glyph.advance.to_be_bytes());
        // The left side bearing is where the outline's box starts, as a
        // font tool writes it.
        hmtx.extend(signed(glyph.outline.map_or(0, |[left, ..]| left)).to_be_bytes());
        loca.extend(u32::try_from(glyf.len()).expect("loca").to_be_bytes());
        if let Some([left, bottom, right, top]) = glyph.outline {
            // One contour of four corners, each on the curve, as deltas.
            glyf.extend(be(&[
                1,
                signed(left),
                signed(bottom),
                signed(right),
                signed(top),
            ]));
            glyf.extend(be(&[3, 0]));
            glyf.extend([1, 1, 1, 1]);
            let (width, height) = (right - left, top - bottom);
            glyf.extend(be(&[signed(left), signed(width), 0, signed(-width)]));
            glyf.extend(be(&[signed(bottom), 0, signed(height), 0]));
        }
    }
    loca.extend(u32::try_from(glyf.len()).expect("loca").to_be_bytes());
    let cmap = character_map(map, &[]);
    let mut tables: Vec<([u8; 4], Vec<u8>)> = vec![
        (*b"cmap", cmap),
        (*b"glyf", glyf),
        (*b"head", head),
        (*b"hhea", hhea),
        (*b"hmtx", hmtx),
        (*b"loca", loca),
        (*b"maxp", maxp),
    ];
    // An extra table of a name above takes that table's place.
    tables.retain(|(name, _)| !extra.iter().any(|(extra, _)| *extra == name));
    tables.extend(extra.iter().map(|(name, bytes)| (**name, bytes.clone())));
    tables.sort_by_key(|table| table.0);
    let mut file = vec![0, 1, 0, 0];
    file.extend(be(&[count(tables.len()), 0, 0, 0]));
    let mut at = 12 + tables.len() * 16;
    let mut body = Vec::new();
    for (name, bytes) in &tables {
        file.extend(name);
        file.extend(0_u32.to_be_bytes());
        file.extend(u32::try_from(at).expect("offset").to_be_bytes());
        file.extend(u32::try_from(bytes.len()).expect("length").to_be_bytes());
        body.extend(bytes);
        // Tables start on four-byte boundaries.
        let padded = bytes.len().next_multiple_of(4);
        body.resize(body.len() + padded - bytes.len(), 0);
        at += padded;
    }
    file.extend(body);
    file
}
