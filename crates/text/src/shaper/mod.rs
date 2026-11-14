//! Shapes one run of text in one face: its characters become the glyphs
//! the font draws them with, each with its advance and its place.
//!
//! A run is text of one script and one direction. Its characters are
//! mapped to glyphs, taking an accented letter apart or putting it
//! together as the face needs; the face's substitutions make ligatures
//! and pick the forms of joined letters; its positioning kerns pairs and
//! sets marks on their letters. The steps and their order follow the
//! OpenType shaping model, so text set in the same face reads the same
//! here as elsewhere.

mod apply;
mod arabic;
pub(crate) mod buffer;
mod fallback;
mod hangul;
mod indic;
mod kern;
mod morx;
mod thai;

use cutlass_font::morx::DELETED_GLYPH;
use cutlass_font::{Font, GlyphClass, KernPass, LayoutTable, Tag, Variant};

use self::apply::{Engine, LookupRef, settle};
use self::buffer::{Buffer, Glyph, Space};
use crate::TextError;
use crate::script::{Script, Shaping};
use crate::unicode::{compose, decompose, gc, mirror, props};

/// The mask bit every glyph has.
pub(super) const EVERY: u32 = 1;
/// A glyph read right to left whose character has no mirror image the
/// face can draw: the face's own mirroring feature applies to it.
const MIRRORED: u32 = 1 << 1;
/// The most marks after one letter that are put in order.
const MAX_MARKS: usize = 32;

/// One run to shape.
pub(crate) struct Run<'t> {
    /// The whole string; the run is bytes `start..end` of it.
    pub(crate) text: &'t str,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) script: Script,
    /// The run is read right to left.
    pub(crate) backward: bool,
    /// Ligatures a reader can do without are made.
    pub(crate) ligatures: bool,
}

/// A feature to apply, the glyphs it applies to, and how it treats
/// joiners.
#[derive(Clone, Copy)]
pub(super) struct Feature {
    tag: Tag,
    mask: u32,
    /// The script's own shaping decides what a zero width joiner does.
    manual_joiners: bool,
}

impl Feature {
    pub(super) const fn every(tag: &[u8; 4]) -> Self {
        Self {
            tag: *tag,
            mask: EVERY,
            manual_joiners: false,
        }
    }

    pub(super) const fn masked(tag: &[u8; 4], mask: u32) -> Self {
        Self {
            tag: *tag,
            mask,
            manual_joiners: false,
        }
    }

    pub(super) const fn manual(mut self) -> Self {
        self.manual_joiners = true;
        self
    }
}

/// Shape `run` in `font`. The glyphs come back in the order the text is
/// stored, whichever way the run is read.
pub(crate) fn shape_run(font: &Font<'_>, run: &Run<'_>) -> Result<Vec<Glyph>, TextError> {
    let part = run.text.get(run.start..run.end).ok_or(TextError::Limit)?;
    let mut glyphs: Vec<Glyph> = part
        .char_indices()
        .map(|(at, character)| described(character, run.start + at))
        .collect();
    if glyphs.is_empty() {
        return Ok(glyphs);
    }
    // What a script needs before its face's tables see it. A face that
    // shapes with its own machines is given the text as it is.
    let own_machines = font.morx().is_some();
    let thai = matches!(&run.script.0, b"Thai" | b"Laoo");
    let korean = run.script == Script(*b"Hang");
    if thai && !own_machines {
        glyphs = thai::split(glyphs, described);
    }
    if korean && !own_machines {
        glyphs = hangul::compose(font, glyphs, described);
    }
    mark_continuations(&mut glyphs);
    // A script is shaped in the direction it is written. A run going the
    // other way is turned around first and turned back at the end. A run
    // of digits alone reads left to right whatever its script.
    let digits = glyphs
        .iter()
        .any(|glyph| glyph.category == gc::DECIMAL_NUMBER)
        && !glyphs.iter().any(|glyph| (1..=5).contains(&glyph.category));
    let native = run.script.right_to_left() && !digits;
    let turned = native != run.backward;
    if turned {
        reverse_graphemes(&mut glyphs);
    }
    let backward = native;
    if run.backward {
        for glyph in &mut glyphs {
            match mirror(glyph.character).filter(|other| has(font, *other)) {
                Some(other) => glyph.character = other,
                None => glyph.mask |= MIRRORED,
            }
        }
    }
    let shaping = run.script.shaping();
    let scripts = run.script.tags();
    // A face places Hebrew points only when its positioning table knows
    // Hebrew; one made for another script would place them wrongly.
    let placing = font.gpos().filter(|table| {
        run.script != Script(*b"Hebr")
            || table
                .language(&scripts, None)
                .is_ok_and(|language| language.script == Some(*b"hebr"))
    });
    // An old Hebrew face draws a letter with its point as one glyph.
    let pointed = run.script == Script(*b"Hebr")
        && !placing.is_some_and(|table| has_feature(table, &scripts, *b"mark"));
    // A face made for Apple's layout engine makes its ligatures, forms and
    // emoji sequences with its own machines, and those are used ahead of
    // any OpenType substitutions it also has: they are what it was drawn
    // for.
    let machines = font.morx().is_some();
    // Its extended kerning likewise places it, unless the face is shaped
    // by both OpenType tables.
    let kerx = font.kerx_passes();
    let placing = placing.filter(|_| kerx.is_empty() || (font.gsub().is_some() && !machines));
    // A script written by syllable is put in order and given its forms
    // here, unless the face does that with its own machines: those are
    // given the text as it is.
    let by_syllable = shaping == Shaping::Syllabic;
    let syllabic = by_syllable && !machines;
    let join = |first: char, second: char| {
        if syllabic {
            return indic::compose(first, second);
        }
        compose(first, second).or_else(|| pointed.then(|| hebrew_form(first, second)).flatten())
    };
    let normal = Normal {
        // These scripts take a letter apart even when the face has it
        // whole, so its parts can take their own forms.
        whole_first: !syllabic,
        keeps_whole: &|character| syllabic && indic::keeps_whole(character),
        join: &join,
    };
    let mut glyphs = normalize(font, glyphs, &normal)?;
    if syllabic {
        indic::describe_all(&mut glyphs, run.script);
    }
    merge_graphemes(&mut glyphs);
    let trusted = classify(font, &mut glyphs, syllabic);
    for glyph in &mut glyphs {
        glyph.mask |= EVERY;
    }
    if shaping == Shaping::Joined {
        arabic::join(&mut glyphs);
    }
    let count = glyphs.len();
    if machines {
        glyphs = morx::apply(font, glyphs, backward)?;
        // What the machines deleted leaves the run before a positioning
        // table places it. The face's own kerning still reads those
        // glyphs: its machines know them, and they go at the end.
        if placing.is_some() {
            remove(&mut glyphs, |glyph| glyph.id == DELETED_GLYPH);
        }
        classify(font, &mut glyphs, true);
    }
    let mut buffer = Buffer::new(glyphs);
    if machines {
    } else if syllabic {
        let engine = Engine::new(font, false, backward, count, trusted);
        indic::substitute(font, engine, &mut buffer, &scripts, run.script)?;
    } else if let Some(mut engine) = Engine::new(font, false, backward, count, trusted)
        && let Some(table) = font.gsub()
    {
        let mut stages = substitution_stages(shaping, backward, run.ligatures);
        if korean {
            hangul::stages(&mut stages);
        }
        for stage in stages {
            for lookup in lookups(table, &scripts, &stage)? {
                engine.apply(&mut buffer, &lookup)?;
            }
        }
    }
    let mut glyphs = buffer.into_glyphs();
    for glyph in &mut glyphs {
        if glyph.id == DELETED_GLYPH {
            continue;
        }
        glyph.x_advance = i32::from(font.advance(glyph.id).map_err(TextError::Font)?);
        if let Some(space) = glyph.space {
            glyph.x_advance = space_width(font, space, glyph.x_advance);
        }
    }
    let positioned = match placing {
        Some(table) => lookups(
            table,
            &scripts,
            &[
                Feature::every(b"abvm"),
                Feature::every(b"blwm"),
                Feature::every(b"mark"),
                Feature::every(b"mkmk"),
                Feature::every(b"curs"),
                Feature::every(b"dist"),
                Feature::every(b"kern"),
            ],
        )?,
        None => Vec::new(),
    };
    let kerned = placing.is_some_and(|table| has_feature(table, &scripts, *b"kern"));
    if placing.is_some()
        && let Some(mut engine) = Engine::new(font, true, backward, count, trusted)
    {
        let mut buffer = Buffer::new(glyphs);
        for lookup in &positioned {
            engine.apply(&mut buffer, lookup)?;
        }
        glyphs = buffer.into_glyphs();
    }
    // Kerning the positioning table did not do. A face placed by its
    // positioning table is never kerned by the extended table as well; it
    // is kerned by its `kern` table when the positioning table has no
    // kerning and the face no extended table. A face with no positioning
    // table is kerned by the extended table when it has one, and else by
    // the `kern` table, as passes or as pairs.
    let own_kerning = match (placing.is_some(), kerx.is_empty()) {
        (true, true) => !kerned,
        (true, false) => false,
        (false, _) => true,
    };
    let passes: &[KernPass<'_>] = if !own_kerning {
        &[]
    } else if placing.is_none() && !kerx.is_empty() {
        kerx
    } else {
        font.kern_passes()
    };
    kern::apply(passes, &mut glyphs, backward);
    if own_kerning && passes.is_empty() && kerx.is_empty() && font.has_kern_table() {
        kern_pairs(font, &mut glyphs, backward);
    }
    // Passes that are machines, or that move glyphs across the line, are
    // how their face places its marks. A face with an extended table
    // keeps its marks' room wherever that table would have kerned it,
    // even when the positioning table is what places it.
    let by_kerx = !kerx.is_empty() && (placing.is_none() || !kerned);
    let machine = passes
        .iter()
        .any(|pass| matches!(pass, KernPass::Machine(_)));
    let across = passes.iter().any(KernPass::cross_stream);
    // A mark takes no room of its own. In a face that places nothing, it
    // is drawn back over the letter before it.
    let unplaced = placing.is_none() && !by_kerx && !across;
    for glyph in &mut glyphs {
        // The marks of a script written by syllable keep their advance:
        // its fonts place them by it. So do those of a script with shaping
        // of its own in a face that shapes by its own machines.
        let own_shaping =
            shaping != Shaping::Plain || thai || korean || run.script == Script(*b"Hebr");
        let kept = by_syllable || (machines && own_shaping) || by_kerx || machine;
        if glyph.is_mark() && !kept {
            if unplaced && !backward {
                glyph.x_offset -= glyph.x_advance;
            }
            glyph.x_advance = 0;
        }
    }
    settle(&mut glyphs, backward);
    // Thai and Lao faces draw their marks where they go, or move them
    // with their own tables: those are not set by their boxes.
    if unplaced && !by_syllable && !thai {
        fallback::place_marks(font, &mut glyphs, backward, !backward);
    }
    // What draws nothing is taken out with the glyphs in the order they
    // are seen, so its place in the text goes to the glyph seen before it.
    if backward {
        glyphs.reverse();
    }
    remove(&mut glyphs, |glyph| glyph.id == DELETED_GLYPH);
    remove(&mut glyphs, Glyph::steps_over);
    if backward != turned {
        glyphs.reverse();
    }
    Ok(glyphs)
}

fn has(font: &Font<'_>, character: char) -> bool {
    matches!(font.glyph_id(character), Ok(Some(_)))
}

/// Whether a face that lacks `character` draws it with its plain space,
/// at a width of its own.
pub(crate) fn has_space_for(font: &Font<'_>, character: char) -> bool {
    Space::of(character).is_some() && has(font, ' ')
}

/// A glyph for `character` with what Unicode says of it.
fn described(character: char, cluster: usize) -> Glyph {
    let props = props(character);
    let mut glyph = Glyph::new(character, cluster);
    glyph.category = props.category();
    glyph.combining = sorting_class(props.combining());
    if props.category() == gc::NONSPACING_MARK {
        glyph.side = fallback::side(character, props.combining());
    }
    glyph.ignorable = props.default_ignorable();
    glyph.joiner = character == '\u{200D}';
    glyph.non_joiner = character == '\u{200C}';
    glyph.continues = props.is_mark();
    glyph
}

/// The order a mark is sorted in among the marks of one letter: its
/// combining class, but for the marks of Hebrew, Arabic and Thai, which
/// are set in the order their fonts expect.
fn sorting_class(class: u8) -> u8 {
    match class {
        // Hebrew.
        10 => 22,
        11 => 15,
        12 => 16,
        13 => 17,
        14 => 23,
        15 => 18,
        16 => 19,
        17 => 20,
        18 => 21,
        19 => 14,
        20 => 24,
        21 => 12,
        22 => 25,
        23 => 13,
        24 => 10,
        25 => 11,
        // Arabic: the shadda comes first.
        27 => 28,
        28 => 29,
        29 => 30,
        30 => 31,
        31 => 32,
        32 => 33,
        33 => 27,
        // Telugu length marks stay where they are typed.
        84 | 91 => 0,
        // Thai and Lao vowels below go before the tone marks.
        103 => 3,
        118 => 122,
        // Tibetan.
        130 => 132,
        132 => 131,
        other => other,
    }
}

/// Mark the characters that go with the one before them: the second of a
/// flag's two halves, a skin tone, and whatever a joiner joins.
fn mark_continuations(glyphs: &mut [Glyph]) {
    let mut index = 0;
    while index < glyphs.len() {
        let character = glyphs[index].character;
        let code = u32::from(character);
        if glyphs[index].category == gc::MODIFIER_SYMBOL && (0x1F3FB..=0x1F3FF).contains(&code) {
            glyphs[index].continues = true;
        } else if props(character).regional_indicator() {
            if index > 0
                && props(glyphs[index - 1].character).regional_indicator()
                && !glyphs[index - 1].continues
            {
                glyphs[index].continues = true;
            }
        } else if glyphs[index].joiner {
            glyphs[index].continues = true;
            if glyphs
                .get(index + 1)
                .is_some_and(|next| props(next.character).pictographic())
            {
                index += 1;
                glyphs[index].continues = true;
            }
        } else if (0xE0020..=0xE007F).contains(&code) || matches!(code, 0xFF9E | 0xFF9F) {
            glyphs[index].continues = true;
        }
        index += 1;
    }
}

/// Give a character and the ones that go with it one cluster.
fn merge_graphemes(glyphs: &mut [Glyph]) {
    let mut start = 0;
    for index in 1..=glyphs.len() {
        if index < glyphs.len() && glyphs[index].continues {
            continue;
        }
        if let Some(first) = glyphs[start..index].iter().map(|glyph| glyph.cluster).min() {
            for glyph in &mut glyphs[start..index] {
                glyph.cluster = first;
            }
        }
        start = index;
    }
}

/// Turn the run around, keeping each character ahead of the ones that go
/// with it.
fn reverse_graphemes(glyphs: &mut [Glyph]) {
    glyphs.reverse();
    let mut start = 0;
    for index in 0..glyphs.len() {
        // Reversed, a character comes after the ones that continue it.
        if !glyphs[index].continues {
            glyphs[start..=index].reverse();
            start = index + 1;
        }
    }
}

/// How a script's characters are taken apart and put together.
struct Normal<'n> {
    /// A character the face has a glyph for is left whole.
    whole_first: bool,
    /// Characters never taken apart.
    keeps_whole: &'n dyn Fn(char) -> bool,
    /// The character two compose, when they should.
    join: &'n dyn Fn(char, char) -> Option<char>,
}

/// Map characters to the face's glyphs. An accented letter the face
/// lacks is drawn as its letter and its accent; a letter and accent the
/// face has as one glyph are drawn as that glyph; and a letter's marks
/// are put in their canonical order.
fn normalize(
    font: &Font<'_>,
    glyphs: Vec<Glyph>,
    normal: &Normal<'_>,
) -> Result<Vec<Glyph>, TextError> {
    let join = normal.join;
    // Take apart what the face cannot draw whole.
    let mut parts: Vec<Glyph> = Vec::with_capacity(glyphs.len());
    for glyph in glyphs {
        if normal.whole_first && has(font, glyph.character) {
            parts.push(glyph);
            continue;
        }
        let from = parts.len();
        let apart = !(normal.keeps_whole)(glyph.character)
            && take_apart(
                font,
                &glyph,
                glyph.character,
                &mut parts,
                !normal.whole_first,
                0,
            );
        if !apart {
            parts.truncate(from);
            let mut whole = glyph;
            // A space the face lacks is drawn with its plain space, and a
            // hyphen that does not break as a hyphen.
            if has(font, ' ') {
                whole.space = Space::of(whole.character);
            }
            if whole.character == '\u{2011}' && has(font, '\u{2010}') {
                whole.character = '\u{2010}';
            }
            parts.push(whole);
        }
    }
    // Sort each letter's marks.
    let mut start = 0;
    while start < parts.len() {
        if parts[start].combining == 0 {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < parts.len() && parts[end].combining != 0 {
            end += 1;
        }
        if end - start <= MAX_MARKS {
            parts[start..end].sort_by_key(|glyph| glyph.combining);
        }
        start = end;
    }
    // Put together what the face draws as one.
    let mut out: Vec<Glyph> = Vec::with_capacity(parts.len());
    let mut starter = 0;
    for glyph in parts {
        let is_mark = props(glyph.character).is_mark();
        let open = out
            .last()
            .is_some_and(|last| starter + 1 == out.len() || last.combining < glyph.combining);
        if is_mark
            && open
            && let Some(base) = out.get(starter)
            && let Some(whole) = join(base.character, glyph.character)
            && has(font, whole)
        {
            let cluster = out[starter..]
                .iter()
                .map(|glyph| glyph.cluster)
                .min()
                .unwrap_or(glyph.cluster)
                .min(glyph.cluster);
            for earlier in &mut out[starter..] {
                earlier.cluster = cluster;
            }
            let mut composed = described(whole, cluster);
            composed.mask = out[starter].mask;
            composed.continues = out[starter].continues;
            out[starter] = composed;
            continue;
        }
        let combining = glyph.combining;
        out.push(glyph);
        if combining == 0 {
            starter = out.len() - 1;
        }
    }
    let mut mapped: Vec<Glyph> = Vec::with_capacity(out.len());
    let mut rest = out.into_iter().peekable();
    while let Some(mut glyph) = rest.next() {
        let drawn = if glyph.space.is_some() {
            ' '
        } else {
            glyph.character
        };
        // A character and the selector after it that the face lists as a
        // pair are one glyph, and the selector is then spent.
        let pair = match rest.peek() {
            Some(next) if selects(next.character) => {
                match font
                    .variant(drawn, next.character)
                    .map_err(TextError::Font)?
                {
                    Variant::Glyph(id) => Some(id),
                    Variant::Plain => font.glyph_id(drawn).map_err(TextError::Font)?,
                    Variant::Unlisted => None,
                }
            }
            _ => None,
        };
        if let Some(id) = pair {
            glyph.id = id;
            let _selector = rest.next();
            mapped.push(glyph);
            continue;
        }
        match font.glyph_id(drawn).map_err(TextError::Font)? {
            Some(id) => glyph.id = id,
            None => {
                glyph.id = font.missing_glyph();
                glyph.missing = !glyph.ignorable;
            }
        }
        mapped.push(glyph);
    }
    Ok(mapped)
}

/// Whether `character` is a variation selector: it picks a form of the
/// character before it and draws nothing itself.
fn selects(character: char) -> bool {
    matches!(
        u32::from(character),
        0x180B..=0x180D | 0x180F | 0xFE00..=0xFE0F | 0xE0100..=0xE01EF
    )
}

/// The Hebrew letter with its point as one character, for the pairs
/// Unicode keeps apart but old faces draw as one glyph.
fn hebrew_form(letter: char, point: char) -> Option<char> {
    const DAGESH: [u32; 27] = [
        0xFB30, 0xFB31, 0xFB32, 0xFB33, 0xFB34, 0xFB35, 0xFB36, 0, 0xFB38, 0xFB39, 0xFB3A, 0xFB3B,
        0xFB3C, 0, 0xFB3E, 0, 0xFB40, 0xFB41, 0, 0xFB43, 0xFB44, 0, 0xFB46, 0xFB47, 0xFB48, 0xFB49,
        0xFB4A,
    ];
    let code = match (u32::from(letter), u32::from(point)) {
        (0x05D9, 0x05B4) => 0xFB1D,
        (0x05F2, 0x05B7) => 0xFB1F,
        (0x05D0, 0x05B7) => 0xFB2E,
        (0x05D0, 0x05B8) => 0xFB2F,
        (0x05D5, 0x05B9) => 0xFB4B,
        (letter @ 0x05D0..=0x05EA, 0x05BC) => DAGESH[usize::try_from(letter - 0x05D0).ok()?],
        (0xFB2A, 0x05BC) | (0xFB49, 0x05C1) => 0xFB2C,
        (0xFB2B, 0x05BC) | (0xFB49, 0x05C2) => 0xFB2D,
        (0x05D1, 0x05BF) => 0xFB4C,
        (0x05DB, 0x05BF) => 0xFB4D,
        (0x05E4, 0x05BF) => 0xFB4E,
        (0x05E9, 0x05C1) => 0xFB2A,
        (0x05E9, 0x05C2) => 0xFB2B,
        _ => 0,
    };
    char::from_u32(code).filter(|_| code != 0)
}

/// Write `character` as parts the face has glyphs for, onto `out`.
/// `false` when it has no such parts. `deep` takes the first part apart
/// too whenever it can be.
fn take_apart(
    font: &Font<'_>,
    origin: &Glyph,
    character: char,
    out: &mut Vec<Glyph>,
    deep: bool,
    depth: u8,
) -> bool {
    let Some((first, second)) = decompose(character) else {
        return false;
    };
    if second.is_some_and(|second| !has(font, second)) || depth > 8 {
        return false;
    }
    let push = |character: char, out: &mut Vec<Glyph>| {
        let mut glyph = described(character, origin.cluster);
        glyph.mask = origin.mask;
        if !out.is_empty() && origin.continues {
            glyph.continues = true;
        }
        out.push(glyph);
    };
    let from = out.len();
    if !deep && has(font, first) {
        push(first, out);
    } else if !take_apart(font, origin, first, out, deep, depth + 1) {
        out.truncate(from);
        if !has(font, first) {
            return false;
        }
        push(first, out);
    }
    if let Some(second) = second {
        push(second, out);
    }
    // The first part stands where the whole did; the rest go with it.
    if let Some(head) = out.get_mut(from) {
        head.continues = origin.continues;
    }
    for part in out.iter_mut().skip(from + 1) {
        part.continues = true;
    }
    true
}

/// Each glyph's kind: what the face says, or a guess from its character.
/// `false` when the face's word is not taken: some faces call a combining
/// accent a base letter, and then every glyph of the run is guessed and
/// the face's mark classes are left unused.
fn classify(font: &Font<'_>, glyphs: &mut [Glyph], syllabic: bool) -> bool {
    // With no classes in the face, a mark is a mark even when the face
    // has no glyph for it. Where the face has classes that are not taken
    // at their word, its missing glyph stays the base it says it is.
    let classed = font.gdef().is_some_and(|gdef| gdef.has_classes());
    let guess = |glyph: &Glyph| {
        if glyph.category == gc::NONSPACING_MARK && !glyph.ignorable && !(classed && glyph.missing)
        {
            GlyphClass::Mark
        } else {
            GlyphClass::Base
        }
    };
    let gdef = font
        .gdef()
        .filter(|gdef| gdef.has_classes())
        .filter(|gdef| {
            // An accent with a combining class is a mark in any face. The
            // signs of a script written by syllable may well be bases: its
            // fonts say, and are taken at their word.
            syllabic
                || !glyphs.iter().any(|glyph| {
                    glyph.combining != 0
                        && !glyph.missing
                        && guess(glyph) == GlyphClass::Mark
                        && matches!(
                            gdef.class(glyph.id),
                            GlyphClass::Base | GlyphClass::Ligature
                        )
                })
        });
    for glyph in glyphs.iter_mut() {
        glyph.class = match gdef {
            // A letter, a digit or a sign is no mark, whatever the face
            // calls its glyph: some widely installed faces class a few
            // letters and quotes as marks, and those would take no room
            // and be drawn over the letter before them. The signs of a
            // script written by syllable are the face's to say, and so is
            // a glyph a machine has put in place of a character's own.
            Some(gdef) => match gdef.class(glyph.id) {
                GlyphClass::Mark
                    if !syllabic && !glyph.ignorable && !props(glyph.character).is_mark() =>
                {
                    GlyphClass::Base
                }
                class => class,
            },
            None => guess(glyph),
        };
    }
    gdef.is_some() || font.gdef().is_none_or(|gdef| !gdef.has_classes())
}

/// How far the pen moves for a space the face lacks, given its plain
/// space's advance.
fn space_width(font: &Font<'_>, space: Space, plain: i32) -> i32 {
    let em = i32::from(font.units_per_em());
    let advance_of = |characters: &[char]| {
        characters.iter().find_map(|character| {
            let glyph = font.glyph_id(*character).ok().flatten()?;
            font.advance(glyph).ok().map(i32::from)
        })
    };
    match space {
        Space::Plain => plain,
        Space::Narrow => plain / 2,
        Space::Em(share) => (em + share / 2) / share,
        Space::Mathematical => em * 4 / 18,
        Space::Figure => {
            advance_of(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']).unwrap_or(plain)
        }
        Space::Punctuation => advance_of(&['.', ',']).unwrap_or(plain),
    }
}

/// The substitution features for a script, in the steps they run in.
fn substitution_stages(shaping: Shaping, backward: bool, ligatures: bool) -> Vec<Vec<Feature>> {
    let direction = if backward {
        [Feature::every(b"rtla"), Feature::masked(b"rtlm", MIRRORED)]
    } else {
        [Feature::every(b"ltra"), Feature::every(b"ltrm")]
    };
    let mut common = vec![
        Feature::every(b"ccmp"),
        Feature::every(b"locl"),
        Feature::every(b"rlig"),
        Feature::every(b"calt"),
        Feature::every(b"rclt"),
    ];
    if ligatures {
        common.extend([Feature::every(b"clig"), Feature::every(b"liga")]);
    }
    let mut stages = vec![vec![Feature::every(b"rvrn")]];
    match shaping {
        Shaping::Plain | Shaping::Syllabic => {
            let mut stage = direction.to_vec();
            stage.extend(common);
            stages.push(stage);
        }
        Shaping::Joined => {
            // Each form is its own step, so one form's lookups see the
            // glyphs the earlier forms made.
            stages.push(direction.to_vec());
            stages.push(vec![
                Feature::every(b"ccmp").manual(),
                Feature::every(b"locl").manual(),
            ]);
            for (tag, mask) in arabic::FORMS {
                stages.push(vec![Feature::masked(tag, mask).manual()]);
            }
            stages.push(vec![Feature::every(b"rlig").manual()]);
            stages.push(vec![Feature::every(b"calt").manual()]);
            let mut last = vec![Feature::every(b"rclt").manual()];
            if ligatures {
                last.extend([
                    Feature::every(b"liga").manual(),
                    Feature::every(b"clig").manual(),
                ]);
            }
            last.push(Feature::every(b"mset").manual());
            stages.push(last);
        }
    }
    stages
}

/// Whether the language the scripts pick turns `tag` on.
fn has_feature(table: &LayoutTable<'_>, scripts: &[Tag], tag: Tag) -> bool {
    table.language(scripts, None).is_ok_and(|language| {
        language
            .features
            .iter()
            .any(|index| table.feature_tag(*index).is_ok_and(|found| found == tag))
    })
}

/// The lookups `features` run, in the font's order, each once.
pub(super) fn lookups(
    table: &LayoutTable<'_>,
    scripts: &[Tag],
    features: &[Feature],
) -> Result<Vec<LookupRef>, TextError> {
    let language = table.language(scripts, None).map_err(TextError::Font)?;
    let mut found: Vec<LookupRef> = Vec::new();
    for feature in features {
        for index in &language.features {
            if table.feature_tag(*index).map_err(TextError::Font)? != feature.tag {
                continue;
            }
            for lookup in table
                .feature_lookups(*index)
                .map_err(TextError::Font)?
                .iter()
            {
                found.push(LookupRef {
                    index: lookup,
                    mask: feature.mask,
                    auto_joiner: !feature.manual_joiners,
                    auto_non_joiner: true,
                    per_syllable: false,
                });
            }
            break;
        }
    }
    found.sort_by_key(|lookup| lookup.index);
    let mut merged: Vec<LookupRef> = Vec::with_capacity(found.len());
    for lookup in found {
        match merged.last_mut().filter(|last| last.index == lookup.index) {
            Some(last) => {
                last.mask |= lookup.mask;
                last.auto_joiner &= lookup.auto_joiner;
                last.auto_non_joiner &= lookup.auto_non_joiner;
            }
            None => merged.push(lookup),
        }
    }
    Ok(merged)
}

/// Kern by the face's `kern` table, for a face whose positioning table
/// does not kern: each pair of glyphs that are not marks, the adjustment
/// shared between the two.
fn kern_pairs(font: &Font<'_>, glyphs: &mut [Glyph], backward: bool) {
    // The table's pairs are the glyphs as they are read.
    if backward {
        glyphs.reverse();
    }
    kern::pairs(glyphs, false, !backward, true, |left, right| {
        i32::from(font.kerning(left, right))
    });
    if backward {
        glyphs.reverse();
    }
}

/// Take out the glyphs that are `gone`: the ones a machine deleted, and
/// the characters that draw nothing, unless a lookup gave one a glyph. A
/// glyph's cluster goes to a neighbour, so no part of the text is left
/// without a glyph that stands for it.
fn remove(glyphs: &mut Vec<Glyph>, gone: impl Fn(&Glyph) -> bool) {
    let mut kept = 0;
    for index in 0..glyphs.len() {
        if !gone(&glyphs[index]) {
            glyphs.swap(kept, index);
            kept += 1;
            continue;
        }
        let cluster = glyphs[index].cluster;
        if glyphs
            .get(index + 1)
            .is_some_and(|next| next.cluster == cluster)
        {
            continue;
        }
        if kept > 0 {
            let before = glyphs[kept - 1].cluster;
            if cluster < before {
                for earlier in glyphs[..kept].iter_mut().rev() {
                    if earlier.cluster != before {
                        break;
                    }
                    earlier.cluster = cluster;
                }
            }
        } else if let Some(next) = glyphs.get_mut(index + 1) {
            next.cluster = next.cluster.min(cluster);
        }
    }
    glyphs.truncate(kept);
}

#[cfg(test)]
mod tests {
    use super::{Run, shape_run};
    use crate::script::Script;

    /// One run as `hb-shape` prints it: `[glyph=cluster@x,y+advance|...]`,
    /// in the order the glyphs are seen.
    fn printed(font: &cutlass_font::Font<'_>, run: &Run<'_>) -> String {
        let mut glyphs = match shape_run(font, run) {
            Ok(glyphs) => glyphs,
            Err(error) => return format!("error: {error}"),
        };
        if run.backward {
            glyphs.reverse();
        }
        let items: Vec<String> = glyphs
            .iter()
            .map(|glyph| {
                let place = if glyph.x_offset == 0 && glyph.y_offset == 0 {
                    String::new()
                } else {
                    format!("@{},{}", glyph.x_offset, glyph.y_offset)
                };
                format!("{}={}{place}+{}", glyph.id, glyph.cluster, glyph.x_advance)
            })
            .collect();
        format!("[{}]", items.join("|"))
    }

    const SHAPING: &[u8] = include_bytes!("../../../font/fixtures/shaping.ttf");
    const HOSTILE: &[u8] = include_bytes!("../../../font/fixtures/hostile.ttf");

    /// `text` shaped in the fixture face, as `hb-shape` prints it. Every
    /// expected string below is what HarfBuzz 14.2 prints for the same
    /// face and text.
    fn shaped(text: &str, script: &[u8; 4], backward: bool) -> String {
        let font = cutlass_font::read(SHAPING).expect("fixture");
        let run = Run {
            text,
            start: 0,
            end: text.len(),
            script: Script(*script),
            backward,
            ligatures: true,
        };
        printed(&font, &run)
    }

    /// Check against Core Text where each glyph is drawn: the glyphs, and
    /// each one's place from the start of the run. The cases are the file
    /// `CUTLASS_SHAPE_CASES` names, as for [`harfbuzz_agrees`], and
    /// `CUTLASS_CORETEXT` is the program built from
    /// `tools/text-oracles/coretext.swift`. Core Text gives a pair's
    /// kerning to its first glyph where this shaper shares it between the
    /// two, so advances are not compared, only where the glyphs land.
    #[test]
    #[ignore = "runs the Core Text probe named in CUTLASS_CORETEXT"]
    fn coretext_agrees() {
        let (Ok(path), Ok(probe)) = (
            std::env::var("CUTLASS_SHAPE_CASES"),
            std::env::var("CUTLASS_CORETEXT"),
        ) else {
            return;
        };
        let cases = std::fs::read_to_string(path).expect("cases");
        let mut wrong = Vec::new();
        let mut total = 0;
        for case in cases.lines().filter(|line| !line.is_empty()) {
            let fields: Vec<&str> = case.split('\t').collect();
            let [file, face, script, direction, escaped] = fields[..] else {
                panic!("case: {case}");
            };
            let text = unescape(escaped);
            let bytes = std::fs::read(file).expect("font");
            let index: u32 = face.parse().expect("face");
            let font = if bytes.starts_with(b"ttcf") {
                cutlass_font::read_collection(&bytes, index)
            } else {
                cutlass_font::read(&bytes)
            }
            .expect("face");
            let mut tag = [0_u8; 4];
            tag.copy_from_slice(script.as_bytes());
            let run = Run {
                text: &text,
                start: 0,
                end: text.len(),
                script: Script(tag),
                backward: direction == "rtl",
                ligatures: true,
            };
            let mut glyphs = shape_run(&font, &run).expect("shaped");
            if run.backward {
                glyphs.reverse();
            }
            let mut pen = 0;
            let mut ours = Vec::new();
            for glyph in &glyphs {
                ours.push(format!(
                    "{}@{},{}",
                    glyph.id,
                    pen + glyph.x_offset,
                    glyph.y_offset
                ));
                pen += glyph.x_advance;
            }
            let output = std::process::Command::new(&probe)
                .arg(file)
                .arg(face)
                .arg(
                    text.chars()
                        .map(|character| format!("U+{:04X}", u32::from(character)))
                        .collect::<Vec<_>>()
                        .join(","),
                )
                .output()
                .expect("probe");
            let printed = String::from_utf8_lossy(&output.stdout);
            // Core Text sets what a face lacks in another face. Such a
            // case says nothing about this face, and is passed over.
            let faces: std::collections::HashSet<&str> = printed
                .trim()
                .split('|')
                .filter(|item| item.starts_with('{'))
                .collect();
            if faces.len() != 1 || glyphs.iter().any(|glyph| glyph.missing) {
                continue;
            }
            // A glyph without its advance; the names of the faces left out.
            let theirs: Vec<String> = printed
                .trim()
                .split('|')
                .filter(|item| !item.starts_with('{'))
                .filter_map(|item| item.split_once('+').map(|(place, _)| place.to_owned()))
                .collect();
            total += 1;
            if ours != theirs {
                let name = file.rsplit('/').next().unwrap_or(file);
                wrong.push(format!(
                    "{name}#{face} {script} {direction} {escaped}\n   ours   {}\n   theirs {}",
                    ours.join("|"),
                    theirs.join("|")
                ));
            }
        }
        let limit: usize = std::env::var("CUTLASS_SHAPE_SHOW")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(25);
        assert!(
            wrong.is_empty(),
            "{} of {total} differ:\n{}",
            wrong.len(),
            wrong
                .iter()
                .take(limit)
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// `text` shaped left to right in one of the kerning fixture faces.
    fn kerned(face: &[u8], text: &str) -> String {
        let font = cutlass_font::read(face).expect("fixture");
        let run = Run {
            text,
            start: 0,
            end: text.len(),
            script: Script(*b"Latn"),
            backward: false,
            ligatures: true,
        };
        printed(&font, &run)
    }

    #[test]
    fn a_kern_table_of_state_machines_moves_glyphs_along_and_across_the_line() {
        const MACHINES: &[u8] = include_bytes!("../../../font/fixtures/kern-machine.ttf");
        // Each line is what HarfBuzz 14.2 prints for the face.
        // B after A: the machine remembers both and moves each back.
        assert_eq!(kerned(MACHINES, "AB"), "[1=0@-50,0+550|2=1@-100,0+500]");
        // Three glyphs remembered and two values in the table: nothing
        // is moved, and nothing is read past the table.
        assert_eq!(
            kerned(MACHINES, "AABC"),
            "[1=0+600|1=1+600|2=2+600|4=3+600]"
        );
        // An acute after B is lifted, and the letter after it is back on
        // the line, where the pairs by class then kern it.
        assert_eq!(kerned(MACHINES, "B\u{301}"), "[2=0+600|3=0@0,300+0]");
        assert_eq!(
            kerned(MACHINES, "B\u{301}CA"),
            "[2=0+600|3=0@0,300+0|4=3+560|1=4@-40,0+560]"
        );
        // Pairs by offset and by compact classes, one after the other.
        assert_eq!(
            kerned(MACHINES, "CAC"),
            "[4=0+560|1=1@-40,0+540|4=2@-20,0+580]"
        );
        assert_eq!(
            kerned(MACHINES, "ACAB\u{301}B"),
            "[1=0+580|4=1@-20,0+540|1=2@-40,0+560|2=3+600|3=3@0,300+0|2=6+600]"
        );
    }

    #[test]
    fn an_extended_kerning_table_kerns_pairs_and_sets_marks_on_anchors() {
        const EXTENDED: &[u8] = include_bytes!("../../../font/fixtures/kerx.ttf");
        // Sorted pairs, a machine, and a mark set with its anchor on its
        // letter's: as HarfBuzz 14.2 prints them.
        assert_eq!(kerned(EXTENDED, "AB"), "[1=0+570|2=1@-30,0+570]");
        assert_eq!(kerned(EXTENDED, "BC"), "[2=0+600|4=1@-70,0+530]");
        assert_eq!(kerned(EXTENDED, "A\u{301}"), "[1=0+600|3=0@-150,100+0]");
        // Pairs by class, whose values are byte offsets into the array:
        // A after C is 90 closer, as Core Text reads such a table.
        // HarfBuzz 14 reads another cell and leaves the pair alone.
        assert_eq!(kerned(EXTENDED, "CA"), "[4=0+555|1=1@-45,0+555]");
        assert_eq!(
            kerned(EXTENDED, "BCA"),
            "[2=0+600|4=1@-70,0+485|1=2@-45,0+555]"
        );
        // These tables know nothing of marks: a pair is two glyphs side
        // by side, so A and B with an acute between them are not kerned,
        // as in Core Text. HarfBuzz steps over the mark and kerns them.
        assert_eq!(
            kerned(EXTENDED, "A\u{301}B"),
            "[1=0+600|3=0@-150,100+0|2=3+600]"
        );
        // Two marks on one letter each meet its anchor.
        assert_eq!(
            kerned(EXTENDED, "A\u{301}\u{301}"),
            "[1=0+600|3=0@-150,100+0|3=0@-150,100+0]"
        );
    }

    #[test]
    fn a_kerning_machine_that_never_moves_on_is_stopped() {
        const STUCK: &[u8] = include_bytes!("../../../font/fixtures/kern-stuck.ttf");
        let started = std::time::Instant::now();
        let long = "AB".repeat(2_000);
        let font = cutlass_font::read(STUCK).expect("fixture");
        let run = Run {
            text: &long,
            start: 0,
            end: long.len(),
            script: Script(*b"Latn"),
            backward: false,
            ligatures: true,
        };
        let glyphs = shape_run(&font, &run).expect("shaped");
        assert_eq!(glyphs.len(), 4_000);
        // Each glyph was read a few times and no more.
        assert!(glyphs.iter().all(|glyph| glyph.x_advance > -200_000));
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    /// `text` of `script` shaped left to right in a fixture face.
    fn shaped_in(face: &[u8], script: &[u8; 4], text: &str) -> String {
        let font = cutlass_font::read(face).expect("fixture");
        let run = Run {
            text,
            start: 0,
            end: text.len(),
            script: Script(*script),
            backward: false,
            ligatures: true,
        };
        printed(&font, &run)
    }

    #[test]
    fn korean_letters_are_drawn_as_the_syllables_the_face_has() {
        const SYLLABLES: &[u8] = include_bytes!("../../../font/fixtures/hangul.ttf");
        const LETTERS: &[u8] = include_bytes!("../../../font/fixtures/hangul-letters.ttf");
        // Each line is what HarfBuzz 14.2 prints for the face.
        // Three letters and two letters: two syllables.
        assert_eq!(
            shaped_in(
                SYLLABLES,
                b"Hang",
                "\u{1112}\u{1161}\u{11AB}\u{1100}\u{1161}"
            ),
            "[3=0+900|1=9+900]"
        );
        // A syllable and the consonant that ends it are one glyph.
        assert_eq!(
            shaped_in(SYLLABLES, b"Hang", "\u{AC00}\u{11A8}a"),
            "[2=0+900|10=6+500]"
        );
        // A syllable the face lacks is drawn from its letters, and one
        // with an old final consonant is taken apart to join it.
        assert_eq!(
            shaped_in(LETTERS, b"Hang", "\u{AC01}"),
            "[1=0+300|2=0+300|3=0+300]"
        );
        assert_eq!(
            shaped_in(SYLLABLES, b"Hang", "\u{AC00}\u{11C3}"),
            "[4=0+300|5=0+300|9=0+300]"
        );
        // In another script's run the letters are left as they are.
        assert_eq!(
            shaped_in(SYLLABLES, b"Latn", "\u{1100}\u{1161}"),
            "[4=0+300|5=3+300]"
        );
    }

    #[test]
    fn thai_sara_am_is_split_and_marks_stay_where_the_face_draws_them() {
        const THAI: &[u8] = include_bytes!("../../../font/fixtures/thai.ttf");
        // Each line is what HarfBuzz 14.2 prints for the face. Sara am is
        // its circle and its vowel, with the circle before a tone mark.
        assert_eq!(
            shaped_in(THAI, b"Thai", "\u{E01}\u{E33}"),
            "[1=0+600|3=0+0|4=0+400]"
        );
        assert_eq!(
            shaped_in(THAI, b"Thai", "\u{E01}\u{E48}\u{E33}"),
            "[1=0+600|3=0+0|2=0+0|4=0+400]"
        );
        assert_eq!(shaped_in(THAI, b"Thai", "\u{E33}"), "[3=0+0|4=0+400]");
        // A Thai mark is not moved over its consonant by its box: the
        // face draws it where it goes.
        assert_eq!(
            shaped_in(THAI, b"Thai", "\u{E01}\u{E48}"),
            "[1=0+600|2=0+0]"
        );
    }

    #[test]
    fn a_letter_a_face_calls_a_mark_keeps_its_room() {
        // The face's class table calls B a mark, as some widely installed
        // faces call a few letters and quotes. HarfBuzz takes the table
        // at its word unless the face is on its list of known ones, and
        // draws B over the letter before it with no room of its own.
        // Here a letter is a letter: it has its advance, and kerns.
        const MISCLASSED: &[u8] = include_bytes!("../../../font/fixtures/misclassed.ttf");
        assert_eq!(kerned(MISCLASSED, "B"), "[2=0+600]");
        assert_eq!(kerned(MISCLASSED, "ABC"), "[1=0+550|2=1@-50,0+550|4=2+600]");
        assert_eq!(kerned(MISCLASSED, "BAB"), "[2=0+600|1=1+550|2=2@-50,0+550]");
        // A mark that is a mark is still one: it takes no room, sits on
        // its letter, and the letters either side of it still kern.
        assert_eq!(
            kerned(MISCLASSED, "A\u{301}B"),
            "[1=0+550|3=0@-100,12+0|2=3@-50,0+550]"
        );
    }

    #[test]
    fn a_character_and_its_selector_draw_the_glyph_the_face_lists_for_them() {
        const VARIED: &[u8] = include_bytes!("../../../font/fixtures/varied.ttf");
        let font = cutlass_font::read(VARIED).expect("fixture");
        let shaped = |text: &str, backward: bool| {
            let run = Run {
                text,
                start: 0,
                end: text.len(),
                script: Script(*b"Zyyy"),
                backward,
                ligatures: true,
            };
            printed(&font, &run)
        };
        // The digit with the emoji selector is its keycap picture, and the
        // selector is spent: the face's own glyph for it is not drawn.
        assert_eq!(shaped("1\u{FE0F}", false), "[3=0+900]");
        assert_eq!(shaped("1\u{FE0F}\u{20E3}", false), "[3=0+900|6=0+800]");
        // A pair listed as plain draws the letter, and another selector
        // picks another glyph of it.
        assert_eq!(shaped("A\u{FE0F}A\u{FE0E}", false), "[1=0+600|5=4+650]");
        assert_eq!(shaped("A\u{FE0F}A\u{FE0E}", true), "[5=4+650|1=0+600]");
        // A pair the face does not list draws the character, and the
        // selector draws nothing.
        assert_eq!(shaped("1\u{FE0E}A\u{FE00}", false), "[2=0+500|1=4+600]");
        // Only the character before the selector is changed.
        assert_eq!(shaped("11\u{FE0F}1", false), "[2=0+500|3=1+900|2=5+500]");
        assert_eq!(shaped("11\u{FE0F}1", true), "[2=5+500|3=1+900|2=0+500]");
        // A selector with nothing before it draws nothing.
        assert_eq!(shaped("\u{FE0F}1", false), "[2=0+500]");
    }

    #[test]
    fn latin_makes_ligatures_and_kerns_pairs() {
        // f and i are one glyph; a non-joiner keeps them apart and draws
        // nothing; A before V is 100 shorter.
        assert_eq!(
            shaped("fi f\u{200C}i AV", b"Latn", false),
            "[6=0+500|1=2+250|4=3+300|5=7+250|1=8+250|2=9+500|3=10+600]"
        );
        let font = cutlass_font::read(SHAPING).expect("fixture");
        let spaced = Run {
            text: "fi",
            start: 0,
            end: 2,
            script: Script::LATIN,
            backward: false,
            ligatures: false,
        };
        assert_eq!(printed(&font, &spaced), "[4=0+300|5=1+250]");
        assert_eq!(shaped("", b"Latn", false), "[]");
        // A character the face lacks is its missing glyph.
        let missing = shape_run(
            &font,
            &Run {
                text: "Z",
                start: 0,
                end: 1,
                script: Script::LATIN,
                backward: false,
                ligatures: true,
            },
        )
        .expect("missing");
        assert!(missing[0].missing && missing[0].id == 0);
    }

    #[test]
    fn accents_compose_come_apart_and_sit_on_their_anchors() {
        // e and an acute are the face's é. A and an acute stay two
        // glyphs, the acute on A's anchor and taking no room. The face
        // has no Á: it is drawn as A and an acute. Over x, which has no
        // anchor, the acute stays where the face draws it.
        assert_eq!(
            shaped("e\u{301} A\u{301} \u{C1} x\u{301}", b"Latn", false),
            "[9=0+500|1=3+250|2=4+600|7=4@-150,-50+0|1=7+250|2=8+600|7=8@-150,-50+0|1=10+250|34=11+500|7=11+0]"
        );
    }

    #[test]
    fn a_space_the_face_lacks_is_its_plain_space_at_another_width() {
        assert_eq!(
            shaped("A\u{A0}V\u{2003}x", b"Latn", false),
            "[2=0+600|1=1+250|3=3+600|1=4+1000|34=7+500]"
        );
    }

    #[test]
    fn arabic_letters_join_and_lam_alef_is_one_glyph() {
        // Three behs: first, middle, last, seen right to left. Lam and
        // alef are one ligature.
        assert_eq!(
            shaped("\u{628}\u{628}\u{628} \u{644}\u{627}", b"Arab", true),
            "[20=7+700|1=6+250|13=4+800|12=2+400|11=0+400]"
        );
        // A fatha does not come between two letters, and sits on its own.
        assert_eq!(
            shaped("\u{628}\u{64E}\u{628} \u{644}\u{64E}\u{627}", b"Arab", true),
            "[21=7+0|20=7+700|1=6+250|13=4+800|21=0@350,-150+0|11=0+400]"
        );
        // The glyphs come back in the order the text is stored.
        let font = cutlass_font::read(SHAPING).expect("fixture");
        let text = "\u{628}\u{628}";
        let glyphs = shape_run(
            &font,
            &Run {
                text,
                start: 0,
                end: text.len(),
                script: Script(*b"Arab"),
                backward: true,
                ligatures: true,
            },
        )
        .expect("arabic");
        let found: Vec<(u16, usize)> = glyphs
            .iter()
            .map(|glyph| (glyph.id, glyph.cluster))
            .collect();
        assert_eq!(found, [(11, 0), (13, 2)]);
    }

    #[test]
    fn brackets_read_right_to_left_are_mirrored() {
        assert_eq!(shaped("(A)", b"Latn", true), "[31=2+300|2=1+600|32=0+300]");
        assert_eq!(shaped("(A)", b"Latn", false), "[31=0+300|2=1+600|32=2+300]");
    }

    #[test]
    fn devanagari_syllables_are_reordered_and_take_their_forms() {
        // ki: the vowel sign is drawn first. k-ssa: one conjunct. k-ta: a
        // half ka. r-ki: the sign, ka, then the hook the r becomes.
        assert_eq!(
            shaped(
                "\u{915}\u{93F} \u{915}\u{94D}\u{937} \u{915}\u{94D}\u{924} \u{930}\u{94D}\u{915}\u{93F}",
                b"Deva",
                false
            ),
            "[27=0+300|22=0+700|1=6+250|26=7+900|1=16+250|25=17+400|30=23+650|1=26+250|27=27+300|22=27+700|29=27+0]"
        );
    }

    #[test]
    fn a_face_that_shapes_by_state_machines_makes_its_ligatures() {
        const MACHINES: &[u8] = include_bytes!("../../../font/fixtures/machines.ttf");
        let font = cutlass_font::read(MACHINES).expect("fixture");
        let shaped = |text: &'static str, backward: bool| {
            printed(
                &font,
                &Run {
                    text,
                    start: 0,
                    end: text.len(),
                    script: Script::LATIN,
                    backward,
                    ligatures: true,
                },
            )
        };
        // The face swaps D for B, then makes A and B one glyph. Each
        // expected string is what HarfBuzz prints for the same face.
        assert_eq!(shaped("ABAB", false), "[4=0+500|4=2+500]");
        assert_eq!(shaped("AD", false), "[4=0+500]");
        assert_eq!(shaped("AAB", false), "[2=0+500|4=1+500]");
        assert_eq!(shaped("BA", false), "[3=0+500|2=1+500]");
        assert_eq!(shaped("DAD", false), "[3=0+500|4=1+500]");
        assert_eq!(shaped("", false), "[]");
        // A long run is walked once, a step a glyph.
        let long = "AB".repeat(20_000);
        let glyphs = shape_run(
            &font,
            &Run {
                text: &long,
                start: 0,
                end: long.len(),
                script: Script::LATIN,
                backward: false,
                ligatures: true,
            },
        )
        .expect("long");
        assert_eq!(glyphs.len(), 20_000);
        assert!(glyphs.iter().all(|glyph| glyph.id == 4));
    }

    #[test]
    fn a_state_machine_that_never_moves_on_is_stopped() {
        const MACHINES: &[u8] = include_bytes!("../../../font/fixtures/machines.ttf");
        // The entry that marks an A now also stays on it, for ever.
        let mut stuck = MACHINES.to_vec();
        let entry = [0, 2, 0x80, 0, 0, 0];
        let at = stuck
            .windows(entry.len())
            .position(|window| window == entry)
            .expect("entry");
        stuck[at + 2] = 0xC0;
        let font = cutlass_font::read(&stuck).expect("font");
        let text = "AB".repeat(2_000);
        let started = std::time::Instant::now();
        let shaped = shape_run(
            &font,
            &Run {
                text: &text,
                start: 0,
                end: text.len(),
                script: Script::LATIN,
                backward: false,
                ligatures: true,
            },
        );
        // It ends, with the run shaped or refused, and not after long.
        assert!(shaped.is_ok() || matches!(shaped, Err(crate::TextError::Limit)));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn a_face_whose_rules_never_settle_is_stopped() {
        let font = cutlass_font::read(HOSTILE).expect("fixture");
        let run = |text: &'static str| Run {
            text,
            start: 0,
            end: text.len(),
            script: Script::LATIN,
            backward: false,
            ligatures: true,
        };
        let started = std::time::Instant::now();
        // Eight lookups that each write four glyphs for one.
        assert!(matches!(
            shape_run(&font, &run("A")),
            Err(crate::TextError::Limit)
        ));
        // Two contextual lookups that call each other.
        assert_eq!(shape_run(&font, &run("BB")).expect("loop").len(), 2);
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    /// Compare with HarfBuzz on real fonts. Runs when `CUTLASS_SHAPE_CASES`
    /// names a file of cases, one a line and tab separated: a font file, a
    /// face index, a script code, `ltr` or `rtl`, and the text with code
    /// points over ASCII written as `\u{...}`. Needs `hb-shape` on the
    /// path.
    #[test]
    #[ignore = "needs hb-shape and a file of cases"]
    fn harfbuzz_agrees() {
        let Ok(path) = std::env::var("CUTLASS_SHAPE_CASES") else {
            return;
        };
        let cases = std::fs::read_to_string(path).expect("cases");
        let mut wrong = Vec::new();
        let mut total = 0;
        let mut fonts: std::collections::HashMap<String, Vec<u8>> =
            std::collections::HashMap::new();
        for case in cases.lines().filter(|line| !line.is_empty()) {
            let fields: Vec<&str> = case.split('\t').collect();
            let [file, face, script, direction, escaped] = fields[..] else {
                panic!("case: {case}");
            };
            let text = unescape(escaped);
            let bytes = fonts
                .entry(file.to_owned())
                .or_insert_with(|| std::fs::read(file).unwrap_or_default());
            let index: u32 = face.parse().expect("face");
            let font = if bytes.starts_with(b"ttcf") {
                cutlass_font::read_collection(bytes, index)
            } else {
                cutlass_font::read(bytes)
            };
            let Ok(font) = font else {
                wrong.push(format!("{file}#{face}: does not read: {:?}", font.err()));
                continue;
            };
            let mut tag = [0_u8; 4];
            tag.copy_from_slice(script.as_bytes());
            let run = Run {
                text: &text,
                start: 0,
                end: text.len(),
                script: Script(tag),
                backward: direction == "rtl",
                ligatures: true,
            };
            let ours = printed(&font, &run);
            let output = std::process::Command::new("hb-shape")
                .arg(format!("--font-file={file}"))
                .arg(format!("--face-index={face}"))
                .arg(format!("--direction={direction}"))
                .arg(format!("--script={script}"))
                .args([
                    "--no-glyph-names",
                    "--utf8-clusters",
                    "--remove-default-ignorables",
                ])
                .arg(format!(
                    "--unicodes={}",
                    text.chars()
                        .map(|character| format!("U+{:04X}", u32::from(character)))
                        .collect::<Vec<_>>()
                        .join(",")
                ))
                .output()
                .expect("hb-shape");
            let theirs = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            total += 1;
            if ours != theirs {
                let name = file.rsplit('/').next().unwrap_or(file);
                wrong.push(format!("{name}#{face} {script} {direction} {escaped}\n   ours   {ours}\n   theirs {theirs}"));
            }
        }
        let limit: usize = std::env::var("CUTLASS_SHAPE_SHOW")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(25);
        let report: Vec<&String> = wrong.iter().take(limit).collect();
        assert!(
            wrong.is_empty(),
            "{} of {total} differ:\n{}",
            wrong.len(),
            report
                .iter()
                .map(|line| line.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    fn unescape(text: &str) -> String {
        let mut out = String::new();
        let mut rest = text;
        while let Some(at) = rest.find("\\u{") {
            out.push_str(&rest[..at]);
            let end = rest[at..].find('}').expect("brace") + at;
            let code = u32::from_str_radix(&rest[at + 3..end], 16).expect("hex");
            out.push(char::from_u32(code).expect("char"));
            rest = &rest[end + 1..];
        }
        out.push_str(rest);
        out
    }
}
