//! Installed faces for text clips, shared by every shell.
//!
//! The draft stores a face's file name. Evaluation does not open that path;
//! this table supplies the bytes when the name matches a face it knows. A
//! shell names the files it trusts ([`FaceTable::load_from`]) or the folders
//! its system keeps fonts in ([`FaceTable::scan`]). A scanned face is read
//! the first time a frame asks for it, so a long font folder costs little.
//!
//! A face of a collection is named `File.ttc#n`, face `n` of that file, and
//! labelled with its full name. Frames get it copied out as its own font.
//!
//! A character a title's own face cannot draw is looked for in the
//! table's fallback faces: the faces each system keeps for other scripts,
//! symbols and emoji, best first. What is found is remembered, so a
//! character is looked for once.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use cutlass_eval::Fonts;

/// The most faces a scan keeps: more than a machine set up for design
/// work has. Only a face's names are read until it is asked for.
pub const MAX_FACES: usize = 4_096;
/// The most bytes one face is read into. A face of a larger file is read
/// alone, without the file's other faces, and of an emoji face one size
/// of its pictures is read: the sharpest that fits. A desktop has the
/// room for the sharpest a Mac's emoji face holds, which is 80 MB; a
/// phone keeps the size below it.
pub const MAX_FACE_BYTES: u64 = if cfg!(any(target_os = "ios", target_os = "android")) {
    40 * 1024 * 1024
} else {
    96 * 1024 * 1024
};
/// The largest font file a scan looks in.
const MAX_FILE_BYTES: u64 = 1024 * 1024 * 1024;
/// Faces new text should start in, best first. A scan puts the first of
/// these it finds ahead of the rest.
pub const PREFERRED_FACES: &[&str] = &[
    "Helvetica.ttf",
    "Helvetica.ttc#0",
    "HelveticaNeue.ttf",
    "HelveticaNeue.ttc#0",
    "Arial.ttf",
    "SFNS.ttf",
];
/// The most faces looked in for a character a title's face cannot draw.
pub const MAX_FALLBACKS: usize = 64;
/// The most characters whose fallback face is remembered.
const MAX_REMEMBERED: usize = 4_096;
/// The files a scan takes fallback faces from, best first: emoji, then
/// the faces macOS, Windows, Linux and Android keep for scripts their
/// usual text face lacks, a face with OpenType shaping ahead of one
/// without. Names are matched without regard to case; of a collection the
/// first face is taken.
pub const FALLBACK_FACES: &[&str] = &[
    "Apple Color Emoji.ttc",
    "seguiemj.ttf",
    "NotoColorEmoji.ttf",
    "NotoColorEmoji-Regular.ttf",
    // Arabic and Hebrew.
    "SFArabic.ttf",
    "Arial.ttf",
    "Times New Roman.ttf",
    "segoeui.ttf",
    "tahoma.ttf",
    "NotoNaskhArabic-Regular.ttf",
    "NotoSansArabic-Regular.ttf",
    "NotoSansHebrew-Regular.ttf",
    "SFHebrew.ttf",
    "ArialHB.ttc",
    // The Mac's own Arabic face, for what the faces above lack: the
    // phrases Arabic writes as one glyph, and the letters of its other
    // languages.
    "GeezaPro.ttc",
    // The scripts of India.
    "Kohinoor.ttc",
    "KohinoorBangla.ttc",
    "KohinoorGujarati.ttc",
    "KohinoorTelugu.ttc",
    "Tamil Sangam MN.ttc",
    "NotoSansKannada.ttc",
    "NotoSansOriya.ttc",
    "Gurmukhi MN.ttc",
    "Nirmala.ttc",
    "Nirmala.ttf",
    "NotoSansDevanagari-Regular.ttf",
    "NotoSansBengali-Regular.ttf",
    "NotoSansGujarati-Regular.ttf",
    "NotoSansGurmukhi-Regular.ttf",
    "NotoSansTamil-Regular.ttf",
    "NotoSansTelugu-Regular.ttf",
    "NotoSansKannada-Regular.ttf",
    "NotoSansMalayalam-Regular.ttf",
    "NotoSansOriya-Regular.ttf",
    // Chinese, Japanese and Korean.
    "PingFang.ttc",
    "Hiragino Sans GB.ttc",
    "ヒラギノ角ゴシック W3.ttc",
    "AppleSDGothicNeo.ttc",
    "msyh.ttc",
    "YuGothR.ttc",
    "meiryo.ttc",
    "malgun.ttf",
    "msjh.ttc",
    "simsun.ttc",
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJKsc-Regular.otf",
    "NotoSansCJKjp-Regular.otf",
    "NotoSansCJKkr-Regular.otf",
    "DroidSansFallback.ttf",
    // Thai and the rest.
    "Thonburi.ttc",
    "LeelawUI.ttf",
    "NotoSansThai-Regular.ttf",
    "Arial Unicode.ttf",
    "NotoSans-Regular.ttf",
    "Roboto-Regular.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "Helvetica.ttc",
    "Apple Symbols.ttf",
    "seguisym.ttf",
    "NotoSansSymbols-Regular.ttf",
    "NotoSansSymbols2-Regular.ttf",
    "Menlo.ttc",
];
/// The fallback faces that hold a little of very many scripts, and
/// signs and pictures. For a letter they are looked in after any face
/// the machine has for the letter's script: they have its glyphs, and
/// often not what joins and stacks them.
const GENERAL_FACES: &[&str] = &[
    "Arial Unicode.ttf",
    "NotoSans-Regular.ttf",
    "Roboto-Regular.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "Helvetica.ttc",
    "Apple Symbols.ttf",
    "seguisym.ttf",
    "NotoSansSymbols-Regular.ttf",
    "NotoSansSymbols2-Regular.ttf",
    "Menlo.ttc",
];
/// How many folders deep a scan looks below each folder it is given.
const MAX_DEPTH: usize = 3;

enum Bytes {
    Read(Vec<u8>),
    /// A face that is part of the program.
    Fixed(&'static [u8]),
    /// Read on the first ask, face `index` copied out of a collection. `None`
    /// once it failed to read as one face.
    Later(Source, OnceLock<Option<Vec<u8>>>),
    /// A named style of a variable face that is part of the program: the
    /// face, the setting of each axis, and the face with that setting
    /// written into it, made on the first ask.
    Pinned(
        &'static [u8],
        Vec<(cutlass_font::Tag, f32)>,
        OnceLock<Option<Vec<u8>>>,
    ),
}

/// Where a face found by a scan is read from: its file, which face of a
/// collection it is, and for a named instance of a variable face, the
/// setting of each axis it is pinned to.
struct Source {
    path: PathBuf,
    index: Option<u32>,
    pinned: Option<Vec<(cutlass_font::Tag, f32)>>,
}

struct Face {
    name: String,
    label: String,
    bytes: Bytes,
    style: Style,
    /// Which characters the face has, read alone from its file the first
    /// time a character no fallback face draws is looked for. `None` once
    /// it failed to read.
    map: OnceLock<Option<cutlass_font::CharacterMap>>,
    /// One of [`GENERAL_FACES`], or another face of such a file: its
    /// light and its bold hold as much.
    general: bool,
    /// Neither bold nor italic, nor any other cut its label names: a
    /// face's label may say what its tables do not.
    plain: bool,
    /// A face of emoji, by its name.
    emoji: bool,
    /// Where the face stands among the faces that ship with the program,
    /// which are given best first for standing in for a letter another
    /// face lacks; the greatest number there is for any other face.
    shipped: usize,
}

impl Face {
    /// Whether the face has a glyph for `character`, by its character
    /// map alone.
    fn has(&self, character: char) -> bool {
        let map = self.map.get_or_init(|| match &self.bytes {
            Bytes::Read(bytes) => character_map(bytes),
            Bytes::Fixed(bytes) | Bytes::Pinned(bytes, ..) => character_map(bytes),
            Bytes::Later(Source { path, index, .. }, _) => {
                let mut file = File::open(path).ok()?;
                cutlass_font::load_character_map(
                    |offset, len| {
                        if len > cutlass_font::MAX_CHARACTER_MAP {
                            return None;
                        }
                        file.seek(SeekFrom::Start(offset)).ok()?;
                        let mut bytes = vec![0; len];
                        file.read_exact(&mut bytes).ok()?;
                        Some(bytes)
                    },
                    *index,
                )
                .ok()
            }
        });
        map.as_ref().is_some_and(|map| map.has(character))
    }
}

/// The character map of a face held as bytes.
fn character_map(bytes: &[u8]) -> Option<cutlass_font::CharacterMap> {
    cutlass_font::load_character_map(
        |offset, len| {
            let start = usize::try_from(offset).ok()?;
            bytes
                .get(start..start.checked_add(len)?)
                .map(<[u8]>::to_vec)
        },
        None,
    )
    .ok()
}

/// The family a face belongs to and which of its faces it is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Style {
    /// The family's name in lower case; empty when the face names none.
    family: String,
    bold: bool,
    italic: bool,
}

impl Style {
    fn of(entry: &cutlass_font::FaceEntry) -> Self {
        Self {
            family: entry
                .family
                .as_deref()
                .map(str::to_lowercase)
                .unwrap_or_default(),
            bold: entry.bold,
            italic: entry.italic,
        }
    }

    /// The style of a face held as bytes.
    fn of_bytes(bytes: &[u8]) -> Self {
        cutlass_font::list_faces(|offset, len| {
            let start = usize::try_from(offset).ok()?;
            bytes
                .get(start..start.checked_add(len)?)
                .map(<[u8]>::to_vec)
        })
        .ok()
        .and_then(|faces| faces.first().map(Self::of))
        .unwrap_or_default()
    }
}

/// A face found by a scan, before the table keeps it.
struct Found {
    name: String,
    label: String,
    source: Source,
    style: Style,
}

/// Faces by name. An empty table supplies nothing.
#[derive(Default)]
pub struct FaceTable {
    faces: Vec<Face>,
    /// The faces looked in for a character a title's face lacks, by their
    /// place in `faces`, best first.
    fallbacks: Vec<usize>,
    /// The fallback face found for a character, or that there is none.
    found: Mutex<HashMap<(char, bool), Option<usize>>>,
}

impl FaceTable {
    /// Every path that is a single readable face, in that order, read now.
    /// A second file with the same name is skipped.
    #[must_use]
    pub fn load_from<P: AsRef<Path>>(paths: &[P]) -> Self {
        let mut table = Self::default();
        for path in paths {
            let path = path.as_ref();
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            if cutlass_font::read(&bytes).is_err() {
                continue;
            }
            if let Some(name) = file_name(path) {
                let style = Style::of_bytes(&bytes);
                table.push(name.clone(), face_label(&name), Bytes::Read(bytes), style);
            }
        }
        table.fallbacks = (0..table.faces.len().min(MAX_FALLBACKS)).collect();
        table
    }

    /// Faces given as bytes, in order. Bytes that are not one face are kept
    /// out.
    #[must_use]
    pub fn from_pairs(faces: Vec<(String, Vec<u8>)>) -> Self {
        let mut table = Self::default();
        for (name, bytes) in faces {
            if cutlass_font::read(&bytes).is_ok() {
                let style = Style::of_bytes(&bytes);
                table.push(name.clone(), face_label(&name), Bytes::Read(bytes), style);
            }
        }
        table.fallbacks = (0..table.faces.len().min(MAX_FALLBACKS)).collect();
        table
    }

    /// This table with the faces that are part of the program, each a
    /// file name and its bytes, after the faces it has. A name the table
    /// already has is the table's: a person's own file of that name is
    /// the one used. They are the last faces a character is looked for
    /// in, so a system with no face for it still has one, and a system
    /// with no fonts at all still has a face to set text in.
    #[must_use]
    pub fn with_built_in(mut self, faces: &[(&str, &'static [u8])]) -> Self {
        for (name, bytes) in faces {
            if self.faces.iter().any(|face| face.name == *name)
                || cutlass_font::read(bytes).is_err()
            {
                continue;
            }
            let style = Style::of_bytes(bytes);
            self.push(
                (*name).to_owned(),
                face_label(name),
                Bytes::Fixed(bytes),
                style,
            );
            self.fallbacks.push(self.faces.len() - 1);
        }
        self
    }

    /// This table with the faces of the families that ship with the
    /// program, each a file name and its bytes, listed among the faces it
    /// has by label. A face is labelled by the name it gives itself, and
    /// a face that varies is listed once for each style its maker named.
    /// A name or a label the table already has is the table's: a
    /// person's own copy of a family is the one used. These faces are
    /// not ones a missing character is looked for in first; they are
    /// looked in with every other face, and ahead of the others in the
    /// order they are given here, so the plainest should come first.
    #[must_use]
    pub fn with_bundled(mut self, files: &[(&str, &'static [u8])]) -> Self {
        for (place, (file, bytes)) in files.iter().enumerate() {
            let Ok(faces) = cutlass_font::list_faces(|offset, len| {
                let start = usize::try_from(offset).ok()?;
                bytes
                    .get(start..start.checked_add(len)?)
                    .map(<[u8]>::to_vec)
            }) else {
                continue;
            };
            // A file of one face: a collection is not pinned or handed
            // out whole.
            let [face] = &faces[..] else {
                continue;
            };
            if face.index.is_some() || cutlass_font::read(bytes).is_err() {
                continue;
            }
            let label = face.full_name.clone().unwrap_or_else(|| face_label(file));
            for listed in styles_of(file, &label, face) {
                let said = listed.label.to_lowercase();
                if self
                    .faces
                    .iter()
                    .any(|face| face.label.to_lowercase() == said)
                {
                    continue;
                }
                let bytes = match listed.pinned {
                    Some(pinned) => Bytes::Pinned(bytes, pinned, OnceLock::new()),
                    None => Bytes::Fixed(bytes),
                };
                let before = self.faces.len();
                self.push(listed.name, listed.label, bytes, listed.style);
                if let Some(face) = self.faces.get_mut(before) {
                    face.shipped = place;
                }
            }
        }
        self.sort_by_label();
        self
    }

    /// Put the faces in the order of their labels, and keep the fallback
    /// faces pointing at the faces they were.
    fn sort_by_label(&mut self) {
        let mut order: Vec<usize> = (0..self.faces.len()).collect();
        order.sort_by(|a, b| {
            let (a, b) = (&self.faces[*a], &self.faces[*b]);
            a.label
                .to_lowercase()
                .cmp(&b.label.to_lowercase())
                .then_with(|| a.name.cmp(&b.name))
        });
        let mut now = vec![0; order.len()];
        for (to, from) in order.iter().enumerate() {
            now[*from] = to;
        }
        let mut faces: Vec<Option<Face>> = std::mem::take(&mut self.faces)
            .into_iter()
            .map(Some)
            .collect();
        self.faces = order
            .iter()
            .filter_map(|from| faces.get_mut(*from).and_then(Option::take))
            .collect();
        for fallback in &mut self.fallbacks {
            if let Some(to) = now.get(*fallback) {
                *fallback = *to;
            }
        }
        if let Ok(mut found) = self.found.lock() {
            found.clear();
        }
    }

    /// One face given as bytes.
    #[must_use]
    pub fn named(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self::from_pairs(vec![(name.into(), bytes)])
    }

    /// The TrueType and OpenType faces in `folders` and the folders below
    /// them, each face of a collection on its own, by label. At most
    /// [`MAX_FACES`] are kept, and a face over [`MAX_FACE_BYTES`] gives no
    /// bytes when it is asked for.
    /// Only headers and names are read now. The first of
    /// [`PREFERRED_FACES`] found comes first.
    #[must_use]
    pub fn scan<P: AsRef<Path>>(folders: &[P]) -> Self {
        let mut found = Vec::new();
        for folder in folders {
            walk(folder.as_ref(), 0, &mut found);
        }
        found.sort_by(|a, b| {
            a.label
                .to_lowercase()
                .cmp(&b.label.to_lowercase())
                .then_with(|| a.name.cmp(&b.name))
        });
        if let Some(at) = PREFERRED_FACES
            .iter()
            .find_map(|preferred| found.iter().position(|face| face.name == *preferred))
        {
            let face = found.remove(at);
            found.insert(0, face);
        }
        let mut table = Self::default();
        for face in found {
            // Past the limit only the faces other scripts and emoji fall
            // back to are kept, so a long font folder does not lose them.
            if table.faces.len() >= MAX_FACES && !is_fallback_file(&face.name) {
                continue;
            }
            let bytes = Bytes::Later(face.source, OnceLock::new());
            table.push(face.name, face.label, bytes, face.style);
        }
        table.rank_fallbacks();
        table
    }

    /// This table with the first of `names` it holds moved to the front,
    /// where new text starts.
    #[must_use]
    pub fn preferring(mut self, names: &[&str]) -> Self {
        let Some(at) = names
            .iter()
            .find_map(|name| self.faces.iter().position(|face| face.name == *name))
        else {
            return self;
        };
        let face = self.faces.remove(at);
        self.faces.insert(0, face);
        // The faces before it each moved one place down.
        for fallback in &mut self.fallbacks {
            *fallback = match (*fallback).cmp(&at) {
                std::cmp::Ordering::Less => *fallback + 1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => *fallback,
            };
        }
        if let Ok(mut found) = self.found.lock() {
            found.clear();
        }
        self
    }

    /// Choose a scanned table's fallback faces: the ones [`FALLBACK_FACES`]
    /// names, in its order, then any other Noto face, which between them
    /// cover most scripts on a system that has them.
    fn rank_fallbacks(&mut self) {
        // The file a face is from, when it is the file's only or first face.
        let file = |name: &str| match name.split_once('#') {
            None => Some(fold(name)),
            Some((file, "0")) => Some(fold(file)),
            Some(_) => None,
        };
        let files: Vec<Option<String>> = self.faces.iter().map(|face| file(&face.name)).collect();
        let mut ranked: Vec<usize> = Vec::new();
        for wanted in FALLBACK_FACES {
            let wanted = fold(wanted);
            if let Some(at) = files
                .iter()
                .position(|file| file.as_deref() == Some(&wanted))
            {
                ranked.push(at);
            }
        }
        for (at, file) in files.iter().enumerate() {
            let noto = file
                .as_deref()
                .is_some_and(|file| file.starts_with("noto") && file.contains("regular"));
            if noto && !ranked.contains(&at) {
                ranked.push(at);
            }
        }
        ranked.truncate(MAX_FALLBACKS);
        self.fallbacks = ranked;
    }

    /// Whether the face at `at` is one of [`GENERAL_FACES`].
    fn general(&self, at: usize) -> bool {
        self.faces.get(at).is_some_and(|face| face.general)
    }

    /// Every other face that has `character`, for one no fallback face
    /// draws: a script the list knows no face for is drawn in whatever
    /// face the machine has for it. Each face is asked through its
    /// character map alone, read once; a plain face comes first, of the
    /// plain ones a face that ships with the program in the order those
    /// were given, and an emoji face only for an emoji.
    fn elsewhere(&self, character: char, emoji: bool) -> Vec<usize> {
        let mut others: Vec<usize> = self
            .faces
            .iter()
            .enumerate()
            .filter(|(at, face)| {
                !face.general
                    && (emoji || !face.emoji)
                    && !self.fallbacks.contains(at)
                    && face.has(character)
            })
            .map(|(at, _)| at)
            .collect();
        others.sort_by_key(|at| {
            let face = &self.faces[*at];
            (!face.plain, face.shipped, *at)
        });
        others
    }

    /// The axes the face `text` is set in varies along, each with where
    /// `text` has it: weight, width, slant and whatever else the face's
    /// maker gave it. None for a face that does not vary or that the
    /// table does not have; an axis the maker hid, or one with nowhere to
    /// go, is left out.
    #[must_use]
    pub fn font_axes(&self, text: &cutlass_model::MaterialText) -> Vec<FontAxis> {
        let Some(font) = self.set_as(text) else {
            return Vec::new();
        };
        font.axes()
            .into_iter()
            .zip(font.settings())
            .filter(|(axis, _)| !axis.hidden && axis.max > axis.min)
            .map(|(axis, (_, value))| FontAxis {
                name: axis.name.unwrap_or_else(|| axis_name(axis.tag)),
                tag: axis.tag,
                min: axis.min,
                default: axis.default,
                max: axis.max,
                value,
            })
            .collect()
    }

    /// What `text`'s setting of its face becomes with the axis `tag` at
    /// `value` and every other axis where it is, written as the text
    /// keeps it; `None` when that is what it already has. Every axis is
    /// written, so the face is where the title shows it whatever style
    /// it started from, and a value past the axis's end is the end.
    pub fn moved_font_axis(
        &self,
        text: &cutlass_model::MaterialText,
        tag: [u8; 4],
        value: f64,
    ) -> Result<Option<String>, AxisError> {
        let font = self.set_as(text).ok_or(AxisError::NoAxes)?;
        let before = font.settings();
        if !before.iter().any(|(have, _)| *have == tag) {
            return Err(AxisError::NoSuchAxis);
        }
        #[allow(clippy::cast_possible_truncation)]
        let wanted = value as f32;
        if !wanted.is_finite() {
            return Err(AxisError::NotANumber);
        }
        let after: Vec<([u8; 4], f32)> = before
            .iter()
            .zip(font.axes())
            .map(|((have, at), axis)| {
                let at = if *have == tag {
                    wanted.clamp(axis.min, axis.max)
                } else {
                    *at
                };
                (*have, at)
            })
            .collect();
        let written = cutlass_model::write_font_axes(&after);
        Ok((written != cutlass_model::write_font_axes(&before)).then_some(written))
    }

    /// The face `text` is set in, as `text` has it set along its axes;
    /// `None` for a face that does not vary.
    fn set_as(&self, text: &cutlass_model::MaterialText) -> Option<cutlass_font::Font<'_>> {
        let mut font = cutlass_font::read(self.face(&text.font_path)?).ok()?;
        if !font.is_variable() {
            return None;
        }
        let axes = text.font_axes();
        if !axes.is_empty() {
            font.set_variations(&axes).ok()?;
        }
        Some(font)
    }

    /// The names of the fallback faces, best first.
    #[must_use]
    pub fn fallback_names(&self) -> Vec<String> {
        self.fallbacks
            .iter()
            .filter_map(|at| self.faces.get(*at))
            .map(|face| face.name.clone())
            .collect()
    }

    /// The bytes of the face at `at`, read on the first ask.
    fn bytes_at(&self, at: usize) -> Option<&[u8]> {
        match &self.faces.get(at)?.bytes {
            Bytes::Read(bytes) => Some(bytes),
            Bytes::Fixed(bytes) => Some(bytes),
            Bytes::Later(source, cell) => cell.get_or_init(|| read_face(source)).as_deref(),
            Bytes::Pinned(bytes, pinned, cell) => cell
                .get_or_init(|| cutlass_font::pin_instance(bytes, pinned).ok())
                .as_deref(),
        }
    }

    /// The first face, which new text starts in. An empty table has an
    /// empty name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.faces.first().map_or("", |face| face.name.as_str())
    }

    /// Names, in order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.faces.iter().map(|face| face.name.clone()).collect()
    }

    /// Names with their labels for people, in order.
    #[must_use]
    pub fn labelled(&self) -> Vec<(String, String)> {
        self.faces
            .iter()
            .map(|face| (face.name.clone(), face.label.clone()))
            .collect()
    }

    fn push(&mut self, name: String, label: String, bytes: Bytes, style: Style) {
        if name.is_empty() || self.faces.iter().any(|face| face.name == name) {
            return;
        }
        const CUTS: [&str; 12] = [
            "bold",
            "italic",
            "oblique",
            "black",
            "heavy",
            "light",
            "thin",
            "medium",
            "condensed",
            "narrow",
            "extended",
            "outline",
        ];
        let file = fold(name.split_once('#').map_or(&name, |(file, _)| file));
        let said = label.to_lowercase();
        self.faces.push(Face {
            general: GENERAL_FACES.iter().any(|general| fold(general) == file),
            plain: !(style.bold || style.italic || CUTS.iter().any(|cut| said.contains(cut))),
            emoji: file.contains("emoji"),
            shipped: usize::MAX,
            name,
            label,
            bytes,
            style,
            map: OnceLock::new(),
        });
    }
}

/// Faces made for Japanese, which come first for kana: a face made for
/// Chinese has kana too, drawn beside Han characters in the Chinese way.
const JAPANESE_FACES: &[&str] = &[
    "ヒラギノ角ゴシック W3.ttc",
    "YuGothR.ttc",
    "meiryo.ttc",
    "msgothic.ttc",
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJKjp-Regular.otf",
    "NotoSansJP-Regular.otf",
    "NotoSansJP-Regular.ttf",
];

/// Faces made for Korean, which come first for Hangul.
const KOREAN_FACES: &[&str] = &[
    "AppleSDGothicNeo.ttc",
    "malgun.ttf",
    "NotoSansCJKkr-Regular.otf",
    "NotoSansKR-Regular.otf",
    "NotoSansKR-Regular.ttf",
];

/// The faces that come first for `character`: those of its language,
/// where the script says which that is.
fn leading(character: char) -> &'static [&'static str] {
    match u32::from(character) {
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => JAPANESE_FACES,
        0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7FF => KOREAN_FACES,
        _ => &[],
    }
}

impl Fonts for FaceTable {
    fn face(&self, path: &str) -> Option<&[u8]> {
        let at = self.faces.iter().position(|face| face.name == path)?;
        self.bytes_at(at)
    }

    fn fallback(&self, character: char, emoji: bool) -> Option<&[u8]> {
        let key = (character, emoji);
        if let Ok(found) = self.found.lock()
            && let Some(known) = found.get(&key)
        {
            return known.and_then(|at| self.bytes_at(at));
        }
        // Emoji faces first for an emoji, and last for anything else.
        let is_emoji = |at: &usize| self.faces.get(*at).is_some_and(|face| face.emoji);
        let (first, second): (Vec<usize>, Vec<usize>) =
            self.fallbacks.iter().partition(|at| is_emoji(at) == emoji);
        // And before those, the faces of the character's own language.
        let leading = leading(character);
        let leads = |at: &usize| {
            self.faces.get(*at).is_some_and(|face| {
                // A collection's first face is named with `#0` after the file.
                let file = fold(face.name.strip_suffix("#0").unwrap_or(&face.name));
                leading.iter().any(|name| fold(name) == file)
            })
        };
        let (own, rest): (Vec<usize>, Vec<usize>) =
            first.into_iter().chain(second).partition(|at| leads(at));
        // A face is asked through its character map alone, and only the
        // one that has the character is read.
        let draws = |at: &usize| {
            self.faces.get(*at).is_some_and(|face| face.has(character))
                && self
                    .bytes_at(*at)
                    .and_then(|bytes| cutlass_font::read(bytes).ok())
                    .is_some_and(|font| matches!(font.glyph_id(character), Ok(Some(_))))
        };
        // The faces that hold a little of every script come after a face
        // the machine has for the character's own script, which shapes
        // it and has its signs to match; for a sign or a picture many
        // scripts share they come first, as the plainest, and so they do
        // for Latin, Greek and Cyrillic, where plain is what is wanted.
        let (general, known): (Vec<usize>, Vec<usize>) = own
            .into_iter()
            .chain(rest)
            .partition(|at| self.general(*at));
        let own_script = !matches!(
            &cutlass_text::script_tag(character),
            b"Zyyy" | b"Zinh" | b"Zzzz" | b"Latn" | b"Grek" | b"Cyrl"
        );
        let hit = if own_script {
            known
                .into_iter()
                .find(draws)
                .or_else(|| self.elsewhere(character, emoji).into_iter().find(draws))
                .or_else(|| general.into_iter().find(draws))
        } else {
            known
                .into_iter()
                .chain(general)
                .find(draws)
                .or_else(|| self.elsewhere(character, emoji).into_iter().find(draws))
        };
        if let Ok(mut found) = self.found.lock()
            && found.len() < MAX_REMEMBERED
        {
            found.insert(key, hit);
        }
        hit.and_then(|at| self.bytes_at(at))
    }

    fn styled(&self, path: &str, bold: bool, italic: bool) -> Option<(&[u8], bool, bool)> {
        let own = &self.faces.iter().find(|face| face.name == path)?.style;
        if own.family.is_empty() {
            return None;
        }
        // What the letters should be, counting what the face already is.
        let (bold, italic) = (bold || own.bold, italic || own.italic);
        let worth = |style: &Style| u8::from(style.bold) + u8::from(style.italic);
        // The family's face with the most of what is asked and nothing
        // that is not, when that is more than the face itself has.
        let (at, best) = self
            .faces
            .iter()
            .enumerate()
            .filter(|(_, face)| {
                face.style.family == own.family
                    && (bold || !face.style.bold)
                    && (italic || !face.style.italic)
            })
            .max_by_key(|(at, face)| (worth(&face.style), std::cmp::Reverse(*at)))?;
        if worth(&best.style) <= worth(own) {
            return None;
        }
        self.bytes_at(at)
            .map(|bytes| (bytes, best.style.bold, best.style.italic))
    }
}

/// One axis a title's face varies along, for a slider.
#[derive(Clone, Debug, PartialEq)]
pub struct FontAxis {
    pub tag: [u8; 4],
    /// The axis's name, from the face or, where it gives none, from its
    /// tag.
    pub name: String,
    pub min: f32,
    pub default: f32,
    pub max: f32,
    /// Where the title has the face along it.
    pub value: f32,
}

/// Why a title's face was not moved along an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisError {
    /// The face does not vary, or the table does not have it.
    NoAxes,
    /// The face has no axis by that tag.
    NoSuchAxis,
    /// The value is no number.
    NotANumber,
}

impl std::fmt::Display for AxisError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NoAxes => "this face has no axes to set",
            Self::NoSuchAxis => "this face has no such axis",
            Self::NotANumber => "that is not a number",
        })
    }
}

impl std::error::Error for AxisError {}

/// What an axis is called when its face does not say: the names of the
/// axes every maker uses, and the tag itself for any other.
fn axis_name(tag: [u8; 4]) -> String {
    match &tag {
        b"wght" => "Weight".to_owned(),
        b"wdth" => "Width".to_owned(),
        b"slnt" => "Slant".to_owned(),
        b"ital" => "Italic".to_owned(),
        b"opsz" => "Optical size".to_owned(),
        _ => String::from_utf8_lossy(&tag).into_owned(),
    }
}

/// The faces of `faces`, each a name and its label, whose labels have
/// every word of `query`, in the order they were given; all of them for
/// no words. Case does not matter.
#[must_use]
pub fn matching_faces<'f>(faces: &'f [(String, String)], query: &str) -> Vec<&'f (String, String)> {
    let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
    faces
        .iter()
        .filter(|(_, label)| {
            let label = label.to_lowercase();
            words.iter().all(|word| label.contains(word.as_str()))
        })
        .collect()
}

/// Whether a scanned face's file is one of [`FALLBACK_FACES`], as its only
/// or its first face.
fn is_fallback_file(name: &str) -> bool {
    let file = match name.split_once('#') {
        None => name,
        Some((file, "0")) => file,
        Some(_) => return false,
    };
    let file = fold(file);
    FALLBACK_FACES.iter().any(|wanted| fold(wanted) == file)
}

/// A file name as names are compared: in lower case, and with a kana's
/// voicing mark left off however it is written. A Mac stores such a mark
/// as a character of its own after the kana, and other systems as one
/// character with it, so the two spellings of a name are one name here.
fn fold(name: &str) -> String {
    // The kana that take a voicing mark as the next character, and a
    // half-voicing mark as the one after that, in both kana blocks.
    const VOICED: &str =
        "かきくけこさしすせそたちつてとはひふへほカキクケコサシスセソタチツテトハヒフヘホ";
    const HALF_VOICED: &str = "はひふへほハヒフヘホ";
    let base = |character: char, back: u32, bases: &str| {
        char::from_u32(u32::from(character).wrapping_sub(back)).filter(|base| bases.contains(*base))
    };
    name.chars()
        .filter(|character| !matches!(character, '\u{3099}' | '\u{309A}'))
        .map(|character| {
            base(character, 1, VOICED)
                .or_else(|| base(character, 2, HALF_VOICED))
                .unwrap_or(character)
        })
        .flat_map(char::to_lowercase)
        .collect()
}

/// A face's name for people: the file name without its extension, with
/// dashes and underscores as spaces.
#[must_use]
pub fn face_label(name: &str) -> String {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(name);
    stem.replace(['-', '_'], " ")
}

fn walk(folder: &Path, depth: usize, found: &mut Vec<Found>) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_dir() {
            if depth < MAX_DEPTH {
                walk(&path, depth + 1, found);
            }
        } else if kind.is_file() {
            faces_in(&path, found);
        }
    }
}

/// The faces of a `.ttf`, `.otf` or `.ttc` file under the size cap. A file
/// that does not read as fonts adds nothing, and neither does a hidden file
/// or a face whose name starts with a dot, which systems keep for their own
/// interface.
fn faces_in(path: &Path, found: &mut Vec<Found>) {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(extension.as_deref(), Some("ttf" | "otf" | "ttc")) {
        return;
    }
    let Some(name) = file_name(path).filter(|name| !name.starts_with('.')) else {
        return;
    };
    let Ok(mut file) = File::open(path) else {
        return;
    };
    if file
        .metadata()
        .map_or(true, |meta| meta.len() > MAX_FILE_BYTES)
    {
        return;
    }
    let Ok(faces) = cutlass_font::list_faces(|offset, len| {
        file.seek(SeekFrom::Start(offset)).ok()?;
        let mut bytes = vec![0; len];
        file.read_exact(&mut bytes).ok()?;
        Some(bytes)
    }) else {
        return;
    };
    for face in faces {
        if face
            .full_name
            .as_deref()
            .is_some_and(|full| full.starts_with('.'))
        {
            continue;
        }
        let label = match face.index {
            None => face_label(&name),
            Some(index) => face
                .full_name
                .clone()
                .unwrap_or_else(|| format!("{} {}", face_label(&name), index + 1)),
        };
        for listed in styles_of(&name, &label, &face) {
            found.push(Found {
                name: listed.name,
                label: listed.label,
                source: Source {
                    path: path.to_owned(),
                    index: face.index,
                    pinned: listed.pinned,
                },
                style: listed.style,
            });
        }
    }
}

/// One entry a face gives a table: the face itself, or one named style
/// of a face that varies, with the setting it is pinned to.
struct Listed {
    name: String,
    label: String,
    style: Style,
    pinned: Option<Vec<(cutlass_font::Tag, f32)>>,
}

/// The entries of `face`, a face of the file named `file` that is
/// labelled `label`. A variable face is listed once for each style its
/// maker named, as a family of separate files would be; any other face
/// once.
fn styles_of(file: &str, label: &str, face: &cutlass_font::FaceEntry) -> Vec<Listed> {
    let style = Style::of(face);
    let name = match face.index {
        None => file.to_owned(),
        Some(index) => format!("{file}#{index}"),
    };
    // The whole design's name, where the face gives one apart from the
    // family its default weight is filed under.
    let named = face.design_family.as_deref().or(face.family.as_deref());
    let family = named.map_or_else(|| label.to_owned(), str::to_owned);
    // A family the system keeps for its own interface, named with a
    // dot, lists no styles; the face stays for what falls back to it.
    let instances: &[cutlass_font::InstanceEntry] = if family.starts_with('.') {
        &[]
    } else {
        &face.instances
    };
    // A face that measures weight on a scale of its own does not say
    // which style is its bold: the one named so is.
    let named_bold = |instance: &cutlass_font::InstanceEntry| {
        let said = instance.name.to_lowercase();
        matches!(said.as_str(), "bold" | "bold italic" | "bold oblique")
    };
    let bold_said = |italic: bool| {
        instances
            .iter()
            .any(|instance| instance.bold && instance.italic == italic)
    };
    let mut listed: Vec<Listed> = instances
        .iter()
        .map(|instance| {
            let bold = instance.bold || (!bold_said(instance.italic) && named_bold(instance));
            let (name, label) = style_names(&name, label, named, &instance.name, instance.default);
            Listed {
                name,
                label,
                style: Style {
                    // The styles of one file are a family, whether or
                    // not the file names one.
                    family: if style.family.is_empty() {
                        family.to_lowercase()
                    } else {
                        style.family.clone()
                    },
                    bold,
                    italic: instance.italic,
                },
                pinned: Some(instance.coordinates.clone()),
            }
        })
        .collect();
    // The face as it is, when no named style is the one it draws with
    // nothing set.
    if !instances.iter().any(|instance| instance.default) {
        listed.push(Listed {
            name,
            label: label.to_owned(),
            style,
            pinned: None,
        });
    }
    listed
}

/// The name and the label of one named style of a variable face, whose
/// own are `name` and `label`. The style the file draws with nothing set
/// keeps the face's name, so a title made before the others were listed
/// still finds it; each other adds the style's name after an `@`. A
/// style is labelled by the family and its own name, and with no family
/// to go by the default keeps the face's label.
fn style_names(
    name: &str,
    label: &str,
    family: Option<&str>,
    style: &str,
    default: bool,
) -> (String, String) {
    let styled = format!("{} {style}", family.unwrap_or(label));
    match (default, family) {
        (true, Some(_)) => (name.to_owned(), styled),
        (true, None) => (name.to_owned(), label.to_owned()),
        (false, _) => (format!("{name}@{style}"), styled),
    }
}

/// One face of a file as its own font, when it reads as one and is no
/// larger than [`MAX_FACE_BYTES`]; a named instance of a variable face
/// with the setting it is pinned to written into it.
fn read_face(source: &Source) -> Option<Vec<u8>> {
    let mut file = File::open(&source.path).ok()?;
    let length = file.metadata().ok()?.len();
    let most = usize::try_from(MAX_FACE_BYTES).ok()?;
    let read_at = |offset: u64, len: usize| {
        // A read past the end, or of more than a face may hold, is
        // refused before any bytes are asked for.
        if len > most || offset.checked_add(len as u64)? > length {
            return None;
        }
        file.seek(SeekFrom::Start(offset)).ok()?;
        let mut bytes = vec![0; len];
        file.read_exact(&mut bytes).ok()?;
        Some(bytes)
    };
    match &source.pinned {
        Some(pinned) => cutlass_font::load_instance(read_at, source.index, pinned, most),
        None => cutlass_font::load_face(read_at, source.index, most),
    }
    .ok()
}

fn file_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    (!name.contains(['/', '\\', '\0'])).then(|| name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const SAMPLE_TTC: &[u8] = include_bytes!("../../font/fixtures/sample.ttc");

    struct Folder(PathBuf);

    impl Folder {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cutlass-faces-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("folder");
            Self(path)
        }
    }

    impl Drop for Folder {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const SHAPING: &[u8] = include_bytes!("../../font/fixtures/shaping.ttf");

    #[test]
    fn a_character_is_found_in_the_first_fallback_face_that_draws_it() {
        // The sample face has A and B; the shaping face has f and more.
        let table = FaceTable::from_pairs(vec![
            ("Sample.ttf".to_owned(), SAMPLE.to_vec()),
            ("Shaping.ttf".to_owned(), SHAPING.to_vec()),
        ]);
        assert_eq!(table.fallback_names(), ["Sample.ttf", "Shaping.ttf"]);
        assert_eq!(table.fallback('A', false), table.face("Sample.ttf"));
        assert_eq!(table.fallback('f', false), table.face("Shaping.ttf"));
        assert!(table.fallback('Q', false).is_none());
        // Asked again, the answer is the one remembered.
        assert_eq!(table.fallback('f', false), table.face("Shaping.ttf"));
        assert!(table.fallback('Q', false).is_none());
        assert!(FaceTable::default().fallback('A', false).is_none());
    }

    /// The sample face made a member of a family: its family name, and
    /// whether it says it is bold or italic.
    fn styled_face(family: &str, bold: bool, italic: bool) -> Vec<u8> {
        // Add a `name` and an `OS/2` table to the sample face.
        let count = usize::from(u16::from_be_bytes([SAMPLE[4], SAMPLE[5]]));
        let mut tables: Vec<([u8; 4], Vec<u8>)> = (0..count)
            .map(|slot| {
                let at = 12 + slot * 16;
                let word = |from: usize| {
                    u32::from_be_bytes([
                        SAMPLE[from],
                        SAMPLE[from + 1],
                        SAMPLE[from + 2],
                        SAMPLE[from + 3],
                    ]) as usize
                };
                let (start, length) = (word(at + 8), word(at + 12));
                let mut tag = [0; 4];
                tag.copy_from_slice(&SAMPLE[at..at + 4]);
                (tag, SAMPLE[start..start + length].to_vec())
            })
            .filter(|(tag, _)| tag != b"name" && tag != b"OS/2")
            .collect();
        let encoded: Vec<u8> = family.encode_utf16().flat_map(u16::to_be_bytes).collect();
        let mut name = Vec::new();
        for value in [0_u16, 1, 18, 3, 1, 0x0409, 1, encoded.len() as u16, 0] {
            name.extend(value.to_be_bytes());
        }
        name.extend(encoded);
        let mut os2 = vec![0_u8; 78];
        os2[62..64].copy_from_slice(&(u16::from(italic) | (u16::from(bold) << 5)).to_be_bytes());
        tables.push((*b"name", name));
        tables.push((*b"OS/2", os2));
        tables.sort_by_key(|table| table.0);
        let mut file = vec![0, 1, 0, 0];
        file.extend((tables.len() as u16).to_be_bytes());
        file.extend([0; 6]);
        let mut at = 12 + tables.len() * 16;
        let mut body = Vec::new();
        for (tag, bytes) in &tables {
            file.extend(tag);
            file.extend([0; 4]);
            file.extend((at as u32).to_be_bytes());
            file.extend((bytes.len() as u32).to_be_bytes());
            body.extend(bytes);
            let padded = bytes.len().next_multiple_of(4);
            body.resize(body.len() + padded - bytes.len(), 0);
            at += padded;
        }
        file.extend(body);
        file
    }

    #[test]
    fn a_preferred_face_moves_to_the_front_and_fallbacks_follow_it() {
        let table = FaceTable::from_pairs(vec![
            ("Sample.ttf".to_owned(), SAMPLE.to_vec()),
            ("Other.ttf".to_owned(), SAMPLE.to_vec()),
            ("Shaping.ttf".to_owned(), SHAPING.to_vec()),
        ])
        .preferring(&["Missing.ttf", "Shaping.ttf", "Other.ttf"]);
        assert_eq!(table.name(), "Shaping.ttf");
        assert_eq!(table.names(), ["Shaping.ttf", "Sample.ttf", "Other.ttf"]);
        // The fallback order is the one it had, by name.
        assert_eq!(
            table.fallback_names(),
            ["Sample.ttf", "Other.ttf", "Shaping.ttf"]
        );
        assert_eq!(table.fallback('f', false), table.face("Shaping.ttf"));
        let same = FaceTable::named("Sample.ttf", SAMPLE.to_vec()).preferring(&["Nope.ttf"]);
        assert_eq!(same.name(), "Sample.ttf");
    }

    #[test]
    fn a_family_gives_its_bold_and_italic_faces() {
        let table = FaceTable::from_pairs(vec![
            ("Fam.ttf".to_owned(), styled_face("Fam", false, false)),
            ("Fam-Bold.ttf".to_owned(), styled_face("Fam", true, false)),
            (
                "Fam-BoldItalic.ttf".to_owned(),
                styled_face("Fam", true, true),
            ),
            ("Other.ttf".to_owned(), styled_face("Other", false, true)),
            ("Plain.ttf".to_owned(), SAMPLE.to_vec()),
        ]);
        let named = |name: &str| table.face(name).map(<[u8]>::as_ptr);
        let styled = |path: &str, bold: bool, italic: bool| {
            table
                .styled(path, bold, italic)
                .map(|(bytes, bold, italic)| (Some(bytes.as_ptr()), bold, italic))
        };
        assert_eq!(
            styled("Fam.ttf", true, false),
            Some((named("Fam-Bold.ttf"), true, false))
        );
        assert_eq!(
            styled("Fam.ttf", true, true),
            Some((named("Fam-BoldItalic.ttf"), true, true))
        );
        // The family has no plain italic: nothing bolder than asked is
        // given, and the lean is left to be drawn.
        assert_eq!(styled("Fam.ttf", false, true), None);
        // A bold face asked for italic too goes to the bold italic.
        assert_eq!(
            styled("Fam-Bold.ttf", false, true),
            Some((named("Fam-BoldItalic.ttf"), true, true))
        );
        // Nothing to add, another family, no family, no such face.
        assert_eq!(styled("Fam-Bold.ttf", true, false), None);
        assert_eq!(styled("Fam.ttf", false, false), None);
        assert_eq!(styled("Other.ttf", true, false), None);
        assert_eq!(styled("Plain.ttf", true, true), None);
        assert_eq!(styled("Missing.ttf", true, false), None);
    }

    #[test]
    fn a_scan_ranks_the_fallback_faces_it_knows_and_emoji_come_first_for_emoji() {
        let folder = Folder::new("fallback");
        for name in [
            "Zeta.ttf",
            "NotoSansFoo-Regular.ttf",
            "NotoSansFoo-Bold.ttf",
            "arial.ttf",
            "NotoColorEmoji.ttf",
        ] {
            std::fs::write(folder.0.join(name), SHAPING).expect("face");
        }
        std::fs::write(folder.0.join("Kohinoor.ttc"), SAMPLE_TTC).expect("collection");
        let table = FaceTable::scan(&[&folder.0]);
        // The listed faces in the list's order, a collection's first face
        // only, then other regular Noto faces; the rest are not fallbacks.
        assert_eq!(
            table.fallback_names(),
            [
                "NotoColorEmoji.ttf",
                "arial.ttf",
                "Kohinoor.ttc#0",
                "NotoSansFoo-Regular.ttf"
            ]
        );
        // Every face here draws f: a letter comes from the first face
        // that is not an emoji face, an emoji from the emoji face.
        assert_eq!(table.fallback('f', false), table.face("arial.ttf"));
        assert_eq!(table.fallback('f', true), table.face("NotoColorEmoji.ttf"));
    }

    const THAI: &[u8] = include_bytes!("../../font/fixtures/thai.ttf");

    #[test]
    fn a_character_no_fallback_face_has_is_found_in_any_other_face() {
        let folder = Folder::new("elsewhere");
        // No name here is one the fallback list knows, but arial's.
        std::fs::write(folder.0.join("arial.ttf"), SHAPING).expect("face");
        std::fs::write(folder.0.join("Alpha.ttf"), SAMPLE).expect("face");
        std::fs::write(folder.0.join("Siam Bold.ttf"), THAI).expect("face");
        std::fs::write(folder.0.join("Siam.ttf"), THAI).expect("face");
        std::fs::write(folder.0.join("Zeta.ttc"), SAMPLE_TTC).expect("collection");
        std::fs::write(folder.0.join("Junk.ttf"), b"not a font").expect("junk");
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.fallback_names(), ["arial.ttf"]);
        // Ko kai is in neither arial nor Alpha: the Thai face has it, and
        // the plain one comes before the bold, whatever their order.
        assert_eq!(table.fallback('\u{E01}', false), table.face("Siam.ttf"));
        // A letter the fallback face has still comes from it.
        assert_eq!(table.fallback('f', false), table.face("arial.ttf"));
        // One only a face no list names has.
        assert_eq!(table.fallback('B', false), table.face("Alpha.ttf"));
        // No face has it: nothing, and nothing again.
        assert!(table.fallback('\u{1780}', false).is_none());
        assert!(table.fallback('\u{1780}', false).is_none());
        // A table of bytes looks through all it holds the same way.
        let table = FaceTable::from_pairs(vec![
            ("Sample.ttf".to_owned(), SAMPLE.to_vec()),
            ("Siam.ttf".to_owned(), THAI.to_vec()),
        ]);
        assert_eq!(table.fallback('\u{E01}', false), table.face("Siam.ttf"));
    }

    #[test]
    fn a_letter_comes_from_its_scripts_face_before_a_face_of_every_script() {
        let folder = Folder::new("general");
        // A face the list keeps for whatever nothing else draws, and a
        // face no list names, with the same glyphs.
        std::fs::write(folder.0.join("Arial Unicode.ttf"), THAI).expect("face");
        std::fs::write(folder.0.join("Siam.ttf"), THAI).expect("face");
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.fallback_names(), ["Arial Unicode.ttf"]);
        // A letter and a mark of the script are set in the face made for
        // it, which shapes them.
        assert_eq!(table.fallback('\u{E01}', false), table.face("Siam.ttf"));
        assert_eq!(table.fallback('\u{E48}', false), table.face("Siam.ttf"));
        // A letter of Latin, and a sign every script shares, are set in
        // the general face though another has them.
        let latin_folder = Folder::new("general-latin");
        std::fs::write(latin_folder.0.join("Arial Unicode.ttf"), SHAPING).expect("face");
        std::fs::write(latin_folder.0.join("Alpha.ttf"), SHAPING).expect("face");
        let latin = FaceTable::scan(&[&latin_folder.0]);
        assert_eq!(latin.fallback('f', false), latin.face("Arial Unicode.ttf"));
        assert_eq!(latin.fallback(' ', false), latin.face("Arial Unicode.ttf"));
        // With no other face for the letter, the general face draws it.
        std::fs::remove_file(folder.0.join("Siam.ttf")).expect("removed");
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(
            table.fallback('\u{E01}', false),
            table.face("Arial Unicode.ttf")
        );
    }

    #[test]
    fn a_long_folder_keeps_every_face_and_is_looked_through_in_time() {
        let folder = Folder::new("many");
        // More faces than a scan once kept, with the one that has the
        // letter last by name.
        for at in 0..600 {
            std::fs::write(folder.0.join(format!("Face {at:03}.ttf")), SAMPLE).expect("face");
        }
        std::fs::write(folder.0.join("Zz Siam.ttf"), THAI).expect("face");
        let started = std::time::Instant::now();
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.names().len(), 601);
        assert!(table.names().iter().any(|name| name == "Zz Siam.ttf"));
        assert_eq!(table.fallback('\u{E01}', false), table.face("Zz Siam.ttf"));
        // Every other character asks each face's map, already read.
        for code in 0x1780..0x17C0_u32 {
            let character = char::from_u32(code).expect("character");
            assert!(table.fallback(character, false).is_none());
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }

    const VARIABLE: &[u8] = include_bytes!("../../font/fixtures/variable.ttf");

    #[test]
    fn a_variable_face_is_listed_once_for_each_style_its_maker_named() {
        let folder = Folder::new("variable");
        std::fs::write(folder.0.join("Vary.ttf"), VARIABLE).expect("face");
        std::fs::write(folder.0.join("Plain.ttf"), SAMPLE).expect("face");
        let table = FaceTable::scan(&[&folder.0]);
        // The style the file draws with nothing set keeps the plain name;
        // each other style is the family and the style's own name.
        assert_eq!(
            table.labelled(),
            [
                ("Plain.ttf".to_owned(), "Plain".to_owned()),
                ("Vary.ttf".to_owned(), "Vary".to_owned()),
                ("Vary.ttf@Bold".to_owned(), "Vary Bold".to_owned()),
                (
                    "Vary.ttf@Condensed Bold".to_owned(),
                    "Vary Condensed Bold".to_owned()
                ),
            ]
        );
        // Each reads as a font set to its style: the same glyph is as
        // wide as the whole file says it is there.
        let whole = cutlass_font::read(VARIABLE).expect("face");
        let glyph = (1..whole.glyph_count())
            .find(|glyph| {
                let bold = whole
                    .varied(&[(*b"wght", 700.0)])
                    .and_then(|bold| bold.advance(*glyph));
                bold.is_ok() && bold != whole.advance(*glyph)
            })
            .expect("a glyph that widens");
        let advance = |name: &str| {
            let bytes = table.face(name).expect("bytes");
            cutlass_font::read(bytes)
                .expect("font")
                .advance(glyph)
                .expect("advance")
        };
        let set = |settings: &[(cutlass_font::Tag, f32)]| {
            whole
                .varied(settings)
                .expect("set")
                .advance(glyph)
                .expect("advance")
        };
        assert_eq!(advance("Vary.ttf"), whole.advance(glyph).expect("advance"));
        assert_eq!(advance("Vary.ttf@Bold"), set(&[(*b"wght", 700.0)]));
        assert_eq!(
            advance("Vary.ttf@Condensed Bold"),
            set(&[(*b"wght", 700.0), (*b"wdth", 75.0)])
        );
        assert_ne!(advance("Vary.ttf@Bold"), advance("Vary.ttf"));
        // The style named bold is the family's bold face.
        let (bold, is_bold, is_italic) = table.styled("Vary.ttf", true, false).expect("bold");
        assert_eq!(Some(bold), table.face("Vary.ttf@Bold"));
        assert!(is_bold && !is_italic);
        assert!(table.styled("Vary.ttf", false, true).is_none());
        // A face that does not vary is listed as it was.
        assert!(table.styled("Plain.ttf", true, false).is_none());
        // A file that changes after the scan gives nothing.
        std::fs::write(folder.0.join("Vary.ttf"), b"gone").expect("rewritten");
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.names(), ["Plain.ttf"]);
    }

    #[test]
    fn a_style_of_a_variable_face_is_named_after_the_face_and_labelled_by_its_family() {
        let named =
            |family, style, default| style_names("Skia.ttf", "Skia file", family, style, default);
        let pair = |name: &str, label: &str| (name.to_owned(), label.to_owned());
        // The default keeps the face's name, and is labelled as the
        // others when the face names its family.
        assert_eq!(
            named(Some("Skia"), "Regular", true),
            pair("Skia.ttf", "Skia Regular")
        );
        assert_eq!(
            named(Some("Skia"), "Black Condensed", false),
            pair("Skia.ttf@Black Condensed", "Skia Black Condensed")
        );
        // With no family the file's label stands in for it.
        assert_eq!(named(None, "Regular", true), pair("Skia.ttf", "Skia file"));
        assert_eq!(
            named(None, "Bold", false),
            pair("Skia.ttf@Bold", "Skia file Bold")
        );
        // A face of a collection is named with its place in the file.
        assert_eq!(
            style_names("Many.ttc#2", "Many Light", Some("Many"), "Thin", false),
            pair("Many.ttc#2@Thin", "Many Thin")
        );
    }

    #[test]
    fn the_programs_own_families_are_listed_among_the_tables_faces_by_label() {
        use cutlass_eval::Fonts;
        let table = FaceTable::from_pairs(vec![
            ("Zeta.ttf".to_owned(), SHAPING.to_vec()),
            ("Middle.ttf".to_owned(), SAMPLE.to_vec()),
        ])
        .with_bundled(&[
            ("Vary.ttf", VARIABLE),
            ("Alpha.ttf", SAMPLE),
            // Not a font, a collection, and a name the table has: none
            // of them is added.
            ("Junk.ttf", b"not a font"),
            ("Many.ttc", SAMPLE_TTC),
            ("Zeta.ttf", SAMPLE),
        ]);
        // In the order of their labels, the variable face once for each
        // style its maker named.
        assert_eq!(
            table.labelled(),
            [
                // The sample face names itself, and is labelled so.
                ("Alpha.ttf".to_owned(), "Cutlass Sample".to_owned()),
                ("Middle.ttf".to_owned(), "Middle".to_owned()),
                ("Vary.ttf".to_owned(), "Vary".to_owned()),
                ("Vary.ttf@Bold".to_owned(), "Vary Bold".to_owned()),
                (
                    "Vary.ttf@Condensed Bold".to_owned(),
                    "Vary Condensed Bold".to_owned()
                ),
                ("Zeta.ttf".to_owned(), "Zeta".to_owned()),
            ]
        );
        // The faces a missing character was looked for in are the same
        // faces, wherever they now stand, and the table's own Zeta is
        // still the one with f.
        assert_eq!(table.fallback_names(), ["Zeta.ttf", "Middle.ttf"]);
        assert_eq!(table.fallback('f', false), table.face("Zeta.ttf"));
        assert_eq!(table.face("Alpha.ttf"), Some(SAMPLE));
        // A style reads as the face set to it, and the one named bold is
        // the family's bold.
        let whole = cutlass_font::read(VARIABLE).expect("face");
        let bold = whole.varied(&[(*b"wght", 700.0)]).expect("bold");
        let pinned = cutlass_font::read(table.face("Vary.ttf@Bold").expect("bytes")).expect("font");
        assert_eq!(pinned.variations(), bold.variations());
        for glyph in 0..whole.glyph_count() {
            assert_eq!(pinned.advance(glyph).ok(), bold.advance(glyph).ok());
        }
        let (styled, is_bold, _) = table.styled("Vary.ttf", true, false).expect("bold");
        assert_eq!(Some(styled), table.face("Vary.ttf@Bold"));
        assert!(is_bold);
        // A letter no fallback face has comes from the shipped face
        // given first that has it, whatever its label: both of these
        // have ko kai, and the one named last by label was given first.
        let thai = FaceTable::default().with_bundled(&[("Zz.ttf", THAI), ("Aa.ttf", THAI)]);
        assert_eq!(thai.names(), ["Aa.ttf", "Zz.ttf"]);
        assert_eq!(thai.fallback('\u{E01}', false), thai.face("Zz.ttf"));
        // A label the table already has is the table's.
        let again = table.with_bundled(&[("Other.ttf", SAMPLE)]);
        assert!(!again.names().iter().any(|name| name.starts_with("Other")));
        // Nothing to add, to nothing.
        assert!(FaceTable::default().with_bundled(&[]).names().is_empty());
    }

    #[test]
    fn a_titles_face_is_moved_along_one_axis_and_kept_on_the_others() {
        use cutlass_model::MaterialText;
        let table = FaceTable::from_pairs(vec![
            ("Sample.ttf".to_owned(), SAMPLE.to_vec()),
            ("Vary.ttf".to_owned(), VARIABLE.to_vec()),
        ])
        .with_bundled(&[("Pinned.ttf", VARIABLE)]);
        let mut text = MaterialText::new("title");
        // A face that does not vary, and one the table lacks: no axes.
        text.font_path = "Sample.ttf".to_owned();
        assert!(table.font_axes(&text).is_empty());
        assert_eq!(
            table.moved_font_axis(&text, *b"wght", 700.0),
            Err(AxisError::NoAxes)
        );
        text.font_path = "Gone.ttf".to_owned();
        assert!(table.font_axes(&text).is_empty());
        // The fixture varies in weight and width, each named by the face.
        text.font_path = "Vary.ttf".to_owned();
        let shown = |text: &MaterialText| -> Vec<(String, f32, f32, f32, f32)> {
            table
                .font_axes(text)
                .into_iter()
                .map(|axis| (axis.name, axis.min, axis.default, axis.max, axis.value))
                .collect()
        };
        assert_eq!(
            shown(&text),
            [
                ("Weight".to_owned(), 100.0, 400.0, 900.0, 400.0),
                ("Width".to_owned(), 50.0, 100.0, 150.0, 100.0)
            ]
        );
        // One axis moves; every axis is written; past an end is the end.
        let moved = table.moved_font_axis(&text, *b"wght", 650.0);
        assert_eq!(moved, Ok(Some("wght=650,wdth=100".to_owned())));
        text.font_variations = "wght=650,wdth=100".to_owned();
        assert_eq!(
            table.moved_font_axis(&text, *b"wdth", 10.0),
            Ok(Some("wght=650,wdth=50".to_owned()))
        );
        assert_eq!(shown(&text)[0].4, 650.0);
        // Where it already is, nothing; an axis it lacks, and a value
        // that is no number.
        assert_eq!(table.moved_font_axis(&text, *b"wght", 650.0), Ok(None));
        assert_eq!(
            table.moved_font_axis(&text, *b"slnt", -8.0),
            Err(AxisError::NoSuchAxis)
        );
        assert_eq!(
            table.moved_font_axis(&text, *b"wght", f64::NAN),
            Err(AxisError::NotANumber)
        );
        // A named style starts from where its maker set it: the bold one
        // moved in width alone is still bold.
        let mut bold = MaterialText::new("bold");
        bold.font_path = "Pinned.ttf@Bold".to_owned();
        assert_eq!(shown(&bold)[0].4, 700.0);
        assert_eq!(
            table.moved_font_axis(&bold, *b"wdth", 80.0),
            Ok(Some("wght=700,wdth=80".to_owned()))
        );
        assert_eq!(
            AxisError::NoAxes.to_string(),
            "this face has no axes to set"
        );
    }

    #[test]
    fn faces_are_searched_by_every_word_typed() {
        let faces = vec![
            ("a.ttf".to_owned(), "Noto Sans Devanagari".to_owned()),
            ("b.ttf".to_owned(), "Noto Serif".to_owned()),
            ("c.ttf".to_owned(), "Inter Bold".to_owned()),
        ];
        let found = |query: &str| -> Vec<&str> {
            matching_faces(&faces, query)
                .into_iter()
                .map(|(name, _)| name.as_str())
                .collect()
        };
        assert_eq!(found(""), ["a.ttf", "b.ttf", "c.ttf"]);
        assert_eq!(found("  noto "), ["a.ttf", "b.ttf"]);
        assert_eq!(found("SANS noto"), ["a.ttf"]);
        assert_eq!(found("ari"), ["a.ttf"]);
        assert_eq!(found("bold inter"), ["c.ttf"]);
        assert!(found("comic").is_empty());
        assert!(found("noto comic").is_empty());
        assert_eq!(found("   ").len(), 3);
        assert!(matching_faces(&[], "noto").is_empty());
    }

    #[test]
    fn faces_that_are_part_of_the_program_come_after_the_systems_own() {
        const BUILT_IN: &[(&str, &[u8])] = &[
            ("Shaping.ttf", SHAPING),
            ("Sample.ttf", SAMPLE),
            ("Junk.ttf", b"not a font"),
        ];
        // A system with no fonts at all: the built-in faces are the table,
        // in their order, and new text starts in the first.
        let bare = FaceTable::default().with_built_in(BUILT_IN);
        assert_eq!(bare.names(), ["Shaping.ttf", "Sample.ttf"]);
        assert_eq!(bare.name(), "Shaping.ttf");
        assert_eq!(bare.face("Shaping.ttf"), Some(SHAPING));
        assert_eq!(bare.fallback('f', false), Some(SHAPING));
        assert_eq!(bare.fallback('B', false), Some(SAMPLE));
        // After a system's faces, and the last looked in for a character:
        // the system's own face that draws it comes first.
        let folder = Folder::new("built-in");
        std::fs::write(folder.0.join("arial.ttf"), SAMPLE).expect("face");
        let table = FaceTable::scan(&[&folder.0]).with_built_in(BUILT_IN);
        assert_eq!(table.names(), ["arial.ttf", "Shaping.ttf", "Sample.ttf"]);
        assert_eq!(
            table.fallback_names(),
            ["arial.ttf", "Shaping.ttf", "Sample.ttf"]
        );
        assert_eq!(table.fallback('B', false), table.face("arial.ttf"));
        assert_eq!(table.fallback('f', false), Some(SHAPING));
        // A person's own file of a built-in face's name is the one used.
        std::fs::write(folder.0.join("Shaping.ttf"), SAMPLE).expect("face");
        let table = FaceTable::scan(&[&folder.0]).with_built_in(BUILT_IN);
        assert_eq!(table.names().len(), 3);
        assert_eq!(table.face("Shaping.ttf"), Some(SAMPLE));
        // A default named for the table still moves to the front.
        let table = table.preferring(&["Sample.ttf"]);
        assert_eq!(table.name(), "Sample.ttf");
        assert_eq!(table.face("Sample.ttf"), Some(SAMPLE));
    }

    #[test]
    fn kana_and_hangul_come_from_a_face_of_their_own_language() {
        const HAN: &[u8] = include_bytes!("../../font/fixtures/han.ttf");
        const KANA: &[u8] = include_bytes!("../../font/fixtures/han-kana.ttf");
        const HANGUL: &[u8] = include_bytes!("../../font/fixtures/han-hangul.ttf");
        // The same face under another name, told apart by bytes after it.
        let padded = |bytes: &[u8]| [bytes, &[0; 4]].concat();
        let folder = Folder::new("languages");
        // Faces made for Chinese are ranked first, and two of them here
        // also draw kana and Hangul.
        for (name, bytes) in [
            ("PingFang.ttc", HAN.to_vec()),
            ("Hiragino Sans GB.ttc", padded(KANA)),
            ("msyh.ttc", padded(HANGUL)),
            // As a Mac stores the name: each voicing mark on its own.
            ("ヒラキ\u{3099}ノ角コ\u{3099}シック W3.ttc", KANA.to_vec()),
            ("malgun.ttf", HANGUL.to_vec()),
        ] {
            std::fs::write(folder.0.join(name), bytes).expect("face");
        }
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.fallback_names().len(), 5, "{:?}", table.names());
        let face = |name: &str| {
            let found = table
                .face(name)
                .or_else(|| table.face(&format!("{name}#0")));
            assert!(found.is_some(), "{name}");
            found
        };
        // A Han character alone is drawn the Chinese way.
        assert_eq!(table.fallback('\u{6C34}', false), face("PingFang.ttc"));
        // Kana and Hangul pass over the Chinese faces that have them.
        assert_eq!(
            table.fallback('\u{3042}', false),
            face("ヒラキ\u{3099}ノ角コ\u{3099}シック W3.ttc")
        );
        // A name is the same name however its voicing marks are written,
        // and whatever its case.
        assert_eq!(
            fold("ガギパ.TTC"),
            fold("カ\u{3099}キ\u{3099}ハ\u{309A}.ttc")
        );
        assert_eq!(fold("Arial.TTF"), "arial.ttf");
        assert_ne!(fold("カナ.ttc"), fold("キナ.ttc"));
        assert_eq!(table.fallback('\u{AC00}', false), face("malgun.ttf"));
        // Without a face of the language, a Chinese face that has it is used.
        std::fs::remove_file(folder.0.join("malgun.ttf")).expect("remove");
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(
            table.fallback('\u{AC00}', false),
            table.face("msyh.ttc").or_else(|| table.face("msyh.ttc#0"))
        );
    }

    #[test]
    fn named_files_are_read_now_and_junk_is_kept_out() {
        let folder = Folder::new("load");
        let face = folder.0.join("Face.ttf");
        let junk = folder.0.join("Junk.ttf");
        std::fs::write(&face, SAMPLE).unwrap();
        std::fs::write(&junk, b"not a font").unwrap();
        let table = FaceTable::load_from(&[&face, &junk, &face]);
        assert_eq!(table.names(), ["Face.ttf"]);
        assert_eq!(table.name(), "Face.ttf");
        assert_eq!(table.face("Face.ttf"), Some(SAMPLE));
        assert!(table.face("Junk.ttf").is_none());

        let empty = FaceTable::load_from::<&Path>(&[]);
        assert!(empty.name().is_empty());
        assert!(
            FaceTable::named("x.ttf", b"junk".to_vec())
                .names()
                .is_empty()
        );
    }

    #[test]
    fn a_scan_finds_single_faces_below_a_folder_and_reads_them_when_asked() {
        let folder = Folder::new("scan");
        let deep = folder.0.join("a/b");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(folder.0.join("Zed.TTF"), SAMPLE).unwrap();
        std::fs::write(deep.join("Alpha.otf"), SAMPLE).unwrap();
        // A fake face and another kind of file stay out.
        std::fs::write(folder.0.join("Fake.ttf"), b"wOFFnope").unwrap();
        std::fs::write(folder.0.join("notes.txt"), SAMPLE).unwrap();
        let too_deep = folder.0.join("1/2/3/4");
        std::fs::create_dir_all(&too_deep).unwrap();
        std::fs::write(too_deep.join("Lost.ttf"), SAMPLE).unwrap();

        let table = FaceTable::scan(&[&folder.0, &folder.0.join("missing")]);
        assert_eq!(table.names(), ["Alpha.otf", "Zed.TTF"]);
        assert_eq!(table.face("Alpha.otf"), Some(SAMPLE));
        assert!(table.face("Lost.ttf").is_none());
    }

    #[test]
    fn each_face_of_a_collection_is_listed_by_its_full_name_and_draws_alone() {
        let folder = Folder::new("ttc");
        std::fs::write(folder.0.join("Zed.ttf"), SAMPLE).unwrap();
        std::fs::write(folder.0.join("Both.ttc"), SAMPLE_TTC).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.names(), ["Both.ttc#0", "Both.ttc#1", "Zed.ttf"]);
        assert_eq!(table.labelled()[0].1, "Cutlass Sample");
        assert_eq!(table.labelled()[2].1, "Zed");
        for name in ["Both.ttc#0", "Both.ttc#1"] {
            let bytes = table.face(name).expect("copied out");
            assert!(cutlass_font::read(bytes).is_ok(), "{name}");
        }
        // The file name alone or a face past the end names nothing.
        assert!(table.face("Both.ttc").is_none());
        assert!(table.face("Both.ttc#2").is_none());
    }

    #[test]
    fn hidden_files_and_dot_named_system_faces_stay_out() {
        let folder = Folder::new("hidden");
        std::fs::write(folder.0.join(".Hidden.ttf"), SAMPLE).unwrap();
        let mut private = SAMPLE_TTC.to_vec();
        let utf16: Vec<u8> = "Cutlass"
            .encode_utf16()
            .flat_map(u16::to_be_bytes)
            .collect();
        let dotted: Vec<u8> = ".utlass"
            .encode_utf16()
            .flat_map(u16::to_be_bytes)
            .collect();
        for (from, to) in [(b"Cutlass".to_vec(), b".utlass".to_vec()), (utf16, dotted)] {
            while let Some(at) = private.windows(from.len()).position(|w| w == from) {
                private[at..at + to.len()].copy_from_slice(&to);
            }
        }
        std::fs::write(folder.0.join("System.ttc"), &private).unwrap();
        std::fs::write(folder.0.join("Shown.ttf"), SAMPLE).unwrap();
        assert_eq!(FaceTable::scan(&[&folder.0]).names(), ["Shown.ttf"]);
    }

    #[test]
    fn a_collection_that_changes_after_the_scan_supplies_nothing() {
        let folder = Folder::new("ttc-gone");
        let path = folder.0.join("Both.ttc");
        std::fs::write(&path, SAMPLE_TTC).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        std::fs::write(&path, &SAMPLE_TTC[..64]).unwrap();
        assert!(table.face("Both.ttc#1").is_none());
    }

    #[test]
    fn a_scan_puts_the_best_preferred_face_first() {
        let folder = Folder::new("preferred");
        for name in ["Zed.ttf", "Arial.ttf", "Alpha.otf"] {
            std::fs::write(folder.0.join(name), SAMPLE).unwrap();
        }
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.names(), ["Arial.ttf", "Alpha.otf", "Zed.ttf"]);
        assert_eq!(table.name(), "Arial.ttf");

        // The first face of a preferred collection beats a later choice.
        std::fs::write(folder.0.join("HelveticaNeue.ttc"), SAMPLE_TTC).unwrap();
        assert_eq!(FaceTable::scan(&[&folder.0]).name(), "HelveticaNeue.ttc#0");

        std::fs::write(folder.0.join("Helvetica.ttf"), SAMPLE).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.name(), "Helvetica.ttf");
        assert_eq!(table.names().len(), 6);

        for name in ["Helvetica.ttf", "Arial.ttf", "HelveticaNeue.ttc"] {
            std::fs::remove_file(folder.0.join(name)).unwrap();
        }
        assert_eq!(FaceTable::scan(&[&folder.0]).name(), "Alpha.otf");
    }

    #[test]
    fn a_scanned_face_that_turns_out_bad_supplies_nothing() {
        let folder = Folder::new("bad");
        // The directory reads, the tables it points at are cut off.
        std::fs::write(folder.0.join("Broken.ttf"), &SAMPLE[..12 + 11 * 16]).unwrap();
        // A directory that does not read keeps the file out.
        let mut header = vec![0, 1, 0, 0];
        header.extend_from_slice(b"truncated");
        std::fs::write(folder.0.join("Torn.ttf"), &header).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.names(), ["Broken.ttf"]);
        assert!(table.face("Broken.ttf").is_none());
        assert!(table.face("Broken.ttf").is_none());
    }

    #[test]
    fn a_label_is_the_name_without_its_extension() {
        assert_eq!(face_label("DINCondensed-bold.ttf"), "DINCondensed bold");
        assert_eq!(face_label("Courier_New.otf"), "Courier New");
        assert_eq!(face_label("plain"), "plain");
    }
}
