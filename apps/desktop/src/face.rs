//! The faces the desktop offers for text clips.
//!
//! The table itself is shared with the phones; this module says where a
//! desktop keeps its fonts: the system's folders, the person's own, the
//! `Fonts` folder of the Cutlass home, and a `fonts` folder shipped beside
//! the app. Only names are read at the start; a face's file is read when a
//! frame first asks for it. A few open faces are part of the program
//! itself, so a system with no fonts still has a face to set text in, and
//! one with no emoji face still draws emoji.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

pub use cutlass_playback::FaceTable;

/// The faces new text starts in, best first: the first one found is the
/// default. These draw Latin with kerning and ligatures from OpenType
/// tables on each system.
const DEFAULT_FACES: &[&str] = &[
    "Arial.ttf",
    "arial.ttf",
    "Geneva.ttf",
    "NewYork.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    // The face that is part of the program, where none of those is.
    "Ubuntu-Light.ttf",
];

/// A face that is part of the program: its file name, its bytes, and its
/// credit for the licenses screen.
#[derive(Clone, Copy, Debug)]
pub struct BuiltIn {
    pub name: &'static str,
    pub bytes: &'static [u8],
    pub family: &'static str,
    pub author: &'static str,
    /// The license's name, as its text gives it.
    pub license: &'static str,
    /// The face's home.
    pub link: &'static str,
    /// The license text, word for word.
    pub text: &'static str,
}

/// The faces that are part of the program. They are the ones the window
/// itself is drawn with, so offering them for titles adds nothing to the
/// download. Which further faces to ship is still to be chosen: a face is
/// added with one entry here and its license text beside the others.
pub const BUILT_IN: &[BuiltIn] = &[
    BuiltIn {
        name: "Ubuntu-Light.ttf",
        bytes: epaint_default_fonts::UBUNTU_LIGHT,
        family: "Ubuntu Light",
        author: "Dalton Maag, for Canonical",
        license: "Ubuntu Font Licence 1.0",
        link: "https://design.ubuntu.com/font",
        text: include_str!("../licenses/fonts/Ubuntu-Light.txt"),
    },
    BuiltIn {
        name: "Hack-Regular.ttf",
        bytes: epaint_default_fonts::HACK_REGULAR,
        family: "Hack",
        author: "Source Foundry",
        license: "MIT, with the Bitstream Vera license",
        link: "https://github.com/source-foundry/Hack",
        text: include_str!("../licenses/fonts/Hack-Regular.txt"),
    },
    BuiltIn {
        name: "NotoEmoji-Regular.ttf",
        bytes: epaint_default_fonts::NOTO_EMOJI_REGULAR,
        family: "Noto Emoji",
        author: "Google",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/noto/specimen/Noto+Emoji",
        text: include_str!("../licenses/fonts/NotoEmoji-Regular.txt"),
    },
];

/// The folders fonts are looked for in, on this system.
fn folders() -> Vec<PathBuf> {
    let mut folders = Vec::new();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if cfg!(target_os = "macos") {
        folders.push(PathBuf::from("/System/Library/Fonts"));
        folders.push(PathBuf::from("/Library/Fonts"));
        folders.extend(home.map(|home| home.join("Library/Fonts")));
    } else if cfg!(target_os = "windows") {
        let windows =
            std::env::var_os("WINDIR").map_or_else(|| PathBuf::from("C:\\Windows"), PathBuf::from);
        folders.push(windows.join("Fonts"));
        folders.extend(
            std::env::var_os("LOCALAPPDATA")
                .map(|local| PathBuf::from(local).join("Microsoft\\Windows\\Fonts")),
        );
    } else {
        folders.push(PathBuf::from("/usr/share/fonts"));
        folders.push(PathBuf::from("/usr/local/share/fonts"));
        if let Some(home) = home {
            folders.push(home.join(".fonts"));
            folders.push(home.join(".local/share/fonts"));
        }
    }
    folders.push(cutlass_storage::desktop_home().fonts());
    // Fonts shipped with the app: beside the program, or in a bundle's
    // resources.
    if let Ok(program) = std::env::current_exe()
        && let Some(beside) = program.parent()
    {
        folders.push(beside.join("fonts"));
        folders.push(beside.join("../Resources/fonts"));
    }
    folders
}

/// The faces found on this system, looked for once.
#[must_use]
pub fn installed() -> Arc<FaceTable> {
    static FACES: OnceLock<Arc<FaceTable>> = OnceLock::new();
    Arc::clone(FACES.get_or_init(|| {
        let built_in: Vec<(&str, &'static [u8])> = BUILT_IN
            .iter()
            .map(|face| (face.name, face.bytes))
            .collect();
        Arc::new(
            FaceTable::scan(&folders())
                .with_built_in(&built_in)
                .preferring(DEFAULT_FACES),
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::{BUILT_IN, DEFAULT_FACES, FaceTable, folders, installed};

    #[test]
    fn the_faces_that_are_part_of_the_program_draw_and_carry_their_license() {
        use cutlass_eval::Fonts;
        let table = FaceTable::default()
            .with_built_in(
                &BUILT_IN
                    .iter()
                    .map(|face| (face.name, face.bytes))
                    .collect::<Vec<_>>(),
            )
            .preferring(DEFAULT_FACES);
        // Every one reads as a face and is offered, and with no other
        // font on the system new text starts in the sans face.
        assert_eq!(table.names().len(), BUILT_IN.len());
        assert_eq!(table.name(), "Ubuntu-Light.ttf");
        for face in BUILT_IN {
            let font = cutlass_font::read(face.bytes).expect(face.name);
            assert!(font.glyph_count() > 100, "{}", face.name);
            // Its license is named and its text is there to show.
            assert!(!face.license.is_empty() && !face.author.is_empty());
            assert!(face.text.len() > 500, "{}", face.name);
            assert!(face.link.starts_with("https://"));
            assert!(
                BUILT_IN
                    .iter()
                    .filter(|other| other.name == face.name)
                    .count()
                    == 1
            );
        }
        // Letters come from the sans face, and an emoji from the emoji
        // face even though it has no color.
        let sans = table.face("Ubuntu-Light.ttf").expect("sans");
        assert_eq!(table.fallback('A', false), Some(sans));
        assert_eq!(
            table.fallback('\u{1F600}', true),
            table.face("NotoEmoji-Regular.ttf")
        );
        // Each has outlines for what it is offered for.
        let drawn = |name: &str, character: char| {
            let font = cutlass_font::read(table.face(name).expect(name)).expect(name);
            let glyph = font.glyph_id(character).expect("map").expect("glyph");
            !font.outline(glyph).expect("outline").contours.is_empty()
        };
        assert!(drawn("Ubuntu-Light.ttf", 'A') && drawn("Ubuntu-Light.ttf", '\u{E9}'));
        assert!(drawn("Hack-Regular.ttf", 'A'));
        assert!(drawn("NotoEmoji-Regular.ttf", '\u{1F600}'));
    }

    #[test]
    fn the_folders_hold_the_homes_own_and_the_apps() {
        let folders = folders();
        assert!(folders.contains(&cutlass_storage::desktop_home().fonts()));
        assert!(folders.iter().any(|folder| folder.ends_with("fonts")));
        assert!(folders.len() >= 4);
    }

    #[test]
    fn the_faces_are_looked_for_once() {
        let first = installed();
        let again = installed();
        assert!(std::sync::Arc::ptr_eq(&first, &again));
        // Whatever the system has, the default is one of its faces.
        assert_eq!(first.name().is_empty(), first.names().is_empty());
    }

    /// How long the scan takes and what it finds, on this machine.
    #[test]
    #[ignore = "prints what this machine's fonts give"]
    fn what_this_machine_gives() {
        let started = std::time::Instant::now();
        let table = FaceTable::scan(&folders()).preferring(DEFAULT_FACES);
        let took = started.elapsed();
        let names = table.names();
        let line = format!(
            "{} faces in {took:?}; default {}; fallbacks {:?}\n",
            names.len(),
            table.name(),
            table.fallback_names()
        );
        std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes()).expect("stderr");
        // A fallback face is read when a character first needs it.
        use cutlass_eval::Fonts as _;
        for (character, emoji) in [('\u{1F600}', true), ('\u{4E2D}', false), ('\u{915}', false)] {
            let started = std::time::Instant::now();
            let size = table.fallback(character, emoji).map(<[u8]>::len);
            let line = format!(
                "U+{:04X}: {size:?} bytes in {:?}\n",
                u32::from(character),
                started.elapsed()
            );
            std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes()).expect("stderr");
        }
    }

    /// Draw titles in several scripts with this machine's faces and write
    /// the frame as raw RGBA, to be looked at. Runs when
    /// `CUTLASS_TEXT_FRAME` names the file to write.
    #[test]
    #[ignore = "writes a frame drawn with this machine's fonts"]
    fn a_frame_of_many_scripts() {
        use cutlass_model::{Draft, MaterialText, Micros, Segment, TimeRange};
        let Ok(path) = std::env::var("CUTLASS_TEXT_FRAME") else {
            return;
        };
        let fonts = installed();
        let mut draft = Draft::empty("scripts");
        draft.canvas_config.width = 1_280;
        draft.canvas_config.height = 720;
        let lines = [
            "Office waffle AVATAR: fi fl, na\u{EF}ve caf\u{E9} \u{1F600}\u{1F44D}\u{1F3FD}\u{1F1EC}\u{1F1ED}",
            "\u{645}\u{631}\u{62D}\u{628}\u{627} \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645} 123 (abc) \u{628}\u{650}\u{633}\u{652}\u{645}\u{650} \u{627}\u{644}\u{644}\u{651}\u{64E}\u{647}\u{650}",
            "\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947} \u{926}\u{941}\u{928}\u{93F}\u{92F}\u{93E} \u{939}\u{93F}\u{928}\u{94D}\u{926}\u{940} \u{915}\u{94D}\u{937}\u{924}\u{94D}\u{930}\u{93F}\u{92F}",
            "\u{5E9}\u{5B8}\u{5C1}\u{5DC}\u{5D5}\u{5B9}\u{5DD} \u{5E2}\u{5D5}\u{5DC}\u{5DD} \u{4E2D}\u{6587}\u{5B57} \u{D55C}\u{AE00} \u{E44}\u{E17}\u{E22}",
            "\u{6771}\u{4EAC}\u{306F}\u{6674}\u{308C}\u{3001}\u{76F4}\u{3059} \u{4E2D}\u{6587}\u{76F4} 1\u{FE0F}\u{20E3} \u{D55C}\u{AD6D}\u{8A9E}",
            "\u{FDFD} bold, italic and underlined.",
        ];
        for (index, line) in lines.iter().enumerate() {
            let id = format!("line{index}");
            let mut text = MaterialText::new(id.as_str());
            text.content = (*line).to_owned();
            text.font_size = 48.0;
            text.font_path = fonts.name().to_owned();
            text.text_color = "#FFFFFF".to_owned();
            text.line_max_width = 1_150.0;
            text.alignment = 1;
            // The last line has a pattern on three of its words, each in
            // the family's own bold or italic face where it has one.
            if index + 1 == lines.len() {
                use cutlass_model::PatternChange;
                let on = Some(true);
                let none = PatternChange::default();
                assert!(text.set_range_pattern(2, 6, PatternChange { bold: on, ..none }));
                assert!(text.set_range_pattern(8, 14, PatternChange { italic: on, ..none }));
                assert!(text.set_range_pattern(
                    19,
                    29,
                    PatternChange {
                        underline: on,
                        ..none
                    }
                ));
            }
            draft.materials.texts.push(text);
            let mut segment = Segment::new(id.as_str(), id.as_str(), TimeRange::new(0, 1_000_000));
            segment.render_index = i32::try_from(index).expect("index") + 1;
            #[allow(clippy::cast_precision_loss)]
            {
                segment.clip.transform.y = 0.8 - index as f64 * 0.27;
            }
            draft.tracks[0].segments.push(segment);
        }
        let frame = cutlass_eval::frame(
            &cutlass_eval::Snapshot::with_fonts(&draft, &*fonts),
            Micros(0),
            &mut cutlass_renderer_software::SoftwareDevice::new(),
            &cutlass_engine::ReaderFactory::new(),
            &mut cutlass_eval::OpenClips::new(),
            0,
        )
        .expect("frame");
        std::fs::write(path, &frame.rgba).expect("write");
    }
}
