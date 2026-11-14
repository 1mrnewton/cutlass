//! The title faces Cutlass ships: open families by other makers, carried
//! inside the program so every system has the same set to set text in.
//!
//! Each family comes with its files as its maker published them, who made
//! it, and its license word for word, for the Licenses window. Where the
//! files are from is in `assets/NOTICE`. A family in one file that varies
//! holds every weight its maker named; the others are a file to a style.

#![forbid(unsafe_code)]

/// One family: its name, who made it, its license and its home, the
/// license's text, and its files by name.
#[derive(Clone, Copy, Debug)]
pub struct Family {
    pub name: &'static str,
    pub author: &'static str,
    /// The license's name.
    pub license: &'static str,
    /// The family's home.
    pub link: &'static str,
    /// The license text, word for word.
    pub text: &'static str,
    /// Each file's name, as its maker named it, and its bytes.
    pub files: &'static [(&'static str, &'static [u8])],
}

/// Every file of every family, by name, in the families' order.
pub fn files() -> impl Iterator<Item = (&'static str, &'static [u8])> {
    FAMILIES
        .iter()
        .flat_map(|family| family.files.iter().copied())
}

/// The families, plain text faces first, then faces for headlines, then
/// faces with serifs, then script and display faces, and a face of one
/// width.
pub const FAMILIES: &[Family] = &[
    Family {
        name: "Inter",
        author: "Rasmus Andersson",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Inter",
        text: include_str!("../assets/inter/OFL.txt"),
        files: &[(
            "Inter[opsz,wght].ttf",
            include_bytes!("../assets/inter/Inter[opsz,wght].ttf"),
        )],
    },
    Family {
        name: "Montserrat",
        author: "Julieta Ulanovsky, Sol Matas, Juan Pablo del Peral, Jacques Le Bailly",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Montserrat",
        text: include_str!("../assets/montserrat/OFL.txt"),
        files: &[(
            "Montserrat[wght].ttf",
            include_bytes!("../assets/montserrat/Montserrat[wght].ttf"),
        )],
    },
    Family {
        name: "Poppins",
        author: "Indian Type Foundry, Jonny Pinhorn, Ninad Kale",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Poppins",
        text: include_str!("../assets/poppins/OFL.txt"),
        files: &[
            (
                "Poppins-Black.ttf",
                include_bytes!("../assets/poppins/Poppins-Black.ttf"),
            ),
            (
                "Poppins-Bold.ttf",
                include_bytes!("../assets/poppins/Poppins-Bold.ttf"),
            ),
            (
                "Poppins-Regular.ttf",
                include_bytes!("../assets/poppins/Poppins-Regular.ttf"),
            ),
            (
                "Poppins-SemiBold.ttf",
                include_bytes!("../assets/poppins/Poppins-SemiBold.ttf"),
            ),
            (
                "Poppins-BoldItalic.ttf",
                include_bytes!("../assets/poppins/Poppins-BoldItalic.ttf"),
            ),
            (
                "Poppins-Italic.ttf",
                include_bytes!("../assets/poppins/Poppins-Italic.ttf"),
            ),
        ],
    },
    Family {
        name: "Rubik",
        author: "Hubert and Fischer, Meir Sadan, Cyreal, Daniel Grumer, Omaima Dajani",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Rubik",
        text: include_str!("../assets/rubik/OFL.txt"),
        files: &[(
            "Rubik[wght].ttf",
            include_bytes!("../assets/rubik/Rubik[wght].ttf"),
        )],
    },
    Family {
        name: "Nunito",
        author: "Vernon Adams, Cyreal, Jacques Le Bailly",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Nunito",
        text: include_str!("../assets/nunito/OFL.txt"),
        files: &[(
            "Nunito[wght].ttf",
            include_bytes!("../assets/nunito/Nunito[wght].ttf"),
        )],
    },
    Family {
        name: "Raleway",
        author: "Matt McInerney, Pablo Impallari, Rodrigo Fuenzalida",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Raleway",
        text: include_str!("../assets/raleway/OFL.txt"),
        files: &[(
            "Raleway[wght].ttf",
            include_bytes!("../assets/raleway/Raleway[wght].ttf"),
        )],
    },
    Family {
        name: "Space Grotesk",
        author: "Florian Karsten",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Space+Grotesk",
        text: include_str!("../assets/spacegrotesk/OFL.txt"),
        files: &[(
            "SpaceGrotesk[wght].ttf",
            include_bytes!("../assets/spacegrotesk/SpaceGrotesk[wght].ttf"),
        )],
    },
    Family {
        name: "Oswald",
        author: "Vernon Adams, Kalapi Gajjar, Cyreal",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Oswald",
        text: include_str!("../assets/oswald/OFL.txt"),
        files: &[(
            "Oswald[wght].ttf",
            include_bytes!("../assets/oswald/Oswald[wght].ttf"),
        )],
    },
    Family {
        name: "Bebas Neue",
        author: "Ryoichi Tsunekawa",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Bebas+Neue",
        text: include_str!("../assets/bebasneue/OFL.txt"),
        files: &[(
            "BebasNeue-Regular.ttf",
            include_bytes!("../assets/bebasneue/BebasNeue-Regular.ttf"),
        )],
    },
    Family {
        name: "Anton",
        author: "Vernon Adams",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Anton",
        text: include_str!("../assets/anton/OFL.txt"),
        files: &[(
            "Anton-Regular.ttf",
            include_bytes!("../assets/anton/Anton-Regular.ttf"),
        )],
    },
    Family {
        name: "Playfair Display",
        author: "Claus Eggers Sørensen",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Playfair+Display",
        text: include_str!("../assets/playfairdisplay/OFL.txt"),
        files: &[
            (
                "PlayfairDisplay[wght].ttf",
                include_bytes!("../assets/playfairdisplay/PlayfairDisplay[wght].ttf"),
            ),
            (
                "PlayfairDisplay-Italic[wght].ttf",
                include_bytes!("../assets/playfairdisplay/PlayfairDisplay-Italic[wght].ttf"),
            ),
        ],
    },
    Family {
        name: "Lora",
        author: "Cyreal",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Lora",
        text: include_str!("../assets/lora/OFL.txt"),
        files: &[
            (
                "Lora[wght].ttf",
                include_bytes!("../assets/lora/Lora[wght].ttf"),
            ),
            (
                "Lora-Italic[wght].ttf",
                include_bytes!("../assets/lora/Lora-Italic[wght].ttf"),
            ),
        ],
    },
    Family {
        name: "Gentium Book Plus",
        author: "SIL International",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Gentium+Book+Plus",
        text: include_str!("../assets/gentiumbookplus/OFL.txt"),
        files: &[
            (
                "GentiumBookPlus-Bold.ttf",
                include_bytes!("../assets/gentiumbookplus/GentiumBookPlus-Bold.ttf"),
            ),
            (
                "GentiumBookPlus-Regular.ttf",
                include_bytes!("../assets/gentiumbookplus/GentiumBookPlus-Regular.ttf"),
            ),
        ],
    },
    Family {
        name: "DM Serif Display",
        author: "Colophon Foundry",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/DM+Serif+Display",
        text: include_str!("../assets/dmserifdisplay/OFL.txt"),
        files: &[
            (
                "DMSerifDisplay-Regular.ttf",
                include_bytes!("../assets/dmserifdisplay/DMSerifDisplay-Regular.ttf"),
            ),
            (
                "DMSerifDisplay-Italic.ttf",
                include_bytes!("../assets/dmserifdisplay/DMSerifDisplay-Italic.ttf"),
            ),
        ],
    },
    Family {
        name: "Abril Fatface",
        author: "TypeTogether",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Abril+Fatface",
        text: include_str!("../assets/abrilfatface/OFL.txt"),
        files: &[(
            "AbrilFatface-Regular.ttf",
            include_bytes!("../assets/abrilfatface/AbrilFatface-Regular.ttf"),
        )],
    },
    Family {
        name: "Pacifico",
        author: "Vernon Adams, Jacques Le Bailly, Botjo Nikoltchev, Ani Petrova",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Pacifico",
        text: include_str!("../assets/pacifico/OFL.txt"),
        files: &[(
            "Pacifico-Regular.ttf",
            include_bytes!("../assets/pacifico/Pacifico-Regular.ttf"),
        )],
    },
    Family {
        name: "Dancing Script",
        author: "Impallari Type",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Dancing+Script",
        text: include_str!("../assets/dancingscript/OFL.txt"),
        files: &[(
            "DancingScript[wght].ttf",
            include_bytes!("../assets/dancingscript/DancingScript[wght].ttf"),
        )],
    },
    Family {
        name: "Caveat",
        author: "Impallari Type",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Caveat",
        text: include_str!("../assets/caveat/OFL.txt"),
        files: &[(
            "Caveat[wght].ttf",
            include_bytes!("../assets/caveat/Caveat[wght].ttf"),
        )],
    },
    Family {
        name: "Lobster",
        author: "Impallari Type",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Lobster",
        text: include_str!("../assets/lobster/OFL.txt"),
        files: &[(
            "Lobster-Regular.ttf",
            include_bytes!("../assets/lobster/Lobster-Regular.ttf"),
        )],
    },
    Family {
        name: "Permanent Marker",
        author: "Font Diner",
        license: "Apache License 2.0",
        link: "https://fonts.google.com/specimen/Permanent+Marker",
        text: include_str!("../assets/permanentmarker/LICENSE.txt"),
        files: &[(
            "PermanentMarker-Regular.ttf",
            include_bytes!("../assets/permanentmarker/PermanentMarker-Regular.ttf"),
        )],
    },
    Family {
        name: "Bangers",
        author: "Vernon Adams",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Bangers",
        text: include_str!("../assets/bangers/OFL.txt"),
        files: &[(
            "Bangers-Regular.ttf",
            include_bytes!("../assets/bangers/Bangers-Regular.ttf"),
        )],
    },
    Family {
        name: "Righteous",
        author: "Astigmatic",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Righteous",
        text: include_str!("../assets/righteous/OFL.txt"),
        files: &[(
            "Righteous-Regular.ttf",
            include_bytes!("../assets/righteous/Righteous-Regular.ttf"),
        )],
    },
    Family {
        name: "Fredoka",
        author: "Milena Brandão, Hafontia",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/Fredoka",
        text: include_str!("../assets/fredoka/OFL.txt"),
        files: &[(
            "Fredoka[wdth,wght].ttf",
            include_bytes!("../assets/fredoka/Fredoka[wdth,wght].ttf"),
        )],
    },
    Family {
        name: "JetBrains Mono",
        author: "JetBrains, Philipp Nurullin, Konstantin Bulenkov",
        license: "SIL Open Font License 1.1",
        link: "https://fonts.google.com/specimen/JetBrains+Mono",
        text: include_str!("../assets/jetbrainsmono/OFL.txt"),
        files: &[(
            "JetBrainsMono[wght].ttf",
            include_bytes!("../assets/jetbrainsmono/JetBrainsMono[wght].ttf"),
        )],
    },
];

#[cfg(test)]
mod tests {
    use super::{FAMILIES, files};

    /// The letters Ghana's languages add to the Latin of English, small
    /// and capital: open e and o, eng, the hooked d and f, gamma, and the
    /// hooked v.
    const GHANA: &str = "\u{25B}\u{254}\u{14B}\u{256}\u{192}\u{263}\u{28B}\u{190}\u{186}\u{14A}\u{189}\u{191}\u{194}\u{1B2}";

    #[test]
    fn every_file_is_a_face_with_outlines_and_every_family_has_its_license() {
        assert_eq!(FAMILIES.len(), 24);
        assert_eq!(files().count(), 33);
        let mut names: Vec<&str> = files().map(|(name, _)| name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 33, "a file name is used twice");
        let mut bytes = 0;
        for family in FAMILIES {
            assert!(!family.author.is_empty(), "{}", family.name);
            assert!(family.link.starts_with("https://"), "{}", family.name);
            // The license it names is the one its text is.
            let words = match family.license {
                "SIL Open Font License 1.1" => "SIL OPEN FONT LICENSE Version 1.1",
                "Apache License 2.0" => "Apache License",
                other => panic!("{}: {other}", family.name),
            };
            assert!(family.text.contains(words), "{}", family.name);
            assert!(family.text.len() > 4_000, "{}", family.name);
            assert!(!family.files.is_empty(), "{}", family.name);
            for (name, file) in family.files {
                bytes += file.len();
                let face =
                    cutlass_font::read(file).unwrap_or_else(|error| panic!("{name}: {error:?}"));
                assert!(face.has_outlines(), "{name}");
                // Every letter and digit of English is drawn.
                for character in ('A'..='Z').chain('a'..='z').chain('0'..='9') {
                    let glyph = face
                        .glyph_id(character)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| panic!("{name}: no {character}"));
                    assert!(face.outline(glyph).is_ok(), "{name}: {character}");
                }
                // The face says it is of this family.
                let listed = cutlass_font::list_faces(|offset, len| {
                    let start = usize::try_from(offset).ok()?;
                    file.get(start..start.checked_add(len)?).map(<[u8]>::to_vec)
                })
                .unwrap_or_else(|error| panic!("{name}: {error:?}"));
                let said = listed[0].family.clone().unwrap_or_default();
                assert!(
                    said.starts_with(family.name),
                    "{name} says it is {said}, not {}",
                    family.name
                );
            }
        }
        // What the set adds to the program, to be seen when it grows.
        assert!(bytes < 9 * 1024 * 1024, "{bytes} bytes");
    }

    #[test]
    fn the_text_faces_draw_the_letters_of_ghanas_languages() {
        // Not every display face has them; the faces for running text
        // that do are named here, so one that loses them is noticed.
        let complete: Vec<&str> = FAMILIES
            .iter()
            .filter(|family| {
                family.files.iter().all(|(_, file)| {
                    let face = cutlass_font::read(file).expect("face");
                    GHANA
                        .chars()
                        .all(|character| matches!(face.glyph_id(character), Ok(Some(_))))
                })
            })
            .map(|family| family.name)
            .collect();
        assert_eq!(complete, COMPLETE);
    }

    /// The families every file of which draws all of [`GHANA`].
    const COMPLETE: &[&str] = &["Inter", "Montserrat", "Anton", "Gentium Book Plus"];
}
