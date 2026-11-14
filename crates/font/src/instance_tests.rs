//! Tests of the named instances a font file lists, and of one instance
//! of a variable face as a face of its own.

use crate::fixture_var::{
    A, B, C, SPACE, fvar_table, names_table, variable_with, variable_with_tables,
};
use crate::{
    FaceEntry, Font, FontError, InstanceEntry, Tag, list_faces, load_face, load_instance,
    pin_instance, read, single_face,
};

const VARIABLE: &[u8] = include_bytes!("../fixtures/variable.ttf");
const STATIC: &[u8] = include_bytes!("../fixtures/sample.ttf");
const COLLECTION: &[u8] = include_bytes!("../fixtures/sample.ttc");

const WGHT: Tag = *b"wght";
const WDTH: Tag = *b"wdth";

fn from(bytes: &[u8]) -> impl FnMut(u64, usize) -> Option<Vec<u8>> + '_ {
    move |offset, len| {
        let start = usize::try_from(offset).ok()?;
        bytes
            .get(start..start.checked_add(len)?)
            .map(<[u8]>::to_vec)
    }
}

fn listed(bytes: &[u8]) -> Vec<FaceEntry> {
    list_faces(from(bytes)).expect("faces")
}

/// What a face draws and how far it advances, for a few glyphs.
fn drawn(font: &Font<'_>) -> Vec<(u16, crate::Outline)> {
    [A, B, C, SPACE]
        .iter()
        .map(|glyph| {
            (
                font.advance(*glyph).expect("advance"),
                font.outline(*glyph).expect("outline"),
            )
        })
        .collect()
}

#[test]
fn a_variable_face_lists_its_named_instances() {
    let faces = listed(VARIABLE);
    assert_eq!(faces.len(), 1);
    assert_eq!(
        faces[0].instances,
        [
            InstanceEntry {
                name: "Regular".to_owned(),
                coordinates: vec![(WGHT, 400.0), (WDTH, 100.0)],
                bold: false,
                italic: false,
                default: true,
            },
            // Of the two at weight 700 the bold is the one whose width
            // is the default.
            InstanceEntry {
                name: "Bold".to_owned(),
                coordinates: vec![(WGHT, 700.0), (WDTH, 100.0)],
                bold: true,
                italic: false,
                default: false,
            },
            InstanceEntry {
                name: "Condensed Bold".to_owned(),
                coordinates: vec![(WGHT, 700.0), (WDTH, 75.0)],
                bold: false,
                italic: false,
                default: false,
            },
        ]
    );
    // A face that does not vary lists none, and is listed as it was.
    let plain = listed(STATIC);
    assert_eq!(plain.len(), 1);
    assert!(plain[0].instances.is_empty());
    assert!(
        listed(COLLECTION)
            .iter()
            .all(|face| face.instances.is_empty())
    );
    // Only the headers, the directory and the tables of names and axes
    // are read.
    let mut most = 0;
    let _faces = list_faces(|offset, len| {
        most = most.max(len);
        from(VARIABLE)(offset, len)
    })
    .expect("faces");
    assert!(most < 1_024, "{most}");
}

/// The fixture with axes of weight and of `second`, the instances
/// `instances`, each a name and a value for each axis, and the names of
/// those instances.
fn styled(second: &[u8; 4], range: (f32, f32, f32), instances: &[(&str, [f32; 2])]) -> Vec<u8> {
    let names: Vec<(u16, &str)> = instances
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (300 + u16::try_from(index).expect("few"), *name))
        .collect();
    let specs: Vec<(u16, &[f32], u16)> = instances
        .iter()
        .enumerate()
        .map(|(index, (_, values))| {
            (
                300 + u16::try_from(index).expect("few"),
                values.as_slice(),
                0xFFFF,
            )
        })
        .collect();
    let axes = fvar_table(
        &[
            (b"wght", 100.0, 400.0, 900.0, 256),
            (second, range.0, range.1, range.2, 257),
        ],
        &specs,
    );
    variable_with_tables(vec![(b"fvar", axes), (b"name", names_table(&names))])
}

/// Each listed instance as its name, whether it is bold, and whether it
/// is italic.
fn styles(bytes: &[u8]) -> Vec<(String, bool, bool)> {
    listed(bytes)[0]
        .instances
        .iter()
        .map(|instance| (instance.name.clone(), instance.bold, instance.italic))
        .collect()
}

fn style(name: &str, bold: bool, italic: bool) -> (String, bool, bool) {
    (name.to_owned(), bold, italic)
}

#[test]
fn one_instance_of_each_slant_is_the_bold() {
    // An axis of slant: an instance leaning a degree or more is italic,
    // and each kind has its own bold, the one at weight 700.
    let bytes = styled(
        b"slnt",
        (-12.0, 0.0, 0.0),
        &[
            ("Regular", [400.0, 0.0]),
            ("Italic", [400.0, -12.0]),
            ("Semibold", [600.0, 0.0]),
            ("Bold", [700.0, 0.0]),
            ("Bold Italic", [700.0, -12.0]),
            ("Black", [900.0, 0.0]),
            ("Black Italic", [900.0, -12.0]),
            ("Barely", [400.0, -0.5]),
        ],
    );
    assert_eq!(
        styles(&bytes),
        [
            style("Regular", false, false),
            style("Italic", false, true),
            style("Semibold", false, false),
            style("Bold", true, false),
            style("Bold Italic", true, true),
            style("Black", false, false),
            style("Black Italic", false, true),
            style("Barely", false, false),
        ]
    );
    // An axis of italic, with no instance at 700: the nearest to it of
    // those from 600 up, and none where every weight is lighter.
    let bytes = styled(
        b"ital",
        (0.0, 0.0, 1.0),
        &[
            ("Light", [300.0, 0.0]),
            ("Medium", [500.0, 0.0]),
            ("Semibold", [600.0, 0.0]),
            ("Extrabold", [800.0, 0.0]),
            ("Light Italic", [300.0, 1.0]),
            ("Medium Italic", [500.0, 1.0]),
            ("Heavy Italic", [850.0, 1.0]),
            ("Black Italic", [900.0, 1.0]),
        ],
    );
    assert_eq!(
        styles(&bytes),
        [
            style("Light", false, false),
            style("Medium", false, false),
            style("Semibold", true, false),
            style("Extrabold", false, false),
            style("Light Italic", false, true),
            style("Medium Italic", false, true),
            style("Heavy Italic", true, true),
            style("Black Italic", false, true),
        ]
    );
    let bytes = styled(
        b"ital",
        (0.0, 0.0, 1.0),
        &[("Light", [300.0, 0.0]), ("Medium", [500.0, 0.0])],
    );
    assert_eq!(
        styles(&bytes),
        [style("Light", false, false), style("Medium", false, false)]
    );
}

#[test]
fn an_instance_is_loaded_as_a_face_of_its_own() {
    let original = read(VARIABLE).expect("face");
    let default = drawn(&original);
    for instance in &listed(VARIABLE)[0].instances {
        let bytes =
            load_instance(from(VARIABLE), None, &instance.coordinates, 1 << 20).expect("instance");
        // Opened like any face, it is the face set to the instance.
        let face = read(&bytes).expect("face");
        let set = original.varied(&instance.coordinates).expect("set");
        assert_eq!(face.variations(), set.variations(), "{}", instance.name);
        assert_eq!(drawn(&face), drawn(&set), "{}", instance.name);
        assert_eq!(
            drawn(&face) == default,
            instance.default,
            "{}",
            instance.name
        );
        // Each instance is another face to what keeps glyphs by face.
        assert_ne!(face.fingerprint(), original.fingerprint());
        // It lists as the variable face it still is.
        assert_eq!(listed(&bytes)[0].instances.len(), 3);
    }
    let pinned = |settings: &[(Tag, f32)]| pin_instance(VARIABLE, settings).expect("pinned");
    let bold = pinned(&[(WGHT, 700.0)]);
    let narrow = pinned(&[(WGHT, 700.0), (WDTH, 75.0)]);
    let fingerprint = |bytes: &[u8]| read(bytes).expect("face").fingerprint();
    assert_ne!(fingerprint(&bold), fingerprint(&narrow));
    assert_eq!(read(&bold).expect("face").advance(A), Ok(540));
    assert_eq!(read(&narrow).expect("face").advance(A), Ok(490));
    // A setting asked of a pinned face is the whole setting: an axis it
    // does not name is at its default, not where the face was pinned.
    let face = read(&narrow).expect("face");
    let wide = face.varied(&[(WGHT, 900.0)]).expect("set");
    assert_eq!(wide.variations(), [16_384, 0]);
    assert_eq!(wide.advance(A), Ok(600));
    assert_eq!(face.advance(A), Ok(490));
    // Pinned anew, the last pin is the one that counts, and the file
    // has one such table.
    let again = pin_instance(&narrow, &[(WGHT, 900.0)]).expect("pinned");
    assert_eq!(read(&again).expect("face").variations(), [16_384, 0]);
    assert_eq!(again.len(), pinned(&[(WGHT, 900.0)]).len());
    // No coordinates is the default instance.
    let plain = pinned(&[]);
    assert_eq!(read(&plain).expect("face").variations(), [0, 0]);
    assert_eq!(drawn(&read(&plain).expect("face")), default);
}

#[test]
fn a_face_that_does_not_vary_is_itself_pinned_or_not() {
    let whole = load_face(from(STATIC), None, 1 << 20).expect("face");
    let pinned = load_instance(from(STATIC), None, &[(WGHT, 700.0)], 1 << 20).expect("face");
    let (one, other) = (read(&whole).expect("face"), read(&pinned).expect("face"));
    assert!(!other.is_variable());
    for glyph in 0..one.glyph_count() {
        assert_eq!(one.advance(glyph), other.advance(glyph));
        assert_eq!(one.outline(glyph), other.outline(glyph));
    }
    // A face of a collection is taken out and pinned; a collection
    // itself is refused, as is a face with no room for its pin.
    let second = load_instance(from(COLLECTION), Some(1), &[(WGHT, 700.0)], 1 << 20).expect("face");
    let copied = single_face(COLLECTION, 1).expect("copied");
    assert_eq!(second.len(), copied.len() + 16 + 12);
    assert_eq!(
        pin_instance(COLLECTION, &[(WGHT, 700.0)]),
        Err(FontError::Collection)
    );
    assert_eq!(
        load_instance(from(VARIABLE), None, &[(WGHT, 700.0)], VARIABLE.len() + 27),
        Err(FontError::Unsupported)
    );
    assert!(load_instance(from(VARIABLE), None, &[(WGHT, 700.0)], VARIABLE.len() + 28).is_ok());
    assert_eq!(
        pin_instance(&[], &[(WGHT, 700.0)]),
        Err(FontError::Truncated)
    );
}

/// The fixture with `table` as its pin.
fn with_pin(table: &[u8]) -> Vec<u8> {
    variable_with(b"Cpin", Some(table.to_vec()))
}

/// A pin of `records`, each an axis and a 16.16 value, that says it is
/// of `version` and holds `count` records.
fn pin(version: u16, count: u16, records: &[(&[u8; 4], i32)]) -> Vec<u8> {
    let mut table = Vec::new();
    table.extend(version.to_be_bytes());
    table.extend(count.to_be_bytes());
    for (tag, value) in records {
        table.extend(*tag);
        table.extend(value.to_be_bytes());
    }
    table
}

#[test]
fn a_pin_that_does_not_read_is_no_pin() {
    let places = |table: &[u8]| {
        let bytes = with_pin(table);
        read(&bytes).expect("face").variations().to_vec()
    };
    let heavy = (b"wght", 900 << 16);
    assert_eq!(places(&pin(1, 1, &[heavy])), [16_384, 0]);
    // Another version, more records than the table has, and a table
    // cut short anywhere: the face opens at its default.
    assert_eq!(places(&pin(2, 1, &[heavy])), [0, 0]);
    assert_eq!(places(&pin(1, 2, &[heavy])), [0, 0]);
    assert_eq!(places(&pin(1, u16::MAX, &[heavy])), [0, 0]);
    let whole = pin(1, 2, &[heavy, (b"wdth", 75 << 16)]);
    assert_eq!(places(&whole), [16_384, -8_192]);
    for length in 0..whole.len() {
        assert_eq!(places(&whole[..length]), [0, 0], "{length}");
    }
    // An axis the face lacks is passed over, and values past an axis's
    // ends, to the largest and smallest a pin can hold, are its ends.
    assert_eq!(
        places(&pin(1, 2, &[(b"slnt", -10 << 16), heavy])),
        [16_384, 0]
    );
    assert_eq!(
        places(&pin(1, 2, &[(b"wght", i32::MAX), (b"wdth", i32::MIN)])),
        [16_384, -16_384]
    );
    // Of two records for one axis the later counts, as with a setting.
    assert_eq!(
        places(&pin(1, 2, &[heavy, (b"wght", 650 << 16)])),
        [4_096, 0]
    );
    // What a caller pins is cleaned first: a value that is no number is
    // left out, and one beyond what a pin holds is the most it holds.
    let pinned = pin_instance(
        VARIABLE,
        &[(WGHT, f32::NAN), (WDTH, f32::NEG_INFINITY), (WGHT, 1.0e30)],
    )
    .expect("pinned");
    assert_eq!(read(&pinned).expect("face").variations(), [16_384, 0]);
    // A face whose bending of its axes is torn cannot be set: pinned, it
    // still opens, at its default.
    let torn = variable_with(b"avar", Some(vec![0, 1, 0, 0, 0, 0, 0, 2, 0, 9]));
    assert!(read(&torn).expect("face").varied(&[(WGHT, 900.0)]).is_err());
    let pinned = pin_instance(&torn, &[(WGHT, 900.0)]).expect("pinned");
    assert_eq!(read(&pinned).expect("face").variations(), [0, 0]);
}

#[test]
fn hostile_lists_of_instances_are_bounded() {
    let started = std::time::Instant::now();
    // More instances than are listed, all with one name.
    let values = [400.0_f32, 100.0];
    let many: Vec<(u16, &[f32], u16)> = (0..1_000).map(|_| (258, &values[..], 0xFFFF)).collect();
    let axes = fvar_table(
        &[
            (b"wght", 100.0, 400.0, 900.0, 256),
            (b"wdth", 50.0, 100.0, 150.0, 257),
        ],
        &many,
    );
    let bytes = variable_with(b"fvar", Some(axes));
    assert_eq!(listed(&bytes)[0].instances.len(), 64);
    // Instances with no name the face has are not listed.
    let nameless: Vec<(u16, &[f32], u16)> =
        (0..1_000).map(|_| (999, &values[..], 0xFFFF)).collect();
    let axes = fvar_table(
        &[
            (b"wght", 100.0, 400.0, 900.0, 256),
            (b"wdth", 50.0, 100.0, 150.0, 257),
        ],
        &nameless,
    );
    let bytes = variable_with(b"fvar", Some(axes));
    assert!(listed(&bytes)[0].instances.is_empty());
    // A table of axes cut short anywhere, or scrambled, lists a face
    // still.
    let whole = crate::fixture_var::table_of(b"fvar");
    for length in 0..whole.len() {
        let bytes = variable_with(b"fvar", Some(whole[..length].to_vec()));
        assert_eq!(listed(&bytes).len(), 1);
    }
    for at in 0..whole.len() {
        let mut table = whole.clone();
        table[at] = 0xFF;
        let bytes = variable_with(b"fvar", Some(table));
        assert_eq!(listed(&bytes).len(), 1);
    }
    // A table of axes too large to be one is not read at all.
    let mut huge = whole.clone();
    huge.resize(300 * 1024, 0);
    let bytes = variable_with(b"fvar", Some(huge));
    let mut most = 0;
    let faces = list_faces(|offset, len| {
        most = most.max(len);
        from(&bytes)(offset, len)
    })
    .expect("faces");
    assert!(faces[0].instances.is_empty());
    assert!(most < 4_096, "{most}");
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
}

/// List the instances of the font files named in `CUTLASS_VARIABLE_FACES`
/// (with `:` between them), and check that each loads as a face that is
/// the variable face set to it.
#[test]
#[ignore = "reads the font files named in CUTLASS_VARIABLE_FACES"]
#[allow(clippy::print_stdout)]
fn the_instances_of_the_named_files_load() {
    let Ok(files) = std::env::var("CUTLASS_VARIABLE_FACES") else {
        return;
    };
    for path in files.split(':') {
        let path = path.split('#').next().unwrap_or(path);
        let bytes = std::fs::read(path).expect("font file");
        for face in listed(&bytes) {
            println!(
                "{path} {:?} {:?}: {} instances",
                face.index,
                face.full_name,
                face.instances.len()
            );
            let whole = load_face(from(&bytes), face.index, 256 << 20).expect("face");
            let original = read(&whole).expect("face");
            for instance in &face.instances {
                println!(
                    "  {}{}{}{}: {:?}",
                    instance.name,
                    if instance.bold { " (bold)" } else { "" },
                    if instance.italic { " (italic)" } else { "" },
                    if instance.default { " (default)" } else { "" },
                    instance
                        .coordinates
                        .iter()
                        .map(|(tag, value)| format!("{}={value}", String::from_utf8_lossy(tag)))
                        .collect::<Vec<_>>()
                );
                let pinned =
                    load_instance(from(&bytes), face.index, &instance.coordinates, 256 << 20)
                        .expect("instance");
                let pinned = read(&pinned).expect("face");
                let set = original.varied(&instance.coordinates).expect("set");
                assert_eq!(pinned.variations(), set.variations());
                for glyph in (0..pinned.glyph_count()).step_by(97) {
                    assert_eq!(pinned.advance(glyph), set.advance(glyph));
                    assert_eq!(pinned.outline(glyph), set.outline(glyph));
                }
            }
        }
    }
}
