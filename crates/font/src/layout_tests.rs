//! The layout tables, read back from fonts built in code.

use crate::fixture::{
    LookupSpec, Null, Off, V, anchor, be, boxed, chain, classes, coverage, coverage_ranges, font,
    gdef, layout_table, ligatures, mark_attach, pairs, sequences, signed, single, table,
};
use crate::{
    Adjust, Anchor, ClassDef, Coverage, FontError, GlyphClass, Positioning, Substitution, read,
};

/// Ten box glyphs; `A` is 2 and `B` is 3.
fn built(extra: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let glyphs = [boxed(500); 10];
    font(&glyphs, &[('A', 2), ('B', 3)], extra)
}

fn gsub() -> Vec<u8> {
    // A single substitution inside an extension lookup.
    let mut wrapped = be(&[1, 1]);
    wrapped.extend(8_u32.to_be_bytes());
    wrapped.extend(single(&[(7, 8)]));
    // A reverse chain: 4 becomes 9 between a 2 and a 3.
    let reverse = table(vec![
        V(1),
        Off(coverage(&[4])),
        V(1),
        Off(coverage(&[2])),
        V(1),
        Off(coverage(&[3])),
        V(1),
        V(9),
    ]);
    // Rules by glyph: 2 then 3 runs lookup 1 on the second.
    let by_glyph = table(vec![
        V(1),
        Off(coverage(&[2])),
        V(1),
        Off(table(vec![V(1), Off(be(&[2, 1, 3, 1, 1]))])),
    ]);
    // Rules by class, with a class before and after: class 1 is 2..3.
    let class_rule = be(&[1, 1, 2, 1, 1, 1, 1, 0, 1]);
    let by_class = table(vec![
        V(2),
        Off(coverage(&[2, 3])),
        Off(classes(&[(5, 5, 1)])),
        Off(classes(&[(2, 3, 1)])),
        Off(classes(&[(6, 6, 1)])),
        V(2),
        Null,
        Off(table(vec![V(1), Off(class_rule)])),
    ]);
    layout_table(
        &[(b"latn", &[0, 1])],
        &[(b"ccmp", &[1, 2]), (b"liga", &[0])],
        vec![
            LookupSpec::new(4, vec![ligatures(&[(4, &[(&[5, 5], 9), (&[5], 6)])])]),
            LookupSpec::new(1, vec![single(&[(2, 3)])]).flag(0x0308),
            LookupSpec::new(2, vec![sequences(&[(3, &[2, 2])])]),
            LookupSpec::new(
                6,
                vec![chain(&[&[2]], &[&[3], &[4, 5]], &[&[2]], &[(1, 1)])],
            ),
            LookupSpec::new(7, vec![wrapped]),
            LookupSpec::new(8, vec![reverse]),
            LookupSpec::new(5, vec![by_glyph]),
            LookupSpec::new(6, vec![by_class]),
            LookupSpec {
                kind: 3,
                flag: 0,
                mark_set: Some(1),
                subtables: vec![sequences(&[(2, &[7, 8])])],
            },
        ],
    )
}

#[test]
fn scripts_features_and_lookups_are_found() {
    let bytes = built(&[(b"GSUB", gsub())]);
    let face = read(&bytes).expect("font");
    let table = face.gsub().expect("gsub");
    assert!(face.gpos().is_none() && face.gdef().is_none());
    let language = table.language(&[*b"latn"], None).expect("language");
    assert_eq!(language.script, Some(*b"latn"));
    assert_eq!(language.required, None);
    assert_eq!(language.features, [0, 1]);
    // A script the font lacks falls back to its Latin one, and a language
    // it lacks to the script's default.
    let other = table
        .language(&[*b"arab"], Some(*b"URD "))
        .expect("fallback");
    assert_eq!(other.script, Some(*b"latn"));
    assert_eq!(other.features, [0, 1]);
    assert_eq!(table.feature_tag(0).expect("tag"), *b"ccmp");
    assert_eq!(table.feature_tag(1).expect("tag"), *b"liga");
    let lookups: Vec<u16> = table.feature_lookups(0).expect("lookups").iter().collect();
    assert_eq!(lookups, [1, 2]);
    assert_eq!(table.lookup_count(), 9);
    assert_eq!(table.feature_tag(2), Err(FontError::BadTable));
    assert_eq!(
        table.feature_lookups(2).map(|_| ()),
        Err(FontError::BadTable)
    );
    assert!(matches!(table.lookup(9), Err(FontError::BadTable)));

    let flagged = table.lookup(1).expect("lookup");
    assert_eq!(flagged.kind, 1);
    assert!(flagged.flags.ignore_marks && !flagged.flags.ignore_bases);
    assert_eq!(flagged.flags.mark_class, 3);
    assert_eq!(flagged.flags.mark_set, None);
    assert_eq!(table.lookup(8).expect("set").flags.mark_set, Some(1));
    // An extension lookup reads as what it wraps.
    let wrapped = table.lookup(4).expect("extension");
    assert_eq!(wrapped.kind, 1);
    let Substitution::Single(inner) = Substitution::read(1, wrapped.subtables[0]).expect("single")
    else {
        panic!("single");
    };
    assert_eq!(inner.substitute(7).expect("covered"), Some(8));
}

#[test]
fn substitutions_answer_for_their_glyphs() {
    let bytes = built(&[(b"GSUB", gsub())]);
    let face = read(&bytes).expect("font");
    let table = face.gsub().expect("gsub");
    let sub = |index: u16| {
        let lookup = table.lookup(index).expect("lookup");
        Substitution::read(lookup.kind, lookup.subtables[0]).expect("subtable")
    };
    let Substitution::Single(one) = sub(1) else {
        panic!("single");
    };
    assert_eq!(one.substitute(2).expect("covered"), Some(3));
    assert_eq!(one.substitute(3).expect("uncovered"), None);

    let Substitution::Multiple(many) = sub(2) else {
        panic!("multiple");
    };
    let written: Vec<u16> = many
        .glyphs(3)
        .expect("covered")
        .expect("list")
        .iter()
        .collect();
    assert_eq!(written, [2, 2]);
    assert!(many.glyphs(2).expect("uncovered").is_none());
    let Substitution::Alternate(choices) = sub(8) else {
        panic!("alternate");
    };
    assert_eq!(
        choices.glyphs(2).expect("covered").expect("list").get(0),
        Some(7)
    );

    let Substitution::Ligature(joined) = sub(0) else {
        panic!("ligature");
    };
    let made = joined.starting(4).expect("covered");
    assert_eq!(made.len(), 2);
    assert_eq!(made[0].glyph, 9);
    assert_eq!(made[0].rest.iter().collect::<Vec<_>>(), [5, 5]);
    assert_eq!(made[1].glyph, 6);
    assert_eq!(made[1].rest.len(), 1);
    assert!(joined.starting(5).expect("uncovered").is_empty());

    let Substitution::Reverse(reverse) = sub(5) else {
        panic!("reverse");
    };
    let (rule, substitute) = reverse.rule(4).expect("covered").expect("rule");
    assert_eq!(substitute, 9);
    assert_eq!(
        (rule.backtrack_len(), rule.input_len(), rule.lookahead_len()),
        (1, 0, 1)
    );
    assert!(rule.backtrack_matches(0, 2).expect("before"));
    assert!(!rule.backtrack_matches(0, 3).expect("before"));
    assert!(rule.lookahead_matches(0, 3).expect("after"));
    assert!(reverse.rule(5).expect("uncovered").is_none());
}

#[test]
fn contextual_rules_match_by_glyph_class_and_coverage() {
    let bytes = built(&[(b"GSUB", gsub())]);
    let face = read(&bytes).expect("font");
    let table = face.gsub().expect("gsub");
    let rules = |index: u16, glyph: u16| {
        let lookup = table.lookup(index).expect("lookup");
        let Substitution::Context(context) =
            Substitution::read(lookup.kind, lookup.subtables[0]).expect("subtable")
        else {
            panic!("context");
        };
        context.rules(glyph).expect("rules")
    };
    // By coverage, with glyphs before and after.
    let found = rules(3, 3);
    assert_eq!(found.len(), 1);
    let rule = found[0];
    assert_eq!(
        (rule.backtrack_len(), rule.input_len(), rule.lookahead_len()),
        (1, 1, 1)
    );
    assert!(rule.backtrack_matches(0, 2).expect("before"));
    assert!(rule.input_matches(0, 5).expect("input"));
    assert!(!rule.input_matches(0, 3).expect("input"));
    assert!(rule.lookahead_matches(0, 2).expect("after"));
    assert_eq!(rule.input_matches(1, 5), Err(FontError::BadTable));
    assert_eq!(rule.records().collect::<Vec<_>>(), [(1, 1)]);
    assert!(rules(3, 2).is_empty());

    // By glyph.
    let found = rules(6, 2);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].input_len(), 1);
    assert!(found[0].input_matches(0, 3).expect("input"));
    assert!(!found[0].input_matches(0, 2).expect("input"));
    assert_eq!(found[0].records().collect::<Vec<_>>(), [(1, 1)]);
    assert!(rules(6, 3).is_empty());

    // By class: class 1 before is glyph 5, the input class is 2..3, and
    // class 1 after is glyph 6.
    let found = rules(7, 2);
    assert_eq!(found.len(), 1);
    let rule = found[0];
    assert_eq!(
        (rule.backtrack_len(), rule.input_len(), rule.lookahead_len()),
        (1, 1, 1)
    );
    assert!(rule.backtrack_matches(0, 5).expect("before"));
    assert!(!rule.backtrack_matches(0, 6).expect("before"));
    assert!(rule.input_matches(0, 3).expect("input"));
    assert!(rule.lookahead_matches(0, 6).expect("after"));
    assert_eq!(rule.records().collect::<Vec<_>>(), [(0, 1)]);
}

fn gpos() -> Vec<u8> {
    // One glyph moved: the same shift for 2 and 3, then one each.
    let shared = table(vec![
        V(1),
        Off(coverage(&[2, 3])),
        V(5),
        V(signed(-10)),
        V(20),
    ]);
    let each = table(vec![
        V(2),
        Off(coverage(&[2, 3])),
        V(2),
        V(2),
        V(7),
        V(signed(-7)),
    ]);
    // Pairs by class: class 1 first glyphs against class 1 second glyphs.
    let by_class = table(vec![
        V(2),
        Off(coverage(&[2])),
        V(4),
        V(1),
        Off(classes(&[(2, 2, 1)])),
        Off(classes(&[(3, 3, 1)])),
        V(2),
        V(2),
        V(0),
        V(0),
        V(0),
        V(0),
        V(0),
        V(0),
        V(signed(-30)),
        V(5),
    ]);
    let cursive = table(vec![
        V(1),
        Off(coverage(&[2, 3])),
        V(2),
        Off(anchor(10, 20)),
        Null,
        Off(anchor(30, 40)),
        Off(anchor(50, 60)),
    ]);
    // A mark on part of a ligature: glyph 6 has two parts.
    let parts = table(vec![V(2), Off(anchor(100, 500)), Off(anchor(400, 500))]);
    let on_ligature = table(vec![
        V(1),
        Off(coverage(&[7])),
        Off(coverage(&[6])),
        V(1),
        Off(table(vec![V(1), V(0), Off(anchor(0, 450))])),
        Off(table(vec![V(1), Off(parts)])),
    ]);
    layout_table(
        &[(b"DFLT", &[0])],
        &[(b"kern", &[2])],
        vec![
            LookupSpec::new(1, vec![shared]),
            LookupSpec::new(1, vec![each]),
            LookupSpec::new(
                2,
                vec![pairs(&[(2, &[(2, 15), (3, -80)]), (3, &[(2, 40)])])],
            ),
            LookupSpec::new(2, vec![by_class]),
            LookupSpec::new(3, vec![cursive]),
            LookupSpec::new(
                4,
                vec![mark_attach(
                    &[(7, 0, (0, 450)), (8, 1, (0, -50))],
                    &[(2, &[(250, 700), (250, 0)]), (3, &[(300, 720), (300, -10)])],
                )],
            ),
            LookupSpec::new(5, vec![on_ligature]),
        ],
    )
}

#[test]
fn positioning_answers_for_its_glyphs() {
    let bytes = built(&[(b"GPOS", gpos())]);
    let face = read(&bytes).expect("font");
    let table = face.gpos().expect("gpos");
    // No script named, and the default script answers.
    assert_eq!(
        table.language(&[], None).expect("language").script,
        Some(*b"DFLT")
    );
    let sub = |index: u16| {
        let lookup = table.lookup(index).expect("lookup");
        Positioning::read(lookup.kind, lookup.subtables[0]).expect("subtable")
    };
    let Positioning::Single(shared) = sub(0) else {
        panic!("single");
    };
    let moved = Adjust {
        x: -10,
        x_advance: 20,
        ..Adjust::default()
    };
    assert_eq!(shared.adjust(2).expect("covered"), Some(moved));
    assert_eq!(shared.adjust(3).expect("covered"), Some(moved));
    assert_eq!(shared.adjust(4).expect("uncovered"), None);
    let Positioning::Single(each) = sub(1) else {
        panic!("single");
    };
    assert_eq!(
        each.adjust(2).expect("covered").map(|adjust| adjust.y),
        Some(7)
    );
    assert_eq!(
        each.adjust(3).expect("covered").map(|adjust| adjust.y),
        Some(-7)
    );

    let Positioning::Pair(kerned) = sub(2) else {
        panic!("pair");
    };
    assert!(!kerned.moves_second());
    let advance = |first, second| {
        kerned
            .adjust(first, second)
            .expect("pair")
            .map(|(first, _)| first.x_advance)
    };
    assert_eq!(advance(2, 3), Some(-80));
    assert_eq!(advance(2, 2), Some(15));
    assert_eq!(advance(3, 2), Some(40));
    assert_eq!(advance(3, 3), None);
    assert_eq!(advance(4, 2), None);
    let Positioning::Pair(classed) = sub(3) else {
        panic!("pair");
    };
    assert!(classed.moves_second());
    let (first, second) = classed.adjust(2, 3).expect("pair").expect("classes");
    assert_eq!((first.x_advance, second.x), (-30, 5));
    let (first, second) = classed.adjust(2, 5).expect("pair").expect("class zero");
    assert_eq!((first, second), (Adjust::default(), Adjust::default()));

    let Positioning::Cursive(joined) = sub(4) else {
        panic!("cursive");
    };
    let first = joined.anchors(2).expect("covered").expect("joins");
    assert_eq!(first.entry, Some(Anchor { x: 10, y: 20 }));
    assert_eq!(first.exit, None);
    let second = joined.anchors(3).expect("covered").expect("joins");
    assert_eq!(second.entry, Some(Anchor { x: 30, y: 40 }));
    assert_eq!(second.exit, Some(Anchor { x: 50, y: 60 }));
    assert!(joined.anchors(4).expect("uncovered").is_none());

    let Positioning::MarkToBase(marks) = sub(5) else {
        panic!("marks");
    };
    assert!(marks.places(7) && marks.places(8) && !marks.places(2));
    assert!(marks.holds(2) && !marks.holds(7));
    let above = marks.anchors(7, 3).expect("attach").expect("anchors");
    assert_eq!(above, (Anchor { x: 0, y: 450 }, Anchor { x: 300, y: 720 }));
    let below = marks.anchors(8, 2).expect("attach").expect("anchors");
    assert_eq!(below, (Anchor { x: 0, y: -50 }, Anchor { x: 250, y: 0 }));
    assert!(marks.anchors(7, 4).expect("no base").is_none());
    assert!(marks.anchors(2, 3).expect("no mark").is_none());

    let Positioning::MarkToLigature(parts) = sub(6) else {
        panic!("ligature marks");
    };
    let on_second = parts
        .ligature_anchors(7, 6, 1)
        .expect("attach")
        .expect("anchors");
    assert_eq!(on_second.1, Anchor { x: 400, y: 500 });
    assert!(parts.ligature_anchors(7, 6, 2).expect("no part").is_none());
    assert_eq!(parts.ligature_parts(6).expect("covered"), Some(2));
    assert_eq!(parts.ligature_parts(2).expect("uncovered"), None);
}

#[test]
fn glyph_kinds_and_mark_sets_come_from_gdef() {
    let table = gdef(
        &[(2, 3, 1), (6, 6, 2), (7, 8, 3)],
        &[(7, 7, 1), (8, 8, 2)],
        &[&[7], &[7, 8]],
    );
    let bytes = built(&[(b"GDEF", table)]);
    let face = read(&bytes).expect("font");
    let gdef = face.gdef().expect("gdef");
    assert!(gdef.has_classes());
    assert_eq!(gdef.class(2), GlyphClass::Base);
    assert_eq!(gdef.class(6), GlyphClass::Ligature);
    assert_eq!(gdef.class(8), GlyphClass::Mark);
    assert_eq!(gdef.class(4), GlyphClass::None);
    assert_eq!(
        (gdef.mark_class(7), gdef.mark_class(8), gdef.mark_class(2)),
        (1, 2, 0)
    );
    assert!(gdef.in_mark_set(0, 7) && !gdef.in_mark_set(0, 8));
    assert!(gdef.in_mark_set(1, 8));
    assert!(!gdef.in_mark_set(2, 7));

    // A table with no glyph classes says so.
    let bare = built(&[(b"GDEF", crate::fixture::gdef(&[], &[], &[]))]);
    let face = read(&bare).expect("font");
    assert!(!face.gdef().expect("gdef").has_classes());
}

#[test]
fn coverage_and_classes_read_in_both_forms() {
    let listed = coverage(&[3, 7, 9]);
    let listed = Coverage::read(&listed).expect("list");
    assert_eq!(
        (listed.index(3), listed.index(7), listed.index(9)),
        (Some(0), Some(1), Some(2))
    );
    assert_eq!(
        (listed.index(2), listed.index(8), listed.index(10)),
        (None, None, None)
    );
    let ranged = coverage_ranges(&[(3, 5), (9, 9), (20, 21)]);
    let ranged = Coverage::read(&ranged).expect("ranges");
    assert_eq!(
        (ranged.index(3), ranged.index(5), ranged.index(9)),
        (Some(0), Some(2), Some(3))
    );
    assert_eq!(
        (ranged.index(21), ranged.index(6), ranged.index(22)),
        (Some(5), None, None)
    );
    assert!(matches!(
        Coverage::read(&be(&[3, 0])),
        Err(FontError::BadTable)
    ));
    assert!(matches!(
        Coverage::read(&be(&[1, 4, 3])),
        Err(FontError::Truncated)
    ));

    let from = be(&[1, 10, 3, 4, 0, 6]);
    let from = ClassDef::read(&from).expect("array");
    assert_eq!(
        (
            from.class(9),
            from.class(10),
            from.class(12),
            from.class(13)
        ),
        (0, 4, 6, 0)
    );
    let ranges = classes(&[(2, 4, 1), (8, 8, 2)]);
    let ranges = ClassDef::read(&ranges).expect("ranges");
    assert_eq!(
        (
            ranges.class(1),
            ranges.class(3),
            ranges.class(8),
            ranges.class(9)
        ),
        (0, 1, 2, 0)
    );
    assert_eq!(ClassDef::default().class(5), 0);
}

#[test]
fn a_torn_layout_table_is_an_error() {
    // The table ends inside its header.
    let short = built(&[(b"GSUB", vec![0, 1, 0, 0, 0])]);
    assert!(matches!(read(&short), Err(FontError::Truncated)));
    // A list offset past the table.
    let past = built(&[(b"GPOS", be(&[1, 0, 400, 0, 0]))]);
    assert!(matches!(read(&past), Err(FontError::Truncated)));
    // A version this reader does not know.
    let version = built(&[(b"GSUB", be(&[2, 0, 0, 0, 0]))]);
    assert!(matches!(read(&version), Err(FontError::Unsupported)));
    // An empty table is no table, and no lists is no lookups.
    let empty = built(&[(b"GSUB", Vec::new())]);
    assert!(read(&empty).expect("font").gsub().is_none());
    let bare = built(&[(b"GSUB", be(&[1, 0, 0, 0, 0]))]);
    let face = read(&bare).expect("font");
    let table = face.gsub().expect("gsub");
    assert_eq!(table.lookup_count(), 0);
    assert_eq!(
        table.language(&[*b"latn"], None).expect("language").script,
        None
    );
    assert!(matches!(table.lookup(0), Err(FontError::BadTable)));

    // A subtable that points past its lookup, and a kind out of range.
    let lost = layout_table(
        &[],
        &[],
        vec![LookupSpec::new(1, vec![be(&[2, 900, 1, 3])])],
    );
    let bytes = built(&[(b"GSUB", lost)]);
    let face = read(&bytes).expect("font");
    let lookup = face.gsub().expect("gsub").lookup(0).expect("lookup");
    assert!(matches!(
        Substitution::read(lookup.kind, lookup.subtables[0]),
        Err(FontError::Truncated)
    ));
    assert!(matches!(
        Substitution::read(9, lookup.subtables[0]),
        Err(FontError::BadTable)
    ));
    assert!(matches!(
        Positioning::read(0, lookup.subtables[0]),
        Err(FontError::BadTable)
    ));
    // An extension that wraps an extension is refused.
    let mut nested = be(&[1, 7]);
    nested.extend(8_u32.to_be_bytes());
    let looped = layout_table(&[], &[], vec![LookupSpec::new(7, vec![nested])]);
    let bytes = built(&[(b"GSUB", looped)]);
    let face = read(&bytes).expect("font");
    assert!(matches!(
        face.gsub().expect("gsub").lookup(0),
        Err(FontError::BadTable)
    ));
}
