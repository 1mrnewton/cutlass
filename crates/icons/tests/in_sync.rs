//! The list, the table, and the phone files agree. When one fails, run
//! `cargo run -p icongen`.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use cutlass_icons::{ANDROID_DRAWABLES, ICONS, IOS_FOLDER_CONTENTS, IOS_ICONS, MANIFEST};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: PathBuf) -> String {
    fs::read_to_string(&path).unwrap_or_else(|error| format!("{}: {error}", path.display()))
}

#[test]
fn the_table_holds_every_listed_icon_in_order() {
    let listed: Vec<(String, Option<String>)> = read(root().join(MANIFEST))
        .lines()
        .map(|line| line.split('#').next().unwrap_or("").trim().to_owned())
        .filter(|line| !line.is_empty() && !line.starts_with("package "))
        .map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next().unwrap_or("").to_owned();
            let source = words.next().filter(|word| *word != "custom");
            (name, source.map(str::to_owned))
        })
        .collect();
    let table: Vec<(String, Option<String>)> = ICONS
        .iter()
        .map(|icon| (icon.name.to_owned(), icon.source.map(str::to_owned)))
        .collect();
    assert_eq!(listed, table);
}

#[test]
fn every_phone_file_matches_the_table() {
    let ios = root().join(IOS_ICONS);
    let android = root().join(ANDROID_DRAWABLES);
    assert_eq!(read(ios.join("Contents.json")), IOS_FOLDER_CONTENTS);
    for icon in ICONS {
        let asset = cutlass_icons::asset_name(icon.name);
        let set = ios.join(format!("{asset}.imageset"));
        assert_eq!(
            read(set.join(format!("{asset}.svg"))),
            cutlass_icons::svg(icon),
            "{asset}.svg"
        );
        assert_eq!(
            read(set.join("Contents.json")),
            cutlass_icons::imageset_contents(&asset),
            "{asset}.imageset"
        );
        assert_eq!(
            read(android.join(format!("{asset}.xml"))),
            cutlass_icons::vector_drawable(icon),
            "{asset}.xml"
        );
    }
}

#[test]
fn no_phone_file_outlives_its_icon() {
    let assets: BTreeSet<String> = ICONS
        .iter()
        .map(|icon| cutlass_icons::asset_name(icon.name))
        .collect();
    for (folder, suffix) in [
        (root().join(IOS_ICONS), ".imageset"),
        (root().join(ANDROID_DRAWABLES), ".xml"),
    ] {
        let entries = fs::read_dir(&folder).map(|entries| entries.flatten().collect::<Vec<_>>());
        assert!(entries.is_ok(), "{}", folder.display());
        for entry in entries.unwrap_or_default() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(asset) = name.strip_suffix(suffix)
                && asset.starts_with("icon_")
            {
                assert!(assets.contains(asset), "{name} is not in the table");
            }
        }
    }
}

#[test]
fn the_hugeicons_notice_stays_with_the_path_data() {
    let source = include_str!("../src/hugeicons.rs");
    assert!(source.contains("MIT License"));
    assert!(source.contains("Copyright (c) 2025 Hugeicons"));
}
