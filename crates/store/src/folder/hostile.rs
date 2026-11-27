//! A hostile project folder fails closed.
//!
//! Every folder here is a fresh temporary one, never a real project. A file
//! past its size, a string or an array past its length, and a draft nested
//! past serde_json's limit each come back from `open` as an error. The same
//! draft inside the bounds still opens, so each refusal is the bound's doing.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use cutlass_model::Draft;
use serde_json::Value;

use super::tests::Temp;
use super::{Folder, open};
use crate::json::{MAX_FILE_BYTES, MAX_ITEMS, MAX_STRING_BYTES, write_draft};
use crate::{JsonLimit, StoreError};

/// A saved one-timeline project inside `temp`.
fn saved(temp: &Temp) -> PathBuf {
    let root = temp.path.join("project");
    Folder::new(Draft::empty("D")).save(&root).unwrap();
    root
}

/// Make the draft carry `value` under an unknown key, in the top file and in
/// the main timeline's copy, so the two still agree.
fn plant(root: &Path, value: &[u8]) {
    let base = write_draft(&Draft::empty("D")).unwrap();
    let mut bytes = br#"{"hostile":"#.to_vec();
    bytes.extend_from_slice(value);
    bytes.push(b',');
    bytes.extend_from_slice(&base[1..]);
    for file in ["draft_info.json", "Timelines/D/draft_info.json"] {
        fs::write(root.join(file), &bytes).unwrap();
    }
}

fn planted(root: &Path) -> Value {
    open(root).unwrap().folder.draft.rest["hostile"].clone()
}

fn string_of(len: usize) -> Vec<u8> {
    let mut bytes = vec![b'a'; len + 2];
    bytes[0] = b'"';
    bytes[len + 1] = b'"';
    bytes
}

fn array_of(items: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(items * 2 + 1);
    bytes.push(b'[');
    for item in 0..items {
        if item > 0 {
            bytes.push(b',');
        }
        bytes.push(b'0');
    }
    bytes.push(b']');
    bytes
}

fn nested(depth: usize) -> Vec<u8> {
    let mut bytes = vec![b'['; depth];
    bytes.resize(depth * 2, b']');
    bytes
}

#[test]
fn a_huge_string_in_the_draft_fails_closed() {
    let temp = Temp::new();
    let root = saved(&temp);
    plant(&root, &string_of(1024));
    assert_eq!(planted(&root).as_str().map(str::len), Some(1024));

    plant(&root, &string_of(MAX_STRING_BYTES + 1));
    let error = open(&root).unwrap_err();
    assert!(
        matches!(
            error,
            StoreError::TooLarge { limit: JsonLimit::String, ref path }
                if path.ends_with("draft_info.json")
        ),
        "{error}"
    );
}

#[test]
fn a_huge_array_in_the_draft_fails_closed() {
    let temp = Temp::new();
    let root = saved(&temp);
    plant(&root, &array_of(1000));
    assert_eq!(planted(&root).as_array().map(Vec::len), Some(1000));

    plant(&root, &array_of(MAX_ITEMS + 1));
    let error = open(&root).unwrap_err();
    assert!(
        matches!(
            error,
            StoreError::TooLarge { limit: JsonLimit::Items, ref path }
                if path.ends_with("draft_info.json")
        ),
        "{error}"
    );
}

#[test]
fn a_deeply_nested_draft_fails_closed() {
    let temp = Temp::new();
    let root = saved(&temp);
    plant(&root, &nested(100));
    assert!(planted(&root).is_array());

    // Just past serde_json's 128 levels, and far enough past them that
    // walking the nesting without the limit would overflow the stack.
    for depth in [200, 1_000_000] {
        plant(&root, &nested(depth));
        let error = open(&root).unwrap_err();
        assert!(
            matches!(
                error,
                StoreError::Json { ref message, ref path }
                    if message.contains("recursion limit") && path.ends_with("draft_info.json")
            ),
            "{depth}: {error}"
        );
    }
}

#[test]
fn an_oversized_project_file_is_refused() {
    // One file for each way open reads the folder: the draft, the project
    // index, a typed sidecar, the id map, an extra sidecar, and a timeline.
    for file in [
        "draft_info.json",
        "project.json",
        "timeline_layout.json",
        "common_attachment/attachment_id_mapping.json",
        "common_attachment/extra.json",
        "Timelines/D/draft_info.json",
    ] {
        let temp = Temp::new();
        let root = saved(&temp);
        // A sparse file: its metadata says one byte past the bound, and no
        // disk or memory holds that much.
        File::create(root.join(file))
            .unwrap()
            .set_len(MAX_FILE_BYTES + 1)
            .unwrap();
        let error = open(&root).unwrap_err();
        assert!(
            matches!(
                error,
                StoreError::TooLarge { limit: JsonLimit::File, ref path } if path.ends_with(file)
            ),
            "{file}: {error}"
        );
    }
}

#[cfg(unix)]
#[test]
fn a_project_file_with_no_size_is_cut_off_at_the_bound() {
    use std::io::Write;

    let temp = Temp::new();
    let root = saved(&temp);
    let draft = root.join("draft_info.json");
    fs::remove_file(&draft).unwrap();
    // A pipe's metadata gives no size, so only the read's own bound stops it.
    let made = std::process::Command::new("mkfifo")
        .arg(&draft)
        .status()
        .unwrap();
    assert!(made.success());
    let writer = std::thread::spawn({
        let draft = draft.clone();
        move || {
            let mut pipe = fs::OpenOptions::new().write(true).open(&draft).unwrap();
            let chunk = vec![0_u8; 1024 * 1024];
            let mut sent = 0_u64;
            while sent <= MAX_FILE_BYTES && pipe.write_all(&chunk).is_ok() {
                sent += chunk.len() as u64;
            }
        }
    });
    let error = open(&root).unwrap_err();
    writer.join().unwrap();
    assert!(
        matches!(
            error,
            StoreError::TooLarge { limit: JsonLimit::File, ref path }
                if path.ends_with("draft_info.json")
        ),
        "{error}"
    );
}
