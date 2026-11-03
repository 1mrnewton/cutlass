//! Writes the icon table and the phone icon files from `crates/icons/icons.txt`.
//!
//! Each line of the list is `name Source`. `Source` is a free Hugeicons glyph
//! such as `Undo03Icon`, or `custom` for a mark in `crates/icons/src/custom.rs`.
//! The `package` line pins the `@hugeicons/core-free-icons` version.
//!
//! The tarball is fetched with `curl` into `target/icongen` once per version.
//! Only the named glyphs are unpacked. A glyph this table cannot draw is an
//! error that names it, never a silent gap.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::{fs, io};

use cutlass_icons::{Cap, IconData, Join, Mark, Shape, Stroke};

/// Bézier handle length for a quarter ellipse.
const KAPPA: f32 = 0.552_284_8;

fn main() -> ExitCode {
    match run() {
        Ok(count) => {
            let _ = writeln!(io::stderr(), "icongen: wrote {count} icons");
            ExitCode::SUCCESS
        }
        Err(error) => {
            let _ = writeln!(io::stderr(), "icongen: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Entry {
    name: String,
    /// `None` for a custom mark.
    source: Option<String>,
}

struct Manifest {
    version: String,
    entries: Vec<Entry>,
}

fn run() -> Result<usize, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest = read_manifest(&root.join(cutlass_icons::MANIFEST))?;
    let sources = fetch(&root, &manifest)?;
    let mut icons = Vec::new();
    for entry in &manifest.entries {
        let icon = match &entry.source {
            Some(source) => {
                let text = sources
                    .get(source)
                    .ok_or_else(|| format!("{source} was not unpacked"))?;
                IconData {
                    name: leak(entry.name.clone()),
                    source: Some(leak(source.clone())),
                    marks: Box::leak(parse_glyph(source, text)?.into_boxed_slice()),
                }
            }
            None => *cutlass_icons::CUSTOM
                .iter()
                .find(|icon| icon.name == entry.name)
                .ok_or_else(|| format!("{} is custom but not in custom.rs", entry.name))?,
        };
        icons.push(icon);
    }
    write_table(&root, &manifest.version, &icons)?;
    write_ios(&root, &icons)?;
    write_android(&root, &icons)?;
    Ok(icons.len())
}

fn read_manifest(path: &Path) -> Result<Manifest, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut version = None;
    let mut entries = Vec::new();
    let mut names = BTreeSet::new();
    let mut assets = BTreeSet::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let [first, second] = words.as_slice() else {
            return Err(format!("icons.txt line {}: want `name Source`", number + 1));
        };
        if *first == "package" {
            version = Some((*second).to_owned());
            continue;
        }
        if !first.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(format!(
                "icons.txt line {}: {first} is not a camelCase name",
                number + 1
            ));
        }
        if !names.insert((*first).to_owned()) || !assets.insert(cutlass_icons::asset_name(first)) {
            return Err(format!(
                "icons.txt line {}: {first} is listed twice",
                number + 1
            ));
        }
        let source = (*second != "custom").then(|| (*second).to_owned());
        entries.push(Entry {
            name: (*first).to_owned(),
            source,
        });
    }
    let version = version.ok_or("icons.txt has no `package <version>` line")?;
    Ok(Manifest { version, entries })
}

/// The ESM source of every named glyph, keyed by the name in the list.
fn fetch(root: &Path, manifest: &Manifest) -> Result<BTreeMap<String, String>, String> {
    let cache = root.join("target/icongen");
    fs::create_dir_all(&cache).map_err(|error| format!("{}: {error}", cache.display()))?;
    let version = &manifest.version;
    let tarball = cache.join(format!("core-free-icons-{version}.tgz"));
    if !tarball.exists() {
        let url = format!(
            "https://registry.npmjs.org/@hugeicons/core-free-icons/-/core-free-icons-{version}.tgz"
        );
        let partial = tarball.with_extension("part");
        command(
            Command::new("curl")
                .arg("-sSfL")
                .arg("-o")
                .arg(&partial)
                .arg(&url),
        )?;
        fs::rename(&partial, &tarball).map_err(|error| error.to_string())?;
    }
    // File names do not always match the export case (`ArrowDownAZIcon.js`).
    let listing = command(Command::new("tar").arg("tzf").arg(&tarball))?;
    let mut by_lower = BTreeMap::new();
    for path in listing.lines() {
        if let Some(file) = path.strip_prefix("package/dist/esm/")
            && let Some(stem) = file.strip_suffix(".js")
            && !stem.contains('/')
        {
            by_lower.insert(stem.to_ascii_lowercase(), path.to_owned());
        }
    }
    let mut wanted = BTreeMap::new();
    for entry in &manifest.entries {
        if let Some(source) = &entry.source {
            let path = by_lower
                .get(&source.to_ascii_lowercase())
                .ok_or_else(|| format!("{source} is not in the free set {version}"))?;
            wanted.insert(source.clone(), path.clone());
        }
    }
    let unpacked = cache.join(version);
    fs::create_dir_all(&unpacked).map_err(|error| error.to_string())?;
    let paths: BTreeSet<&String> = wanted.values().collect();
    if !paths.is_empty() {
        command(
            Command::new("tar")
                .arg("xzf")
                .arg(&tarball)
                .arg("-C")
                .arg(&unpacked)
                .args(paths),
        )?;
    }
    let mut sources = BTreeMap::new();
    for (source, path) in wanted {
        let file = unpacked.join(&path);
        let text =
            fs::read_to_string(&file).map_err(|error| format!("{}: {error}", file.display()))?;
        sources.insert(source, text);
    }
    Ok(sources)
}

fn command(command: &mut Command) -> Result<String, String> {
    let output = command
        .output()
        .map_err(|error| format!("{command:?}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{command:?}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| error.to_string())
}

/// Read `["tag", { key: "value", ... }]` rows from the package's ESM file.
fn parse_glyph(source: &str, text: &str) -> Result<Vec<Mark>, String> {
    let mut marks = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("[\"") else {
            continue;
        };
        let (tag, rest) = rest
            .split_once('"')
            .ok_or_else(|| format!("{source}: unclosed tag"))?;
        let body = rest
            .split_once('{')
            .and_then(|(_, body)| body.rsplit_once('}'))
            .map(|(body, _)| body)
            .ok_or_else(|| format!("{source}: {tag} has no attributes"))?;
        let attributes = parse_attributes(source, body)?;
        marks.push(mark(source, tag, &attributes)?);
    }
    if marks.is_empty() {
        return Err(format!("{source}: no marks"));
    }
    Ok(marks)
}

fn parse_attributes(source: &str, body: &str) -> Result<BTreeMap<String, String>, String> {
    let mut attributes = BTreeMap::new();
    let mut rest = body;
    loop {
        rest = rest.trim_start_matches([' ', ',']);
        if rest.is_empty() {
            return Ok(attributes);
        }
        let (key, after) = rest
            .split_once(':')
            .ok_or_else(|| format!("{source}: attribute without a value"))?;
        let after = after
            .trim_start()
            .strip_prefix('"')
            .ok_or_else(|| format!("{source}: {key} is not a string"))?;
        let (value, after) = after
            .split_once('"')
            .ok_or_else(|| format!("{source}: {key} is not closed"))?;
        attributes.insert(key.trim().to_owned(), value.to_owned());
        rest = after;
    }
}

fn mark(source: &str, tag: &str, attributes: &BTreeMap<String, String>) -> Result<Mark, String> {
    let number = |key: &str| -> Result<f32, String> {
        attributes
            .get(key)
            .ok_or_else(|| format!("{source}: {tag} has no {key}"))?
            .parse()
            .map_err(|_| format!("{source}: {tag} {key} is not a number"))
    };
    let geometry: &[&str] = match tag {
        "path" => &["d"],
        "circle" => &["cx", "cy", "r"],
        "ellipse" => &["cx", "cy", "rx", "ry"],
        "rect" => &["x", "y", "width", "height", "rx"],
        _ => return Err(format!("{source}: {tag} is not drawn here")),
    };
    let paint = [
        "key",
        "fill",
        "fillRule",
        "clipRule",
        "stroke",
        "strokeWidth",
        "strokeLinecap",
        "strokeLinejoin",
    ];
    for key in attributes.keys() {
        if !geometry.contains(&key.as_str()) && !paint.contains(&key.as_str()) {
            return Err(format!(
                "{source}: {tag} uses {key}, which this table cannot draw"
            ));
        }
    }
    let shape = match tag {
        "path" => {
            let data = attributes
                .get("d")
                .ok_or_else(|| format!("{source}: path has no d"))?;
            if let Some(command) = data
                .chars()
                .find(|c| c.is_ascii_alphabetic() && !"MLHVCZ".contains(*c) && *c != 'e')
            {
                return Err(format!(
                    "{source}: path command {command} is not drawn here"
                ));
            }
            Shape::Path(leak(data.clone()))
        }
        "circle" => Shape::Circle {
            cx: number("cx")?,
            cy: number("cy")?,
            r: number("r")?,
        },
        "ellipse" => Shape::Path(leak(ellipse(
            number("cx")?,
            number("cy")?,
            number("rx")?,
            number("ry")?,
        ))),
        _ => {
            let rx = if attributes.contains_key("rx") {
                number("rx")?
            } else {
                0.0
            };
            Shape::Path(leak(rect(
                number("x").unwrap_or(0.0),
                number("y").unwrap_or(0.0),
                number("width")?,
                number("height")?,
                rx,
            )))
        }
    };
    let fill = match attributes.get("fill").map(String::as_str) {
        None | Some("none") => false,
        Some("currentColor") => true,
        Some(other) => return Err(format!("{source}: fill {other} is not the tint")),
    };
    let stroke = match attributes.get("stroke").map(String::as_str) {
        None | Some("none") => None,
        Some("currentColor") => Some(Stroke {
            width: number("strokeWidth").unwrap_or(1.0),
            cap: match attributes.get("strokeLinecap").map(String::as_str) {
                None | Some("butt") => Cap::Butt,
                Some("round") => Cap::Round,
                Some("square") => Cap::Square,
                Some(other) => return Err(format!("{source}: cap {other}")),
            },
            join: match attributes.get("strokeLinejoin").map(String::as_str) {
                None | Some("miter") => Join::Miter,
                Some("round") => Join::Round,
                Some("bevel") => Join::Bevel,
                Some(other) => return Err(format!("{source}: join {other}")),
            },
        }),
        Some(other) => return Err(format!("{source}: stroke {other} is not the tint")),
    };
    let even_odd = match attributes.get("fillRule").map(String::as_str) {
        None | Some("nonzero") => false,
        Some("evenodd") => true,
        Some(other) => return Err(format!("{source}: fill rule {other}")),
    };
    // Nothing clips, so a clip rule only has to agree with the fill rule.
    if let Some(clip) = attributes.get("clipRule")
        && Some(clip) != attributes.get("fillRule")
    {
        return Err(format!(
            "{source}: clip rule {clip} differs from the fill rule"
        ));
    }
    if !fill && stroke.is_none() {
        return Err(format!("{source}: {tag} has neither fill nor stroke"));
    }
    Ok(Mark {
        shape,
        fill,
        even_odd,
        stroke,
    })
}

fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> String {
    let (kx, ky) = (rx * KAPPA, ry * KAPPA);
    let (left, right, top, bottom) = (cx - rx, cx + rx, cy - ry, cy + ry);
    format!(
        "M{} {}C{} {} {} {} {} {}C{} {} {} {} {} {}C{} {} {} {} {} {}C{} {} {} {} {} {}Z",
        num(right),
        num(cy),
        num(right),
        num(cy + ky),
        num(cx + kx),
        num(bottom),
        num(cx),
        num(bottom),
        num(cx - kx),
        num(bottom),
        num(left),
        num(cy + ky),
        num(left),
        num(cy),
        num(left),
        num(cy - ky),
        num(cx - kx),
        num(top),
        num(cx),
        num(top),
        num(cx + kx),
        num(top),
        num(right),
        num(cy - ky),
        num(right),
        num(cy),
    )
}

fn rect(x: f32, y: f32, width: f32, height: f32, rx: f32) -> String {
    let r = rx.min(width / 2.0).min(height / 2.0).max(0.0);
    let (right, bottom) = (x + width, y + height);
    if r == 0.0 {
        return format!(
            "M{} {}H{}V{}H{}Z",
            num(x),
            num(y),
            num(right),
            num(bottom),
            num(x)
        );
    }
    let k = r * (1.0 - KAPPA);
    let mut out = String::new();
    let _ = write!(out, "M{} {}", num(x + r), num(y));
    let _ = write!(out, "H{}", num(right - r));
    let _ = write!(
        out,
        "C{} {} {} {} {} {}",
        num(right - k),
        num(y),
        num(right),
        num(y + k),
        num(right),
        num(y + r)
    );
    let _ = write!(out, "V{}", num(bottom - r));
    let _ = write!(
        out,
        "C{} {} {} {} {} {}",
        num(right),
        num(bottom - k),
        num(right - k),
        num(bottom),
        num(right - r),
        num(bottom)
    );
    let _ = write!(out, "H{}", num(x + r));
    let _ = write!(
        out,
        "C{} {} {} {} {} {}",
        num(x + k),
        num(bottom),
        num(x),
        num(bottom - k),
        num(x),
        num(bottom - r)
    );
    let _ = write!(out, "V{}", num(y + r));
    let _ = write!(
        out,
        "C{} {} {} {} {} {}Z",
        num(x),
        num(y + k),
        num(x + k),
        num(y),
        num(x + r),
        num(y)
    );
    out
}

/// Four decimals at most, no trailing zeros.
fn num(value: f32) -> String {
    let text = format!("{value:.4}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" {
        "0".to_owned()
    } else {
        text.to_owned()
    }
}

fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

fn write_table(root: &Path, version: &str, icons: &[IconData]) -> Result<(), String> {
    let mut out = String::new();
    let _ = write!(
        out,
        "//! Written by `tools/icongen` from `icons.txt`. Do not edit.
//!
//! Path data from `@hugeicons/core-free-icons` {version}.
//!
//! MIT License
//!
//! Copyright (c) 2025 Hugeicons
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the \"Software\"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

#[allow(unused_imports)]
use crate::{{Cap, IconData, Join, Mark, Shape, Stroke}};

pub static ICONS: &[IconData] = &[
"
    );
    for icon in icons {
        let Some(source) = icon.source else {
            let _ = writeln!(out, "crate::custom::{},", icon.name.to_ascii_uppercase());
            continue;
        };
        let _ = write!(
            out,
            "IconData {{ name: {:?}, source: Some({source:?}), marks: &[",
            icon.name
        );
        for mark in icon.marks {
            let shape = match mark.shape {
                Shape::Circle { cx, cy, r } => {
                    format!("Shape::Circle {{ cx: {cx:?}, cy: {cy:?}, r: {r:?} }}")
                }
                Shape::Path(data) => format!("Shape::Path({data:?})"),
            };
            let stroke = match mark.stroke {
                None => "None".to_owned(),
                Some(stroke) => format!(
                    "Some(Stroke {{ width: {:?}, cap: Cap::{:?}, join: Join::{:?} }})",
                    stroke.width, stroke.cap, stroke.join
                ),
            };
            let _ = write!(
                out,
                "Mark {{ shape: {shape}, fill: {}, even_odd: {}, stroke: {stroke} }},",
                mark.fill, mark.even_odd
            );
        }
        out.push_str("] },\n");
    }
    out.push_str("];\n");
    let path = root.join("crates/icons/src/hugeicons.rs");
    write(&path, &out)?;
    command(
        Command::new("rustfmt")
            .arg("--edition")
            .arg("2024")
            .arg(&path),
    )?;
    Ok(())
}

fn write_ios(root: &Path, icons: &[IconData]) -> Result<(), String> {
    let folder = root.join(cutlass_icons::IOS_ICONS);
    write(
        &folder.join("Contents.json"),
        cutlass_icons::IOS_FOLDER_CONTENTS,
    )?;
    let mut keep = BTreeSet::new();
    for icon in icons {
        let asset = cutlass_icons::asset_name(icon.name);
        let set = folder.join(format!("{asset}.imageset"));
        write(&set.join(format!("{asset}.svg")), &cutlass_icons::svg(icon))?;
        write(
            &set.join("Contents.json"),
            &cutlass_icons::imageset_contents(&asset),
        )?;
        keep.insert(format!("{asset}.imageset"));
    }
    remove_stale(&folder, &keep, ".imageset")
}

fn write_android(root: &Path, icons: &[IconData]) -> Result<(), String> {
    let folder = root.join(cutlass_icons::ANDROID_DRAWABLES);
    let mut keep = BTreeSet::new();
    for icon in icons {
        let asset = cutlass_icons::asset_name(icon.name);
        write(
            &folder.join(format!("{asset}.xml")),
            &cutlass_icons::vector_drawable(icon),
        )?;
        keep.insert(format!("{asset}.xml"));
    }
    remove_stale(&folder, &keep, ".xml")
}

/// Remove `icon_*` files the list no longer names.
fn remove_stale(folder: &Path, keep: &BTreeSet<String>, suffix: &str) -> Result<(), String> {
    let entries = fs::read_dir(folder).map_err(|error| format!("{}: {error}", folder.display()))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("icon_") && name.ends_with(suffix) && !keep.contains(&name) {
            let path: PathBuf = entry.path();
            let removed = if path.is_dir() {
                fs::remove_dir_all(&path)
            } else {
                fs::remove_file(&path)
            };
            removed.map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    Ok(())
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }
    fs::write(path, text).map_err(|error| format!("{}: {error}", path.display()))
}
