//! Collect the notices for every crate the desktop app ships.
//!
//! The walk starts at the `cutlass` package and follows normal
//! dependencies on every platform; build scripts and tests are not shipped.
//! Each crate's own license files are kept word for word, and a text that
//! several crates share is stored once.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

type Failure = Box<dyn std::error::Error>;

const ROOT: &str = "cutlass";
const OUT: &str = "crates/licenses/third_party.json";

fn main() -> Result<(), Failure> {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--locked"])
        .current_dir(&workspace)
        .output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned().into());
    }
    let metadata: Value = serde_json::from_slice(&output.stdout)?;
    let notices = collect(&metadata)?;
    let bytes = serde_json::to_vec_pretty(&notices)?;
    std::fs::write(workspace.join(OUT), [bytes.as_slice(), b"\n"].concat())?;
    Ok(())
}

fn collect(metadata: &Value) -> Result<Value, Failure> {
    let packages: HashMap<&str, &Value> = list(metadata, "packages")?
        .iter()
        .filter_map(|package| Some((package["id"].as_str()?, package)))
        .collect();
    let root = packages
        .values()
        .find(|package| package["name"] == ROOT && package["source"].is_null())
        .and_then(|package| package["id"].as_str())
        .ok_or("no cutlass package")?;
    let nodes: HashMap<&str, &Value> = list(&metadata["resolve"], "nodes")?
        .iter()
        .filter_map(|node| Some((node["id"].as_str()?, node)))
        .collect();
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::from([root]);
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = nodes.get(id) else {
            continue;
        };
        for dep in node["deps"].as_array().into_iter().flatten() {
            let shipped = dep["dep_kinds"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|kind| kind["kind"].is_null());
            if shipped && let Some(pkg) = dep["pkg"].as_str() {
                queue.push_back(pkg);
            }
        }
    }
    let mut texts: Vec<String> = Vec::new();
    let mut crates = BTreeMap::new();
    let standard = Standard::read()?;
    for id in seen {
        let Some(package) = packages.get(id) else {
            continue;
        };
        if package["source"].is_null() {
            continue;
        }
        let name = package["name"].as_str().unwrap_or_default();
        let version = package["version"].as_str().unwrap_or_default();
        let manifest = package["manifest_path"].as_str().unwrap_or_default();
        let license = package["license"].as_str().unwrap_or("see the files");
        let mut refs = Vec::new();
        let mut found = license_files(Path::new(manifest).parent().unwrap_or(Path::new(".")))?;
        if found.is_empty() {
            // The package left its license files out; its terms are the
            // standard ones it names.
            found = standard.for_expression(license);
        }
        for text in found {
            let index = match texts.iter().position(|known| *known == text) {
                Some(index) => index,
                None => {
                    texts.push(text);
                    texts.len() - 1
                }
            };
            refs.push(index);
        }
        crates.insert(
            (name.to_owned(), version.to_owned()),
            json!({
                "name": name,
                "version": version,
                "license": license,
                "repository": package["repository"].as_str().unwrap_or_default(),
                "texts": refs,
            }),
        );
    }
    Ok(json!({
        "crates": crates.into_values().collect::<Vec<_>>(),
        "texts": texts,
    }))
}

/// The standard license texts, for a crate that does not ship its own.
struct Standard {
    mit: String,
    apache: String,
}

impl Standard {
    fn read() -> Result<Self, Failure> {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let ours = std::fs::read_to_string(workspace.join("LICENSE-MIT"))?;
        let mit = ours
            .lines()
            .map(|line| {
                if line.starts_with("Copyright") {
                    "Copyright (c) the authors of this crate"
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let apache = std::fs::read_to_string(workspace.join("LICENSE-APACHE"))?;
        Ok(Self {
            mit: mit.trim().to_owned(),
            apache: apache.trim().to_owned(),
        })
    }

    /// The text of the terms the app takes the crate under: MIT where it is
    /// offered, else Apache 2.0, and the font licenses named beside them.
    fn for_expression(&self, license: &str) -> Vec<String> {
        let mut texts = Vec::new();
        if license.contains("MIT") {
            texts.push(self.mit.clone());
        } else if license.contains("Apache-2.0") {
            texts.push(self.apache.clone());
        }
        if license.contains("OFL-1.1") || license.contains("Ubuntu-font-1.0") {
            texts.push(
                "The fonts bundled in this crate are under the SIL Open Font License 1.1 \
                 (https://openfontlicense.org) and the Ubuntu Font Licence 1.0 \
                 (https://ubuntu.com/legal/font-licence)."
                    .to_owned(),
            );
        }
        texts
    }
}

fn list<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, Failure> {
    value[key]
        .as_array()
        .ok_or_else(|| format!("cargo metadata has no {key}").into())
}

/// The license, copying and notice files at the top of a crate, by name.
fn license_files(dir: &Path) -> Result<Vec<String>, Failure> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_uppercase();
        let named = ["LICENSE", "LICENCE", "COPYING", "NOTICE", "UNLICENSE"]
            .iter()
            .any(|prefix| name.starts_with(prefix));
        if named && entry.path().is_file() {
            names.push(entry.path());
        }
    }
    names.sort();
    let mut texts = Vec::new();
    for path in names {
        let bytes = std::fs::read(&path)?;
        let text = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
        let text = text.trim().to_owned();
        if !text.is_empty() {
            texts.push(text);
        }
    }
    Ok(texts)
}
