//! Minimal template authoring CLI — the bundle half of the authoring path.
//!
//! Authoring today: build the finished timeline in the editor (or
//! `cutlass-py`), save the project, then use this tool to mark the fill
//! slots and pack the distributable bundle. A slot-marking UI in the app
//! comes later; this unblocks first-party template production now.
//!
//! ```text
//! template-bundle mark <project.cutlass> <out.cutlasst> \
//!     --name "Vlog intro" [--category vlog] [--slots 12,15,19] \
//!     [--music 22] [--texts 30,31]
//! template-bundle pack <template.cutlasst> <out.cutlassb>
//! template-bundle inspect <bundle.cutlassb>
//! ```
//!
//! Clip ids are the raw ids the editor and `.cutlass` JSON show. `--slots`
//! order is fill order.

use std::path::Path;
use std::process::ExitCode;

use cutlass_models::{
    ClipId, Project, Replaceable, SlotMedia, Template, TemplateCategory, TemplateMeta,
    template_bundle,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("mark") => mark(&args[1..]),
        Some("pack") => pack(&args[1..]),
        Some("inspect") => inspect(&args[1..]),
        _ => {
            eprintln!("usage: template-bundle <mark|pack|inspect> …  (see module docs)");
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn mark(args: &[String]) -> Result<(), String> {
    let (positional, options) = split_options(args);
    let [project_path, out_path] = positional.as_slice() else {
        return Err("mark needs <project.cutlass> <out.cutlasst>".into());
    };

    let mut project =
        Project::load_from_file(Path::new(project_path)).map_err(|e| e.to_string())?;

    for (order, id) in id_list(&options, "slots")?.into_iter().enumerate() {
        project
            .set_replaceable(id, Some(Replaceable::new(order as u32)))
            .map_err(|e| format!("slot {}: {e}", id.raw()))?;
    }
    if let Some(id) = id_list(&options, "music")?.first().copied() {
        project
            .set_replaceable(
                id,
                Some(Replaceable::new(0).with_accepts(SlotMedia::AudioOnly)),
            )
            .map_err(|e| format!("music {}: {e}", id.raw()))?;
    }
    for id in id_list(&options, "texts")? {
        project
            .set_text_editable(id, true)
            .map_err(|e| format!("text {}: {e}", id.raw()))?;
    }

    let name = options
        .iter()
        .find(|(k, _)| k == "name")
        .map(|(_, v)| v.clone())
        .ok_or("--name is required")?;
    let mut meta = TemplateMeta::new(name);
    if let Some((_, category)) = options.iter().find(|(k, _)| k == "category") {
        meta.category = parse_category(category)?;
    }

    let template = Template::from_project(project, meta);
    if template.slot_count() == 0 {
        return Err("no visual slots marked — pass --slots with at least one clip id".into());
    }
    template
        .save_to_file(Path::new(out_path))
        .map_err(|e| e.to_string())?;
    println!(
        "wrote {out_path}: {} slot(s), {} editable text(s){}",
        template.slot_count(),
        template.editable_texts().len(),
        if template.music().is_some() {
            ", swappable music"
        } else {
            ""
        }
    );
    Ok(())
}

fn pack(args: &[String]) -> Result<(), String> {
    let (positional, _) = split_options(args);
    let [template_path, out_path] = positional.as_slice() else {
        return Err("pack needs <template.cutlasst> <out.cutlassb>".into());
    };
    let template = Template::load_from_file(Path::new(template_path)).map_err(|e| e.to_string())?;
    template_bundle::write(&template, Path::new(out_path)).map_err(|e| e.to_string())?;
    let size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "wrote {out_path}: {} slot(s), {} sample file(s), {:.1} MiB",
        template.slot_count(),
        template.project().media_count(),
        size as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn inspect(args: &[String]) -> Result<(), String> {
    let (positional, _) = split_options(args);
    let [bundle_path] = positional.as_slice() else {
        return Err("inspect needs <bundle.cutlassb>".into());
    };
    let manifest =
        template_bundle::read_manifest(Path::new(bundle_path)).map_err(|e| e.to_string())?;
    println!(
        "{}: bundle format v{}, requires project schema >= v{}",
        manifest.name, manifest.format_version, manifest.min_schema_version
    );
    Ok(())
}

// --- tiny arg plumbing (no clap: two commands, three options) ---------------

fn split_options(args: &[String]) -> (Vec<String>, Vec<(String, String)>) {
    let mut positional = Vec::new();
    let mut options = Vec::new();
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if let Some(key) = arg.strip_prefix("--") {
            let value = iter.next().cloned().unwrap_or_default();
            options.push((key.to_string(), value));
        } else {
            positional.push(arg.clone());
        }
    }
    (positional, options)
}

fn id_list(options: &[(String, String)], key: &str) -> Result<Vec<ClipId>, String> {
    let Some((_, raw)) = options.iter().find(|(k, _)| k == key) else {
        return Ok(Vec::new());
    };
    raw.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<u64>()
                .map(ClipId::from_raw)
                .map_err(|_| format!("--{key}: {part:?} is not a clip id"))
        })
        .collect()
}

fn parse_category(raw: &str) -> Result<TemplateCategory, String> {
    serde_json::from_value(serde_json::Value::String(raw.to_lowercase()))
        .map_err(|_| format!("unknown category {raw:?}"))
}
