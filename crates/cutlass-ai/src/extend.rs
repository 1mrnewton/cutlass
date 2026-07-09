//! Prompt extensibility: user rules, skills, and slash commands
//! (cloud-roadmap Workstream 10).
//!
//! Everything here is **prompt-level only** — the closed command
//! vocabulary, validation, dry-run, and one-undo-group invariants are
//! untouched, so a bad or malicious rule/skill can at worst propose bad
//! edits, which dry-run surfaces and one undo reverts.
//!
//! - **Rules** (`~/.cutlass/agent/rules/*.md`, plus per-project rules from
//!   `ProjectMetadata`) are always-on text injected into the system
//!   prompt, under a hard byte cap ([`MAX_RULES_BYTES`]) because local
//!   models with small contexts are a product target. Over-budget rules
//!   truncate with a *visible* flag the UI must surface — never silently.
//! - **Skills** (`~/.cutlass/agent/skills/<id>/SKILL.md`) are on-demand
//!   procedural workflows with YAML frontmatter. Only names + descriptions
//!   enter the system prompt; the body loads through the read-only
//!   `read_skill` tool when the model asks for it. (Client-side keyword
//!   matching that pre-injects bodies was considered and rejected:
//!   brittle, and wastes tokens on misses.)
//! - **Slash commands** (`~/.cutlass/agent/commands/*.md`) are prompt
//!   templates expanded client-side before the prompt reaches the loop;
//!   the loop itself never sees them.
//!
//! The agent loop stays file-free: callers load everything up front (the
//! [`load_agent_dir`] helper plus [`bundled_skills`]) and pass an
//! [`AgentExtensions`] into `run_prompt`.

use std::path::Path;

use crate::wire::ToolSpec;

/// Hard cap on injected rules text (user + project combined). A few KB —
/// rules share the context window with the state snapshot and the tool
/// schema, and small local models degrade fast past that.
pub const MAX_RULES_BYTES: usize = 4096;

/// One on-demand skill: a procedural workflow the model can fetch by id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    /// Stable id — the directory name under `skills/`, or the bundled id.
    pub id: String,
    pub name: String,
    /// One-line summary; this (with the name) is all the system prompt
    /// carries.
    pub description: String,
    /// The full SKILL.md body (frontmatter stripped), returned by
    /// `read_skill`.
    pub body: String,
}

/// One slash-command template: typing `/name rest` in the chat panel
/// expands to the template body (with `$ARGUMENTS` replaced by `rest`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommand {
    /// The command name — the file stem of `commands/<name>.md`.
    pub name: String,
    pub body: String,
}

/// Everything the caller loaded for this prompt: composed rules text
/// (already capped) and the skill set behind `read_skill`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentExtensions {
    /// Composed rules text, `""` when there are none. Callers build this
    /// with [`compose_rules`] so the cap is always applied.
    pub rules: String,
    pub skills: Vec<Skill>,
}

/// Compose rule sections (label, text) into the single prompt block,
/// enforcing [`MAX_RULES_BYTES`]. Returns the block and whether anything
/// was truncated — when true the UI must warn visibly.
pub fn compose_rules(sections: &[(String, String)]) -> (String, bool) {
    let mut out = String::new();
    for (label, text) in sections {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&format!("[{label}]\n{text}"));
    }
    if out.len() <= MAX_RULES_BYTES {
        return (out, false);
    }
    // Truncate on a char boundary and say so in-band, so the model knows
    // the rules are incomplete rather than mid-sentence garbage.
    let mut cut = MAX_RULES_BYTES;
    while !out.is_char_boundary(cut) {
        cut -= 1;
    }
    out.truncate(cut);
    out.push_str("\n[…rules truncated at the size cap]");
    (out, true)
}

/// A SKILL.md failed to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillParseError(pub String);

impl std::fmt::Display for SkillParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SkillParseError {}

/// Parse a SKILL.md: YAML frontmatter (`name`, `description` — flat
/// `key: value` lines only, no nesting) between `---` fences, body after.
pub fn parse_skill(id: &str, text: &str) -> Result<Skill, SkillParseError> {
    let text = text.trim_start_matches('\u{feff}');
    let rest = text
        .strip_prefix("---")
        .ok_or_else(|| SkillParseError("missing frontmatter (expected leading ---)".into()))?;
    let (front, body) = rest
        .split_once("\n---")
        .ok_or_else(|| SkillParseError("unterminated frontmatter (missing closing ---)".into()))?;

    let mut name = None;
    let mut description = None;
    for line in front.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        match key.trim() {
            "name" => name = Some(value.to_string()),
            "description" => description = Some(value.to_string()),
            _ => {}
        }
    }
    let name = name
        .filter(|n| !n.is_empty())
        .ok_or_else(|| SkillParseError("frontmatter is missing 'name'".into()))?;
    let description = description
        .filter(|d| !d.is_empty())
        .ok_or_else(|| SkillParseError("frontmatter is missing 'description'".into()))?;
    let body = body.trim_start_matches('\n').trim().to_string();
    if body.is_empty() {
        return Err(SkillParseError("skill body is empty".into()));
    }
    Ok(Skill {
        id: id.to_string(),
        name,
        description,
        body,
    })
}

/// The `read_skill` tool spec. Read-only: the loop answers it from the
/// preloaded [`AgentExtensions::skills`] without touching dispatch, the
/// same trust class as `describe_project`. Only offered to the model when
/// at least one skill exists.
pub fn read_skill_spec() -> ToolSpec {
    ToolSpec {
        name: "read_skill",
        description: "Fetch the full step-by-step procedure of a skill by id. The \
                      available skills (id, name, and what they're for) are listed in \
                      the system prompt. Call this before starting a task a skill \
                      covers, then follow the returned procedure.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The skill id, exactly as listed in the system prompt.",
                },
            },
            "required": ["id"],
        }),
    }
}

/// Expand a slash command: `/name rest of prompt` becomes the template
/// body with every `$ARGUMENTS` replaced by `rest of prompt` (appended at
/// the end when the template has no placeholder and arguments exist).
/// `None` when the prompt is not a slash command or no template matches —
/// the caller sends the prompt unchanged.
pub fn expand_slash_command(prompt: &str, commands: &[SlashCommand]) -> Option<String> {
    let rest = prompt.strip_prefix('/')?;
    let (name, args) = match rest.split_once(char::is_whitespace) {
        Some((name, args)) => (name, args.trim()),
        None => (rest.trim_end(), ""),
    };
    if name.is_empty() {
        return None;
    }
    let command = commands.iter().find(|c| c.name == name)?;
    let mut expanded = if command.body.contains("$ARGUMENTS") {
        command.body.replace("$ARGUMENTS", args)
    } else if args.is_empty() {
        command.body.clone()
    } else {
        format!("{}\n\n{args}", command.body)
    };
    expanded = expanded.trim().to_string();
    (!expanded.is_empty()).then_some(expanded)
}

// --- filesystem loading ---------------------------------------------------

/// Everything found in an agent config dir (`~/.cutlass/agent/`). Missing
/// dir ⇔ all empty; unreadable or malformed files are skipped and reported
/// in `warnings` so the UI can surface them without failing the prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentDir {
    /// (file stem, contents) of `rules/*.md`, sorted by stem.
    pub rules: Vec<(String, String)>,
    /// Parsed `skills/<id>/SKILL.md`, sorted by id.
    pub skills: Vec<Skill>,
    /// Parsed `commands/*.md`, sorted by name.
    pub commands: Vec<SlashCommand>,
    /// Human-readable notes about skipped files.
    pub warnings: Vec<String>,
}

/// Per-file size cap: a rules or skill file past this is a mistake (or
/// hostile), not a workflow.
const MAX_FILE_BYTES: u64 = 64 * 1024;

/// Load `dir` (the `~/.cutlass/agent/` layout). Never errors: a missing
/// or partially unreadable dir degrades to fewer entries plus warnings.
pub fn load_agent_dir(dir: &Path) -> AgentDir {
    let mut out = AgentDir::default();

    for (stem, text) in read_md_files(dir.join("rules"), &mut out.warnings) {
        out.rules.push((stem, text));
    }
    out.rules.sort_by(|a, b| a.0.cmp(&b.0));

    let skills_dir = dir.join("skills");
    if let Ok(entries) = std::fs::read_dir(&skills_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            let skill_path = path.join("SKILL.md");
            let Some(text) = read_capped(&skill_path, &mut out.warnings) else {
                continue;
            };
            match parse_skill(&id, &text) {
                Ok(skill) => out.skills.push(skill),
                Err(e) => out.warnings.push(format!(
                    "skill '{id}' skipped: {e} ({})",
                    skill_path.display()
                )),
            }
        }
    }
    out.skills.sort_by(|a, b| a.id.cmp(&b.id));

    for (stem, text) in read_md_files(dir.join("commands"), &mut out.warnings) {
        out.commands.push(SlashCommand {
            name: stem,
            body: text.trim().to_string(),
        });
    }
    out.commands.sort_by(|a, b| a.name.cmp(&b.name));

    out
}

/// (file stem, contents) for every readable `*.md` directly in `dir`.
fn read_md_files(dir: std::path::PathBuf, warnings: &mut Vec<String>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") || !path.is_file() {
            continue;
        }
        let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
            continue;
        };
        if let Some(text) = read_capped(&path, warnings) {
            out.push((stem, text));
        }
    }
    out
}

/// Read a file under the size cap; over-cap or unreadable pushes a warning
/// and returns `None`.
fn read_capped(path: &Path, warnings: &mut Vec<String>) -> Option<String> {
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > MAX_FILE_BYTES => {
            warnings.push(format!(
                "{} skipped: larger than {} KB",
                path.display(),
                MAX_FILE_BYTES / 1024
            ));
            return None;
        }
        Ok(_) => {}
        Err(_) => return None,
    }
    match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(e) => {
            warnings.push(format!("{} skipped: {e}", path.display()));
            None
        }
    }
}

// --- bundled first-party skills -------------------------------------------

macro_rules! bundled {
    ($id:literal) => {
        (
            $id,
            include_str!(concat!("../../../assets/skills/", $id, "/SKILL.md")),
        )
    };
}

/// First-party skills shipped with the app (the embedded-catalog pattern
/// from `cutlass-models::sticker`), so the feature demonstrates itself.
/// User skills with the same id take precedence at merge time.
const BUNDLED: &[(&str, &str)] = &[
    bundled!("social-media-cut"),
    bundled!("podcast-cleanup"),
    bundled!("highlights-reel"),
];

/// Parse the bundled skill set. Panics only on a malformed bundled file —
/// a build defect, pinned by the unit test below.
pub fn bundled_skills() -> Vec<Skill> {
    BUNDLED
        .iter()
        .map(|(id, text)| parse_skill(id, text).expect("bundled SKILL.md files are well-formed"))
        .collect()
}

/// Bundled skills plus `user` skills, user winning on id collisions,
/// sorted by id (deterministic prompt order).
pub fn merge_skills(user: Vec<Skill>) -> Vec<Skill> {
    let mut merged = user;
    for skill in bundled_skills() {
        if !merged.iter().any(|s| s.id == skill.id) {
            merged.push(skill);
        }
    }
    merged.sort_by(|a, b| a.id.cmp(&b.id));
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKILL: &str = "---\nname: Podcast cleanup\ndescription: Clean up a talk recording.\n---\n\nStep 1: denoise.\n";

    #[test]
    fn parse_skill_reads_frontmatter_and_body() {
        let skill = parse_skill("podcast-cleanup", SKILL).unwrap();
        assert_eq!(skill.id, "podcast-cleanup");
        assert_eq!(skill.name, "Podcast cleanup");
        assert_eq!(skill.description, "Clean up a talk recording.");
        assert_eq!(skill.body, "Step 1: denoise.");
    }

    #[test]
    fn parse_skill_rejects_missing_pieces() {
        assert!(parse_skill("x", "no frontmatter").is_err());
        assert!(parse_skill("x", "---\nname: A\n---\n\nbody").is_err());
        assert!(parse_skill("x", "---\nname: A\ndescription: B\n---\n\n").is_err());
    }

    #[test]
    fn compose_rules_caps_and_flags() {
        let (text, truncated) = compose_rules(&[
            ("user".into(), "always 9:16".into()),
            ("p".into(), "".into()),
        ]);
        assert_eq!(text, "[user]\nalways 9:16");
        assert!(!truncated);

        let big = "x".repeat(MAX_RULES_BYTES * 2);
        let (text, truncated) = compose_rules(&[("user".into(), big)]);
        assert!(truncated);
        assert!(text.len() <= MAX_RULES_BYTES + 64);
        assert!(text.ends_with("[…rules truncated at the size cap]"));
    }

    #[test]
    fn slash_commands_expand_with_arguments() {
        let commands = vec![
            SlashCommand {
                name: "vertical".into(),
                body: "Convert the timeline to 9:16.".into(),
            },
            SlashCommand {
                name: "title".into(),
                body: "Add a title that says $ARGUMENTS at the playhead.".into(),
            },
        ];
        assert_eq!(
            expand_slash_command("/vertical", &commands).as_deref(),
            Some("Convert the timeline to 9:16.")
        );
        assert_eq!(
            expand_slash_command("/title Hello World", &commands).as_deref(),
            Some("Add a title that says Hello World at the playhead.")
        );
        // Extra args with no placeholder append.
        assert_eq!(
            expand_slash_command("/vertical but keep audio", &commands).as_deref(),
            Some("Convert the timeline to 9:16.\n\nbut keep audio")
        );
        assert_eq!(expand_slash_command("/unknown", &commands), None);
        assert_eq!(expand_slash_command("plain prompt", &commands), None);
    }

    #[test]
    fn bundled_skills_parse_and_have_unique_ids() {
        let skills = bundled_skills();
        assert_eq!(skills.len(), 3);
        for (i, skill) in skills.iter().enumerate() {
            assert!(!skill.body.is_empty(), "{} has an empty body", skill.id);
            assert!(
                !skills[..i].iter().any(|s| s.id == skill.id),
                "duplicate id {}",
                skill.id
            );
        }
    }

    #[test]
    fn merge_prefers_user_skills_on_collision() {
        let user = vec![Skill {
            id: "podcast-cleanup".into(),
            name: "My cleanup".into(),
            description: "Custom".into(),
            body: "Do it my way.".into(),
        }];
        let merged = merge_skills(user);
        let podcast = merged.iter().find(|s| s.id == "podcast-cleanup").unwrap();
        assert_eq!(podcast.name, "My cleanup");
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn load_agent_dir_reads_the_layout() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("rules")).unwrap();
        std::fs::create_dir_all(root.join("skills/my-skill")).unwrap();
        std::fs::create_dir_all(root.join("commands")).unwrap();
        std::fs::write(root.join("rules/style.md"), "prefer crossfades").unwrap();
        std::fs::write(root.join("skills/my-skill/SKILL.md"), SKILL).unwrap();
        std::fs::write(root.join("skills/my-skill/notes.txt"), "ignored").unwrap();
        std::fs::write(root.join("commands/vertical.md"), "Go 9:16.").unwrap();

        let loaded = load_agent_dir(root);
        assert_eq!(
            loaded.rules,
            vec![("style".into(), "prefer crossfades".into())]
        );
        assert_eq!(loaded.skills.len(), 1);
        assert_eq!(loaded.skills[0].id, "my-skill");
        assert_eq!(
            loaded.commands,
            vec![SlashCommand {
                name: "vertical".into(),
                body: "Go 9:16.".into()
            }]
        );
        assert!(loaded.warnings.is_empty());

        // Missing dir degrades to empty.
        assert_eq!(load_agent_dir(&root.join("nope")), AgentDir::default());
    }

    #[test]
    fn malformed_skill_becomes_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("skills/broken")).unwrap();
        std::fs::write(root.join("skills/broken/SKILL.md"), "no frontmatter").unwrap();
        let loaded = load_agent_dir(root);
        assert!(loaded.skills.is_empty());
        assert_eq!(loaded.warnings.len(), 1);
        assert!(loaded.warnings[0].contains("broken"));
    }
}
