//! Bundled skills and one external skill the host has allowed.

use std::fmt;
use std::path::Path;

use crate::approval::{ApprovalAction, Approver, authorize_then};

/// Instructions and the tools one ask may use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    /// Skill name.
    pub title: String,
    /// Words matched against the ask.
    pub description: String,
    /// Tool names this skill may call, in order.
    pub allowed_tools: Vec<String>,
    /// Instructions for the model.
    pub body: String,
}

/// The tool names a skill tells the model to call, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillPlan {
    /// Tool names, in the order the skill lists them.
    pub calls: Vec<String>,
}

/// Why a skill was not loaded or added.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillError {
    /// No skill matched the ask.
    NotFound,
    /// The skill text is missing a title, a description, or tools.
    Invalid {
        /// What the header failed to provide.
        reason: String,
    },
    /// The host refused to add a skill from outside the app.
    ApprovalRefused,
    /// The skill file could not be read.
    Io {
        /// The filesystem error text.
        message: String,
    },
}

impl fmt::Display for SkillError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => formatter.write_str("skill not found"),
            Self::Invalid { reason } => write!(formatter, "invalid skill: {reason}"),
            Self::ApprovalRefused => formatter.write_str("adding an external skill was refused"),
            Self::Io { message } => write!(formatter, "skill file could not be read: {message}"),
        }
    }
}

impl std::error::Error for SkillError {}

/// Bundled skills and skills the host allowed from outside the app.
#[derive(Debug)]
pub struct SkillLibrary {
    skills: Vec<Skill>,
}

impl Skill {
    /// The tool names, in listed order.
    #[must_use]
    pub fn plan(&self) -> SkillPlan {
        SkillPlan {
            calls: self.allowed_tools.clone(),
        }
    }
}

impl SkillLibrary {
    /// The five skills that ship with the app.
    #[must_use]
    pub fn bundled() -> Self {
        const TEXTS: [&str; 5] = [
            include_str!("../skills/remove-pauses/SKILL.md"),
            include_str!("../skills/motion-graphics/SKILL.md"),
            include_str!("../skills/product-demo/SKILL.md"),
            include_str!("../skills/captions/SKILL.md"),
            include_str!("../skills/project-setup/SKILL.md"),
        ];
        let mut skills = Vec::with_capacity(TEXTS.len());
        for text in TEXTS {
            if let Ok(skill) = parse_skill_md(text) {
                skills.push(skill);
            }
        }
        Self { skills }
    }

    /// The one skill that matches the ask, if any.
    #[must_use]
    pub fn load(&self, ask: &str) -> Option<&Skill> {
        let ask_words = words(ask);
        let mut best: Option<(u32, &Skill)> = None;
        for skill in &self.skills {
            let score = word_score(&skill.description, &ask_words);
            if score < 2 && !title_in_ask(&skill.title, ask) {
                continue;
            }
            match best {
                Some((best_score, _)) if score <= best_score => {}
                _ => best = Some((score, skill)),
            }
        }
        best.map(|(_, skill)| skill)
    }

    /// At most one skill to put in a prompt.
    #[must_use]
    pub fn for_prompt(&self, ask: &str) -> Vec<&Skill> {
        self.load(ask).into_iter().collect()
    }

    /// Read one skill file after the host allows it.
    pub fn add_outside(
        &mut self,
        path: &Path,
        approver: &mut dyn Approver,
        model_says_approved: bool,
    ) -> Result<(), SkillError> {
        let action = ApprovalAction::AddExternalSkill {
            path: path.display().to_string(),
        };
        let text = authorize_then(approver, action, model_says_approved, || {
            std::fs::read_to_string(path)
        })
        .map_err(|_| SkillError::ApprovalRefused)?;
        let text = text.map_err(|error| SkillError::Io {
            message: error.to_string(),
        })?;
        let skill = parse_skill_md(&text)?;
        self.skills.insert(0, skill);
        Ok(())
    }
}

/// Parse a skill file's title, description, tools, and body.
pub fn parse_skill_md(text: &str) -> Result<Skill, SkillError> {
    let (header, body) = split_header(text);
    let mut title = None;
    let mut description = None;
    let mut tools = None;
    for line in header.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(value) = value_after(line, "title") {
            title = Some(value.to_owned());
        } else if let Some(value) = value_after(line, "description") {
            description = Some(value.to_owned());
        } else if let Some(value) = value_after(line, "tools") {
            tools = Some(split_tools(value));
        }
    }
    let Some(title) = title.filter(|value| !value.is_empty()) else {
        return Err(invalid("missing title"));
    };
    let Some(description) = description.filter(|value| !value.is_empty()) else {
        return Err(invalid("missing description"));
    };
    let Some(allowed_tools) = tools.filter(|names| !names.is_empty()) else {
        return Err(invalid("missing tools"));
    };
    Ok(Skill {
        title,
        description,
        allowed_tools,
        body: body.trim().to_owned(),
    })
}

fn invalid(reason: &str) -> SkillError {
    SkillError::Invalid {
        reason: reason.to_owned(),
    }
}

fn split_header(text: &str) -> (&str, &str) {
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if line.trim().is_empty() {
            return (&text[..offset], &text[offset + line.len()..]);
        }
        offset += line.len();
    }
    (text, "")
}

fn value_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(key)?;
    let rest = rest.strip_prefix(':')?;
    Some(rest.trim())
}

fn split_tools(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

fn words(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn word_score(description: &str, ask_words: &[String]) -> u32 {
    let mut score = 0u32;
    for word in words(description) {
        if word.len() >= 4 && ask_words.iter().any(|ask| ask == &word) {
            score = score.saturating_add(1);
        }
    }
    score
}

fn title_in_ask(title: &str, ask: &str) -> bool {
    !title.is_empty() && ask.to_lowercase().contains(&title.to_lowercase())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::approval::{ApprovalAction, Approver};

    use super::{SkillError, SkillLibrary, parse_skill_md};

    struct Flag {
        allow: bool,
        saw_external: bool,
    }

    impl Approver for Flag {
        fn decide(&mut self, action: &ApprovalAction) -> bool {
            self.saw_external = matches!(
                action,
                ApprovalAction::AddExternalSkill { path } if !path.is_empty()
            );
            self.allow
        }
    }

    struct TempSkill {
        directory: PathBuf,
        path: PathBuf,
    }

    impl TempSkill {
        fn write(text: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "cutlass-agent-skill-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&directory).unwrap();
            let temp = Self {
                path: directory.join("SKILL.md"),
                directory,
            };
            fs::write(&temp.path, text).unwrap();
            temp
        }
    }

    impl Drop for TempSkill {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    struct MissingPath {
        path: PathBuf,
    }

    impl MissingPath {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "cutlass-missing-skill-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            Self { path }
        }
    }

    impl Drop for MissingPath {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn five_bundled_skills_are_data() {
        let library = SkillLibrary::bundled();
        let titles: Vec<&str> = library
            .skills
            .iter()
            .map(|skill| skill.title.as_str())
            .collect();
        assert_eq!(
            titles,
            [
                "Remove pauses",
                "Motion graphics",
                "Product demo",
                "Captions",
                "Project setup",
            ]
        );
        assert_eq!(
            library.skills[0].description,
            "Remove pauses and silent gaps from a range."
        );
        assert_eq!(
            library.skills[0].allowed_tools,
            [
                "speech.start",
                "speech.poll",
                "sandbox.run",
                "draft.query",
                "draft.apply",
            ]
        );
        assert_eq!(
            library.skills[1].description,
            "Build motion graphics with shapes, text, keyframes, and catalog effects."
        );
        assert_eq!(
            library.skills[1].allowed_tools,
            [
                "catalog.search",
                "package.ensure",
                "draft.query",
                "draft.apply",
                "sandbox.run",
            ]
        );
        assert_eq!(
            library.skills[2].description,
            "Cut a product demo with the named media, titles, callouts, zooms, and captions."
        );
        assert_eq!(
            library.skills[2].allowed_tools,
            [
                "media.probe",
                "draft.query",
                "draft.apply",
                "catalog.search",
                "speech.start",
                "speech.poll",
            ]
        );
        assert_eq!(
            library.skills[3].description,
            "Write captions from speech recognition."
        );
        assert_eq!(
            library.skills[3].allowed_tools,
            ["speech.start", "speech.poll", "draft.query", "draft.apply"]
        );
        assert_eq!(
            library.skills[4].description,
            "Set the canvas, frame rate, and background for a new project."
        );
        assert_eq!(
            library.skills[4].allowed_tools,
            ["draft.query", "draft.apply"]
        );
        assert!(library.skills.iter().all(|skill| !skill.body.is_empty()));
        assert!(
            library.skills[0]
                .body
                .contains("does not write the draft during a frame")
        );
    }

    #[test]
    fn load_matches_the_description_and_omits_the_others() {
        let library = SkillLibrary::bundled();
        let skill = library.load("please remove pauses").unwrap();
        assert_eq!(skill.title, "Remove pauses");
        let prompt = library.for_prompt("please remove pauses");
        assert_eq!(prompt.len(), 1);
        assert_eq!(prompt[0].title, "Remove pauses");
        assert!(prompt.iter().all(|skill| skill.title != "Motion graphics"
            && skill.title != "Product demo"
            && skill.title != "Captions"
            && skill.title != "Project setup"));
    }

    #[test]
    fn prompt_does_not_include_every_skill() {
        let library = SkillLibrary::bundled();
        let prompt =
            library.for_prompt("please remove pauses and write captions from speech recognition");
        assert_eq!(prompt.len(), 1);
        assert_ne!(prompt.len(), library.skills.len());
        assert_eq!(prompt[0].title, "Captions");
        assert_eq!(library.for_prompt("hello").len(), 0);
        assert_ne!(library.skills.len(), 1);
    }

    #[test]
    fn hello_matches_nothing() {
        let library = SkillLibrary::bundled();
        assert!(library.load("hello").is_none());
        assert!(library.for_prompt("hello").is_empty());
    }

    #[test]
    fn a_skill_plans_tool_calls_and_not_a_document() {
        let library = SkillLibrary::bundled();
        let skill = library.load("please remove pauses").unwrap();
        let plan = skill.plan();
        assert_eq!(
            plan.calls,
            [
                "speech.start".to_owned(),
                "speech.poll".to_owned(),
                "sandbox.run".to_owned(),
                "draft.query".to_owned(),
                "draft.apply".to_owned(),
            ]
        );
        assert!(plan.calls.iter().all(|call| !call.is_empty()));
        assert_eq!(plan.calls, skill.allowed_tools);
        assert!(!skill.body.contains("SKILL.md"));
        assert!(!skill.body.to_lowercase().contains("document"));
        assert!(
            skill
                .body
                .contains("does not write the draft during a frame")
        );
    }

    #[test]
    fn refused_external_skill_is_not_added() {
        let mut library = SkillLibrary::bundled();
        let mut approver = Flag {
            allow: false,
            saw_external: false,
        };
        let missing = MissingPath::new();
        assert!(!missing.path.exists());
        let error = library
            .add_outside(&missing.path, &mut approver, true)
            .unwrap_err();
        assert_eq!(error, SkillError::ApprovalRefused);
        assert!(error.to_string().contains("refused"));
        assert!(approver.saw_external);
        assert!(!missing.path.exists());
        assert_eq!(library.skills.len(), 5);
        assert!(library.load("gather ocean debris").is_none());
    }

    #[test]
    fn allowed_external_skill_loads() {
        let mut library = SkillLibrary::bundled();
        let ocean = TempSkill::write(
            "title: Ocean cleanup\ndescription: Gather ocean debris from shorelines.\ntools: draft.query, draft.apply\n\nQuery the draft, then apply.\n",
        );
        let mut approver = Flag {
            allow: true,
            saw_external: false,
        };
        library
            .add_outside(&ocean.path, &mut approver, false)
            .unwrap();
        assert!(approver.saw_external);
        let loaded = library.load("gather ocean debris").unwrap();
        assert_eq!(loaded.title, "Ocean cleanup");
        assert_eq!(library.for_prompt("gather ocean debris").len(), 1);

        let tied = TempSkill::write(
            "title: Quiet gaps\ndescription: Remove pauses from a talking range.\ntools: draft.query\n\nQuery the draft.\n",
        );
        library
            .add_outside(&tied.path, &mut approver, false)
            .unwrap();
        let tied_load = library.load("please remove pauses").unwrap();
        assert_eq!(tied_load.title, "Quiet gaps");
        assert_eq!(library.for_prompt("please remove pauses").len(), 1);
    }

    #[test]
    fn invalid_skill_text_is_refused() {
        assert!(matches!(
            parse_skill_md("description: Something long enough\ntools: draft.query\n\nBody"),
            Err(SkillError::Invalid { .. })
        ));
        assert!(matches!(
            parse_skill_md("title: Title\ntools: draft.query\n\nBody"),
            Err(SkillError::Invalid { .. })
        ));
        assert!(matches!(
            parse_skill_md("title: Title\ndescription: Something long enough\n\nBody"),
            Err(SkillError::Invalid { .. })
        ));
        assert!(matches!(
            parse_skill_md(""),
            Err(SkillError::Invalid { .. })
        ));
        assert!(matches!(
            parse_skill_md("title: Title\ndescription: Something\ntools: , ,\n\nBody"),
            Err(SkillError::Invalid { reason }) if reason.contains("tools")
        ));

        let broken = TempSkill::write("title: Broken\n\nno tools here\n");
        let mut library = SkillLibrary::bundled();
        let mut approver = Flag {
            allow: true,
            saw_external: false,
        };
        let error = library
            .add_outside(&broken.path, &mut approver, false)
            .unwrap_err();
        assert!(matches!(error, SkillError::Invalid { .. }));
        assert_eq!(library.skills.len(), 5);
        assert!(library.load("Broken").is_none());
    }
}
