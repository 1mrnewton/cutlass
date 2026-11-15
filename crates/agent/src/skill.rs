//! Bundled skills, the skill folders the user added in settings, and one
//! external skill the host has allowed.
//!
//! A skill folder holds a `SKILL.md`, or folders that each hold one. The
//! file is the bundled form (`title:`, `description:`, `tools:` lines, a
//! blank line, then the body) or a front matter block between `---` lines
//! naming `name` and `description`, where `tools` may be left out.

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
    /// Other words an ask may use for this skill, from a `keywords:` line.
    pub keywords: Vec<String>,
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
    /// Skill files in the user's folders that could not be read, with why.
    problems: Vec<String>,
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
        Self {
            skills,
            problems: Vec::new(),
        }
    }

    /// The bundled skills, then each skill in `folders`, which the user
    /// already allowed when adding them in settings. A user skill comes
    /// first, so it wins over a bundled one that matches as well. A file
    /// that does not read is kept as a problem, not a failure.
    #[must_use]
    pub fn with_folders(mut self, folders: &[cutlass_settings::SkillFolder]) -> Self {
        let mut found = Vec::new();
        for folder in folders {
            if folder.path.trim().is_empty() {
                continue;
            }
            let root = Path::new(&folder.path);
            let direct = root.join("SKILL.md");
            if direct.is_file() {
                found.push(direct);
                continue;
            }
            if root.is_file() {
                found.push(root.to_path_buf());
                continue;
            }
            match std::fs::read_dir(root) {
                Ok(entries) => {
                    let mut inner: Vec<_> = entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path().join("SKILL.md"))
                        .filter(|path| path.is_file())
                        .collect();
                    inner.sort();
                    if inner.is_empty() {
                        self.problems
                            .push(format!("{}: no SKILL.md in it", root.display()));
                    }
                    found.extend(inner);
                }
                Err(error) => self.problems.push(format!("{}: {error}", root.display())),
            }
        }
        let mut user = Vec::new();
        for path in found {
            match std::fs::read_to_string(&path).map_err(|error| error.to_string()) {
                Ok(text) => match parse_skill_md(&text) {
                    Ok(skill) => user.push(skill),
                    Err(error) => self.problems.push(format!("{}: {error}", path.display())),
                },
                Err(error) => self.problems.push(format!("{}: {error}", path.display())),
            }
        }
        user.append(&mut self.skills);
        self.skills = user;
        self
    }

    /// Every skill, in the order `load` prefers them.
    #[must_use]
    pub fn skills(&self) -> &[Skill] {
        &self.skills
    }

    /// Skill files in the user's folders that did not read.
    #[must_use]
    pub fn problems(&self) -> &[String] {
        &self.problems
    }

    /// One line a skill: its title and what it is for, then any skill file
    /// that did not read.
    #[must_use]
    pub fn listing(&self) -> String {
        let mut out = String::new();
        for skill in &self.skills {
            out.push_str(&skill.title);
            out.push_str(": ");
            out.push_str(&skill.description);
            out.push('\n');
        }
        for problem in &self.problems {
            out.push_str("did not load: ");
            out.push_str(problem);
            out.push('\n');
        }
        out.trim_end().to_owned()
    }

    /// The one skill `ask` is for, or none. A skill is found by its title,
    /// or by the words of its title, description and keywords: an ask word
    /// counts once, twice when only that skill has it, and a skill needs
    /// two. Words are compared without their endings, so "removing pauses"
    /// finds "Remove pauses".
    #[must_use]
    pub fn load(&self, ask: &str) -> Option<&Skill> {
        let ask = plain(ask);
        let ask_stems = stems(&ask);
        let skill_stems: Vec<Vec<String>> = self
            .skills
            .iter()
            .map(|skill| {
                stems(&format!(
                    "{} {} {}",
                    skill.title,
                    skill.description,
                    skill.keywords.join(" ")
                ))
            })
            .collect();
        let mut best: Option<(u32, &Skill)> = None;
        for (skill, own) in self.skills.iter().zip(&skill_stems) {
            let mut score = 0u32;
            for stem in &ask_stems {
                if !own.contains(stem) {
                    continue;
                }
                let holders = skill_stems
                    .iter()
                    .filter(|stems| stems.contains(stem))
                    .count();
                score = score.saturating_add(if holders == 1 { 2 } else { 1 });
            }
            if score < 2 && !ask.contains(&plain(&skill.title)) {
                continue;
            }
            match best {
                Some((best_score, _)) if score <= best_score => {}
                _ => best = Some((score, skill)),
            }
        }
        best.map(|(_, skill)| skill)
    }

    /// The skills' titles, for an ask that found none.
    #[must_use]
    pub fn titles(&self) -> Vec<&str> {
        self.skills
            .iter()
            .map(|skill| skill.title.as_str())
            .collect()
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
    let (header, body, front_matter) = match front_matter(text) {
        Some((header, body)) => (header, body, true),
        None => {
            let (header, body) = split_header(text);
            (header, body, false)
        }
    };
    let mut title = None;
    let mut description = None;
    let mut tools = None;
    let mut keywords = Vec::new();
    for line in header.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(value) = value_after(line, "title").or_else(|| value_after(line, "name")) {
            title = Some(value.to_owned());
        } else if let Some(value) = value_after(line, "description") {
            description = Some(value.to_owned());
        } else if let Some(value) =
            value_after(line, "tools").or_else(|| value_after(line, "allowed-tools"))
        {
            tools = Some(split_tools(value));
        } else if let Some(value) = value_after(line, "keywords") {
            keywords = split_tools(value);
        }
    }
    let Some(title) = title.filter(|value| !value.is_empty()) else {
        return Err(invalid("missing title"));
    };
    let Some(description) = description.filter(|value| !value.is_empty()) else {
        return Err(invalid("missing description"));
    };
    // A front matter skill may leave its tools out; it may then call any.
    let allowed_tools = match tools.filter(|names| !names.is_empty()) {
        Some(names) => names,
        None if front_matter => Vec::new(),
        None => return Err(invalid("missing tools")),
    };
    Ok(Skill {
        title,
        description,
        allowed_tools,
        body: body.trim().to_owned(),
        keywords,
    })
}

fn invalid(reason: &str) -> SkillError {
    SkillError::Invalid {
        reason: reason.to_owned(),
    }
}

/// The header between a first `---` line and the next, and the body after.
fn front_matter(text: &str) -> Option<(&str, &str)> {
    let rest = text.trim_start_matches('\u{feff}');
    let rest = rest
        .strip_prefix("---\n")
        .or_else(|| rest.strip_prefix("---\r\n"))?;
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim() == "---" {
            return Some((&rest[..offset], &rest[offset + line.len()..]));
        }
        offset += line.len();
    }
    None
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
    Some(rest.trim().trim_matches(|c| c == '"' || c == '\''))
}

fn split_tools(value: &str) -> Vec<String> {
    value
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|name| name.trim().trim_matches(|c| c == '"' || c == '\''))
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

/// `text` in lower case with `-` and `_` as spaces, so a folder name such
/// as `project-setup` reads as the title does.
fn plain(text: &str) -> String {
    text.to_lowercase().replace(['-', '_'], " ")
}

/// The words of `text` of four letters or more, each without its ending
/// and once only.
fn stems(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in plain(text).split(|character: char| !character.is_alphanumeric()) {
        if word.chars().count() < 4 {
            continue;
        }
        let stem = stem(word);
        if !out.contains(&stem) {
            out.push(stem);
        }
    }
    out
}

/// `word` without a common English ending, so "removing", "removes" and
/// "remove" compare equal. It is a match key, not a word.
fn stem(word: &str) -> String {
    for ending in ["ing", "es", "ed", "s", "e"] {
        if let Some(stem) = word.strip_suffix(ending)
            && stem.chars().count() >= 3
        {
            return stem.to_owned();
        }
    }
    word.to_owned()
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
    fn a_skill_is_found_by_other_wordings_of_its_name_and_by_its_keywords() {
        let library = SkillLibrary::bundled();
        let found = |ask: &str| library.load(ask).map(|skill| skill.title.as_str());
        // The wordings the system prompt itself uses.
        for (ask, title) in [
            ("captions", "Captions"),
            ("removing pauses", "Remove pauses"),
            ("motion graphics", "Motion graphics"),
            ("a product demo", "Product demo"),
            ("setting up a project", "Project setup"),
        ] {
            assert_eq!(found(ask), Some(title), "{ask}");
        }
        // Folder names, and words only the keywords hold.
        for (ask, title) in [
            ("project-setup", "Project setup"),
            ("remove_pauses", "Remove pauses"),
            ("subtitles", "Captions"),
            ("add subtitles to my video", "Captions"),
            ("cut the silences", "Remove pauses"),
            ("make it vertical for reels", "Project setup"),
            ("animate a lower third", "Motion graphics"),
        ] {
            assert_eq!(found(ask), Some(title), "{ask}");
        }
        for ask in ["hello", "export the video", "what time is it", ""] {
            assert_eq!(found(ask), None, "{ask}");
        }
    }

    #[test]
    fn each_bundled_skill_names_requests_that_exist() {
        let library = SkillLibrary::bundled();
        let known: Vec<&str> = crate::ops::ops().iter().map(|op| op.name).collect();
        for skill in &library.skills {
            let words: Vec<&str> = skill
                .body
                .split(|character: char| !(character.is_ascii_lowercase() || character == '_'))
                .collect();
            let named = words.iter().filter(|word| known.contains(word)).count();
            assert!(named >= 3, "{} names few requests", skill.title);
            // Every word written like a request kind is one.
            for word in words {
                let written_as_one = ["set_", "add_", "key_"]
                    .iter()
                    .any(|start| word.starts_with(start) && word.len() > start.len());
                assert!(
                    !written_as_one || known.contains(&word),
                    "{}: no request {word}",
                    skill.title
                );
            }
            assert!(skill.body.len() > 600, "{} says little", skill.title);
        }
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

    #[test]
    fn skills_in_the_users_folders_load_first_and_bad_ones_are_listed() {
        let root =
            std::env::temp_dir().join(format!("cutlass-skill-folders-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let one = root.join("shop").join("intro");
        let two = root.join("shop").join("broken");
        fs::create_dir_all(&one).unwrap();
        fs::create_dir_all(&two).unwrap();
        fs::write(
            one.join("SKILL.md"),
            "---\nname: Shop intro\ndescription: \"Open a shop video with the logo and a title\"\n---\n\nPlace the logo, then a title.\n",
        )
        .unwrap();
        fs::write(two.join("SKILL.md"), "no header at all").unwrap();
        let single = root.join("single");
        fs::create_dir_all(&single).unwrap();
        fs::write(
            single.join("SKILL.md"),
            "title: Captions\ndescription: Write captions from speech recognition, my way.\ntools: speech.start, draft.apply\n\nMine.\n",
        )
        .unwrap();
        let folders = [
            cutlass_settings::SkillFolder {
                path: root.join("shop").display().to_string(),
                ..Default::default()
            },
            cutlass_settings::SkillFolder {
                path: single.display().to_string(),
                ..Default::default()
            },
            cutlass_settings::SkillFolder {
                path: root.join("gone").display().to_string(),
                ..Default::default()
            },
        ];
        let library = SkillLibrary::bundled().with_folders(&folders);
        assert_eq!(library.skills().len(), 7);
        let intro = library.load("open my shop video with the logo").unwrap();
        assert_eq!(intro.title, "Shop intro");
        assert!(intro.allowed_tools.is_empty());
        assert_eq!(intro.body, "Place the logo, then a title.");
        // The user's Captions wins over the bundled one.
        let captions = library.load("write captions from speech").unwrap();
        assert_eq!(captions.body, "Mine.");
        assert_eq!(library.problems().len(), 2, "{:?}", library.problems());
        let listing = library.listing();
        assert!(
            listing.contains("Shop intro: Open a shop video"),
            "{listing}"
        );
        assert!(listing.contains("did not load:"), "{listing}");
        let _ = fs::remove_dir_all(&root);
    }
}
