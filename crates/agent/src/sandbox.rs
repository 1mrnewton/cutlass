//! Local and online code runners take source and JSON and do not open the project folder.

use std::fmt;

use cutlass_settings::Settings;

/// Why a sandbox call did not return JSON.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxError {
    /// The input or the runner output is not one JSON value.
    NotJson,
    /// The source names a network address.
    Network,
    /// The source or the input names the home directory.
    Home,
    /// The source or the input names the project folder.
    ProjectPath,
    /// The source tries to save the project.
    SaveProject,
    /// The source tries to change a shortcut.
    ChangeShortcut,
    /// The source tries to publish a package.
    PublishPackage,
    /// The online runner is off.
    OnlineDisabled,
    /// The injected runner failed.
    Executor(String),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotJson => formatter.write_str("not json"),
            Self::Network => formatter.write_str("network"),
            Self::Home => formatter.write_str("home"),
            Self::ProjectPath => formatter.write_str("project path"),
            Self::SaveProject => formatter.write_str("save project"),
            Self::ChangeShortcut => formatter.write_str("change shortcut"),
            Self::PublishPackage => formatter.write_str("publish package"),
            Self::OnlineDisabled => formatter.write_str("online sandbox is off"),
            Self::Executor(message) => write!(formatter, "executor: {message}"),
        }
    }
}

impl std::error::Error for SandboxError {}

/// Runs source against a JSON input and returns JSON.
pub trait Sandbox {
    /// Run one call and return the runner's JSON text.
    fn run(&mut self, source: &str, input_json: &str) -> Result<String, SandboxError>;
}

/// The injected runner. The sandbox calls it only after the checks pass.
pub trait SandboxExecutor {
    /// Run the source and return JSON text.
    fn execute(&mut self, source: &str, input_json: &str) -> Result<String, SandboxError>;
}

/// Paths the local runner must not see in the source or the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxPolicy {
    /// Project folder text. Empty skips the check.
    pub project_path: String,
    /// Home directory text. Empty skips the check.
    pub home_path: String,
}

/// The local runner compares strings and does not open a socket, a home
/// directory, or the project folder.
#[derive(Debug)]
pub struct LocalSandbox<E> {
    policy: SandboxPolicy,
    executor: E,
}

impl<E> LocalSandbox<E> {
    /// A local runner with the given path checks and injected executor.
    #[must_use]
    pub fn new(policy: SandboxPolicy, executor: E) -> Self {
        Self { policy, executor }
    }

    /// The injected runner.
    #[must_use]
    pub fn executor(&self) -> &E {
        &self.executor
    }

    /// The injected runner, mutable.
    #[must_use]
    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }
}

impl<E: SandboxExecutor> Sandbox for LocalSandbox<E> {
    fn run(&mut self, source: &str, input_json: &str) -> Result<String, SandboxError> {
        screen(source, input_json, &self.policy)?;
        finish(self.executor.execute(source, input_json))
    }
}

/// The online runner stays off until settings enable it and does not follow
/// the cloud model switch.
#[derive(Debug)]
pub struct OnlineSandbox<E> {
    enabled: bool,
    /// Empty paths. This runner is not given a project folder or a home directory.
    policy: SandboxPolicy,
    executor: E,
}

impl<E> OnlineSandbox<E> {
    /// Build an online runner from the sandbox switch.
    #[must_use]
    pub fn from_settings(settings: &Settings, executor: E) -> Self {
        Self {
            enabled: settings.sandbox.online,
            policy: SandboxPolicy {
                project_path: String::new(),
                home_path: String::new(),
            },
            executor,
        }
    }

    /// Whether the online runner may run.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl<E: SandboxExecutor> Sandbox for OnlineSandbox<E> {
    fn run(&mut self, source: &str, input_json: &str) -> Result<String, SandboxError> {
        if !self.enabled {
            return Err(SandboxError::OnlineDisabled);
        }
        screen(source, input_json, &self.policy)?;
        finish(self.executor.execute(source, input_json))
    }
}

/// Which runner a call asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxRoute {
    /// The local runner.
    Local,
    /// The online runner.
    Online,
}

/// Return the runner the call asked for.
#[must_use]
pub fn route_for(settings: &Settings, requested: SandboxRoute) -> SandboxRoute {
    match (requested, settings.model.cloud, settings.sandbox.online) {
        (SandboxRoute::Local, _, _) => SandboxRoute::Local,
        (SandboxRoute::Online, _, _) => SandboxRoute::Online,
    }
}

fn finish(result: Result<String, SandboxError>) -> Result<String, SandboxError> {
    let output = result?;
    if is_json(&output) {
        Ok(output)
    } else {
        Err(SandboxError::NotJson)
    }
}

fn screen(source: &str, input_json: &str, policy: &SandboxPolicy) -> Result<(), SandboxError> {
    if !is_json(input_json) {
        return Err(SandboxError::NotJson);
    }
    if path_hit(&policy.project_path, source, input_json) {
        return Err(SandboxError::ProjectPath);
    }
    if path_hit(&policy.home_path, source, input_json) {
        return Err(SandboxError::Home);
    }
    if source.contains("://") {
        return Err(SandboxError::Network);
    }
    if source.contains("save_project") {
        return Err(SandboxError::SaveProject);
    }
    if source.contains("change_shortcut") {
        return Err(SandboxError::ChangeShortcut);
    }
    if source.contains("publish_package") {
        return Err(SandboxError::PublishPackage);
    }
    Ok(())
}

fn path_hit(path: &str, source: &str, input_json: &str) -> bool {
    !path.is_empty() && (source.contains(path) || input_json.contains(path))
}

const MAX_JSON_DEPTH: u32 = 64;

fn is_json(input: &str) -> bool {
    let mut scan = Scan {
        bytes: input.as_bytes(),
        index: 0,
    };
    if !scan.value(0) {
        return false;
    }
    scan.skip_ws();
    scan.index == scan.bytes.len()
}

struct Scan<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl Scan<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.index).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.index += 1;
        Some(byte)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.bump();
        }
    }

    fn value(&mut self, depth: u32) -> bool {
        if depth >= MAX_JSON_DEPTH {
            return false;
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string(),
            Some(b't') => self.literal(b"true"),
            Some(b'f') => self.literal(b"false"),
            Some(b'n') => self.literal(b"null"),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => false,
        }
    }

    fn object(&mut self, depth: u32) -> bool {
        if self.bump() != Some(b'{') {
            return false;
        }
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.bump();
            return true;
        }
        loop {
            self.skip_ws();
            if !self.string() {
                return false;
            }
            self.skip_ws();
            if self.bump() != Some(b':') {
                return false;
            }
            if !self.value(depth + 1) {
                return false;
            }
            self.skip_ws();
            match self.bump() {
                Some(b'}') => return true,
                Some(b',') => {}
                _ => return false,
            }
        }
    }

    fn array(&mut self, depth: u32) -> bool {
        if self.bump() != Some(b'[') {
            return false;
        }
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.bump();
            return true;
        }
        loop {
            if !self.value(depth + 1) {
                return false;
            }
            self.skip_ws();
            match self.bump() {
                Some(b']') => return true,
                Some(b',') => {}
                _ => return false,
            }
        }
    }

    fn string(&mut self) -> bool {
        if self.bump() != Some(b'"') {
            return false;
        }
        loop {
            match self.bump() {
                None => return false,
                Some(b'"') => return true,
                Some(b'\\') => {
                    if !self.escape() {
                        return false;
                    }
                }
                Some(byte) if byte < 0x20 => return false,
                Some(_) => {}
            }
        }
    }

    fn escape(&mut self) -> bool {
        match self.bump() {
            Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => true,
            Some(b'u') => {
                (0..4).all(|_| matches!(self.bump(), Some(byte) if byte.is_ascii_hexdigit()))
            }
            _ => false,
        }
    }

    fn number(&mut self) -> bool {
        if self.peek() == Some(b'-') {
            self.bump();
        }
        match self.bump() {
            Some(b'0') => {}
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.bump();
                }
            }
            _ => return false,
        }
        if self.peek() == Some(b'.') {
            self.bump();
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return false;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.bump();
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.bump();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump();
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return false;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.bump();
            }
        }
        true
    }

    fn literal(&mut self, expected: &[u8]) -> bool {
        expected.iter().all(|byte| self.bump() == Some(*byte))
    }
}

#[cfg(test)]
mod tests {
    use cutlass_settings::Settings;

    use super::{
        LocalSandbox, OnlineSandbox, Sandbox, SandboxError, SandboxExecutor, SandboxPolicy,
        SandboxRoute, route_for,
    };

    struct Counting {
        calls: u32,
        output: Option<String>,
    }

    impl SandboxExecutor for Counting {
        fn execute(&mut self, _source: &str, input_json: &str) -> Result<String, SandboxError> {
            self.calls += 1;
            Ok(match &self.output {
                Some(output) => output.clone(),
                None => input_json.to_owned(),
            })
        }
    }

    fn policy() -> SandboxPolicy {
        SandboxPolicy {
            project_path: "/tmp/project".to_owned(),
            home_path: "/Users/someone".to_owned(),
        }
    }

    fn local_sandbox() -> LocalSandbox<Counting> {
        LocalSandbox::new(
            policy(),
            Counting {
                calls: 0,
                output: None,
            },
        )
    }

    #[test]
    fn local_runner_returns_json() {
        let mut sandbox = local_sandbox();
        let input = "{\"gaps\":[]}";
        let output = sandbox.run("gaps", input).unwrap();
        assert_eq!(output, input);
        assert_eq!(sandbox.executor().calls, 1);

        for input in [
            "{}",
            "[]",
            "null",
            "true",
            "false",
            "0",
            "-2.5",
            "\"hi\"",
            "  {\"gaps\":[]}  ",
            "\"say \\\"hi\\\"\"",
        ] {
            let mut sandbox = local_sandbox();
            assert_eq!(sandbox.run("gaps", input).unwrap(), input);
            assert_eq!(sandbox.executor().calls, 1);
        }
    }

    #[test]
    fn local_runner_rejects_non_json() {
        for input in [
            "",
            " ",
            "nope",
            "'hi'",
            "{",
            "{\"a\":1} trailing",
            "01",
            "+1",
            "[1,]",
            "/tmp/project",
        ] {
            let mut sandbox = local_sandbox();
            let error = sandbox.run("gaps", input).unwrap_err();
            assert_eq!(error, SandboxError::NotJson);
            assert_eq!(error.to_string(), "not json");
            assert_eq!(sandbox.executor().calls, 0);
        }

        let mut sandbox = LocalSandbox::new(
            policy(),
            Counting {
                calls: 0,
                output: Some("not-json".to_owned()),
            },
        );
        let error = sandbox.run("gaps", "{}").unwrap_err();
        assert_eq!(error, SandboxError::NotJson);
        assert_eq!(sandbox.executor().calls, 1);
        assert_eq!(
            SandboxError::Executor("down".to_owned()).to_string(),
            "executor: down"
        );
    }

    #[test]
    fn local_runner_rejects_network() {
        let mut sandbox = local_sandbox();
        let error = sandbox.run("fetch https://example.com", "{}").unwrap_err();
        assert_eq!(error, SandboxError::Network);
        assert_eq!(error.to_string(), "network");
        assert_eq!(sandbox.executor().calls, 0);
    }

    #[test]
    fn local_runner_rejects_home() {
        let mut sandbox = local_sandbox();
        let error = sandbox.run("read /Users/someone/note", "{}").unwrap_err();
        assert_eq!(error, SandboxError::Home);
        assert_eq!(error.to_string(), "home");
        assert_eq!(sandbox.executor().calls, 0);

        let error = sandbox
            .run("https://example.com/Users/someone", "{}")
            .unwrap_err();
        assert_eq!(error, SandboxError::Home);

        let error = sandbox
            .run("gaps", "{\"home\":\"/Users/someone\"}")
            .unwrap_err();
        assert_eq!(error, SandboxError::Home);
        assert_eq!(sandbox.executor().calls, 0);
    }

    #[test]
    fn local_runner_rejects_project_path() {
        let mut sandbox = local_sandbox();
        let error = sandbox.run("read /tmp/project/draft", "{}").unwrap_err();
        assert_eq!(error, SandboxError::ProjectPath);
        assert_eq!(error.to_string(), "project path");
        assert_eq!(sandbox.executor().calls, 0);

        let error = sandbox
            .run("https://example.com/tmp/project", "{}")
            .unwrap_err();
        assert_eq!(error, SandboxError::ProjectPath);

        let error = sandbox
            .run("gaps", "{\"path\":\"/tmp/project\"}")
            .unwrap_err();
        assert_eq!(error, SandboxError::ProjectPath);
        assert_eq!(sandbox.executor().calls, 0);
    }

    #[test]
    fn local_runner_cannot_save_the_project() {
        let mut sandbox = local_sandbox();
        let error = sandbox.run("save_project()", "{}").unwrap_err();
        assert_eq!(error, SandboxError::SaveProject);
        assert_eq!(error.to_string(), "save project");
        assert_eq!(sandbox.executor().calls, 0);
    }

    #[test]
    fn local_runner_cannot_change_a_shortcut() {
        let mut sandbox = local_sandbox();
        let error = sandbox.run("change_shortcut()", "{}").unwrap_err();
        assert_eq!(error, SandboxError::ChangeShortcut);
        assert_eq!(error.to_string(), "change shortcut");
        assert_eq!(sandbox.executor().calls, 0);
    }

    #[test]
    fn local_runner_cannot_publish_a_package() {
        let mut sandbox = local_sandbox();
        let error = sandbox.run("publish_package()", "{}").unwrap_err();
        assert_eq!(error, SandboxError::PublishPackage);
        assert_eq!(error.to_string(), "publish package");
        assert_eq!(sandbox.executor().calls, 0);
    }

    #[test]
    fn denied_call_does_not_reach_the_executor() {
        let mut sandbox = local_sandbox();
        let denied = [
            ("gaps", "nope"),
            ("use https://example.com", "{}"),
            ("read /Users/someone/note", "{}"),
            ("read /tmp/project/draft", "{}"),
            ("save_project()", "{}"),
            ("change_shortcut()", "{}"),
            ("publish_package()", "{}"),
        ];
        for (source, input) in denied {
            assert!(sandbox.run(source, input).is_err());
        }
        assert_eq!(sandbox.executor().calls, 0);
    }

    #[test]
    fn online_runner_stays_off_until_enabled() {
        let mut online = OnlineSandbox::from_settings(
            &Settings::default(),
            Counting {
                calls: 0,
                output: None,
            },
        );
        assert!(!online.is_enabled());
        let error = online.run("save_project()", "not-json").unwrap_err();
        assert_eq!(error, SandboxError::OnlineDisabled);
        assert_eq!(error.to_string(), "online sandbox is off");
        assert_eq!(online.executor.calls, 0);
    }

    #[test]
    fn online_runner_does_not_follow_the_cloud_model_switch() {
        let mut settings = Settings::default();
        settings.model.cloud = true;
        settings.sandbox.online = false;
        let mut online = OnlineSandbox::from_settings(
            &settings,
            Counting {
                calls: 0,
                output: None,
            },
        );
        assert!(!online.is_enabled());
        let error = online.run("return input", "{}").unwrap_err();
        assert_eq!(error, SandboxError::OnlineDisabled);
        assert_eq!(online.executor.calls, 0);
    }

    #[test]
    fn enabled_online_runner_uses_the_injected_executor() {
        let mut settings = Settings::default();
        settings.sandbox.online = true;
        settings.model.cloud = false;
        let mut online = OnlineSandbox::from_settings(
            &settings,
            Counting {
                calls: 0,
                output: None,
            },
        );
        assert!(online.is_enabled());
        assert_eq!(online.run("return input", "{}").unwrap(), "{}");
        assert_eq!(online.executor.calls, 1);
        let error = online.run("save_project()", "{}").unwrap_err();
        assert_eq!(error, SandboxError::SaveProject);
        assert_eq!(online.executor.calls, 1);
    }

    #[test]
    fn route_ignores_both_switches() {
        let mut settings = Settings::default();
        assert_eq!(
            route_for(&settings, SandboxRoute::Local),
            SandboxRoute::Local
        );
        assert_eq!(
            route_for(&settings, SandboxRoute::Online),
            SandboxRoute::Online
        );
        settings.model.cloud = true;
        settings.sandbox.online = false;
        assert_eq!(
            route_for(&settings, SandboxRoute::Online),
            SandboxRoute::Online
        );
        assert_eq!(
            route_for(&settings, SandboxRoute::Local),
            SandboxRoute::Local
        );
        settings.model.cloud = false;
        settings.sandbox.online = true;
        assert_eq!(
            route_for(&settings, SandboxRoute::Online),
            SandboxRoute::Online
        );
        assert_eq!(
            route_for(&settings, SandboxRoute::Local),
            SandboxRoute::Local
        );
        settings.model.cloud = true;
        settings.sandbox.online = true;
        assert_eq!(
            route_for(&settings, SandboxRoute::Local),
            SandboxRoute::Local
        );
        assert_eq!(
            route_for(&settings, SandboxRoute::Online),
            SandboxRoute::Online
        );
    }
}
