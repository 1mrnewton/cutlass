//! One log for the desktop app.
//!
//! The file is `cutlass.log` under the user logs folder. A line names the
//! open project. When the file passes 4 MiB it becomes `cutlass.log.1` and a
//! new file starts. wgpu warnings land here too. A codec crate still returns
//! an error. The desktop code that handles it writes the line.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// A full log is renamed aside. One previous file is kept.
const LIMIT: u64 = 4 * 1024 * 1024;

static INSTALLED: OnceLock<&'static Logger> = OnceLock::new();

struct Logger {
    inner: Mutex<State>,
}

struct State {
    path: PathBuf,
    file: Option<File>,
    limit: u64,
    project: String,
}

/// Opens the log before the window exists. A missing home directory leaves the app quiet.
pub(crate) fn open() {
    let Some(dir) = logs_dir() else {
        return;
    };
    let path = dir.join("cutlass.log");
    let Ok(state) = State::create(&path, LIMIT) else {
        return;
    };
    let logger = Box::leak(Box::new(Logger {
        inner: Mutex::new(state),
    }));
    if log::set_logger(logger).is_err() {
        return;
    }
    let _ = INSTALLED.set(logger);
    log::set_max_level(log::LevelFilter::Info);
    log::info!("The log is {}.", path.display());
}

/// The name written on the following lines. An empty name reads as no project.
pub(crate) fn project(name: &str) {
    let Some(logger) = INSTALLED.get().copied() else {
        return;
    };
    let mut state = logger.lock();
    if state.project != name {
        state.project.clear();
        state.project.push_str(name);
    }
}

impl Logger {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        keeps(metadata.level(), metadata.target())
    }

    fn log(&self, record: &log::Record) {
        if !keeps(record.level(), record.target()) {
            return;
        }
        let message = record.args().to_string();
        self.lock().write(record.level(), &message);
    }

    fn flush(&self) {
        if let Some(file) = &mut self.lock().file {
            let _ = file.flush();
        }
    }
}

impl State {
    fn create(path: &Path, limit: u64) -> std::io::Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = Some(open_append(path)?);
        Ok(Self {
            path: path.to_owned(),
            file,
            limit,
            project: String::new(),
        })
    }

    fn write(&mut self, level: log::Level, message: &str) {
        if self.roll().is_err() {
            return;
        }
        let Some(file) = &mut self.file else {
            return;
        };
        let line = format_line(
            &stamp(SystemTime::now()),
            level_word(level),
            &self.project,
            message,
        );
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }

    fn roll(&mut self) -> std::io::Result<()> {
        if self.file.is_none() {
            self.file = Some(open_append(&self.path)?);
        }
        let Some(file) = &self.file else {
            return Ok(());
        };
        if file.metadata()?.len() < self.limit {
            return Ok(());
        }
        self.file.take();
        let previous = previous_path(&self.path);
        let _ = std::fs::remove_file(&previous);
        if std::fs::rename(&self.path, &previous).is_err() {
            self.file = Some(open_append(&self.path)?);
            return Ok(());
        }
        self.file = Some(open_append(&self.path)?);
        Ok(())
    }
}

fn open_append(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

fn logs_dir() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        home_dir().map(|home| macos_dir(&home))
    }
    #[cfg(target_os = "windows")]
    {
        local_dir().map(|local| windows_dir(&local))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        state_dir().map(|state| linux_dir(&state))
    }
}

#[cfg(any(test, target_os = "macos"))]
fn macos_dir(home: &Path) -> PathBuf {
    home.join("Library").join("Logs").join("Cutlass")
}

#[cfg(any(test, target_os = "windows"))]
fn windows_dir(local: &Path) -> PathBuf {
    local.join("Cutlass").join("Logs")
}

#[cfg(any(test, not(any(target_os = "macos", target_os = "windows"))))]
fn linux_dir(state: &Path) -> PathBuf {
    state.join("cutlass")
}

#[cfg(not(target_os = "windows"))]
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(target_os = "windows")]
fn local_dir() -> Option<PathBuf> {
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        return Some(PathBuf::from(local));
    }
    std::env::var_os("USERPROFILE").map(|home| PathBuf::from(home).join("AppData").join("Local"))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn state_dir() -> Option<PathBuf> {
    if let Some(state) = std::env::var_os("XDG_STATE_HOME") {
        let state = PathBuf::from(state);
        if !state.as_os_str().is_empty() {
            return Some(state);
        }
    }
    home_dir().map(|home| home.join(".local").join("state"))
}

fn previous_path(path: &Path) -> PathBuf {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return path.with_extension("1");
    };
    path.with_file_name(format!("{name}.1"))
}

/// Cutlass lines keep info. Other crates keep a warning or an error.
fn keeps(level: log::Level, target: &str) -> bool {
    let ceiling = if target.starts_with("cutlass") {
        log::Level::Info
    } else {
        log::Level::Warn
    };
    level <= ceiling
}

fn level_word(level: log::Level) -> &'static str {
    match level {
        log::Level::Error => "error",
        log::Level::Warn => "warn",
        log::Level::Info => "info",
        log::Level::Debug => "debug",
        log::Level::Trace => "trace",
    }
}

fn format_line(stamp: &str, level: &str, project: &str, message: &str) -> String {
    let project = one_line(project);
    let project = if project.is_empty() {
        "(none)"
    } else {
        &project
    };
    format!("{stamp} {level} \"{project}\" {}\n", one_line(message))
}

fn one_line(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '\n' | '\r' => ' ',
            '"' => '\'',
            _ => ch,
        })
        .collect()
}

fn stamp(time: SystemTime) -> String {
    let Ok(since) = time.duration_since(UNIX_EPOCH) else {
        return "1970-01-01T00:00:00.000Z".to_owned();
    };
    let secs = since.as_secs();
    let millis = since.subsec_millis();
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let tod = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = tod / 3_600;
    let minute = (tod % 3_600) / 60;
    let second = tod % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

/// Days since 1970-01-01, as a civil date. Howard Hinnant's algorithm.
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = u64::try_from(z - era * 146_097).unwrap_or(0);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = i64::try_from(yoe).unwrap_or(0) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    if month <= 2 {
        year += 1;
    }
    (
        i32::try_from(year).unwrap_or(0),
        u32::try_from(month).unwrap_or(0),
        u32::try_from(day).unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::{
        State, civil_from_days, format_line, keeps, linux_dir, macos_dir, previous_path, stamp,
        windows_dir,
    };

    #[test]
    fn a_line_names_the_project_and_stays_on_one_line() {
        assert_eq!(
            format_line(
                "2026-10-22T20:09:01.123Z",
                "warn",
                "Birthday",
                "The device stopped."
            ),
            "2026-10-22T20:09:01.123Z warn \"Birthday\" The device stopped.\n"
        );
        assert_eq!(
            format_line("t", "info", "", "Started."),
            "t info \"(none)\" Started.\n"
        );
        assert_eq!(
            format_line("t", "warn", "A\nB \"C\"", "line\r\n"),
            "t warn \"A B 'C'\" line  \n"
        );
    }

    #[test]
    fn a_stamp_is_utc() {
        assert_eq!(stamp(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        let later = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        assert_eq!(stamp(later), "2023-11-14T22:13:20.000Z");
    }

    #[test]
    fn cutlass_info_is_kept_and_other_info_is_not() {
        assert!(keeps(log::Level::Info, "cutlass::hear"));
        assert!(keeps(log::Level::Warn, "wgpu_core"));
        assert!(!keeps(log::Level::Info, "wgpu"));
        assert!(!keeps(log::Level::Debug, "cutlass::frames"));
    }

    #[test]
    fn the_log_sits_in_the_user_logs_folder() {
        assert_eq!(
            macos_dir(std::path::Path::new("/Users/ada")),
            std::path::PathBuf::from("/Users/ada/Library/Logs/Cutlass")
        );
        assert_eq!(
            windows_dir(std::path::Path::new("C:/Users/ada/AppData/Local")),
            std::path::PathBuf::from("C:/Users/ada/AppData/Local/Cutlass/Logs")
        );
        assert_eq!(
            linux_dir(std::path::Path::new("/home/ada/.local/state")),
            std::path::PathBuf::from("/home/ada/.local/state/cutlass")
        );
        assert_eq!(
            previous_path(std::path::Path::new("/tmp/cutlass.log")),
            std::path::PathBuf::from("/tmp/cutlass.log.1")
        );
    }

    #[test]
    fn a_full_log_keeps_the_previous_file() {
        let dir = std::env::temp_dir().join(format!("cutlass-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("cutlass.log");
        let mut state = State::create(&path, 1).expect("log");
        state.project = "Cut".to_owned();
        state.write(log::Level::Warn, "first line that is long enough");
        state.write(log::Level::Info, "second");
        let kept = std::fs::read_to_string(previous_path(&path)).expect("rolled");
        let current = std::fs::read_to_string(&path).expect("new");
        assert!(kept.contains("first line that is long enough"));
        assert!(current.contains("second"));
        assert!(!current.contains("first line that is long enough"));
        drop(state);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
