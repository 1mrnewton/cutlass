//! What the command line asks for, read before anything starts.

use std::ffi::OsString;
use std::path::PathBuf;

/// One reading of the arguments after the program name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// The editor window, on the project folder or files named, if any.
    Window,
    /// `cutlass studio`: the window, open on the effect studio.
    Studio,
    /// `cutlass mcp [project]`: the editing tools on stdin and stdout.
    Mcp(Option<PathBuf>),
    /// `cutlass new <folder>`: an empty project under the folder.
    New(Option<PathBuf>),
    /// `cutlass sandbox`: one agent script, started by `sandbox.run`.
    Sandbox,
    /// `--help`, `-h` or `help`: say how to call it.
    Help,
    /// `--version` or `-V`.
    Version,
    /// An option this program does not have. A window is not opened on it.
    Unknown(String),
}

/// Read the arguments after the program name.
///
/// A word that starts with `-` is an option, and one that is not known is
/// refused, so `cutlass --help` answers on the terminal and no window opens
/// on a file called `--help`. A file whose name does start with `-` is
/// named after `--`.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Command {
    let args: Vec<OsString> = args.into_iter().collect();
    let Some(first) = args.first() else {
        return Command::Window;
    };
    let rest = args.get(1);
    let asks_help = |word: &OsString| word == "--help" || word == "-h";
    if asks_help(first) || first == "help" || rest.is_some_and(asks_help) {
        return Command::Help;
    }
    if first == "--version" || first == "-V" {
        return Command::Version;
    }
    if first == "sandbox" {
        return Command::Sandbox;
    }
    if first == "mcp" {
        return Command::Mcp(rest.map(PathBuf::from));
    }
    if first == "new" {
        return Command::New(rest.map(PathBuf::from));
    }
    if first == "studio" {
        return Command::Studio;
    }
    if first == "--" {
        return Command::Window;
    }
    let word = first.to_string_lossy();
    // The system once passed a process serial number to a launched app.
    if word.starts_with('-') && !word.starts_with("-psn_") {
        return Command::Unknown(word.into_owned());
    }
    Command::Window
}

/// The files and folders a window opens on: every argument that is not an
/// option, with `--` ending the options.
pub fn named_files(args: impl IntoIterator<Item = OsString>) -> Vec<PathBuf> {
    let mut literal = false;
    let mut named = Vec::new();
    for word in args {
        if !literal && word == "--" {
            literal = true;
            continue;
        }
        if !literal && word.to_string_lossy().starts_with('-') {
            continue;
        }
        named.push(PathBuf::from(word));
    }
    named
}

/// The version line `--version` prints.
pub fn version() -> String {
    format!("Cutlass {}\n", env!("CARGO_PKG_VERSION"))
}

/// The text `--help` prints.
pub fn usage() -> String {
    format!(
        "Cutlass {version}, a video editor.

Usage:
  cutlass [project folder or files]
      Open the editor, on that project or with those files added.
  cutlass new <folder>
      Make an empty project in the folder and print where it is.
  cutlass mcp [project folder]
      Serve the editing tools on stdin and stdout for an MCP client, with
      no window.
  cutlass studio [effect or transition id]
      Open the editor on the effect studio.
  cutlass --help
      Show this.
  cutlass --version
      Show the version.
",
        version = env!("CARGO_PKG_VERSION")
    )
}

#[cfg(test)]
mod tests {
    use super::{Command, named_files, parse, usage, version};
    use std::ffi::OsString;
    use std::path::PathBuf;

    fn words(line: &[&str]) -> Vec<OsString> {
        line.iter().map(OsString::from).collect()
    }

    #[test]
    fn no_arguments_open_the_window() {
        assert_eq!(parse(words(&[])), Command::Window);
    }

    #[test]
    fn help_is_answered_and_opens_no_window() {
        for line in [
            &["--help"][..],
            &["-h"],
            &["help"],
            &["mcp", "--help"],
            &["new", "-h"],
        ] {
            assert_eq!(parse(words(line)), Command::Help, "{line:?}");
        }
    }

    #[test]
    fn version_is_answered() {
        assert_eq!(parse(words(&["--version"])), Command::Version);
        assert_eq!(parse(words(&["-V"])), Command::Version);
        assert!(version().contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn an_unknown_option_is_refused_by_name() {
        assert_eq!(
            parse(words(&["--halp"])),
            Command::Unknown("--halp".to_owned())
        );
    }

    #[test]
    fn the_commands_keep_their_paths() {
        assert_eq!(
            parse(words(&["mcp", "/tmp/Project"])),
            Command::Mcp(Some(PathBuf::from("/tmp/Project")))
        );
        assert_eq!(parse(words(&["mcp"])), Command::Mcp(None));
        assert_eq!(
            parse(words(&["new", "/tmp/films"])),
            Command::New(Some(PathBuf::from("/tmp/films")))
        );
        assert_eq!(parse(words(&["new"])), Command::New(None));
        assert_eq!(parse(words(&["studio", "recipe/shake"])), Command::Studio);
        assert_eq!(parse(words(&["sandbox"])), Command::Sandbox);
    }

    #[test]
    fn a_file_or_folder_opens_the_window() {
        assert_eq!(parse(words(&["/tmp/clip.mp4"])), Command::Window);
        assert_eq!(parse(words(&["--", "-odd.mp4"])), Command::Window);
        assert_eq!(parse(words(&["-psn_0_12345"])), Command::Window);
    }

    #[test]
    fn only_files_are_handed_to_the_window() {
        assert_eq!(
            named_files(words(&["a.mp4", "-psn_0_1", "b.mov"])),
            [PathBuf::from("a.mp4"), PathBuf::from("b.mov")]
        );
        assert_eq!(
            named_files(words(&["--", "-odd.mp4"])),
            [PathBuf::from("-odd.mp4")]
        );
        assert!(named_files(words(&[])).is_empty());
    }

    #[test]
    fn the_usage_names_every_command_a_person_runs() {
        let text = usage();
        for word in [
            "cutlass new",
            "cutlass mcp",
            "cutlass studio",
            "--help",
            "--version",
        ] {
            assert!(text.contains(word), "{word}");
        }
        assert!(!text.contains("sandbox"));
    }
}
