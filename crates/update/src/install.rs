//! Putting a checked package in place of the running app, and starting the
//! new one once this process has quit.
//!
//! macOS swaps the `.app` bundle from the DMG. Windows runs the installer
//! silently into the same folder after this process exits. Linux renames
//! the new AppImage over `$APPIMAGE`.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::fetch::io;
use crate::{Platform, UpdateError, Version};

/// Where the running app was installed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Installed {
    /// The `.app` bundle on macOS.
    Bundle(PathBuf),
    /// The folder the Windows installer wrote, with `uninstall.exe`.
    Folder(PathBuf),
    /// The AppImage file on Linux.
    AppImage(PathBuf),
}

impl Installed {
    /// The running app's install, from its executable and `$APPIMAGE`.
    pub fn current(platform: Platform) -> Result<Self, UpdateError> {
        let exe = std::env::current_exe()
            .map_err(|error| UpdateError::NotInstalled(format!("this app's path: {error}")))?;
        let exe = exe.canonicalize().unwrap_or(exe);
        let appimage = std::env::var_os("APPIMAGE");
        Self::detect(platform, &exe, appimage.as_deref())
    }

    /// The install `exe` runs from. A binary outside a bundle, an installer
    /// folder or an AppImage is refused: only the packaged app updates.
    pub fn detect(
        platform: Platform,
        exe: &Path,
        appimage: Option<&OsStr>,
    ) -> Result<Self, UpdateError> {
        match platform {
            Platform::MacOs => {
                let bundle = exe
                    .parent()
                    .filter(|dir| dir.file_name() == Some(OsStr::new("MacOS")))
                    .and_then(Path::parent)
                    .filter(|dir| dir.file_name() == Some(OsStr::new("Contents")))
                    .and_then(Path::parent)
                    .filter(|dir| dir.extension() == Some(OsStr::new("app")))
                    .filter(|dir| dir.join("Contents").join("Info.plist").is_file());
                bundle
                    .map(|dir| Self::Bundle(dir.to_path_buf()))
                    .ok_or_else(|| {
                        UpdateError::NotInstalled(
                            "updates need Cutlass.app; this copy runs outside an app bundle"
                                .to_owned(),
                        )
                    })
            }
            Platform::Windows => exe
                .parent()
                .filter(|dir| dir.join("uninstall.exe").is_file())
                .map(|dir| Self::Folder(dir.to_path_buf()))
                .ok_or_else(|| {
                    UpdateError::NotInstalled(
                        "updates need Cutlass installed by its installer".to_owned(),
                    )
                }),
            Platform::Linux { .. } => appimage
                .map(PathBuf::from)
                .filter(|path| path.is_absolute() && path.is_file())
                .map(Self::AppImage)
                .ok_or_else(|| {
                    UpdateError::NotInstalled("updates need the Cutlass AppImage".to_owned())
                }),
        }
    }
}

/// A command run after this process exits: it waits for `pid`, then starts
/// the app (on Windows, installs first). [`Relaunch::spawn`] starts it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relaunch {
    pub program: OsString,
    pub args: Vec<OsString>,
}

/// The POSIX wait: `$1` is the pid, the rest the command. It gives up
/// waiting after two minutes.
const WAIT_THEN_RUN: &str = "pid=$1; shift; i=0; \
    while kill -0 \"$pid\" 2>/dev/null && [ $i -lt 600 ]; do sleep 0.2; i=$((i+1)); done; \
    exec \"$@\"";

impl Relaunch {
    /// `command` once process `pid` is gone, through `/bin/sh`.
    #[must_use]
    pub fn after(pid: u32, command: &[&OsStr]) -> Self {
        let mut args: Vec<OsString> = vec![
            "-c".into(),
            WAIT_THEN_RUN.into(),
            "cutlass-relaunch".into(),
            pid.to_string().into(),
        ];
        args.extend(command.iter().map(|part| part.to_os_string()));
        Self {
            program: "/bin/sh".into(),
            args,
        }
    }

    /// Windows: wait for `pid`, run `installer` silently into `folder`, then
    /// start `folder\cutlass.exe`, through PowerShell.
    #[must_use]
    pub fn windows_installer(pid: u32, installer: &Path, folder: &Path) -> Self {
        let quote = |path: &Path| path.display().to_string().replace('\'', "''");
        let folder_text = quote(folder);
        let script = format!(
            "Wait-Process -Id {pid} -Timeout 120 -ErrorAction SilentlyContinue; \
             Start-Process -Wait -FilePath '{}' -ArgumentList '/S','/D={folder_text}'; \
             Start-Process -FilePath '{folder_text}\\cutlass.exe'",
            quote(installer),
        );
        Self {
            program: "powershell.exe".into(),
            args: [
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-Command",
                &script,
            ]
            .iter()
            .map(OsString::from)
            .collect(),
        }
    }

    /// Start the command on its own, detached from this process, so it
    /// outlives it.
    pub fn spawn(&self) -> std::io::Result<()> {
        let mut command = Command::new(&self.program);
        command
            .args(&self.args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const DETACHED_PROCESS: u32 = 0x0000_0008;
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
        }
        command.spawn().map(drop)
    }
}

/// Install `package` (checked already) over `installed`, as `version`, and
/// say what to run once process `pid` has quit.
pub fn install(
    installed: &Installed,
    package: &Path,
    version: &Version,
    pid: u32,
) -> Result<Relaunch, UpdateError> {
    if !package.is_file() {
        return Err(UpdateError::Install(format!(
            "{} is not there",
            package.display()
        )));
    }
    match installed {
        Installed::Bundle(bundle) => {
            swap_bundle(package, bundle, version)?;
            Ok(Relaunch::after(
                pid,
                &[
                    OsStr::new("/usr/bin/open"),
                    OsStr::new("-n"),
                    bundle.as_os_str(),
                ],
            ))
        }
        Installed::Folder(folder) => Ok(Relaunch::windows_installer(pid, package, folder)),
        Installed::AppImage(appimage) => {
            replace_appimage(package, appimage)?;
            Ok(Relaunch::after(pid, &[appimage.as_os_str()]))
        }
    }
}

/// `<parent>/.<name>.<suffix>` beside `path`.
fn beside(path: &Path, suffix: &str) -> Result<PathBuf, UpdateError> {
    let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else {
        return Err(UpdateError::Install(format!(
            "{} has no folder",
            path.display()
        )));
    };
    let mut hidden = OsString::from(".");
    hidden.push(name);
    hidden.push(".");
    hidden.push(suffix);
    Ok(parent.join(hidden))
}

/// Run a tool and fail with what it said.
fn run(program: &str, args: &[&OsStr]) -> Result<String, UpdateError> {
    let output = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| UpdateError::Install(format!("{program}: {error}")))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    } else {
        Err(UpdateError::Install(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

/// One key of a bundle's Info.plist.
fn plist_value(bundle: &Path, key: &str) -> Result<String, UpdateError> {
    let plist = bundle.join("Contents").join("Info.plist");
    run(
        "/usr/bin/plutil",
        &[
            OsStr::new("-extract"),
            OsStr::new(key),
            OsStr::new("raw"),
            OsStr::new("-o"),
            OsStr::new("-"),
            plist.as_os_str(),
        ],
    )
}

/// Detaches the DMG when dropped, whatever happened in between.
struct Mounted(PathBuf);

impl Drop for Mounted {
    fn drop(&mut self) {
        let point = self.0.as_os_str();
        if run(
            "/usr/bin/hdiutil",
            &[OsStr::new("detach"), OsStr::new("-quiet"), point],
        )
        .is_err()
        {
            let _ = run(
                "/usr/bin/hdiutil",
                &[
                    OsStr::new("detach"),
                    OsStr::new("-force"),
                    OsStr::new("-quiet"),
                    point,
                ],
            );
        }
        let _ = fs::remove_dir(&self.0);
    }
}

/// Mount `dmg`, copy its `Cutlass.app` beside `bundle`, check its signature,
/// version and identifier, and swap it in: the old bundle is renamed aside,
/// the new one renamed in, the old one removed.
fn swap_bundle(dmg: &Path, bundle: &Path, version: &Version) -> Result<(), UpdateError> {
    let point = std::env::temp_dir().join(format!(
        "cutlass-update-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos())
    ));
    fs::create_dir_all(&point).map_err(|error| io(&point, &error))?;
    run(
        "/usr/bin/hdiutil",
        &[
            OsStr::new("attach"),
            OsStr::new("-nobrowse"),
            OsStr::new("-readonly"),
            OsStr::new("-noautoopen"),
            OsStr::new("-quiet"),
            OsStr::new("-mountpoint"),
            point.as_os_str(),
            dmg.as_os_str(),
        ],
    )
    .inspect_err(|_| {
        let _ = fs::remove_dir(&point);
    })?;
    let mounted = Mounted(point);
    let source = mounted.0.join("Cutlass.app");
    if !source.join("Contents").join("Info.plist").is_file() {
        return Err(UpdateError::Install(
            "the DMG holds no Cutlass.app".to_owned(),
        ));
    }
    let staged = beside(bundle, "update")?;
    let old = beside(bundle, "old")?;
    for leftover in [&staged, &old] {
        if leftover.exists() {
            fs::remove_dir_all(leftover).map_err(|error| io(leftover, &error))?;
        }
    }
    let checked = (|| {
        run("/usr/bin/ditto", &[source.as_os_str(), staged.as_os_str()])?;
        run(
            "/usr/bin/codesign",
            &[
                OsStr::new("--verify"),
                OsStr::new("--strict"),
                staged.as_os_str(),
            ],
        )?;
        let found = plist_value(&staged, "CFBundleShortVersionString")?;
        if Version::parse(&found).as_ref() != Some(version) {
            return Err(UpdateError::Install(format!(
                "the package holds version {found}, not {version}"
            )));
        }
        let ours = plist_value(bundle, "CFBundleIdentifier")?;
        let theirs = plist_value(&staged, "CFBundleIdentifier")?;
        if ours != theirs {
            return Err(UpdateError::Install(format!(
                "the package is {theirs}, not {ours}"
            )));
        }
        Ok(())
    })();
    drop(mounted);
    if let Err(error) = checked {
        let _ = fs::remove_dir_all(&staged);
        return Err(error);
    }
    fs::rename(bundle, &old).map_err(|error| {
        let _ = fs::remove_dir_all(&staged);
        io(bundle, &error)
    })?;
    if let Err(error) = fs::rename(&staged, bundle) {
        let _ = fs::rename(&old, bundle);
        let _ = fs::remove_dir_all(&staged);
        return Err(io(bundle, &error));
    }
    let _ = fs::remove_dir_all(&old);
    Ok(())
}

/// Copy `package` beside `appimage`, make it executable, and rename it over
/// `appimage`.
fn replace_appimage(package: &Path, appimage: &Path) -> Result<(), UpdateError> {
    let staged = beside(appimage, "update")?;
    fs::copy(package, &staged).map_err(|error| io(&staged, &error))?;
    let finish = || -> Result<(), UpdateError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&staged, fs::Permissions::from_mode(0o755))
                .map_err(|error| io(&staged, &error))?;
        }
        fs::rename(&staged, appimage).map_err(|error| io(appimage, &error))
    };
    finish().inspect_err(|_| {
        let _ = fs::remove_file(&staged);
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::{Installed, Relaunch};
    use crate::{Platform, UpdateError};

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cutlass-update-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(path: &Path) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"x").unwrap();
    }

    #[test]
    fn only_a_bundle_counts_on_macos() {
        let dir = scratch("bundle");
        let app = dir.join("Cutlass.app");
        let exe = app.join("Contents/MacOS/cutlass");
        touch(&exe);
        assert!(
            matches!(
                Installed::detect(Platform::MacOs, &exe, None),
                Err(UpdateError::NotInstalled(_))
            ),
            "no Info.plist yet"
        );
        touch(&app.join("Contents/Info.plist"));
        assert_eq!(
            Installed::detect(Platform::MacOs, &exe, None).unwrap(),
            Installed::Bundle(app.clone())
        );
        let bare = dir.join("target/debug/cutlass");
        touch(&bare);
        let refused = Installed::detect(Platform::MacOs, &bare, None).unwrap_err();
        assert!(refused.to_string().contains("Cutlass.app"), "{refused}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_the_installer_folder_counts_on_windows() {
        let dir = scratch("folder");
        let exe = dir.join("Cutlass/cutlass.exe");
        touch(&exe);
        assert!(Installed::detect(Platform::Windows, &exe, None).is_err());
        touch(&dir.join("Cutlass/uninstall.exe"));
        assert_eq!(
            Installed::detect(Platform::Windows, &exe, None).unwrap(),
            Installed::Folder(dir.join("Cutlass"))
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_an_appimage_counts_on_linux() {
        let dir = scratch("appimage");
        let image = dir.join("Cutlass-x86_64.AppImage");
        let linux = Platform::Linux { arch: "x86_64" };
        let exe = Path::new("/tmp/.mount_x/usr/bin/cutlass");
        assert!(Installed::detect(linux, exe, None).is_err());
        assert!(
            Installed::detect(linux, exe, Some(image.as_os_str())).is_err(),
            "missing"
        );
        touch(&image);
        assert!(Installed::detect(linux, exe, Some(OsStr::new("relative.AppImage"))).is_err());
        assert_eq!(
            Installed::detect(linux, exe, Some(image.as_os_str())).unwrap(),
            Installed::AppImage(image.clone())
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn an_appimage_is_replaced_whole_and_made_executable() {
        use std::os::unix::fs::PermissionsExt;

        use super::{beside, install};
        use crate::Version;

        let dir = scratch("replace");
        let image = dir.join("Cutlass.AppImage");
        fs::write(&image, b"old").unwrap();
        let package = dir.join("download/Cutlass-x86_64.AppImage");
        touch(&package);
        fs::write(&package, b"new image").unwrap();
        let version = Version::parse("0.2.0").unwrap();
        let relaunch =
            install(&Installed::AppImage(image.clone()), &package, &version, 42).unwrap();
        assert_eq!(fs::read(&image).unwrap(), b"new image");
        assert_eq!(
            fs::metadata(&image).unwrap().permissions().mode() & 0o777,
            0o755
        );
        assert!(!beside(&image, "update").unwrap().exists());
        assert_eq!(relaunch.program, "/bin/sh");
        assert_eq!(relaunch.args[3], "42");
        assert_eq!(relaunch.args[4], image.as_os_str());
        let missing = install(
            &Installed::AppImage(image.clone()),
            &dir.join("nothing"),
            &version,
            42,
        );
        assert!(matches!(missing, Err(UpdateError::Install(_))));
        assert_eq!(fs::read(&image).unwrap(), b"new image");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn windows_installs_silently_into_its_folder_after_the_app_quits() {
        let folder = Path::new(r"C:\Users\O'Neil Ama\AppData\Local\Programs\Cutlass");
        let installer = Path::new(
            r"C:\Users\O'Neil Ama\AppData\Roaming\Cutlass\Cache\Updates\0.2.0\Cutlass-setup.exe",
        );
        let relaunch = Relaunch::windows_installer(4321, installer, folder);
        assert_eq!(relaunch.program, "powershell.exe");
        let script = relaunch.args.last().unwrap().to_str().unwrap();
        assert!(script.starts_with("Wait-Process -Id 4321 "), "{script}");
        assert!(
            script.contains(r"-FilePath 'C:\Users\O''Neil Ama\AppData\Roaming"),
            "{script}"
        );
        assert!(
            script.contains(
                r"-ArgumentList '/S','/D=C:\Users\O''Neil Ama\AppData\Local\Programs\Cutlass'"
            ),
            "{script}"
        );
        // The install waits; the relaunch comes after it.
        let waits = script.find("Start-Process -Wait").unwrap();
        let starts = script.rfind("Start-Process -FilePath").unwrap();
        assert!(waits < starts);
        assert!(
            script[starts..].contains(r"Cutlass\cutlass.exe'"),
            "{script}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_relaunch_waits_for_the_app_to_quit() {
        use std::time::{Duration, Instant};

        let dir = scratch("relaunch");
        let mark = dir.join("started");
        let mut app = std::process::Command::new("/bin/sleep")
            .arg("1")
            .spawn()
            .unwrap();
        let began = Instant::now();
        Relaunch::after(app.id(), &[OsStr::new("/usr/bin/touch"), mark.as_os_str()])
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_millis(400));
        assert!(!mark.exists(), "it ran while the app was up");
        app.wait().unwrap();
        while !mark.exists() && began.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(mark.exists(), "it never ran");
        assert!(began.elapsed() >= Duration::from_millis(900));
        fs::remove_dir_all(&dir).unwrap();
    }
}
