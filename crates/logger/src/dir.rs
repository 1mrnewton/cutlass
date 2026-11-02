//! Where a shell keeps `cutlass.log` when it does not pass a directory.

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::InitError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Host {
    Mac,
    Windows,
    Other,
}

pub(crate) const fn current_host() -> Host {
    if cfg!(target_os = "macos") {
        Host::Mac
    } else if cfg!(target_os = "windows") {
        Host::Windows
    } else {
        Host::Other
    }
}

pub fn default_directory() -> Result<PathBuf, InitError> {
    let home = argument(env::var_os("HOME"));
    let local = argument(env::var_os("LOCALAPPDATA"));
    let state = argument(env::var_os("XDG_STATE_HOME"));
    directory_from(
        current_host(),
        home.as_deref(),
        local.as_deref(),
        state.as_deref(),
    )
}

pub(crate) fn directory_from(
    host: Host,
    home: Option<&Path>,
    local_app_data: Option<&Path>,
    xdg_state: Option<&Path>,
) -> Result<PathBuf, InitError> {
    match host {
        Host::Mac => {
            let home = home.ok_or(InitError::NoDirectory)?;
            Ok(home.join("Library").join("Logs").join("Cutlass"))
        }
        Host::Windows => {
            if let Some(local) = local_app_data {
                return Ok(local.join("Cutlass").join("logs"));
            }
            let home = home.ok_or(InitError::NoDirectory)?;
            Ok(home
                .join("AppData")
                .join("Local")
                .join("Cutlass")
                .join("logs"))
        }
        Host::Other => {
            if let Some(state) = xdg_state {
                return Ok(state.join("cutlass").join("logs"));
            }
            let home = home.ok_or(InitError::NoDirectory)?;
            Ok(home
                .join(".local")
                .join("state")
                .join("cutlass")
                .join("logs"))
        }
    }
}

fn argument(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    use super::{Host, argument, current_host, default_directory, directory_from};
    use crate::InitError;

    #[test]
    fn an_empty_env_value_is_missing() {
        assert!(argument(None).is_none());
        assert!(argument(Some(OsString::new())).is_none());
        assert_eq!(
            argument(Some(OsString::from("/home/ada"))).as_deref(),
            Some(Path::new("/home/ada"))
        );
    }

    #[test]
    fn each_os_has_its_log_directory() {
        let home = Path::new("/home/ada");
        let local = Path::new("/home/ada/AppData/Local");
        let state = Path::new("/state");

        assert_eq!(
            directory_from(Host::Mac, Some(home), None, None).unwrap(),
            PathBuf::from("/home/ada/Library/Logs/Cutlass")
        );
        assert!(matches!(
            directory_from(Host::Mac, None, Some(local), Some(state)),
            Err(InitError::NoDirectory)
        ));

        assert_eq!(
            directory_from(Host::Windows, None, Some(local), None).unwrap(),
            local.join("Cutlass").join("logs")
        );
        assert_eq!(
            directory_from(Host::Windows, Some(home), None, None).unwrap(),
            home.join("AppData")
                .join("Local")
                .join("Cutlass")
                .join("logs")
        );
        assert!(matches!(
            directory_from(Host::Windows, None, None, Some(state)),
            Err(InitError::NoDirectory)
        ));

        assert_eq!(
            directory_from(Host::Other, Some(home), Some(local), Some(state)).unwrap(),
            state.join("cutlass").join("logs")
        );
        assert_eq!(
            directory_from(Host::Other, Some(home), None, None).unwrap(),
            home.join(".local")
                .join("state")
                .join("cutlass")
                .join("logs")
        );
        assert!(matches!(
            directory_from(Host::Other, None, Some(local), None),
            Err(InitError::NoDirectory)
        ));
    }

    #[test]
    fn this_process_uses_its_own_os_directory() {
        let path = default_directory().unwrap();
        assert!(path.is_absolute());
        match current_host() {
            Host::Mac => assert!(path.ends_with("Library/Logs/Cutlass")),
            Host::Windows => assert!(path.ends_with("Cutlass/logs")),
            Host::Other => assert!(path.ends_with("cutlass/logs")),
        }
    }
}
