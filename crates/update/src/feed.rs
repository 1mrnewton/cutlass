//! The release feed: GitHub's latest release, and the package in it for
//! this platform.

use serde::Deserialize;

use crate::{UpdateError, Version};

/// The latest release of Cutlass, as GitHub's API answers it.
pub const FEED_URL: &str = "https://api.github.com/repos/1mrnewton/cutlass/releases/latest";

/// The most of the feed read: GitHub's answer for one release is a few
/// kilobytes.
pub const FEED_LIMIT: usize = 1 << 20;

/// Which package a machine installs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    /// `Cutlass.dmg`.
    MacOs,
    /// `Cutlass-setup.exe`.
    Windows,
    /// `Cutlass-<arch>.AppImage`.
    Linux { arch: &'static str },
}

impl Platform {
    /// The platform this build runs on, when it ships a package.
    #[must_use]
    pub fn current() -> Option<Self> {
        if cfg!(target_os = "macos") {
            Some(Self::MacOs)
        } else if cfg!(target_os = "windows") {
            Some(Self::Windows)
        } else if cfg!(target_os = "linux") {
            Some(Self::Linux {
                arch: std::env::consts::ARCH,
            })
        } else {
            None
        }
    }

    /// The release asset this platform installs.
    #[must_use]
    pub fn asset_name(self) -> String {
        match self {
            Self::MacOs => "Cutlass.dmg".to_owned(),
            Self::Windows => "Cutlass-setup.exe".to_owned(),
            Self::Linux { arch } => format!("Cutlass-{arch}.AppImage"),
        }
    }
}

/// One file of a release.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Asset {
    pub name: String,
    #[serde(rename = "browser_download_url")]
    pub url: String,
    #[serde(default)]
    pub size: u64,
}

/// What the feed says about one release.
#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<Asset>,
}

/// A newer release this machine can install.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Offer {
    pub version: Version,
    /// The release notes, as written.
    pub notes: String,
    /// The package for this platform.
    pub package: Asset,
    /// `<package>.sha256`: the package's SHA-256 in hex, first.
    pub checksum: Asset,
}

impl Offer {
    /// The notes' first `count` lines that say something.
    #[must_use]
    pub fn notes_head(&self, count: usize) -> Vec<&str> {
        self.notes
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .take(count)
            .collect()
    }
}

/// The release in `json` when it is newer than `running` and has a package
/// and its checksum for `platform`. A draft, a pre-release, and a release no
/// newer than this one offer nothing.
pub fn offer(
    json: &str,
    running: &Version,
    platform: Platform,
) -> Result<Option<Offer>, UpdateError> {
    let release: Release = serde_json::from_str(json)
        .map_err(|error| UpdateError::Feed(format!("the release feed did not read: {error}")))?;
    let Some(version) = Version::parse(&release.tag_name) else {
        return Err(UpdateError::Feed(format!(
            "the release tag {:?} is not a version",
            release.tag_name
        )));
    };
    if release.draft || release.prerelease || version.is_pre_release() || version <= *running {
        return Ok(None);
    }
    let name = platform.asset_name();
    let sum_name = format!("{name}.sha256");
    let find = |wanted: &str| release.assets.iter().find(|asset| asset.name == wanted);
    let (Some(package), Some(checksum)) = (find(&name), find(&sum_name)) else {
        return Ok(None);
    };
    for asset in [package, checksum] {
        if !asset.url.starts_with("https://") && !asset.url.starts_with("http://") {
            return Err(UpdateError::Feed(format!(
                "{} has no download address",
                asset.name
            )));
        }
    }
    Ok(Some(Offer {
        version,
        notes: release.body.unwrap_or_default(),
        package: package.clone(),
        checksum: checksum.clone(),
    }))
}

/// The SHA-256 a `.sha256` asset names: its first token, in hex, as
/// `sha256sum` writes it.
pub fn parse_checksum(text: &str) -> Result<cutlass_download::Checksum, UpdateError> {
    let token = text.split_whitespace().next().unwrap_or_default();
    cutlass_download::Checksum::parse(token)
        .map_err(|error| UpdateError::Feed(format!("the package checksum did not read: {error}")))
}

/// Whether a day has passed since `last` (seconds since 1970), or the clock
/// went back past it. `0` is never.
#[must_use]
pub fn due(last: u64, now: u64) -> bool {
    const DAY: u64 = 24 * 60 * 60;
    last == 0 || now < last || now - last >= DAY
}

#[cfg(test)]
mod tests {
    use super::{Platform, due, offer, parse_checksum};
    use crate::{UpdateError, Version};

    const HEX: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

    /// A feed with every package and checksum, as the release workflow
    /// publishes them.
    fn feed(tag: &str) -> String {
        let assets: Vec<String> = [
            "Cutlass.dmg",
            "Cutlass.dmg.sha256",
            "Cutlass-setup.exe",
            "Cutlass-setup.exe.sha256",
            "Cutlass-x86_64.AppImage",
            "Cutlass-x86_64.AppImage.sha256",
        ]
        .iter()
        .enumerate()
        .map(|(index, name)| {
            format!(
                r#"{{"name":"{name}","size":{},"browser_download_url":"https://github.com/1mrnewton/cutlass/releases/download/{tag}/{name}","state":"uploaded"}}"#,
                1000 + index
            )
        })
        .collect();
        format!(
            r#"{{"url":"https://api.github.com/repos/1mrnewton/cutlass/releases/1","tag_name":"{tag}","name":"Cutlass {tag}","draft":false,"prerelease":false,"body":"Faster export.\r\n\r\n- Captions keep their style\n- Fixes","assets":[{}]}}"#,
            assets.join(",")
        )
    }

    fn running() -> Version {
        Version::parse("0.1.0").unwrap()
    }

    #[test]
    fn each_platform_gets_its_package_and_checksum() {
        let json = feed("v0.2.0");
        for (platform, name) in [
            (Platform::MacOs, "Cutlass.dmg"),
            (Platform::Windows, "Cutlass-setup.exe"),
            (
                Platform::Linux { arch: "x86_64" },
                "Cutlass-x86_64.AppImage",
            ),
        ] {
            let found = offer(&json, &running(), platform).unwrap().unwrap();
            assert_eq!(found.version.to_string(), "0.2.0");
            assert_eq!(found.package.name, name);
            assert_eq!(found.checksum.name, format!("{name}.sha256"));
            assert!(found.package.url.ends_with(&format!("/v0.2.0/{name}")));
            assert!(found.package.size >= 1000);
        }
        let found = offer(&json, &running(), Platform::MacOs).unwrap().unwrap();
        assert_eq!(
            found.notes_head(3),
            ["Faster export.", "- Captions keep their style", "- Fixes"]
        );
        assert_eq!(found.notes_head(1), ["Faster export."]);
    }

    #[test]
    fn the_same_or_an_older_release_offers_nothing() {
        for tag in ["v0.1.0", "0.0.9", "v0.1.0+rebuild"] {
            assert_eq!(
                offer(&feed(tag), &running(), Platform::MacOs).unwrap(),
                None,
                "{tag}"
            );
        }
        let newer = Version::parse("0.3.0").unwrap();
        assert_eq!(
            offer(&feed("v0.2.0"), &newer, Platform::Windows).unwrap(),
            None
        );
    }

    #[test]
    fn a_pre_release_or_a_draft_is_ignored() {
        assert_eq!(
            offer(&feed("v0.2.0-rc.1"), &running(), Platform::MacOs).unwrap(),
            None
        );
        let flagged = feed("v0.2.0").replace(r#""prerelease":false"#, r#""prerelease":true"#);
        assert_eq!(offer(&flagged, &running(), Platform::MacOs).unwrap(), None);
        let draft = feed("v0.2.0").replace(r#""draft":false"#, r#""draft":true"#);
        assert_eq!(offer(&draft, &running(), Platform::MacOs).unwrap(), None);
        // A pre-release running build is offered the release it leads to.
        let dev = Version::parse("0.2.0-dev").unwrap();
        assert!(
            offer(&feed("v0.2.0"), &dev, Platform::MacOs)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn a_release_without_this_platform_or_its_checksum_offers_nothing() {
        let json = feed("v0.2.0");
        assert_eq!(
            offer(&json, &running(), Platform::Linux { arch: "aarch64" }).unwrap(),
            None
        );
        let unsummed = json.replace("Cutlass.dmg.sha256", "Cutlass.dmg.md5");
        assert_eq!(offer(&unsummed, &running(), Platform::MacOs).unwrap(), None);
        assert!(
            offer(&unsummed, &running(), Platform::Windows)
                .unwrap()
                .is_some()
        );
        let empty = r#"{"tag_name":"v9.0.0","assets":[]}"#;
        assert_eq!(offer(empty, &running(), Platform::MacOs).unwrap(), None);
    }

    #[test]
    fn a_broken_feed_names_the_cause() {
        for json in [
            "",
            "<html>rate limited</html>",
            r#"{"message":"API rate limit exceeded"}"#,
            r#"{"tag_name":"nightly","assets":[]}"#,
            r#"{"tag_name":"v0.2.0","assets":"none"}"#,
        ] {
            assert!(
                matches!(
                    offer(json, &running(), Platform::MacOs),
                    Err(UpdateError::Feed(_))
                ),
                "{json}"
            );
        }
        let local = feed("v0.2.0").replace("https://github.com", "file:///etc");
        assert!(matches!(
            offer(&local, &running(), Platform::MacOs),
            Err(UpdateError::Feed(_))
        ));
    }

    #[test]
    fn a_checksum_file_gives_its_first_token() {
        let expected = cutlass_download::Checksum::parse(HEX).unwrap();
        for text in [
            HEX.to_owned(),
            format!("{HEX}  Cutlass.dmg\n"),
            format!("  {}\tCutlass.dmg", HEX.to_uppercase()),
        ] {
            assert_eq!(parse_checksum(&text).unwrap(), expected, "{text:?}");
        }
        for text in ["", "abc  Cutlass.dmg", "Cutlass.dmg  9f86"] {
            assert!(parse_checksum(text).is_err(), "{text:?}");
        }
    }

    #[test]
    fn a_check_is_due_once_a_day() {
        let day = 24 * 60 * 60;
        assert!(due(0, 1_000), "never checked");
        assert!(!due(10 * day, 10 * day + 60));
        assert!(!due(10 * day, 11 * day - 1));
        assert!(due(10 * day, 11 * day));
        assert!(due(10 * day, 9 * day), "the clock went back");
    }
}
