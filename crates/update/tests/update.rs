//! The whole update over a loopback server: the feed, the checksum, the
//! package, and on macOS the bundle swap from a real DMG.

#![cfg(test)]

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cutlass_download::{Checksum, Http};
use cutlass_update::{Platform, UpdateError, Version, check, download};

/// Serves `files` by path, with ranges, and counts the requests.
struct Server {
    base: String,
    requests: Arc<AtomicUsize>,
}

fn serve(files: HashMap<String, Vec<u8>>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let files = Arc::new(files);
    let requests = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&requests);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let files = Arc::clone(&files);
            let counted = Arc::clone(&counted);
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut stream = stream;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    counted.fetch_add(1, Ordering::SeqCst);
                    let path = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
                    let mut range = None;
                    loop {
                        let mut header = String::new();
                        if reader.read_line(&mut header).unwrap_or(0) == 0 {
                            return;
                        }
                        let header = header.trim_end().to_ascii_lowercase();
                        if header.is_empty() {
                            break;
                        }
                        if let Some(value) = header.strip_prefix("range: bytes=") {
                            let (start, end) = value.split_once('-').unwrap();
                            range = Some((
                                start.parse::<usize>().unwrap(),
                                end.parse::<usize>().unwrap() + 1,
                            ));
                        }
                    }
                    let Some(body) = files.get(&path) else {
                        let _ = stream
                            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                        continue;
                    };
                    let written = match range {
                        Some((start, end)) => {
                            let end = end.min(body.len());
                            let head = format!(
                                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                                 Content-Range: bytes {start}-{}/{}\r\n\r\n",
                                end - start,
                                end - 1,
                                body.len()
                            );
                            stream
                                .write_all(head.as_bytes())
                                .and_then(|()| stream.write_all(&body[start..end]))
                        }
                        None => {
                            let head = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                                body.len()
                            );
                            stream
                                .write_all(head.as_bytes())
                                .and_then(|()| stream.write_all(body))
                        }
                    };
                    if written.is_err() {
                        return;
                    }
                }
            });
        }
    });
    Server { base, requests }
}

/// A feed for `tag` whose package for `platform` is served at `/pkg/<name>`.
fn feed(base: &str, tag: &str, platform: Platform, size: usize) -> String {
    let name = platform.asset_name();
    format!(
        r#"{{"tag_name":"{tag}","draft":false,"prerelease":false,"body":"New in {tag}\n\n- Faster","assets":[
            {{"name":"{name}","size":{size},"browser_download_url":"{base}/pkg/{name}"}},
            {{"name":"{name}.sha256","size":80,"browser_download_url":"{base}/pkg/{name}.sha256"}}]}}"#
    )
}

fn files_for(platform: Platform, package: &[u8]) -> HashMap<String, Vec<u8>> {
    let name = platform.asset_name();
    let mut files = HashMap::new();
    files.insert(
        format!("/pkg/{name}.sha256"),
        format!("{}  {name}\n", Checksum::of(package).to_hex()).into_bytes(),
    );
    files.insert(format!("/pkg/{name}"), package.to_vec());
    files
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-update-it-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Serve `package` for `platform` as release `tag`; the feed lives at
/// `/feed`. The feed names the server's own address, so it is served from a
/// second server that knows it.
fn release(platform: Platform, package: &[u8], tag: &str) -> (Server, Server) {
    let files = serve(files_for(platform, package));
    let mut feed_files = HashMap::new();
    feed_files.insert(
        "/feed".to_owned(),
        feed(&files.base, tag, platform, package.len()).into_bytes(),
    );
    (serve(feed_files), files)
}

#[test]
fn a_newer_release_downloads_checked_and_an_old_one_offers_nothing() {
    let platform = Platform::Linux { arch: "x86_64" };
    let package: Vec<u8> = (0..2_500_000u32).map(|index| (index % 251) as u8).collect();
    let (feed_server, files) = release(platform, &package, "v0.2.0");
    let feed_url = format!("{}/feed", feed_server.base);
    let running = Version::parse("0.1.0").unwrap();
    let offer = check(&Http::new(), &feed_url, &running, platform)
        .unwrap()
        .unwrap();
    assert_eq!(offer.version.to_string(), "0.2.0");
    assert_eq!(offer.notes_head(5), ["New in v0.2.0", "- Faster"]);
    let current = Version::parse("0.2.0").unwrap();
    assert_eq!(
        check(&Http::new(), &feed_url, &current, platform).unwrap(),
        None
    );

    let folder = scratch("download");
    fs::create_dir_all(folder.join("0.1.5")).unwrap();
    let mut shares = Vec::new();
    let path = download(
        Http::new(),
        &offer,
        &folder,
        &mut |share| shares.push(share),
        &|| false,
    )
    .unwrap();
    assert_eq!(path, folder.join("0.2.0").join("Cutlass-x86_64.AppImage"));
    assert_eq!(fs::read(&path).unwrap(), package);
    assert_eq!(shares.last().copied(), Some(1.0));
    assert!(
        !folder.join("0.1.5").exists(),
        "an older download is cleared"
    );
    assert!(
        !folder
            .join("0.2.0/Cutlass-x86_64.AppImage.partial")
            .exists()
    );

    // A second try finds the checked package and fetches only the checksum.
    let before = files.requests.load(Ordering::SeqCst);
    let again = download(Http::new(), &offer, &folder, &mut |_| {}, &|| false).unwrap();
    assert_eq!(again, path);
    assert_eq!(files.requests.load(Ordering::SeqCst), before + 1);

    // Stopped at once: nothing lands.
    fs::remove_file(&path).unwrap();
    let stopped = download(Http::new(), &offer, &folder, &mut |_| {}, &|| true);
    assert_eq!(stopped, Err(UpdateError::Stopped));
    assert!(!path.exists());
    fs::remove_dir_all(&folder).unwrap();
}

#[test]
fn a_package_that_does_not_match_its_checksum_is_refused() {
    let platform = Platform::Windows;
    let package = vec![7u8; 300_000];
    let mut files = files_for(platform, &package);
    files.insert("/pkg/Cutlass-setup.exe".to_owned(), vec![8u8; 300_000]);
    let files = serve(files);
    let mut feed_files = HashMap::new();
    feed_files.insert(
        "/feed".to_owned(),
        feed(&files.base, "v0.2.0", platform, package.len()).into_bytes(),
    );
    let feed_server = serve(feed_files);
    let running = Version::parse("0.1.0").unwrap();
    let offer = check(
        &Http::new(),
        &format!("{}/feed", feed_server.base),
        &running,
        platform,
    )
    .unwrap()
    .unwrap();
    let folder = scratch("mismatch");
    let error = download(Http::new(), &offer, &folder, &mut |_| {}, &|| false).unwrap_err();
    assert!(
        matches!(&error, UpdateError::Network(message) if message.contains("checksum")),
        "{error}"
    );
    assert!(!folder.join("0.2.0/Cutlass-setup.exe").exists());
    fs::remove_dir_all(&folder).unwrap();
}

#[test]
fn an_unreachable_or_missing_feed_is_a_network_error() {
    let running = Version::parse("0.1.0").unwrap();
    let empty = serve(HashMap::new());
    let missing = check(
        &Http::new(),
        &format!("{}/feed", empty.base),
        &running,
        Platform::MacOs,
    );
    assert!(matches!(missing, Err(UpdateError::Network(message)) if message.contains("404")));
    let closed = check(
        &Http::new(),
        "http://127.0.0.1:9/feed",
        &running,
        Platform::MacOs,
    );
    assert!(matches!(closed, Err(UpdateError::Network(_))));
}

/// The Linux update as it runs: the feed offers a newer AppImage, it is
/// fetched and checked, put over the running one, and started once the old
/// process has quit. The AppImages are shell scripts that say they ran.
#[cfg(unix)]
#[test]
fn an_appimage_is_replaced_and_the_new_one_starts_after_the_old_quits() {
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};

    use cutlass_update::{Installed, install};

    let platform = Platform::Linux { arch: "x86_64" };
    let dir = scratch("appimage");
    let ran = dir.join("ran");
    let appimage = dir.join("Cutlass.AppImage");
    fs::write(&appimage, "#!/bin/sh\necho old > \"$0.old\"\n").unwrap();
    fs::set_permissions(&appimage, fs::Permissions::from_mode(0o755)).unwrap();
    let package = format!("#!/bin/sh\necho new > '{}'\n", ran.display()).into_bytes();
    let (feed_server, _files) = release(platform, &package, "v0.2.0");
    let running = Version::parse("0.1.0").unwrap();
    let offer = check(
        &Http::new(),
        &format!("{}/feed", feed_server.base),
        &running,
        platform,
    )
    .unwrap()
    .unwrap();
    let downloaded = download(
        Http::new(),
        &offer,
        &dir.join("Updates"),
        &mut |_| {},
        &|| false,
    )
    .unwrap();

    let installed = Installed::detect(
        platform,
        &dir.join("squashfs-root/AppRun"),
        Some(appimage.as_os_str()),
    )
    .unwrap();
    assert_eq!(installed, Installed::AppImage(appimage.clone()));
    // The "running app": a process the relaunch waits for.
    let mut app = std::process::Command::new("sleep")
        .arg("1")
        .spawn()
        .unwrap();
    let relaunch = install(&installed, &downloaded, &offer.version, app.id()).unwrap();
    assert_eq!(fs::read(&appimage).unwrap(), package);
    assert_eq!(
        fs::metadata(&appimage).unwrap().permissions().mode() & 0o777,
        0o755
    );
    assert!(!dir.join(".Cutlass.AppImage.update").exists());
    relaunch.spawn().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    assert!(!ran.exists(), "the new app waits for the old one to quit");
    app.wait().unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    while !ran.exists() {
        assert!(Instant::now() < until, "the new AppImage never started");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(fs::read_to_string(&ran).unwrap(), "new\n");
    assert!(
        !dir.join("Cutlass.AppImage.old").exists(),
        "the old one never ran"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(target_os = "macos")]
mod macos {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::{env, fs};

    use cutlass_download::Http;
    use cutlass_update::{Installed, Platform, UpdateError, Version, check, download, install};

    use super::{release, scratch};

    fn run(program: &str, args: &[&str]) {
        let status = Command::new(program).args(args).status().unwrap();
        assert!(status.success(), "{program} {args:?}");
    }

    fn plist(app: &Path, key: &str) -> String {
        let out = Command::new("/usr/bin/plutil")
            .args(["-extract", key, "raw", "-o", "-"])
            .arg(app.join("Contents/Info.plist"))
            .output()
            .unwrap();
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    /// A tiny signed Cutlass.app at `version`, as the bundle script lays it out.
    fn fake_app(dir: &Path, version: &str, id: &str) -> PathBuf {
        let app = dir.join("Cutlass.app");
        let macos = app.join("Contents/MacOS");
        fs::create_dir_all(&macos).unwrap();
        fs::write(
            macos.join("cutlass"),
            format!("#!/bin/sh\necho {version}\n"),
        )
        .unwrap();
        run(
            "/bin/chmod",
            &["755", macos.join("cutlass").to_str().unwrap()],
        );
        let info = app.join("Contents/Info.plist");
        fs::write(
            &info,
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>cutlass</string>
<key>CFBundleIdentifier</key><string>{id}</string>
<key>CFBundleName</key><string>Cutlass</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>{version}</string>
<key>CFBundleVersion</key><string>{version}</string>
</dict></plist>"#
            ),
        )
        .unwrap();
        run(
            "/usr/bin/codesign",
            &["--force", "--sign", "-", app.to_str().unwrap()],
        );
        app
    }

    fn dmg_of(app: &Path, out: &Path) {
        let stage = out.with_extension("stage");
        fs::create_dir_all(&stage).unwrap();
        run(
            "/usr/bin/ditto",
            &[
                app.to_str().unwrap(),
                stage.join("Cutlass.app").to_str().unwrap(),
            ],
        );
        run(
            "/usr/bin/hdiutil",
            &[
                "create",
                "-quiet",
                "-volname",
                "Cutlass",
                "-srcfolder",
                stage.to_str().unwrap(),
                "-fs",
                "HFS+",
                "-format",
                "UDZO",
                "-ov",
                out.to_str().unwrap(),
            ],
        );
        fs::remove_dir_all(&stage).unwrap();
    }

    /// Feed, download and swap for `old_app` and the DMG holding `version`.
    fn update_bundle(
        work: &Path,
        old_app: &Path,
        dmg: &Path,
        version: &str,
    ) -> Result<PathBuf, UpdateError> {
        let installed_dir = work.join("Applications");
        fs::create_dir_all(&installed_dir).unwrap();
        let bundle = installed_dir.join("Cutlass.app");
        run(
            "/usr/bin/ditto",
            &[old_app.to_str().unwrap(), bundle.to_str().unwrap()],
        );
        let running = Version::parse(&plist(&bundle, "CFBundleShortVersionString")).unwrap();
        let package = fs::read(dmg).unwrap();
        let (feed_server, _files) = release(Platform::MacOs, &package, &format!("v{version}"));
        let offer = check(
            &Http::new(),
            &format!("{}/feed", feed_server.base),
            &running,
            Platform::MacOs,
        )?
        .ok_or_else(|| UpdateError::Feed("nothing offered".to_owned()))?;
        let downloaded = download(
            Http::new(),
            &offer,
            &work.join("Updates"),
            &mut |_| {},
            &|| false,
        )?;
        let exe = bundle.join("Contents/MacOS/cutlass");
        let installed = Installed::detect(Platform::MacOs, &exe, None)?;
        assert_eq!(installed, Installed::Bundle(bundle.clone()));
        let relaunch = install(&installed, &downloaded, &offer.version, 99_999)?;
        assert_eq!(relaunch.args.last().unwrap(), bundle.as_os_str());
        assert!(relaunch.args.iter().any(|arg| arg == "/usr/bin/open"));
        Ok(bundle)
    }

    fn verified(app: &Path) -> bool {
        Command::new("/usr/bin/codesign")
            .args(["--verify", "--strict"])
            .arg(app)
            .status()
            .unwrap()
            .success()
    }

    fn leftovers(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "Cutlass.app")
            .collect()
    }

    #[test]
    fn a_dmg_swaps_in_for_the_bundle_and_verifies() {
        let work = scratch("macos");
        let old = fake_app(&work.join("old"), "0.1.0", "app.cutlass.desktop");
        let new = fake_app(&work.join("new"), "0.2.0", "app.cutlass.desktop");
        let dmg = work.join("Cutlass.dmg");
        dmg_of(&new, &dmg);
        let bundle = update_bundle(&work, &old, &dmg, "0.2.0").unwrap();
        assert_eq!(plist(&bundle, "CFBundleShortVersionString"), "0.2.0");
        assert!(verified(&bundle));
        let said = Command::new(bundle.join("Contents/MacOS/cutlass"))
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&said.stdout).trim(), "0.2.0");
        assert!(
            leftovers(&work.join("Applications")).is_empty(),
            "the old bundle is gone"
        );
        fs::remove_dir_all(&work).unwrap();
    }

    #[test]
    fn a_dmg_with_another_app_or_version_leaves_the_bundle() {
        let work = scratch("macos-refused");
        let old = fake_app(&work.join("old"), "0.1.0", "app.cutlass.desktop");
        let other = fake_app(&work.join("other"), "0.2.0", "com.example.other");
        let dmg = work.join("Other.dmg");
        dmg_of(&other, &dmg);
        let error = update_bundle(&work, &old, &dmg, "0.2.0").unwrap_err();
        assert!(error.to_string().contains("com.example.other"), "{error}");
        let bundle = work.join("Applications/Cutlass.app");
        assert_eq!(plist(&bundle, "CFBundleShortVersionString"), "0.1.0");
        assert!(verified(&bundle));
        assert!(leftovers(&work.join("Applications")).is_empty());

        // The feed says 0.3.0 but the DMG holds 0.2.0.
        fs::remove_dir_all(work.join("Applications")).unwrap();
        let new = fake_app(&work.join("new"), "0.2.0", "app.cutlass.desktop");
        let dmg = work.join("Cutlass.dmg");
        dmg_of(&new, &dmg);
        let error = update_bundle(&work, &old, &dmg, "0.3.0").unwrap_err();
        assert!(error.to_string().contains("0.2.0, not 0.3.0"), "{error}");
        assert_eq!(plist(&bundle, "CFBundleShortVersionString"), "0.1.0");

        // A bundle whose files changed after signing fails codesign.
        fs::remove_dir_all(work.join("Applications")).unwrap();
        fs::write(
            new.join("Contents/MacOS/cutlass"),
            "#!/bin/sh\necho tampered\n",
        )
        .unwrap();
        dmg_of(&new, &dmg);
        let error = update_bundle(&work, &old, &dmg, "0.2.0").unwrap_err();
        assert!(error.to_string().contains("codesign"), "{error}");
        assert_eq!(plist(&bundle, "CFBundleShortVersionString"), "0.1.0");
        fs::remove_dir_all(&work).unwrap();
    }

    /// The real desktop bundles: `CUTLASS_UPDATE_OLD_APP` (a Cutlass.app)
    /// updated from `CUTLASS_UPDATE_NEW_DMG` holding
    /// `CUTLASS_UPDATE_NEW_VERSION`.
    #[test]
    #[ignore = "needs two bundles from scripts/macos_bundle.py"]
    fn the_real_bundle_updates_from_the_real_dmg() {
        let old = PathBuf::from(env::var("CUTLASS_UPDATE_OLD_APP").unwrap());
        let dmg = PathBuf::from(env::var("CUTLASS_UPDATE_NEW_DMG").unwrap());
        let version = env::var("CUTLASS_UPDATE_NEW_VERSION").unwrap();
        let work = scratch("real");
        let bundle = update_bundle(&work, &old, &dmg, &version).unwrap();
        assert_eq!(plist(&bundle, "CFBundleShortVersionString"), version);
        assert!(verified(&bundle));
        assert!(leftovers(&work.join("Applications")).is_empty());
        if let Ok(keep) = env::var("CUTLASS_UPDATE_KEEP") {
            run("/usr/bin/ditto", &[bundle.to_str().unwrap(), &keep]);
        }
        fs::remove_dir_all(&work).unwrap();
    }
}
