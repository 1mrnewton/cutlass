"""Package a built macOS desktop binary as Cutlass.app and a DMG.

Usage: python3 scripts/macos_bundle.py path/to/cutlass out/dir

It lays out `Cutlass.app` (the binary, Info.plist and the icon), signs it,
checks it, and writes `Cutlass.dmg` beside it with an Applications link.

Info.plist declares the documents the app opens: a project folder, and
movies, sound and pictures to import. Finder sends them as an "open
documents" event, which the app handles (`listen_for_opened_files`).

Signing is ad hoc (`codesign -s -`) unless `CODESIGN_IDENTITY` names a
Developer ID Application certificate in the keychain; then it signs with
the hardened runtime and a timestamp. With `NOTARY_PROFILE` also set (a
`xcrun notarytool store-credentials` profile), the DMG is notarized and
stapled. `BUNDLE_ID` sets the bundle identifier, and `BUNDLE_VERSION` the
version in Info.plist (default: the workspace version).
"""

import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MEDIA = [
    ("Movie", ["public.movie"]),
    ("Sound", ["public.audio"]),
    ("Picture", ["public.image"]),
    ("Lottie animation", ["public.json"]),
]


def version() -> str:
    if os.environ.get("BUNDLE_VERSION"):
        return os.environ["BUNDLE_VERSION"]
    for line in (ROOT / "Cargo.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    return "0.1.0"


def info_plist(bundle_id: str) -> dict:
    documents = [{
        "CFBundleTypeName": "Cutlass project",
        "CFBundleTypeRole": "Editor",
        # A project is a folder; Cutlass opens one, and other apps stay the
        # default for folders.
        "LSItemContentTypes": ["public.folder"],
        "LSHandlerRank": "Alternate",
    }]
    for name, types in MEDIA:
        documents.append({
            "CFBundleTypeName": name,
            "CFBundleTypeRole": "Viewer",
            "LSItemContentTypes": types,
            "LSHandlerRank": "Alternate",
        })
    return {
        "CFBundleName": "Cutlass",
        "CFBundleDisplayName": "Cutlass",
        "CFBundleIdentifier": bundle_id,
        "CFBundleExecutable": "cutlass",
        "CFBundleIconFile": "Cutlass",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": version(),
        "CFBundleVersion": version(),
        "LSMinimumSystemVersion": "11.0",
        "LSApplicationCategoryType": "public.app-category.video",
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
        "CFBundleDocumentTypes": documents,
    }


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def icns(png: Path, out: Path) -> None:
    """The icon set iconutil wants, scaled from the one PNG."""
    with tempfile.TemporaryDirectory() as scratch:
        iconset = Path(scratch) / "Cutlass.iconset"
        iconset.mkdir()
        for side in [16, 32, 128, 256, 512]:
            for scale in [1, 2]:
                pixels = side * scale
                suffix = "" if scale == 1 else "@2x"
                target = iconset / f"icon_{side}x{side}{suffix}.png"
                run("sips", "-z", str(pixels), str(pixels), str(png), "--out", str(target))
        run("iconutil", "-c", "icns", str(iconset), "-o", str(out))


def lay_out(binary: Path, app: Path, bundle_id: str) -> None:
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()
    shutil.copy2(binary, contents / "MacOS" / "cutlass")
    with open(contents / "Info.plist", "wb") as file:
        plistlib.dump(info_plist(bundle_id), file)
    (contents / "PkgInfo").write_text("APPL????")
    icns(ROOT / "apps" / "desktop" / "icons" / "macos.png", contents / "Resources" / "Cutlass.icns")


def sign(app: Path) -> None:
    identity = os.environ.get("CODESIGN_IDENTITY")
    if identity:
        run("codesign", "--force", "--options", "runtime", "--timestamp",
            "--sign", identity, str(app))
    else:
        run("codesign", "--force", "--sign", "-", str(app))
    run("codesign", "--verify", "--strict", "--verbose=2", str(app))


def check(app: Path) -> None:
    """What Finder and Launch Services need is there and reads back."""
    with open(app / "Contents" / "Info.plist", "rb") as file:
        plist = plistlib.load(file)
    assert plist["CFBundleExecutable"] == "cutlass", plist
    kinds = [kind for entry in plist["CFBundleDocumentTypes"] for kind in entry["LSItemContentTypes"]]
    assert "public.folder" in kinds and "public.movie" in kinds, kinds
    assert os.access(app / "Contents" / "MacOS" / "cutlass", os.X_OK)
    assert (app / "Contents" / "Resources" / "Cutlass.icns").stat().st_size > 1000
    run("plutil", "-lint", str(app / "Contents" / "Info.plist"))


def dmg(app: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        stage = Path(scratch) / "Cutlass"
        stage.mkdir()
        shutil.copytree(app, stage / app.name, symlinks=True)
        (stage / "Applications").symlink_to("/Applications")
        if out.exists():
            out.unlink()
        run("hdiutil", "create", "-volname", "Cutlass", "-srcfolder", str(stage),
            "-fs", "HFS+", "-format", "UDZO", "-ov", str(out))
    if os.environ.get("CODESIGN_IDENTITY"):
        run("codesign", "--force", "--timestamp", "--sign", os.environ["CODESIGN_IDENTITY"], str(out))
    profile = os.environ.get("NOTARY_PROFILE")
    if profile:
        run("xcrun", "notarytool", "submit", str(out), "--keychain-profile", profile, "--wait")
        run("xcrun", "stapler", "staple", str(out))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: macos_bundle.py path/to/cutlass out/dir")
    binary = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    if not binary.is_file():
        raise SystemExit(f"{binary} is not a file; build it first")
    bundle_id = os.environ.get("BUNDLE_ID", "app.cutlass.desktop")
    out.mkdir(parents=True, exist_ok=True)
    app = out / "Cutlass.app"
    if app.exists():
        shutil.rmtree(app)
    lay_out(binary, app, bundle_id)
    sign(app)
    check(app)
    dmg(app, out / "Cutlass.dmg")
    print(app)
    print(out / "Cutlass.dmg")


if __name__ == "__main__":
    main()
