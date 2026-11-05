"""Package a built Linux desktop binary as an AppImage.

Usage: python3 scripts/appimage.py path/to/cutlass out/Cutlass.AppImage

It lays out an AppDir (the binary, a launcher, the desktop entry and the
icon), checks it, and hands it to appimagetool, which the `APPIMAGETOOL`
variable names (default `appimagetool` on the PATH). The machine's
architecture is the AppImage's.

GTK, Vulkan and ALSA come from the system, as on any Linux desktop: the
file dialogs need GTK 3, the preview a Vulkan or GL driver, and the sound
libasound, which Cutlass loads only when it is there.
"""

import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DESKTOP_ENTRY = """[Desktop Entry]
Type=Application
Name=Cutlass
GenericName=Video editor
Comment=Edit videos
Exec=cutlass %F
Icon=cutlass
Terminal=false
Categories=AudioVideo;Video;AudioVideoEditing;
"""

# The launcher: run the binary inside the image with what it was given.
APP_RUN = """#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/cutlass" "$@"
"""

ARCHES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}


def write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def lay_out(binary: Path, app_dir: Path) -> None:
    """The AppDir appimagetool packs."""
    bin_dir = app_dir / "usr" / "bin"
    bin_dir.mkdir(parents=True)
    shutil.copy2(binary, bin_dir / "cutlass")
    write(app_dir / "AppRun", APP_RUN, executable=True)
    write(app_dir / "cutlass.desktop", DESKTOP_ENTRY)
    icon = ROOT / "apps" / "desktop" / "icons" / "square.png"
    shutil.copy2(icon, app_dir / "cutlass.png")
    icons = app_dir / "usr" / "share" / "icons" / "hicolor" / "512x512" / "apps"
    icons.mkdir(parents=True)
    shutil.copy2(icon, icons / "cutlass.png")
    applications = app_dir / "usr" / "share" / "applications"
    applications.mkdir(parents=True)
    shutil.copy2(app_dir / "cutlass.desktop", applications / "cutlass.desktop")
    check(app_dir)


def check(app_dir: Path) -> None:
    """What appimagetool and desktop menus need is there and runs."""
    for name in ["AppRun", "cutlass.desktop", "cutlass.png", "usr/bin/cutlass"]:
        assert (app_dir / name).is_file(), f"{name} is missing from the AppDir"
    for name in ["AppRun", "usr/bin/cutlass"]:
        assert os.access(app_dir / name, os.X_OK), f"{name} is not executable"
    entry = (app_dir / "cutlass.desktop").read_text()
    assert "Exec=cutlass" in entry and "Icon=cutlass" in entry, entry
    png = (app_dir / "cutlass.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "the icon is not a PNG"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: appimage.py path/to/cutlass out/Cutlass.AppImage")
    binary = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if not binary.is_file():
        raise SystemExit(f"{binary} is not a file; build it first")
    arch = ARCHES.get(platform.machine().lower())
    if arch is None:
        raise SystemExit(f"no AppImage for {platform.machine()}")
    tool = os.environ.get("APPIMAGETOOL", "appimagetool")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        app_dir = Path(scratch) / "Cutlass.AppDir"
        lay_out(binary, app_dir)
        env = dict(os.environ, ARCH=arch)
        # appimagetool is itself an AppImage; this runs it without FUSE.
        env.setdefault("APPIMAGE_EXTRACT_AND_RUN", "1")
        subprocess.run(
            [tool, "--no-appstream", str(app_dir), str(output)],
            env=env, check=True,
        )
    assert output.is_file(), f"appimagetool wrote no {output}"
    print(output)


if __name__ == "__main__":
    main()
