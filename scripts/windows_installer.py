"""Package a built Windows desktop binary as an NSIS installer.

Usage: python3 scripts/windows_installer.py path/to/cutlass.exe out/Cutlass-setup.exe

It stages the binary and the icon, runs makensis on
`installer/windows/cutlass.nsi` (`MAKENSIS` names it; default `makensis`
on the PATH), and checks the installer was written. The installer is for
the current user and needs no administrator.

Signing: with `SIGNTOOL_CERT` (a .pfx file) and `SIGNTOOL_PASSWORD` set,
the binary and the installer are signed with signtool and timestamped.
Without them both stay unsigned, and SmartScreen warns on first run.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMESTAMP = "http://timestamp.digicert.com"


def version() -> str:
    for line in (ROOT / "Cargo.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    return "0.1.0"


def sign(path: Path) -> None:
    cert = os.environ.get("SIGNTOOL_CERT")
    if not cert:
        return
    subprocess.run(
        ["signtool", "sign", "/f", cert, "/p", os.environ.get("SIGNTOOL_PASSWORD", ""),
         "/fd", "sha256", "/tr", TIMESTAMP, "/td", "sha256", str(path)],
        check=True,
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: windows_installer.py path/to/cutlass.exe out/Cutlass-setup.exe")
    binary = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if not binary.is_file():
        raise SystemExit(f"{binary} is not a file; build it first")
    tool = os.environ.get("MAKENSIS", "makensis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        stage = Path(scratch)
        shutil.copy2(binary, stage / "cutlass.exe")
        shutil.copy2(ROOT / "apps" / "desktop" / "icons" / "windows.ico", stage / "cutlass.ico")
        sign(stage / "cutlass.exe")
        subprocess.run(
            [tool, "-V2", f"-DVERSION={version()}", f"-DSOURCE={stage}",
             f"-DOUTFILE={output}", str(ROOT / "installer" / "windows" / "cutlass.nsi")],
            check=True,
        )
    assert output.is_file() and output.stat().st_size > 1000, f"makensis wrote no {output}"
    with open(output, "rb") as file:
        assert file.read(2) == b"MZ", "the installer is not a Windows program"
    sign(output)
    print(output)


if __name__ == "__main__":
    main()
