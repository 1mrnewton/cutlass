"""Package a built Windows desktop binary as an NSIS installer.

Usage: python3 scripts/windows_installer.py path/to/cutlass.exe out/Cutlass-x86_64-setup.exe

It stages the binary and the icon, runs makensis on
`installer/windows/cutlass.nsi` (`MAKENSIS` names it; default `makensis`
on the PATH), and checks the installer was written. The installer is for
the current user and needs no administrator. It is for the processor the
binary was built for, read from the binary itself (x86_64 or aarch64),
and refuses to install on a machine that cannot run it.

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


# The PE header's machine field: IMAGE_FILE_MACHINE_AMD64 and _ARM64.
MACHINES = {0x8664: "x86_64", 0xAA64: "aarch64"}


def arch(binary: Path) -> str:
    """The processor a Windows program is built for."""
    data = binary.read_bytes()[:4096]
    if data[:2] != b"MZ":
        raise SystemExit(f"{binary} is not a Windows program")
    header = int.from_bytes(data[0x3C:0x40], "little")
    if data[header:header + 4] != b"PE\0\0":
        raise SystemExit(f"{binary} has no PE header")
    machine = int.from_bytes(data[header + 4:header + 6], "little")
    if machine not in MACHINES:
        raise SystemExit(f"{binary} is built for machine {machine:#x}, not x86_64 or aarch64")
    return MACHINES[machine]


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
        raise SystemExit("usage: windows_installer.py path/to/cutlass.exe out/Cutlass-x86_64-setup.exe")
    binary = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if not binary.is_file():
        raise SystemExit(f"{binary} is not a file; build it first")
    built_for = arch(binary)
    named = [name for name in MACHINES.values() if name in output.name]
    if named and named != [built_for]:
        raise SystemExit(f"{output.name} names {named[0]}, but {binary.name} is built for {built_for}")
    tool = os.environ.get("MAKENSIS", "makensis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        stage = Path(scratch)
        shutil.copy2(binary, stage / "cutlass.exe")
        shutil.copy2(ROOT / "apps" / "desktop" / "icons" / "windows.ico", stage / "cutlass.ico")
        sign(stage / "cutlass.exe")
        subprocess.run(
            [tool, "-V2", f"-DVERSION={version()}", f"-DARCH={built_for}", f"-DSOURCE={stage}",
             f"-DOUTFILE={output}", str(ROOT / "installer" / "windows" / "cutlass.nsi")],
            check=True,
        )
    assert output.is_file() and output.stat().st_size > 1000, f"makensis wrote no {output}"
    with open(output, "rb") as file:
        assert file.read(2) == b"MZ", "the installer is not a Windows program"
    sign(output)
    print(f"{output} ({built_for})")


if __name__ == "__main__":
    main()
