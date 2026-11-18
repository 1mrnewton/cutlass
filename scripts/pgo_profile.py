"""Profile the AV1 decoder for a profile-guided release build.

Usage: python3 scripts/pgo_profile.py path/to/av1.profdata

It builds the `av1bench` example with instrumentation, decodes every IVF
stream in the AV1 fixtures on one thread and on four, and merges the counts
into the given file with the toolchain's `llvm-profdata` (the `llvm-tools`
rustup component). The release build then names it in `PGO_MERGE` for
`scripts/pgo.py`, which gives the merged profile to the workspace's crates.
The profile is for the host's target only.

The fixtures cover the coding tools rather than a typical stream, but their
profile keeps most of what one from long 4K clips gives: on an M5, 2.7%
and 4.5% fewer instructions decoding two 4K clips on one thread, where a
profile of the clips themselves gives 3.1% and 4.6%. Both thread counts
are run so that the code that shares a frame between threads is not taken
for cold. Functions the decodes never reach, and the crates the bench does
not link, build as without a profile.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pgo import build_wrapper, wrapped

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "crates" / "codecs" / "av1" / "fixtures"


def llvm_profdata() -> Path:
    sysroot = subprocess.run(
        ["rustc", "--print", "sysroot"], capture_output=True, text=True, check=True
    ).stdout.strip()
    host = next(
        line.split(":", 1)[1].strip()
        for line in subprocess.run(
            ["rustc", "-vV"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        if line.startswith("host:")
    )
    exe = ".exe" if os.name == "nt" else ""
    tool = Path(sysroot) / "lib" / "rustlib" / host / "bin" / f"llvm-profdata{exe}"
    if not tool.is_file():
        sys.exit(f"{tool} is missing: run `rustup component add llvm-tools`")
    return tool


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1]).resolve()
    tool = llvm_profdata()
    streams = sorted(FIXTURES.glob("*.ivf"))
    if not streams:
        sys.exit(f"no IVF streams in {FIXTURES}")
    with tempfile.TemporaryDirectory() as scratch:
        raw = Path(scratch) / "raw"
        target = Path(scratch) / "target"
        # Instrumented through pgo.py's wrapper at the path pgo.py uses, so
        # cutlass_av1 gets the -C metadata the release build gives it and
        # the counts match its functions there.
        wrapper = build_wrapper(ROOT / "target" / "pgo")
        env = dict(os.environ, **wrapped(wrapper, f"-Cprofile-generate={raw}"))
        subprocess.run(
            [
                "cargo", "build", "--release", "-p", "cutlass-av1",
                "--example", "av1bench", "--target-dir", str(target),
            ],
            cwd=ROOT, env=env, check=True,
        )
        exe = ".exe" if os.name == "nt" else ""
        bench = target / "release" / "examples" / f"av1bench{exe}"
        env = dict(os.environ)
        env["LLVM_PROFILE_FILE"] = str(raw / "av1-%p-%m.profraw")
        decoded = 0
        for threads in ("1", "4"):
            for stream in streams:
                # A stream the decoder refuses still counts what it ran.
                run = subprocess.run(
                    [bench, stream, threads], env=env, capture_output=True
                )
                decoded += run.returncode == 0
        # wide.ivf is refused on purpose; most of the rest must decode.
        if decoded < len(streams):
            sys.exit(f"only {decoded} of {2 * len(streams)} decodes succeeded")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [tool, "merge", "-o", out, *sorted(raw.glob("*.profraw"))], check=True
        )
    print(out)


if __name__ == "__main__":
    main()
