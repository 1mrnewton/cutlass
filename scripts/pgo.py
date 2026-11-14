"""A release build with the H.264 codec optimised from a profile (PGO).

    python3 scripts/pgo.py -p cutlass --features public-release

The arguments are those of `cargo build --release`, which this runs last;
the binaries land where that build puts them. Before it:

1. `cutlass-h264-pgo` is built with `-Cprofile-generate` into
   `target/pgo/build`, and run. It decodes and encodes the synthetic
   streams in `tools/h264-pgo/streams`.
2. The counts are merged with `llvm-profdata` (`rustup component add
   llvm-tools`) and cut down to the functions of `cutlass-h264` and
   `cutlass-h264-kernels`.
3. The build runs with `-Cprofile-use` on that profile. Every other crate
   has no counts in it and is built as it would be without one.

`PGO_MERGE` names other `.profdata` files, separated as in `PATH`, that are
merged into that profile as they are, so one build carries the counts of
other crates too: the release build passes the AV1 decoder's profile from
`scripts/pgo_profile.py` this way.

On five test films (phone, 1080p and 4K footage) the decoder retired
4.4-6.5% fewer instructions this way and took 1.6-7% fewer cycles on one
thread, with the same output. Trained on the films themselves instead of
the synthetic streams, it retired up to 1.3% fewer again.

A profile only reaches a function whose name and shape are the same in
both builds, so this checks that both gave `cutlass_h264` the same
`-C metadata` and fails the build when they did not. `RUSTFLAGS` set by
the caller is kept, and must be the same for both builds.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAINER = "cutlass-h264-pgo"
# A function is kept when its symbol names either crate, as its own or as
# the crate that instantiated it.
KEEP = "cutlass_h264"
# Terminal colour codes, which Cargo writes when told to colour always.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main(args: list[str]) -> int:
    if "--profile" in args or any(a.startswith("--profile=") for a in args):
        return fail("pass the arguments of `cargo build --release`, not a --profile")
    target = value_of(args, "--target")
    profdata = llvm_profdata()
    # The counts and profiles are kept out of the trainer's target
    # directory, which Cargo only cleans when it made it itself.
    out = ROOT / "target" / "pgo"
    build = out / "build"
    raw = out / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    flags = os.environ.get("RUSTFLAGS", "").strip()

    # Cargo does not see that a profile changed, only RUSTFLAGS, so the two
    # crates are cleaned before each build: they are always built against
    # this run's profile, and their -C metadata is printed.
    at = ["--target", target] if target else []
    print(f"pgo: building {TRAINER} instrumented", flush=True)
    clean(["--target-dir", str(build), *at])
    train = ["build", "--release", "-p", TRAINER, "--target-dir", str(build), *at]
    trained_meta = cargo(train, f"{flags} -Cprofile-generate={raw}")
    if trained_meta is None:
        return fail("the instrumented build did not compile cutlass-h264")
    exe = build / (target or "") / "release" / (TRAINER + (".exe" if os.name == "nt" else ""))
    print("pgo: training", flush=True)
    env = dict(os.environ, LLVM_PROFILE_FILE=str(raw / "%m-%p.profraw"))
    subprocess.run([str(exe), str(ROOT / "tools" / "h264-pgo" / "streams")], env=env, check=True)

    merged = out / "merged.profdata"
    text = out / "merged.proftext"
    kept = out / "h264.proftext"
    profile = out / "h264.profdata"
    raws = sorted(str(p) for p in raw.glob("*.profraw"))
    if not raws:
        return fail("the training run wrote no profile")
    subprocess.run([profdata, "merge", "-o", str(merged), *raws], check=True)
    subprocess.run([profdata, "merge", "--text", "-o", str(text), str(merged)], check=True)
    count = keep_h264(text, kept)
    if count == 0:
        return fail(f"the profile has no function of {KEEP}")
    extra = [p for p in os.environ.get("PGO_MERGE", "").split(os.pathsep) if p]
    missing = [p for p in extra if not pathlib.Path(p).is_file()]
    if missing:
        return fail(f"PGO_MERGE names no such file: {', '.join(missing)}")
    subprocess.run([profdata, "merge", "-o", str(profile), str(kept), *extra], check=True)
    print(f"pgo: {count} functions profiled; building", flush=True)

    target_dir = value_of(args, "--target-dir")
    clean(at + (["--target-dir", target_dir] if target_dir else []))
    built_meta = cargo(["build", "--release", *args], f"{flags} -Cprofile-use={profile}")
    if built_meta is None:
        return fail(
            "the build did not compile cutlass-h264, so the profile was not used: "
            "build a package that depends on it"
        )
    if built_meta != trained_meta:
        return fail(
            f"cutlass_h264 was built with -C metadata={built_meta}, "
            f"trained with {trained_meta}: the profile matched none of its functions"
        )
    print("pgo: done", flush=True)
    return 0


def value_of(args: list[str], flag: str) -> str | None:
    """The value of a cargo option given as `flag value` or `flag=value`."""
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def clean(args: list[str]) -> None:
    """Removes the release build of the two H.264 crates."""
    subprocess.run(
        ["cargo", "clean", "--release", "-p", "cutlass-h264", "-p", "cutlass-h264-kernels", *args],
        cwd=ROOT,
        check=True,
    )


def llvm_profdata() -> str:
    sysroot = subprocess.run(
        ["rustc", "--print", "sysroot"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    version = subprocess.run(
        ["rustc", "-vV"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    host = re.search(r"^host: (\S+)", version, re.M)
    name = "llvm-profdata" + (".exe" if os.name == "nt" else "")
    path = pathlib.Path(sysroot) / "lib" / "rustlib" / (host.group(1) if host else "") / "bin" / name
    if not path.is_file():
        sys.exit(f"pgo: {path} is missing: run `rustup component add llvm-tools`")
    return str(path)


def cargo(args: list[str], rustflags: str) -> str | None:
    """Runs cargo verbosely, prints all but its rustc command lines, and
    returns the `-C metadata` it gave cutlass_h264, if it compiled it."""
    env = dict(os.environ, RUSTFLAGS=rustflags.strip())
    proc = subprocess.Popen(
        ["cargo", *args, "-v"], cwd=ROOT, env=env, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )
    meta = None
    assert proc.stderr is not None
    for line in proc.stderr:
        # CI sets CARGO_TERM_COLOR=always, which colours the word Running.
        plain = ANSI.sub("", line)
        if plain.lstrip().startswith("Running `"):
            found = re.search(r"--crate-name cutlass_h264 .*?-C metadata=(\w+)", plain)
            if found:
                meta = found.group(1)
            continue
        sys.stderr.write(line)
    if proc.wait() != 0:
        sys.exit(f"pgo: cargo {' '.join(args)} failed")
    return meta


def keep_h264(text: pathlib.Path, out: pathlib.Path) -> int:
    """Copies the text profile's header and the records of functions whose
    name contains KEEP. A record is the function's name and its counts, up
    to a blank line."""
    lines = text.read_text(encoding="utf-8").split("\n")
    header = []
    while lines and (lines[0].startswith("#") or lines[0].startswith(":")):
        header.append(lines.pop(0))
    kept, record = [], []
    for line in lines + [""]:
        if line.strip():
            record.append(line)
            continue
        if record and KEEP in record[0]:
            kept.append("\n".join(record))
        record = []
    out.write_text("\n".join(header) + "\n" + "\n\n".join(kept) + "\n", encoding="utf-8")
    return len(kept)


def fail(message: str) -> int:
    print(f"pgo: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
