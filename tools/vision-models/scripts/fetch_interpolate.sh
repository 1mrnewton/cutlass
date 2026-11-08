#!/usr/bin/env bash
# Downloads RIFE v4.26's weights, checks their hash, converts them into
# the vision crate's .bpk, and writes the PyTorch reference its
# interpolate test checks burn against. Everything lands in
# local-files/models/interpolate beside the repository, and the .bpk is also
# copied into the desktop app's Models/interpolate folder, where a release
# build looks for it. Set CUTLASS_MODELS to put it in another Models
# folder (a home the user chose), or to an empty string to skip the copy.
#
# Needs curl, cargo, and uv (PyTorch runs in a throwaway environment).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="$repo/../local-files/models/interpolate"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out"

# vs-rife (MIT) mirrors Practical-RIFE's weights, which are only on
# Google Drive, as a GitHub release; the hash pins the file.
url="https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.26.pkl"
want="45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

pth="$work/flownet_v4.26.pkl"
curl -fsSL --retry 3 -o "$pth" "$url"
got="$(sha256 "$pth")"
if [ "$got" != "$want" ]; then
    echo "flownet_v4.26.pkl: hash $got, expected $want" >&2
    exit 1
fi

(cd "$repo" && cargo run -q -p cutlass-vision --example convert_interpolate -- "$pth" "$out")
(cd "$here" && uv run --no-project --with torch --with numpy --with safetensors \
    python -I scripts/interpolate_reference.py "$pth" "$out/reference_pair.safetensors")

# The app's own Models folder, as cutlass-storage's desktop_home names it.
app_models() {
    if [ -n "${CUTLASS_MODELS+set}" ]; then printf '%s' "$CUTLASS_MODELS"; return; fi
    case "$(uname -s)" in
        Darwin) printf '%s' "$HOME/Library/Application Support/Cutlass/Models" ;;
        MINGW*|MSYS*|CYGWIN*) printf '%s' "${APPDATA:-$HOME/AppData/Roaming}/Cutlass/Models" ;;
        *) printf '%s' "${XDG_DATA_HOME:-$HOME/.local/share}/cutlass/Models" ;;
    esac
}

models="$(app_models)"
if [ -n "$models" ]; then
    mkdir -p "$models/interpolate"
    cp "$out/rife_v4_26.bpk" "$models/interpolate/rife_v4_26.bpk.part"
    mv "$models/interpolate/rife_v4_26.bpk.part" "$models/interpolate/rife_v4_26.bpk"
    echo "installed $models/interpolate/rife_v4_26.bpk"
fi
