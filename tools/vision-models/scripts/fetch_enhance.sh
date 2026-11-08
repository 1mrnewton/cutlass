#!/usr/bin/env bash
# Downloads Real-ESRGAN's realesr-general-x4v3 weights, checks their hash,
# converts them into the vision crate's .bpk, and writes the PyTorch
# reference its enhance test checks burn against. Everything lands in
# local-files/models/enhance beside the repository, and the .bpk is also
# copied into the desktop app's Models/enhance folder, where a release
# build looks for it. Set CUTLASS_MODELS to put it in another Models
# folder (a home the user chose), or to an empty string to skip the copy.
#
# Needs curl, cargo, and uv (PyTorch runs in a throwaway environment).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="$repo/../local-files/models/enhance"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out"

url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
want="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

pth="$work/realesr-general-x4v3.pth"
curl -fsSL --retry 3 -o "$pth" "$url"
got="$(sha256 "$pth")"
if [ "$got" != "$want" ]; then
    echo "realesr-general-x4v3.pth: hash $got, expected $want" >&2
    exit 1
fi

(cd "$repo" && cargo run -q -p cutlass-vision --example convert_enhance -- "$pth" "$out")
(cd "$here" && uv run --no-project --with torch --with numpy --with safetensors --with pillow \
    python -I scripts/enhance_reference.py "$pth" "$out/reference_pattern.safetensors")

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
    mkdir -p "$models/enhance"
    cp "$out/realesr_general_x4v3.bpk" "$models/enhance/realesr_general_x4v3.bpk.part"
    mv "$models/enhance/realesr_general_x4v3.bpk.part" "$models/enhance/realesr_general_x4v3.bpk"
    echo "installed $models/enhance/realesr_general_x4v3.bpk"
fi
