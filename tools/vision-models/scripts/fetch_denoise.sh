#!/usr/bin/env bash
# Downloads FastDVDnet's weights for clipped noise, checks their hash,
# converts them into the vision crate's .bpk, and writes the PyTorch
# reference its denoise test checks burn against. Everything lands in
# local-files/models/denoise beside the repository, and the .bpk is also
# copied into the desktop app's Models/denoise folder, where a release
# build looks for it. Set CUTLASS_MODELS to put it in another Models
# folder (a home the user chose), or to an empty string to skip the copy.
#
# Needs curl, cargo, and uv (PyTorch runs in a throwaway environment).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="$repo/../local-files/models/denoise"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out"

# Pinned to a commit: the file lives in the repository, not a release.
url="https://raw.githubusercontent.com/m-tassano/fastdvdnet/c8fdf6182a0340e89dd18f5df25b47337cbede6f/model_clipped_noise.pth"
want="8118974ac7defaa5037f73caf87e0cb53efcfa49ae77d55c05ab187f59e55949"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

pth="$work/model_clipped_noise.pth"
curl -fsSL --retry 3 -o "$pth" "$url"
got="$(sha256 "$pth")"
if [ "$got" != "$want" ]; then
    echo "model_clipped_noise.pth: hash $got, expected $want" >&2
    exit 1
fi

(cd "$repo" && cargo run -q -p cutlass-vision --example convert_denoise -- "$pth" "$out")
(cd "$here" && uv run --no-project --with torch --with numpy --with safetensors \
    python -I scripts/denoise_reference.py "$pth" "$out/reference_clip.safetensors")

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
    mkdir -p "$models/denoise"
    cp "$out/fastdvdnet_clipped.bpk" "$models/denoise/fastdvdnet_clipped.bpk.part"
    mv "$models/denoise/fastdvdnet_clipped.bpk.part" "$models/denoise/fastdvdnet_clipped.bpk"
    echo "installed $models/denoise/fastdvdnet_clipped.bpk"
fi
