#!/usr/bin/env bash
# Downloads Depth Anything V2 Small (Apache-2.0) as published for
# transformers, checks its hash, converts it into the vision crate's .bpk,
# and writes the reference its depth test checks burn against. Everything
# lands in local-files/models/relight beside the repository, and the .bpk
# is also copied into the desktop app's Models/relight folder, where a
# release build looks for it. Set CUTLASS_MODELS to put it in another
# Models folder (a home the user chose), or to an empty string to skip the
# copy.
#
# Only the Small model is Apache-2.0; Base, Large and Giant are
# non-commercial and are not fetched.
#
# Needs curl, cargo, and uv (PyTorch runs in a throwaway environment).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="$repo/../local-files/models/relight"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out/hf"

# The revision pins both files; the hash pins the weights.
base="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/5426e4f0f36572d16453bbda7a8389317b1bef99"
want="3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

curl -fsSL --retry 3 -o "$work/model.safetensors" "$base/model.safetensors"
curl -fsSL --retry 3 -o "$work/config.json" "$base/config.json"
got="$(sha256 "$work/model.safetensors")"
if [ "$got" != "$want" ]; then
    echo "model.safetensors: hash $got, expected $want" >&2
    exit 1
fi
mv "$work/model.safetensors" "$work/config.json" "$out/hf/"

(cd "$repo" && cargo run -q -p cutlass-vision --example convert_depth -- "$out/hf/model.safetensors" "$out")
(cd "$here" && uv run --no-project --with torch --with transformers --with numpy --with safetensors \
    python -I scripts/depth_reference.py "$out/hf" "$out/reference_frame.safetensors")

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
    mkdir -p "$models/relight"
    cp "$out/depth_anything_v2_small.bpk" "$models/relight/depth_anything_v2_small.bpk.part"
    mv "$models/relight/depth_anything_v2_small.bpk.part" "$models/relight/depth_anything_v2_small.bpk"
    echo "installed $models/relight/depth_anything_v2_small.bpk"
fi
