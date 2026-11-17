#!/usr/bin/env bash
# Downloads MODNet (Apache-2.0) as Xenova published it for transformers.js,
# checks its hash, reads its weights out of the ONNX file, converts them
# into the vision crate's .bpk, and writes the reference its portrait test
# checks burn against. Everything lands in local-files/models/portrait
# beside the repository, and the .bpk is also copied into the desktop app's
# Models/portrait folder, where a release build looks for it. Set
# CUTLASS_MODELS to put it in another Models folder (a home the user
# chose), or to an empty string to skip the copy.
#
# There is no PyTorch checkpoint of MODNet on Hugging Face; the ONNX file
# is the full-precision export of the official one.
#
# Needs curl, cargo, and uv (onnx and onnxruntime run in a throwaway
# environment).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
# local-files sits beside the checkout; a worktree under .claude/worktrees
# finds the same one further up.
files="$repo/../local-files"
dir="$repo"
while [ "$dir" != "/" ]; do
    if [ -d "$dir/local-files" ]; then files="$dir/local-files"; break; fi
    dir="$(dirname "$dir")"
done
out="$files/models/portrait"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out/hf"

# The revision pins every file; the hash pins the weights.
base="https://huggingface.co/Xenova/modnet/resolve/fa2fa546052fba4c08921230a26cc69a333fca12"
want="07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

curl -fsSL --retry 3 -o "$work/model.onnx" "$base/onnx/model.onnx"
curl -fsSL --retry 3 -o "$work/preprocessor_config.json" "$base/preprocessor_config.json"
got="$(sha256 "$work/model.onnx")"
if [ "$got" != "$want" ]; then
    echo "model.onnx: hash $got, expected $want" >&2
    exit 1
fi
mv "$work/model.onnx" "$work/preprocessor_config.json" "$out/hf/"

(cd "$here" && uv run --no-project --with onnx --with numpy --with safetensors \
    python -I scripts/portrait_weights.py "$out/hf/model.onnx" "$out/hf/modnet.safetensors")
(cd "$repo" && cargo run -q -p cutlass-vision --example convert_portrait -- "$out/hf/modnet.safetensors" "$out")
(cd "$here" && uv run --no-project --with onnxruntime --with numpy --with safetensors \
    python -I scripts/portrait_reference.py "$out/hf/model.onnx" "$out/reference_frame.safetensors")

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
    mkdir -p "$models/portrait"
    cp "$out/modnet.bpk" "$models/portrait/modnet.bpk.part"
    mv "$models/portrait/modnet.bpk.part" "$models/portrait/modnet.bpk"
    echo "installed $models/portrait/modnet.bpk"
fi
