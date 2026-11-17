#!/usr/bin/env bash
# Downloads BiRefNet_lite (MIT) as its author published it, checks its
# hash, converts it into the vision crate's .bpk, and writes the reference
# its matting test checks burn against. Everything lands in
# local-files/models/matting beside the repository, and the .bpk is also
# copied into the desktop app's Models/matting folder, where a release
# build looks for it. Set CUTLASS_MODELS to put it in another Models folder
# (a home the user chose), or to an empty string to skip the copy.
#
# The reference runs the model code published beside the weights, at the
# same pinned revision, under PyTorch.
#
# Needs curl, cargo, and uv (PyTorch runs in a throwaway environment).
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
out="$files/models/matting"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out/hf"

# The revision pins every file; the hash pins the weights.
base="https://huggingface.co/ZhengPeng7/BiRefNet_lite/resolve/aa62cd87eafb9cc43056d08ef3615a14628b831d"
want="4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

for file in model.safetensors config.json BiRefNet_config.py birefnet.py; do
    curl -fsSL --retry 3 -o "$work/$file" "$base/$file"
done
got="$(sha256 "$work/model.safetensors")"
if [ "$got" != "$want" ]; then
    echo "model.safetensors: hash $got, expected $want" >&2
    exit 1
fi
mv "$work/model.safetensors" "$work/config.json" "$work/BiRefNet_config.py" "$work/birefnet.py" "$out/hf/"

(cd "$repo" && cargo run -q -p cutlass-vision --example convert_matting -- "$out/hf/model.safetensors" "$out")
(cd "$here" && uv run --no-project --with torch --with torchvision --with transformers --with timm \
    --with kornia --with einops --with numpy --with safetensors \
    python -I scripts/matting_reference.py "$out/hf" "$out/reference_frame.safetensors")

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
    mkdir -p "$models/matting"
    cp "$out/birefnet_lite.bpk" "$models/matting/birefnet_lite.bpk.part"
    mv "$models/matting/birefnet_lite.bpk.part" "$models/matting/birefnet_lite.bpk"
    echo "installed $models/matting/birefnet_lite.bpk"
fi
