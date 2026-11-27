#!/usr/bin/env bash
# Downloads Beat This!'s small model (MIT, code and weights) as its authors
# published it, checks its hash, reads its weights out of the checkpoint,
# converts them into the vision crate's .bpk, and writes the reference its
# beats test checks burn against. Everything lands in
# local-files/models/beats beside the repository, and the .bpk is also
# copied into the desktop app's Models/beats folder, where a release build
# looks for it. Set CUTLASS_MODELS to put it in another Models folder (a
# home the user chose), or to an empty string to skip the copy.
#
# The checkpoint is `small0`: the smaller model, trained on all the
# authors' data but GTZAN. The reference runs the authors' own code, at a
# pinned revision, under PyTorch.
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
out="$files/models/beats"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out/hf"

# The authors' share has no revisions; the hash pins the weights.
url="https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt"
want="6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b"
# The authors' code the reference runs.
code="beat-this @ git+https://github.com/CPJKU/beat_this@b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

curl -fsSL --retry 3 -o "$work/small0.ckpt" "$url"
got="$(sha256 "$work/small0.ckpt")"
if [ "$got" != "$want" ]; then
    echo "small0.ckpt: hash $got, expected $want" >&2
    exit 1
fi
mv "$work/small0.ckpt" "$out/hf/"

(cd "$here" && uv run --no-project --with torch --with numpy --with safetensors \
    python -I scripts/beats_weights.py "$out/hf/small0.ckpt" "$out/hf/beat_this_small.safetensors")
(cd "$repo" && cargo run -q -p cutlass-vision --example convert_beats -- "$out/hf/beat_this_small.safetensors" "$out")
(cd "$here" && uv run --no-project --with "$code" --with numpy --with safetensors \
    python -I scripts/beats_reference.py "$out/hf/small0.ckpt" "$out/reference_piece.safetensors")

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
    mkdir -p "$models/beats"
    cp "$out/beat_this_small.bpk" "$models/beats/beat_this_small.bpk.part"
    mv "$models/beats/beat_this_small.bpk.part" "$models/beats/beat_this_small.bpk"
    echo "installed $models/beats/beat_this_small.bpk"
fi
