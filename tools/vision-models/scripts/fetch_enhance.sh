#!/usr/bin/env bash
# Downloads Real-ESRGAN's realesr-general-x4v3 weights, checks their hash,
# converts them into the vision crate's .bpk, and writes the PyTorch
# reference its enhance test checks burn against. Everything lands in
# local-files/models/enhance beside the repository.
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
