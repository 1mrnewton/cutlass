#!/usr/bin/env bash
# Downloads the MobileSAM files the bench needs into models/, checks each
# against its hash, and cuts the decoder down to its low-res masks.
#
# Needs curl, and Python 3 with the onnx package for the cut:
#     python3 -m pip install onnx
# or, with uv, nothing to install: the script uses `uv run --with onnx`.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
models="$here/models"
mkdir -p "$models"

# Pinned revisions, so the bytes never change under the hashes.
acly="https://huggingface.co/Acly/MobileSAM/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8"
candle="https://huggingface.co/lmz/candle-sam/resolve/8b4cb7c743f3b3cb8afd212a86ae15b1bbfdac97"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fetch() {
    local url="$1" file="$models/$2" want="$3"
    if [ -f "$file" ] && [ "$(sha256 "$file")" = "$want" ]; then
        echo "have $2"
        return
    fi
    echo "fetching $2"
    curl -fsSL --retry 3 -o "$file.part" "$url"
    local got
    got="$(sha256 "$file.part")"
    if [ "$got" != "$want" ]; then
        rm -f "$file.part"
        echo "$2: hash $got, expected $want" >&2
        exit 1
    fi
    mv "$file.part" "$file"
}

fetch "$acly/mobile_sam_image_encoder.onnx" mobile_sam_image_encoder.onnx \
    580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749
fetch "$acly/sam_mask_decoder_multi.onnx" sam_mask_decoder_multi.onnx \
    8976b90a87ba50a6a72217a5ff994f7d25ce16f2229fcc1ed259e1294c622ffe
fetch "$candle/mobile_sam-tiny-vitt.safetensors" mobile_sam-tiny-vitt.safetensors \
    5ad43a841b57cc798e2837e251c15aa9f602868a1780d195459383d117784cf7

trim=(python3 -I)
if command -v uv >/dev/null; then trim=(uv run --quiet --no-project --with onnx python -I); fi
"${trim[@]}" "$here/scripts/trim_decoder.py" \
    "$models/sam_mask_decoder_multi.onnx" "$models/sam_mask_decoder_lowres.onnx"
echo "models ready in $models"
