#!/usr/bin/env bash
# Downloads the subject detectors tracking starts from, as OpenCV Zoo
# publishes them, checks their hashes, converts them into the vision
# crate's burn code and .bpk weights, and writes the reference its detector
# tests check burn against:
#
#   YuNet face detector, 2023mar (MIT)
#   MediaPipe person detector, 2023mar (Apache-2.0)
#   MediaPipe palm detector, 2023feb (Apache-2.0)
#
# Everything lands in local-files/models/detect beside the repository, and
# the .bpk files are also copied into the desktop app's Models/detect
# folder, where a release build looks for them. Set CUTLASS_MODELS to put
# them in another Models folder, or to an empty string to skip the copy.
#
# The reference frames come from stock clips in local-files; without them
# the reference is skipped and the tests that need it pass by skipping.
#
# Needs curl, cargo, ffmpeg and uv (OpenCV and onnxruntime run in a
# throwaway environment).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
local_files="$repo/../local-files"
out="$local_files/models/detect"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out"

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# Repository, revision, file, hash.
fetch() {
    curl -fsSL --retry 3 -o "$work/$3" "https://huggingface.co/opencv/$1/resolve/$2/$3"
    local got
    got="$(sha256 "$work/$3")"
    if [ "$got" != "$4" ]; then
        echo "$3: hash $got, expected $4" >&2
        exit 1
    fi
}

fetch face_detection_yunet 3cc26e7f1014a5ee5d74a42acee58bafc9d0a310 \
    face_detection_yunet_2023mar.onnx 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
fetch person_detection_mediapipe 531b6da87204aa40aa5cc680b4d032dd09b5055e \
    person_detection_mediapipe_2023mar.onnx 47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f
fetch palm_detection_mediapipe 233e619dcea1759bf6de707b9b904fe30881ea55 \
    palm_detection_mediapipe_2023feb.onnx 78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c

(cd "$here" && cargo run --release -q -- "$work" "$out" \
    face_detection_yunet_2023mar person_detection_mediapipe_2023mar palm_detection_mediapipe_2023feb)

# A face on green, a dancer whose hands hold her headphones, three
# children, and a man walking away down a pier: frame, second, width,
# clip. The tests name places in these frames at these sizes.
frames=()
take() {
    local clip="$local_files/$4"
    if [ -f "$clip" ]; then
        ffmpeg -v error -ss "$2" -i "$clip" -frames:v 1 -vf "scale=$3:-2" -y "$work/$1.png"
        frames+=("$work/$1.png")
    fi
}
take face 2 1280 9489379-uhd_4096_2160_25fps.mp4
take body 2 1280 6905095-hd_1920_1080_25fps.mp4
take party 2 720 7150172-hd_720_1280_50fps.mp4
take walk 1 1920 16929689_1920_1080_30fps.mp4
if [ "${#frames[@]}" -eq 4 ]; then
    (cd "$work" && uv run --no-project --with opencv-python==4.10.0.84 --with onnxruntime \
        --with numpy --with safetensors \
        python -I "$here/scripts/detect_reference.py" "$work" "$out/reference_detect.safetensors" "${frames[@]}")
else
    echo "skipped the reference: the stock clips are not in $local_files" >&2
fi

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
    mkdir -p "$models/detect"
    for weights in face_detection_yunet_2023mar person_detection_mediapipe_2023mar palm_detection_mediapipe_2023feb; do
        cp "$out/$weights.bpk" "$models/detect/$weights.bpk.part"
        mv "$models/detect/$weights.bpk.part" "$models/detect/$weights.bpk"
    done
    echo "installed the detectors in $models/detect"
fi
