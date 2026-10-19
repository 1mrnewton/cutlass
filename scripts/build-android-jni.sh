#!/usr/bin/env bash
#
# Rebuild libcutlass_mobile.so (the JNI engine the Android app loads) from the
# `cutlass-mobile` Rust crate, into apps/cutlass-android/app/src/main/jniLibs/.
# The build is FFmpeg-free: the engine selects native MediaCodec decode for
# target_os=android automatically.
#
# Requirements:
#   - cargo install cargo-ndk
#   - rustup target add aarch64-linux-android x86_64-linux-android
#   - Android NDK (auto-detected under $ANDROID_HOME/ndk, or set ANDROID_NDK_HOME)
#
# Usage: scripts/build-android-jni.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# cargo-ndk needs ANDROID_NDK_HOME; fall back to the newest installed NDK whose
# major version cargo-ndk 4.x understands (r29 and earlier; r30+ canaries panic).
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
  if [[ -d "$sdk/ndk" ]]; then
    for ndk in $(ls -d "$sdk"/ndk/* | sort -Vr); do
      major="$(basename "$ndk" | cut -d. -f1)"
      if [[ "$major" -le 29 ]]; then
        ANDROID_NDK_HOME="$ndk"
        export ANDROID_NDK_HOME
        break
      fi
    done
  fi
fi
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  echo "error: ANDROID_NDK_HOME not set and no NDK found under \$ANDROID_HOME/ndk" >&2
  exit 1
fi
echo "==> Using NDK: $ANDROID_NDK_HOME"

out="apps/cutlass-android/app/src/main/jniLibs"
echo "==> Building cutlass-mobile (.so) for arm64-v8a, x86_64 -> $out"
# -P 26 matches the ndk crate's api-level-26 feature (AMediaExtractor/AMediaCodec).
cargo ndk -t arm64-v8a -t x86_64 -P 26 -o "$out" build -p cutlass-mobile --release

echo "==> Done. Libraries:"
find "$out" -name 'libcutlass_mobile.so'
