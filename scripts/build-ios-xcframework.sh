#!/usr/bin/env bash
#
# Rebuild CutlassMobileFFI.xcframework (consumed by the iOS/macOS app) from the
# `cutlass-mobile` Rust crate. Produces three slices: iOS device, iOS simulator,
# and macOS — matching the app's Package.swift binary target.
#
# iOS slices are FFmpeg-free (the engine selects native decode for target_os=ios
# automatically). The macOS slice uses the desktop FFmpeg path, so a working
# FFmpeg dev install is required on the host to build it.
#
# Requirements:
#   - Xcode command line tools (xcodebuild)
#   - rustup target add aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin
#
# Usage: scripts/build-ios-xcframework.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pkg="cutlass-mobile"
lib="libcutlass_mobile.a"
app_dir="apps/cutlass-ios-macos/CutlassMobile"
xcframework="$app_dir/CutlassMobileFFI.xcframework"
headers="$app_dir/include"

# device, simulator, macOS — the three slices Package.swift expects.
targets=(aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin)

echo "==> Building $pkg (release) for: ${targets[*]}"
for target in "${targets[@]}"; do
  echo "    - $target"
  cargo build --release -p "$pkg" --target "$target"
done

echo "==> Assembling $xcframework"
rm -rf "$xcframework"
create_args=()
for target in "${targets[@]}"; do
  create_args+=(-library "target/$target/release/$lib" -headers "$headers")
done
xcodebuild -create-xcframework "${create_args[@]}" -output "$xcframework"

echo "==> Done: $xcframework"
