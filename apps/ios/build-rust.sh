#!/bin/sh
# Build the Rust library the iOS app links, and write its Swift bindings.
#
# Run from anywhere. The static library lands in target/<triple>/<profile>.
# The Swift file goes in ios/Generated, which Xcode compiles with the app.
# The C header and its module map go in Rust/include.
#
#   apps/ios/build-rust.sh           debug, simulator and device
#   apps/ios/build-rust.sh release   release, simulator and device
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
profile=${1:-debug}
case "$profile" in
debug) flag= ;;
release) flag=--release ;;
*) echo "usage: $0 [debug|release]" >&2; exit 2 ;;
esac

cd "$root"
for triple in aarch64-apple-ios-sim aarch64-apple-ios; do
	cargo build -p cutlass-ffi-bindings --lib $flag --target "$triple"
done

# The generator reads the host library's metadata.
cargo build -p cutlass-ffi-bindings --lib
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cargo run --quiet --manifest-path tools/uniffi-bindgen/Cargo.toml -- \
	generate --library target/debug/libcutlass_ffi_bindings.dylib \
	--language swift --out-dir "$out"

mkdir -p "$here/ios/Generated" "$here/Rust/include"
cp "$out/cutlass_ffi_bindings.swift" "$here/ios/Generated/"
cp "$out/cutlass_ffi_bindingsFFI.h" "$here/Rust/include/"
cp "$out/cutlass_ffi_bindingsFFI.modulemap" "$here/Rust/include/module.modulemap"
