#!/bin/sh
# Build the Rust library the Android app loads, and write its Kotlin bindings.
#
# Run from anywhere. The shared libraries land in app/src/main/jniLibs, one
# folder per ABI, which Gradle packs into the APK; they are not committed.
# The Kotlin file goes in app/src/main/java/uniffi, which is.
#
# Both Gradle build types load the release library: an unoptimized one takes
# seconds to draw a movie frame. Pass debug to step through Rust.
#
#   apps/android/build-rust.sh           release, arm64-v8a and x86_64
#   apps/android/build-rust.sh debug     debug, arm64-v8a and x86_64
#
# Needs cargo-ndk and an NDK of r28 or later, whose linker lays the library
# out on 16 KB pages, as Android 15 phones with 16 KB pages require.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
profile=${1:-release}
case "$profile" in
debug) flag= ;;
release) flag=--release ;;
*) echo "usage: $0 [debug|release]" >&2; exit 2 ;;
esac

cd "$root"
cargo ndk -t arm64-v8a -t x86_64 -P 28 -o "$here/app/src/main/jniLibs" \
	build -p cutlass-ffi-bindings --lib $flag

# The generator reads the host library's metadata: a dylib on a Mac, a
# shared object on Linux, as in CI.
cargo build -p cutlass-ffi-bindings --lib
case "$(uname -s)" in
Darwin) host=target/debug/libcutlass_ffi_bindings.dylib ;;
*) host=target/debug/libcutlass_ffi_bindings.so ;;
esac
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cargo run --quiet --manifest-path tools/uniffi-bindgen/Cargo.toml -- \
	generate --library "$host" \
	--language kotlin --out-dir "$out"

rm -rf "$here/app/src/main/java/uniffi"
mkdir -p "$here/app/src/main/java"
cp -R "$out/uniffi" "$here/app/src/main/java/"
