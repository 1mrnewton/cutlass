# ml-bench

Times one AI model on two Rust engines, candle and burn, so Cutlass can pick
one. The model is MobileSAM: you click a point in a frame and it returns a
mask of the object under the click. It runs in two steps:

- **encode**: reads the whole frame once. This is the slow part.
- **decode**: turns one click into masks. This runs on every click.

Both engines run the same MobileSAM weights. candle runs the model code that
ships in `candle-transformers`. burn runs Rust code that `burn-onnx` writes
from the ONNX files at build time. The bench checks that both give the same
masks as onnxruntime, then times them.

This crate is its own Cargo workspace. burn and its GPU stack stay out of the
editor's build, its lockfile, and `cargo test --workspace`.

## Get the models

```bash
./scripts/fetch.sh
```

This downloads about 85 MB into `models/` and checks each file's hash. It
then removes the decoder's last step, which resizes masks to the frame's size,
because candle's decoder does not do that step. Removing it needs Python with
the `onnx` package, or `uv`.

## Run it

```bash
cargo run --release -- --image frame.png --click 1010,650
cargo run --release --features metal -- --gpu --image frame.png --click 1010,650
```

| Flag | Meaning |
| --- | --- |
| `--image PATH --click X,Y` | A PNG or JPEG, and the click in its own pixels |
| `--reference PATH` | A frame, click and answer from `scripts/reference.py` (see below) |
| `--engine candle\|burn\|both` | Default `both` |
| `--gpu` | Use the GPU. Without it, both run on the CPU |
| `--runs N` / `--clicks N` | How many encodes and decodes to time after the first. Default 10 and 30 |
| `--models DIR` / `--out DIR` | Default `models` and `out` |

The report is printed and also written to `out/report.json`. Each engine's
best mask is saved to `out/<engine>_mask.png` so you can look at it.

Cargo features:

| Feature | What it turns on |
| --- | --- |
| `metal` | candle on Metal, and burn on wgpu over Metal (Apple devices) |
| `vulkan` | burn on wgpu over Vulkan (Android, Windows, Linux). It needs Vulkan 1.2 features. candle has no Vulkan backend, so it stays on the CPU |
| `wgpu` | burn on wgpu's portable path (WGSL), for GPUs the `vulkan` path refuses. This build stores the model's shape numbers as 32-bit integers, because many phone GPUs can't run 64-bit ones |
| `ios` | candle on Metal, and burn on wgpu's portable path with the Metal API pinned (iPhone, iPad) |
| `cache` | burn saves its compiled GPU kernels to disk, so later launches skip most of the warm-up. **It builds C code**; see the findings |

To check the masks against onnxruntime, make a reference file, then pass it
with `--reference`:

```bash
python -I scripts/reference.py models frame.png 1010 650 out/ref.safetensors
cargo run --release -- --reference out/ref.safetensors
```

### On an Android phone

You need the Android NDK and `cargo-ndk`.

```bash
cargo ndk -t arm64-v8a --platform 26 build --release --features vulkan
```

```bash
adb push target/aarch64-linux-android/release/ml-bench /data/local/tmp/
```

```bash
adb push models/mobile_sam-tiny-vitt.safetensors /data/local/tmp/models/
```

```bash
adb push frame.png /data/local/tmp/
```

On the CPU, both engines:

```bash
adb shell "cd /data/local/tmp && ./ml-bench --image frame.png --click 500,900 --runs 5 --clicks 20"
```

On the GPU (burn only). Build with `--features wgpu` instead if the
phone's driver lacks Vulkan 1.2:

```bash
adb shell "cd /data/local/tmp && ./ml-bench --gpu --engine burn --image frame.png --click 500,900 --runs 5 --clicks 20"
```

burn's weights are built into the binary. candle reads its weights from
`models/`.

### On an iPhone

You need Xcode, `xcodegen`, and the Rust `aarch64-apple-ios` target. Build
the library, then the app:

```bash
cargo build --release --target aarch64-apple-ios --features ios --lib
```

```bash
cd ios && xcodegen && xcodebuild -project MLBench.xcodeproj -scheme MLBench -configuration Release -destination 'generic/platform=iOS' -derivedDataPath build -allowProvisioningUpdates build
```

The app copies the candle weights and the two reference files from `models/`
into its bundle; make the references first with `scripts/reference.py`, named
`reference_landscape.safetensors` and `reference_portrait.safetensors`. On
launch it runs both engines on the GPU, then on the CPU, prints each report to
the console, writes them to the app's `Documents/`, and shows "Done." Set
`ML_BENCH_ONLY` (for example `gpu_landscape`) and `ML_BENCH_ENGINE` in the
launch environment to run one part.

### On Windows

Install Rust with the MSVC toolchain, and run `scripts/fetch.sh` from Git
Bash. Then:

```bash
cargo run --release -- --image frame.png --click 1010,650
```

```bash
cargo run --release --features vulkan -- --gpu --engine burn --image frame.png --click 1010,650
```

## Findings on an M5 Pro Mac (macOS, Rust 1.99.1)

Two 1080p frames from `../local-files`, one landscape and one portrait, one
click each. Times are the median of 10 encodes and 30 clicks, after a first
warm-up run. The two frames gave the same times to within 10%.

| Engine | Encode a frame | One click | First run of the process |
| --- | --- | --- | --- |
| candle, CPU | 1,050 ms | 34 ms | about 1.1 s |
| burn, CPU (flex) | 1,750 ms | 80 ms | about 1.8 s |
| candle, Metal | 452 ms | 19 ms | up to 8 s |
| burn, wgpu on Metal | **48 ms** | **5.5 ms** | **about 35 s every launch**; about 4 s with `cache` |

Masks:

- burn matched onnxruntime to within 0.000004, and its masks were the same
  pixels.
- candle drifted up to 0.0016 in the frame encoding. Its masks overlapped
  onnxruntime's by 99.7% or more, and it picked the same best mask both
  times.

What this shows:

1. **burn imported both ONNX files with no changes.** That includes the
   decoder's prompt handling, which has some awkward steps. The CPU numbers
   don't depend on that conversion, and the GPU numbers do.
2. **On the GPU, burn is about 10 times faster** than candle on Metal.
   candle's Metal backend is only about twice as fast as its own CPU.
3. **On the CPU, candle is faster**: 1.7 times for encode and 2.3 times per
   click.
4. **burn's warm-up is the catch.** Each launch compiles its GPU kernels.
   Lowering the autotune level (`CUBECL_AUTOTUNE_LEVEL=minimal`) still took
   28 s, and turning fusion off still took 29 s. Only the `cache` feature
   brings it down to about 4 s, and it needs `turso`, which builds C code
   (`aegis`, `simsimd`) that the repo's `deny.toml` does not allow. Without
   `cache`, burn builds no C code. The cache writes to `target/environment/`
   here; in an app it would be pointed at Cutlass's `Cache/` folder.
5. **burn's generated code `unwrap`s and `panic!`s** on bad weights and on
   shapes it does not expect. The editor's own crates may not do that.
6. Not measured yet: a Windows laptop with integrated graphics. The Windows
   build compiles from a Mac without `cache`. With `cache`, its build script
   needs a Windows resource compiler.

## Findings on a Galaxy Note10+ (Snapdragon 855, Adreno 640, Android 12)

A 2019 flagship, about as fast as a mid-range phone today. The same two
frames and clicks as on the Mac. Times are the median of 3 to 5 encodes and
10 to 20 clicks.

| Engine | Encode a frame | One click |
| --- | --- | --- |
| candle, CPU | 15.5 s to 24.8 s (it varied between runs) | 0.80 to 0.85 s |
| burn, CPU (flex) | **11.3 s** (steady) | **0.52 s** |
| burn, GPU | did not run | did not run |

Both engines' masks matched onnxruntime here as well. burn's matched
exactly, and candle's overlapped by 99.7% or more and picked the same best
mask.

What this shows:

1. **On this phone's CPU, burn beats candle**: about 1.4 to 2.2 times faster
   to encode, and 1.6 times faster per click. That is the reverse of the Mac.
2. **Neither is fast enough on the CPU** for a click-to-mask preview: about
   11 seconds per frame before the first mask.
3. **burn could not use this phone's GPU.** It hit three failures:
   - The `vulkan` path aborts the process. The driver lacks
     `buffer_device_address`, a Vulkan 1.2 feature, and burn panics inside a
     function that cannot unwind instead of returning an error.
   - The `wgpu` path then refused the generated code's 64-bit shape integers
     (no `SHADER_INT64`). The `wgpu` build narrows them to 32 bits to get past
     this.
   - Even then, Qualcomm's shader compiler failed on one of burn's kernels
     (`AdrenoVK: Failed to link shaders`, an internal assertion), and the
     device was lost.

   The last failure is a driver bug, so this phone has no GPU route for burn
   without changes inside burn. Newer Adreno drivers may behave differently;
   this was not tested.
4. candle has no Android GPU backend at all, so on Android it is CPU only
   either way.

## Findings on an iPhone 16 Pro Max (A18 Pro, iOS)

The same landscape frame and click as on the Mac. Times are the median of 10
encodes and 30 clicks.

| Engine | Encode a frame | One click | Masks |
| --- | --- | --- | --- |
| candle, CPU | 1,653 ms | 70 ms | correct |
| burn, CPU (flex) | 2,547 ms | 106 ms | correct |
| candle, Metal | 2,021 ms | 100 ms | correct |
| burn, wgpu (WGSL) on Metal | **878 ms** | **35 ms** | **wrong** |

What this shows:

1. **The CPU works for both**, at about 1.7 to 2.5 s a frame. candle is
   faster here, as on the Mac.
2. **candle's Metal backend is slower than its own CPU** on this phone.
3. **burn's GPU path is fast but gives wrong masks.** burn's Metal compiler
   only runs on macOS, and its automatic mode looks for Vulkan on iOS, so the
   `ios` feature uses the portable WGSL compiler with the Metal API pinned.
   On that path the frame encoding matches onnxruntime (to 0.000003), but the
   decoder's masks overlap onnxruntime's by under 6%, and its scores are
   about 0.35 where they should be about 0.98. Keeping the shape integers in
   32 bits did not fix it. The bug is in burn's WGSL kernels for something
   in the decoder; it was not narrowed further.
4. **burn's warm-up is about 28 s for the first encode and 13 to 29 s for the
   first click**, on every launch, without `cache`.

## Licenses

- MobileSAM's code and weights are Apache-2.0.
- The ONNX files from `Acly/MobileSAM` are MIT.
- `lmz/candle-sam` converts the same MobileSAM weights to safetensors. It
  states no license of its own.
