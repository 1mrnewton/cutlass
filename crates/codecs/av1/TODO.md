# AV1: open work

State of the branch when work moved off the cloud machine. Numbers are in
`STATUS.md`; they were measured on a shared, loaded 4-core machine, so
re-measure locally before relying on them.

## Checks before every merge

```
cargo fmt
cargo clippy -p cutlass-av1 --all-targets -- -D warnings
cargo test -p cutlass-av1 --release      # includes the libdav1d fixture digests
AV1TEST=<dir with conf.py streams> crates/codecs/av1/tools/checkall.sh   # 46 ok
crates/codecs/av1/tools/encmatrix.sh                                     # 13 match
```

Plus every test stream through `examples/av1bench <ivf> <threads> <out.yuv>`
at 1/2/4/8 threads, byte-identical to the single-threaded libdav1d output,
and `examples/av1fuzz`. Encoder BD-rate: `tools/kbd.sh` (q 60..220 against
libaom `-cpu-used 6`). Regenerate test streams with `tools/conf.py`.

## Feature gaps

Decoder:
- Large-scale tile (`OBU_TILE_LIST`); libdav1d/FFmpeg lack it too.

Decoder speed (multi-tile frames, `STATUS.md`):
- Start a multi-tile frame while its references are still decoding
  (rows they publish, as the one-column pipeline does), instead of after.
- Shells of the tile threads: about 33 MB each at 4K, and a frame beside
  another may find none left.

Encoder tools the decoder supports but the encoder never chooses:
- Intra block copy (screen content; ts10 is ~+21% vs libaom).
- Wedge and difference-weighted compound, inter-intra.
- Loop restoration on by default (search exists in `enc/lr_search.rs`).
- Super resolution, segmentation (incl. per-segment q).
- 128x128 superblocks / partition search above 64x64.

## Encoder quality (BD-rate vs libaom cpu-used 6)

c30 -4.3%, c60 +0.7%, k1 +3.1%, ts10 +20.9%, hd20 (1080p) +23.9%;
natural content upscaled to 704x576 about +42%.
- Reference q ladder much steeper than libaom's (key q85 / alt-ref q92 vs
  libaom q112 / q106 at leaf q160); leaf frames ~2x libaom's size at SD+.
- Inter-frame partition merge check (tune `AV1_PRDI`, `AV1_PRDN`,
  `AV1_PRDA`): c30 -7..-9%, c60 -3..-4%, but +11..50% time. Needs a cheaper
  whole-block trial, or enable at speed <= 5.
- hd20: compound +-254 vectors with skip where we code single-ref residual.

## Encoder speed

Single thread about 13% behind libaom cpu-used 6 on CIF; 1080p with 4
threads about level. Profile (c30): SATD 13%, 8-tap interpolation 16%,
coefficient coding/quantize 12%, estimate DCT 10%, TPL 7%, SAD 5%.
- Keep each cached prediction's SATD in `pred_cache`.
- One worker: code directly instead of logging and replaying symbols.
- Untested BD trades: eighth-sample step only up to 32x32, cheaper TPL
  search, prune transform types by the DCT result.

## Decoder speed

bench1080 single thread 4.60G instructions vs libdav1d 2.49G (about 1.65x
the time); threaded CPU time about 1.7x libdav1d's. Remaining by size:
- Inverse transforms ~390M: 8 columns as `[i32; 8]` lanes, skip zero
  rows/columns from the eob region.
- CDEF ~750M (filter 461M, direction 160M scalar, padding 87M).
- Loop filter ~270M: vertical edges gather/transpose several rows at once.
- MC leftovers: broadcast taps once in the horizontal pass, 4-wide
  horizontal, h-only/v-only prep, `avg_copy`.
- memset 73M, find_mv_stack 120M, decode_block 139M, motion field 91M.
- Referenced-frame priority in the worker pool.

Check generated code with callgrind per-instruction dumps or `objdump`:
LLVM easily turns lane-shaped rewrites into scalar shuffles (CDEF direction
and two-row MC attempts both got slower).
