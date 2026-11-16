#!/usr/bin/env bash
# Decodes every IVF file given with tools/media-bench and with FFmpeg's own
# `vp9` decoder, and compares the raw samples. Prints one line a file.
# BIN overrides the media-bench binary.
set -uo pipefail
bin=${BIN:-$(git rev-parse --show-toplevel)/target/release/media-bench}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0
for f in "$@"; do
  name=$(basename "$f" .ivf)
  pix=$(ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 "$f")
  ffmpeg -loglevel error -y -c:v vp9 -i "$f" -autoscale 0 -f rawvideo -pix_fmt "$pix" "$tmp/ff.yuv" 2>/dev/null
  if ! "$bin" decode ivf "$f" --dump "$tmp/us.yuv" > /dev/null 2> "$tmp/err"; then
    echo "FAIL  $name ($pix): $(head -c 200 "$tmp/err")"; fail=1; continue
  fi
  if cmp -s "$tmp/ff.yuv" "$tmp/us.yuv"; then
    echo "ok    $name ($pix)"
  else
    echo "DIFF  $name ($pix): $(cmp "$tmp/ff.yuv" "$tmp/us.yuv" 2>&1 | head -1) sizes $(stat -c %s "$tmp/ff.yuv") $(stat -c %s "$tmp/us.yuv")"; fail=1
  fi
done
exit $fail
