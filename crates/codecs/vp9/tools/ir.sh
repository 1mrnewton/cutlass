#!/usr/bin/env bash
# Instructions (in millions) media-bench executes decoding an IVF file on
# one thread, from cachegrind: a measure that does not move with the
# machine's load. Usage: ir.sh <file.ivf> [media-bench binary]
bin=${2:-$(git rev-parse --show-toplevel)/target/release/media-bench}
out=$(mktemp)
taskset -c 0 valgrind --tool=cachegrind --cache-sim=no --cachegrind-out-file=/dev/null "$bin" decode ivf "$1" 2> "$out" > /dev/null
grep -E "I *refs" "$out" | awk '{gsub(",","",$NF); printf "%.1f M instructions\n", $NF/1e6}'
rm -f "$out"
