#!/usr/bin/env bash
# Times the Rust Matroska demuxer and muxer against libavformat.
#
#     crates/containers/matroska/bench/against_ffmpeg.sh <work folder> [runs]
#
# Makes three files in the work folder with ffmpeg (H.264 + AAC in MKV,
# VP9 + Opus in WebM, FLAC in MKA), builds the `bench` example in release
# and `ffmpeg_matroska.c` against the FFmpeg pkg-config finds, then runs
# each job on each file, one side after the other: one warm-up each, then
# `runs` timed runs each (5 by default). Each run is a whole process under
# `/usr/bin/time -l`, start-up and file reading included; the table gives
# the median of instructions retired, CPU time and peak memory. The `open`
# row is start-up and the header alone; "past open" takes its median off
# each other row's, which leaves the work of the job itself. The two
# sides must print the same packet count and bytes for `read` before the
# times are kept. Both sides run on one thread.
set -euo pipefail

work=${1:?usage: against_ffmpeg.sh <work folder> [runs]}
runs=${2:-5}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
mkdir -p "$work"
cd "$work"

[ -f h264_aac.mkv ] || ffmpeg -v error -y -f lavfi -i testsrc2=size=1920x1080:rate=30 \
    -f lavfi -i sine=frequency=440:sample_rate=48000 -t 120 -c:v libx264 -preset veryfast \
    -b:v 6M -g 60 -bf 2 -c:a aac -b:a 128k -ac 2 h264_aac.mkv
[ -f vp9_opus.webm ] || ffmpeg -v error -y -f lavfi -i testsrc2=size=640x360:rate=30 \
    -f lavfi -i sine=frequency=440:sample_rate=48000 -t 60 -c:v libvpx-vp9 -deadline realtime \
    -cpu-used 8 -b:v 1M -g 60 -c:a libopus -b:a 96k vp9_opus.webm
[ -f flac.mka ] || ffmpeg -v error -y -f lavfi -i sine=frequency=440:sample_rate=48000 -t 600 \
    -ac 2 -c:a flac flac.mka

(cd "$root" && cargo build --quiet --release -p cutlass-matroska --example bench)
ours="$root/target/release/examples/bench"
# shellcheck disable=SC2046
cc -O2 -o ffmpeg_matroska "$here/ffmpeg_matroska.c" $(pkg-config --cflags --libs libavformat libavutil)
theirs="$work/ffmpeg_matroska"

# One run: prints "instructions cpu_seconds peak_bytes" and leaves the
# command's own output in last.txt.
one() {
    /usr/bin/time -l "$@" > last.txt 2> time.txt
    awk '/instructions retired/ {i=$1} / real / {c=$3+$5} /maximum resident/ {m=$1}
         END {print i, c, m}' time.txt
}

# The median of `runs` timed runs after one warm-up.
median() {
    one "$@" > /dev/null
    for _ in $(seq "$runs"); do one "$@"; done | sort -n | awk -v n="$runs" '
        {i[NR]=$1; c[NR]=$2; m[NR]=$3} END {k=int((n+1)/2); print i[k], c[k], m[k]}'
}

row() {
    local job=$1 file=$2
    shift 2
    local mine other
    case $job in
        open)  mine=("$ours" open "$file"); other=("$theirs" open "$file") ;;
        read)  mine=("$ours" read "$file"); other=("$theirs" read "$file") ;;
        seek)  mine=("$ours" seek "$file" 200); other=("$theirs" seek "$file" 200) ;;
        remux) mine=("$ours" remux "$file" "ours.${file##*.}")
               other=("$theirs" remux "$file" "theirs.${file##*.}") ;;
    esac
    "${mine[@]}" > mine.txt
    "${other[@]}" > other.txt
    if [ "$job" = read ] && ! diff -q mine.txt other.txt > /dev/null; then
        echo "$file: the two sides read different packets:" >&2
        cat mine.txt other.txt >&2
        exit 1
    fi
    read -r mi mc mm <<< "$(median "${mine[@]}")"
    read -r ti tc tm <<< "$(median "${other[@]}")"
    if [ "$job" = open ]; then
        open_mine=$mi
        open_theirs=$ti
    fi
    awk -v f="$file" -v j="$job" -v mi="$mi" -v ti="$ti" -v mc="$mc" -v tc="$tc" \
        -v mm="$mm" -v tm="$tm" -v om="$open_mine" -v ot="$open_theirs" 'BEGIN {
        past = j == "open" ? "" : sprintf("%.1f M / %.1f M (%.2f)", (mi-om)/1e6, (ti-ot)/1e6,
            (mi-om)/(ti-ot))
        printf "| %s | %s | %.1f M | %.1f M | %.2f | %s | %.2f / %.2f s | %.1f / %.1f MB |\n",
            f, j, mi/1e6, ti/1e6, mi/ti, past, mc, tc, mm/1048576, tm/1048576 }'
}

echo "| File | Job | Rust instructions | libavformat instructions | Ratio | Past open: Rust / lavf (ratio) | CPU (Rust / lavf) | Peak memory (Rust / lavf) |"
echo "|---|---|---:|---:|---:|---:|---:|---:|"
for file in h264_aac.mkv vp9_opus.webm flac.mka; do
    for job in open read seek remux; do
        row "$job" "$file"
    done
done
