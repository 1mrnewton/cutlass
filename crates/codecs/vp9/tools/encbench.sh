#!/bin/sh
# Rate, quality and speed of our encoder against libvpx-vp9 on one raw clip.
#   encbench.sh <in.yuv> <w> <h> <frames> ours <q>... | vpx <deadline> <cpu-used> <kbps>...
# Prints: label  bytes  kb/s  PSNR-Y  PSNR-avg  seconds
set -e
in=$1 w=$2 h=$3 n=$4 who=$5; shift 5
tmp=${TMPDIR:-/tmp}/encbench.$$; mkdir -p "$tmp"
enc=${VP9ENC:-target/release/examples/vp9enc}
psnr() {
	ffmpeg -hide_banner -loglevel error -i "$1" -frames:v "$n" -f rawvideo -pix_fmt yuv420p -y "$tmp/d.yuv"
	ffmpeg -hide_banner -s "${w}x$h" -pix_fmt yuv420p -f rawvideo -i "$tmp/d.yuv" \
		-s "${w}x$h" -pix_fmt yuv420p -f rawvideo -i "$in" -frames:v "$n" -lavfi psnr -f null - 2>&1 |
		sed -n 's/.*PSNR y:\([0-9.]*\).*average:\([0-9.]*\).*/\1 \2/p'
}
if [ "$who" = ours ]; then
	for q in "$@"; do
		t0=$(date +%s.%N)
		$enc "$in" "$w" "$h" "$tmp/o.ivf" q="$q" frames="$n" ${SPEED:+speed=$SPEED} threads=${THREADS:-1} ${LAG:+lag=$LAG} > /dev/null
		t1=$(date +%s.%N)
		b=$(($(stat -c %s "$tmp/o.ivf") - 32 - 12 * n))
		echo "ours-q$q $b $((b * 8 * 30 / n / 1000)) $(psnr "$tmp/o.ivf") $(echo "$t1 - $t0" | bc)"
	done
else
	dl=$1 cpu=$2; shift 2
	for k in "$@"; do
		t0=$(date +%s.%N)
		ffmpeg -hide_banner -loglevel error -s "${w}x$h" -pix_fmt yuv420p -f rawvideo -r 30 -i "$in" -frames:v "$n" \
			-c:v libvpx-vp9 -b:v "${k}k" -deadline "$dl" -cpu-used "$cpu" -threads ${THREADS:-1} -row-mt $([ "${THREADS:-1}" -gt 1 ] && echo 1 || echo 0) -tile-columns $([ "${THREADS:-1}" -gt 1 ] && echo 2 || echo 0) -y "$tmp/v.ivf"
		t1=$(date +%s.%N)
		b=$(($(stat -c %s "$tmp/v.ivf") - 32 - 12 * n))
		echo "vpx-$dl$cpu-${k}k $b $((b * 8 * 30 / n / 1000)) $(psnr "$tmp/v.ivf") $(echo "$t1 - $t0" | bc)"
	done
fi
rm -rf "$tmp"
