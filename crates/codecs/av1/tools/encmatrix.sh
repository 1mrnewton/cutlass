#!/bin/bash
# Encode test clips in many formats; check libdav1d decodes to our reconstruction; print PSNR/size/fps.
HERE=$(cd "$(dirname "$0")" && pwd)
ENC=${ENC:-$HERE/../../../../target/release/examples/av1enc}
D=${TMPDIR:-/tmp}/encm; mkdir -p $D
run() { # name pix size frames encargs
  local name=$1 pix=$2 size=$3 n=$4; shift 4
  ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=$size:rate=30 -frames:v $n -pix_fmt $pix -strict -1 $D/$name.y4m -y
  local info; info=$($ENC $D/$name.y4m $D/$name.ivf --recon $D/$name.rec "$@" 2>&1 | tail -1)
  ffmpeg -hide_banner -loglevel error -c:v libdav1d -i $D/$name.ivf -f rawvideo -pix_fmt $pix -strict -1 $D/$name.dav -y 2>/dev/null
  local m="MISMATCH"; cmp -s $D/$name.rec $D/$name.dav && m="match"
  local p; p=$(ffmpeg -hide_banner -i $D/$name.ivf -i $D/$name.y4m -lavfi psnr -f null - 2>&1 | grep -o "average:[0-9.inf]*")
  printf "%-22s %-8s %s | %s\n" "$name" "$m" "$p" "$info"
}
run e420 yuv420p 352x288 10 --q 100
run e420_q30 yuv420p 352x288 6 --q 30
run e420_q220 yuv420p 352x288 6 --q 220
run e420_10b yuv420p10le 352x288 8 --q 100
run e420_12b yuv420p12le 352x288 6 --q 100
run e444 yuv444p 352x288 6 --q 100
run e422_10b yuv422p10le 352x288 6 --q 100
run egray gray 352x288 6 --q 100
run eodd yuv420p 97x61 6 --q 100
run etiles yuv420p 640x480 6 --q 100 --tiles 1x1
run ekbps yuv420p 352x288 30 --kbps 300
run efast yuv420p 352x288 10 --q 100 --speed 10
run eslow yuv420p 352x288 10 --q 100 --speed 0
