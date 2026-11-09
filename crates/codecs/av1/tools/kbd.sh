#!/bin/bash
# kbd.sh in.y4m frames : BD-rate of our encoder (q 60 100 140 180) vs libaom cpu-used 6 (crf 20 30 40 50)
IN=$1; N=${2:-1}; S=/tmp/kbd$$; mkdir -p $S
HERE=$(cd "$(dirname "$0")" && pwd)
E=${ENC:-$HERE/../../../../target/release/examples/av1enc}
ps(){ ffmpeg -hide_banner -i $1 -i $IN -frames:v $N -lavfi psnr -f null - 2>&1 | grep -o "average:[0-9.]*" | cut -d: -f2; }
A=${ANCHOR_CACHE:-$S/anchor}; 
if [ ! -s "$A" ]; then for c in 20 30 40 50; do ffmpeg -v error -y -i $IN -frames:v $N -c:v libaom-av1 -crf $c -cpu-used 6 -g 999 $S/a.ivf; echo "$(stat -c %s $S/a.ivf),$(ps $S/a.ivf)"; done > $A; fi
T=""; for q in ${QS:-60 100 140 180}; do $E $IN $S/o.ivf --q $q --frames $N $EXTRA >/dev/null 2>&1; T="$T $(stat -c %s $S/o.ivf),$(ps $S/o.ivf)"; done
echo "ours:$T"; python3 $HERE/bd.py "$(cat $A)" "$T"; rm -rf $S
