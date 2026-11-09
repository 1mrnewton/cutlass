#!/bin/bash
# Rate-distortion and speed comparison. Usage: rdcmp.sh <src.y4m> <label>
SRC=$1; L=$2; D=${TMPDIR:-/tmp}/rd/$L; mkdir -p $D
HERE=$(cd "$(dirname "$0")" && pwd)
ENC=${ENC:-$HERE/../../../../target/release/examples/av1enc}
N=$(grep -c FRAME $SRC 2>/dev/null || echo 0)
psnr() { ffmpeg -hide_banner -i $1 -i $SRC -lavfi psnr -f null - 2>&1 | grep -o "average:[0-9.]*" | cut -d: -f2; }
kbps() { local b=$(stat -c %s $1); echo "scale=1; $b*8/1000/($N/30)" | bc; }
t0() { date +%s.%N; }
for q in 60 100 140 180; do
  s=$(t0); $ENC $SRC $D/ours_$q.ivf --q $q --speed ${SPEED:-6} 2>/dev/null; e=$(t0)
  echo "ours q=$q $(kbps $D/ours_$q.ivf) kbps $(psnr $D/ours_$q.ivf) dB $(echo "scale=2; $N/($e-$s)" | bc) fps"
done
for crf in 25 33 41 49; do
  s=$(t0); ffmpeg -hide_banner -loglevel error -threads 1 -i $SRC -c:v libaom-av1 -cpu-used ${AOMSPEED:-6} -crf $crf -b:v 0 -row-mt 0 -tiles 1x1 $D/aom_$crf.ivf -y; e=$(t0)
  echo "aom crf=$crf $(kbps $D/aom_$crf.ivf) kbps $(psnr $D/aom_$crf.ivf) dB $(echo "scale=2; $N/($e-$s)" | bc) fps"
done
for crf in 25 33 41 49; do
  s=$(t0); ffmpeg -hide_banner -loglevel error -i $SRC -c:v libsvtav1 -preset ${SVTPRESET:-8} -crf $crf -svtav1-params lp=1 $D/svt_$crf.ivf -y 2>/dev/null; e=$(t0)
  echo "svt crf=$crf $(kbps $D/svt_$crf.ivf) kbps $(psnr $D/svt_$crf.ivf) dB $(echo "scale=2; $N/($e-$s)" | bc) fps"
done
for qp in 60 100 140 180; do
  s=$(t0); ffmpeg -hide_banner -loglevel error -i $SRC -c:v librav1e -speed ${RAV1ESPEED:-8} -qp $qp -rav1e-params threads=1 $D/rav1e_$qp.ivf -y; e=$(t0)
  echo "rav1e qp=$qp $(kbps $D/rav1e_$qp.ivf) kbps $(psnr $D/rav1e_$qp.ivf) dB $(echo "scale=2; $N/($e-$s)" | bc) fps"
done
