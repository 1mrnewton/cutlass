#!/usr/bin/env bash
# Writes a set of VP9 streams with FFmpeg's libvpx-vp9 into $1 (IVF), each
# exercising one part of the format, for checking a decoder frame by frame
# against FFmpeg's own `vp9` decoder (see tools/check.sh).
set -euo pipefail
out=${1:?folder}
mkdir -p "$out"
f() { # name, size, frames, source, extra args...
  local name=$1 size=$2 n=$3 src=$4; shift 4
  [ -s "$out/$name.ivf" ] && return 0
  case $src in *=*) ;; *) src="$src=size=$size:rate=30" ;; esac
  ffmpeg -loglevel error -y -f lavfi -i "$src" -frames:v "$n" \
    -c:v libvpx-vp9 -pix_fmt yuv420p "$@" "$out/$name.ivf"
}
f basic      352x288 30 mandelbrot -b:v 400k -deadline good -cpu-used 2
f altref     352x288 40 mandelbrot -b:v 300k -auto-alt-ref 1 -lag-in-frames 25 -deadline good -cpu-used 1
f odd        99x77   20 testsrc2   -b:v 200k -cpu-used 4
f odd2       8x8     5  testsrc2   -b:v 20k -vf scale=2:2
f odd3       17x33   10 mandelbrot -b:v 50k
f tiles      1280x720 12 mandelbrot -b:v 2M -tile-columns 2 -tile-rows 2 -cpu-used 4 -row-mt 1
f lossless   176x144 10 testsrc2   -lossless 1
f hiq        352x288 12 testsrc2   -crf 4 -b:v 0 -cpu-used 1
f loq        352x288 20 mandelbrot -crf 60 -b:v 0
f errres     352x288 20 mandelbrot -b:v 300k -error-resilient 1
f fparallel  352x288 20 mandelbrot -b:v 300k -frame-parallel 1
f rt         640x360 30 mandelbrot -b:v 500k -deadline realtime -cpu-used 8
f aq3        352x288 30 mandelbrot -b:v 300k -aq-mode 3 -deadline realtime -cpu-used 6
f aq1        352x288 20 testsrc2   -b:v 300k -aq-mode 1
f aq2        352x288 20 mandelbrot -b:v 300k -aq-mode 2
f sharp      352x288 10 mandelbrot -b:v 300k -sharpness 5
f p1_444     176x144 12 mandelbrot -pix_fmt yuv444p -b:v 300k
f p1_422     176x144 12 mandelbrot -pix_fmt yuv422p -b:v 300k
f p1_440     176x144 12 mandelbrot -pix_fmt yuv440p -b:v 300k
f p2_10      176x144 12 mandelbrot -pix_fmt yuv420p10le -b:v 300k
f p2_12      176x144 12 mandelbrot -pix_fmt yuv420p12le -b:v 300k
f p3_444_10  176x144 12 mandelbrot -pix_fmt yuv444p10le -b:v 300k
f p3_422_12  176x144 12 mandelbrot -pix_fmt yuv422p12le -b:v 300k
f p2_lossless 64x64   6 testsrc2  -pix_fmt yuv420p10le -lossless 1
f gbr        176x144 8  mandelbrot -pix_fmt gbrp -b:v 300k
f gbr10      176x144 8  mandelbrot -pix_fmt gbrp10le -b:v 300k
f noise      352x288 10 "testsrc2=size=352x288:rate=30,noise=alls=30:allf=t" -b:v 2M
f noise1080  1920x1080 6 "testsrc2=size=1920x1080:rate=30,noise=alls=20:allf=t" -crf 20 -b:v 0 -cpu-used 4 -row-mt 1
