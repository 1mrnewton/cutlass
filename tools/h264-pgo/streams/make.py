"""Writes the H.264 streams `h264-pgo` trains on, next to this script.

    python3 tools/h264-pgo/streams/make.py

Each stream is one of FFmpeg's synthetic sources (`testsrc2`, with and
without noise, and `mandelbrot`) encoded by libx264, so no footage is in
the repository. libx264 is used, not this repository's
encoder, because the profile must weigh the syntax that real files use:
B-pyramids, direct and weighted prediction, 8x8 transforms, several
references. Trained on our own encoder's streams instead, a profile won
about half as much on real films. The files are written once and committed;
the release build only reads them, so CI needs no FFmpeg.
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

# (file, source, frames, libx264 options). 1080p is above the size from
# which the decoder prefetches motion, as 4K films do; Baseline keeps CAVLC
# and High 10 the deeper samples from being treated as never run. Noise
# gives the residual and motion search of camera footage; the Mandelbrot
# zoom, motion across the whole picture.
STREAMS = [
    ("high_1080p.h264", "testsrc2=size=1920x1080:rate=30", 60, ["-preset", "medium"]),
    ("high_720p.h264", "testsrc2=size=1280x720:rate=30", 60, ["-preset", "medium"]),
    (
        "noisy_720p.h264",
        "testsrc2=size=1280x720:rate=30,noise=alls=10:allf=t+u",
        30,
        ["-preset", "medium"],
    ),
    ("zoom_720p.h264", "mandelbrot=size=1280x720:rate=30", 30, ["-preset", "medium"]),
    ("baseline_360p.h264", "testsrc2=size=640x360:rate=30", 30, ["-profile:v", "baseline"]),
    (
        "high10_360p.h264",
        "testsrc2=size=640x360:rate=30",
        30,
        ["-profile:v", "high10", "-pix_fmt", "yuv420p10le"],
    ),
]


def main() -> int:
    for name, source, frames, options in STREAMS:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", source,
                "-frames:v", str(frames), "-an", "-c:v", "libx264", *options,
                "-f", "h264", str(HERE / name),
            ],
            check=True,
        )
        print(f"{name}: {(HERE / name).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
