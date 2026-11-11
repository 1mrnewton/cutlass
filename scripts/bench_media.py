#!/usr/bin/env python3
"""Time the Rust codecs and containers against FFmpeg on the same input.

    cargo build --release -p media-bench
    python3 scripts/bench_media.py --video clip.mp4 --audio song.m4a --work /tmp/bench
    python3 scripts/bench_media.py --work /tmp/bench --report

There are three kinds of case.

Decode. FFmpeg writes a file from the source clip, so both decoders are
handed bytes neither of them wrote. Both decode it once to raw samples on
disk and the two dumps are compared; then both decode it again to nothing
and are timed. Where the Rust decoder returns its pictures in another layout
than FFmpeg's decoder (RGB from a JPEG, say), FFmpeg converts each picture
to that layout too, so both sides end with the same kind of pixels.

Demux. Both read every packet of a file and decode none.

Encode. Both encode the same raw pictures or raw sound at the same bitrate
and write the result to disk. For pictures, each result is decoded again and
compared with the source, so the table can say what each encoder kept.

How a run is timed, the same way on both sides: as a whole process, start-up
and file reading included, under `/usr/bin/time -l`, which reports wall
time, CPU time, peak memory and instructions retired. The two sides run one
after the other, `--runs` times each, and the median run is reported.

The timed FFmpeg decoder and demuxer are `scripts/ffmpeg_decode.c`:
libavformat reads the packets and libavcodec decodes them, with no filter
graph and no output stage. The `ffmpeg` command runs those extra stages on
other threads, which is work the Rust tool does not do, so for decoding it
only makes the files and the dumps. Encoding is timed with the `ffmpeg`
command, reading the same raw file the Rust tool reads.

Threads: FFmpeg gets one thread when the Rust side used one, and sixteen,
the Rust worker pool's cap, when it used more.

Results go to `<work>/results.json`. `--report` turns them into one Markdown
block per crate under `<work>/tables/`, and `--fill` writes each block into
that crate's README in place of its `<!-- SPEED -->` marker.
"""

import argparse
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys

WIDTH, HEIGHT, RATE = 1920, 1080, 30
VIDEO_SECONDS = 20
ENCODE_SECONDS = 10
AUDIO_LOOPS = 2  # the source sound is played this many extra times
AUDIO_RATE = 48000
STILL_REPEATS = 200
POOL_CAP = 16
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def video(name, crate, ext, encode, decoder, seconds=VIDEO_SECONDS, how="picture", convert=None):
    return dict(name=name, crate=crate, kind="picture", ext=ext, encode=encode,
                decoder=decoder, seconds=seconds, how=how, convert=convert)


def audio(name, crate, ext, encode, decoder, seconds=None):
    return dict(name=name, crate=crate, kind="sound", ext=ext, encode=encode,
                decoder=decoder, seconds=seconds, how="sound")


def still(name, crate, ext, encode, decoder, frames=1, repeat=STILL_REPEATS, writer="FFmpeg"):
    return dict(name=name, crate=crate, kind="still", ext=ext, encode=encode,
                decoder=decoder, frames=frames, repeat=repeat, how="picture", convert=None,
                writer=writer)


DECODES = [
    video("h264", "codecs/h264", "mp4", ["-c:v", "libx264", "-preset", "medium", "-crf", "23"], "h264"),
    video("hevc", "codecs/hevc", "mp4",
          ["-c:v", "libx265", "-preset", "medium", "-crf", "26", "-tag:v", "hvc1",
           "-x265-params", "log-level=error"], "hevc"),
    video("av1", "codecs/av1", "mp4", ["-c:v", "libsvtav1", "-preset", "8", "-crf", "35"], "libdav1d"),
    video("vp8", "codecs/vp8", "ivf", ["-c:v", "libvpx", "-b:v", "4M", "-deadline", "good",
                                         "-cpu-used", "4"], "vp8", how="ivf"),
    video("vp9", "codecs/vp9", "ivf", ["-c:v", "libvpx-vp9", "-b:v", "3M", "-deadline", "good",
                                         "-cpu-used", "4", "-row-mt", "1"], "vp9", how="ivf"),
    video("mpeg4", "codecs/mpeg4", "mp4", ["-c:v", "mpeg4", "-q:v", "4"], "mpeg4"),
    video("mpeg2", "codecs/mpeg2", "mp4", ["-c:v", "mpeg2video", "-q:v", "4"], "mpeg2video"),
    video("mjpeg", "codecs/mjpeg", "mov", ["-c:v", "mjpeg", "-q:v", "4"], "mjpeg"),
    video("prores", "codecs/prores", "mov",
          ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"], "prores", seconds=10,
          convert="yuv420p"),
    audio("aac", "codecs/aac", "m4a", ["-c:a", "aac", "-b:a", "128k"], "aac"),
    audio("mp3", "codecs/mp3", "mp3", ["-c:a", "libmp3lame", "-b:a", "128k"], "mp3float"),
    audio("flac", "codecs/flac", "flac", ["-c:a", "flac"], "flac"),
    audio("alac", "codecs/alac", "m4a", ["-c:a", "alac"], "alac"),
    audio("opus", "codecs/opus", "opus", ["-c:a", "libopus", "-b:a", "96k"], "opus"),
    audio("vorbis", "codecs/vorbis", "ogg", ["-c:a", "vorbis", "-strict", "-2", "-q:a", "4"], "vorbis"),
    audio("ac3", "codecs/ac3", "mp4", ["-c:a", "ac3", "-b:a", "192k"], "ac3"),
    audio("eac3", "codecs/ac3", "mp4", ["-c:a", "eac3", "-b:a", "192k"], "eac3"),
    audio("pcm", "codecs/pcm", "wav", ["-c:a", "pcm_s16le"], "pcm_s16le", seconds=300),
    still("jpeg", "codecs/jpeg", "jpg", ["-c:v", "mjpeg", "-q:v", "3", "-pix_fmt", "yuvj420p"], "mjpeg"),
    still("png", "codecs/png", "png", ["-c:v", "png", "-pix_fmt", "rgb24"], "png"),
    still("gif", "codecs/gif", "gif", ["-vf", "fps=25,scale=960:540"], "gif", frames=250, repeat=10),
    # This FFmpeg build has no WebP encoder, so libwebp's own tool writes the file.
    still("webp", "codecs/webp", "webp", ["-q", "80"], "webp", writer="cwebp"),
    still("webp-lossless", "codecs/webp", "webp", ["-lossless"], "webp", writer="cwebp"),
]

# What the tables do not cover, said in the README beside them.
NOTES = {
    "codecs/jpeg": "Encoding was not measured: the tool encodes moving pictures and sound only.",
    "codecs/png": "Encoding was not measured: the tool encodes moving pictures and sound only.",
    "codecs/gif": "Encoding was not measured: the tool encodes moving pictures and sound only.",
    "codecs/webp": ("Encoding and animated files were not measured: the tool encodes moving "
                    "pictures and sound only, and the FFmpeg here cannot write WebP."),
    "codecs/vp9": ("The encoder was not measured. The decoder is timed from an IVF file, because "
                   "the desktop app does not register this crate."),
    "codecs/vp8": "The decoder is timed from an IVF file, read by the tool itself.",
    "codecs/prores": ("The stream is 10-bit 4:2:2. The Rust decoder hands the app 8-bit 4:2:0 and "
                      "converts to it inside the decoder, so FFmpeg is asked for 8-bit 4:2:0 too, "
                      "and the two conversions round differently."),
    "containers/still": ("Not measured. A still file is one packet, or one packet per picture for "
                         "a GIF or an animated WebP, so there is no packet stream to time against "
                         "libavformat. The codec crates time the decoding."),
    "containers/isobmff": "Writing files was not measured.",
    "containers/riff": "Writing files was not measured.",
    "containers/aiff": "Writing files was not measured.",
    "containers/flac": "Writing files was not measured.",
    "containers/ogg": "Writing files was not measured.",
    "containers/mp3": "Writing files was not measured.",
    "containers/adts": "Writing files was not measured. LOAS/LATM was not measured.",
}

# name, crate, the file to read (made by a decode case, or by the command given).
DEMUXES = [
    dict(name="mp4", crate="containers/isobmff", file="long.mp4",
         make=["-stream_loop", "89", "-i", "h264.mp4", "-i", "aac.m4a", "-map", "0:v", "-map", "1:a",
               "-c", "copy", "-shortest"]),
    dict(name="wav", crate="containers/riff", file="pcm.wav", make=None),
    dict(name="aiff", crate="containers/aiff", file="pcm.aiff",
         make=["-i", "pcm.wav", "-c:a", "pcm_s16be"]),
    dict(name="flac", crate="containers/flac", file="flac.flac", make=None),
    dict(name="ogg-opus", crate="containers/ogg", file="opus.opus", make=None),
    dict(name="ogg-vorbis", crate="containers/ogg", file="vorbis.ogg", make=None),
    dict(name="mp3", crate="containers/mp3", file="mp3.mp3", make=None),
    dict(name="adts", crate="containers/adts", file="aac.aac",
         make=["-i", "aac.m4a", "-c", "copy", "-f", "adts"]),
]

# Rust codec name, crate, bitrate, then each FFmpeg encoder it is set against.
PICTURE_ENCODES = [
    dict(name="h264", crate="codecs/h264", bitrate=8_000_000, form="h264", rivals=[
        ("libx264 ultrafast", ["-c:v", "libx264", "-preset", "ultrafast"]),
        ("libx264 medium", ["-c:v", "libx264", "-preset", "medium"])]),
    dict(name="av1", crate="codecs/av1", bitrate=4_000_000, form="ivf", rivals=[
        ("libsvtav1 preset 10", ["-c:v", "libsvtav1", "-preset", "10"]),
        ("libsvtav1 preset 6", ["-c:v", "libsvtav1", "-preset", "6"])]),
    dict(name="vp8", crate="codecs/vp8", bitrate=6_000_000, form="ivf", rivals=[
        ("libvpx realtime", ["-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "4"]),
        ("libvpx good", ["-c:v", "libvpx", "-deadline", "good", "-cpu-used", "2"])]),
    dict(name="mpeg4", crate="codecs/mpeg4", bitrate=8_000_000, form="m4v", rivals=[
        ("mpeg4", ["-c:v", "mpeg4"])]),
    dict(name="mpeg2", crate="codecs/mpeg2", bitrate=10_000_000, form="mpeg2video", rivals=[
        ("mpeg2video", ["-c:v", "mpeg2video"])]),
    # The Rust Motion JPEG encoder takes RGBA pictures, so both sides get those.
    dict(name="mjpeg", crate="codecs/mjpeg", bitrate=40_000_000, form="mjpeg", pixel="rgba", rivals=[
        ("mjpeg", ["-c:v", "mjpeg", "-pix_fmt", "yuvj420p"])]),
    dict(name="prores", crate="codecs/prores", bitrate=0, form="mov", rivals=[
        ("prores_ks 422", ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"])]),
]

SOUND_ENCODES = [
    dict(name="aac", crate="codecs/aac", bitrate=128_000, form="adts", rivals=[("aac", ["-c:a", "aac"])]),
    dict(name="mp3", crate="codecs/mp3", bitrate=128_000, form="mp3", rivals=[
        ("libmp3lame", ["-c:a", "libmp3lame"])]),
    dict(name="opus", crate="codecs/opus", bitrate=96_000, form="ogg", rivals=[
        ("libopus", ["-c:a", "libopus"])]),
    dict(name="vorbis", crate="codecs/vorbis", bitrate=128_000, form="ogg", rivals=[
        ("vorbis", ["-c:a", "vorbis", "-strict", "-2"])]),
    dict(name="ac3", crate="codecs/ac3", bitrate=192_000, form="ac3", rivals=[("ac3", ["-c:a", "ac3"])]),
    dict(name="eac3", crate="codecs/ac3", bitrate=192_000, form="eac3", rivals=[
        ("eac3", ["-c:a", "eac3"])]),
    dict(name="flac", crate="codecs/flac", bitrate=0, form="flac", rivals=[
        ("flac", ["-c:a", "flac", "-sample_fmt", "s16"])]),
    dict(name="alac", crate="codecs/alac", bitrate=0, form="ipod", rivals=[
        ("alac", ["-c:a", "alac", "-sample_fmt", "s16p"])]),
    dict(name="pcm", crate="codecs/pcm", bitrate=0, form="s16le", rivals=[
        ("pcm_s16le", ["-c:a", "pcm_s16le"])]),
]


def run(command, **kwargs):
    done = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(command)}\n{(done.stderr or done.stdout)[-1500:]}")
    return done


def timed(command):
    """One run under /usr/bin/time -l: wall, CPU, peak memory, instructions."""
    done = subprocess.run(["/usr/bin/time", "-l", *command], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(command)}\n{done.stderr[-1500:]}")
    text = done.stderr
    real = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
    memory = re.search(r"(\d+)\s+maximum resident set size", text)
    instructions = re.search(r"(\d+)\s+instructions retired", text)
    return dict(wall=float(real.group(1)), cpu=float(real.group(2)) + float(real.group(3)),
                memory=int(memory.group(1)), instructions=int(instructions.group(1)),
                line=done.stdout.strip())


def middle(runs):
    keys = ("wall", "cpu", "memory", "instructions")
    return {key: statistics.median(run[key] for run in runs) for key in keys}


def race(args, mine, other):
    """The two commands in turn, `--runs` times each, after one warm-up each."""
    timed(mine)
    timed(other)
    ours, theirs = [], []
    for _ in range(args.runs):
        ours.append(timed(mine))
        theirs.append(timed(other))
    return middle(ours), middle(theirs)


def threads_for(probe):
    threaded = probe["cpu"] > probe["wall"] * 1.25 and probe["wall"] > 0.05
    return POOL_CAP if threaded else 1


def ffmpeg(args):
    return [args.ffmpeg, "-v", "error", "-nostdin", "-y"]


# Decode ---------------------------------------------------------------------


def make(case, args, path):
    """ffmpeg writes the file for one decode case from the source clip."""
    if os.path.exists(path):
        return
    if case["kind"] == "picture":
        command = ffmpeg(args) + ["-i", args.video, "-t", str(case["seconds"]), "-an",
                                  "-vf", f"fps={RATE},scale={WIDTH}:{HEIGHT},format=yuv420p"]
        command += case["encode"] + [path]
    elif case["kind"] == "sound":
        command = ffmpeg(args) + ["-stream_loop", str(AUDIO_LOOPS), "-i", args.audio, "-vn",
                                  "-ac", "2"]
        if case["seconds"]:
            command += ["-t", str(case["seconds"])]
        command += case["encode"] + [path]
    elif case["frames"] > 1:
        command = ffmpeg(args) + ["-i", args.video, "-t", "10", "-an"] + case["encode"] + [path]
    elif case["writer"] == "cwebp":
        plain = path + ".png"
        run(ffmpeg(args) + ["-ss", "5", "-i", args.video, "-frames:v", "1", "-an",
                            "-vf", f"scale={WIDTH}:{HEIGHT}", "-pix_fmt", "rgb24", plain])
        command = ["cwebp", "-quiet", *case["encode"], plain, "-o", path]
    else:
        command = ffmpeg(args) + ["-ss", "5", "-i", args.video, "-frames:v", "1", "-an",
                                  "-vf", f"scale={WIDTH}:{HEIGHT}"] + case["encode"] + [path]
    run(command)


def ours(args, case, path, extra=()):
    return [args.tool, "decode", case["how"], path, *extra]


def layout(case, shape):
    """The FFmpeg pixel format that matches what the Rust decoder returned.

    `None` when FFmpeg's decoder already returns that layout.
    """
    if case["kind"] == "sound":
        return None
    still = re.search(r" (Rgb|Rgba|Gray|GrayAlpha) (Eight|Sixteen)$", shape)
    if still:
        deep = still.group(2) == "Sixteen"
        return {("Rgb", False): "rgb24", ("Rgba", False): "rgba", ("Gray", False): "gray",
                ("GrayAlpha", False): "ya8", ("Rgb", True): "rgb48le", ("Rgba", True): "rgba64le",
                ("Gray", True): "gray16le", ("GrayAlpha", True): "ya16le"}[
                    (still.group(1), deep)]
    if shape.endswith("Rgba8"):
        return "rgba"
    return case.get("convert")


def dump(args, case, path, sink):
    """The ffmpeg command that writes one stream's decoded samples to `sink`."""
    stream = "a" if case["kind"] == "sound" else "v"
    command = ffmpeg(args) + ["-threads", "1", f"-c:{stream}", case["decoder"], "-i", path,
                              "-map", f"0:{stream}:0"]
    if case["kind"] != "sound":
        # Each picture as stored, not timed to a frame rate.
        command += ["-fps_mode", "passthrough"]
    return command + sink


def theirs(args, case, path, threads, repeat, pixel):
    """The timed ffmpeg side: libavformat and libavcodec, nothing else."""
    kind = "sound" if case["kind"] == "sound" else (f"to:{pixel}" if pixel else "picture")
    return [args.harness, kind, path, case["decoder"], str(threads), str(repeat)]


def check(args, case, path, shape, pixel):
    """Dump both decoders' output and compare the dumps."""
    mine, other = path + ".ours.raw", path + ".ffmpeg.raw"
    run(ours(args, case, path, ["--dump", mine]))
    if case["kind"] == "sound":
        form = "s16le" if "16" in shape.split()[-1] else "f32le"
        codec = "pcm_s16le" if form == "s16le" else "pcm_f32le"
        run(dump(args, case, path, ["-c:a", codec, "-f", form, other]))
        kind = "i16" if form == "s16le" else "f32"
        result = run([args.tool, "compare", kind, mine, other, "--channels", "2"]).stdout.strip()
    else:
        sink = (["-pix_fmt", pixel] if pixel else []) + ["-f", "rawvideo", other]
        run(dump(args, case, path, sink))
        still = re.search(r" (Eight|Sixteen)$", shape)
        kind = "u8"
        if still and still.group(1) == "Sixteen":
            kind = "u16"
        elif still:
            # A still is dumped as one 16-bit value per sample, whatever its depth.
            with open(mine, "rb") as handle:
                low = handle.read()[::2]
            with open(mine, "wb") as handle:
                handle.write(low)
        elif re.search(r"p1[026]", shape.lower()):
            kind = "u16"
        result = run([args.tool, "compare", kind, mine, other]).stdout.strip()
    for leftover in (mine, other):
        os.remove(leftover)
    return result


def decode(args, case):
    path = os.path.join(args.work, f"{case['name']}.{case['ext']}")
    make(case, args, path)
    record = dict(bytes=os.path.getsize(path), made=" ".join(case["encode"]),
                  rival=case["decoder"], kind=case["kind"], writer=case.get("writer", "FFmpeg"))
    times = case.get("repeat", 1)
    repeat = ["--repeat", str(times)] if times > 1 else []
    probe = timed(ours(args, case, path, repeat))
    record["decoded"] = probe["line"]
    shape = probe["line"]
    if case["kind"] != "sound":
        shape = probe["line"].split(" bytes ")[-1].split(" ", 1)[-1]
    pixel = layout(case, shape)
    record["converted"] = pixel
    record["check"] = check(args, case, path, shape, pixel)
    record["threads"] = threads_for(probe)
    record["cutlass"], record["ffmpeg"] = race(
        args, ours(args, case, path, repeat),
        theirs(args, case, path, record["threads"], times, pixel))
    count = re.search(r"(?:pictures|frames) (\d+)", probe["line"])
    record["units"] = int(count.group(1)) * times
    record["repeat"] = times
    rate = re.search(r"(\d+) Hz", probe["line"])
    if rate:
        record["rate"] = int(rate.group(1))
    return record


# Demux ----------------------------------------------------------------------


def demux(args, case):
    path = os.path.join(args.work, case["file"])
    if not os.path.exists(path):
        if case["make"] is None:
            raise RuntimeError(f"{case['file']} is made by a decode case; run that first")
        run(ffmpeg(args) + case["make"] + [path], cwd=args.work)
    mine = [args.tool, "demux", path]
    other = [args.harness, "demux", path, "none", "1", "1"]
    probe = timed(mine)
    record = dict(bytes=os.path.getsize(path), read=probe["line"],
                  ffmpeg_read=timed(other)["line"], threads=1)
    record["cutlass"], record["ffmpeg"] = race(args, mine, other)
    return record


# Encode ---------------------------------------------------------------------


def sources(args):
    """The raw pictures and raw sound every encoder is given."""
    pictures = os.path.join(args.work, "source.yuv")
    sound = os.path.join(args.work, "source.f32")
    for path, pixel in ((pictures, "yuv420p"), (pictures[:-3] + "rgba", "rgba")):
        if not os.path.exists(path):
            run(ffmpeg(args) + ["-i", args.video, "-t", str(ENCODE_SECONDS), "-an",
                                "-vf", f"fps={RATE},scale={WIDTH}:{HEIGHT},format={pixel}",
                                "-f", "rawvideo", path])
    if not os.path.exists(sound):
        run(ffmpeg(args) + ["-stream_loop", str(AUDIO_LOOPS), "-i", args.audio, "-vn", "-ac", "2",
                            "-ar", str(AUDIO_RATE), "-f", "f32le", sound])
    return pictures, sound


def psnr(args, decoded, source):
    try:
        return run([args.tool, "compare", "u8", decoded, source]).stdout.strip()
    except RuntimeError as error:
        return f"not measured: {str(error).strip().splitlines()[-1][:160]}"


def encode_pictures(args, case, source):
    out = os.path.join(args.work, f"enc-{case['name']}.bin")
    size = [str(WIDTH), str(HEIGHT), str(RATE), str(case["bitrate"])]
    pixel = case.get("pixel", "yuv420p")
    if pixel == "rgba":
        source = source[:-3] + "rgba"
    mine = [args.tool, "encode", "picture", case["name"], *size, source, "--out", out]
    if pixel == "rgba":
        mine.append("--rgba")
    probe = timed(mine)
    frames = int(re.search(r"pictures (\d+)", probe["line"]).group(1))
    threads = threads_for(probe)
    decoded = out + ".yuv"
    try:
        run(mine + ["--decoded", decoded])
        kept = psnr(args, decoded, source)
    except RuntimeError as error:
        kept = f"not measured: {str(error).strip().splitlines()[-1][:160]}"
    if os.path.exists(decoded):
        os.remove(decoded)
    record = dict(kind="picture", bitrate=case["bitrate"], units=frames, threads=threads,
                  encoded=probe["line"], bytes=os.path.getsize(out), kept=kept, pixel=pixel,
                  rivals=[])
    record["cutlass"] = None
    for label, options in case["rivals"]:
        theirs_out = os.path.join(args.work, f"enc-{case['name']}-ffmpeg.{case['form']}")
        rate = ["-b:v", str(case["bitrate"])] if case["bitrate"] else []
        other = ffmpeg(args) + ["-f", "rawvideo", "-pix_fmt", pixel, "-s", f"{WIDTH}x{HEIGHT}",
                                "-r", str(RATE), "-i", source, "-threads", str(threads),
                                *options, *rate, "-f", case["form"], theirs_out]
        a, b = race(args, mine, other)
        back = theirs_out + ".yuv"
        # Every picture as stored: a bare stream has no clock to time them by.
        run(ffmpeg(args) + ["-i", theirs_out, "-fps_mode", "passthrough", "-pix_fmt", pixel,
                            "-f", "rawvideo", back])
        rival = dict(label=label, options=" ".join(options + rate), ffmpeg=b,
                     bytes=os.path.getsize(theirs_out), kept=psnr(args, back, source))
        os.remove(back)
        record["rivals"].append(rival)
        if record["cutlass"] is None:
            record["cutlass"] = a
    return record


def encode_sound(args, case, source):
    out = os.path.join(args.work, f"enc-{case['name']}.bin")
    mine = [args.tool, "encode", "sound", case["name"], str(AUDIO_RATE), "2",
            str(case["bitrate"]), source, "--out", out]
    probe = timed(mine)
    frames = int(re.search(r"frames (\d+)", probe["line"]).group(1))
    threads = threads_for(probe)
    record = dict(kind="sound", bitrate=case["bitrate"], units=frames, rate=AUDIO_RATE,
                  threads=threads, encoded=probe["line"], bytes=os.path.getsize(out), rivals=[])
    record["cutlass"] = None
    for label, options in case["rivals"]:
        theirs_out = os.path.join(args.work, f"enc-{case['name']}-ffmpeg.{case['form']}")
        rate = ["-b:a", str(case["bitrate"])] if case["bitrate"] else []
        other = ffmpeg(args) + ["-f", "f32le", "-ar", str(AUDIO_RATE), "-ac", "2", "-i", source,
                                "-threads", str(threads), *options, *rate, "-f", case["form"],
                                theirs_out]
        a, b = race(args, mine, other)
        record["rivals"].append(dict(label=label, options=" ".join(options + rate), ffmpeg=b,
                                     bytes=os.path.getsize(theirs_out)))
        if record["cutlass"] is None:
            record["cutlass"] = a
    return record


# Report ---------------------------------------------------------------------


def seconds(value):
    return f"{value:.2f} s" if value < 100 else f"{value:.0f} s"


def megabytes(value):
    if value < 100e3:
        return f"{value / 1e3:.0f} kB"
    return f"{value / 1e6:.0f} MB" if value >= 10e6 else f"{value / 1e6:.1f} MB"


def speed(record, wall):
    if wall <= 0:
        return "too fast to time"
    if record["kind"] == "sound":
        return f"{record['units'] / record['rate'] / wall:,.0f}× real time"
    unit = "pictures/s" if record["kind"] == "still" else "frames/s"
    value = record["units"] / wall
    return f"{value:,.0f} {unit}" if value >= 10 else f"{value:.1f} {unit}"


def verdict(ours, theirs, against="FFmpeg"):
    """How the Rust side's wall time stands against the other's, in words."""
    if ours <= 0 or theirs <= 0:
        return f"The run is too short to compare with {against}."
    if ours >= theirs * 1.05:
        return f"Cutlass is **{ours / theirs:.1f}× slower** than {against} here."
    if theirs >= ours * 1.05:
        return f"Cutlass is **{theirs / ours:.1f}× faster** than {against} here."
    return f"Cutlass and {against} take **about the same time** here."


def same(check):
    """The output comparison, in words."""
    if check.startswith("identical") or check.startswith("equal"):
        return "Both decoders produced exactly the same samples."
    worst = re.search(r"psnr ([\d.]+) dB worst ([\d.e+-]+)", check)
    if worst:
        text = (f"The two outputs are not bit-identical: {worst.group(1)} dB PSNR between them, "
                f"largest single difference {float(worst.group(2)):.3g}")
        lag = re.search(r"lag (\d+) frames, lengths (\d+) and (\d+)", check)
        if lag and lag.group(2) != lag.group(3):
            text += (f" (lengths {int(lag.group(2)):,} and {int(lag.group(3)):,} sample frames, "
                     f"compared after sliding one by {lag.group(1)})")
        return text + "."
    return f"Output comparison: {check}."


def table(rows, heads):
    lines = ["| " + " | ".join(heads) + " |",
             "| --- |" + " ---: |" * (len(heads) - 1)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def what(record):
    """One line on the input of a decode case."""
    size = megabytes(record["bytes"])
    writer = record.get("writer", "FFmpeg")
    if record["kind"] == "sound":
        minutes = record["units"] / record["rate"] / 60
        return (f"{minutes:.0f} minutes of stereo sound at {record['rate']:,} Hz ({size}), "
                f"written by FFmpeg with `{record['made']}`")
    shape = record["decoded"].split(" bytes ")[-1].split(" ", 1)[-1].split()[0].replace("x", "×")
    if record["kind"] == "still":
        each = record["units"] // record["repeat"]
        many = "one picture" if each == 1 else f"{each} pictures in one file"
        return (f"{many} of {shape} ({size}), written by {writer} with `{record['made']}`, "
                f"opened and decoded {record['repeat']} times")
    return (f"{record['units']} frames of {shape} at {RATE} frames a second ({size}), "
            f"written by FFmpeg with `{record['made']}`")


def block_decode(name, record):
    a, b = record["cutlass"], record["ffmpeg"]
    rows = [
        ["Wall time", seconds(a["wall"]), seconds(b["wall"])],
        ["Speed", speed(record, a["wall"]), speed(record, b["wall"])],
        ["CPU time", seconds(a["cpu"]), seconds(b["cpu"])],
        ["Peak memory", megabytes(a["memory"]), megabytes(b["memory"])],
        ["Threads allowed", str(record["threads"]), str(record["threads"])],
    ]
    text = f"**Decode, {name}.** Input: {what(record)}.\n\n"
    text += table(rows, ["", "Cutlass", f"FFmpeg (`{record['rival']}`)"]) + "\n\n"
    text += verdict(a["wall"], b["wall"]) + " " + same(record["check"])
    if record.get("converted"):
        text += (f" The Rust decoder returns `{record['converted']}` pictures, so FFmpeg's time "
                 "includes converting each picture to that layout with libswscale.")
    return text


def block_demux(name, record):
    a, b = record["cutlass"], record["ffmpeg"]

    def packets(line):
        count = int(re.search(r"packets (\d+)", line).group(1))
        return f"{count:,}"

    rows = [
        ["Wall time", seconds(a["wall"]), seconds(b["wall"])],
        ["CPU time", seconds(a["cpu"]), seconds(b["cpu"])],
        ["Peak memory", megabytes(a["memory"]), megabytes(b["memory"])],
        ["Packets returned", packets(record["read"]), packets(record["ffmpeg_read"])],
    ]
    text = (f"**Read every packet, {name}.** Input: one {megabytes(record['bytes'])} file. "
            "Nothing is decoded.\n\n")
    text += table(rows, ["", "Cutlass", "FFmpeg (libavformat)"]) + "\n\n"
    return text + verdict(a["wall"], b["wall"])


def kept_text(kept):
    found = re.search(r"psnr ([\d.]+) dB", kept)
    if found:
        return f"{found.group(1)} dB"
    if kept.startswith("identical") or kept.startswith("equal"):
        return "exact"
    return "not measured"


def block_encode(name, record):
    a = record["cutlass"]
    heads = ["", "Cutlass"] + [f"FFmpeg (`{rival['label']}`)" for rival in record["rivals"]]
    sides = [dict(ffmpeg=a, bytes=record["bytes"], kept=record.get("kept", ""))] + record["rivals"]
    rows = [
        ["Wall time"] + [seconds(side["ffmpeg"]["wall"]) for side in sides],
        ["Speed"] + [speed(record, side["ffmpeg"]["wall"]) for side in sides],
        ["CPU time"] + [seconds(side["ffmpeg"]["cpu"]) for side in sides],
        ["Peak memory"] + [megabytes(side["ffmpeg"]["memory"]) for side in sides],
        ["Output size"] + [megabytes(side["bytes"]) for side in sides],
    ]
    if record["kind"] == "picture":
        rows.append(["Quality kept (PSNR)"] + [kept_text(side["kept"]) for side in sides])
    rows.append(["Threads allowed"] + [str(record["threads"])] * len(sides))
    if record["kind"] == "picture":
        given = f"{record['units']} raw frames of {WIDTH}×{HEIGHT} at {RATE} frames a second"
        if record.get("pixel") == "rgba":
            given += ", as RGBA because that is what the Rust encoder takes"
    else:
        given = (f"{record['units'] / record['rate'] / 60:.0f} minutes of raw stereo sound at "
                 f"{record['rate']:,} Hz")
    asked = ""
    if record["bitrate"] >= 1e6:
        asked = f", asked for {record['bitrate'] / 1e6:g} Mbit/s"
    elif record["bitrate"]:
        asked = f", asked for {record['bitrate'] // 1000} kbit/s"
    text = f"**Encode, {name}.** Input: {given}{asked}.\n\n"
    text += table(rows, heads) + "\n\n"
    text += " ".join(verdict(a["wall"], rival["ffmpeg"]["wall"], f"`{rival['label']}`")
                     for rival in record["rivals"])
    if record["kind"] == "picture":
        text += (" Quality is the PSNR between the source pictures and what each encoder's output "
                 "decodes to; higher is closer to the source. The encoders do not make the same "
                 "choices, so read speed, size and quality together.")
    else:
        text += " Sound quality was not compared, so this is speed and size only."
    return text


def report(args, results):
    meta = results["meta"]
    by_crate = {}
    for key, record in results["cases"].items():
        by_crate.setdefault(record["crate"], []).append((key, record))
    os.makedirs(os.path.join(args.work, "tables"), exist_ok=True)
    order = {"decode": 0, "encode": 1, "demux": 2}
    written = {}
    for crate in NOTES:
        by_crate.setdefault(crate, [])
    for crate, records in by_crate.items():
        if not records:
            written[crate] = NOTES[crate]
            continue
        blocks, failures = [], []
        for key, record in sorted(records, key=lambda item: order[item[0].split(":")[0]]):
            mode, name = key.split(":")
            if "failed" in record:
                reason = record["failed"].strip().splitlines()[-1][:200]
                failures.append(f"**{mode.capitalize()}, {name}: not measured.** The run stopped "
                                f"with: `{reason}`")
                continue
            make = {"decode": block_decode, "demux": block_demux, "encode": block_encode}[mode]
            blocks.append(make(name, record))
        head = (f"Measured on {meta['date']} at commit `{meta['commit']}`, on an {meta['machine']} "
                f"running macOS {meta['os']}, against FFmpeg {meta['ffmpeg']}. "
                f"Built with `cargo build --release` and {meta['rustc']}.")
        how = (
            "How it was measured: both sides get the same input from disk, do all of the work, "
            "and throw the result away. Each side is a whole process timed with "
            "`/usr/bin/time -l`, start-up included, run five times in turn; the tables show the "
            "median run. FFmpeg is given as many threads as the Rust code used. The Rust side is "
            "[`tools/media-bench`](../../../tools/media-bench), which registers only the Rust "
            "codecs. For decoding and packet reading the FFmpeg side is "
            "[`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls "
            "libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. "
            "The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a "
            "music recording.\n\n"
            "To run it again: `cargo build --release -p media-bench`, then "
            "`python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.")
        notes = [NOTES[crate]] if crate in NOTES else []
        text = "\n\n".join([head] + blocks + failures + notes + [how])
        path = os.path.join(args.work, "tables", crate.replace("/", "-") + ".md")
        with open(path, "w") as handle:
            handle.write(text + "\n")
        written[crate] = text
    return written


def fill(written):
    """Put each crate's block where its README has the marker."""
    for crate, text in sorted(written.items()):
        path = os.path.join(ROOT, "crates", crate, "README.md")
        if not os.path.exists(path):
            print(f"{crate}: no README.md")
            continue
        with open(path) as handle:
            page = handle.read()
        if "<!-- SPEED -->" not in page:
            print(f"{crate}: no marker in README.md")
            continue
        with open(path, "w") as handle:
            handle.write(page.replace("<!-- SPEED -->", text))
        print(f"{crate}: filled")


# Main -----------------------------------------------------------------------


def describe(args):
    def text(command):
        return run(command).stdout.strip()

    cores = text(["sysctl", "-n", "hw.ncpu"])
    memory = int(text(["sysctl", "-n", "hw.memsize"])) // 2**30
    return dict(
        date=datetime.date.today().strftime("%-d %B %Y"),
        commit=text(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]),
        machine=f"{text(['sysctl', '-n', 'machdep.cpu.brand_string'])} ({cores} cores, {memory} GB)",
        os=platform.mac_ver()[0],
        ffmpeg=text([args.ffmpeg, "-version"]).split()[2],
        rustc=text(["rustc", "--version"]).rsplit(" (", 1)[0],
    )


def build_harness(args):
    """Compile scripts/ffmpeg_decode.c against the installed FFmpeg."""
    source = os.path.join(ROOT, "scripts", "ffmpeg_decode.c")
    flags = run(["pkg-config", "--cflags", "--libs", "libavformat", "libavcodec",
                 "libswscale", "libavutil"]).stdout.split()
    args.harness = os.path.join(args.work, "ffmpeg_decode")
    run(["cc", "-O2", "-o", args.harness, source, *flags])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", help="a clip with real pictures")
    parser.add_argument("--audio", help="a file with real sound")
    parser.add_argument("--work", required=True, help="folder for the files and results")
    parser.add_argument("--tool", default=os.path.join(ROOT, "target/release/media-bench"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--only", default="", help="comma-separated cases, as mode:name or name")
    parser.add_argument("--modes", default="decode,demux,encode")
    parser.add_argument("--report", action="store_true", help="write the tables and stop")
    parser.add_argument("--fill", action="store_true", help="with --report, fill the READMEs")
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)
    out = os.path.join(args.work, "results.json")
    results = dict(meta={}, cases={})
    if os.path.exists(out):
        with open(out) as handle:
            results = json.load(handle)
    if args.report:
        written = report(args, results)
        if args.fill:
            fill(written)
        return 0
    if not args.video or not args.audio:
        parser.error("--video and --audio are needed to measure")
    build_harness(args)
    results["meta"] = describe(args)
    wanted = set(filter(None, args.only.split(",")))
    modes = args.modes.split(",")
    plan = []
    if "decode" in modes:
        plan += [("decode", case, decode) for case in DECODES]
    if "demux" in modes:
        plan += [("demux", case, demux) for case in DEMUXES]
    if "encode" in modes:
        pictures, sound = sources(args)
        plan += [("encode", case, lambda a, c: encode_pictures(a, c, pictures))
                 for case in PICTURE_ENCODES]
        plan += [("encode", case, lambda a, c: encode_sound(a, c, sound)) for case in SOUND_ENCODES]
    for mode, case, measure in plan:
        key = f"{mode}:{case['name']}"
        if wanted and key not in wanted and case["name"] not in wanted:
            continue
        try:
            record = measure(args, case)
        except (RuntimeError, AttributeError) as error:
            record = dict(failed=str(error)[-600:])
        record["crate"] = case["crate"]
        results["cases"][key] = record
        with open(out, "w") as handle:
            json.dump(results, handle, indent=1)
        if "failed" in record:
            print(f"{key}: FAILED {record['failed'].strip().splitlines()[-1][:200]}", flush=True)
            continue
        a = record["cutlass"]
        rivals = [record["ffmpeg"]] if "ffmpeg" in record else [r["ffmpeg"] for r in record["rivals"]]
        note = record.get("check") or record.get("kept") or record.get("read") or ""
        print(f"{key}: wall {a['wall']:.2f}s vs " + ", ".join(f"{b['wall']:.2f}s" for b in rivals)
              + f" | cpu {a['cpu']:.2f}s vs " + ", ".join(f"{b['cpu']:.2f}s" for b in rivals)
              + f" | threads {record['threads']} | {note}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
