#!/usr/bin/env python3
"""Times media-bench against FFmpeg (scripts/ffmpeg_decode.c) decoding the
same IVF files, each a whole process run in turn N times; prints the
median wall time, CPU time and peak memory of each side.

    bench.py <ffmpeg_decode binary> <threads> <runs> <file.ivf>...
"""
import os, resource, statistics, subprocess, sys, time

def run(cmd):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, usage = os.wait4(p.pid, 0)
    wall = time.perf_counter() - t
    if status != 0:
        raise SystemExit(f"failed: {cmd}")
    return wall, usage.ru_utime + usage.ru_stime, usage.ru_maxrss / 1024

def main():
    ff, threads, runs, files = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ours = os.environ.get("BIN", f"{root}/target/release/media-bench")
    env_threads = ["--threads", threads] if os.environ.get("OUR_THREADS") else []
    for f in files:
        a, b = [], []
        for _ in range(runs):
            a.append(run([ours, "decode", "ivf", f] + env_threads))
            b.append(run([ff, "picture", f, "vp9", threads, "1"]))
        med = lambda xs, i: statistics.median(x[i] for x in xs)
        wins = sum(1 for x, y in zip(a, b) if x[0] < y[0])
        best = lambda xs: min(x[0] for x in xs)
        print(f"{os.path.basename(f):14} cutlass {med(a,0):6.3f}s cpu {med(a,1):6.3f}s {med(a,2):6.0f}MB | "
              f"ffmpeg {med(b,0):6.3f}s cpu {med(b,1):6.3f}s {med(b,2):6.0f}MB | ratio {med(a,0)/med(b,0):5.2f} "
              f"(fastest {best(a):.3f}/{best(b):.3f} = {best(a)/best(b):.2f}) wins {wins}/{runs}")

main()
