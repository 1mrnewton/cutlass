#!/usr/bin/env python3
"""BD-rate of B against A from encbench.sh output: bdrate.py A.txt B.txt.
Piecewise-linear log-rate over PSNR-Y on the overlapping PSNR range;
negative means B needs fewer bits for the same quality."""
import math, sys

def curve(path):
    pts = []
    for line in open(path):
        f = line.split()
        if len(f) >= 5:
            pts.append((float(f[3]), math.log(float(f[1]))))
    return sorted(pts)

def at(c, q):
    for (q0, r0), (q1, r1) in zip(c, c[1:]):
        if q0 <= q <= q1:
            return r0 + (r1 - r0) * (q - q0) / (q1 - q0) if q1 > q0 else r0
    return None

a, b = curve(sys.argv[1]), curve(sys.argv[2])
lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
if hi <= lo:
    sys.exit("no overlap")
n = 200
d = sum(at(b, lo + (hi - lo) * i / n) - at(a, lo + (hi - lo) * i / n) for i in range(n + 1)) / (n + 1)
print(f"BD-rate {100 * (math.exp(d) - 1):+.1f}% over PSNR-Y {lo:.2f}..{hi:.2f}")
