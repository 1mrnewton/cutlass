#!/usr/bin/env python3
"""Writes line-break cases with ICU's answers, to check ours against.

    python3 tools/text-oracles/line_breaks.py cases.txt 30000 [seed low high]
    CUTLASS_LINEBREAK_CASES=cases.txt cargo test -p cutlass-text --lib -- \
        --ignored linebreak::tests::another_implementation_agrees

Each case is a random string over an alphabet that covers the line
breaking classes: code points in hex, a semicolon, then the character
indices where ICU's line iterator breaks, with `!` after a mandatory one.
Needs ICU 78 from Homebrew; change LIBRARY for another install.
"""
import ctypes
import random
import sys

LIBRARY = "/opt/homebrew/opt/icu4c/lib/libicuuc.78.dylib"
uc = ctypes.CDLL(LIBRARY)
ubrk_open = uc.ubrk_open_78; ubrk_open.restype = ctypes.c_void_p
ubrk_open.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int)]
ubrk_next = uc.ubrk_next_78; ubrk_next.restype = ctypes.c_int32; ubrk_next.argtypes = [ctypes.c_void_p]
ubrk_close = uc.ubrk_close_78; ubrk_close.argtypes = [ctypes.c_void_p]
ubrk_status = uc.ubrk_getRuleStatus_78; ubrk_status.restype = ctypes.c_int32; ubrk_status.argtypes=[ctypes.c_void_p]

def breaks(text):
    data = text.encode("utf-16-le")
    n = len(data)//2
    buf = ctypes.create_string_buffer(data, len(data)+2)
    err = ctypes.c_int(0)
    it = ubrk_open(2, b"en", buf, n, ctypes.byref(err))
    out = []
    # map utf16 offsets to char index
    offs = {}
    o = 0
    for i, ch in enumerate(text):
        offs[o] = i
        o += 2 if ord(ch) > 0xFFFF else 1
    offs[o] = len(text)
    while True:
        b = ubrk_next(it)
        if b < 0: break
        hard = ubrk_status(it) >= 100
        if b != n:
            out.append((offs[b], hard))
    ubrk_close(it)
    return out

alphabet = [0x41,0x62,0x63,0x20,0x20,0x20,0x2D,0x2010,0x2E,0x2C,0x3A,0x3B,0x21,0x3F,0x28,0x29,0x5B,0x5D,0x22,0x27,0x201C,0x201D,0x2018,0x2019,0xAB,0xBB,
  0x31,0x32,0x24,0x25,0x2F,0x5C,0xA0,0x2060,0x200B,0x200D,0xAD,0x2014,0x2026,0x301,0x308,0x4E2D,0x6587,0x3002,0x3001,0x300C,0x300D,0x30AD,0x30E3,0x30C3,0x3041,0x30FC,
  0xFF08,0xFF09,0xAC00,0xAC01,0x1100,0x1161,0x11A8,0x1F600,0x1F44D,0x1F3FD,0x1F1EC,0x1F1ED,0x5D0,0x5D1,0x5BE,0x627,0x628,0x660,0x915,0x94D,0x93F,0xE01,0xE31,
  0x0A,0x0D,0x85,0x2028,0x9,0xB0,0x20AC,0xA3,0x2030,0x2B,0x3D,0x7E,0x2013,0x2012,0x58A,0xF0B,0x1B05,0x1B44,0x1B3A,0x25CC,0x11003,0x11046,0x2D,0x3000,0xFF01,0x23,0x40,0x2A,0x7C,0xB7,0x2032,0x2103,0x2116]
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
low = int(sys.argv[4]) if len(sys.argv) > 4 else 1
high = int(sys.argv[5]) if len(sys.argv) > 5 else 9
cases = set()
for _ in range(int(sys.argv[2]) if len(sys.argv)>2 else 20000):
    n = random.randint(low, high)
    cases.add("".join(chr(random.choice(alphabet)) for _ in range(n)))
with open(sys.argv[1], "w") as f:
    for text in sorted(cases):
        b = breaks(text)
        f.write(" ".join(f"{ord(c):X}" for c in text) + ";" + " ".join(f"{i}{'!' if h else ''}" for i,h in b) + "\n")
print(len(cases))
