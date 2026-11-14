#!/usr/bin/env python3
"""Writes bidi cases with ICU's embedding levels, to check ours against.

    python3 tools/text-oracles/bidi_levels.py cases.txt 40000 3 1 8
    CUTLASS_BIDI_CASES=cases.txt cargo test -p cutlass-text --lib -- \
        --ignored bidi::tests::another_implementation_agrees

The arguments are the file to write, how many cases, a random seed, and
the shortest and longest string. Each case is code points in hex, a
semicolon, then a level for each: `x` for a character the algorithm
removes, and `e` or `o` (even, odd) for a line all one way, whose levels
ICU does not report. ICU keeps a mark that follows a bracket neutral where
the Unicode text and FriBidi give it the bracket's direction; a handful
of long cases differ there, and ours follow the text.
Needs ICU 78 from Homebrew; change the library path for another install.
"""
import ctypes, random, sys
uc = ctypes.CDLL("/opt/homebrew/opt/icu4c/lib/libicuuc.78.dylib")
f = lambda n, r, *a: (setattr(getattr(uc, n + "_78"), "restype", r), setattr(getattr(uc, n + "_78"), "argtypes", list(a)), getattr(uc, n + "_78"))[2]
ubidi_open = f("ubidi_open", ctypes.c_void_p)
ubidi_setPara = f("ubidi_setPara", None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint8, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
ubidi_getDirection = f("ubidi_getDirection", ctypes.c_int, ctypes.c_void_p)
ubidi_getLevels = f("ubidi_getLevels", ctypes.POINTER(ctypes.c_uint8), ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
char_dir = f("u_charDirection", ctypes.c_int, ctypes.c_int32)
bidi = ubidi_open()
REMOVED = {11, 12, 14, 15, 16, 18}
def levels(text):
    data = text.encode("utf-16-le"); n = len(data)//2
    buf = ctypes.create_string_buffer(data, len(data)+2)
    err = ctypes.c_int(0)
    ubidi_setPara(bidi, buf, n, 0xFE, None, ctypes.byref(err))   # UBIDI_DEFAULT_LTR
    lv = ubidi_getLevels(bidi, ctypes.byref(err))
    out = []; o = 0
    plain = ubidi_getDirection(bidi) != 2
    for ch in text:
        out.append("x" if char_dir(ord(ch)) in REMOVED else ("eo"[lv[o] % 2] if plain else str(lv[o])))
        o += 2 if ord(ch) > 0xFFFF else 1
    return out
alphabet = [0x61,0x62,0x41,0x5D0,0x5D1,0x5D2,0x627,0x628,0x644,0x30,0x31,0x39,0x660,0x661,0x6F1,0x20,0x20,0x20,0x9,0x2E,0x2C,0x3A,0x2B,0x2D,0x24,0x25,0x23,0x2F,
  0x28,0x29,0x5B,0x5D,0x7B,0x7D,0x3C,0x3E,0x2329,0x232A,0x3008,0x3009,0x21,0x3F,0x22,0x300,0x5B0,0x64B,0x200E,0x200F,0x61C,0x202A,0x202B,0x202C,0x202D,0x202E,
  0x2066,0x2067,0x2068,0x2069,0x200B,0x200D,0xAD,0x1F600,0x4E2D,0x10800,0x1E900,0x7C0,0x2212,0xA0,0x3000,0xB0,0x20AC,0x600,0x66B,0x66C,0xFB1E,0x1F]
random.seed(int(sys.argv[3]))
lo, hi = int(sys.argv[4]), int(sys.argv[5])
cases = set()
for _ in range(int(sys.argv[2])):
    cases.add("".join(chr(random.choice(alphabet)) for _ in range(random.randint(lo, hi))))
with open(sys.argv[1], "w") as out:
    for text in sorted(cases):
        out.write(" ".join(f"{ord(c):X}" for c in text) + ";" + " ".join(levels(text)) + "\n")
print(len(cases))
