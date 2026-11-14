#!/usr/bin/env python3
"""Writes crates/text/src/unicode/data.bin from the Unicode data in ICU.

Run from the repository root on a machine with ICU installed:

    python3 tools/unicode-tables/generate.py /opt/homebrew/opt/icu4c/lib

The argument is the folder holding libicuuc. The build does not run this
script and does not link ICU; the table it writes is checked in. Run it
again when a newer Unicode version should be taken, and keep the constants
in crates/text/src/unicode/mod.rs equal to the numbering ICU uses.

The file is little endian:

    "CUTU"  u16 version  u16 block bits
    u32 counts: stage one, stage two, records, scripts, mirrors, brackets,
                decompositions, compositions
    u8  Unicode version: major, minor, patch, 0
    u16 stage one   an index of a block for each run of code points
    u16 stage two   a record index for each code point of each block
    u64 records     the properties, packed as `pack` packs them below
    u32 scripts     ISO 15924 tags, four bytes each
    u32 pairs       mirrors: a character and its mirror image
    u32 pairs       brackets: an opening bracket and its closing one
    u32 triples     decompositions: a character and its one or two parts
    u32 triples     compositions: two parts and the character they make
"""

import ctypes
import glob
import os
import struct
import sys

VERSION = 1
BLOCK_BITS = 6
LAST = 0x10FFFF

UCHAR_DEFAULT_IGNORABLE = 5
UCHAR_EMOJI = 57
UCHAR_EMOJI_PRESENTATION = 58
UCHAR_EMOJI_MODIFIER = 59
UCHAR_REGIONAL_INDICATOR = 62
UCHAR_EXTENDED_PICTOGRAPHIC = 64
UCHAR_EAST_ASIAN_WIDTH = 0x1004
UCHAR_JOINING_TYPE = 0x1007
UCHAR_LINE_BREAK = 0x1008
UCHAR_GRAPHEME_CLUSTER_BREAK = 0x1012
UCHAR_BIDI_PAIRED_BRACKET_TYPE = 0x1015
UCHAR_INDIC_POSITIONAL_CATEGORY = 0x1016
UCHAR_INDIC_SYLLABIC_CATEGORY = 0x1017


def load(folder):
    paths = [
        path
        for path in sorted(glob.glob(os.path.join(folder, "libicuuc.*")), key=len)
        if not path.endswith(".a")
    ]
    if not paths:
        sys.exit(f"no libicuuc in {folder}")
    library = ctypes.CDLL(paths[0])
    version = None
    for number in range(60, 120):
        if hasattr(library, f"u_charType_{number}"):
            version = number
            break
    suffix = "" if version is None else f"_{version}"

    def function(name, result, *arguments):
        found = getattr(library, name + suffix)
        found.restype = result
        found.argtypes = list(arguments)
        return found

    return function


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    function = load(sys.argv[1])
    c_int, c_int8, c_uint8 = ctypes.c_int32, ctypes.c_int8, ctypes.c_uint8
    error = ctypes.c_int(0)
    char_type = function("u_charType", c_int8, c_int)
    direction = function("u_charDirection", c_int, c_int)
    int_property = function("u_getIntPropertyValue", c_int, c_int, c_int)
    binary = function("u_hasBinaryProperty", c_int8, c_int, c_int)
    combining = function("u_getCombiningClass", c_uint8, c_int)
    script_of = function("uscript_getScript", c_int, c_int, ctypes.POINTER(ctypes.c_int))
    script_name = function("uscript_getShortName", ctypes.c_char_p, c_int)
    mirror = function("u_charMirror", c_int, c_int)
    bracket = function("u_getBidiPairedBracket", c_int, c_int)
    unicode_version = function("u_getUnicodeVersion", None, ctypes.POINTER(c_uint8 * 4))
    nfc = function("unorm2_getNFCInstance", ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
    raw = function(
        "unorm2_getRawDecomposition",
        c_int,
        ctypes.c_void_p,
        c_int,
        ctypes.POINTER(ctypes.c_uint16),
        c_int,
        ctypes.POINTER(ctypes.c_int),
    )
    compose = function("unorm2_composePair", c_int, ctypes.c_void_p, c_int, c_int)

    version = (c_uint8 * 4)()
    unicode_version(ctypes.byref(version))
    normalizer = nfc(ctypes.byref(error))

    # Common, inherited and unknown keep fixed places.
    scripts = [b"Zyyy", b"Zinh", b"Zzzz"]
    script_index = {}

    def script(code_point):
        code = script_of(code_point, ctypes.byref(error))
        if code not in script_index:
            tag = script_name(code)
            if tag not in scripts:
                scripts.append(tag)
            script_index[code] = scripts.index(tag)
        return script_index[code]

    def pack(code_point):
        fields = (
            (char_type(code_point), 5),
            (direction(code_point), 5),
            (int_property(code_point, UCHAR_JOINING_TYPE), 3),
            (int_property(code_point, UCHAR_LINE_BREAK), 6),
            (int_property(code_point, UCHAR_EAST_ASIAN_WIDTH), 3),
            (binary(code_point, UCHAR_EXTENDED_PICTOGRAPHIC), 1),
            (combining(code_point), 8),
            (script(code_point), 8),
            (int_property(code_point, UCHAR_INDIC_SYLLABIC_CATEGORY), 6),
            (int_property(code_point, UCHAR_INDIC_POSITIONAL_CATEGORY), 4),
            (int_property(code_point, UCHAR_GRAPHEME_CLUSTER_BREAK), 5),
            (int_property(code_point, UCHAR_BIDI_PAIRED_BRACKET_TYPE), 2),
            (binary(code_point, UCHAR_DEFAULT_IGNORABLE), 1),
            (binary(code_point, UCHAR_EMOJI_PRESENTATION), 1),
            (binary(code_point, UCHAR_EMOJI), 1),
            (binary(code_point, UCHAR_EMOJI_MODIFIER), 1),
            (binary(code_point, UCHAR_REGIONAL_INDICATOR), 1),
        )
        packed, shift = 0, 0
        for value, bits in fields:
            value = int(value) & 0xFF if bits == 8 else int(value)
            if value < 0 or value >= 1 << bits:
                sys.exit(f"U+{code_point:04X}: {value} does not fit {bits} bits")
            packed |= value << shift
            shift += bits
        return packed

    records, record_index, values = [], {}, []
    mirrors, brackets, parts, pairs = [], [], [], []
    buffer = (ctypes.c_uint16 * 8)()
    for code_point in range(LAST + 1):
        packed = pack(code_point)
        if packed not in record_index:
            record_index[packed] = len(records)
            records.append(packed)
        values.append(record_index[packed])
        mirrored = mirror(code_point)
        if mirrored != code_point:
            mirrors.append((code_point, mirrored))
        if int_property(code_point, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == 1:
            brackets.append((code_point, bracket(code_point)))
        # Hangul syllables are composed by arithmetic and left out here.
        if 0xAC00 <= code_point <= 0xD7A3:
            continue
        error.value = 0
        length = raw(normalizer, code_point, buffer, 8, ctypes.byref(error))
        if length > 0:
            text = bytes(buffer)[: length * 2].decode("utf-16-le")
            points = [ord(character) for character in text]
            if len(points) > 2:
                sys.exit(f"U+{code_point:04X} has {len(points)} parts")
            first, second = points[0], points[1] if len(points) == 2 else 0
            parts.append((code_point, first, second))
            if second and compose(normalizer, first, second) == code_point:
                pairs.append((first, second, code_point))
    pairs.sort()

    size = 1 << BLOCK_BITS
    blocks, block_index, stage_one = [], {}, []
    for start in range(0, LAST + 1, size):
        block = tuple(values[start : start + size])
        if block not in block_index:
            block_index[block] = len(blocks)
            blocks.append(block)
        stage_one.append(block_index[block])
    stage_two = [value for block in blocks for value in block]
    if len(records) > 0xFFFF or len(blocks) > 0xFFFF or len(scripts) > 0xFF:
        sys.exit("an index does not fit")

    out = bytearray(b"CUTU")
    out += struct.pack("<HH", VERSION, BLOCK_BITS)
    out += struct.pack(
        "<8I",
        len(stage_one),
        len(stage_two),
        len(records),
        len(scripts),
        len(mirrors),
        len(brackets),
        len(parts),
        len(pairs),
    )
    out += bytes(version)
    out += struct.pack(f"<{len(stage_one)}H", *stage_one)
    out += struct.pack(f"<{len(stage_two)}H", *stage_two)
    out += struct.pack(f"<{len(records)}Q", *records)
    for tag in scripts:
        out += tag
    for table in (mirrors, brackets, parts, pairs):
        for row in table:
            out += struct.pack(f"<{len(row)}I", *row)
    target = os.path.join("crates", "text", "src", "unicode", "data.bin")
    with open(target, "wb") as file:
        file.write(out)
    print(
        f"Unicode {version[0]}.{version[1]}.{version[2]}: {len(out)} bytes, "
        f"{len(blocks)} blocks, {len(records)} records, {len(scripts)} scripts, "
        f"{len(mirrors)} mirrors, {len(brackets)} brackets, "
        f"{len(parts)} decompositions, {len(pairs)} compositions"
    )


if __name__ == "__main__":
    main()
