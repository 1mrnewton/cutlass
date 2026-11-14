#!/usr/bin/env python3
"""Writes the word lists Cutlass breaks Thai, Lao, Khmer and Burmese lines by.

    python3 tools/text-dictionaries/generate.py crates/text/src/linebreak/words.bin
    python3 tools/text-dictionaries/generate.py --check     # proves the decoding
    python3 tools/text-dictionaries/generate.py --words DIR # one .txt per language
    python3 tools/text-dictionaries/generate.py --license crates/text/src/linebreak/WORDS-LICENSE.txt

The words are ICU's: its four dictionaries are opened through ICU's own
data API (`udata_open`), each is a byte trie whose bytes are a letter's
place after the first of its script, and every word of the trie is read
out. Needs ICU 78 from Homebrew; change ROOT for another install.

`--check` writes each list back through ICU's `gendict` with the options
the dictionary was built with and compares the trie it makes, byte for
byte, with the one ICU ships.

The file written is read in place by `crates/text/src/linebreak/words.rs`:

    "CLWD"  u32 version = 1  u32 count of languages
    for each language, 20 bytes: u32 first code point of its letters,
        u32 offset of its trie from the start of the file, u32 length of
        the trie, u32 the root's place in the trie, u32 count of words
    the tries

A trie is nodes, each written after everything below it, so the root is
last; a node is found by how far back it starts. Its first byte holds two
flags and a number: bit 7, a word ends here; bit 6, the node is a run.
A node that is not a run has the number of its children in bits 0 to 5
(63 meaning the next byte is the number), then one byte for each child's
letter, in rising order, then for each child a LEB128 number: how many
bytes before this node the child's node starts. A run stands for letters
that can only follow one another with no word ending between them: bits
0 to 5 are how many, the letters follow, and then one LEB128 number for
the node after the last of them. A letter is its place after the first
code point of its script, with 0xFF for the zero width joiner and 0xFE
for the non-joiner, as ICU has them. Nodes with the same words below
them are stored once.

Each language's entry in the header gives the offset of its trie, its
length, and where in it the root starts.
"""
import ctypes
import os
import struct
import subprocess
import sys
import tempfile

ROOT = "/opt/homebrew/opt/icu4c"
LIBRARY = ROOT + "/lib/libicuuc.78.dylib"
PACKAGE = b"icudt78l-brkitr"
GENDICT = ROOT + "/bin/gendict"
# The item, the name of its script, and the first code point its bytes count from.
LANGUAGES = [("thaidict", "Thai", 0x0E00), ("laodict", "Lao", 0x0E80),
             ("khmerdict", "Khmer", 0x1780), ("burmesedict", "Burmese", 0x1000)]


def item(name):
    """The body of one dictionary, after ICU's data header."""
    uc = ctypes.CDLL(LIBRARY)
    opened = uc.udata_open_78
    opened.restype = ctypes.c_void_p
    opened.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    memory = uc.udata_getMemory_78
    memory.restype = ctypes.c_void_p
    memory.argtypes = [ctypes.c_void_p]
    error = ctypes.c_int(0)
    data = opened(PACKAGE, b"dict", name.encode(), ctypes.byref(error))
    if not data or error.value > 0:
        sys.exit("ICU has no %s: error %d" % (name, error.value))
    start = memory(data)
    indexes = struct.unpack("<8i", ctypes.string_at(start, 32))
    return ctypes.string_at(start, indexes[3])


def read_value(trie, at, lead):
    """A value whose lead byte, halved, is `lead`; and where it ends."""
    if lead < 0x51:
        return lead - 0x10, at
    if lead < 0x6C:
        return ((lead - 0x51) << 8) | trie[at], at + 1
    if lead < 0x7E:
        return ((lead - 0x6C) << 16) | (trie[at] << 8) | trie[at + 1], at + 2
    if lead == 0x7E:
        return (trie[at] << 16) | (trie[at + 1] << 8) | trie[at + 2], at + 3
    return (trie[at] << 24) | (trie[at + 1] << 16) | (trie[at + 2] << 8) | trie[at + 3], at + 4


def jump(trie, at):
    """Where a branch's delta at `at` leads."""
    delta = trie[at]
    at += 1
    if delta >= 0xC0:
        if delta < 0xF0:
            delta = ((delta - 0xC0) << 8) | trie[at]
            at += 1
        elif delta < 0xFE:
            delta = ((delta - 0xF0) << 16) | (trie[at] << 8) | trie[at + 1]
            at += 2
        elif delta == 0xFE:
            delta = (trie[at] << 16) | (trie[at + 1] << 8) | trie[at + 2]
            at += 3
        else:
            delta = (trie[at] << 24) | (trie[at + 1] << 16) | (trie[at + 2] << 8) | trie[at + 3]
            at += 4
    return at + delta


def skip_delta(trie, at):
    delta = trie[at]
    at += 1
    if delta >= 0xC0:
        if delta < 0xF0:
            at += 1
        elif delta < 0xFE:
            at += 2
        else:
            at += 3 + (delta & 1)
    return at


def keys(trie):
    """Every key of an ICU byte trie, as bytes, with its value."""
    found = []
    # Places still to read: a node and the bytes that lead to it, or a
    # branch with how many of its entries are left.
    todo = [("node", 0, b"", 0)]
    while todo:
        kind, at, prefix, length = todo.pop()
        if kind == "node":
            while True:
                node = trie[at]
                at += 1
                if node >= 0x20:
                    value, at = read_value(trie, at, node >> 1)
                    found.append((prefix, value))
                    if node & 1:
                        break
                elif node < 0x10:
                    length = node
                    if length == 0:
                        length = trie[at]
                        at += 1
                    todo.append(("branch", at, prefix, length + 1))
                    break
                else:
                    count = node - 0x10 + 1
                    prefix += bytes(trie[at:at + count])
                    at += count
            continue
        while length > 5:
            at += 1
            todo.append(("branch", jump(trie, at), prefix, length >> 1))
            length -= length >> 1
            at = skip_delta(trie, at)
        for _ in range(length - 1):
            byte = bytes([trie[at]])
            node = trie[at + 1]
            value, after = read_value(trie, at + 2, node >> 1)
            if node & 1:
                found.append((prefix + byte, value))
            else:
                todo.append(("node", after + value, prefix + byte, 0))
            at = after
        todo.append(("node", at + 1, prefix + bytes([trie[at]]), 0))
    return found


def words(name, first):
    """The words of one dictionary, sorted, and its trie as ICU has it."""
    body = item(name)
    indexes = struct.unpack("<8i", body[:32])
    if indexes[4] != 0 or indexes[5] != (0x1000000 | first):
        sys.exit("%s is not a byte trie of letters counted from U+%04X" % (name, first))
    trie = body[indexes[0]:indexes[3]]

    def letter(byte):
        # The two joiners have bytes of their own at the top.
        return {0xFF: chr(0x200D), 0xFE: chr(0x200C)}.get(byte) or chr(first + byte)
    found = keys(trie)
    if any(value != 0 for _, value in found):
        sys.exit("%s has values, which this does not keep" % name)
    return sorted("".join(letter(byte) for byte in key) for key, _ in found), trie


def leb(number):
    out = bytearray()
    while True:
        byte = number & 0x7F
        number >>= 7
        if number:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


MOST_IN_A_RUN = 63


def packed(listed, first):
    """Our trie of `listed`, whose letters are bytes counted from `first`,
    and where its root starts."""
    root = {}
    for word in listed:
        node = root
        for character in word:
            code = {0x200D: 0xFF, 0x200C: 0xFE}.get(ord(character), ord(character) - first)
            if not 0 <= code <= 0xFF:
                sys.exit("a letter outside its script's block: U+%04X" % ord(character))
            node = node.setdefault(code, {})
        node[-1] = True
    names = {}

    def name(node):
        """A number that is the same for nodes with the same words below."""
        if -2 not in node:
            key = (bool(node.get(-1)),) + tuple(
                (letter, name(node[letter])) for letter in sorted(k for k in node if k >= 0))
            node[-2] = names.setdefault(key, len(names))
        return node[-2]

    out = bytearray()
    placed = {}

    def place(node):
        """Write `node` after what is below it, once; where it starts."""
        known = placed.get(name(node))
        if known is not None:
            return known
        letters = sorted(key for key in node if key >= 0)
        final = 0x80 if node.get(-1) else 0
        # Letters that can only follow one another, none ending a word.
        run = []
        after = node
        while len(run) < MOST_IN_A_RUN:
            below = [key for key in after if key >= 0]
            if len(below) != 1 or (run and after.get(-1)):
                break
            run.append(below[0])
            after = after[below[0]]
        if len(run) >= 2:
            target = place(after)
            start = len(out)
            out.append(final | 0x40 | len(run))
            out.extend(run)
            out.extend(leb(start - target))
        else:
            targets = [place(node[letter]) for letter in letters]
            start = len(out)
            if len(letters) < 63:
                out.append(final | len(letters))
            else:
                out.append(final | 63)
                out.append(len(letters))
            out.extend(letters)
            for target in targets:
                out.extend(leb(start - target))
        placed[name(node)] = start
        return start

    start = place(root)
    return bytes(out), start


def check():
    """Rebuild each dictionary with ICU's own tool and compare the tries."""
    for name, script, first in LANGUAGES:
        listed, trie = words(name, first)
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, name + ".txt")
            built = os.path.join(folder, name + ".dict")
            with open(source, "w", encoding="utf-8") as out:
                out.write("\n".join(listed) + "\n")
            subprocess.run([GENDICT, "--bytes", "--transform", "offset-0x%04x" % first, source, built],
                           check=True, stdout=subprocess.DEVNULL)
            data = open(built, "rb").read()
            header = struct.unpack("<H", data[:2])[0]
            indexes = struct.unpack("<8i", data[header:header + 32])
            again = data[header + indexes[0]:header + indexes[3]]
        same = again == trie
        print("%s: %d words, ICU's trie %d bytes, rebuilt by gendict %s" % (
            script, len(listed), len(trie), "the same, byte for byte" if same else "DIFFERENT"))
        if not same:
            sys.exit(1)


RULE = "-" * 70


def license_text():
    """Where the lists are from, and the terms ICU's license file gives
    for its data and for the lists that have notices of their own."""
    # A section ends at a line that is the rule alone; the notices have
    # rules of their own inside, behind a comment sign.
    sections = [""]
    for line in open(ROOT + "/LICENSE", encoding="utf-8"):
        if line.rstrip("\n") == RULE:
            sections.append("")
        else:
            sections[-1] += line
    own = [section for section in sections
           if section.lstrip().startswith(("Lao Word Break Dictionary Data", "Burmese Word Break Dictionary Data"))]
    if not sections[0].startswith("UNICODE LICENSE V3") or len(own) != 2:
        sys.exit("ICU's license file is not laid out as this expects")
    # The library's own name says which ICU this is: libicuuc.78.3.dylib.
    version = os.path.basename(os.path.realpath(LIBRARY)).split(".", 1)[1].rsplit(".", 1)[0]
    counts = ", ".join("%s %d" % (script, len(words(name, first)[0])) for name, script, first in LANGUAGES)
    note = (
        "Word lists for breaking lines of Thai, Lao, Khmer and Burmese\n\n"
        "The words in words.bin are those of the four dictionaries of ICU %s\n"
        "(International Components for Unicode, https://github.com/unicode-org/icu):\n"
        "brkitr/thaidict.dict, laodict.dict, khmerdict.dict and burmesedict.dict of\n"
        "its data, read out word for word and stored in another layout by\n"
        "tools/text-dictionaries/generate.py, which also writes this file. No word\n"
        "is added, changed or left out: %s.\n\n"
        "ICU and its data are under the Unicode License V3, which follows. Its\n"
        "license file gives the Lao and the Burmese list notices of their own,\n"
        "which follow it; the Thai and the Khmer list have none apart from ICU's.\n\n" % (version, counts))
    return note + RULE + "\n\n" + sections[0] + RULE + "\n" + own[0] + RULE + "\n" + own[1]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--license":
        with open(sys.argv[2], "w", encoding="utf-8") as out:
            out.write(license_text())
        return
    if sys.argv[1] == "--check":
        check()
        return
    if sys.argv[1] == "--words":
        os.makedirs(sys.argv[2], exist_ok=True)
        for name, script, first in LANGUAGES:
            listed, _ = words(name, first)
            with open(os.path.join(sys.argv[2], script.lower() + ".txt"), "w", encoding="utf-8") as out:
                out.write("\n".join(listed) + "\n")
            print("%s: %d words" % (script, len(listed)))
        return
    tries = []
    for name, script, first in LANGUAGES:
        listed, _ = words(name, first)
        trie, root = packed(listed, first)
        tries.append((first, trie, root, len(listed)))
        print("%s: %d words, %d bytes" % (script, len(listed), len(trie)))
    out = bytearray(b"CLWD" + struct.pack("<II", 1, len(tries)))
    at = len(out) + 20 * len(tries)
    for first, trie, root, count in tries:
        out.extend(struct.pack("<IIIII", first, at, len(trie), root, count))
        at += len(trie)
    for _, trie, _, _ in tries:
        out.extend(trie)
    with open(sys.argv[1], "wb") as file:
        file.write(out)
    print("%s: %d bytes" % (sys.argv[1], len(out)))


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    main()
