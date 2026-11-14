#!/usr/bin/env python3
"""Writes line-break cases in Thai, Lao, Khmer and Burmese with ICU's answers.

    python3 tools/text-dictionaries/generate.py --words words
    python3 tools/text-oracles/line_breaks_words.py cases.txt words 4000 [seed [thai,lao|all [words]]]
    CUTLASS_LINEBREAK_CASES=cases.txt cargo test -p cutlass-text --lib -- --ignored another_implementation_agrees

ICU breaks these four scripts by its word lists, and so does Cutlass,
with the same lists. `words` is the folder the first command writes the
lists into. For each script this writes about as many cases as asked
for: sentences of words from the list, the same with spaces, signs,
digits and Latin among them, words with a letter dropped, doubled or
changed, the script's letters and marks at random, short runs of every
length, two of the scripts side by side, and Thai's signs that follow a
word. A list of scripts keeps to those, and `words` after it to
sentences of the lists' words alone. The format is that of `line_breaks.py`: code points in hex, a
semicolon, then the character indices where ICU's line iterator breaks.
Needs ICU 78 from Homebrew; change LIBRARY for another install.
"""
import ctypes
import os
import random
import sys

LIBRARY = "/opt/homebrew/opt/icu4c/lib/libicuuc.78.dylib"
uc = ctypes.CDLL(LIBRARY)
ubrk_open = uc.ubrk_open_78
ubrk_open.restype = ctypes.c_void_p
ubrk_open.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int)]
ubrk_next = uc.ubrk_next_78
ubrk_next.restype = ctypes.c_int32
ubrk_next.argtypes = [ctypes.c_void_p]
ubrk_close = uc.ubrk_close_78
ubrk_close.argtypes = [ctypes.c_void_p]
ubrk_status = uc.ubrk_getRuleStatus_78
ubrk_status.restype = ctypes.c_int32
ubrk_status.argtypes = [ctypes.c_void_p]


HARD = [chr(code) for code in (0x0A, 0x0B, 0x0C, 0x0D, 0x85, 0x2028, 0x2029)]


def breaks(text):
    data = text.encode("utf-16-le")
    count = len(data) // 2
    buffer = ctypes.create_string_buffer(data, len(data) + 2)
    error = ctypes.c_int(0)
    iterator = ubrk_open(2, b"en", buffer, count, ctypes.byref(error))
    places = {}
    unit = 0
    for index, character in enumerate(text):
        places[unit] = index
        unit += 2 if ord(character) > 0xFFFF else 1
    places[unit] = len(text)
    found = []
    while True:
        place = ubrk_next(iterator)
        if place < 0:
            break
        if place != count:
            # ICU gives a break its word lists found the status of the
            # break its rules found after it, so one that says it must
            # be taken is believed only after a character that ends a
            # line.
            at = places[place]
            hard = ubrk_status(iterator) >= 100 and text[at - 1] in HARD
            found.append((at, hard))
    ubrk_close(iterator)
    return found


# Each script's letters and marks the lists are looked up with, and a few
# of its characters that are not: digits and signs.
SCRIPTS = {
    "thai": (list(range(0x0E01, 0x0E3B)) + list(range(0x0E40, 0x0E4F)), [0x0E3F, 0x0E4F, 0x0E50, 0x0E55, 0x0E5A]),
    "lao": ([c for c in range(0x0E81, 0x0EC7) if c not in (0x0E83, 0x0E85, 0x0E8B, 0x0EA4, 0x0EA6, 0x0EBE, 0x0EBF)]
            + [0x0EC8, 0x0EC9, 0x0ECA, 0x0ECB, 0x0ECC, 0x0ECD, 0x0EDC, 0x0EDD, 0x0EDE, 0x0EDF], [0x0ED0, 0x0ED5]),
    "khmer": (list(range(0x1780, 0x17D4)) + [0x17D7, 0x17DC, 0x17DD], [0x17D4, 0x17D5, 0x17DB, 0x17E0, 0x17E5]),
    "burmese": (list(range(0x1000, 0x1040)) + list(range(0x1050, 0x1090)) + list(range(0x109A, 0x10A0))
                + [0xA9E0, 0xA9E5, 0xA9E6, 0xAA60, 0xAA70, 0xAA7B], [0x1040, 0x1045, 0x104A, 0x104B]),
}
OTHERS = [0x20, 0x20, 0x20, 0x2C, 0x2E, 0x21, 0x3F, 0x28, 0x29, 0x22, 0x2D, 0x41, 0x62, 0x63, 0x31, 0x32,
          0x200B, 0x200C, 0x200D, 0x2060, 0xA0, 0x301, 0x0A, 0x4E2D, 0x3002, 0x1F600, 0x2014, 0x2F]
PAIYANNOI = chr(0x0E2F)
MAIYAMOK = chr(0x0E46)


def cases_of(script, lists, wanted, plain):
    letters, signs = SCRIPTS[script]
    words = lists[script]

    def word():
        return random.choice(words)

    def sentence(low, high):
        return [word() for _ in range(random.randint(low, high))]

    def letters_at_random(low, high):
        return "".join(chr(random.choice(letters)) for _ in range(random.randint(low, high)))

    def spoiled(text):
        if not text:
            return text
        at = random.randrange(len(text))
        kind = random.randrange(4)
        if kind == 0:
            return text[:at] + text[at + 1:]
        if kind == 1:
            return text[:at] + text[at] + text[at:]
        if kind == 2:
            return text[:at] + chr(random.choice(letters)) + text[at + 1:]
        return text[:at] + chr(random.choice(letters)) + text[at:]

    def with_others():
        between = ["", "", " ", chr(random.choice(OTHERS)), chr(random.choice(signs))]
        return "".join(part + random.choice(between) for part in sentence(1, 6))

    def with_signs():
        if script != "thai":
            return "".join(sentence(3, 12))
        after = ["", PAIYANNOI, MAIYAMOK, MAIYAMOK * 2, PAIYANNOI + MAIYAMOK]
        return "".join(part + random.choice(after) for part in sentence(1, 5))

    def two_scripts():
        other = lists[random.choice(list(SCRIPTS))]
        return "".join(sentence(1, 3)) + "".join(random.choice(other) for _ in range(random.randint(1, 3)))

    makers = [
        lambda: "".join(sentence(1, 8)),
        lambda: "".join(sentence(2, 4)),
        with_others,
        lambda: "".join(spoiled(part) for part in sentence(1, 5)),
        lambda: spoiled(spoiled("".join(sentence(2, 5)))),
        lambda: letters_at_random(1, 12),
        lambda: letters_at_random(1, 6),
        lambda: "".join(chr(random.choice(letters + OTHERS + signs)) for _ in range(random.randint(1, 14))),
        lambda: word()[:random.randint(1, 5)] + word()[:random.randint(0, 5)],
        two_scripts,
        with_signs,
        lambda: letters_at_random(0, 4) + "".join(sentence(1, 4)) + letters_at_random(0, 4),
        # Long ones: a paragraph of words, a long stretch of letters, and
        # words with stretches no list knows between them.
        lambda: "".join(sentence(10, 40)),
        lambda: letters_at_random(20, 120),
        lambda: "".join(part + letters_at_random(0, 6) for part in sentence(4, 16)),
    ]
    if plain:
        # Only words of the list, one after another.
        makers = [makers[0], makers[1], makers[12]]
    made = set()
    while len(made) < wanted:
        text = random.choice(makers)()
        if text:
            made.add(text)
    return made


def main():
    out, folder = sys.argv[1], sys.argv[2]
    wanted = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else 11)
    only = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] != "all" else list(SCRIPTS)
    plain = len(sys.argv) > 6 and sys.argv[6] == "words"
    lists = {}
    for script in SCRIPTS:
        with open(os.path.join(folder, script + ".txt"), encoding="utf-8") as file:
            lists[script] = [word for word in file.read().split("\n") if word]
    cases = set()
    for script in only:
        cases |= cases_of(script, lists, wanted, plain)
    with open(out, "w", encoding="utf-8") as file:
        for text in sorted(cases):
            found = breaks(text)
            file.write("%s;%s\n" % (
                " ".join("%X" % ord(character) for character in text),
                " ".join("%d%s" % (place, "!" if hard else "") for place, hard in found)))
    print(len(cases))


if __name__ == "__main__":
    main()
