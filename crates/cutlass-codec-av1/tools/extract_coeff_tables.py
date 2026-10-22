#!/usr/bin/env python3
"""Extract AV1 coefficient CDFs and quantizer lookups into Rust.

Source: libaom v3.13.1 (d772e334cc724105040382a977ebb10dfd393293)
  av1/common/token_cdfs.h
  av1/common/quant_common.c

AOM_CDF arguments are the specification's forward cumulative probabilities.
Each row is those values, then the 32768 sentinel, then an adaptation counter
of 0. libaom stores the inverse. This script emits the forward form.

EOB-512 and EOB-1024 tables in libaom carry a second context that the
specification's Default_Eob_Pt_512/1024 tables do not. The context-0 row is
the specification row; the other row is a flat placeholder and is dropped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LIBAOM = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/refs/libaom")
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "src")


def find_initializer(text: str, name: str) -> str:
    idx = text.find(name)
    if idx < 0:
        raise SystemExit(f"missing {name}")
    eq = text.find("=", idx)
    start = text.find("{", eq)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise SystemExit(f"unbalanced {name}")


def expand_cdf_macros(body: str) -> str:
    def repl(match: re.Match[str]) -> str:
        args = []
        for part in match.group(1).replace("\n", " ").split(","):
            part = part.strip()
            if not part:
                continue
            if re.fullmatch(r"\d+", part):
                args.append(int(part))
            elif re.fullmatch(r"\d+\s*\*\s*\d+", part):
                left, right = (int(piece) for piece in part.split("*"))
                args.append(left * right)
            else:
                raise SystemExit(f"unparsed CDF argument {part!r}")
        vals = args + [32768, 0]
        return "[" + ", ".join(str(v) for v in vals) + "]"

    return re.sub(r"AOM_CDF\d+\(([^)]*)\)", repl, body, flags=re.S)


def parse_node(src: str) -> list:
    src = src.strip()
    if src.startswith("["):
        inner = src[1:-1]
        return [int(part) for part in inner.split(",") if part.strip()]
    if not (src.startswith("{") and src.endswith("}")):
        raise SystemExit(f"bad node {src[:40]!r}")
    inner = src[1:-1]
    items: list = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch.isspace() or ch == ",":
            i += 1
            continue
        if ch not in "{[":
            raise SystemExit(f"unexpected {ch!r}")
        close = "}" if ch == "{" else "]"
        depth = 0
        j = i
        while j < len(inner):
            if inner[j] == ch:
                depth += 1
            elif inner[j] == close:
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        items.append(parse_node(inner[i:j]))
        i = j
    return items


def unwrap_singleton_rows(node):
    """Drop the extra braces libaom puts around each CDF row."""
    if node and isinstance(node[0], int):
        return node
    node = [unwrap_singleton_rows(child) for child in node]
    if len(node) == 1 and node[0] and isinstance(node[0][0], int):
        return node[0]
    return node


def shape_of(node) -> tuple:
    if node and isinstance(node[0], int):
        return (len(node),)
    return (len(node),) + shape_of(node[0])


def emit(node, indent: int) -> str:
    pad = " " * indent
    if node and isinstance(node[0], int):
        return pad + "[" + ", ".join(str(v) for v in node) + "]"
    lines = [pad + "["]
    for i, child in enumerate(node):
        comma = "," if i + 1 < len(node) else ""
        lines.append(emit(child, indent + 4) + comma)
    lines.append(pad + "]")
    return "\n".join(lines)


def rust_type(shape: tuple) -> str:
    ty = "u16"
    for dim in reversed(shape):
        ty = f"[{ty}; {dim}]"
    return ty


def load_cdf(text: str, name: str, drop_ctx0: bool = False):
    raw = expand_cdf_macros(find_initializer(text, name))
    tree = unwrap_singleton_rows(parse_node(raw))
    if drop_ctx0:
        # [q][plane][ctx][cdf] -> [q][plane][cdf] using ctx 0.
        tree = [[plane[0] for plane in qctx] for qctx in tree]
    return tree


def load_i16_array(text: str, name: str) -> list[int]:
    body = find_initializer(text, name)
    nums = [int(part) for part in re.findall(r"-?\d+", body)]
    if len(nums) != 256:
        raise SystemExit(f"{name} has {len(nums)} values")
    return nums


def main() -> None:
    token = (LIBAOM / "av1/common/token_cdfs.h").read_text()
    quant = (LIBAOM / "av1/common/quant_common.c").read_text()
    tables = {
        "TXB_SKIP": load_cdf(token, "av1_default_txb_skip_cdfs"),
        "EOB_EXTRA": load_cdf(token, "av1_default_eob_extra_cdfs"),
        "EOB_PT_16": load_cdf(token, "av1_default_eob_multi16_cdfs"),
        "EOB_PT_32": load_cdf(token, "av1_default_eob_multi32_cdfs"),
        "EOB_PT_64": load_cdf(token, "av1_default_eob_multi64_cdfs"),
        "EOB_PT_128": load_cdf(token, "av1_default_eob_multi128_cdfs"),
        "EOB_PT_256": load_cdf(token, "av1_default_eob_multi256_cdfs"),
        "EOB_PT_512": load_cdf(token, "av1_default_eob_multi512_cdfs", drop_ctx0=True),
        "EOB_PT_1024": load_cdf(token, "av1_default_eob_multi1024_cdfs", drop_ctx0=True),
        "DC_SIGN": load_cdf(token, "av1_default_dc_sign_cdfs"),
        "COEFF_BASE": load_cdf(token, "av1_default_coeff_base_multi_cdfs"),
        "COEFF_BASE_EOB": load_cdf(token, "av1_default_coeff_base_eob_multi_cdfs"),
        "COEFF_BR": load_cdf(token, "av1_default_coeff_lps_multi_cdfs"),
    }
    expected = {
        "TXB_SKIP": (4, 5, 13, 3),
        "EOB_EXTRA": (4, 5, 2, 9, 3),
        "EOB_PT_16": (4, 2, 2, 6),
        "EOB_PT_32": (4, 2, 2, 7),
        "EOB_PT_64": (4, 2, 2, 8),
        "EOB_PT_128": (4, 2, 2, 9),
        "EOB_PT_256": (4, 2, 2, 10),
        "EOB_PT_512": (4, 2, 11),
        "EOB_PT_1024": (4, 2, 12),
        "DC_SIGN": (4, 2, 3, 3),
        "COEFF_BASE": (4, 5, 2, 42, 5),
        "COEFF_BASE_EOB": (4, 5, 2, 4, 4),
        "COEFF_BR": (4, 5, 2, 21, 5),
    }
    parts = [
        "//! Generated coefficient CDFs. Do not edit.",
        "//!",
        "//! Regenerate with `tools/extract_coeff_tables.py`.",
        "//! Provenance: libaom v3.13.1 `av1/common/token_cdfs.h`",
        "//! (`d772e334cc724105040382a977ebb10dfd393293`). Forward CDF form.",
        "//! See `NOTICE`.",
        "",
        "#![allow(dead_code)]",
        "",
    ]
    for name, tree in tables.items():
        got = shape_of(tree)
        if got != expected[name]:
            raise SystemExit(f"{name} shape {got} != {expected[name]}")
        parts.append(f"pub static {name}: {rust_type(got)} = {emit(tree, 0).lstrip()};")
        parts.append("")
    if tables["TXB_SKIP"][0][0][0] != [31849, 32768, 0]:
        raise SystemExit("txb_skip[0][0][0] did not match the header")
    (OUT_DIR / "coeff_cdf.rs").write_text("\n".join(parts) + "\n")

    qparts = [
        "//! Quantizer lookup tables.",
        "//!",
        "//! Regenerate with `tools/extract_coeff_tables.py`.",
        "//! Provenance: libaom v3.13.1 `av1/common/quant_common.c`, which matches",
        "//! `Dc_Qlookup` / `Ac_Qlookup` in the specification. See `NOTICE`.",
        "",
    ]
    arrays = [
        ("DC_QLOOKUP_8", "dc_qlookup_QTX"),
        ("DC_QLOOKUP_10", "dc_qlookup_10_QTX"),
        ("DC_QLOOKUP_12", "dc_qlookup_12_QTX"),
        ("AC_QLOOKUP_8", "ac_qlookup_QTX"),
        ("AC_QLOOKUP_10", "ac_qlookup_10_QTX"),
        ("AC_QLOOKUP_12", "ac_qlookup_12_QTX"),
    ]
    for rust_name, c_name in arrays:
        vals = load_i16_array(quant, c_name)
        lines = [f"pub static {rust_name}: [i16; 256] = ["]
        for row in range(0, 256, 8):
            chunk = ", ".join(str(v) for v in vals[row : row + 8])
            lines.append(f"    {chunk},")
        lines.append("];")
        lines.append("")
        qparts.extend(lines)
    (OUT_DIR / "quant.rs").write_text("\n".join(qparts) + "\n")
    print("wrote coeff_cdf.rs and quant.rs")


if __name__ == "__main__":
    main()
