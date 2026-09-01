#!/usr/bin/env python3
"""Convert talkie's tiktoken tokenizer (vocab.txt + pat_str + specials) into a
HuggingFace `tokenizer.json` that Ollama's MLX runner can load.

Run from the talkie repo so `talkie` + `tiktoken` are importable, e.g.:

    cd ~/Documents/AI/talkie
    uv run --with tokenizers python \
        ~/Documents/dev/ollama-v0.30.2/x/models/talkie/convert_tokenizer.py \
        --vocab ~/models/talkie-1930-13b-it-mlx/vocab.txt \
        --out   ~/models/talkie-1930-13b-it-mlx/tokenizer.json \
        --style it --validate

Verdict-grade test: with --validate it builds talkie's real tiktoken encoder and
the generated HF tokenizer and asserts byte-identical token ids on a battery of
strings (period prose, contractions, digits, unicode, whitespace, chat markers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken  # noqa: F401  (proves we're in the talkie env)
from tiktoken.load import load_tiktoken_bpe

from talkie.tokenizer import (
    BASE_VOCAB_SIZE,
    IT_VOCAB_SIZE,
    _BASE_SPECIAL_TOKENS,
    _IT_SPECIAL_TOKENS,
    _PAT_STR,
    build_tokenizer,
)


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 / ByteLevel reversible byte->unicode map (same one tiktenken &
    HF ByteLevel use). Lets raw bytes live as printable chars in vocab/merges."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()


def tok_to_str(token: bytes) -> str:
    return "".join(BYTE_ENCODER[b] for b in token)


def bpe_split(mergeable: dict[bytes, int], token: bytes, max_rank: int):
    """Re-derive how `token` is built by greedily applying the lowest-rank merge,
    stopping before `max_rank`. Returns the final list of parts (2 for a real
    merge token)."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i in range(len(parts) - 1):
            rank = mergeable.get(parts[i] + parts[i + 1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or min_rank >= max_rank:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def build_tokenizer_json(vocab_path: str, style: str) -> dict:
    raw = load_tiktoken_bpe(vocab_path)  # {token_bytes: rank}

    # talkie drops the merge whose rank == BASE_VOCAB_SIZE-1 and reserves that id
    # for <|endoftext|>; mirror that filter exactly so ids line up.
    cutoff = BASE_VOCAB_SIZE - 1  # 65535
    mergeable = {tok: rank for tok, rank in raw.items() if rank < cutoff}

    # vocab: byte->unicode mapped piece -> id (== rank)
    vocab = {tok_to_str(tok): rank for tok, rank in mergeable.items()}

    # merges: for every multi-byte token, in rank order, recover its 2 parts.
    merges: list[list[str]] = []
    for tok, rank in sorted(mergeable.items(), key=lambda kv: kv[1]):
        if len(tok) == 1:
            continue
        parts = bpe_split(mergeable, tok, max_rank=rank)
        assert len(parts) == 2, f"token {tok!r} (rank {rank}) split into {len(parts)} parts"
        merges.append([tok_to_str(parts[0]), tok_to_str(parts[1])])

    specials = _IT_SPECIAL_TOKENS if style == "it" else _BASE_SPECIAL_TOKENS
    added = [
        {"id": tid, "content": name, "special": True,
         "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
        for name, tid in sorted(specials.items(), key=lambda kv: kv[1])
    ]

    return {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                # talkie's findall regex: matches ARE the pieces -> Isolated.
                {"type": "Split",
                 "pattern": {"Regex": _PAT_STR},
                 "behavior": "Isolated",
                 "invert": False},
                # byte->unicode mapping only; no extra splitting, no prefix space.
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


TEST_STRINGS = [
    "If scientists discover life on other planets,",
    "The electric telegraph is the most marvelous invention of the modern age.",
    "I shouldn't've thought it possible; we'll see, won't we?",
    "In 1912 there were 1,234 motor-cars and 56 aeroplanes.",
    "Café naïve résumé — façade, jalapeño, Zürich.",
    "Multiple    spaces\tand\ttabs\nand\nnewlines.",
    "ALLCAPS lowercase MixedCase camelCase snake_case",
    "Punctuation!?... (parentheses) [brackets] {braces} \"quotes\"",
    "<|user|>What ails the modern young man?<|end|><|assistant|>",
    "The quick brown fox jumps over the lazy dog. 0123456789.",
]


def validate(vocab_path: str, out_path: str, style: str) -> bool:
    from tokenizers import Tokenizer  # lazy: only needed for --validate

    gt = build_tokenizer(vocab_path, style=style)            # talkie's real tiktoken
    hf = Tokenizer.from_file(out_path)                        # our generated json

    print(f"\nVocab sizes: tiktoken n_vocab={gt.n_vocab}, hf={hf.get_vocab_size()} "
          f"(expected {IT_VOCAB_SIZE if style == 'it' else BASE_VOCAB_SIZE})")

    ok = True
    for s in TEST_STRINGS:
        a = gt.encode(s, allowed_special="all")
        b = hf.encode(s).ids
        match = a == b
        ok = ok and match
        flag = "OK " if match else "MISMATCH"
        print(f"[{flag}] {s[:48]!r:50} tk={len(a)} hf={len(b)}")
        if not match:
            print(f"        tiktoken: {a}")
            print(f"        hf:       {b}")
            # show first divergence
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"        first diff at {i}: {x} != {y}  "
                          f"({gt.decode([x])!r} vs {gt.decode([y]) if y < gt.n_vocab else '?'!r})")
                    break
    # round-trip decode check on the trickiest one
    s = TEST_STRINGS[4]
    dec = hf.decode(hf.encode(s).ids)
    print(f"\nDecode round-trip ({s[:24]!r}...): {'OK' if dec == s else f'DIFF -> {dec!r}'}")
    print("\n" + ("ALL ENCODINGS MATCH ✅" if ok else "ENCODING MISMATCHES ❌"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="it", choices=["base", "it"])
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    data = build_tokenizer_json(args.vocab, args.style)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False))
    nmerges = len(data["model"]["merges"])
    nvocab = len(data["model"]["vocab"])
    print(f"Wrote {args.out}: {nvocab} vocab, {nmerges} merges, "
          f"{len(data['added_tokens'])} special tokens")

    if args.validate:
        import sys
        sys.exit(0 if validate(args.vocab, args.out, args.style) else 1)


if __name__ == "__main__":
    main()
