import os
from collections import Counter
from multiprocessing import Pool

import regex

from cs336_basics.pretokenization_example import find_chunk_boundaries

# GPT-2 pre-tokenizer pattern (from the handout)
PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def count_chunk(path: str | os.PathLike, start: int, end: int, special_tokens: list[str]) -> Counter[str]:
    """Worker: read bytes [start, end), drop special tokens, return local pre-token counts."""
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # Split on special tokens so they're never counted and no pre-token spans a document boundary.
    # Longest first so overlapping special tokens prefer the longest match.
    if special_tokens:
        split_pat = "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        pieces = regex.split(split_pat, text)
    else:
        pieces = [text]

    counts: Counter[str] = Counter()
    for piece in pieces:
        counts.update(m.group() for m in PAT.finditer(piece))
    return counts


def pretokenize_counts(
    path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 1,
) -> Counter[str]:
    """Parent: find boundaries, count each chunk (serially or in a pool), merge the maps."""
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    jobs = [(path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]

    if num_processes <= 1:
        results = [count_chunk(*job) for job in jobs]
    else:
        with Pool(num_processes) as pool:
            results = pool.starmap(count_chunk, jobs)

    total: Counter[str] = Counter()
    for r in results:
        total.update(r)  # "add if present, else insert"
    return total


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    special = ["<|endoftext|>"]
    serial = pretokenize_counts(path, special, num_processes=1)
    parallel = pretokenize_counts(path, special, num_processes=4)
    print("unique pre-tokens:", len(serial), "total:", sum(serial.values()))
    print("serial == parallel:", serial == parallel)
    print("top 10:", serial.most_common(10))
