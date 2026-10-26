import os
from collections import defaultdict

from cs336_basics.pretokenize import pretokenize_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Vocab layout: special tokens, then the 256 single bytes, then merges in creation order.
    vocab: dict[int, bytes] = {}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])
    num_merges = max(0, vocab_size - len(vocab))

    if num_processes is None:
        # Process startup costs ~100s of ms on Windows; only parallelize big files.
        big = os.path.getsize(input_path) > 50 * 1024 * 1024
        num_processes = (os.cpu_count() or 1) if big else 1

    pretoken_counts = pretokenize_counts(input_path, special_tokens, num_processes)

    # Each unique pre-token becomes a mutable list of byte tokens, plus its frequency.
    words: list[list[bytes]] = []
    freqs: list[int] = []
    for pretoken, count in pretoken_counts.items():
        words.append([bytes([b]) for b in pretoken.encode("utf-8")])
        freqs.append(count)

    # pair -> weighted count, and pair -> indices of words that (may) contain it.
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for i, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freqs[i]
            pair_to_words[pair].add(i)

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        # Highest count wins; ties go to the lexicographically greater pair.
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        a, b = best
        new_token = a + b
        merges.append(best)
        vocab[len(vocab)] = new_token

        # Only words containing `best` change. Remove their old pair contributions,
        # merge in place, then add back the new pair contributions.
        for i in list(pair_to_words.pop(best, ())):
            word, freq = words[i], freqs[i]
            if len(word) < 2:
                continue

            for pair in zip(word, word[1:]):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            merged: list[bytes] = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                    merged.append(new_token)
                    j += 2
                else:
                    merged.append(word[j])
                    j += 1
            words[i] = merged

            for pair in zip(merged, merged[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(i)

    return vocab, merges
