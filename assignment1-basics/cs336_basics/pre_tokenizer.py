import os
from typing import BinaryIO
import regex as re
import multiprocessing
from dataclasses import dataclass


@dataclass
class WordState:
    token: list[bytes, ...]
    count: int = 0


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _pre_tokenize(chunk: str) -> dict[str, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    result = {}
    for match in re.finditer(PAT, chunk):
        word = match.group()
        result[word] = result.get(word, 0) + 1

    return result


def merge_dicts(dict1: dict[str, int], dict2: dict[str, int]):
    for key, value in dict2.items():
        dict1[key] = dict1.get(key, 0) + value
    return dict1


def pre_tokenize(path: str, num_processes: int, special_tokens) -> list[WordState]:
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        count_results = []

        pool = multiprocessing.Pool(processes=num_processes)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_parts = re.split(re.escape("|".join(special_tokens)), chunk)
            for chunk_part in chunk_parts:
                res = pool.apply_async(_pre_tokenize, args=(chunk_part,))
                count_results.append(res)

        pool.close()

        full_result = {}
        for res in count_results:
            full_result = merge_dicts(full_result, res.get())

        word_states = []
        for key, value in full_result.items():
            word_states.append(WordState(token=list(tuple(bytes([b]) for b in key.encode("utf-8"))), count=value))

        return word_states
