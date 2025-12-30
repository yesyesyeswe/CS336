from cs336_basics.pre_tokenizer import pre_tokenize, pre_tokenize_from_str, WordState
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import os
from pathlib import Path
import json
import heapq
import regex as re


def decode_tuple_bytes_to_str(tuple_bytes: tuple[bytes]):
    return (b"".join(tuple_bytes)).decode("utf-8")


class ReverseBytes:
    def __init__(self, data):
        self.data = data

    def __lt__(self, other):
        # reverse order in heapq
        return self.data > other.data


@dataclass
class PairContext:
    context: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))  # Word index, Character index
    count: int = 0  # Total Count

    def add(self, index: tuple[int, int]):
        w_i, c_i = index
        # in the case word: "iii", we just record pair (i,i)
        # since the merge of the previous (i,i) will destroy the latter one
        if w_i in self.context and c_i - 1 in self.context[w_i]:
            return
        self.context[w_i].add(c_i)

    def remove(self, index: tuple[int, int]):
        w_i, c_i = index
        self.context[w_i].discard(c_i)
        if len(self.context[w_i]) == 0:
            del self.context[w_i]

    def get_cindex(self, w_index):
        assert w_index in self.context
        return self.context[w_index]

    def items(self):
        return self.context.items()

    def __iter__(self):
        return iter(self.context)

    def __contains__(self, word_idx):
        return word_idx in self.context


@dataclass
class BPETrainer:
    input_path: str | os.PathLike
    vocab_size: int
    special_tokens: list[str]
    isTrain: bool = True
    pretoken: list[WordState] = field(init=False)
    pair_context_dict: dict[tuple[bytes, bytes], PairContext] = field(
        default_factory=lambda: defaultdict(PairContext), init=False
    )
    pair_queue: list = field(default_factory=list, init=False)
    updated_pairs: set[tuple[bytes, bytes]] = field(default_factory=set, init=False)
    vocab: dict[int, bytes] = field(default_factory=dict, init=False)
    merge_list: list[tuple[bytes, bytes]] = field(default_factory=list, init=False)
    vocab_count: int = field(init=False)

    def initial_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab_count = 256
        for token in self.special_tokens:
            self.vocab[self.vocab_count] = token.encode("utf-8")
            self.vocab_count += 1

    def initial_pair_context(self):
        for index, wordstate in enumerate(self.pretoken):
            token = wordstate.token
            if len(token) == 1:
                continue
            for i in range(0, len(token) - 1):
                pair = (token[i], token[i + 1])
                self.pair_context_dict[pair].add((index, i))
                self.pair_context_dict[pair].count += wordstate.count

        if self.isTrain:
            for pair, pair_ctx in self.pair_context_dict.items():
                heapq.heappush(self.pair_queue, (-pair_ctx.count, ReverseBytes(pair)))

    def update_context_dict(self, merged_pair, w_index, c_indexes, isNextNotPrev: bool):
        word = self.pretoken[w_index]
        token = word.token

        for i, c_index in enumerate(c_indexes):
            if not isNextNotPrev:
                if c_index == 0:
                    continue
                # if the merged pair is adjacent, merge once
                if i > 0 and c_indexes[i] - c_indexes[i - 1] == 2:
                    continue

                new_pair = (token[c_index - 1], b"".join(merged_pair))
                new_c_index = c_index - 1 - i

                old_pair = (token[c_index - 1], token[c_index])
                old_c_index = c_index - 1
            else:
                if c_index + 2 == len(token):
                    return
                # if the merged pair is adjacent, merge once
                if i < len(c_indexes) - 1 and c_indexes[i + 1] - c_indexes[i] == 2:
                    new_pair = (b"".join(merged_pair), b"".join(merged_pair))
                else:
                    new_pair = (b"".join(merged_pair), token[c_index + 2])
                new_c_index = c_index - i
                old_pair = (token[c_index + 1], token[c_index + 2])
                old_c_index = c_index + 1

            if old_pair != merged_pair:
                self.updated_pairs.add(new_pair)
                self.updated_pairs.add(old_pair)

            # Update New Pair
            new_pair_ctx = self.pair_context_dict[new_pair]
            new_pair_ctx.count += word.count
            new_pair_ctx.add((w_index, new_c_index))

            # Update Old Pair
            assert old_pair in self.pair_context_dict, f"{old_pair}"
            old_pair_ctx = self.pair_context_dict[old_pair]
            old_pair_ctx.count -= word.count

            if old_pair_ctx.count == 0:
                del self.pair_context_dict[old_pair]
            elif old_pair_ctx.count > 0:
                old_pair_ctx.remove((w_index, old_c_index))
            else:
                assert False, "count should not small than 0!"

    def update_follow_pairs(self, w_index, c_indexes):
        word = self.pretoken[w_index]
        token = word.token

        for k in range(len(c_indexes)):
            begin = c_indexes[k] + 2
            if begin + 1 >= len(token):
                return

            end = len(token) - 1 if k == len(c_indexes) - 1 else c_indexes[k + 1] - 1

            for i in range(begin, end):
                follow_pair = (token[i], token[i + 1])
                follow_pair_ctx = self.pair_context_dict[follow_pair]
                follow_pair_ctx.remove((w_index, i))
                follow_pair_ctx.add((w_index, i - 1 - k))

    def get_token(self, w_index, c_indexes, merged_pair):
        word = self.pretoken[w_index]
        merged_bytes = b"".join(merged_pair)
        old_token = word.token
        new_tokens = []

        # Align with the c_indexes
        extended_indexes = [-2] + c_indexes

        for i in range(len(extended_indexes)):
            if i == len(extended_indexes) - 1:
                new_tokens += old_token[extended_indexes[i] + 2 :]
            else:
                new_tokens += old_token[extended_indexes[i] + 2 : extended_indexes[i + 1]] + [merged_bytes]

        assert c_indexes != []

        return new_tokens

    def merge_pair(self, merged_pair: str = ""):
        if self.isTrain:
            while True:
                nge_count, merged_pair = heapq.heappop(self.pair_queue)
                merged_pair = merged_pair.data
                if -nge_count == self.pair_context_dict[merged_pair].count:
                    break

        pair_ctx = self.pair_context_dict[merged_pair]

        # In case pair_ctx change
        for w_index, c_indexes in list(pair_ctx.items()):
            if w_index not in pair_ctx:
                continue

            sorted_indexes = sorted(c_indexes)
            self.update_context_dict(merged_pair, w_index, sorted_indexes, False)
            self.update_context_dict(merged_pair, w_index, sorted_indexes, True)
            self.update_follow_pairs(w_index, sorted_indexes)

            self.pretoken[w_index] = WordState(
                token=self.get_token(w_index, sorted_indexes, merged_pair), count=self.pretoken[w_index].count
            )

        if self.isTrain:
            for pair in self.updated_pairs:
                if pair not in self.pair_context_dict:
                    continue
                count = self.pair_context_dict[pair].count
                heapq.heappush(self.pair_queue, (-count, ReverseBytes(pair)))

            self.updated_pairs.clear()

        self.pair_context_dict.pop(merged_pair, None)

        return merged_pair

    def train(self):
        # Initial the vocabulary
        self.initial_vocab()

        # PreTokenize
        self.pretoken = pre_tokenize(self.input_path, 12, self.special_tokens)

        # Merge the pair
        ## Initial the pair context
        self.initial_pair_context()

        ## Merge pair
        self.merge_list = []
        length = self.vocab_size - self.vocab_count
        for _ in range(length):
            merged_pair = self.merge_pair()
            self.merge_list.append(merged_pair)
            self.vocab[self.vocab_count] = b"".join(merged_pair)
            self.vocab_count += 1

        return self.vocab, self.merge_list

    def pre_encode(self, pretoken, merges, vocab):
        # Prepare for encode
        self.isTrain = False
        self.merge_list = merges
        self.vocab = vocab
        self.pretoken = pretoken

        # Initial the pair context
        self.initial_pair_context()

        # Merge pair
        for merged_pair in self.merge_list:
            self.merge_pair(merged_pair)

        return self.pretoken

    def save(self, vocab_path, merges_path):
        readable_vocab = {k: v.hex() for k, v in self.vocab.items()}
        with open(vocab_path, "w") as f:
            json.dump(readable_vocab, f)

        readable_merges = [(a.hex(), b.hex()) for a, b in self.merge_list]
        with open(merges_path, "w") as f:
            json.dump(readable_merges, f)


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        if not special_tokens:
            self.special_tokens = ["<|endoftext|>"]
        else:
            self.special_tokens = special_tokens
            self.special_tokens.append("<|endoftext|>")

        self.encoder = {v: k for k, v in self.vocab.items()}
        if self.special_tokens:
            for token in self.special_tokens:
                if token not in self.encoder:
                    self.encoder[token] = len(self.vocab)
                    self.vocab[len(self.vocab)] = token

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath) as f:
            _vocab = json.load(f)

        vocab = {int(k): bytes.fromhex(v) for k, v in _vocab.items()}

        with open(merges_filepath) as f:
            _merges = json.load(f)

        merges = [(bytes.fromhex(pair[0]), bytes.fromhex(pair[1])) for pair in _merges]

        bpe_tokenizer = cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

        return bpe_tokenizer

    def encode(self, text: str) -> list[int]:
        self.pretoken = pre_tokenize_from_str(text, 12, self.special_tokens)

        bpe_trainer = BPETrainer("", 0, self.special_tokens)
        bpe_trainer.pretoken = self.pretoken
        pre_encode = bpe_trainer.pre_encode(self.pretoken, self.merges, self.vocab)

        pre_encode_dict = {}
        for wordstate in pre_encode:
            pre_encode_dict[b"".join(wordstate.token)] = [self.encoder[tk] for tk in wordstate.token]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        encode_result = []
        chunk_parts = re.split(re.escape("|".join(self.special_tokens)), text)
        for chunk_part in chunk_parts:
            for match in re.finditer(PAT, chunk_part):
                word = match.group()
                encode_result.extend(pre_encode_dict[word.encode("utf-8")])

        return encode_result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[str]:
        for text in iterable:
            ids = self.encode(text)
            yield from ids

    def decode(self, ids: list[int]) -> str:
        result = b""
        for id in ids:
            word = self.vocab[id]
            result += word

        return result.decode("utf-8", errors="replace")


if __name__ == "__main__":
    current_dir = Path(__file__).parent.parent
    input_path = current_dir / "tests/fixtures/corpus.en"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 256 + len(special_tokens) + (500 - 256 - 1)

    bpe_trainer = BPETrainer(input_path, vocab_size, special_tokens)
    vocab, merge_list = bpe_trainer.train()
    bpe_trainer.save("data/vocab.json", "data/merges.json")

    # for merge_pair in merge_list:
    #     print(decode_tuple_bytes_to_str(merge_pair))
