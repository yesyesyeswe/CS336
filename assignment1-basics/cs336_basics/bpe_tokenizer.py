from cs336_basics.pre_tokenizer import pre_tokenize, WordState
from collections import defaultdict
from dataclasses import dataclass, field

import os
from pathlib import Path


def decode_tuple_bytes_to_str(tuple_bytes: tuple[bytes]):
    return (b"".join(tuple_bytes)).decode("utf-8")


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
class BPETokenizer:
    input_path: str | os.PathLike
    vocab_size: int
    special_tokens: list[str]
    pretoken: list[WordState] = field(init=False)
    pair_context_dict: dict[tuple[bytes, bytes], PairContext] = field(
        default_factory=lambda: defaultdict(PairContext), init=False
    )
    vocab: dict[int, bytes] = field(default_factory=dict, init=False)
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

    def merge_pair(self):
        merged_pair, pair_ctx = max(self.pair_context_dict.items(), key=lambda x: (x[1].count, x[0]))

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

        self.pair_context_dict.pop(merged_pair, None)

        return merged_pair

    def train(self):
        # Initial the vocabulary
        self.initial_vocab()

        # PreTokenize
        self.pretoken = pre_tokenize(self.input_path, 8, self.special_tokens)

        # Merge the pair
        ## Initial the pair context
        self.initial_pair_context()

        ## Merge pair
        merge_list = []
        length = self.vocab_size - self.vocab_count
        for _ in range(length):
            merged_pair = self.merge_pair()
            merge_list.append(merged_pair)
            self.vocab[self.vocab_count] = b"".join(merged_pair)
            self.vocab_count += 1

        return self.vocab, merge_list


if __name__ == "__main__":
    current_dir = Path(__file__).parent.parent
    input_path = current_dir / "LICENSE"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 256 + len(special_tokens) + 10

    bpe_tokenize = BPETokenizer(input_path, vocab_size, special_tokens)

    vocab, merge_list = bpe_tokenize.train()

    for merge_pair in merge_list:
        print(decode_tuple_bytes_to_str(merge_pair))
