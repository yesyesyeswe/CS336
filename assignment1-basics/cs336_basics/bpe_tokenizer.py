from cs336_basics.pre_tokenizer import pre_tokenize
from collections import defaultdict
from dataclasses import dataclass, field

import os


def decode_tuple_bytes_to_str(tuple_bytes: tuple[bytes]):
    return (b"".join(tuple_bytes)).decode("utf-8")


@dataclass
class PairContext:
    context: dict[tuple[bytes, bytes], int] = field(default_factory=lambda: defaultdict(int))
    count: int = 0  # Total Count


@dataclass
class BPETokenizer:
    input_path: str | os.PathLike
    vocab_size: int
    special_tokens: list[str]
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

        self.kkcount = 4

    def update_pair_context_dict(self, pair, count, prev_bytes, next_bytes):
        pair_context = self.pair_context_dict[pair]
        pair_context.count += count
        pair_ctx = (prev_bytes, next_bytes)
        pair_context.context[pair_ctx] += count

    def initial_pair_context(self, pretoken):
        for token, count in pretoken.items():
            for i in range(0, len(token) - 1):
                pair = (token[i], token[i + 1])
                next_bytes = token[i + 2] if i + 2 < len(token) else None
                prev_bytes = token[i - 1] if i >= 1 else None
                self.update_pair_context_dict(pair, count, prev_bytes, next_bytes)

    def update_second_order_pair(self, p1, p1_prev_next, p1_count, isNextNotPrev: bool, force: bool = False):
        if p1_prev_next[isNextNotPrev] is None:
            return False

        if isNextNotPrev:
            p2 = (p1[1], p1_prev_next[1])
        else:
            p2 = (p1_prev_next[0], p1[0])

        if not force and p1 == p2:
            return True

        p2_ctx = self.pair_context_dict[p2]
        p2_delete_keys = []
        update_p2_prev_nexts = []
        for p2_prev_next, count in p2_ctx.context.items():
            if p2_prev_next[not isNextNotPrev] != p1[not isNextNotPrev]:
                continue
            update_p2_prev_nexts.append(p2_prev_next)
            p2_ctx.context[p2_prev_next] -= p1_count
            if p2_ctx.context[p2_prev_next] == 0:
                p2_delete_keys.append(p2_prev_next)
            elif p2_ctx.context[p2_prev_next] > 0:
                pass
            else:
                assert False, "count should not small than 0!"

        for key in p2_delete_keys:
            p2_ctx.context.pop(key, None)

        for p2_prev_next in update_p2_prev_nexts:
            if isNextNotPrev:
                p2_ctx.context[(b"".join(p1_prev_next[0], p1[0]), p2_prev_next[1])] = p1_count
            else:
                p2_ctx.context[(p2_prev_next[0], b"".join(p1[1], p1_prev_next[1]))] = p1_count

        return False

    def update_context_dict(self, merged_pair, prev_next, isNextNotPrev: bool, same: bool = False):
        if prev_next[isNextNotPrev] is None:
            return False

        if isNextNotPrev:
            old_pair = (merged_pair[1], prev_next[1])
            new_pair = (b"".join(merged_pair), prev_next[1])
        else:
            old_pair = (prev_next[0], merged_pair[0])
            new_pair = (prev_next[0], b"".join(merged_pair))

        assert old_pair != new_pair, "pair could not be the same!"
        assert new_pair != merged_pair, "pair could not be the same!"
        if not same and old_pair == merged_pair:
            return True

        if not isNextNotPrev and merged_pair == (b"h", b"e") and old_pair == (b" t", b"h"):
            print("YesYes!")
            print(f"{new_pair}")

        # Update new pair
        old_pair_ctx = self.pair_context_dict[old_pair]
        new_pair_ctx = self.pair_context_dict[new_pair]
        old_pair_ctx_delete_keys = []
        delete_count = 0
        second_order_pair_info = []
        for old_pair_prev_next, count in old_pair_ctx.context.items():
            if old_pair_prev_next[not isNextNotPrev] != merged_pair[not isNextNotPrev]:
                continue
            if isNextNotPrev:
                new_pair_ctx.context[(prev_next[0], old_pair_prev_next[1])] = count
            else:
                new_pair_ctx.context[(old_pair_prev_next[0], prev_next[1])] = count
            new_pair_ctx.count += count
            delete_count += count
            old_pair_ctx_delete_keys.append(old_pair_prev_next)

            ## Update Second order pair
            same = self.update_second_order_pair(old_pair, old_pair_prev_next, count, isNextNotPrev)
            if same:
                second_order_pair_info.append([old_pair_prev_next, count])

        for old_prev_next, count in second_order_pair_info:
            self.update_second_order_pair(old_pair, old_prev_next, count, isNextNotPrev, True)

        # Update old pair
        if not isNextNotPrev and merged_pair == (b"h", b"e") and old_pair == (b" t", b"h"):
            print(f"delete_count:{delete_count}")
        old_pair_ctx.count -= delete_count
        if old_pair_ctx.count == 0:
            assert old_pair in self.pair_context_dict
            self.pair_context_dict.pop(old_pair, None)
        elif old_pair_ctx.count > 0:
            for old_pair_prev_next in old_pair_ctx_delete_keys:
                old_pair_ctx.context.pop(old_pair_prev_next, None)
        else:
            info = "when deal with "
            info += "Next," if isNextNotPrev else "Prev,"
            info += f" old_pair:{old_pair}, new_pair:{new_pair},"
            info += f" old_pair_ctx.count:{old_pair_ctx.count + new_pair_ctx.count} should not < new_pair_ctx.count:{new_pair_ctx.count}!"
            assert False, info

        return False

    def merge_pair(self):
        merged_pair, pair_ctx = max(self.pair_context_dict.items(), key=lambda x: (x[1].count, x[0]))

        # if self.kkcount:
        #     print(f"{merged_pair}, count={self.pair_context_dict[merged_pair].count}")
        #     #print(pair_ctx.context)
        #     print(f"(b'h', b'e') count :{self.pair_context_dict[(b'h', b'e')].count}")
        #     if (b' t', b'h') in self.pair_context_dict:
        #         print(self.pair_context_dict[(b' t', b'h')].count)

        #     self.kkcount -= 1

        for prev_next, count in pair_ctx.context.items():
            if merged_pair == (b"h", b"e"):
                print(f"prev_next:{prev_next}")
            prev_same = self.update_context_dict(merged_pair, prev_next, False)
            next_same = self.update_context_dict(merged_pair, prev_next, True)

        if prev_same:
            self.update_context_dict(merged_pair, prev_next, False, True)
        if next_same:
            self.update_context_dict(merged_pair, prev_next, True, True)

        self.pair_context_dict.pop(merged_pair, None)

        return merged_pair

    def train(self):
        # Initial the vocabulary
        self.initial_vocab()

        # PreTokenize
        pretoken = pre_tokenize(self.input_path, 8, self.special_tokens)

        # Merge the pair
        ## Initial the pair context
        self.initial_pair_context(pretoken)

        ## Merge pair
        merge_list = []
        for _ in range(self.vocab_size - self.vocab_count):
            merged_pair = self.merge_pair()
            merge_list.append(merged_pair)

        return self.vocab, merge_list


if __name__ == "__main__":
    input_path = "/home/yesyesyeswe/cs336/assignment1-basics/LICENSE"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 256 + len(special_tokens) + 3

    bpe_tokenize = BPETokenizer(input_path, vocab_size, special_tokens)

    vocab, merge_list = bpe_tokenize.train()

    for merge_pair in merge_list:
        print(decode_tuple_bytes_to_str(merge_pair))
