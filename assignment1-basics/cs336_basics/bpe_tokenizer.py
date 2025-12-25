from cs336_basics.pre_tokenizer import pre_tokenize


def decode_tuple_bytes_to_str(tuple_bytes: tuple[bytes]):
    return (b"".join(max_item[0])).decode("utf-8")


results = pre_tokenize("/home/yesyesyeswe/cs336/assignment1-basics/LICENSE", 4)

max_item = max(results.items(), key=lambda x: (x[1], x[0]))
print(decode_tuple_bytes_to_str(max_item[0]))


def initial_vocab(special_tokens: list[str]):
    vocab = {i: bytes([i]) for i in range(256)}
    size = 256
    for token in special_tokens:
        vocab[size] = token.encode("utf-8")
        size += 1

    return vocab, size


def initial_pair_info(pretoken):
    pair_to_count = {}
    pair_to_prev_pair = {}
    pair_to_next_pair = {}
    for token, count in pretoken.items():
        # i == 0
        pair = (token[0], token[1])
        if len(token) >= 3:
            next_pair = (token[1], token[2])
            pair_to_next_pair[pair] = pair_to_next_pair.get(pair, []).append(next_pair)

        for i in range(1, len(token) - 2):
            pair = (token[i], token[i + 1])
            next_pair = (token[i + 1], token[i + 2])
            prev_pair = (token[i - 1], token[i])
            pair_to_count[pair] = pair_to_count.get(pair, 0) + count
            pair_to_next_pair[pair] = pair_to_next_pair.get(pair, []).append(next_pair)
            pair_to_prev_pair[pair] = pair_to_prev_pair.get(pair, []).append(prev_pair)

        # i == len(token) - 2
        if len(token) >= 3:
            i = len(token) - 2
            pair = (token[i], token[i + 1])
            prev_pair = (token[i - 1], token[i])
            pair_to_prev_pair[pair] = pair_to_prev_pair.get(pair, []).append(prev_pair)

    return pair_to_count, pair_to_prev_pair, pair_to_next_pair


def get_prev_pair(oldpair, newpair, pair_to_prev_pair, pair_to_next_pair):
    prev_pair = []
    oldpair_reserver_prev_pair = []
    oldpair_prev_pair = pair_to_prev_pair[oldpair]
    for pair in oldpair_prev_pair:
        if oldpair in pair_to_next_pair[pair]:
            prev_pair.append(pair)
        else:
            oldpair_reserver_prev_pair.append(pair)
    return prev_pair, oldpair_reserver_prev_pair


def get_next_pair(oldpair, newpair, pair_to_prev_pair, pair_to_next_pair):
    next_pair = []
    oldpair_reserver_next_pair = []
    oldpair_next_pair = pair_to_next_pair[oldpair]
    for pair in oldpair_next_pair:
        if oldpair in pair_to_prev_pair[pair]:
            next_pair.append(pair)
        else:
            oldpair_reserver_next_pair.append(pair)
    return next_pair, oldpair_reserver_next_pair


def process_new_pair(oldpair, newpair, pair_to_count, pair_to_prev_pair, pair_to_next_pair):
    # Update pair_to_count
    pair_to_count[newpair] = pair_to_count.get(newpair, 0) + 1
    pair_to_count[oldpair] = pair_to_count[oldpair] - 1
    if pair_to_count[oldpair] == 0:
        del pair_to_count[oldpair]

    # Update pair_to_prev_pair
    prev_pair, oldpair_reserver_prev_pair = get_prev_pair(oldpair, newpair, pair_to_prev_pair, pair_to_next_pair)
    ## Update oldpair pre pair
    if len(oldpair_reserver_prev_pair) == 0:
        del pair_to_prev_pair[oldpair]
    else:
        pair_to_prev_pair[oldpair] = oldpair_reserver_prev_pair
    ## Update newpair pre pair
    pair_to_prev_pair[newpair] = prev_pair

    # Update pair_to_next_pair
    next_pair, oldpair_reserver_next_pair = get_next_pair(oldpair, newpair, pair_to_prev_pair, pair_to_next_pair)
    ## Update oldpair pre pair
    if len(oldpair_reserver_next_pair) == 0:
        del pair_to_next_pair[oldpair]
    else:
        pair_to_next_pair[oldpair] = oldpair_reserver_next_pair
    ## Update newpair next pair
    pair_to_next_pair[newpair] = newpair

    return pair_to_count, pair_to_prev_pair, pair_to_next_pair


def merge_pair(pair_to_count, pair_to_prev_pair, pair_to_next_pair):
    merged_pair, count = max(pair_to_count.items(), key=lambda x: (x[1], x[0]))
    pair_to_count_delete_keys = [merged_pair]
    for pair in pair_to_prev_pair.get(merged_pair, []):
        new_pair = (pair[0], b"".join(merged_pair))

    for pair in pair_to_next_pair.get(merged_pair, []):
        new_pair = (b"".join(merged_pair), pair[1])
        pair_to_count[new_pair] = pair_to_count.get(new_pair, 0) + 1
        pair_to_count[pair] = pair_to_count[pair] - 1
        if pair_to_count[pair] == 0:
            pair_to_count_delete_keys.append(pair)


def bpe_tokenize(input_path: str, vocab_size: int, special_tokens: list[str]):
    # Initial the vocabulary
    vocab, size = initial_vocab(special_tokens)

    # PreTokenize
    pretoken = pre_tokenize(input_path, 4)

    # Merge the pair
    ## Initial the pair_to_count
    pair_to_count, pair_to_prev_pair, pair_to_next_pair = initial_pair_info(pretoken)

    ## Merge pair
    merge_list = []
    for _ in range(vocab_size - size):
        merged_pair = merge_pair(pair_to_count, pair_to_prev_pair, pair_to_next_pair)
        merge_list.append(merged_pair)

    return vocab, merge_list
