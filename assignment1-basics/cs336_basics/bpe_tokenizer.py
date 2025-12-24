from cs336_basics.pre_tokenizer import pre_tokenize


def decode_tuple_bytes_to_str(tuple_bytes: tuple[bytes]):
    return (b"".join(max_item[0])).decode("utf-8")


results = pre_tokenize("/home/yesyesyeswe/cs336/assignment1-basics/LICENSE", 4)

max_item = max(results.items(), key=lambda x: (x[1], x[0]))
print(decode_tuple_bytes_to_str(max_item[0]))
