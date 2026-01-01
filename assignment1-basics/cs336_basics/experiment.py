from cs336_basics.bpe_tokenizer import BPETokenizer
from pathlib import Path
import time
import numpy as np

current_dir = Path(__file__).parent.parent
tokenizer_dir = current_dir / "tokenizer"
data_dir = current_dir / "data"

owt_vocab_path = tokenizer_dir / "owt_valid_vocab.json"
owt_merges_path = tokenizer_dir / "owt_valid_merges.json"
owt_data_path = data_dir / "owt_small.txt"
TinyStories_vocab_path = tokenizer_dir / "TinyStoriesV2-GPT4-valid_vocab.json"
TinyStories_merges_path = tokenizer_dir / "TinyStoriesV2-GPT4-valid_merges.json"
TinyStories_data_path = data_dir / "TinyStories_small.txt"
special_tokens = ["<|endoftext|>"]


owt_tokenizer = BPETokenizer.from_files(owt_vocab_path, owt_merges_path, special_tokens)
TinyStories_tokenizer = BPETokenizer.from_files(TinyStories_vocab_path, TinyStories_merges_path, special_tokens)


def find_longest_token(tokenizer, name):
    longest_token = None
    max_len = 0
    for token_bytes in tokenizer.vocab.values():
        current_len = len(token_bytes)
        if current_len > max_len:
            max_len = current_len
            longest_token = token_bytes

    print(f"[{name}] Longest token length: {max_len} bytes")
    print(f"[{name}] Longest token content: {longest_token}")
    try:
        print(f"[{name}] Longest token decoded: {longest_token.decode('utf-8')}")
    except Exception:
        print(f"[{name}] Longest token (raw hex): {longest_token.hex()}")


print("=================== Experiment 1: Longest token ===================")
find_longest_token(owt_tokenizer, "OWT")
find_longest_token(TinyStories_tokenizer, "TinyStories")
"""
=================== Experiment 1: Longest token ===================
[OWT] Longest token length: 19 bytes
[OWT] Longest token content: b' disproportionately'
[OWT] Longest token decoded:  disproportionately
[TinyStories] Longest token length: 15 bytes
[TinyStories] Longest token content: b' accomplishment'
[TinyStories] Longest token decoded:  accomplishment
"""

print("\n================ Experiment 2: Vocabulary Overlap ================")
owt_vocab_set = set(owt_tokenizer.vocab.values())
ts_vocab_set = set(TinyStories_tokenizer.vocab.values())

intersection = owt_vocab_set.intersection(ts_vocab_set)
print(f"OWT Vocab Size: {len(owt_vocab_set)}")
print(f"TinyStories Vocab Size: {len(ts_vocab_set)}")
print(f"Shared Tokens: {len(intersection)}")
print(f"Overlap Percentage: {len(intersection) / len(owt_vocab_set) * 100:.2f}%")
"""
OWT Vocab Size: 32000
TinyStories Vocab Size: 10000
Shared Tokens: 7096
Overlap Percentage: 22.18%
"""

print("\n========== Experiment 3 & 4: Compression & Cross-Domain ===========")


def run_compression_test(tokenizer, data_path, tokenizer_name, data_name):
    with open(data_path, encoding="utf-8") as f:
        text = f.read()

    encoded = tokenizer.encode(text)
    num_tokens = len(encoded)
    original_bytes = len(text.encode("utf-8"))
    compression_ratio = original_bytes / num_tokens

    print(f"[{tokenizer_name} on {data_name}] Tokens: {num_tokens}")
    print(f"[{tokenizer_name} on {data_name}] Compression Ratio: {compression_ratio:.3f} bytes/token")


run_compression_test(owt_tokenizer, owt_data_path, "OWT_Tokenizer", "OWT_Data")
run_compression_test(owt_tokenizer, TinyStories_data_path, "OWT_Tokenizer", "TinyStories_Data")
run_compression_test(TinyStories_tokenizer, TinyStories_data_path, "TS_Tokenizer", "TinyStories_Data")
run_compression_test(TinyStories_tokenizer, owt_data_path, "TS_Tokenizer", "OWT_Data")
"""
[OWT_Tokenizer on OWT_Data] Tokens: 11203
[OWT_Tokenizer on OWT_Data] Compression Ratio: 4.516 bytes/token
[OWT_Tokenizer on TinyStories_Data] Tokens: 2800
[OWT_Tokenizer on TinyStories_Data] Compression Ratio: 3.896 bytes/token
[TS_Tokenizer on TinyStories_Data] Tokens: 2698
[TS_Tokenizer on TinyStories_Data] Compression Ratio: 4.043 bytes/token
[TS_Tokenizer on OWT_Data] Tokens: 14991
[TS_Tokenizer on OWT_Data] Compression Ratio: 3.375 bytes/token
"""

print("\n=================== Experiment 5: Throughput ===================")


def run_throughput_test(tokenizer, data_path, tokenizer_name):
    with open(data_path, encoding="utf-8") as f:
        text = f.read()

    num_runs = 5
    start_time = time.time()
    for _ in range(num_runs):
        _ = tokenizer.encode(text)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_runs
    total_bytes = len(text.encode("utf-8"))

    throughput_bps = total_bytes / avg_time
    throughput_mbps = throughput_bps / (1024 * 1024)

    print(f"[{tokenizer_name}] Throughput: {throughput_mbps:.2f} MB/s")

    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * 1024 * 1024 * 1024
    estimated_seconds = pile_size_bytes / throughput_bps
    estimated_hours = estimated_seconds / 3600

    print(f"[{tokenizer_name}] Estimated time for Pile (825GB): {estimated_hours:.2f} hours")


run_throughput_test(owt_tokenizer, owt_data_path, "OWT_Tokenizer")
run_throughput_test(TinyStories_tokenizer, TinyStories_data_path, "TS_Tokenizer")
"""
[OWT_Tokenizer] Throughput: 0.39 MB/s
[OWT_Tokenizer] Estimated time for Pile (825GB): 600.14 hours
[TS_Tokenizer] Throughput: 0.42 MB/s
[TS_Tokenizer] Estimated time for Pile (825GB): 555.43 hours
"""

print("\n================ Experiment 6: Dataset Encoding ================")
owt_path = data_dir / "owt_small.txt"
ts_path = data_dir / "TinyStories_small.txt"


def encode_and_save(tokenizer, input_path, output_path):
    print(f"Encoding {input_path.name}...")
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text)
    # uint16 is appropriate because:
    # 1. Our vocabulary sizes (OWT ~32k, TS ~10k) fit within the uint16 range (0-65,535).
    # 2. It saves 50%-75% of storage and memory compared to uint32 or int64,
    #    which is crucial for large-scale datasets.
    ids_array = np.array(ids, dtype=np.uint16)
    np.save(output_path, ids_array)
    print(f"Saved to {output_path} (Shape: {ids_array.shape}, Dtype: {ids_array.dtype})")


encode_and_save(owt_tokenizer, owt_path, data_dir / "owt_small_train.npy")
encode_and_save(TinyStories_tokenizer, ts_path, data_dir / "ts_small_train.npy")
