from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.train import run_train
import optuna
from pathlib import Path
import time
import numpy as np

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cs336_basics.utils import SGD, AdamW, TFLM, load_checkpoint
from cs336_basics.decodeLM import generate
import os

# Control variable to select the experiment to run
# 1: Longest token
# 2: Vocabulary Overlap
# 3: Compression & Cross-Domain (3 & 4)
# 5: Throughput
# 6: Dataset Encoding
# 7: Transformer Memory & FLOPs Calculator
# 8: GPT-2 XL Context Length Scaling
# 9: SGD Learning Rate Comparison
# 10: Hyperparameter Optimization
# 11: LLM Decoding
EXPERIMENT_TYPE = 11

current_dir = Path(__file__).parent.parent
tokenizer_dir = current_dir / "tokenizer"
data_dir = current_dir / "data"

owt_vocab_path = tokenizer_dir / "owt_valid_vocab.json"
owt_merges_path = tokenizer_dir / "owt_valid_merges.json"
owt_data_path = data_dir / "owt_small.txt"
TinyStories_vocab_path = tokenizer_dir / "TinyStoriesV2-GPT4-train_vocab.json"
TinyStories_merges_path = tokenizer_dir / "TinyStoriesV2-GPT4-train_merges.json"
TinyStories_train_data_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
TinyStories_valid_data_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"
special_tokens = ["<|endoftext|>"]

# Initialize tokenizers only if needed for tokenizer-related experiments
if EXPERIMENT_TYPE in [1, 2, 3, 4, 5, 6]:
    owt_tokenizer = BPETokenizer.from_files(owt_vocab_path, owt_merges_path, special_tokens)
    TinyStories_tokenizer = BPETokenizer.from_files(TinyStories_vocab_path, TinyStories_merges_path, special_tokens)


# Common function for analyzing model performance (used in Exp 7 and Exp 8)
def analyze_model_performance(
    name, n, d, h, p=50257, q=1024, bytes_per_param=4, results_memory=None, results_flops=None, results_stats=None
):
    d_f = 4 * d
    print(f"\n--- Model: {name} ---")
    print(f"Parameters: p={p}, q={q}, n={n}, d={d}, h={h}, d_f={d_f}")
    print(f"Precision: {bytes_per_param} bytes/param")

    # Memory Calculation (Bytes)
    mem_embedding = p * d * bytes_per_param
    mem_rms_norm = (2 * n + 1) * d * bytes_per_param
    mem_mha = 4 * n * d * d * bytes_per_param
    mem_rope = 2 * q * d * bytes_per_param
    mem_ffn = 3 * n * d * d_f * bytes_per_param
    mem_linear_head = d * p * bytes_per_param

    total_memory_bytes = mem_embedding + mem_rms_norm + mem_mha + mem_rope + mem_ffn + mem_linear_head
    total_memory_gb = total_memory_bytes / (1024**3)

    mem_percentages = {
        "Embedding": mem_embedding / total_memory_bytes * 100,
        "RMSNorm": mem_rms_norm / total_memory_bytes * 100,
        "MHA": mem_mha / total_memory_bytes * 100,
        "Rope": mem_rope / total_memory_bytes * 100,
        "FFN": mem_ffn / total_memory_bytes * 100,
        "Linear Head": mem_linear_head / total_memory_bytes * 100,
    }
    if results_memory is not None:
        results_memory.append({"Model": name, **mem_percentages})

    print(f"Total Memory: {total_memory_gb:.4f} GB")
    print("Memory Breakdown:")
    for k, v in mem_percentages.items():
        print(f"  {k}: {v:.2f}%")

    # FLOPs Calculation
    flops_embedding = 0  # No FLOPs for embedding
    flops_rms_norm = (2 * n + 1) * q * d
    flops_mha = 6 * n * (q * d * d + q * q * d)
    flops_rope = 4 * q * d * n
    flops_ffn = 6 * n * q * d * d_f
    flops_linear_head = 2 * q * d * p

    total_flops = flops_embedding + flops_rms_norm + flops_mha + flops_rope + flops_ffn + flops_linear_head

    flops_percentages = {
        "Embedding": 0.0,
        "RMSNorm": flops_rms_norm / total_flops * 100,
        "MHA": flops_mha / total_flops * 100,
        "Rope": flops_rope / total_flops * 100,
        "FFN": flops_ffn / total_flops * 100,
        "Linear Head": flops_linear_head / total_flops * 100,
    }
    if results_flops is not None:
        results_flops.append({"Model": name, **flops_percentages})

    # Store general stats
    if results_stats is not None:
        results_stats.append(
            {
                "Model": name,
                "p": p,
                "q": q,
                "n": n,
                "d": d,
                "h": h,
                "d_f": d_f,
                "Precision": f"{bytes_per_param} bytes/param",
                "Total Memory": f"{total_memory_gb:.4f} GB",
                "Total FLOPs": f"{total_flops:.2e}",
            }
        )

    print(f"Total FLOPs: {total_flops:.2e}")
    print("FLOPs Breakdown:")
    for k, v in flops_percentages.items():
        print(f"  {k}: {v:.2f}%")


if EXPERIMENT_TYPE == 1:

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

elif EXPERIMENT_TYPE == 2:
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

elif EXPERIMENT_TYPE == 3 or EXPERIMENT_TYPE == 4:
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
    run_compression_test(owt_tokenizer, TinyStories_train_data_path, "OWT_Tokenizer", "TinyStories_Data")
    run_compression_test(TinyStories_tokenizer, TinyStories_train_data_path, "TS_Tokenizer", "TinyStories_Data")
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

elif EXPERIMENT_TYPE == 5:
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
    run_throughput_test(TinyStories_tokenizer, TinyStories_train_data_path, "TS_Tokenizer")
    """
    [OWT_Tokenizer] Throughput: 0.39 MB/s
    [OWT_Tokenizer] Estimated time for Pile (825GB): 600.14 hours
    [TS_Tokenizer] Throughput: 0.42 MB/s
    [TS_Tokenizer] Estimated time for Pile (825GB): 555.43 hours
    """

elif EXPERIMENT_TYPE == 6:
    print("\n================ Experiment 6: Dataset Encoding ================")
    owt_path = data_dir / "owt_small.txt"

    def encode_and_save(tokenizer, input_path, output_path):
        print(f"Encoding {input_path.name}...")
        with open(input_path, encoding="utf-8") as f:
            text = f.read()

        ids = tokenizer.encode(text)
        ids_array = np.array(ids, dtype=np.uint16)
        np.save(output_path, ids_array)
        print(f"Saved to {output_path} (Shape: {ids_array.shape}, Dtype: {ids_array.dtype})")

    def encode_and_save_efficiently(tokenizer, input_path, output_path):
        print(f"Encoding {input_path.name} iteratively...")

        with open(input_path, encoding="utf-8") as f:
            token_id_generator = tokenizer.encode_iterable(f)
            ids_array = np.fromiter(token_id_generator, dtype=np.uint16)

        np.save(output_path, ids_array)
        print(f"Saved to {output_path} (Shape: {ids_array.shape})")

    # encode_and_save(owt_tokenizer, owt_path, data_dir / "owt_small_train.npy")
    encode_and_save_efficiently(
        TinyStories_tokenizer, TinyStories_train_data_path, data_dir / "TinyStoriesV2-GPT4-train.npy"
    )
    encode_and_save_efficiently(
        TinyStories_tokenizer, TinyStories_valid_data_path, data_dir / "TinyStoriesV2-GPT4-valid.npy"
    )

elif EXPERIMENT_TYPE == 7:
    print("\n============ Experiment 7: Model Memory & FLOPs ============")

    results_memory = []
    results_flops = []
    results_stats = []

    # Run analysis for different GPT-2 configurations
    analyze_model_performance(
        "GPT-2 Small",
        n=12,
        d=768,
        h=12,
        results_memory=results_memory,
        results_flops=results_flops,
        results_stats=results_stats,
    )
    analyze_model_performance(
        "GPT-2 Medium",
        n=24,
        d=1024,
        h=16,
        results_memory=results_memory,
        results_flops=results_flops,
        results_stats=results_stats,
    )
    analyze_model_performance(
        "GPT-2 Large",
        n=36,
        d=1280,
        h=20,
        results_memory=results_memory,
        results_flops=results_flops,
        results_stats=results_stats,
    )
    analyze_model_performance(
        "GPT-2 XL",
        n=48,
        d=1600,
        h=25,
        results_memory=results_memory,
        results_flops=results_flops,
        results_stats=results_stats,
    )

    # Generate Markdown Table
    md_output_path = current_dir / "experiment7_results.md"
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 7 Results: Memory & FLOPs Breakdown\n\n")

        # Shared Parameters Note
        first_row = results_stats[0]
        f.write(f"**Common Parameters:** vocab_size={first_row['p']}, context_length={first_row['q']}\n\n")

        # Model Stats Table
        f.write("## Model Statistics\n")
        f.write("| Model | num_layers | d_model | num_heads | d_ff | Precision | Total Memory | Total FLOPs |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in results_stats:
            f.write(
                f"| {row['Model']} | {row['n']} | {row['d']} | {row['h']} | {row['d_f']} | {row['Precision']} | {row['Total Memory']} | {row['Total FLOPs']} |\n"
            )

        f.write("\n")

        # Memory Table
        f.write("## Memory Distribution (%)\n")
        f.write("| Model | Embedding | RMSNorm | MHA | Rope | FFN | Linear Head |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in results_memory:
            f.write(
                f"| {row['Model']} | {row['Embedding']:.2f} | {row['RMSNorm']:.2f} | {row['MHA']:.2f} | {row['Rope']:.2f} | {row['FFN']:.2f} | {row['Linear Head']:.2f} |\n"
            )

        f.write("\n")

        # FLOPs Table
        f.write("## FLOPs Distribution (%)\n")
        f.write("| Model | Embedding | RMSNorm | MHA | Rope | FFN | Linear Head |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in results_flops:
            f.write(
                f"| {row['Model']} | {row['Embedding']:.2f} | {row['RMSNorm']:.2f} | {row['MHA']:.2f} | {row['Rope']:.2f} | {row['FFN']:.2f} | {row['Linear Head']:.2f} |\n"
            )

    print(f"\nResults saved to {md_output_path}")

elif EXPERIMENT_TYPE == 8:
    print("\n============ Experiment 8: GPT-2 XL Context Length Scaling ============")

    results_memory = []
    results_flops = []
    results_stats = []

    # GPT-2 XL (1024 context)
    analyze_model_performance(
        "GPT-2 XL (1k Context)",
        n=48,
        d=1600,
        h=25,
        q=1024,
        results_memory=results_memory,
        results_flops=results_flops,
        results_stats=results_stats,
    )

    # GPT-2 XL Extended (16384 context)
    analyze_model_performance(
        "GPT-2 XL (16k Context)",
        n=48,
        d=1600,
        h=25,
        q=16384,
        results_memory=results_memory,
        results_flops=results_flops,
        results_stats=results_stats,
    )

    # Generate Markdown Table for Exp 8
    md_output_path = current_dir / "experiment8_results.md"
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 8 Results: GPT-2 XL Context Length Scaling\n\n")

        # Model Stats Table
        f.write("## Model Statistics\n")
        f.write(
            "| Model | num_layers | d_model | num_heads | d_ff | vocab_size | context_length | Precision | Total Memory | Total FLOPs |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for row in results_stats:
            f.write(
                f"| {row['Model']} | {row['n']} | {row['d']} | {row['h']} | {row['d_f']} | {row['p']} | {row['q']} | {row['Precision']} | {row['Total Memory']} | {row['Total FLOPs']} |\n"
            )

        f.write("\n")

        # Memory Table
        f.write("## Memory Distribution (%)\n")
        f.write("| Model | Embedding | RMSNorm | MHA | Rope | FFN | Linear Head |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in results_memory:
            f.write(
                f"| {row['Model']} | {row['Embedding']:.2f} | {row['RMSNorm']:.2f} | {row['MHA']:.2f} | {row['Rope']:.2f} | {row['FFN']:.2f} | {row['Linear Head']:.2f} |\n"
            )

        f.write("\n")

        # FLOPs Table
        f.write("## FLOPs Distribution (%)\n")
        f.write("| Model | Embedding | RMSNorm | MHA | Rope | FFN | Linear Head |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in results_flops:
            f.write(
                f"| {row['Model']} | {row['Embedding']:.2f} | {row['RMSNorm']:.2f} | {row['MHA']:.2f} | {row['Rope']:.2f} | {row['FFN']:.2f} | {row['Linear Head']:.2f} |\n"
            )

    print(f"\nResults saved to {md_output_path}")

elif EXPERIMENT_TYPE == 9:

    def run_original_experiment():
        print("\n=== Running Original Experiment (LR=1, 100 iterations) ===")
        torch.manual_seed(42)
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=1)

        losses = []
        for t in range(100):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            loss_val = loss.cpu().item()
            losses.append(loss_val)
            # print(loss_val) # Reduced verbosity for bulk run, but keeping logic
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.

        print(f"Final Loss (Iter 100): {losses[-1]:.4f}")
        return losses

    def run_lr_comparison():
        print("\n=== Running LR Comparison Experiment (LR=[1, 10, 100, 1000], 10 iterations) ===")
        lrs = [1.0, 10.0, 100.0, 1000.0]
        results = {}

        # Set seed for reproducibility to compare across LRs fairly
        # Using a fixed initial tensor for all runs in this comparison
        torch.manual_seed(42)
        initial_data = 5 * torch.randn((10, 10))

        print(f"{'LR':<10} | {'Iter 0':<12} | {'Iter 9':<12} | {'Trend'}")
        print("-" * 60)

        for lr in lrs:
            weights = torch.nn.Parameter(initial_data.clone())
            opt = SGD([weights], lr=lr)

            losses = []
            for t in range(10):
                opt.zero_grad()
                loss = (weights**2).mean()
                loss_val = loss.item()
                losses.append(loss_val)

                loss.backward()
                opt.step()

            results[lr] = losses

            # Determine trend
            trend = "Decreasing" if losses[-1] < losses[0] else "Increasing/Diverging"
            print(f"{lr:<10} | {losses[0]:<12.4f} | {losses[-1]:<12.4f} | {trend}")

        # Plotting
        plt.figure(figsize=(10, 6))
        for lr, losses in results.items():
            plt.plot(range(len(losses)), losses, marker="o", label=f"LR={lr}")

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("SGD Training Loss with Different Learning Rates (10 Iterations)")
        plt.legend()
        plt.grid(True)

        output_path = "loss_comparison.png"
        plt.savefig(output_path)
        print(f"\nPlot saved to {os.path.abspath(output_path)}")

    # Original experiment requested by user prompt
    run_original_experiment()

    # New experiment requested (comparison)
    run_lr_comparison()

elif EXPERIMENT_TYPE == 10:
    print("\n================ Experiment 10: Hyperparameter Optimization ================")

    def objective(trial):
        # Define search space around the default values provided
        # Defaults: lr=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8

        # Log-uniform search for Learning Rate (centered around 1e-3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # Log-uniform search for Weight Decay (centered around 0.1)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 1.0, log=True)

        # Uniform search for Beta1 (around 0.9)
        beta1 = trial.suggest_float("beta1", 0.85, 0.95)

        # Uniform search for Beta2 (around 0.95)
        beta2 = trial.suggest_float("beta2", 0.90, 0.99)

        # Log-uniform search for Epsilon (around 1e-8)
        eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)

        # Integer search for Warmup Iterations (around 100)
        warmup_iters = trial.suggest_int("warmup_iters", 30, 70)

        # Integer search for Cosine Cycle Iterations
        # We search around the total number of iterations (200), allowing it to be shorter or longer
        cosine_cycle_iters = trial.suggest_int("cosine_cycle_iters", 120, 150)

        print(f"\n--- Trial {trial.number} ---")
        print(
            f"Params: lr={lr:.2e}, wd={weight_decay:.2e}, b1={beta1:.4f}, b2={beta2:.4f}, eps={eps:.2e}, warmup={warmup_iters}, cycle={cosine_cycle_iters}"
        )

        # Run training
        train_path = data_dir / "TinyStoriesV2-GPT4-train.npy"
        valid_path = data_dir / "TinyStoriesV2-GPT4-valid.npy"

        if not train_path.exists():
            print(f"Warning: {train_path} does not exist. Please run Experiment 6 first to generate .npy files.")
            pass

        try:
            _, val_losses = run_train(
                vocab_size=10000,
                context_length=256,
                d_model=512,
                num_layers=4,
                num_heads=16,
                d_ff=1344,
                rope_theta=10000.0,
                train_path=train_path,
                valid_path=valid_path,
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                warmup_iters=warmup_iters,
                cosine_cycle_iters=cosine_cycle_iters,
                num_iterations=150,
            )
            final_val_loss = val_losses[-1]
            return final_val_loss
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    # Create a study and optimize
    study = optuna.create_study(direction="minimize")

    # We limit the number of trials to 8 to demonstrate "searching" without "traversing" the full grid.
    study.optimize(objective, n_trials=8)

    print("\n================ Optimization Results ================")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Val Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Visualization
    print("\nSaving optimization visualizations...")
    try:
        # Use matplotlib backend instead of plotly to avoid browser requirements
        from optuna.visualization import matplotlib as optuna_plt

        # 1. Optimization History
        plt.figure()
        optuna_plt.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig("opt_history.png")
        print("Saved opt_history.png")

        # 2. Parallel Coordinate Plot
        # Create a wide figure
        plt.figure(figsize=(20, 10))
        optuna_plt.plot_parallel_coordinate(study)
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.9)  # Leave space for labels
        plt.savefig("opt_parallel.png", bbox_inches="tight")
        print("Saved opt_parallel.png")

        # 3. Parameter Importances
        # Requires scikit-learn
        # try:
        #     plt.figure()
        #     optuna_plt.plot_param_importances(study)
        #     plt.tight_layout()
        #     plt.savefig("opt_importance.png")
        #     print("Saved opt_importance.png")
        # except Exception as e:
        #     print(f"Could not plot parameter importance (might need more than 1 completed trial): {e}")

    except Exception as e:
        print(f"Error saving visualizations: {e}")
    """
    ================ Optimization Results ================
    Best trial:
    Value (Val Loss): 3.4611518383026123
    Params:
        lr: 0.0011729107375170252
        weight_decay: 0.10978009021658337
        beta1: 0.8985580334915605
        beta2: 0.9811726002887176
        eps: 1.4622374563481237e-08
        warmup_iters: 38
        cosine_cycle_iters: 149
    """
elif EXPERIMENT_TYPE == 11:
    print("\n================ Experiment 11: Decode LM Checkpoint ================")

    device = "cpu" if torch.cuda.is_available() else "cpu"

    vocab_size = 10000
    context_length = 256
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000.0

    transform_lm = TFLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    optimizer = AdamW(transform_lm.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.98), eps=1e-8)

    checkpoint_path = current_dir / "checkpoints" / "checkpoint_1000.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
    else:
        iteration = load_checkpoint(str(checkpoint_path), transform_lm, optimizer)
        print(f"Loaded checkpoint from iteration {iteration} at {checkpoint_path}")

        tokenizer = BPETokenizer.from_files(TinyStories_vocab_path, TinyStories_merges_path, special_tokens)

        max_len = 128
        temperature = 0.8
        top_p = 0.9

        print("Type 'exit' or 'quit' to stop.")

        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue

            ids = tokenizer.encode(prompt)
            x = torch.tensor([ids], dtype=torch.long, device=device)

            response = generate(x, max_len, temperature, top_p, transform_lm, tokenizer)
            print(f"Model: {response}")
