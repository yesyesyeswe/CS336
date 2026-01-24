import os
import torch
from pathlib import Path
from timeit import default_timer
from cs336_basics.utils import TFLM, AdamW, cross_entropy, lr_cosine_schedule, load_data, get_batch
import numpy as np


def naive_profile(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    train_path: str | Path,
    valid_path: str | Path,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    warmup_iters: int = 100,
    test_iterations: int = 100,
    min_lr: float | None = None,
    cosine_cycle_iters: int | None = None,
    forward_only: bool = False,
):
    if min_lr is None:
        min_lr = lr * 0.1

    if cosine_cycle_iters is None:
        cosine_cycle_iters = test_iterations

    sequence_length = context_length
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_lm = TFLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    opt = AdamW(transform_lm.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    train_data, val_data = load_data(train_path, valid_path)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = -1
    total_iters = test_iterations + warmup_iters
    forward_time_record = []
    backward_time_record = []
    for t in range(total_iters):
        # Update learning rate
        current_lr = lr_cosine_schedule(t, lr, min_lr, warmup_iters, cosine_cycle_iters)
        for group in opt.param_groups:
            group["lr"] = current_lr

        # Training step
        transform_lm.train()
        opt.zero_grad()
        x_train, y_train = get_batch(train_data, batch_size, sequence_length, device)

        if t >= warmup_iters:
            start_time = default_timer()
        logits = transform_lm(x_train)
        torch.cuda.synchronize()
        if t >= warmup_iters:
            forward_time_record.append(default_timer() - start_time)

        if not forward_only and t >= warmup_iters:
            start_time = default_timer()
            loss = cross_entropy(logits, y_train)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            backward_time_record.append(default_timer() - start_time)

            train_losses = loss.item()

    if not forward_only:
        print(f"Final Train Loss: {train_losses:.4f}")

    print(f"average forward time used for {test_iterations} iteritions: {np.mean(forward_time_record)}")
    print(f"forward time used Variation for {test_iterations} iteritions: {np.var(forward_time_record)}")

    if not forward_only:
        print(f"average backward time used for {test_iterations} iteritions: {np.mean(backward_time_record)}")
        print(f"backward time used Variation for {test_iterations} iteritions: {np.var(backward_time_record)}")
        print(
            f"average total time used for {test_iterations} iteritions: {np.mean(np.array(backward_time_record) + np.array(forward_time_record))}"
        )
        print(
            f"total time used Variation for {test_iterations} iteritions: {np.var(np.array(backward_time_record) + np.array(forward_time_record))}"
        )


if __name__ == "__main__":
    import pandas as pd

    data = [
        ["small", 768, 3072, 12, 12],
        # ["medium", 1024, 4096, 24, 16],
        # ["large", 1280, 5120, 36, 20],
        # ["xl", 1600, 6400, 48, 25],
        # ["2.7B", 2560, 10240, 32, 32],
    ]
    columns = ["Size", "d_model", "d_ff", "num_layers", "num_heads"]
    df = pd.DataFrame(data, columns=columns)

    print("Experiment Configurations:")
    print(df.to_markdown(index=False))

    # Path to data files (relative to this script)
    project_root = Path(__file__).parent.parent.parent.parent
    train_path = project_root / "assignment1-basics" / "data" / "TinyStoriesV2-GPT4-train.npy"
    valid_path = project_root / "assignment1-basics" / "data" / "TinyStoriesV2-GPT4-valid.npy"

    warmup_iters = 5
    test_iterations = 10

    for index, row in df.iterrows():
        print(f"\n{'=' * 50}")
        print(f"Profiling Model Size: {row['Size']}, warmup_iters: {warmup_iters}, test_iterations: {test_iterations}")
        print(f"{'=' * 50}")

        naive_profile(
            vocab_size=10000,
            context_length=512,
            d_model=int(row["d_model"]),
            num_layers=int(row["num_layers"]),
            num_heads=int(row["num_heads"]),
            d_ff=int(row["d_ff"]),
            rope_theta=10000.0,
            train_path=train_path,
            valid_path=valid_path,
            batch_size=4,
            warmup_iters=warmup_iters,  # Reduced for profiling speed
            test_iterations=test_iterations,  # Reduced for profiling speed
        )

"""
==================================================
Profiling Model Size: small, warmup_iters: 5, test_iterations: 10
==================================================
Final Train Loss: 6.2311
average forward time used for 10 iteritions: 4.607869516500796
forward time used Variation for 10 iteritions: 0.6078592694446666
average backward time used for 10 iteritions: 8.63726463489984
backward time used Variation for 10 iteritions: 0.7354710869782133
average total time used for 10 iteritions: 13.245134151400634
total time used Variation for 10 iteritions: 0.6255006028507073

==================================================
Profiling Model Size: small, warmup_iters: 0, test_iterations: 10
==================================================
Final Train Loss: 5.6034
average forward time used for 10 iteritions: 3.453805025999827
forward time used Variation for 10 iteritions: 0.12513927088104573
average backward time used for 10 iteritions: 7.272198764199129
backward time used Variation for 10 iteritions: 1.0779655278887286
average total time used for 10 iteritions: 10.726003790198956
total time used Variation for 10 iteritions: 1.8657957032517885


==================================================
Profiling Model Size: small, warmup_iters: 1, test_iterations: 10
==================================================
Final Train Loss: 5.6895
average forward time used for 10 iteritions: 4.3821981571003565
forward time used Variation for 10 iteritions: 0.6517095160219464
average backward time used for 10 iteritions: 7.6523379074000335
backward time used Variation for 10 iteritions: 1.1566172836085964
average total time used for 10 iteritions: 12.03453606450039
total time used Variation for 10 iteritions: 0.22597211608881557

从数据上看，使用 5 步 warm‑up 后 forward 和 backward 的平均耗时略高，但标准差较小，说明结果更稳定、更能代表模型在长期训练中的真实性能；而 0 或 1 步 warm‑up 给出更低的平均时间，此原因不明。但方差更大、波动更明显，是因为混入了冷启动和初始化。
"""
