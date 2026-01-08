import torch
from cs336_basics.utils import AdamW, TFLM, get_batch, cross_entropy, save_checkpoint, lr_cosine_schedule
from numpy import load
import matplotlib
import argparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import wandb
from pathlib import Path


def load_data(train_path: str | Path, valid_path: str | Path, ratio: float = 0.1):
    if valid_path is None:
        train_dataset = load(train_path, mmap_mode="r")
        train_num = int(len(train_dataset) * (1 - ratio))
        return train_dataset[:train_num], train_dataset[train_num:]

    train_dataset = load(train_path, mmap_mode="r")
    valid_dataset = load(valid_path, mmap_mode="r")
    return train_dataset, valid_dataset


def run_train(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    train_path: str | Path,
    valid_path: str | Path,
    lr: float = 1e-3,
    weight_decay: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    warmup_iters: int = 100,
    num_iterations: int = 100,
    min_lr: float | None = None,
    cosine_cycle_iters: int | None = None,
):
    if min_lr is None:
        min_lr = lr * 0.1

    if cosine_cycle_iters is None:
        cosine_cycle_iters = num_iterations

    batch_size = 10
    sequence_length = context_length
    device = "cpu" if torch.cuda.is_available() else "cpu"

    # Initialize wandb
    entity = None
    if os.path.exists("wandb_entity.txt"):
        with open("wandb_entity.txt") as f:
            entity = f.read().strip()

    wandb.init(
        entity=entity,
        project="cs336-assignment1",
        config={
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "lr": lr,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "warmup_iters": warmup_iters,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "min_lr": min_lr,
        },
    )

    transform_lm = TFLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    opt = AdamW(transform_lm.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    train_data, val_data = load_data(train_path, valid_path)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    for t in range(num_iterations):
        # Update learning rate
        current_lr = lr_cosine_schedule(t, lr, min_lr, warmup_iters, cosine_cycle_iters)
        for group in opt.param_groups:
            group["lr"] = current_lr

        # Training step
        transform_lm.train()
        opt.zero_grad()
        x_train, y_train = get_batch(train_data, batch_size, sequence_length, device)
        logits = transform_lm(x_train)
        loss = cross_entropy(logits, y_train)
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

        metrics = {"train_loss": loss.item(), "iteration": t + 1}

        # Validation step

        transform_lm.eval()
        with torch.no_grad():
            x_val, y_val = get_batch(val_data, batch_size, sequence_length, device)
            val_logits = transform_lm(x_val)
            val_loss = cross_entropy(val_logits, y_val)
            val_losses.append(val_loss.item())

            # Add validation metrics to the same dictionary
            metrics["val_loss"] = val_loss.item()

        # Log all metrics at once
        wandb.log(metrics)

        # Save checkpoint every 5 iterations or at the last iteration
        if (t + 1) % 200 == 0 or (t + 1) == num_iterations:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{t + 1}.pt")
            save_checkpoint(transform_lm, opt, t + 1, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if (t + 1) == num_iterations:
            print(f"Final Validation Perplexity: {torch.exp(val_loss).item():.4f}")

    print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Val Loss: {val_losses[-1]:.4f}")

    # Finish wandb run
    wandb.finish()

    return train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_iters", type=int, default=38)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--train_path", type=str, default="data/TinyStoriesV2-GPT4-train.npy")
    parser.add_argument("--valid_path", type=str, default="data/TinyStoriesV2-GPT4-valid.npy")

    args = parser.parse_args()

    train_losses, val_losses = run_train(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        train_path=args.train_path,
        valid_path=args.valid_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        warmup_iters=args.warmup_iters,
        num_iterations=args.num_iterations,
    )

    # Plotting losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss", marker="o")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss", marker="x")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    output_path = "train_val_loss.png"
    plt.savefig(output_path)
    print(f"Plot saved to {os.path.abspath(output_path)}")
