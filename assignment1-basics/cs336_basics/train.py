import torch
from cs336_basics.utils import AdamW, TFLM, get_batch, cross_entropy, save_checkpoint
from numpy import load
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import wandb
from pathlib import Path


def load_data(path: str | Path, ratio: float = 0.1):
    dataset = load(path, mmap_mode="r")
    train_num = int(len(dataset) * (1 - ratio))
    return dataset[:train_num], dataset[train_num:]


def run_train(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    data_path: str | Path,
    lr: float = 1e-3,
    weight_decay: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
):
    batch_size = 10
    num_iterations = 1000
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
        },
    )

    transform_lm = TFLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    opt = AdamW(transform_lm.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    train_data, val_data = load_data(data_path)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    for t in range(num_iterations):
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
        if (t + 1) % 5 == 0 or (t + 1) == num_iterations:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{t + 1}.pt")
            save_checkpoint(transform_lm, opt, t + 1, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if (t + 1) == num_iterations:
            print(f"Final Validation Perplexity: {torch.exp(val_loss).item():.4f}")

    print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Val Loss: {val_losses[-1]:.4f}")

    # Finish wandb run
    wandb.finish()

    return train_losses, val_losses


def train():
    data_path = Path(__file__).parent.parent / "data" / "ts_small_train.npy"

    vocab_size = 10000
    context_length = 256
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000.0

    train_losses, val_losses = run_train(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        data_path=data_path,
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


if __name__ == "__main__":
    train()
