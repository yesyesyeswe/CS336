from torch import Tensor, exp, log, arange, zeros_like, sqrt, norm, from_numpy, save, load
from torch.nn import Module

from einops import reduce, rearrange

from collections.abc import Callable
from torch.optim import Optimizer
import math

import numpy.typing as npt
from numpy import random, stack

import os
import typing


def cross_entropy(inputs: Tensor, targets: Tensor):
    # Numerical stability: subtract max before exp
    max_val = reduce(inputs, "... vocab -> ... 1", "max")
    # Calculate log(sum(exp(x - max)))
    exp_sum = reduce(exp(inputs - max_val), "... vocab -> ... 1", "sum")
    # log_sum_exp: shape (..., 1)
    log_sum_exp = max_val + log(exp_sum)

    # Flatten inputs and targets to handle arbitrary dimensions (e.g., batch, seq)
    # inputs: (..., vocab) -> (N, vocab)
    # targets: (...) -> (N,)
    inputs_flat = rearrange(inputs, "... vocab -> (...) vocab")
    targets_flat = rearrange(targets, "... -> (...)")

    # Select the logits corresponding to the target classes
    target_logits = inputs_flat[arange(inputs_flat.shape[0]), targets_flat]

    # Reshape target_logits back to (...) or just align with log_sum_exp for subtraction
    # log_sum_exp is (..., 1), we can flatten it to (N,) to match target_logits
    log_sum_exp_flat = rearrange(log_sum_exp, "... 1 -> (...)")

    # Loss = -x[class] + log(sum(exp(x)))
    loss = -target_logits + log_sum_exp_flat

    return reduce(loss, "batch -> ", "mean")


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(Optimizer):
    def __init__(self, params, lr=0.0, weight_decay=0.9, betas=(0.9, 0.95), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            _cache_alpha_t = None
            _cache_t = 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                m = state.get("m", zeros_like(grad))
                v = state.get("v", zeros_like(grad))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                if t != _cache_t:
                    _cache_alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                    _cache_t = t

                alpta_t = _cache_alpha_t
                p.data -= alpta_t * m / (sqrt(v) + eps)
                p.data = (1 - lr * weight_decay) * p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss


def lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate

    return (
        min_learning_rate
        + (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))
        * (max_learning_rate - min_learning_rate)
        / 2
    )


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    total_norm = 0.0
    for p in parameters:
        if p is None or p.grad is None:
            continue
        total_norm += norm(p.grad) ** 2

    total_norm = total_norm**0.5
    for p in parameters:
        if p is None or p.grad is None:
            continue
        if norm(p.grad) < max_l2_norm:
            continue
        p.grad *= max_l2_norm / (total_norm + 1e-6)


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    idx = random.randint(0, len(dataset) - context_length, size=(batch_size,))
    x_batch = stack([dataset[i : i + context_length] for i in idx])
    y_batch = stack([dataset[i + 1 : i + 1 + context_length] for i in idx])

    return from_numpy(x_batch).to(device).long(), from_numpy(y_batch).to(device).long()


def save_checkpoint(
    model: Module, optimizer: Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    dict_save = {}
    dict_save["model"] = model.state_dict()
    dict_save["optimizer"] = optimizer.state_dict()
    dict_save["iteration"] = iteration

    save(dict_save, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: Module, optimizer: Optimizer):
    dict_save = load(src, weights_only=False)
    model.load_state_dict(dict_save["model"])
    optimizer.load_state_dict(dict_save["optimizer"])
    return dict_save["iteration"]
