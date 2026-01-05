from torch import Tensor, max, sum, exp, log, arange, zeros_like, sqrt, norm

from einops import reduce

from collections.abc import Callable
from torch.optim import Optimizer
import math


def cross_entropy(inputs: Tensor, targets: Tensor):
    max_val, _ = max(inputs, -1, keepdim=True)
    exp_sum = sum(exp(inputs - max_val), -1)
    return reduce(-inputs[arange(inputs.size(0)), targets] + max_val + log(exp_sum), "... batch_size -> ", "mean")


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
