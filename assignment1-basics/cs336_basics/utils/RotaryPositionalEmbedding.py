from torch.nn import Module
from torch import Tensor, empty, arange, sin, cos, exp, log, tensor

from einops import einsum, rearrange


class RotaryPositionalEmbedding(Module):
    _R_cache: dict[tuple, Tensor] = {}

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        cache_key = (theta, d_k, max_seq_len, str(device))
        if cache_key in RotaryPositionalEmbedding._R_cache:
            R = RotaryPositionalEmbedding._R_cache[cache_key]
        else:
            # Stable way
            thetas = einsum(
                arange(max_seq_len, device=device),
                exp(-2 * arange(0, d_k // 2, device=device) * log(tensor(theta, device=device)) / d_k),
                "i, k -> i k",
            )
            cos_thetas = cos(thetas)
            sin_thetas = sin(thetas)

            R = empty(max_seq_len, 2, d_k, device=device)
            R[:, 0, 0::2] = cos_thetas
            R[:, 1, 1::2] = cos_thetas
            R[:, 0, 1::2] = -sin_thetas
            R[:, 1, 0::2] = sin_thetas
            RotaryPositionalEmbedding._R_cache[cache_key] = R

        self.register_buffer("R", R, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # faster(3s), but take more memory
        x = rearrange(x, "... seq_len (d w) -> ... seq_len d w", w=2)
        Rx = rearrange(self.R[token_positions], "... seq_len w1 (d w2) -> ... seq_len d w1 w2", w2=2)
        result = einsum(x, Rx, "... seq_len d w, ... seq_len d w1 w -> ... seq_len d w1")
        return rearrange(result, "... seq_len d w1 -> ... seq_len (d w1)")
        # Slower(4s), but take less memory
        # return dot(
        #     "... seq_len (d w), seq_len w1 (d w) -> ... seq_len (d w1)",
        #     x, self.R[token_positions],
        #     w=2
        # )
