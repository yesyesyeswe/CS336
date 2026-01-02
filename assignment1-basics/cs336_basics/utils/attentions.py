from torch import Tensor, triu, bool, ones
from torch.nn import Module
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
from cs336_basics.utils import softmax, Linear, RotaryPositionalEmbedding


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    QK = einsum(Q / Q.size(-1) ** 0.5, K, " ... queries d_k, ... keys d_k -> ... queries keys")
    QK = QK.masked_fill(~mask, float("-inf"))

    return einsum(softmax(QK, -1), V, " ... queries seq_len, ... seq_len d_v -> ... queries d_v")


class MHA(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 0,
        theta: float = 0.0,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.W_QKV = Linear(d_model, 3 * d_model)
        self.W_O = Linear(d_model, d_model)
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions

    def forward(self, x: Tensor):
        Qs, Ks, Vs = rearrange(
            self.W_QKV(x), "... sequence_length (k h d_k) -> k h ... sequence_length d_k", k=3, h=self.num_heads
        )

        if self.max_seq_len:
            rope = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len, x.device)
            Qs = rope(Qs, self.token_positions)
            Ks = rope(Ks, self.token_positions)

        seq_len = x.size(-2)
        mask = ~triu(ones(seq_len, seq_len, device=x.device), diagonal=1).to(bool)

        multi_head_attn = rearrange(
            scaled_dot_product_attention(Qs, Ks, Vs, mask), "h ... sequence_length d_k -> ... sequence_length (h d_k)"
        )
        return self.W_O(multi_head_attn)
