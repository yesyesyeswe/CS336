from torch import Tensor, arange
from torch.nn import Module
from einops import repeat
from cs336_basics.utils import MHA, RMSNorm, SwiGLU


class TFBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 0, theta: float = 0.0):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.mha = MHA(d_model, num_heads, max_seq_len, theta)
        self.rmsnorm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        token_positions = repeat(arange(seq_len, device=x.device), "seq -> batch seq", batch=batch)
        x = x + self.mha(self.rmsnorm1(x), token_positions)
        return x + self.ffn(self.rmsnorm2(x))
