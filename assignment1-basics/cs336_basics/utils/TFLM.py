from torch import Tensor
from torch.nn import Module, ModuleList
from cs336_basics.utils import Linear, RMSNorm, TFBlock, Embedding


class TFLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.transform_blocks = ModuleList(
            [TFBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        )
        self.embedding = Embedding(vocab_size, d_model)
        self.rmsnorm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        for transform_block in self.transform_blocks:
            x = transform_block(x)
        return self.lm_head(self.rmsnorm(x))
