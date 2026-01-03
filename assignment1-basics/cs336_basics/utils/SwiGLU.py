from torch.nn import Module
from torch import Tensor, sigmoid

from cs336_basics.utils.Linear import Linear


def SiLU(x: Tensor) -> Tensor:
    return x * sigmoid(x)


class SwiGLU(Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))
