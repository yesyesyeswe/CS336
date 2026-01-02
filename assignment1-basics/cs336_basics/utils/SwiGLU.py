from torch.nn import Module
from torch import Tensor, sigmoid


from cs336_basics.utils.Linear import Linear


class SwiGLU(Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.w2 = Linear(self.d_ff, self.d_model, self.device, self.dtype)
        self.w3 = Linear(self.d_model, self.d_ff, self.device, self.dtype)

    def SiLU(self, x: Tensor) -> Tensor:
        return x * sigmoid(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.SiLU(self.w1(x)) * self.w3(x))
