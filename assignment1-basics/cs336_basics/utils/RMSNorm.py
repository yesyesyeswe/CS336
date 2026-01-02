from torch.nn import Module, Parameter
from torch import Tensor, ones, rsqrt, float32

from einops import reduce


class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = Parameter(ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(float32)

        # Compute the inverse RMS factor: 1 / sqrt(mean(x^2) + eps)
        inv_rms = rsqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)

        # Apply scale and gain in a single element-wise multiplication to save memory
        x = x * (inv_rms * self.g)

        return x.to(in_dtype)
