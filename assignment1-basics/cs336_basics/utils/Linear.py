from torch.nn import Module, Parameter
from torch import Tensor, empty
from torch.nn.init import trunc_normal_

from einops import einsum


class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = Parameter(empty(out_features, in_features, device=device, dtype=dtype))
        sigma = (2 / (out_features + in_features)) ** 0.5
        trunc_normal_(
            self.W,
            mean=0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.W, "... in_feature, out_feature in_feature -> ... out_feature")
