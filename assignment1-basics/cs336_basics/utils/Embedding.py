from torch.nn import Module, Parameter
from torch import Tensor, empty
from torch.nn.init import trunc_normal_


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embeddings_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.vocab = Parameter(empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        trunc_normal_(self.vocab, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.vocab[token_ids]
