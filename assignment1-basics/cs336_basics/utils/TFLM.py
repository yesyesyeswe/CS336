from torch import Tensor
import torch
from torch.nn import Module, ModuleList
from cs336_basics.utils import Linear, RMSNorm, TFBlock, Embedding, softmax


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

    @torch.no_grad()
    def generate(self, x, max_len: int, t: float, p: float, bpe_tokenizer):
        response_id = []
        while len(response_id) < max_len:
            logits = self.forward(x)

            logits = logits[0, -1, :]

            # 1. Apply temperature and calculate probabilities
            probs = softmax(logits / t)

            # 2. Sort probabilities: value descending, index ascending.
            # Since torch.sort is stable, sorting descending by value will
            # keep indices in ascending order for equal values.
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, stable=True)

            # 3. Top-p (Nucleus) sampling
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold p.
            # We keep the first token that exceeds p, so we shift the mask.
            # keep_mask[i] is True if sum(sorted_probs[:i]) < p
            keep_mask = (cumulative_probs - sorted_probs) < p

            # Ensure we keep at least one token
            if not keep_mask.any():
                keep_mask[0] = True

            filtered_probs = sorted_probs[keep_mask]
            filtered_indices = sorted_indices[keep_mask]

            # 4. Renormalize and sample
            filtered_probs = filtered_probs / filtered_probs.sum()
            sample_idx = torch.multinomial(filtered_probs, 1).item()
            next_token_id = filtered_indices[sample_idx].item()

            response_id.append(next_token_id)

            if bpe_tokenizer.decode([next_token_id]) == "<|endoftext|>":
                break

            # Update x for the next iteration
            x = torch.cat([x, torch.tensor([[next_token_id]], device=x.device)], dim=1)

        return bpe_tokenizer.decode(response_id)
